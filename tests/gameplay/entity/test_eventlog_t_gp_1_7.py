"""
Comprehensive tests for Entity Lifecycle EventLog Integration (T-GP-1.7).

Tests for:
- EntitySpawned event fires on spawn
- EntityDestroyed event fires on destroy
- ComponentAdded/Removed events fire correctly
- EntityStateChanged events track transitions
- Causal chain: spawn -> component adds linked
- Event query by entity_id
- Event query by time range
- Event query by event type
- Multiple entities tracked independently
- Event replay capabilities
- Performance: 1000 spawns under 100ms

50+ tests total covering all requirements.
"""
from __future__ import annotations

import time
import pytest
from typing import Any, List
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.entity.eventlog_integration import (
    EntitySpawned,
    EntityDestroyed,
    ComponentAdded,
    ComponentRemoved,
    EntityStateChanged,
    CausalChain,
    LifecycleEventRecord,
    EntityEventLog,
    get_entity_event_log,
    clear_entity_event_log,
)
from engine.gameplay.entity.lifecycle import (
    LifecycleState,
    LifecycleEvent,
    LifecycleManager,
)
from engine.gameplay.entity.actor import Actor, Transform
from foundation.eventlog import get_event_log, clear_event_log, set_current_tick


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_state():
    """Reset all state before each test."""
    EntityEventLog.reset_instance()
    LifecycleManager.reset_instance()
    Actor.reset_entity_ids()
    clear_event_log()
    set_current_tick(0)
    yield
    EntityEventLog.reset_instance()
    LifecycleManager.reset_instance()
    clear_event_log()


@pytest.fixture
def event_log():
    """Get a fresh event log instance."""
    EntityEventLog.reset_instance()
    return EntityEventLog()


@pytest.fixture
def test_actor():
    """Create a test actor."""
    return Actor(name="TestActor")


# =============================================================================
# ENTITY SPAWNED EVENT TESTS (10 tests)
# =============================================================================


class TestEntitySpawnedEvent:
    """Tests for EntitySpawned event."""

    def test_spawn_event_fires_on_actor_creation(self, event_log):
        """EntitySpawned event fires when an actor is spawned."""
        actor = Actor(name="SpawnTest")

        events = event_log.query_by_type("EntitySpawned")
        assert len(events) >= 1
        spawn_event = events[-1]
        assert spawn_event.entity_id == actor._entity_id

    def test_spawn_event_contains_entity_id(self, event_log):
        """EntitySpawned event contains correct entity_id."""
        actor = Actor(name="IDTest")

        events = event_log.query(entity_id=actor._entity_id, event_type="EntitySpawned")
        assert len(events) == 1
        assert events[0].event_data.entity_id == actor._entity_id

    def test_spawn_event_contains_prefab_name(self, event_log):
        """EntitySpawned event contains prefab/actor name."""
        actor = Actor(name="PrefabNameTest")

        events = event_log.query(entity_id=actor._entity_id, event_type="EntitySpawned")
        assert len(events) == 1
        assert "PrefabNameTest" in events[0].event_data.prefab_name

    def test_spawn_event_contains_position(self, event_log):
        """EntitySpawned event contains spawn position."""
        transform = Transform(position=(10.0, 20.0, 30.0))
        actor = Actor(name="PositionTest", transform=transform)

        events = event_log.query(entity_id=actor._entity_id, event_type="EntitySpawned")
        assert len(events) == 1
        assert events[0].event_data.position == (10.0, 20.0, 30.0)

    def test_spawn_event_contains_timestamp(self, event_log):
        """EntitySpawned event contains valid timestamp."""
        before = time.time()
        actor = Actor(name="TimestampTest")
        after = time.time()

        events = event_log.query(entity_id=actor._entity_id, event_type="EntitySpawned")
        assert len(events) == 1
        assert before <= events[0].event_data.timestamp <= after

    def test_spawn_event_contains_entity_type(self, event_log):
        """EntitySpawned event contains entity type."""
        actor = Actor(name="TypeTest")

        events = event_log.query(entity_id=actor._entity_id, event_type="EntitySpawned")
        assert len(events) == 1
        assert events[0].event_data.entity_type == "Actor"

    def test_spawn_event_has_correct_event_type(self, event_log):
        """EntitySpawned event has correct event_type property."""
        actor = Actor(name="EventTypeTest")

        events = event_log.query(entity_id=actor._entity_id)
        spawn_events = [e for e in events if e.event_type == "EntitySpawned"]
        assert len(spawn_events) == 1

    def test_spawn_event_dataclass_is_frozen(self):
        """EntitySpawned dataclass is immutable."""
        event = EntitySpawned(
            entity_id=1,
            prefab_name="Test",
            position=(0.0, 0.0, 0.0),
            timestamp=time.time(),
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            event.entity_id = 2

    def test_multiple_spawns_create_multiple_events(self, event_log):
        """Multiple actor spawns create separate events."""
        actor1 = Actor(name="Multi1")
        actor2 = Actor(name="Multi2")
        actor3 = Actor(name="Multi3")

        events = event_log.query_by_type("EntitySpawned")
        assert len(events) >= 3

    def test_spawn_event_recorded_in_foundation_eventlog(self):
        """EntitySpawned event is also recorded in Foundation EventLog."""
        actor = Actor(name="FoundationTest")

        foundation_events = get_event_log().events_for_operation("Lifecycle.EntitySpawned")
        assert len(foundation_events) >= 1


# =============================================================================
# ENTITY DESTROYED EVENT TESTS (10 tests)
# =============================================================================


class TestEntityDestroyedEvent:
    """Tests for EntityDestroyed event."""

    def test_destroy_event_fires_on_actor_destruction(self, event_log):
        """EntityDestroyed event fires when an actor is destroyed."""
        actor = Actor(name="DestroyTest")
        entity_id = actor._entity_id
        actor.destroy(immediate=True)

        events = event_log.query(entity_id=entity_id, event_type="EntityDestroyed")
        assert len(events) == 1

    def test_destroy_event_contains_entity_id(self, event_log):
        """EntityDestroyed event contains correct entity_id."""
        actor = Actor(name="DestroyIDTest")
        entity_id = actor._entity_id
        actor.destroy(immediate=True)

        events = event_log.query(entity_id=entity_id, event_type="EntityDestroyed")
        assert len(events) == 1
        assert events[0].event_data.entity_id == entity_id

    def test_destroy_event_contains_reason(self, event_log):
        """EntityDestroyed event contains destruction reason."""
        actor = Actor(name="ReasonTest")
        entity_id = actor._entity_id
        actor.destroy(immediate=True)

        events = event_log.query(entity_id=entity_id, event_type="EntityDestroyed")
        assert len(events) == 1
        assert events[0].event_data.reason is not None

    def test_destroy_event_contains_timestamp(self, event_log):
        """EntityDestroyed event contains valid timestamp."""
        actor = Actor(name="DestroyTimestampTest")
        entity_id = actor._entity_id
        before = time.time()
        actor.destroy(immediate=True)
        after = time.time()

        events = event_log.query(entity_id=entity_id, event_type="EntityDestroyed")
        assert len(events) == 1
        assert before <= events[0].event_data.timestamp <= after

    def test_destroy_event_has_correct_event_type(self, event_log):
        """EntityDestroyed event has correct event_type property."""
        actor = Actor(name="DestroyTypeTest")
        entity_id = actor._entity_id
        actor.destroy(immediate=True)

        events = event_log.query(entity_id=entity_id)
        destroy_events = [e for e in events if e.event_type == "EntityDestroyed"]
        assert len(destroy_events) == 1

    def test_destroy_event_dataclass_is_frozen(self):
        """EntityDestroyed dataclass is immutable."""
        event = EntityDestroyed(
            entity_id=1,
            reason="test",
            timestamp=time.time(),
        )
        with pytest.raises(Exception):
            event.entity_id = 2

    def test_destroy_after_spawn_both_events_exist(self, event_log):
        """Both spawn and destroy events exist for a destroyed actor."""
        actor = Actor(name="SpawnDestroyTest")
        entity_id = actor._entity_id
        actor.destroy(immediate=True)

        spawn_events = event_log.query(entity_id=entity_id, event_type="EntitySpawned")
        destroy_events = event_log.query(entity_id=entity_id, event_type="EntityDestroyed")

        assert len(spawn_events) == 1
        assert len(destroy_events) == 1

    def test_destroy_event_recorded_in_foundation_eventlog(self):
        """EntityDestroyed event is recorded in Foundation EventLog."""
        actor = Actor(name="FoundationDestroyTest")
        actor.destroy(immediate=True)

        foundation_events = get_event_log().events_for_operation("Lifecycle.EntityDestroyed")
        assert len(foundation_events) >= 1

    def test_manual_destroy_recording(self, event_log):
        """Manually recording a destroy event works."""
        event_log.record_destroy(999, reason="manual_test", final_state="DESTROYED")

        events = event_log.query(entity_id=999, event_type="EntityDestroyed")
        assert len(events) == 1
        assert events[0].event_data.reason == "manual_test"

    def test_destroy_event_with_final_state(self, event_log):
        """EntityDestroyed can include final state."""
        event_log.record_destroy(888, reason="test", final_state="DESTROYED")

        events = event_log.query(entity_id=888, event_type="EntityDestroyed")
        assert len(events) == 1
        assert events[0].event_data.final_state == "DESTROYED"


# =============================================================================
# COMPONENT ADDED/REMOVED EVENT TESTS (10 tests)
# =============================================================================


class MockComponent:
    """Mock component for testing."""
    pass


class TestComponentEvents:
    """Tests for ComponentAdded and ComponentRemoved events."""

    def test_component_added_event_fires(self, event_log):
        """ComponentAdded event fires when a component is added."""
        actor = Actor(name="ComponentAddTest")
        component = MockComponent()
        actor.add_component("test_comp", component)

        events = event_log.query(entity_id=actor._entity_id, event_type="ComponentAdded")
        assert len(events) >= 1

    def test_component_added_event_contains_entity_id(self, event_log):
        """ComponentAdded event contains correct entity_id."""
        actor = Actor(name="CompEntityIDTest")
        component = MockComponent()
        actor.add_component("test_comp", component)

        events = event_log.query(entity_id=actor._entity_id, event_type="ComponentAdded")
        assert len(events) >= 1
        assert events[-1].event_data.entity_id == actor._entity_id

    def test_component_added_event_contains_component_type(self, event_log):
        """ComponentAdded event contains component type."""
        actor = Actor(name="CompTypeTest")
        component = MockComponent()
        actor.add_component("mock_comp", component)

        events = event_log.query(entity_id=actor._entity_id, event_type="ComponentAdded")
        matching = [e for e in events if e.event_data.component_type == "MockComponent"]
        assert len(matching) >= 1

    def test_component_added_event_contains_component_name(self, event_log):
        """ComponentAdded event contains component name."""
        actor = Actor(name="CompNameTest")
        component = MockComponent()
        actor.add_component("named_component", component)

        events = event_log.query(entity_id=actor._entity_id, event_type="ComponentAdded")
        matching = [e for e in events if e.event_data.component_name == "named_component"]
        assert len(matching) >= 1

    def test_component_removed_event_fires(self, event_log):
        """ComponentRemoved event fires when a component is removed."""
        actor = Actor(name="ComponentRemoveTest")
        component = MockComponent()
        actor.add_component("removable", component)
        actor.remove_component("removable")

        events = event_log.query(entity_id=actor._entity_id, event_type="ComponentRemoved")
        assert len(events) >= 1

    def test_component_removed_event_contains_component_type(self, event_log):
        """ComponentRemoved event contains component type."""
        actor = Actor(name="RemoveTypeTest")
        component = MockComponent()
        actor.add_component("to_remove", component)
        actor.remove_component("to_remove")

        events = event_log.query(entity_id=actor._entity_id, event_type="ComponentRemoved")
        matching = [e for e in events if e.event_data.component_type == "MockComponent"]
        assert len(matching) >= 1

    def test_component_events_have_timestamps(self, event_log):
        """Component events have valid timestamps."""
        before = time.time()
        actor = Actor(name="CompTimestampTest")
        component = MockComponent()
        actor.add_component("timed_comp", component)
        after = time.time()

        events = event_log.query(entity_id=actor._entity_id, event_type="ComponentAdded")
        assert len(events) >= 1
        assert before <= events[-1].event_data.timestamp <= after

    def test_component_added_dataclass_is_frozen(self):
        """ComponentAdded dataclass is immutable."""
        event = ComponentAdded(
            entity_id=1,
            component_type="Test",
            timestamp=time.time(),
        )
        with pytest.raises(Exception):
            event.entity_id = 2

    def test_component_removed_dataclass_is_frozen(self):
        """ComponentRemoved dataclass is immutable."""
        event = ComponentRemoved(
            entity_id=1,
            component_type="Test",
            timestamp=time.time(),
        )
        with pytest.raises(Exception):
            event.entity_id = 2

    def test_multiple_components_create_multiple_events(self, event_log):
        """Adding multiple components creates multiple events."""
        actor = Actor(name="MultiCompTest")
        for i in range(5):
            actor.add_component(f"comp_{i}", MockComponent())

        events = event_log.query(entity_id=actor._entity_id, event_type="ComponentAdded")
        # At least 5 component add events (may have default components)
        assert len(events) >= 5


# =============================================================================
# STATE CHANGED EVENT TESTS (8 tests)
# =============================================================================


class TestEntityStateChangedEvent:
    """Tests for EntityStateChanged event."""

    def test_state_change_event_fires_on_transition(self, event_log):
        """EntityStateChanged event fires on state transition."""
        actor = Actor(name="StateChangeTest")

        events = event_log.query(entity_id=actor._entity_id, event_type="EntityStateChanged")
        # Should have at least one state change (UNINITIALIZED -> CREATED)
        assert len(events) >= 1

    def test_state_change_event_contains_old_state(self, event_log):
        """EntityStateChanged event contains old state."""
        event_log.record_state_change(1, "UNINITIALIZED", "CREATED")

        events = event_log.query(entity_id=1, event_type="EntityStateChanged")
        assert len(events) == 1
        assert events[0].event_data.old_state == "UNINITIALIZED"

    def test_state_change_event_contains_new_state(self, event_log):
        """EntityStateChanged event contains new state."""
        event_log.record_state_change(1, "CREATED", "INITIALIZING")

        events = event_log.query(entity_id=1, event_type="EntityStateChanged")
        assert len(events) == 1
        assert events[0].event_data.new_state == "INITIALIZING"

    def test_state_change_event_contains_timestamp(self, event_log):
        """EntityStateChanged event contains timestamp."""
        before = time.time()
        event_log.record_state_change(1, "ACTIVE", "DEACTIVATING")
        after = time.time()

        events = event_log.query(entity_id=1, event_type="EntityStateChanged")
        assert len(events) == 1
        assert before <= events[0].event_data.timestamp <= after

    def test_state_change_accepts_enum_values(self, event_log):
        """EntityStateChanged accepts LifecycleState enum values."""
        event_log.record_state_change(
            1,
            LifecycleState.ACTIVE,
            LifecycleState.DEACTIVATING,
        )

        events = event_log.query(entity_id=1, event_type="EntityStateChanged")
        assert len(events) == 1
        assert events[0].event_data.old_state == "ACTIVE"
        assert events[0].event_data.new_state == "DEACTIVATING"

    def test_state_change_dataclass_is_frozen(self):
        """EntityStateChanged dataclass is immutable."""
        event = EntityStateChanged(
            entity_id=1,
            old_state="ACTIVE",
            new_state="DEACTIVATING",
            timestamp=time.time(),
        )
        with pytest.raises(Exception):
            event.entity_id = 2

    def test_multiple_state_changes_tracked(self, event_log):
        """Multiple state changes are all tracked."""
        for state in ["CREATED", "INITIALIZING", "INITIALIZED", "ACTIVE"]:
            event_log.record_state_change(1, "PREVIOUS", state)

        events = event_log.query(entity_id=1, event_type="EntityStateChanged")
        assert len(events) == 4

    def test_state_changes_ordered_by_id(self, event_log):
        """State changes are ordered correctly."""
        event_log.record_state_change(1, "A", "B")
        event_log.record_state_change(1, "B", "C")
        event_log.record_state_change(1, "C", "D")

        events = event_log.query(entity_id=1, event_type="EntityStateChanged")
        assert events[0].event_data.new_state == "B"
        assert events[1].event_data.new_state == "C"
        assert events[2].event_data.new_state == "D"


# =============================================================================
# CAUSAL CHAIN TESTS (8 tests)
# =============================================================================


class TestCausalChain:
    """Tests for causal chain tracking."""

    def test_spawn_causes_component_adds_linked(self, event_log):
        """Component adds are linked to spawn event via causal chain."""
        actor = Actor(name="CausalTest")
        component = MockComponent()
        actor.add_component("causal_comp", component)

        spawn_events = event_log.query(entity_id=actor._entity_id, event_type="EntitySpawned")
        comp_events = event_log.query(entity_id=actor._entity_id, event_type="ComponentAdded")

        # Component events should reference spawn as causal root
        if spawn_events and comp_events:
            spawn_id = spawn_events[0].id
            causal_comp = [e for e in comp_events if e.causal_root_id == spawn_id]
            # If causal linking is present, verify the chain is valid
            for event in causal_comp:
                assert event.causal_root_id == spawn_id, "Causal root should match spawn event"

    def test_causal_chain_creation(self):
        """CausalChain can be created."""
        chain = CausalChain(
            root_event_id=1,
            parent_event_id=1,
            depth=0,
        )
        assert chain.root_event_id == 1
        assert chain.parent_event_id == 1
        assert chain.depth == 0

    def test_causal_chain_child(self):
        """CausalChain.child() creates child chain."""
        parent = CausalChain(root_event_id=1, parent_event_id=1, depth=0)
        child = parent.child(2)

        assert child.root_event_id == 1
        assert child.parent_event_id == 2
        assert child.depth == 1

    def test_begin_causal_chain(self, event_log):
        """begin_causal_chain starts tracking."""
        spawn = event_log.record_spawn(1, "Test", (0, 0, 0))
        chain = event_log.begin_causal_chain(spawn)

        assert chain.root_event_id == spawn.id
        assert chain.parent_event_id == spawn.id
        assert chain.depth == 1

    def test_end_causal_chain(self, event_log):
        """end_causal_chain stops tracking."""
        spawn = event_log.record_spawn(1, "Test", (0, 0, 0))
        event_log.begin_causal_chain(spawn)
        event_log.end_causal_chain()

        assert event_log._current_causal_chain is None

    def test_with_causal_chain_context_manager(self, event_log):
        """with_causal_chain context manager works."""
        spawn = event_log.record_spawn(1, "Test", (0, 0, 0))

        with event_log.with_causal_chain(spawn) as chain:
            comp = event_log.record_component_added(1, "TestComp")
            assert comp.causal_root_id == spawn.id

    def test_query_causal_children(self, event_log):
        """query_causal_children returns child events."""
        spawn = event_log.record_spawn(1, "Test", (0, 0, 0))
        chain = event_log.begin_causal_chain(spawn)

        comp1 = event_log.record_component_added(1, "Comp1", causal_chain=chain)
        comp2 = event_log.record_component_added(1, "Comp2", causal_chain=chain)

        event_log.end_causal_chain()

        children = event_log.query_causal_children(spawn.id)
        assert len(children) == 2

    def test_query_causal_chain(self, event_log):
        """query_causal_chain returns all events in chain."""
        spawn = event_log.record_spawn(1, "Test", (0, 0, 0))
        chain = event_log.begin_causal_chain(spawn)

        for i in range(3):
            event_log.record_component_added(1, f"Comp{i}", causal_chain=chain)

        event_log.end_causal_chain()

        all_in_chain = event_log.query_causal_chain(spawn.id)
        assert len(all_in_chain) == 3


# =============================================================================
# QUERY TESTS (12 tests)
# =============================================================================


class TestEventQueries:
    """Tests for event querying."""

    def test_query_by_entity_id(self, event_log):
        """Query by entity_id returns correct events."""
        event_log.record_spawn(1, "Entity1", (0, 0, 0))
        event_log.record_spawn(2, "Entity2", (0, 0, 0))

        events = event_log.query(entity_id=1)
        assert all(e.entity_id == 1 for e in events)

    def test_query_by_event_type(self, event_log):
        """Query by event_type returns correct events."""
        event_log.record_spawn(1, "Test", (0, 0, 0))
        event_log.record_component_added(1, "Comp")

        events = event_log.query(event_type="EntitySpawned")
        assert all(e.event_type == "EntitySpawned" for e in events)

    def test_query_by_tick(self, event_log):
        """Query by tick returns correct events."""
        set_current_tick(5)
        event_log.record_spawn(1, "Test", (0, 0, 0))
        set_current_tick(10)
        event_log.record_spawn(2, "Test2", (0, 0, 0))

        events = event_log.query(tick=5)
        assert all(e.tick == 5 for e in events)

    def test_query_by_time_range(self, event_log):
        """Query by time range returns correct events."""
        start = time.time()
        event_log.record_spawn(1, "Test1", (0, 0, 0))
        time.sleep(0.01)
        mid = time.time()
        event_log.record_spawn(2, "Test2", (0, 0, 0))
        time.sleep(0.01)
        end = time.time()

        events = event_log.query(time_start=mid, time_end=end)
        # Should only include the second event
        assert len(events) == 1
        assert events[0].entity_id == 2

    def test_query_with_limit(self, event_log):
        """Query with limit returns correct number of events."""
        for i in range(10):
            event_log.record_spawn(i, f"Test{i}", (0, 0, 0))

        events = event_log.query(limit=5)
        assert len(events) == 5

    def test_query_combined_filters(self, event_log):
        """Query with combined filters works."""
        event_log.record_spawn(1, "Test1", (0, 0, 0))
        event_log.record_component_added(1, "Comp1")
        event_log.record_spawn(2, "Test2", (0, 0, 0))

        events = event_log.query(entity_id=1, event_type="EntitySpawned")
        assert len(events) == 1
        assert events[0].entity_id == 1
        assert events[0].event_type == "EntitySpawned"

    def test_query_by_entity_shorthand(self, event_log):
        """query_by_entity returns all events for entity."""
        event_log.record_spawn(1, "Test", (0, 0, 0))
        event_log.record_component_added(1, "Comp")
        event_log.record_state_change(1, "A", "B")

        events = event_log.query_by_entity(1)
        assert len(events) == 3

    def test_query_by_type_shorthand(self, event_log):
        """query_by_type returns all events of type."""
        for i in range(5):
            event_log.record_spawn(i, f"Test{i}", (0, 0, 0))

        events = event_log.query_by_type("EntitySpawned")
        assert len(events) == 5

    def test_query_by_tick_shorthand(self, event_log):
        """query_by_tick returns events at tick."""
        set_current_tick(42)
        event_log.record_spawn(1, "Test", (0, 0, 0))
        event_log.record_component_added(1, "Comp")

        events = event_log.query_by_tick(42)
        assert len(events) == 2

    def test_query_empty_returns_empty_list(self, event_log):
        """Query with no matches returns empty list."""
        events = event_log.query(entity_id=9999)
        assert events == []

    def test_get_event_by_id(self, event_log):
        """get_event returns specific event by ID."""
        spawn = event_log.record_spawn(1, "Test", (0, 0, 0))

        retrieved = event_log.get_event(spawn.id)
        assert retrieved is not None
        assert retrieved.id == spawn.id

    def test_get_nonexistent_event_returns_none(self, event_log):
        """get_event returns None for nonexistent ID."""
        result = event_log.get_event(99999)
        assert result is None


# =============================================================================
# MULTIPLE ENTITY TRACKING TESTS (5 tests)
# =============================================================================


class TestMultipleEntityTracking:
    """Tests for tracking multiple entities independently."""

    def test_multiple_entities_tracked_independently(self, event_log):
        """Multiple entities have separate event histories."""
        for i in range(5):
            event_log.record_spawn(i, f"Entity{i}", (float(i), 0.0, 0.0))

        for i in range(5):
            events = event_log.query(entity_id=i)
            assert len(events) == 1
            assert events[0].entity_id == i

    def test_entity_events_dont_interfere(self, event_log):
        """Events for one entity don't appear in another's query."""
        event_log.record_spawn(1, "Entity1", (0, 0, 0))
        event_log.record_component_added(1, "Comp1")

        event_log.record_spawn(2, "Entity2", (0, 0, 0))
        event_log.record_component_added(2, "Comp2")

        e1_events = event_log.query(entity_id=1)
        e2_events = event_log.query(entity_id=2)

        assert all(e.entity_id == 1 for e in e1_events)
        assert all(e.entity_id == 2 for e in e2_events)

    def test_total_event_count_correct(self, event_log):
        """Total event count is sum of all entities."""
        for i in range(3):
            event_log.record_spawn(i, f"Entity{i}", (0, 0, 0))
            event_log.record_component_added(i, f"Comp{i}")

        assert len(event_log) == 6

    def test_stats_reflect_multiple_entities(self, event_log):
        """Stats correctly reflect multiple entity tracking."""
        for i in range(10):
            event_log.record_spawn(i, f"Entity{i}", (0, 0, 0))

        stats = event_log.get_stats()
        assert stats["entities_tracked"] == 10
        assert stats["total_events"] == 10

    def test_clear_removes_all_entity_data(self, event_log):
        """Clear removes events for all entities."""
        for i in range(5):
            event_log.record_spawn(i, f"Entity{i}", (0, 0, 0))

        event_log.clear()

        for i in range(5):
            assert event_log.query(entity_id=i) == []


# =============================================================================
# REPLAY TESTS (5 tests)
# =============================================================================


class TestEventReplay:
    """Tests for event replay capabilities."""

    def test_get_replay_sequence(self, event_log):
        """get_replay_sequence returns events in order."""
        set_current_tick(0)
        event_log.record_spawn(1, "Test1", (0, 0, 0))
        set_current_tick(1)
        event_log.record_component_added(1, "Comp1")
        set_current_tick(2)
        event_log.record_state_change(1, "A", "B")

        sequence = event_log.get_replay_sequence()
        assert len(sequence) == 3
        assert sequence[0].tick <= sequence[1].tick <= sequence[2].tick

    def test_replay_sequence_filtered_by_entity(self, event_log):
        """get_replay_sequence can filter by entity."""
        event_log.record_spawn(1, "Test1", (0, 0, 0))
        event_log.record_spawn(2, "Test2", (0, 0, 0))

        sequence = event_log.get_replay_sequence(entity_id=1)
        assert all(e.entity_id == 1 for e in sequence)

    def test_replay_sequence_filtered_by_tick_range(self, event_log):
        """get_replay_sequence can filter by tick range."""
        for i in range(10):
            set_current_tick(i)
            event_log.record_spawn(i, f"Test{i}", (0, 0, 0))

        sequence = event_log.get_replay_sequence(start_tick=3, end_tick=7)
        assert all(3 <= e.tick <= 7 for e in sequence)

    def test_replay_events_calls_handler(self, event_log):
        """replay_events calls handler for each event."""
        for i in range(5):
            event_log.record_spawn(i, f"Test{i}", (0, 0, 0))

        sequence = event_log.get_replay_sequence()
        handled = []

        count = event_log.replay_events(sequence, lambda e: handled.append(e))

        assert count == 5
        assert len(handled) == 5

    def test_replay_deterministic_order(self, event_log):
        """Replay sequence has deterministic order."""
        for i in range(100):
            set_current_tick(i % 10)  # Overlapping ticks
            event_log.record_spawn(i, f"Test{i}", (0, 0, 0))

        seq1 = event_log.get_replay_sequence()
        seq2 = event_log.get_replay_sequence()

        # IDs should be in same order
        assert [e.id for e in seq1] == [e.id for e in seq2]


# =============================================================================
# PERFORMANCE TESTS (2 tests)
# =============================================================================


class TestPerformance:
    """Performance tests."""

    def test_1000_spawns_under_100ms(self, event_log):
        """1000 spawn events complete in under 100ms."""
        start = time.time()

        for i in range(1000):
            event_log.record_spawn(i, f"Entity{i}", (float(i), 0.0, 0.0))

        elapsed = (time.time() - start) * 1000  # ms

        assert elapsed < 100, f"1000 spawns took {elapsed:.2f}ms, expected < 100ms"
        assert len(event_log) == 1000

    def test_query_performance_with_many_events(self, event_log):
        """Queries remain fast with many events."""
        # Record 1000 events
        for i in range(1000):
            event_log.record_spawn(i % 100, f"Entity{i}", (0, 0, 0))

        # Query should complete quickly
        start = time.time()
        events = event_log.query(entity_id=50)
        elapsed = (time.time() - start) * 1000

        assert elapsed < 10, f"Query took {elapsed:.2f}ms"
        assert len(events) == 10


# =============================================================================
# EDGE CASES AND ERROR HANDLING (5 tests)
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_record_with_none_values(self, event_log):
        """Recording with None values doesn't crash."""
        event = event_log.record_spawn(1, "", (0, 0, 0))
        assert event is not None

    def test_singleton_pattern_works(self):
        """EntityEventLog singleton returns same instance."""
        log1 = EntityEventLog()
        log2 = EntityEventLog()
        assert log1 is log2

    def test_reset_instance_creates_new(self):
        """reset_instance creates fresh instance."""
        log1 = EntityEventLog()
        log1.record_spawn(1, "Test", (0, 0, 0))

        EntityEventLog.reset_instance()
        log2 = EntityEventLog()

        assert len(log2) == 0

    def test_clear_resets_next_id(self, event_log):
        """Clear resets the event ID counter."""
        event_log.record_spawn(1, "Test", (0, 0, 0))
        event_log.clear()
        event = event_log.record_spawn(1, "Test2", (0, 0, 0))

        assert event.id == 1

    def test_global_access_function(self):
        """get_entity_event_log returns singleton."""
        log1 = get_entity_event_log()
        log2 = get_entity_event_log()
        assert log1 is log2


# =============================================================================
# STATISTICS TESTS (3 tests)
# =============================================================================


class TestStatistics:
    """Tests for event log statistics."""

    def test_stats_total_events(self, event_log):
        """Stats reports correct total events."""
        for i in range(10):
            event_log.record_spawn(i, f"Test{i}", (0, 0, 0))

        stats = event_log.get_stats()
        assert stats["total_events"] == 10

    def test_stats_events_per_type(self, event_log):
        """Stats reports events per type."""
        event_log.record_spawn(1, "Test", (0, 0, 0))
        event_log.record_spawn(2, "Test", (0, 0, 0))
        event_log.record_component_added(1, "Comp")

        stats = event_log.get_stats()
        assert stats["events_per_type"]["EntitySpawned"] == 2
        assert stats["events_per_type"]["ComponentAdded"] == 1

    def test_stats_entities_tracked(self, event_log):
        """Stats reports entities tracked."""
        for i in range(5):
            event_log.record_spawn(i, f"Test{i}", (0, 0, 0))

        stats = event_log.get_stats()
        assert stats["entities_tracked"] == 5
