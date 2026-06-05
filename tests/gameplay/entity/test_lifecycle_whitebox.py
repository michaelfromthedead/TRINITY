"""
WHITEBOX Tests for Entity Lifecycle System

Tests:
- LifecycleState transitions
- LifecycleManager singleton
- Deferred transitions
- Lifecycle callbacks
- LifecycleMixin functionality
- Lifecycle decorators
"""
import pytest
import threading
from typing import List, Optional, Tuple
from collections import deque

from engine.gameplay.entity.lifecycle import (
    LifecycleEvent,
    LifecycleStateDescriptor,
    LifecycleCallback,
    LifecycleManager,
    LifecycleMixin,
    lifecycle_hook,
    on_spawn,
    begin_play,
    tick,
    end_play,
    on_destroy,
)
from engine.gameplay.entity.constants import (
    LifecycleState,
    VALID_LIFECYCLE_TRANSITIONS,
)
from engine.gameplay.entity.actor import Actor, ActorMeta


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_lifecycle():
    """Reset lifecycle manager before each test."""
    LifecycleManager.reset_instance()
    ActorMeta.clear_registry()
    Actor.reset_entity_ids()
    yield


@pytest.fixture
def lifecycle_manager():
    """Get lifecycle manager instance."""
    return LifecycleManager()


@pytest.fixture
def test_actor():
    """Create a test actor."""
    return Actor(name="TestActor")


# =============================================================================
# LIFECYCLE STATE TESTS
# =============================================================================


class TestLifecycleState:
    """Tests for lifecycle state enum and transitions."""

    def test_all_states_exist(self):
        """All expected lifecycle states should exist."""
        expected_states = [
            "UNINITIALIZED",
            "CREATED",
            "INITIALIZING",
            "INITIALIZED",
            "BEGINNING_PLAY",
            "ACTIVE",
            "DEACTIVATING",
            "DEACTIVATED",
            "DESTROYING",
            "DESTROYED",
        ]
        for state_name in expected_states:
            assert hasattr(LifecycleState, state_name)

    def test_valid_transitions_defined(self):
        """All states should have valid transitions defined."""
        for state in LifecycleState:
            assert state in VALID_LIFECYCLE_TRANSITIONS

    def test_destroyed_is_terminal(self):
        """DESTROYED should have no valid transitions."""
        valid = VALID_LIFECYCLE_TRANSITIONS[LifecycleState.DESTROYED]
        assert len(valid) == 0

    def test_transition_uninitialized_to_created(self):
        """UNINITIALIZED should only transition to CREATED."""
        valid = VALID_LIFECYCLE_TRANSITIONS[LifecycleState.UNINITIALIZED]
        assert LifecycleState.CREATED in valid

    def test_all_states_can_destroy(self):
        """Most states should be able to transition to DESTROYING."""
        destroyable_states = [
            LifecycleState.CREATED,
            LifecycleState.INITIALIZING,
            LifecycleState.INITIALIZED,
            LifecycleState.BEGINNING_PLAY,
            LifecycleState.ACTIVE,
            LifecycleState.DEACTIVATING,
            LifecycleState.DEACTIVATED,
        ]
        for state in destroyable_states:
            valid = VALID_LIFECYCLE_TRANSITIONS[state]
            assert LifecycleState.DESTROYING in valid


# =============================================================================
# LIFECYCLE EVENT TESTS
# =============================================================================


class TestLifecycleEvent:
    """Tests for lifecycle event identifiers."""

    def test_event_names_defined(self):
        """All expected event names should be defined."""
        expected_events = [
            "ON_SPAWN",
            "BEGIN_PLAY",
            "TICK",
            "END_PLAY",
            "ON_DESTROY",
            "ON_ACTIVATE",
            "ON_DEACTIVATE",
            "ON_STATE_CHANGED",
        ]
        for event_name in expected_events:
            assert hasattr(LifecycleEvent, event_name)

    def test_event_values_are_strings(self):
        """Event values should be strings."""
        assert isinstance(LifecycleEvent.ON_SPAWN, str)
        assert isinstance(LifecycleEvent.BEGIN_PLAY, str)


# =============================================================================
# LIFECYCLE STATE DESCRIPTOR TESTS
# =============================================================================


class TestLifecycleStateDescriptor:
    """Tests for LifecycleStateDescriptor."""

    def test_descriptor_id(self):
        """Descriptor should have correct ID."""
        desc = LifecycleStateDescriptor()
        assert desc.descriptor_id == "lifecycle_state"

    def test_pre_set_validates_type(self):
        """pre_set should reject non-LifecycleState values."""
        desc = LifecycleStateDescriptor()

        class MockObj:
            pass

        obj = MockObj()
        with pytest.raises(TypeError, match="Expected LifecycleState"):
            desc.pre_set(obj, "invalid")

    def test_pre_set_validates_transition(self):
        """pre_set should validate state transitions."""
        desc = LifecycleStateDescriptor(validate_transitions=True)

        class MockObj:
            pass

        obj = MockObj()
        # Set initial state (bypass validation by setting directly)
        desc._set_stored(obj, LifecycleState.CREATED)

        # Invalid transition: CREATED -> ACTIVE (should skip intermediate states)
        with pytest.raises(ValueError, match="Invalid state transition"):
            desc.pre_set(obj, LifecycleState.ACTIVE)

    def test_pre_set_allows_valid_transition(self):
        """pre_set should allow valid transitions."""
        desc = LifecycleStateDescriptor(validate_transitions=True)

        class MockObj:
            pass

        obj = MockObj()
        desc._set_stored(obj, LifecycleState.CREATED)

        # Valid transition: CREATED -> INITIALIZING
        result = desc.pre_set(obj, LifecycleState.INITIALIZING)
        assert result == LifecycleState.INITIALIZING

    def test_post_set_tracks_history(self):
        """post_set should track state history."""
        desc = LifecycleStateDescriptor(track_history=True, max_history=5)
        desc._name = "_lifecycle_state"

        class MockObj:
            _lifecycle_callbacks = {}

        obj = MockObj()
        desc.post_set(obj, LifecycleState.CREATED, LifecycleState.UNINITIALIZED)

        history_attr = f"_{desc._name}_history"
        assert hasattr(obj, history_attr)
        history = getattr(obj, history_attr)
        assert len(history) == 1
        assert history[0] == (LifecycleState.UNINITIALIZED, LifecycleState.CREATED)


# =============================================================================
# LIFECYCLE CALLBACK TESTS
# =============================================================================


class TestLifecycleCallback:
    """Tests for LifecycleCallback dataclass."""

    def test_basic_creation(self):
        """Basic callback creation."""
        callback = LifecycleCallback(
            event="begin_play",
            callback=lambda entity: None,
        )
        assert callback.event == "begin_play"
        assert callback.priority == 0
        assert callback.once is False

    def test_with_priority(self):
        """Callback with priority."""
        callback = LifecycleCallback(
            event="tick",
            callback=lambda entity: None,
            priority=10,
        )
        assert callback.priority == 10

    def test_once_flag(self):
        """Callback with once flag."""
        callback = LifecycleCallback(
            event="on_spawn",
            callback=lambda entity: None,
            once=True,
        )
        assert callback.once is True


# =============================================================================
# LIFECYCLE MANAGER TESTS
# =============================================================================


class TestLifecycleManager:
    """Whitebox tests for LifecycleManager."""

    def test_singleton(self, lifecycle_manager):
        """Manager should be singleton."""
        manager2 = LifecycleManager()
        assert lifecycle_manager is manager2

    def test_register_entity(self, lifecycle_manager, test_actor):
        """register_entity should track entity."""
        lifecycle_manager.register_entity(test_actor)
        entities = list(lifecycle_manager._entities.values())
        assert test_actor in entities

    def test_unregister_entity(self, lifecycle_manager, test_actor):
        """unregister_entity should remove entity."""
        lifecycle_manager.register_entity(test_actor)
        lifecycle_manager.unregister_entity(test_actor)
        entities = list(lifecycle_manager._entities.values())
        assert test_actor not in entities

    def test_request_transition_valid(self, lifecycle_manager, test_actor):
        """request_transition should accept valid transitions."""
        lifecycle_manager.register_entity(test_actor)
        test_actor._lifecycle_state = LifecycleState.CREATED

        success = lifecycle_manager.request_transition(
            test_actor,
            LifecycleState.INITIALIZING,
            immediate=False,
        )
        assert success is True
        # Should be pending (not immediate)
        assert len(lifecycle_manager._pending_transitions) > 0

    def test_request_transition_invalid(self, lifecycle_manager, test_actor):
        """request_transition should reject invalid transitions."""
        lifecycle_manager.register_entity(test_actor)
        test_actor._lifecycle_state = LifecycleState.CREATED

        # Invalid: CREATED -> ACTIVE (skips states)
        success = lifecycle_manager.request_transition(
            test_actor,
            LifecycleState.ACTIVE,
            immediate=False,
        )
        assert success is False

    def test_request_transition_immediate(self, lifecycle_manager, test_actor):
        """immediate transition should happen instantly."""
        lifecycle_manager.register_entity(test_actor)
        test_actor._lifecycle_state = LifecycleState.CREATED

        success = lifecycle_manager.request_transition(
            test_actor,
            LifecycleState.INITIALIZING,
            immediate=True,
        )
        assert success is True
        assert test_actor._lifecycle_state == LifecycleState.INITIALIZING

    def test_process_pending_transitions(self, lifecycle_manager, test_actor):
        """process_pending_transitions should apply pending transitions."""
        lifecycle_manager.register_entity(test_actor)
        test_actor._lifecycle_state = LifecycleState.CREATED

        lifecycle_manager.request_transition(test_actor, LifecycleState.INITIALIZING)
        count = lifecycle_manager.process_pending_transitions()

        assert count == 1
        assert test_actor._lifecycle_state == LifecycleState.INITIALIZING

    def test_global_callback_registration(self, lifecycle_manager):
        """register_global_callback should add callback."""
        calls = []
        callback = lambda entity: calls.append(entity)

        lifecycle_manager.register_global_callback("begin_play", callback)
        assert "begin_play" in lifecycle_manager._global_callbacks
        assert callback in lifecycle_manager._global_callbacks["begin_play"]

    def test_global_callback_unregistration(self, lifecycle_manager):
        """unregister_global_callback should remove callback."""
        callback = lambda entity: None
        lifecycle_manager.register_global_callback("begin_play", callback)
        success = lifecycle_manager.unregister_global_callback("begin_play", callback)

        assert success is True
        assert callback not in lifecycle_manager._global_callbacks.get("begin_play", [])

    def test_get_entities_in_state(self, lifecycle_manager):
        """get_entities_in_state should filter by state."""
        actor1 = Actor(name="Actor1")
        actor2 = Actor(name="Actor2")
        actor3 = Actor(name="Actor3")

        lifecycle_manager.register_entity(actor1)
        lifecycle_manager.register_entity(actor2)
        lifecycle_manager.register_entity(actor3)

        actor1._lifecycle_state = LifecycleState.ACTIVE
        actor2._lifecycle_state = LifecycleState.ACTIVE
        actor3._lifecycle_state = LifecycleState.DEACTIVATED

        active = lifecycle_manager.get_entities_in_state(LifecycleState.ACTIVE)
        assert len(active) == 2
        assert actor1 in active
        assert actor2 in active

    def test_get_state_count(self, lifecycle_manager, test_actor):
        """get_state_count should return count for state."""
        lifecycle_manager.register_entity(test_actor)
        count = lifecycle_manager.get_state_count(LifecycleState.CREATED)
        assert count >= 1

    def test_get_stats(self, lifecycle_manager, test_actor):
        """get_stats should return statistics."""
        lifecycle_manager.register_entity(test_actor)
        stats = lifecycle_manager.get_stats()

        assert "total_entities" in stats
        assert "pending_transitions" in stats
        assert "state_counts" in stats
        assert "global_callbacks" in stats

    def test_clear(self, lifecycle_manager, test_actor):
        """clear should reset all state."""
        lifecycle_manager.register_entity(test_actor)
        lifecycle_manager.register_global_callback("test", lambda e: None)

        lifecycle_manager.clear()

        assert len(lifecycle_manager._entities) == 0
        assert len(lifecycle_manager._pending_transitions) == 0
        assert len(lifecycle_manager._global_callbacks) == 0


class TestLifecycleManagerCallbacks:
    """Tests for lifecycle event firing."""

    def test_spawn_event_fires(self, lifecycle_manager):
        """ON_SPAWN should fire on UNINITIALIZED -> CREATED."""
        calls = []

        class SpawnActor(Actor):
            def on_spawn(self):
                calls.append("spawn")

        actor = SpawnActor(name="Spawner")
        lifecycle_manager.register_entity(actor)

        # Simulate spawn event manually
        lifecycle_manager._fire_lifecycle_event(
            actor,
            LifecycleState.UNINITIALIZED,
            LifecycleState.CREATED,
        )
        assert "spawn" in calls

    def test_destroy_event_fires(self, lifecycle_manager, test_actor):
        """ON_DESTROY should fire on transition to DESTROYING."""
        calls = []
        test_actor.on_destroy = lambda: calls.append("destroy")

        lifecycle_manager.register_entity(test_actor)
        lifecycle_manager._fire_lifecycle_event(
            test_actor,
            LifecycleState.ACTIVE,
            LifecycleState.DESTROYING,
        )
        assert "destroy" in calls

    def test_begin_play_event_fires(self, lifecycle_manager, test_actor):
        """BEGIN_PLAY should fire on appropriate transition."""
        calls = []
        test_actor.begin_play = lambda: calls.append("begin")

        lifecycle_manager.register_entity(test_actor)
        lifecycle_manager._fire_lifecycle_event(
            test_actor,
            LifecycleState.INITIALIZED,
            LifecycleState.BEGINNING_PLAY,
        )
        assert "begin" in calls


# =============================================================================
# LIFECYCLE MIXIN TESTS
# =============================================================================


class TestLifecycleMixin:
    """Tests for LifecycleMixin functionality."""

    def test_actor_has_mixin(self, test_actor):
        """Actor should have LifecycleMixin functionality."""
        assert hasattr(test_actor, "_lifecycle_state")
        assert hasattr(test_actor, "_lifecycle_callbacks")
        assert hasattr(test_actor, "transition_to")

    def test_register_lifecycle_callback(self, test_actor):
        """register_lifecycle_callback should add callback."""
        callback = lambda entity: None
        test_actor.register_lifecycle_callback("begin_play", callback)

        assert "begin_play" in test_actor._lifecycle_callbacks
        callbacks = test_actor._lifecycle_callbacks["begin_play"]
        assert any(cb.callback == callback for cb in callbacks)

    def test_unregister_lifecycle_callback(self, test_actor):
        """unregister_lifecycle_callback should remove callback."""
        callback = lambda entity: None
        test_actor.register_lifecycle_callback("begin_play", callback)
        success = test_actor.unregister_lifecycle_callback("begin_play", callback)

        assert success is True

    def test_transition_to(self, test_actor, lifecycle_manager):
        """transition_to should request transition."""
        lifecycle_manager.register_entity(test_actor)
        test_actor._lifecycle_state = LifecycleState.CREATED

        success = test_actor.transition_to(LifecycleState.INITIALIZING)
        assert success is True

    def test_get_lifecycle_state(self, test_actor):
        """get_lifecycle_state should return current state."""
        assert test_actor.get_lifecycle_state() == LifecycleState.CREATED

    def test_is_active(self, test_actor):
        """is_active should check ACTIVE state."""
        assert test_actor.is_active() is False
        test_actor._lifecycle_state = LifecycleState.ACTIVE
        assert test_actor.is_active() is True

    def test_is_destroyed(self, test_actor):
        """is_destroyed should check DESTROYING/DESTROYED states."""
        assert test_actor.is_destroyed() is False
        test_actor._lifecycle_state = LifecycleState.DESTROYING
        assert test_actor.is_destroyed() is True
        test_actor._lifecycle_state = LifecycleState.DESTROYED
        assert test_actor.is_destroyed() is True


# =============================================================================
# LIFECYCLE DECORATOR TESTS
# =============================================================================


class TestLifecycleDecorators:
    """Tests for lifecycle decorator functions."""

    def test_lifecycle_hook_decorator(self):
        """@lifecycle_hook should add metadata to method."""
        @lifecycle_hook(event=LifecycleEvent.BEGIN_PLAY, priority=5)
        def my_method(self):
            pass

        assert hasattr(my_method, "_lifecycle_hook")
        assert my_method._lifecycle_hook is True
        assert my_method._lifecycle_event == LifecycleEvent.BEGIN_PLAY
        assert my_method._lifecycle_priority == 5

    def test_on_spawn_decorator(self):
        """@on_spawn should set ON_SPAWN event."""
        @on_spawn
        def spawn_handler(self):
            pass

        assert spawn_handler._lifecycle_event == LifecycleEvent.ON_SPAWN

    def test_begin_play_decorator(self):
        """@begin_play should set BEGIN_PLAY event."""
        @begin_play
        def play_handler(self):
            pass

        assert play_handler._lifecycle_event == LifecycleEvent.BEGIN_PLAY

    def test_tick_decorator(self):
        """@tick should set TICK event."""
        @tick
        def tick_handler(self):
            pass

        assert tick_handler._lifecycle_event == LifecycleEvent.TICK

    def test_end_play_decorator(self):
        """@end_play should set END_PLAY event."""
        @end_play
        def end_handler(self):
            pass

        assert end_handler._lifecycle_event == LifecycleEvent.END_PLAY

    def test_on_destroy_decorator(self):
        """@on_destroy should set ON_DESTROY event."""
        @on_destroy
        def destroy_handler(self):
            pass

        assert destroy_handler._lifecycle_event == LifecycleEvent.ON_DESTROY


class TestLifecycleHookCollection:
    """Tests for automatic hook collection in subclasses."""

    def test_hooks_collected_from_class(self):
        """Decorated methods should be collected automatically."""
        class HookedActor(Actor):
            @begin_play
            def custom_begin_play(self):
                pass

        assert "begin_play" in getattr(HookedActor, "_collected_lifecycle_hooks", {})

    def test_multiple_hooks_collected(self):
        """Multiple hooks should be collected."""
        class MultiHookedActor(Actor):
            @on_spawn
            def spawn_handler(self):
                pass

            @begin_play
            def play_handler(self):
                pass

            @on_destroy
            def destroy_handler(self):
                pass

        hooks = getattr(MultiHookedActor, "_collected_lifecycle_hooks", {})
        assert LifecycleEvent.ON_SPAWN in hooks
        assert LifecycleEvent.BEGIN_PLAY in hooks
        assert LifecycleEvent.ON_DESTROY in hooks


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================


class TestLifecycleManagerThreadSafety:
    """Tests for thread safety of lifecycle manager."""

    def test_concurrent_transitions(self, lifecycle_manager):
        """Concurrent transition requests should be safe."""
        actors = [Actor(name=f"Actor{i}") for i in range(100)]
        for actor in actors:
            lifecycle_manager.register_entity(actor)
            actor._lifecycle_state = LifecycleState.CREATED

        results = []

        def request_transitions():
            for actor in actors:
                success = lifecycle_manager.request_transition(
                    actor,
                    LifecycleState.INITIALIZING,
                )
                results.append(success)

        threads = [
            threading.Thread(target=request_transitions)
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All requests should succeed
        assert all(results)

    def test_concurrent_process_pending(self, lifecycle_manager):
        """Concurrent process_pending calls should be safe."""
        actors = [Actor(name=f"Actor{i}") for i in range(50)]
        for actor in actors:
            lifecycle_manager.register_entity(actor)
            actor._lifecycle_state = LifecycleState.CREATED
            lifecycle_manager.request_transition(actor, LifecycleState.INITIALIZING)

        counts = []

        def process():
            count = lifecycle_manager.process_pending_transitions()
            counts.append(count)

        threads = [
            threading.Thread(target=process)
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total processed should equal actors
        assert sum(counts) == 50


# =============================================================================
# STATE TRANSITION SEQUENCE TESTS
# =============================================================================


class TestStateTransitionSequences:
    """Tests for valid state transition sequences."""

    def test_full_lifecycle_sequence(self, lifecycle_manager):
        """Test complete lifecycle from creation to destruction."""
        actor = Actor(name="FullLifecycle")
        lifecycle_manager.register_entity(actor)

        # CREATED -> INITIALIZING -> INITIALIZED -> BEGINNING_PLAY -> ACTIVE
        transitions = [
            LifecycleState.INITIALIZING,
            LifecycleState.INITIALIZED,
            LifecycleState.BEGINNING_PLAY,
            LifecycleState.ACTIVE,
        ]

        for target in transitions:
            success = lifecycle_manager.request_transition(actor, target, immediate=True)
            assert success is True
            assert actor._lifecycle_state == target

    def test_deactivation_sequence(self, lifecycle_manager):
        """Test deactivation sequence."""
        actor = Actor(name="Deactivating")
        lifecycle_manager.register_entity(actor)
        actor._lifecycle_state = LifecycleState.ACTIVE

        transitions = [
            LifecycleState.DEACTIVATING,
            LifecycleState.DEACTIVATED,
        ]

        for target in transitions:
            success = lifecycle_manager.request_transition(actor, target, immediate=True)
            assert success is True

        assert actor._lifecycle_state == LifecycleState.DEACTIVATED

    def test_reactivation_sequence(self, lifecycle_manager):
        """Test reactivation from deactivated state."""
        actor = Actor(name="Reactivating")
        lifecycle_manager.register_entity(actor)
        actor._lifecycle_state = LifecycleState.DEACTIVATED

        # DEACTIVATED -> BEGINNING_PLAY -> ACTIVE
        transitions = [
            LifecycleState.BEGINNING_PLAY,
            LifecycleState.ACTIVE,
        ]

        for target in transitions:
            success = lifecycle_manager.request_transition(actor, target, immediate=True)
            assert success is True

        assert actor._lifecycle_state == LifecycleState.ACTIVE

    def test_destruction_from_any_state(self, lifecycle_manager):
        """Test destruction from various states."""
        destroyable_states = [
            LifecycleState.CREATED,
            LifecycleState.INITIALIZING,
            LifecycleState.INITIALIZED,
            LifecycleState.BEGINNING_PLAY,
            LifecycleState.ACTIVE,
            LifecycleState.DEACTIVATING,
            LifecycleState.DEACTIVATED,
        ]

        for initial_state in destroyable_states:
            actor = Actor(name=f"Destroy_{initial_state.name}")
            lifecycle_manager.register_entity(actor)
            actor._lifecycle_state = initial_state

            success = lifecycle_manager.request_transition(
                actor,
                LifecycleState.DESTROYING,
                immediate=True,
            )
            assert success is True, f"Failed to destroy from {initial_state.name}"
            assert actor._lifecycle_state == LifecycleState.DESTROYING
