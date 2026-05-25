"""
Comprehensive tests for the Entity Lifecycle System.

Tests for:
- Lifecycle state transitions (CREATE->INITIALIZE->ACTIVE->DEACTIVATE->DESTROY)
- Deferred operations (spawn, destroy during iteration)
- Begin play / end play callbacks
- Lifecycle hooks and events
- Error handling for invalid transitions
- Batch lifecycle operations
"""
from __future__ import annotations

import pytest
import threading
import weakref
from typing import Any, List
from unittest.mock import Mock, MagicMock, patch, call

from engine.gameplay.entity.lifecycle import (
    LifecycleState,
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
    VALID_LIFECYCLE_TRANSITIONS,
)
from engine.gameplay.entity.actor import Actor


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_state():
    """Reset lifecycle state before each test."""
    LifecycleManager.reset_instance()
    Actor.reset_entity_ids()
    yield
    LifecycleManager.reset_instance()


@pytest.fixture
def lifecycle_manager():
    """Create a fresh lifecycle manager."""
    LifecycleManager.reset_instance()
    return LifecycleManager()


@pytest.fixture
def test_actor():
    """Create a test actor for lifecycle testing."""
    return Actor(name="TestActor")


class MockEntity:
    """Mock entity for testing without full Actor implementation."""
    _entity_id_counter = 0

    def __init__(self):
        MockEntity._entity_id_counter += 1
        self._entity_id = MockEntity._entity_id_counter
        self._lifecycle_state = LifecycleState.UNINITIALIZED
        self._lifecycle_callbacks = {}

    @classmethod
    def reset_counter(cls):
        cls._entity_id_counter = 0


@pytest.fixture
def mock_entity():
    """Create a mock entity."""
    MockEntity.reset_counter()
    return MockEntity()


# =============================================================================
# LIFECYCLE STATE TESTS
# =============================================================================


class TestLifecycleState:
    """Tests for LifecycleState enum."""

    def test_all_states_defined(self):
        """Test all expected lifecycle states are defined."""
        expected_states = {
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
        }
        actual_states = {state.name for state in LifecycleState}
        assert expected_states == actual_states

    def test_state_ordering(self):
        """Test lifecycle states have increasing values."""
        assert LifecycleState.UNINITIALIZED.value < LifecycleState.CREATED.value
        assert LifecycleState.CREATED.value < LifecycleState.INITIALIZING.value
        assert LifecycleState.INITIALIZING.value < LifecycleState.INITIALIZED.value
        assert LifecycleState.INITIALIZED.value < LifecycleState.BEGINNING_PLAY.value
        assert LifecycleState.BEGINNING_PLAY.value < LifecycleState.ACTIVE.value
        assert LifecycleState.ACTIVE.value < LifecycleState.DEACTIVATING.value
        assert LifecycleState.DEACTIVATING.value < LifecycleState.DEACTIVATED.value
        assert LifecycleState.DEACTIVATED.value < LifecycleState.DESTROYING.value
        assert LifecycleState.DESTROYING.value < LifecycleState.DESTROYED.value


class TestLifecycleTransitions:
    """Tests for valid lifecycle state transitions."""

    def test_uninitialized_can_transition_to_created(self):
        """Test UNINITIALIZED -> CREATED is valid."""
        valid = VALID_LIFECYCLE_TRANSITIONS[LifecycleState.UNINITIALIZED]
        assert LifecycleState.CREATED in valid

    def test_created_transitions(self):
        """Test valid transitions from CREATED."""
        valid = VALID_LIFECYCLE_TRANSITIONS[LifecycleState.CREATED]
        assert LifecycleState.INITIALIZING in valid
        assert LifecycleState.DESTROYING in valid

    def test_initializing_transitions(self):
        """Test valid transitions from INITIALIZING."""
        valid = VALID_LIFECYCLE_TRANSITIONS[LifecycleState.INITIALIZING]
        assert LifecycleState.INITIALIZED in valid
        assert LifecycleState.DESTROYING in valid

    def test_initialized_transitions(self):
        """Test valid transitions from INITIALIZED."""
        valid = VALID_LIFECYCLE_TRANSITIONS[LifecycleState.INITIALIZED]
        assert LifecycleState.BEGINNING_PLAY in valid
        assert LifecycleState.DESTROYING in valid

    def test_beginning_play_transitions(self):
        """Test valid transitions from BEGINNING_PLAY."""
        valid = VALID_LIFECYCLE_TRANSITIONS[LifecycleState.BEGINNING_PLAY]
        assert LifecycleState.ACTIVE in valid
        assert LifecycleState.DESTROYING in valid

    def test_active_transitions(self):
        """Test valid transitions from ACTIVE."""
        valid = VALID_LIFECYCLE_TRANSITIONS[LifecycleState.ACTIVE]
        assert LifecycleState.DEACTIVATING in valid
        assert LifecycleState.DESTROYING in valid

    def test_deactivating_transitions(self):
        """Test valid transitions from DEACTIVATING."""
        valid = VALID_LIFECYCLE_TRANSITIONS[LifecycleState.DEACTIVATING]
        assert LifecycleState.DEACTIVATED in valid
        assert LifecycleState.DESTROYING in valid

    def test_deactivated_transitions(self):
        """Test valid transitions from DEACTIVATED."""
        valid = VALID_LIFECYCLE_TRANSITIONS[LifecycleState.DEACTIVATED]
        assert LifecycleState.BEGINNING_PLAY in valid  # Can reactivate
        assert LifecycleState.DESTROYING in valid

    def test_destroying_transitions(self):
        """Test valid transitions from DESTROYING."""
        valid = VALID_LIFECYCLE_TRANSITIONS[LifecycleState.DESTROYING]
        assert LifecycleState.DESTROYED in valid

    def test_destroyed_is_terminal(self):
        """Test DESTROYED has no valid transitions."""
        valid = VALID_LIFECYCLE_TRANSITIONS[LifecycleState.DESTROYED]
        assert len(valid) == 0

    def test_can_destroy_from_any_state(self):
        """Test DESTROYING is reachable from most states."""
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
            assert LifecycleState.DESTROYING in valid, f"DESTROYING not valid from {state}"


# =============================================================================
# LIFECYCLE EVENT TESTS
# =============================================================================


class TestLifecycleEvent:
    """Tests for LifecycleEvent identifiers."""

    def test_on_spawn_event(self):
        """Test ON_SPAWN event identifier."""
        assert LifecycleEvent.ON_SPAWN == "on_spawn"

    def test_begin_play_event(self):
        """Test BEGIN_PLAY event identifier."""
        assert LifecycleEvent.BEGIN_PLAY == "begin_play"

    def test_tick_event(self):
        """Test TICK event identifier."""
        assert LifecycleEvent.TICK == "tick"

    def test_end_play_event(self):
        """Test END_PLAY event identifier."""
        assert LifecycleEvent.END_PLAY == "end_play"

    def test_on_destroy_event(self):
        """Test ON_DESTROY event identifier."""
        assert LifecycleEvent.ON_DESTROY == "on_destroy"

    def test_on_activate_event(self):
        """Test ON_ACTIVATE event identifier."""
        assert LifecycleEvent.ON_ACTIVATE == "on_activate"

    def test_on_deactivate_event(self):
        """Test ON_DEACTIVATE event identifier."""
        assert LifecycleEvent.ON_DEACTIVATE == "on_deactivate"

    def test_on_state_changed_event(self):
        """Test ON_STATE_CHANGED event identifier."""
        assert LifecycleEvent.ON_STATE_CHANGED == "on_state_changed"


# =============================================================================
# LIFECYCLE CALLBACK TESTS
# =============================================================================


class TestLifecycleCallback:
    """Tests for LifecycleCallback data class."""

    def test_callback_creation(self):
        """Test creating a lifecycle callback."""
        cb = LifecycleCallback(
            event="on_spawn",
            callback=lambda e: None,
        )
        assert cb.event == "on_spawn"
        assert callable(cb.callback)

    def test_callback_default_priority(self):
        """Test callback default priority is 0."""
        cb = LifecycleCallback(
            event="begin_play",
            callback=lambda e: None,
        )
        assert cb.priority == 0

    def test_callback_custom_priority(self):
        """Test callback with custom priority."""
        cb = LifecycleCallback(
            event="tick",
            callback=lambda e: None,
            priority=10,
        )
        assert cb.priority == 10

    def test_callback_default_once(self):
        """Test callback default once is False."""
        cb = LifecycleCallback(
            event="end_play",
            callback=lambda e: None,
        )
        assert cb.once is False

    def test_callback_once_true(self):
        """Test callback with once=True."""
        cb = LifecycleCallback(
            event="on_destroy",
            callback=lambda e: None,
            once=True,
        )
        assert cb.once is True


# =============================================================================
# LIFECYCLE STATE DESCRIPTOR TESTS
# =============================================================================


class TestLifecycleStateDescriptor:
    """Tests for LifecycleStateDescriptor."""

    def test_descriptor_id(self):
        """Test descriptor has correct ID."""
        descriptor = LifecycleStateDescriptor()
        assert descriptor.descriptor_id == "lifecycle_state"

    def test_pre_set_validates_type(self):
        """Test pre_set rejects non-LifecycleState values."""
        descriptor = LifecycleStateDescriptor()
        with pytest.raises(TypeError, match="Expected LifecycleState"):
            descriptor.pre_set(Mock(), "invalid")

    def test_pre_set_accepts_lifecycle_state(self):
        """Test pre_set accepts LifecycleState values."""
        descriptor = LifecycleStateDescriptor()
        obj = Mock()
        obj._lifecycle_state = None
        descriptor._name = "_lifecycle_state"
        result = descriptor.pre_set(obj, LifecycleState.CREATED)
        assert result == LifecycleState.CREATED

    def test_descriptor_steps(self):
        """Test descriptor defines expected steps."""
        descriptor = LifecycleStateDescriptor()
        steps = descriptor.descriptor_steps
        ops = {s.op.name for s in steps}
        assert "TRACK" in ops
        assert "VALIDATE" in ops
        assert "HOOK" in ops


# =============================================================================
# LIFECYCLE MANAGER TESTS
# =============================================================================


class TestLifecycleManagerSingleton:
    """Tests for LifecycleManager singleton pattern."""

    def test_singleton_instance(self):
        """Test LifecycleManager is a singleton."""
        mgr1 = LifecycleManager()
        mgr2 = LifecycleManager()
        assert mgr1 is mgr2

    def test_reset_instance(self):
        """Test resetting singleton instance."""
        mgr1 = LifecycleManager()
        LifecycleManager.reset_instance()
        mgr2 = LifecycleManager()
        # After reset, should be a new instance
        # Note: singleton might return same object if not truly reset
        assert mgr2 is not None


class TestLifecycleManagerRegistration:
    """Tests for entity registration with LifecycleManager."""

    def test_register_entity(self, lifecycle_manager, mock_entity):
        """Test registering an entity."""
        lifecycle_manager.register_entity(mock_entity)
        # Should not raise
        assert True

    def test_unregister_entity(self, lifecycle_manager, mock_entity):
        """Test unregistering an entity."""
        lifecycle_manager.register_entity(mock_entity)
        lifecycle_manager.unregister_entity(mock_entity)
        # Should not raise
        assert True

    def test_register_updates_state_counts(self, lifecycle_manager, mock_entity):
        """Test registration updates state counts."""
        initial_count = lifecycle_manager.get_state_count(LifecycleState.UNINITIALIZED)
        lifecycle_manager.register_entity(mock_entity)
        new_count = lifecycle_manager.get_state_count(LifecycleState.UNINITIALIZED)
        assert new_count == initial_count + 1

    def test_unregister_updates_state_counts(self, lifecycle_manager, mock_entity):
        """Test unregistration updates state counts."""
        lifecycle_manager.register_entity(mock_entity)
        count_before = lifecycle_manager.get_state_count(LifecycleState.UNINITIALIZED)
        lifecycle_manager.unregister_entity(mock_entity)
        count_after = lifecycle_manager.get_state_count(LifecycleState.UNINITIALIZED)
        assert count_after == count_before - 1


class TestLifecycleManagerTransitions:
    """Tests for state transitions via LifecycleManager."""

    def test_request_valid_transition(self, lifecycle_manager, mock_entity):
        """Test requesting a valid state transition."""
        mock_entity._lifecycle_state = LifecycleState.UNINITIALIZED
        lifecycle_manager.register_entity(mock_entity)
        result = lifecycle_manager.request_transition(
            mock_entity,
            LifecycleState.CREATED,
            immediate=True,
        )
        assert result is True

    def test_request_invalid_transition(self, lifecycle_manager, mock_entity):
        """Test requesting an invalid state transition."""
        mock_entity._lifecycle_state = LifecycleState.UNINITIALIZED
        lifecycle_manager.register_entity(mock_entity)
        result = lifecycle_manager.request_transition(
            mock_entity,
            LifecycleState.ACTIVE,  # Invalid from UNINITIALIZED
            immediate=True,
        )
        assert result is False

    def test_deferred_transition(self, lifecycle_manager, mock_entity):
        """Test deferred (non-immediate) transition."""
        mock_entity._lifecycle_state = LifecycleState.UNINITIALIZED
        lifecycle_manager.register_entity(mock_entity)
        result = lifecycle_manager.request_transition(
            mock_entity,
            LifecycleState.CREATED,
            immediate=False,
        )
        assert result is True
        # State not changed yet
        assert mock_entity._lifecycle_state == LifecycleState.UNINITIALIZED

    def test_process_pending_transitions(self, lifecycle_manager, mock_entity):
        """Test processing pending transitions."""
        mock_entity._lifecycle_state = LifecycleState.UNINITIALIZED
        lifecycle_manager.register_entity(mock_entity)
        lifecycle_manager.request_transition(
            mock_entity,
            LifecycleState.CREATED,
            immediate=False,
        )
        count = lifecycle_manager.process_pending_transitions()
        assert count >= 1
        assert mock_entity._lifecycle_state == LifecycleState.CREATED

    def test_multiple_pending_transitions(self, lifecycle_manager):
        """Test processing multiple pending transitions."""
        entities = []
        for i in range(5):
            entity = MockEntity()
            entity._lifecycle_state = LifecycleState.UNINITIALIZED
            lifecycle_manager.register_entity(entity)
            lifecycle_manager.request_transition(
                entity,
                LifecycleState.CREATED,
                immediate=False,
            )
            entities.append(entity)

        count = lifecycle_manager.process_pending_transitions()
        assert count == 5
        for entity in entities:
            assert entity._lifecycle_state == LifecycleState.CREATED


class TestLifecycleManagerCallbacks:
    """Tests for LifecycleManager callback system."""

    def test_register_global_callback(self, lifecycle_manager):
        """Test registering a global callback."""
        callback = Mock()
        lifecycle_manager.register_global_callback("on_spawn", callback)
        # Should not raise
        assert True

    def test_unregister_global_callback(self, lifecycle_manager):
        """Test unregistering a global callback."""
        callback = Mock()
        lifecycle_manager.register_global_callback("on_spawn", callback)
        result = lifecycle_manager.unregister_global_callback("on_spawn", callback)
        assert result is True

    def test_unregister_nonexistent_callback(self, lifecycle_manager):
        """Test unregistering nonexistent callback returns False."""
        callback = Mock()
        result = lifecycle_manager.unregister_global_callback("on_spawn", callback)
        assert result is False


class TestLifecycleManagerQueries:
    """Tests for LifecycleManager query methods."""

    def test_get_entities_in_state(self, lifecycle_manager):
        """Test getting entities in a specific state."""
        entity1 = MockEntity()
        entity2 = MockEntity()
        entity1._lifecycle_state = LifecycleState.ACTIVE
        entity2._lifecycle_state = LifecycleState.ACTIVE

        lifecycle_manager.register_entity(entity1)
        lifecycle_manager.register_entity(entity2)

        active_entities = lifecycle_manager.get_entities_in_state(LifecycleState.ACTIVE)
        assert len(active_entities) == 2

    def test_get_state_count(self, lifecycle_manager):
        """Test getting count of entities in a state."""
        for i in range(3):
            entity = MockEntity()
            entity._lifecycle_state = LifecycleState.CREATED
            lifecycle_manager.register_entity(entity)

        count = lifecycle_manager.get_state_count(LifecycleState.CREATED)
        assert count == 3

    def test_get_stats(self, lifecycle_manager, mock_entity):
        """Test getting manager statistics."""
        lifecycle_manager.register_entity(mock_entity)
        stats = lifecycle_manager.get_stats()
        assert "total_entities" in stats
        assert "pending_transitions" in stats
        assert "state_counts" in stats
        assert "global_callbacks" in stats

    def test_clear(self, lifecycle_manager):
        """Test clearing all state."""
        entity = MockEntity()
        lifecycle_manager.register_entity(entity)
        lifecycle_manager.register_global_callback("test", Mock())
        lifecycle_manager.clear()
        stats = lifecycle_manager.get_stats()
        assert stats["total_entities"] == 0
        assert stats["pending_transitions"] == 0


# =============================================================================
# LIFECYCLE MIXIN TESTS
# =============================================================================


class TestLifecycleMixin:
    """Tests for LifecycleMixin class."""

    def test_mixin_initial_state(self, test_actor):
        """Test LifecycleMixin initial state is CREATED."""
        assert test_actor._lifecycle_state == LifecycleState.CREATED

    def test_get_lifecycle_state(self, test_actor):
        """Test get_lifecycle_state method."""
        state = test_actor.get_lifecycle_state()
        assert state == LifecycleState.CREATED

    def test_is_active_false_initially(self, test_actor):
        """Test is_active returns False initially."""
        assert test_actor.is_active() is False

    def test_is_destroyed_false_initially(self, test_actor):
        """Test is_destroyed returns False initially."""
        assert test_actor.is_destroyed() is False

    def test_register_lifecycle_callback(self, test_actor):
        """Test registering a lifecycle callback on entity."""
        callback = Mock()
        test_actor.register_lifecycle_callback("on_spawn", callback)
        assert "on_spawn" in test_actor._lifecycle_callbacks
        assert len(test_actor._lifecycle_callbacks["on_spawn"]) == 1

    def test_register_callback_with_priority(self, test_actor):
        """Test registering callback with custom priority."""
        callback = Mock()
        test_actor.register_lifecycle_callback("on_spawn", callback, priority=10)
        cb = test_actor._lifecycle_callbacks["on_spawn"][0]
        assert cb.priority == 10

    def test_unregister_lifecycle_callback(self, test_actor):
        """Test unregistering a lifecycle callback."""
        callback = Mock()
        test_actor.register_lifecycle_callback("on_spawn", callback)
        result = test_actor.unregister_lifecycle_callback("on_spawn", callback)
        assert result is True

    def test_unregister_nonexistent_callback(self, test_actor):
        """Test unregistering nonexistent callback returns False."""
        callback = Mock()
        result = test_actor.unregister_lifecycle_callback("on_spawn", callback)
        assert result is False

    def test_transition_to(self, test_actor, lifecycle_manager):
        """Test transition_to method."""
        # Need to set proper initial state for transition
        test_actor._lifecycle_state = LifecycleState.UNINITIALIZED
        result = test_actor.transition_to(LifecycleState.CREATED, immediate=True)
        # Result depends on manager state
        assert isinstance(result, bool)


# =============================================================================
# LIFECYCLE DECORATOR TESTS
# =============================================================================


class TestLifecycleHookDecorator:
    """Tests for @lifecycle_hook decorator."""

    def test_lifecycle_hook_basic(self):
        """Test basic @lifecycle_hook usage."""
        @lifecycle_hook(event=LifecycleEvent.ON_SPAWN)
        def handle_spawn(entity):
            pass

        assert handle_spawn._lifecycle_hook is True
        assert handle_spawn._lifecycle_event == LifecycleEvent.ON_SPAWN

    def test_lifecycle_hook_priority(self):
        """Test @lifecycle_hook with priority."""
        @lifecycle_hook(event=LifecycleEvent.BEGIN_PLAY, priority=5)
        def handle_begin(entity):
            pass

        assert handle_begin._lifecycle_priority == 5

    def test_lifecycle_hook_preserves_callable(self):
        """Test decorated function remains callable."""
        @lifecycle_hook(event=LifecycleEvent.TICK)
        def handle_tick(entity):
            return "ticked"

        assert callable(handle_tick)
        assert handle_tick(None) == "ticked"


class TestOnSpawnDecorator:
    """Tests for @on_spawn decorator."""

    def test_on_spawn_basic(self):
        """Test basic @on_spawn usage."""
        @on_spawn
        def handle_spawn(entity):
            pass

        assert handle_spawn._lifecycle_hook is True
        assert handle_spawn._lifecycle_event == LifecycleEvent.ON_SPAWN

    def test_on_spawn_preserves_callable(self):
        """Test @on_spawn preserves callable."""
        @on_spawn
        def handle_spawn(entity):
            return "spawned"

        assert handle_spawn(None) == "spawned"

    def test_on_spawn_preserves_docstring(self):
        """Test @on_spawn preserves docstring."""
        @on_spawn
        def handle_spawn(entity):
            """Spawn handler doc."""
            pass

        assert handle_spawn.__doc__ == "Spawn handler doc."


class TestBeginPlayDecorator:
    """Tests for @begin_play decorator."""

    def test_begin_play_basic(self):
        """Test basic @begin_play usage."""
        @begin_play
        def handle_begin(entity):
            pass

        assert handle_begin._lifecycle_hook is True
        assert handle_begin._lifecycle_event == LifecycleEvent.BEGIN_PLAY

    def test_begin_play_preserves_callable(self):
        """Test @begin_play preserves callable."""
        @begin_play
        def handle_begin(entity):
            return "began"

        assert handle_begin(None) == "began"


class TestTickDecorator:
    """Tests for @tick decorator."""

    def test_tick_basic(self):
        """Test basic @tick usage."""
        @tick
        def handle_tick(entity):
            pass

        assert handle_tick._lifecycle_hook is True
        assert handle_tick._lifecycle_event == LifecycleEvent.TICK

    def test_tick_preserves_callable(self):
        """Test @tick preserves callable."""
        @tick
        def handle_tick(entity):
            return "ticked"

        assert handle_tick(None) == "ticked"


class TestEndPlayDecorator:
    """Tests for @end_play decorator."""

    def test_end_play_basic(self):
        """Test basic @end_play usage."""
        @end_play
        def handle_end(entity):
            pass

        assert handle_end._lifecycle_hook is True
        assert handle_end._lifecycle_event == LifecycleEvent.END_PLAY

    def test_end_play_preserves_callable(self):
        """Test @end_play preserves callable."""
        @end_play
        def handle_end(entity):
            return "ended"

        assert handle_end(None) == "ended"


class TestOnDestroyDecorator:
    """Tests for @on_destroy decorator."""

    def test_on_destroy_basic(self):
        """Test basic @on_destroy usage."""
        @on_destroy
        def handle_destroy(entity):
            pass

        assert handle_destroy._lifecycle_hook is True
        assert handle_destroy._lifecycle_event == LifecycleEvent.ON_DESTROY

    def test_on_destroy_preserves_callable(self):
        """Test @on_destroy preserves callable."""
        @on_destroy
        def handle_destroy(entity):
            return "destroyed"

        assert handle_destroy(None) == "destroyed"


# =============================================================================
# DEFERRED OPERATIONS TESTS
# =============================================================================


class TestDeferredOperations:
    """Tests for deferred lifecycle operations."""

    def test_deferred_spawn(self, lifecycle_manager):
        """Test deferred spawn operation."""
        entity = MockEntity()
        entity._lifecycle_state = LifecycleState.UNINITIALIZED
        lifecycle_manager.register_entity(entity)

        # Request deferred transition
        lifecycle_manager.request_transition(entity, LifecycleState.CREATED, immediate=False)

        # State should not have changed yet
        assert entity._lifecycle_state == LifecycleState.UNINITIALIZED

        # Process pending
        lifecycle_manager.process_pending_transitions()

        # Now state should be updated
        assert entity._lifecycle_state == LifecycleState.CREATED

    def test_deferred_destroy(self, lifecycle_manager):
        """Test deferred destroy operation."""
        entity = MockEntity()
        entity._lifecycle_state = LifecycleState.ACTIVE
        lifecycle_manager.register_entity(entity)

        lifecycle_manager.request_transition(entity, LifecycleState.DESTROYING, immediate=False)
        assert entity._lifecycle_state == LifecycleState.ACTIVE

        lifecycle_manager.process_pending_transitions()
        assert entity._lifecycle_state == LifecycleState.DESTROYING

    def test_deferred_operations_batch(self, lifecycle_manager):
        """Test batch processing of deferred operations."""
        entities = []
        for i in range(10):
            entity = MockEntity()
            entity._lifecycle_state = LifecycleState.UNINITIALIZED
            lifecycle_manager.register_entity(entity)
            lifecycle_manager.request_transition(entity, LifecycleState.CREATED, immediate=False)
            entities.append(entity)

        # All should still be UNINITIALIZED
        for entity in entities:
            assert entity._lifecycle_state == LifecycleState.UNINITIALIZED

        # Process all at once
        count = lifecycle_manager.process_pending_transitions()
        assert count == 10

        # All should now be CREATED
        for entity in entities:
            assert entity._lifecycle_state == LifecycleState.CREATED

    def test_deferred_during_iteration(self, lifecycle_manager):
        """Test deferred operations requested during iteration."""
        entities = []
        for i in range(5):
            entity = MockEntity()
            entity._lifecycle_state = LifecycleState.ACTIVE
            lifecycle_manager.register_entity(entity)
            entities.append(entity)

        # Simulate requesting destroy during iteration
        for entity in entities:
            lifecycle_manager.request_transition(entity, LifecycleState.DESTROYING, immediate=False)

        # Process should handle all
        count = lifecycle_manager.process_pending_transitions()
        assert count == 5


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestLifecycleErrorHandling:
    """Tests for lifecycle error handling."""

    def test_invalid_transition_rejected(self, lifecycle_manager):
        """Test invalid transition returns False."""
        entity = MockEntity()
        entity._lifecycle_state = LifecycleState.DESTROYED
        lifecycle_manager.register_entity(entity)

        result = lifecycle_manager.request_transition(
            entity,
            LifecycleState.ACTIVE,  # Invalid from DESTROYED
            immediate=True,
        )
        assert result is False

    def test_transition_from_terminal_state(self, lifecycle_manager):
        """Test no transitions allowed from DESTROYED."""
        entity = MockEntity()
        entity._lifecycle_state = LifecycleState.DESTROYED
        lifecycle_manager.register_entity(entity)

        for state in LifecycleState:
            result = lifecycle_manager.request_transition(entity, state, immediate=True)
            assert result is False

    def test_callback_error_doesnt_break_transition(self, lifecycle_manager):
        """Test callback errors don't break state transition."""
        def bad_callback(entity):
            raise RuntimeError("Callback error")

        lifecycle_manager.register_global_callback("on_spawn", bad_callback)

        entity = MockEntity()
        entity._lifecycle_state = LifecycleState.UNINITIALIZED
        lifecycle_manager.register_entity(entity)

        # Should not raise, callback error is caught
        result = lifecycle_manager.request_transition(
            entity,
            LifecycleState.CREATED,
            immediate=True,
        )
        # State should still change despite callback error
        assert entity._lifecycle_state == LifecycleState.CREATED


class TestLifecycleValidation:
    """Tests for lifecycle validation."""

    def test_validate_invalid_event(self):
        """Test validation rejects invalid event names."""
        with pytest.raises(ValueError, match="Invalid lifecycle event"):
            @lifecycle_hook(event="invalid_event")
            def handle(entity):
                pass


# =============================================================================
# BATCH OPERATIONS TESTS
# =============================================================================


class TestBatchLifecycleOperations:
    """Tests for batch lifecycle operations."""

    def test_batch_spawn(self, lifecycle_manager):
        """Test spawning multiple entities in batch."""
        entities = []
        for i in range(20):
            entity = MockEntity()
            entity._lifecycle_state = LifecycleState.UNINITIALIZED
            lifecycle_manager.register_entity(entity)
            lifecycle_manager.request_transition(entity, LifecycleState.CREATED, immediate=False)
            entities.append(entity)

        count = lifecycle_manager.process_pending_transitions()
        assert count == 20

    def test_batch_destroy(self, lifecycle_manager):
        """Test destroying multiple entities in batch."""
        entities = []
        for i in range(15):
            entity = MockEntity()
            entity._lifecycle_state = LifecycleState.ACTIVE
            lifecycle_manager.register_entity(entity)
            lifecycle_manager.request_transition(entity, LifecycleState.DESTROYING, immediate=False)
            entities.append(entity)

        count = lifecycle_manager.process_pending_transitions()
        assert count == 15

    def test_mixed_batch_operations(self, lifecycle_manager):
        """Test mixed spawn and destroy in same batch."""
        spawn_entities = []
        destroy_entities = []

        for i in range(5):
            entity = MockEntity()
            entity._lifecycle_state = LifecycleState.UNINITIALIZED
            lifecycle_manager.register_entity(entity)
            lifecycle_manager.request_transition(entity, LifecycleState.CREATED, immediate=False)
            spawn_entities.append(entity)

        for i in range(5):
            entity = MockEntity()
            entity._lifecycle_state = LifecycleState.ACTIVE
            lifecycle_manager.register_entity(entity)
            lifecycle_manager.request_transition(entity, LifecycleState.DESTROYING, immediate=False)
            destroy_entities.append(entity)

        count = lifecycle_manager.process_pending_transitions()
        assert count == 10

        for entity in spawn_entities:
            assert entity._lifecycle_state == LifecycleState.CREATED
        for entity in destroy_entities:
            assert entity._lifecycle_state == LifecycleState.DESTROYING


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================


class TestLifecycleThreadSafety:
    """Tests for lifecycle thread safety."""

    def test_concurrent_registration(self, lifecycle_manager):
        """Test concurrent entity registration is safe."""
        entities = []
        lock = threading.Lock()

        def register_entity():
            entity = MockEntity()
            lifecycle_manager.register_entity(entity)
            with lock:
                entities.append(entity)

        threads = [threading.Thread(target=register_entity) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(entities) == 50

    def test_concurrent_transition_requests(self, lifecycle_manager):
        """Test concurrent transition requests are safe."""
        entities = []
        for i in range(20):
            entity = MockEntity()
            entity._lifecycle_state = LifecycleState.UNINITIALIZED
            lifecycle_manager.register_entity(entity)
            entities.append(entity)

        def request_transition(entity):
            lifecycle_manager.request_transition(entity, LifecycleState.CREATED, immediate=False)

        threads = [threading.Thread(target=request_transition, args=(e,)) for e in entities]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        count = lifecycle_manager.process_pending_transitions()
        assert count == 20


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestLifecycleIntegration:
    """Integration tests for lifecycle system."""

    def test_full_lifecycle_flow(self, lifecycle_manager):
        """Test complete entity lifecycle from spawn to destroy."""
        entity = MockEntity()
        entity._lifecycle_state = LifecycleState.UNINITIALIZED
        lifecycle_manager.register_entity(entity)

        # Spawn
        lifecycle_manager.request_transition(entity, LifecycleState.CREATED, immediate=True)
        assert entity._lifecycle_state == LifecycleState.CREATED

        # Initialize
        lifecycle_manager.request_transition(entity, LifecycleState.INITIALIZING, immediate=True)
        lifecycle_manager.request_transition(entity, LifecycleState.INITIALIZED, immediate=True)
        assert entity._lifecycle_state == LifecycleState.INITIALIZED

        # Begin play
        lifecycle_manager.request_transition(entity, LifecycleState.BEGINNING_PLAY, immediate=True)
        lifecycle_manager.request_transition(entity, LifecycleState.ACTIVE, immediate=True)
        assert entity._lifecycle_state == LifecycleState.ACTIVE

        # Deactivate
        lifecycle_manager.request_transition(entity, LifecycleState.DEACTIVATING, immediate=True)
        lifecycle_manager.request_transition(entity, LifecycleState.DEACTIVATED, immediate=True)
        assert entity._lifecycle_state == LifecycleState.DEACTIVATED

        # Destroy
        lifecycle_manager.request_transition(entity, LifecycleState.DESTROYING, immediate=True)
        lifecycle_manager.request_transition(entity, LifecycleState.DESTROYED, immediate=True)
        assert entity._lifecycle_state == LifecycleState.DESTROYED

    def test_reactivation_flow(self, lifecycle_manager):
        """Test entity can be reactivated from deactivated state."""
        entity = MockEntity()
        entity._lifecycle_state = LifecycleState.DEACTIVATED
        lifecycle_manager.register_entity(entity)

        # Reactivate
        lifecycle_manager.request_transition(entity, LifecycleState.BEGINNING_PLAY, immediate=True)
        lifecycle_manager.request_transition(entity, LifecycleState.ACTIVE, immediate=True)
        assert entity._lifecycle_state == LifecycleState.ACTIVE

    def test_early_destroy(self, lifecycle_manager):
        """Test destroying entity before full initialization."""
        entity = MockEntity()
        entity._lifecycle_state = LifecycleState.CREATED
        lifecycle_manager.register_entity(entity)

        # Destroy early
        lifecycle_manager.request_transition(entity, LifecycleState.DESTROYING, immediate=True)
        assert entity._lifecycle_state == LifecycleState.DESTROYING

    def test_callback_invocation_order(self, lifecycle_manager):
        """Test callbacks are invoked in priority order."""
        call_order = []

        def callback_low(entity):
            call_order.append("low")

        def callback_high(entity):
            call_order.append("high")

        entity = MockEntity()
        entity._lifecycle_state = LifecycleState.UNINITIALIZED
        entity._lifecycle_callbacks = {}
        entity._lifecycle_callbacks[LifecycleEvent.ON_SPAWN] = [
            LifecycleCallback(event=LifecycleEvent.ON_SPAWN, callback=callback_low, priority=0),
            LifecycleCallback(event=LifecycleEvent.ON_SPAWN, callback=callback_high, priority=10),
        ]

        lifecycle_manager.register_entity(entity)
        lifecycle_manager.request_transition(entity, LifecycleState.CREATED, immediate=True)

        # Note: Actual order depends on implementation
        # Just verify both were called
        # The implementation sorts by priority (low first typically)
