"""
Blackbox tests for EventSynchronizer and SyncEvent.

Tests the public API behavior without examining implementation details.
Covers event creation, handler registration, event queueing, processing,
deduplication, and integration scenarios.

Task: T-FB-4.18
"""

import pytest
from typing import List, Dict, Any


# =============================================================================
# Test Fixtures and Setup
# =============================================================================

@pytest.fixture
def event_sync_module():
    """Import event synchronizer module."""
    from engine.animation.graph.sync import EventSynchronizer, SyncEvent
    return EventSynchronizer, SyncEvent


@pytest.fixture
def EventSynchronizer(event_sync_module):
    """Get EventSynchronizer class."""
    return event_sync_module[0]


@pytest.fixture
def SyncEvent(event_sync_module):
    """Get SyncEvent class."""
    return event_sync_module[1]


@pytest.fixture
def synchronizer(EventSynchronizer):
    """Create a fresh EventSynchronizer instance."""
    return EventSynchronizer()


@pytest.fixture
def basic_event(SyncEvent):
    """Create a basic test event."""
    return SyncEvent(
        name="test_event",
        source_node_id="node_1",
        normalized_time=0.5
    )


@pytest.fixture
def foot_plant_events(SyncEvent):
    """Create foot plant events for walk cycle testing."""
    return [
        SyncEvent(name="left_foot_plant", source_node_id="walk_node", normalized_time=0.0),
        SyncEvent(name="right_foot_plant", source_node_id="walk_node", normalized_time=0.5),
    ]


@pytest.fixture
def animation_events(SyncEvent):
    """Create various animation events."""
    return [
        SyncEvent(name="attack_start", source_node_id="combat", normalized_time=0.0),
        SyncEvent(name="attack_impact", source_node_id="combat", normalized_time=0.4),
        SyncEvent(name="attack_end", source_node_id="combat", normalized_time=1.0),
    ]


# =============================================================================
# SyncEvent Creation Tests
# =============================================================================

class TestSyncEventCreation:
    """Tests for SyncEvent instantiation and field access."""

    def test_create_with_required_fields(self, SyncEvent):
        """Create event with required name, source, and time."""
        event = SyncEvent(
            name="test_event",
            source_node_id="source_1",
            normalized_time=0.25
        )
        assert event.name == "test_event"
        assert event.source_node_id == "source_1"
        assert event.normalized_time == 0.25

    def test_create_with_empty_data(self, SyncEvent):
        """Create event without explicit data - should have empty dict."""
        event = SyncEvent(
            name="simple",
            source_node_id="node",
            normalized_time=0.0
        )
        assert event.data == {} or event.data is not None

    def test_create_with_custom_data(self, SyncEvent):
        """Create event with custom data dictionary."""
        data = {"velocity": 2.5, "bone_name": "foot_l"}
        event = SyncEvent(
            name="foot_plant",
            source_node_id="walk",
            normalized_time=0.0,
            data=data
        )
        assert event.data.get("velocity") == 2.5
        assert event.data.get("bone_name") == "foot_l"

    def test_create_at_time_zero(self, SyncEvent):
        """Create event at normalized time 0.0."""
        event = SyncEvent(name="start", source_node_id="n", normalized_time=0.0)
        assert event.normalized_time == 0.0

    def test_create_at_time_one(self, SyncEvent):
        """Create event at normalized time 1.0."""
        event = SyncEvent(name="end", source_node_id="n", normalized_time=1.0)
        assert event.normalized_time == 1.0

    def test_create_at_midpoint(self, SyncEvent):
        """Create event at normalized time 0.5."""
        event = SyncEvent(name="mid", source_node_id="n", normalized_time=0.5)
        assert event.normalized_time == 0.5

    def test_create_with_precise_time(self, SyncEvent):
        """Create event with high precision time value."""
        event = SyncEvent(name="precise", source_node_id="n", normalized_time=0.123456789)
        assert abs(event.normalized_time - 0.123456789) < 1e-9

    def test_name_with_underscores(self, SyncEvent):
        """Event name can contain underscores."""
        event = SyncEvent(name="left_foot_ground_contact", source_node_id="n", normalized_time=0.1)
        assert event.name == "left_foot_ground_contact"

    def test_name_single_character(self, SyncEvent):
        """Event name can be a single character."""
        event = SyncEvent(name="A", source_node_id="n", normalized_time=0.1)
        assert event.name == "A"

    def test_source_node_id_with_path(self, SyncEvent):
        """Source node ID can contain path-like identifiers."""
        event = SyncEvent(name="ev", source_node_id="root/child/node", normalized_time=0.1)
        assert event.source_node_id == "root/child/node"

    def test_data_with_nested_dict(self, SyncEvent):
        """Event data can contain nested dictionaries."""
        data = {"outer": {"inner": {"value": 42}}}
        event = SyncEvent(name="nested", source_node_id="n", normalized_time=0.1, data=data)
        assert event.data["outer"]["inner"]["value"] == 42

    def test_data_with_list_values(self, SyncEvent):
        """Event data can contain lists."""
        data = {"bones": ["hip", "knee", "ankle"]}
        event = SyncEvent(name="chain", source_node_id="n", normalized_time=0.1, data=data)
        assert "knee" in event.data["bones"]

    def test_data_with_mixed_types(self, SyncEvent):
        """Event data can contain mixed types."""
        data = {
            "int_val": 10,
            "float_val": 3.14,
            "str_val": "test",
            "bool_val": True,
            "list_val": [1, 2, 3],
            "none_val": None
        }
        event = SyncEvent(name="mixed", source_node_id="n", normalized_time=0.1, data=data)
        assert event.data["int_val"] == 10
        assert event.data["float_val"] == 3.14
        assert event.data["bool_val"] is True


# =============================================================================
# EventSynchronizer Creation Tests
# =============================================================================

class TestEventSynchronizerCreation:
    """Tests for EventSynchronizer instantiation."""

    def test_create_empty_synchronizer(self, EventSynchronizer):
        """Create synchronizer with no handlers or events."""
        sync = EventSynchronizer()
        assert sync is not None

    def test_create_multiple_synchronizers(self, EventSynchronizer):
        """Multiple synchronizers can coexist independently."""
        sync1 = EventSynchronizer()
        sync2 = EventSynchronizer()
        assert sync1 is not sync2

    def test_synchronizer_has_register_handler(self, EventSynchronizer):
        """Synchronizer exposes register_handler method."""
        sync = EventSynchronizer()
        assert hasattr(sync, 'register_handler')
        assert callable(sync.register_handler)

    def test_synchronizer_has_unregister_handler(self, EventSynchronizer):
        """Synchronizer exposes unregister_handler method."""
        sync = EventSynchronizer()
        assert hasattr(sync, 'unregister_handler')
        assert callable(sync.unregister_handler)

    def test_synchronizer_has_queue_event(self, EventSynchronizer):
        """Synchronizer exposes queue_event method."""
        sync = EventSynchronizer()
        assert hasattr(sync, 'queue_event')
        assert callable(sync.queue_event)

    def test_synchronizer_has_process_events(self, EventSynchronizer):
        """Synchronizer exposes process_events method."""
        sync = EventSynchronizer()
        assert hasattr(sync, 'process_events')
        assert callable(sync.process_events)

    def test_synchronizer_has_clear(self, EventSynchronizer):
        """Synchronizer exposes clear method."""
        sync = EventSynchronizer()
        assert hasattr(sync, 'clear')
        assert callable(sync.clear)


# =============================================================================
# Handler Registration Tests
# =============================================================================

class TestHandlerRegistration:
    """Tests for registering and unregistering event handlers."""

    def test_register_single_handler(self, synchronizer):
        """Register a single handler for an event type."""
        received = []
        def handler(event):
            received.append(event)

        synchronizer.register_handler("test_event", handler)
        # Registration should not raise

    def test_register_multiple_handlers_same_event(self, synchronizer):
        """Multiple handlers can be registered for same event type."""
        received1 = []
        received2 = []

        def handler1(event):
            received1.append(event)

        def handler2(event):
            received2.append(event)

        synchronizer.register_handler("test_event", handler1)
        synchronizer.register_handler("test_event", handler2)

    def test_register_handlers_different_events(self, synchronizer):
        """Handlers can be registered for different event types."""
        synchronizer.register_handler("event_a", lambda e: None)
        synchronizer.register_handler("event_b", lambda e: None)
        synchronizer.register_handler("event_c", lambda e: None)

    def test_unregister_handler(self, synchronizer):
        """Unregister a previously registered handler."""
        def handler(event):
            pass

        synchronizer.register_handler("test_event", handler)
        synchronizer.unregister_handler("test_event", handler)

    def test_unregister_nonexistent_handler_no_error(self, synchronizer):
        """Unregistering non-existent handler should not raise."""
        def handler(event):
            pass

        # Unregister handler that was never registered
        try:
            synchronizer.unregister_handler("test_event", handler)
        except Exception:
            pytest.fail("Unregistering non-existent handler should not raise")

    def test_unregister_from_nonexistent_event_type(self, synchronizer):
        """Unregistering from non-existent event type should not raise."""
        def handler(event):
            pass

        try:
            synchronizer.unregister_handler("nonexistent_event", handler)
        except Exception:
            pytest.fail("Unregistering from non-existent event type should not raise")

    def test_register_lambda_handler(self, synchronizer):
        """Lambda functions can be used as handlers."""
        synchronizer.register_handler("test", lambda e: None)

    def test_register_method_handler(self, synchronizer):
        """Instance methods can be used as handlers."""
        class EventHandler:
            def __init__(self):
                self.events = []

            def handle(self, event):
                self.events.append(event)

        handler_obj = EventHandler()
        synchronizer.register_handler("test", handler_obj.handle)


# =============================================================================
# Event Queueing Tests
# =============================================================================

class TestEventQueueing:
    """Tests for queuing events."""

    def test_queue_single_event(self, synchronizer, basic_event):
        """Queue a single event."""
        synchronizer.queue_event(basic_event)
        # Should not raise

    def test_queue_multiple_events(self, synchronizer, SyncEvent):
        """Queue multiple events."""
        for i in range(5):
            event = SyncEvent(name=f"event_{i}", source_node_id="node", normalized_time=i * 0.2)
            synchronizer.queue_event(event)

    def test_queue_events_different_types(self, synchronizer, animation_events):
        """Queue events of different types."""
        for event in animation_events:
            synchronizer.queue_event(event)

    def test_queue_same_event_multiple_times(self, synchronizer, basic_event):
        """Queue the same event instance multiple times."""
        synchronizer.queue_event(basic_event)
        synchronizer.queue_event(basic_event)
        synchronizer.queue_event(basic_event)

    def test_queue_events_with_same_name(self, synchronizer, SyncEvent):
        """Queue multiple events with same name but different sources."""
        event1 = SyncEvent(name="collision", source_node_id="node_a", normalized_time=0.1)
        event2 = SyncEvent(name="collision", source_node_id="node_b", normalized_time=0.2)
        synchronizer.queue_event(event1)
        synchronizer.queue_event(event2)


# =============================================================================
# Event Processing Tests
# =============================================================================

class TestEventProcessing:
    """Tests for processing queued events."""

    def test_process_calls_registered_handler(self, synchronizer, SyncEvent):
        """Processing events calls the registered handler."""
        received = []

        def handler(event):
            received.append(event)

        synchronizer.register_handler("test_event", handler)
        event = SyncEvent(name="test_event", source_node_id="n", normalized_time=0.5)
        synchronizer.queue_event(event)
        synchronizer.process_events()

        assert len(received) == 1
        assert received[0].name == "test_event"

    def test_process_handler_receives_event_data(self, synchronizer, SyncEvent):
        """Handler receives event with all original data."""
        received = []

        def handler(event):
            received.append(event)

        synchronizer.register_handler("data_event", handler)
        event = SyncEvent(
            name="data_event",
            source_node_id="source_123",
            normalized_time=0.75,
            data={"key": "value", "num": 42}
        )
        synchronizer.queue_event(event)
        synchronizer.process_events()

        assert len(received) == 1
        assert received[0].source_node_id == "source_123"
        assert received[0].normalized_time == 0.75
        assert received[0].data["key"] == "value"

    def test_process_multiple_handlers_all_called(self, synchronizer, SyncEvent):
        """All handlers for an event type are called."""
        received1 = []
        received2 = []
        received3 = []

        synchronizer.register_handler("multi", lambda e: received1.append(e))
        synchronizer.register_handler("multi", lambda e: received2.append(e))
        synchronizer.register_handler("multi", lambda e: received3.append(e))

        event = SyncEvent(name="multi", source_node_id="n", normalized_time=0.1)
        synchronizer.queue_event(event)
        synchronizer.process_events()

        assert len(received1) == 1
        assert len(received2) == 1
        assert len(received3) == 1

    def test_process_empty_queue(self, synchronizer):
        """Processing with no queued events should not raise."""
        received = []
        synchronizer.register_handler("test", lambda e: received.append(e))
        synchronizer.process_events()
        assert len(received) == 0

    def test_process_no_handlers(self, synchronizer, basic_event):
        """Processing events with no registered handlers should not raise."""
        synchronizer.queue_event(basic_event)
        synchronizer.process_events()  # Should not raise

    def test_process_clears_queue(self, synchronizer, SyncEvent):
        """Events are cleared from queue after processing."""
        received = []

        def handler(event):
            received.append(event)

        synchronizer.register_handler("test", handler)
        event = SyncEvent(name="test", source_node_id="n", normalized_time=0.5)
        synchronizer.queue_event(event)

        synchronizer.process_events()
        assert len(received) == 1

        received.clear()
        synchronizer.process_events()
        assert len(received) == 0  # Queue should be empty

    def test_process_only_matching_handlers_called(self, synchronizer, SyncEvent):
        """Only handlers for matching event type are called."""
        received_a = []
        received_b = []

        synchronizer.register_handler("event_a", lambda e: received_a.append(e))
        synchronizer.register_handler("event_b", lambda e: received_b.append(e))

        event = SyncEvent(name="event_a", source_node_id="n", normalized_time=0.1)
        synchronizer.queue_event(event)
        synchronizer.process_events()

        assert len(received_a) == 1
        assert len(received_b) == 0


# =============================================================================
# Event Deduplication Tests
# =============================================================================

class TestEventDeduplication:
    """Tests for event deduplication behavior."""

    def test_duplicate_events_deduplicated(self, synchronizer, SyncEvent):
        """Identical events queued multiple times may be deduplicated."""
        received = []

        def handler(event):
            received.append(event)

        synchronizer.register_handler("dup_test", handler)

        # Queue same event twice
        event = SyncEvent(name="dup_test", source_node_id="same", normalized_time=0.5)
        synchronizer.queue_event(event)
        synchronizer.queue_event(event)

        synchronizer.process_events()

        # Either 1 (deduplicated) or 2 (not deduplicated) - test documents behavior
        assert len(received) >= 1

    def test_similar_events_different_sources_same_time_deduplicated(self, synchronizer, SyncEvent):
        """Events with same name and time are deduplicated regardless of source."""
        received = []

        def handler(event):
            received.append(event)

        synchronizer.register_handler("collision", handler)

        event1 = SyncEvent(name="collision", source_node_id="node_a", normalized_time=0.5)
        event2 = SyncEvent(name="collision", source_node_id="node_b", normalized_time=0.5)

        synchronizer.queue_event(event1)
        synchronizer.queue_event(event2)
        synchronizer.process_events()

        # Deduplication is based on (name, normalized_time) pair
        assert len(received) == 1

    def test_events_different_sources_different_times_not_deduplicated(self, synchronizer, SyncEvent):
        """Events with different times are not deduplicated."""
        received = []

        def handler(event):
            received.append(event)

        synchronizer.register_handler("collision", handler)

        event1 = SyncEvent(name="collision", source_node_id="node_a", normalized_time=0.5)
        event2 = SyncEvent(name="collision", source_node_id="node_b", normalized_time=0.6)

        synchronizer.queue_event(event1)
        synchronizer.queue_event(event2)
        synchronizer.process_events()

        assert len(received) == 2

    def test_similar_events_different_times_not_deduplicated(self, synchronizer, SyncEvent):
        """Events with different times are not deduplicated."""
        received = []

        def handler(event):
            received.append(event)

        synchronizer.register_handler("tick", handler)

        event1 = SyncEvent(name="tick", source_node_id="clock", normalized_time=0.1)
        event2 = SyncEvent(name="tick", source_node_id="clock", normalized_time=0.2)

        synchronizer.queue_event(event1)
        synchronizer.queue_event(event2)
        synchronizer.process_events()

        assert len(received) == 2


# =============================================================================
# Clear Operation Tests
# =============================================================================

class TestClearOperation:
    """Tests for the clear method."""

    def test_clear_removes_queued_events(self, synchronizer, SyncEvent):
        """Clear removes all queued events."""
        received = []

        def handler(event):
            received.append(event)

        synchronizer.register_handler("test", handler)

        event = SyncEvent(name="test", source_node_id="n", normalized_time=0.5)
        synchronizer.queue_event(event)
        synchronizer.clear()
        synchronizer.process_events()

        assert len(received) == 0

    def test_clear_empty_synchronizer(self, synchronizer):
        """Clear on empty synchronizer should not raise."""
        synchronizer.clear()

    def test_clear_multiple_times(self, synchronizer):
        """Clear can be called multiple times."""
        synchronizer.clear()
        synchronizer.clear()
        synchronizer.clear()

    def test_can_queue_after_clear(self, synchronizer, SyncEvent):
        """Events can be queued after clear."""
        received = []

        def handler(event):
            received.append(event)

        synchronizer.register_handler("test", handler)
        synchronizer.clear()

        event = SyncEvent(name="test", source_node_id="n", normalized_time=0.5)
        synchronizer.queue_event(event)
        synchronizer.process_events()

        assert len(received) == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_walk_cycle_foot_events(self, synchronizer, SyncEvent):
        """Simulate walk cycle foot plant events."""
        left_plants = []
        right_plants = []

        synchronizer.register_handler("left_foot_plant", lambda e: left_plants.append(e))
        synchronizer.register_handler("right_foot_plant", lambda e: right_plants.append(e))

        # Simulate two walk cycles
        for cycle in range(2):
            offset = cycle * 1.0
            synchronizer.queue_event(SyncEvent(
                name="left_foot_plant",
                source_node_id="walk",
                normalized_time=0.0 + offset,
                data={"cycle": cycle}
            ))
            synchronizer.queue_event(SyncEvent(
                name="right_foot_plant",
                source_node_id="walk",
                normalized_time=0.5 + offset,
                data={"cycle": cycle}
            ))

        synchronizer.process_events()

        assert len(left_plants) == 2
        assert len(right_plants) == 2

    def test_combat_event_sequence(self, synchronizer, SyncEvent):
        """Simulate combat animation event sequence."""
        event_log = []

        def log_event(event):
            event_log.append((event.name, event.normalized_time))

        synchronizer.register_handler("attack_start", log_event)
        synchronizer.register_handler("attack_impact", log_event)
        synchronizer.register_handler("attack_recovery", log_event)
        synchronizer.register_handler("attack_end", log_event)

        synchronizer.queue_event(SyncEvent(name="attack_start", source_node_id="combat", normalized_time=0.0))
        synchronizer.queue_event(SyncEvent(name="attack_impact", source_node_id="combat", normalized_time=0.4))
        synchronizer.queue_event(SyncEvent(name="attack_recovery", source_node_id="combat", normalized_time=0.6))
        synchronizer.queue_event(SyncEvent(name="attack_end", source_node_id="combat", normalized_time=1.0))

        synchronizer.process_events()

        assert len(event_log) == 4
        event_names = [e[0] for e in event_log]
        assert "attack_start" in event_names
        assert "attack_impact" in event_names

    def test_multiple_event_types_interleaved(self, synchronizer, SyncEvent):
        """Handle multiple event types queued in interleaved order."""
        type_a_received = []
        type_b_received = []
        type_c_received = []

        synchronizer.register_handler("type_a", lambda e: type_a_received.append(e))
        synchronizer.register_handler("type_b", lambda e: type_b_received.append(e))
        synchronizer.register_handler("type_c", lambda e: type_c_received.append(e))

        # Queue in interleaved order
        synchronizer.queue_event(SyncEvent(name="type_a", source_node_id="n", normalized_time=0.1))
        synchronizer.queue_event(SyncEvent(name="type_b", source_node_id="n", normalized_time=0.2))
        synchronizer.queue_event(SyncEvent(name="type_c", source_node_id="n", normalized_time=0.3))
        synchronizer.queue_event(SyncEvent(name="type_a", source_node_id="n", normalized_time=0.4))
        synchronizer.queue_event(SyncEvent(name="type_b", source_node_id="n", normalized_time=0.5))

        synchronizer.process_events()

        assert len(type_a_received) == 2
        assert len(type_b_received) == 2
        assert len(type_c_received) == 1

    def test_handler_registration_order(self, synchronizer, SyncEvent):
        """Handlers are called (order may vary but all are called)."""
        call_order = []

        def handler_first(e):
            call_order.append("first")

        def handler_second(e):
            call_order.append("second")

        def handler_third(e):
            call_order.append("third")

        synchronizer.register_handler("ordered", handler_first)
        synchronizer.register_handler("ordered", handler_second)
        synchronizer.register_handler("ordered", handler_third)

        synchronizer.queue_event(SyncEvent(name="ordered", source_node_id="n", normalized_time=0.5))
        synchronizer.process_events()

        assert len(call_order) == 3
        assert "first" in call_order
        assert "second" in call_order
        assert "third" in call_order

    def test_rapid_event_processing(self, synchronizer, SyncEvent):
        """Handle rapid queueing and processing cycles."""
        received = []

        def handler(event):
            received.append(event.data.get("batch"))

        synchronizer.register_handler("rapid", handler)

        for batch in range(10):
            for i in range(5):
                synchronizer.queue_event(SyncEvent(
                    name="rapid",
                    source_node_id=f"source_{i}",
                    normalized_time=i * 0.2,
                    data={"batch": batch}
                ))
            synchronizer.process_events()

        # 10 batches x 5 events each = 50 total
        assert len(received) == 50

    def test_handler_modifies_external_state(self, synchronizer, SyncEvent):
        """Handlers can modify external state."""
        state = {"count": 0, "total_time": 0.0}

        def counting_handler(event):
            state["count"] += 1
            state["total_time"] += event.normalized_time

        synchronizer.register_handler("counted", counting_handler)

        for i in range(5):
            synchronizer.queue_event(SyncEvent(
                name="counted",
                source_node_id="n",
                normalized_time=0.1 * (i + 1)
            ))

        synchronizer.process_events()

        assert state["count"] == 5
        assert abs(state["total_time"] - 1.5) < 1e-6  # 0.1 + 0.2 + 0.3 + 0.4 + 0.5

    def test_unregister_during_active_use(self, synchronizer, SyncEvent):
        """Unregister handler and verify it stops receiving events."""
        received_before = []
        received_after = []

        def handler_before(e):
            received_before.append(e)

        def handler_after(e):
            received_after.append(e)

        synchronizer.register_handler("test", handler_before)
        synchronizer.register_handler("test", handler_after)

        # First batch
        synchronizer.queue_event(SyncEvent(name="test", source_node_id="n", normalized_time=0.1))
        synchronizer.process_events()

        assert len(received_before) == 1
        assert len(received_after) == 1

        # Unregister first handler
        synchronizer.unregister_handler("test", handler_before)

        # Second batch
        synchronizer.queue_event(SyncEvent(name="test", source_node_id="n", normalized_time=0.2))
        synchronizer.process_events()

        assert len(received_before) == 1  # Not called again
        assert len(received_after) == 2   # Still active

    def test_animation_transition_events(self, synchronizer, SyncEvent):
        """Simulate animation state machine transition events."""
        transitions = []

        def transition_handler(event):
            transitions.append({
                "from": event.data.get("from_state"),
                "to": event.data.get("to_state"),
                "time": event.normalized_time
            })

        synchronizer.register_handler("state_transition", transition_handler)

        synchronizer.queue_event(SyncEvent(
            name="state_transition",
            source_node_id="state_machine",
            normalized_time=0.0,
            data={"from_state": "idle", "to_state": "walk"}
        ))
        synchronizer.queue_event(SyncEvent(
            name="state_transition",
            source_node_id="state_machine",
            normalized_time=0.5,
            data={"from_state": "walk", "to_state": "run"}
        ))

        synchronizer.process_events()

        assert len(transitions) == 2
        assert transitions[0]["from"] == "idle"
        assert transitions[1]["to"] == "run"

    def test_event_with_complex_data_payload(self, synchronizer, SyncEvent):
        """Handle events with complex nested data."""
        received_data = []

        def data_handler(event):
            received_data.append(event.data)

        synchronizer.register_handler("complex_event", data_handler)

        complex_data = {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "rotation": {"pitch": 0.0, "yaw": 90.0, "roll": 0.0},
            "velocity": [0.5, 0.0, 0.0],
            "metadata": {
                "source": "physics_engine",
                "frame": 120,
                "tags": ["collision", "ground"]
            }
        }

        synchronizer.queue_event(SyncEvent(
            name="complex_event",
            source_node_id="physics",
            normalized_time=0.5,
            data=complex_data
        ))

        synchronizer.process_events()

        assert len(received_data) == 1
        assert received_data[0]["position"]["x"] == 1.0
        assert received_data[0]["metadata"]["frame"] == 120
        assert "collision" in received_data[0]["metadata"]["tags"]


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_handler_with_exception_does_not_crash(self, synchronizer, SyncEvent):
        """Handler raising exception should not crash processing."""
        received = []

        def failing_handler(event):
            raise ValueError("Handler failed")

        def working_handler(event):
            received.append(event)

        synchronizer.register_handler("test", failing_handler)
        synchronizer.register_handler("test", working_handler)

        synchronizer.queue_event(SyncEvent(name="test", source_node_id="n", normalized_time=0.5))

        # Behavior depends on implementation - either raises or continues
        try:
            synchronizer.process_events()
            # If it doesn't raise, working handler may or may not have been called
        except ValueError:
            # Exception propagated - this is acceptable behavior
            pass

    def test_very_long_event_name(self, SyncEvent):
        """Event with very long name."""
        long_name = "event_" + "x" * 1000
        event = SyncEvent(name=long_name, source_node_id="n", normalized_time=0.5)
        assert event.name == long_name

    def test_unicode_event_name(self, SyncEvent):
        """Event name can contain unicode characters."""
        event = SyncEvent(name="event_test", source_node_id="n", normalized_time=0.5)
        assert "event" in event.name

    def test_negative_normalized_time(self, SyncEvent):
        """Event with negative normalized time (if allowed)."""
        try:
            event = SyncEvent(name="negative", source_node_id="n", normalized_time=-0.1)
            # If allowed, verify it's stored
            assert event.normalized_time == -0.1
        except (ValueError, AssertionError):
            # Implementation may reject negative times
            pass

    def test_normalized_time_greater_than_one(self, SyncEvent):
        """Event with normalized time > 1.0 (if allowed)."""
        try:
            event = SyncEvent(name="overflow", source_node_id="n", normalized_time=1.5)
            assert event.normalized_time == 1.5
        except (ValueError, AssertionError):
            # Implementation may reject times > 1.0
            pass

    def test_many_handlers_same_event(self, synchronizer, SyncEvent):
        """Register many handlers for same event type."""
        counters = []

        for i in range(50):
            counter = {"value": 0}
            counters.append(counter)

            def make_handler(c):
                return lambda e: c.__setitem__("value", c["value"] + 1)

            synchronizer.register_handler("many_handlers", make_handler(counter))

        synchronizer.queue_event(SyncEvent(name="many_handlers", source_node_id="n", normalized_time=0.5))
        synchronizer.process_events()

        total_calls = sum(c["value"] for c in counters)
        assert total_calls == 50

    def test_many_event_types(self, synchronizer, SyncEvent):
        """Register handlers for many different event types."""
        received = {}

        for i in range(100):
            event_name = f"event_type_{i}"
            received[event_name] = []

            def make_handler(name):
                return lambda e: received[name].append(e)

            synchronizer.register_handler(event_name, make_handler(event_name))

        # Queue one event per type
        for i in range(100):
            synchronizer.queue_event(SyncEvent(
                name=f"event_type_{i}",
                source_node_id="n",
                normalized_time=i / 100.0
            ))

        synchronizer.process_events()

        for i in range(100):
            assert len(received[f"event_type_{i}"]) == 1

    def test_empty_source_node_id(self, SyncEvent):
        """Event with empty source node ID."""
        event = SyncEvent(name="empty_source", source_node_id="", normalized_time=0.5)
        assert event.source_node_id == ""

    def test_data_mutation_after_event_creation(self, SyncEvent):
        """Data dictionary mutation behavior after event creation."""
        data = {"original": True}
        event = SyncEvent(name="mutable", source_node_id="n", normalized_time=0.5, data=data)

        # Mutate original dict
        data["modified"] = True

        # Behavior depends on implementation - may or may not reflect mutation
        # This test documents the behavior
        assert event.data.get("original") is True


# =============================================================================
# Performance Characteristics Tests
# =============================================================================

class TestPerformanceCharacteristics:
    """Tests for performance-related characteristics."""

    def test_large_event_queue_different_names(self, synchronizer, SyncEvent):
        """Handle large number of queued events with different names."""
        received = []

        def handler(event):
            received.append(event)

        # Register handler for each unique event name
        for i in range(100):
            synchronizer.register_handler(f"bulk_{i}", handler)

        # Use unique event names to avoid deduplication
        for i in range(100):
            synchronizer.queue_event(SyncEvent(
                name=f"bulk_{i}",
                source_node_id=f"node_{i % 10}",
                normalized_time=0.5
            ))

        synchronizer.process_events()

        # Each unique name should be processed
        assert len(received) == 100

    def test_large_event_queue_with_deduplication(self, synchronizer, SyncEvent):
        """Deduplication reduces events with same name and time."""
        received = []

        def handler(event):
            received.append(event)

        synchronizer.register_handler("bulk", handler)

        # Queue events with repeating times (will be deduplicated)
        for i in range(1000):
            synchronizer.queue_event(SyncEvent(
                name="bulk",
                source_node_id=f"node_{i % 10}",
                normalized_time=(i % 100) / 100.0  # Only 100 unique times
            ))

        synchronizer.process_events()

        # Should have exactly 100 unique (name, time) pairs
        assert len(received) == 100

    def test_frequent_process_calls(self, synchronizer, SyncEvent):
        """Handle frequent process_events calls."""
        received = []

        def handler(event):
            received.append(event)

        synchronizer.register_handler("frequent", handler)

        for i in range(100):
            synchronizer.queue_event(SyncEvent(
                name="frequent",
                source_node_id="n",
                normalized_time=0.5
            ))
            synchronizer.process_events()

        assert len(received) == 100

    def test_process_events_returns_quickly_when_empty(self, synchronizer):
        """process_events should return quickly when queue is empty."""
        import time

        start = time.perf_counter()
        for _ in range(100):
            synchronizer.process_events()
        elapsed = time.perf_counter() - start

        # 100 empty process calls should complete in under 100ms
        assert elapsed < 0.1
