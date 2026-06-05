"""WHITEBOX tests for engine/animation/graph/sync.py SyncEvent and EventSynchronizer.

Tests for T-FB-4.18 (EventSynchronizer).

WHITEBOX coverage plan:
  [SyncEvent dataclass fields]
    Path A1:  name field stores string correctly
    Path A2:  source_node_id field stores string
    Path A3:  normalized_time field stores float
    Path A4:  data default is empty dict
    Path A5:  data accepts custom dict
    Path A6:  data dict is mutable after creation
    Path A7:  multiple SyncEvents have independent data dicts

  [EventSynchronizer.__init__]
    Path B1:  event_handlers initialized as empty dict
    Path B2:  _pending_events initialized as empty list
    Path B3:  multiple EventSynchronizer instances are independent

  [EventSynchronizer.register_handler creates list]
    Path C1:  new event_name creates new list
    Path C2:  event_handlers dict contains new event_name key
    Path C3:  handler added to newly created list
    Path C4:  list has length 1 after first registration

  [EventSynchronizer.register_handler multiple handlers]
    Path D1:  second handler for same event appended
    Path D2:  list has length 2 after two registrations
    Path D3:  handlers preserved in registration order
    Path D4:  same handler can be registered multiple times
    Path D5:  different events have separate handler lists

  [EventSynchronizer.unregister_handler returns True/False]
    Path E1:  unregister existing handler returns True
    Path E2:  unregister non-existent handler returns False
    Path E3:  unregister from non-existent event returns False
    Path E4:  handler removed from list after unregister

  [EventSynchronizer.unregister_handler edge cases]
    Path F1:  unregister first of multiple handlers
    Path F2:  unregister last of multiple handlers
    Path F3:  unregister middle handler
    Path F4:  unregister duplicate handler removes only one
    Path F5:  unregister same handler twice (second returns False)
    Path F6:  empty event_handlers dict returns False

  [EventSynchronizer.queue_event adds to pending]
    Path G1:  queue single event adds to _pending_events
    Path G2:  queue multiple events accumulates
    Path G3:  events preserved in queue order
    Path G4:  same event can be queued multiple times
    Path G5:  queue does not modify event

  [EventSynchronizer.process_events deduplication by name and rounded_time]
    Path H1:  identical events deduplicated
    Path H2:  events with same name but different rounded_time kept
    Path H3:  events with different name but same rounded_time kept
    Path H4:  normalized_time rounded to 2 decimal places
    Path H5:  0.001 and 0.004 round to same value (deduplicated)
    Path H6:  0.001 and 0.006 round to different values (both kept)
    Path H7:  negative time differences handled correctly

  [EventSynchronizer.process_events dispatches to handlers]
    Path I1:  handler called for matching event
    Path I2:  handler receives correct event object
    Path I3:  all handlers for event called
    Path I4:  handler order is preserved (FIFO)
    Path I5:  events dispatched in queue order (after dedup)
    Path I6:  no handlers means no dispatch (no error)

  [EventSynchronizer.process_events clears pending list]
    Path J1:  _pending_events empty after process_events
    Path J2:  _pending_events cleared even if no handlers
    Path J3:  _pending_events cleared even if handlers throw
    Path J4:  process_events on empty queue does nothing

  [EventSynchronizer.process_events exception handling]
    Path K1:  handler exception caught and ignored
    Path K2:  subsequent handlers still called after exception
    Path K3:  subsequent events still processed after exception
    Path K4:  multiple exceptions in handlers all caught

  [EventSynchronizer.process_events sync_group parameter]
    Path L1:  sync_group=None processes all events
    Path L2:  process_events accepts sync_group parameter

  [EventSynchronizer.clear]
    Path M1:  clear empties _pending_events
    Path M2:  clear on empty list succeeds
    Path M3:  clear does not affect event_handlers
    Path M4:  events can be queued after clear
"""

import pytest
from dataclasses import fields, field
from typing import List, Any

from engine.animation.graph.sync import SyncEvent, EventSynchronizer


# =============================================================================
# SyncEvent DATACLASS FIELDS
# =============================================================================


class TestSyncEventFields:
    """Tests for SyncEvent dataclass field access and defaults."""

    def test_A1_name_field_stores_string(self):
        """Path A1: name field stores string correctly."""
        event = SyncEvent(name="footstep", source_node_id="node1", normalized_time=0.5)
        assert event.name == "footstep"

    def test_A1_name_field_empty_string(self):
        """Path A1: name field accepts empty string."""
        event = SyncEvent(name="", source_node_id="node1", normalized_time=0.5)
        assert event.name == ""

    def test_A1_name_field_unicode(self):
        """Path A1: name field accepts unicode strings."""
        event = SyncEvent(name="event_α", source_node_id="node1", normalized_time=0.5)
        assert event.name == "event_α"

    def test_A1_name_field_whitespace(self):
        """Path A1: name field accepts whitespace."""
        event = SyncEvent(name="  spaced event  ", source_node_id="node1", normalized_time=0.5)
        assert event.name == "  spaced event  "

    def test_A2_source_node_id_stores_string(self):
        """Path A2: source_node_id field stores string."""
        event = SyncEvent(name="test", source_node_id="animation_node_42", normalized_time=0.5)
        assert event.source_node_id == "animation_node_42"

    def test_A2_source_node_id_empty_string(self):
        """Path A2: source_node_id accepts empty string."""
        event = SyncEvent(name="test", source_node_id="", normalized_time=0.5)
        assert event.source_node_id == ""

    def test_A2_source_node_id_uuid_format(self):
        """Path A2: source_node_id accepts UUID-like strings."""
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        event = SyncEvent(name="test", source_node_id=uuid_str, normalized_time=0.5)
        assert event.source_node_id == uuid_str

    def test_A3_normalized_time_stores_float(self):
        """Path A3: normalized_time field stores float."""
        event = SyncEvent(name="test", source_node_id="node1", normalized_time=0.75)
        assert event.normalized_time == 0.75
        assert isinstance(event.normalized_time, float)

    def test_A3_normalized_time_zero(self):
        """Path A3: normalized_time accepts zero."""
        event = SyncEvent(name="test", source_node_id="node1", normalized_time=0.0)
        assert event.normalized_time == 0.0

    def test_A3_normalized_time_one(self):
        """Path A3: normalized_time accepts one."""
        event = SyncEvent(name="test", source_node_id="node1", normalized_time=1.0)
        assert event.normalized_time == 1.0

    def test_A3_normalized_time_negative(self):
        """Path A3: normalized_time stores negative values (no clamping)."""
        event = SyncEvent(name="test", source_node_id="node1", normalized_time=-0.5)
        assert event.normalized_time == -0.5

    def test_A3_normalized_time_greater_than_one(self):
        """Path A3: normalized_time stores values > 1 (no clamping)."""
        event = SyncEvent(name="test", source_node_id="node1", normalized_time=2.5)
        assert event.normalized_time == 2.5

    def test_A4_data_default_empty_dict(self):
        """Path A4: data default is empty dict."""
        event = SyncEvent(name="test", source_node_id="node1", normalized_time=0.5)
        assert event.data == {}
        assert isinstance(event.data, dict)

    def test_A5_data_accepts_custom_dict(self):
        """Path A5: data accepts custom dict."""
        custom_data = {"velocity": 1.5, "intensity": 0.8}
        event = SyncEvent(name="test", source_node_id="node1", normalized_time=0.5, data=custom_data)
        assert event.data == {"velocity": 1.5, "intensity": 0.8}

    def test_A5_data_accepts_nested_dict(self):
        """Path A5: data accepts nested dict structures."""
        nested_data = {"metadata": {"bone": "foot_l", "weight": 1.0}, "flags": [True, False]}
        event = SyncEvent(name="test", source_node_id="node1", normalized_time=0.5, data=nested_data)
        assert event.data["metadata"]["bone"] == "foot_l"
        assert event.data["flags"] == [True, False]

    def test_A6_data_dict_mutable_after_creation(self):
        """Path A6: data dict is mutable after creation."""
        event = SyncEvent(name="test", source_node_id="node1", normalized_time=0.5)
        event.data["new_key"] = "new_value"
        assert event.data["new_key"] == "new_value"

    def test_A6_data_dict_can_be_cleared(self):
        """Path A6: data dict can be cleared after creation."""
        event = SyncEvent(name="test", source_node_id="node1", normalized_time=0.5, data={"key": "value"})
        event.data.clear()
        assert event.data == {}

    def test_A7_multiple_events_independent_data_dicts(self):
        """Path A7: multiple SyncEvents have independent data dicts."""
        event1 = SyncEvent(name="test1", source_node_id="node1", normalized_time=0.5)
        event2 = SyncEvent(name="test2", source_node_id="node2", normalized_time=0.6)
        event1.data["key"] = "value1"
        event2.data["key"] = "value2"
        assert event1.data["key"] == "value1"
        assert event2.data["key"] == "value2"

    def test_A7_default_data_not_shared(self):
        """Path A7: default data dicts are not shared between instances."""
        event1 = SyncEvent(name="test1", source_node_id="node1", normalized_time=0.5)
        event2 = SyncEvent(name="test2", source_node_id="node2", normalized_time=0.6)
        event1.data["exclusive"] = True
        assert "exclusive" not in event2.data


# =============================================================================
# EventSynchronizer.__init__
# =============================================================================


class TestEventSynchronizerInit:
    """Tests for EventSynchronizer initialization."""

    def test_B1_event_handlers_initialized_empty_dict(self):
        """Path B1: event_handlers initialized as empty dict."""
        sync = EventSynchronizer()
        assert sync.event_handlers == {}
        assert isinstance(sync.event_handlers, dict)

    def test_B2_pending_events_initialized_empty_list(self):
        """Path B2: _pending_events initialized as empty list."""
        sync = EventSynchronizer()
        assert sync._pending_events == []
        assert isinstance(sync._pending_events, list)

    def test_B3_multiple_instances_independent(self):
        """Path B3: multiple EventSynchronizer instances are independent."""
        sync1 = EventSynchronizer()
        sync2 = EventSynchronizer()

        def handler(event):
            pass

        sync1.register_handler("test", handler)
        sync1.queue_event(SyncEvent(name="test", source_node_id="n1", normalized_time=0.5))

        assert "test" in sync1.event_handlers
        assert "test" not in sync2.event_handlers
        assert len(sync1._pending_events) == 1
        assert len(sync2._pending_events) == 0


# =============================================================================
# EventSynchronizer.register_handler CREATES LIST
# =============================================================================


class TestRegisterHandlerCreatesList:
    """Tests for register_handler creating new handler lists."""

    def test_C1_new_event_name_creates_list(self):
        """Path C1: new event_name creates new list."""
        sync = EventSynchronizer()

        def handler(event):
            pass

        sync.register_handler("footstep", handler)
        assert isinstance(sync.event_handlers["footstep"], list)

    def test_C2_event_handlers_contains_new_key(self):
        """Path C2: event_handlers dict contains new event_name key."""
        sync = EventSynchronizer()

        def handler(event):
            pass

        sync.register_handler("impact", handler)
        assert "impact" in sync.event_handlers

    def test_C3_handler_added_to_new_list(self):
        """Path C3: handler added to newly created list."""
        sync = EventSynchronizer()

        def my_handler(event):
            pass

        sync.register_handler("test", my_handler)
        assert my_handler in sync.event_handlers["test"]

    def test_C4_list_length_one_after_first_registration(self):
        """Path C4: list has length 1 after first registration."""
        sync = EventSynchronizer()

        def handler(event):
            pass

        sync.register_handler("event", handler)
        assert len(sync.event_handlers["event"]) == 1


# =============================================================================
# EventSynchronizer.register_handler MULTIPLE HANDLERS
# =============================================================================


class TestRegisterHandlerMultiple:
    """Tests for register_handler with multiple handlers."""

    def test_D1_second_handler_appended(self):
        """Path D1: second handler for same event appended."""
        sync = EventSynchronizer()

        def handler1(event):
            pass

        def handler2(event):
            pass

        sync.register_handler("test", handler1)
        sync.register_handler("test", handler2)
        assert handler1 in sync.event_handlers["test"]
        assert handler2 in sync.event_handlers["test"]

    def test_D2_list_length_two_after_two_registrations(self):
        """Path D2: list has length 2 after two registrations."""
        sync = EventSynchronizer()

        def handler1(event):
            pass

        def handler2(event):
            pass

        sync.register_handler("test", handler1)
        sync.register_handler("test", handler2)
        assert len(sync.event_handlers["test"]) == 2

    def test_D3_handlers_preserved_in_order(self):
        """Path D3: handlers preserved in registration order."""
        sync = EventSynchronizer()
        handlers_order = []

        def handler1(event):
            handlers_order.append(1)

        def handler2(event):
            handlers_order.append(2)

        def handler3(event):
            handlers_order.append(3)

        sync.register_handler("test", handler1)
        sync.register_handler("test", handler2)
        sync.register_handler("test", handler3)

        assert sync.event_handlers["test"][0] is handler1
        assert sync.event_handlers["test"][1] is handler2
        assert sync.event_handlers["test"][2] is handler3

    def test_D4_same_handler_registered_multiple_times(self):
        """Path D4: same handler can be registered multiple times."""
        sync = EventSynchronizer()

        def handler(event):
            pass

        sync.register_handler("test", handler)
        sync.register_handler("test", handler)
        assert len(sync.event_handlers["test"]) == 2
        assert sync.event_handlers["test"].count(handler) == 2

    def test_D5_different_events_separate_lists(self):
        """Path D5: different events have separate handler lists."""
        sync = EventSynchronizer()

        def handler_a(event):
            pass

        def handler_b(event):
            pass

        sync.register_handler("event_a", handler_a)
        sync.register_handler("event_b", handler_b)

        assert handler_a in sync.event_handlers["event_a"]
        assert handler_a not in sync.event_handlers["event_b"]
        assert handler_b in sync.event_handlers["event_b"]
        assert handler_b not in sync.event_handlers["event_a"]


# =============================================================================
# EventSynchronizer.unregister_handler RETURNS TRUE/FALSE
# =============================================================================


class TestUnregisterHandlerReturn:
    """Tests for unregister_handler return values."""

    def test_E1_unregister_existing_returns_true(self):
        """Path E1: unregister existing handler returns True."""
        sync = EventSynchronizer()

        def handler(event):
            pass

        sync.register_handler("test", handler)
        result = sync.unregister_handler("test", handler)
        assert result is True

    def test_E2_unregister_nonexistent_handler_returns_false(self):
        """Path E2: unregister non-existent handler returns False."""
        sync = EventSynchronizer()

        def handler1(event):
            pass

        def handler2(event):
            pass

        sync.register_handler("test", handler1)
        result = sync.unregister_handler("test", handler2)
        assert result is False

    def test_E3_unregister_from_nonexistent_event_returns_false(self):
        """Path E3: unregister from non-existent event returns False."""
        sync = EventSynchronizer()

        def handler(event):
            pass

        result = sync.unregister_handler("nonexistent", handler)
        assert result is False

    def test_E4_handler_removed_from_list(self):
        """Path E4: handler removed from list after unregister."""
        sync = EventSynchronizer()

        def handler(event):
            pass

        sync.register_handler("test", handler)
        sync.unregister_handler("test", handler)
        assert handler not in sync.event_handlers["test"]


# =============================================================================
# EventSynchronizer.unregister_handler EDGE CASES
# =============================================================================


class TestUnregisterHandlerEdgeCases:
    """Tests for unregister_handler edge cases."""

    def test_F1_unregister_first_of_multiple(self):
        """Path F1: unregister first of multiple handlers."""
        sync = EventSynchronizer()

        def handler1(event):
            pass

        def handler2(event):
            pass

        def handler3(event):
            pass

        sync.register_handler("test", handler1)
        sync.register_handler("test", handler2)
        sync.register_handler("test", handler3)

        sync.unregister_handler("test", handler1)

        assert handler1 not in sync.event_handlers["test"]
        assert handler2 in sync.event_handlers["test"]
        assert handler3 in sync.event_handlers["test"]
        assert len(sync.event_handlers["test"]) == 2

    def test_F2_unregister_last_of_multiple(self):
        """Path F2: unregister last of multiple handlers."""
        sync = EventSynchronizer()

        def handler1(event):
            pass

        def handler2(event):
            pass

        sync.register_handler("test", handler1)
        sync.register_handler("test", handler2)

        sync.unregister_handler("test", handler2)

        assert handler1 in sync.event_handlers["test"]
        assert handler2 not in sync.event_handlers["test"]

    def test_F3_unregister_middle_handler(self):
        """Path F3: unregister middle handler."""
        sync = EventSynchronizer()

        def handler1(event):
            pass

        def handler2(event):
            pass

        def handler3(event):
            pass

        sync.register_handler("test", handler1)
        sync.register_handler("test", handler2)
        sync.register_handler("test", handler3)

        sync.unregister_handler("test", handler2)

        assert sync.event_handlers["test"] == [handler1, handler3]

    def test_F4_unregister_duplicate_removes_only_one(self):
        """Path F4: unregister duplicate handler removes only one."""
        sync = EventSynchronizer()

        def handler(event):
            pass

        sync.register_handler("test", handler)
        sync.register_handler("test", handler)
        sync.register_handler("test", handler)

        result = sync.unregister_handler("test", handler)

        assert result is True
        assert sync.event_handlers["test"].count(handler) == 2

    def test_F5_unregister_same_handler_twice(self):
        """Path F5: unregister same handler twice (second returns False)."""
        sync = EventSynchronizer()

        def handler(event):
            pass

        sync.register_handler("test", handler)

        result1 = sync.unregister_handler("test", handler)
        result2 = sync.unregister_handler("test", handler)

        assert result1 is True
        assert result2 is False

    def test_F6_empty_event_handlers_returns_false(self):
        """Path F6: empty event_handlers dict returns False."""
        sync = EventSynchronizer()

        def handler(event):
            pass

        result = sync.unregister_handler("test", handler)
        assert result is False


# =============================================================================
# EventSynchronizer.queue_event
# =============================================================================


class TestQueueEvent:
    """Tests for queue_event adding events to pending list."""

    def test_G1_queue_single_event(self):
        """Path G1: queue single event adds to _pending_events."""
        sync = EventSynchronizer()
        event = SyncEvent(name="test", source_node_id="node1", normalized_time=0.5)

        sync.queue_event(event)

        assert len(sync._pending_events) == 1
        assert sync._pending_events[0] is event

    def test_G2_queue_multiple_events_accumulates(self):
        """Path G2: queue multiple events accumulates."""
        sync = EventSynchronizer()
        event1 = SyncEvent(name="test1", source_node_id="node1", normalized_time=0.5)
        event2 = SyncEvent(name="test2", source_node_id="node2", normalized_time=0.6)
        event3 = SyncEvent(name="test3", source_node_id="node3", normalized_time=0.7)

        sync.queue_event(event1)
        sync.queue_event(event2)
        sync.queue_event(event3)

        assert len(sync._pending_events) == 3

    def test_G3_events_preserved_in_order(self):
        """Path G3: events preserved in queue order."""
        sync = EventSynchronizer()
        events = [
            SyncEvent(name=f"event_{i}", source_node_id=f"node{i}", normalized_time=i * 0.1)
            for i in range(5)
        ]

        for event in events:
            sync.queue_event(event)

        for i, event in enumerate(events):
            assert sync._pending_events[i] is event

    def test_G4_same_event_queued_multiple_times(self):
        """Path G4: same event can be queued multiple times."""
        sync = EventSynchronizer()
        event = SyncEvent(name="test", source_node_id="node1", normalized_time=0.5)

        sync.queue_event(event)
        sync.queue_event(event)
        sync.queue_event(event)

        assert len(sync._pending_events) == 3
        assert all(e is event for e in sync._pending_events)

    def test_G5_queue_does_not_modify_event(self):
        """Path G5: queue does not modify event."""
        sync = EventSynchronizer()
        event = SyncEvent(
            name="original",
            source_node_id="node_original",
            normalized_time=0.5,
            data={"key": "value"}
        )

        sync.queue_event(event)

        assert event.name == "original"
        assert event.source_node_id == "node_original"
        assert event.normalized_time == 0.5
        assert event.data == {"key": "value"}


# =============================================================================
# EventSynchronizer.process_events DEDUPLICATION
# =============================================================================


class TestProcessEventsDeduplication:
    """Tests for process_events deduplication logic."""

    def test_H1_identical_events_deduplicated(self):
        """Path H1: identical events deduplicated."""
        sync = EventSynchronizer()
        received_events: List[SyncEvent] = []

        def handler(event):
            received_events.append(event)

        sync.register_handler("test", handler)

        # Queue identical events
        event1 = SyncEvent(name="test", source_node_id="node1", normalized_time=0.5)
        event2 = SyncEvent(name="test", source_node_id="node2", normalized_time=0.5)

        sync.queue_event(event1)
        sync.queue_event(event2)
        sync.process_events()

        # Only one should be dispatched (first one)
        assert len(received_events) == 1

    def test_H2_same_name_different_rounded_time_kept(self):
        """Path H2: events with same name but different rounded_time kept."""
        sync = EventSynchronizer()
        received_events: List[SyncEvent] = []

        def handler(event):
            received_events.append(event)

        sync.register_handler("test", handler)

        event1 = SyncEvent(name="test", source_node_id="node1", normalized_time=0.50)
        event2 = SyncEvent(name="test", source_node_id="node2", normalized_time=0.60)

        sync.queue_event(event1)
        sync.queue_event(event2)
        sync.process_events()

        assert len(received_events) == 2

    def test_H3_different_name_same_rounded_time_kept(self):
        """Path H3: events with different name but same rounded_time kept."""
        sync = EventSynchronizer()
        received_names: List[str] = []

        def handler(event):
            received_names.append(event.name)

        sync.register_handler("event_a", handler)
        sync.register_handler("event_b", handler)

        event1 = SyncEvent(name="event_a", source_node_id="node1", normalized_time=0.5)
        event2 = SyncEvent(name="event_b", source_node_id="node2", normalized_time=0.5)

        sync.queue_event(event1)
        sync.queue_event(event2)
        sync.process_events()

        assert "event_a" in received_names
        assert "event_b" in received_names

    def test_H4_normalized_time_rounded_to_2_decimals(self):
        """Path H4: normalized_time rounded to 2 decimal places."""
        sync = EventSynchronizer()
        received_events: List[SyncEvent] = []

        def handler(event):
            received_events.append(event)

        sync.register_handler("test", handler)

        # These should round to the same value (0.50)
        event1 = SyncEvent(name="test", source_node_id="node1", normalized_time=0.501)
        event2 = SyncEvent(name="test", source_node_id="node2", normalized_time=0.504)

        sync.queue_event(event1)
        sync.queue_event(event2)
        sync.process_events()

        # Both round to 0.50, so deduplicated
        assert len(received_events) == 1

    def test_H5_001_and_004_round_to_same(self):
        """Path H5: 0.001 and 0.004 round to same value (deduplicated)."""
        sync = EventSynchronizer()
        received_events: List[SyncEvent] = []

        def handler(event):
            received_events.append(event)

        sync.register_handler("test", handler)

        # 0.001 rounds to 0.00, 0.004 rounds to 0.00
        event1 = SyncEvent(name="test", source_node_id="node1", normalized_time=0.001)
        event2 = SyncEvent(name="test", source_node_id="node2", normalized_time=0.004)

        sync.queue_event(event1)
        sync.queue_event(event2)
        sync.process_events()

        assert len(received_events) == 1

    def test_H6_001_and_006_round_to_different(self):
        """Path H6: 0.001 and 0.006 round to different values (both kept)."""
        sync = EventSynchronizer()
        received_events: List[SyncEvent] = []

        def handler(event):
            received_events.append(event)

        sync.register_handler("test", handler)

        # 0.001 rounds to 0.00, 0.006 rounds to 0.01
        event1 = SyncEvent(name="test", source_node_id="node1", normalized_time=0.001)
        event2 = SyncEvent(name="test", source_node_id="node2", normalized_time=0.006)

        sync.queue_event(event1)
        sync.queue_event(event2)
        sync.process_events()

        assert len(received_events) == 2

    def test_H7_negative_time_handled(self):
        """Path H7: negative time differences handled correctly."""
        sync = EventSynchronizer()
        received_events: List[SyncEvent] = []

        def handler(event):
            received_events.append(event)

        sync.register_handler("test", handler)

        # -0.501 rounds to -0.50, -0.504 also rounds to -0.50
        event1 = SyncEvent(name="test", source_node_id="node1", normalized_time=-0.501)
        event2 = SyncEvent(name="test", source_node_id="node2", normalized_time=-0.504)

        sync.queue_event(event1)
        sync.queue_event(event2)
        sync.process_events()

        assert len(received_events) == 1


# =============================================================================
# EventSynchronizer.process_events DISPATCH
# =============================================================================


class TestProcessEventsDispatch:
    """Tests for process_events dispatching to handlers."""

    def test_I1_handler_called_for_matching_event(self):
        """Path I1: handler called for matching event."""
        sync = EventSynchronizer()
        called = [False]

        def handler(event):
            called[0] = True

        sync.register_handler("test", handler)
        sync.queue_event(SyncEvent(name="test", source_node_id="node1", normalized_time=0.5))
        sync.process_events()

        assert called[0] is True

    def test_I2_handler_receives_correct_event(self):
        """Path I2: handler receives correct event object."""
        sync = EventSynchronizer()
        received_event = [None]

        def handler(event):
            received_event[0] = event

        sync.register_handler("test", handler)
        original_event = SyncEvent(
            name="test",
            source_node_id="specific_node",
            normalized_time=0.75,
            data={"marker": "start"}
        )
        sync.queue_event(original_event)
        sync.process_events()

        assert received_event[0] is original_event

    def test_I3_all_handlers_for_event_called(self):
        """Path I3: all handlers for event called."""
        sync = EventSynchronizer()
        call_count = [0]

        def handler1(event):
            call_count[0] += 1

        def handler2(event):
            call_count[0] += 1

        def handler3(event):
            call_count[0] += 1

        sync.register_handler("test", handler1)
        sync.register_handler("test", handler2)
        sync.register_handler("test", handler3)

        sync.queue_event(SyncEvent(name="test", source_node_id="node1", normalized_time=0.5))
        sync.process_events()

        assert call_count[0] == 3

    def test_I4_handler_order_preserved(self):
        """Path I4: handler order is preserved (FIFO)."""
        sync = EventSynchronizer()
        call_order: List[int] = []

        def handler1(event):
            call_order.append(1)

        def handler2(event):
            call_order.append(2)

        def handler3(event):
            call_order.append(3)

        sync.register_handler("test", handler1)
        sync.register_handler("test", handler2)
        sync.register_handler("test", handler3)

        sync.queue_event(SyncEvent(name="test", source_node_id="node1", normalized_time=0.5))
        sync.process_events()

        assert call_order == [1, 2, 3]

    def test_I5_events_dispatched_in_queue_order(self):
        """Path I5: events dispatched in queue order (after dedup)."""
        sync = EventSynchronizer()
        received_times: List[float] = []

        def handler(event):
            received_times.append(event.normalized_time)

        sync.register_handler("test", handler)

        # Queue events with different times
        for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
            sync.queue_event(SyncEvent(name="test", source_node_id="node1", normalized_time=t))

        sync.process_events()

        assert received_times == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_I6_no_handlers_no_error(self):
        """Path I6: no handlers means no dispatch (no error)."""
        sync = EventSynchronizer()

        sync.queue_event(SyncEvent(name="unhandled", source_node_id="node1", normalized_time=0.5))

        # Should not raise
        sync.process_events()

        # Pending events should be cleared
        assert len(sync._pending_events) == 0


# =============================================================================
# EventSynchronizer.process_events CLEARS PENDING
# =============================================================================


class TestProcessEventsClearsPending:
    """Tests for process_events clearing pending list."""

    def test_J1_pending_empty_after_process(self):
        """Path J1: _pending_events empty after process_events."""
        sync = EventSynchronizer()

        def handler(event):
            pass

        sync.register_handler("test", handler)
        sync.queue_event(SyncEvent(name="test", source_node_id="node1", normalized_time=0.5))
        sync.queue_event(SyncEvent(name="test", source_node_id="node2", normalized_time=0.6))

        sync.process_events()

        assert len(sync._pending_events) == 0

    def test_J2_cleared_even_if_no_handlers(self):
        """Path J2: _pending_events cleared even if no handlers."""
        sync = EventSynchronizer()

        sync.queue_event(SyncEvent(name="test", source_node_id="node1", normalized_time=0.5))
        sync.process_events()

        assert len(sync._pending_events) == 0

    def test_J3_cleared_even_if_handlers_throw(self):
        """Path J3: _pending_events cleared even if handlers throw."""
        sync = EventSynchronizer()

        def throwing_handler(event):
            raise ValueError("Handler error")

        sync.register_handler("test", throwing_handler)
        sync.queue_event(SyncEvent(name="test", source_node_id="node1", normalized_time=0.5))

        sync.process_events()

        assert len(sync._pending_events) == 0

    def test_J4_process_empty_queue_does_nothing(self):
        """Path J4: process_events on empty queue does nothing."""
        sync = EventSynchronizer()
        call_count = [0]

        def handler(event):
            call_count[0] += 1

        sync.register_handler("test", handler)
        sync.process_events()

        assert call_count[0] == 0


# =============================================================================
# EventSynchronizer.process_events EXCEPTION HANDLING
# =============================================================================


class TestProcessEventsExceptionHandling:
    """Tests for process_events exception handling."""

    def test_K1_handler_exception_caught(self):
        """Path K1: handler exception caught and ignored."""
        sync = EventSynchronizer()

        def throwing_handler(event):
            raise RuntimeError("Handler crashed")

        sync.register_handler("test", throwing_handler)
        sync.queue_event(SyncEvent(name="test", source_node_id="node1", normalized_time=0.5))

        # Should not raise
        sync.process_events()

    def test_K2_subsequent_handlers_called_after_exception(self):
        """Path K2: subsequent handlers still called after exception."""
        sync = EventSynchronizer()
        called_after = [False]

        def throwing_handler(event):
            raise RuntimeError("First handler crashed")

        def second_handler(event):
            called_after[0] = True

        sync.register_handler("test", throwing_handler)
        sync.register_handler("test", second_handler)
        sync.queue_event(SyncEvent(name="test", source_node_id="node1", normalized_time=0.5))

        sync.process_events()

        assert called_after[0] is True

    def test_K3_subsequent_events_processed_after_exception(self):
        """Path K3: subsequent events still processed after exception."""
        sync = EventSynchronizer()
        processed_events: List[str] = []

        def throwing_handler(event):
            if event.name == "crash":
                raise RuntimeError("Crash event handler")
            processed_events.append(event.name)

        sync.register_handler("crash", throwing_handler)
        sync.register_handler("safe", throwing_handler)

        sync.queue_event(SyncEvent(name="crash", source_node_id="node1", normalized_time=0.1))
        sync.queue_event(SyncEvent(name="safe", source_node_id="node2", normalized_time=0.2))

        sync.process_events()

        assert "safe" in processed_events

    def test_K4_multiple_exceptions_all_caught(self):
        """Path K4: multiple exceptions in handlers all caught."""
        sync = EventSynchronizer()

        def thrower1(event):
            raise ValueError("Error 1")

        def thrower2(event):
            raise TypeError("Error 2")

        def thrower3(event):
            raise RuntimeError("Error 3")

        sync.register_handler("test", thrower1)
        sync.register_handler("test", thrower2)
        sync.register_handler("test", thrower3)

        sync.queue_event(SyncEvent(name="test", source_node_id="node1", normalized_time=0.5))

        # Should not raise any of the exceptions
        sync.process_events()


# =============================================================================
# EventSynchronizer.process_events SYNC_GROUP PARAMETER
# =============================================================================


class TestProcessEventsSyncGroup:
    """Tests for process_events sync_group parameter."""

    def test_L1_sync_group_none_processes_all(self):
        """Path L1: sync_group=None processes all events."""
        sync = EventSynchronizer()
        processed: List[str] = []

        def handler(event):
            processed.append(event.name)

        sync.register_handler("event1", handler)
        sync.register_handler("event2", handler)
        sync.register_handler("event3", handler)

        sync.queue_event(SyncEvent(name="event1", source_node_id="node1", normalized_time=0.1))
        sync.queue_event(SyncEvent(name="event2", source_node_id="node2", normalized_time=0.2))
        sync.queue_event(SyncEvent(name="event3", source_node_id="node3", normalized_time=0.3))

        sync.process_events(sync_group=None)

        assert set(processed) == {"event1", "event2", "event3"}

    def test_L2_accepts_sync_group_parameter(self):
        """Path L2: process_events accepts sync_group parameter."""
        sync = EventSynchronizer()

        def handler(event):
            pass

        sync.register_handler("test", handler)
        sync.queue_event(SyncEvent(name="test", source_node_id="node1", normalized_time=0.5))

        # Should accept any object as sync_group (currently unused in implementation)
        sync.process_events(sync_group="some_group")
        sync.process_events(sync_group=123)
        sync.process_events(sync_group={"key": "value"})


# =============================================================================
# EventSynchronizer.clear
# =============================================================================


class TestEventSynchronizerClear:
    """Tests for EventSynchronizer.clear method."""

    def test_M1_clear_empties_pending_events(self):
        """Path M1: clear empties _pending_events."""
        sync = EventSynchronizer()

        sync.queue_event(SyncEvent(name="test1", source_node_id="node1", normalized_time=0.1))
        sync.queue_event(SyncEvent(name="test2", source_node_id="node2", normalized_time=0.2))
        sync.queue_event(SyncEvent(name="test3", source_node_id="node3", normalized_time=0.3))

        assert len(sync._pending_events) == 3

        sync.clear()

        assert len(sync._pending_events) == 0

    def test_M2_clear_on_empty_list_succeeds(self):
        """Path M2: clear on empty list succeeds."""
        sync = EventSynchronizer()

        assert len(sync._pending_events) == 0

        # Should not raise
        sync.clear()

        assert len(sync._pending_events) == 0

    def test_M3_clear_does_not_affect_handlers(self):
        """Path M3: clear does not affect event_handlers."""
        sync = EventSynchronizer()

        def handler(event):
            pass

        sync.register_handler("test", handler)
        sync.queue_event(SyncEvent(name="test", source_node_id="node1", normalized_time=0.5))

        sync.clear()

        assert "test" in sync.event_handlers
        assert handler in sync.event_handlers["test"]

    def test_M4_events_can_be_queued_after_clear(self):
        """Path M4: events can be queued after clear."""
        sync = EventSynchronizer()

        sync.queue_event(SyncEvent(name="old", source_node_id="node1", normalized_time=0.1))
        sync.clear()

        sync.queue_event(SyncEvent(name="new", source_node_id="node2", normalized_time=0.2))

        assert len(sync._pending_events) == 1
        assert sync._pending_events[0].name == "new"


# =============================================================================
# ADDITIONAL EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Additional edge case tests for completeness."""

    def test_register_handler_lambda(self):
        """Test that lambda handlers work correctly."""
        sync = EventSynchronizer()
        received = []

        sync.register_handler("test", lambda e: received.append(e.name))
        sync.queue_event(SyncEvent(name="test", source_node_id="node1", normalized_time=0.5))
        sync.process_events()

        assert received == ["test"]

    def test_process_events_multiple_times(self):
        """Test that process_events can be called multiple times."""
        sync = EventSynchronizer()
        call_count = [0]

        def handler(event):
            call_count[0] += 1

        sync.register_handler("test", handler)

        sync.queue_event(SyncEvent(name="test", source_node_id="node1", normalized_time=0.1))
        sync.process_events()

        sync.queue_event(SyncEvent(name="test", source_node_id="node2", normalized_time=0.2))
        sync.process_events()

        sync.queue_event(SyncEvent(name="test", source_node_id="node3", normalized_time=0.3))
        sync.process_events()

        assert call_count[0] == 3

    def test_handler_modifies_event_data(self):
        """Test that handler can modify event data."""
        sync = EventSynchronizer()

        def modifying_handler(event):
            event.data["modified"] = True

        sync.register_handler("test", modifying_handler)
        event = SyncEvent(name="test", source_node_id="node1", normalized_time=0.5)
        sync.queue_event(event)
        sync.process_events()

        assert event.data.get("modified") is True

    def test_deduplication_preserves_first_event(self):
        """Test that deduplication preserves the first queued event."""
        sync = EventSynchronizer()
        received_sources: List[str] = []

        def handler(event):
            received_sources.append(event.source_node_id)

        sync.register_handler("test", handler)

        # Queue duplicate events with same name and time but different sources
        event1 = SyncEvent(name="test", source_node_id="first_node", normalized_time=0.5)
        event2 = SyncEvent(name="test", source_node_id="second_node", normalized_time=0.5)
        event3 = SyncEvent(name="test", source_node_id="third_node", normalized_time=0.5)

        sync.queue_event(event1)
        sync.queue_event(event2)
        sync.queue_event(event3)
        sync.process_events()

        assert received_sources == ["first_node"]

    def test_many_events_performance(self):
        """Test handling many events efficiently."""
        sync = EventSynchronizer()
        received_count = [0]

        def handler(event):
            received_count[0] += 1

        sync.register_handler("test", handler)

        # Queue 100 events with unique rounded times (spaced 0.01 apart)
        for i in range(100):
            sync.queue_event(SyncEvent(
                name="test",
                source_node_id=f"node{i}",
                normalized_time=i * 0.01
            ))

        sync.process_events()

        # All should be processed (different rounded times)
        assert received_count[0] == 100

    def test_rounding_boundary_cases(self):
        """Test rounding at various boundaries."""
        sync = EventSynchronizer()
        received_times: List[float] = []

        def handler(event):
            received_times.append(event.normalized_time)

        sync.register_handler("test", handler)

        # These times each round to clearly different values at 2 decimal places
        # 0.10, 0.20, 0.30, 0.40, 0.50
        times = [0.10, 0.20, 0.30, 0.40, 0.50]
        for t in times:
            sync.queue_event(SyncEvent(name="test", source_node_id="node", normalized_time=t))

        sync.process_events()

        # Each rounds to a different hundredth
        assert len(received_times) == 5

    def test_zero_and_one_time_events(self):
        """Test events at exactly 0.0 and 1.0 normalized time."""
        sync = EventSynchronizer()
        received_times: List[float] = []

        def handler(event):
            received_times.append(event.normalized_time)

        sync.register_handler("test", handler)

        sync.queue_event(SyncEvent(name="test", source_node_id="start", normalized_time=0.0))
        sync.queue_event(SyncEvent(name="test", source_node_id="end", normalized_time=1.0))

        sync.process_events()

        assert 0.0 in received_times
        assert 1.0 in received_times

    def test_handler_with_closure(self):
        """Test handler that captures variables via closure."""
        sync = EventSynchronizer()
        results = {"sum": 0}

        def create_handler(multiplier):
            def handler(event):
                results["sum"] += event.normalized_time * multiplier
            return handler

        sync.register_handler("test", create_handler(2))
        sync.register_handler("test", create_handler(3))

        sync.queue_event(SyncEvent(name="test", source_node_id="node", normalized_time=0.5))
        sync.process_events()

        # 0.5 * 2 + 0.5 * 3 = 2.5
        assert results["sum"] == 2.5
