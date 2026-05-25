"""
Tests for Phase 7 descriptors: Priority, Mirror, EventSourced, Batched, Broadcast.
"""

from __future__ import annotations

import pytest

from trinity.decorators.ops import Op
from trinity.descriptors.priority import PriorityDescriptor
from trinity.descriptors.mirror import MirrorDescriptor
from trinity.descriptors.event_sourced import (
    EventSourcedDescriptor,
    get_events,
    replay_events,
)
from trinity.descriptors.batched import BatchedDescriptor, flush_batch
from trinity.descriptors.broadcast import (
    BroadcastDescriptor,
    subscribe,
    unsubscribe,
)


# =============================================================================
# PriorityDescriptor
# =============================================================================


class TestPriorityDescriptor:
    def test_tag_value(self):
        desc = PriorityDescriptor(priority=5)
        steps = desc.descriptor_steps
        assert len(steps) == 1
        assert steps[0].op == Op.TAG
        assert steps[0].args["key"] == "priority"
        assert steps[0].args["value"] == 5

    def test_metadata(self):
        desc = PriorityDescriptor(priority=10)
        meta = desc.get_metadata()
        assert meta["priority"] == 10

    def test_default_priority(self):
        desc = PriorityDescriptor()
        assert desc.descriptor_steps[0].args["value"] == 0

    def test_get_set(self):
        class Obj:
            val = PriorityDescriptor(priority=3)

        Obj.val.__set_name__(Obj, "val")
        o = Obj()
        o.val = 42
        assert o.val == 42


# =============================================================================
# MirrorDescriptor
# =============================================================================


class TestMirrorDescriptor:
    def test_reads_from_source(self):
        class Obj:
            source = "hello"
            mirror = MirrorDescriptor(source_field="source")

        Obj.mirror.__set_name__(Obj, "mirror")
        o = Obj()
        o.source = "world"
        assert o.mirror == "world"

    def test_set_raises(self):
        class Obj:
            source = "hello"
            mirror = MirrorDescriptor(source_field="source")

        Obj.mirror.__set_name__(Obj, "mirror")
        o = Obj()
        with pytest.raises(AttributeError, match="Cannot set mirrored field"):
            o.mirror = "nope"

    def test_steps(self):
        desc = MirrorDescriptor(source_field="x")
        steps = desc.descriptor_steps
        assert len(steps) == 1
        assert steps[0].op == Op.INTERCEPT
        assert steps[0].args["set"] == "deny"


# =============================================================================
# EventSourcedDescriptor
# =============================================================================


class TestEventSourcedDescriptor:
    def _make(self, max_events=1000):
        class Obj:
            val = EventSourcedDescriptor(max_events=max_events)

        Obj.val.__set_name__(Obj, "val")
        return Obj, Obj()

    def test_events_recorded(self):
        Cls, o = self._make()
        o.val = 1
        o.val = 2
        events = get_events(o, "val")
        assert len(events) == 2
        assert events[0]["old"] is None
        assert events[0]["new"] == 1
        assert events[1]["old"] == 1
        assert events[1]["new"] == 2

    def test_get_events_limit(self):
        Cls, o = self._make()
        for i in range(5):
            o.val = i
        events = get_events(o, "val", limit=2)
        assert len(events) == 2
        assert events[0]["new"] == 3
        assert events[1]["new"] == 4

    def test_max_trimming(self):
        Cls, o = self._make(max_events=3)
        for i in range(10):
            o.val = i
        events = get_events(o, "val")
        assert len(events) == 3
        assert events[0]["new"] == 7

    def test_replay_events(self):
        Cls, o = self._make()
        o.val = "a"
        o.val = "b"
        o.val = "c"
        assert replay_events(o, "val") == "c"

    def test_steps(self):
        desc = EventSourcedDescriptor()
        desc.__set_name__(type("X", (), {}), "x")
        steps = desc.descriptor_steps
        assert len(steps) == 2
        assert steps[0].op == Op.TRACK
        assert steps[1].op == Op.TAG
        assert steps[1].args["key"] == "event_sourced"


# =============================================================================
# BatchedDescriptor
# =============================================================================


class TestBatchedDescriptor:
    def test_counter_increments(self):
        class Obj:
            val = BatchedDescriptor(batch_size=5)

        Obj.val.__set_name__(Obj, "val")
        o = Obj()
        o.val = 1
        o.val = 2
        o.val = 3
        assert o.__dict__["_batch_count_val"] == 3
        assert len(o.__dict__["_batch_val"]) == 3

    def test_flush_at_batch_size(self):
        flushed = []

        class Obj:
            val = BatchedDescriptor(batch_size=3)

            def _flush_batch(self, field_name):
                flushed.append(field_name)

        Obj.val.__set_name__(Obj, "val")
        o = Obj()
        o.val = 1
        o.val = 2
        assert len(flushed) == 0
        o.val = 3  # triggers flush
        assert len(flushed) == 1
        assert flushed[0] == "val"
        # Counter and batch reset
        assert o.__dict__["_batch_count_val"] == 0
        assert o.__dict__["_batch_val"] == []

    def test_manual_flush(self):
        flushed = []

        class Obj:
            val = BatchedDescriptor(batch_size=100)

            def _flush_batch(self, field_name):
                flushed.append(field_name)

        Obj.val.__set_name__(Obj, "val")
        o = Obj()
        o.val = 1
        flush_batch(o, "val")
        assert len(flushed) == 1

    def test_steps(self):
        desc = BatchedDescriptor()
        steps = desc.descriptor_steps
        assert steps[0].op == Op.INTERCEPT
        assert steps[1].op == Op.TAG


# =============================================================================
# BroadcastDescriptor
# =============================================================================


class TestBroadcastDescriptor:
    def test_subscribe_and_notify(self):
        notifications = []

        class Obj:
            val = BroadcastDescriptor(channel="test")

        Obj.val.__set_name__(Obj, "val")

        def on_change(obj, name, old, new):
            notifications.append((name, old, new))

        subscribe(Obj, "val", on_change)
        o = Obj()
        o.val = 42
        assert len(notifications) == 1
        assert notifications[0] == ("val", None, 42)

        o.val = 99
        assert len(notifications) == 2
        assert notifications[1] == ("val", 42, 99)

        # Cleanup
        unsubscribe(Obj, "val", on_change)

    def test_unsubscribe(self):
        notifications = []

        class Obj:
            val = BroadcastDescriptor(channel="test2")

        Obj.val.__set_name__(Obj, "val")

        def on_change(obj, name, old, new):
            notifications.append(new)

        subscribe(Obj, "val", on_change)
        o = Obj()
        o.val = 1
        assert len(notifications) == 1

        unsubscribe(Obj, "val", on_change)
        o.val = 2
        assert len(notifications) == 1  # No new notification

    def test_steps(self):
        desc = BroadcastDescriptor(channel="ch1")
        steps = desc.descriptor_steps
        assert len(steps) == 2
        assert steps[0].op == Op.HOOK
        assert steps[1].op == Op.TAG
        assert steps[1].args["key"] == "broadcast_channel"
        assert steps[1].args["value"] == "ch1"

    def test_metadata(self):
        desc = BroadcastDescriptor(channel="events")
        meta = desc.get_metadata()
        assert meta["channel"] == "events"
