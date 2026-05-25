"""Tests for engine.core.ecs.event_bus -- EventBus."""

from engine.core.ecs.event_bus import EventBus


class _MyEvent:
    def __init__(self, value: int = 0):
        self.value = value


class _OtherEvent:
    pass


class TestEventBus:
    def test_emit_and_drain(self):
        bus = EventBus()
        bus.emit(_MyEvent(1))
        bus.emit(_MyEvent(2))
        events = bus.drain(_MyEvent)
        assert len(events) == 2
        assert events[0].value == 1
        assert events[1].value == 2
        # Drain again should be empty
        assert bus.drain(_MyEvent) == []

    def test_subscribe_callback_called(self):
        bus = EventBus()
        received = []
        bus.subscribe(_MyEvent, lambda e: received.append(e.value))
        bus.emit(_MyEvent(42))
        assert received == [42]

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        cb = lambda e: received.append(e.value)
        bus.subscribe(_MyEvent, cb)
        bus.emit(_MyEvent(1))
        assert received == [1]
        bus.unsubscribe(_MyEvent, cb)
        bus.emit(_MyEvent(99))
        assert received == [1]  # callback not called after unsubscribe

    def test_clear_clears_all(self):
        bus = EventBus()
        received = []
        bus.subscribe(_MyEvent, lambda e: received.append(e.value))
        bus.emit(_MyEvent(1))
        bus.emit(_OtherEvent())
        bus.clear()
        assert bus.drain(_MyEvent) == []
        assert bus.drain(_OtherEvent) == []
        # Subscribers should also be cleared
        bus.emit(_MyEvent(2))
        assert received == [1]  # callback not called after clear

    def test_clear_events_only(self):
        """clear_events() should clear queues but preserve subscribers."""
        bus = EventBus()
        received = []
        bus.subscribe(_MyEvent, lambda e: received.append(e.value))
        bus.emit(_MyEvent(1))
        bus.clear_events()
        assert bus.drain(_MyEvent) == []
        # Subscribers should still be active after clear_events
        bus.emit(_MyEvent(2))
        assert 2 in received
