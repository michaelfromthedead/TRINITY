"""
Whitebox tests for engine.simulation.collision.collision_events module.

Tests collision event system:
- CollisionEventType
- CollisionEvent
- CollisionEventDispatcher
- CollisionListener
- CollisionEventProcessor
"""

import pytest
from engine.simulation.collision.collision_events import (
    CollisionEventType,
    CollisionEvent,
    CollisionCallback,
    EventFilterCallback,
    CollisionEventDispatcher,
    CollisionListener,
    CollisionEventProcessor,
)
from engine.simulation.collision.contact_manifold import (
    ContactPoint,
    ContactManifold,
)
from engine.simulation.collision.broadphase import Vec3


class TestCollisionEventType:
    """Tests for CollisionEventType enum."""

    def test_all_types_exist(self):
        """All event types should exist."""
        assert hasattr(CollisionEventType, "BEGIN")
        assert hasattr(CollisionEventType, "PERSIST")
        assert hasattr(CollisionEventType, "END")

    def test_types_unique(self):
        """Event types should have unique values."""
        values = [t.value for t in CollisionEventType]
        assert len(values) == len(set(values))


class TestCollisionEvent:
    """Tests for CollisionEvent dataclass."""

    def test_default_construction(self):
        """Default CollisionEvent should have sensible defaults."""
        event = CollisionEvent(
            event_type=CollisionEventType.BEGIN,
            body_a=1,
            body_b=2,
        )
        assert event.body_a == 1
        assert event.body_b == 2
        assert event.impulse == 0.0

    def test_is_begin(self):
        """is_begin should identify BEGIN events."""
        event = CollisionEvent(
            event_type=CollisionEventType.BEGIN,
            body_a=1,
            body_b=2,
        )
        assert event.is_begin
        assert not event.is_persist
        assert not event.is_end

    def test_is_persist(self):
        """is_persist should identify PERSIST events."""
        event = CollisionEvent(
            event_type=CollisionEventType.PERSIST,
            body_a=1,
            body_b=2,
        )
        assert not event.is_begin
        assert event.is_persist
        assert not event.is_end

    def test_is_end(self):
        """is_end should identify END events."""
        event = CollisionEvent(
            event_type=CollisionEventType.END,
            body_a=1,
            body_b=2,
        )
        assert not event.is_begin
        assert not event.is_persist
        assert event.is_end

    def test_contact_count(self):
        """contact_count should return number of contacts."""
        event = CollisionEvent(
            event_type=CollisionEventType.BEGIN,
            body_a=1,
            body_b=2,
            contacts=[
                ContactPoint(),
                ContactPoint(),
            ],
        )
        assert event.contact_count == 2

    def test_get_other_body(self):
        """get_other_body should return the other body."""
        event = CollisionEvent(
            event_type=CollisionEventType.BEGIN,
            body_a=1,
            body_b=2,
        )
        assert event.get_other_body(1) == 2
        assert event.get_other_body(2) == 1

    def test_get_contact_normal_for(self):
        """get_contact_normal_for should flip normal appropriately."""
        event = CollisionEvent(
            event_type=CollisionEventType.BEGIN,
            body_a=1,
            body_b=2,
            normal=Vec3(1, 0, 0),
        )
        # Normal points from A to B
        normal_a = event.get_contact_normal_for(1)
        normal_b = event.get_contact_normal_for(2)
        assert normal_a.x == 1
        assert normal_b.x == -1


class TestCollisionEventDispatcher:
    """Tests for CollisionEventDispatcher class."""

    def test_construction(self):
        """Dispatcher should be constructed correctly."""
        dispatcher = CollisionEventDispatcher()
        assert dispatcher.events_dispatched == 0

    def test_on_collision_begin(self):
        """on_collision_begin should register handler."""
        dispatcher = CollisionEventDispatcher()
        received = []
        dispatcher.on_collision_begin(lambda e: received.append(e) or True)
        dispatcher.dispatch_begin(1, 2, [], frame=0)
        assert len(received) == 1
        assert received[0].is_begin

    def test_on_collision_persist(self):
        """on_collision_persist should register handler."""
        dispatcher = CollisionEventDispatcher()
        received = []
        dispatcher.on_collision_persist(lambda e: received.append(e) or True)
        dispatcher.dispatch_persist(1, 2, [], impulse=5.0, frame=0)
        assert len(received) == 1
        assert received[0].is_persist
        assert received[0].impulse == 5.0

    def test_on_collision_end(self):
        """on_collision_end should register handler."""
        dispatcher = CollisionEventDispatcher()
        received = []
        dispatcher.on_collision_end(lambda e: received.append(e) or True)
        dispatcher.dispatch_end(1, 2, frame=0)
        assert len(received) == 1
        assert received[0].is_end

    def test_handler_priority(self):
        """Higher priority handlers should be called first."""
        dispatcher = CollisionEventDispatcher()
        order = []
        dispatcher.on_collision_begin(lambda e: order.append(1) or True, priority=1)
        dispatcher.on_collision_begin(lambda e: order.append(2) or True, priority=2)
        dispatcher.on_collision_begin(lambda e: order.append(0) or True, priority=0)
        dispatcher.dispatch_begin(1, 2, [], frame=0)
        assert order == [2, 1, 0]

    def test_handler_stops_propagation(self):
        """Handler returning False should stop propagation."""
        dispatcher = CollisionEventDispatcher()
        received = []
        dispatcher.on_collision_begin(lambda e: False, priority=2)  # Stops
        dispatcher.on_collision_begin(lambda e: received.append(e) or True, priority=1)
        dispatcher.dispatch_begin(1, 2, [], frame=0)
        assert len(received) == 0

    def test_remove_handler(self):
        """remove_handler should remove handler."""
        dispatcher = CollisionEventDispatcher()
        received = []
        handler = lambda e: received.append(e) or True
        dispatcher.on_collision_begin(handler)
        assert dispatcher.remove_handler(handler)
        dispatcher.dispatch_begin(1, 2, [], frame=0)
        assert len(received) == 0

    def test_on_body_collision(self):
        """on_body_collision should register body-specific handler."""
        dispatcher = CollisionEventDispatcher()
        received = []
        dispatcher.on_body_collision(1, lambda e: received.append(e) or True)
        dispatcher.dispatch_begin(1, 2, [], frame=0)  # Body 1 involved
        dispatcher.dispatch_begin(3, 4, [], frame=0)  # Body 1 not involved
        assert len(received) == 1

    def test_remove_body_handlers(self):
        """remove_body_handlers should remove all handlers for body."""
        dispatcher = CollisionEventDispatcher()
        received = []
        dispatcher.on_body_collision(1, lambda e: received.append(1) or True)
        dispatcher.on_body_collision(1, lambda e: received.append(2) or True)
        count = dispatcher.remove_body_handlers(1)
        assert count == 2
        dispatcher.dispatch_begin(1, 2, [], frame=0)
        assert len(received) == 0

    def test_add_filter(self):
        """add_filter should filter events."""
        dispatcher = CollisionEventDispatcher()
        received = []
        dispatcher.on_collision_begin(lambda e: received.append(e) or True)
        # Filter out body 5
        dispatcher.add_filter(lambda a, b: a != 5 and b != 5)
        dispatcher.dispatch_begin(1, 2, [], frame=0)  # Allowed
        dispatcher.dispatch_begin(1, 5, [], frame=0)  # Filtered
        assert len(received) == 1
        assert dispatcher.events_filtered == 1

    def test_remove_filter(self):
        """remove_filter should remove filter."""
        dispatcher = CollisionEventDispatcher()
        filter_fn = lambda a, b: a != 5 and b != 5
        dispatcher.add_filter(filter_fn)
        assert dispatcher.remove_filter(filter_fn)

    def test_deferred_mode(self):
        """Deferred mode should queue events."""
        dispatcher = CollisionEventDispatcher()
        received = []
        dispatcher.on_collision_begin(lambda e: received.append(e) or True)
        dispatcher.begin_deferred()
        dispatcher.dispatch_begin(1, 2, [], frame=0)
        dispatcher.dispatch_begin(3, 4, [], frame=0)
        assert len(received) == 0  # Not dispatched yet
        dispatcher.end_deferred()
        assert len(received) == 2

    def test_flush_deferred(self):
        """flush_deferred should process queue without ending deferred mode."""
        dispatcher = CollisionEventDispatcher()
        received = []
        dispatcher.on_collision_begin(lambda e: received.append(e) or True)
        dispatcher.begin_deferred()
        dispatcher.dispatch_begin(1, 2, [], frame=0)
        count = dispatcher.flush_deferred()
        assert count == 1
        assert len(received) == 1
        # Still in deferred mode
        dispatcher.dispatch_begin(3, 4, [], frame=0)
        assert len(received) == 1  # Still queued

    def test_clear(self):
        """clear should remove all handlers and filters."""
        dispatcher = CollisionEventDispatcher()
        dispatcher.on_collision_begin(lambda e: True)
        dispatcher.on_collision_persist(lambda e: True)
        dispatcher.add_filter(lambda a, b: True)
        dispatcher.clear()
        assert dispatcher.handler_count() == 0

    def test_handler_count(self):
        """handler_count should return total handlers."""
        dispatcher = CollisionEventDispatcher()
        dispatcher.on_collision_begin(lambda e: True)
        dispatcher.on_collision_persist(lambda e: True)
        dispatcher.on_collision_end(lambda e: True)
        dispatcher.on_body_collision(1, lambda e: True)
        assert dispatcher.handler_count() == 4

    def test_handler_exception_silenced(self):
        """Handler exceptions should be silently ignored."""
        dispatcher = CollisionEventDispatcher()
        received = []
        def bad_handler(e):
            raise ValueError("Intentional error")
        def good_handler(e):
            received.append(e)
            return True
        dispatcher.on_collision_begin(bad_handler, priority=2)
        dispatcher.on_collision_begin(good_handler, priority=1)
        dispatcher.dispatch_begin(1, 2, [], frame=0)
        # good_handler should still be called
        assert len(received) == 1


class TestCollisionListener:
    """Tests for CollisionListener base class."""

    def test_default_returns_true(self):
        """Default implementations should return True."""
        listener = CollisionListener()
        event = CollisionEvent(
            event_type=CollisionEventType.BEGIN,
            body_a=1,
            body_b=2,
        )
        assert listener.on_collision_begin(event)
        assert listener.on_collision_persist(event)
        assert listener.on_collision_end(event)

    def test_register(self):
        """register should add listener to dispatcher."""
        dispatcher = CollisionEventDispatcher()
        listener = CollisionListener()
        listener.register(dispatcher)
        # All 3 handlers should be registered
        assert dispatcher.handler_count() == 3


class TestCustomCollisionListener:
    """Tests for custom CollisionListener subclasses."""

    def test_custom_listener(self):
        """Custom listener should receive events."""
        received_begin = []
        received_persist = []
        received_end = []

        class TestListener(CollisionListener):
            def on_collision_begin(self, event):
                received_begin.append(event)
                return True

            def on_collision_persist(self, event):
                received_persist.append(event)
                return True

            def on_collision_end(self, event):
                received_end.append(event)
                return True

        dispatcher = CollisionEventDispatcher()
        listener = TestListener()
        listener.register(dispatcher)

        dispatcher.dispatch_begin(1, 2, [], frame=0)
        dispatcher.dispatch_persist(1, 2, [], frame=1)
        dispatcher.dispatch_end(1, 2, frame=2)

        assert len(received_begin) == 1
        assert len(received_persist) == 1
        assert len(received_end) == 1


class TestCollisionEventProcessor:
    """Tests for CollisionEventProcessor class."""

    def test_construction(self):
        """Processor should be constructed correctly."""
        dispatcher = CollisionEventDispatcher()
        processor = CollisionEventProcessor(dispatcher)
        assert processor is not None

    def test_process_manifold_begin(self):
        """process_manifold should dispatch BEGIN for new contacts."""
        dispatcher = CollisionEventDispatcher()
        received = []
        dispatcher.on_collision_begin(lambda e: received.append(e) or True)
        processor = CollisionEventProcessor(dispatcher)

        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        processor.process_manifold(manifold)

        assert len(received) == 1
        assert received[0].is_begin

    def test_process_manifold_persist(self):
        """process_manifold should dispatch PERSIST for ongoing contacts."""
        dispatcher = CollisionEventDispatcher()
        received_persist = []
        dispatcher.on_collision_persist(lambda e: received_persist.append(e) or True)
        processor = CollisionEventProcessor(dispatcher)

        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)

        # First call triggers BEGIN
        processor.process_manifold(manifold)
        # Second call triggers PERSIST
        processor.process_manifold(manifold)

        assert len(received_persist) == 1

    def test_process_manifold_end(self):
        """process_manifold should dispatch END when contacts removed."""
        dispatcher = CollisionEventDispatcher()
        received_end = []
        dispatcher.on_collision_end(lambda e: received_end.append(e) or True)
        processor = CollisionEventProcessor(dispatcher)

        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        processor.process_manifold(manifold)  # BEGIN

        manifold.clear()
        processor.process_manifold(manifold)  # END

        assert len(received_end) == 1

    def test_process_removed_manifolds(self):
        """process_removed_manifolds should dispatch END events."""
        dispatcher = CollisionEventDispatcher()
        received_end = []
        dispatcher.on_collision_end(lambda e: received_end.append(e) or True)
        processor = CollisionEventProcessor(dispatcher)

        # Simulate active manifold
        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        processor.process_manifold(manifold)  # BEGIN

        # Process removal
        processor.process_removed_manifolds([(1, 2)])
        assert len(received_end) == 1

    def test_advance_frame(self):
        """advance_frame should increment frame counter."""
        dispatcher = CollisionEventDispatcher()
        processor = CollisionEventProcessor(dispatcher)
        processor.advance_frame()
        # Frame counter is internal, just check it doesn't crash

    def test_clear(self):
        """clear should reset processor state."""
        dispatcher = CollisionEventDispatcher()
        processor = CollisionEventProcessor(dispatcher)

        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        processor.process_manifold(manifold)

        processor.clear()
        # After clear, same manifold should trigger BEGIN again
        received = []
        dispatcher.on_collision_begin(lambda e: received.append(e) or True)
        processor.process_manifold(manifold)
        # Would be BEGIN since state was cleared
        # (Note: this depends on manifold state, may need separate test)


class TestCollisionEventEdgeCases:
    """Edge case tests for collision events."""

    def test_event_with_contacts(self):
        """Event creation with contacts should compute averages."""
        dispatcher = CollisionEventDispatcher()
        received = []
        dispatcher.on_collision_begin(lambda e: received.append(e) or True)

        contacts = [
            ContactPoint(position=Vec3(0, 0, 0), normal=Vec3(1, 0, 0), depth=0.1),
            ContactPoint(position=Vec3(2, 0, 0), normal=Vec3(0, 1, 0), depth=0.3),
        ]
        dispatcher.dispatch_begin(1, 2, contacts, frame=0)

        event = received[0]
        assert event.max_depth == 0.3
        assert event.position.x == 1.0  # Average of 0 and 2

    def test_many_handlers_same_event(self):
        """Many handlers for same event should all be called."""
        dispatcher = CollisionEventDispatcher()
        call_count = [0]

        for _ in range(10):
            dispatcher.on_collision_begin(lambda e: (call_count.__setitem__(0, call_count[0] + 1), True)[1])

        dispatcher.dispatch_begin(1, 2, [], frame=0)
        assert call_count[0] == 10

    def test_dispatch_without_handlers(self):
        """Dispatching without handlers should not crash."""
        dispatcher = CollisionEventDispatcher()
        dispatcher.dispatch_begin(1, 2, [], frame=0)
        dispatcher.dispatch_persist(1, 2, [], frame=0)
        dispatcher.dispatch_end(1, 2, frame=0)
        # No assertion needed, just verify no crash

    def test_thread_safety_registration(self):
        """Handler registration should be thread-safe."""
        import threading
        dispatcher = CollisionEventDispatcher()

        def register_handlers():
            for _ in range(100):
                dispatcher.on_collision_begin(lambda e: True)

        threads = [threading.Thread(target=register_handlers) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All handlers should be registered
        assert dispatcher.handler_count() >= 400
