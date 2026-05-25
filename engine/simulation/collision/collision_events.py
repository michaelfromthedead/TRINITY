"""
Collision Event System.

This module implements the event system for collision detection,
providing callbacks for collision begin, persist, and end events.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Any
from weakref import WeakMethod, ref
import threading

from .broadphase import Vec3
from .contact_manifold import ContactPoint, ContactManifold


# =============================================================================
# Event Types
# =============================================================================


class CollisionEventType(Enum):
    """Types of collision events."""

    BEGIN = auto()    # First frame of contact
    PERSIST = auto()  # Ongoing contact
    END = auto()      # Contact ended


# =============================================================================
# Collision Event
# =============================================================================


@dataclass
class CollisionEvent:
    """
    Represents a collision event between two bodies.

    Contains all relevant collision information for event handlers.
    """

    # Event type
    event_type: CollisionEventType

    # Body identifiers
    body_a: int
    body_b: int

    # Contact information
    contacts: list[ContactPoint] = field(default_factory=list)

    # Accumulated impulse from solver (for PERSIST/END events)
    impulse: float = 0.0
    impulse_tangent_1: float = 0.0
    impulse_tangent_2: float = 0.0

    # Collision normal (average of contact normals)
    normal: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))

    # Contact position (average of contact positions)
    position: Vec3 = field(default_factory=Vec3)

    # Relative velocity at contact point
    relative_velocity: Vec3 = field(default_factory=Vec3)

    # Maximum penetration depth
    max_depth: float = 0.0

    # Frame number when event occurred
    frame: int = 0

    # User data attached to bodies
    user_data_a: Any = None
    user_data_b: Any = None

    @property
    def is_begin(self) -> bool:
        """Check if this is a begin event."""
        return self.event_type == CollisionEventType.BEGIN

    @property
    def is_persist(self) -> bool:
        """Check if this is a persist event."""
        return self.event_type == CollisionEventType.PERSIST

    @property
    def is_end(self) -> bool:
        """Check if this is an end event."""
        return self.event_type == CollisionEventType.END

    @property
    def contact_count(self) -> int:
        """Get number of contact points."""
        return len(self.contacts)

    def get_other_body(self, body_id: int) -> int:
        """Get the other body in the collision."""
        return self.body_b if body_id == self.body_a else self.body_a

    def get_contact_normal_for(self, body_id: int) -> Vec3:
        """Get contact normal pointing away from specified body."""
        if body_id == self.body_a:
            return self.normal
        return self.normal * -1


# =============================================================================
# Event Callback Types
# =============================================================================

# Callback type: receives CollisionEvent, returns bool (True to continue)
CollisionCallback = Callable[[CollisionEvent], bool]

# Filter callback: receives (body_a, body_b), returns bool (True to allow event)
EventFilterCallback = Callable[[int, int], bool]


# =============================================================================
# Collision Event Dispatcher
# =============================================================================


class CollisionEventDispatcher:
    """
    Dispatches collision events to registered handlers.

    Features:
    - Separate handlers for begin/persist/end events
    - Priority-based handler ordering
    - Event filtering
    - Thread-safe callback management
    """

    def __init__(self):
        self._begin_handlers: list[tuple[int, CollisionCallback]] = []
        self._persist_handlers: list[tuple[int, CollisionCallback]] = []
        self._end_handlers: list[tuple[int, CollisionCallback]] = []

        # Body-specific handlers
        self._body_handlers: dict[int, list[tuple[int, CollisionCallback]]] = {}

        # Event filters
        self._filters: list[EventFilterCallback] = []

        # Lock for thread safety
        self._lock = threading.Lock()

        # Event queue for deferred processing
        self._event_queue: list[CollisionEvent] = []
        self._deferred_mode = False

        # Statistics
        self._events_dispatched = 0
        self._events_filtered = 0

    @property
    def events_dispatched(self) -> int:
        """Get total events dispatched."""
        return self._events_dispatched

    @property
    def events_filtered(self) -> int:
        """Get total events filtered out."""
        return self._events_filtered

    # -------------------------------------------------------------------------
    # Handler Registration
    # -------------------------------------------------------------------------

    def on_collision_begin(
        self,
        callback: CollisionCallback,
        priority: int = 0,
    ) -> None:
        """
        Register handler for collision begin events.

        Args:
            callback: Callback function
            priority: Handler priority (higher = called first)
        """
        with self._lock:
            self._begin_handlers.append((priority, callback))
            self._begin_handlers.sort(key=lambda x: -x[0])

    def on_collision_persist(
        self,
        callback: CollisionCallback,
        priority: int = 0,
    ) -> None:
        """
        Register handler for collision persist events.

        Args:
            callback: Callback function
            priority: Handler priority
        """
        with self._lock:
            self._persist_handlers.append((priority, callback))
            self._persist_handlers.sort(key=lambda x: -x[0])

    def on_collision_end(
        self,
        callback: CollisionCallback,
        priority: int = 0,
    ) -> None:
        """
        Register handler for collision end events.

        Args:
            callback: Callback function
            priority: Handler priority
        """
        with self._lock:
            self._end_handlers.append((priority, callback))
            self._end_handlers.sort(key=lambda x: -x[0])

    def on_body_collision(
        self,
        body_id: int,
        callback: CollisionCallback,
        priority: int = 0,
    ) -> None:
        """
        Register handler for specific body's collisions.

        Args:
            body_id: Body to monitor
            callback: Callback function
            priority: Handler priority
        """
        with self._lock:
            if body_id not in self._body_handlers:
                self._body_handlers[body_id] = []
            self._body_handlers[body_id].append((priority, callback))
            self._body_handlers[body_id].sort(key=lambda x: -x[0])

    def remove_handler(self, callback: CollisionCallback) -> bool:
        """
        Remove a handler from all event types.

        Args:
            callback: Callback to remove

        Returns:
            True if handler was found and removed
        """
        removed = False
        with self._lock:
            for handlers in [
                self._begin_handlers,
                self._persist_handlers,
                self._end_handlers,
            ]:
                for i, (_, cb) in enumerate(handlers):
                    if cb is callback:
                        handlers.pop(i)
                        removed = True
                        break

            for handlers in self._body_handlers.values():
                for i, (_, cb) in enumerate(handlers):
                    if cb is callback:
                        handlers.pop(i)
                        removed = True
                        break

        return removed

    def remove_body_handlers(self, body_id: int) -> int:
        """
        Remove all handlers for a specific body.

        Args:
            body_id: Body to remove handlers for

        Returns:
            Number of handlers removed
        """
        with self._lock:
            if body_id in self._body_handlers:
                count = len(self._body_handlers[body_id])
                del self._body_handlers[body_id]
                return count
        return 0

    # -------------------------------------------------------------------------
    # Event Filtering
    # -------------------------------------------------------------------------

    def add_filter(self, filter_fn: EventFilterCallback) -> None:
        """
        Add event filter.

        Filter receives (body_a, body_b) and returns True to allow event.

        Args:
            filter_fn: Filter function
        """
        with self._lock:
            self._filters.append(filter_fn)

    def remove_filter(self, filter_fn: EventFilterCallback) -> bool:
        """
        Remove event filter.

        Args:
            filter_fn: Filter to remove

        Returns:
            True if filter was removed
        """
        with self._lock:
            if filter_fn in self._filters:
                self._filters.remove(filter_fn)
                return True
        return False

    def _passes_filters(self, body_a: int, body_b: int) -> bool:
        """Check if event passes all filters."""
        for filter_fn in self._filters:
            if not filter_fn(body_a, body_b):
                return False
        return True

    # -------------------------------------------------------------------------
    # Event Dispatch
    # -------------------------------------------------------------------------

    def dispatch(self, event: CollisionEvent) -> None:
        """
        Dispatch a collision event to handlers.

        If in deferred mode, queues event for later processing.

        Args:
            event: Event to dispatch
        """
        if self._deferred_mode:
            self._event_queue.append(event)
            return

        self._dispatch_immediate(event)

    def _dispatch_immediate(self, event: CollisionEvent) -> None:
        """Immediately dispatch event to handlers."""
        # Apply filters
        if not self._passes_filters(event.body_a, event.body_b):
            self._events_filtered += 1
            return

        self._events_dispatched += 1

        # Get appropriate handlers
        if event.event_type == CollisionEventType.BEGIN:
            handlers = self._begin_handlers
        elif event.event_type == CollisionEventType.PERSIST:
            handlers = self._persist_handlers
        else:
            handlers = self._end_handlers

        # Call general handlers
        for _, callback in handlers:
            try:
                if not callback(event):
                    break  # Handler returned False, stop propagation
            except Exception:
                pass  # Silently ignore handler errors

        # Call body-specific handlers
        for body_id in (event.body_a, event.body_b):
            if body_id in self._body_handlers:
                for _, callback in self._body_handlers[body_id]:
                    try:
                        if not callback(event):
                            break
                    except Exception:
                        pass

    def dispatch_begin(
        self,
        body_a: int,
        body_b: int,
        contacts: list[ContactPoint],
        frame: int = 0,
    ) -> None:
        """
        Dispatch collision begin event.

        Args:
            body_a: First body ID
            body_b: Second body ID
            contacts: Contact points
            frame: Current frame number
        """
        event = self._create_event(
            CollisionEventType.BEGIN, body_a, body_b, contacts, frame
        )
        self.dispatch(event)

    def dispatch_persist(
        self,
        body_a: int,
        body_b: int,
        contacts: list[ContactPoint],
        impulse: float = 0.0,
        frame: int = 0,
    ) -> None:
        """
        Dispatch collision persist event.

        Args:
            body_a: First body ID
            body_b: Second body ID
            contacts: Contact points
            impulse: Accumulated impulse
            frame: Current frame number
        """
        event = self._create_event(
            CollisionEventType.PERSIST, body_a, body_b, contacts, frame
        )
        event.impulse = impulse
        self.dispatch(event)

    def dispatch_end(
        self,
        body_a: int,
        body_b: int,
        frame: int = 0,
    ) -> None:
        """
        Dispatch collision end event.

        Args:
            body_a: First body ID
            body_b: Second body ID
            frame: Current frame number
        """
        event = CollisionEvent(
            event_type=CollisionEventType.END,
            body_a=body_a,
            body_b=body_b,
            frame=frame,
        )
        self.dispatch(event)

    def _create_event(
        self,
        event_type: CollisionEventType,
        body_a: int,
        body_b: int,
        contacts: list[ContactPoint],
        frame: int,
    ) -> CollisionEvent:
        """Create event from contact points."""
        if not contacts:
            return CollisionEvent(
                event_type=event_type,
                body_a=body_a,
                body_b=body_b,
                frame=frame,
            )

        # Compute average normal and position
        avg_normal = Vec3()
        avg_position = Vec3()
        max_depth = 0.0

        for contact in contacts:
            avg_normal = avg_normal + contact.normal
            avg_position = avg_position + contact.position
            max_depth = max(max_depth, contact.depth)

        n = len(contacts)
        avg_normal = avg_normal.normalized()
        avg_position = avg_position * (1.0 / n)

        return CollisionEvent(
            event_type=event_type,
            body_a=body_a,
            body_b=body_b,
            contacts=contacts,
            normal=avg_normal,
            position=avg_position,
            max_depth=max_depth,
            frame=frame,
        )

    # -------------------------------------------------------------------------
    # Deferred Processing
    # -------------------------------------------------------------------------

    def begin_deferred(self) -> None:
        """Begin deferred event processing mode."""
        self._deferred_mode = True

    def end_deferred(self) -> None:
        """End deferred mode and process queued events."""
        self._deferred_mode = False
        for event in self._event_queue:
            self._dispatch_immediate(event)
        self._event_queue.clear()

    def flush_deferred(self) -> int:
        """
        Flush deferred events without ending deferred mode.

        Returns:
            Number of events processed
        """
        count = len(self._event_queue)
        for event in self._event_queue:
            self._dispatch_immediate(event)
        self._event_queue.clear()
        return count

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all handlers and filters."""
        with self._lock:
            self._begin_handlers.clear()
            self._persist_handlers.clear()
            self._end_handlers.clear()
            self._body_handlers.clear()
            self._filters.clear()
            self._event_queue.clear()
            self._events_dispatched = 0
            self._events_filtered = 0

    def handler_count(self) -> int:
        """Get total number of registered handlers."""
        with self._lock:
            count = (
                len(self._begin_handlers)
                + len(self._persist_handlers)
                + len(self._end_handlers)
            )
            for handlers in self._body_handlers.values():
                count += len(handlers)
            return count


# =============================================================================
# Collision Listener Interface
# =============================================================================


class CollisionListener:
    """
    Base class for collision event listeners.

    Subclass and override methods to handle collision events.
    """

    def on_collision_begin(self, event: CollisionEvent) -> bool:
        """
        Called when collision begins.

        Args:
            event: Collision event

        Returns:
            True to continue event propagation
        """
        return True

    def on_collision_persist(self, event: CollisionEvent) -> bool:
        """
        Called each frame while collision persists.

        Args:
            event: Collision event

        Returns:
            True to continue event propagation
        """
        return True

    def on_collision_end(self, event: CollisionEvent) -> bool:
        """
        Called when collision ends.

        Args:
            event: Collision event

        Returns:
            True to continue event propagation
        """
        return True

    def register(self, dispatcher: CollisionEventDispatcher) -> None:
        """Register this listener with a dispatcher."""
        dispatcher.on_collision_begin(self.on_collision_begin)
        dispatcher.on_collision_persist(self.on_collision_persist)
        dispatcher.on_collision_end(self.on_collision_end)


# =============================================================================
# Event Processor
# =============================================================================


class CollisionEventProcessor:
    """
    Processes collision manifolds into events.

    Tracks manifold state changes to generate appropriate events.
    """

    def __init__(self, dispatcher: CollisionEventDispatcher):
        self._dispatcher = dispatcher
        self._active_manifolds: set[tuple[int, int]] = set()
        self._frame = 0

    def process_manifold(self, manifold: ContactManifold) -> None:
        """
        Process a manifold and generate events.

        Args:
            manifold: Contact manifold to process
        """
        key = (min(manifold.body_a, manifold.body_b),
               max(manifold.body_a, manifold.body_b))

        began, persist, ended = manifold.update_touching_state()

        if began:
            self._active_manifolds.add(key)
            self._dispatcher.dispatch_begin(
                manifold.body_a,
                manifold.body_b,
                manifold.contacts,
                self._frame,
            )
        elif persist:
            self._dispatcher.dispatch_persist(
                manifold.body_a,
                manifold.body_b,
                manifold.contacts,
                manifold.get_total_impulse(),
                self._frame,
            )
        elif ended:
            self._active_manifolds.discard(key)
            self._dispatcher.dispatch_end(
                manifold.body_a,
                manifold.body_b,
                self._frame,
            )

    def process_removed_manifolds(
        self,
        removed_body_pairs: list[tuple[int, int]],
    ) -> None:
        """
        Process removed manifolds and generate end events.

        Args:
            removed_body_pairs: List of (body_a, body_b) tuples
        """
        for body_a, body_b in removed_body_pairs:
            key = (min(body_a, body_b), max(body_a, body_b))
            if key in self._active_manifolds:
                self._active_manifolds.discard(key)
                self._dispatcher.dispatch_end(body_a, body_b, self._frame)

    def advance_frame(self) -> None:
        """Advance frame counter."""
        self._frame += 1

    def clear(self) -> None:
        """Clear processor state."""
        self._active_manifolds.clear()
        self._frame = 0
