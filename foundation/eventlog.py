"""
EventLog - Unified operation and change tracking with causal chains.

Part of Core Foundation Layer 0. Provides:
- Recording of operations with changes
- Entity-centric causal chain tracking
- Multiple indexes for efficient querying
- @traced decorator for automatic recording
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

F = TypeVar('F', bound=Callable)


@dataclass
class Change:
    """
    Represents a single field change.

    Attributes:
        entity: The entity ID that was changed.
        field: The field name that was modified.
        old_value: The value before the change.
        new_value: The value after the change.
    """
    entity: int
    field: str
    old_value: Any
    new_value: Any


@dataclass
class Event:
    """
    Represents an operation that may cause changes.

    An event captures:
    - What operation was performed
    - Which entity was the "self" (if any)
    - What changes occurred
    - The causal chain (which operation triggered this one)

    Attributes:
        tick: The game tick when this occurred.
        operation: Qualified operation name (e.g., "Player.take_damage").
        operation_args: Arguments passed to the operation.
        entity: The entity ID this operation was called on (if method call).
        changes: List of field changes caused by this operation.
        result: Return value of the operation.
        error: Exception if the operation failed.
        immediate_parent: The operation that directly called this one.
        immediate_parent_entity: Entity of the immediate parent operation.
        root_cause: The first entity-bound operation in the causal chain.
        root_cause_entity: Entity of the root cause operation.
        depth: Nesting depth in the call chain.
    """
    tick: int
    operation: str
    operation_args: dict[str, Any] = field(default_factory=dict)
    entity: Optional[int] = None
    changes: list[Change] = field(default_factory=list)
    result: Any = None
    error: Optional[Exception] = None

    # Causal chain (entity-centric)
    immediate_parent: Optional[str] = None
    immediate_parent_entity: Optional[int] = None
    root_cause: Optional[str] = None
    root_cause_entity: Optional[int] = None
    depth: int = 0


class EventLog:
    """
    Unified log of operations and changes with multiple indexes.

    Provides efficient querying by:
    - Entity ID
    - Game tick
    - Operation name
    - Root cause (for causal analysis)
    """

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._by_entity: dict[int, list[Event]] = {}
        self._by_tick: dict[int, list[Event]] = {}
        self._by_operation: dict[str, list[Event]] = {}
        self._by_root_cause: dict[int, list[Event]] = {}

    def record(self, event: Event) -> None:
        """
        Record an event to the log.

        Args:
            event: The Event to record.
        """
        self._events.append(event)

        # Index by entity
        if event.entity is not None:
            if event.entity not in self._by_entity:
                self._by_entity[event.entity] = []
            self._by_entity[event.entity].append(event)

        # Index by tick
        if event.tick not in self._by_tick:
            self._by_tick[event.tick] = []
        self._by_tick[event.tick].append(event)

        # Index by operation
        if event.operation not in self._by_operation:
            self._by_operation[event.operation] = []
        self._by_operation[event.operation].append(event)

        # Index by root cause entity
        if event.root_cause_entity is not None:
            if event.root_cause_entity not in self._by_root_cause:
                self._by_root_cause[event.root_cause_entity] = []
            self._by_root_cause[event.root_cause_entity].append(event)

    def events_at(self, tick: int) -> list[Event]:
        """Get all events at a specific tick."""
        return list(self._by_tick.get(tick, []))

    def events_for_entity(self, entity_id: int) -> list[Event]:
        """Get all events for a specific entity."""
        return list(self._by_entity.get(entity_id, []))

    def events_for_operation(self, operation: str) -> list[Event]:
        """Get all events for a specific operation."""
        return list(self._by_operation.get(operation, []))

    def events_caused_by(self, entity_id: int) -> list[Event]:
        """Get all events where the given entity was the root cause."""
        return list(self._by_root_cause.get(entity_id, []))

    def events_where(self, **kwargs) -> list[Event]:
        """
        Query events with multiple filters.

        Args:
            tick: Filter by tick.
            entity: Filter by entity ID.
            operation: Filter by operation name.
            root_cause_entity: Filter by root cause entity.
            has_error: Filter by whether event has an error.
            min_depth: Minimum call depth.
            max_depth: Maximum call depth.

        Returns:
            List of matching events.
        """
        result = self._events

        if 'tick' in kwargs:
            result = [e for e in result if e.tick == kwargs['tick']]

        if 'entity' in kwargs:
            result = [e for e in result if e.entity == kwargs['entity']]

        if 'operation' in kwargs:
            result = [e for e in result if e.operation == kwargs['operation']]

        if 'root_cause_entity' in kwargs:
            result = [e for e in result if e.root_cause_entity == kwargs['root_cause_entity']]

        if 'has_error' in kwargs:
            result = [e for e in result if (e.error is not None) == kwargs['has_error']]

        if 'min_depth' in kwargs:
            result = [e for e in result if e.depth >= kwargs['min_depth']]

        if 'max_depth' in kwargs:
            result = [e for e in result if e.depth <= kwargs['max_depth']]

        return result

    def changes_where(self, **kwargs) -> list[Change]:
        """
        Query changes with filters.

        Args:
            entity: Filter by entity ID.
            field: Filter by field name.
            tick: Filter by tick (from parent event).

        Returns:
            List of matching changes.
        """
        events = self._events

        if 'tick' in kwargs:
            events = [e for e in events if e.tick == kwargs['tick']]

        changes: list[Change] = []
        for event in events:
            for change in event.changes:
                include = True

                if 'entity' in kwargs and change.entity != kwargs['entity']:
                    include = False

                if 'field' in kwargs and change.field != kwargs['field']:
                    include = False

                if include:
                    changes.append(change)

        return changes

    def all_events(self) -> list[Event]:
        """Get all recorded events."""
        return list(self._events)

    def clear(self) -> None:
        """Clear all events and indexes."""
        self._events.clear()
        self._by_entity.clear()
        self._by_tick.clear()
        self._by_operation.clear()
        self._by_root_cause.clear()

    def __len__(self) -> int:
        """Return the number of recorded events."""
        return len(self._events)


# =============================================================================
# Context Variables for Causal Chain Tracking
# =============================================================================

_current_event: ContextVar[Optional[Event]] = ContextVar('current_event', default=None)
_root_cause: ContextVar[Optional[str]] = ContextVar('root_cause', default=None)
_root_cause_entity: ContextVar[Optional[int]] = ContextVar('root_cause_entity', default=None)
_immediate_parent: ContextVar[Optional[str]] = ContextVar('immediate_parent', default=None)
_immediate_parent_entity: ContextVar[Optional[int]] = ContextVar('immediate_parent_entity', default=None)
_depth: ContextVar[int] = ContextVar('depth', default=0)
_current_tick: ContextVar[int] = ContextVar('current_tick', default=0)

# Global event log singleton
_event_log = EventLog()


def set_current_tick(tick: int) -> None:
    """Set the current game tick for event recording."""
    _current_tick.set(tick)


def get_current_tick() -> int:
    """Get the current game tick."""
    return _current_tick.get()


def get_event_log() -> EventLog:
    """Get the global event log instance."""
    return _event_log


def get_current_event() -> Optional[Event]:
    """Get the currently executing event (if any)."""
    return _current_event.get()


# =============================================================================
# @traced Decorator
# =============================================================================

def traced(fn: F) -> F:
    """
    Decorator that automatically records operations to the event log.

    Key insight: Systems are pass-through, entities are actors.
    - If the first argument has an 'id' attribute, it's an entity-bound operation.
    - The first entity-bound operation in a call chain becomes the root cause.

    Usage:
        class Player:
            id: int

            @traced
            def take_damage(self, amount: int) -> None:
                self.health -= amount

    The decorator captures:
    - Operation name (qualified: "Player.take_damage")
    - Entity ID (from self.id)
    - Arguments passed
    - Any changes made (when integrated with TrackedDescriptor)
    - Return value or exception
    - Causal chain (immediate parent, root cause)
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Extract entity ID from first argument if present
        entity_id: Optional[int] = None
        if args:
            first_arg = args[0]
            entity_id = getattr(first_arg, 'id', None)

        # Get operation name
        operation = fn.__qualname__

        # Get current context
        current_root = _root_cause.get()
        current_root_entity = _root_cause_entity.get()
        parent = _immediate_parent.get()
        parent_entity = _immediate_parent_entity.get()
        current_depth = _depth.get()

        # Root cause = first entity-bound operation in the chain
        is_new_root = (current_root is None and entity_id is not None)
        if is_new_root:
            new_root = operation
            new_root_entity = entity_id
        else:
            new_root = current_root
            new_root_entity = current_root_entity

        # Create event
        event = Event(
            tick=_current_tick.get(),
            operation=operation,
            operation_args={'args': args[1:], 'kwargs': kwargs},  # Exclude self
            entity=entity_id,
            immediate_parent=parent,
            immediate_parent_entity=parent_entity,
            root_cause=new_root,
            root_cause_entity=new_root_entity,
            depth=current_depth,
        )

        # Set new context for nested calls
        token_event = _current_event.set(event)
        token_root = _root_cause.set(new_root)
        token_root_entity = _root_cause_entity.set(new_root_entity)
        token_parent = _immediate_parent.set(operation)
        token_parent_entity = _immediate_parent_entity.set(entity_id)
        token_depth = _depth.set(current_depth + 1)

        try:
            result = fn(*args, **kwargs)
            event.result = result
            return result
        except Exception as e:
            event.error = e
            raise
        finally:
            # Record event
            _event_log.record(event)

            # Restore context
            _current_event.reset(token_event)
            _root_cause.reset(token_root)
            _root_cause_entity.reset(token_root_entity)
            _immediate_parent.reset(token_parent)
            _immediate_parent_entity.reset(token_parent_entity)
            _depth.reset(token_depth)

    return wrapper  # type: ignore


def add_change_to_current_event(change: Change) -> bool:
    """
    Add a change to the currently executing event.

    This is called by TrackedDescriptor when a field is modified
    within a @traced operation.

    Returns:
        True if added to current event, False if no event context.
    """
    event = _current_event.get()
    if event is not None:
        event.changes.append(change)
        return True
    return False


def clear_event_log() -> None:
    """Clear the global event log."""
    _event_log.clear()


__all__ = [
    "Change",
    "Event",
    "EventLog",
    "traced",
    "set_current_tick",
    "get_current_tick",
    "get_event_log",
    "get_current_event",
    "add_change_to_current_event",
    "clear_event_log",
]
