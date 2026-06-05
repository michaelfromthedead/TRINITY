"""
Entity Lifecycle EventLog Integration
======================================
Integrates Foundation EventLog with entity lifecycle system.

Provides:
- Typed lifecycle events (EntitySpawned, EntityDestroyed, etc.)
- Automatic event firing at lifecycle points
- Causal chain tracking (spawn causes component adds, etc.)
- Query interface: EntityEventLog.query(entity_id=X)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

from foundation.eventlog import (
    Event,
    EventLog,
    get_event_log,
    get_current_tick,
    set_current_tick,
    get_current_event,
    clear_event_log,
)

from .constants import LifecycleState


# =============================================================================
# LIFECYCLE EVENT TYPES
# =============================================================================


@dataclass(frozen=True)
class EntitySpawned:
    """Event fired when an entity is spawned."""
    entity_id: int
    prefab_name: str
    position: tuple[float, float, float]
    timestamp: float
    entity_type: str = ""

    @property
    def event_type(self) -> str:
        return "EntitySpawned"


@dataclass(frozen=True)
class EntityDestroyed:
    """Event fired when an entity is destroyed."""
    entity_id: int
    reason: str
    timestamp: float
    final_state: Optional[str] = None

    @property
    def event_type(self) -> str:
        return "EntityDestroyed"


@dataclass(frozen=True)
class ComponentAdded:
    """Event fired when a component is added to an entity."""
    entity_id: int
    component_type: str
    timestamp: float
    component_name: str = ""

    @property
    def event_type(self) -> str:
        return "ComponentAdded"


@dataclass(frozen=True)
class ComponentRemoved:
    """Event fired when a component is removed from an entity."""
    entity_id: int
    component_type: str
    timestamp: float
    component_name: str = ""

    @property
    def event_type(self) -> str:
        return "ComponentRemoved"


@dataclass(frozen=True)
class EntityStateChanged:
    """Event fired when an entity's lifecycle state changes."""
    entity_id: int
    old_state: str
    new_state: str
    timestamp: float

    @property
    def event_type(self) -> str:
        return "EntityStateChanged"


# Union type for all lifecycle events
LifecycleEventData = Union[
    EntitySpawned,
    EntityDestroyed,
    ComponentAdded,
    ComponentRemoved,
    EntityStateChanged,
]

# Type variable for event types
E = TypeVar("E", bound=LifecycleEventData)


# =============================================================================
# CAUSAL CHAIN TRACKING
# =============================================================================


@dataclass
class CausalChain:
    """
    Tracks causal relationships between events.

    Used to link events together, e.g., EntitySpawned causes ComponentAdded.
    """
    root_event_id: int
    parent_event_id: Optional[int]
    depth: int = 0

    def child(self, event_id: int) -> "CausalChain":
        """Create a child chain from this one."""
        return CausalChain(
            root_event_id=self.root_event_id,
            parent_event_id=event_id,
            depth=self.depth + 1,
        )


# =============================================================================
# LIFECYCLE EVENT RECORD
# =============================================================================


@dataclass
class LifecycleEventRecord:
    """
    A recorded lifecycle event with metadata.

    Wraps a lifecycle event data object with tracking information.
    """
    id: int
    tick: int
    event_data: LifecycleEventData
    entity_id: int
    causal_parent_id: Optional[int] = None
    causal_root_id: Optional[int] = None
    depth: int = 0

    @property
    def event_type(self) -> str:
        """Get the event type name."""
        return self.event_data.event_type

    @property
    def timestamp(self) -> float:
        """Get the event timestamp."""
        return self.event_data.timestamp


# =============================================================================
# ENTITY EVENT LOG
# =============================================================================


class EntityEventLog:
    """
    Specialized event log for entity lifecycle events.

    Provides:
    - Recording of typed lifecycle events
    - Multiple indexes for efficient querying
    - Causal chain tracking
    - Integration with Foundation EventLog
    """

    _instance: Optional["EntityEventLog"] = None

    def __new__(cls) -> "EntityEventLog":
        """Singleton pattern for global event log."""
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._events: List[LifecycleEventRecord] = []
        self._next_id: int = 1

        # Indexes
        self._by_entity: Dict[int, List[LifecycleEventRecord]] = {}
        self._by_type: Dict[str, List[LifecycleEventRecord]] = {}
        self._by_tick: Dict[int, List[LifecycleEventRecord]] = {}
        self._by_time_range: List[LifecycleEventRecord] = []  # Sorted by timestamp

        # Causal tracking
        self._causal_children: Dict[int, List[int]] = {}  # parent_id -> [child_ids]
        self._current_causal_chain: Optional[CausalChain] = None

        self._initialized = True

    def record(
        self,
        event_data: LifecycleEventData,
        entity_id: int,
        causal_chain: Optional[CausalChain] = None,
    ) -> LifecycleEventRecord:
        """
        Record a lifecycle event.

        Args:
            event_data: The event data to record
            entity_id: The entity this event relates to
            causal_chain: Optional causal chain for tracking relationships

        Returns:
            The recorded event with its assigned ID
        """
        event_id = self._next_id
        self._next_id += 1

        tick = get_current_tick()

        # Use current causal chain if none provided
        chain = causal_chain or self._current_causal_chain

        record = LifecycleEventRecord(
            id=event_id,
            tick=tick,
            event_data=event_data,
            entity_id=entity_id,
            causal_parent_id=chain.parent_event_id if chain else None,
            causal_root_id=chain.root_event_id if chain else None,
            depth=chain.depth if chain else 0,
        )

        # Store event
        self._events.append(record)

        # Index by entity
        if entity_id not in self._by_entity:
            self._by_entity[entity_id] = []
        self._by_entity[entity_id].append(record)

        # Index by event type
        event_type = record.event_type
        if event_type not in self._by_type:
            self._by_type[event_type] = []
        self._by_type[event_type].append(record)

        # Index by tick
        if tick not in self._by_tick:
            self._by_tick[tick] = []
        self._by_tick[tick].append(record)

        # Track causal relationships
        if chain and chain.parent_event_id is not None:
            parent_id = chain.parent_event_id
            if parent_id not in self._causal_children:
                self._causal_children[parent_id] = []
            self._causal_children[parent_id].append(event_id)

        # Also record to Foundation EventLog for integration
        foundation_event = Event(
            tick=tick,
            operation=f"Lifecycle.{event_type}",
            operation_args={"event_data": event_data.__dict__},
            entity=entity_id,
            immediate_parent=f"Lifecycle.{chain.parent_event_id}" if chain and chain.parent_event_id else None,
            root_cause=f"Lifecycle.{chain.root_event_id}" if chain and chain.root_event_id else None,
            depth=chain.depth if chain else 0,
        )
        get_event_log().record(foundation_event)

        return record

    def begin_causal_chain(self, event: LifecycleEventRecord) -> CausalChain:
        """
        Begin a new causal chain rooted at the given event.

        Use this when an event may cause other events (e.g., spawn causes component adds).

        Args:
            event: The root event for the chain

        Returns:
            A CausalChain that can be used for child events
        """
        chain = CausalChain(
            root_event_id=event.id,
            parent_event_id=event.id,
            depth=1,
        )
        self._current_causal_chain = chain
        return chain

    def end_causal_chain(self) -> None:
        """End the current causal chain."""
        self._current_causal_chain = None

    def with_causal_chain(self, event: LifecycleEventRecord):
        """
        Context manager for causal chain tracking.

        Usage:
            spawn_event = log.record_spawn(...)
            with log.with_causal_chain(spawn_event):
                log.record_component_added(...)  # Automatically linked
        """
        class CausalChainContext:
            def __init__(ctx, log: EntityEventLog, evt: LifecycleEventRecord):
                ctx.log = log
                ctx.event = evt
                ctx.old_chain = None

            def __enter__(ctx) -> CausalChain:
                ctx.old_chain = ctx.log._current_causal_chain
                return ctx.log.begin_causal_chain(ctx.event)

            def __exit__(ctx, exc_type, exc_val, exc_tb):
                ctx.log._current_causal_chain = ctx.old_chain
                return False

        return CausalChainContext(self, event)

    # =========================================================================
    # CONVENIENCE RECORDING METHODS
    # =========================================================================

    def record_spawn(
        self,
        entity_id: int,
        prefab_name: str,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        entity_type: str = "",
        causal_chain: Optional[CausalChain] = None,
    ) -> LifecycleEventRecord:
        """Record an entity spawn event."""
        event_data = EntitySpawned(
            entity_id=entity_id,
            prefab_name=prefab_name,
            position=position,
            timestamp=time.time(),
            entity_type=entity_type,
        )
        return self.record(event_data, entity_id, causal_chain)

    def record_destroy(
        self,
        entity_id: int,
        reason: str = "normal",
        final_state: Optional[str] = None,
        causal_chain: Optional[CausalChain] = None,
    ) -> LifecycleEventRecord:
        """Record an entity destroy event."""
        event_data = EntityDestroyed(
            entity_id=entity_id,
            reason=reason,
            timestamp=time.time(),
            final_state=final_state,
        )
        return self.record(event_data, entity_id, causal_chain)

    def record_component_added(
        self,
        entity_id: int,
        component_type: str,
        component_name: str = "",
        causal_chain: Optional[CausalChain] = None,
    ) -> LifecycleEventRecord:
        """Record a component added event."""
        event_data = ComponentAdded(
            entity_id=entity_id,
            component_type=component_type,
            timestamp=time.time(),
            component_name=component_name,
        )
        return self.record(event_data, entity_id, causal_chain)

    def record_component_removed(
        self,
        entity_id: int,
        component_type: str,
        component_name: str = "",
        causal_chain: Optional[CausalChain] = None,
    ) -> LifecycleEventRecord:
        """Record a component removed event."""
        event_data = ComponentRemoved(
            entity_id=entity_id,
            component_type=component_type,
            timestamp=time.time(),
            component_name=component_name,
        )
        return self.record(event_data, entity_id, causal_chain)

    def record_state_change(
        self,
        entity_id: int,
        old_state: Union[str, LifecycleState],
        new_state: Union[str, LifecycleState],
        causal_chain: Optional[CausalChain] = None,
    ) -> LifecycleEventRecord:
        """Record an entity state change event."""
        old_str = old_state.name if isinstance(old_state, LifecycleState) else old_state
        new_str = new_state.name if isinstance(new_state, LifecycleState) else new_state

        event_data = EntityStateChanged(
            entity_id=entity_id,
            old_state=old_str,
            new_state=new_str,
            timestamp=time.time(),
        )
        return self.record(event_data, entity_id, causal_chain)

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def query(
        self,
        entity_id: Optional[int] = None,
        event_type: Optional[str] = None,
        tick: Optional[int] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        causal_root_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[LifecycleEventRecord]:
        """
        Query events with multiple filters.

        Args:
            entity_id: Filter by entity ID
            event_type: Filter by event type name
            tick: Filter by game tick
            time_start: Filter events after this timestamp
            time_end: Filter events before this timestamp
            causal_root_id: Filter by causal chain root
            limit: Maximum number of results

        Returns:
            List of matching events
        """
        # Start with most specific index
        if entity_id is not None:
            result = list(self._by_entity.get(entity_id, []))
        elif event_type is not None:
            result = list(self._by_type.get(event_type, []))
        elif tick is not None:
            result = list(self._by_tick.get(tick, []))
        else:
            result = list(self._events)

        # Apply additional filters
        if entity_id is not None and event_type is not None:
            result = [e for e in result if e.event_type == event_type]

        if entity_id is not None and tick is not None:
            result = [e for e in result if e.tick == tick]

        if event_type is not None and tick is not None and entity_id is None:
            result = [e for e in result if e.tick == tick]

        if time_start is not None:
            result = [e for e in result if e.timestamp >= time_start]

        if time_end is not None:
            result = [e for e in result if e.timestamp <= time_end]

        if causal_root_id is not None:
            result = [e for e in result if e.causal_root_id == causal_root_id]

        if limit is not None:
            result = result[:limit]

        return result

    def query_by_entity(self, entity_id: int) -> List[LifecycleEventRecord]:
        """Get all events for a specific entity."""
        return list(self._by_entity.get(entity_id, []))

    def query_by_type(self, event_type: str) -> List[LifecycleEventRecord]:
        """Get all events of a specific type."""
        return list(self._by_type.get(event_type, []))

    def query_by_tick(self, tick: int) -> List[LifecycleEventRecord]:
        """Get all events at a specific tick."""
        return list(self._by_tick.get(tick, []))

    def query_by_time_range(
        self,
        start: float,
        end: float,
    ) -> List[LifecycleEventRecord]:
        """Get all events within a time range."""
        return [e for e in self._events if start <= e.timestamp <= end]

    def query_causal_children(self, event_id: int) -> List[LifecycleEventRecord]:
        """Get all events caused by a specific event."""
        child_ids = self._causal_children.get(event_id, [])
        return [e for e in self._events if e.id in child_ids]

    def query_causal_chain(self, root_id: int) -> List[LifecycleEventRecord]:
        """Get all events in a causal chain."""
        return [e for e in self._events if e.causal_root_id == root_id]

    def get_event(self, event_id: int) -> Optional[LifecycleEventRecord]:
        """Get a specific event by ID."""
        for event in self._events:
            if event.id == event_id:
                return event
        return None

    # =========================================================================
    # REPLAY CAPABILITIES
    # =========================================================================

    def get_replay_sequence(
        self,
        entity_id: Optional[int] = None,
        start_tick: int = 0,
        end_tick: Optional[int] = None,
    ) -> List[LifecycleEventRecord]:
        """
        Get events in replay order for a given entity or all entities.

        Args:
            entity_id: Optional entity to filter by
            start_tick: Starting tick (inclusive)
            end_tick: Ending tick (inclusive), None for all

        Returns:
            Events sorted by (tick, id) for deterministic replay
        """
        if entity_id is not None:
            events = self._by_entity.get(entity_id, [])
        else:
            events = self._events

        # Filter by tick range
        filtered = [e for e in events if e.tick >= start_tick]
        if end_tick is not None:
            filtered = [e for e in filtered if e.tick <= end_tick]

        # Sort by tick then ID for deterministic order
        return sorted(filtered, key=lambda e: (e.tick, e.id))

    def replay_events(
        self,
        events: List[LifecycleEventRecord],
        handler: Callable[[LifecycleEventRecord], None],
    ) -> int:
        """
        Replay a sequence of events through a handler.

        Args:
            events: Events to replay
            handler: Function to call for each event

        Returns:
            Number of events replayed
        """
        count = 0
        for event in events:
            handler(event)
            count += 1
        return count

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get event log statistics."""
        return {
            "total_events": len(self._events),
            "entities_tracked": len(self._by_entity),
            "event_types": list(self._by_type.keys()),
            "events_per_type": {k: len(v) for k, v in self._by_type.items()},
            "ticks_with_events": len(self._by_tick),
            "causal_chains": len(set(e.causal_root_id for e in self._events if e.causal_root_id)),
        }

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def clear(self) -> None:
        """Clear all recorded events."""
        self._events.clear()
        self._by_entity.clear()
        self._by_type.clear()
        self._by_tick.clear()
        self._causal_children.clear()
        self._current_causal_chain = None
        self._next_id = 1

    def __len__(self) -> int:
        """Return the number of recorded events."""
        return len(self._events)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        if cls._instance is not None:
            cls._instance.clear()
        cls._instance = None


# =============================================================================
# DECORATOR FOR AUTO-LOGGING
# =============================================================================


def logs_lifecycle_event(event_type: str):
    """
    Decorator to automatically log lifecycle events from methods.

    Usage:
        @logs_lifecycle_event("spawn")
        def spawn_entity(self, prefab_name: str) -> Actor:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)

            # Try to extract entity info from result or self
            entity_id = None
            if hasattr(result, '_entity_id'):
                entity_id = result._entity_id
            elif hasattr(result, 'entity_id'):
                entity_id = result.entity_id
            elif len(args) > 0 and hasattr(args[0], '_entity_id'):
                entity_id = args[0]._entity_id

            if entity_id is not None:
                log = EntityEventLog()
                if event_type == "spawn":
                    prefab = kwargs.get('prefab_name', kwargs.get('name', 'unknown'))
                    pos = kwargs.get('position', (0.0, 0.0, 0.0))
                    log.record_spawn(entity_id, prefab, pos)
                elif event_type == "destroy":
                    reason = kwargs.get('reason', 'normal')
                    log.record_destroy(entity_id, reason)

            return result
        return wrapper
    return decorator


# =============================================================================
# GLOBAL ACCESS
# =============================================================================


def get_entity_event_log() -> EntityEventLog:
    """Get the global entity event log instance."""
    return EntityEventLog()


def clear_entity_event_log() -> None:
    """Clear the global entity event log."""
    EntityEventLog().clear()


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Event types
    "EntitySpawned",
    "EntityDestroyed",
    "ComponentAdded",
    "ComponentRemoved",
    "EntityStateChanged",
    "LifecycleEventData",
    # Causal tracking
    "CausalChain",
    # Event record
    "LifecycleEventRecord",
    # Event log
    "EntityEventLog",
    # Decorator
    "logs_lifecycle_event",
    # Global access
    "get_entity_event_log",
    "clear_entity_event_log",
]
