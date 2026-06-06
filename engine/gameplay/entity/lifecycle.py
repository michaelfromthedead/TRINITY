"""
Entity Lifecycle Manager
========================
Manages entity state transitions through the lifecycle:
CREATE -> INITIALIZE -> ACTIVE -> DEACTIVATE -> DESTROY

Uses the Trinity Pattern with:
- LifecycleMeta metaclass for automatic registration
- LifecycleDescriptor for state tracking
- @lifecycle_hook decorator for callback registration
"""
from __future__ import annotations

import logging
import threading
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from trinity.decorators.ops import Op, Step, make_decorator, run_steps
from trinity.descriptors.base import BaseDescriptor
from trinity.metaclasses.engine_meta import EngineMeta

from .constants import (
    VALID_LIFECYCLE_TRANSITIONS,
    LifecycleState,
)

def _get_entity_event_log():
    """Get the EntityEventLog singleton (lazy import to avoid circular imports)."""
    from .eventlog_integration import EntityEventLog
    return EntityEventLog()

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .actor import Actor

T = TypeVar("T")


# =============================================================================
# LIFECYCLE HOOK EVENTS
# =============================================================================


class LifecycleEvent:
    """Lifecycle event identifiers."""

    ON_SPAWN = "on_spawn"
    BEGIN_PLAY = "begin_play"
    TICK = "tick"
    END_PLAY = "end_play"
    ON_DESTROY = "on_destroy"
    ON_ACTIVATE = "on_activate"
    ON_DEACTIVATE = "on_deactivate"
    ON_STATE_CHANGED = "on_state_changed"


# =============================================================================
# LIFECYCLE DESCRIPTOR
# =============================================================================


class LifecycleStateDescriptor(BaseDescriptor[LifecycleState]):
    """
    Descriptor for tracking entity lifecycle state.

    Features:
    - Validates state transitions
    - Records state history
    - Triggers lifecycle callbacks
    """

    descriptor_id: str = "lifecycle_state"
    accepts_inner: tuple[str, ...] = ("*",)
    accepts_outer: tuple[str, ...] = ("tracked", "observable")
    excludes: tuple[str, ...] = ()

    def __init__(
        self,
        field_type: type = LifecycleState,
        inner: Optional[BaseDescriptor] = None,
        validate_transitions: bool = True,
        track_history: bool = True,
        max_history: int = 10,
        **config: Any,
    ) -> None:
        super().__init__(field_type, inner, **config)
        self._validate_transitions = validate_transitions
        self._track_history = track_history
        self._max_history = max_history

    def pre_set(self, obj: Any, value: LifecycleState) -> LifecycleState:
        """Validate state transition before setting."""
        if not isinstance(value, LifecycleState):
            raise TypeError(f"Expected LifecycleState, got {type(value).__name__}")

        if self._validate_transitions:
            current = self._get_stored_safe(obj)
            if current is not None:
                valid_next = VALID_LIFECYCLE_TRANSITIONS.get(current, frozenset())
                if value not in valid_next:
                    raise ValueError(
                        f"Invalid state transition: {current.name} -> {value.name}. "
                        f"Valid transitions: {', '.join(s.name for s in valid_next)}"
                    )

        return value

    def post_set(
        self,
        obj: Any,
        value: LifecycleState,
        old_value: Optional[LifecycleState],
    ) -> None:
        """Record state change and trigger callbacks."""
        if self._track_history and old_value is not None:
            history_attr = f"_{self._name}_history"
            if not hasattr(obj, history_attr):
                setattr(obj, history_attr, deque(maxlen=self._max_history))
            history: deque = getattr(obj, history_attr)
            history.append((old_value, value))

        # Trigger state change callbacks
        if hasattr(obj, "_lifecycle_callbacks"):
            callbacks = obj._lifecycle_callbacks.get(LifecycleEvent.ON_STATE_CHANGED, [])
            for callback in callbacks:
                try:
                    callback(obj, old_value, value)
                except Exception as e:
                    # Log error but don't let callback errors break state transition
                    _logger.warning(
                        "Lifecycle state change callback failed for %s: %s",
                        type(obj).__name__,
                        e,
                    )

    @property
    def descriptor_steps(self) -> list[Step]:
        """Return steps this descriptor performs."""
        return [
            Step(Op.TRACK, {"field": self._name}),
            Step(Op.VALIDATE, {"constraint": "lifecycle_transition"}),
            Step(Op.HOOK, {"event": LifecycleEvent.ON_STATE_CHANGED}),
        ]


# =============================================================================
# LIFECYCLE CALLBACKS DATA STRUCTURE
# =============================================================================


@dataclass
class LifecycleCallback:
    """Registered lifecycle callback."""

    event: str
    callback: Callable
    priority: int = 0
    once: bool = False


# =============================================================================
# LIFECYCLE MANAGER
# =============================================================================


class LifecycleManager:
    """
    Manages entity lifecycle states and transitions.

    Features:
    - Deferred state transitions (batched to end of frame)
    - Callback registration and invocation
    - State history tracking
    - Thread-safe operations
    """

    _instance: ClassVar[Optional["LifecycleManager"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "LifecycleManager":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._entities: weakref.WeakValueDictionary[int, "Actor"] = weakref.WeakValueDictionary()
        self._pending_transitions: deque[Tuple[int, LifecycleState]] = deque()
        self._callbacks: Dict[str, List[LifecycleCallback]] = {}
        self._global_callbacks: Dict[str, List[Callable]] = {}
        self._state_counts: Dict[LifecycleState, int] = {s: 0 for s in LifecycleState}
        self._transition_lock = threading.Lock()
        self._initialized = True

    def register_entity(self, entity: "Actor") -> None:
        """Register an entity with the lifecycle manager."""
        entity_id = getattr(entity, "_entity_id", id(entity))
        self._entities[entity_id] = entity
        state = getattr(entity, "_lifecycle_state", LifecycleState.UNINITIALIZED)
        self._state_counts[state] += 1

    def unregister_entity(self, entity: "Actor") -> None:
        """Unregister an entity from the lifecycle manager."""
        entity_id = getattr(entity, "_entity_id", id(entity))
        if entity_id in self._entities:
            state = getattr(entity, "_lifecycle_state", LifecycleState.UNINITIALIZED)
            self._state_counts[state] -= 1
            del self._entities[entity_id]

    def request_transition(
        self,
        entity: "Actor",
        target_state: LifecycleState,
        immediate: bool = False,
    ) -> bool:
        """
        Request a state transition for an entity.

        Args:
            entity: The entity to transition
            target_state: The desired target state
            immediate: If True, perform transition now; otherwise defer

        Returns:
            True if transition is valid and queued/performed
        """
        current_state = getattr(entity, "_lifecycle_state", LifecycleState.UNINITIALIZED)
        valid_next = VALID_LIFECYCLE_TRANSITIONS.get(current_state, frozenset())

        if target_state not in valid_next:
            return False

        entity_id = getattr(entity, "_entity_id", id(entity))

        if immediate:
            return self._perform_transition(entity, target_state)

        with self._transition_lock:
            self._pending_transitions.append((entity_id, target_state))
        return True

    def _perform_transition(
        self,
        entity: "Actor",
        target_state: LifecycleState,
    ) -> bool:
        """Perform an immediate state transition."""
        old_state = getattr(entity, "_lifecycle_state", LifecycleState.UNINITIALIZED)

        # Update state counts
        self._state_counts[old_state] -= 1
        self._state_counts[target_state] += 1

        # Set new state (descriptor handles validation and callbacks)
        try:
            entity._lifecycle_state = target_state
        except ValueError:
            # Rollback counts on failure
            self._state_counts[old_state] += 1
            self._state_counts[target_state] -= 1
            return False

        # Log state change to EventLog
        entity_id = getattr(entity, "_entity_id", id(entity))
        try:
            event_log = _get_entity_event_log()
            event_log.record_state_change(entity_id, old_state, target_state)
        except Exception as e:
            _logger.debug("Failed to log state change event: %s", e)

        # Fire lifecycle event based on transition
        self._fire_lifecycle_event(entity, old_state, target_state)

        return True

    def _fire_lifecycle_event(
        self,
        entity: "Actor",
        old_state: LifecycleState,
        new_state: LifecycleState,
    ) -> None:
        """Fire appropriate lifecycle events for a state transition."""
        event_map = {
            (LifecycleState.CREATED, LifecycleState.INITIALIZING): None,
            (LifecycleState.INITIALIZED, LifecycleState.BEGINNING_PLAY): LifecycleEvent.BEGIN_PLAY,
            (LifecycleState.BEGINNING_PLAY, LifecycleState.ACTIVE): LifecycleEvent.ON_ACTIVATE,
            (LifecycleState.ACTIVE, LifecycleState.DEACTIVATING): LifecycleEvent.ON_DEACTIVATE,
            (LifecycleState.DEACTIVATING, LifecycleState.DEACTIVATED): LifecycleEvent.END_PLAY,
            (LifecycleState.DEACTIVATED, LifecycleState.BEGINNING_PLAY): LifecycleEvent.ON_ACTIVATE,
        }

        # Handle destroy transition from any state
        if new_state == LifecycleState.DESTROYING:
            # Log destroy event
            entity_id = getattr(entity, "_entity_id", id(entity))
            try:
                event_log = _get_entity_event_log()
                event_log.record_destroy(entity_id, reason="lifecycle_transition")
            except Exception as e:
                _logger.debug("Failed to log destroy event: %s", e)
            self._invoke_entity_callback(entity, LifecycleEvent.ON_DESTROY)
            return

        # Handle spawn (first activation)
        if old_state == LifecycleState.UNINITIALIZED and new_state == LifecycleState.CREATED:
            # Log spawn event
            entity_id = getattr(entity, "_entity_id", id(entity))
            entity_name = getattr(entity, "_name", type(entity).__name__)
            position = (0.0, 0.0, 0.0)
            if hasattr(entity, "_transform") and hasattr(entity._transform, "position"):
                position = entity._transform.position
            try:
                event_log = _get_entity_event_log()
                spawn_event = event_log.record_spawn(
                    entity_id,
                    prefab_name=entity_name,
                    position=position,
                    entity_type=type(entity).__name__,
                )
                # Store spawn event for causal chain tracking
                entity._spawn_event_id = spawn_event.id
            except Exception as e:
                _logger.debug("Failed to log spawn event: %s", e)
            self._invoke_entity_callback(entity, LifecycleEvent.ON_SPAWN)
            return

        event = event_map.get((old_state, new_state))
        if event:
            self._invoke_entity_callback(entity, event)

    def _invoke_entity_callback(self, entity: "Actor", event: str) -> None:
        """Invoke entity-specific lifecycle callbacks."""
        entity_name = getattr(entity, "_name", type(entity).__name__)

        # Entity-specific callbacks
        if hasattr(entity, "_lifecycle_callbacks"):
            callbacks = entity._lifecycle_callbacks.get(event, [])
            for cb in sorted(callbacks, key=lambda c: c.priority):
                try:
                    cb.callback(entity)
                except Exception as e:
                    _logger.warning(
                        "Entity callback '%s' failed for %s: %s",
                        event,
                        entity_name,
                        e,
                    )

        # Method-based callbacks
        method_name = event
        if hasattr(entity, method_name) and callable(getattr(entity, method_name)):
            try:
                getattr(entity, method_name)()
            except Exception as e:
                _logger.warning(
                    "Lifecycle method '%s' failed for %s: %s",
                    method_name,
                    entity_name,
                    e,
                )

        # Global callbacks
        global_callbacks = self._global_callbacks.get(event, [])
        for callback in global_callbacks:
            try:
                callback(entity)
            except Exception as e:
                _logger.warning(
                    "Global lifecycle callback '%s' failed for %s: %s",
                    event,
                    entity_name,
                    e,
                )

    def process_pending_transitions(self) -> int:
        """
        Process all pending state transitions.

        Called at the end of each frame to batch transitions.

        Returns:
            Number of transitions processed
        """
        count = 0
        with self._transition_lock:
            while self._pending_transitions:
                entity_id, target_state = self._pending_transitions.popleft()
                entity = self._entities.get(entity_id)
                if entity is not None:
                    if self._perform_transition(entity, target_state):
                        count += 1
        return count

    def register_global_callback(self, event: str, callback: Callable) -> None:
        """Register a global lifecycle callback for all entities."""
        if event not in self._global_callbacks:
            self._global_callbacks[event] = []
        self._global_callbacks[event].append(callback)

    def unregister_global_callback(self, event: str, callback: Callable) -> bool:
        """Unregister a global lifecycle callback."""
        if event in self._global_callbacks:
            try:
                self._global_callbacks[event].remove(callback)
                return True
            except ValueError:
                pass
        return False

    def get_entities_in_state(self, state: LifecycleState) -> List["Actor"]:
        """Get all entities currently in a specific state."""
        return [
            entity
            for entity in self._entities.values()
            if getattr(entity, "_lifecycle_state", None) == state
        ]

    def get_state_count(self, state: LifecycleState) -> int:
        """Get the count of entities in a specific state."""
        return self._state_counts.get(state, 0)

    def get_stats(self) -> Dict[str, Any]:
        """Get lifecycle manager statistics."""
        return {
            "total_entities": len(self._entities),
            "pending_transitions": len(self._pending_transitions),
            "state_counts": dict(self._state_counts),
            "global_callbacks": {k: len(v) for k, v in self._global_callbacks.items()},
        }

    def clear(self) -> None:
        """Clear all registered entities and callbacks (for testing)."""
        self._entities.clear()
        self._pending_transitions.clear()
        self._callbacks.clear()
        self._global_callbacks.clear()
        self._state_counts = {s: 0 for s in LifecycleState}

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear()
            cls._instance = None


# =============================================================================
# LIFECYCLE DECORATORS
# =============================================================================


def _build_lifecycle_hook_steps(params: dict) -> list[Step]:
    """Build steps for lifecycle hook decorator."""
    event = params.get("event", "")
    priority = params.get("priority", 0)
    return [
        Step(Op.TAG, {"key": "lifecycle_hook", "value": True}),
        Step(Op.TAG, {"key": "lifecycle_event", "value": event}),
        Step(Op.TAG, {"key": "lifecycle_priority", "value": priority}),
        Step(Op.HOOK, {"event": event}),
        Step(Op.REGISTER, {"registry": "lifecycle_hooks"}),
    ]


def _validate_lifecycle_hook_params(event: str = "", priority: int = 0, **kwargs: Any) -> None:
    """Validate lifecycle hook parameters."""
    valid_events = {
        LifecycleEvent.ON_SPAWN,
        LifecycleEvent.BEGIN_PLAY,
        LifecycleEvent.TICK,
        LifecycleEvent.END_PLAY,
        LifecycleEvent.ON_DESTROY,
        LifecycleEvent.ON_ACTIVATE,
        LifecycleEvent.ON_DEACTIVATE,
        LifecycleEvent.ON_STATE_CHANGED,
    }
    if event and event not in valid_events:
        raise ValueError(f"Invalid lifecycle event: {event}. Valid events: {valid_events}")


def _after_lifecycle_hook_steps(target: Any, params: dict) -> Any:
    """Post-processing for lifecycle hook decorator."""
    event = params.get("event", "")
    priority = params.get("priority", 0)

    # Store as lifecycle hook metadata
    target._lifecycle_event = event
    target._lifecycle_priority = priority
    target._lifecycle_hook = True

    return target


lifecycle_hook = make_decorator(
    name="lifecycle_hook",
    steps=_build_lifecycle_hook_steps,
    doc="Register a method as a lifecycle hook.",
    validate=_validate_lifecycle_hook_params,
    after_steps=_after_lifecycle_hook_steps,
)


# Convenience decorators for specific lifecycle events
def on_spawn(fn: Callable) -> Callable:
    """Decorator to register a spawn callback."""
    return lifecycle_hook(event=LifecycleEvent.ON_SPAWN)(fn)


def begin_play(fn: Callable) -> Callable:
    """Decorator to register a begin_play callback."""
    return lifecycle_hook(event=LifecycleEvent.BEGIN_PLAY)(fn)


def tick(fn: Callable) -> Callable:
    """Decorator to register a tick callback."""
    return lifecycle_hook(event=LifecycleEvent.TICK)(fn)


def end_play(fn: Callable) -> Callable:
    """Decorator to register an end_play callback."""
    return lifecycle_hook(event=LifecycleEvent.END_PLAY)(fn)


def on_destroy(fn: Callable) -> Callable:
    """Decorator to register a destroy callback."""
    return lifecycle_hook(event=LifecycleEvent.ON_DESTROY)(fn)


# =============================================================================
# LIFECYCLE MIXIN
# =============================================================================


class LifecycleMixin:
    """
    Mixin class providing lifecycle management functionality.

    Provides:
    - State tracking via descriptor
    - Callback registration
    - State transition methods
    """

    _lifecycle_state: LifecycleState = LifecycleState.UNINITIALIZED
    _lifecycle_callbacks: Dict[str, List[LifecycleCallback]]
    _lifecycle_manager: ClassVar[LifecycleManager]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Collect lifecycle hooks from methods
        cls._collected_lifecycle_hooks: Dict[str, List[Callable]] = {}
        for name in dir(cls):
            method = getattr(cls, name, None)
            if callable(method) and getattr(method, "_lifecycle_hook", False):
                event = getattr(method, "_lifecycle_event", "")
                if event:
                    if event not in cls._collected_lifecycle_hooks:
                        cls._collected_lifecycle_hooks[event] = []
                    cls._collected_lifecycle_hooks[event].append(method)

    def _init_lifecycle(self) -> None:
        """Initialize lifecycle tracking for this instance."""
        self._lifecycle_callbacks = {}
        self._lifecycle_manager = LifecycleManager()
        self._lifecycle_manager.register_entity(self)

        # Register collected hooks
        for event, methods in getattr(self.__class__, "_collected_lifecycle_hooks", {}).items():
            for method in methods:
                priority = getattr(method, "_lifecycle_priority", 0)
                self.register_lifecycle_callback(
                    event,
                    lambda e, m=method: m(e),
                    priority=priority,
                )

    def register_lifecycle_callback(
        self,
        event: str,
        callback: Callable,
        priority: int = 0,
        once: bool = False,
    ) -> None:
        """Register a lifecycle callback for this entity."""
        if event not in self._lifecycle_callbacks:
            self._lifecycle_callbacks[event] = []
        self._lifecycle_callbacks[event].append(
            LifecycleCallback(event=event, callback=callback, priority=priority, once=once)
        )

    def unregister_lifecycle_callback(self, event: str, callback: Callable) -> bool:
        """Unregister a lifecycle callback."""
        if event in self._lifecycle_callbacks:
            for i, cb in enumerate(self._lifecycle_callbacks[event]):
                if cb.callback == callback:
                    del self._lifecycle_callbacks[event][i]
                    return True
        return False

    def transition_to(self, state: LifecycleState, immediate: bool = False) -> bool:
        """Request a state transition."""
        return self._lifecycle_manager.request_transition(self, state, immediate)

    def get_lifecycle_state(self) -> LifecycleState:
        """Get the current lifecycle state."""
        return self._lifecycle_state

    def is_active(self) -> bool:
        """Check if entity is in active state."""
        return self._lifecycle_state == LifecycleState.ACTIVE

    def is_destroyed(self) -> bool:
        """Check if entity has been destroyed."""
        return self._lifecycle_state in (
            LifecycleState.DESTROYING,
            LifecycleState.DESTROYED,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "LifecycleState",
    "LifecycleEvent",
    # Descriptor
    "LifecycleStateDescriptor",
    # Callback
    "LifecycleCallback",
    # Manager
    "LifecycleManager",
    # Decorators
    "lifecycle_hook",
    "on_spawn",
    "begin_play",
    "tick",
    "end_play",
    "on_destroy",
    # Mixin
    "LifecycleMixin",
]
