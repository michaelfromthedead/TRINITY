"""Foundation Tracker Integration for Animation Systems (T-AN-9.10).

This module wires the Foundation Tracker pattern to animation systems for
dirty-flag optimization. It provides:

- TrackedDescriptor on animation parameters (speed, direction, state)
- tracker.all_dirty() in animation graph system
- Only re-evaluate graph when parameters change
- tracker.on_change(AnimationState, callback) for type-level subscriptions
- Track current_state changes for state machine validation

Key Features:
- AnimationTrackedParameter class wrapping TrackedDescriptor
- AnimationParameterSet with dirty flag tracking
- on_change callbacks for parameter updates
- Integration with AnimationGraphSystem and StateMachineSystem
- Support for type-level subscriptions for AnimationState changes
- clear_dirty() and mark_dirty() utilities

Dependencies:
- Foundation Tracker system (from engine.rendering.demoscene.sdf_ast)
- T-AN-9.3 (AnimationGraphSystem)
- T-AN-5.7 (StateMachineSystem)

Usage:
    >>> from engine.animation.tracker_integration import (
    ...     AnimationTrackedParameter,
    ...     AnimationParameterSet,
    ...     TrackedAnimationComponent,
    ... )
    >>>
    >>> # Create a tracked parameter set
    >>> params = AnimationParameterSet()
    >>> params.register("speed", ParameterType.FLOAT, 0.0)
    >>> params.register("direction", ParameterType.FLOAT, 0.0)
    >>>
    >>> # Subscribe to changes
    >>> params.on_change("speed", lambda name, old, new: print(f"{name}: {old} -> {new}"))
    >>>
    >>> # Set value (triggers callback)
    >>> params.set("speed", 5.0)  # Prints: speed: 0.0 -> 5.0
    >>>
    >>> # Check dirty state
    >>> if params.all_dirty():
    ...     evaluate_graph()
    ...     params.clear_all_dirty()
"""

from __future__ import annotations

import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    Protocol,
    runtime_checkable,
)

from engine.animation.graph import (
    GraphParameter,
    ParameterType,
    AnimationGraph,
    StateMachine,
    GraphContext,
    Pose,
)

__all__ = [
    # Core tracker types
    "TrackedDescriptor",
    "TrackedField",
    "AnimationTracker",
    # Parameter tracking
    "AnimationTrackedParameter",
    "AnimationParameterSet",
    # State tracking
    "AnimationStateTracker",
    "StateChangeEvent",
    "StateTransitionRecord",
    # Component integration
    "TrackedAnimationComponent",
    "TrackedIKGoal",
    # Utilities
    "clear_dirty",
    "mark_dirty",
    "all_dirty",
    "any_dirty",
    # Type subscriptions
    "ChangeCallback",
    "TypeSubscription",
    "AnimationStateSubscription",
    # Integration helpers
    "wrap_parameter",
    "wrap_state_machine",
    "create_tracked_parameter_set",
]


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

T = TypeVar("T")
ChangeCallback = Callable[[str, Any, Any], None]  # (name, old_value, new_value) -> None
StateChangeCallback = Callable[["StateChangeEvent"], None]


# =============================================================================
# TRACKED DESCRIPTOR (Core dirty-tracking descriptor)
# =============================================================================


class TrackedDescriptor(Generic[T]):
    """A descriptor that tracks value changes and dirty state.

    This is the foundation for all tracked values in the animation system.
    It provides:
    - Change detection
    - Dirty flag tracking
    - Version counting
    - Callback notification

    Attributes:
        name: Field name
        default: Default value
        _storage_attr: Private attribute name for storage
        _dirty_attr: Private attribute name for dirty flag
        _version_attr: Private attribute name for version

    Example:
        >>> class MyClass:
        ...     speed = TrackedDescriptor[float]("speed", 0.0)
        ...
        >>> obj = MyClass()
        >>> obj.speed = 5.0  # Sets dirty flag, increments version
    """

    def __init__(
        self,
        name: str,
        default: T,
        notify_callback: Optional[Callable[[str, T, T], None]] = None,
    ) -> None:
        """Initialize tracked descriptor.

        Args:
            name: Field name for tracking
            default: Default value
            notify_callback: Optional callback(name, old, new) on change
        """
        self.name = name
        self.default = default
        self._storage_attr = f"_tracked_{name}_value"
        self._dirty_attr = f"_tracked_{name}_dirty"
        self._version_attr = f"_tracked_{name}_version"
        self._callback_attr = f"_tracked_{name}_callback"
        self._initial_callback = notify_callback

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when descriptor is assigned to a class attribute."""
        if not hasattr(owner, "_tracked_fields"):
            owner._tracked_fields = set()
        owner._tracked_fields.add(self.name)

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> T:
        """Get the tracked value."""
        if obj is None:
            return self  # type: ignore
        return getattr(obj, self._storage_attr, self.default)

    def __set__(self, obj: Any, value: T) -> None:
        """Set the tracked value and mark dirty if changed."""
        old_value = getattr(obj, self._storage_attr, self.default)

        if old_value != value:
            setattr(obj, self._storage_attr, value)
            setattr(obj, self._dirty_attr, True)

            # Increment version
            version = getattr(obj, self._version_attr, 0)
            setattr(obj, self._version_attr, version + 1)

            # Fire callback if set
            callback = getattr(obj, self._callback_attr, self._initial_callback)
            if callback is not None:
                try:
                    callback(self.name, old_value, value)
                except Exception:
                    pass  # Callbacks should not break the setter

    def is_dirty(self, obj: Any) -> bool:
        """Check if value has been modified."""
        return getattr(obj, self._dirty_attr, False)

    def clear_dirty(self, obj: Any) -> None:
        """Clear the dirty flag."""
        setattr(obj, self._dirty_attr, False)

    def mark_dirty(self, obj: Any) -> None:
        """Manually mark as dirty."""
        setattr(obj, self._dirty_attr, True)
        version = getattr(obj, self._version_attr, 0)
        setattr(obj, self._version_attr, version + 1)

    def get_version(self, obj: Any) -> int:
        """Get the version number."""
        return getattr(obj, self._version_attr, 0)

    def set_callback(
        self,
        obj: Any,
        callback: Optional[Callable[[str, T, T], None]],
    ) -> None:
        """Set or clear the change callback."""
        setattr(obj, self._callback_attr, callback)


# =============================================================================
# TRACKED FIELD (Runtime tracked field wrapper)
# =============================================================================


@dataclass
class TrackedField(Generic[T]):
    """Runtime wrapper for a tracked field.

    Unlike TrackedDescriptor which is a descriptor protocol,
    TrackedField is a data container that can be used dynamically.

    Attributes:
        name: Field name
        value: Current value
        is_dirty: Whether value has changed
        version: Change counter
        callbacks: List of change callbacks

    Example:
        >>> field = TrackedField("speed", 0.0)
        >>> field.on_change(lambda n, o, v: print(f"{n}: {o} -> {v}"))
        >>> field.set(5.0)  # Prints: speed: 0.0 -> 5.0
    """

    name: str
    value: T
    is_dirty: bool = False
    version: int = 0
    callbacks: List[ChangeCallback] = field(default_factory=list)

    def get(self) -> T:
        """Get the current value."""
        return self.value

    def set(self, new_value: T) -> bool:
        """Set value, returning True if changed."""
        if self.value != new_value:
            old_value = self.value
            self.value = new_value
            self.is_dirty = True
            self.version += 1

            # Fire callbacks
            for callback in self.callbacks:
                try:
                    callback(self.name, old_value, new_value)
                except Exception:
                    pass

            return True
        return False

    def on_change(self, callback: ChangeCallback) -> None:
        """Register a change callback."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: ChangeCallback) -> bool:
        """Remove a change callback."""
        try:
            self.callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def clear_dirty(self) -> None:
        """Clear dirty flag."""
        self.is_dirty = False

    def mark_dirty(self) -> None:
        """Mark as dirty."""
        self.is_dirty = True
        self.version += 1


# =============================================================================
# ANIMATION TRACKER (Aggregate dirty tracking for animation nodes)
# =============================================================================


class AnimationTracker:
    """Tracks dirty state for animation parameters and state.

    Provides aggregate dirty tracking across multiple fields, with support
    for hierarchical tracking (parent/child relationships).

    Attributes:
        fields: Dictionary of tracked fields
        _version: Global version counter
        _callbacks: Type-level callbacks

    Example:
        >>> tracker = AnimationTracker()
        >>> tracker.add_field("speed", 0.0)
        >>> tracker.add_field("direction", 0.0)
        >>> tracker.set("speed", 5.0)
        >>> tracker.is_dirty("speed")  # True
        >>> tracker.all_dirty()  # True (any field dirty)
    """

    __slots__ = ("_fields", "_version", "_callbacks", "_parent", "_children")

    def __init__(self) -> None:
        """Initialize tracker with empty fields."""
        self._fields: Dict[str, TrackedField] = {}
        self._version: int = 0
        self._callbacks: List[Tuple[str, ChangeCallback]] = []
        self._parent: Optional[weakref.ref["AnimationTracker"]] = None
        self._children: List[weakref.ref["AnimationTracker"]] = []

    def add_field(self, name: str, default_value: Any) -> TrackedField:
        """Add a tracked field."""
        field_obj = TrackedField(name=name, value=default_value)
        self._fields[name] = field_obj
        return field_obj

    def remove_field(self, name: str) -> bool:
        """Remove a tracked field."""
        if name in self._fields:
            del self._fields[name]
            return True
        return False

    def get(self, name: str) -> Any:
        """Get field value."""
        if name in self._fields:
            return self._fields[name].get()
        raise KeyError(f"Unknown field: {name}")

    def set(self, name: str, value: Any) -> bool:
        """Set field value."""
        if name in self._fields:
            changed = self._fields[name].set(value)
            if changed:
                self._version += 1
                self._notify_callbacks(name)
            return changed
        raise KeyError(f"Unknown field: {name}")

    def is_dirty(self, name: str) -> bool:
        """Check if specific field is dirty."""
        if name in self._fields:
            return self._fields[name].is_dirty
        return False

    def any_dirty(self) -> bool:
        """Check if any field is dirty."""
        return any(f.is_dirty for f in self._fields.values())

    def all_dirty(self) -> bool:
        """Check if any field is dirty (alias for any_dirty for API compatibility)."""
        return self.any_dirty()

    def get_dirty_fields(self) -> FrozenSet[str]:
        """Get names of all dirty fields."""
        return frozenset(n for n, f in self._fields.items() if f.is_dirty)

    def clear_dirty(self, name: str) -> None:
        """Clear dirty flag for specific field."""
        if name in self._fields:
            self._fields[name].clear_dirty()

    def clear_all_dirty(self) -> None:
        """Clear all dirty flags."""
        for field_obj in self._fields.values():
            field_obj.clear_dirty()

    def mark_dirty(self, name: str) -> None:
        """Mark specific field as dirty."""
        if name in self._fields:
            self._fields[name].mark_dirty()
            self._version += 1
            self._notify_callbacks(name)

    def mark_all_dirty(self) -> None:
        """Mark all fields as dirty."""
        for field_obj in self._fields.values():
            field_obj.mark_dirty()
        self._version += 1

    def on_change(
        self,
        field_name_or_type: Union[str, type],
        callback: ChangeCallback,
    ) -> None:
        """Register callback for field changes.

        Args:
            field_name_or_type: Field name or type to subscribe to
            callback: Callback(name, old_value, new_value)
        """
        if isinstance(field_name_or_type, str):
            # Field-level subscription
            if field_name_or_type in self._fields:
                self._fields[field_name_or_type].on_change(callback)
            self._callbacks.append((field_name_or_type, callback))
        else:
            # Type-level subscription (subscribe to all fields of this type)
            type_name = field_name_or_type.__name__
            self._callbacks.append((f"__type__{type_name}", callback))

    def _notify_callbacks(self, field_name: str) -> None:
        """Notify registered callbacks for a field change."""
        for pattern, callback in self._callbacks:
            if pattern == field_name:
                field_obj = self._fields.get(field_name)
                if field_obj:
                    try:
                        callback(field_name, None, field_obj.value)
                    except Exception:
                        pass

    def set_parent(self, parent: "AnimationTracker") -> None:
        """Set parent tracker for hierarchical tracking."""
        self._parent = weakref.ref(parent)
        parent._children.append(weakref.ref(self))

    def get_version(self) -> int:
        """Get global version number."""
        return self._version

    @property
    def fields(self) -> Dict[str, TrackedField]:
        """Get all tracked fields."""
        return self._fields

    def __repr__(self) -> str:
        dirty_count = sum(1 for f in self._fields.values() if f.is_dirty)
        return f"<AnimationTracker fields={len(self._fields)} dirty={dirty_count}>"


# =============================================================================
# ANIMATION TRACKED PARAMETER (Wrapper for GraphParameter)
# =============================================================================


@dataclass
class AnimationTrackedParameter:
    """Tracked wrapper for animation graph parameters.

    Wraps a GraphParameter with dirty tracking and change callbacks.

    Attributes:
        parameter: The underlying GraphParameter
        field: TrackedField for dirty tracking
        callbacks: Registered change callbacks

    Example:
        >>> param = GraphParameter("speed", ParameterType.FLOAT, 0.0)
        >>> tracked = AnimationTrackedParameter(param)
        >>> tracked.on_change(lambda n, o, v: print(f"Changed: {v}"))
        >>> tracked.set(5.0)  # Prints: Changed: 5.0
    """

    parameter: GraphParameter
    field: TrackedField = field(init=False)

    def __post_init__(self) -> None:
        """Initialize tracked field from parameter."""
        self.field = TrackedField(
            name=self.parameter.name,
            value=self.parameter.value,
        )

    @property
    def name(self) -> str:
        """Get parameter name."""
        return self.parameter.name

    @property
    def value(self) -> Any:
        """Get current value."""
        return self.field.value

    @property
    def param_type(self) -> ParameterType:
        """Get parameter type."""
        return self.parameter.param_type

    @property
    def is_dirty(self) -> bool:
        """Check if value changed."""
        return self.field.is_dirty

    @property
    def version(self) -> int:
        """Get version number."""
        return self.field.version

    def get(self) -> Any:
        """Get current value."""
        return self.field.get()

    def set(self, value: Any) -> bool:
        """Set value, syncing to underlying parameter."""
        if self.field.set(value):
            self.parameter.value = value
            return True
        return False

    def on_change(self, callback: ChangeCallback) -> None:
        """Register change callback."""
        self.field.on_change(callback)

    def clear_dirty(self) -> None:
        """Clear dirty flag."""
        self.field.clear_dirty()

    def mark_dirty(self) -> None:
        """Mark as dirty."""
        self.field.mark_dirty()

    def sync_from_parameter(self) -> bool:
        """Sync value from underlying parameter."""
        return self.field.set(self.parameter.value)


# =============================================================================
# ANIMATION PARAMETER SET (Collection of tracked parameters)
# =============================================================================


class AnimationParameterSet:
    """Collection of tracked animation parameters.

    Provides aggregate operations over multiple tracked parameters,
    with support for:
    - Bulk dirty checking
    - Type-level subscriptions
    - Integration with AnimationGraph

    Attributes:
        parameters: Dictionary of tracked parameters
        tracker: Underlying AnimationTracker
        _type_subscriptions: Type-level change subscriptions

    Example:
        >>> params = AnimationParameterSet()
        >>> params.register("speed", ParameterType.FLOAT, 0.0)
        >>> params.register("direction", ParameterType.FLOAT, 0.0)
        >>> params.set("speed", 5.0)
        >>> params.all_dirty()  # True
        >>> params.get_dirty_names()  # {"speed"}
    """

    __slots__ = ("_parameters", "_tracker", "_type_subscriptions", "_graph")

    def __init__(self, graph: Optional[AnimationGraph] = None) -> None:
        """Initialize parameter set.

        Args:
            graph: Optional AnimationGraph to sync parameters from
        """
        self._parameters: Dict[str, AnimationTrackedParameter] = {}
        self._tracker = AnimationTracker()
        self._type_subscriptions: Dict[type, List[ChangeCallback]] = {}
        self._graph = graph

        if graph:
            self._sync_from_graph(graph)

    def _sync_from_graph(self, graph: AnimationGraph) -> None:
        """Sync parameters from an AnimationGraph."""
        for name, param in graph.parameters.items():
            self.register(name, param.param_type, param.value)

    def register(
        self,
        name: str,
        param_type: ParameterType,
        default_value: Any,
    ) -> AnimationTrackedParameter:
        """Register a new tracked parameter.

        Args:
            name: Parameter name
            param_type: Parameter type
            default_value: Initial value

        Returns:
            The registered tracked parameter.
        """
        param = GraphParameter(name=name, param_type=param_type, default_value=default_value)
        tracked = AnimationTrackedParameter(parameter=param)
        self._parameters[name] = tracked
        self._tracker.add_field(name, default_value)
        return tracked

    def unregister(self, name: str) -> bool:
        """Unregister a parameter."""
        if name in self._parameters:
            del self._parameters[name]
            self._tracker.remove_field(name)
            return True
        return False

    def get(self, name: str) -> Any:
        """Get parameter value."""
        if name in self._parameters:
            return self._parameters[name].get()
        raise KeyError(f"Unknown parameter: {name}")

    def set(self, name: str, value: Any) -> bool:
        """Set parameter value."""
        if name in self._parameters:
            param = self._parameters[name]
            changed = param.set(value)
            if changed:
                self._tracker.set(name, value)
                self._notify_type_subscriptions(name, param)
            return changed
        raise KeyError(f"Unknown parameter: {name}")

    def _notify_type_subscriptions(
        self,
        name: str,
        param: AnimationTrackedParameter,
    ) -> None:
        """Notify type-level subscriptions."""
        value = param.value
        value_type = type(value)
        if value_type in self._type_subscriptions:
            for callback in self._type_subscriptions[value_type]:
                try:
                    callback(name, None, value)
                except Exception:
                    pass

    def is_dirty(self, name: str) -> bool:
        """Check if specific parameter is dirty."""
        if name in self._parameters:
            return self._parameters[name].is_dirty
        return False

    def any_dirty(self) -> bool:
        """Check if any parameter is dirty."""
        return any(p.is_dirty for p in self._parameters.values())

    def all_dirty(self) -> bool:
        """Check if any parameter is dirty (API compatibility with tracker.all_dirty())."""
        return self.any_dirty()

    def get_dirty_names(self) -> FrozenSet[str]:
        """Get names of all dirty parameters."""
        return frozenset(n for n, p in self._parameters.items() if p.is_dirty)

    def clear_dirty(self, name: str) -> None:
        """Clear dirty flag for specific parameter."""
        if name in self._parameters:
            self._parameters[name].clear_dirty()
            self._tracker.clear_dirty(name)

    def clear_all_dirty(self) -> None:
        """Clear all dirty flags."""
        for param in self._parameters.values():
            param.clear_dirty()
        self._tracker.clear_all_dirty()

    def mark_dirty(self, name: str) -> None:
        """Mark specific parameter as dirty."""
        if name in self._parameters:
            self._parameters[name].mark_dirty()
            self._tracker.mark_dirty(name)

    def mark_all_dirty(self) -> None:
        """Mark all parameters as dirty."""
        for param in self._parameters.values():
            param.mark_dirty()
        self._tracker.mark_all_dirty()

    def on_change(
        self,
        name_or_type: Union[str, type],
        callback: ChangeCallback,
    ) -> None:
        """Register change callback.

        Args:
            name_or_type: Parameter name or type to subscribe to
            callback: Callback(name, old_value, new_value)
        """
        if isinstance(name_or_type, str):
            if name_or_type in self._parameters:
                self._parameters[name_or_type].on_change(callback)
        else:
            # Type-level subscription
            if name_or_type not in self._type_subscriptions:
                self._type_subscriptions[name_or_type] = []
            self._type_subscriptions[name_or_type].append(callback)

    def sync_to_graph(self, graph: AnimationGraph) -> int:
        """Sync dirty parameters to an AnimationGraph.

        Returns:
            Number of parameters synced.
        """
        synced = 0
        for name, param in self._parameters.items():
            if param.is_dirty and name in graph.parameters:
                graph.parameters[name].value = param.value
                synced += 1
        return synced

    def get_version(self) -> int:
        """Get global version number."""
        return self._tracker.get_version()

    @property
    def parameters(self) -> Dict[str, AnimationTrackedParameter]:
        """Get all parameters."""
        return self._parameters

    def __len__(self) -> int:
        return len(self._parameters)

    def __contains__(self, name: str) -> bool:
        return name in self._parameters

    def __iter__(self):
        return iter(self._parameters.values())

    def __repr__(self) -> str:
        dirty_count = sum(1 for p in self._parameters.values() if p.is_dirty)
        return f"<AnimationParameterSet count={len(self)} dirty={dirty_count}>"


# =============================================================================
# STATE CHANGE EVENT
# =============================================================================


@dataclass(frozen=True)
class StateChangeEvent:
    """Event fired when animation state changes.

    Attributes:
        old_state: Previous state name
        new_state: New state name
        transition_duration: Duration of transition (if any)
        is_transition: Whether this is a transition or instant change
        timestamp: Event timestamp (frame or time)
    """

    old_state: str
    new_state: str
    transition_duration: float = 0.0
    is_transition: bool = False
    timestamp: float = 0.0


@dataclass
class StateTransitionRecord:
    """Records a state transition for validation.

    Attributes:
        from_state: Source state
        to_state: Target state
        conditions_met: Conditions that triggered transition
        timestamp: When transition occurred
        frame: Frame number when transition occurred
    """

    from_state: str
    to_state: str
    conditions_met: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    frame: int = 0


# =============================================================================
# ANIMATION STATE TRACKER
# =============================================================================


class AnimationStateTracker:
    """Tracks animation state machine state for validation.

    Provides:
    - State change detection
    - Transition validation
    - State history tracking
    - Type-level subscriptions for AnimationState

    Attributes:
        current_state: Current state name
        previous_state: Previous state name
        history: List of state transitions
        callbacks: State change callbacks

    Example:
        >>> tracker = AnimationStateTracker()
        >>> tracker.on_change(AnimationState, lambda e: print(f"State: {e.new_state}"))
        >>> tracker.set_state("idle")  # Prints: State: idle
        >>> tracker.transition_to("walk", 0.3)  # Prints: State: walk
    """

    __slots__ = (
        "_current_state",
        "_previous_state",
        "_is_dirty",
        "_version",
        "_history",
        "_callbacks",
        "_max_history",
        "_frame",
        "_time",
    )

    def __init__(self, max_history: int = 100) -> None:
        """Initialize state tracker.

        Args:
            max_history: Maximum state transitions to keep in history
        """
        self._current_state: str = ""
        self._previous_state: str = ""
        self._is_dirty: bool = False
        self._version: int = 0
        self._history: List[StateTransitionRecord] = []
        self._callbacks: List[StateChangeCallback] = []
        self._max_history = max_history
        self._frame: int = 0
        self._time: float = 0.0

    @property
    def current_state(self) -> str:
        """Get current state name."""
        return self._current_state

    @property
    def previous_state(self) -> str:
        """Get previous state name."""
        return self._previous_state

    @property
    def is_dirty(self) -> bool:
        """Check if state has changed."""
        return self._is_dirty

    @property
    def version(self) -> int:
        """Get version number."""
        return self._version

    @property
    def history(self) -> List[StateTransitionRecord]:
        """Get transition history."""
        return self._history

    def set_state(self, state_name: str) -> bool:
        """Set current state directly (instant transition).

        Args:
            state_name: New state name

        Returns:
            True if state changed.
        """
        if self._current_state != state_name:
            old_state = self._current_state
            self._previous_state = old_state
            self._current_state = state_name
            self._is_dirty = True
            self._version += 1

            # Record transition
            record = StateTransitionRecord(
                from_state=old_state,
                to_state=state_name,
                timestamp=self._time,
                frame=self._frame,
            )
            self._add_history(record)

            # Fire callbacks
            event = StateChangeEvent(
                old_state=old_state,
                new_state=state_name,
                is_transition=False,
                timestamp=self._time,
            )
            self._fire_callbacks(event)

            return True
        return False

    def transition_to(
        self,
        state_name: str,
        duration: float = 0.0,
        conditions: Optional[List[str]] = None,
    ) -> bool:
        """Start transition to new state.

        Args:
            state_name: Target state name
            duration: Transition duration in seconds
            conditions: List of condition names that triggered transition

        Returns:
            True if transition started.
        """
        if self._current_state != state_name:
            old_state = self._current_state
            self._previous_state = old_state
            self._current_state = state_name
            self._is_dirty = True
            self._version += 1

            # Record transition
            record = StateTransitionRecord(
                from_state=old_state,
                to_state=state_name,
                conditions_met=conditions or [],
                timestamp=self._time,
                frame=self._frame,
            )
            self._add_history(record)

            # Fire callbacks
            event = StateChangeEvent(
                old_state=old_state,
                new_state=state_name,
                transition_duration=duration,
                is_transition=duration > 0,
                timestamp=self._time,
            )
            self._fire_callbacks(event)

            return True
        return False

    def _add_history(self, record: StateTransitionRecord) -> None:
        """Add record to history, trimming if needed."""
        self._history.append(record)
        while len(self._history) > self._max_history:
            self._history.pop(0)

    def _fire_callbacks(self, event: StateChangeEvent) -> None:
        """Fire all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def on_change(
        self,
        type_or_callback: Union[type, StateChangeCallback],
        callback: Optional[StateChangeCallback] = None,
    ) -> None:
        """Register state change callback.

        Can be called as:
        - on_change(callback) - Direct callback registration
        - on_change(AnimationState, callback) - Type-level subscription

        Args:
            type_or_callback: Either a type (for type-level subscription) or callback
            callback: Callback when type_or_callback is a type
        """
        if callback is not None:
            # Type-level subscription: on_change(AnimationState, callback)
            self._callbacks.append(callback)
        elif callable(type_or_callback):
            # Direct callback: on_change(callback)
            self._callbacks.append(type_or_callback)

    def clear_dirty(self) -> None:
        """Clear dirty flag."""
        self._is_dirty = False

    def mark_dirty(self) -> None:
        """Mark as dirty."""
        self._is_dirty = True
        self._version += 1

    def update_time(self, time: float, frame: int) -> None:
        """Update time tracking.

        Args:
            time: Current time in seconds
            frame: Current frame number
        """
        self._time = time
        self._frame = frame

    def get_transition_count(self, from_state: str, to_state: str) -> int:
        """Count transitions between two states in history."""
        return sum(
            1 for r in self._history
            if r.from_state == from_state and r.to_state == to_state
        )

    def validate_transition(self, from_state: str, to_state: str) -> bool:
        """Check if transition from->to has occurred.

        Args:
            from_state: Source state
            to_state: Target state

        Returns:
            True if this transition exists in history.
        """
        return any(
            r.from_state == from_state and r.to_state == to_state
            for r in self._history
        )

    def clear_history(self) -> None:
        """Clear transition history."""
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"<AnimationStateTracker state='{self._current_state}' "
            f"dirty={self._is_dirty} version={self._version}>"
        )


# =============================================================================
# TYPE SUBSCRIPTION (For type-level change subscriptions)
# =============================================================================


@dataclass
class TypeSubscription(Generic[T]):
    """Subscription to changes of a specific type.

    Attributes:
        target_type: Type to subscribe to
        callback: Callback to invoke on change
        filter_fn: Optional filter function
    """

    target_type: Type[T]
    callback: Callable[[str, T, T], None]
    filter_fn: Optional[Callable[[T], bool]] = None

    def matches(self, value: Any) -> bool:
        """Check if value matches this subscription."""
        if not isinstance(value, self.target_type):
            return False
        if self.filter_fn and not self.filter_fn(value):
            return False
        return True


# Convenience alias for AnimationState type subscriptions
AnimationStateSubscription = TypeSubscription


# =============================================================================
# TRACKED ANIMATION COMPONENT
# =============================================================================


@dataclass
class TrackedAnimationComponent:
    """Animation component with integrated tracking.

    Combines:
    - Parameter tracking via AnimationParameterSet
    - State tracking via AnimationStateTracker
    - Integration with AnimationGraph and StateMachine

    Attributes:
        parameters: Tracked parameter set
        state_tracker: State change tracker
        graph: Optional AnimationGraph
        state_machine: Optional StateMachine
        enabled: Whether component is enabled
    """

    parameters: AnimationParameterSet = field(default_factory=AnimationParameterSet)
    state_tracker: AnimationStateTracker = field(default_factory=AnimationStateTracker)
    graph: Optional[AnimationGraph] = None
    state_machine: Optional[StateMachine] = None
    enabled: bool = True

    # Dirty tracking
    _needs_evaluation: bool = True
    _last_eval_frame: int = -1

    def needs_evaluation(self, current_frame: int) -> bool:
        """Check if graph needs re-evaluation.

        Returns True if:
        - Any parameter is dirty
        - State has changed
        - Never evaluated before
        - Force evaluation requested
        """
        if not self.enabled:
            return False

        if self._last_eval_frame < 0:
            return True

        if self._needs_evaluation:
            return True

        if self.parameters.any_dirty():
            return True

        if self.state_tracker.is_dirty:
            return True

        return False

    def mark_evaluated(self, frame: int) -> None:
        """Mark component as evaluated for this frame."""
        self._last_eval_frame = frame
        self._needs_evaluation = False
        self.parameters.clear_all_dirty()
        self.state_tracker.clear_dirty()

    def invalidate(self) -> None:
        """Force re-evaluation on next frame."""
        self._needs_evaluation = True
        self.parameters.mark_all_dirty()
        self.state_tracker.mark_dirty()

    def set_parameter(self, name: str, value: Any) -> bool:
        """Set a parameter value."""
        return self.parameters.set(name, value)

    def get_parameter(self, name: str) -> Any:
        """Get a parameter value."""
        return self.parameters.get(name)

    def set_state(self, state_name: str) -> bool:
        """Set current animation state."""
        return self.state_tracker.set_state(state_name)

    def transition_to(
        self,
        state_name: str,
        duration: float = 0.0,
    ) -> bool:
        """Start transition to new state."""
        return self.state_tracker.transition_to(state_name, duration)

    def sync_to_graph(self) -> int:
        """Sync dirty parameters to graph."""
        if self.graph:
            return self.parameters.sync_to_graph(self.graph)
        return 0


# =============================================================================
# TRACKED IK GOAL
# =============================================================================


@dataclass
class TrackedIKGoal:
    """IK goal with dirty tracking for optimization.

    Tracks changes to IK goal position/rotation to skip re-solving
    when goals haven't changed.

    Attributes:
        goal_id: Unique goal identifier
        chain_name: Name of IK chain
        tracker: AnimationTracker for position/rotation
        enabled: Whether goal is active
    """

    goal_id: str
    chain_name: str
    tracker: AnimationTracker = field(default_factory=AnimationTracker)
    enabled: bool = True

    def __post_init__(self) -> None:
        """Initialize tracked fields."""
        self.tracker.add_field("position_x", 0.0)
        self.tracker.add_field("position_y", 0.0)
        self.tracker.add_field("position_z", 0.0)
        self.tracker.add_field("rotation_x", 0.0)
        self.tracker.add_field("rotation_y", 0.0)
        self.tracker.add_field("rotation_z", 0.0)
        self.tracker.add_field("rotation_w", 1.0)
        self.tracker.add_field("weight", 1.0)

    def set_position(self, x: float, y: float, z: float) -> bool:
        """Set goal position."""
        changed = False
        changed |= self.tracker.set("position_x", x)
        changed |= self.tracker.set("position_y", y)
        changed |= self.tracker.set("position_z", z)
        return changed

    def set_rotation(self, x: float, y: float, z: float, w: float) -> bool:
        """Set goal rotation (quaternion)."""
        changed = False
        changed |= self.tracker.set("rotation_x", x)
        changed |= self.tracker.set("rotation_y", y)
        changed |= self.tracker.set("rotation_z", z)
        changed |= self.tracker.set("rotation_w", w)
        return changed

    def set_weight(self, weight: float) -> bool:
        """Set goal weight."""
        return self.tracker.set("weight", weight)

    @property
    def is_dirty(self) -> bool:
        """Check if goal has changed."""
        return self.tracker.any_dirty()

    def clear_dirty(self) -> None:
        """Clear dirty flags."""
        self.tracker.clear_all_dirty()

    def needs_solving(self) -> bool:
        """Check if goal needs IK solving."""
        return self.enabled and self.is_dirty


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def clear_dirty(obj: Any) -> None:
    """Clear dirty state on an object.

    Works with:
    - AnimationTracker
    - AnimationParameterSet
    - AnimationStateTracker
    - TrackedAnimationComponent
    - Any object with clear_dirty or clear_all_dirty method
    """
    if hasattr(obj, "clear_all_dirty"):
        obj.clear_all_dirty()
    elif hasattr(obj, "clear_dirty"):
        obj.clear_dirty()


def mark_dirty(obj: Any, field_name: Optional[str] = None) -> None:
    """Mark object or field as dirty.

    Args:
        obj: Object to mark dirty
        field_name: Optional specific field to mark
    """
    if field_name and hasattr(obj, "mark_dirty"):
        obj.mark_dirty(field_name)
    elif hasattr(obj, "mark_all_dirty"):
        obj.mark_all_dirty()
    elif hasattr(obj, "mark_dirty"):
        obj.mark_dirty()


def all_dirty(obj: Any) -> bool:
    """Check if any field/parameter is dirty.

    Works with:
    - AnimationTracker
    - AnimationParameterSet
    - AnimationStateTracker
    - TrackedAnimationComponent
    """
    if hasattr(obj, "all_dirty"):
        return obj.all_dirty()
    elif hasattr(obj, "any_dirty"):
        return obj.any_dirty()
    elif hasattr(obj, "is_dirty"):
        return obj.is_dirty
    return False


def any_dirty(obj: Any) -> bool:
    """Alias for all_dirty for API consistency."""
    return all_dirty(obj)


def wrap_parameter(param: GraphParameter) -> AnimationTrackedParameter:
    """Wrap a GraphParameter with tracking.

    Args:
        param: GraphParameter to wrap

    Returns:
        Tracked parameter wrapper.
    """
    return AnimationTrackedParameter(parameter=param)


def wrap_state_machine(sm: StateMachine) -> AnimationStateTracker:
    """Create a state tracker for a StateMachine.

    Args:
        sm: StateMachine to track

    Returns:
        State tracker for the machine.
    """
    tracker = AnimationStateTracker()
    if sm.current_state:
        tracker.set_state(sm.current_state.name)
    return tracker


def create_tracked_parameter_set(
    graph: AnimationGraph,
) -> AnimationParameterSet:
    """Create a tracked parameter set from an AnimationGraph.

    Args:
        graph: AnimationGraph to extract parameters from

    Returns:
        Parameter set with all graph parameters tracked.
    """
    return AnimationParameterSet(graph=graph)
