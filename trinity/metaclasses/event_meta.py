"""
EventMeta - Metaclass for event types.

Handles event registration and schema validation.
Events are data-only objects used for communication between systems.
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar, Optional, get_type_hints

from trinity.constants import EVENT_POOL_MAX_SIZE
from trinity.decorators.ops import Op, Step
from trinity.metaclasses.engine_meta import EngineMeta


class EventMeta(EngineMeta):
    """
    Metaclass for event types.

    Created classes will:
    - Be registered in the event registry
    - Have a unique event type ID
    - Have fields validated (events must be data-only)
    - Support event inheritance for dispatch
    - Enable fast isinstance checks via ID comparison

    Optional class attributes (set by decorators):
    - _event_priority: int (dispatch priority, higher = earlier)
    - _event_channels: tuple[str, ...] (channels this event uses)
    - _event_pooled: bool (whether to pool event instances)

    Attached attributes:
    - _event_id: int (unique identifier)
    - _event_name: str (qualified name)
    - _event_fields: dict[str, type] (field name -> type)
    - _event_parent_ids: tuple[int, ...] (parent event IDs for inheritance)
    """

    _registry: ClassVar[dict[int, type]] = {}
    _name_to_id: ClassVar[dict[str, int]] = {}
    _next_id: ClassVar[int] = 1
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _event_pools: ClassVar[dict[type, list]] = {}  # event_cls -> pool of instances
    _event_pool_max_size: ClassVar[int] = EVENT_POOL_MAX_SIZE

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> EventMeta:
        """Create a new event type."""
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base Event class
        if name == "Event":
            return cls

        with mcs._lock:
            # === 1. GENERATE UNIQUE ID ===
            cls._event_id = mcs._next_id
            mcs._next_id += 1
            cls._event_name = f"{cls.__module__}.{name}"

            # 3.5.2: Record TAG steps for event_id and event_name
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "event_id", "value": cls._event_id})
            )
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "event_name", "value": cls._event_name})
            )

            # === 2. COLLECT FIELDS ===
            cls._event_fields = mcs._collect_fields(cls)

            # 3.5.3: Record DESCRIBE step for each field
            for field_name, field_type in cls._event_fields.items():
                type_name = field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)
                cls._metaclass_steps.append(
                    Step(Op.DESCRIBE, {"field": field_name, "type": type_name})
                )

            # === 3. TRACK INHERITANCE ===
            cls._event_parent_ids = mcs._collect_parent_ids(cls, bases)

            # 3.5.4: Record TAG for event_parents
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "event_parents", "value": cls._event_parent_ids})
            )

            # === 4. SET DEFAULTS ===
            if not hasattr(cls, "_event_priority"):
                cls._event_priority = 0
            if not hasattr(cls, "_event_channels"):
                cls._event_channels = ()
            if not hasattr(cls, "_event_pooled"):
                cls._event_pooled = False

            # 3.5.5: Record TAG steps for defaults
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "event_priority", "value": cls._event_priority})
            )
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "event_channels", "value": cls._event_channels})
            )
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "event_pooled", "value": cls._event_pooled})
            )

            # === 5. VALIDATE ===
            mcs._validate_event(cls)

            # 3.5.6: Record VALIDATE step
            cls._metaclass_steps.append(
                Step(Op.VALIDATE, {"constraint": "event_data_only"})
            )

            # === 6. REGISTER ===
            mcs._registry[cls._event_id] = cls
            mcs._name_to_id[cls._event_name] = cls._event_id

            # 3.5.7: Record REGISTER step
            cls._metaclass_steps.append(
                Step(Op.REGISTER, {"registry": "event_registry"})
            )

            # 3.5.8: If pooled, record HOOK steps for pool lifecycle
            if cls._event_pooled:
                cls._metaclass_steps.append(
                    Step(Op.HOOK, {"event": "pool_acquire"})
                )
                cls._metaclass_steps.append(
                    Step(Op.HOOK, {"event": "pool_release"})
                )

        return cls

    @classmethod
    def _collect_fields(mcs, cls: type) -> dict[str, type]:
        """Collect field annotations from the event class."""
        try:
            annotations = get_type_hints(cls)
        except Exception:
            annotations = getattr(cls, "__annotations__", {})

        fields = {}
        for field_name, field_type in annotations.items():
            if not field_name.startswith("_"):
                fields[field_name] = field_type

        return fields

    @classmethod
    def _collect_parent_ids(mcs, cls: type, bases: tuple[type, ...]) -> tuple[int, ...]:
        """Collect event IDs of parent event classes."""
        parent_ids = []
        for base in bases:
            if hasattr(base, "_event_id"):
                parent_ids.append(base._event_id)
                # Also include grandparent IDs for transitive inheritance
                if hasattr(base, "_event_parent_ids"):
                    parent_ids.extend(base._event_parent_ids)
        return tuple(dict.fromkeys(parent_ids))  # Remove duplicates, preserve order

    @classmethod
    def _validate_event(mcs, cls: type) -> None:
        """Validate event definition."""
        # Events must be data-only (no methods except __init__, __repr__, etc.)
        allowed_methods = {
            "__init__",
            "__repr__",
            "__str__",
            "__eq__",
            "__hash__",
            "__post_init__",  # For dataclasses
        }

        for name, value in vars(cls).items():
            if name.startswith("_"):
                continue

            if callable(value) and not isinstance(
                value, (classmethod, staticmethod, property)
            ):
                if name not in allowed_methods:
                    raise TypeError(
                        f"{cls.__name__}.{name}(): Events must be data-only. "
                        f"No methods allowed except {allowed_methods}."
                    )

    # =========================================================================
    # REGISTRY ACCESS CLASS METHODS
    # =========================================================================

    @classmethod
    def get_by_id(mcs, event_id: int) -> Optional[type]:
        """Get event class by ID."""
        return mcs._registry.get(event_id)

    @classmethod
    def get_by_name(mcs, name: str) -> Optional[type]:
        """Get event class by qualified name."""
        event_id = mcs._name_to_id.get(name)
        return mcs._registry.get(event_id) if event_id else None

    @classmethod
    def all_events(mcs) -> list[type]:
        """Get all registered event classes."""
        return list(mcs._registry.values())

    @classmethod
    def is_subtype(mcs, event_id: int, parent_id: int) -> bool:
        """
        Check if event_id is a subtype of parent_id.

        Includes the case where event_id == parent_id.
        """
        if event_id == parent_id:
            return True

        cls = mcs._registry.get(event_id)
        if cls is None:
            return False

        return parent_id in cls._event_parent_ids

    @classmethod
    def get_subtypes(mcs, parent_id: int) -> list[type]:
        """Get all event types that are subtypes of the given event."""
        return [
            cls
            for cls in mcs._registry.values()
            if mcs.is_subtype(cls._event_id, parent_id)
        ]

    @classmethod
    def get_by_channel(mcs, channel: str) -> list[type]:
        """Get all event types registered for a channel."""
        return [
            cls
            for cls in mcs._registry.values()
            if channel in getattr(cls, "_event_channels", ())
        ]

    @classmethod
    def clear_registry(mcs) -> None:
        """Clear the event registry. Useful for testing."""
        with mcs._lock:
            mcs._registry.clear()
            mcs._name_to_id.clear()
            mcs._next_id = 1
            mcs._event_pools.clear()
        super().clear_registry()

    # =========================================================================
    # EVENT POOLING
    # =========================================================================

    @classmethod
    def acquire(mcs, event_cls: type, **kwargs: Any) -> Any:
        """
        Acquire an event instance from the pool or create a new one.

        Args:
            event_cls: Event class to acquire.
            **kwargs: Arguments to pass to __init__.

        Returns:
            Event instance, either from pool or newly created.

        Raises:
            ValueError: If event_cls is not pooled but acquire() is called.
        """
        if not getattr(event_cls, "_event_pooled", False):
            # Non-pooled events should use regular constructor
            return event_cls(**kwargs)

        instance = None
        with mcs._lock:
            pool = mcs._event_pools.get(event_cls)
            if pool and len(pool) > 0:
                instance = pool.pop()

        # Initialize outside lock to prevent deadlock if __init__ acquires lock
        if instance is not None:
            if hasattr(instance, "__init__"):
                instance.__init__(**kwargs)
            return instance

        # No pooled instance available, create new
        return event_cls(**kwargs)

    @classmethod
    def release(mcs, instance: Any) -> None:
        """
        Return an event instance to the pool for reuse.

        Args:
            instance: Event instance to release.
        """
        event_cls = type(instance)
        if not getattr(event_cls, "_event_pooled", False):
            return

        with mcs._lock:
            if event_cls not in mcs._event_pools:
                mcs._event_pools[event_cls] = []

            pool = mcs._event_pools[event_cls]
            if len(pool) < mcs._event_pool_max_size:
                pool.append(instance)

    @classmethod
    def pool_stats(mcs, event_cls: type) -> dict[str, Any]:
        """
        Get pool statistics for an event class.

        Args:
            event_cls: Event class to get stats for.

        Returns:
            Dict with 'pooled', 'current_size', 'max_size'.
        """
        with mcs._lock:
            pooled = getattr(event_cls, "_event_pooled", False)
            pool = mcs._event_pools.get(event_cls, [])
            return {
                "pooled": pooled,
                "current_size": len(pool),
                "max_size": mcs._event_pool_max_size if pooled else 0,
            }

    # =========================================================================
    # EVENT SERIALIZATION
    # =========================================================================

    @classmethod
    def serialize(mcs, event_instance: Any) -> dict[str, Any]:
        """
        Serialize an event instance to a dictionary.

        Args:
            event_instance: Event instance to serialize.

        Returns:
            Dict with field_name -> value for all annotated fields.
            Nested events are recursively serialized.
            None values and missing fields are preserved.
        """
        event_cls = type(event_instance)
        result = {}

        fields = getattr(event_cls, "_event_fields", {})
        for field_name in fields:
            if not hasattr(event_instance, field_name):
                # Field not set - skip it (optional field)
                continue

            value = getattr(event_instance, field_name)

            # Handle None explicitly
            if value is None:
                result[field_name] = None
            # Recursively serialize nested events
            elif hasattr(type(value), "_event_id"):
                result[field_name] = mcs.serialize(value)
            elif isinstance(value, (list, tuple)):
                serialized_list = []
                for item in value:
                    if item is None:
                        serialized_list.append(None)
                    elif hasattr(type(item), "_event_id"):
                        serialized_list.append(mcs.serialize(item))
                    else:
                        serialized_list.append(item)
                result[field_name] = serialized_list
            else:
                result[field_name] = value

        return result

    @classmethod
    def deserialize(mcs, event_cls: type, data: dict[str, Any]) -> Any:
        """
        Deserialize a dictionary into an event instance.

        Args:
            event_cls: Event class to instantiate.
            data: Dict with field_name -> value.

        Returns:
            New event instance with fields populated from data.

        Raises:
            TypeError: If required fields are missing from data.
            ValueError: If data contains invalid values.
        """
        if not isinstance(data, dict):
            raise ValueError(f"deserialize() expects dict, got {type(data).__name__}")

        fields = getattr(event_cls, "_event_fields", {})
        kwargs = {}

        for field_name, field_type in fields.items():
            if field_name not in data:
                # Field not in data - will be handled by __init__
                continue

            value = data[field_name]

            # Handle None explicitly
            if value is None:
                kwargs[field_name] = None
            # Recursively deserialize nested events
            elif isinstance(value, dict) and hasattr(field_type, "_event_id"):
                kwargs[field_name] = mcs.deserialize(field_type, value)
            elif isinstance(value, (list, tuple)):
                # Try to deserialize list items if they look like event dicts
                deserialized_list = []
                for item in value:
                    if item is None:
                        deserialized_list.append(None)
                    elif isinstance(item, dict):
                        # Attempt to find event type (could be enhanced)
                        deserialized_list.append(item)  # Fallback to dict
                    else:
                        deserialized_list.append(item)
                kwargs[field_name] = deserialized_list
            else:
                kwargs[field_name] = value

        # Let __init__ raise TypeError if required fields are missing
        try:
            return event_cls(**kwargs)
        except TypeError as e:
            raise TypeError(f"Failed to deserialize {event_cls.__name__}: {e}") from e
