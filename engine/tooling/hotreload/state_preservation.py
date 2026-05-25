"""
State Preservation - Serialize and restore object state across hot-reloads.

Provides strategies for preserving object state using Foundation's Serializer
and custom preservation logic for complex scenarios.
"""
from __future__ import annotations

import copy
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar

from foundation import to_dict, from_dict, mirror, deep_copy, schema_hash


T = TypeVar("T")


class PreservationStrategy(Enum):
    """Strategies for preserving object state."""

    # Use Foundation's Serializer (default)
    SERIALIZER = auto()

    # Use Mirror to extract field values directly
    MIRROR = auto()

    # Use __getstate__/__setstate__ if available
    PICKLE_PROTOCOL = auto()

    # Custom preservation using __preserve__/__restore__ methods
    CUSTOM = auto()

    # No preservation - reset to defaults
    NONE = auto()

    # Shallow copy of __dict__
    SHALLOW_COPY = auto()

    # Deep copy of __dict__
    DEEP_COPY = auto()


@dataclass
class StateSnapshot:
    """A snapshot of an object's state."""

    obj_id: int
    class_name: str
    module_name: str
    schema_hash: str
    timestamp: float
    state: Dict[str, Any]
    strategy: PreservationStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)

    def age(self) -> float:
        """Get the age of this snapshot in seconds."""
        return time.time() - self.timestamp

    def is_stale(self, max_age: float = 60.0) -> bool:
        """Check if the snapshot is older than max_age seconds."""
        return self.age() > max_age


@dataclass
class PreservationConfig:
    """Configuration for state preservation."""

    strategy: PreservationStrategy = PreservationStrategy.SERIALIZER
    include_fields: Optional[Set[str]] = None  # None = all fields
    exclude_fields: Set[str] = field(default_factory=set)
    max_depth: int = 10  # For nested objects
    preserve_refs: bool = True  # Preserve object references
    validate_types: bool = True  # Validate field types on restore


class StatePreserver:
    """
    Manages state preservation and restoration for hot-reloaded objects.

    Features:
    - Multiple preservation strategies
    - Configurable field inclusion/exclusion
    - Type validation on restore
    - Reference tracking for object graphs
    - Migration support for schema changes
    """

    # Constants
    DEFAULT_MAX_SNAPSHOTS = 100
    DEFAULT_SNAPSHOT_TTL = 300.0  # 5 minutes

    def __init__(
        self,
        max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
        snapshot_ttl: float = DEFAULT_SNAPSHOT_TTL,
    ):
        """
        Initialize the state preserver.

        Args:
            max_snapshots: Maximum number of snapshots to keep.
            snapshot_ttl: Time-to-live for snapshots in seconds.
        """
        self._max_snapshots = max_snapshots
        self._snapshot_ttl = snapshot_ttl
        self._snapshots: Dict[int, StateSnapshot] = {}
        self._configs: Dict[str, PreservationConfig] = {}  # class_name -> config
        self._ref_map: Dict[int, weakref.ref] = {}

    def configure(
        self,
        class_name: str,
        config: PreservationConfig,
    ) -> None:
        """
        Configure preservation for a class.

        Args:
            class_name: Full class name (module.ClassName).
            config: Preservation configuration.
        """
        self._configs[class_name] = config

    def get_config(self, class_name: str) -> PreservationConfig:
        """Get preservation config for a class."""
        return self._configs.get(class_name, PreservationConfig())

    def preserve(
        self,
        obj: Any,
        strategy: Optional[PreservationStrategy] = None,
    ) -> StateSnapshot:
        """
        Preserve the state of an object.

        Args:
            obj: Object to preserve.
            strategy: Override the default strategy.

        Returns:
            StateSnapshot containing the preserved state.
        """
        obj_id = id(obj)
        cls = type(obj)
        class_name = f"{cls.__module__}.{cls.__name__}"

        config = self.get_config(class_name)
        used_strategy = strategy or config.strategy

        # Get state based on strategy
        state = self._extract_state(obj, used_strategy, config)

        snapshot = StateSnapshot(
            obj_id=obj_id,
            class_name=cls.__name__,
            module_name=cls.__module__,
            schema_hash=schema_hash(cls),
            timestamp=time.time(),
            state=state,
            strategy=used_strategy,
        )

        # Store snapshot
        self._snapshots[obj_id] = snapshot
        self._ref_map[obj_id] = weakref.ref(obj)

        # Cleanup old snapshots
        self._cleanup_snapshots()

        return snapshot

    def restore(
        self,
        obj: Any,
        snapshot: Optional[StateSnapshot] = None,
        migration_fn: Optional[Callable[[Dict, str, str], Dict]] = None,
    ) -> bool:
        """
        Restore state to an object.

        Args:
            obj: Object to restore state to.
            snapshot: Snapshot to restore from (or use stored snapshot).
            migration_fn: Optional migration function for schema changes.

        Returns:
            True if restore was successful.
        """
        obj_id = id(obj)

        if snapshot is None:
            snapshot = self._snapshots.get(obj_id)
            if snapshot is None:
                return False

        cls = type(obj)
        class_name = f"{cls.__module__}.{cls.__name__}"
        config = self.get_config(class_name)

        state = snapshot.state

        # Apply migration if schema changed
        current_hash = schema_hash(cls)
        if snapshot.schema_hash != current_hash and migration_fn:
            state = migration_fn(state, snapshot.schema_hash, current_hash)

        # Restore based on strategy
        return self._apply_state(obj, state, snapshot.strategy, config)

    def get_snapshot(self, obj_or_id: Any) -> Optional[StateSnapshot]:
        """
        Get the snapshot for an object.

        Args:
            obj_or_id: Object or object ID.

        Returns:
            StateSnapshot or None.
        """
        obj_id = obj_or_id if isinstance(obj_or_id, int) else id(obj_or_id)
        return self._snapshots.get(obj_id)

    def has_snapshot(self, obj_or_id: Any) -> bool:
        """Check if a snapshot exists for an object."""
        obj_id = obj_or_id if isinstance(obj_or_id, int) else id(obj_or_id)
        return obj_id in self._snapshots

    def clear_snapshot(self, obj_or_id: Any) -> bool:
        """
        Remove a snapshot.

        Args:
            obj_or_id: Object or object ID.

        Returns:
            True if snapshot was removed.
        """
        obj_id = obj_or_id if isinstance(obj_or_id, int) else id(obj_or_id)
        if obj_id in self._snapshots:
            del self._snapshots[obj_id]
            self._ref_map.pop(obj_id, None)
            return True
        return False

    def clear_all(self) -> int:
        """
        Clear all snapshots.

        Returns:
            Number of snapshots cleared.
        """
        count = len(self._snapshots)
        self._snapshots.clear()
        self._ref_map.clear()
        return count

    def _extract_state(
        self,
        obj: Any,
        strategy: PreservationStrategy,
        config: PreservationConfig,
    ) -> Dict[str, Any]:
        """Extract state from an object using the specified strategy."""

        if strategy == PreservationStrategy.NONE:
            return {}

        elif strategy == PreservationStrategy.SERIALIZER:
            try:
                state = to_dict(obj, include_schema_hash=False)
                return self._filter_fields(state, config)
            except Exception:
                # Fallback to mirror
                return self._extract_via_mirror(obj, config)

        elif strategy == PreservationStrategy.MIRROR:
            return self._extract_via_mirror(obj, config)

        elif strategy == PreservationStrategy.PICKLE_PROTOCOL:
            if hasattr(obj, "__getstate__"):
                state = obj.__getstate__()
                if isinstance(state, dict):
                    return self._filter_fields(state, config)
                return {"__state__": state}
            return self._extract_via_mirror(obj, config)

        elif strategy == PreservationStrategy.CUSTOM:
            if hasattr(obj, "__preserve__"):
                return obj.__preserve__()
            return self._extract_via_mirror(obj, config)

        elif strategy == PreservationStrategy.SHALLOW_COPY:
            if hasattr(obj, "__dict__"):
                return self._filter_fields(dict(obj.__dict__), config)
            return self._extract_via_mirror(obj, config)

        elif strategy == PreservationStrategy.DEEP_COPY:
            if hasattr(obj, "__dict__"):
                try:
                    return self._filter_fields(copy.deepcopy(obj.__dict__), config)
                except Exception:
                    return self._filter_fields(dict(obj.__dict__), config)
            return self._extract_via_mirror(obj, config)

        return {}

    def _extract_via_mirror(
        self,
        obj: Any,
        config: PreservationConfig,
    ) -> Dict[str, Any]:
        """Extract state using Foundation's Mirror."""
        m = mirror(obj)
        state = {}

        for name, info in m.fields.items():
            if config.include_fields and name not in config.include_fields:
                continue
            if name in config.exclude_fields:
                continue
            if info.metadata.get("transient"):
                continue

            try:
                value = m.get(name)
                state[name] = self._serialize_value(value, config.max_depth)
            except (AttributeError, KeyError):
                pass

        return state

    def _serialize_value(self, value: Any, max_depth: int, depth: int = 0) -> Any:
        """Serialize a value, handling nested objects."""
        if depth >= max_depth:
            return None

        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v, max_depth, depth + 1) for v in value]

        if isinstance(value, dict):
            return {
                k: self._serialize_value(v, max_depth, depth + 1)
                for k, v in value.items()
            }

        if isinstance(value, set):
            return list(value)

        # For objects, try to serialize
        try:
            return to_dict(value, include_schema_hash=False)
        except Exception:
            return str(value)

    def _filter_fields(
        self,
        state: Dict[str, Any],
        config: PreservationConfig,
    ) -> Dict[str, Any]:
        """Filter state fields based on configuration."""
        result = {}

        for key, value in state.items():
            if key.startswith("__"):
                continue
            if config.include_fields and key not in config.include_fields:
                continue
            if key in config.exclude_fields:
                continue
            result[key] = value

        return result

    def _apply_state(
        self,
        obj: Any,
        state: Dict[str, Any],
        strategy: PreservationStrategy,
        config: PreservationConfig,
    ) -> bool:
        """Apply state to an object."""

        if strategy == PreservationStrategy.NONE:
            return True

        if strategy == PreservationStrategy.PICKLE_PROTOCOL:
            if hasattr(obj, "__setstate__"):
                if "__state__" in state:
                    obj.__setstate__(state["__state__"])
                else:
                    obj.__setstate__(state)
                return True

        if strategy == PreservationStrategy.CUSTOM:
            if hasattr(obj, "__restore__"):
                obj.__restore__(state)
                return True

        # Default: apply via mirror
        return self._apply_via_mirror(obj, state, config)

    def _apply_via_mirror(
        self,
        obj: Any,
        state: Dict[str, Any],
        config: PreservationConfig,
    ) -> bool:
        """Apply state using Foundation's Mirror."""
        m = mirror(obj)
        current_fields = set(m.fields.keys())
        success = True

        for name, value in state.items():
            if name.startswith("__"):
                continue
            if name not in current_fields:
                continue
            if name in config.exclude_fields:
                continue

            try:
                # Type validation
                if config.validate_types:
                    field_info = m.fields.get(name)
                    if field_info and field_info.type:
                        if not self._validate_type(value, field_info.type):
                            continue

                m.set(name, value)
            except (AttributeError, TypeError, ValueError):
                success = False

        return success

    def _validate_type(self, value: Any, expected_type: type) -> bool:
        """Validate that a value matches the expected type."""
        if value is None:
            return True  # Allow None for any type

        if expected_type in (Any, object):
            return True

        return isinstance(value, expected_type)

    def _cleanup_snapshots(self) -> None:
        """Remove old and excess snapshots."""
        # Remove stale snapshots
        stale_ids = [
            obj_id for obj_id, snapshot in self._snapshots.items()
            if snapshot.is_stale(self._snapshot_ttl)
        ]
        for obj_id in stale_ids:
            del self._snapshots[obj_id]
            self._ref_map.pop(obj_id, None)

        # Remove excess snapshots (oldest first)
        if len(self._snapshots) > self._max_snapshots:
            sorted_items = sorted(
                self._snapshots.items(),
                key=lambda x: x[1].timestamp,
            )
            to_remove = len(self._snapshots) - self._max_snapshots
            for obj_id, _ in sorted_items[:to_remove]:
                del self._snapshots[obj_id]
                self._ref_map.pop(obj_id, None)


__all__ = [
    "PreservationStrategy",
    "StateSnapshot",
    "PreservationConfig",
    "StatePreserver",
]
