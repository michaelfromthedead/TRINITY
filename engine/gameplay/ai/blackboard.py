"""
Blackboard System - Key-value storage with observers.

The blackboard is a shared memory space for AI components to communicate.
It supports hierarchical keys, namespaces, observers for change notifications,
and scoped access patterns.
"""

from __future__ import annotations

import time
import weakref
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

from .constants import (
    BLACKBOARD_DEFAULT_NAMESPACE,
    BLACKBOARD_KEY_SEPARATOR,
    BLACKBOARD_MAX_OBSERVERS,
)

T = TypeVar("T")
ObserverCallback = Callable[["BlackboardKey", Any, Any], None]


@dataclass
class BlackboardEntry:
    """A single entry in the blackboard."""
    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # Time-to-live in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if this entry has expired."""
        if self.ttl is None:
            return False
        current_time = current_time or time.time()
        return (current_time - self.timestamp) > self.ttl


@dataclass
class BlackboardKey:
    """A key in the blackboard with namespace support."""
    name: str
    namespace: str = BLACKBOARD_DEFAULT_NAMESPACE

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Key name cannot be empty")

    @property
    def full_key(self) -> str:
        """Get the full key including namespace."""
        return f"{self.namespace}{BLACKBOARD_KEY_SEPARATOR}{self.name}"

    @classmethod
    def from_string(cls, key_str: str) -> "BlackboardKey":
        """Create a key from a string representation."""
        parts = key_str.split(BLACKBOARD_KEY_SEPARATOR, 1)
        if len(parts) == 2:
            return cls(name=parts[1], namespace=parts[0])
        return cls(name=key_str)

    def __hash__(self) -> int:
        return hash(self.full_key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BlackboardKey):
            return self.full_key == other.full_key
        return False

    def __str__(self) -> str:
        return self.full_key


class Observer:
    """An observer that watches for blackboard changes."""

    def __init__(
        self,
        callback: ObserverCallback,
        key_pattern: Optional[str] = None,
        namespace: Optional[str] = None,
        once: bool = False,
    ) -> None:
        self.callback = callback
        self.key_pattern = key_pattern
        self.namespace = namespace
        self.once = once
        self._triggered = False

    def matches(self, key: BlackboardKey) -> bool:
        """Check if this observer matches the given key."""
        if self.once and self._triggered:
            return False

        if self.namespace and key.namespace != self.namespace:
            return False

        if self.key_pattern:
            # Simple glob matching
            if self.key_pattern.endswith("*"):
                prefix = self.key_pattern[:-1]
                if not key.name.startswith(prefix):
                    return False
            elif self.key_pattern != key.name:
                return False

        return True

    def notify(self, key: BlackboardKey, old_value: Any, new_value: Any) -> None:
        """Notify this observer of a change."""
        if self.matches(key):
            self.callback(key, old_value, new_value)
            if self.once:
                self._triggered = True


# =============================================================================
# Blackboard Registry (must be before Blackboard class for from_registry method)
# =============================================================================

# Valid scopes for blackboards
VALID_SCOPES = frozenset({"entity", "shared", "team", "group", "zone", "session"})

# Registry tag
TAG_BLACKBOARD = "blackboard"

# Registry storage
_blackboard_registry: Dict[str, Dict[str, Any]] = {}


class Blackboard:
    """
    A shared memory space for AI agents to communicate.

    Supports:
    - Key-value storage with any Python values
    - Hierarchical keys with namespace support
    - Observers for change notifications
    - TTL for automatic expiration
    - Scoped access through BlackboardScope
    """

    def __init__(
        self,
        name: str = "default",
        enable_event_logging: bool = True,
        entity_id: Optional[int] = None,
    ) -> None:
        self.name = name
        self.enable_event_logging = enable_event_logging
        self.entity_id = entity_id
        self._data: Dict[str, BlackboardEntry] = {}
        self._observers: List[Observer] = []
        self._parent: Optional["Blackboard"] = None

    def set(
        self,
        key: Union[str, BlackboardKey],
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set a value in the blackboard."""
        if isinstance(key, str):
            key = BlackboardKey.from_string(key)

        full_key = key.full_key
        old_entry = self._data.get(full_key)
        old_value = old_entry.value if old_entry else None

        self._data[full_key] = BlackboardEntry(
            value=value,
            timestamp=time.time(),
            ttl=ttl,
            metadata=metadata or {},
        )

        # Notify observers and log events
        if old_value != value:
            self._notify_observers(key, old_value, value)
            # Log event if enabled and entity_id is set
            if self.enable_event_logging and self.entity_id is not None:
                self._log_value_changed(full_key, old_value, value)

    def get(
        self,
        key: Union[str, BlackboardKey],
        default: Any = None,
        check_parent: bool = True,
    ) -> Any:
        """Get a value from the blackboard."""
        if isinstance(key, str):
            key = BlackboardKey.from_string(key)

        full_key = key.full_key
        entry = self._data.get(full_key)

        if entry is not None:
            if entry.is_expired():
                del self._data[full_key]
            else:
                return entry.value

        # Check parent blackboard if not found
        if check_parent and self._parent is not None:
            return self._parent.get(key, default, check_parent=True)

        return default

    def has(self, key: Union[str, BlackboardKey], check_parent: bool = True) -> bool:
        """Check if a key exists in the blackboard."""
        if isinstance(key, str):
            key = BlackboardKey.from_string(key)

        full_key = key.full_key
        entry = self._data.get(full_key)

        if entry is not None:
            if entry.is_expired():
                del self._data[full_key]
                entry = None
            else:
                return True

        if check_parent and self._parent is not None:
            return self._parent.has(key, check_parent=True)

        return False

    def remove(self, key: Union[str, BlackboardKey]) -> bool:
        """Remove a key from the blackboard."""
        if isinstance(key, str):
            key = BlackboardKey.from_string(key)

        full_key = key.full_key
        if full_key in self._data:
            old_value = self._data[full_key].value
            del self._data[full_key]
            self._notify_observers(key, old_value, None)
            return True
        return False

    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear all keys, optionally only in a specific namespace."""
        if namespace is None:
            self._data.clear()
        else:
            keys_to_remove = [
                k for k in self._data.keys()
                if k.startswith(f"{namespace}{BLACKBOARD_KEY_SEPARATOR}")
            ]
            for k in keys_to_remove:
                del self._data[k]

    def keys(self, namespace: Optional[str] = None) -> List[BlackboardKey]:
        """Get all keys, optionally filtered by namespace."""
        result = []
        for key_str in self._data.keys():
            key = BlackboardKey.from_string(key_str)
            if namespace is None or key.namespace == namespace:
                entry = self._data[key_str]
                if not entry.is_expired():
                    result.append(key)
        return result

    def namespaces(self) -> Set[str]:
        """Get all namespaces in the blackboard."""
        return {
            key_str.split(BLACKBOARD_KEY_SEPARATOR)[0]
            for key_str in self._data.keys()
        }

    def add_observer(
        self,
        callback: ObserverCallback,
        key_pattern: Optional[str] = None,
        namespace: Optional[str] = None,
        once: bool = False,
    ) -> Observer:
        """Add an observer for key changes."""
        if len(self._observers) >= BLACKBOARD_MAX_OBSERVERS:
            raise RuntimeError(
                f"Maximum number of observers ({BLACKBOARD_MAX_OBSERVERS}) exceeded"
            )

        observer = Observer(
            callback=callback,
            key_pattern=key_pattern,
            namespace=namespace,
            once=once,
        )
        self._observers.append(observer)
        return observer

    def remove_observer(self, observer: Observer) -> bool:
        """Remove an observer."""
        try:
            self._observers.remove(observer)
            return True
        except ValueError:
            return False

    def _notify_observers(
        self, key: BlackboardKey, old_value: Any, new_value: Any
    ) -> None:
        """Notify all matching observers of a change."""
        # Remove one-time observers that have been triggered
        self._observers = [
            obs for obs in self._observers
            if not (obs.once and obs._triggered)
        ]

        for observer in self._observers:
            if observer.matches(key):
                observer.notify(key, old_value, new_value)

    def _log_value_changed(self, key: str, old_value: Any, new_value: Any) -> None:
        """Log a value change event to the AI event logger."""
        try:
            from .ai_events import get_ai_event_logger
            logger = get_ai_event_logger()
            logger.log_blackboard_value_changed(
                entity_id=self.entity_id,
                key=key,
                old_value=old_value,
                new_value=new_value,
            )
        except ImportError:
            pass  # AI events module not available

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        current_time = time.time()
        expired_keys = [
            k for k, v in self._data.items()
            if v.is_expired(current_time)
        ]
        for k in expired_keys:
            del self._data[k]
        return len(expired_keys)

    def create_scope(self, namespace: str) -> "BlackboardScope":
        """Create a scoped view of the blackboard."""
        return BlackboardScope(self, namespace)

    def create_child(self, name: str) -> "Blackboard":
        """Create a child blackboard that inherits from this one."""
        child = Blackboard(name)
        child._parent = self
        return child

    def get_entry(self, key: Union[str, BlackboardKey]) -> Optional[BlackboardEntry]:
        """Get the full entry including metadata."""
        if isinstance(key, str):
            key = BlackboardKey.from_string(key)
        entry = self._data.get(key.full_key)
        if entry and entry.is_expired():
            del self._data[key.full_key]
            return None
        return entry

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self._data)

    def __contains__(self, key: Union[str, BlackboardKey]) -> bool:
        """Check if a key exists."""
        return self.has(key, check_parent=False)

    def __iter__(self) -> Iterator[BlackboardKey]:
        """Iterate over all keys."""
        return iter(self.keys())

    @classmethod
    def from_registry(cls, name: str, **kwargs: Any) -> "Blackboard":
        """
        Create a Blackboard instance from a registered blackboard class.

        Args:
            name: The registered name of the blackboard class.
            **kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            A new instance of the registered blackboard class.

        Raises:
            ValueError: If the blackboard name is not found in registry.
        """
        if name not in _blackboard_registry:
            raise ValueError(f"Blackboard '{name}' not found in registry")
        registered_cls = _blackboard_registry[name]["class"]
        return registered_cls(**kwargs)


class BlackboardScope:
    """
    A scoped view of a blackboard with a fixed namespace.

    This provides a convenient way to work with a subset of keys
    without having to specify the namespace every time.
    """

    def __init__(self, blackboard: Blackboard, namespace: str) -> None:
        self._blackboard = blackboard
        self._namespace = namespace

    @property
    def namespace(self) -> str:
        """Get the namespace of this scope."""
        return self._namespace

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set a value in this scope."""
        bb_key = BlackboardKey(name=key, namespace=self._namespace)
        self._blackboard.set(bb_key, value, ttl, metadata)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from this scope."""
        bb_key = BlackboardKey(name=key, namespace=self._namespace)
        return self._blackboard.get(bb_key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists in this scope."""
        bb_key = BlackboardKey(name=key, namespace=self._namespace)
        return self._blackboard.has(bb_key)

    def remove(self, key: str) -> bool:
        """Remove a key from this scope."""
        bb_key = BlackboardKey(name=key, namespace=self._namespace)
        return self._blackboard.remove(bb_key)

    def clear(self) -> None:
        """Clear all keys in this scope."""
        self._blackboard.clear(self._namespace)

    def keys(self) -> List[BlackboardKey]:
        """Get all keys in this scope."""
        return self._blackboard.keys(self._namespace)

    def add_observer(
        self,
        callback: ObserverCallback,
        key_pattern: Optional[str] = None,
        once: bool = False,
    ) -> Observer:
        """Add an observer for this scope."""
        return self._blackboard.add_observer(
            callback=callback,
            key_pattern=key_pattern,
            namespace=self._namespace,
            once=once,
        )

    def __len__(self) -> int:
        """Return the number of entries in this scope."""
        return len(self.keys())


class TypedBlackboardKey(Generic[T]):
    """
    A typed blackboard key for type-safe access.

    Usage:
        target_key = TypedBlackboardKey[Entity]("target", "combat")
        bb.set_typed(target_key, enemy)
        target = bb.get_typed(target_key)  # Returns Optional[Entity]
    """

    def __init__(self, name: str, namespace: str = BLACKBOARD_DEFAULT_NAMESPACE) -> None:
        self._key = BlackboardKey(name=name, namespace=namespace)

    @property
    def key(self) -> BlackboardKey:
        """Get the underlying blackboard key."""
        return self._key


class TypedBlackboard(Blackboard):
    """A blackboard with typed key support."""

    def set_typed(
        self,
        key: TypedBlackboardKey[T],
        value: T,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set a typed value in the blackboard."""
        self.set(key.key, value, ttl, metadata)

    def get_typed(
        self,
        key: TypedBlackboardKey[T],
        default: Optional[T] = None,
    ) -> Optional[T]:
        """Get a typed value from the blackboard."""
        return self.get(key.key, default)


def blackboard(
    scope_or_cls: Union[str, type] = "entity",
    name: Optional[str] = None,
    key_types: Optional[Union[Dict[str, type], List[str]]] = None,
    description: str = "",
    track_instances: bool = False,
    scope: Optional[str] = None,
) -> Union[Callable[[type], type], type]:
    """
    Decorator to register a blackboard class with the registry.

    Can be used with or without parentheses:
        @blackboard
        class MyBlackboard(Blackboard):
            pass

        @blackboard(scope="entity", name="combat_ai")
        class CombatBlackboard(Blackboard):
            pass

    Args:
        scope_or_cls: Either the scope string, or the class (when used without parens)
        name: Optional name for the blackboard
        key_types: Optional dict of key names to types, or list of key names
        description: Human-readable description
        track_instances: If True, track all instances via WeakSet
        scope: Alternative way to pass scope (for backwards compat)

    Raises:
        ValueError: If scope is not one of the valid scopes.
    """
    # Handle @blackboard without parentheses
    if isinstance(scope_or_cls, type):
        # Called as @blackboard directly on a class
        cls = scope_or_cls
        actual_scope = "entity"

        original_init = getattr(cls, "__init__", lambda self: None)

        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            self._blackboard = kwargs.pop("blackboard", None) or Blackboard(
                name=name or cls.__name__
            )
            if hasattr(original_init, "__call__") and original_init is not object.__init__:
                original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        cls._blackboard_registered = True
        cls._blackboard_decorated = True
        cls._blackboard_scope = actual_scope
        cls._blackboard_name = cls.__name__
        cls._blackboard_key_types = frozenset()
        cls._blackboard_description = ""

        # Register in the blackboard registry
        bb_name = cls.__name__
        _blackboard_registry[bb_name] = {
            "class": cls,
            "bb_name": bb_name,
            "scope": actual_scope,
            "key_types": frozenset(),
            "description": "",
        }

        # Also register with Foundation Registry if available
        try:
            from foundation import registry
            registry_name = f"bb.{bb_name}"
            try:
                registry.register(cls, name=registry_name, track_instances=False)
                registry.add_tag(cls, TAG_BLACKBOARD)
                registry.set_metadata(cls, "scope", actual_scope)
                registry.set_metadata(cls, "bb_name", bb_name)
                registry.set_metadata(cls, "key_types", frozenset())
                registry.set_metadata(cls, "description", "")
            except ValueError:
                pass  # Already registered
        except ImportError:
            pass

        return cls

    # Called with arguments: @blackboard(...) or @blackboard()
    actual_scope = scope if scope is not None else scope_or_cls if isinstance(scope_or_cls, str) else "entity"

    # Validate scope
    if actual_scope not in VALID_SCOPES:
        raise ValueError(
            f"Invalid blackboard scope '{actual_scope}'. Must be one of: {VALID_SCOPES}"
        )

    # Handle key_types - can be a list of strings or a dict
    if isinstance(key_types, list):
        resolved_key_types = frozenset(key_types)
    elif key_types is not None:
        resolved_key_types = frozenset(key_types.keys())
    else:
        resolved_key_types = frozenset()

    def decorator(cls: type) -> type:
        original_init = getattr(cls, "__init__", lambda self: None)

        # Use explicit name if provided (even empty string), otherwise use class name
        bb_name = name if name is not None else cls.__name__

        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            self._blackboard = kwargs.pop("blackboard", None) or Blackboard(
                name=bb_name or cls.__name__
            )
            if hasattr(original_init, "__call__") and original_init is not object.__init__:
                original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        cls._blackboard_registered = True
        cls._blackboard_decorated = True
        cls._blackboard_scope = actual_scope
        cls._blackboard_name = bb_name
        cls._blackboard_key_types = resolved_key_types
        cls._blackboard_description = description

        # Register in the blackboard registry
        _blackboard_registry[bb_name] = {
            "class": cls,
            "bb_name": bb_name,
            "scope": actual_scope,
            "key_types": resolved_key_types,
            "description": description,
        }

        # Also register with Foundation Registry if available
        try:
            from foundation import registry
            registry_name = f"bb.{bb_name}"
            try:
                registry.register(cls, name=registry_name, track_instances=track_instances)
                registry.add_tag(cls, TAG_BLACKBOARD)
                registry.set_metadata(cls, "scope", actual_scope)
                registry.set_metadata(cls, "bb_name", bb_name)
                registry.set_metadata(cls, "key_types", resolved_key_types)
                registry.set_metadata(cls, "description", description or "")
            except ValueError:
                pass  # Already registered
        except ImportError:
            pass

        return cls

    return decorator


# =============================================================================
# Blackboard Registry Helper Functions
# =============================================================================


def get_all_blackboards() -> Dict[str, type]:
    """Get all registered blackboard classes."""
    return [info["class"] for info in _blackboard_registry.values()]


def get_blackboards_by_scope(scope: str) -> List[type]:
    """Get all blackboard classes with the specified scope."""
    return [
        info["class"]
        for info in _blackboard_registry.values()
        if info["scope"] == scope
    ]


def get_blackboard_metadata(name: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a registered blackboard."""
    return _blackboard_registry.get(name)


def create_blackboard_from_registry(name: str, **kwargs: Any) -> Blackboard:
    """Create a blackboard instance from the registry."""
    if name not in _blackboard_registry:
        raise KeyError(f"Blackboard '{name}' not found in registry")
    cls = _blackboard_registry[name]["class"]
    return cls(**kwargs)


def clear_blackboard_registry() -> None:
    """Clear the blackboard registry (mainly for testing)."""
    _blackboard_registry.clear()


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "Blackboard",
    "BlackboardEntry",
    "BlackboardKey",
    "BlackboardScope",
    "Observer",
    "ObserverCallback",
    "TAG_BLACKBOARD",
    "TypedBlackboard",
    "TypedBlackboardKey",
    "VALID_SCOPES",
    "blackboard",
    "clear_blackboard_registry",
    "create_blackboard_from_registry",
    "get_all_blackboards",
    "get_blackboard_metadata",
    "get_blackboards_by_scope",
]
