"""
Registry - Unified type registry for the Core Foundation. Part of Layer 1 (Structural).
Tracks all registered types, enables lookup by name, and provides instance tracking.
"""
from __future__ import annotations
import functools
import threading
from typing import Any, Callable, Iterator, Optional
from weakref import WeakSet

from foundation.mirror import mirror


class Registry:
    """
    Unified registry for all engine types.

    Complements per-metaclass registries by providing a single point of access
    for type lookup, queries, and optional instance tracking.

    Thread-safe: All modifications are protected by locks.
    """
    __slots__ = ("_lock", "_types", "_names", "_instances", "_metadata")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._types: dict[str, type] = {}  # name -> type
        self._names: dict[type, str] = {}  # type -> name
        self._instances: dict[type, WeakSet] = {}  # type -> weak set of instances
        self._metadata: dict[type, dict[str, Any]] = {}  # type -> metadata dict

    # --- Registration ---

    def register(self, cls: type, name: Optional[str] = None, track_instances: bool = False) -> None:
        """
        Register a type with the registry.

        Args:
            cls: The class to register.
            name: Optional custom name. Defaults to '{module}.{classname}'.
            track_instances: If True, track all instances via WeakSet.
        """
        if not isinstance(cls, type):
            raise TypeError(f"Expected a class, got {type(cls).__name__}")

        resolved_name = name or f"{cls.__module__}.{cls.__name__}"

        with self._lock:
            if cls in self._names:
                return  # Already registered

            if resolved_name in self._types:
                raise ValueError(f"Name '{resolved_name}' already registered to {self._types[resolved_name]}")

            self._types[resolved_name] = cls
            self._names[cls] = resolved_name
            self._metadata[cls] = {}

            if track_instances:
                self._instances[cls] = WeakSet()
                self._wrap_init(cls)

    def unregister(self, cls: type) -> None:
        """Remove a type from the registry."""
        with self._lock:
            name = self._names.pop(cls, None)
            if name:
                self._types.pop(name, None)
            self._instances.pop(cls, None)
            self._metadata.pop(cls, None)

    def is_registered(self, cls: type) -> bool:
        """Check if a type is registered."""
        with self._lock:
            return cls in self._names

    # --- Type Lookup ---

    def get(self, name: str) -> Optional[type]:
        """Get a type by its registered name."""
        with self._lock:
            return self._types.get(name)

    def get_name(self, cls: type) -> Optional[str]:
        """Get the registered name for a type."""
        with self._lock:
            return self._names.get(cls)

    def all_types(self) -> list[type]:
        """Return all registered types."""
        with self._lock:
            return list(self._names.keys())

    # --- Type Queries ---

    def subclasses(self, base: type) -> list[type]:
        """Return all registered types that are subclasses of base (excluding base)."""
        with self._lock:
            return [cls for cls in self._names if cls is not base and issubclass(cls, base)]

    def types_with_decorator(self, decorator_name: str) -> list[type]:
        """Return all types that have the named decorator in _applied_decorators."""
        with self._lock:
            result = []
            for cls in self._names:
                applied = getattr(cls, "_applied_decorators", None)
                if applied and decorator_name in applied:
                    result.append(cls)
            return result

    def types_where(self, predicate: Callable[[type], bool]) -> list[type]:
        """Return all types matching the predicate."""
        with self._lock:
            return [cls for cls in self._names if predicate(cls)]

    # --- Instance Tracking ---

    def _wrap_init(self, cls: type) -> None:
        """Wrap __init__ to track instances. Called under lock."""
        original_init = cls.__init__
        weak_set = self._instances[cls]

        @functools.wraps(original_init)
        def tracking_init(self_obj: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self_obj, *args, **kwargs)
            weak_set.add(self_obj)

        cls.__init__ = tracking_init  # type: ignore[method-assign]

    def instances(self, cls: type) -> Iterator[object]:
        """Iterate over all live instances of a tracked type."""
        with self._lock:
            weak_set = self._instances.get(cls)
            if weak_set is None:
                return iter([])
            # Copy to avoid modification during iteration
            return iter(list(weak_set))

    def instance_count(self, cls: type) -> int:
        """Return the count of live instances for a tracked type."""
        with self._lock:
            weak_set = self._instances.get(cls)
            return len(weak_set) if weak_set else 0

    # --- Metadata ---

    def set_metadata(self, cls: type, key: str, value: Any) -> None:
        """Set metadata for a registered type."""
        with self._lock:
            if cls not in self._metadata:
                raise ValueError(f"Type {cls.__name__} is not registered")
            self._metadata[cls][key] = value

    def get_metadata(self, cls: type, key: str) -> Any:
        """Get metadata value for a registered type."""
        with self._lock:
            meta = self._metadata.get(cls)
            return meta.get(key) if meta else None

    def get_all_metadata(self, cls: type) -> dict[str, Any]:
        """Get all metadata for a registered type."""
        with self._lock:
            return dict(self._metadata.get(cls, {}))

    # --- Utilities ---

    def clear(self) -> None:
        """Clear all registrations. Useful for testing."""
        with self._lock:
            self._types.clear()
            self._names.clear()
            self._instances.clear()
            self._metadata.clear()

    def describe(self, cls: type) -> str:
        """Return a human-readable description of a registered type."""
        with self._lock:
            name = self._names.get(cls)
            if not name:
                return f"<unregistered {cls.__name__}>"

            m = mirror(cls)
            meta = self._metadata.get(cls, {})
            tracked = cls in self._instances
            count = len(self._instances[cls]) if tracked else 0

            lines = [f"Registry: {name}"]
            if tracked:
                lines.append(f"  instances: {count}")
            if meta:
                lines.append(f"  metadata: {meta}")
            lines.append(m.describe())
            return "\n".join(lines)


# Module-level singleton
registry = Registry()

__all__ = ["Registry", "registry"]
