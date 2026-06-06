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

    def query(
        self,
        tag: Optional[str] = None,
        **metadata_filters: Any,
    ) -> list[type]:
        """
        Query registered types by tag and/or metadata filters.

        Args:
            tag: If provided, filter types that have this tag.
            **metadata_filters: Key-value pairs to match against type metadata.

        Returns:
            List of types matching all specified criteria.

        Examples:
            >>> registry.query(tag="bt_node")  # All BT nodes
            >>> registry.query(tag="bt_node", node_type="action")  # Action BT nodes
            >>> registry.query(tag="goap_action", effect="has_weapon")  # GOAP with effect
        """
        with self._lock:
            result = []
            for cls in self._names:
                meta = self._metadata.get(cls, {})

                # Check tag filter
                if tag is not None:
                    tags = meta.get("_tags")
                    if not tags or tag not in tags:
                        continue

                # Check metadata filters
                match = True
                for key, expected in metadata_filters.items():
                    actual = meta.get(key)
                    # Handle set membership for list-based filters (e.g., effects, preconditions)
                    if isinstance(expected, str) and isinstance(actual, (list, set, frozenset)):
                        if expected not in actual:
                            match = False
                            break
                    elif actual != expected:
                        match = False
                        break

                if match:
                    result.append(cls)

            return result

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

    def add_tag(self, cls: type, tag: str) -> None:
        """Add a tag to a registered type for query-based discovery."""
        with self._lock:
            if cls not in self._metadata:
                raise ValueError(f"Type {cls.__name__} is not registered")
            tags = self._metadata[cls].setdefault("_tags", set())
            tags.add(tag)

    def remove_tag(self, cls: type, tag: str) -> bool:
        """Remove a tag from a registered type. Returns True if tag was present."""
        with self._lock:
            if cls not in self._metadata:
                return False
            tags = self._metadata[cls].get("_tags")
            if tags and tag in tags:
                tags.remove(tag)
                return True
            return False

    def has_tag(self, cls: type, tag: str) -> bool:
        """Check if a registered type has the given tag."""
        with self._lock:
            if cls not in self._metadata:
                return False
            tags = self._metadata[cls].get("_tags")
            return tags is not None and tag in tags

    def get_tags(self, cls: type) -> set[str]:
        """Get all tags for a registered type."""
        with self._lock:
            if cls not in self._metadata:
                return set()
            tags = self._metadata[cls].get("_tags")
            return set(tags) if tags else set()

    def get_metadata(self, cls: type, key: str) -> Any:
        """Get metadata value for a registered type."""
        with self._lock:
            meta = self._metadata.get(cls)
            return meta.get(key) if meta else None

    def get_all_metadata(self, cls: type) -> dict[str, Any]:
        """Get all metadata for a registered type."""
        with self._lock:
            return dict(self._metadata.get(cls, {}))

    # --- Tags ---

    def add_tag(self, cls: type, tag: str) -> None:
        """Add a tag to a registered type."""
        with self._lock:
            if cls not in self._metadata:
                raise ValueError(f"Type {cls.__name__} is not registered")
            tags = self._metadata[cls].get("_tags", set())
            if isinstance(tags, frozenset):
                tags = set(tags)
            tags.add(tag)
            self._metadata[cls]["_tags"] = tags

    def remove_tag(self, cls: type, tag: str) -> bool:
        """Remove a tag from a registered type. Returns True if tag was removed."""
        with self._lock:
            meta = self._metadata.get(cls)
            if not meta:
                return False
            tags = meta.get("_tags", set())
            if isinstance(tags, frozenset):
                tags = set(tags)
            if tag in tags:
                tags.discard(tag)
                self._metadata[cls]["_tags"] = tags
                return True
            return False

    def has_tag(self, cls: type, tag: str) -> bool:
        """Check if a registered type has a specific tag."""
        with self._lock:
            meta = self._metadata.get(cls)
            if not meta:
                return False
            tags = meta.get("_tags", set())
            return tag in tags

    def get_tags(self, cls: type) -> set[str]:
        """Get all tags for a registered type."""
        with self._lock:
            meta = self._metadata.get(cls)
            if not meta:
                return set()
            tags = meta.get("_tags", set())
            return set(tags)

    def query(self, tag: Optional[str] = None, **metadata_filters: Any) -> list[type]:
        """
        Query registered types by tag and/or metadata filters.

        Args:
            tag: If provided, only return types with this tag.
            **metadata_filters: Key-value pairs to filter by metadata.
                For set-valued metadata (like 'effects' or 'preconditions'),
                a single value will match if it's contained in the set.

        Returns:
            List of types matching the query.

        Example:
            >>> registry.query(tag="bt_node")  # All BT nodes
            >>> registry.query(tag="bt_node", node_type="action")  # Action nodes
            >>> registry.query(tag="goap_action", effects="target_damaged")
        """
        with self._lock:
            results = []
            for cls, meta in self._metadata.items():
                # Check tag filter
                if tag is not None:
                    tags = meta.get("_tags", set())
                    if tag not in tags:
                        continue

                # Check metadata filters
                matches = True
                for key, expected in metadata_filters.items():
                    actual = meta.get(key)
                    if actual is None:
                        matches = False
                        break
                    # Handle set membership for set-valued metadata
                    if isinstance(actual, (set, frozenset)):
                        if expected not in actual:
                            matches = False
                            break
                    elif actual != expected:
                        matches = False
                        break

                if matches:
                    results.append(cls)

            return results

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
