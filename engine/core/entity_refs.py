"""T-CC-2.6: Entity reference handling with resolution on deserialize.

Provides EntityRef[T] for referencing serializable entities, with cycle detection,
forward reference resolution, and integration with SerializationContext.
"""
from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from .serialization import (
    Serializable,
    SerializationContext,
    SerializationError,
    _serialize_value,
    _deserialize_value,
)


T = TypeVar("T")
S = TypeVar("S", bound="Referenceable")


class ReferenceError(SerializationError):
    """Error related to entity reference handling."""

    def __init__(
        self,
        message: str,
        ref_id: Optional[str] = None,
        path: str = "",
        is_cycle: bool = False,
        is_missing: bool = False,
        is_forward: bool = False,
    ):
        super().__init__(message, path)
        self.ref_id = ref_id
        self.is_cycle = is_cycle
        self.is_missing = is_missing
        self.is_forward = is_forward

    def __str__(self) -> str:
        prefix = ""
        if self.is_cycle:
            prefix = "[CYCLE] "
        elif self.is_missing:
            prefix = "[MISSING] "
        elif self.is_forward:
            prefix = "[FORWARD] "
        base = super().__str__()
        return f"{prefix}{base}"


class RefState(Enum):
    """State of an entity reference."""
    UNRESOLVED = auto()  # Reference not yet resolved
    RESOLVED = auto()    # Reference resolved to live object
    DEFERRED = auto()    # Forward reference, will resolve later
    BROKEN = auto()      # Reference target not found
    CYCLE = auto()       # Circular reference detected


@dataclass
class RefInfo:
    """Metadata about a reference."""
    ref_id: str
    target_type: Optional[Type] = None
    state: RefState = RefState.UNRESOLVED
    cycle_path: Optional[List[str]] = None


class Referenceable:
    """Mixin for objects that can be referenced by EntityRef.

    Any class that wants to be referenceable must:
    1. Inherit from Referenceable
    2. Implement get_ref_id() to return a unique string ID
    3. Optionally implement set_ref_id() for deserialization
    """

    def get_ref_id(self) -> str:
        """Return unique identifier for this entity.

        Override this to provide custom ID generation.
        Default uses id() which only works for in-memory references.
        """
        return f"ref_{id(self):x}"

    def set_ref_id(self, ref_id: str) -> None:
        """Set the reference ID during deserialization.

        Override if your class needs to track its own ID.
        """
        pass


class EntityRef(Generic[T]):
    """Type-safe reference to a serializable entity.

    EntityRef provides lazy resolution of references during deserialization,
    with support for forward references and cycle detection.

    Example:
        @serializable()
        @dataclass
        class Player(Referenceable):
            name: str
            team: EntityRef[Team]  # Reference to Team entity

        @serializable()
        @dataclass
        class Team(Referenceable):
            name: str
            captain: EntityRef[Player]  # Can reference back
    """

    __slots__ = ("_ref_id", "_target", "_target_type", "_state", "_resolver")

    def __init__(
        self,
        target: Optional[T] = None,
        ref_id: Optional[str] = None,
        target_type: Optional[Type[T]] = None,
    ):
        """Create a reference.

        Args:
            target: The target object (if already resolved)
            ref_id: The reference ID (for unresolved refs)
            target_type: Expected type of target (for validation)
        """
        self._target: Optional[T] = target
        self._target_type: Optional[Type[T]] = target_type
        self._resolver: Optional[ReferenceResolver] = None

        if target is not None:
            if isinstance(target, Referenceable):
                self._ref_id = target.get_ref_id()
            elif ref_id is not None:
                self._ref_id = ref_id
            else:
                self._ref_id = f"ref_{id(target):x}"
            self._state = RefState.RESOLVED
        elif ref_id is not None:
            self._ref_id = ref_id
            self._state = RefState.UNRESOLVED
        else:
            self._ref_id = ""
            self._state = RefState.BROKEN

    @classmethod
    def null(cls) -> "EntityRef[T]":
        """Create a null reference."""
        ref = cls.__new__(cls)
        ref._ref_id = ""
        ref._target = None
        ref._target_type = None
        ref._state = RefState.BROKEN
        ref._resolver = None
        return ref

    @classmethod
    def from_id(cls, ref_id: str, target_type: Optional[Type[T]] = None) -> "EntityRef[T]":
        """Create an unresolved reference from an ID."""
        ref = cls.__new__(cls)
        ref._ref_id = ref_id
        ref._target = None
        ref._target_type = target_type
        ref._state = RefState.UNRESOLVED
        ref._resolver = None
        return ref

    @property
    def ref_id(self) -> str:
        """Get the reference ID."""
        return self._ref_id

    @property
    def state(self) -> RefState:
        """Get the current state of this reference."""
        return self._state

    @property
    def is_resolved(self) -> bool:
        """Check if the reference is resolved."""
        return self._state == RefState.RESOLVED

    @property
    def is_valid(self) -> bool:
        """Check if the reference is valid (resolved or deferred)."""
        return self._state in (RefState.RESOLVED, RefState.DEFERRED)

    @property
    def is_null(self) -> bool:
        """Check if this is a null reference."""
        return self._ref_id == "" or self._state == RefState.BROKEN

    def get(self) -> Optional[T]:
        """Get the referenced object, or None if not resolved."""
        if self._state == RefState.RESOLVED:
            return self._target
        if self._state == RefState.DEFERRED and self._resolver is not None:
            # Try to resolve deferred reference
            self._try_resolve()
        return self._target

    def get_or_raise(self) -> T:
        """Get the referenced object, raising if not resolved."""
        result = self.get()
        if result is None:
            raise ReferenceError(
                f"Reference {self._ref_id} is not resolved",
                ref_id=self._ref_id,
                is_missing=True,
            )
        return result

    def _try_resolve(self) -> bool:
        """Attempt to resolve this reference using the attached resolver."""
        if self._resolver is None:
            return False

        resolved = self._resolver.resolve(self._ref_id)
        if resolved is not None:
            self._target = cast(T, resolved)
            self._state = RefState.RESOLVED
            return True
        return False

    def _set_resolver(self, resolver: "ReferenceResolver") -> None:
        """Attach a resolver for deferred resolution."""
        self._resolver = resolver

    def _mark_deferred(self) -> None:
        """Mark this reference as deferred (forward reference)."""
        self._state = RefState.DEFERRED

    def _mark_cycle(self) -> None:
        """Mark this reference as part of a cycle."""
        self._state = RefState.CYCLE

    def _resolve(self, target: T) -> None:
        """Resolve this reference to a target object."""
        self._target = target
        self._state = RefState.RESOLVED

    def serialize(self, ctx: Optional[SerializationContext] = None) -> Dict[str, Any]:
        """Serialize this reference to a compact ID form."""
        return {
            "__ref__": self._ref_id,
            "__ref_type__": self._target_type.__name__ if self._target_type else None,
        }

    @classmethod
    def deserialize(
        cls,
        data: Dict[str, Any],
        ctx: Optional[SerializationContext] = None,
        target_type: Optional[Type[T]] = None,
    ) -> "EntityRef[T]":
        """Deserialize a reference from data."""
        if not isinstance(data, dict) or "__ref__" not in data:
            raise ReferenceError("Invalid reference data format")

        ref_id = data["__ref__"]
        type_name = data.get("__ref_type__")

        ref = cls.from_id(ref_id, target_type)
        return ref

    def __repr__(self) -> str:
        type_name = self._target_type.__name__ if self._target_type else "?"
        if self.is_null:
            return f"EntityRef[{type_name}](null)"
        return f"EntityRef[{type_name}](id={self._ref_id}, state={self._state.name})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EntityRef):
            return NotImplemented
        return self._ref_id == other._ref_id

    def __hash__(self) -> int:
        return hash(self._ref_id)


class ReferenceRegistry:
    """Registry for tracking referenceable objects.

    Maintains a bidirectional mapping between ref_ids and objects.
    Thread-safe for concurrent access.
    """

    def __init__(self, use_weak_refs: bool = True):
        """Create a new registry.

        Args:
            use_weak_refs: If True, use weak references to allow GC
        """
        self._use_weak_refs = use_weak_refs
        self._by_id: Dict[str, Any] = {}
        self._by_object: Dict[int, str] = {}
        self._lock = threading.RLock()

    def register(self, obj: Any, ref_id: Optional[str] = None) -> str:
        """Register an object and return its reference ID.

        Args:
            obj: Object to register
            ref_id: Optional explicit ref_id (auto-generated if not provided)

        Returns:
            The reference ID for this object
        """
        with self._lock:
            obj_id = id(obj)

            # Already registered?
            if obj_id in self._by_object:
                return self._by_object[obj_id]

            # Generate ref_id if not provided
            if ref_id is None:
                if isinstance(obj, Referenceable):
                    ref_id = obj.get_ref_id()
                else:
                    ref_id = f"ref_{obj_id:x}"

            # Store mapping
            if self._use_weak_refs:
                try:
                    self._by_id[ref_id] = weakref.ref(obj, lambda _: self._cleanup(ref_id, obj_id))
                except TypeError:
                    # Object doesn't support weak refs
                    self._by_id[ref_id] = obj
            else:
                self._by_id[ref_id] = obj

            self._by_object[obj_id] = ref_id
            return ref_id

    def _cleanup(self, ref_id: str, obj_id: int) -> None:
        """Clean up weak reference when object is collected."""
        with self._lock:
            self._by_id.pop(ref_id, None)
            self._by_object.pop(obj_id, None)

    def unregister(self, obj: Any) -> bool:
        """Unregister an object.

        Returns:
            True if object was registered and removed
        """
        with self._lock:
            obj_id = id(obj)
            ref_id = self._by_object.pop(obj_id, None)
            if ref_id is not None:
                self._by_id.pop(ref_id, None)
                return True
            return False

    def unregister_by_id(self, ref_id: str) -> bool:
        """Unregister by reference ID.

        Returns:
            True if ref_id was registered and removed
        """
        with self._lock:
            entry = self._by_id.pop(ref_id, None)
            if entry is not None:
                obj = entry() if callable(entry) else entry
                if obj is not None:
                    self._by_object.pop(id(obj), None)
                return True
            return False

    def get(self, ref_id: str) -> Optional[Any]:
        """Get object by reference ID."""
        with self._lock:
            entry = self._by_id.get(ref_id)
            if entry is None:
                return None
            if callable(entry):  # weak reference
                return entry()
            return entry

    def get_ref_id(self, obj: Any) -> Optional[str]:
        """Get reference ID for an object."""
        with self._lock:
            return self._by_object.get(id(obj))

    def contains(self, ref_id: str) -> bool:
        """Check if a ref_id is registered."""
        with self._lock:
            return ref_id in self._by_id

    def contains_object(self, obj: Any) -> bool:
        """Check if an object is registered."""
        with self._lock:
            return id(obj) in self._by_object

    def clear(self) -> None:
        """Clear all registrations."""
        with self._lock:
            self._by_id.clear()
            self._by_object.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_id)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._by_id.keys()))

    def items(self) -> List[Tuple[str, Any]]:
        """Get all (ref_id, object) pairs."""
        with self._lock:
            result = []
            for ref_id, entry in self._by_id.items():
                obj = entry() if callable(entry) else entry
                if obj is not None:
                    result.append((ref_id, obj))
            return result


class ReferenceResolver:
    """Resolves entity references during deserialization.

    Handles:
    - Immediate resolution for already-deserialized objects
    - Deferred resolution for forward references
    - Cycle detection and prevention
    """

    def __init__(
        self,
        registry: Optional[ReferenceRegistry] = None,
        max_depth: int = 100,
        allow_broken: bool = False,
    ):
        """Create a resolver.

        Args:
            registry: Registry of live objects (created if not provided)
            max_depth: Maximum reference chain depth (cycle detection)
            allow_broken: If True, allow broken references without error
        """
        self._registry = registry or ReferenceRegistry()
        self._max_depth = max_depth
        self._allow_broken = allow_broken

        # Tracking for deserialization
        self._pending: Dict[str, List[EntityRef]] = {}  # refs waiting to be resolved
        self._in_progress: Set[str] = set()  # ref_ids currently being resolved
        self._resolved_order: List[str] = []  # order of resolution for debugging
        self._errors: List[ReferenceError] = []
        self._lock = threading.RLock()

    @property
    def registry(self) -> ReferenceRegistry:
        """Get the underlying registry."""
        return self._registry

    def register(self, obj: Any, ref_id: Optional[str] = None) -> str:
        """Register an object and resolve any pending references to it.

        Args:
            obj: Object to register
            ref_id: Optional explicit ref_id

        Returns:
            The reference ID
        """
        with self._lock:
            ref_id = self._registry.register(obj, ref_id)

            # Resolve any pending references
            pending = self._pending.pop(ref_id, [])
            for ref in pending:
                ref._resolve(obj)

            self._resolved_order.append(ref_id)
            return ref_id

    def resolve(self, ref_id: str) -> Optional[Any]:
        """Resolve a reference ID to an object.

        Args:
            ref_id: The reference ID to resolve

        Returns:
            The resolved object, or None if not found
        """
        with self._lock:
            return self._registry.get(ref_id)

    def resolve_ref(self, ref: EntityRef[T]) -> Optional[T]:
        """Resolve an EntityRef to its target.

        Args:
            ref: The reference to resolve

        Returns:
            The resolved object, or None if not resolvable yet
        """
        with self._lock:
            target = self._registry.get(ref.ref_id)
            if target is not None:
                ref._resolve(target)
                return cast(T, target)

            # Not found - defer resolution
            ref._set_resolver(self)
            ref._mark_deferred()

            if ref.ref_id not in self._pending:
                self._pending[ref.ref_id] = []
            self._pending[ref.ref_id].append(ref)

            return None

    def create_ref(
        self,
        target: T,
        target_type: Optional[Type[T]] = None,
    ) -> EntityRef[T]:
        """Create and register a reference to an object.

        Args:
            target: The target object
            target_type: Optional type hint

        Returns:
            A resolved EntityRef
        """
        ref_id = self.register(target)
        ref = EntityRef(target=target, ref_id=ref_id, target_type=target_type)
        return ref

    def finalize(self) -> List[ReferenceError]:
        """Finalize resolution and check for unresolved references.

        Call this after all objects have been deserialized.

        Returns:
            List of errors for broken references
        """
        with self._lock:
            errors = list(self._errors)

            for ref_id, refs in self._pending.items():
                if not self._allow_broken:
                    err = ReferenceError(
                        f"Unresolved reference: {ref_id}",
                        ref_id=ref_id,
                        is_missing=True,
                    )
                    errors.append(err)

                for ref in refs:
                    ref._state = RefState.BROKEN

            self._pending.clear()
            return errors

    def detect_cycle(
        self,
        start_id: str,
        visited: Optional[Set[str]] = None,
        path: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """Detect if following references from start_id creates a cycle.

        Args:
            start_id: Starting reference ID
            visited: Already visited IDs
            path: Current path for error reporting

        Returns:
            The cycle path if found, None otherwise
        """
        if visited is None:
            visited = set()
        if path is None:
            path = []

        if start_id in visited:
            cycle_start = path.index(start_id) if start_id in path else 0
            return path[cycle_start:] + [start_id]

        visited.add(start_id)
        path.append(start_id)

        obj = self._registry.get(start_id)
        if obj is not None:
            # Check all EntityRef fields
            for attr_name in dir(obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                    if isinstance(attr, EntityRef) and attr.ref_id:
                        result = self.detect_cycle(attr.ref_id, visited, path)
                        if result:
                            return result
                except Exception:
                    pass

        path.pop()
        visited.discard(start_id)
        return None

    def check_all_cycles(self) -> List[Tuple[str, List[str]]]:
        """Check all registered objects for cycles.

        Returns:
            List of (start_id, cycle_path) tuples for any cycles found
        """
        cycles = []
        checked: Set[str] = set()

        with self._lock:
            for ref_id in self._registry:
                if ref_id in checked:
                    continue

                cycle = self.detect_cycle(ref_id)
                if cycle:
                    cycles.append((ref_id, cycle))
                    checked.update(cycle)

        return cycles

    def clear(self) -> None:
        """Clear all state."""
        with self._lock:
            self._registry.clear()
            self._pending.clear()
            self._in_progress.clear()
            self._resolved_order.clear()
            self._errors.clear()

    def get_pending_count(self) -> int:
        """Get count of pending (unresolved) references."""
        with self._lock:
            return sum(len(refs) for refs in self._pending.values())

    def get_resolution_order(self) -> List[str]:
        """Get the order in which references were resolved."""
        with self._lock:
            return list(self._resolved_order)


class DeserializationContext:
    """Enhanced context for reference-aware deserialization.

    Extends SerializationContext with reference resolution capabilities.
    """

    def __init__(
        self,
        base_ctx: Optional[SerializationContext] = None,
        resolver: Optional[ReferenceResolver] = None,
    ):
        """Create a deserialization context.

        Args:
            base_ctx: Base serialization context
            resolver: Reference resolver (created if not provided)
        """
        self._base = base_ctx or SerializationContext()
        self._resolver = resolver or ReferenceResolver()
        self._type_registry: Dict[str, Type] = {}

    @property
    def base(self) -> SerializationContext:
        """Get the base serialization context."""
        return self._base

    @property
    def resolver(self) -> ReferenceResolver:
        """Get the reference resolver."""
        return self._resolver

    def register_type(self, type_cls: Type, name: Optional[str] = None) -> None:
        """Register a type for reference resolution.

        Args:
            type_cls: The type class
            name: Optional name (defaults to class name)
        """
        type_name = name or type_cls.__name__
        self._type_registry[type_name] = type_cls

    def get_type(self, name: str) -> Optional[Type]:
        """Get a registered type by name."""
        return self._type_registry.get(name)

    def deserialize_ref(
        self,
        data: Dict[str, Any],
        target_type: Optional[Type[T]] = None,
    ) -> EntityRef[T]:
        """Deserialize an EntityRef from data.

        Args:
            data: Serialized reference data
            target_type: Expected target type

        Returns:
            An EntityRef (may be deferred)
        """
        ref = EntityRef.deserialize(data, self._base, target_type)
        self._resolver.resolve_ref(ref)
        return ref

    def finalize(self) -> List[ReferenceError]:
        """Finalize deserialization and check for errors."""
        return self._resolver.finalize()


def serialize_with_refs(
    obj: Any,
    ctx: Optional[SerializationContext] = None,
    registry: Optional[ReferenceRegistry] = None,
) -> Tuple[Dict[str, Any], ReferenceRegistry]:
    """Serialize an object graph with references.

    Converts object references to compact ref_id format.

    Args:
        obj: Root object to serialize
        ctx: Serialization context
        registry: Registry for tracking refs

    Returns:
        Tuple of (serialized_data, registry)
    """
    if ctx is None:
        ctx = SerializationContext()
    if registry is None:
        registry = ReferenceRegistry()

    visited: Set[int] = set()
    ref_map: Dict[int, str] = {}

    def serialize_value(value: Any, depth: int = 0) -> Any:
        if value is None:
            return None

        if isinstance(value, (bool, int, float, str)):
            return value

        obj_id = id(value)

        # Check for cycles
        if obj_id in visited:
            # Return reference instead of value
            ref_id = ref_map.get(obj_id)
            if ref_id:
                return {"__ref__": ref_id}
            # Fallback
            return {"__ref__": f"ref_{obj_id:x}"}

        visited.add(obj_id)

        # Handle EntityRef specially
        if isinstance(value, EntityRef):
            return value.serialize(ctx)

        # Handle Referenceable objects
        if isinstance(value, Referenceable):
            ref_id = registry.register(value)
            ref_map[obj_id] = ref_id

            # Serialize the object contents
            if hasattr(value, "serialize"):
                data = value.serialize(ctx)
            else:
                data = {"__type__": type(value).__name__}
                for attr_name in dir(value):
                    if attr_name.startswith("_"):
                        continue
                    try:
                        attr = getattr(value, attr_name)
                        if callable(attr):
                            continue
                        data[attr_name] = serialize_value(attr, depth + 1)
                    except Exception:
                        pass

            data["__ref_id__"] = ref_id
            return data

        # Handle collections
        if isinstance(value, (list, tuple)):
            return [serialize_value(v, depth + 1) for v in value]

        if isinstance(value, set):
            return {"__set__": [serialize_value(v, depth + 1) for v in value]}

        if isinstance(value, dict):
            return {str(k): serialize_value(v, depth + 1) for k, v in value.items()}

        # Handle other serializable objects
        if hasattr(value, "serialize"):
            return value.serialize(ctx)

        # Fallback
        return _serialize_value(value, ctx)

    data = serialize_value(obj)
    return data, registry


def deserialize_with_refs(
    data: Dict[str, Any],
    root_type: Type[T],
    ctx: Optional[DeserializationContext] = None,
) -> Tuple[T, List[ReferenceError]]:
    """Deserialize an object graph with references.

    Args:
        data: Serialized data
        root_type: Expected type of root object
        ctx: Deserialization context

    Returns:
        Tuple of (deserialized_object, list_of_errors)
    """
    if ctx is None:
        ctx = DeserializationContext()

    # First pass: deserialize all objects
    result = _deserialize_object(data, root_type, ctx)

    # Finalize and check for unresolved refs
    errors = ctx.finalize()

    return result, errors


def _deserialize_object(
    data: Any,
    expected_type: Type[T],
    ctx: DeserializationContext,
) -> T:
    """Internal deserialization with reference handling."""
    if data is None:
        return cast(T, None)

    if isinstance(data, dict):
        # Check for reference
        if "__ref__" in data and "__ref_id__" not in data:
            # This is a reference to another object
            ref = ctx.deserialize_ref(data, expected_type)
            # Return the ref's target or the ref itself depending on expected type
            if get_origin(expected_type) is EntityRef or expected_type is EntityRef:
                return cast(T, ref)
            return cast(T, ref.get())

        # Check for inline object with ref_id
        if "__ref_id__" in data:
            ref_id = data["__ref_id__"]
            # Deserialize the object and register it
            clean_data = {k: v for k, v in data.items() if k != "__ref_id__"}

            if hasattr(expected_type, "deserialize"):
                obj = expected_type.deserialize(clean_data, ctx.base)
            else:
                obj = expected_type(**clean_data)

            if isinstance(obj, Referenceable):
                obj.set_ref_id(ref_id)
            ctx.resolver.register(obj, ref_id)
            return cast(T, obj)

    # Standard deserialization
    if hasattr(expected_type, "deserialize"):
        return expected_type.deserialize(data, ctx.base)

    return cast(T, _deserialize_value(data, expected_type, ctx.base))


# Global registry for convenience
_global_registry = ReferenceRegistry()


def get_global_registry() -> ReferenceRegistry:
    """Get the global reference registry."""
    return _global_registry


def register_global(obj: Any, ref_id: Optional[str] = None) -> str:
    """Register an object in the global registry."""
    return _global_registry.register(obj, ref_id)


def resolve_global(ref_id: str) -> Optional[Any]:
    """Resolve a reference ID in the global registry."""
    return _global_registry.get(ref_id)
