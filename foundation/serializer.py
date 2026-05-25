"""
Serializer - Convert objects to/from portable formats. Part of Core Foundation Layer 0.
Handles primitives, containers, objects, references, and circular references.
Supports schema hashing for versioned serialization and auto-migration.
"""
from __future__ import annotations
import pickle, json
from dataclasses import dataclass, is_dataclass, fields as dc_fields
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union, TYPE_CHECKING
from foundation.mirror import mirror, FieldInfo, schema_hash

if TYPE_CHECKING:
    from foundation.migrations import MigrationRegistry

T = TypeVar("T")
_type_registry: dict[str, type] = {}
_PRIMITIVES = (type(None), bool, int, float, str)

# Global migration registry for auto-migration during deserialization
_migration_registry: Optional["MigrationRegistry"] = None


def set_migration_registry(registry: "MigrationRegistry") -> None:
    """Set the migration registry for auto-migration during deserialization."""
    global _migration_registry
    _migration_registry = registry


def get_migration_registry() -> Optional["MigrationRegistry"]:
    """Get the current migration registry."""
    return _migration_registry


def register_type(cls: type, name: Optional[str] = None) -> type:
    """Register a type for deserialization lookup."""
    _type_registry[name or _get_type_name(cls)] = cls
    return cls


def _get_type_name(cls: type) -> str:
    module = getattr(cls, "__module__", "")
    return f"{module}.{cls.__name__}" if module and module != "builtins" else cls.__name__


def _get_type(name: str) -> Optional[type]:
    if name in _type_registry: return _type_registry[name]
    if name in {"list": list, "tuple": tuple, "set": set, "dict": dict}:
        return {"list": list, "tuple": tuple, "set": set, "dict": dict}[name]
    if "." in name:
        try:
            import importlib
            parts = name.rsplit(".", 1)
            cls = getattr(importlib.import_module(parts[0]), parts[1], None)
            if cls: _type_registry[name] = cls
            return cls
        except (ImportError, ModuleNotFoundError, AttributeError): pass
    return None


class SerializationContext:
    """Track seen objects for circular reference detection."""
    __slots__ = ("_seen", "_next_id", "include_schema")
    def __init__(self, include_schema: bool = False):
        self._seen, self._next_id = {}, 1
        self.include_schema = include_schema
    def has_seen(self, obj: Any) -> bool: return id(obj) in self._seen
    def get_id(self, obj: Any) -> int: return self._seen[id(obj)]
    def assign_id(self, obj: Any) -> int:
        self._seen[id(obj)], self._next_id = self._next_id, self._next_id + 1
        return self._next_id - 1


class DeserializationContext:
    """Track object reconstruction for two-pass deserialization."""
    __slots__ = ("_objects", "_pending")
    def __init__(self): self._objects, self._pending = {}, []
    def register(self, obj_id: int, obj: Any): self._objects[obj_id] = obj
    def get(self, obj_id: int) -> Optional[Any]: return self._objects.get(obj_id)
    def add_pending(self, container: Any, key: Union[str, int], ref_id: int):
        self._pending.append((container, key, ref_id))
    def resolve_refs(self):
        for container, key, ref_id in self._pending:
            obj = self._objects.get(ref_id)
            if obj is not None:
                if isinstance(container, (dict, list)): container[key] = obj
                elif hasattr(container, "__dict__"): setattr(container, key, obj)


class _PendingRef:
    __slots__ = ("ref_id",)
    def __init__(self, ref_id: int): self.ref_id = ref_id


def _serialize_value(value: Any, ctx: SerializationContext) -> Any:
    if isinstance(value, _PRIMITIVES): return value
    if ctx.has_seen(value): return {"__ref__": ctx.get_id(value)}
    if isinstance(value, (list, tuple)):
        result = [_serialize_value(item, ctx) for item in value]
        return {"__type__": "tuple", "items": result} if isinstance(value, tuple) else result
    if isinstance(value, set): return {"__type__": "set", "items": [_serialize_value(i, ctx) for i in value]}
    if isinstance(value, dict): return {str(k): _serialize_value(v, ctx) for k, v in value.items()}
    return _serialize_object(value, ctx)


def _serialize_object(obj: Any, ctx: SerializationContext) -> dict[str, Any]:
    if hasattr(obj, "__before_serialize__"): obj.__before_serialize__()
    result = {"__type__": _get_type_name(type(obj)), "__id__": ctx.assign_id(obj)}
    if ctx.include_schema:
        result["__schema__"] = schema_hash(type(obj))
    m = mirror(obj)
    for name, fi in m.fields.items():
        if fi.metadata.get("transient"): continue
        try: result[name] = _serialize_value(m.get(name), ctx)
        except AttributeError: pass
    return result


def _deserialize_value(data: Any, ctx: DeserializationContext) -> Any:
    if isinstance(data, _PRIMITIVES): return data
    if isinstance(data, list): return [_deserialize_value(item, ctx) for item in data]
    if isinstance(data, dict):
        if "__ref__" in data:
            existing = ctx.get(data["__ref__"])
            return existing if existing else _PendingRef(data["__ref__"])
        if "__value__" in data: return data["__value__"]
        if "__type__" in data:
            t = data["__type__"]
            if t == "tuple": return tuple(_deserialize_value(i, ctx) for i in data["items"])
            if t == "set": return set(_deserialize_value(i, ctx) for i in data["items"])
            return _deserialize_object(data, ctx)
        return {k: _deserialize_value(v, ctx) for k, v in data.items()}
    return data


class SchemaMismatchError(Exception):
    """Raised when serialized schema doesn't match current class schema."""
    def __init__(self, type_name: str, stored_hash: str, current_hash: str):
        self.type_name = type_name
        self.stored_hash = stored_hash
        self.current_hash = current_hash
        super().__init__(
            f"Schema mismatch for {type_name}: stored={stored_hash}, current={current_hash}"
        )


def _deserialize_object(data: dict[str, Any], ctx: DeserializationContext) -> Any:
    cls = _get_type(data["__type__"])
    if not cls: raise TypeError(f"Unknown type: {data['__type__']}. Register with register_type().")

    # Schema verification and auto-migration
    stored_schema = data.get("__schema__")
    if stored_schema:
        current_schema = schema_hash(cls)
        if stored_schema != current_schema:
            # Try auto-migration if registry is available
            if _migration_registry and _migration_registry.has_path(stored_schema, current_schema):
                data = _migration_registry.migrate(data, stored_schema, current_schema)
            else:
                raise SchemaMismatchError(data["__type__"], stored_schema, current_schema)

    obj = object.__new__(cls)
    if data.get("__id__"): ctx.register(data["__id__"], obj)
    for key, value in data.items():
        if key in ("__type__", "__id__", "__schema__"): continue
        d = _deserialize_value(value, ctx)
        if isinstance(d, _PendingRef): ctx.add_pending(obj, key, d.ref_id)
        elif hasattr(obj, "__dict__"): obj.__dict__[key] = d
        else: setattr(obj, key, d)
    return obj


def _resolve_pending(obj: Any, ctx: DeserializationContext) -> Any:
    if isinstance(obj, _PendingRef): return ctx.get(obj.ref_id)
    if isinstance(obj, list):
        for i, item in enumerate(obj): obj[i] = _resolve_pending(item, ctx)
    elif isinstance(obj, dict):
        for k in list(obj): obj[k] = _resolve_pending(obj[k], ctx)
    return obj


def _call_after_deserialize(obj: Any, visited: Optional[set] = None):
    if visited is None: visited = set()
    if id(obj) in visited: return
    visited.add(id(obj))
    if isinstance(obj, (list, tuple)):
        for item in obj: _call_after_deserialize(item, visited)
    elif isinstance(obj, dict):
        for v in obj.values(): _call_after_deserialize(v, visited)
    elif hasattr(obj, "__dict__"):
        if hasattr(obj, "__after_deserialize__"): obj.__after_deserialize__()
        for v in obj.__dict__.values(): _call_after_deserialize(v, visited)


def to_dict(obj: Any, include_schema_hash: bool = False) -> dict[str, Any]:
    """
    Serialize an object to a JSON-compatible dictionary.

    Args:
        obj: The object to serialize.
        include_schema_hash: If True, include __schema__ hash for each object.
            This enables schema verification and auto-migration on deserialization.

    Returns:
        A JSON-compatible dictionary representation.
    """
    if isinstance(obj, _PRIMITIVES): return {"__value__": obj}
    return _serialize_value(obj, SerializationContext(include_schema=include_schema_hash))


def from_dict(data: dict[str, Any]) -> Any:
    """Deserialize an object from a dictionary."""
    ctx = DeserializationContext()
    result = _resolve_pending(_deserialize_value(data, ctx), ctx)
    ctx.resolve_refs()
    _call_after_deserialize(result)
    return result


def to_bytes(obj: Any) -> bytes:
    """Serialize an object to compact binary format using pickle."""
    if hasattr(obj, "__before_serialize__"): obj.__before_serialize__()
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def from_bytes(data: bytes) -> Any:
    """Deserialize an object from binary format."""
    obj = pickle.loads(data)
    _call_after_deserialize(obj)
    return obj


def to_file(obj: Any, path: Union[str, Path], binary: bool = False, include_schema_hash: bool = False):
    """
    Serialize an object to a file (JSON or binary).

    Args:
        obj: The object to serialize.
        path: File path to write to.
        binary: If True, use binary pickle format. If False, use JSON.
        include_schema_hash: If True (and binary=False), include __schema__ hash
            for schema verification and auto-migration on deserialization.
    """
    path = Path(path)
    if binary: path.write_bytes(to_bytes(obj))
    else: path.write_text(json.dumps(to_dict(obj, include_schema_hash=include_schema_hash), indent=2, default=str))


def from_file(path: Union[str, Path], binary: bool = False) -> Any:
    """Deserialize an object from a file."""
    path = Path(path)
    return from_bytes(path.read_bytes()) if binary else from_dict(json.loads(path.read_text()))


def deep_copy(obj: T) -> T:
    """Create a deep copy of an object via serialize/deserialize."""
    return from_dict(to_dict(obj))


@dataclass
class Delta:
    """Represents differences between two objects."""
    added: dict[str, Any]
    removed: dict[str, Any]
    changed: dict[str, tuple[Any, Any]]
    def is_empty(self) -> bool: return not self.added and not self.removed and not self.changed


def diff(obj_a: Any, obj_b: Any) -> Delta:
    """Compare two objects and return their differences."""
    if type(obj_a) != type(obj_b):
        raise TypeError(f"Cannot diff objects of different types: {type(obj_a)} vs {type(obj_b)}")
    m_a, m_b = mirror(obj_a), mirror(obj_b)
    fields_a, fields_b = set(m_a.fields.keys()), set(m_b.fields.keys())
    added = {n: m_b.get(n) for n in fields_b - fields_a}
    removed = {n: m_a.get(n) for n in fields_a - fields_b}
    changed = {n: (m_a.get(n), m_b.get(n)) for n in fields_a & fields_b if m_a.get(n) != m_b.get(n)}
    return Delta(added=added, removed=removed, changed=changed)


def patch(obj: T, delta: Delta) -> T:
    """Apply a delta to an object, returning a modified copy."""
    result = deep_copy(obj)
    m = mirror(result)
    for name in delta.removed:
        if hasattr(result, "__dict__") and name in result.__dict__: del result.__dict__[name]
    for name, value in delta.added.items(): m.set(name, value)
    for name, (_, new_val) in delta.changed.items(): m.set(name, new_val)
    return result


__all__ = [
    "register_type", "to_dict", "from_dict", "to_bytes", "from_bytes",
    "to_file", "from_file", "deep_copy", "diff", "patch", "Delta",
    "SerializationContext", "DeserializationContext",
    "SchemaMismatchError", "set_migration_registry", "get_migration_registry",
]
