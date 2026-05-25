"""
Mirror - Uniform reflection for any object. Part of Core Foundation Layer 0.
Inspired by Pharo/Smalltalk's live image philosophy.
"""
from __future__ import annotations
import dataclasses, inspect, hashlib, json
from dataclasses import dataclass, field, fields as dc_fields, is_dataclass
from typing import Any, Optional, Union, get_type_hints, get_origin, get_args

# Import path utilities for get_path/set_path methods
from foundation.paths import get_path as _get_path, set_path as _set_path

STANDARD_METADATA_KEYS = frozenset({
    "transient", "serialize_as", "readonly", "hidden", "range", "choices",
    "required", "validator", "widget", "label", "description",
    "replicated", "authority", "interpolated", "version_added",
})

# Schema hash configuration
SCHEMA_HASH_LENGTH = 16  # Characters in the hex hash (64 bits of entropy)

@dataclass(frozen=True)
class FieldInfo:
    """Information about a field."""
    name: str
    type: Optional[type] = None
    has_default: bool = False
    default: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class MethodInfo:
    """Information about a method."""
    name: str
    signature: inspect.Signature
    is_property: bool = False

def _extract_metadata(ann: Any) -> dict[str, Any]:
    """Extract metadata from Annotated[T, ...] types."""
    if get_origin(ann) is None: return {}
    try:
        from typing import Annotated
        if get_origin(ann) is Annotated:
            return {k: v for arg in get_args(ann)[1:] if isinstance(arg, dict) for k, v in arg.items()}
    except ImportError: pass
    return {}

def _get_base_type(ann: Any) -> Optional[type]:
    """Extract base type from annotation."""
    origin = get_origin(ann)
    if origin is None: return ann if isinstance(ann, type) else None
    try:
        from typing import Annotated
        if origin is Annotated: return get_args(ann)[0] if get_args(ann) else None
    except ImportError: pass
    return origin

def _collect_fields(cls: type, obj: Any = None) -> dict[str, FieldInfo]:
    """Collect field info from class and optional instance."""
    result: dict[str, FieldInfo] = {}
    hints = {}
    try: hints = get_type_hints(cls, include_extras=True)
    except (TypeError, NameError, AttributeError): pass

    if is_dataclass(cls):
        for f in dc_fields(cls):
            ann = hints.get(f.name, f.type)
            has_def = f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
            default = f.default if f.default is not dataclasses.MISSING else None
            meta = _extract_metadata(ann); meta.update(f.metadata)
            result[f.name] = FieldInfo(f.name, _get_base_type(ann), has_def, default, meta)
        return result

    all_ann = {}
    for k in reversed(cls.__mro__):
        if k is not object: all_ann.update(getattr(k, "__annotations__", {}))

    for name, ann in all_ann.items():
        if name.startswith("_"): continue
        has_def, default = False, None
        for k in cls.__mro__:
            if name in getattr(k, "__dict__", {}):
                attr = k.__dict__[name]
                if not callable(attr) and not hasattr(attr, "__get__"): has_def, default = True, attr
                break
        result[name] = FieldInfo(name, _get_base_type(ann), has_def, default, _extract_metadata(ann))

    for k in cls.__mro__:
        if k is object: continue
        slots = getattr(k, "__slots__", ())
        for s in ((slots,) if isinstance(slots, str) else slots):
            if not s.startswith("_") and s not in result:
                ann = all_ann.get(s)
                result[s] = FieldInfo(s, _get_base_type(ann) if ann else None, False, None, _extract_metadata(ann) if ann else {})

    if obj and hasattr(obj, "__dict__"):
        for n, v in obj.__dict__.items():
            if not n.startswith("_") and n not in result:
                result[n] = FieldInfo(n, type(v), True, v, {})
    return result

def _collect_methods(cls: type) -> dict[str, MethodInfo]:
    """Collect method info from class."""
    result = {}
    for name in dir(cls):
        if name.startswith("_"): continue
        attr = getattr(cls, name, None)
        if attr is None: continue
        if isinstance(attr, property): result[name] = MethodInfo(name, inspect.Signature(), True)
        elif callable(attr):
            try: sig = inspect.signature(attr)
            except (ValueError, TypeError): sig = inspect.Signature()
            result[name] = MethodInfo(name, sig, False)
    return result

class ObjectMirror:
    """Mirror for object instances."""
    __slots__ = ("_obj", "_fields", "_methods")

    def __init__(self, obj: Any):
        if isinstance(obj, type): raise TypeError("Use ClassMirror for classes")
        self._obj, self._fields, self._methods = obj, None, None

    @property
    def type_name(self) -> str: return type(self._obj).__name__
    @property
    def type_class(self) -> type: return type(self._obj)
    @property
    def fields(self) -> dict[str, FieldInfo]:
        if self._fields is None: self._fields = _collect_fields(type(self._obj), self._obj)
        return self._fields
    @property
    def methods(self) -> dict[str, MethodInfo]:
        if self._methods is None: self._methods = _collect_methods(type(self._obj))
        return self._methods

    def get(self, name: str) -> Any: return getattr(self._obj, name)
    def set(self, name: str, value: Any): setattr(self._obj, name, value)
    def has(self, name: str) -> bool: return hasattr(self._obj, name)
    def to_dict(self) -> dict[str, Any]: return {n: self.get(n) for n in self.fields}

    def get_path(self, path: str) -> Any:
        """
        Get value at dotted path starting from the mirrored object.

        Args:
            path: Dotted path string with optional array indices.

        Returns:
            The value at the specified path.

        Examples:
            >>> m.get_path("inventory.items[0].damage")
            10
        """
        return _get_path(self._obj, path)

    def set_path(self, path: str, value: Any) -> None:
        """
        Set value at dotted path starting from the mirrored object.

        Args:
            path: Dotted path string with optional array indices.
            value: Value to set at the path.

        Examples:
            >>> m.set_path("inventory.items[0].damage", 20)
        """
        _set_path(self._obj, path, value)
    def describe(self) -> str:
        lines = [f"{self.type_name}:"]
        for n, f in self.fields.items():
            lines.append(f"  {n}: {f.type.__name__ if f.type else 'Any'} = {self.get(n)!r}")
        return "\n".join(lines)

class ClassMirror:
    """Mirror for class definitions."""
    __slots__ = ("_cls", "_fields", "_methods")

    def __init__(self, cls: type):
        if not isinstance(cls, type): raise TypeError("ClassMirror requires a class")
        self._cls, self._fields, self._methods = cls, None, None

    @property
    def type_name(self) -> str: return self._cls.__name__
    @property
    def type_class(self) -> type: return self._cls
    @property
    def fields(self) -> dict[str, FieldInfo]:
        if self._fields is None: self._fields = _collect_fields(self._cls)
        return self._fields
    @property
    def methods(self) -> dict[str, MethodInfo]:
        if self._methods is None: self._methods = _collect_methods(self._cls)
        return self._methods

    def has(self, name: str) -> bool: return name in self.fields or name in self.methods
    def describe(self) -> str:
        lines = [f"class {self.type_name}:"]
        for n, f in self.fields.items():
            t = f.type.__name__ if f.type else "Any"
            lines.append(f"  {n}: {t}{f' = {f.default!r}' if f.has_default else ''}")
        return "\n".join(lines)

def mirror(obj_or_cls: Any) -> Union[ObjectMirror, ClassMirror]:
    """Create a mirror for an object or class."""
    return ClassMirror(obj_or_cls) if isinstance(obj_or_cls, type) else ObjectMirror(obj_or_cls)


def schema_hash(cls: type) -> str:
    """
    Generate a stable hash of a class's schema for migration detection.

    The hash is based on:
    - Class name
    - Field names, types, defaults, and metadata

    Returns:
        16-character hexadecimal hash string.

    Use cases:
    - Hot reload detection (hash changed → migration needed)
    - Versioned save files
    - Network protocol compatibility
    """
    if not isinstance(cls, type):
        cls = type(cls)

    m = mirror(cls)

    def _serialize_metadata(metadata: dict) -> list:
        """Serialize metadata to a stable list of tuples."""
        result = []
        for k, v in sorted(metadata.items()):
            # Convert non-JSON-serializable values to repr strings
            try:
                json.dumps(v)
                result.append([k, v])
            except (TypeError, ValueError):
                result.append([k, repr(v)])
        return result

    def _type_name(t: Optional[type]) -> str:
        """Get stable type name."""
        if t is None:
            return "Any"
        return t.__name__ if hasattr(t, "__name__") else str(t)

    def _default_repr(info: FieldInfo) -> Optional[str]:
        """Get stable default representation."""
        if not info.has_default:
            return None
        # Use repr for stable string representation
        return repr(info.default)

    canonical = {
        "name": m.type_name,
        "fields": {
            name: {
                "type": _type_name(info.type),
                "default": _default_repr(info),
                "metadata": _serialize_metadata(info.metadata)
            }
            for name, info in sorted(m.fields.items())
        }
    }

    canonical_json = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()[:SCHEMA_HASH_LENGTH]


__all__ = ["FieldInfo", "MethodInfo", "ObjectMirror", "ClassMirror", "mirror", "schema_hash", "SCHEMA_HASH_LENGTH", "STANDARD_METADATA_KEYS"]
