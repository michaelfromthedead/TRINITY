"""T-CC-2.7: Partial serialization with SerializationContext.

Extends the serialization framework to support scoped/partial serialization
with configurable depth limits, reference handling, and field filtering.

Features:
- max_depth: Limit nested object depth
- include_refs: Control entity reference serialization
- field_filter: Dynamic field inclusion/exclusion
- root_only: Serialize only root object
- Snapshot vs Full serialization modes
- Graceful handling of missing fields during deserialization
"""
from __future__ import annotations

import copy
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)
from weakref import WeakValueDictionary

from .serialization import (
    SchemaInfo,
    SchemaVersion,
    Serializable,
    SerializationContext,
    SerializationError,
    SerializationFormat,
    serializable,
    _serialize_value,
    _deserialize_value,
)


T = TypeVar("T")


class SerializationMode(Enum):
    """Serialization mode controlling output detail level."""

    SNAPSHOT = auto()  # Minimal output - IDs only for refs, shallow
    STANDARD = auto()  # Normal serialization
    FULL = auto()  # Complete output - all refs expanded, deep


class DepthPolicy(Enum):
    """Policy for handling depth limit."""

    TRUNCATE = auto()  # Stop at depth limit, omit deeper fields
    REFERENCE = auto()  # Replace deep objects with reference IDs
    PLACEHOLDER = auto()  # Replace with placeholder marker
    ERROR = auto()  # Raise error when depth exceeded


class MissingFieldPolicy(Enum):
    """Policy for handling missing fields during deserialization."""

    DEFAULT = auto()  # Use field default or None
    ERROR = auto()  # Raise error
    SKIP = auto()  # Skip the field entirely
    SENTINEL = auto()  # Use a sentinel value (MISSING marker)


@dataclass(frozen=True)
class FieldDescriptor:
    """Descriptor for a field with serialization metadata."""

    name: str
    type_hint: Any
    default: Any = None
    has_default: bool = False
    sensitive: bool = False
    large: bool = False
    lazy: bool = False
    reference: bool = False
    max_depth: Optional[int] = None


@dataclass
class ScopeConfig:
    """Configuration for serialization scope.

    Controls what gets serialized and how deep serialization goes.
    """

    max_depth: Optional[int] = None
    include_refs: bool = True
    field_filter: Optional[Callable[[str, Any], bool]] = None
    root_only: bool = False
    mode: SerializationMode = SerializationMode.STANDARD
    depth_policy: DepthPolicy = DepthPolicy.TRUNCATE
    missing_field_policy: MissingFieldPolicy = MissingFieldPolicy.DEFAULT
    include_type_info: bool = True
    exclude_fields: Optional[Set[str]] = None
    include_fields: Optional[Set[str]] = None
    exclude_sensitive: bool = False
    exclude_large: bool = False
    max_collection_size: Optional[int] = None
    lazy_refs: bool = False

    def __post_init__(self):
        if self.exclude_fields is None:
            self.exclude_fields = set()
        if self.include_fields is None:
            self.include_fields = set()

    def should_include_field(self, field_name: str, value: Any) -> bool:
        """Check if a field should be included based on all filters."""
        # Explicit exclusion
        if self.exclude_fields and field_name in self.exclude_fields:
            return False

        # Explicit inclusion (if specified, only these fields included)
        if self.include_fields and field_name not in self.include_fields:
            return False

        # Custom filter function
        if self.field_filter is not None:
            if not self.field_filter(field_name, value):
                return False

        return True

    def copy_with(self, **kwargs) -> "ScopeConfig":
        """Create a copy with modified values."""
        config_dict = {
            "max_depth": self.max_depth,
            "include_refs": self.include_refs,
            "field_filter": self.field_filter,
            "root_only": self.root_only,
            "mode": self.mode,
            "depth_policy": self.depth_policy,
            "missing_field_policy": self.missing_field_policy,
            "include_type_info": self.include_type_info,
            "exclude_fields": self.exclude_fields.copy() if self.exclude_fields else None,
            "include_fields": self.include_fields.copy() if self.include_fields else None,
            "exclude_sensitive": self.exclude_sensitive,
            "exclude_large": self.exclude_large,
            "max_collection_size": self.max_collection_size,
            "lazy_refs": self.lazy_refs,
        }
        config_dict.update(kwargs)
        return ScopeConfig(**config_dict)

    @classmethod
    def snapshot(cls) -> "ScopeConfig":
        """Create a snapshot configuration (minimal output)."""
        return cls(
            max_depth=1,
            include_refs=False,
            root_only=False,
            mode=SerializationMode.SNAPSHOT,
            depth_policy=DepthPolicy.REFERENCE,
            include_type_info=False,
            exclude_sensitive=True,
            exclude_large=True,
        )

    @classmethod
    def full(cls) -> "ScopeConfig":
        """Create a full configuration (complete output)."""
        return cls(
            max_depth=None,
            include_refs=True,
            root_only=False,
            mode=SerializationMode.FULL,
            depth_policy=DepthPolicy.TRUNCATE,
            include_type_info=True,
            exclude_sensitive=False,
            exclude_large=False,
        )


@dataclass
class PartialSerializationContext:
    """Extended context for partial serialization operations."""

    scope: ScopeConfig = field(default_factory=ScopeConfig)
    format: SerializationFormat = SerializationFormat.JSON
    include_schema: bool = True

    _current_depth: int = field(default=0, init=False)
    _path: List[str] = field(default_factory=list, init=False)
    _references: Dict[int, Any] = field(default_factory=dict, init=False)
    _ref_ids: Dict[int, str] = field(default_factory=dict, init=False)
    _seen_objects: Set[int] = field(default_factory=set, init=False)
    _deferred_refs: List[Tuple[str, int]] = field(default_factory=list, init=False)
    _stats: Dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._stats = {
            "fields_serialized": 0,
            "fields_skipped": 0,
            "refs_replaced": 0,
            "depth_truncated": 0,
            "collections_truncated": 0,
        }

    @property
    def current_path(self) -> str:
        return ".".join(self._path)

    @property
    def current_depth(self) -> int:
        return self._current_depth

    @property
    def stats(self) -> Dict[str, int]:
        return self._stats.copy()

    def enter(self, name: str) -> "PartialSerializationContext":
        """Enter a nested field."""
        self._path.append(name)
        self._current_depth += 1
        return self

    def exit(self) -> None:
        """Exit a nested field."""
        if self._path:
            self._path.pop()
        self._current_depth -= 1

    def at_depth_limit(self) -> bool:
        """Check if at the depth limit."""
        if self.scope.max_depth is None:
            return False
        return self._current_depth >= self.scope.max_depth

    def should_serialize_field(self, field_name: str, value: Any = None) -> bool:
        """Check if a field should be serialized."""
        # Check scope config filters
        if not self.scope.should_include_field(field_name, value):
            self._stats["fields_skipped"] += 1
            return False

        # Check depth limit
        if self.scope.root_only and self._current_depth > 0:
            self._stats["depth_truncated"] += 1
            return False

        if self.at_depth_limit():
            if self.scope.depth_policy == DepthPolicy.ERROR:
                raise SerializationError(
                    f"Depth limit {self.scope.max_depth} exceeded at {self.current_path}",
                    path=self.current_path,
                )
            if self.scope.depth_policy == DepthPolicy.TRUNCATE:
                self._stats["depth_truncated"] += 1
                return False

        self._stats["fields_serialized"] += 1
        return True

    def register_object(self, obj: Any) -> Tuple[bool, Optional[str]]:
        """Register an object and check for cycles.

        Returns (is_new, ref_id) where is_new is False if object was seen before.
        """
        obj_id = id(obj)
        if obj_id in self._seen_objects:
            ref_id = self._ref_ids.get(obj_id)
            return False, ref_id

        self._seen_objects.add(obj_id)
        ref_id = f"ref_{len(self._ref_ids)}"
        self._ref_ids[obj_id] = ref_id
        self._references[obj_id] = obj
        return True, ref_id

    def get_reference_marker(self, obj: Any) -> Dict[str, Any]:
        """Get a reference marker for an object."""
        obj_id = id(obj)
        ref_id = self._ref_ids.get(obj_id, f"ref_{obj_id}")
        self._stats["refs_replaced"] += 1

        marker = {"__ref__": ref_id}

        # Include ID if object has one
        if hasattr(obj, "id"):
            marker["id"] = getattr(obj, "id")
        elif hasattr(obj, "entity_id"):
            marker["entity_id"] = getattr(obj, "entity_id")

        # Include type info if configured
        if self.scope.include_type_info:
            marker["__type__"] = type(obj).__name__

        return marker

    def to_serialization_context(self) -> SerializationContext:
        """Convert to standard SerializationContext for compatibility."""
        return SerializationContext(
            format=self.format,
            include_schema=self.include_schema,
            include_defaults=self.scope.mode == SerializationMode.FULL,
            max_depth=self.scope.max_depth,
            skip_fields=self.scope.exclude_fields or set(),
        )


@runtime_checkable
class HasEntityId(Protocol):
    """Protocol for objects with entity IDs."""

    @property
    def entity_id(self) -> Any:
        ...


@runtime_checkable
class HasId(Protocol):
    """Protocol for objects with generic IDs."""

    @property
    def id(self) -> Any:
        ...


class PartialSerializer:
    """Serializer that supports partial/scoped serialization.

    Features:
    - Configurable depth limits
    - Field filtering
    - Reference handling (full vs ID-only)
    - Snapshot and full modes
    - Cycle detection
    """

    def __init__(self, scope: Optional[ScopeConfig] = None):
        self.scope = scope or ScopeConfig()
        self._field_descriptors: Dict[Type, List[FieldDescriptor]] = {}

    def serialize(
        self,
        obj: Any,
        ctx: Optional[PartialSerializationContext] = None,
    ) -> Any:
        """Serialize an object with partial serialization support."""
        if ctx is None:
            ctx = PartialSerializationContext(scope=self.scope)

        return self._serialize_value(obj, ctx)

    def _serialize_value(self, value: Any, ctx: PartialSerializationContext) -> Any:
        """Serialize a single value with context awareness."""
        if value is None:
            return None

        # Primitives
        if isinstance(value, (bool, int, float, str)):
            return value

        # Enum
        if isinstance(value, Enum):
            if ctx.scope.include_type_info:
                return {"__enum__": type(value).__name__, "value": value.value}
            return value.value

        # Check for cycles and references for complex objects
        if isinstance(value, (list, tuple, set, dict)):
            pass  # Collections handled below
        elif hasattr(value, "__dict__") or is_dataclass(value):
            is_new, ref_id = ctx.register_object(value)
            if not is_new:
                # Already serialized - return reference
                return ctx.get_reference_marker(value)

            # Check if refs should be replaced with IDs only (only for nested objects)
            if not ctx.scope.include_refs and ctx.current_depth > 0:
                return ctx.get_reference_marker(value)

            # Check depth limit BEFORE entering nested serialization
            if ctx.at_depth_limit():
                if ctx.scope.depth_policy == DepthPolicy.REFERENCE:
                    return ctx.get_reference_marker(value)
                if ctx.scope.depth_policy == DepthPolicy.PLACEHOLDER:
                    return {"__truncated__": True, "__type__": type(value).__name__}
                if ctx.scope.depth_policy == DepthPolicy.ERROR:
                    raise SerializationError(
                        f"Depth limit exceeded at {ctx.current_path}",
                        path=ctx.current_path,
                    )
                # TRUNCATE - return minimal reference
                return ctx.get_reference_marker(value)

        # Collections - apply size limits
        if isinstance(value, (list, tuple)):
            items = list(value)
            if ctx.scope.max_collection_size is not None:
                if len(items) > ctx.scope.max_collection_size:
                    ctx._stats["collections_truncated"] += 1
                    items = items[: ctx.scope.max_collection_size]
            result = [self._serialize_value(v, ctx) for v in items]
            if isinstance(value, tuple):
                return result  # JSON doesn't distinguish
            return result

        if isinstance(value, set):
            items = list(value)
            if ctx.scope.max_collection_size is not None:
                if len(items) > ctx.scope.max_collection_size:
                    ctx._stats["collections_truncated"] += 1
                    items = items[: ctx.scope.max_collection_size]
            return {"__set__": [self._serialize_value(v, ctx) for v in items]}

        if isinstance(value, dict):
            items = list(value.items())
            if ctx.scope.max_collection_size is not None:
                if len(items) > ctx.scope.max_collection_size:
                    ctx._stats["collections_truncated"] += 1
                    items = items[: ctx.scope.max_collection_size]
            return {str(k): self._serialize_value(v, ctx) for k, v in items}

        # Dataclasses (including those with serialize method) - use our own serialization
        # to properly respect scope config
        if is_dataclass(value):
            return self._serialize_dataclass(value, ctx)

        # Objects with serialize method (non-dataclass)
        if hasattr(value, "serialize"):
            return self._serialize_object_with_method(value, ctx)

        # Fallback
        return str(value)

    def _serialize_object_with_method(
        self, obj: Any, ctx: PartialSerializationContext
    ) -> Dict[str, Any]:
        """Serialize object that has a serialize method."""
        # Check if at depth limit
        if ctx.at_depth_limit():
            if ctx.scope.depth_policy == DepthPolicy.REFERENCE:
                return ctx.get_reference_marker(obj)
            if ctx.scope.depth_policy == DepthPolicy.PLACEHOLDER:
                return {"__truncated__": True, "__type__": type(obj).__name__}
            if ctx.scope.depth_policy == DepthPolicy.ERROR:
                raise SerializationError(
                    f"Depth limit exceeded at {ctx.current_path}",
                    path=ctx.current_path,
                )
            # TRUNCATE - return minimal
            return ctx.get_reference_marker(obj)

        # Use standard serialize method but filter fields
        std_ctx = ctx.to_serialization_context()
        std_ctx._current_depth = ctx.current_depth

        try:
            data = obj.serialize(std_ctx)
        except Exception as e:
            # Fall back to manual serialization
            data = self._serialize_dataclass(obj, ctx) if is_dataclass(obj) else {"__error__": str(e)}

        # Post-process to apply additional filters
        if isinstance(data, dict):
            filtered = {}
            for key, value in data.items():
                if key.startswith("__"):  # Preserve metadata
                    filtered[key] = value
                elif ctx.should_serialize_field(key, value):
                    filtered[key] = value
            return filtered

        return data

    def _serialize_dataclass(
        self, obj: Any, ctx: PartialSerializationContext
    ) -> Dict[str, Any]:
        """Serialize a dataclass with partial serialization."""
        if ctx.at_depth_limit():
            if ctx.scope.depth_policy == DepthPolicy.REFERENCE:
                return ctx.get_reference_marker(obj)
            if ctx.scope.depth_policy == DepthPolicy.PLACEHOLDER:
                return {"__truncated__": True, "__type__": type(obj).__name__}
            if ctx.scope.depth_policy == DepthPolicy.ERROR:
                raise SerializationError(
                    f"Depth limit exceeded at {ctx.current_path}",
                    path=ctx.current_path,
                )
            # TRUNCATE
            return ctx.get_reference_marker(obj)

        result = {}

        # Add type info if configured
        if ctx.scope.include_type_info:
            result["__type__"] = type(obj).__name__

        # Get field descriptors
        descriptors = self._get_field_descriptors(type(obj))

        for desc in descriptors:
            # Check sensitive/large exclusions
            if ctx.scope.exclude_sensitive and desc.sensitive:
                continue
            if ctx.scope.exclude_large and desc.large:
                continue

            value = getattr(obj, desc.name, None)

            if not ctx.should_serialize_field(desc.name, value):
                continue

            ctx.enter(desc.name)
            try:
                # Handle reference fields specially
                if desc.reference and not ctx.scope.include_refs:
                    if value is not None:
                        result[desc.name] = ctx.get_reference_marker(value)
                    else:
                        result[desc.name] = None
                else:
                    result[desc.name] = self._serialize_value(value, ctx)
            finally:
                ctx.exit()

        return result

    def _get_field_descriptors(self, cls: Type) -> List[FieldDescriptor]:
        """Get field descriptors for a type, with caching."""
        if cls in self._field_descriptors:
            return self._field_descriptors[cls]

        descriptors = []

        if is_dataclass(cls):
            try:
                hints = get_type_hints(cls)
            except Exception:
                hints = {}

            for f in fields(cls):
                has_default = f.default is not MISSING or f.default_factory is not MISSING
                default = f.default if f.default is not MISSING else None

                desc = FieldDescriptor(
                    name=f.name,
                    type_hint=hints.get(f.name, Any),
                    default=default,
                    has_default=has_default,
                    sensitive=f.metadata.get("sensitive", False),
                    large=f.metadata.get("large", False),
                    lazy=f.metadata.get("lazy", False),
                    reference=f.metadata.get("reference", False),
                    max_depth=f.metadata.get("max_depth"),
                )
                descriptors.append(desc)

        self._field_descriptors[cls] = descriptors
        return descriptors

    def to_json(
        self,
        obj: Any,
        pretty: bool = False,
        ctx: Optional[PartialSerializationContext] = None,
    ) -> str:
        """Serialize to JSON string."""
        data = self.serialize(obj, ctx)
        if pretty:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, default=str)


class PartialDeserializer:
    """Deserializer that handles partial/incomplete data gracefully.

    Features:
    - Missing field handling with defaults
    - Type coercion
    - Reference resolution
    - Partial object reconstruction
    """

    def __init__(
        self,
        missing_policy: MissingFieldPolicy = MissingFieldPolicy.DEFAULT,
        strict: bool = False,
    ):
        self.missing_policy = missing_policy
        self.strict = strict
        self._reference_cache: Dict[str, Any] = {}
        self._pending_refs: List[Tuple[Any, str, str]] = []
        self._default_factories: Dict[Type, Callable[[], Any]] = {}

    def register_default_factory(self, cls: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function for creating default instances."""
        self._default_factories[cls] = factory

    def deserialize(
        self,
        data: Any,
        target_type: Type[T],
        ctx: Optional[PartialSerializationContext] = None,
    ) -> T:
        """Deserialize data into target type with partial support."""
        if ctx is None:
            ctx = PartialSerializationContext()

        result = self._deserialize_value(data, target_type, ctx)

        # Resolve pending references
        self._resolve_pending_refs()

        return result

    def _deserialize_value(
        self,
        value: Any,
        type_hint: Any,
        ctx: PartialSerializationContext,
    ) -> Any:
        """Deserialize a single value."""
        if value is None:
            return self._get_default_for_type(type_hint)

        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # Handle reference markers
        if isinstance(value, dict) and "__ref__" in value:
            return self._handle_reference(value, type_hint)

        # Handle truncated markers
        if isinstance(value, dict) and "__truncated__" in value:
            return self._handle_truncated(value, type_hint)

        # Handle special markers
        if isinstance(value, dict):
            if "__enum__" in value:
                return value["value"]  # Simplified enum handling

            if "__set__" in value:
                item_type = args[0] if args else Any
                return {
                    self._deserialize_value(v, item_type, ctx)
                    for v in value["__set__"]
                }

        # Basic types (primitives only)
        if type_hint in (bool, int, float, str):
            return self._coerce_primitive(value, type_hint)

        # For non-generic types where origin is None but it's a complex type
        # we need to check if it's a dataclass or has deserialize before coercing
        if origin is None and isinstance(type_hint, type):
            if is_dataclass(type_hint):
                return self._deserialize_dataclass(value, type_hint, ctx)
            if hasattr(type_hint, "deserialize"):
                return self._deserialize_object_with_method(value, type_hint, ctx)
            # Unknown type - return as-is or try coercion for basic types
            if not isinstance(value, dict):
                return self._coerce_primitive(value, type_hint)
            return value

        # List
        if origin is list:
            item_type = args[0] if args else Any
            if not isinstance(value, list):
                return []
            return [self._deserialize_value(v, item_type, ctx) for v in value]

        # Tuple
        if origin is tuple:
            if not isinstance(value, (list, tuple)):
                return tuple()
            if args:
                return tuple(
                    self._deserialize_value(v, t, ctx) for v, t in zip(value, args)
                )
            return tuple(value)

        # Set
        if origin is set:
            item_type = args[0] if args else Any
            if not isinstance(value, (list, set)):
                return set()
            return {self._deserialize_value(v, item_type, ctx) for v in value}

        # Dict
        if origin is dict:
            key_type = args[0] if args else str
            val_type = args[1] if len(args) > 1 else Any
            if not isinstance(value, dict):
                return {}
            return {
                self._deserialize_value(k, key_type, ctx): self._deserialize_value(
                    v, val_type, ctx
                )
                for k, v in value.items()
                if not k.startswith("__")  # Skip metadata
            }

        # Optional
        if origin is Union:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return self._deserialize_value(value, non_none[0], ctx)

        # Objects - prefer dataclass handling for partial support
        if isinstance(type_hint, type):
            # Always use dataclass handling for dataclasses to properly support partial
            if is_dataclass(type_hint):
                return self._deserialize_dataclass(value, type_hint, ctx)
            # Non-dataclass with deserialize method
            if hasattr(type_hint, "deserialize"):
                return self._deserialize_object_with_method(value, type_hint, ctx)

        return value

    def _deserialize_object_with_method(
        self,
        data: Dict[str, Any],
        cls: Type[T],
        ctx: PartialSerializationContext,
    ) -> T:
        """Deserialize object that has a deserialize method (non-dataclass)."""
        if not isinstance(data, dict):
            if self.missing_policy == MissingFieldPolicy.ERROR:
                raise SerializationError(f"Expected dict for {cls.__name__}, got {type(data)}")
            return self._get_default_for_type(cls)

        try:
            std_ctx = ctx.to_serialization_context()
            return cls.deserialize(data, std_ctx)
        except SerializationError as e:
            if self.strict:
                raise
            # For non-dataclass, we can't do partial deserialization
            raise SerializationError(f"Failed to deserialize {cls.__name__}: {e}")

    def _deserialize_dataclass(
        self,
        data: Dict[str, Any],
        cls: Type[T],
        ctx: PartialSerializationContext,
    ) -> T:
        """Deserialize a dataclass with missing field handling."""
        if not isinstance(data, dict):
            if self.strict or self.missing_policy == MissingFieldPolicy.ERROR:
                raise SerializationError(f"Expected dict for {cls.__name__}, got {type(data)}")
            # Try to create with defaults
            return self._create_with_defaults(cls)

        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = {}

        kwargs = {}
        missing_fields = []

        for f in fields(cls):
            field_name = f.name
            field_type = hints.get(field_name, Any)

            if field_name in data:
                ctx.enter(field_name)
                try:
                    kwargs[field_name] = self._deserialize_value(
                        data[field_name], field_type, ctx
                    )
                finally:
                    ctx.exit()
            else:
                # Handle missing field
                default_value = self._get_field_default(f, field_type)

                if default_value is MISSING:
                    missing_fields.append(field_name)
                    if self.missing_policy == MissingFieldPolicy.ERROR:
                        raise SerializationError(
                            f"Missing required field: {field_name}",
                            path=ctx.current_path,
                        )
                    elif self.missing_policy == MissingFieldPolicy.SENTINEL:
                        kwargs[field_name] = MISSING
                    elif self.missing_policy == MissingFieldPolicy.SKIP:
                        continue
                    else:  # DEFAULT
                        kwargs[field_name] = self._get_default_for_type(field_type)
                else:
                    kwargs[field_name] = default_value

        try:
            return cls(**kwargs)
        except TypeError as e:
            if self.strict:
                raise SerializationError(f"Failed to create {cls.__name__}: {e}")
            return self._create_with_defaults(cls)

    def _get_field_default(self, f: Any, field_type: Any) -> Any:
        """Get the default value for a field."""
        if f.default is not MISSING:
            return f.default
        if f.default_factory is not MISSING:
            return f.default_factory()
        return MISSING

    def _get_default_for_type(self, type_hint: Any) -> Any:
        """Get a sensible default for a type."""
        if type_hint is None or type_hint is type(None):
            return None

        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # Handle Optional
        if origin is Union:
            non_none = [a for a in args if a is not type(None)]
            if not non_none or len(args) > len(non_none):
                return None

        # Check registered factories
        if isinstance(type_hint, type) and type_hint in self._default_factories:
            return self._default_factories[type_hint]()

        # Basic types
        if type_hint is bool:
            return False
        if type_hint is int:
            return 0
        if type_hint is float:
            return 0.0
        if type_hint is str:
            return ""

        # Collections
        if origin is list:
            return []
        if origin is tuple:
            return tuple()
        if origin is set:
            return set()
        if origin is dict:
            return {}

        return None

    def _coerce_primitive(self, value: Any, type_hint: Any) -> Any:
        """Coerce a value to a primitive type."""
        if type_hint is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)

        if type_hint is int:
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return 0
            return 0

        if type_hint is float:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return 0.0
            return 0.0

        if type_hint is str:
            return str(value)

        return value

    def _handle_reference(self, data: Dict[str, Any], type_hint: Any) -> Any:
        """Handle a reference marker."""
        ref_id = data["__ref__"]

        # Check cache first
        if ref_id in self._reference_cache:
            return self._reference_cache[ref_id]

        # Create a placeholder if we can
        if isinstance(type_hint, type):
            if type_hint in self._default_factories:
                placeholder = self._default_factories[type_hint]()
                self._reference_cache[ref_id] = placeholder
                return placeholder

            # Try to get ID from the reference
            obj_id = data.get("id") or data.get("entity_id")
            if obj_id is not None:
                # Return a reference object
                return ReferenceMarker(ref_id=ref_id, object_id=obj_id, type_name=data.get("__type__"))

        return ReferenceMarker(ref_id=ref_id, object_id=None, type_name=data.get("__type__"))

    def _handle_truncated(self, data: Dict[str, Any], type_hint: Any) -> Any:
        """Handle a truncated marker."""
        type_name = data.get("__type__", "Unknown")

        if isinstance(type_hint, type) and type_hint in self._default_factories:
            return self._default_factories[type_hint]()

        return TruncatedMarker(type_name=type_name)

    def _create_with_defaults(self, cls: Type[T]) -> T:
        """Create an instance using all defaults."""
        if cls in self._default_factories:
            return self._default_factories[cls]()

        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = {}

        kwargs = {}
        for f in fields(cls):
            if f.default is not MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not MISSING:
                kwargs[f.name] = f.default_factory()
            else:
                field_type = hints.get(f.name, Any)
                kwargs[f.name] = self._get_default_for_type(field_type)

        return cls(**kwargs)

    def _resolve_pending_refs(self) -> None:
        """Resolve any pending reference after initial deserialization."""
        for obj, attr_name, ref_id in self._pending_refs:
            if ref_id in self._reference_cache:
                setattr(obj, attr_name, self._reference_cache[ref_id])
        self._pending_refs.clear()

    def from_json(
        self,
        json_str: str,
        target_type: Type[T],
        ctx: Optional[PartialSerializationContext] = None,
    ) -> T:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return self.deserialize(data, target_type, ctx)


@dataclass(frozen=True)
class ReferenceMarker:
    """Marker for an unresolved reference."""

    ref_id: str
    object_id: Any
    type_name: Optional[str] = None

    def __repr__(self) -> str:
        if self.object_id:
            return f"<Ref({self.type_name or 'Unknown'}, id={self.object_id})>"
        return f"<Ref({self.ref_id})>"


@dataclass(frozen=True)
class TruncatedMarker:
    """Marker for truncated data."""

    type_name: str

    def __repr__(self) -> str:
        return f"<Truncated({self.type_name})>"


def partial_serializable(
    version: str = "1.0.0",
    name: Optional[str] = None,
    sensitive_fields: Optional[Set[str]] = None,
    large_fields: Optional[Set[str]] = None,
    reference_fields: Optional[Set[str]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator to make a dataclass partial-serializable.

    Extends @serializable with support for field metadata.
    """
    sensitive_fields = sensitive_fields or set()
    large_fields = large_fields or set()
    reference_fields = reference_fields or set()

    def decorator(cls: Type[T]) -> Type[T]:
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")

        # Apply base serializable decorator
        cls = serializable(version=version, name=name)(cls)

        # Store metadata for partial serialization
        cls.__partial_sensitive__ = sensitive_fields
        cls.__partial_large__ = large_fields
        cls.__partial_references__ = reference_fields

        # Add partial serialization method
        def partial_serialize(
            self,
            scope: Optional[ScopeConfig] = None,
            ctx: Optional[PartialSerializationContext] = None,
        ) -> Dict[str, Any]:
            if ctx is None:
                ctx = PartialSerializationContext(scope=scope or ScopeConfig())
            serializer = PartialSerializer(ctx.scope)
            return serializer.serialize(self, ctx)

        @classmethod
        def partial_deserialize(
            klass: Type[T],
            data: Dict[str, Any],
            missing_policy: MissingFieldPolicy = MissingFieldPolicy.DEFAULT,
            ctx: Optional[PartialSerializationContext] = None,
        ) -> T:
            deserializer = PartialDeserializer(missing_policy=missing_policy)
            return deserializer.deserialize(data, klass, ctx)

        cls.partial_serialize = partial_serialize
        cls.partial_deserialize = partial_deserialize

        return cls

    return decorator


def create_snapshot(obj: Any, scope: Optional[ScopeConfig] = None) -> Dict[str, Any]:
    """Create a minimal snapshot of an object.

    Convenience function for snapshot mode serialization.
    """
    if scope is None:
        scope = ScopeConfig.snapshot()
    else:
        scope = scope.copy_with(mode=SerializationMode.SNAPSHOT)

    serializer = PartialSerializer(scope)
    ctx = PartialSerializationContext(scope=scope)
    return serializer.serialize(obj, ctx)


def create_full_serialization(obj: Any, scope: Optional[ScopeConfig] = None) -> Dict[str, Any]:
    """Create a complete serialization of an object.

    Convenience function for full mode serialization.
    """
    if scope is None:
        scope = ScopeConfig.full()
    else:
        scope = scope.copy_with(mode=SerializationMode.FULL)

    serializer = PartialSerializer(scope)
    ctx = PartialSerializationContext(scope=scope)
    return serializer.serialize(obj, ctx)


def deserialize_partial(
    data: Dict[str, Any],
    target_type: Type[T],
    missing_policy: MissingFieldPolicy = MissingFieldPolicy.DEFAULT,
) -> T:
    """Deserialize partial data into target type.

    Convenience function for partial deserialization.
    """
    deserializer = PartialDeserializer(missing_policy=missing_policy)
    return deserializer.deserialize(data, target_type)


class PartialSerializationStats:
    """Statistics for partial serialization operations."""

    def __init__(self):
        self._lock = threading.Lock()
        self.serializations = 0
        self.deserializations = 0
        self.fields_serialized = 0
        self.fields_skipped = 0
        self.refs_replaced = 0
        self.depth_truncated = 0
        self.missing_fields_defaulted = 0
        self.type_coercions = 0

    def record_serialization(self, ctx: PartialSerializationContext) -> None:
        """Record stats from a serialization context."""
        with self._lock:
            self.serializations += 1
            stats = ctx.stats
            self.fields_serialized += stats.get("fields_serialized", 0)
            self.fields_skipped += stats.get("fields_skipped", 0)
            self.refs_replaced += stats.get("refs_replaced", 0)
            self.depth_truncated += stats.get("depth_truncated", 0)

    def record_deserialization(
        self,
        missing_defaulted: int = 0,
        type_coercions: int = 0,
    ) -> None:
        """Record deserialization statistics."""
        with self._lock:
            self.deserializations += 1
            self.missing_fields_defaulted += missing_defaulted
            self.type_coercions += type_coercions

    def to_dict(self) -> Dict[str, int]:
        """Convert stats to dictionary."""
        return {
            "serializations": self.serializations,
            "deserializations": self.deserializations,
            "fields_serialized": self.fields_serialized,
            "fields_skipped": self.fields_skipped,
            "refs_replaced": self.refs_replaced,
            "depth_truncated": self.depth_truncated,
            "missing_fields_defaulted": self.missing_fields_defaulted,
            "type_coercions": self.type_coercions,
        }

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self.serializations = 0
            self.deserializations = 0
            self.fields_serialized = 0
            self.fields_skipped = 0
            self.refs_replaced = 0
            self.depth_truncated = 0
            self.missing_fields_defaulted = 0
            self.type_coercions = 0


_global_partial_stats = PartialSerializationStats()


def get_partial_serialization_stats() -> PartialSerializationStats:
    """Get global partial serialization statistics."""
    return _global_partial_stats
