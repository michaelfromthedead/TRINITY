"""T-CC-2.4: Serializable trait with schema versioning.

Provides a unified serialization framework with versioning support for
cross-format compatibility (binary, JSON, diff).
"""
from __future__ import annotations

import hashlib
import json
import struct
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, auto, IntEnum
from io import BytesIO
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .data_driven import SchemaGenerator, TypeMapper


T = TypeVar('T')


class SerializationFormat(Enum):
    """Supported serialization formats."""
    BINARY = auto()
    JSON = auto()
    COMPACT_JSON = auto()
    DIFF = auto()


class VersionCompatibility(Enum):
    """Compatibility levels between schema versions."""
    COMPATIBLE = auto()  # Can load directly
    MIGRATABLE = auto()  # Can load with migration
    INCOMPATIBLE = auto()  # Cannot load


@dataclass(frozen=True)
class SchemaVersion:
    """Version identifier for a schema."""
    major: int
    minor: int
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: "SchemaVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "SchemaVersion") -> bool:
        return (self.major, self.minor, self.patch) <= (other.major, other.minor, other.patch)

    def is_compatible(self, other: "SchemaVersion") -> VersionCompatibility:
        """Check compatibility with another version."""
        if self.major != other.major:
            return VersionCompatibility.INCOMPATIBLE
        if self.minor != other.minor:
            return VersionCompatibility.MIGRATABLE
        return VersionCompatibility.COMPATIBLE

    @classmethod
    def from_string(cls, s: str) -> "SchemaVersion":
        """Parse version from string."""
        parts = s.split(".")
        if len(parts) == 2:
            return cls(int(parts[0]), int(parts[1]))
        elif len(parts) == 3:
            return cls(int(parts[0]), int(parts[1]), int(parts[2]))
        raise ValueError(f"Invalid version string: {s}")


@dataclass
class SchemaInfo:
    """Metadata about a serializable schema."""
    name: str
    version: SchemaVersion
    hash: str
    fields: List[str]
    optional_fields: Set[str] = field(default_factory=set)
    deprecated_fields: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": str(self.version),
            "hash": self.hash,
            "fields": self.fields,
            "optional_fields": list(self.optional_fields),
            "deprecated_fields": list(self.deprecated_fields),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaInfo":
        return cls(
            name=data["name"],
            version=SchemaVersion.from_string(data["version"]),
            hash=data["hash"],
            fields=data["fields"],
            optional_fields=set(data.get("optional_fields", [])),
            deprecated_fields=set(data.get("deprecated_fields", [])),
        )


@dataclass
class SerializationError(Exception):
    """Error during serialization or deserialization."""
    message: str
    path: str = ""
    version_mismatch: bool = False
    schema_info: Optional[SchemaInfo] = None

    def __str__(self) -> str:
        if self.path:
            return f"{self.path}: {self.message}"
        return self.message


class SerializationContext:
    """Context for serialization operations."""

    def __init__(
        self,
        format: SerializationFormat = SerializationFormat.JSON,
        include_schema: bool = True,
        include_defaults: bool = False,
        max_depth: Optional[int] = None,
        skip_fields: Optional[Set[str]] = None,
    ):
        self.format = format
        self.include_schema = include_schema
        self.include_defaults = include_defaults
        self.max_depth = max_depth
        self.skip_fields = skip_fields or set()
        self._current_depth = 0
        self._path: List[str] = []
        self._references: Dict[int, Any] = {}

    @property
    def current_path(self) -> str:
        return ".".join(self._path)

    def enter(self, name: str) -> "SerializationContext":
        """Enter a nested field."""
        self._path.append(name)
        self._current_depth += 1
        return self

    def exit(self) -> None:
        """Exit a nested field."""
        if self._path:
            self._path.pop()
        self._current_depth -= 1

    def should_serialize_field(self, field_name: str) -> bool:
        """Check if a field should be serialized."""
        if field_name in self.skip_fields:
            return False
        if self.max_depth and self._current_depth > self.max_depth:
            return False
        return True

    def register_reference(self, obj: Any, ref_id: int) -> None:
        """Register a reference for cycle detection."""
        self._references[ref_id] = obj

    def get_reference(self, ref_id: int) -> Optional[Any]:
        """Get a registered reference."""
        return self._references.get(ref_id)


class Serializable(ABC):
    """Base class for serializable types.

    Provides serialization/deserialization with schema versioning.
    """

    # Class-level version information
    __schema_version__: ClassVar[SchemaVersion] = SchemaVersion(1, 0, 0)
    __schema_name__: ClassVar[Optional[str]] = None

    @classmethod
    def get_schema_info(cls) -> SchemaInfo:
        """Get schema information for this type."""
        name = cls.__schema_name__ or cls.__name__

        # Get fields from type hints
        try:
            hints = get_type_hints(cls)
            field_names = [k for k in hints if not k.startswith("_")]
        except Exception:
            field_names = []

        # Compute schema hash
        hash_data = f"{name}:{cls.__schema_version__}:{','.join(sorted(field_names))}"
        schema_hash = hashlib.md5(hash_data.encode()).hexdigest()[:8]

        optional = set()
        deprecated = set()

        if is_dataclass(cls):
            for f in fields(cls):
                if f.metadata.get("optional"):
                    optional.add(f.name)
                if f.metadata.get("deprecated"):
                    deprecated.add(f.name)

        return SchemaInfo(
            name=name,
            version=cls.__schema_version__,
            hash=schema_hash,
            fields=field_names,
            optional_fields=optional,
            deprecated_fields=deprecated,
        )

    @abstractmethod
    def serialize(self, ctx: Optional[SerializationContext] = None) -> Dict[str, Any]:
        """Serialize to dictionary representation."""
        pass

    @classmethod
    @abstractmethod
    def deserialize(
        cls: Type[T],
        data: Dict[str, Any],
        ctx: Optional[SerializationContext] = None,
    ) -> T:
        """Deserialize from dictionary representation."""
        pass

    def to_json(self, pretty: bool = False) -> str:
        """Serialize to JSON string."""
        ctx = SerializationContext(format=SerializationFormat.JSON)
        data = self.serialize(ctx)
        if pretty:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, default=str)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.deserialize(data)

    def to_bytes(self) -> bytes:
        """Serialize to binary format."""
        ctx = SerializationContext(format=SerializationFormat.BINARY)
        data = self.serialize(ctx)
        return json.dumps(data).encode("utf-8")

    @classmethod
    def from_bytes(cls: Type[T], data: bytes) -> T:
        """Deserialize from binary format."""
        dict_data = json.loads(data.decode("utf-8"))
        return cls.deserialize(dict_data)


def serializable(
    version: str = "1.0.0",
    name: Optional[str] = None,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator to make a dataclass serializable."""

    def decorator(cls: Type[T]) -> Type[T]:
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")

        cls.__schema_version__ = SchemaVersion.from_string(version)
        cls.__schema_name__ = name or cls.__name__

        original_init = cls.__init__

        def serialize(self, ctx: Optional[SerializationContext] = None) -> Dict[str, Any]:
            if ctx is None:
                ctx = SerializationContext()

            result = {}

            if ctx.include_schema:
                result["__schema__"] = self.get_schema_info().to_dict()

            for f in fields(self):
                if not ctx.should_serialize_field(f.name):
                    continue

                value = getattr(self, f.name)

                # Skip None values unless include_defaults
                if value is None and not ctx.include_defaults:
                    continue

                # Skip default values unless include_defaults
                if not ctx.include_defaults:
                    if f.default is not field().default and value == f.default:
                        continue
                    if f.default_factory is not field().default_factory:
                        try:
                            if value == f.default_factory():
                                continue
                        except Exception:
                            pass

                ctx.enter(f.name)
                try:
                    result[f.name] = _serialize_value(value, ctx)
                finally:
                    ctx.exit()

            return result

        @classmethod
        def deserialize(
            klass: Type[T],
            data: Dict[str, Any],
            ctx: Optional[SerializationContext] = None,
        ) -> T:
            if ctx is None:
                ctx = SerializationContext()

            # Check schema compatibility
            if "__schema__" in data:
                stored_info = SchemaInfo.from_dict(data["__schema__"])
                current_info = klass.get_schema_info()

                compat = current_info.version.is_compatible(stored_info.version)
                if compat == VersionCompatibility.INCOMPATIBLE:
                    raise SerializationError(
                        f"Incompatible schema versions: stored={stored_info.version}, "
                        f"current={current_info.version}",
                        version_mismatch=True,
                        schema_info=stored_info,
                    )

            # Get type hints for deserialization
            try:
                hints = get_type_hints(klass)
            except Exception:
                hints = {}

            kwargs = {}
            for f in fields(klass):
                if f.name not in data:
                    # Check for default
                    if f.default is not field().default:
                        kwargs[f.name] = f.default
                    elif f.default_factory is not field().default_factory:
                        kwargs[f.name] = f.default_factory()
                    elif f.name in klass.get_schema_info().optional_fields:
                        kwargs[f.name] = None
                    else:
                        raise SerializationError(
                            f"Missing required field: {f.name}",
                            path=ctx.current_path,
                        )
                else:
                    ctx.enter(f.name)
                    try:
                        field_type = hints.get(f.name, Any)
                        kwargs[f.name] = _deserialize_value(data[f.name], field_type, ctx)
                    finally:
                        ctx.exit()

            return klass(**kwargs)

        @classmethod
        def get_schema_info(klass) -> SchemaInfo:
            return Serializable.get_schema_info.__func__(klass)

        cls.serialize = serialize
        cls.deserialize = deserialize
        cls.get_schema_info = get_schema_info
        cls.to_json = Serializable.to_json
        cls.from_json = classmethod(lambda klass, s: Serializable.from_json.__func__(klass, s))
        cls.to_bytes = Serializable.to_bytes
        cls.from_bytes = classmethod(lambda klass, b: Serializable.from_bytes.__func__(klass, b))

        return cls

    return decorator


def _serialize_value(value: Any, ctx: SerializationContext) -> Any:
    """Serialize a single value."""
    if value is None:
        return None

    if isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, Enum):
        return {"__enum__": type(value).__name__, "value": value.value}

    if isinstance(value, (list, tuple)):
        return [_serialize_value(v, ctx) for v in value]

    if isinstance(value, set):
        return {"__set__": [_serialize_value(v, ctx) for v in value]}

    if isinstance(value, dict):
        return {str(k): _serialize_value(v, ctx) for k, v in value.items()}

    if hasattr(value, "serialize"):
        return value.serialize(ctx)

    if is_dataclass(value):
        result = {}
        for f in fields(value):
            ctx.enter(f.name)
            try:
                result[f.name] = _serialize_value(getattr(value, f.name), ctx)
            finally:
                ctx.exit()
        return result

    # Fallback to string representation
    return str(value)


def _deserialize_value(value: Any, type_hint: Any, ctx: SerializationContext) -> Any:
    """Deserialize a single value."""
    if value is None:
        return None

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # Handle special markers
    if isinstance(value, dict):
        if "__enum__" in value:
            # Look up enum by name - simplified
            return value["value"]

        if "__set__" in value:
            item_type = args[0] if args else Any
            return {_deserialize_value(v, item_type, ctx) for v in value["__set__"]}

    # Handle basic types
    if type_hint in (bool, int, float, str) or origin is None:
        if type_hint == bool and isinstance(value, bool):
            return value
        if type_hint == int and isinstance(value, (int, float)):
            return int(value)
        if type_hint == float and isinstance(value, (int, float)):
            return float(value)
        if type_hint == str:
            return str(value)

    # Handle List
    if origin is list:
        item_type = args[0] if args else Any
        return [_deserialize_value(v, item_type, ctx) for v in value]

    # Handle Tuple
    if origin is tuple:
        if args:
            return tuple(_deserialize_value(v, t, ctx) for v, t in zip(value, args))
        return tuple(value)

    # Handle Set
    if origin is set:
        item_type = args[0] if args else Any
        return {_deserialize_value(v, item_type, ctx) for v in value}

    # Handle Dict
    if origin is dict:
        key_type = args[0] if args else str
        val_type = args[1] if len(args) > 1 else Any
        return {
            _deserialize_value(k, key_type, ctx): _deserialize_value(v, val_type, ctx)
            for k, v in value.items()
        }

    # Handle Optional
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _deserialize_value(value, non_none[0], ctx)

    # Handle serializable types
    if isinstance(type_hint, type):
        if hasattr(type_hint, "deserialize"):
            return type_hint.deserialize(value, ctx)
        if is_dataclass(type_hint):
            return type_hint(**value)

    return value


class SchemaRegistry:
    """Registry for serializable schemas."""

    _instance: Optional["SchemaRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SchemaRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._schemas: Dict[str, Type[Serializable]] = {}
                cls._instance._migrations: Dict[Tuple[str, str], Callable] = {}
            return cls._instance

    @classmethod
    def get_instance(cls) -> "SchemaRegistry":
        return cls()

    def register(self, cls: Type[Serializable]) -> None:
        """Register a serializable type."""
        info = cls.get_schema_info()
        key = f"{info.name}:{info.version}"
        self._schemas[key] = cls

    def get(self, name: str, version: Optional[str] = None) -> Optional[Type[Serializable]]:
        """Get a registered type."""
        if version:
            key = f"{name}:{version}"
            return self._schemas.get(key)

        # Find latest version
        matches = [(k, v) for k, v in self._schemas.items() if k.startswith(f"{name}:")]
        if not matches:
            return None
        return max(matches, key=lambda x: x[0])[1]

    def register_migration(
        self,
        from_version: str,
        to_version: str,
        migration: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        """Register a migration function between versions."""
        self._migrations[(from_version, to_version)] = migration

    def migrate(
        self,
        data: Dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> Dict[str, Any]:
        """Migrate data between versions."""
        key = (from_version, to_version)
        if key not in self._migrations:
            raise SerializationError(f"No migration path from {from_version} to {to_version}")
        return self._migrations[key](data)

    def list_schemas(self) -> List[SchemaInfo]:
        """List all registered schemas."""
        return [cls.get_schema_info() for cls in self._schemas.values()]

    def clear(self) -> None:
        """Clear all registrations."""
        self._schemas.clear()
        self._migrations.clear()


def register_serializable(cls: Type[T]) -> Type[T]:
    """Decorator to register a serializable type."""
    SchemaRegistry.get_instance().register(cls)
    return cls


class SerializationStats:
    """Statistics about serialization operations."""

    def __init__(self):
        self.serialize_count = 0
        self.deserialize_count = 0
        self.total_bytes_written = 0
        self.total_bytes_read = 0
        self.errors = 0
        self._lock = threading.Lock()

    def record_serialize(self, byte_count: int) -> None:
        with self._lock:
            self.serialize_count += 1
            self.total_bytes_written += byte_count

    def record_deserialize(self, byte_count: int) -> None:
        with self._lock:
            self.deserialize_count += 1
            self.total_bytes_read += byte_count

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "serialize_count": self.serialize_count,
            "deserialize_count": self.deserialize_count,
            "total_bytes_written": self.total_bytes_written,
            "total_bytes_read": self.total_bytes_read,
            "errors": self.errors,
        }


_global_stats = SerializationStats()


def get_serialization_stats() -> SerializationStats:
    """Get global serialization statistics."""
    return _global_stats


def serialize_to_file(obj: Serializable, path: str, format: SerializationFormat = SerializationFormat.JSON) -> None:
    """Serialize an object to a file."""
    ctx = SerializationContext(format=format)
    data = obj.serialize(ctx)

    if format == SerializationFormat.JSON:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    else:
        with open(path, "wb") as f:
            f.write(json.dumps(data).encode("utf-8"))

    _global_stats.record_serialize(len(json.dumps(data)))


def deserialize_from_file(
    cls: Type[T],
    path: str,
    format: SerializationFormat = SerializationFormat.JSON,
) -> T:
    """Deserialize an object from a file."""
    if format == SerializationFormat.JSON:
        with open(path, "r") as f:
            data = json.load(f)
    else:
        with open(path, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))

    _global_stats.record_deserialize(len(json.dumps(data)))
    return cls.deserialize(data)
