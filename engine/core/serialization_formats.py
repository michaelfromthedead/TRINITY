"""T-CC-2.5: Multi-format serialization writers/readers.

Provides BinaryWriter/Reader, JSONWriter/Reader, and DiffWriter/Reader
with streaming support and common interfaces.
"""
from __future__ import annotations

import hashlib
import json
import struct
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, IntEnum
from io import BytesIO, StringIO
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    TextIO,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .serialization import (
    SchemaInfo,
    SchemaVersion,
    Serializable,
    SerializationContext,
    SerializationError,
    SerializationFormat,
    _serialize_value,
    _deserialize_value,
)


T = TypeVar('T')

# Binary format constants
BINARY_MAGIC = b'TRNY'
BINARY_VERSION = 1
BINARY_HEADER_SIZE = 16  # magic(4) + version(4) + flags(4) + checksum(4)

# Type tags for binary encoding
class BinaryTypeTag(IntEnum):
    """Type tags for binary serialization."""
    NULL = 0
    BOOL_FALSE = 1
    BOOL_TRUE = 2
    INT8 = 3
    INT16 = 4
    INT32 = 5
    INT64 = 6
    FLOAT32 = 7
    FLOAT64 = 8
    STRING = 9
    BYTES = 10
    LIST = 11
    DICT = 12
    SET = 13
    OBJECT = 14
    ENUM = 15
    SCHEMA_REF = 16


class BinaryFlags(IntEnum):
    """Flags for binary format."""
    NONE = 0
    COMPRESSED = 1
    HAS_SCHEMA = 2
    HAS_CHECKSUM = 4


# Base interfaces

class SerializationWriter(ABC):
    """Base interface for serialization writers."""

    @abstractmethod
    def write(self, obj: Any) -> None:
        """Write an object."""
        pass

    @abstractmethod
    def write_field(self, name: str, value: Any) -> None:
        """Write a named field."""
        pass

    @abstractmethod
    def begin_object(self, schema: Optional[SchemaInfo] = None) -> None:
        """Begin writing an object."""
        pass

    @abstractmethod
    def end_object(self) -> None:
        """End writing an object."""
        pass

    @abstractmethod
    def begin_list(self, length: int) -> None:
        """Begin writing a list."""
        pass

    @abstractmethod
    def end_list(self) -> None:
        """End writing a list."""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered data."""
        pass

    @abstractmethod
    def get_output(self) -> Any:
        """Get the serialized output."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset writer state for reuse."""
        pass


class SerializationReader(ABC):
    """Base interface for serialization readers."""

    @abstractmethod
    def read(self) -> Any:
        """Read the next value."""
        pass

    @abstractmethod
    def read_field(self, name: str) -> Any:
        """Read a named field."""
        pass

    @abstractmethod
    def begin_object(self) -> Optional[SchemaInfo]:
        """Begin reading an object, returns schema if present."""
        pass

    @abstractmethod
    def end_object(self) -> None:
        """End reading an object."""
        pass

    @abstractmethod
    def begin_list(self) -> int:
        """Begin reading a list, returns length."""
        pass

    @abstractmethod
    def end_list(self) -> None:
        """End reading a list."""
        pass

    @abstractmethod
    def has_more(self) -> bool:
        """Check if there's more data to read."""
        pass

    @abstractmethod
    def peek_type(self) -> Optional[str]:
        """Peek at the type of the next value without consuming it."""
        pass


# Binary Writer/Reader

class BinaryWriter(SerializationWriter):
    """Binary format writer with compact encoding and version headers."""

    def __init__(
        self,
        stream: Optional[BinaryIO] = None,
        compress: bool = False,
        include_checksum: bool = True,
        include_schema: bool = True,
    ):
        self._stream = stream or BytesIO()
        self._compress = compress
        self._include_checksum = include_checksum
        self._include_schema = include_schema
        self._buffer = BytesIO()
        self._depth = 0
        self._written_schemas: Dict[str, int] = {}
        self._schema_counter = 0
        self._finalized = False

    def write(self, obj: Any) -> None:
        """Write any value."""
        if obj is None:
            self._write_tag(BinaryTypeTag.NULL)
        elif isinstance(obj, bool):
            self._write_tag(BinaryTypeTag.BOOL_TRUE if obj else BinaryTypeTag.BOOL_FALSE)
        elif isinstance(obj, int):
            self._write_int(obj)
        elif isinstance(obj, float):
            self._write_float(obj)
        elif isinstance(obj, str):
            self._write_string(obj)
        elif isinstance(obj, bytes):
            self._write_bytes(obj)
        elif isinstance(obj, (list, tuple)):
            self.begin_list(len(obj))
            for item in obj:
                self.write(item)
            self.end_list()
        elif isinstance(obj, set):
            self._write_tag(BinaryTypeTag.SET)
            self._write_varint(len(obj))
            for item in obj:
                self.write(item)
        elif isinstance(obj, dict):
            if "__schema__" in obj and self._include_schema:
                schema_info = SchemaInfo.from_dict(obj["__schema__"])
                self.begin_object(schema_info)
                for k, v in obj.items():
                    if k != "__schema__":
                        self.write_field(k, v)
                self.end_object()
            elif "__set__" in obj:
                self._write_tag(BinaryTypeTag.SET)
                items = obj["__set__"]
                self._write_varint(len(items))
                for item in items:
                    self.write(item)
            elif "__enum__" in obj:
                self._write_tag(BinaryTypeTag.ENUM)
                self._write_string(obj["__enum__"])
                self.write(obj["value"])
            else:
                self._write_tag(BinaryTypeTag.DICT)
                self._write_varint(len(obj))
                for k, v in obj.items():
                    self._write_string(k)
                    self.write(v)
        elif isinstance(obj, Enum):
            self._write_tag(BinaryTypeTag.ENUM)
            self._write_string(type(obj).__name__)
            self.write(obj.value)
        elif hasattr(obj, "serialize"):
            ctx = SerializationContext(
                format=SerializationFormat.BINARY,
                include_schema=self._include_schema,
            )
            data = obj.serialize(ctx)
            self.write(data)
        elif is_dataclass(obj):
            self.begin_object()
            for f in fields(obj):
                self.write_field(f.name, getattr(obj, f.name))
            self.end_object()
        else:
            self._write_string(str(obj))

    def write_field(self, name: str, value: Any) -> None:
        """Write a named field."""
        self._write_string(name)
        self.write(value)

    def begin_object(self, schema: Optional[SchemaInfo] = None) -> None:
        """Begin writing an object with optional schema."""
        self._write_tag(BinaryTypeTag.OBJECT)
        self._depth += 1

        if schema and self._include_schema:
            schema_key = f"{schema.name}:{schema.version}"
            if schema_key in self._written_schemas:
                # Schema reference - write special marker then ref id
                self._buffer.write(b'\x02')  # Has schema reference
                self._write_varint(self._written_schemas[schema_key])
            else:
                self._written_schemas[schema_key] = self._schema_counter
                self._schema_counter += 1
                self._buffer.write(b'\x01')  # Has full schema
                self._write_string(schema.name)
                self._write_string(str(schema.version))
                self._write_string(schema.hash)
                self._write_varint(len(schema.fields))
                for field_name in schema.fields:
                    self._write_string(field_name)
        else:
            self._buffer.write(b'\x00')  # No schema

    def end_object(self) -> None:
        """End writing an object."""
        self._buffer.write(b'\xFF')  # End marker
        self._depth -= 1

    def begin_list(self, length: int) -> None:
        """Begin writing a list with known length."""
        self._write_tag(BinaryTypeTag.LIST)
        self._write_varint(length)

    def end_list(self) -> None:
        """End writing a list (no-op for binary, length is known)."""
        pass

    def flush(self) -> None:
        """Flush buffered data to stream with header."""
        if self._finalized:
            return

        data = self._buffer.getvalue()

        # Compress if requested
        if self._compress and len(data) > 64:
            compressed = zlib.compress(data)
            if len(compressed) < len(data):
                data = compressed
                flags = BinaryFlags.COMPRESSED
            else:
                flags = BinaryFlags.NONE
        else:
            flags = BinaryFlags.NONE

        if self._include_schema:
            flags |= BinaryFlags.HAS_SCHEMA
        if self._include_checksum:
            flags |= BinaryFlags.HAS_CHECKSUM

        # Calculate checksum
        checksum = zlib.crc32(data) if self._include_checksum else 0

        # Write header
        self._stream.write(BINARY_MAGIC)
        self._stream.write(struct.pack('<I', BINARY_VERSION))
        self._stream.write(struct.pack('<I', flags))
        self._stream.write(struct.pack('<I', checksum))

        # Write data length and data
        self._stream.write(struct.pack('<I', len(data)))
        self._stream.write(data)

        self._finalized = True

    def get_output(self) -> bytes:
        """Get the complete binary output."""
        self.flush()
        if isinstance(self._stream, BytesIO):
            return self._stream.getvalue()
        raise ValueError("Cannot get output from external stream")

    def reset(self) -> None:
        """Reset writer state."""
        self._buffer = BytesIO()
        self._stream = BytesIO()
        self._depth = 0
        self._written_schemas.clear()
        self._schema_counter = 0
        self._finalized = False

    def _write_tag(self, tag: BinaryTypeTag) -> None:
        self._buffer.write(bytes([tag]))

    def _write_varint(self, value: int) -> None:
        """Write a variable-length integer."""
        if value < 0:
            # Handle negative numbers
            self._buffer.write(b'\xFF')
            value = -value
        while value > 0x7F:
            self._buffer.write(bytes([(value & 0x7F) | 0x80]))
            value >>= 7
        self._buffer.write(bytes([value & 0x7F]))

    def _write_int(self, value: int) -> None:
        """Write an integer with appropriate size tag."""
        if -128 <= value <= 127:
            self._write_tag(BinaryTypeTag.INT8)
            self._buffer.write(struct.pack('<b', value))
        elif -32768 <= value <= 32767:
            self._write_tag(BinaryTypeTag.INT16)
            self._buffer.write(struct.pack('<h', value))
        elif -2147483648 <= value <= 2147483647:
            self._write_tag(BinaryTypeTag.INT32)
            self._buffer.write(struct.pack('<i', value))
        else:
            self._write_tag(BinaryTypeTag.INT64)
            self._buffer.write(struct.pack('<q', value))

    def _write_float(self, value: float) -> None:
        """Write a float."""
        self._write_tag(BinaryTypeTag.FLOAT64)
        self._buffer.write(struct.pack('<d', value))

    def _write_string(self, value: str) -> None:
        """Write a string with length prefix."""
        encoded = value.encode('utf-8')
        self._write_tag(BinaryTypeTag.STRING)
        self._write_varint(len(encoded))
        self._buffer.write(encoded)

    def _write_bytes(self, value: bytes) -> None:
        """Write raw bytes with length prefix."""
        self._write_tag(BinaryTypeTag.BYTES)
        self._write_varint(len(value))
        self._buffer.write(value)


class BinaryReader(SerializationReader):
    """Binary format reader with header validation."""

    def __init__(self, data: Union[bytes, BinaryIO]):
        if isinstance(data, bytes):
            self._stream = BytesIO(data)
        else:
            self._stream = data

        self._schemas: Dict[int, SchemaInfo] = {}
        self._schema_counter = 0
        self._current_object_fields: List[Dict[str, Any]] = []
        self._validated = False
        self._data_start = 0
        self._data_end = 0

    def _validate_header(self) -> None:
        """Validate and read header."""
        if self._validated:
            return

        magic = self._stream.read(4)
        if magic != BINARY_MAGIC:
            raise SerializationError(f"Invalid binary magic: {magic!r}")

        version = struct.unpack('<I', self._stream.read(4))[0]
        if version > BINARY_VERSION:
            raise SerializationError(f"Unsupported binary version: {version}")

        self._flags = struct.unpack('<I', self._stream.read(4))[0]
        expected_checksum = struct.unpack('<I', self._stream.read(4))[0]

        data_length = struct.unpack('<I', self._stream.read(4))[0]
        self._data_start = self._stream.tell()
        self._data_end = self._data_start + data_length

        # Read and validate data
        data = self._stream.read(data_length)

        # Verify checksum
        if self._flags & BinaryFlags.HAS_CHECKSUM:
            actual_checksum = zlib.crc32(data)
            if actual_checksum != expected_checksum:
                raise SerializationError(
                    f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
                )

        # Decompress if needed
        if self._flags & BinaryFlags.COMPRESSED:
            data = zlib.decompress(data)

        # Replace stream with decompressed data
        self._stream = BytesIO(data)
        self._validated = True

    def read(self) -> Any:
        """Read the next value."""
        self._validate_header()

        if not self.has_more():
            raise SerializationError("No more data to read")

        tag = self._read_tag()

        if tag == BinaryTypeTag.NULL:
            return None
        elif tag == BinaryTypeTag.BOOL_FALSE:
            return False
        elif tag == BinaryTypeTag.BOOL_TRUE:
            return True
        elif tag == BinaryTypeTag.INT8:
            return struct.unpack('<b', self._stream.read(1))[0]
        elif tag == BinaryTypeTag.INT16:
            return struct.unpack('<h', self._stream.read(2))[0]
        elif tag == BinaryTypeTag.INT32:
            return struct.unpack('<i', self._stream.read(4))[0]
        elif tag == BinaryTypeTag.INT64:
            return struct.unpack('<q', self._stream.read(8))[0]
        elif tag == BinaryTypeTag.FLOAT32:
            return struct.unpack('<f', self._stream.read(4))[0]
        elif tag == BinaryTypeTag.FLOAT64:
            return struct.unpack('<d', self._stream.read(8))[0]
        elif tag == BinaryTypeTag.STRING:
            length = self._read_varint()
            return self._stream.read(length).decode('utf-8')
        elif tag == BinaryTypeTag.BYTES:
            length = self._read_varint()
            return self._stream.read(length)
        elif tag == BinaryTypeTag.LIST:
            length = self._read_varint()
            return [self.read() for _ in range(length)]
        elif tag == BinaryTypeTag.SET:
            length = self._read_varint()
            return {self.read() for _ in range(length)}
        elif tag == BinaryTypeTag.DICT:
            length = self._read_varint()
            result = {}
            for _ in range(length):
                key = self._read_string_value()
                value = self.read()
                result[key] = value
            return result
        elif tag == BinaryTypeTag.OBJECT:
            return self._read_object()
        elif tag == BinaryTypeTag.ENUM:
            enum_name = self._read_string_value()
            enum_value = self.read()
            return {"__enum__": enum_name, "value": enum_value}
        elif tag == BinaryTypeTag.SCHEMA_REF:
            schema_id = self._read_varint()
            return self._schemas.get(schema_id)
        else:
            raise SerializationError(f"Unknown type tag: {tag}")

    def _read_object(self) -> Dict[str, Any]:
        """Read an object with optional schema."""
        result = {}

        schema_marker = self._stream.read(1)[0]
        if schema_marker == 0x01:
            # Full schema follows
            name = self._read_string_value()
            version = self._read_string_value()
            hash_val = self._read_string_value()
            field_count = self._read_varint()
            field_names = [self._read_string_value() for _ in range(field_count)]

            schema = SchemaInfo(
                name=name,
                version=SchemaVersion.from_string(version),
                hash=hash_val,
                fields=field_names,
            )
            self._schemas[self._schema_counter] = schema
            self._schema_counter += 1
            result["__schema__"] = schema.to_dict()
        elif schema_marker == 0x02:
            # Schema reference
            schema_id = self._read_varint()
            schema = self._schemas.get(schema_id)
            if schema:
                result["__schema__"] = schema.to_dict()
        # else schema_marker == 0x00: no schema

        # Read fields until end marker
        while True:
            pos = self._stream.tell()
            marker = self._stream.read(1)
            if not marker or marker[0] == 0xFF:
                break
            self._stream.seek(pos)  # Put back the byte

            field_name = self._read_string_value()
            field_value = self.read()
            result[field_name] = field_value

        return result

    def read_field(self, name: str) -> Any:
        """Read a named field from current object."""
        if not self._current_object_fields:
            raise SerializationError("Not inside an object")
        return self._current_object_fields[-1].get(name)

    def begin_object(self) -> Optional[SchemaInfo]:
        """Begin reading an object."""
        self._validate_header()
        tag = self._read_tag()
        if tag != BinaryTypeTag.OBJECT:
            raise SerializationError(f"Expected object, got tag {tag}")

        obj = self._read_object()
        self._current_object_fields.append(obj)

        if "__schema__" in obj:
            return SchemaInfo.from_dict(obj["__schema__"])
        return None

    def end_object(self) -> None:
        """End reading an object."""
        if self._current_object_fields:
            self._current_object_fields.pop()

    def begin_list(self) -> int:
        """Begin reading a list, returns length."""
        self._validate_header()
        tag = self._read_tag()
        if tag != BinaryTypeTag.LIST:
            raise SerializationError(f"Expected list, got tag {tag}")
        return self._read_varint()

    def end_list(self) -> None:
        """End reading a list."""
        pass

    def has_more(self) -> bool:
        """Check if there's more data."""
        self._validate_header()
        pos = self._stream.tell()
        data = self._stream.read(1)
        if data:
            self._stream.seek(pos)
            return True
        return False

    def peek_type(self) -> Optional[str]:
        """Peek at the type of next value."""
        self._validate_header()
        if not self.has_more():
            return None

        pos = self._stream.tell()
        tag = self._read_tag()
        self._stream.seek(pos)

        tag_names = {
            BinaryTypeTag.NULL: "null",
            BinaryTypeTag.BOOL_FALSE: "bool",
            BinaryTypeTag.BOOL_TRUE: "bool",
            BinaryTypeTag.INT8: "int",
            BinaryTypeTag.INT16: "int",
            BinaryTypeTag.INT32: "int",
            BinaryTypeTag.INT64: "int",
            BinaryTypeTag.FLOAT32: "float",
            BinaryTypeTag.FLOAT64: "float",
            BinaryTypeTag.STRING: "string",
            BinaryTypeTag.BYTES: "bytes",
            BinaryTypeTag.LIST: "list",
            BinaryTypeTag.DICT: "dict",
            BinaryTypeTag.SET: "set",
            BinaryTypeTag.OBJECT: "object",
            BinaryTypeTag.ENUM: "enum",
        }
        return tag_names.get(tag, "unknown")

    def _read_tag(self) -> BinaryTypeTag:
        data = self._stream.read(1)
        if not data:
            raise SerializationError("Unexpected end of data")
        return BinaryTypeTag(data[0])

    def _read_varint(self) -> int:
        """Read a variable-length integer."""
        first = self._stream.read(1)
        if not first:
            raise SerializationError("Unexpected end of data")

        if first[0] == 0xFF:
            # Negative number
            result = self._read_varint_unsigned()
            return -result

        self._stream.seek(self._stream.tell() - 1)
        return self._read_varint_unsigned()

    def _read_varint_unsigned(self) -> int:
        result = 0
        shift = 0
        while True:
            byte = self._stream.read(1)
            if not byte:
                raise SerializationError("Unexpected end of data")
            b = byte[0]
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        return result

    def _read_string_value(self) -> str:
        """Read a string value (expects STRING tag already consumed or reads it)."""
        tag = self._read_tag()
        if tag != BinaryTypeTag.STRING:
            raise SerializationError(f"Expected string, got tag {tag}")
        length = self._read_varint()
        return self._stream.read(length).decode('utf-8')


# JSON Writer/Reader

class JSONWriter(SerializationWriter):
    """JSON format writer with pretty printing and schema metadata."""

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        pretty: bool = True,
        indent: int = 2,
        include_schema: bool = True,
        sort_keys: bool = False,
    ):
        self._stream = stream or StringIO()
        self._pretty = pretty
        self._indent = indent
        self._include_schema = include_schema
        self._sort_keys = sort_keys
        self._current_object: List[Dict[str, Any]] = [{}]
        self._current_list: List[List[Any]] = []
        self._in_list = False
        self._root_written = False

    def write(self, obj: Any) -> None:
        """Write any value."""
        if hasattr(obj, "serialize"):
            ctx = SerializationContext(
                format=SerializationFormat.JSON,
                include_schema=self._include_schema,
            )
            value = obj.serialize(ctx)
        elif is_dataclass(obj):
            value = self._dataclass_to_dict(obj)
        elif isinstance(obj, Enum):
            value = {"__enum__": type(obj).__name__, "value": obj.value}
        elif isinstance(obj, set):
            value = {"__set__": list(obj)}
        elif isinstance(obj, (list, tuple)):
            value = [self._convert_value(v) for v in obj]
        elif isinstance(obj, dict):
            value = {k: self._convert_value(v) for k, v in obj.items()}
        else:
            value = obj

        if self._in_list and self._current_list:
            self._current_list[-1].append(value)
        elif self._current_object:
            self._current_object[-1] = value
            self._root_written = True

    def _convert_value(self, obj: Any) -> Any:
        """Convert a value for JSON serialization."""
        if hasattr(obj, "serialize"):
            ctx = SerializationContext(
                format=SerializationFormat.JSON,
                include_schema=self._include_schema,
            )
            return obj.serialize(ctx)
        elif is_dataclass(obj):
            return self._dataclass_to_dict(obj)
        elif isinstance(obj, Enum):
            return {"__enum__": type(obj).__name__, "value": obj.value}
        elif isinstance(obj, set):
            return {"__set__": [self._convert_value(v) for v in obj]}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_value(v) for v in obj]
        elif isinstance(obj, dict):
            return {str(k): self._convert_value(v) for k, v in obj.items()}
        return obj

    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert a dataclass to dict."""
        result = {}
        for f in fields(obj):
            result[f.name] = self._convert_value(getattr(obj, f.name))
        return result

    def write_field(self, name: str, value: Any) -> None:
        """Write a named field to current object."""
        if not self._current_object:
            raise SerializationError("Not inside an object")

        converted = self._convert_value(value)
        self._current_object[-1][name] = converted

    def begin_object(self, schema: Optional[SchemaInfo] = None) -> None:
        """Begin writing an object."""
        new_obj: Dict[str, Any] = {}

        if schema and self._include_schema:
            new_obj["__schema__"] = schema.to_dict()

        if self._in_list and self._current_list:
            self._current_list[-1].append(new_obj)

        self._current_object.append(new_obj)

    def end_object(self) -> None:
        """End writing an object."""
        if len(self._current_object) > 1:
            completed = self._current_object.pop()
            if not self._in_list and len(self._current_object) == 1:
                self._current_object[0] = completed
                self._root_written = True

    def begin_list(self, length: int) -> None:
        """Begin writing a list."""
        new_list: List[Any] = []
        self._current_list.append(new_list)
        self._in_list = True

    def end_list(self) -> None:
        """End writing a list."""
        if self._current_list:
            completed = self._current_list.pop()
            self._in_list = bool(self._current_list)

            if self._current_object and not self._in_list:
                if len(self._current_object) == 1 and not self._root_written:
                    self._current_object[0] = completed
                    self._root_written = True

    def flush(self) -> None:
        """Flush to stream."""
        if self._root_written or self._current_object:
            output = self._current_object[0] if self._current_object else {}

            if self._pretty:
                json_str = json.dumps(
                    output,
                    indent=self._indent,
                    sort_keys=self._sort_keys,
                    default=str,
                )
            else:
                json_str = json.dumps(output, sort_keys=self._sort_keys, default=str)

            self._stream.write(json_str)

    def get_output(self) -> str:
        """Get the JSON string output."""
        self.flush()
        if isinstance(self._stream, StringIO):
            return self._stream.getvalue()
        raise ValueError("Cannot get output from external stream")

    def reset(self) -> None:
        """Reset writer state."""
        self._stream = StringIO()
        self._current_object = [{}]
        self._current_list = []
        self._in_list = False
        self._root_written = False


class JSONReader(SerializationReader):
    """JSON format reader with schema validation."""

    def __init__(self, data: Union[str, TextIO, Dict[str, Any], List[Any]]):
        if isinstance(data, str):
            self._data = json.loads(data)
        elif isinstance(data, (dict, list)):
            self._data = data
        else:
            self._data = json.load(data)

        self._stack: List[Tuple[Any, int]] = [(self._data, 0)]
        self._consumed = False

    def read(self) -> Any:
        """Read the complete value."""
        if self._consumed:
            raise SerializationError("Data already consumed")
        self._consumed = True
        return self._data

    def read_field(self, name: str) -> Any:
        """Read a named field."""
        if not self._stack:
            raise SerializationError("No object context")

        current, _ = self._stack[-1]
        if not isinstance(current, dict):
            raise SerializationError("Current context is not an object")

        return current.get(name)

    def begin_object(self) -> Optional[SchemaInfo]:
        """Begin reading an object."""
        if not self._stack:
            raise SerializationError("No data")

        current, idx = self._stack[-1]

        if isinstance(current, list):
            if idx >= len(current):
                raise SerializationError("List index out of bounds")
            obj = current[idx]
            self._stack[-1] = (current, idx + 1)
        elif isinstance(current, dict):
            obj = current
        else:
            raise SerializationError(f"Expected object, got {type(current)}")

        self._stack.append((obj, 0))

        if isinstance(obj, dict) and "__schema__" in obj:
            schema_data = obj["__schema__"]
            # Only parse if it's a valid schema dict with required fields
            if isinstance(schema_data, dict) and "name" in schema_data:
                return SchemaInfo.from_dict(schema_data)
        return None

    def end_object(self) -> None:
        """End reading an object."""
        if len(self._stack) > 1:
            self._stack.pop()

    def begin_list(self) -> int:
        """Begin reading a list."""
        if not self._stack:
            raise SerializationError("No data")

        current, _ = self._stack[-1]

        if isinstance(current, dict):
            # Look for list in current context
            for v in current.values():
                if isinstance(v, list):
                    self._stack.append((v, 0))
                    return len(v)
        elif isinstance(current, list):
            self._stack.append((current, 0))
            return len(current)

        raise SerializationError("No list found")

    def end_list(self) -> None:
        """End reading a list."""
        if len(self._stack) > 1:
            self._stack.pop()

    def has_more(self) -> bool:
        """Check if there's more data."""
        if not self._stack:
            return False

        current, idx = self._stack[-1]
        if isinstance(current, list):
            return idx < len(current)
        return not self._consumed

    def peek_type(self) -> Optional[str]:
        """Peek at the type of next value."""
        if not self._stack:
            return None

        current, idx = self._stack[-1]

        if isinstance(current, list):
            if idx < len(current):
                value = current[idx]
            else:
                return None
        else:
            value = current

        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "list"
        elif isinstance(value, dict):
            if "__set__" in value:
                return "set"
            if "__enum__" in value:
                return "enum"
            return "object"
        return "unknown"

    def get_all_fields(self) -> Dict[str, Any]:
        """Get all fields from current object context."""
        if not self._stack:
            return {}

        current, _ = self._stack[-1]
        if isinstance(current, dict):
            return {k: v for k, v in current.items() if not k.startswith("__")}
        return {}


# Diff Writer/Reader

@dataclass
class DiffEntry:
    """A single diff entry representing a change."""
    path: str
    operation: str  # 'add', 'remove', 'replace'
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "path": self.path,
            "op": self.operation,
        }
        if self.old_value is not None:
            result["old"] = self.old_value
        if self.new_value is not None:
            result["new"] = self.new_value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiffEntry":
        return cls(
            path=data["path"],
            operation=data["op"],
            old_value=data.get("old"),
            new_value=data.get("new"),
        )


@dataclass
class DiffPatch:
    """A collection of diff entries forming a patch."""
    entries: List[DiffEntry] = field(default_factory=list)
    source_version: Optional[str] = None
    target_version: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "entries": [e.to_dict() for e in self.entries],
        }
        if self.source_version:
            result["source_version"] = self.source_version
        if self.target_version:
            result["target_version"] = self.target_version
        if self.timestamp:
            result["timestamp"] = self.timestamp
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiffPatch":
        return cls(
            entries=[DiffEntry.from_dict(e) for e in data.get("entries", [])],
            source_version=data.get("source_version"),
            target_version=data.get("target_version"),
            timestamp=data.get("timestamp"),
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[DiffEntry]:
        return iter(self.entries)


class DiffWriter(SerializationWriter):
    """Differential format writer for undo/network deltas."""

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        source_version: Optional[str] = None,
        target_version: Optional[str] = None,
    ):
        self._stream = stream or StringIO()
        self._entries: List[DiffEntry] = []
        self._source_version = source_version
        self._target_version = target_version
        self._path_stack: List[str] = []
        self._old_values: Dict[str, Any] = {}
        self._base_object: Optional[Dict[str, Any]] = None

    def set_base(self, obj: Any) -> None:
        """Set the base object for diff computation."""
        if hasattr(obj, "serialize"):
            ctx = SerializationContext(include_schema=False)
            self._base_object = obj.serialize(ctx)
        elif isinstance(obj, dict):
            self._base_object = obj
        else:
            self._base_object = None

        # Flatten base object for path lookups
        if self._base_object:
            self._old_values = self._flatten_object(self._base_object, "")

    def _flatten_object(self, obj: Any, prefix: str) -> Dict[str, Any]:
        """Flatten nested object to path -> value mapping."""
        result = {}

        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.startswith("__"):
                    continue
                path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)) and not (isinstance(v, dict) and "__" in str(v.keys())):
                    result.update(self._flatten_object(v, path))
                else:
                    result[path] = v
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                path = f"{prefix}[{i}]"
                if isinstance(v, (dict, list)):
                    result.update(self._flatten_object(v, path))
                else:
                    result[path] = v
        else:
            result[prefix] = obj

        return result

    def write(self, obj: Any) -> None:
        """Write object and compute diff against base."""
        if hasattr(obj, "serialize"):
            ctx = SerializationContext(include_schema=False)
            new_data = obj.serialize(ctx)
        elif isinstance(obj, dict):
            new_data = obj
        else:
            # For simple values, create replace entry
            path = ".".join(self._path_stack) if self._path_stack else "$"
            old_val = self._old_values.get(path)
            self._entries.append(DiffEntry(
                path=path,
                operation="replace" if old_val is not None else "add",
                old_value=old_val,
                new_value=obj,
            ))
            return

        # Compute diff between base and new
        new_values = self._flatten_object(new_data, "")

        # Find additions and changes
        for path, new_val in new_values.items():
            old_val = self._old_values.get(path)
            if old_val is None:
                self._entries.append(DiffEntry(
                    path=path,
                    operation="add",
                    new_value=new_val,
                ))
            elif old_val != new_val:
                self._entries.append(DiffEntry(
                    path=path,
                    operation="replace",
                    old_value=old_val,
                    new_value=new_val,
                ))

        # Find removals
        for path, old_val in self._old_values.items():
            if path not in new_values:
                self._entries.append(DiffEntry(
                    path=path,
                    operation="remove",
                    old_value=old_val,
                ))

    def write_field(self, name: str, value: Any) -> None:
        """Write a field change."""
        path = ".".join(self._path_stack + [name]) if self._path_stack else name
        old_val = self._old_values.get(path)

        # Convert value if needed
        if hasattr(value, "serialize"):
            ctx = SerializationContext(include_schema=False)
            new_val = value.serialize(ctx)
        elif isinstance(value, Enum):
            new_val = {"__enum__": type(value).__name__, "value": value.value}
        elif isinstance(value, set):
            new_val = {"__set__": list(value)}
        else:
            new_val = value

        if old_val is None:
            self._entries.append(DiffEntry(
                path=path,
                operation="add",
                new_value=new_val,
            ))
        elif old_val != new_val:
            self._entries.append(DiffEntry(
                path=path,
                operation="replace",
                old_value=old_val,
                new_value=new_val,
            ))

    def write_remove(self, path: str) -> None:
        """Write a removal entry."""
        old_val = self._old_values.get(path)
        self._entries.append(DiffEntry(
            path=path,
            operation="remove",
            old_value=old_val,
        ))

    def begin_object(self, schema: Optional[SchemaInfo] = None) -> None:
        """Begin writing an object context."""
        pass

    def end_object(self) -> None:
        """End writing an object context."""
        pass

    def begin_list(self, length: int) -> None:
        """Begin writing a list context."""
        pass

    def end_list(self) -> None:
        """End writing a list context."""
        pass

    def flush(self) -> None:
        """Flush diff entries to stream."""
        patch = DiffPatch(
            entries=self._entries,
            source_version=self._source_version,
            target_version=self._target_version,
        )
        json.dump(patch.to_dict(), self._stream, indent=2, default=str)

    def get_output(self) -> str:
        """Get the diff output as JSON string."""
        self.flush()
        if isinstance(self._stream, StringIO):
            return self._stream.getvalue()
        raise ValueError("Cannot get output from external stream")

    def get_patch(self) -> DiffPatch:
        """Get the diff patch object."""
        return DiffPatch(
            entries=self._entries.copy(),
            source_version=self._source_version,
            target_version=self._target_version,
        )

    def reset(self) -> None:
        """Reset writer state."""
        self._stream = StringIO()
        self._entries = []
        self._path_stack = []
        self._old_values = {}
        self._base_object = None

    def add_entry(self, entry: DiffEntry) -> None:
        """Add a diff entry directly."""
        self._entries.append(entry)


class DiffReader(SerializationReader):
    """Differential format reader for applying patches."""

    def __init__(self, data: Union[str, TextIO, Dict[str, Any], DiffPatch]):
        if isinstance(data, DiffPatch):
            self._patch = data
        elif isinstance(data, str):
            self._patch = DiffPatch.from_dict(json.loads(data))
        elif isinstance(data, dict):
            self._patch = DiffPatch.from_dict(data)
        else:
            self._patch = DiffPatch.from_dict(json.load(data))

        self._entry_index = 0

    def read(self) -> DiffPatch:
        """Read the complete patch."""
        return self._patch

    def read_entry(self) -> Optional[DiffEntry]:
        """Read the next diff entry."""
        if self._entry_index >= len(self._patch.entries):
            return None
        entry = self._patch.entries[self._entry_index]
        self._entry_index += 1
        return entry

    def read_field(self, name: str) -> Any:
        """Read a field value from current entry."""
        if self._entry_index == 0 or self._entry_index > len(self._patch.entries):
            return None
        entry = self._patch.entries[self._entry_index - 1]
        if name == "path":
            return entry.path
        elif name == "operation":
            return entry.operation
        elif name == "old_value":
            return entry.old_value
        elif name == "new_value":
            return entry.new_value
        return None

    def begin_object(self) -> Optional[SchemaInfo]:
        """Begin reading (returns None for diff format)."""
        return None

    def end_object(self) -> None:
        """End reading object context."""
        pass

    def begin_list(self) -> int:
        """Begin reading entries list."""
        return len(self._patch.entries)

    def end_list(self) -> None:
        """End reading list."""
        pass

    def has_more(self) -> bool:
        """Check if more entries exist."""
        return self._entry_index < len(self._patch.entries)

    def peek_type(self) -> Optional[str]:
        """Peek at next entry type."""
        if not self.has_more():
            return None
        return "entry"

    def apply_to(self, obj: Any) -> Any:
        """Apply patch to an object and return modified copy."""
        if hasattr(obj, "serialize"):
            ctx = SerializationContext(include_schema=False)
            data = obj.serialize(ctx)
        elif isinstance(obj, dict):
            data = dict(obj)
        else:
            data = obj

        for entry in self._patch.entries:
            data = self._apply_entry(data, entry)

        return data

    def _apply_entry(self, data: Any, entry: DiffEntry) -> Any:
        """Apply a single diff entry to data."""
        path_parts = self._parse_path(entry.path)

        if not path_parts:
            # Root level change
            if entry.operation == "replace":
                return entry.new_value
            elif entry.operation == "add":
                return entry.new_value
            elif entry.operation == "remove":
                return None
            return data

        return self._set_nested(data, path_parts, entry)

    def _parse_path(self, path: str) -> List[Union[str, int]]:
        """Parse a dot-notation path with array indices."""
        if path == "$":
            return []

        parts: List[Union[str, int]] = []
        current = ""
        i = 0

        while i < len(path):
            char = path[i]
            if char == ".":
                if current:
                    parts.append(current)
                    current = ""
            elif char == "[":
                if current:
                    parts.append(current)
                    current = ""
                # Find closing bracket
                j = i + 1
                while j < len(path) and path[j] != "]":
                    j += 1
                parts.append(int(path[i+1:j]))
                i = j
            else:
                current += char
            i += 1

        if current:
            parts.append(current)

        return parts

    def _set_nested(
        self,
        data: Any,
        path_parts: List[Union[str, int]],
        entry: DiffEntry,
    ) -> Any:
        """Set a nested value in data structure."""
        if not path_parts:
            if entry.operation == "remove":
                return None
            return entry.new_value

        if isinstance(data, dict):
            data = dict(data)
            key = path_parts[0]
            if len(path_parts) == 1:
                if entry.operation == "remove":
                    data.pop(str(key), None)
                else:
                    data[str(key)] = entry.new_value
            else:
                if str(key) in data:
                    data[str(key)] = self._set_nested(
                        data[str(key)],
                        path_parts[1:],
                        entry,
                    )
                elif entry.operation == "add":
                    # Create nested structure
                    data[str(key)] = self._create_nested(path_parts[1:], entry.new_value)
            return data

        elif isinstance(data, list):
            data = list(data)
            idx = path_parts[0]
            if not isinstance(idx, int):
                return data

            # Extend list if needed
            while len(data) <= idx:
                data.append(None)

            if len(path_parts) == 1:
                if entry.operation == "remove":
                    if idx < len(data):
                        data.pop(idx)
                else:
                    data[idx] = entry.new_value
            else:
                data[idx] = self._set_nested(data[idx], path_parts[1:], entry)
            return data

        return data

    def _create_nested(self, path_parts: List[Union[str, int]], value: Any) -> Any:
        """Create a nested structure for remaining path parts."""
        if not path_parts:
            return value

        key = path_parts[0]
        if isinstance(key, int):
            result: List[Any] = []
            while len(result) <= key:
                result.append(None)
            result[key] = self._create_nested(path_parts[1:], value)
            return result
        else:
            return {str(key): self._create_nested(path_parts[1:], value)}

    def get_entries(self) -> List[DiffEntry]:
        """Get all diff entries."""
        return self._patch.entries.copy()

    def get_patch(self) -> DiffPatch:
        """Get the diff patch object."""
        return self._patch


# Utility functions

def create_writer(
    format: SerializationFormat,
    **kwargs: Any,
) -> SerializationWriter:
    """Factory function to create appropriate writer."""
    if format == SerializationFormat.BINARY:
        return BinaryWriter(**kwargs)
    elif format == SerializationFormat.JSON:
        return JSONWriter(**kwargs)
    elif format == SerializationFormat.COMPACT_JSON:
        return JSONWriter(pretty=False, **kwargs)
    elif format == SerializationFormat.DIFF:
        return DiffWriter(**kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_reader(
    format: SerializationFormat,
    data: Any,
) -> SerializationReader:
    """Factory function to create appropriate reader."""
    if format == SerializationFormat.BINARY:
        return BinaryReader(data)
    elif format in (SerializationFormat.JSON, SerializationFormat.COMPACT_JSON):
        return JSONReader(data)
    elif format == SerializationFormat.DIFF:
        return DiffReader(data)
    else:
        raise ValueError(f"Unsupported format: {format}")


def compute_diff(
    old_obj: Any,
    new_obj: Any,
    source_version: Optional[str] = None,
    target_version: Optional[str] = None,
) -> DiffPatch:
    """Compute diff between two objects."""
    writer = DiffWriter(
        source_version=source_version,
        target_version=target_version,
    )
    writer.set_base(old_obj)
    writer.write(new_obj)
    return writer.get_patch()


def apply_diff(obj: Any, patch: DiffPatch) -> Any:
    """Apply a diff patch to an object."""
    reader = DiffReader(patch)
    return reader.apply_to(obj)
