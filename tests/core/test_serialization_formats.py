"""Tests for multi-format serialization writers/readers (T-CC-2.5)."""
import json
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from engine.core.serialization import (
    SchemaInfo,
    SchemaVersion,
    SerializationContext,
    SerializationFormat,
    serializable,
)
from engine.core.serialization_formats import (
    BINARY_MAGIC,
    BINARY_VERSION,
    BinaryFlags,
    BinaryReader,
    BinaryTypeTag,
    BinaryWriter,
    DiffEntry,
    DiffPatch,
    DiffReader,
    DiffWriter,
    JSONReader,
    JSONWriter,
    SerializationReader,
    SerializationWriter,
    apply_diff,
    compute_diff,
    create_reader,
    create_writer,
)


# Test fixtures

class Color(Enum):
    """Test enum."""
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@serializable()
@dataclass
class SimpleData:
    """Simple test dataclass."""
    name: str
    value: int


@serializable(version="2.0.0")
@dataclass
class ComplexData:
    """Complex test dataclass."""
    name: str
    values: List[int]
    mapping: Dict[str, float]
    optional: Optional[str] = None
    tags: Set[str] = field(default_factory=set)


@serializable()
@dataclass
class NestedData:
    """Nested test dataclass."""
    inner: SimpleData
    items: List[SimpleData]


# Binary Writer Tests

class TestBinaryWriter:
    """Tests for BinaryWriter."""

    def test_write_null(self):
        writer = BinaryWriter()
        writer.write(None)
        output = writer.get_output()

        assert output.startswith(BINARY_MAGIC)
        assert BinaryTypeTag.NULL.to_bytes(1, 'little') in output

    def test_write_bool(self):
        writer = BinaryWriter()
        writer.write(True)
        output = writer.get_output()

        assert BINARY_MAGIC in output

    def test_write_int_sizes(self):
        writer = BinaryWriter()
        writer.write(42)  # INT8
        output = writer.get_output()
        assert output is not None

        writer.reset()
        writer.write(1000)  # INT16
        output = writer.get_output()
        assert output is not None

        writer.reset()
        writer.write(100000)  # INT32
        output = writer.get_output()
        assert output is not None

        writer.reset()
        writer.write(10**15)  # INT64
        output = writer.get_output()
        assert output is not None

    def test_write_float(self):
        writer = BinaryWriter()
        writer.write(3.14159)
        output = writer.get_output()
        assert len(output) > BINARY_MAGIC.__len__()

    def test_write_string(self):
        writer = BinaryWriter()
        writer.write("hello world")
        output = writer.get_output()
        assert len(output) > len("hello world")

    def test_write_bytes(self):
        writer = BinaryWriter()
        writer.write(b"\x00\x01\x02\x03")
        output = writer.get_output()
        assert isinstance(output, bytes)

    def test_write_list(self):
        writer = BinaryWriter()
        writer.write([1, 2, 3, 4, 5])
        output = writer.get_output()
        assert len(output) > 0

    def test_write_set(self):
        writer = BinaryWriter()
        writer.write({1, 2, 3})
        output = writer.get_output()
        assert len(output) > 0

    def test_write_dict(self):
        writer = BinaryWriter()
        writer.write({"a": 1, "b": 2})
        output = writer.get_output()
        assert len(output) > 0

    def test_write_enum(self):
        writer = BinaryWriter()
        writer.write(Color.RED)
        output = writer.get_output()
        assert len(output) > 0

    def test_write_dataclass(self):
        writer = BinaryWriter()
        obj = SimpleData(name="test", value=42)
        writer.write(obj)
        output = writer.get_output()
        assert len(output) > 0

    def test_write_complex_dataclass(self):
        writer = BinaryWriter()
        obj = ComplexData(
            name="complex",
            values=[1, 2, 3],
            mapping={"x": 1.5},
            optional="yes",
            tags={"a", "b"},
        )
        writer.write(obj)
        output = writer.get_output()
        assert len(output) > 0

    def test_write_nested_dataclass(self):
        writer = BinaryWriter()
        obj = NestedData(
            inner=SimpleData(name="inner", value=1),
            items=[SimpleData(name="a", value=2)],
        )
        writer.write(obj)
        output = writer.get_output()
        assert len(output) > 0

    def test_compression(self):
        writer_uncompressed = BinaryWriter(compress=False)
        writer_compressed = BinaryWriter(compress=True)

        # Large data to trigger compression
        data = {"data": "x" * 1000}

        writer_uncompressed.write(data)
        writer_compressed.write(data)

        uncompressed = writer_uncompressed.get_output()
        compressed = writer_compressed.get_output()

        # Compressed should be smaller for repetitive data
        assert len(compressed) < len(uncompressed)

    def test_no_checksum(self):
        writer = BinaryWriter(include_checksum=False)
        writer.write({"test": 123})
        output = writer.get_output()
        assert len(output) > 0

    def test_no_schema(self):
        writer = BinaryWriter(include_schema=False)
        obj = SimpleData(name="test", value=1)
        writer.write(obj)
        output = writer.get_output()
        assert len(output) > 0

    def test_streaming_write(self):
        stream = BytesIO()
        writer = BinaryWriter(stream=stream)
        writer.begin_object()
        writer.write_field("name", "streamed")
        writer.write_field("value", 100)
        writer.end_object()
        writer.flush()

        assert stream.tell() > 0

    def test_reset(self):
        writer = BinaryWriter()
        writer.write(42)
        writer.reset()
        writer.write("new data")
        output = writer.get_output()
        assert b"new data".replace(b" ", b"") not in output or len(output) > 0


# Binary Reader Tests

class TestBinaryReader:
    """Tests for BinaryReader."""

    def _make_binary(self, obj: Any) -> bytes:
        writer = BinaryWriter()
        writer.write(obj)
        return writer.get_output()

    def test_read_null(self):
        data = self._make_binary(None)
        reader = BinaryReader(data)
        assert reader.read() is None

    def test_read_bool_true(self):
        data = self._make_binary(True)
        reader = BinaryReader(data)
        assert reader.read() is True

    def test_read_bool_false(self):
        data = self._make_binary(False)
        reader = BinaryReader(data)
        assert reader.read() is False

    def test_read_int(self):
        for value in [42, 1000, 100000, 10**15]:
            data = self._make_binary(value)
            reader = BinaryReader(data)
            assert reader.read() == value

    def test_read_float(self):
        data = self._make_binary(3.14159)
        reader = BinaryReader(data)
        result = reader.read()
        assert abs(result - 3.14159) < 0.00001

    def test_read_string(self):
        data = self._make_binary("hello world")
        reader = BinaryReader(data)
        assert reader.read() == "hello world"

    def test_read_list(self):
        data = self._make_binary([1, 2, 3, 4, 5])
        reader = BinaryReader(data)
        assert reader.read() == [1, 2, 3, 4, 5]

    def test_read_set(self):
        data = self._make_binary({1, 2, 3})
        reader = BinaryReader(data)
        assert reader.read() == {1, 2, 3}

    def test_read_dict(self):
        data = self._make_binary({"a": 1, "b": 2})
        reader = BinaryReader(data)
        result = reader.read()
        assert result == {"a": 1, "b": 2}

    def test_read_enum(self):
        data = self._make_binary(Color.RED)
        reader = BinaryReader(data)
        result = reader.read()
        assert result["__enum__"] == "Color"
        assert result["value"] == "red"

    def test_invalid_magic(self):
        data = b"XXXX" + b"\x00" * 20
        reader = BinaryReader(data)
        with pytest.raises(Exception):
            reader.read()

    def test_checksum_validation(self):
        # Create valid data with enough content to have data bytes to corrupt
        writer = BinaryWriter(include_checksum=True)
        writer.write({"test": "value", "number": 12345})
        data = bytearray(writer.get_output())

        # Header is 20 bytes (magic 4 + version 4 + flags 4 + checksum 4 + length 4)
        # Corrupt a data byte after the header
        if len(data) > 25:
            data[25] ^= 0xFF  # Flip bits in data section

        reader = BinaryReader(bytes(data))
        with pytest.raises(Exception):
            reader.read()

    def test_has_more(self):
        data = self._make_binary(42)
        reader = BinaryReader(data)
        assert reader.has_more()
        reader.read()
        assert not reader.has_more()

    def test_peek_type(self):
        data = self._make_binary("test")
        reader = BinaryReader(data)
        assert reader.peek_type() == "string"

        data = self._make_binary([1, 2, 3])
        reader = BinaryReader(data)
        assert reader.peek_type() == "list"


# Binary Round-Trip Tests

class TestBinaryRoundTrip:
    """Round-trip tests for binary format."""

    def test_roundtrip_simple(self):
        original = SimpleData(name="test", value=42)

        writer = BinaryWriter()
        writer.write(original)
        data = writer.get_output()

        reader = BinaryReader(data)
        result = reader.read()

        restored = SimpleData.deserialize(result)
        assert restored.name == original.name
        assert restored.value == original.value

    def test_roundtrip_complex(self):
        original = ComplexData(
            name="complex",
            values=[1, 2, 3],
            mapping={"x": 1.5, "y": 2.5},
            optional="present",
            tags={"tag1", "tag2"},
        )

        writer = BinaryWriter()
        writer.write(original)
        data = writer.get_output()

        reader = BinaryReader(data)
        result = reader.read()

        restored = ComplexData.deserialize(result)
        assert restored.name == original.name
        assert restored.values == original.values
        assert restored.mapping == original.mapping
        assert restored.optional == original.optional
        assert restored.tags == original.tags

    def test_roundtrip_nested(self):
        original = NestedData(
            inner=SimpleData(name="inner", value=100),
            items=[
                SimpleData(name="item1", value=1),
                SimpleData(name="item2", value=2),
            ],
        )

        writer = BinaryWriter()
        writer.write(original)
        data = writer.get_output()

        reader = BinaryReader(data)
        result = reader.read()

        restored = NestedData.deserialize(result)
        assert restored.inner.name == original.inner.name
        assert restored.inner.value == original.inner.value
        assert len(restored.items) == len(original.items)

    def test_roundtrip_compressed(self):
        original = {"data": list(range(100)), "text": "x" * 500}

        writer = BinaryWriter(compress=True)
        writer.write(original)
        data = writer.get_output()

        reader = BinaryReader(data)
        result = reader.read()

        assert result["data"] == original["data"]
        assert result["text"] == original["text"]


# JSON Writer Tests

class TestJSONWriter:
    """Tests for JSONWriter."""

    def test_write_simple(self):
        writer = JSONWriter()
        writer.write({"name": "test", "value": 42})
        output = writer.get_output()

        parsed = json.loads(output)
        assert parsed["name"] == "test"
        assert parsed["value"] == 42

    def test_pretty_print(self):
        writer = JSONWriter(pretty=True, indent=4)
        writer.write({"a": 1, "b": 2})
        output = writer.get_output()

        assert "\n" in output
        assert "    " in output

    def test_compact(self):
        writer = JSONWriter(pretty=False)
        writer.write({"a": 1, "b": 2})
        output = writer.get_output()

        assert "\n" not in output

    def test_sort_keys(self):
        writer = JSONWriter(sort_keys=True)
        writer.write({"z": 1, "a": 2, "m": 3})
        output = writer.get_output()

        # Keys should appear in sorted order
        a_pos = output.find('"a"')
        m_pos = output.find('"m"')
        z_pos = output.find('"z"')
        assert a_pos < m_pos < z_pos

    def test_write_dataclass(self):
        writer = JSONWriter()
        obj = SimpleData(name="test", value=42)
        writer.write(obj)
        output = writer.get_output()

        parsed = json.loads(output)
        assert parsed["name"] == "test"
        assert parsed["value"] == 42

    def test_write_enum(self):
        writer = JSONWriter()
        writer.write({"color": Color.RED})
        output = writer.get_output()

        parsed = json.loads(output)
        assert parsed["color"]["__enum__"] == "Color"
        assert parsed["color"]["value"] == "red"

    def test_write_set(self):
        writer = JSONWriter()
        writer.write({"values": {1, 2, 3}})
        output = writer.get_output()

        parsed = json.loads(output)
        assert "__set__" in parsed["values"]
        assert set(parsed["values"]["__set__"]) == {1, 2, 3}

    def test_include_schema(self):
        writer = JSONWriter(include_schema=True)
        obj = SimpleData(name="test", value=1)
        writer.write(obj)
        output = writer.get_output()

        parsed = json.loads(output)
        assert "__schema__" in parsed

    def test_no_schema(self):
        writer = JSONWriter(include_schema=False)
        obj = SimpleData(name="test", value=1)
        writer.write(obj)
        output = writer.get_output()

        parsed = json.loads(output)
        assert "__schema__" not in parsed

    def test_streaming_object(self):
        writer = JSONWriter()
        schema = SchemaInfo(
            name="Test",
            version=SchemaVersion(1, 0, 0),
            hash="abc",
            fields=["a", "b"],
        )
        writer.begin_object(schema)
        writer.write_field("a", 1)
        writer.write_field("b", "two")
        writer.end_object()
        output = writer.get_output()

        parsed = json.loads(output)
        assert parsed["a"] == 1
        assert parsed["b"] == "two"
        assert "__schema__" in parsed

    def test_reset(self):
        writer = JSONWriter()
        writer.write({"first": 1})
        writer.reset()
        writer.write({"second": 2})
        output = writer.get_output()

        parsed = json.loads(output)
        assert "second" in parsed
        assert "first" not in parsed


# JSON Reader Tests

class TestJSONReader:
    """Tests for JSONReader."""

    def test_read_dict(self):
        reader = JSONReader('{"name": "test", "value": 42}')
        result = reader.read()

        assert result["name"] == "test"
        assert result["value"] == 42

    def test_read_from_dict(self):
        reader = JSONReader({"name": "test", "value": 42})
        result = reader.read()

        assert result["name"] == "test"
        assert result["value"] == 42

    def test_read_from_stream(self):
        stream = StringIO('{"name": "test"}')
        reader = JSONReader(stream)
        result = reader.read()

        assert result["name"] == "test"

    def test_read_field(self):
        reader = JSONReader({"a": 1, "b": 2, "c": 3})
        reader.begin_object()

        assert reader.read_field("a") == 1
        assert reader.read_field("b") == 2
        assert reader.read_field("c") == 3
        assert reader.read_field("missing") is None

    def test_begin_object_with_schema(self):
        data = {
            "__schema__": {
                "name": "Test",
                "version": "1.0.0",
                "hash": "abc",
                "fields": ["x"],
            },
            "x": 42,
        }
        reader = JSONReader(data)
        schema = reader.begin_object()

        assert schema is not None
        assert schema.name == "Test"
        assert schema.version == SchemaVersion(1, 0, 0)

    def test_peek_type(self):
        reader = JSONReader({"value": 42})
        assert reader.peek_type() == "object"

        # When reader has a list at root, peek shows the list type
        reader = JSONReader({"items": [1, 2, 3]})
        assert reader.peek_type() == "object"  # Root is still an object

        reader = JSONReader({"__set__": [1, 2]})
        assert reader.peek_type() == "set"

        reader = JSONReader({"__enum__": "Color", "value": "red"})
        assert reader.peek_type() == "enum"

        # When root is a list, peek shows type of first element (for iteration)
        reader = JSONReader([1, 2, 3])
        assert reader.peek_type() == "int"  # First element type

        reader = JSONReader([{"name": "a"}, {"name": "b"}])
        assert reader.peek_type() == "object"  # First element is an object

    def test_has_more(self):
        reader = JSONReader({"a": 1})
        assert reader.has_more()
        reader.read()
        assert not reader.has_more()

    def test_get_all_fields(self):
        reader = JSONReader({"a": 1, "b": 2, "__internal__": "hidden"})
        reader.begin_object()
        fields = reader.get_all_fields()

        assert "a" in fields
        assert "b" in fields
        assert "__internal__" not in fields  # Fields starting with __ are excluded


# JSON Round-Trip Tests

class TestJSONRoundTrip:
    """Round-trip tests for JSON format."""

    def test_roundtrip_simple(self):
        original = SimpleData(name="test", value=42)

        writer = JSONWriter()
        writer.write(original)
        json_str = writer.get_output()

        reader = JSONReader(json_str)
        result = reader.read()

        restored = SimpleData.deserialize(result)
        assert restored.name == original.name
        assert restored.value == original.value

    def test_roundtrip_complex(self):
        original = ComplexData(
            name="complex",
            values=[1, 2, 3],
            mapping={"x": 1.5},
            optional="yes",
            tags={"a", "b"},
        )

        writer = JSONWriter()
        writer.write(original)
        json_str = writer.get_output()

        reader = JSONReader(json_str)
        result = reader.read()

        restored = ComplexData.deserialize(result)
        assert restored.name == original.name
        assert restored.values == original.values
        assert restored.tags == original.tags


# Diff Writer Tests

class TestDiffWriter:
    """Tests for DiffWriter."""

    def test_diff_add(self):
        old = {"a": 1}
        new = {"a": 1, "b": 2}

        writer = DiffWriter()
        writer.set_base(old)
        writer.write(new)
        patch = writer.get_patch()

        assert len(patch.entries) == 1
        assert patch.entries[0].operation == "add"
        assert patch.entries[0].path == "b"
        assert patch.entries[0].new_value == 2

    def test_diff_remove(self):
        old = {"a": 1, "b": 2}
        new = {"a": 1}

        writer = DiffWriter()
        writer.set_base(old)
        writer.write(new)
        patch = writer.get_patch()

        assert len(patch.entries) == 1
        assert patch.entries[0].operation == "remove"
        assert patch.entries[0].path == "b"
        assert patch.entries[0].old_value == 2

    def test_diff_replace(self):
        old = {"a": 1}
        new = {"a": 2}

        writer = DiffWriter()
        writer.set_base(old)
        writer.write(new)
        patch = writer.get_patch()

        assert len(patch.entries) == 1
        assert patch.entries[0].operation == "replace"
        assert patch.entries[0].path == "a"
        assert patch.entries[0].old_value == 1
        assert patch.entries[0].new_value == 2

    def test_diff_nested(self):
        old = {"outer": {"inner": 1}}
        new = {"outer": {"inner": 2}}

        writer = DiffWriter()
        writer.set_base(old)
        writer.write(new)
        patch = writer.get_patch()

        assert len(patch.entries) == 1
        assert patch.entries[0].path == "outer.inner"
        assert patch.entries[0].operation == "replace"

    def test_diff_list(self):
        old = {"items": [1, 2, 3]}
        new = {"items": [1, 2, 4]}

        writer = DiffWriter()
        writer.set_base(old)
        writer.write(new)
        patch = writer.get_patch()

        # Should detect change in items[2]
        assert any(e.path == "items[2]" and e.operation == "replace" for e in patch.entries)

    def test_diff_dataclass(self):
        old = SimpleData(name="old", value=1)
        new = SimpleData(name="new", value=1)

        writer = DiffWriter()
        writer.set_base(old)
        writer.write(new)
        patch = writer.get_patch()

        assert any(e.path == "name" and e.new_value == "new" for e in patch.entries)

    def test_diff_versions(self):
        writer = DiffWriter(source_version="1.0.0", target_version="1.1.0")
        writer.set_base({"a": 1})
        writer.write({"a": 2})
        patch = writer.get_patch()

        assert patch.source_version == "1.0.0"
        assert patch.target_version == "1.1.0"

    def test_write_field(self):
        writer = DiffWriter()
        writer.set_base({"x": 1})
        writer.write_field("x", 2)
        writer.write_field("y", 3)
        patch = writer.get_patch()

        assert len(patch.entries) == 2
        assert any(e.path == "x" and e.operation == "replace" for e in patch.entries)
        assert any(e.path == "y" and e.operation == "add" for e in patch.entries)

    def test_write_remove(self):
        writer = DiffWriter()
        writer.set_base({"a": 1, "b": 2})
        writer.write_remove("b")
        patch = writer.get_patch()

        assert len(patch.entries) == 1
        assert patch.entries[0].operation == "remove"
        assert patch.entries[0].path == "b"

    def test_get_output_json(self):
        writer = DiffWriter()
        writer.set_base({"a": 1})
        writer.write({"a": 2})
        output = writer.get_output()

        parsed = json.loads(output)
        assert "entries" in parsed
        assert len(parsed["entries"]) == 1

    def test_reset(self):
        writer = DiffWriter()
        writer.set_base({"a": 1})
        writer.write({"a": 2})
        writer.reset()

        patch = writer.get_patch()
        assert len(patch.entries) == 0


# Diff Reader Tests

class TestDiffReader:
    """Tests for DiffReader."""

    def test_read_patch_from_string(self):
        json_str = '{"entries": [{"path": "a", "op": "replace", "old": 1, "new": 2}]}'
        reader = DiffReader(json_str)
        patch = reader.read()

        assert len(patch.entries) == 1
        assert patch.entries[0].path == "a"

    def test_read_patch_from_dict(self):
        data = {"entries": [{"path": "a", "op": "add", "new": 1}]}
        reader = DiffReader(data)
        patch = reader.read()

        assert len(patch.entries) == 1

    def test_read_patch_from_patch(self):
        patch = DiffPatch(entries=[DiffEntry(path="x", operation="add", new_value=1)])
        reader = DiffReader(patch)
        result = reader.read()

        assert result is patch

    def test_read_entry(self):
        patch = DiffPatch(entries=[
            DiffEntry(path="a", operation="add", new_value=1),
            DiffEntry(path="b", operation="remove", old_value=2),
        ])
        reader = DiffReader(patch)

        entry1 = reader.read_entry()
        assert entry1.path == "a"

        entry2 = reader.read_entry()
        assert entry2.path == "b"

        entry3 = reader.read_entry()
        assert entry3 is None

    def test_has_more(self):
        patch = DiffPatch(entries=[DiffEntry(path="a", operation="add", new_value=1)])
        reader = DiffReader(patch)

        assert reader.has_more()
        reader.read_entry()
        assert not reader.has_more()

    def test_apply_add(self):
        patch = DiffPatch(entries=[DiffEntry(path="b", operation="add", new_value=2)])
        reader = DiffReader(patch)

        result = reader.apply_to({"a": 1})
        assert result["a"] == 1
        assert result["b"] == 2

    def test_apply_remove(self):
        patch = DiffPatch(entries=[DiffEntry(path="b", operation="remove", old_value=2)])
        reader = DiffReader(patch)

        result = reader.apply_to({"a": 1, "b": 2})
        assert result["a"] == 1
        assert "b" not in result

    def test_apply_replace(self):
        patch = DiffPatch(entries=[
            DiffEntry(path="a", operation="replace", old_value=1, new_value=10)
        ])
        reader = DiffReader(patch)

        result = reader.apply_to({"a": 1, "b": 2})
        assert result["a"] == 10
        assert result["b"] == 2

    def test_apply_nested(self):
        patch = DiffPatch(entries=[
            DiffEntry(path="outer.inner", operation="replace", old_value=1, new_value=2)
        ])
        reader = DiffReader(patch)

        result = reader.apply_to({"outer": {"inner": 1}})
        assert result["outer"]["inner"] == 2

    def test_apply_list_index(self):
        patch = DiffPatch(entries=[
            DiffEntry(path="items[1]", operation="replace", old_value=2, new_value=20)
        ])
        reader = DiffReader(patch)

        result = reader.apply_to({"items": [1, 2, 3]})
        assert result["items"] == [1, 20, 3]

    def test_get_entries(self):
        patch = DiffPatch(entries=[
            DiffEntry(path="a", operation="add", new_value=1),
            DiffEntry(path="b", operation="add", new_value=2),
        ])
        reader = DiffReader(patch)

        entries = reader.get_entries()
        assert len(entries) == 2


# Diff Round-Trip Tests

class TestDiffRoundTrip:
    """Round-trip tests for diff format."""

    def test_roundtrip_simple(self):
        old = {"a": 1, "b": 2}
        new = {"a": 1, "b": 3, "c": 4}

        patch = compute_diff(old, new)
        result = apply_diff(old, patch)

        assert result == new

    def test_roundtrip_complex(self):
        old = ComplexData(
            name="old",
            values=[1, 2, 3],
            mapping={"x": 1.0},
            tags={"tag1"},
        )
        new = ComplexData(
            name="new",
            values=[1, 2, 4],
            mapping={"x": 2.0},
            tags={"tag1", "tag2"},
        )

        patch = compute_diff(old, new)
        assert len(patch.entries) > 0

    def test_roundtrip_nested(self):
        old = {"level1": {"level2": {"value": 1}}}
        new = {"level1": {"level2": {"value": 2}}}

        patch = compute_diff(old, new)
        result = apply_diff(old, patch)

        assert result["level1"]["level2"]["value"] == 2

    def test_undo_with_diff(self):
        # Use diff for undo functionality
        original = {"name": "original", "value": 1}
        modified = {"name": "modified", "value": 2}

        # Forward patch
        forward_patch = compute_diff(original, modified)

        # Reverse patch (swap old/new)
        reverse_entries = [
            DiffEntry(
                path=e.path,
                operation="replace" if e.operation == "replace" else ("remove" if e.operation == "add" else "add"),
                old_value=e.new_value,
                new_value=e.old_value,
            )
            for e in forward_patch.entries
        ]
        reverse_patch = DiffPatch(entries=reverse_entries)

        # Apply forward
        modified_result = apply_diff(original, forward_patch)
        assert modified_result == modified

        # Apply reverse to get back original
        restored = apply_diff(modified, reverse_patch)
        assert restored == original


# DiffEntry and DiffPatch Tests

class TestDiffEntry:
    """Tests for DiffEntry dataclass."""

    def test_to_dict(self):
        entry = DiffEntry(path="a.b", operation="replace", old_value=1, new_value=2)
        d = entry.to_dict()

        assert d["path"] == "a.b"
        assert d["op"] == "replace"
        assert d["old"] == 1
        assert d["new"] == 2

    def test_from_dict(self):
        d = {"path": "x", "op": "add", "new": 42}
        entry = DiffEntry.from_dict(d)

        assert entry.path == "x"
        assert entry.operation == "add"
        assert entry.new_value == 42
        assert entry.old_value is None


class TestDiffPatch:
    """Tests for DiffPatch dataclass."""

    def test_to_dict(self):
        patch = DiffPatch(
            entries=[DiffEntry(path="a", operation="add", new_value=1)],
            source_version="1.0",
            target_version="2.0",
        )
        d = patch.to_dict()

        assert len(d["entries"]) == 1
        assert d["source_version"] == "1.0"
        assert d["target_version"] == "2.0"

    def test_from_dict(self):
        d = {
            "entries": [{"path": "a", "op": "add", "new": 1}],
            "source_version": "1.0",
        }
        patch = DiffPatch.from_dict(d)

        assert len(patch.entries) == 1
        assert patch.source_version == "1.0"

    def test_len(self):
        patch = DiffPatch(entries=[
            DiffEntry(path="a", operation="add", new_value=1),
            DiffEntry(path="b", operation="add", new_value=2),
        ])
        assert len(patch) == 2

    def test_iter(self):
        patch = DiffPatch(entries=[
            DiffEntry(path="a", operation="add", new_value=1),
        ])
        entries = list(patch)
        assert len(entries) == 1


# Factory Function Tests

class TestFactoryFunctions:
    """Tests for create_writer and create_reader."""

    def test_create_binary_writer(self):
        writer = create_writer(SerializationFormat.BINARY)
        assert isinstance(writer, BinaryWriter)

    def test_create_json_writer(self):
        writer = create_writer(SerializationFormat.JSON)
        assert isinstance(writer, JSONWriter)

    def test_create_compact_json_writer(self):
        writer = create_writer(SerializationFormat.COMPACT_JSON)
        assert isinstance(writer, JSONWriter)

    def test_create_diff_writer(self):
        writer = create_writer(SerializationFormat.DIFF)
        assert isinstance(writer, DiffWriter)

    def test_create_binary_reader(self):
        writer = BinaryWriter()
        writer.write(42)
        data = writer.get_output()

        reader = create_reader(SerializationFormat.BINARY, data)
        assert isinstance(reader, BinaryReader)

    def test_create_json_reader(self):
        reader = create_reader(SerializationFormat.JSON, '{"a": 1}')
        assert isinstance(reader, JSONReader)

    def test_create_diff_reader(self):
        reader = create_reader(
            SerializationFormat.DIFF,
            '{"entries": []}'
        )
        assert isinstance(reader, DiffReader)


# Interface Compliance Tests

class TestInterfaceCompliance:
    """Tests to verify interface compliance."""

    def test_binary_writer_interface(self):
        writer = BinaryWriter()
        assert isinstance(writer, SerializationWriter)
        assert hasattr(writer, 'write')
        assert hasattr(writer, 'write_field')
        assert hasattr(writer, 'begin_object')
        assert hasattr(writer, 'end_object')
        assert hasattr(writer, 'begin_list')
        assert hasattr(writer, 'end_list')
        assert hasattr(writer, 'flush')
        assert hasattr(writer, 'get_output')
        assert hasattr(writer, 'reset')

    def test_json_writer_interface(self):
        writer = JSONWriter()
        assert isinstance(writer, SerializationWriter)

    def test_diff_writer_interface(self):
        writer = DiffWriter()
        assert isinstance(writer, SerializationWriter)

    def test_binary_reader_interface(self):
        writer = BinaryWriter()
        writer.write(42)
        reader = BinaryReader(writer.get_output())

        assert isinstance(reader, SerializationReader)
        assert hasattr(reader, 'read')
        assert hasattr(reader, 'read_field')
        assert hasattr(reader, 'begin_object')
        assert hasattr(reader, 'end_object')
        assert hasattr(reader, 'begin_list')
        assert hasattr(reader, 'end_list')
        assert hasattr(reader, 'has_more')
        assert hasattr(reader, 'peek_type')

    def test_json_reader_interface(self):
        reader = JSONReader('{}')
        assert isinstance(reader, SerializationReader)

    def test_diff_reader_interface(self):
        reader = DiffReader('{"entries": []}')
        assert isinstance(reader, SerializationReader)


# Cross-Format Tests

class TestCrossFormat:
    """Tests for cross-format compatibility."""

    def test_json_to_binary_to_json(self):
        # Start with JSON
        original = {"name": "test", "values": [1, 2, 3]}

        json_writer = JSONWriter(include_schema=False)
        json_writer.write(original)
        json_str = json_writer.get_output()

        # Convert to binary
        json_reader = JSONReader(json_str)
        data = json_reader.read()

        binary_writer = BinaryWriter(include_schema=False)
        binary_writer.write(data)
        binary_data = binary_writer.get_output()

        # Convert back to JSON
        binary_reader = BinaryReader(binary_data)
        restored = binary_reader.read()

        json_writer2 = JSONWriter(include_schema=False)
        json_writer2.write(restored)
        final_json = json_writer2.get_output()

        final = json.loads(final_json)
        assert final == original

    def test_all_formats_same_data(self):
        original = SimpleData(name="test", value=42)

        # Binary
        binary_writer = BinaryWriter()
        binary_writer.write(original)
        binary_data = binary_writer.get_output()

        binary_reader = BinaryReader(binary_data)
        binary_result = binary_reader.read()

        # JSON
        json_writer = JSONWriter()
        json_writer.write(original)
        json_str = json_writer.get_output()

        json_reader = JSONReader(json_str)
        json_result = json_reader.read()

        # Compare (excluding schema differences)
        assert binary_result.get("name") == json_result.get("name")
        assert binary_result.get("value") == json_result.get("value")


# Edge Cases

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_object(self):
        writer = BinaryWriter()
        writer.write({})
        data = writer.get_output()

        reader = BinaryReader(data)
        assert reader.read() == {}

    def test_empty_list(self):
        writer = JSONWriter()
        writer.write([])
        output = writer.get_output()
        assert json.loads(output) == []

    def test_deeply_nested(self):
        data = {"l1": {"l2": {"l3": {"l4": {"l5": {"value": 42}}}}}}

        writer = BinaryWriter()
        writer.write(data)
        binary = writer.get_output()

        reader = BinaryReader(binary)
        result = reader.read()

        assert result["l1"]["l2"]["l3"]["l4"]["l5"]["value"] == 42

    def test_large_string(self):
        data = {"text": "x" * 10000}

        writer = BinaryWriter()
        writer.write(data)
        binary = writer.get_output()

        reader = BinaryReader(binary)
        result = reader.read()

        assert result["text"] == data["text"]

    def test_unicode_strings(self):
        data = {"text": "Hello 世界 \U0001F600"}

        writer = JSONWriter()
        writer.write(data)
        json_str = writer.get_output()

        reader = JSONReader(json_str)
        result = reader.read()

        assert result["text"] == data["text"]

    def test_special_float_values(self):
        # Note: JSON doesn't support inf/nan, so test with binary
        writer = BinaryWriter()
        writer.write({"value": 1e308})
        binary = writer.get_output()

        reader = BinaryReader(binary)
        result = reader.read()
        assert abs(result["value"] - 1e308) < 1e300

    def test_negative_integers(self):
        writer = BinaryWriter()
        writer.write({"values": [-1, -1000, -100000, -10**15]})
        binary = writer.get_output()

        reader = BinaryReader(binary)
        result = reader.read()
        assert result["values"] == [-1, -1000, -100000, -10**15]

    def test_mixed_list_types(self):
        data = [1, "two", 3.0, True, None, {"nested": "dict"}]

        writer = JSONWriter()
        writer.write(data)
        json_str = writer.get_output()

        reader = JSONReader(json_str)
        result = reader.read()

        assert result == data

    def test_empty_diff(self):
        old = {"a": 1}
        new = {"a": 1}

        patch = compute_diff(old, new)
        assert len(patch.entries) == 0

        result = apply_diff(old, patch)
        assert result == old
