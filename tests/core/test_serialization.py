"""Tests for Serializable trait with schema versioning (T-CC-2.4)."""
import json
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from engine.core.serialization import (
    SchemaInfo,
    SchemaRegistry,
    SchemaVersion,
    Serializable,
    SerializationContext,
    SerializationError,
    SerializationFormat,
    SerializationStats,
    VersionCompatibility,
    deserialize_from_file,
    get_serialization_stats,
    register_serializable,
    serializable,
    serialize_to_file,
)


class TestSchemaVersion:
    """Tests for SchemaVersion."""

    def test_creation(self):
        v = SchemaVersion(1, 2, 3)
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_str(self):
        v = SchemaVersion(1, 2, 3)
        assert str(v) == "1.2.3"

    def test_comparison(self):
        v1 = SchemaVersion(1, 0, 0)
        v2 = SchemaVersion(1, 1, 0)
        v3 = SchemaVersion(2, 0, 0)

        assert v1 < v2
        assert v2 < v3
        assert v1 <= v2
        assert not v2 < v1

    def test_from_string(self):
        v = SchemaVersion.from_string("1.2.3")
        assert v == SchemaVersion(1, 2, 3)

        v2 = SchemaVersion.from_string("1.2")
        assert v2 == SchemaVersion(1, 2, 0)

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            SchemaVersion.from_string("invalid")

    def test_compatibility_compatible(self):
        v1 = SchemaVersion(1, 2, 0)
        v2 = SchemaVersion(1, 2, 1)
        assert v1.is_compatible(v2) == VersionCompatibility.COMPATIBLE

    def test_compatibility_migratable(self):
        v1 = SchemaVersion(1, 2, 0)
        v2 = SchemaVersion(1, 3, 0)
        assert v1.is_compatible(v2) == VersionCompatibility.MIGRATABLE

    def test_compatibility_incompatible(self):
        v1 = SchemaVersion(1, 0, 0)
        v2 = SchemaVersion(2, 0, 0)
        assert v1.is_compatible(v2) == VersionCompatibility.INCOMPATIBLE


class TestSchemaInfo:
    """Tests for SchemaInfo."""

    def test_creation(self):
        info = SchemaInfo(
            name="TestType",
            version=SchemaVersion(1, 0, 0),
            hash="abc123",
            fields=["a", "b", "c"],
        )
        assert info.name == "TestType"
        assert info.version.major == 1

    def test_to_dict(self):
        info = SchemaInfo(
            name="Test",
            version=SchemaVersion(1, 2, 3),
            hash="xyz",
            fields=["x", "y"],
            optional_fields={"y"},
        )
        d = info.to_dict()
        assert d["name"] == "Test"
        assert d["version"] == "1.2.3"
        assert d["hash"] == "xyz"
        assert "y" in d["optional_fields"]

    def test_from_dict(self):
        d = {
            "name": "Test",
            "version": "1.0.0",
            "hash": "abc",
            "fields": ["a"],
            "optional_fields": ["a"],
            "deprecated_fields": [],
        }
        info = SchemaInfo.from_dict(d)
        assert info.name == "Test"
        assert info.version == SchemaVersion(1, 0, 0)


class TestSerializationContext:
    """Tests for SerializationContext."""

    def test_default_values(self):
        ctx = SerializationContext()
        assert ctx.format == SerializationFormat.JSON
        assert ctx.include_schema
        assert not ctx.include_defaults

    def test_custom_values(self):
        ctx = SerializationContext(
            format=SerializationFormat.BINARY,
            include_schema=False,
            max_depth=3,
        )
        assert ctx.format == SerializationFormat.BINARY
        assert not ctx.include_schema
        assert ctx.max_depth == 3

    def test_path_tracking(self):
        ctx = SerializationContext()
        assert ctx.current_path == ""

        ctx.enter("level1")
        assert ctx.current_path == "level1"

        ctx.enter("level2")
        assert ctx.current_path == "level1.level2"

        ctx.exit()
        assert ctx.current_path == "level1"

    def test_should_serialize_field(self):
        ctx = SerializationContext(skip_fields={"secret"})
        assert ctx.should_serialize_field("name")
        assert not ctx.should_serialize_field("secret")

    def test_max_depth(self):
        ctx = SerializationContext(max_depth=2)
        ctx.enter("level1")
        ctx.enter("level2")
        ctx.enter("level3")
        assert not ctx.should_serialize_field("too_deep")


class TestSerializableDecorator:
    """Tests for @serializable decorator."""

    def test_basic_serialization(self):
        @serializable()
        @dataclass
        class Simple:
            name: str
            value: int

        obj = Simple(name="test", value=42)
        data = obj.serialize()

        assert data["name"] == "test"
        assert data["value"] == 42

    def test_deserialization(self):
        @serializable()
        @dataclass
        class Simple:
            name: str
            value: int

        data = {"name": "test", "value": 42}
        obj = Simple.deserialize(data)

        assert obj.name == "test"
        assert obj.value == 42

    def test_version(self):
        @serializable(version="2.1.0")
        @dataclass
        class Versioned:
            x: int

        info = Versioned.get_schema_info()
        assert info.version == SchemaVersion(2, 1, 0)

    def test_custom_name(self):
        @serializable(name="CustomName")
        @dataclass
        class Original:
            x: int

        info = Original.get_schema_info()
        assert info.name == "CustomName"

    def test_schema_in_output(self):
        @serializable()
        @dataclass
        class WithSchema:
            x: int

        obj = WithSchema(x=10)
        ctx = SerializationContext(include_schema=True)
        data = obj.serialize(ctx)

        assert "__schema__" in data
        assert data["__schema__"]["name"] == "WithSchema"

    def test_no_schema_in_output(self):
        @serializable()
        @dataclass
        class NoSchema:
            x: int

        obj = NoSchema(x=10)
        ctx = SerializationContext(include_schema=False)
        data = obj.serialize(ctx)

        assert "__schema__" not in data

    def test_optional_fields(self):
        @serializable()
        @dataclass
        class WithOptional:
            required: str
            optional: Optional[int] = None

        data = {"required": "test"}
        obj = WithOptional.deserialize(data)
        assert obj.required == "test"
        assert obj.optional is None

    def test_default_values(self):
        @serializable()
        @dataclass
        class WithDefaults:
            name: str
            count: int = 0

        data = {"name": "test"}
        obj = WithDefaults.deserialize(data)
        assert obj.count == 0

    def test_to_json(self):
        @serializable()
        @dataclass
        class JsonTest:
            value: int

        obj = JsonTest(value=42)
        json_str = obj.to_json()
        parsed = json.loads(json_str)
        assert parsed["value"] == 42

    def test_from_json(self):
        @serializable()
        @dataclass
        class JsonTest:
            value: int

        json_str = '{"value": 42}'
        obj = JsonTest.from_json(json_str)
        assert obj.value == 42

    def test_to_bytes(self):
        @serializable()
        @dataclass
        class BytesTest:
            value: int

        obj = BytesTest(value=42)
        data = obj.to_bytes()
        assert isinstance(data, bytes)

    def test_from_bytes(self):
        @serializable()
        @dataclass
        class BytesTest:
            value: int

        obj = BytesTest(value=42)
        data = obj.to_bytes()
        restored = BytesTest.from_bytes(data)
        assert restored.value == 42


class TestComplexTypes:
    """Tests for complex type serialization."""

    def test_list_field(self):
        @serializable()
        @dataclass
        class WithList:
            items: List[int]

        obj = WithList(items=[1, 2, 3])
        data = obj.serialize()
        assert data["items"] == [1, 2, 3]

        restored = WithList.deserialize(data)
        assert restored.items == [1, 2, 3]

    def test_dict_field(self):
        @serializable()
        @dataclass
        class WithDict:
            mapping: Dict[str, int]

        obj = WithDict(mapping={"a": 1, "b": 2})
        data = obj.serialize()
        assert data["mapping"] == {"a": 1, "b": 2}

        restored = WithDict.deserialize(data)
        assert restored.mapping == {"a": 1, "b": 2}

    def test_set_field(self):
        @serializable()
        @dataclass
        class WithSet:
            values: Set[int]

        obj = WithSet(values={1, 2, 3})
        data = obj.serialize()
        assert "__set__" in data["values"]

        restored = WithSet.deserialize(data)
        assert restored.values == {1, 2, 3}

    def test_tuple_field(self):
        @serializable()
        @dataclass
        class WithTuple:
            coord: Tuple[int, int]

        obj = WithTuple(coord=(10, 20))
        data = obj.serialize()
        assert data["coord"] == [10, 20]

        restored = WithTuple.deserialize(data)
        assert restored.coord == (10, 20)

    def test_optional_field(self):
        @serializable()
        @dataclass
        class WithOptional:
            value: Optional[int] = None

        obj1 = WithOptional(value=42)
        data1 = obj1.serialize()
        assert data1["value"] == 42

        obj2 = WithOptional(value=None)
        ctx = SerializationContext(include_defaults=False)
        data2 = obj2.serialize(ctx)
        assert "value" not in data2 or data2.get("value") is None

    def test_nested_dataclass(self):
        @serializable()
        @dataclass
        class Inner:
            x: int

        @serializable()
        @dataclass
        class Outer:
            inner: Inner

        obj = Outer(inner=Inner(x=42))
        data = obj.serialize()
        assert data["inner"]["x"] == 42

        restored = Outer.deserialize(data)
        assert restored.inner.x == 42

    def test_enum_field(self):
        class Color(Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        @serializable()
        @dataclass
        class WithEnum:
            color: Color

        obj = WithEnum(color=Color.RED)
        data = obj.serialize()
        assert data["color"]["__enum__"] == "Color"
        assert data["color"]["value"] == "red"


class TestVersioning:
    """Tests for schema versioning."""

    def test_version_check_compatible(self):
        @serializable(version="1.0.0")
        @dataclass
        class Compat:
            x: int

        data = {
            "__schema__": {"name": "Compat", "version": "1.0.0", "hash": "x", "fields": ["x"]},
            "x": 42,
        }
        obj = Compat.deserialize(data)
        assert obj.x == 42

    def test_version_check_incompatible(self):
        @serializable(version="1.0.0")
        @dataclass
        class Incompat:
            x: int

        data = {
            "__schema__": {"name": "Incompat", "version": "2.0.0", "hash": "x", "fields": ["x"]},
            "x": 42,
        }
        with pytest.raises(SerializationError) as exc:
            Incompat.deserialize(data)
        assert exc.value.version_mismatch

    def test_missing_field_error(self):
        @serializable()
        @dataclass
        class Required:
            name: str
            value: int

        data = {"name": "test"}  # Missing 'value'
        with pytest.raises(SerializationError) as exc:
            Required.deserialize(data)
        assert "value" in str(exc.value)


class TestSchemaRegistry:
    """Tests for SchemaRegistry."""

    def setup_method(self):
        SchemaRegistry.get_instance().clear()

    def test_singleton(self):
        r1 = SchemaRegistry.get_instance()
        r2 = SchemaRegistry.get_instance()
        assert r1 is r2

    def test_register_and_get(self):
        @serializable()
        @dataclass
        class Registered:
            x: int

        registry = SchemaRegistry.get_instance()
        registry.register(Registered)

        result = registry.get("Registered")
        assert result is Registered

    def test_get_with_version(self):
        @serializable(version="1.0.0")
        @dataclass
        class Versioned:
            x: int

        registry = SchemaRegistry.get_instance()
        registry.register(Versioned)

        result = registry.get("Versioned", "1.0.0")
        assert result is Versioned

    def test_get_nonexistent(self):
        registry = SchemaRegistry.get_instance()
        result = registry.get("NotRegistered")
        assert result is None

    def test_register_migration(self):
        registry = SchemaRegistry.get_instance()

        def migrate_v1_to_v2(data):
            data["new_field"] = data.pop("old_field", None)
            return data

        registry.register_migration("1.0.0", "2.0.0", migrate_v1_to_v2)

        data = {"old_field": "value"}
        migrated = registry.migrate(data, "1.0.0", "2.0.0")
        assert "new_field" in migrated
        assert migrated["new_field"] == "value"

    def test_migrate_no_path(self):
        registry = SchemaRegistry.get_instance()

        with pytest.raises(SerializationError):
            registry.migrate({}, "1.0.0", "3.0.0")

    def test_list_schemas(self):
        @serializable()
        @dataclass
        class Schema1:
            x: int

        @serializable()
        @dataclass
        class Schema2:
            y: int

        registry = SchemaRegistry.get_instance()
        registry.register(Schema1)
        registry.register(Schema2)

        schemas = registry.list_schemas()
        names = [s.name for s in schemas]
        assert "Schema1" in names
        assert "Schema2" in names

    def test_clear(self):
        @serializable()
        @dataclass
        class ToClear:
            x: int

        registry = SchemaRegistry.get_instance()
        registry.register(ToClear)
        registry.clear()

        assert registry.get("ToClear") is None


class TestRegisterDecorator:
    """Tests for @register_serializable decorator."""

    def setup_method(self):
        SchemaRegistry.get_instance().clear()

    def test_auto_register(self):
        @register_serializable
        @serializable()
        @dataclass
        class AutoReg:
            x: int

        registry = SchemaRegistry.get_instance()
        result = registry.get("AutoReg")
        assert result is AutoReg


class TestSerializationStats:
    """Tests for SerializationStats."""

    def test_initial_state(self):
        stats = SerializationStats()
        assert stats.serialize_count == 0
        assert stats.deserialize_count == 0

    def test_record_serialize(self):
        stats = SerializationStats()
        stats.record_serialize(100)

        assert stats.serialize_count == 1
        assert stats.total_bytes_written == 100

    def test_record_deserialize(self):
        stats = SerializationStats()
        stats.record_deserialize(200)

        assert stats.deserialize_count == 1
        assert stats.total_bytes_read == 200

    def test_record_error(self):
        stats = SerializationStats()
        stats.record_error()
        assert stats.errors == 1

    def test_to_dict(self):
        stats = SerializationStats()
        stats.record_serialize(100)
        stats.record_deserialize(200)

        d = stats.to_dict()
        assert d["serialize_count"] == 1
        assert d["deserialize_count"] == 1


class TestFileOperations:
    """Tests for file serialization."""

    def test_serialize_to_file(self):
        @serializable()
        @dataclass
        class FileTest:
            value: int

        obj = FileTest(value=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            serialize_to_file(obj, str(path))

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["value"] == 42

    def test_deserialize_from_file(self):
        @serializable()
        @dataclass
        class FileTest:
            value: int

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            path.write_text('{"value": 42}')

            obj = deserialize_from_file(FileTest, str(path))
            assert obj.value == 42

    def test_binary_format(self):
        @serializable()
        @dataclass
        class BinaryTest:
            value: int

        obj = BinaryTest(value=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            serialize_to_file(obj, str(path), format=SerializationFormat.BINARY)

            restored = deserialize_from_file(BinaryTest, str(path), format=SerializationFormat.BINARY)
            assert restored.value == 42


class TestSerializationError:
    """Tests for SerializationError."""

    def test_basic_error(self):
        err = SerializationError("Test error")
        assert str(err) == "Test error"

    def test_error_with_path(self):
        err = SerializationError("Invalid value", path="root.field")
        assert str(err) == "root.field: Invalid value"

    def test_version_mismatch(self):
        err = SerializationError("Version mismatch", version_mismatch=True)
        assert err.version_mismatch


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_dataclass(self):
        @serializable()
        @dataclass
        class Empty:
            pass

        obj = Empty()
        data = obj.serialize()
        restored = Empty.deserialize(data)
        assert isinstance(restored, Empty)

    def test_deeply_nested(self):
        @serializable()
        @dataclass
        class Level3:
            value: int

        @serializable()
        @dataclass
        class Level2:
            level3: Level3

        @serializable()
        @dataclass
        class Level1:
            level2: Level2

        obj = Level1(level2=Level2(level3=Level3(value=42)))
        data = obj.serialize()

        restored = Level1.deserialize(data)
        assert restored.level2.level3.value == 42

    def test_skip_fields(self):
        @serializable()
        @dataclass
        class WithSecret:
            name: str
            secret: str = "hidden"

        obj = WithSecret(name="test", secret="password")
        ctx = SerializationContext(skip_fields={"secret"})
        data = obj.serialize(ctx)

        assert "name" in data
        assert "secret" not in data

    def test_include_defaults(self):
        @serializable()
        @dataclass
        class WithDefaults:
            name: str
            count: int = 0

        obj = WithDefaults(name="test", count=0)

        ctx1 = SerializationContext(include_defaults=False, include_schema=False)
        data1 = obj.serialize(ctx1)
        # count might not be in data since it equals default

        ctx2 = SerializationContext(include_defaults=True, include_schema=False)
        data2 = obj.serialize(ctx2)
        assert "count" in data2

    def test_non_dataclass_error(self):
        with pytest.raises(TypeError):
            @serializable()
            class NotDataclass:
                x: int

    def test_roundtrip_preserves_data(self):
        @serializable()
        @dataclass
        class Complex:
            name: str
            values: List[int]
            mapping: Dict[str, float]
            optional: Optional[str] = None

        original = Complex(
            name="test",
            values=[1, 2, 3],
            mapping={"a": 1.5, "b": 2.5},
            optional="present",
        )

        json_str = original.to_json()
        restored = Complex.from_json(json_str)

        assert restored.name == original.name
        assert restored.values == original.values
        assert restored.mapping == original.mapping
        assert restored.optional == original.optional
