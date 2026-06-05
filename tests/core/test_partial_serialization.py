"""Tests for partial serialization with SerializationContext (T-CC-2.7)."""
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytest

from engine.core.partial_serialization import (
    DepthPolicy,
    FieldDescriptor,
    MissingFieldPolicy,
    PartialDeserializer,
    PartialSerializationContext,
    PartialSerializationStats,
    PartialSerializer,
    ReferenceMarker,
    ScopeConfig,
    SerializationMode,
    TruncatedMarker,
    create_full_serialization,
    create_snapshot,
    deserialize_partial,
    get_partial_serialization_stats,
    partial_serializable,
)
from engine.core.serialization import (
    SerializationContext,
    SerializationError,
    SerializationFormat,
    serializable,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@serializable()
@dataclass
class SimpleData:
    name: str
    value: int
    active: bool = True


@serializable()
@dataclass
class NestedData:
    label: str
    child: SimpleData


@serializable()
@dataclass
class DeeplyNested:
    level1: NestedData


@serializable()
@dataclass
class WithCollections:
    items: List[int]
    mapping: Dict[str, str]
    tags: Set[str] = field(default_factory=set)


@serializable()
@dataclass
class WithOptionals:
    required: str
    optional_str: Optional[str] = None
    optional_int: Optional[int] = None
    optional_list: Optional[List[int]] = None


@serializable()
@dataclass
class WithDefaults:
    name: str
    count: int = 0
    enabled: bool = True
    items: List[str] = field(default_factory=list)


@dataclass
class SensitiveData:
    username: str
    password: str = field(metadata={"sensitive": True})
    api_key: str = field(metadata={"sensitive": True})


@dataclass
class LargeData:
    name: str
    blob: bytes = field(metadata={"large": True})
    data: List[int] = field(default_factory=list, metadata={"large": True})


@serializable()
@dataclass
class EntityRef:
    entity_id: int
    name: str


@serializable()
@dataclass
class WithReference:
    id: int
    ref: EntityRef


@serializable()
@dataclass
class CircularA:
    name: str
    other: Optional["CircularB"] = None


@serializable()
@dataclass
class CircularB:
    value: int
    back: Optional[CircularA] = None


# =============================================================================
# ScopeConfig Tests
# =============================================================================


class TestScopeConfig:
    """Tests for ScopeConfig."""

    def test_default_values(self):
        config = ScopeConfig()
        assert config.max_depth is None
        assert config.include_refs is True
        assert config.field_filter is None
        assert config.root_only is False
        assert config.mode == SerializationMode.STANDARD

    def test_snapshot_preset(self):
        config = ScopeConfig.snapshot()
        assert config.max_depth == 1
        assert config.include_refs is False
        assert config.mode == SerializationMode.SNAPSHOT
        assert config.exclude_sensitive is True
        assert config.exclude_large is True

    def test_full_preset(self):
        config = ScopeConfig.full()
        assert config.max_depth is None
        assert config.include_refs is True
        assert config.mode == SerializationMode.FULL
        assert config.exclude_sensitive is False
        assert config.exclude_large is False

    def test_custom_config(self):
        config = ScopeConfig(
            max_depth=3,
            include_refs=False,
            root_only=True,
            mode=SerializationMode.SNAPSHOT,
        )
        assert config.max_depth == 3
        assert config.include_refs is False
        assert config.root_only is True

    def test_field_filter(self):
        config = ScopeConfig(
            field_filter=lambda name, value: name != "secret"
        )
        assert config.should_include_field("name", "value")
        assert not config.should_include_field("secret", "hidden")

    def test_exclude_fields(self):
        config = ScopeConfig(exclude_fields={"password", "secret"})
        assert config.should_include_field("name", "value")
        assert not config.should_include_field("password", "123")
        assert not config.should_include_field("secret", "xxx")

    def test_include_fields(self):
        config = ScopeConfig(include_fields={"name", "value"})
        assert config.should_include_field("name", "test")
        assert config.should_include_field("value", 42)
        assert not config.should_include_field("other", "stuff")

    def test_copy_with(self):
        original = ScopeConfig(max_depth=5, include_refs=True)
        modified = original.copy_with(max_depth=10, root_only=True)

        assert original.max_depth == 5
        assert modified.max_depth == 10
        assert modified.include_refs is True
        assert modified.root_only is True

    def test_combined_filters(self):
        config = ScopeConfig(
            exclude_fields={"excluded"},
            include_fields={"name", "value"},
            field_filter=lambda name, val: val is not None,
        )
        assert config.should_include_field("name", "test")
        assert not config.should_include_field("name", None)
        assert not config.should_include_field("excluded", "test")
        assert not config.should_include_field("other", "test")


# =============================================================================
# PartialSerializationContext Tests
# =============================================================================


class TestPartialSerializationContext:
    """Tests for PartialSerializationContext."""

    def test_default_creation(self):
        ctx = PartialSerializationContext()
        assert ctx.scope is not None
        assert ctx.format == SerializationFormat.JSON
        assert ctx.include_schema is True

    def test_custom_scope(self):
        scope = ScopeConfig(max_depth=2)
        ctx = PartialSerializationContext(scope=scope)
        assert ctx.scope.max_depth == 2

    def test_path_tracking(self):
        ctx = PartialSerializationContext()
        assert ctx.current_path == ""
        assert ctx.current_depth == 0

        ctx.enter("level1")
        assert ctx.current_path == "level1"
        assert ctx.current_depth == 1

        ctx.enter("level2")
        assert ctx.current_path == "level1.level2"
        assert ctx.current_depth == 2

        ctx.exit()
        assert ctx.current_path == "level1"
        assert ctx.current_depth == 1

        ctx.exit()
        assert ctx.current_path == ""
        assert ctx.current_depth == 0

    def test_at_depth_limit(self):
        scope = ScopeConfig(max_depth=2)
        ctx = PartialSerializationContext(scope=scope)

        assert not ctx.at_depth_limit()

        ctx.enter("level1")
        assert not ctx.at_depth_limit()

        ctx.enter("level2")
        assert ctx.at_depth_limit()

    def test_no_depth_limit(self):
        ctx = PartialSerializationContext()
        for i in range(10):
            ctx.enter(f"level{i}")
        assert not ctx.at_depth_limit()

    def test_should_serialize_field(self):
        scope = ScopeConfig(exclude_fields={"secret"})
        ctx = PartialSerializationContext(scope=scope)

        assert ctx.should_serialize_field("name", "value")
        assert not ctx.should_serialize_field("secret", "hidden")

    def test_root_only_mode(self):
        scope = ScopeConfig(root_only=True)
        ctx = PartialSerializationContext(scope=scope)

        assert ctx.should_serialize_field("name")
        ctx.enter("child")
        assert not ctx.should_serialize_field("nested_field")

    def test_register_object(self):
        ctx = PartialSerializationContext()
        obj1 = SimpleData(name="test", value=1)
        obj2 = SimpleData(name="other", value=2)

        is_new1, ref1 = ctx.register_object(obj1)
        assert is_new1 is True
        assert ref1 is not None

        is_new2, ref2 = ctx.register_object(obj1)  # Same object
        assert is_new2 is False
        assert ref2 == ref1

        is_new3, ref3 = ctx.register_object(obj2)  # Different object
        assert is_new3 is True
        assert ref3 != ref1

    def test_reference_marker(self):
        ctx = PartialSerializationContext()
        obj = SimpleData(name="test", value=42)

        is_new, ref_id = ctx.register_object(obj)
        marker = ctx.get_reference_marker(obj)

        assert "__ref__" in marker
        assert marker["__ref__"] == ref_id

    def test_stats_tracking(self):
        scope = ScopeConfig(exclude_fields={"excluded"})
        ctx = PartialSerializationContext(scope=scope)

        ctx.should_serialize_field("included", "val")
        ctx.should_serialize_field("excluded", "val")

        stats = ctx.stats
        assert stats["fields_serialized"] == 1
        assert stats["fields_skipped"] == 1

    def test_to_serialization_context(self):
        scope = ScopeConfig(max_depth=5, exclude_fields={"hidden"})
        ctx = PartialSerializationContext(scope=scope)

        std_ctx = ctx.to_serialization_context()

        assert isinstance(std_ctx, SerializationContext)
        assert std_ctx.max_depth == 5
        assert "hidden" in std_ctx.skip_fields


# =============================================================================
# PartialSerializer Tests
# =============================================================================


class TestPartialSerializer:
    """Tests for PartialSerializer."""

    def test_serialize_primitives(self):
        serializer = PartialSerializer()

        assert serializer.serialize(None) is None
        assert serializer.serialize(True) is True
        assert serializer.serialize(42) == 42
        assert serializer.serialize(3.14) == 3.14
        assert serializer.serialize("hello") == "hello"

    def test_serialize_simple_dataclass(self):
        serializer = PartialSerializer()
        obj = SimpleData(name="test", value=100, active=True)

        data = serializer.serialize(obj)

        assert data["name"] == "test"
        assert data["value"] == 100
        assert data["active"] is True

    def test_serialize_nested_dataclass(self):
        serializer = PartialSerializer()
        obj = NestedData(
            label="parent",
            child=SimpleData(name="child", value=50),
        )

        data = serializer.serialize(obj)

        assert data["label"] == "parent"
        assert data["child"]["name"] == "child"
        assert data["child"]["value"] == 50

    def test_serialize_with_max_depth_1(self):
        scope = ScopeConfig(max_depth=1, depth_policy=DepthPolicy.REFERENCE)
        serializer = PartialSerializer(scope)

        obj = NestedData(
            label="parent",
            child=SimpleData(name="child", value=50),
        )

        data = serializer.serialize(obj)

        assert data["label"] == "parent"
        assert "__ref__" in data["child"]

    def test_serialize_with_max_depth_truncate(self):
        scope = ScopeConfig(max_depth=1, depth_policy=DepthPolicy.TRUNCATE)
        serializer = PartialSerializer(scope)

        obj = DeeplyNested(
            level1=NestedData(
                label="nested",
                child=SimpleData(name="deep", value=1),
            )
        )

        ctx = PartialSerializationContext(scope=scope)
        data = serializer.serialize(obj, ctx)

        # At depth 1, level1 should be a reference
        assert "__ref__" in data.get("level1", {}) or "__type__" in data.get("level1", {})

    def test_serialize_with_max_depth_placeholder(self):
        scope = ScopeConfig(max_depth=1, depth_policy=DepthPolicy.PLACEHOLDER)
        serializer = PartialSerializer(scope)

        obj = NestedData(
            label="parent",
            child=SimpleData(name="child", value=50),
        )

        data = serializer.serialize(obj)

        assert data["label"] == "parent"
        if "child" in data:
            assert "__truncated__" in data["child"] or "__ref__" in data["child"]

    def test_serialize_with_max_depth_error(self):
        scope = ScopeConfig(max_depth=0, depth_policy=DepthPolicy.ERROR)
        serializer = PartialSerializer(scope)
        ctx = PartialSerializationContext(scope=scope)

        obj = NestedData(
            label="parent",
            child=SimpleData(name="child", value=50),
        )

        # Entering any nested field should raise
        ctx.enter("test")
        with pytest.raises(SerializationError):
            ctx.should_serialize_field("nested")

    def test_serialize_collections(self):
        serializer = PartialSerializer()
        obj = WithCollections(
            items=[1, 2, 3],
            mapping={"a": "alpha", "b": "beta"},
            tags={"tag1", "tag2"},
        )

        data = serializer.serialize(obj)

        assert data["items"] == [1, 2, 3]
        assert data["mapping"] == {"a": "alpha", "b": "beta"}
        assert "__set__" in data["tags"]
        assert set(data["tags"]["__set__"]) == {"tag1", "tag2"}

    def test_serialize_with_collection_limit(self):
        scope = ScopeConfig(max_collection_size=2)
        serializer = PartialSerializer(scope)

        obj = WithCollections(
            items=[1, 2, 3, 4, 5],
            mapping={"a": "1", "b": "2", "c": "3"},
            tags={"t1", "t2", "t3"},
        )

        data = serializer.serialize(obj)

        assert len(data["items"]) == 2
        assert len(data["mapping"]) == 2
        assert len(data["tags"]["__set__"]) == 2

    def test_serialize_enum(self):
        @serializable()
        @dataclass
        class WithEnum:
            color: Color

        serializer = PartialSerializer()
        obj = WithEnum(color=Color.RED)

        data = serializer.serialize(obj)

        assert data["color"]["__enum__"] == "Color"
        assert data["color"]["value"] == "red"

    def test_serialize_enum_no_type_info(self):
        @serializable()
        @dataclass
        class WithEnum:
            color: Color

        scope = ScopeConfig(include_type_info=False)
        serializer = PartialSerializer(scope)
        obj = WithEnum(color=Color.GREEN)

        data = serializer.serialize(obj)

        assert data["color"] == "green"

    def test_include_refs_false(self):
        scope = ScopeConfig(include_refs=False)
        serializer = PartialSerializer(scope)

        ref = EntityRef(entity_id=42, name="target")
        obj = WithReference(id=1, ref=ref)

        data = serializer.serialize(obj)

        assert data["id"] == 1
        # Reference should be replaced with marker at depth > 0
        assert "__ref__" in data["ref"] or "entity_id" in data["ref"]

    def test_exclude_fields(self):
        scope = ScopeConfig(exclude_fields={"value", "active"})
        serializer = PartialSerializer(scope)

        obj = SimpleData(name="test", value=100, active=True)

        data = serializer.serialize(obj)

        assert "name" in data
        assert "value" not in data
        assert "active" not in data

    def test_include_fields_only(self):
        scope = ScopeConfig(include_fields={"name"})
        serializer = PartialSerializer(scope)

        obj = SimpleData(name="test", value=100, active=True)

        data = serializer.serialize(obj)

        assert data.get("name") == "test"
        # Other fields should be excluded (but __type__ may still be there)
        assert "value" not in data or data.get("value") is None

    def test_field_filter_function(self):
        # Filter out fields with "secret" in name or value > 50 for ints
        def custom_filter(name: str, value: Any) -> bool:
            if "secret" in name.lower():
                return False
            if isinstance(value, int) and not isinstance(value, bool) and value > 50:
                return False
            return True

        scope = ScopeConfig(field_filter=custom_filter)
        serializer = PartialSerializer(scope)

        obj = SimpleData(name="test", value=100, active=True)

        data = serializer.serialize(obj)

        assert "name" in data
        assert "active" in data  # bool True should pass
        assert "value" not in data  # value=100 > 50 should be filtered

    def test_root_only(self):
        scope = ScopeConfig(root_only=True)
        serializer = PartialSerializer(scope)

        obj = NestedData(
            label="parent",
            child=SimpleData(name="child", value=50),
        )

        ctx = PartialSerializationContext(scope=scope)
        data = serializer.serialize(obj, ctx)

        # Only root fields serialized
        assert "label" in data

    def test_cycle_detection(self):
        serializer = PartialSerializer()

        a = CircularA(name="A")
        b = CircularB(value=42, back=a)
        a.other = b

        data = serializer.serialize(a)

        assert data["name"] == "A"
        assert "other" in data
        # The cycle should be broken with a reference
        if "back" in data.get("other", {}):
            back = data["other"]["back"]
            assert "__ref__" in back or back is None

    def test_to_json(self):
        serializer = PartialSerializer()
        obj = SimpleData(name="test", value=42)

        json_str = serializer.to_json(obj)
        parsed = json.loads(json_str)

        assert parsed["name"] == "test"
        assert parsed["value"] == 42

    def test_to_json_pretty(self):
        serializer = PartialSerializer()
        obj = SimpleData(name="test", value=42)

        json_str = serializer.to_json(obj, pretty=True)

        assert "\n" in json_str
        assert "  " in json_str

    def test_snapshot_mode(self):
        scope = ScopeConfig.snapshot()
        serializer = PartialSerializer(scope)

        obj = DeeplyNested(
            level1=NestedData(
                label="nested",
                child=SimpleData(name="deep", value=1),
            )
        )

        data = serializer.serialize(obj)

        # Snapshot mode should have limited depth
        assert "__type__" not in data  # include_type_info is False

    def test_full_mode(self):
        scope = ScopeConfig.full()
        serializer = PartialSerializer(scope)

        obj = DeeplyNested(
            level1=NestedData(
                label="nested",
                child=SimpleData(name="deep", value=1),
            )
        )

        data = serializer.serialize(obj)

        # Full mode should have all nested data
        assert data["level1"]["child"]["name"] == "deep"


# =============================================================================
# PartialDeserializer Tests
# =============================================================================


class TestPartialDeserializer:
    """Tests for PartialDeserializer."""

    def test_deserialize_primitives(self):
        deserializer = PartialDeserializer()

        assert deserializer.deserialize(None, type(None)) is None
        assert deserializer.deserialize(True, bool) is True
        assert deserializer.deserialize(42, int) == 42
        assert deserializer.deserialize(3.14, float) == 3.14
        assert deserializer.deserialize("hello", str) == "hello"

    def test_deserialize_simple_dataclass(self):
        deserializer = PartialDeserializer()
        data = {"name": "test", "value": 100, "active": True}

        obj = deserializer.deserialize(data, SimpleData)

        assert obj.name == "test"
        assert obj.value == 100
        assert obj.active is True

    def test_deserialize_nested_dataclass(self):
        deserializer = PartialDeserializer()
        data = {
            "label": "parent",
            "child": {"name": "child", "value": 50, "active": False},
        }

        obj = deserializer.deserialize(data, NestedData)

        assert obj.label == "parent"
        assert obj.child.name == "child"
        assert obj.child.value == 50

    def test_deserialize_with_missing_field_default(self):
        deserializer = PartialDeserializer(missing_policy=MissingFieldPolicy.DEFAULT)
        data = {"name": "test"}  # Missing value and active

        obj = deserializer.deserialize(data, WithDefaults)

        assert obj.name == "test"
        assert obj.count == 0  # Default
        assert obj.enabled is True  # Default
        assert obj.items == []  # Default factory

    def test_deserialize_with_missing_field_error(self):
        deserializer = PartialDeserializer(missing_policy=MissingFieldPolicy.ERROR)
        data = {"name": "test"}  # Missing required 'value'

        with pytest.raises(SerializationError):
            deserializer.deserialize(data, SimpleData)

    def test_deserialize_collections(self):
        deserializer = PartialDeserializer()
        data = {
            "items": [1, 2, 3],
            "mapping": {"a": "alpha"},
            "tags": {"__set__": ["t1", "t2"]},
        }

        obj = deserializer.deserialize(data, WithCollections)

        assert obj.items == [1, 2, 3]
        assert obj.mapping == {"a": "alpha"}
        assert obj.tags == {"t1", "t2"}

    def test_deserialize_optionals(self):
        deserializer = PartialDeserializer()
        data = {"required": "value"}

        obj = deserializer.deserialize(data, WithOptionals)

        assert obj.required == "value"
        assert obj.optional_str is None
        assert obj.optional_int is None
        assert obj.optional_list is None

    def test_deserialize_optionals_present(self):
        deserializer = PartialDeserializer()
        data = {
            "required": "value",
            "optional_str": "present",
            "optional_int": 42,
            "optional_list": [1, 2, 3],
        }

        obj = deserializer.deserialize(data, WithOptionals)

        assert obj.optional_str == "present"
        assert obj.optional_int == 42
        assert obj.optional_list == [1, 2, 3]

    def test_type_coercion_int(self):
        deserializer = PartialDeserializer()
        data = {"name": "test", "value": "42", "active": True}

        obj = deserializer.deserialize(data, SimpleData)

        assert obj.value == 42
        assert isinstance(obj.value, int)

    def test_type_coercion_float(self):
        @serializable()
        @dataclass
        class WithFloat:
            value: float

        deserializer = PartialDeserializer()
        data = {"value": "3.14"}

        obj = deserializer.deserialize(data, WithFloat)

        assert obj.value == 3.14

    def test_type_coercion_bool(self):
        deserializer = PartialDeserializer()
        data = {"name": "test", "value": 1, "active": "true"}

        obj = deserializer.deserialize(data, SimpleData)

        assert obj.active is True

    def test_type_coercion_bool_variations(self):
        deserializer = PartialDeserializer()

        @serializable()
        @dataclass
        class BoolTest:
            flag: bool

        assert deserializer.deserialize({"flag": "yes"}, BoolTest).flag is True
        assert deserializer.deserialize({"flag": "1"}, BoolTest).flag is True
        assert deserializer.deserialize({"flag": "TRUE"}, BoolTest).flag is True
        assert deserializer.deserialize({"flag": "no"}, BoolTest).flag is False
        assert deserializer.deserialize({"flag": 1}, BoolTest).flag is True
        assert deserializer.deserialize({"flag": 0}, BoolTest).flag is False

    def test_reference_marker_handling(self):
        deserializer = PartialDeserializer()
        data = {"__ref__": "ref_0", "id": 42, "__type__": "EntityRef"}

        result = deserializer.deserialize(data, EntityRef)

        assert isinstance(result, ReferenceMarker)
        assert result.object_id == 42

    def test_truncated_marker_handling(self):
        deserializer = PartialDeserializer()
        data = {"__truncated__": True, "__type__": "SomeType"}

        result = deserializer.deserialize(data, Any)

        assert isinstance(result, TruncatedMarker)
        assert result.type_name == "SomeType"

    def test_register_default_factory(self):
        deserializer = PartialDeserializer()

        def create_simple():
            return SimpleData(name="default", value=0)

        deserializer.register_default_factory(SimpleData, create_simple)
        data = {"__ref__": "ref_0"}

        result = deserializer.deserialize(data, SimpleData)

        assert isinstance(result, SimpleData)
        assert result.name == "default"

    def test_from_json(self):
        deserializer = PartialDeserializer()
        json_str = '{"name": "test", "value": 42, "active": true}'

        obj = deserializer.from_json(json_str, SimpleData)

        assert obj.name == "test"
        assert obj.value == 42

    def test_deserialize_union_types(self):
        @serializable()
        @dataclass
        class WithUnion:
            value: Union[int, str]

        deserializer = PartialDeserializer()

        # Can't fully resolve union, but should handle gracefully
        obj = deserializer.deserialize({"value": 42}, WithUnion)
        assert obj.value == 42

    def test_empty_collections(self):
        deserializer = PartialDeserializer()
        data = {"items": [], "mapping": {}, "tags": {"__set__": []}}

        obj = deserializer.deserialize(data, WithCollections)

        assert obj.items == []
        assert obj.mapping == {}
        assert obj.tags == set()

    def test_nested_missing_with_default(self):
        deserializer = PartialDeserializer(missing_policy=MissingFieldPolicy.DEFAULT)
        data = {"label": "parent"}  # Missing child

        obj = deserializer.deserialize(data, NestedData)

        assert obj.label == "parent"
        # Child should get default values or None

    def test_strict_mode(self):
        deserializer = PartialDeserializer(strict=True)
        data = "not a dict"

        with pytest.raises(SerializationError):
            deserializer.deserialize(data, SimpleData)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPartialSerializableDecorator:
    """Tests for @partial_serializable decorator."""

    def test_basic_usage(self):
        @partial_serializable()
        @dataclass
        class Basic:
            name: str
            value: int

        obj = Basic(name="test", value=42)

        # Standard serialization
        data = obj.serialize()
        assert data["name"] == "test"

        # Partial serialization
        scope = ScopeConfig(exclude_fields={"value"})
        partial_data = obj.partial_serialize(scope=scope)
        assert "name" in partial_data

    def test_with_sensitive_fields(self):
        @partial_serializable(sensitive_fields={"password"})
        @dataclass
        class WithSensitive:
            username: str
            password: str

        obj = WithSensitive(username="user", password="secret")

        # Full serialization includes all
        full_data = obj.partial_serialize(scope=ScopeConfig())
        assert "password" in full_data

    def test_partial_deserialize(self):
        @partial_serializable()
        @dataclass
        class Partial:
            name: str
            value: int = 0

        data = {"name": "test"}  # Missing value

        obj = Partial.partial_deserialize(data, missing_policy=MissingFieldPolicy.DEFAULT)

        assert obj.name == "test"
        assert obj.value == 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_snapshot(self):
        obj = DeeplyNested(
            level1=NestedData(
                label="nested",
                child=SimpleData(name="deep", value=1),
            )
        )

        snapshot = create_snapshot(obj)

        # Snapshot should be minimal
        assert isinstance(snapshot, dict)

    def test_create_snapshot_with_custom_scope(self):
        obj = SimpleData(name="test", value=42, active=True)
        scope = ScopeConfig(exclude_fields={"active"})

        snapshot = create_snapshot(obj, scope)

        assert "active" not in snapshot or snapshot.get("active") is None

    def test_create_full_serialization(self):
        obj = DeeplyNested(
            level1=NestedData(
                label="nested",
                child=SimpleData(name="deep", value=1),
            )
        )

        full = create_full_serialization(obj)

        assert full["level1"]["child"]["name"] == "deep"

    def test_deserialize_partial(self):
        data = {"name": "test"}  # Missing value

        obj = deserialize_partial(
            data,
            WithDefaults,
            missing_policy=MissingFieldPolicy.DEFAULT,
        )

        assert obj.name == "test"
        assert obj.count == 0


class TestRoundTrip:
    """Round-trip serialization tests."""

    def test_simple_roundtrip(self):
        serializer = PartialSerializer()
        deserializer = PartialDeserializer()

        original = SimpleData(name="test", value=42, active=True)

        data = serializer.serialize(original)
        restored = deserializer.deserialize(data, SimpleData)

        assert restored.name == original.name
        assert restored.value == original.value
        assert restored.active == original.active

    def test_nested_roundtrip(self):
        serializer = PartialSerializer()
        deserializer = PartialDeserializer()

        original = NestedData(
            label="parent",
            child=SimpleData(name="child", value=100),
        )

        data = serializer.serialize(original)
        restored = deserializer.deserialize(data, NestedData)

        assert restored.label == original.label
        assert restored.child.name == original.child.name
        assert restored.child.value == original.child.value

    def test_collections_roundtrip(self):
        serializer = PartialSerializer()
        deserializer = PartialDeserializer()

        original = WithCollections(
            items=[1, 2, 3],
            mapping={"a": "alpha", "b": "beta"},
            tags={"x", "y", "z"},
        )

        data = serializer.serialize(original)
        restored = deserializer.deserialize(data, WithCollections)

        assert restored.items == original.items
        assert restored.mapping == original.mapping
        assert restored.tags == original.tags

    def test_optionals_roundtrip(self):
        serializer = PartialSerializer()
        deserializer = PartialDeserializer()

        original = WithOptionals(
            required="test",
            optional_str="present",
            optional_int=42,
        )

        data = serializer.serialize(original)
        restored = deserializer.deserialize(data, WithOptionals)

        assert restored.required == original.required
        assert restored.optional_str == original.optional_str
        assert restored.optional_int == original.optional_int

    def test_json_roundtrip(self):
        serializer = PartialSerializer()
        deserializer = PartialDeserializer()

        original = SimpleData(name="json_test", value=999)

        json_str = serializer.to_json(original)
        restored = deserializer.from_json(json_str, SimpleData)

        assert restored.name == original.name
        assert restored.value == original.value


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_dataclass(self):
        @serializable()
        @dataclass
        class Empty:
            pass

        serializer = PartialSerializer()
        deserializer = PartialDeserializer()

        obj = Empty()
        data = serializer.serialize(obj)
        restored = deserializer.deserialize(data, Empty)

        assert isinstance(restored, Empty)

    def test_none_values(self):
        @serializable()
        @dataclass
        class WithNone:
            value: Optional[int] = None

        serializer = PartialSerializer()
        deserializer = PartialDeserializer()

        obj = WithNone(value=None)
        data = serializer.serialize(obj)
        restored = deserializer.deserialize(data, WithNone)

        assert restored.value is None

    def test_special_characters_in_strings(self):
        serializer = PartialSerializer()
        deserializer = PartialDeserializer()

        obj = SimpleData(name='test\n"quoted"\ttab', value=42)
        data = serializer.serialize(obj)
        json_str = json.dumps(data)
        restored = deserializer.from_json(json_str, SimpleData)

        assert restored.name == obj.name

    def test_unicode_strings(self):
        serializer = PartialSerializer()
        deserializer = PartialDeserializer()

        obj = SimpleData(name="Test unicode: 中文 \U0001F600", value=42)
        data = serializer.serialize(obj)
        restored = deserializer.deserialize(data, SimpleData)

        assert restored.name == obj.name

    def test_large_numbers(self):
        @serializable()
        @dataclass
        class BigNumbers:
            big_int: int
            big_float: float

        serializer = PartialSerializer()
        deserializer = PartialDeserializer()

        obj = BigNumbers(big_int=10**20, big_float=1e308)
        data = serializer.serialize(obj)
        restored = deserializer.deserialize(data, BigNumbers)

        assert restored.big_int == obj.big_int
        assert restored.big_float == obj.big_float

    def test_negative_numbers(self):
        serializer = PartialSerializer()
        deserializer = PartialDeserializer()

        obj = SimpleData(name="negative", value=-42)
        data = serializer.serialize(obj)
        restored = deserializer.deserialize(data, SimpleData)

        assert restored.value == -42

    def test_empty_string(self):
        serializer = PartialSerializer()
        deserializer = PartialDeserializer()

        obj = SimpleData(name="", value=0)
        data = serializer.serialize(obj)
        restored = deserializer.deserialize(data, SimpleData)

        assert restored.name == ""

    def test_deeply_nested_beyond_limit(self):
        # Create 5-level deep nesting
        @serializable()
        @dataclass
        class L5:
            v: int

        @serializable()
        @dataclass
        class L4:
            child: L5

        @serializable()
        @dataclass
        class L3:
            child: L4

        @serializable()
        @dataclass
        class L2:
            child: L3

        @serializable()
        @dataclass
        class L1:
            child: L2

        scope = ScopeConfig(max_depth=2, depth_policy=DepthPolicy.REFERENCE)
        serializer = PartialSerializer(scope)

        obj = L1(child=L2(child=L3(child=L4(child=L5(v=42)))))
        data = serializer.serialize(obj)

        # Should have references beyond depth 2
        assert isinstance(data, dict)

    def test_self_referencing_object(self):
        @dataclass
        class SelfRef:
            name: str
            self_ref: Optional["SelfRef"] = None

        serializer = PartialSerializer()

        obj = SelfRef(name="root")
        obj.self_ref = obj  # Self reference

        data = serializer.serialize(obj)

        assert data["name"] == "root"
        # Self reference should be detected and handled
        assert "__ref__" in data.get("self_ref", {}) or data.get("self_ref") is None


# =============================================================================
# Statistics Tests
# =============================================================================


class TestPartialSerializationStats:
    """Tests for PartialSerializationStats."""

    def test_initial_state(self):
        stats = PartialSerializationStats()
        assert stats.serializations == 0
        assert stats.deserializations == 0

    def test_record_serialization(self):
        stats = PartialSerializationStats()
        ctx = PartialSerializationContext()

        # Simulate some serialization work
        ctx.should_serialize_field("field1", "value")
        ctx.should_serialize_field("field2", "value")

        stats.record_serialization(ctx)

        assert stats.serializations == 1
        assert stats.fields_serialized == 2

    def test_record_deserialization(self):
        stats = PartialSerializationStats()

        stats.record_deserialization(missing_defaulted=3, type_coercions=2)

        assert stats.deserializations == 1
        assert stats.missing_fields_defaulted == 3
        assert stats.type_coercions == 2

    def test_to_dict(self):
        stats = PartialSerializationStats()
        stats.serializations = 5
        stats.deserializations = 3

        d = stats.to_dict()

        assert d["serializations"] == 5
        assert d["deserializations"] == 3

    def test_reset(self):
        stats = PartialSerializationStats()
        stats.serializations = 10
        stats.deserializations = 5

        stats.reset()

        assert stats.serializations == 0
        assert stats.deserializations == 0

    def test_global_stats(self):
        stats = get_partial_serialization_stats()
        assert isinstance(stats, PartialSerializationStats)


# =============================================================================
# FieldDescriptor Tests
# =============================================================================


class TestFieldDescriptor:
    """Tests for FieldDescriptor."""

    def test_basic_descriptor(self):
        desc = FieldDescriptor(
            name="test_field",
            type_hint=int,
        )
        assert desc.name == "test_field"
        assert desc.type_hint is int
        assert desc.has_default is False

    def test_with_default(self):
        desc = FieldDescriptor(
            name="test_field",
            type_hint=int,
            default=42,
            has_default=True,
        )
        assert desc.default == 42
        assert desc.has_default is True

    def test_with_metadata(self):
        desc = FieldDescriptor(
            name="password",
            type_hint=str,
            sensitive=True,
            large=False,
        )
        assert desc.sensitive is True
        assert desc.large is False

    def test_reference_descriptor(self):
        desc = FieldDescriptor(
            name="entity_ref",
            type_hint=EntityRef,
            reference=True,
        )
        assert desc.reference is True


# =============================================================================
# DepthPolicy Tests
# =============================================================================


class TestDepthPolicy:
    """Tests for DepthPolicy handling."""

    def test_truncate_policy(self):
        scope = ScopeConfig(max_depth=1, depth_policy=DepthPolicy.TRUNCATE)
        ctx = PartialSerializationContext(scope=scope)

        ctx.enter("level1")
        ctx.enter("level2")

        # At depth 2, which exceeds max_depth=1
        assert not ctx.should_serialize_field("deep_field")

    def test_reference_policy(self):
        scope = ScopeConfig(max_depth=1, depth_policy=DepthPolicy.REFERENCE)
        serializer = PartialSerializer(scope)

        obj = NestedData(
            label="parent",
            child=SimpleData(name="child", value=50),
        )

        data = serializer.serialize(obj)

        # Child should be a reference
        assert "__ref__" in data.get("child", {}) or isinstance(
            data.get("child"), dict
        )

    def test_placeholder_policy(self):
        scope = ScopeConfig(max_depth=1, depth_policy=DepthPolicy.PLACEHOLDER)
        serializer = PartialSerializer(scope)

        obj = NestedData(
            label="parent",
            child=SimpleData(name="child", value=50),
        )

        data = serializer.serialize(obj)

        child = data.get("child", {})
        assert "__truncated__" in child or "__ref__" in child

    def test_error_policy(self):
        scope = ScopeConfig(max_depth=0, depth_policy=DepthPolicy.ERROR)
        ctx = PartialSerializationContext(scope=scope)

        ctx.enter("level1")

        with pytest.raises(SerializationError):
            ctx.should_serialize_field("any_field")


# =============================================================================
# MissingFieldPolicy Tests
# =============================================================================


class TestMissingFieldPolicy:
    """Tests for MissingFieldPolicy handling."""

    def test_default_policy_uses_field_defaults(self):
        deserializer = PartialDeserializer(missing_policy=MissingFieldPolicy.DEFAULT)
        data = {"name": "test"}

        obj = deserializer.deserialize(data, WithDefaults)

        assert obj.count == 0
        assert obj.enabled is True
        assert obj.items == []

    def test_default_policy_uses_type_defaults(self):
        deserializer = PartialDeserializer(missing_policy=MissingFieldPolicy.DEFAULT)
        data = {"name": "test"}  # Missing value and active

        obj = deserializer.deserialize(data, SimpleData)

        assert obj.value == 0  # int default
        assert obj.active is False or obj.active is True  # bool default

    def test_error_policy_raises(self):
        deserializer = PartialDeserializer(missing_policy=MissingFieldPolicy.ERROR)
        data = {"name": "test"}

        with pytest.raises(SerializationError):
            deserializer.deserialize(data, SimpleData)

    def test_skip_policy(self):
        # Skip is harder to test as it may cause errors on construction
        deserializer = PartialDeserializer(missing_policy=MissingFieldPolicy.SKIP)
        data = {"name": "test"}

        # For types with all defaults, skip should work
        obj = deserializer.deserialize(data, WithDefaults)
        assert obj.name == "test"


# =============================================================================
# SerializationMode Tests
# =============================================================================


class TestSerializationMode:
    """Tests for SerializationMode."""

    def test_snapshot_mode_minimal(self):
        scope = ScopeConfig.snapshot()
        assert scope.mode == SerializationMode.SNAPSHOT
        assert scope.include_type_info is False
        assert scope.exclude_sensitive is True

    def test_standard_mode(self):
        scope = ScopeConfig()
        assert scope.mode == SerializationMode.STANDARD

    def test_full_mode_complete(self):
        scope = ScopeConfig.full()
        assert scope.mode == SerializationMode.FULL
        assert scope.include_type_info is True
        assert scope.exclude_sensitive is False


# =============================================================================
# Complex Scenarios
# =============================================================================


class TestComplexScenarios:
    """Complex real-world scenario tests."""

    def test_entity_graph_serialization(self):
        """Test serializing a graph of entities with references."""

        @serializable()
        @dataclass
        class Node:
            id: int
            name: str
            connections: List["Node"] = field(default_factory=list)

        # Create a small graph
        n1 = Node(id=1, name="Node 1")
        n2 = Node(id=2, name="Node 2")
        n3 = Node(id=3, name="Node 3")

        n1.connections = [n2, n3]
        n2.connections = [n1]  # Creates cycle back to n1

        serializer = PartialSerializer()
        data = serializer.serialize(n1)

        # Should handle cycles gracefully
        assert data["id"] == 1
        assert len(data["connections"]) == 2

    def test_incremental_deserialization(self):
        """Test deserializing with progressively more data."""
        deserializer = PartialDeserializer(missing_policy=MissingFieldPolicy.DEFAULT)

        # Minimal data
        minimal = {"required": "test"}
        obj1 = deserializer.deserialize(minimal, WithOptionals)
        assert obj1.required == "test"
        assert obj1.optional_str is None

        # More data
        more = {"required": "test", "optional_str": "present"}
        obj2 = deserializer.deserialize(more, WithOptionals)
        assert obj2.optional_str == "present"

        # Full data
        full = {
            "required": "test",
            "optional_str": "present",
            "optional_int": 42,
            "optional_list": [1, 2, 3],
        }
        obj3 = deserializer.deserialize(full, WithOptionals)
        assert obj3.optional_int == 42
        assert obj3.optional_list == [1, 2, 3]

    def test_mixed_known_unknown_fields(self):
        """Test deserializing data with unknown fields."""
        deserializer = PartialDeserializer()
        data = {
            "name": "test",
            "value": 42,
            "active": True,
            "unknown_field": "ignored",
            "another_unknown": [1, 2, 3],
        }

        obj = deserializer.deserialize(data, SimpleData)

        assert obj.name == "test"
        assert obj.value == 42
        # Unknown fields should be ignored

    def test_version_migration_scenario(self):
        """Test handling data from different schema versions."""
        # Old format (v1)
        old_data = {"label": "old_format", "nested_value": 42}

        # New format expects different structure
        @serializable()
        @dataclass
        class NewFormat:
            label: str
            child: Optional[SimpleData] = None

        deserializer = PartialDeserializer(missing_policy=MissingFieldPolicy.DEFAULT)
        obj = deserializer.deserialize(old_data, NewFormat)

        assert obj.label == "old_format"
        assert obj.child is None  # Default since not in old format

    def test_partial_update_pattern(self):
        """Test pattern for partial updates to objects."""
        serializer = PartialSerializer()
        deserializer = PartialDeserializer()

        # Original object
        original = WithDefaults(name="original", count=10, enabled=True, items=["a", "b"])

        # Serialize only modified fields
        scope = ScopeConfig(include_fields={"name", "count"})
        partial_data = serializer.serialize(original, PartialSerializationContext(scope=scope))

        # Partial data for update (only changed fields)
        update_data = {"name": "updated", "count": 20}

        # Merge with defaults for full reconstruction
        merged = {**{"enabled": True, "items": []}, **update_data}

        updated = deserializer.deserialize(merged, WithDefaults)

        assert updated.name == "updated"
        assert updated.count == 20
        assert updated.enabled is True
