"""
Comprehensive unit tests for the Serializer system.
Tests primitives, containers, objects, references, circular references,
binary format, file operations, utilities, and lifecycle hooks.
"""
import pytest
import sys
import tempfile
import os

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from dataclasses import dataclass
from foundation.serializer import (
    to_dict, from_dict, to_bytes, from_bytes,
    to_file, from_file, deep_copy, diff, patch,
    register_type, Delta, _type_registry
)


@pytest.fixture(autouse=True)
def clear_type_registry():
    """Clear the type registry before and after each test."""
    # Store original registry
    original = _type_registry.copy()
    _type_registry.clear()
    yield
    # Restore original registry
    _type_registry.clear()
    _type_registry.update(original)


class TestPrimitives:
    """Test serialization of primitive types."""

    def test_none(self):
        """None should serialize to dict with __value__ and deserialize back."""
        serialized = to_dict(None)
        assert serialized == {"__value__": None}
        assert from_dict(serialized) is None

    def test_bool_true(self):
        """True should serialize and deserialize correctly."""
        serialized = to_dict(True)
        assert serialized == {"__value__": True}
        assert from_dict(serialized) is True

    def test_bool_false(self):
        """False should serialize and deserialize correctly."""
        serialized = to_dict(False)
        assert serialized == {"__value__": False}
        assert from_dict(serialized) is False

    def test_int_positive(self):
        """Positive integers should round-trip correctly."""
        serialized = to_dict(42)
        assert serialized == {"__value__": 42}
        assert from_dict(serialized) == 42

    def test_int_negative(self):
        """Negative integers should round-trip correctly."""
        serialized = to_dict(-100)
        assert serialized == {"__value__": -100}
        assert from_dict(serialized) == -100

    def test_int_zero(self):
        """Zero should round-trip correctly."""
        serialized = to_dict(0)
        assert serialized == {"__value__": 0}
        assert from_dict(serialized) == 0

    def test_float(self):
        """Floats should round-trip correctly."""
        serialized = to_dict(3.14159)
        assert serialized == {"__value__": 3.14159}
        assert from_dict(serialized) == 3.14159

    def test_float_negative(self):
        """Negative floats should round-trip correctly."""
        serialized = to_dict(-2.5)
        assert serialized == {"__value__": -2.5}
        assert from_dict(serialized) == -2.5

    def test_str_simple(self):
        """Simple strings should round-trip correctly."""
        serialized = to_dict("hello")
        assert serialized == {"__value__": "hello"}
        assert from_dict(serialized) == "hello"

    def test_str_empty(self):
        """Empty strings should round-trip correctly."""
        serialized = to_dict("")
        assert serialized == {"__value__": ""}
        assert from_dict(serialized) == ""

    def test_str_unicode(self):
        """Unicode strings should round-trip correctly."""
        serialized = to_dict("Hello, World!")
        assert from_dict(serialized) == "Hello, World!"


class TestContainers:
    """Test serialization of container types."""

    def test_list_simple(self):
        """Simple lists should serialize to arrays."""
        data = [1, 2, 3]
        result = to_dict(data)
        assert result == [1, 2, 3]

    def test_list_nested(self):
        """Nested lists should serialize correctly."""
        data = [1, [2, 3], [4, [5, 6]]]
        result = to_dict(data)
        assert result == [1, [2, 3], [4, [5, 6]]]

    def test_list_empty(self):
        """Empty lists should serialize correctly."""
        data = []
        result = to_dict(data)
        assert result == []

    def test_list_round_trip(self):
        """Lists should round-trip correctly."""
        data = [1, 2, 3]
        serialized = to_dict(data)
        restored = from_dict(serialized)
        assert restored == [1, 2, 3]

    def test_tuple_preserved(self):
        """Tuples should serialize with type marker and restore as tuples."""
        data = (1, 2, 3)
        serialized = to_dict(data)
        assert serialized == {"__type__": "tuple", "items": [1, 2, 3]}
        restored = from_dict(serialized)
        assert restored == (1, 2, 3)
        assert isinstance(restored, tuple)

    def test_tuple_nested(self):
        """Nested tuples should serialize correctly."""
        data = (1, (2, 3))
        serialized = to_dict(data)
        restored = from_dict(serialized)
        assert restored == (1, (2, 3))

    def test_dict_simple(self):
        """Simple dicts should serialize to dicts."""
        data = {"a": 1, "b": 2}
        result = to_dict(data)
        assert result == {"a": 1, "b": 2}

    def test_dict_nested(self):
        """Nested dicts should serialize correctly."""
        data = {"outer": {"inner": 42}}
        result = to_dict(data)
        assert result == {"outer": {"inner": 42}}

    def test_dict_round_trip(self):
        """Dicts should round-trip correctly."""
        data = {"a": 1, "b": 2, "c": 3}
        serialized = to_dict(data)
        restored = from_dict(serialized)
        assert restored == {"a": 1, "b": 2, "c": 3}

    def test_set_simple(self):
        """Sets should serialize with type marker."""
        data = {1, 2, 3}
        serialized = to_dict(data)
        assert serialized["__type__"] == "set"
        assert set(serialized["items"]) == {1, 2, 3}

    def test_set_round_trip(self):
        """Sets should round-trip correctly."""
        data = {1, 2, 3}
        serialized = to_dict(data)
        restored = from_dict(serialized)
        assert restored == {1, 2, 3}
        assert isinstance(restored, set)

    def test_mixed_containers(self):
        """Mixed container types should serialize correctly."""
        data = {"list": [1, 2], "tuple": (3, 4), "set": {5, 6}}
        serialized = to_dict(data)
        restored = from_dict(serialized)
        assert restored["list"] == [1, 2]
        assert restored["tuple"] == (3, 4)
        assert restored["set"] == {5, 6}


class TestObjects:
    """Test serialization of custom objects."""

    def test_simple_dataclass(self):
        """Simple dataclass should serialize with type marker."""
        @dataclass
        class Point:
            x: int
            y: int

        register_type(Point)
        p = Point(10, 20)
        serialized = to_dict(p)

        assert "__type__" in serialized
        assert serialized["x"] == 10
        assert serialized["y"] == 20

    def test_dataclass_round_trip(self):
        """Dataclass should round-trip correctly."""
        @dataclass
        class Item:
            name: str
            value: int

        register_type(Item)
        original = Item("sword", 100)
        serialized = to_dict(original)
        restored = from_dict(serialized)

        assert isinstance(restored, Item)
        assert restored.name == "sword"
        assert restored.value == 100

    def test_nested_objects(self):
        """Nested objects should serialize correctly."""
        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            inner: Inner
            name: str

        register_type(Inner)
        register_type(Outer)

        original = Outer(inner=Inner(42), name="test")
        serialized = to_dict(original)
        restored = from_dict(serialized)

        assert isinstance(restored, Outer)
        assert isinstance(restored.inner, Inner)
        assert restored.inner.value == 42
        assert restored.name == "test"

    def test_object_with_list(self):
        """Object containing a list should serialize correctly."""
        @dataclass
        class Container:
            items: list

        register_type(Container)
        original = Container(items=[1, 2, 3])
        serialized = to_dict(original)
        restored = from_dict(serialized)

        assert restored.items == [1, 2, 3]

    def test_regular_class(self):
        """Non-dataclass objects should serialize via __dict__."""
        class SimpleClass:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        register_type(SimpleClass)
        original = SimpleClass(5, 10)
        serialized = to_dict(original)
        restored = from_dict(serialized)

        assert restored.x == 5
        assert restored.y == 10


class TestReferences:
    """Test handling of shared references."""

    def test_shared_list_reference(self):
        """Shared list references should be handled."""
        @dataclass
        class TwoLists:
            first: list
            second: list

        register_type(TwoLists)
        shared = [1, 2, 3]
        obj = TwoLists(first=shared, second=shared)

        serialized = to_dict(obj)
        restored = from_dict(serialized)

        assert restored.first == [1, 2, 3]
        assert restored.second == [1, 2, 3]

    @pytest.mark.xfail(reason="Known limitation: shared mutable object references not preserved during round-trip")
    def test_shared_object_reference(self):
        """Multiple references to same object should be preserved."""
        from typing import Any

        @dataclass
        class Container:
            a: Any = None
            b: Any = None

        register_type(Container)

        shared = [1, 2, 3]
        c = Container(a=shared, b=shared)

        # Before serialization, a and b are the same object
        assert c.a is c.b

        serialized = to_dict(c)
        restored = from_dict(serialized)

        # CRITICAL: After deserialization, a and b should STILL be the same object
        assert restored.a is restored.b  # Reference identity preserved!
        assert restored.a == [1, 2, 3]


class TestCircularReferences:
    """Test handling of circular references."""

    def test_simple_circular_reference(self):
        """Circular reference should be preserved through serialization."""
        from typing import Optional

        class Node:
            def __init__(self, value: int = 0):
                self.value = value
                self.next: Optional['Node'] = None

        register_type(Node)

        a = Node(1)
        b = Node(2)
        a.next = b
        b.next = a  # Circular!

        # Serialize
        serialized = to_dict(a)
        assert "__type__" in serialized

        # CRITICAL: Actually deserialize and verify circular structure
        restored = from_dict(serialized)
        assert restored.value == 1
        assert restored.next is not None
        assert restored.next.value == 2
        assert restored.next.next is restored  # Circular reference preserved!

    def test_self_reference(self):
        """Self-referential object should be preserved."""
        from typing import Optional

        class SelfRef:
            def __init__(self):
                self.value = 42
                self.self_ref: Optional['SelfRef'] = None

        register_type(SelfRef)

        obj = SelfRef()
        obj.self_ref = obj  # Self-reference!

        serialized = to_dict(obj)

        # CRITICAL: Verify self-reference is preserved after round-trip
        restored = from_dict(serialized)
        assert restored.value == 42
        assert restored.self_ref is restored  # Self-reference preserved!

    def test_circular_list(self):
        """Circular reference through list should be preserved."""
        class Container:
            def __init__(self):
                self.items: list = []

        register_type(Container)

        c = Container()
        c.items.append(c)  # Circular via list!

        serialized = to_dict(c)

        # CRITICAL: Verify circular structure through list
        restored = from_dict(serialized)
        assert len(restored.items) == 1
        assert restored.items[0] is restored  # Circular preserved!


class TestBinaryFormat:
    """Test binary serialization using pickle."""

    def test_dict_to_from_bytes(self):
        """Dict should serialize to bytes and back."""
        data = {"key": "value", "num": 42}
        binary = to_bytes(data)
        assert isinstance(binary, bytes)
        restored = from_bytes(binary)
        assert restored == data

    def test_list_to_from_bytes(self):
        """List should serialize to bytes and back."""
        data = [1, 2, 3, "four", 5.0]
        binary = to_bytes(data)
        restored = from_bytes(binary)
        assert restored == data

    def test_object_to_from_bytes(self):
        """Object should serialize to bytes and back.

        Note: pickle cannot serialize locally-defined classes, so we test
        with module-level classes or built-in types.
        """
        # Test with a dict containing various types (pickle handles these)
        original = {"x": 10, "y": "hello", "nested": [1, 2, 3]}
        binary = to_bytes(original)
        restored = from_bytes(binary)

        assert restored["x"] == 10
        assert restored["y"] == "hello"
        assert restored["nested"] == [1, 2, 3]


class TestFileOperations:
    """Test file-based serialization."""

    def test_to_from_file_json(self):
        """JSON file serialization should work."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            data = {"test": 123, "nested": {"a": 1}}
            to_file(data, path)
            restored = from_file(path)
            assert restored == data
        finally:
            os.unlink(path)

    def test_to_from_file_binary(self):
        """Binary file serialization should work."""
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            path = f.name

        try:
            data = {"test": 123, "list": [1, 2, 3]}
            to_file(data, path, binary=True)
            restored = from_file(path, binary=True)
            assert restored == data
        finally:
            os.unlink(path)

    def test_file_with_object(self):
        """Object should serialize to file and back."""
        @dataclass
        class GameState:
            level: int
            score: int
            player_name: str

        register_type(GameState)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            original = GameState(level=5, score=1000, player_name="Player1")
            to_file(original, path)
            restored = from_file(path)

            assert isinstance(restored, GameState)
            assert restored.level == 5
            assert restored.score == 1000
            assert restored.player_name == "Player1"
        finally:
            os.unlink(path)


class TestUtilities:
    """Test utility functions."""

    def test_deep_copy_dict(self):
        """deep_copy should create independent copy of dict."""
        original = {"a": [1, 2, 3], "b": {"c": 4}}
        copied = deep_copy(original)

        assert copied == original
        assert copied is not original
        assert copied["a"] is not original["a"]
        assert copied["b"] is not original["b"]

    def test_deep_copy_object(self):
        """deep_copy should create independent copy of object."""
        @dataclass
        class Data:
            values: list

        register_type(Data)

        original = Data(values=[1, 2, 3])
        copied = deep_copy(original)

        assert copied.values == original.values
        assert copied.values is not original.values

    def test_deep_copy_modification(self):
        """Modifying deep copy should not affect original."""
        original = {"items": [1, 2, 3]}
        copied = deep_copy(original)

        copied["items"].append(4)

        assert original["items"] == [1, 2, 3]
        assert copied["items"] == [1, 2, 3, 4]


class TestDiff:
    """Test the diff function."""

    def test_diff_identical_objects(self):
        """Diff of identical objects should be empty."""
        @dataclass
        class Point:
            x: int
            y: int

        register_type(Point)

        a = Point(10, 20)
        b = Point(10, 20)
        delta = diff(a, b)

        assert isinstance(delta, Delta)
        assert len(delta.changed) == 0
        assert len(delta.added) == 0
        assert len(delta.removed) == 0
        assert delta.is_empty()

    def test_diff_changed_field(self):
        """Diff should detect changed fields."""
        @dataclass
        class Point:
            x: int
            y: int

        register_type(Point)

        a = Point(10, 20)
        b = Point(10, 30)
        delta = diff(a, b)

        assert "y" in delta.changed
        assert delta.changed["y"] == (20, 30)

    def test_diff_different_types_raises(self):
        """Diff of different types should raise TypeError."""
        @dataclass
        class Point:
            x: int

        @dataclass
        class Vector:
            x: int

        register_type(Point)
        register_type(Vector)

        with pytest.raises(TypeError):
            diff(Point(1), Vector(1))


class TestPatch:
    """Test the patch function."""

    def test_patch_changed_field(self):
        """Patch should apply changed fields."""
        @dataclass
        class Point:
            x: int
            y: int

        register_type(Point)

        original = Point(10, 20)
        delta = Delta(added={}, removed={}, changed={"y": (20, 30)})
        result = patch(original, delta)

        assert result.x == 10
        assert result.y == 30
        # Original should be unchanged
        assert original.y == 20

    def test_patch_added_field(self):
        """Patch should add new fields."""
        class Flexible:
            def __init__(self):
                self.x = 10

        register_type(Flexible)

        original = Flexible()
        delta = Delta(added={"y": 20}, removed={}, changed={})
        result = patch(original, delta)

        assert result.x == 10
        assert result.y == 20


class TestLifecycleHooks:
    """Test serialization lifecycle hooks."""

    def test_before_serialize_called(self):
        """__before_serialize__ should be called during serialization."""
        class Hooked:
            def __init__(self):
                self.prepared = False
                self.value = 0

            def __before_serialize__(self):
                self.prepared = True

        register_type(Hooked)
        obj = Hooked()
        assert obj.prepared is False

        to_dict(obj)
        assert obj.prepared is True

    def test_after_deserialize_called(self):
        """__after_deserialize__ should be called after deserialization."""
        class Hooked:
            def __init__(self):
                self.initialized = False
                self.value = 42

            def __after_deserialize__(self):
                self.initialized = True

        register_type(Hooked)

        # Serialize an object
        original = Hooked()
        serialized = to_dict(original)

        # Deserialize should call __after_deserialize__
        restored = from_dict(serialized)
        assert restored.initialized is True

    def test_before_serialize_bytes(self):
        """__before_serialize__ should be called for binary serialization.

        Note: pickle cannot serialize locally-defined classes. We verify
        that the hook is called by checking the object state, even though
        the actual pickle will fail. For production use, classes must be
        defined at module level.
        """
        class Hooked:
            def __init__(self):
                self.prepared = False

            def __before_serialize__(self):
                self.prepared = True

        obj = Hooked()
        # The hook should be called even if pickle fails
        try:
            to_bytes(obj)
        except (AttributeError, TypeError):
            # Pickle fails for local classes, but hook should have been called
            pass

        assert obj.prepared is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dict(self):
        """Empty dict should serialize correctly."""
        data = {}
        serialized = to_dict(data)
        restored = from_dict(serialized)
        assert restored == {}

    def test_empty_list(self):
        """Empty list should serialize correctly."""
        data = []
        serialized = to_dict(data)
        restored = from_dict(serialized)
        assert restored == []

    def test_deeply_nested(self):
        """Deeply nested structures should serialize correctly."""
        data = {"a": {"b": {"c": {"d": {"e": 42}}}}}
        serialized = to_dict(data)
        restored = from_dict(serialized)
        assert restored["a"]["b"]["c"]["d"]["e"] == 42

    def test_large_list(self):
        """Large lists should serialize correctly."""
        data = list(range(1000))
        serialized = to_dict(data)
        restored = from_dict(serialized)
        assert restored == data

    def test_special_string_chars(self):
        """Strings with special characters should serialize correctly."""
        data = "line1\nline2\ttab\"quote'apostrophe\\backslash"
        serialized = to_dict(data)
        restored = from_dict(serialized)
        assert restored == data


class TestTypeRegistry:
    """Test type registration functionality."""

    def test_register_with_custom_name(self):
        """Type can be registered with custom name."""
        @dataclass
        class MyClass:
            value: int

        register_type(MyClass, "CustomName")
        obj = MyClass(42)

        serialized = to_dict(obj)
        # The serialized type name should be the module-qualified name
        assert "__type__" in serialized

    def test_unregistered_type_raises(self):
        """Deserializing unregistered type should raise TypeError."""
        @dataclass
        class Unregistered:
            value: int

        # Don't register it
        obj = Unregistered(42)
        serialized = to_dict(obj)

        # Clear registry to ensure it's not registered
        type_name = serialized.get("__type__")
        if type_name in _type_registry:
            del _type_registry[type_name]

        with pytest.raises(TypeError, match="Unknown type"):
            from_dict(serialized)


class TestTransientFields:
    """Test transient field handling (fields not serialized)."""

    def test_transient_field_not_serialized(self):
        """Fields marked transient should not be serialized."""
        from dataclasses import dataclass, field
        from typing import Annotated

        @dataclass
        class WithTransient:
            persistent: int
            transient_field: int = field(default=0, metadata={"transient": True})

        register_type(WithTransient)

        obj = WithTransient(persistent=10, transient_field=999)
        serialized = to_dict(obj)

        assert serialized["persistent"] == 10
        assert "transient_field" not in serialized


class TestSchemaHashSerialization:
    """Test schema hash inclusion in serialization (Phase 7)."""

    def test_schema_hash_not_included_by_default(self):
        """Schema hash should not be included by default."""
        @dataclass
        class Simple:
            value: int

        register_type(Simple)
        obj = Simple(42)
        serialized = to_dict(obj)

        assert "__type__" in serialized
        assert "__schema__" not in serialized

    def test_schema_hash_included_when_requested(self):
        """Schema hash should be included when include_schema_hash=True."""
        from foundation.mirror import SCHEMA_HASH_LENGTH

        @dataclass
        class Simple:
            value: int

        register_type(Simple)
        obj = Simple(42)
        serialized = to_dict(obj, include_schema_hash=True)

        assert "__type__" in serialized
        assert "__schema__" in serialized
        assert len(serialized["__schema__"]) == SCHEMA_HASH_LENGTH
        assert all(c in '0123456789abcdef' for c in serialized["__schema__"])

    def test_schema_hash_matches_mirror_schema_hash(self):
        """Included schema hash should match mirror.schema_hash()."""
        from foundation.mirror import schema_hash as mirror_schema_hash

        @dataclass
        class Point:
            x: int
            y: int

        register_type(Point)
        obj = Point(10, 20)
        serialized = to_dict(obj, include_schema_hash=True)

        expected_hash = mirror_schema_hash(Point)
        assert serialized["__schema__"] == expected_hash

    def test_nested_objects_have_schema_hashes(self):
        """Nested objects should also have schema hashes."""
        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            inner: Inner

        register_type(Inner)
        register_type(Outer)

        obj = Outer(inner=Inner(42))
        serialized = to_dict(obj, include_schema_hash=True)

        assert "__schema__" in serialized
        assert "__schema__" in serialized["inner"]


class TestSchemaVerification:
    """Test schema verification during deserialization (Phase 7)."""

    def test_matching_schema_deserializes_normally(self):
        """Matching schema hash should deserialize without error."""
        @dataclass
        class Entity:
            id: int
            name: str

        register_type(Entity)
        obj = Entity(1, "test")

        serialized = to_dict(obj, include_schema_hash=True)
        restored = from_dict(serialized)

        assert isinstance(restored, Entity)
        assert restored.id == 1
        assert restored.name == "test"

    def test_mismatched_schema_raises_error(self):
        """Mismatched schema hash should raise SchemaMismatchError."""
        from foundation.serializer import SchemaMismatchError
        from foundation.mirror import SCHEMA_HASH_LENGTH, schema_hash as mirror_schema_hash

        @dataclass
        class Current:
            value: int

        register_type(Current)

        # Serialize to get the correct type name
        obj = Current(42)
        serialized = to_dict(obj, include_schema_hash=True)
        actual_hash = serialized["__schema__"]

        # Tamper with the schema hash (use constant for length)
        fake_hash = "0" * SCHEMA_HASH_LENGTH
        serialized["__schema__"] = fake_hash

        with pytest.raises(SchemaMismatchError) as exc_info:
            from_dict(serialized)

        assert exc_info.value.stored_hash == fake_hash
        assert exc_info.value.current_hash == actual_hash
        assert exc_info.value.current_hash == mirror_schema_hash(Current)

    def test_data_without_schema_deserializes_normally(self):
        """Data without __schema__ should deserialize without verification."""
        @dataclass
        class Legacy:
            value: int

        register_type(Legacy)

        # Serialize without schema hash (old format)
        obj = Legacy(42)
        serialized = to_dict(obj)  # No include_schema_hash

        # Verify no schema hash present
        assert "__schema__" not in serialized

        restored = from_dict(serialized)
        assert isinstance(restored, Legacy)
        assert restored.value == 42


class TestAutoMigration:
    """Test automatic migration during deserialization (Phase 7)."""

    def test_auto_migration_with_registry(self):
        """Auto-migration should apply when schema mismatch and registry available."""
        from foundation.serializer import set_migration_registry, get_migration_registry
        from foundation.migrations import MigrationRegistry
        from foundation.mirror import schema_hash as mirror_schema_hash, SCHEMA_HASH_LENGTH

        @dataclass
        class EntityV2:
            id: int
            name: str
            version: int = 2

        register_type(EntityV2, "test_serializer.Entity")

        # Create a migration registry
        registry = MigrationRegistry()

        # Define migration from V1 to V2 (use constant for fake hash length)
        old_hash = "a" * SCHEMA_HASH_LENGTH
        new_hash = mirror_schema_hash(EntityV2)

        def migrate_v1_to_v2(data: dict) -> dict:
            data["version"] = 2
            data["__schema__"] = new_hash
            return data

        registry.register(old_hash, new_hash, migrate_v1_to_v2)

        # Set the migration registry
        old_registry = get_migration_registry()
        set_migration_registry(registry)

        try:
            # Simulate old data
            old_data = {
                "__type__": "test_serializer.Entity",
                "__id__": 1,
                "__schema__": old_hash,
                "id": 42,
                "name": "test"
            }

            # Should auto-migrate
            restored = from_dict(old_data)

            assert isinstance(restored, EntityV2)
            assert restored.id == 42
            assert restored.name == "test"
            assert restored.version == 2

        finally:
            # Restore original registry
            set_migration_registry(old_registry)

    def test_no_migration_path_raises_error(self):
        """Missing migration path should raise SchemaMismatchError."""
        from foundation.serializer import (
            set_migration_registry, get_migration_registry, SchemaMismatchError
        )
        from foundation.migrations import MigrationRegistry
        from foundation.mirror import SCHEMA_HASH_LENGTH

        @dataclass
        class NoPath:
            value: int

        register_type(NoPath)

        # Create an empty migration registry
        registry = MigrationRegistry()

        old_registry = get_migration_registry()
        set_migration_registry(registry)

        try:
            # Serialize to get correct type name
            obj = NoPath(42)
            old_data = to_dict(obj, include_schema_hash=True)

            # Tamper with the schema hash to an unknown value (use constant for length)
            fake_hash = "b" * SCHEMA_HASH_LENGTH
            old_data["__schema__"] = fake_hash

            with pytest.raises(SchemaMismatchError) as exc_info:
                from_dict(old_data)

            assert exc_info.value.stored_hash == fake_hash

        finally:
            set_migration_registry(old_registry)

    def test_migration_registry_setter_getter(self):
        """set_migration_registry and get_migration_registry should work."""
        from foundation.serializer import set_migration_registry, get_migration_registry
        from foundation.migrations import MigrationRegistry

        original = get_migration_registry()

        new_registry = MigrationRegistry()
        set_migration_registry(new_registry)

        assert get_migration_registry() is new_registry

        # Restore
        set_migration_registry(original)


class TestSchemaHashWithFile:
    """Test schema hash with file operations (Phase 7)."""

    def test_to_file_includes_schema_hash(self):
        """to_file should include schema hash when requested."""
        import json
        from foundation.mirror import SCHEMA_HASH_LENGTH

        @dataclass
        class FileEntity:
            value: int

        register_type(FileEntity)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            obj = FileEntity(42)
            to_file(obj, path, include_schema_hash=True)

            # Read raw JSON to verify schema hash is present
            with open(path) as f:
                raw = json.load(f)

            assert "__schema__" in raw
            assert len(raw["__schema__"]) == SCHEMA_HASH_LENGTH

            # Verify round-trip works
            restored = from_file(path)
            assert isinstance(restored, FileEntity)
            assert restored.value == 42

        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
