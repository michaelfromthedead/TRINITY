"""
Comprehensive unit tests for the Mirror system.
Tests reflection capabilities for objects, classes, dataclasses, and slots.
"""
import pytest
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from dataclasses import dataclass
from typing import Annotated
from foundation.mirror import mirror, ObjectMirror, ClassMirror, FieldInfo, MethodInfo


class TestMirrorBasics:
    """Test basic mirror() function behavior."""

    def test_mirror_returns_object_mirror_for_instance(self):
        """mirror() should return ObjectMirror for object instances."""
        obj = object()
        m = mirror(obj)
        assert isinstance(m, ObjectMirror)

    def test_mirror_returns_class_mirror_for_class(self):
        """mirror() should return ClassMirror for class types."""
        m = mirror(int)
        assert isinstance(m, ClassMirror)

    def test_type_name_property(self):
        """type_name should return the class name."""
        class Foo:
            pass
        m = mirror(Foo())
        assert m.type_name == "Foo"

    def test_mirror_builtin_types(self):
        """mirror() should work with builtin types."""
        m = mirror(str)
        assert isinstance(m, ClassMirror)
        assert m.type_name == "str"

    def test_mirror_builtin_instances(self):
        """mirror() should work with builtin instances."""
        m = mirror("hello")
        assert isinstance(m, ObjectMirror)
        assert m.type_name == "str"


class TestObjectMirror:
    """Test ObjectMirror functionality."""

    def test_fields_from_annotations(self):
        """ObjectMirror should detect fields from class annotations."""
        class Player:
            health: int = 100
            name: str = "Unknown"

        m = mirror(Player())
        assert "health" in m.fields
        assert "name" in m.fields

    def test_get_returns_value(self):
        """get() should return the current attribute value."""
        class Box:
            value: int = 42

        obj = Box()
        m = mirror(obj)
        assert m.get("value") == 42

    def test_set_modifies_value(self):
        """set() should modify the object's attribute."""
        class Box:
            value: int = 0

        obj = Box()
        m = mirror(obj)
        m.set("value", 99)
        assert obj.value == 99

    def test_has_returns_true_for_existing(self):
        """has() should return True for existing attributes."""
        class Foo:
            x: int = 1
        m = mirror(Foo())
        assert m.has("x") is True
        assert m.has("nonexistent") is False

    def test_to_dict_returns_dict(self):
        """to_dict() should return a dictionary of all field values."""
        class Point:
            x: int = 10
            y: int = 20

        m = mirror(Point())
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d.get("x") == 10
        assert d.get("y") == 20

    def test_type_class_property(self):
        """type_class should return the actual class type."""
        class Widget:
            pass
        obj = Widget()
        m = mirror(obj)
        assert m.type_class is Widget

    def test_get_modified_instance_value(self):
        """get() should return modified instance values, not defaults."""
        class Counter:
            count: int = 0

        obj = Counter()
        obj.count = 42
        m = mirror(obj)
        assert m.get("count") == 42

    def test_multiple_set_operations(self):
        """Multiple set() operations should all take effect."""
        class Container:
            a: int = 0
            b: int = 0

        obj = Container()
        m = mirror(obj)
        m.set("a", 10)
        m.set("b", 20)
        assert obj.a == 10
        assert obj.b == 20


class TestDataclassSupport:
    """Test mirror support for dataclasses."""

    def test_mirror_dataclass(self):
        """mirror() should work with dataclass instances."""
        @dataclass
        class Item:
            name: str
            count: int = 0

        item = Item(name="sword", count=5)
        m = mirror(item)
        assert m.get("name") == "sword"
        assert m.get("count") == 5

    def test_mirror_dataclass_fields(self):
        """Dataclass fields should be properly detected."""
        @dataclass
        class Entity:
            id: int
            label: str = "default"

        m = mirror(Entity(id=1))
        assert "id" in m.fields
        assert "label" in m.fields

    def test_dataclass_field_defaults(self):
        """Dataclass field defaults should be captured in FieldInfo."""
        @dataclass
        class Config:
            enabled: bool = True
            max_size: int = 100

        m = mirror(Config())
        enabled_field = m.fields.get("enabled")
        assert enabled_field is not None
        assert enabled_field.has_default is True

    def test_dataclass_to_dict(self):
        """to_dict() should work correctly with dataclasses."""
        @dataclass
        class Point3D:
            x: float
            y: float
            z: float = 0.0

        p = Point3D(x=1.0, y=2.0, z=3.0)
        m = mirror(p)
        d = m.to_dict()
        assert d == {"x": 1.0, "y": 2.0, "z": 3.0}


class TestSlotsSupport:
    """Test mirror support for __slots__ classes."""

    def test_mirror_slots_class(self):
        """mirror() should work with __slots__ classes."""
        class Slotted:
            __slots__ = ['x', 'y']
            def __init__(self):
                self.x = 1
                self.y = 2

        obj = Slotted()
        m = mirror(obj)
        assert m.get("x") == 1
        assert m.get("y") == 2

    def test_slots_fields_detected(self):
        """Slots should be detected as fields."""
        class SlottedClass:
            __slots__ = ['a', 'b']
            a: int
            b: str
            def __init__(self):
                self.a = 10
                self.b = "test"

        obj = SlottedClass()
        m = mirror(obj)
        assert "a" in m.fields
        assert "b" in m.fields

    def test_slots_set_value(self):
        """set() should work with slotted attributes."""
        class SlottedBox:
            __slots__ = ['value']
            def __init__(self):
                self.value = 0

        obj = SlottedBox()
        m = mirror(obj)
        m.set("value", 999)
        assert obj.value == 999


class TestAnnotatedMetadata:
    """Test extraction of Annotated type metadata."""

    def test_extracts_dict_metadata(self):
        """Metadata from Annotated should be extracted."""
        class Config:
            value: Annotated[int, {"readonly": True, "range": (0, 100)}] = 50

        m = mirror(Config())
        field = m.fields.get("value")
        assert field is not None
        # Metadata should be extracted from Annotated
        assert field.metadata.get("readonly") is True
        assert field.metadata.get("range") == (0, 100)

    def test_multiple_metadata_dicts(self):
        """Multiple dict annotations should be merged."""
        class Settings:
            level: Annotated[int, {"min": 0}, {"max": 100}] = 50

        m = mirror(Settings())
        field = m.fields.get("level")
        assert field is not None
        # Both dicts should be merged
        assert field.metadata.get("min") == 0
        assert field.metadata.get("max") == 100

    def test_field_type_extracted_from_annotated(self):
        """Base type should be extracted from Annotated."""
        class Data:
            count: Annotated[int, {"label": "Count"}] = 0

        m = mirror(Data())
        field = m.fields.get("count")
        assert field is not None
        assert field.type is int


class TestMethodInfo:
    """Test method detection and information."""

    def test_methods_detected(self):
        """Public methods should be detected."""
        class Service:
            def process(self):
                pass
            def validate(self):
                pass

        m = mirror(Service())
        assert "process" in m.methods
        assert "validate" in m.methods

    def test_private_methods_excluded(self):
        """Private methods (starting with _) should be excluded."""
        class Hidden:
            def public_method(self):
                pass
            def _private_method(self):
                pass

        m = mirror(Hidden())
        assert "public_method" in m.methods
        assert "_private_method" not in m.methods

    def test_method_info_structure(self):
        """MethodInfo should have correct structure."""
        class Calculator:
            def add(self, a: int, b: int) -> int:
                return a + b

        m = mirror(Calculator())
        method = m.methods.get("add")
        assert method is not None
        assert isinstance(method, MethodInfo)
        assert method.name == "add"
        assert method.is_property is False

    def test_property_detected_as_method(self):
        """Properties should be detected with is_property=True."""
        class WithProperty:
            @property
            def computed(self) -> int:
                return 42

        m = mirror(WithProperty())
        method = m.methods.get("computed")
        assert method is not None
        assert method.is_property is True


class TestDescribe:
    """Test the describe() method."""

    def test_describe_returns_string(self):
        """describe() should return a string."""
        class Simple:
            value: int = 1

        m = mirror(Simple())
        desc = m.describe()
        assert isinstance(desc, str)
        assert "Simple" in desc

    def test_describe_includes_fields(self):
        """describe() should include field information."""
        class Entity:
            name: str = "test"
            active: bool = True

        m = mirror(Entity())
        desc = m.describe()
        assert "name" in desc
        assert "active" in desc

    def test_describe_shows_values(self):
        """describe() should show current field values."""
        class Data:
            count: int = 42

        obj = Data()
        obj.count = 100
        m = mirror(obj)
        desc = m.describe()
        assert "100" in desc


class TestClassMirror:
    """Test ClassMirror functionality."""

    def test_class_mirror_type_name(self):
        """ClassMirror type_name should return class name."""
        class MyClass:
            pass
        m = mirror(MyClass)
        assert m.type_name == "MyClass"

    def test_class_mirror_type_class(self):
        """ClassMirror type_class should return the class itself."""
        class Target:
            pass
        m = mirror(Target)
        assert m.type_class is Target

    def test_class_mirror_fields(self):
        """ClassMirror should detect class fields."""
        class Blueprint:
            width: int = 100
            height: int = 200

        m = mirror(Blueprint)
        assert "width" in m.fields
        assert "height" in m.fields

    def test_class_mirror_methods(self):
        """ClassMirror should detect class methods."""
        class Handler:
            def handle(self):
                pass
            def cleanup(self):
                pass

        m = mirror(Handler)
        assert "handle" in m.methods
        assert "cleanup" in m.methods

    def test_class_mirror_has(self):
        """ClassMirror has() should check both fields and methods."""
        class Mixed:
            value: int = 0
            def action(self):
                pass

        m = mirror(Mixed)
        assert m.has("value") is True
        assert m.has("action") is True
        assert m.has("nonexistent") is False

    def test_class_mirror_describe(self):
        """ClassMirror describe() should format class info."""
        class Sample:
            x: int = 1
            y: int = 2

        m = mirror(Sample)
        desc = m.describe()
        assert "class Sample" in desc
        assert "x" in desc
        assert "y" in desc


class TestFieldInfo:
    """Test FieldInfo dataclass."""

    def test_field_info_attributes(self):
        """FieldInfo should have all expected attributes."""
        fi = FieldInfo(name="test", type=int, has_default=True, default=42, metadata={"key": "value"})
        assert fi.name == "test"
        assert fi.type is int
        assert fi.has_default is True
        assert fi.default == 42
        assert fi.metadata == {"key": "value"}

    def test_field_info_frozen(self):
        """FieldInfo should be frozen (immutable)."""
        fi = FieldInfo(name="test")
        with pytest.raises(Exception):  # FrozenInstanceError
            fi.name = "changed"


class TestInheritance:
    """Test mirror support for class inheritance."""

    def test_inherited_fields(self):
        """Fields from parent classes should be included."""
        class Base:
            base_field: int = 1

        class Child(Base):
            child_field: str = "child"

        m = mirror(Child())
        assert "base_field" in m.fields
        assert "child_field" in m.fields

    def test_inherited_methods(self):
        """Methods from parent classes should be included."""
        class Parent:
            def parent_method(self):
                pass

        class Child(Parent):
            def child_method(self):
                pass

        m = mirror(Child())
        assert "parent_method" in m.methods
        assert "child_method" in m.methods


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_object_mirror_rejects_class(self):
        """ObjectMirror should raise TypeError for class types."""
        with pytest.raises(TypeError):
            ObjectMirror(int)

    def test_class_mirror_rejects_instance(self):
        """ClassMirror should raise TypeError for instances."""
        with pytest.raises(TypeError):
            ClassMirror(42)

    def test_empty_class(self):
        """mirror() should handle empty classes."""
        class Empty:
            pass
        m = mirror(Empty())
        assert isinstance(m.fields, dict)
        assert isinstance(m.methods, dict)

    def test_dynamic_attributes(self):
        """Dynamic attributes should be detected."""
        class Dynamic:
            pass

        obj = Dynamic()
        obj.new_attr = "dynamic"
        m = mirror(obj)
        assert "new_attr" in m.fields
        assert m.get("new_attr") == "dynamic"
