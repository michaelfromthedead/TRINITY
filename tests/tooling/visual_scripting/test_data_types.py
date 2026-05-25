"""
Tests for FlowForge data types.

Tests type conversion, validation, and operations for all blueprint data types.
"""

import pytest
import math

from engine.tooling.visual_scripting.data_types import (
    BlueprintType,
    DataTypeCategory,
    TypeColor,
    WireColors,
    BoolType,
    IntType,
    FloatType,
    StringType,
    Vector2,
    Vector3,
    Vector2Type,
    Vector3Type,
    Rotator,
    RotatorType,
    Transform,
    TransformType,
    ObjectRef,
    ObjectType,
    ArrayValue,
    ArrayType,
    MapValue,
    MapType,
    SetValue,
    SetType,
    ExecutionType,
    WildcardType,
    TYPE_REGISTRY,
    get_type_by_name,
    register_type,
    can_connect_types,
    convert_value,
    create_object_type,
    create_array_type,
)


class TestTypeColor:
    """Tests for TypeColor."""

    def test_to_hex(self):
        color = TypeColor(255, 128, 64)
        assert color.to_hex() == "#ff8040"

    def test_to_tuple(self):
        color = TypeColor(100, 200, 50, 128)
        assert color.to_tuple() == (100, 200, 50, 128)

    def test_default_alpha(self):
        color = TypeColor(0, 0, 0)
        assert color.a == 255


class TestBoolType:
    """Tests for BoolType."""

    def test_type_name(self):
        assert BoolType.type_name() == "Bool"

    def test_validate_true(self):
        assert BoolType.validate(True)
        assert BoolType.validate(False)

    def test_validate_false(self):
        assert not BoolType.validate(1)
        assert not BoolType.validate("true")

    def test_coerce_bool(self):
        assert BoolType.coerce(True) is True
        assert BoolType.coerce(False) is False

    def test_coerce_int(self):
        assert BoolType.coerce(1) is True
        assert BoolType.coerce(0) is False
        assert BoolType.coerce(42) is True

    def test_coerce_float(self):
        assert BoolType.coerce(1.0) is True
        assert BoolType.coerce(0.0) is False

    def test_coerce_string(self):
        assert BoolType.coerce("true") is True
        assert BoolType.coerce("TRUE") is True
        assert BoolType.coerce("1") is True
        assert BoolType.coerce("yes") is True
        assert BoolType.coerce("false") is False
        assert BoolType.coerce("0") is False
        assert BoolType.coerce("no") is False
        assert BoolType.coerce("") is False

    def test_can_convert_from(self):
        assert BoolType.can_convert_from(IntType)
        assert BoolType.can_convert_from(FloatType)
        assert BoolType.can_convert_from(StringType)
        assert not BoolType.can_convert_from(Vector3Type)

    def test_wire_color(self):
        assert BoolType.wire_color == WireColors.BOOL


class TestIntType:
    """Tests for IntType."""

    def test_type_name(self):
        assert IntType.type_name() == "Int"

    def test_validate(self):
        assert IntType.validate(42)
        assert IntType.validate(0)
        assert IntType.validate(-100)
        assert not IntType.validate(True)  # bool is not int for blueprint purposes
        assert not IntType.validate(3.14)
        assert not IntType.validate("42")

    def test_coerce_int(self):
        assert IntType.coerce(42) == 42

    def test_coerce_bool(self):
        assert IntType.coerce(True) == 1
        assert IntType.coerce(False) == 0

    def test_coerce_float(self):
        assert IntType.coerce(3.7) == 3
        assert IntType.coerce(3.2) == 3

    def test_coerce_string(self):
        assert IntType.coerce("42") == 42
        assert IntType.coerce("3.7") == 3
        assert IntType.coerce("invalid") == 0

    def test_default_value(self):
        assert IntType.default_value == 0


class TestFloatType:
    """Tests for FloatType."""

    def test_type_name(self):
        assert FloatType.type_name() == "Float"

    def test_validate(self):
        assert FloatType.validate(3.14)
        assert FloatType.validate(42)  # int is valid for float
        assert not FloatType.validate(True)
        assert not FloatType.validate("3.14")

    def test_coerce_float(self):
        assert FloatType.coerce(3.14) == 3.14

    def test_coerce_int(self):
        assert FloatType.coerce(42) == 42.0

    def test_coerce_bool(self):
        assert FloatType.coerce(True) == 1.0
        assert FloatType.coerce(False) == 0.0

    def test_coerce_string(self):
        assert FloatType.coerce("3.14") == 3.14
        assert FloatType.coerce("42") == 42.0
        assert FloatType.coerce("invalid") == 0.0


class TestStringType:
    """Tests for StringType."""

    def test_type_name(self):
        assert StringType.type_name() == "String"

    def test_validate(self):
        assert StringType.validate("hello")
        assert StringType.validate("")
        assert not StringType.validate(42)

    def test_coerce(self):
        assert StringType.coerce("hello") == "hello"
        assert StringType.coerce(42) == "42"
        assert StringType.coerce(3.14) == "3.14"
        assert StringType.coerce(True) == "True"

    def test_can_convert_from_all(self):
        # String can convert from almost anything
        assert StringType.can_convert_from(IntType)
        assert StringType.can_convert_from(FloatType)
        assert StringType.can_convert_from(BoolType)
        assert StringType.can_convert_from(Vector3Type)


class TestVector2:
    """Tests for Vector2 value type."""

    def test_create(self):
        v = Vector2(1.0, 2.0)
        assert v.x == 1.0
        assert v.y == 2.0

    def test_default_values(self):
        v = Vector2()
        assert v.x == 0.0
        assert v.y == 0.0

    def test_add(self):
        v1 = Vector2(1.0, 2.0)
        v2 = Vector2(3.0, 4.0)
        result = v1 + v2
        assert result.x == 4.0
        assert result.y == 6.0

    def test_sub(self):
        v1 = Vector2(5.0, 7.0)
        v2 = Vector2(2.0, 3.0)
        result = v1 - v2
        assert result.x == 3.0
        assert result.y == 4.0

    def test_mul(self):
        v = Vector2(2.0, 3.0)
        result = v * 2
        assert result.x == 4.0
        assert result.y == 6.0

    def test_div(self):
        v = Vector2(4.0, 6.0)
        result = v / 2
        assert result.x == 2.0
        assert result.y == 3.0

    def test_neg(self):
        v = Vector2(1.0, -2.0)
        result = -v
        assert result.x == -1.0
        assert result.y == 2.0

    def test_magnitude(self):
        v = Vector2(3.0, 4.0)
        assert v.magnitude() == 5.0

    def test_normalized(self):
        v = Vector2(3.0, 4.0)
        n = v.normalized()
        assert abs(n.magnitude() - 1.0) < 0.0001

    def test_normalized_zero(self):
        v = Vector2(0.0, 0.0)
        n = v.normalized()
        assert n.x == 0.0
        assert n.y == 0.0

    def test_dot(self):
        v1 = Vector2(1.0, 2.0)
        v2 = Vector2(3.0, 4.0)
        assert v1.dot(v2) == 11.0

    def test_to_tuple(self):
        v = Vector2(1.0, 2.0)
        assert v.to_tuple() == (1.0, 2.0)


class TestVector3:
    """Tests for Vector3 value type."""

    def test_create(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_magnitude(self):
        v = Vector3(1.0, 2.0, 2.0)
        assert v.magnitude() == 3.0

    def test_cross(self):
        v1 = Vector3(1.0, 0.0, 0.0)
        v2 = Vector3(0.0, 1.0, 0.0)
        result = v1.cross(v2)
        assert result.x == 0.0
        assert result.y == 0.0
        assert result.z == 1.0

    def test_normalized(self):
        v = Vector3(1.0, 2.0, 2.0)
        n = v.normalized()
        assert abs(n.magnitude() - 1.0) < 0.0001


class TestVector2Type:
    """Tests for Vector2Type blueprint type."""

    def test_type_name(self):
        assert Vector2Type.type_name() == "Vector2"

    def test_validate(self):
        assert Vector2Type.validate(Vector2(1, 2))
        assert not Vector2Type.validate([1, 2])

    def test_coerce_vector2(self):
        v = Vector2(1, 2)
        assert Vector2Type.coerce(v) == v

    def test_coerce_vector3(self):
        v3 = Vector3(1, 2, 3)
        result = Vector2Type.coerce(v3)
        assert result.x == 1
        assert result.y == 2

    def test_coerce_list(self):
        result = Vector2Type.coerce([1.0, 2.0])
        assert result.x == 1.0
        assert result.y == 2.0

    def test_coerce_dict(self):
        result = Vector2Type.coerce({"x": 1.0, "y": 2.0})
        assert result.x == 1.0
        assert result.y == 2.0


class TestVector3Type:
    """Tests for Vector3Type blueprint type."""

    def test_coerce_vector2(self):
        v2 = Vector2(1, 2)
        result = Vector3Type.coerce(v2)
        assert result.x == 1
        assert result.y == 2
        assert result.z == 0

    def test_can_convert_from_vector2(self):
        assert Vector3Type.can_convert_from(Vector2Type)


class TestRotator:
    """Tests for Rotator value type."""

    def test_create(self):
        r = Rotator(10.0, 20.0, 30.0)
        assert r.pitch == 10.0
        assert r.yaw == 20.0
        assert r.roll == 30.0

    def test_to_radians(self):
        r = Rotator(180.0, 90.0, 45.0)
        pitch, yaw, roll = r.to_radians()
        assert abs(pitch - math.pi) < 0.0001
        assert abs(yaw - math.pi / 2) < 0.0001
        assert abs(roll - math.pi / 4) < 0.0001

    def test_from_radians(self):
        r = Rotator.from_radians(math.pi, math.pi / 2, math.pi / 4)
        assert abs(r.pitch - 180.0) < 0.0001
        assert abs(r.yaw - 90.0) < 0.0001
        assert abs(r.roll - 45.0) < 0.0001

    def test_add(self):
        r1 = Rotator(10, 20, 30)
        r2 = Rotator(5, 10, 15)
        result = r1 + r2
        assert result.pitch == 15
        assert result.yaw == 30
        assert result.roll == 45


class TestTransform:
    """Tests for Transform value type."""

    def test_create_default(self):
        t = Transform()
        assert t.location.x == 0
        assert t.rotation.pitch == 0
        assert t.scale.x == 1

    def test_get_forward_vector(self):
        t = Transform()
        forward = t.get_forward_vector()
        assert abs(forward.x - 1.0) < 0.0001
        assert abs(forward.y) < 0.0001
        assert abs(forward.z) < 0.0001


class TestObjectRef:
    """Tests for ObjectRef."""

    def test_bool_valid(self):
        ref = ObjectRef(object_id=42, is_valid=True)
        assert bool(ref) is True

    def test_bool_invalid(self):
        ref = ObjectRef(object_id=0, is_valid=False)
        assert bool(ref) is False

    def test_bool_zero_id(self):
        ref = ObjectRef(object_id=0, is_valid=True)
        assert bool(ref) is False


class TestArrayValue:
    """Tests for ArrayValue container."""

    def test_create_empty(self):
        arr = ArrayValue()
        assert len(arr) == 0

    def test_append(self):
        arr = ArrayValue()
        arr.append(1)
        arr.append(2)
        assert len(arr) == 2

    def test_getitem(self):
        arr = ArrayValue(items=[1, 2, 3])
        assert arr[0] == 1
        assert arr[2] == 3

    def test_setitem(self):
        arr = ArrayValue(items=[1, 2, 3])
        arr[1] = 5
        assert arr[1] == 5

    def test_remove(self):
        arr = ArrayValue(items=[1, 2, 3])
        arr.remove(2)
        assert len(arr) == 2
        assert 2 not in arr.items

    def test_clear(self):
        arr = ArrayValue(items=[1, 2, 3])
        arr.clear()
        assert len(arr) == 0

    def test_iter(self):
        arr = ArrayValue(items=[1, 2, 3])
        assert list(arr) == [1, 2, 3]


class TestMapValue:
    """Tests for MapValue container."""

    def test_create_empty(self):
        m = MapValue()
        assert len(m) == 0

    def test_setitem_getitem(self):
        m = MapValue()
        m["key1"] = "value1"
        assert m["key1"] == "value1"

    def test_contains(self):
        m = MapValue(items={"a": 1, "b": 2})
        assert "a" in m
        assert "c" not in m

    def test_get_default(self):
        m = MapValue(items={"a": 1})
        assert m.get("a") == 1
        assert m.get("b", 0) == 0


class TestSetValue:
    """Tests for SetValue container."""

    def test_create_empty(self):
        s = SetValue()
        assert len(s) == 0

    def test_add(self):
        s = SetValue()
        s.add(1)
        s.add(2)
        s.add(1)  # duplicate
        assert len(s) == 2

    def test_contains(self):
        s = SetValue(items={1, 2, 3})
        assert 1 in s
        assert 4 not in s

    def test_remove(self):
        s = SetValue(items={1, 2, 3})
        s.remove(2)
        assert 2 not in s


class TestExecutionType:
    """Tests for ExecutionType."""

    def test_type_name(self):
        assert ExecutionType.type_name() == "Exec"

    def test_validate_always_true(self):
        assert ExecutionType.validate(None)
        assert ExecutionType.validate(42)
        assert ExecutionType.validate("anything")

    def test_coerce_returns_none(self):
        assert ExecutionType.coerce("anything") is None


class TestWildcardType:
    """Tests for WildcardType."""

    def test_type_name(self):
        assert WildcardType.type_name() == "Wildcard"

    def test_validate_always_true(self):
        assert WildcardType.validate(42)
        assert WildcardType.validate("anything")
        assert WildcardType.validate(None)

    def test_coerce_passthrough(self):
        assert WildcardType.coerce(42) == 42
        assert WildcardType.coerce("hello") == "hello"

    def test_can_convert_from_all(self):
        assert WildcardType.can_convert_from(IntType)
        assert WildcardType.can_convert_from(StringType)
        assert WildcardType.can_convert_from(Vector3Type)


class TestTypeRegistry:
    """Tests for type registry functions."""

    def test_get_type_by_name(self):
        assert get_type_by_name("Bool") == BoolType
        assert get_type_by_name("Int") == IntType
        assert get_type_by_name("Float") == FloatType
        assert get_type_by_name("String") == StringType
        assert get_type_by_name("Vector3") == Vector3Type

    def test_get_type_by_name_not_found(self):
        assert get_type_by_name("InvalidType") is None

    def test_register_type(self):
        class CustomType(BlueprintType):
            @classmethod
            def type_name(cls):
                return "Custom"

            @classmethod
            def validate(cls, value):
                return True

            @classmethod
            def coerce(cls, value):
                return value

        register_type(CustomType)
        assert get_type_by_name("Custom") == CustomType


class TestTypeConnections:
    """Tests for type connection compatibility."""

    def test_same_type_connects(self):
        assert can_connect_types(IntType, IntType)
        assert can_connect_types(StringType, StringType)
        assert can_connect_types(Vector3Type, Vector3Type)

    def test_wildcard_connects_all(self):
        assert can_connect_types(IntType, WildcardType)
        assert can_connect_types(WildcardType, StringType)
        assert can_connect_types(WildcardType, WildcardType)

    def test_numeric_conversion(self):
        assert can_connect_types(IntType, FloatType)
        assert can_connect_types(FloatType, IntType)

    def test_bool_to_numeric(self):
        assert can_connect_types(BoolType, IntType)
        assert can_connect_types(BoolType, FloatType)

    def test_anything_to_string(self):
        assert can_connect_types(IntType, StringType)
        assert can_connect_types(FloatType, StringType)
        assert can_connect_types(BoolType, StringType)

    def test_incompatible_types(self):
        assert not can_connect_types(Vector3Type, IntType)
        assert not can_connect_types(TransformType, BoolType)


class TestTypeConversion:
    """Tests for convert_value function."""

    def test_same_type_no_conversion(self):
        assert convert_value(42, IntType, IntType) == 42

    def test_int_to_float(self):
        assert convert_value(42, IntType, FloatType) == 42.0

    def test_float_to_int(self):
        assert convert_value(3.7, FloatType, IntType) == 3

    def test_bool_to_int(self):
        assert convert_value(True, BoolType, IntType) == 1

    def test_int_to_string(self):
        assert convert_value(42, IntType, StringType) == "42"

    def test_incompatible_raises(self):
        with pytest.raises(TypeError):
            convert_value(Vector3(1, 2, 3), Vector3Type, IntType)


class TestDynamicTypeCreation:
    """Tests for dynamic type creation."""

    def test_create_object_type(self):
        ActorType = create_object_type("Actor")
        assert ActorType.type_name() == "Actor"
        assert ActorType.object_class == "Actor"

    def test_create_array_type(self):
        IntArrayType = create_array_type(IntType)
        assert "Int" in IntArrayType.type_name()
        assert IntArrayType.element_type == IntType

    def test_array_type_inherits_color(self):
        IntArrayType = create_array_type(IntType)
        assert IntArrayType.wire_color == IntType.wire_color
