"""Tests for @data_driven decorator with schema generation (T-CC-1.7)."""
import json
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import pytest

from engine.core.data_driven import (
    DataDrivenMeta,
    SchemaGenerator,
    SchemaProperty,
    SchemaType,
    SchemaValidator,
    TypeMapper,
    ValidationError,
    ValidationResult,
    data_driven,
    export_schema,
    load_and_validate,
    schema_field,
)


class TestSchemaType:
    """Tests for SchemaType enum."""

    def test_all_types_exist(self):
        types = [
            SchemaType.STRING,
            SchemaType.NUMBER,
            SchemaType.INTEGER,
            SchemaType.BOOLEAN,
            SchemaType.ARRAY,
            SchemaType.OBJECT,
            SchemaType.NULL,
        ]
        assert len(types) == 7

    def test_type_values(self):
        assert SchemaType.STRING.value == "string"
        assert SchemaType.NUMBER.value == "number"
        assert SchemaType.INTEGER.value == "integer"


class TestSchemaProperty:
    """Tests for SchemaProperty dataclass."""

    def test_basic_property(self):
        prop = SchemaProperty(name="test", schema_type=SchemaType.STRING)
        d = prop.to_dict()
        assert d["type"] == "string"
        assert "description" not in d

    def test_property_with_description(self):
        prop = SchemaProperty(
            name="name",
            schema_type=SchemaType.STRING,
            description="The user's name",
        )
        d = prop.to_dict()
        assert d["description"] == "The user's name"

    def test_property_with_constraints(self):
        prop = SchemaProperty(
            name="age",
            schema_type=SchemaType.INTEGER,
            min_value=0,
            max_value=150,
        )
        d = prop.to_dict()
        assert d["minimum"] == 0
        assert d["maximum"] == 150

    def test_property_with_enum(self):
        prop = SchemaProperty(
            name="status",
            schema_type=SchemaType.STRING,
            enum_values=["active", "inactive"],
        )
        d = prop.to_dict()
        assert d["enum"] == ["active", "inactive"]

    def test_property_with_string_constraints(self):
        prop = SchemaProperty(
            name="email",
            schema_type=SchemaType.STRING,
            min_length=5,
            max_length=100,
            pattern=r"^[\w.-]+@[\w.-]+\.\w+$",
        )
        d = prop.to_dict()
        assert d["minLength"] == 5
        assert d["maxLength"] == 100
        assert "pattern" in d

    def test_array_property(self):
        prop = SchemaProperty(
            name="items",
            schema_type=SchemaType.ARRAY,
            items_schema={"type": "string"},
        )
        d = prop.to_dict()
        assert d["type"] == "array"
        assert d["items"] == {"type": "string"}

    def test_object_property(self):
        prop = SchemaProperty(
            name="config",
            schema_type=SchemaType.OBJECT,
            properties={"key": {"type": "string"}},
            additional_properties=False,
        )
        d = prop.to_dict()
        assert d["type"] == "object"
        assert d["additionalProperties"] is False


class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_error_str(self):
        error = ValidationError(path="$.name", message="Required field missing")
        assert str(error) == "$.name: Required field missing"

    def test_error_with_value(self):
        error = ValidationError(path="$.age", message="Must be integer", value="abc")
        assert error.value == "abc"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_success(self):
        result = ValidationResult.success()
        assert result.valid
        assert len(result.errors) == 0
        assert bool(result)

    def test_failure(self):
        errors = [ValidationError("$.x", "error")]
        result = ValidationResult.failure(errors)
        assert not result.valid
        assert len(result.errors) == 1
        assert not bool(result)


class TestTypeMapper:
    """Tests for TypeMapper."""

    def test_simple_types(self):
        assert TypeMapper.get_schema_type(str) == SchemaType.STRING
        assert TypeMapper.get_schema_type(int) == SchemaType.INTEGER
        assert TypeMapper.get_schema_type(float) == SchemaType.NUMBER
        assert TypeMapper.get_schema_type(bool) == SchemaType.BOOLEAN

    def test_str_to_schema(self):
        schema = TypeMapper.python_type_to_schema(str)
        assert schema == {"type": "string"}

    def test_int_to_schema(self):
        schema = TypeMapper.python_type_to_schema(int)
        assert schema == {"type": "integer"}

    def test_float_to_schema(self):
        schema = TypeMapper.python_type_to_schema(float)
        assert schema == {"type": "number"}

    def test_bool_to_schema(self):
        schema = TypeMapper.python_type_to_schema(bool)
        assert schema == {"type": "boolean"}

    def test_list_to_schema(self):
        schema = TypeMapper.python_type_to_schema(List[str])
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_set_to_schema(self):
        schema = TypeMapper.python_type_to_schema(Set[int])
        assert schema["type"] == "array"
        assert schema["uniqueItems"] is True
        assert schema["items"]["type"] == "integer"

    def test_dict_to_schema(self):
        schema = TypeMapper.python_type_to_schema(Dict[str, int])
        assert schema["type"] == "object"
        assert schema["additionalProperties"]["type"] == "integer"

    def test_tuple_to_schema(self):
        schema = TypeMapper.python_type_to_schema(Tuple[str, int])
        assert schema["type"] == "array"
        assert len(schema["items"]) == 2
        assert schema["minItems"] == 2
        assert schema["maxItems"] == 2

    def test_optional_to_schema(self):
        schema = TypeMapper.python_type_to_schema(Optional[str])
        assert "oneOf" in schema
        types = [s.get("type") for s in schema["oneOf"]]
        assert "string" in types
        assert "null" in types

    def test_union_to_schema(self):
        schema = TypeMapper.python_type_to_schema(Union[str, int])
        assert "oneOf" in schema
        types = [s.get("type") for s in schema["oneOf"]]
        assert "string" in types
        assert "integer" in types

    def test_literal_to_schema(self):
        schema = TypeMapper.python_type_to_schema(Literal["a", "b", "c"])
        assert schema["enum"] == ["a", "b", "c"]

    def test_enum_to_schema(self):
        class Color(Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        schema = TypeMapper.python_type_to_schema(Color)
        assert schema["enum"] == ["red", "green", "blue"]

    def test_any_to_schema(self):
        schema = TypeMapper.python_type_to_schema(Any)
        assert schema == {}

    def test_nested_list(self):
        schema = TypeMapper.python_type_to_schema(List[List[int]])
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "array"
        assert schema["items"]["items"]["type"] == "integer"


class TestSchemaGenerator:
    """Tests for SchemaGenerator."""

    def test_generate_simple_dataclass(self):
        @dataclass
        class Person:
            name: str
            age: int

        generator = SchemaGenerator()
        schema = generator.generate(Person)

        assert schema["title"] == "Person"
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["age"]["type"] == "integer"

    def test_generate_with_optional(self):
        @dataclass
        class Config:
            name: str
            debug: Optional[bool] = None

        generator = SchemaGenerator()
        schema = generator.generate(Config)

        assert "name" in schema.get("required", [])
        assert "debug" not in schema.get("required", [])

    def test_generate_nested_dataclass(self):
        @dataclass
        class Address:
            street: str
            city: str

        @dataclass
        class Person:
            name: str
            address: Address

        generator = SchemaGenerator()
        schema = generator.generate(Person)

        assert "$defs" in schema or "address" in schema.get("properties", {})

    def test_generate_with_list_field(self):
        @dataclass
        class Team:
            name: str
            members: List[str]

        generator = SchemaGenerator()
        schema = generator.generate(Team)

        assert schema["properties"]["members"]["type"] == "array"

    def test_generate_from_class(self):
        class Config:
            name: str
            value: int

        generator = SchemaGenerator()
        schema = generator.generate(Config)

        assert schema["type"] == "object"
        assert "name" in schema["properties"]


class TestSchemaValidator:
    """Tests for SchemaValidator."""

    def test_validate_string(self):
        validator = SchemaValidator()
        schema = {"type": "string"}

        assert validator.validate("hello", schema).valid
        assert not validator.validate(123, schema).valid

    def test_validate_integer(self):
        validator = SchemaValidator()
        schema = {"type": "integer"}

        assert validator.validate(42, schema).valid
        assert not validator.validate(3.14, schema).valid
        assert not validator.validate("42", schema).valid

    def test_validate_number(self):
        validator = SchemaValidator()
        schema = {"type": "number"}

        assert validator.validate(42, schema).valid
        assert validator.validate(3.14, schema).valid
        assert not validator.validate("42", schema).valid

    def test_validate_boolean(self):
        validator = SchemaValidator()
        schema = {"type": "boolean"}

        assert validator.validate(True, schema).valid
        assert validator.validate(False, schema).valid
        assert not validator.validate(1, schema).valid

    def test_validate_null(self):
        validator = SchemaValidator()
        schema = {"type": "null"}

        assert validator.validate(None, schema).valid
        assert not validator.validate("null", schema).valid

    def test_validate_array(self):
        validator = SchemaValidator()
        schema = {"type": "array", "items": {"type": "string"}}

        assert validator.validate(["a", "b"], schema).valid
        assert not validator.validate([1, 2], schema).valid

    def test_validate_object(self):
        validator = SchemaValidator()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        assert validator.validate({"name": "test"}, schema).valid
        assert not validator.validate({}, schema).valid

    def test_validate_enum(self):
        validator = SchemaValidator()
        schema = {"enum": ["a", "b", "c"]}

        assert validator.validate("a", schema).valid
        assert not validator.validate("d", schema).valid

    def test_validate_minimum(self):
        validator = SchemaValidator()
        schema = {"type": "integer", "minimum": 0}

        assert validator.validate(5, schema).valid
        assert not validator.validate(-1, schema).valid

    def test_validate_maximum(self):
        validator = SchemaValidator()
        schema = {"type": "integer", "maximum": 100}

        assert validator.validate(50, schema).valid
        assert not validator.validate(150, schema).valid

    def test_validate_min_length(self):
        validator = SchemaValidator()
        schema = {"type": "string", "minLength": 3}

        assert validator.validate("hello", schema).valid
        assert not validator.validate("hi", schema).valid

    def test_validate_max_length(self):
        validator = SchemaValidator()
        schema = {"type": "string", "maxLength": 5}

        assert validator.validate("hello", schema).valid
        assert not validator.validate("hello world", schema).valid

    def test_validate_pattern(self):
        validator = SchemaValidator()
        schema = {"type": "string", "pattern": "^[a-z]+$"}

        assert validator.validate("hello", schema).valid
        assert not validator.validate("Hello", schema).valid

    def test_validate_min_items(self):
        validator = SchemaValidator()
        schema = {"type": "array", "minItems": 2}

        assert validator.validate([1, 2, 3], schema).valid
        assert not validator.validate([1], schema).valid

    def test_validate_max_items(self):
        validator = SchemaValidator()
        schema = {"type": "array", "maxItems": 3}

        assert validator.validate([1, 2], schema).valid
        assert not validator.validate([1, 2, 3, 4], schema).valid

    def test_validate_unique_items(self):
        validator = SchemaValidator()
        schema = {"type": "array", "uniqueItems": True}

        assert validator.validate([1, 2, 3], schema).valid
        assert not validator.validate([1, 1, 2], schema).valid

    def test_validate_one_of(self):
        validator = SchemaValidator()
        schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}

        assert validator.validate("hello", schema).valid
        assert validator.validate(42, schema).valid
        assert not validator.validate(3.14, schema).valid

    def test_validate_nested(self):
        validator = SchemaValidator()
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
        }

        assert validator.validate({"items": [1, 2, 3]}, schema).valid
        result = validator.validate({"items": ["a", "b"]}, schema)
        assert not result.valid
        assert "items[0]" in result.errors[0].path


class TestDataDrivenDecorator:
    """Tests for @data_driven decorator."""

    def test_decorator_adds_schema(self):
        @data_driven
        @dataclass
        class Config:
            name: str
            value: int

        schema = Config.__schema__()
        assert schema["title"] == "Config"
        assert "name" in schema["properties"]

    def test_decorator_adds_validate(self):
        @data_driven
        @dataclass
        class Config:
            name: str
            value: int

        result = Config.validate({"name": "test", "value": 42})
        assert result.valid

        result = Config.validate({"name": "test"})
        assert not result.valid

    def test_decorator_adds_from_dict(self):
        @data_driven
        @dataclass
        class Person:
            name: str
            age: int

        person = Person.from_dict({"name": "Alice", "age": 30})
        assert person.name == "Alice"
        assert person.age == 30

    def test_from_dict_validates(self):
        @data_driven
        @dataclass
        class Person:
            name: str
            age: int

        with pytest.raises(ValueError):
            Person.from_dict({"name": "Alice"})  # Missing age

    def test_from_dict_skip_validation(self):
        @data_driven
        @dataclass
        class Person:
            name: str
            age: int = 0

        person = Person.from_dict({"name": "Alice"}, validate=False)
        assert person.name == "Alice"

    def test_decorator_adds_to_dict(self):
        @data_driven
        @dataclass
        class Person:
            name: str
            age: int

        person = Person(name="Alice", age=30)
        d = person.to_dict()
        assert d == {"name": "Alice", "age": 30}

    def test_decorator_on_regular_class(self):
        @data_driven
        class Config:
            name: str
            value: int

        schema = Config.__schema__()
        assert "name" in schema["properties"]

    def test_decorator_preserves_class(self):
        @data_driven
        @dataclass
        class Original:
            x: int

        assert Original.__name__ == "Original"
        obj = Original(x=5)
        assert obj.x == 5


class TestSchemaField:
    """Tests for schema_field helper."""

    def test_field_with_description(self):
        @dataclass
        class Config:
            name: str = schema_field(description="The config name")

        f = [field for field in Config.__dataclass_fields__.values()][0]
        assert f.metadata.get("description") == "The config name"

    def test_field_with_constraints(self):
        @dataclass
        class Config:
            age: int = schema_field(min_value=0, max_value=150)

        f = Config.__dataclass_fields__["age"]
        assert f.metadata.get("min") == 0
        assert f.metadata.get("max") == 150

    def test_field_with_default(self):
        @dataclass
        class Config:
            debug: bool = schema_field(default=False, description="Debug mode")

        config = Config()
        assert config.debug is False


class TestExportSchema:
    """Tests for export_schema function."""

    def test_export_to_file(self):
        @dataclass
        class Config:
            name: str
            value: int

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "schema.json"
            export_schema(Config, path)

            assert path.exists()
            with open(path) as f:
                schema = json.load(f)
            assert schema["title"] == "Config"


class TestLoadAndValidate:
    """Tests for load_and_validate function."""

    def test_load_valid_json(self):
        @dataclass
        class Config:
            name: str
            value: int

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text('{"name": "test", "value": 42}')

            instance, result = load_and_validate(Config, path)
            assert result.valid
            assert instance.name == "test"
            assert instance.value == 42

    def test_load_invalid_json(self):
        @dataclass
        class Config:
            name: str
            value: int

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text('{"name": "test"}')  # Missing value

            instance, result = load_and_validate(Config, path)
            assert not result.valid
            assert instance is None

    def test_load_missing_file(self):
        @dataclass
        class Config:
            name: str

        instance, result = load_and_validate(Config, "/nonexistent.json")
        assert not result.valid
        assert instance is None

    def test_load_malformed_json(self):
        @dataclass
        class Config:
            name: str

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text("{invalid json}")

            instance, result = load_and_validate(Config, path)
            assert not result.valid
            assert instance is None


class TestComplexScenarios:
    """Complex integration tests."""

    def test_deeply_nested_structure(self):
        @dataclass
        class Coord:
            x: float
            y: float

        @dataclass
        class Bounds:
            min: Coord
            max: Coord

        @dataclass
        class Region:
            name: str
            bounds: Bounds

        generator = SchemaGenerator()
        schema = generator.generate(Region)

        assert "$defs" in schema or "bounds" in schema["properties"]

    def test_recursive_type_hint(self):
        @dataclass
        class Node:
            value: int
            children: List["Node"] = field(default_factory=list)

        generator = SchemaGenerator()
        schema = generator.generate(Node)
        assert "value" in schema.get("properties", {})

    def test_complex_union(self):
        @data_driven
        @dataclass
        class Response:
            status: Literal["ok", "error"]
            data: Optional[Dict[str, Any]] = None
            error: Optional[str] = None

        result = Response.validate({"status": "ok", "data": {"x": 1}})
        assert result.valid

        result = Response.validate({"status": "invalid"})
        assert not result.valid

    def test_enum_field(self):
        class Priority(Enum):
            LOW = 1
            MEDIUM = 2
            HIGH = 3

        @data_driven
        @dataclass
        class Task:
            name: str
            priority: Priority

        schema = Task.__schema__()
        # Enums are serialized by value
        assert schema["properties"]["priority"]["enum"] == [1, 2, 3]

    def test_validation_error_paths(self):
        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            inner: Inner
            items: List[int]

        validator = SchemaValidator()
        generator = SchemaGenerator()
        schema = generator.generate(Outer)

        data = {
            "inner": {"value": "not-int"},
            "items": [1, "bad", 3],
        }

        result = validator.validate(data, schema)
        assert not result.valid
        paths = [e.path for e in result.errors]
        assert any("inner" in p for p in paths) or any("value" in p for p in paths)


class TestDataDrivenMeta:
    """Tests for DataDrivenMeta metaclass."""

    def test_get_schema(self):
        @dataclass
        class Config:
            name: str

        schema = DataDrivenMeta.get_schema(Config)
        assert schema["title"] == "Config"

    def test_schema_cached(self):
        @dataclass
        class Config:
            name: str

        schema1 = DataDrivenMeta.get_schema(Config)
        schema2 = DataDrivenMeta.get_schema(Config)
        assert schema1 is schema2

    def test_validate_via_meta(self):
        @dataclass
        class Config:
            name: str
            value: int

        result = DataDrivenMeta.validate(Config, {"name": "test", "value": 42})
        assert result.valid


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_dataclass(self):
        @dataclass
        class Empty:
            pass

        generator = SchemaGenerator()
        schema = generator.generate(Empty)
        assert schema["type"] == "object"

    def test_class_with_methods(self):
        @data_driven
        class WithMethods:
            name: str

            def greet(self) -> str:
                return f"Hello, {self.name}"

        schema = WithMethods.__schema__()
        assert "greet" not in schema.get("properties", {})

    def test_private_fields_excluded(self):
        @data_driven
        class Config:
            name: str
            _internal: int = 0

        schema = Config.__schema__()
        assert "_internal" not in schema.get("properties", {})

    def test_default_factory(self):
        @dataclass
        class Config:
            items: List[str] = field(default_factory=list)

        generator = SchemaGenerator()
        schema = generator.generate(Config)
        assert "items" not in schema.get("required", [])

    def test_tuple_validation(self):
        validator = SchemaValidator()
        schema = {
            "type": "array",
            "items": [{"type": "string"}, {"type": "integer"}],
            "minItems": 2,
            "maxItems": 2,
        }

        assert validator.validate(["hello", 42], schema).valid
        assert not validator.validate(["hello"], schema).valid
        assert not validator.validate([42, "hello"], schema).valid
