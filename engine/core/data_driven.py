"""T-CC-1.7: Data-driven decorator with schema generation.

Provides @data_driven decorator that auto-generates JSON schema from
type annotations and supports runtime validation and coercion.
"""
from __future__ import annotations

import json
import re
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    Generic,
    List,
    Literal,
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


T = TypeVar('T')


class SchemaType(Enum):
    """JSON Schema types."""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


@dataclass
class SchemaProperty:
    """A property in a JSON schema."""
    name: str
    schema_type: SchemaType
    description: Optional[str] = None
    default: Any = None
    required: bool = True
    enum_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    items_schema: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None
    additional_properties: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON schema dict."""
        result: Dict[str, Any] = {"type": self.schema_type.value}

        if self.description:
            result["description"] = self.description
        if self.default is not None:
            result["default"] = self.default
        if self.enum_values:
            result["enum"] = self.enum_values
        if self.min_value is not None:
            result["minimum"] = self.min_value
        if self.max_value is not None:
            result["maximum"] = self.max_value
        if self.min_length is not None:
            result["minLength"] = self.min_length
        if self.max_length is not None:
            result["maxLength"] = self.max_length
        if self.pattern:
            result["pattern"] = self.pattern
        if self.items_schema:
            result["items"] = self.items_schema
        if self.properties:
            result["properties"] = self.properties
            result["additionalProperties"] = self.additional_properties

        return result


@dataclass
class ValidationError:
    """A validation error."""
    path: str
    message: str
    value: Any = None

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass
class ValidationResult:
    """Result of schema validation."""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)

    @classmethod
    def success(cls) -> "ValidationResult":
        return cls(valid=True)

    @classmethod
    def failure(cls, errors: List[ValidationError]) -> "ValidationResult":
        return cls(valid=False, errors=errors)

    def __bool__(self) -> bool:
        return self.valid


class TypeMapper:
    """Maps Python types to JSON Schema types."""

    _SIMPLE_TYPES: Dict[type, SchemaType] = {
        str: SchemaType.STRING,
        int: SchemaType.INTEGER,
        float: SchemaType.NUMBER,
        bool: SchemaType.BOOLEAN,
        type(None): SchemaType.NULL,
    }

    @classmethod
    def get_schema_type(cls, python_type: type) -> Optional[SchemaType]:
        """Get JSON schema type for a Python type."""
        return cls._SIMPLE_TYPES.get(python_type)

    @classmethod
    def python_type_to_schema(
        cls,
        python_type: Any,
        definitions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Convert a Python type annotation to JSON schema."""
        if definitions is None:
            definitions = {}

        origin = get_origin(python_type)
        args = get_args(python_type)

        if python_type in cls._SIMPLE_TYPES:
            return {"type": cls._SIMPLE_TYPES[python_type].value}

        if python_type is Any:
            return {}

        if origin is Union:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1 and type(None) in args:
                inner = cls.python_type_to_schema(non_none[0], definitions)
                return {"oneOf": [inner, {"type": "null"}]}
            return {"oneOf": [cls.python_type_to_schema(a, definitions) for a in args]}

        if origin is Literal:
            return {"enum": list(args)}

        if origin is list or origin is List:
            item_type = args[0] if args else Any
            return {
                "type": "array",
                "items": cls.python_type_to_schema(item_type, definitions),
            }

        if origin is set or origin is Set:
            item_type = args[0] if args else Any
            return {
                "type": "array",
                "uniqueItems": True,
                "items": cls.python_type_to_schema(item_type, definitions),
            }

        if origin is dict or origin is Dict:
            value_type = args[1] if len(args) > 1 else Any
            return {
                "type": "object",
                "additionalProperties": cls.python_type_to_schema(value_type, definitions),
            }

        if origin is tuple or origin is Tuple:
            if args:
                return {
                    "type": "array",
                    "items": [cls.python_type_to_schema(a, definitions) for a in args],
                    "minItems": len(args),
                    "maxItems": len(args),
                }
            return {"type": "array"}

        if isinstance(python_type, type) and issubclass(python_type, Enum):
            return {"enum": [e.value for e in python_type]}

        if is_dataclass(python_type):
            return cls._dataclass_to_schema(python_type, definitions)

        if isinstance(python_type, type):
            return {"type": "object"}

        if isinstance(python_type, ForwardRef):
            return {"$ref": f"#/$defs/{python_type.__forward_arg__}"}

        return {}

    @classmethod
    def _dataclass_to_schema(
        cls,
        dc_type: type,
        definitions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert a dataclass to JSON schema."""
        type_name = dc_type.__name__

        if type_name in definitions:
            return {"$ref": f"#/$defs/{type_name}"}

        definitions[type_name] = {}  # Prevent recursion

        try:
            hints = get_type_hints(dc_type)
        except Exception:
            hints = {}

        properties = {}
        required = []

        for f in fields(dc_type):
            field_type = hints.get(f.name, Any)
            prop_schema = cls.python_type_to_schema(field_type, definitions)

            has_default = f.default is not MISSING
            has_factory = f.default_factory is not MISSING

            if has_default and f.default is not None:
                prop_schema["default"] = f.default
            # Factory defaults can't be serialized

            if f.metadata:
                if "description" in f.metadata:
                    prop_schema["description"] = f.metadata["description"]
                if "min" in f.metadata:
                    prop_schema["minimum"] = f.metadata["min"]
                if "max" in f.metadata:
                    prop_schema["maximum"] = f.metadata["max"]

            properties[f.name] = prop_schema

            if not has_default and not has_factory:
                required.append(f.name)

        schema = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        definitions[type_name] = schema
        return {"$ref": f"#/$defs/{type_name}"}


class SchemaGenerator:
    """Generates JSON schemas from type annotations."""

    def __init__(self):
        self._definitions: Dict[str, Any] = {}

    def generate(self, cls: type) -> Dict[str, Any]:
        """Generate a JSON schema for a class."""
        self._definitions = {}

        if is_dataclass(cls):
            root = TypeMapper._dataclass_to_schema(cls, self._definitions)
        else:
            root = self._generate_from_annotations(cls)

        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": cls.__name__,
        }

        if "$ref" in root:
            ref_name = root["$ref"].split("/")[-1]
            if ref_name in self._definitions:
                schema.update(self._definitions[ref_name])
                del self._definitions[ref_name]
        else:
            schema.update(root)

        if self._definitions:
            schema["$defs"] = self._definitions

        return schema

    def _generate_from_annotations(self, cls: type) -> Dict[str, Any]:
        """Generate schema from class annotations."""
        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = getattr(cls, "__annotations__", {})

        properties = {}
        required = []

        for name, type_hint in hints.items():
            if name.startswith("_"):
                continue

            prop_schema = TypeMapper.python_type_to_schema(type_hint, self._definitions)
            properties[name] = prop_schema

            origin = get_origin(type_hint)
            args = get_args(type_hint)
            is_optional = origin is Union and type(None) in args

            if not is_optional and not hasattr(cls, name):
                required.append(name)

        schema = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        return schema


class SchemaValidator:
    """Validates data against JSON schemas."""

    def validate(self, data: Any, schema: Dict[str, Any]) -> ValidationResult:
        """Validate data against a schema."""
        errors = []
        self._validate_value(data, schema, "", errors, schema.get("$defs", {}))
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def _validate_value(
        self,
        value: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
        definitions: Dict[str, Any],
    ) -> None:
        """Validate a single value."""
        if "$ref" in schema:
            ref_name = schema["$ref"].split("/")[-1]
            if ref_name in definitions:
                schema = definitions[ref_name]
            else:
                return

        if "oneOf" in schema:
            valid_for_any = False
            for sub_schema in schema["oneOf"]:
                sub_errors: List[ValidationError] = []
                self._validate_value(value, sub_schema, path, sub_errors, definitions)
                if not sub_errors:
                    valid_for_any = True
                    break
            if not valid_for_any:
                errors.append(ValidationError(
                    path=path or "$",
                    message="Value does not match any of the allowed schemas",
                    value=value,
                ))
            return

        if "enum" in schema:
            if value not in schema["enum"]:
                errors.append(ValidationError(
                    path=path or "$",
                    message=f"Value must be one of {schema['enum']}",
                    value=value,
                ))
            return

        schema_type = schema.get("type")

        if schema_type == "null":
            if value is not None:
                errors.append(ValidationError(path=path or "$", message="Expected null", value=value))

        elif schema_type == "boolean":
            if not isinstance(value, bool):
                errors.append(ValidationError(path=path or "$", message="Expected boolean", value=value))

        elif schema_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(ValidationError(path=path or "$", message="Expected integer", value=value))
            else:
                self._validate_number_constraints(value, schema, path, errors)

        elif schema_type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(ValidationError(path=path or "$", message="Expected number", value=value))
            else:
                self._validate_number_constraints(value, schema, path, errors)

        elif schema_type == "string":
            if not isinstance(value, str):
                errors.append(ValidationError(path=path or "$", message="Expected string", value=value))
            else:
                self._validate_string_constraints(value, schema, path, errors)

        elif schema_type == "array":
            if not isinstance(value, (list, tuple, set)):
                errors.append(ValidationError(path=path or "$", message="Expected array", value=value))
            else:
                self._validate_array(list(value), schema, path, errors, definitions)

        elif schema_type == "object":
            if not isinstance(value, dict):
                errors.append(ValidationError(path=path or "$", message="Expected object", value=value))
            else:
                self._validate_object(value, schema, path, errors, definitions)

    def _validate_number_constraints(
        self,
        value: Union[int, float],
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
    ) -> None:
        """Validate number constraints."""
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(ValidationError(
                path=path or "$",
                message=f"Value must be >= {schema['minimum']}",
                value=value,
            ))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(ValidationError(
                path=path or "$",
                message=f"Value must be <= {schema['maximum']}",
                value=value,
            ))

    def _validate_string_constraints(
        self,
        value: str,
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
    ) -> None:
        """Validate string constraints."""
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(ValidationError(
                path=path or "$",
                message=f"String must be at least {schema['minLength']} characters",
                value=value,
            ))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(ValidationError(
                path=path or "$",
                message=f"String must be at most {schema['maxLength']} characters",
                value=value,
            ))
        if "pattern" in schema:
            if not re.match(schema["pattern"], value):
                errors.append(ValidationError(
                    path=path or "$",
                    message=f"String must match pattern {schema['pattern']}",
                    value=value,
                ))

    def _validate_array(
        self,
        value: List[Any],
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
        definitions: Dict[str, Any],
    ) -> None:
        """Validate array constraints."""
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(ValidationError(
                path=path or "$",
                message=f"Array must have at least {schema['minItems']} items",
                value=value,
            ))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(ValidationError(
                path=path or "$",
                message=f"Array must have at most {schema['maxItems']} items",
                value=value,
            ))

        if "items" in schema:
            items_schema = schema["items"]
            if isinstance(items_schema, list):
                for i, (item, item_schema) in enumerate(zip(value, items_schema)):
                    self._validate_value(item, item_schema, f"{path}[{i}]", errors, definitions)
            else:
                for i, item in enumerate(value):
                    self._validate_value(item, items_schema, f"{path}[{i}]", errors, definitions)

        if schema.get("uniqueItems") and len(value) != len(set(map(repr, value))):
            errors.append(ValidationError(
                path=path or "$",
                message="Array items must be unique",
                value=value,
            ))

    def _validate_object(
        self,
        value: Dict[str, Any],
        schema: Dict[str, Any],
        path: str,
        errors: List[ValidationError],
        definitions: Dict[str, Any],
    ) -> None:
        """Validate object constraints."""
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for req_prop in required:
            if req_prop not in value:
                errors.append(ValidationError(
                    path=f"{path}.{req_prop}" if path else req_prop,
                    message=f"Required property '{req_prop}' is missing",
                ))

        for prop_name, prop_value in value.items():
            if prop_name in properties:
                prop_path = f"{path}.{prop_name}" if path else prop_name
                self._validate_value(prop_value, properties[prop_name], prop_path, errors, definitions)


class DataDrivenMeta(type):
    """Metaclass for data-driven classes."""

    _schema_cache: Dict[type, Dict[str, Any]] = {}
    _generator = SchemaGenerator()
    _validator = SchemaValidator()

    def __new__(mcs, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any]) -> type:
        cls = super().__new__(mcs, name, bases, namespace)
        return cls

    @classmethod
    def get_schema(mcs, cls: type) -> Dict[str, Any]:
        """Get JSON schema for a class."""
        if cls not in mcs._schema_cache:
            mcs._schema_cache[cls] = mcs._generator.generate(cls)
        return mcs._schema_cache[cls]

    @classmethod
    def validate(mcs, cls: type, data: Any) -> ValidationResult:
        """Validate data against a class's schema."""
        schema = mcs.get_schema(cls)
        return mcs._validator.validate(data, schema)


def data_driven(cls: Type[T]) -> Type[T]:
    """Decorator that adds data-driven capabilities to a class.

    Provides:
    - __schema__: JSON schema generated from type annotations
    - validate(data): Validate data against the schema
    - from_dict(data): Create instance from dict with validation
    """
    generator = SchemaGenerator()
    validator = SchemaValidator()
    schema = generator.generate(cls)

    def get_schema() -> Dict[str, Any]:
        return schema

    def validate_data(data: Any) -> ValidationResult:
        return validator.validate(data, schema)

    def from_dict(data: Dict[str, Any], validate: bool = True) -> T:
        if validate:
            result = validator.validate(data, schema)
            if not result:
                raise ValueError(f"Validation failed: {result.errors}")

        if is_dataclass(cls):
            return cls(**data)
        else:
            instance = cls.__new__(cls)
            for key, value in data.items():
                setattr(instance, key, value)
            return instance

    def to_dict(instance: T) -> Dict[str, Any]:
        if is_dataclass(instance):
            return {f.name: getattr(instance, f.name) for f in fields(instance)}
        else:
            hints = get_type_hints(cls) if hasattr(cls, "__annotations__") else {}
            return {k: getattr(instance, k) for k in hints if not k.startswith("_")}

    cls.__schema__ = staticmethod(get_schema)
    cls.validate = staticmethod(validate_data)
    cls.from_dict = staticmethod(from_dict)
    cls.to_dict = to_dict

    return cls


def schema_field(
    default: Any = None,
    description: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
) -> Any:
    """Create a field with schema metadata for dataclasses."""
    metadata = {}
    if description:
        metadata["description"] = description
    if min_value is not None:
        metadata["min"] = min_value
    if max_value is not None:
        metadata["max"] = max_value
    if min_length is not None:
        metadata["minLength"] = min_length
    if max_length is not None:
        metadata["maxLength"] = max_length
    if pattern:
        metadata["pattern"] = pattern

    if default is None:
        return field(metadata=metadata)
    else:
        return field(default=default, metadata=metadata)


def export_schema(cls: type, path: Union[str, Path]) -> None:
    """Export a class's schema to a JSON file."""
    generator = SchemaGenerator()
    schema = generator.generate(cls)
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)


def load_and_validate(
    cls: Type[T],
    path: Union[str, Path],
) -> Tuple[Optional[T], ValidationResult]:
    """Load JSON file and validate against class schema."""
    generator = SchemaGenerator()
    validator = SchemaValidator()
    schema = generator.generate(cls)

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return None, ValidationResult(
            valid=False,
            errors=[ValidationError(path=str(path), message=str(e))],
        )

    result = validator.validate(data, schema)
    if not result:
        return None, result

    if is_dataclass(cls):
        instance = cls(**data)
    else:
        instance = cls.__new__(cls)
        for key, value in data.items():
            setattr(instance, key, value)

    return instance, result
