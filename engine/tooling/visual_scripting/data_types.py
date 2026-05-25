"""
FlowForge Data Types - Blueprint data types with type conversion and validation.

Provides the type system for visual scripting with:
- Primitive types (Bool, Int, Float, String)
- Complex types (Vector2, Vector3, Rotator, Transform)
- Object types (Object, Actor, Component references)
- Collection types (Array, Map, Set)
- Type conversion and validation
- Color coding for wire visualization
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)


class DataTypeCategory(Enum):
    """Categories of blueprint data types."""
    PRIMITIVE = auto()
    NUMERIC = auto()
    STRING = auto()
    VECTOR = auto()
    TRANSFORM = auto()
    OBJECT = auto()
    COLLECTION = auto()
    STRUCT = auto()
    ENUM = auto()
    WILDCARD = auto()
    EXECUTION = auto()


@dataclass(frozen=True)
class TypeColor:
    """RGB color for wire visualization."""
    r: int
    g: int
    b: int
    a: int = 255

    def to_hex(self) -> str:
        """Convert to hex string."""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to RGBA tuple."""
        return (self.r, self.g, self.b, self.a)


# Standard wire colors
class WireColors:
    """Standard colors for data type wires."""
    EXECUTION = TypeColor(255, 255, 255)  # White
    BOOL = TypeColor(139, 0, 0)  # Dark red
    INT = TypeColor(0, 230, 180)  # Cyan
    FLOAT = TypeColor(0, 200, 80)  # Green
    STRING = TypeColor(255, 0, 200)  # Magenta
    VECTOR = TypeColor(255, 200, 0)  # Gold/Yellow
    ROTATOR = TypeColor(150, 130, 255)  # Lavender
    TRANSFORM = TypeColor(255, 140, 0)  # Orange
    OBJECT = TypeColor(0, 160, 255)  # Blue
    STRUCT = TypeColor(0, 80, 180)  # Dark blue
    ENUM = TypeColor(0, 100, 0)  # Dark green
    ARRAY = TypeColor(100, 100, 100)  # Gray (inner color varies)
    WILDCARD = TypeColor(200, 200, 200)  # Light gray


class BlueprintType(ABC):
    """Base class for all blueprint data types."""

    category: DataTypeCategory = DataTypeCategory.PRIMITIVE
    wire_color: TypeColor = WireColors.WILDCARD
    default_value: Any = None

    @classmethod
    @abstractmethod
    def type_name(cls) -> str:
        """Return the display name of this type."""
        pass

    @classmethod
    @abstractmethod
    def validate(cls, value: Any) -> bool:
        """Check if a value is valid for this type."""
        pass

    @classmethod
    @abstractmethod
    def coerce(cls, value: Any) -> Any:
        """Attempt to convert a value to this type."""
        pass

    @classmethod
    def can_convert_from(cls, other_type: Type[BlueprintType]) -> bool:
        """Check if conversion from another type is possible."""
        return False

    @classmethod
    def convert_from(cls, value: Any, source_type: Type[BlueprintType]) -> Any:
        """Convert a value from another type."""
        if cls.can_convert_from(source_type):
            return cls.coerce(value)
        raise TypeError(f"Cannot convert from {source_type.type_name()} to {cls.type_name()}")


class BoolType(BlueprintType):
    """Boolean type (true/false)."""

    category = DataTypeCategory.PRIMITIVE
    wire_color = WireColors.BOOL
    default_value = False

    @classmethod
    def type_name(cls) -> str:
        return "Bool"

    @classmethod
    def validate(cls, value: Any) -> bool:
        return isinstance(value, bool)

    @classmethod
    def coerce(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                return True
            if value.lower() in ("false", "0", "no", ""):
                return False
        return bool(value)

    @classmethod
    def can_convert_from(cls, other_type: Type[BlueprintType]) -> bool:
        return other_type in (IntType, FloatType, StringType)


class IntType(BlueprintType):
    """Integer type."""

    category = DataTypeCategory.NUMERIC
    wire_color = WireColors.INT
    default_value = 0

    @classmethod
    def type_name(cls) -> str:
        return "Int"

    @classmethod
    def validate(cls, value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    @classmethod
    def coerce(cls, value: Any) -> int:
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                return 0
        return 0

    @classmethod
    def can_convert_from(cls, other_type: Type[BlueprintType]) -> bool:
        return other_type in (BoolType, FloatType, StringType)


class FloatType(BlueprintType):
    """Floating-point number type."""

    category = DataTypeCategory.NUMERIC
    wire_color = WireColors.FLOAT
    default_value = 0.0

    @classmethod
    def type_name(cls) -> str:
        return "Float"

    @classmethod
    def validate(cls, value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @classmethod
    def coerce(cls, value: Any) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    @classmethod
    def can_convert_from(cls, other_type: Type[BlueprintType]) -> bool:
        return other_type in (BoolType, IntType, StringType)


class StringType(BlueprintType):
    """String/text type."""

    category = DataTypeCategory.STRING
    wire_color = WireColors.STRING
    default_value = ""

    @classmethod
    def type_name(cls) -> str:
        return "String"

    @classmethod
    def validate(cls, value: Any) -> bool:
        return isinstance(value, str)

    @classmethod
    def coerce(cls, value: Any) -> str:
        if isinstance(value, str):
            return value
        return str(value)

    @classmethod
    def can_convert_from(cls, other_type: Type[BlueprintType]) -> bool:
        # String can convert from almost any type
        return True


@dataclass
class Vector2:
    """2D vector value."""
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: Vector2) -> Vector2:
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2) -> Vector2:
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vector2:
        return Vector2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> Vector2:
        return Vector2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> Vector2:
        return Vector2(-self.x, -self.y)

    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalized(self) -> Vector2:
        mag = self.magnitude()
        if mag == 0:
            return Vector2(0, 0)
        return self / mag

    def dot(self, other: Vector2) -> float:
        return self.x * other.x + self.y * other.y

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Vector3:
    """3D vector value."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> Vector3:
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> Vector3:
        return Vector3(-self.x, -self.y, -self.z)

    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> Vector3:
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return self / mag

    def dot(self, other: Vector3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class Vector2Type(BlueprintType):
    """2D vector type."""

    category = DataTypeCategory.VECTOR
    wire_color = WireColors.VECTOR
    default_value = Vector2(0.0, 0.0)

    @classmethod
    def type_name(cls) -> str:
        return "Vector2"

    @classmethod
    def validate(cls, value: Any) -> bool:
        return isinstance(value, Vector2)

    @classmethod
    def coerce(cls, value: Any) -> Vector2:
        if isinstance(value, Vector2):
            return value
        if isinstance(value, Vector3):
            return Vector2(value.x, value.y)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return Vector2(float(value[0]), float(value[1]))
        if isinstance(value, dict):
            return Vector2(float(value.get("x", 0)), float(value.get("y", 0)))
        return Vector2(0.0, 0.0)

    @classmethod
    def can_convert_from(cls, other_type: Type[BlueprintType]) -> bool:
        return other_type == Vector3Type


class Vector3Type(BlueprintType):
    """3D vector type."""

    category = DataTypeCategory.VECTOR
    wire_color = WireColors.VECTOR
    default_value = Vector3(0.0, 0.0, 0.0)

    @classmethod
    def type_name(cls) -> str:
        return "Vector3"

    @classmethod
    def validate(cls, value: Any) -> bool:
        return isinstance(value, Vector3)

    @classmethod
    def coerce(cls, value: Any) -> Vector3:
        if isinstance(value, Vector3):
            return value
        if isinstance(value, Vector2):
            return Vector3(value.x, value.y, 0.0)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return Vector3(float(value[0]), float(value[1]), float(value[2]))
        if isinstance(value, dict):
            return Vector3(
                float(value.get("x", 0)),
                float(value.get("y", 0)),
                float(value.get("z", 0))
            )
        return Vector3(0.0, 0.0, 0.0)

    @classmethod
    def can_convert_from(cls, other_type: Type[BlueprintType]) -> bool:
        return other_type == Vector2Type


@dataclass
class Rotator:
    """Rotation represented as pitch, yaw, roll in degrees."""
    pitch: float = 0.0  # Rotation around Y axis
    yaw: float = 0.0    # Rotation around Z axis
    roll: float = 0.0   # Rotation around X axis

    def to_radians(self) -> Tuple[float, float, float]:
        """Convert to radians."""
        return (
            math.radians(self.pitch),
            math.radians(self.yaw),
            math.radians(self.roll)
        )

    @classmethod
    def from_radians(cls, pitch: float, yaw: float, roll: float) -> Rotator:
        """Create from radians."""
        return cls(
            math.degrees(pitch),
            math.degrees(yaw),
            math.degrees(roll)
        )

    def __add__(self, other: Rotator) -> Rotator:
        return Rotator(
            self.pitch + other.pitch,
            self.yaw + other.yaw,
            self.roll + other.roll
        )

    def __sub__(self, other: Rotator) -> Rotator:
        return Rotator(
            self.pitch - other.pitch,
            self.yaw - other.yaw,
            self.roll - other.roll
        )


class RotatorType(BlueprintType):
    """Rotation type (pitch, yaw, roll)."""

    category = DataTypeCategory.VECTOR
    wire_color = WireColors.ROTATOR
    default_value = Rotator(0.0, 0.0, 0.0)

    @classmethod
    def type_name(cls) -> str:
        return "Rotator"

    @classmethod
    def validate(cls, value: Any) -> bool:
        return isinstance(value, Rotator)

    @classmethod
    def coerce(cls, value: Any) -> Rotator:
        if isinstance(value, Rotator):
            return value
        if isinstance(value, Vector3):
            return Rotator(value.x, value.y, value.z)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return Rotator(float(value[0]), float(value[1]), float(value[2]))
        if isinstance(value, dict):
            return Rotator(
                float(value.get("pitch", 0)),
                float(value.get("yaw", 0)),
                float(value.get("roll", 0))
            )
        return Rotator(0.0, 0.0, 0.0)

    @classmethod
    def can_convert_from(cls, other_type: Type[BlueprintType]) -> bool:
        return other_type == Vector3Type


@dataclass
class Transform:
    """Transform with location, rotation, and scale."""
    location: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    rotation: Rotator = field(default_factory=lambda: Rotator(0, 0, 0))
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))

    def get_forward_vector(self) -> Vector3:
        """Get the forward direction vector."""
        yaw_rad = math.radians(self.rotation.yaw)
        pitch_rad = math.radians(self.rotation.pitch)
        return Vector3(
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad)
        )

    def get_right_vector(self) -> Vector3:
        """Get the right direction vector."""
        yaw_rad = math.radians(self.rotation.yaw + 90)
        return Vector3(math.cos(yaw_rad), math.sin(yaw_rad), 0)


class TransformType(BlueprintType):
    """Transform type (location, rotation, scale)."""

    category = DataTypeCategory.TRANSFORM
    wire_color = WireColors.TRANSFORM
    default_value = Transform()

    @classmethod
    def type_name(cls) -> str:
        return "Transform"

    @classmethod
    def validate(cls, value: Any) -> bool:
        return isinstance(value, Transform)

    @classmethod
    def coerce(cls, value: Any) -> Transform:
        if isinstance(value, Transform):
            return value
        if isinstance(value, dict):
            loc = value.get("location", Vector3())
            if not isinstance(loc, Vector3):
                loc = Vector3Type.coerce(loc)
            rot = value.get("rotation", Rotator())
            if not isinstance(rot, Rotator):
                rot = RotatorType.coerce(rot)
            scale = value.get("scale", Vector3(1, 1, 1))
            if not isinstance(scale, Vector3):
                scale = Vector3Type.coerce(scale)
            return Transform(loc, rot, scale)
        return Transform()


@dataclass
class ObjectRef:
    """Reference to a game object."""
    object_id: int = 0
    object_type: str = "Object"
    is_valid: bool = False

    def __bool__(self) -> bool:
        return self.is_valid and self.object_id != 0


class ObjectType(BlueprintType):
    """Object reference type."""

    category = DataTypeCategory.OBJECT
    wire_color = WireColors.OBJECT
    default_value = ObjectRef()
    object_class: str = "Object"

    @classmethod
    def type_name(cls) -> str:
        return cls.object_class

    @classmethod
    def validate(cls, value: Any) -> bool:
        if not isinstance(value, ObjectRef):
            return False
        if cls.object_class != "Object":
            return value.object_type == cls.object_class
        return True

    @classmethod
    def coerce(cls, value: Any) -> ObjectRef:
        if isinstance(value, ObjectRef):
            return value
        if isinstance(value, int):
            return ObjectRef(object_id=value, object_type=cls.object_class, is_valid=value != 0)
        if value is None:
            return ObjectRef()
        return ObjectRef()

    @classmethod
    def can_convert_from(cls, other_type: Type[BlueprintType]) -> bool:
        if not issubclass(other_type, ObjectType):
            return False
        # Can convert from derived types (Actor -> Object, etc.)
        return True


def create_object_type(class_name: str) -> Type[ObjectType]:
    """Create a new object type for a specific class."""
    return type(
        f"{class_name}Type",
        (ObjectType,),
        {"object_class": class_name}
    )


# Common object types
ActorType = create_object_type("Actor")
ComponentType = create_object_type("Component")
WidgetType = create_object_type("Widget")


T = TypeVar("T")


@dataclass
class ArrayValue(Generic[T]):
    """Array value container."""
    items: List[T] = field(default_factory=list)
    element_type: Type[BlueprintType] = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]

    def __setitem__(self, index: int, value: T) -> None:
        self.items[index] = value

    def append(self, value: T) -> None:
        self.items.append(value)

    def remove(self, value: T) -> None:
        self.items.remove(value)

    def clear(self) -> None:
        self.items.clear()

    def __iter__(self):
        return iter(self.items)


class ArrayType(BlueprintType):
    """Array/list type (generic container)."""

    category = DataTypeCategory.COLLECTION
    wire_color = WireColors.ARRAY
    default_value = ArrayValue()
    element_type: Type[BlueprintType] = None

    @classmethod
    def type_name(cls) -> str:
        if cls.element_type:
            return f"Array<{cls.element_type.type_name()}>"
        return "Array"

    @classmethod
    def validate(cls, value: Any) -> bool:
        if not isinstance(value, (ArrayValue, list)):
            return False
        if cls.element_type and isinstance(value, ArrayValue):
            return value.element_type == cls.element_type
        return True

    @classmethod
    def coerce(cls, value: Any) -> ArrayValue:
        if isinstance(value, ArrayValue):
            return value
        if isinstance(value, (list, tuple)):
            arr = ArrayValue(element_type=cls.element_type)
            for item in value:
                if cls.element_type:
                    arr.append(cls.element_type.coerce(item))
                else:
                    arr.append(item)
            return arr
        return ArrayValue(element_type=cls.element_type)


def create_array_type(element_type: Type[BlueprintType]) -> Type[ArrayType]:
    """Create a new array type for a specific element type."""
    # Use element type's wire color for inner color
    return type(
        f"ArrayOf{element_type.type_name()}Type",
        (ArrayType,),
        {"element_type": element_type, "wire_color": element_type.wire_color}
    )


@dataclass
class MapValue(Generic[T]):
    """Map/dictionary value container."""
    items: Dict[str, T] = field(default_factory=dict)
    key_type: Type[BlueprintType] = StringType
    value_type: Type[BlueprintType] = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, key: str) -> T:
        return self.items[key]

    def __setitem__(self, key: str, value: T) -> None:
        self.items[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.items

    def keys(self):
        return self.items.keys()

    def values(self):
        return self.items.values()

    def get(self, key: str, default: T = None) -> T:
        return self.items.get(key, default)


class MapType(BlueprintType):
    """Map/dictionary type."""

    category = DataTypeCategory.COLLECTION
    wire_color = WireColors.STRUCT
    default_value = MapValue()
    key_type: Type[BlueprintType] = StringType
    value_type: Type[BlueprintType] = None

    @classmethod
    def type_name(cls) -> str:
        key_name = cls.key_type.type_name() if cls.key_type else "String"
        val_name = cls.value_type.type_name() if cls.value_type else "Any"
        return f"Map<{key_name}, {val_name}>"

    @classmethod
    def validate(cls, value: Any) -> bool:
        return isinstance(value, (MapValue, dict))

    @classmethod
    def coerce(cls, value: Any) -> MapValue:
        if isinstance(value, MapValue):
            return value
        if isinstance(value, dict):
            map_val = MapValue(key_type=cls.key_type, value_type=cls.value_type)
            for k, v in value.items():
                key = cls.key_type.coerce(k) if cls.key_type else str(k)
                val = cls.value_type.coerce(v) if cls.value_type else v
                map_val[key] = val
            return map_val
        return MapValue(key_type=cls.key_type, value_type=cls.value_type)


@dataclass
class SetValue(Generic[T]):
    """Set value container."""
    items: Set[T] = field(default_factory=set)
    element_type: Type[BlueprintType] = None

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, item: T) -> bool:
        return item in self.items

    def add(self, item: T) -> None:
        self.items.add(item)

    def remove(self, item: T) -> None:
        self.items.discard(item)

    def __iter__(self):
        return iter(self.items)


class SetType(BlueprintType):
    """Set type (unique elements)."""

    category = DataTypeCategory.COLLECTION
    wire_color = WireColors.STRUCT
    default_value = SetValue()
    element_type: Type[BlueprintType] = None

    @classmethod
    def type_name(cls) -> str:
        if cls.element_type:
            return f"Set<{cls.element_type.type_name()}>"
        return "Set"

    @classmethod
    def validate(cls, value: Any) -> bool:
        return isinstance(value, (SetValue, set, frozenset))

    @classmethod
    def coerce(cls, value: Any) -> SetValue:
        if isinstance(value, SetValue):
            return value
        if isinstance(value, (set, frozenset, list, tuple)):
            set_val = SetValue(element_type=cls.element_type)
            for item in value:
                if cls.element_type:
                    set_val.add(cls.element_type.coerce(item))
                else:
                    set_val.add(item)
            return set_val
        return SetValue(element_type=cls.element_type)


class ExecutionType(BlueprintType):
    """Execution wire type (flow control)."""

    category = DataTypeCategory.EXECUTION
    wire_color = WireColors.EXECUTION
    default_value = None

    @classmethod
    def type_name(cls) -> str:
        return "Exec"

    @classmethod
    def validate(cls, value: Any) -> bool:
        return True  # Execution pins don't carry data

    @classmethod
    def coerce(cls, value: Any) -> None:
        return None


class WildcardType(BlueprintType):
    """Wildcard type that can match any type."""

    category = DataTypeCategory.WILDCARD
    wire_color = WireColors.WILDCARD
    default_value = None

    @classmethod
    def type_name(cls) -> str:
        return "Wildcard"

    @classmethod
    def validate(cls, value: Any) -> bool:
        return True

    @classmethod
    def coerce(cls, value: Any) -> Any:
        return value

    @classmethod
    def can_convert_from(cls, other_type: Type[BlueprintType]) -> bool:
        return True


# Type registry for lookup by name
TYPE_REGISTRY: Dict[str, Type[BlueprintType]] = {
    "Bool": BoolType,
    "Int": IntType,
    "Float": FloatType,
    "String": StringType,
    "Vector2": Vector2Type,
    "Vector3": Vector3Type,
    "Rotator": RotatorType,
    "Transform": TransformType,
    "Object": ObjectType,
    "Actor": ActorType,
    "Component": ComponentType,
    "Widget": WidgetType,
    "Array": ArrayType,
    "Map": MapType,
    "Set": SetType,
    "Exec": ExecutionType,
    "Wildcard": WildcardType,
}


def get_type_by_name(name: str) -> Optional[Type[BlueprintType]]:
    """Get a blueprint type by its name."""
    return TYPE_REGISTRY.get(name)


def register_type(bp_type: Type[BlueprintType]) -> None:
    """Register a new blueprint type."""
    TYPE_REGISTRY[bp_type.type_name()] = bp_type


def can_connect_types(
    source_type: Type[BlueprintType],
    target_type: Type[BlueprintType]
) -> bool:
    """Check if two types can be connected (compatible)."""
    # Same type is always compatible
    if source_type == target_type:
        return True

    # Wildcard accepts anything
    if target_type == WildcardType or source_type == WildcardType:
        return True

    # Check if target can convert from source
    if target_type.can_convert_from(source_type):
        return True

    # Object type inheritance
    if issubclass(source_type, ObjectType) and issubclass(target_type, ObjectType):
        return True

    return False


def convert_value(
    value: Any,
    source_type: Type[BlueprintType],
    target_type: Type[BlueprintType]
) -> Any:
    """Convert a value from one type to another."""
    if source_type == target_type:
        return value

    if not can_connect_types(source_type, target_type):
        raise TypeError(
            f"Cannot convert from {source_type.type_name()} to {target_type.type_name()}"
        )

    return target_type.coerce(value)
