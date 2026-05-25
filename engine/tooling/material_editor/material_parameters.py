"""Material parameters - exposed parameters for material instances."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import copy


class ParameterType(Enum):
    """Type of material parameter."""
    SCALAR = auto()
    VECTOR2 = auto()
    VECTOR3 = auto()
    VECTOR4 = auto()
    COLOR = auto()
    TEXTURE = auto()
    BOOLEAN = auto()
    INTEGER = auto()


class ParameterSemantics(Enum):
    """Semantic hints for parameters."""
    NONE = auto()
    COLOR_RGB = auto()
    COLOR_RGBA = auto()
    NORMAL = auto()
    POSITION = auto()
    UV = auto()
    ROUGHNESS = auto()
    METALLIC = auto()
    ALBEDO = auto()
    EMISSIVE = auto()
    AO = auto()
    HEIGHT = auto()


@dataclass
class ParameterRange:
    """Range constraints for a parameter."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    soft_min: Optional[float] = None
    soft_max: Optional[float] = None

    def clamp(self, value: float) -> float:
        """Clamp value to range."""
        if self.min_value is not None and value < self.min_value:
            return self.min_value
        if self.max_value is not None and value > self.max_value:
            return self.max_value
        return value

    def is_valid(self, value: float) -> bool:
        """Check if value is within range."""
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


@dataclass
class TextureSettings:
    """Settings for texture parameters."""
    default_path: str = ""
    filter_mode: str = "linear"
    address_mode: str = "wrap"
    srgb: bool = True
    generate_mips: bool = True
    max_size: int = 4096


class MaterialParameter(ABC):
    """Base class for material parameters."""

    def __init__(
        self,
        name: str,
        display_name: Optional[str] = None,
        group: str = "Default",
        description: str = "",
        semantics: ParameterSemantics = ParameterSemantics.NONE,
        visible: bool = True,
        animatable: bool = False
    ):
        self._name = name
        self._display_name = display_name or name
        self._group = group
        self._description = description
        self._semantics = semantics
        self._visible = visible
        self._animatable = animatable
        self._binding: Optional[str] = None

    @property
    def name(self) -> str:
        """Get parameter name."""
        return self._name

    @property
    def display_name(self) -> str:
        """Get display name."""
        return self._display_name

    @property
    def group(self) -> str:
        """Get parameter group."""
        return self._group

    @property
    def description(self) -> str:
        """Get description."""
        return self._description

    @property
    def semantics(self) -> ParameterSemantics:
        """Get semantics."""
        return self._semantics

    @property
    def visible(self) -> bool:
        """Check if parameter is visible in UI."""
        return self._visible

    @property
    def animatable(self) -> bool:
        """Check if parameter can be animated."""
        return self._animatable

    @property
    def binding(self) -> Optional[str]:
        """Get shader binding name."""
        return self._binding

    @binding.setter
    def binding(self, value: str) -> None:
        """Set shader binding name."""
        self._binding = value

    @property
    @abstractmethod
    def parameter_type(self) -> ParameterType:
        """Get parameter type."""
        pass

    @property
    @abstractmethod
    def value(self) -> Any:
        """Get current value."""
        pass

    @value.setter
    @abstractmethod
    def value(self, new_value: Any) -> None:
        """Set current value."""
        pass

    @property
    @abstractmethod
    def default_value(self) -> Any:
        """Get default value."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset to default value."""
        pass

    @abstractmethod
    def clone(self) -> 'MaterialParameter':
        """Create a deep copy of this parameter."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self._name,
            "display_name": self._display_name,
            "group": self._group,
            "description": self._description,
            "semantics": self._semantics.name,
            "visible": self._visible,
            "animatable": self._animatable,
            "binding": self._binding,
            "type": self.parameter_type.name,
            "value": self.value,
            "default_value": self.default_value
        }


class ScalarParameter(MaterialParameter):
    """Scalar (float) parameter."""

    def __init__(
        self,
        name: str,
        default: float = 0.0,
        range: Optional[ParameterRange] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self._default = default
        self._value = default
        self._range = range or ParameterRange()

    @property
    def parameter_type(self) -> ParameterType:
        return ParameterType.SCALAR

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, new_value: float) -> None:
        self._value = self._range.clamp(float(new_value))

    @property
    def default_value(self) -> float:
        return self._default

    @property
    def range(self) -> ParameterRange:
        return self._range

    def reset(self) -> None:
        self._value = self._default

    def clone(self) -> 'ScalarParameter':
        param = ScalarParameter(
            self._name,
            default=self._default,
            range=copy.deepcopy(self._range),
            display_name=self._display_name,
            group=self._group,
            description=self._description,
            semantics=self._semantics,
            visible=self._visible,
            animatable=self._animatable
        )
        param._value = self._value
        param._binding = self._binding
        return param


class VectorParameter(MaterialParameter):
    """Vector parameter (2-4 components)."""

    def __init__(
        self,
        name: str,
        components: int = 3,
        default: Optional[Tuple[float, ...]] = None,
        range: Optional[ParameterRange] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        if components not in (2, 3, 4):
            raise ValueError("Components must be 2, 3, or 4")
        self._components = components
        self._default = default if default else tuple([0.0] * components)
        if len(self._default) != components:
            raise ValueError(f"Default must have {components} components")
        self._value = list(self._default)
        self._range = range or ParameterRange()

    @property
    def parameter_type(self) -> ParameterType:
        if self._components == 2:
            return ParameterType.VECTOR2
        elif self._components == 3:
            return ParameterType.VECTOR3
        return ParameterType.VECTOR4

    @property
    def components(self) -> int:
        return self._components

    @property
    def value(self) -> Tuple[float, ...]:
        return tuple(self._value)

    @value.setter
    def value(self, new_value: Tuple[float, ...]) -> None:
        if len(new_value) != self._components:
            raise ValueError(f"Value must have {self._components} components")
        self._value = [self._range.clamp(float(v)) for v in new_value]

    @property
    def default_value(self) -> Tuple[float, ...]:
        return self._default

    @property
    def range(self) -> ParameterRange:
        return self._range

    def reset(self) -> None:
        self._value = list(self._default)

    def clone(self) -> 'VectorParameter':
        param = VectorParameter(
            self._name,
            components=self._components,
            default=self._default,
            range=copy.deepcopy(self._range),
            display_name=self._display_name,
            group=self._group,
            description=self._description,
            semantics=self._semantics,
            visible=self._visible,
            animatable=self._animatable
        )
        param._value = list(self._value)
        param._binding = self._binding
        return param


class ColorParameter(MaterialParameter):
    """Color parameter (RGB or RGBA)."""

    def __init__(
        self,
        name: str,
        default: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
        has_alpha: bool = True,
        hdr: bool = False,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self._has_alpha = has_alpha
        self._hdr = hdr
        self._components = 4 if has_alpha else 3
        if len(default) < self._components:
            default = default + (1.0,) * (self._components - len(default))
        self._default = default[:self._components]
        self._value = list(self._default)

    @property
    def parameter_type(self) -> ParameterType:
        return ParameterType.COLOR

    @property
    def has_alpha(self) -> bool:
        return self._has_alpha

    @property
    def hdr(self) -> bool:
        return self._hdr

    @property
    def value(self) -> Tuple[float, ...]:
        return tuple(self._value)

    @value.setter
    def value(self, new_value: Tuple[float, ...]) -> None:
        if len(new_value) < self._components:
            new_value = new_value + (1.0,) * (self._components - len(new_value))
        new_value = new_value[:self._components]
        if self._hdr:
            self._value = [max(0.0, float(v)) for v in new_value]
        else:
            self._value = [max(0.0, min(1.0, float(v))) for v in new_value]

    @property
    def default_value(self) -> Tuple[float, ...]:
        return self._default

    def reset(self) -> None:
        self._value = list(self._default)

    def clone(self) -> 'ColorParameter':
        param = ColorParameter(
            self._name,
            default=self._default,
            has_alpha=self._has_alpha,
            hdr=self._hdr,
            display_name=self._display_name,
            group=self._group,
            description=self._description,
            semantics=self._semantics,
            visible=self._visible,
            animatable=self._animatable
        )
        param._value = list(self._value)
        param._binding = self._binding
        return param


class TextureParameter(MaterialParameter):
    """Texture parameter."""

    def __init__(
        self,
        name: str,
        default_path: str = "",
        settings: Optional[TextureSettings] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self._default_path = default_path
        self._path = default_path
        self._settings = settings or TextureSettings()
        self._texture_handle: Optional[int] = None

    @property
    def parameter_type(self) -> ParameterType:
        return ParameterType.TEXTURE

    @property
    def settings(self) -> TextureSettings:
        return self._settings

    @property
    def value(self) -> str:
        return self._path

    @value.setter
    def value(self, new_value: str) -> None:
        self._path = str(new_value)
        self._texture_handle = None  # Invalidate cached handle

    @property
    def default_value(self) -> str:
        return self._default_path

    @property
    def texture_handle(self) -> Optional[int]:
        return self._texture_handle

    @texture_handle.setter
    def texture_handle(self, handle: int) -> None:
        self._texture_handle = handle

    def reset(self) -> None:
        self._path = self._default_path
        self._texture_handle = None

    def clone(self) -> 'TextureParameter':
        param = TextureParameter(
            self._name,
            default_path=self._default_path,
            settings=copy.deepcopy(self._settings),
            display_name=self._display_name,
            group=self._group,
            description=self._description,
            semantics=self._semantics,
            visible=self._visible,
            animatable=self._animatable
        )
        param._path = self._path
        param._binding = self._binding
        return param


class BooleanParameter(MaterialParameter):
    """Boolean parameter."""

    def __init__(
        self,
        name: str,
        default: bool = False,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self._default = default
        self._value = default

    @property
    def parameter_type(self) -> ParameterType:
        return ParameterType.BOOLEAN

    @property
    def value(self) -> bool:
        return self._value

    @value.setter
    def value(self, new_value: bool) -> None:
        self._value = bool(new_value)

    @property
    def default_value(self) -> bool:
        return self._default

    def reset(self) -> None:
        self._value = self._default

    def clone(self) -> 'BooleanParameter':
        param = BooleanParameter(
            self._name,
            default=self._default,
            display_name=self._display_name,
            group=self._group,
            description=self._description,
            semantics=self._semantics,
            visible=self._visible,
            animatable=self._animatable
        )
        param._value = self._value
        param._binding = self._binding
        return param


class IntegerParameter(MaterialParameter):
    """Integer parameter."""

    def __init__(
        self,
        name: str,
        default: int = 0,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self._default = default
        self._value = default
        self._min = min_value
        self._max = max_value

    @property
    def parameter_type(self) -> ParameterType:
        return ParameterType.INTEGER

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, new_value: int) -> None:
        val = int(new_value)
        if self._min is not None:
            val = max(self._min, val)
        if self._max is not None:
            val = min(self._max, val)
        self._value = val

    @property
    def default_value(self) -> int:
        return self._default

    @property
    def min_value(self) -> Optional[int]:
        return self._min

    @property
    def max_value(self) -> Optional[int]:
        return self._max

    def reset(self) -> None:
        self._value = self._default

    def clone(self) -> 'IntegerParameter':
        param = IntegerParameter(
            self._name,
            default=self._default,
            min_value=self._min,
            max_value=self._max,
            display_name=self._display_name,
            group=self._group,
            description=self._description,
            semantics=self._semantics,
            visible=self._visible,
            animatable=self._animatable
        )
        param._value = self._value
        param._binding = self._binding
        return param


@dataclass
class ParameterCollection:
    """Collection of material parameters."""

    _parameters: Dict[str, MaterialParameter] = field(default_factory=dict)

    def add(self, param: MaterialParameter) -> None:
        """Add a parameter to the collection."""
        self._parameters[param.name] = param

    def remove(self, name: str) -> Optional[MaterialParameter]:
        """Remove and return a parameter."""
        return self._parameters.pop(name, None)

    def get(self, name: str) -> Optional[MaterialParameter]:
        """Get a parameter by name."""
        return self._parameters.get(name)

    def __getitem__(self, name: str) -> MaterialParameter:
        """Get parameter by name."""
        return self._parameters[name]

    def __contains__(self, name: str) -> bool:
        """Check if parameter exists."""
        return name in self._parameters

    def __len__(self) -> int:
        """Get number of parameters."""
        return len(self._parameters)

    def __iter__(self):
        """Iterate over parameters."""
        return iter(self._parameters.values())

    @property
    def names(self) -> List[str]:
        """Get all parameter names."""
        return list(self._parameters.keys())

    def get_by_group(self, group: str) -> List[MaterialParameter]:
        """Get all parameters in a group."""
        return [p for p in self._parameters.values() if p.group == group]

    def get_groups(self) -> List[str]:
        """Get all unique group names."""
        return list(set(p.group for p in self._parameters.values()))

    def get_by_type(self, param_type: ParameterType) -> List[MaterialParameter]:
        """Get all parameters of a specific type."""
        return [p for p in self._parameters.values() if p.parameter_type == param_type]

    def reset_all(self) -> None:
        """Reset all parameters to defaults."""
        for param in self._parameters.values():
            param.reset()

    def clone(self) -> 'ParameterCollection':
        """Create a deep copy of the collection."""
        new_collection = ParameterCollection()
        for param in self._parameters.values():
            new_collection.add(param.clone())
        return new_collection

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Serialize to dictionary."""
        return {name: param.to_dict() for name, param in self._parameters.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Any]]) -> 'ParameterCollection':
        """Deserialize from dictionary."""
        collection = cls()
        for name, param_data in data.items():
            param_type = ParameterType[param_data["type"]]
            if param_type == ParameterType.SCALAR:
                param = ScalarParameter(
                    name,
                    default=param_data.get("default_value", 0.0),
                    display_name=param_data.get("display_name"),
                    group=param_data.get("group", "Default"),
                    description=param_data.get("description", ""),
                    visible=param_data.get("visible", True),
                    animatable=param_data.get("animatable", False)
                )
                param.value = param_data.get("value", param_data.get("default_value", 0.0))
            elif param_type in (ParameterType.VECTOR2, ParameterType.VECTOR3, ParameterType.VECTOR4):
                components = {ParameterType.VECTOR2: 2, ParameterType.VECTOR3: 3, ParameterType.VECTOR4: 4}[param_type]
                param = VectorParameter(
                    name,
                    components=components,
                    default=tuple(param_data.get("default_value", [0.0] * components)),
                    display_name=param_data.get("display_name"),
                    group=param_data.get("group", "Default"),
                    description=param_data.get("description", ""),
                    visible=param_data.get("visible", True),
                    animatable=param_data.get("animatable", False)
                )
                param.value = tuple(param_data.get("value", param_data.get("default_value", [0.0] * components)))
            elif param_type == ParameterType.COLOR:
                param = ColorParameter(
                    name,
                    default=tuple(param_data.get("default_value", (1.0, 1.0, 1.0, 1.0))),
                    display_name=param_data.get("display_name"),
                    group=param_data.get("group", "Default"),
                    description=param_data.get("description", ""),
                    visible=param_data.get("visible", True),
                    animatable=param_data.get("animatable", False)
                )
                param.value = tuple(param_data.get("value", param_data.get("default_value", (1.0, 1.0, 1.0, 1.0))))
            elif param_type == ParameterType.TEXTURE:
                param = TextureParameter(
                    name,
                    default_path=param_data.get("default_value", ""),
                    display_name=param_data.get("display_name"),
                    group=param_data.get("group", "Default"),
                    description=param_data.get("description", ""),
                    visible=param_data.get("visible", True),
                    animatable=param_data.get("animatable", False)
                )
                param.value = param_data.get("value", param_data.get("default_value", ""))
            elif param_type == ParameterType.BOOLEAN:
                param = BooleanParameter(
                    name,
                    default=param_data.get("default_value", False),
                    display_name=param_data.get("display_name"),
                    group=param_data.get("group", "Default"),
                    description=param_data.get("description", ""),
                    visible=param_data.get("visible", True),
                    animatable=param_data.get("animatable", False)
                )
                param.value = param_data.get("value", param_data.get("default_value", False))
            elif param_type == ParameterType.INTEGER:
                param = IntegerParameter(
                    name,
                    default=param_data.get("default_value", 0),
                    display_name=param_data.get("display_name"),
                    group=param_data.get("group", "Default"),
                    description=param_data.get("description", ""),
                    visible=param_data.get("visible", True),
                    animatable=param_data.get("animatable", False)
                )
                param.value = param_data.get("value", param_data.get("default_value", 0))
            else:
                continue

            param.binding = param_data.get("binding")
            collection.add(param)

        return collection
