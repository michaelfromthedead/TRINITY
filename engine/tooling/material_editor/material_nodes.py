"""Material nodes - Node types for material graph."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Type, Callable
import uuid
import math


class NodeCategory(Enum):
    """Category of material node."""
    INPUT = auto()
    MATH = auto()
    TEXTURE = auto()
    UTILITY = auto()
    PBR = auto()
    OUTPUT = auto()
    CUSTOM = auto()


class DataType(Enum):
    """Data type for node pins."""
    FLOAT = auto()
    FLOAT2 = auto()
    FLOAT3 = auto()
    FLOAT4 = auto()
    TEXTURE2D = auto()
    TEXTURE_CUBE = auto()
    SAMPLER = auto()
    MATRIX = auto()
    BOOL = auto()
    INT = auto()
    ANY = auto()  # Can connect to any type


@dataclass
class NodePin:
    """Input or output pin on a node."""
    name: str
    data_type: DataType
    is_output: bool
    default_value: Any = None
    description: str = ""
    hidden: bool = False

    def is_compatible(self, other: 'NodePin') -> bool:
        """Check if this pin can connect to another."""
        if self.is_output == other.is_output:
            return False  # Can't connect output to output or input to input

        # ANY type is compatible with everything
        if self.data_type == DataType.ANY or other.data_type == DataType.ANY:
            return True

        # Same types are always compatible
        if self.data_type == other.data_type:
            return True

        # Float types can be implicitly converted
        float_types = {DataType.FLOAT, DataType.FLOAT2, DataType.FLOAT3, DataType.FLOAT4}
        if self.data_type in float_types and other.data_type in float_types:
            return True

        return False


class MaterialNode(ABC):
    """Base class for material graph nodes."""

    def __init__(self, name: str = ""):
        self._id = str(uuid.uuid4())
        self._name = name or self.__class__.__name__
        self._position: Tuple[float, float] = (0.0, 0.0)
        self._inputs: Dict[str, NodePin] = {}
        self._outputs: Dict[str, NodePin] = {}
        self._setup_pins()
        self._preview_enabled = True
        self._collapsed = False
        self._comment = ""
        self._color: Optional[Tuple[int, int, int]] = None

    @abstractmethod
    def _setup_pins(self) -> None:
        """Set up input and output pins."""
        pass

    @property
    @abstractmethod
    def category(self) -> NodeCategory:
        """Get node category."""
        pass

    @abstractmethod
    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate node with given inputs and return outputs."""
        pass

    @abstractmethod
    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        """Generate shader code for this node."""
        pass

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def position(self) -> Tuple[float, float]:
        return self._position

    @position.setter
    def position(self, value: Tuple[float, float]) -> None:
        self._position = value

    @property
    def inputs(self) -> Dict[str, NodePin]:
        return self._inputs

    @property
    def outputs(self) -> Dict[str, NodePin]:
        return self._outputs

    @property
    def preview_enabled(self) -> bool:
        return self._preview_enabled

    @preview_enabled.setter
    def preview_enabled(self, value: bool) -> None:
        self._preview_enabled = value

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    @collapsed.setter
    def collapsed(self, value: bool) -> None:
        self._collapsed = value

    @property
    def comment(self) -> str:
        return self._comment

    @comment.setter
    def comment(self, value: str) -> None:
        self._comment = value

    @property
    def color(self) -> Optional[Tuple[int, int, int]]:
        return self._color

    @color.setter
    def color(self, value: Tuple[int, int, int]) -> None:
        self._color = value

    def add_input(self, pin: NodePin) -> None:
        """Add an input pin."""
        pin.is_output = False
        self._inputs[pin.name] = pin

    def add_output(self, pin: NodePin) -> None:
        """Add an output pin."""
        pin.is_output = True
        self._outputs[pin.name] = pin

    def get_input(self, name: str) -> Optional[NodePin]:
        """Get input pin by name."""
        return self._inputs.get(name)

    def get_output(self, name: str) -> Optional[NodePin]:
        """Get output pin by name."""
        return self._outputs.get(name)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        # Use registry key if available, fallback to class name
        from . import material_nodes as nodes_module
        node_type = getattr(nodes_module, 'NODE_CLASS_TO_KEY', {}).get(
            self.__class__, self.__class__.__name__
        )
        return {
            "id": self._id,
            "type": node_type,
            "name": self._name,
            "position": list(self._position),
            "preview_enabled": self._preview_enabled,
            "collapsed": self._collapsed,
            "comment": self._comment,
            "color": list(self._color) if self._color else None,
            "inputs": {name: pin.default_value for name, pin in self._inputs.items()},
        }


# ============================================================================
# INPUT NODES
# ============================================================================

class ConstantNode(MaterialNode):
    """Constant scalar value node."""

    def __init__(self, value: float = 0.0, name: str = "Constant"):
        self._value = value
        super().__init__(name)

    def _setup_pins(self) -> None:
        self.add_output(NodePin("Value", DataType.FLOAT, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.INPUT

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, v: float) -> None:
        self._value = v

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"Value": self._value}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        return f"float {output_vars['Value']} = {self._value};"


class Constant2Node(MaterialNode):
    """Constant float2 value node."""

    def __init__(self, value: Tuple[float, float] = (0.0, 0.0), name: str = "Constant2"):
        self._value = value
        super().__init__(name)

    def _setup_pins(self) -> None:
        self.add_output(NodePin("Value", DataType.FLOAT2, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.INPUT

    @property
    def value(self) -> Tuple[float, float]:
        return self._value

    @value.setter
    def value(self, v: Tuple[float, float]) -> None:
        self._value = v

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"Value": self._value}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        return f"float2 {output_vars['Value']} = float2({self._value[0]}, {self._value[1]});"


class Constant3Node(MaterialNode):
    """Constant float3 value node."""

    def __init__(self, value: Tuple[float, float, float] = (0.0, 0.0, 0.0), name: str = "Constant3"):
        self._value = value
        super().__init__(name)

    def _setup_pins(self) -> None:
        self.add_output(NodePin("Value", DataType.FLOAT3, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.INPUT

    @property
    def value(self) -> Tuple[float, float, float]:
        return self._value

    @value.setter
    def value(self, v: Tuple[float, float, float]) -> None:
        self._value = v

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"Value": self._value}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        return f"float3 {output_vars['Value']} = float3({self._value[0]}, {self._value[1]}, {self._value[2]});"


class Constant4Node(MaterialNode):
    """Constant float4 value node."""

    def __init__(self, value: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0), name: str = "Constant4"):
        self._value = value
        super().__init__(name)

    def _setup_pins(self) -> None:
        self.add_output(NodePin("Value", DataType.FLOAT4, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.INPUT

    @property
    def value(self) -> Tuple[float, float, float, float]:
        return self._value

    @value.setter
    def value(self, v: Tuple[float, float, float, float]) -> None:
        self._value = v

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"Value": self._value}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        return f"float4 {output_vars['Value']} = float4({self._value[0]}, {self._value[1]}, {self._value[2]}, {self._value[3]});"


class ParameterNode(MaterialNode):
    """Exposed parameter node."""

    def __init__(self, param_name: str = "Parameter", param_type: DataType = DataType.FLOAT, name: str = "Parameter"):
        self._param_name = param_name
        self._param_type = param_type
        self._default_value: Any = 0.0
        super().__init__(name)

    def _setup_pins(self) -> None:
        self.add_output(NodePin("Value", self._param_type, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.INPUT

    @property
    def param_name(self) -> str:
        return self._param_name

    @param_name.setter
    def param_name(self, v: str) -> None:
        self._param_name = v

    @property
    def param_type(self) -> DataType:
        return self._param_type

    @property
    def default_value(self) -> Any:
        return self._default_value

    @default_value.setter
    def default_value(self, v: Any) -> None:
        self._default_value = v

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"Value": self._default_value}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        type_map = {
            DataType.FLOAT: "float",
            DataType.FLOAT2: "float2",
            DataType.FLOAT3: "float3",
            DataType.FLOAT4: "float4",
        }
        shader_type = type_map.get(self._param_type, "float")
        return f"{shader_type} {output_vars['Value']} = {self._param_name};"


# ============================================================================
# MATH NODES
# ============================================================================

class AddNode(MaterialNode):
    """Add two values."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("A", DataType.ANY, False, 0.0))
        self.add_input(NodePin("B", DataType.ANY, False, 0.0))
        self.add_output(NodePin("Result", DataType.ANY, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        a = inputs.get("A", 0.0)
        b = inputs.get("B", 0.0)
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return {"Result": tuple(x + y for x, y in zip(a, b))}
        elif isinstance(a, (tuple, list)):
            return {"Result": tuple(x + b for x in a)}
        elif isinstance(b, (tuple, list)):
            return {"Result": tuple(a + x for x in b)}
        return {"Result": a + b}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        a = inputs.get("A", "0.0")
        b = inputs.get("B", "0.0")
        return f"auto {output_vars['Result']} = {a} + {b};"


class SubtractNode(MaterialNode):
    """Subtract two values."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("A", DataType.ANY, False, 0.0))
        self.add_input(NodePin("B", DataType.ANY, False, 0.0))
        self.add_output(NodePin("Result", DataType.ANY, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        a = inputs.get("A", 0.0)
        b = inputs.get("B", 0.0)
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return {"Result": tuple(x - y for x, y in zip(a, b))}
        elif isinstance(a, (tuple, list)):
            return {"Result": tuple(x - b for x in a)}
        elif isinstance(b, (tuple, list)):
            return {"Result": tuple(a - x for x in b)}
        return {"Result": a - b}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        a = inputs.get("A", "0.0")
        b = inputs.get("B", "0.0")
        return f"auto {output_vars['Result']} = {a} - {b};"


class MultiplyNode(MaterialNode):
    """Multiply two values."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("A", DataType.ANY, False, 1.0))
        self.add_input(NodePin("B", DataType.ANY, False, 1.0))
        self.add_output(NodePin("Result", DataType.ANY, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        a = inputs.get("A", 1.0)
        b = inputs.get("B", 1.0)
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return {"Result": tuple(x * y for x, y in zip(a, b))}
        elif isinstance(a, (tuple, list)):
            return {"Result": tuple(x * b for x in a)}
        elif isinstance(b, (tuple, list)):
            return {"Result": tuple(a * x for x in b)}
        return {"Result": a * b}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        a = inputs.get("A", "1.0")
        b = inputs.get("B", "1.0")
        return f"auto {output_vars['Result']} = {a} * {b};"


class DivideNode(MaterialNode):
    """Divide two values."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("A", DataType.ANY, False, 1.0))
        self.add_input(NodePin("B", DataType.ANY, False, 1.0))
        self.add_output(NodePin("Result", DataType.ANY, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        a = inputs.get("A", 1.0)
        b = inputs.get("B", 1.0)

        def safe_div(x, y):
            return x / y if y != 0 else 0.0

        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return {"Result": tuple(safe_div(x, y) for x, y in zip(a, b))}
        elif isinstance(a, (tuple, list)):
            return {"Result": tuple(safe_div(x, b) for x in a)}
        elif isinstance(b, (tuple, list)):
            return {"Result": tuple(safe_div(a, x) for x in b)}
        return {"Result": safe_div(a, b)}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        a = inputs.get("A", "1.0")
        b = inputs.get("B", "1.0")
        return f"auto {output_vars['Result']} = {a} / max({b}, 0.0001);"


class LerpNode(MaterialNode):
    """Linear interpolation between two values."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("A", DataType.ANY, False, 0.0))
        self.add_input(NodePin("B", DataType.ANY, False, 1.0))
        self.add_input(NodePin("Alpha", DataType.FLOAT, False, 0.5))
        self.add_output(NodePin("Result", DataType.ANY, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        a = inputs.get("A", 0.0)
        b = inputs.get("B", 1.0)
        alpha = inputs.get("Alpha", 0.5)

        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return {"Result": tuple(x + (y - x) * alpha for x, y in zip(a, b))}
        return {"Result": a + (b - a) * alpha}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        a = inputs.get("A", "0.0")
        b = inputs.get("B", "1.0")
        alpha = inputs.get("Alpha", "0.5")
        return f"auto {output_vars['Result']} = lerp({a}, {b}, {alpha});"


class ClampNode(MaterialNode):
    """Clamp value between min and max."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Value", DataType.ANY, False, 0.5))
        self.add_input(NodePin("Min", DataType.ANY, False, 0.0))
        self.add_input(NodePin("Max", DataType.ANY, False, 1.0))
        self.add_output(NodePin("Result", DataType.ANY, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        value = inputs.get("Value", 0.5)
        min_val = inputs.get("Min", 0.0)
        max_val = inputs.get("Max", 1.0)

        if isinstance(value, (tuple, list)):
            return {"Result": tuple(max(min_val, min(max_val, x)) for x in value)}
        return {"Result": max(min_val, min(max_val, value))}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        value = inputs.get("Value", "0.5")
        min_val = inputs.get("Min", "0.0")
        max_val = inputs.get("Max", "1.0")
        return f"auto {output_vars['Result']} = clamp({value}, {min_val}, {max_val});"


class SaturateNode(MaterialNode):
    """Clamp value between 0 and 1."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Value", DataType.ANY, False, 0.5))
        self.add_output(NodePin("Result", DataType.ANY, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        value = inputs.get("Value", 0.5)
        if isinstance(value, (tuple, list)):
            return {"Result": tuple(max(0.0, min(1.0, x)) for x in value)}
        return {"Result": max(0.0, min(1.0, value))}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        value = inputs.get("Value", "0.5")
        return f"auto {output_vars['Result']} = saturate({value});"


class PowerNode(MaterialNode):
    """Raise base to exponent."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Base", DataType.ANY, False, 2.0))
        self.add_input(NodePin("Exponent", DataType.FLOAT, False, 2.0))
        self.add_output(NodePin("Result", DataType.ANY, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        base = inputs.get("Base", 2.0)
        exp = inputs.get("Exponent", 2.0)
        if isinstance(base, (tuple, list)):
            return {"Result": tuple(pow(max(0, x), exp) for x in base)}
        return {"Result": pow(max(0, base), exp)}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        base = inputs.get("Base", "2.0")
        exp = inputs.get("Exponent", "2.0")
        return f"auto {output_vars['Result']} = pow(max({base}, 0.0), {exp});"


class DotNode(MaterialNode):
    """Dot product of two vectors."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("A", DataType.FLOAT3, False, (1.0, 0.0, 0.0)))
        self.add_input(NodePin("B", DataType.FLOAT3, False, (0.0, 1.0, 0.0)))
        self.add_output(NodePin("Result", DataType.FLOAT, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        a = inputs.get("A", (1.0, 0.0, 0.0))
        b = inputs.get("B", (0.0, 1.0, 0.0))
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return {"Result": sum(x * y for x, y in zip(a, b))}
        return {"Result": 0.0}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        a = inputs.get("A", "float3(1, 0, 0)")
        b = inputs.get("B", "float3(0, 1, 0)")
        return f"float {output_vars['Result']} = dot({a}, {b});"


class CrossNode(MaterialNode):
    """Cross product of two vectors."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("A", DataType.FLOAT3, False, (1.0, 0.0, 0.0)))
        self.add_input(NodePin("B", DataType.FLOAT3, False, (0.0, 1.0, 0.0)))
        self.add_output(NodePin("Result", DataType.FLOAT3, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        a = inputs.get("A", (1.0, 0.0, 0.0))
        b = inputs.get("B", (0.0, 1.0, 0.0))
        if len(a) >= 3 and len(b) >= 3:
            return {"Result": (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]
            )}
        return {"Result": (0.0, 0.0, 0.0)}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        a = inputs.get("A", "float3(1, 0, 0)")
        b = inputs.get("B", "float3(0, 1, 0)")
        return f"float3 {output_vars['Result']} = cross({a}, {b});"


class NormalizeNode(MaterialNode):
    """Normalize a vector."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Vector", DataType.FLOAT3, False, (1.0, 0.0, 0.0)))
        self.add_output(NodePin("Result", DataType.FLOAT3, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        v = inputs.get("Vector", (1.0, 0.0, 0.0))
        if isinstance(v, (tuple, list)):
            length = math.sqrt(sum(x * x for x in v))
            if length > 0:
                return {"Result": tuple(x / length for x in v)}
        return {"Result": (0.0, 0.0, 0.0)}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        v = inputs.get("Vector", "float3(1, 0, 0)")
        return f"float3 {output_vars['Result']} = normalize({v});"


class AbsNode(MaterialNode):
    """Absolute value."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Value", DataType.ANY, False, 0.0))
        self.add_output(NodePin("Result", DataType.ANY, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        value = inputs.get("Value", 0.0)
        if isinstance(value, (tuple, list)):
            return {"Result": tuple(abs(x) for x in value)}
        return {"Result": abs(value)}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        value = inputs.get("Value", "0.0")
        return f"auto {output_vars['Result']} = abs({value});"


class FloorNode(MaterialNode):
    """Floor value."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Value", DataType.ANY, False, 0.0))
        self.add_output(NodePin("Result", DataType.ANY, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        value = inputs.get("Value", 0.0)
        if isinstance(value, (tuple, list)):
            return {"Result": tuple(math.floor(x) for x in value)}
        return {"Result": math.floor(value)}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        value = inputs.get("Value", "0.0")
        return f"auto {output_vars['Result']} = floor({value});"


class CeilNode(MaterialNode):
    """Ceiling value."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Value", DataType.ANY, False, 0.0))
        self.add_output(NodePin("Result", DataType.ANY, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        value = inputs.get("Value", 0.0)
        if isinstance(value, (tuple, list)):
            return {"Result": tuple(math.ceil(x) for x in value)}
        return {"Result": math.ceil(value)}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        value = inputs.get("Value", "0.0")
        return f"auto {output_vars['Result']} = ceil({value});"


class FracNode(MaterialNode):
    """Fractional part of value."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Value", DataType.ANY, False, 0.0))
        self.add_output(NodePin("Result", DataType.ANY, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        value = inputs.get("Value", 0.0)
        if isinstance(value, (tuple, list)):
            return {"Result": tuple(x - math.floor(x) for x in value)}
        return {"Result": value - math.floor(value)}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        value = inputs.get("Value", "0.0")
        return f"auto {output_vars['Result']} = frac({value});"


class SinNode(MaterialNode):
    """Sine function."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Value", DataType.FLOAT, False, 0.0))
        self.add_output(NodePin("Result", DataType.FLOAT, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        value = inputs.get("Value", 0.0)
        return {"Result": math.sin(value)}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        value = inputs.get("Value", "0.0")
        return f"float {output_vars['Result']} = sin({value});"


class CosNode(MaterialNode):
    """Cosine function."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Value", DataType.FLOAT, False, 0.0))
        self.add_output(NodePin("Result", DataType.FLOAT, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        value = inputs.get("Value", 0.0)
        return {"Result": math.cos(value)}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        value = inputs.get("Value", "0.0")
        return f"float {output_vars['Result']} = cos({value});"


class OneMinusNode(MaterialNode):
    """1 - Value."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Value", DataType.ANY, False, 0.0))
        self.add_output(NodePin("Result", DataType.ANY, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.MATH

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        value = inputs.get("Value", 0.0)
        if isinstance(value, (tuple, list)):
            return {"Result": tuple(1.0 - x for x in value)}
        return {"Result": 1.0 - value}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        value = inputs.get("Value", "0.0")
        return f"auto {output_vars['Result']} = 1.0 - {value};"


# ============================================================================
# TEXTURE NODES
# ============================================================================

class TextureSampleNode(MaterialNode):
    """Sample a 2D texture."""

    def __init__(self, texture_path: str = "", name: str = "TextureSample"):
        self._texture_path = texture_path
        super().__init__(name)

    def _setup_pins(self) -> None:
        self.add_input(NodePin("UV", DataType.FLOAT2, False, (0.0, 0.0)))
        self.add_input(NodePin("Texture", DataType.TEXTURE2D, False))
        self.add_output(NodePin("RGBA", DataType.FLOAT4, True))
        self.add_output(NodePin("RGB", DataType.FLOAT3, True))
        self.add_output(NodePin("R", DataType.FLOAT, True))
        self.add_output(NodePin("G", DataType.FLOAT, True))
        self.add_output(NodePin("B", DataType.FLOAT, True))
        self.add_output(NodePin("A", DataType.FLOAT, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.TEXTURE

    @property
    def texture_path(self) -> str:
        return self._texture_path

    @texture_path.setter
    def texture_path(self, v: str) -> None:
        self._texture_path = v

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # In evaluation, return a default color
        rgba = (1.0, 1.0, 1.0, 1.0)
        return {
            "RGBA": rgba,
            "RGB": rgba[:3],
            "R": rgba[0],
            "G": rgba[1],
            "B": rgba[2],
            "A": rgba[3]
        }

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        uv = inputs.get("UV", "input.uv")
        tex = inputs.get("Texture", self._texture_path.replace("/", "_").replace(".", "_") + "_tex")
        lines = [
            f"float4 {output_vars['RGBA']} = {tex}.Sample(sampler_{tex}, {uv});",
            f"float3 {output_vars['RGB']} = {output_vars['RGBA']}.rgb;",
            f"float {output_vars['R']} = {output_vars['RGBA']}.r;",
            f"float {output_vars['G']} = {output_vars['RGBA']}.g;",
            f"float {output_vars['B']} = {output_vars['RGBA']}.b;",
            f"float {output_vars['A']} = {output_vars['RGBA']}.a;"
        ]
        return "\n".join(lines)


class UVNode(MaterialNode):
    """Get UV coordinates."""

    def __init__(self, uv_channel: int = 0, name: str = "UV"):
        self._uv_channel = uv_channel
        super().__init__(name)

    def _setup_pins(self) -> None:
        self.add_output(NodePin("UV", DataType.FLOAT2, True))
        self.add_output(NodePin("U", DataType.FLOAT, True))
        self.add_output(NodePin("V", DataType.FLOAT, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.TEXTURE

    @property
    def uv_channel(self) -> int:
        return self._uv_channel

    @uv_channel.setter
    def uv_channel(self, v: int) -> None:
        self._uv_channel = v

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "UV": (0.5, 0.5),
            "U": 0.5,
            "V": 0.5
        }

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        uv_name = f"input.uv{self._uv_channel}" if self._uv_channel > 0 else "input.uv"
        lines = [
            f"float2 {output_vars['UV']} = {uv_name};",
            f"float {output_vars['U']} = {output_vars['UV']}.x;",
            f"float {output_vars['V']} = {output_vars['UV']}.y;"
        ]
        return "\n".join(lines)


class TilingOffsetNode(MaterialNode):
    """Apply tiling and offset to UVs."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("UV", DataType.FLOAT2, False, (0.0, 0.0)))
        self.add_input(NodePin("Tiling", DataType.FLOAT2, False, (1.0, 1.0)))
        self.add_input(NodePin("Offset", DataType.FLOAT2, False, (0.0, 0.0)))
        self.add_output(NodePin("Result", DataType.FLOAT2, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.TEXTURE

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        uv = inputs.get("UV", (0.0, 0.0))
        tiling = inputs.get("Tiling", (1.0, 1.0))
        offset = inputs.get("Offset", (0.0, 0.0))
        return {"Result": (uv[0] * tiling[0] + offset[0], uv[1] * tiling[1] + offset[1])}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        uv = inputs.get("UV", "input.uv")
        tiling = inputs.get("Tiling", "float2(1, 1)")
        offset = inputs.get("Offset", "float2(0, 0)")
        return f"float2 {output_vars['Result']} = {uv} * {tiling} + {offset};"


class NormalMapNode(MaterialNode):
    """Unpack normal from normal map."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Normal", DataType.FLOAT3, False, (0.5, 0.5, 1.0)))
        self.add_input(NodePin("Strength", DataType.FLOAT, False, 1.0))
        self.add_output(NodePin("Result", DataType.FLOAT3, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.TEXTURE

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        normal = inputs.get("Normal", (0.5, 0.5, 1.0))
        strength = inputs.get("Strength", 1.0)
        # Unpack from [0,1] to [-1,1]
        unpacked = (
            (normal[0] * 2.0 - 1.0) * strength,
            (normal[1] * 2.0 - 1.0) * strength,
            normal[2] * 2.0 - 1.0
        )
        return {"Result": unpacked}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        normal = inputs.get("Normal", "float3(0.5, 0.5, 1.0)")
        strength = inputs.get("Strength", "1.0")
        lines = [
            f"float3 {output_vars['Result']} = {normal} * 2.0 - 1.0;",
            f"{output_vars['Result']}.xy *= {strength};",
            f"{output_vars['Result']} = normalize({output_vars['Result']});"
        ]
        return "\n".join(lines)


class ParallaxNode(MaterialNode):
    """Parallax mapping offset."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("UV", DataType.FLOAT2, False, (0.0, 0.0)))
        self.add_input(NodePin("Height", DataType.FLOAT, False, 0.5))
        self.add_input(NodePin("Scale", DataType.FLOAT, False, 0.05))
        self.add_input(NodePin("ViewDir", DataType.FLOAT3, False, (0.0, 0.0, 1.0)))
        self.add_output(NodePin("Result", DataType.FLOAT2, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.TEXTURE

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        uv = inputs.get("UV", (0.0, 0.0))
        height = inputs.get("Height", 0.5)
        scale = inputs.get("Scale", 0.05)
        view_dir = inputs.get("ViewDir", (0.0, 0.0, 1.0))

        offset_x = view_dir[0] * height * scale
        offset_y = view_dir[1] * height * scale

        return {"Result": (uv[0] - offset_x, uv[1] - offset_y)}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        uv = inputs.get("UV", "input.uv")
        height = inputs.get("Height", "0.5")
        scale = inputs.get("Scale", "0.05")
        view_dir = inputs.get("ViewDir", "input.viewDir")
        lines = [
            f"float2 {output_vars['Result']} = {uv} - {view_dir}.xy * {height} * {scale};"
        ]
        return "\n".join(lines)


# ============================================================================
# UTILITY NODES
# ============================================================================

class TimeNode(MaterialNode):
    """Get time values."""

    def _setup_pins(self) -> None:
        self.add_output(NodePin("Time", DataType.FLOAT, True))
        self.add_output(NodePin("SinTime", DataType.FLOAT, True))
        self.add_output(NodePin("CosTime", DataType.FLOAT, True))
        self.add_output(NodePin("DeltaTime", DataType.FLOAT, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.UTILITY

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Time": 0.0,
            "SinTime": 0.0,
            "CosTime": 1.0,
            "DeltaTime": 0.016
        }

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        lines = [
            f"float {output_vars['Time']} = _Time.y;",
            f"float {output_vars['SinTime']} = _SinTime.w;",
            f"float {output_vars['CosTime']} = _CosTime.w;",
            f"float {output_vars['DeltaTime']} = unity_DeltaTime.x;"
        ]
        return "\n".join(lines)


class WorldPositionNode(MaterialNode):
    """Get world position."""

    def _setup_pins(self) -> None:
        self.add_output(NodePin("Position", DataType.FLOAT3, True))
        self.add_output(NodePin("X", DataType.FLOAT, True))
        self.add_output(NodePin("Y", DataType.FLOAT, True))
        self.add_output(NodePin("Z", DataType.FLOAT, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.UTILITY

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Position": (0.0, 0.0, 0.0),
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.0
        }

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        lines = [
            f"float3 {output_vars['Position']} = input.worldPos;",
            f"float {output_vars['X']} = {output_vars['Position']}.x;",
            f"float {output_vars['Y']} = {output_vars['Position']}.y;",
            f"float {output_vars['Z']} = {output_vars['Position']}.z;"
        ]
        return "\n".join(lines)


class ViewDirectionNode(MaterialNode):
    """Get view direction."""

    def _setup_pins(self) -> None:
        self.add_output(NodePin("ViewDir", DataType.FLOAT3, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.UTILITY

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"ViewDir": (0.0, 0.0, 1.0)}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        return f"float3 {output_vars['ViewDir']} = normalize(input.viewDir);"


class ScreenPositionNode(MaterialNode):
    """Get screen position."""

    def _setup_pins(self) -> None:
        self.add_output(NodePin("Position", DataType.FLOAT4, True))
        self.add_output(NodePin("UV", DataType.FLOAT2, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.UTILITY

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Position": (0.5, 0.5, 0.0, 1.0),
            "UV": (0.5, 0.5)
        }

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        lines = [
            f"float4 {output_vars['Position']} = input.screenPos;",
            f"float2 {output_vars['UV']} = {output_vars['Position']}.xy / {output_vars['Position']}.w;"
        ]
        return "\n".join(lines)


class VertexColorNode(MaterialNode):
    """Get vertex color."""

    def _setup_pins(self) -> None:
        self.add_output(NodePin("Color", DataType.FLOAT4, True))
        self.add_output(NodePin("RGB", DataType.FLOAT3, True))
        self.add_output(NodePin("R", DataType.FLOAT, True))
        self.add_output(NodePin("G", DataType.FLOAT, True))
        self.add_output(NodePin("B", DataType.FLOAT, True))
        self.add_output(NodePin("A", DataType.FLOAT, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.UTILITY

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Color": (1.0, 1.0, 1.0, 1.0),
            "RGB": (1.0, 1.0, 1.0),
            "R": 1.0,
            "G": 1.0,
            "B": 1.0,
            "A": 1.0
        }

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        lines = [
            f"float4 {output_vars['Color']} = input.color;",
            f"float3 {output_vars['RGB']} = {output_vars['Color']}.rgb;",
            f"float {output_vars['R']} = {output_vars['Color']}.r;",
            f"float {output_vars['G']} = {output_vars['Color']}.g;",
            f"float {output_vars['B']} = {output_vars['Color']}.b;",
            f"float {output_vars['A']} = {output_vars['Color']}.a;"
        ]
        return "\n".join(lines)


class SplitNode(MaterialNode):
    """Split vector into components."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Vector", DataType.FLOAT4, False, (0.0, 0.0, 0.0, 1.0)))
        self.add_output(NodePin("X", DataType.FLOAT, True))
        self.add_output(NodePin("Y", DataType.FLOAT, True))
        self.add_output(NodePin("Z", DataType.FLOAT, True))
        self.add_output(NodePin("W", DataType.FLOAT, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.UTILITY

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        v = inputs.get("Vector", (0.0, 0.0, 0.0, 1.0))
        if len(v) < 4:
            v = tuple(v) + (0.0,) * (4 - len(v))
        return {"X": v[0], "Y": v[1], "Z": v[2], "W": v[3]}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        v = inputs.get("Vector", "float4(0, 0, 0, 1)")
        lines = [
            f"float {output_vars['X']} = {v}.x;",
            f"float {output_vars['Y']} = {v}.y;",
            f"float {output_vars['Z']} = {v}.z;",
            f"float {output_vars['W']} = {v}.w;"
        ]
        return "\n".join(lines)


class CombineNode(MaterialNode):
    """Combine components into vector."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("X", DataType.FLOAT, False, 0.0))
        self.add_input(NodePin("Y", DataType.FLOAT, False, 0.0))
        self.add_input(NodePin("Z", DataType.FLOAT, False, 0.0))
        self.add_input(NodePin("W", DataType.FLOAT, False, 1.0))
        self.add_output(NodePin("XYZW", DataType.FLOAT4, True))
        self.add_output(NodePin("XYZ", DataType.FLOAT3, True))
        self.add_output(NodePin("XY", DataType.FLOAT2, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.UTILITY

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        x = inputs.get("X", 0.0)
        y = inputs.get("Y", 0.0)
        z = inputs.get("Z", 0.0)
        w = inputs.get("W", 1.0)
        return {
            "XYZW": (x, y, z, w),
            "XYZ": (x, y, z),
            "XY": (x, y)
        }

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        x = inputs.get("X", "0.0")
        y = inputs.get("Y", "0.0")
        z = inputs.get("Z", "0.0")
        w = inputs.get("W", "1.0")
        lines = [
            f"float4 {output_vars['XYZW']} = float4({x}, {y}, {z}, {w});",
            f"float3 {output_vars['XYZ']} = float3({x}, {y}, {z});",
            f"float2 {output_vars['XY']} = float2({x}, {y});"
        ]
        return "\n".join(lines)


# ============================================================================
# PBR NODES
# ============================================================================

class FresnelNode(MaterialNode):
    """Calculate Fresnel effect."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Normal", DataType.FLOAT3, False, (0.0, 0.0, 1.0)))
        self.add_input(NodePin("ViewDir", DataType.FLOAT3, False, (0.0, 0.0, 1.0)))
        self.add_input(NodePin("Power", DataType.FLOAT, False, 5.0))
        self.add_output(NodePin("Result", DataType.FLOAT, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.PBR

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        normal = inputs.get("Normal", (0.0, 0.0, 1.0))
        view_dir = inputs.get("ViewDir", (0.0, 0.0, 1.0))
        power = inputs.get("Power", 5.0)

        # Calculate dot product
        n_dot_v = sum(n * v for n, v in zip(normal, view_dir))
        n_dot_v = max(0.0, n_dot_v)

        fresnel = pow(1.0 - n_dot_v, power)
        return {"Result": fresnel}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        normal = inputs.get("Normal", "input.normal")
        view_dir = inputs.get("ViewDir", "input.viewDir")
        power = inputs.get("Power", "5.0")
        lines = [
            f"float NdotV = saturate(dot({normal}, {view_dir}));",
            f"float {output_vars['Result']} = pow(1.0 - NdotV, {power});"
        ]
        return "\n".join(lines)


class GGXNode(MaterialNode):
    """GGX/Trowbridge-Reitz normal distribution."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("NdotH", DataType.FLOAT, False, 1.0))
        self.add_input(NodePin("Roughness", DataType.FLOAT, False, 0.5))
        self.add_output(NodePin("Result", DataType.FLOAT, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.PBR

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        n_dot_h = inputs.get("NdotH", 1.0)
        roughness = inputs.get("Roughness", 0.5)

        a = roughness * roughness
        a2 = a * a
        n_dot_h2 = n_dot_h * n_dot_h

        denom = n_dot_h2 * (a2 - 1.0) + 1.0
        result = a2 / (math.pi * denom * denom + 0.0001)

        return {"Result": result}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        n_dot_h = inputs.get("NdotH", "1.0")
        roughness = inputs.get("Roughness", "0.5")
        lines = [
            f"float a = {roughness} * {roughness};",
            f"float a2 = a * a;",
            f"float NdotH2 = {n_dot_h} * {n_dot_h};",
            f"float denom = NdotH2 * (a2 - 1.0) + 1.0;",
            f"float {output_vars['Result']} = a2 / (PI * denom * denom + 0.0001);"
        ]
        return "\n".join(lines)


class LambertNode(MaterialNode):
    """Lambert diffuse term."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Normal", DataType.FLOAT3, False, (0.0, 0.0, 1.0)))
        self.add_input(NodePin("LightDir", DataType.FLOAT3, False, (0.0, 1.0, 0.0)))
        self.add_output(NodePin("Result", DataType.FLOAT, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.PBR

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        normal = inputs.get("Normal", (0.0, 0.0, 1.0))
        light_dir = inputs.get("LightDir", (0.0, 1.0, 0.0))

        n_dot_l = sum(n * l for n, l in zip(normal, light_dir))
        return {"Result": max(0.0, n_dot_l)}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        normal = inputs.get("Normal", "input.normal")
        light_dir = inputs.get("LightDir", "_WorldSpaceLightPos0.xyz")
        return f"float {output_vars['Result']} = saturate(dot({normal}, {light_dir}));"


class BRDFNode(MaterialNode):
    """Complete PBR BRDF calculation."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Albedo", DataType.FLOAT3, False, (0.5, 0.5, 0.5)))
        self.add_input(NodePin("Normal", DataType.FLOAT3, False, (0.0, 0.0, 1.0)))
        self.add_input(NodePin("Metallic", DataType.FLOAT, False, 0.0))
        self.add_input(NodePin("Roughness", DataType.FLOAT, False, 0.5))
        self.add_input(NodePin("AO", DataType.FLOAT, False, 1.0))
        self.add_output(NodePin("Color", DataType.FLOAT3, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.PBR

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        albedo = inputs.get("Albedo", (0.5, 0.5, 0.5))
        ao = inputs.get("AO", 1.0)
        # Simplified evaluation
        return {"Color": tuple(c * ao for c in albedo)}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        albedo = inputs.get("Albedo", "float3(0.5, 0.5, 0.5)")
        normal = inputs.get("Normal", "input.normal")
        metallic = inputs.get("Metallic", "0.0")
        roughness = inputs.get("Roughness", "0.5")
        ao = inputs.get("AO", "1.0")

        lines = [
            f"// PBR BRDF calculation",
            f"float3 F0 = lerp(float3(0.04, 0.04, 0.04), {albedo}, {metallic});",
            f"float3 {output_vars['Color']} = ComputePBR({albedo}, {normal}, {metallic}, {roughness}, F0) * {ao};"
        ]
        return "\n".join(lines)


# ============================================================================
# OUTPUT NODES
# ============================================================================

class PBROutputNode(MaterialNode):
    """Standard PBR output node."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Albedo", DataType.FLOAT3, False, (1.0, 1.0, 1.0), "Base color"))
        self.add_input(NodePin("Normal", DataType.FLOAT3, False, (0.0, 0.0, 1.0), "Surface normal"))
        self.add_input(NodePin("Metallic", DataType.FLOAT, False, 0.0, "Metallic value"))
        self.add_input(NodePin("Roughness", DataType.FLOAT, False, 0.5, "Surface roughness"))
        self.add_input(NodePin("AO", DataType.FLOAT, False, 1.0, "Ambient occlusion"))
        self.add_input(NodePin("Emissive", DataType.FLOAT3, False, (0.0, 0.0, 0.0), "Emissive color"))
        self.add_input(NodePin("Alpha", DataType.FLOAT, False, 1.0, "Opacity"))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.OUTPUT

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        albedo = inputs.get("Albedo", "float3(1, 1, 1)")
        normal = inputs.get("Normal", "float3(0, 0, 1)")
        metallic = inputs.get("Metallic", "0.0")
        roughness = inputs.get("Roughness", "0.5")
        ao = inputs.get("AO", "1.0")
        emissive = inputs.get("Emissive", "float3(0, 0, 0)")
        alpha = inputs.get("Alpha", "1.0")

        lines = [
            f"output.Albedo = {albedo};",
            f"output.Normal = {normal};",
            f"output.Metallic = {metallic};",
            f"output.Roughness = {roughness};",
            f"output.AO = {ao};",
            f"output.Emissive = {emissive};",
            f"output.Alpha = {alpha};"
        ]
        return "\n".join(lines)


class UnlitOutputNode(MaterialNode):
    """Unlit output node."""

    def _setup_pins(self) -> None:
        self.add_input(NodePin("Color", DataType.FLOAT3, False, (1.0, 1.0, 1.0), "Output color"))
        self.add_input(NodePin("Alpha", DataType.FLOAT, False, 1.0, "Opacity"))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.OUTPUT

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        color = inputs.get("Color", "float3(1, 1, 1)")
        alpha = inputs.get("Alpha", "1.0")

        lines = [
            f"output.Color = {color};",
            f"output.Alpha = {alpha};"
        ]
        return "\n".join(lines)


# ============================================================================
# CUSTOM NODES
# ============================================================================

class CustomCodeNode(MaterialNode):
    """Node with custom shader code."""

    def __init__(self, code: str = "", name: str = "Custom"):
        self._code = code
        self._custom_inputs: List[Tuple[str, DataType]] = []
        self._custom_outputs: List[Tuple[str, DataType]] = []
        super().__init__(name)

    def _setup_pins(self) -> None:
        for pin_name, data_type in self._custom_inputs:
            self.add_input(NodePin(pin_name, data_type, False))
        for pin_name, data_type in self._custom_outputs:
            self.add_output(NodePin(pin_name, data_type, True))

    def add_custom_input(self, name: str, data_type: DataType) -> None:
        """Add a custom input pin."""
        self._custom_inputs.append((name, data_type))
        self.add_input(NodePin(name, data_type, False))

    def add_custom_output(self, name: str, data_type: DataType) -> None:
        """Add a custom output pin."""
        self._custom_outputs.append((name, data_type))
        self.add_output(NodePin(name, data_type, True))

    @property
    def category(self) -> NodeCategory:
        return NodeCategory.CUSTOM

    @property
    def code(self) -> str:
        return self._code

    @code.setter
    def code(self, v: str) -> None:
        self._code = v

    def evaluate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Custom code can't be evaluated
        return {name: 0.0 for name, _ in self._custom_outputs}

    def generate_code(self, inputs: Dict[str, str], output_vars: Dict[str, str]) -> str:
        code = self._code
        for name, var in inputs.items():
            code = code.replace(f"${{{name}}}", var)
        for name, var in output_vars.items():
            code = code.replace(f"${{{name}}}", var)
        return code


# Node registry for factory
NODE_REGISTRY: Dict[str, Type[MaterialNode]] = {
    # Input
    "Constant": ConstantNode,
    "Constant2": Constant2Node,
    "Constant3": Constant3Node,
    "Constant4": Constant4Node,
    "Parameter": ParameterNode,
    # Math
    "Add": AddNode,
    "Subtract": SubtractNode,
    "Multiply": MultiplyNode,
    "Divide": DivideNode,
    "Lerp": LerpNode,
    "Clamp": ClampNode,
    "Saturate": SaturateNode,
    "Power": PowerNode,
    "Dot": DotNode,
    "Cross": CrossNode,
    "Normalize": NormalizeNode,
    "Abs": AbsNode,
    "Floor": FloorNode,
    "Ceil": CeilNode,
    "Frac": FracNode,
    "Sin": SinNode,
    "Cos": CosNode,
    "OneMinus": OneMinusNode,
    # Texture
    "TextureSample": TextureSampleNode,
    "UV": UVNode,
    "TilingOffset": TilingOffsetNode,
    "NormalMap": NormalMapNode,
    "Parallax": ParallaxNode,
    # Utility
    "Time": TimeNode,
    "WorldPosition": WorldPositionNode,
    "ViewDirection": ViewDirectionNode,
    "ScreenPosition": ScreenPositionNode,
    "VertexColor": VertexColorNode,
    "Split": SplitNode,
    "Combine": CombineNode,
    # PBR
    "Fresnel": FresnelNode,
    "GGX": GGXNode,
    "Lambert": LambertNode,
    "BRDF": BRDFNode,
    # Output
    "PBROutput": PBROutputNode,
    "UnlitOutput": UnlitOutputNode,
    # Custom
    "Custom": CustomCodeNode,
}

# Reverse registry: class -> registry key name (for serialization)
NODE_CLASS_TO_KEY: Dict[Type[MaterialNode], str] = {
    cls: key for key, cls in NODE_REGISTRY.items()
}
