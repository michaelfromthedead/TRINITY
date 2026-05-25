"""Node-based material authoring system.

This module provides a graph-based material authoring system:
- MaterialNode: Base class for all nodes
- MathNodes: Add, Multiply, Lerp, etc.
- TextureNodes: Sample, UV manipulation
- MaterialGraph: Connect nodes, compile to shader
- GraphCompiler: Convert graph to shader code
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from engine.core.math.vec import Vec2, Vec3, Vec4

__all__ = [
    "DataType",
    "NodePort",
    "NodeConnection",
    "MaterialNode",
    "ConstantNode",
    "ParameterNode",
    "TextureSampleNode",
    "UVNode",
    "AddNode",
    "SubtractNode",
    "MultiplyNode",
    "DivideNode",
    "LerpNode",
    "ClampNode",
    "DotNode",
    "NormalizeNode",
    "PowerNode",
    "SqrtNode",
    "AbsNode",
    "FracNode",
    "FloorNode",
    "CeilNode",
    "MinNode",
    "MaxNode",
    "SinNode",
    "CosNode",
    "OneMinus",
    "ComponentMask",
    "AppendNode",
    "OutputNode",
    "MaterialGraph",
    "GraphCompiler",
    "GraphValidationError",
]


class DataType(Enum):
    """Data types for node connections."""
    FLOAT = auto()
    VEC2 = auto()
    VEC3 = auto()
    VEC4 = auto()
    INT = auto()
    BOOL = auto()
    TEXTURE2D = auto()
    TEXTURECUBE = auto()
    SAMPLER = auto()


@dataclass(slots=True)
class NodePort:
    """Input or output port on a material node.

    Attributes:
        name: Port name
        data_type: Data type of the port
        is_output: Whether this is an output (True) or input (False)
        default_value: Default value for unconnected inputs
        hidden: Whether to hide in UI
    """
    name: str
    data_type: DataType
    is_output: bool = False
    default_value: Any = None
    hidden: bool = False

    def is_compatible_with(self, other: NodePort) -> bool:
        """Check if this port can connect to another."""
        if self.is_output == other.is_output:
            return False  # Can't connect same types

        # Exact match
        if self.data_type == other.data_type:
            return True

        # Float can promote to any vector type
        if self.data_type == DataType.FLOAT and other.data_type in (
            DataType.VEC2,
            DataType.VEC3,
            DataType.VEC4,
        ):
            return True

        if other.data_type == DataType.FLOAT and self.data_type in (
            DataType.VEC2,
            DataType.VEC3,
            DataType.VEC4,
        ):
            return True

        return False


@dataclass(slots=True)
class NodeConnection:
    """Connection between two node ports.

    Attributes:
        source_node: Node providing the value
        source_port: Output port name on source
        target_node: Node receiving the value
        target_port: Input port name on target
    """
    source_node: str  # Node ID
    source_port: str
    target_node: str  # Node ID
    target_port: str

    def get_id(self) -> str:
        """Get unique connection identifier."""
        return (
            f"{self.source_node}.{self.source_port}->"
            f"{self.target_node}.{self.target_port}"
        )


class MaterialNode(ABC):
    """Base class for all material graph nodes.

    Nodes are the building blocks of material graphs. Each node has
    input and output ports that can be connected to other nodes.

    Attributes:
        node_id: Unique identifier
        name: Display name
        inputs: Input port definitions
        outputs: Output port definitions
        position: Position in graph editor (x, y)
    """

    __slots__ = (
        "_node_id",
        "_name",
        "_inputs",
        "_outputs",
        "_position",
        "_metadata",
    )

    def __init__(self, name: Optional[str] = None) -> None:
        self._node_id = str(uuid.uuid4())
        self._name = name or self.__class__.__name__
        self._inputs: Dict[str, NodePort] = {}
        self._outputs: Dict[str, NodePort] = {}
        self._position: Tuple[float, float] = (0.0, 0.0)
        self._metadata: Dict[str, Any] = {}

        self._define_ports()

    @property
    def node_id(self) -> str:
        return self._node_id

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
    def inputs(self) -> Dict[str, NodePort]:
        return self._inputs.copy()

    @property
    def outputs(self) -> Dict[str, NodePort]:
        return self._outputs.copy()

    def add_input(
        self,
        name: str,
        data_type: DataType,
        default_value: Any = None,
    ) -> None:
        """Add an input port."""
        self._inputs[name] = NodePort(
            name=name,
            data_type=data_type,
            is_output=False,
            default_value=default_value,
        )

    def add_output(self, name: str, data_type: DataType) -> None:
        """Add an output port."""
        self._outputs[name] = NodePort(
            name=name,
            data_type=data_type,
            is_output=True,
        )

    def get_input(self, name: str) -> Optional[NodePort]:
        """Get input port by name."""
        return self._inputs.get(name)

    def get_output(self, name: str) -> Optional[NodePort]:
        """Get output port by name."""
        return self._outputs.get(name)

    @abstractmethod
    def _define_ports(self) -> None:
        """Define input and output ports. Override in subclasses."""
        pass

    @abstractmethod
    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        """Generate shader code for this node.

        Args:
            input_vars: Map of input port name to variable name
            output_var: Variable name for output

        Returns:
            Shader code string
        """
        pass

    def get_node_type(self) -> str:
        """Get the node type identifier."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self._node_id[:8]}>"


# --- Constant and Parameter Nodes ---


class ConstantNode(MaterialNode):
    """Node that outputs a constant value."""

    __slots__ = ("_value", "_data_type")

    def __init__(
        self,
        value: Any,
        data_type: DataType = DataType.FLOAT,
        name: Optional[str] = None,
    ) -> None:
        self._value = value
        self._data_type = data_type
        super().__init__(name)

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, val: Any) -> None:
        self._value = val

    def _define_ports(self) -> None:
        self.add_output("value", self._data_type)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        if self._data_type == DataType.FLOAT:
            return f"float {output_var} = {self._value};"
        elif self._data_type == DataType.VEC2:
            return f"vec2 {output_var} = vec2({self._value.x}, {self._value.y});"
        elif self._data_type == DataType.VEC3:
            return f"vec3 {output_var} = vec3({self._value.x}, {self._value.y}, {self._value.z});"
        elif self._data_type == DataType.VEC4:
            return f"vec4 {output_var} = vec4({self._value.x}, {self._value.y}, {self._value.z}, {self._value.w});"
        return f"// Unknown type for {output_var}"


class ParameterNode(MaterialNode):
    """Node that exposes a material parameter."""

    __slots__ = ("_param_name", "_data_type", "_default_value")

    def __init__(
        self,
        param_name: str,
        data_type: DataType = DataType.FLOAT,
        default_value: Any = None,
        name: Optional[str] = None,
    ) -> None:
        self._param_name = param_name
        self._data_type = data_type
        self._default_value = default_value
        super().__init__(name or param_name)

    @property
    def param_name(self) -> str:
        return self._param_name

    def _define_ports(self) -> None:
        self.add_output("value", self._data_type)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        type_map = {
            DataType.FLOAT: "float",
            DataType.VEC2: "vec2",
            DataType.VEC3: "vec3",
            DataType.VEC4: "vec4",
        }
        glsl_type = type_map.get(self._data_type, "float")
        return f"{glsl_type} {output_var} = u_{self._param_name};"


# --- Texture Nodes ---


class TextureSampleNode(MaterialNode):
    """Node that samples a texture."""

    __slots__ = ("_texture_name",)

    def __init__(
        self,
        texture_name: str,
        name: Optional[str] = None,
    ) -> None:
        self._texture_name = texture_name
        super().__init__(name or f"Sample_{texture_name}")

    @property
    def texture_name(self) -> str:
        return self._texture_name

    def _define_ports(self) -> None:
        self.add_input("uv", DataType.VEC2, default_value=Vec2(0, 0))
        self.add_output("rgba", DataType.VEC4)
        self.add_output("rgb", DataType.VEC3)
        self.add_output("r", DataType.FLOAT)
        self.add_output("g", DataType.FLOAT)
        self.add_output("b", DataType.FLOAT)
        self.add_output("a", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        uv = input_vars.get("uv", "v_uv")
        lines = [
            f"vec4 {output_var}_rgba = texture(tex_{self._texture_name}, {uv});",
            f"vec3 {output_var}_rgb = {output_var}_rgba.rgb;",
            f"float {output_var}_r = {output_var}_rgba.r;",
            f"float {output_var}_g = {output_var}_rgba.g;",
            f"float {output_var}_b = {output_var}_rgba.b;",
            f"float {output_var}_a = {output_var}_rgba.a;",
        ]
        return "\n".join(lines)


class UVNode(MaterialNode):
    """Node that provides UV coordinates."""

    __slots__ = ("_uv_index",)

    def __init__(self, uv_index: int = 0, name: Optional[str] = None) -> None:
        self._uv_index = uv_index
        super().__init__(name or f"UV{uv_index}")

    def _define_ports(self) -> None:
        self.add_output("uv", DataType.VEC2)
        self.add_output("u", DataType.FLOAT)
        self.add_output("v", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        uv_attr = f"v_uv{self._uv_index}" if self._uv_index > 0 else "v_uv"
        return (
            f"vec2 {output_var}_uv = {uv_attr};\n"
            f"float {output_var}_u = {uv_attr}.x;\n"
            f"float {output_var}_v = {uv_attr}.y;"
        )


# --- Math Nodes ---


class AddNode(MaterialNode):
    """Add two values."""

    def _define_ports(self) -> None:
        self.add_input("a", DataType.FLOAT)
        self.add_input("b", DataType.FLOAT)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        a = input_vars.get("a", "0.0")
        b = input_vars.get("b", "0.0")
        return f"float {output_var} = {a} + {b};"


class SubtractNode(MaterialNode):
    """Subtract two values."""

    def _define_ports(self) -> None:
        self.add_input("a", DataType.FLOAT)
        self.add_input("b", DataType.FLOAT)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        a = input_vars.get("a", "0.0")
        b = input_vars.get("b", "0.0")
        return f"float {output_var} = {a} - {b};"


class MultiplyNode(MaterialNode):
    """Multiply two values."""

    def _define_ports(self) -> None:
        self.add_input("a", DataType.FLOAT)
        self.add_input("b", DataType.FLOAT)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        a = input_vars.get("a", "1.0")
        b = input_vars.get("b", "1.0")
        return f"float {output_var} = {a} * {b};"


class DivideNode(MaterialNode):
    """Divide two values."""

    def _define_ports(self) -> None:
        self.add_input("a", DataType.FLOAT)
        self.add_input("b", DataType.FLOAT, default_value=1.0)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        a = input_vars.get("a", "0.0")
        b = input_vars.get("b", "1.0")
        return f"float {output_var} = {a} / max({b}, 0.0001);"


class LerpNode(MaterialNode):
    """Linear interpolation between two values."""

    def _define_ports(self) -> None:
        self.add_input("a", DataType.FLOAT)
        self.add_input("b", DataType.FLOAT)
        self.add_input("t", DataType.FLOAT, default_value=0.5)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        a = input_vars.get("a", "0.0")
        b = input_vars.get("b", "1.0")
        t = input_vars.get("t", "0.5")
        return f"float {output_var} = mix({a}, {b}, {t});"


class ClampNode(MaterialNode):
    """Clamp value between min and max."""

    def _define_ports(self) -> None:
        self.add_input("value", DataType.FLOAT)
        self.add_input("min", DataType.FLOAT, default_value=0.0)
        self.add_input("max", DataType.FLOAT, default_value=1.0)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        value = input_vars.get("value", "0.0")
        min_val = input_vars.get("min", "0.0")
        max_val = input_vars.get("max", "1.0")
        return f"float {output_var} = clamp({value}, {min_val}, {max_val});"


class DotNode(MaterialNode):
    """Dot product of two vectors."""

    def _define_ports(self) -> None:
        self.add_input("a", DataType.VEC3)
        self.add_input("b", DataType.VEC3)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        a = input_vars.get("a", "vec3(0.0)")
        b = input_vars.get("b", "vec3(0.0)")
        return f"float {output_var} = dot({a}, {b});"


class NormalizeNode(MaterialNode):
    """Normalize a vector."""

    def _define_ports(self) -> None:
        self.add_input("vector", DataType.VEC3)
        self.add_output("result", DataType.VEC3)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        vector = input_vars.get("vector", "vec3(0.0, 1.0, 0.0)")
        return f"vec3 {output_var} = normalize({vector});"


class PowerNode(MaterialNode):
    """Raise value to a power."""

    def _define_ports(self) -> None:
        self.add_input("base", DataType.FLOAT)
        self.add_input("exponent", DataType.FLOAT, default_value=2.0)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        base = input_vars.get("base", "1.0")
        exp = input_vars.get("exponent", "2.0")
        return f"float {output_var} = pow({base}, {exp});"


class SqrtNode(MaterialNode):
    """Square root of value."""

    def _define_ports(self) -> None:
        self.add_input("value", DataType.FLOAT)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        value = input_vars.get("value", "1.0")
        return f"float {output_var} = sqrt(max({value}, 0.0));"


class AbsNode(MaterialNode):
    """Absolute value."""

    def _define_ports(self) -> None:
        self.add_input("value", DataType.FLOAT)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        value = input_vars.get("value", "0.0")
        return f"float {output_var} = abs({value});"


class FracNode(MaterialNode):
    """Fractional part of value."""

    def _define_ports(self) -> None:
        self.add_input("value", DataType.FLOAT)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        value = input_vars.get("value", "0.0")
        return f"float {output_var} = fract({value});"


class FloorNode(MaterialNode):
    """Floor of value."""

    def _define_ports(self) -> None:
        self.add_input("value", DataType.FLOAT)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        value = input_vars.get("value", "0.0")
        return f"float {output_var} = floor({value});"


class CeilNode(MaterialNode):
    """Ceiling of value."""

    def _define_ports(self) -> None:
        self.add_input("value", DataType.FLOAT)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        value = input_vars.get("value", "0.0")
        return f"float {output_var} = ceil({value});"


class MinNode(MaterialNode):
    """Minimum of two values."""

    def _define_ports(self) -> None:
        self.add_input("a", DataType.FLOAT)
        self.add_input("b", DataType.FLOAT)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        a = input_vars.get("a", "0.0")
        b = input_vars.get("b", "0.0")
        return f"float {output_var} = min({a}, {b});"


class MaxNode(MaterialNode):
    """Maximum of two values."""

    def _define_ports(self) -> None:
        self.add_input("a", DataType.FLOAT)
        self.add_input("b", DataType.FLOAT)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        a = input_vars.get("a", "0.0")
        b = input_vars.get("b", "0.0")
        return f"float {output_var} = max({a}, {b});"


class SinNode(MaterialNode):
    """Sine of value."""

    def _define_ports(self) -> None:
        self.add_input("value", DataType.FLOAT)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        value = input_vars.get("value", "0.0")
        return f"float {output_var} = sin({value});"


class CosNode(MaterialNode):
    """Cosine of value."""

    def _define_ports(self) -> None:
        self.add_input("value", DataType.FLOAT)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        value = input_vars.get("value", "0.0")
        return f"float {output_var} = cos({value});"


class OneMinus(MaterialNode):
    """One minus value (1 - x)."""

    def _define_ports(self) -> None:
        self.add_input("value", DataType.FLOAT)
        self.add_output("result", DataType.FLOAT)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        value = input_vars.get("value", "0.0")
        return f"float {output_var} = 1.0 - {value};"


class ComponentMask(MaterialNode):
    """Extract components from a vector."""

    __slots__ = ("_components",)

    def __init__(
        self,
        components: str = "xyz",
        name: Optional[str] = None,
    ) -> None:
        self._components = components
        super().__init__(name)

    def _define_ports(self) -> None:
        self.add_input("vector", DataType.VEC4)
        # Output type depends on number of components
        if len(self._components) == 1:
            self.add_output("result", DataType.FLOAT)
        elif len(self._components) == 2:
            self.add_output("result", DataType.VEC2)
        elif len(self._components) == 3:
            self.add_output("result", DataType.VEC3)
        else:
            self.add_output("result", DataType.VEC4)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        vector = input_vars.get("vector", "vec4(0.0)")
        type_map = {1: "float", 2: "vec2", 3: "vec3", 4: "vec4"}
        out_type = type_map.get(len(self._components), "float")
        return f"{out_type} {output_var} = {vector}.{self._components};"


class AppendNode(MaterialNode):
    """Append values to create a vector."""

    def _define_ports(self) -> None:
        self.add_input("a", DataType.FLOAT)
        self.add_input("b", DataType.FLOAT)
        self.add_output("result", DataType.VEC2)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        a = input_vars.get("a", "0.0")
        b = input_vars.get("b", "0.0")
        return f"vec2 {output_var} = vec2({a}, {b});"


class OutputNode(MaterialNode):
    """Final output node for material graph."""

    def _define_ports(self) -> None:
        self.add_input("base_color", DataType.VEC3, default_value=Vec3(1, 1, 1))
        self.add_input("metallic", DataType.FLOAT, default_value=0.0)
        self.add_input("roughness", DataType.FLOAT, default_value=0.5)
        self.add_input("normal", DataType.VEC3, default_value=Vec3(0, 0, 1))
        self.add_input("emissive", DataType.VEC3, default_value=Vec3(0, 0, 0))
        self.add_input("ao", DataType.FLOAT, default_value=1.0)
        self.add_input("opacity", DataType.FLOAT, default_value=1.0)

    def generate_code(
        self,
        input_vars: Dict[str, str],
        output_var: str,
    ) -> str:
        base_color = input_vars.get("base_color", "vec3(1.0)")
        metallic = input_vars.get("metallic", "0.0")
        roughness = input_vars.get("roughness", "0.5")
        normal = input_vars.get("normal", "vec3(0.0, 0.0, 1.0)")
        emissive = input_vars.get("emissive", "vec3(0.0)")
        ao = input_vars.get("ao", "1.0")
        opacity = input_vars.get("opacity", "1.0")

        return f"""
    // Material outputs
    out_BaseColor = {base_color};
    out_Metallic = {metallic};
    out_Roughness = {roughness};
    out_Normal = {normal};
    out_Emissive = {emissive};
    out_AO = {ao};
    out_Opacity = {opacity};
"""


class GraphValidationError(Exception):
    """Raised when material graph validation fails."""

    def __init__(
        self,
        message: str,
        node_id: Optional[str] = None,
        port_name: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.node_id = node_id
        self.port_name = port_name


class MaterialGraph:
    """Container for a node-based material.

    Manages nodes and connections, provides validation and compilation.

    Attributes:
        name: Graph name
        nodes: Dictionary of nodes by ID
        connections: List of connections
        output_node: The output node (required)
    """

    __slots__ = (
        "_name",
        "_nodes",
        "_connections",
        "_output_node",
        "_parameters",
        "_textures",
    )

    def __init__(self, name: str = "MaterialGraph") -> None:
        self._name = name
        self._nodes: Dict[str, MaterialNode] = {}
        self._connections: List[NodeConnection] = []
        self._output_node: Optional[OutputNode] = None
        self._parameters: Dict[str, ParameterNode] = {}
        self._textures: Dict[str, TextureSampleNode] = {}

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def nodes(self) -> Dict[str, MaterialNode]:
        return self._nodes.copy()

    @property
    def connections(self) -> List[NodeConnection]:
        return self._connections.copy()

    @property
    def output_node(self) -> Optional[OutputNode]:
        return self._output_node

    def add_node(self, node: MaterialNode) -> str:
        """Add a node to the graph.

        Args:
            node: Node to add

        Returns:
            Node ID
        """
        self._nodes[node.node_id] = node

        if isinstance(node, OutputNode):
            if self._output_node is not None:
                raise GraphValidationError("Graph can only have one output node")
            self._output_node = node
        elif isinstance(node, ParameterNode):
            self._parameters[node.param_name] = node
        elif isinstance(node, TextureSampleNode):
            self._textures[node.texture_name] = node

        return node.node_id

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its connections.

        Args:
            node_id: Node to remove
        """
        node = self._nodes.pop(node_id, None)
        if node is None:
            return

        if isinstance(node, OutputNode):
            self._output_node = None
        elif isinstance(node, ParameterNode):
            self._parameters.pop(node.param_name, None)
        elif isinstance(node, TextureSampleNode):
            self._textures.pop(node.texture_name, None)

        # Remove associated connections
        self._connections = [
            c for c in self._connections
            if c.source_node != node_id and c.target_node != node_id
        ]

    def get_node(self, node_id: str) -> Optional[MaterialNode]:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def connect(
        self,
        source_node: Union[str, MaterialNode],
        source_port: str,
        target_node: Union[str, MaterialNode],
        target_port: str,
    ) -> NodeConnection:
        """Connect two nodes.

        Args:
            source_node: Source node or ID
            source_port: Output port name
            target_node: Target node or ID
            target_port: Input port name

        Returns:
            Created connection

        Raises:
            GraphValidationError: If connection is invalid
        """
        source_id = (
            source_node if isinstance(source_node, str) else source_node.node_id
        )
        target_id = (
            target_node if isinstance(target_node, str) else target_node.node_id
        )

        src = self._nodes.get(source_id)
        tgt = self._nodes.get(target_id)

        if src is None:
            raise GraphValidationError(f"Source node not found: {source_id}")
        if tgt is None:
            raise GraphValidationError(f"Target node not found: {target_id}")

        src_port = src.get_output(source_port)
        tgt_port = tgt.get_input(target_port)

        if src_port is None:
            raise GraphValidationError(
                f"Output port not found: {source_port}",
                node_id=source_id,
            )
        if tgt_port is None:
            raise GraphValidationError(
                f"Input port not found: {target_port}",
                node_id=target_id,
            )

        if not src_port.is_compatible_with(tgt_port):
            raise GraphValidationError(
                f"Incompatible types: {src_port.data_type} -> {tgt_port.data_type}"
            )

        # Check for existing connection to target port
        for conn in self._connections:
            if conn.target_node == target_id and conn.target_port == target_port:
                raise GraphValidationError(
                    f"Input port already connected: {target_port}",
                    node_id=target_id,
                )

        connection = NodeConnection(
            source_node=source_id,
            source_port=source_port,
            target_node=target_id,
            target_port=target_port,
        )
        self._connections.append(connection)
        return connection

    def disconnect(
        self,
        target_node: Union[str, MaterialNode],
        target_port: str,
    ) -> bool:
        """Disconnect an input port.

        Args:
            target_node: Target node or ID
            target_port: Input port name

        Returns:
            True if connection was removed
        """
        target_id = (
            target_node if isinstance(target_node, str) else target_node.node_id
        )

        for i, conn in enumerate(self._connections):
            if conn.target_node == target_id and conn.target_port == target_port:
                del self._connections[i]
                return True
        return False

    def get_input_connection(
        self,
        node_id: str,
        port_name: str,
    ) -> Optional[NodeConnection]:
        """Get connection to an input port."""
        for conn in self._connections:
            if conn.target_node == node_id and conn.target_port == port_name:
                return conn
        return None

    def get_output_connections(
        self,
        node_id: str,
        port_name: str,
    ) -> List[NodeConnection]:
        """Get all connections from an output port."""
        return [
            conn for conn in self._connections
            if conn.source_node == node_id and conn.source_port == port_name
        ]

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the graph structure.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: List[str] = []

        if self._output_node is None:
            errors.append("Graph must have an output node")

        # Check for cycles
        if self._has_cycle():
            errors.append("Graph contains a cycle")

        # Check all connections are valid
        for conn in self._connections:
            if conn.source_node not in self._nodes:
                errors.append(f"Invalid connection source: {conn.source_node}")
            if conn.target_node not in self._nodes:
                errors.append(f"Invalid connection target: {conn.target_node}")

        return len(errors) == 0, errors

    def _has_cycle(self) -> bool:
        """Check if graph contains cycles using DFS."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for conn in self._connections:
                if conn.source_node == node_id:
                    if conn.target_node not in visited:
                        if dfs(conn.target_node):
                            return True
                    elif conn.target_node in rec_stack:
                        return True

            rec_stack.discard(node_id)
            return False

        for node_id in self._nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    def get_topological_order(self) -> List[str]:
        """Get nodes in topological order for code generation."""
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        for conn in self._connections:
            in_degree[conn.target_node] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for conn in self._connections:
                if conn.source_node == node_id:
                    in_degree[conn.target_node] -= 1
                    if in_degree[conn.target_node] == 0:
                        queue.append(conn.target_node)

        return result

    def get_parameters(self) -> List[ParameterNode]:
        """Get all parameter nodes."""
        return list(self._parameters.values())

    def get_textures(self) -> List[TextureSampleNode]:
        """Get all texture nodes."""
        return list(self._textures.values())


class GraphCompiler:
    """Compiles a MaterialGraph to shader code.

    Attributes:
        target_language: Target shader language
    """

    __slots__ = ("_target_language",)

    def __init__(self, target_language: str = "glsl") -> None:
        self._target_language = target_language

    def compile(self, graph: MaterialGraph) -> str:
        """Compile graph to shader code.

        Args:
            graph: Material graph to compile

        Returns:
            Generated shader code

        Raises:
            GraphValidationError: If graph is invalid
        """
        is_valid, errors = graph.validate()
        if not is_valid:
            raise GraphValidationError(
                f"Invalid graph: {', '.join(errors)}"
            )

        lines: List[str] = []
        var_counter = 0

        def get_var() -> str:
            nonlocal var_counter
            var_counter += 1
            return f"_v{var_counter}"

        # Generate uniform declarations
        lines.append("// Uniforms")
        for param in graph.get_parameters():
            type_map = {
                DataType.FLOAT: "float",
                DataType.VEC2: "vec2",
                DataType.VEC3: "vec3",
                DataType.VEC4: "vec4",
            }
            glsl_type = type_map.get(param._data_type, "float")
            lines.append(f"uniform {glsl_type} u_{param.param_name};")

        # Generate sampler declarations
        lines.append("\n// Samplers")
        for tex in graph.get_textures():
            lines.append(f"uniform sampler2D tex_{tex.texture_name};")

        lines.append("\n// Main shader code")
        lines.append("void materialMain() {")

        # Track output variable for each node
        node_outputs: Dict[str, Dict[str, str]] = {}

        # Generate code in topological order
        for node_id in graph.get_topological_order():
            node = graph.get_node(node_id)
            if node is None:
                continue

            # Build input variable map
            input_vars: Dict[str, str] = {}
            for port_name, port in node.inputs.items():
                conn = graph.get_input_connection(node_id, port_name)
                if conn:
                    src_outputs = node_outputs.get(conn.source_node, {})
                    input_vars[port_name] = src_outputs.get(
                        conn.source_port,
                        self._default_value(port.data_type),
                    )
                elif port.default_value is not None:
                    input_vars[port_name] = self._format_value(port.default_value)

            # Generate output variable
            output_var = get_var()
            code = node.generate_code(input_vars, output_var)
            lines.append(f"    // {node.name}")
            for line in code.strip().split("\n"):
                lines.append(f"    {line}")

            # Track outputs for this node
            node_outputs[node_id] = {}
            for port_name in node.outputs:
                node_outputs[node_id][port_name] = f"{output_var}_{port_name}" if len(node.outputs) > 1 else output_var

        lines.append("}")

        return "\n".join(lines)

    def _default_value(self, data_type: DataType) -> str:
        """Get default value string for a data type."""
        defaults = {
            DataType.FLOAT: "0.0",
            DataType.VEC2: "vec2(0.0)",
            DataType.VEC3: "vec3(0.0)",
            DataType.VEC4: "vec4(0.0)",
            DataType.INT: "0",
            DataType.BOOL: "false",
        }
        return defaults.get(data_type, "0.0")

    def _format_value(self, value: Any) -> str:
        """Format a value for shader code."""
        if isinstance(value, Vec2):
            return f"vec2({value.x}, {value.y})"
        elif isinstance(value, Vec3):
            return f"vec3({value.x}, {value.y}, {value.z})"
        elif isinstance(value, Vec4):
            return f"vec4({value.x}, {value.y}, {value.z}, {value.w})"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, float):
            return f"{value}"
        elif isinstance(value, int):
            return f"{value}"
        return str(value)
