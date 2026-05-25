"""Node factory - Node creation with defaults and presets."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type

from .material_nodes import (
    MaterialNode, NodeCategory, DataType, NODE_REGISTRY,
    ConstantNode, Constant2Node, Constant3Node, Constant4Node,
    ParameterNode, AddNode, MultiplyNode, LerpNode, TextureSampleNode,
    PBROutputNode, UnlitOutputNode, FresnelNode, NormalMapNode,
    CustomCodeNode
)


@dataclass
class NodePreset:
    """Preset configuration for creating nodes."""
    name: str
    description: str
    node_type: str
    default_values: Dict[str, Any] = field(default_factory=dict)
    custom_config: Dict[str, Any] = field(default_factory=dict)
    category: str = "Default"
    tags: List[str] = field(default_factory=list)


class NodeFactory:
    """Factory for creating material nodes with presets."""

    def __init__(self):
        self._registry: Dict[str, Type[MaterialNode]] = NODE_REGISTRY.copy()
        self._presets: Dict[str, NodePreset] = {}
        self._preset_creators: Dict[str, Callable[[], MaterialNode]] = {}
        self._setup_default_presets()

    def _setup_default_presets(self) -> None:
        """Set up default node presets."""
        # Color presets
        self.register_preset(NodePreset(
            name="White",
            description="White color constant",
            node_type="Constant4",
            custom_config={"value": (1.0, 1.0, 1.0, 1.0)},
            category="Colors",
            tags=["color", "white"]
        ))

        self.register_preset(NodePreset(
            name="Black",
            description="Black color constant",
            node_type="Constant4",
            custom_config={"value": (0.0, 0.0, 0.0, 1.0)},
            category="Colors",
            tags=["color", "black"]
        ))

        self.register_preset(NodePreset(
            name="Red",
            description="Red color constant",
            node_type="Constant4",
            custom_config={"value": (1.0, 0.0, 0.0, 1.0)},
            category="Colors",
            tags=["color", "red"]
        ))

        self.register_preset(NodePreset(
            name="Green",
            description="Green color constant",
            node_type="Constant4",
            custom_config={"value": (0.0, 1.0, 0.0, 1.0)},
            category="Colors",
            tags=["color", "green"]
        ))

        self.register_preset(NodePreset(
            name="Blue",
            description="Blue color constant",
            node_type="Constant4",
            custom_config={"value": (0.0, 0.0, 1.0, 1.0)},
            category="Colors",
            tags=["color", "blue"]
        ))

        self.register_preset(NodePreset(
            name="Gray 50%",
            description="50% gray constant",
            node_type="Constant",
            custom_config={"value": 0.5},
            category="Values",
            tags=["gray", "half"]
        ))

        # PBR presets
        self.register_preset(NodePreset(
            name="Metal Roughness 0.3",
            description="Standard metal roughness",
            node_type="Constant",
            custom_config={"value": 0.3},
            category="PBR",
            tags=["pbr", "roughness", "metal"]
        ))

        self.register_preset(NodePreset(
            name="Dielectric Roughness 0.5",
            description="Standard dielectric roughness",
            node_type="Constant",
            custom_config={"value": 0.5},
            category="PBR",
            tags=["pbr", "roughness", "dielectric"]
        ))

        self.register_preset(NodePreset(
            name="Full Metal",
            description="Fully metallic value",
            node_type="Constant",
            custom_config={"value": 1.0},
            category="PBR",
            tags=["pbr", "metallic", "metal"]
        ))

        self.register_preset(NodePreset(
            name="Non-Metal",
            description="Non-metallic value",
            node_type="Constant",
            custom_config={"value": 0.0},
            category="PBR",
            tags=["pbr", "metallic", "dielectric"]
        ))

        # Normal map presets
        self.register_preset(NodePreset(
            name="Default Normal",
            description="Default up-facing normal",
            node_type="Constant3",
            custom_config={"value": (0.5, 0.5, 1.0)},
            category="Normals",
            tags=["normal", "default"]
        ))

        # UV presets
        self.register_preset(NodePreset(
            name="UV 0",
            description="Primary UV channel",
            node_type="UV",
            custom_config={"uv_channel": 0},
            category="UV",
            tags=["uv", "texcoord"]
        ))

        self.register_preset(NodePreset(
            name="UV 1",
            description="Secondary UV channel (lightmap)",
            node_type="UV",
            custom_config={"uv_channel": 1},
            category="UV",
            tags=["uv", "lightmap"]
        ))

    def register_node_type(self, name: str, node_class: Type[MaterialNode]) -> None:
        """Register a custom node type."""
        self._registry[name] = node_class

    def unregister_node_type(self, name: str) -> None:
        """Unregister a node type."""
        if name in self._registry:
            del self._registry[name]

    def register_preset(self, preset: NodePreset) -> None:
        """Register a node preset."""
        self._presets[preset.name] = preset

    def unregister_preset(self, name: str) -> None:
        """Unregister a preset."""
        if name in self._presets:
            del self._presets[name]

    def register_preset_creator(self, name: str, creator: Callable[[], MaterialNode]) -> None:
        """Register a custom preset creator function."""
        self._preset_creators[name] = creator

    def get_node_types(self) -> List[str]:
        """Get list of available node types."""
        return list(self._registry.keys())

    def get_node_types_by_category(self, category: NodeCategory) -> List[str]:
        """Get node types filtered by category."""
        result = []
        for name, node_class in self._registry.items():
            try:
                # Create temporary instance to check category
                node = node_class()
                if node.category == category:
                    result.append(name)
            except Exception:
                pass
        return result

    def get_presets(self) -> List[NodePreset]:
        """Get list of available presets."""
        return list(self._presets.values())

    def get_presets_by_category(self, category: str) -> List[NodePreset]:
        """Get presets filtered by category."""
        return [p for p in self._presets.values() if p.category == category]

    def get_preset_categories(self) -> List[str]:
        """Get list of preset categories."""
        return list(set(p.category for p in self._presets.values()))

    def search_presets(self, query: str) -> List[NodePreset]:
        """Search presets by name, description, or tags."""
        query_lower = query.lower()
        results = []
        for preset in self._presets.values():
            if (query_lower in preset.name.lower() or
                query_lower in preset.description.lower() or
                any(query_lower in tag for tag in preset.tags)):
                results.append(preset)
        return results

    def create_node(self, node_type: str, name: str = "", **kwargs) -> Optional[MaterialNode]:
        """
        Create a node of the specified type.

        Args:
            node_type: Type of node to create
            name: Optional name for the node
            **kwargs: Additional arguments to pass to node constructor

        Returns:
            Created node or None if type not found
        """
        node_class = self._registry.get(node_type)
        if node_class is None:
            return None

        try:
            if name:
                kwargs['name'] = name
            node = node_class(**kwargs)
            return node
        except Exception as e:
            print(f"Error creating node '{node_type}': {e}")
            return None

    def create_from_preset(self, preset_name: str) -> Optional[MaterialNode]:
        """
        Create a node from a preset.

        Args:
            preset_name: Name of the preset

        Returns:
            Created node or None if preset not found
        """
        # Check for custom creator first
        if preset_name in self._preset_creators:
            return self._preset_creators[preset_name]()

        preset = self._presets.get(preset_name)
        if preset is None:
            return None

        node = self.create_node(preset.node_type)
        if node is None:
            return None

        # Apply custom config
        for key, value in preset.custom_config.items():
            if hasattr(node, key):
                setattr(node, key, value)
            elif hasattr(node, f"_{key}"):
                setattr(node, f"_{key}", value)

        return node

    def create_constant(self, value: float, name: str = "Constant") -> ConstantNode:
        """Create a constant node with a value."""
        node = ConstantNode(value=value, name=name)
        return node

    def create_constant2(self, x: float, y: float, name: str = "Constant2") -> Constant2Node:
        """Create a float2 constant node."""
        node = Constant2Node(value=(x, y), name=name)
        return node

    def create_constant3(self, x: float, y: float, z: float, name: str = "Constant3") -> Constant3Node:
        """Create a float3 constant node."""
        node = Constant3Node(value=(x, y, z), name=name)
        return node

    def create_constant4(self, x: float, y: float, z: float, w: float, name: str = "Constant4") -> Constant4Node:
        """Create a float4 constant node."""
        node = Constant4Node(value=(x, y, z, w), name=name)
        return node

    def create_color(self, r: float, g: float, b: float, a: float = 1.0, name: str = "Color") -> Constant4Node:
        """Create a color constant node."""
        node = Constant4Node(value=(r, g, b, a), name=name)
        return node

    def create_parameter(
        self,
        param_name: str,
        param_type: DataType = DataType.FLOAT,
        name: str = "Parameter"
    ) -> ParameterNode:
        """Create a parameter node."""
        node = ParameterNode(param_name=param_name, param_type=param_type, name=name)
        return node

    def create_texture_sample(self, texture_path: str = "", name: str = "Texture") -> TextureSampleNode:
        """Create a texture sample node."""
        node = TextureSampleNode(texture_path=texture_path, name=name)
        return node

    def create_pbr_output(self, name: str = "PBR Output") -> PBROutputNode:
        """Create a PBR output node."""
        node = PBROutputNode(name=name)
        return node

    def create_unlit_output(self, name: str = "Unlit Output") -> UnlitOutputNode:
        """Create an unlit output node."""
        node = UnlitOutputNode(name=name)
        return node

    def create_custom(self, code: str = "", name: str = "Custom") -> CustomCodeNode:
        """Create a custom code node."""
        node = CustomCodeNode(code=code, name=name)
        return node


# Common node creation patterns
class NodeTemplates:
    """Pre-defined node graph templates."""

    @staticmethod
    def create_pbr_material_nodes(factory: NodeFactory) -> List[MaterialNode]:
        """Create basic PBR material node setup."""
        nodes = []

        # Albedo
        albedo = factory.create_color(0.8, 0.8, 0.8, 1.0, "Albedo")
        albedo.position = (0, 0)
        nodes.append(albedo)

        # Metallic
        metallic = factory.create_constant(0.0, "Metallic")
        metallic.position = (0, 100)
        nodes.append(metallic)

        # Roughness
        roughness = factory.create_constant(0.5, "Roughness")
        roughness.position = (0, 200)
        nodes.append(roughness)

        # Output
        output = factory.create_pbr_output()
        output.position = (300, 100)
        nodes.append(output)

        return nodes

    @staticmethod
    def create_textured_pbr_nodes(factory: NodeFactory) -> List[MaterialNode]:
        """Create textured PBR material node setup."""
        nodes = []

        # UV node
        uv = factory.create_node("UV", "UV Coords")
        uv.position = (-300, 0)
        nodes.append(uv)

        # Albedo texture
        albedo_tex = factory.create_texture_sample("", "Albedo Map")
        albedo_tex.position = (0, 0)
        nodes.append(albedo_tex)

        # Normal texture
        normal_tex = factory.create_texture_sample("", "Normal Map")
        normal_tex.position = (0, 150)
        nodes.append(normal_tex)

        # Normal unpack
        normal_unpack = factory.create_node("NormalMap", "Unpack Normal")
        normal_unpack.position = (200, 150)
        nodes.append(normal_unpack)

        # Metallic/Roughness texture
        mr_tex = factory.create_texture_sample("", "MetallicRoughness")
        mr_tex.position = (0, 300)
        nodes.append(mr_tex)

        # Output
        output = factory.create_pbr_output()
        output.position = (500, 150)
        nodes.append(output)

        return nodes

    @staticmethod
    def create_unlit_material_nodes(factory: NodeFactory) -> List[MaterialNode]:
        """Create basic unlit material node setup."""
        nodes = []

        # Color
        color = factory.create_color(1.0, 1.0, 1.0, 1.0, "Color")
        color.position = (0, 0)
        nodes.append(color)

        # Output
        output = factory.create_unlit_output()
        output.position = (300, 0)
        nodes.append(output)

        return nodes

    @staticmethod
    def create_fresnel_effect_nodes(factory: NodeFactory) -> List[MaterialNode]:
        """Create fresnel rim lighting effect nodes."""
        nodes = []

        # Fresnel
        fresnel = factory.create_node("Fresnel", "Fresnel")
        fresnel.position = (0, 0)
        nodes.append(fresnel)

        # Power input
        power = factory.create_constant(3.0, "Power")
        power.position = (-200, 50)
        nodes.append(power)

        # Color
        rim_color = factory.create_color(0.5, 0.8, 1.0, 1.0, "Rim Color")
        rim_color.position = (0, 150)
        nodes.append(rim_color)

        # Multiply
        multiply = factory.create_node("Multiply", "Multiply")
        multiply.position = (200, 75)
        nodes.append(multiply)

        return nodes

    @staticmethod
    def create_animated_uv_nodes(factory: NodeFactory) -> List[MaterialNode]:
        """Create animated UV scrolling nodes."""
        nodes = []

        # Time
        time = factory.create_node("Time", "Time")
        time.position = (-300, 0)
        nodes.append(time)

        # Speed
        speed = factory.create_constant2(0.5, 0.0, "Speed")
        speed.position = (-300, 100)
        nodes.append(speed)

        # Multiply time by speed
        multiply = factory.create_node("Multiply", "Animate")
        multiply.position = (-100, 50)
        nodes.append(multiply)

        # UV
        uv = factory.create_node("UV", "Base UV")
        uv.position = (-100, 150)
        nodes.append(uv)

        # Add animated offset
        add = factory.create_node("Add", "Offset UV")
        add.position = (100, 100)
        nodes.append(add)

        return nodes


# Global factory instance
_default_factory: Optional[NodeFactory] = None


def get_default_factory() -> NodeFactory:
    """Get the default node factory instance."""
    global _default_factory
    if _default_factory is None:
        _default_factory = NodeFactory()
    return _default_factory
