"""
Material Editor Subsystem

A comprehensive node-based material editing system for the AI Game Engine.

This module provides:
- Node-based material graphs with type-safe connections
- Real-time material preview with configurable lighting
- Shader compilation to HLSL, GLSL, and Metal
- Material instances with parameter overrides
- Material library with organization, search, and favorites

Example Usage:
    from engine.tooling.material_editor import (
        MaterialGraph, MaterialCompiler, MaterialPreview,
        NodeFactory, MaterialLibrary
    )

    # Create a material graph
    graph = MaterialGraph("MyMaterial")

    # Add nodes using factory
    factory = NodeFactory()
    albedo = factory.create_color(0.8, 0.2, 0.2, 1.0, "BaseColor")
    output = factory.create_pbr_output()
    graph.add_node(albedo)
    graph.add_node(output)

    # Connect nodes
    graph.connect(albedo.id, "Value", output.id, "Albedo")

    # Compile to shader
    compiler = MaterialCompiler()
    result = compiler.compile(graph, ShaderLanguage.HLSL)

    # Preview material
    preview = MaterialPreview()
    preview.initialize(512, 512)
    preview.set_material_data({"albedo": (0.8, 0.2, 0.2)})
    preview.render()
"""

# Material Parameters
from .material_parameters import (
    ParameterType,
    ParameterSemantics,
    ParameterRange,
    TextureSettings,
    MaterialParameter,
    ScalarParameter,
    VectorParameter,
    ColorParameter,
    TextureParameter,
    BooleanParameter,
    IntegerParameter,
    ParameterCollection,
)

# Material Nodes
from .material_nodes import (
    NodeCategory,
    DataType,
    NodePin,
    MaterialNode,
    # Input nodes
    ConstantNode,
    Constant2Node,
    Constant3Node,
    Constant4Node,
    ParameterNode,
    # Math nodes
    AddNode,
    SubtractNode,
    MultiplyNode,
    DivideNode,
    LerpNode,
    ClampNode,
    SaturateNode,
    PowerNode,
    DotNode,
    CrossNode,
    NormalizeNode,
    AbsNode,
    FloorNode,
    CeilNode,
    FracNode,
    SinNode,
    CosNode,
    OneMinusNode,
    # Texture nodes
    TextureSampleNode,
    UVNode,
    TilingOffsetNode,
    NormalMapNode,
    ParallaxNode,
    # Utility nodes
    TimeNode,
    WorldPositionNode,
    ViewDirectionNode,
    ScreenPositionNode,
    VertexColorNode,
    SplitNode,
    CombineNode,
    # PBR nodes
    FresnelNode,
    GGXNode,
    LambertNode,
    BRDFNode,
    # Output nodes
    PBROutputNode,
    UnlitOutputNode,
    # Custom nodes
    CustomCodeNode,
    # Registry
    NODE_REGISTRY,
)

# Connection Validator
from .connection_validator import (
    ValidationError,
    ValidationResult,
    Connection,
    TypeConversion,
    ConnectionValidator,
)

# Node Factory
from .node_factory import (
    NodePreset,
    NodeFactory,
    NodeTemplates,
    get_default_factory,
)

# Material Graph
from .material_graph import (
    GraphState,
    GraphError,
    GraphMetadata,
    MaterialGraph,
)

# Material Compiler
from .material_compiler import (
    ShaderLanguage,
    ShaderStage,
    ShaderParameter,
    TextureBinding,
    CompilationResult,
    ShaderGenerator,
    HLSLGenerator,
    GLSLGenerator,
    MetalGenerator,
    MaterialCompiler,
)

# Material Preview
from .material_preview import (
    PreviewShape,
    LightType,
    PreviewLight,
    PreviewCamera,
    LightingPreset,
    PreviewSettings,
    PreviewRenderer,
    NullPreviewRenderer,
    MaterialPreview,
)

# Material Instances
from .material_instances import (
    InstanceState,
    ParameterOverride,
    MaterialDefinition,
    MaterialInstance,
    MaterialInstanceManager,
)

# Material Library
from .material_library import (
    LibraryItemType,
    LibraryMetadata,
    LibraryItem,
    SearchFilter,
    SortOrder,
    MaterialLibrary,
)

__all__ = [
    # Parameters
    "ParameterType",
    "ParameterSemantics",
    "ParameterRange",
    "TextureSettings",
    "MaterialParameter",
    "ScalarParameter",
    "VectorParameter",
    "ColorParameter",
    "TextureParameter",
    "BooleanParameter",
    "IntegerParameter",
    "ParameterCollection",
    # Nodes
    "NodeCategory",
    "DataType",
    "NodePin",
    "MaterialNode",
    "ConstantNode",
    "Constant2Node",
    "Constant3Node",
    "Constant4Node",
    "ParameterNode",
    "AddNode",
    "SubtractNode",
    "MultiplyNode",
    "DivideNode",
    "LerpNode",
    "ClampNode",
    "SaturateNode",
    "PowerNode",
    "DotNode",
    "CrossNode",
    "NormalizeNode",
    "AbsNode",
    "FloorNode",
    "CeilNode",
    "FracNode",
    "SinNode",
    "CosNode",
    "OneMinusNode",
    "TextureSampleNode",
    "UVNode",
    "TilingOffsetNode",
    "NormalMapNode",
    "ParallaxNode",
    "TimeNode",
    "WorldPositionNode",
    "ViewDirectionNode",
    "ScreenPositionNode",
    "VertexColorNode",
    "SplitNode",
    "CombineNode",
    "FresnelNode",
    "GGXNode",
    "LambertNode",
    "BRDFNode",
    "PBROutputNode",
    "UnlitOutputNode",
    "CustomCodeNode",
    "NODE_REGISTRY",
    # Validator
    "ValidationError",
    "ValidationResult",
    "Connection",
    "TypeConversion",
    "ConnectionValidator",
    # Factory
    "NodePreset",
    "NodeFactory",
    "NodeTemplates",
    "get_default_factory",
    # Graph
    "GraphState",
    "GraphError",
    "GraphMetadata",
    "MaterialGraph",
    # Compiler
    "ShaderLanguage",
    "ShaderStage",
    "ShaderParameter",
    "TextureBinding",
    "CompilationResult",
    "ShaderGenerator",
    "HLSLGenerator",
    "GLSLGenerator",
    "MetalGenerator",
    "MaterialCompiler",
    # Preview
    "PreviewShape",
    "LightType",
    "PreviewLight",
    "PreviewCamera",
    "LightingPreset",
    "PreviewSettings",
    "PreviewRenderer",
    "NullPreviewRenderer",
    "MaterialPreview",
    # Instances
    "InstanceState",
    "ParameterOverride",
    "MaterialDefinition",
    "MaterialInstance",
    "MaterialInstanceManager",
    # Library
    "LibraryItemType",
    "LibraryMetadata",
    "LibraryItem",
    "SearchFilter",
    "SortOrder",
    "MaterialLibrary",
]
