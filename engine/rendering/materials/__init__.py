"""Materials & Shading subsystem.

This module provides comprehensive material management for the rendering layer:

Core Material System:
- MaterialTemplate: Base shader definition with parameter schema
- MaterialInstance: Override parameters for a template
- MaterialFunction: Reusable shader snippets
- MaterialLayer: Composable material stacking
- MaterialSystem: Resource managing all materials

PBR Model:
- PBRParameters: Core PBR metallic-roughness parameters
- PBRMaterial: Component with tracked descriptors for dirty flags
- PBRTextureSet: Standard PBR texture map bindings

Shader Compilation:
- ShaderSource: HLSL/GLSL/Metal source management
- ShaderPermutation: Static permutations (compile-time variants)
- PSOCache: Pipeline State Object caching
- ShaderCompiler: Compile variants, hot-reload support
- PermutationKey: Variant selection

Material Graph:
- MaterialNode: Base class for all nodes
- Math nodes (Add, Multiply, Lerp, etc.)
- Texture nodes (Sample, UV manipulation)
- MaterialGraph: Connect nodes, compile to shader
- GraphCompiler: Convert graph to shader code

Material Functions:
- Common shader functions (Fresnel, NormalBlend, etc.)
- MaterialFunctionLibrary singleton

Advanced Shading Models:
- SubsurfaceScattering: Burley diffusion profiles
- ClearCoat: Secondary specular layer
- Anisotropy: Directional roughness
- Sheen: Fabric, velvet
- Iridescence: Thin film interference
- Transmission: Glass, thin surfaces
"""

# Constants
from engine.rendering.materials.constants import (
    PBRParameterRange,
    PBR_METALLIC_RANGE,
    PBR_ROUGHNESS_RANGE,
    PBR_NORMAL_SCALE_RANGE,
    PBR_AO_RANGE,
    PBR_BASE_COLOR_RANGE,
    PBR_EMISSIVE_MIN,
    PSO_CACHE_DEFAULT_MAX_SIZE,
    HOT_RELOAD_POLL_INTERVAL_SECONDS,
    SHADER_HASH_LENGTH,
    SAFE_DIVISION_EPSILON,
)

# Material System
from engine.rendering.materials.material_system import (
    MaterialDomain,
    BlendMode,
    ShadingModel,
    ParameterType,
    MaterialParameter,
    MaterialTemplate,
    MaterialInstance,
    MaterialFunction,
    MaterialLayer,
    MaterialSystem,
    DirtyFlags,
)

# PBR Model
from engine.rendering.materials.pbr_model import (
    PBRParameters,
    PBRMaterial,
    PBRTextureSet,
    PBRWorkflow,
    TextureChannel,
    validate_pbr_parameter,
    clamp_pbr_parameter,
)

# Shader Compilation
from engine.rendering.materials.shader_compiler import (
    ShaderStage,
    ShaderLanguage,
    ShaderSource,
    ShaderDefine,
    PermutationKey,
    ShaderPermutation,
    CompiledShader,
    PSODescriptor,
    PSOCache,
    ShaderCompiler,
    CompilationError,
    HotReloadWatcher,
)

# Material Graph
from engine.rendering.materials.material_graph import (
    DataType,
    NodePort,
    NodeConnection,
    MaterialNode,
    ConstantNode,
    ParameterNode,
    TextureSampleNode,
    UVNode,
    AddNode,
    SubtractNode,
    MultiplyNode,
    DivideNode,
    LerpNode,
    ClampNode,
    DotNode,
    NormalizeNode,
    PowerNode,
    SqrtNode,
    AbsNode,
    FracNode,
    FloorNode,
    CeilNode,
    MinNode,
    MaxNode,
    SinNode,
    CosNode,
    OneMinus,
    ComponentMask,
    AppendNode,
    OutputNode,
    MaterialGraph,
    GraphCompiler,
    GraphValidationError,
)

# Material Functions Library
from engine.rendering.materials.material_functions import (
    MaterialFunctionLibrary,
    create_fresnel_function,
    create_normal_blend_function,
    create_parallax_function,
    create_triplanar_function,
    create_detail_normal_function,
    create_height_blend_function,
    create_srgb_to_linear_function,
    create_linear_to_srgb_function,
    create_luminance_function,
    create_saturation_function,
    create_contrast_function,
    create_noise_function,
    create_voronoi_function,
    create_gradient_noise_function,
)

# Advanced Shading Models
from engine.rendering.materials.advanced_models import (
    SubsurfaceProfile,
    SubsurfaceScattering,
    ClearCoat,
    Anisotropy,
    Sheen,
    Iridescence,
    Transmission,
    AdvancedShadingModel,
    ShadingModelType,
)

__all__ = [
    # Constants
    "PBRParameterRange",
    "PBR_METALLIC_RANGE",
    "PBR_ROUGHNESS_RANGE",
    "PBR_NORMAL_SCALE_RANGE",
    "PBR_AO_RANGE",
    "PBR_BASE_COLOR_RANGE",
    "PBR_EMISSIVE_MIN",
    "PSO_CACHE_DEFAULT_MAX_SIZE",
    "HOT_RELOAD_POLL_INTERVAL_SECONDS",
    "SHADER_HASH_LENGTH",
    "SAFE_DIVISION_EPSILON",
    # Material System
    "MaterialDomain",
    "BlendMode",
    "ShadingModel",
    "ParameterType",
    "MaterialParameter",
    "MaterialTemplate",
    "MaterialInstance",
    "MaterialFunction",
    "MaterialLayer",
    "MaterialSystem",
    "DirtyFlags",
    # PBR Model
    "PBRParameters",
    "PBRMaterial",
    "PBRTextureSet",
    "PBRWorkflow",
    "TextureChannel",
    "validate_pbr_parameter",
    "clamp_pbr_parameter",
    # Shader Compilation
    "ShaderStage",
    "ShaderLanguage",
    "ShaderSource",
    "ShaderDefine",
    "PermutationKey",
    "ShaderPermutation",
    "CompiledShader",
    "PSODescriptor",
    "PSOCache",
    "ShaderCompiler",
    "CompilationError",
    "HotReloadWatcher",
    # Material Graph
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
    # Material Functions
    "MaterialFunctionLibrary",
    "create_fresnel_function",
    "create_normal_blend_function",
    "create_parallax_function",
    "create_triplanar_function",
    "create_detail_normal_function",
    "create_height_blend_function",
    "create_srgb_to_linear_function",
    "create_linear_to_srgb_function",
    "create_luminance_function",
    "create_saturation_function",
    "create_contrast_function",
    "create_noise_function",
    "create_voronoi_function",
    "create_gradient_noise_function",
    # Advanced Models
    "SubsurfaceProfile",
    "SubsurfaceScattering",
    "ClearCoat",
    "Anisotropy",
    "Sheen",
    "Iridescence",
    "Transmission",
    "AdvancedShadingModel",
    "ShadingModelType",
]
