"""Material compiler - Compile graph to shader code (HLSL, GLSL, Metal)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
import re

from .material_graph import MaterialGraph, GraphError
from .material_nodes import (
    MaterialNode, NodeCategory, DataType,
    PBROutputNode, UnlitOutputNode, TextureSampleNode, ParameterNode
)
from .connection_validator import Connection, TypeConversion


class ShaderLanguage(Enum):
    """Target shader language."""
    HLSL = auto()
    GLSL = auto()
    METAL = auto()
    SPIRV = auto()


class ShaderStage(Enum):
    """Shader pipeline stage."""
    VERTEX = auto()
    FRAGMENT = auto()
    COMPUTE = auto()


@dataclass
class ShaderParameter:
    """Exposed shader parameter."""
    name: str
    data_type: str
    default_value: Any
    binding: int
    semantic: str = ""


@dataclass
class TextureBinding:
    """Texture binding information."""
    name: str
    slot: int
    sampler_slot: int
    type: str = "Texture2D"


@dataclass
class CompilationResult:
    """Result of material compilation."""
    success: bool
    vertex_shader: str = ""
    fragment_shader: str = ""
    parameters: List[ShaderParameter] = field(default_factory=list)
    textures: List[TextureBinding] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ShaderGenerator(ABC):
    """Abstract base class for shader code generation."""

    def __init__(self):
        self._indent_level = 0
        self._indent_char = "    "

    @property
    @abstractmethod
    def language(self) -> ShaderLanguage:
        """Get target shader language."""
        pass

    @abstractmethod
    def generate_header(self) -> str:
        """Generate shader header."""
        pass

    @abstractmethod
    def generate_structs(self) -> str:
        """Generate input/output structs."""
        pass

    @abstractmethod
    def generate_parameters(self, params: List[ShaderParameter]) -> str:
        """Generate parameter declarations."""
        pass

    @abstractmethod
    def generate_textures(self, textures: List[TextureBinding]) -> str:
        """Generate texture declarations."""
        pass

    @abstractmethod
    def generate_function_start(self, stage: ShaderStage) -> str:
        """Generate main function start."""
        pass

    @abstractmethod
    def generate_function_end(self) -> str:
        """Generate main function end."""
        pass

    @abstractmethod
    def generate_type(self, data_type: DataType) -> str:
        """Generate type name for data type."""
        pass

    def indent(self) -> str:
        """Get current indentation."""
        return self._indent_char * self._indent_level

    def push_indent(self) -> None:
        """Increase indentation."""
        self._indent_level += 1

    def pop_indent(self) -> None:
        """Decrease indentation."""
        self._indent_level = max(0, self._indent_level - 1)


class HLSLGenerator(ShaderGenerator):
    """HLSL shader code generator."""

    @property
    def language(self) -> ShaderLanguage:
        return ShaderLanguage.HLSL

    def generate_header(self) -> str:
        return """// Generated HLSL shader
#pragma pack_matrix(row_major)

"""

    def generate_structs(self) -> str:
        return """struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
    float4 tangent : TANGENT;
};

struct PSInput {
    float4 position : SV_POSITION;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 color : COLOR0;
    float3 viewDir : TEXCOORD3;
    float4 screenPos : TEXCOORD4;
    float3 tangent : TEXCOORD5;
    float3 bitangent : TEXCOORD6;
};

struct PSOutput {
    float4 Albedo : SV_Target0;
    float4 Normal : SV_Target1;
    float4 MetallicRoughness : SV_Target2;
    float4 Emissive : SV_Target3;
};

"""

    def generate_parameters(self, params: List[ShaderParameter]) -> str:
        if not params:
            return ""

        lines = ["cbuffer MaterialParameters : register(b0) {"]
        for param in params:
            lines.append(f"    {param.data_type} {param.name};")
        lines.append("};\n")
        return "\n".join(lines)

    def generate_textures(self, textures: List[TextureBinding]) -> str:
        if not textures:
            return ""

        lines = []
        for tex in textures:
            lines.append(f"{tex.type} {tex.name} : register(t{tex.slot});")
            lines.append(f"SamplerState sampler_{tex.name} : register(s{tex.sampler_slot});")
        lines.append("")
        return "\n".join(lines)

    def generate_function_start(self, stage: ShaderStage) -> str:
        if stage == ShaderStage.VERTEX:
            return """PSInput VSMain(VSInput input) {
    PSInput output;
"""
        else:
            return """PSOutput PSMain(PSInput input) {
    PSOutput output;
"""

    def generate_function_end(self) -> str:
        return """    return output;
}
"""

    def generate_type(self, data_type: DataType) -> str:
        type_map = {
            DataType.FLOAT: "float",
            DataType.FLOAT2: "float2",
            DataType.FLOAT3: "float3",
            DataType.FLOAT4: "float4",
            DataType.INT: "int",
            DataType.BOOL: "bool",
            DataType.MATRIX: "float4x4",
            DataType.TEXTURE2D: "Texture2D",
            DataType.TEXTURE_CUBE: "TextureCube",
        }
        return type_map.get(data_type, "float")


class GLSLGenerator(ShaderGenerator):
    """GLSL shader code generator."""

    @property
    def language(self) -> ShaderLanguage:
        return ShaderLanguage.GLSL

    def generate_header(self) -> str:
        return """// Generated GLSL shader
#version 450 core

"""

    def generate_structs(self) -> str:
        return """// Vertex shader inputs
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_color;
layout(location = 4) in vec4 a_tangent;

// Fragment shader inputs (from vertex shader)
layout(location = 0) out vec3 v_worldPos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out vec4 v_color;
layout(location = 4) out vec3 v_viewDir;
layout(location = 5) out vec4 v_screenPos;

// Fragment outputs
layout(location = 0) out vec4 o_albedo;
layout(location = 1) out vec4 o_normal;
layout(location = 2) out vec4 o_metallicRoughness;
layout(location = 3) out vec4 o_emissive;

"""

    def generate_parameters(self, params: List[ShaderParameter]) -> str:
        if not params:
            return ""

        lines = ["layout(std140, binding = 0) uniform MaterialParameters {"]
        for param in params:
            lines.append(f"    {param.data_type} {param.name};")
        lines.append("};\n")
        return "\n".join(lines)

    def generate_textures(self, textures: List[TextureBinding]) -> str:
        if not textures:
            return ""

        lines = []
        for tex in textures:
            glsl_type = "sampler2D" if tex.type == "Texture2D" else "samplerCube"
            lines.append(f"layout(binding = {tex.slot}) uniform {glsl_type} {tex.name};")
        lines.append("")
        return "\n".join(lines)

    def generate_function_start(self, stage: ShaderStage) -> str:
        if stage == ShaderStage.VERTEX:
            return "void main() {\n"
        else:
            return "void main() {\n"

    def generate_function_end(self) -> str:
        return "}\n"

    def generate_type(self, data_type: DataType) -> str:
        type_map = {
            DataType.FLOAT: "float",
            DataType.FLOAT2: "vec2",
            DataType.FLOAT3: "vec3",
            DataType.FLOAT4: "vec4",
            DataType.INT: "int",
            DataType.BOOL: "bool",
            DataType.MATRIX: "mat4",
            DataType.TEXTURE2D: "sampler2D",
            DataType.TEXTURE_CUBE: "samplerCube",
        }
        return type_map.get(data_type, "float")


class MetalGenerator(ShaderGenerator):
    """Metal shader code generator."""

    @property
    def language(self) -> ShaderLanguage:
        return ShaderLanguage.METAL

    def generate_header(self) -> str:
        return """// Generated Metal shader
#include <metal_stdlib>
using namespace metal;

"""

    def generate_structs(self) -> str:
        return """struct VertexInput {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 uv [[attribute(2)]];
    float4 color [[attribute(3)]];
    float4 tangent [[attribute(4)]];
};

struct FragmentInput {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float2 uv;
    float4 color;
    float3 viewDir;
    float4 screenPos;
};

struct FragmentOutput {
    float4 albedo [[color(0)]];
    float4 normal [[color(1)]];
    float4 metallicRoughness [[color(2)]];
    float4 emissive [[color(3)]];
};

"""

    def generate_parameters(self, params: List[ShaderParameter]) -> str:
        if not params:
            return ""

        lines = ["struct MaterialParameters {"]
        for param in params:
            lines.append(f"    {param.data_type} {param.name};")
        lines.append("};\n")
        return "\n".join(lines)

    def generate_textures(self, textures: List[TextureBinding]) -> str:
        # Metal handles textures in function arguments
        return ""

    def generate_function_start(self, stage: ShaderStage) -> str:
        if stage == ShaderStage.VERTEX:
            return """vertex FragmentInput vertexMain(
    VertexInput input [[stage_in]],
    constant MaterialParameters& params [[buffer(0)]]
) {
    FragmentInput output;
"""
        else:
            return """fragment FragmentOutput fragmentMain(
    FragmentInput input [[stage_in]],
    constant MaterialParameters& params [[buffer(0)]]
) {
    FragmentOutput output;
"""

    def generate_function_end(self) -> str:
        return """    return output;
}
"""

    def generate_type(self, data_type: DataType) -> str:
        type_map = {
            DataType.FLOAT: "float",
            DataType.FLOAT2: "float2",
            DataType.FLOAT3: "float3",
            DataType.FLOAT4: "float4",
            DataType.INT: "int",
            DataType.BOOL: "bool",
            DataType.MATRIX: "float4x4",
            DataType.TEXTURE2D: "texture2d<float>",
            DataType.TEXTURE_CUBE: "texturecube<float>",
        }
        return type_map.get(data_type, "float")


class MaterialCompiler:
    """Compiles material graphs to shader code."""

    def __init__(self):
        self._generators: Dict[ShaderLanguage, ShaderGenerator] = {
            ShaderLanguage.HLSL: HLSLGenerator(),
            ShaderLanguage.GLSL: GLSLGenerator(),
            ShaderLanguage.METAL: MetalGenerator(),
        }
        self._var_counter = 0

    def compile(
        self,
        graph: MaterialGraph,
        language: ShaderLanguage = ShaderLanguage.HLSL,
        optimize: bool = True
    ) -> CompilationResult:
        """
        Compile a material graph to shader code.

        Args:
            graph: Material graph to compile
            language: Target shader language
            optimize: Whether to optimize the output

        Returns:
            CompilationResult with generated shaders
        """
        self._var_counter = 0

        # Validate graph first
        errors = graph.validate()
        if any(e.severity == "error" for e in errors):
            return CompilationResult(
                success=False,
                errors=[e.message for e in errors if e.severity == "error"],
                warnings=[e.message for e in errors if e.severity == "warning"]
            )

        generator = self._generators.get(language)
        if not generator:
            return CompilationResult(
                success=False,
                errors=[f"Unsupported shader language: {language.name}"]
            )

        try:
            # Collect parameters and textures
            params = self._collect_parameters(graph, generator)
            textures = self._collect_textures(graph)

            # Generate fragment shader body
            fragment_body = self._generate_fragment_code(graph, generator)

            # Build complete fragment shader
            fragment_shader = (
                generator.generate_header() +
                generator.generate_structs() +
                generator.generate_parameters(params) +
                generator.generate_textures(textures) +
                self._generate_helper_functions(language) +
                generator.generate_function_start(ShaderStage.FRAGMENT) +
                fragment_body +
                generator.generate_function_end()
            )

            # Generate vertex shader (basic transform)
            vertex_shader = self._generate_vertex_shader(generator, params)

            if optimize:
                fragment_shader = self._optimize_shader(fragment_shader)
                vertex_shader = self._optimize_shader(vertex_shader)

            return CompilationResult(
                success=True,
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader,
                parameters=params,
                textures=textures,
                warnings=[e.message for e in errors if e.severity == "warning"]
            )

        except Exception as e:
            return CompilationResult(
                success=False,
                errors=[f"Compilation error: {str(e)}"]
            )

    def _generate_unique_var(self, prefix: str = "var") -> str:
        """Generate a unique variable name."""
        self._var_counter += 1
        return f"{prefix}_{self._var_counter}"

    def _collect_parameters(
        self,
        graph: MaterialGraph,
        generator: ShaderGenerator
    ) -> List[ShaderParameter]:
        """Collect all exposed parameters from the graph."""
        params = []
        binding = 0

        for node in graph.nodes.values():
            if isinstance(node, ParameterNode):
                params.append(ShaderParameter(
                    name=node.param_name,
                    data_type=generator.generate_type(node.param_type),
                    default_value=node.default_value,
                    binding=binding
                ))
                binding += 1

        return params

    def _collect_textures(self, graph: MaterialGraph) -> List[TextureBinding]:
        """Collect all texture bindings from the graph."""
        textures = []
        slot = 0

        for node in graph.nodes.values():
            if isinstance(node, TextureSampleNode):
                tex_name = node.texture_path.replace("/", "_").replace(".", "_") + "_tex"
                if not tex_name or tex_name == "_tex":
                    tex_name = f"texture_{slot}"
                textures.append(TextureBinding(
                    name=tex_name,
                    slot=slot,
                    sampler_slot=slot
                ))
                slot += 1

        return textures

    def _generate_fragment_code(
        self,
        graph: MaterialGraph,
        generator: ShaderGenerator
    ) -> str:
        """Generate the fragment shader code body."""
        lines = []

        # Get nodes in topological order
        try:
            node_order = graph.get_evaluation_order()
        except ValueError as e:
            raise ValueError(f"Invalid graph structure: {e}")

        # Track output variables for each node
        node_outputs: Dict[str, Dict[str, str]] = {}

        for node in node_order:
            # Collect input variable names
            input_vars: Dict[str, str] = {}
            for pin_name, pin in node.inputs.items():
                conn = graph.get_connection_to_pin(node.id, pin_name)
                if conn:
                    source_outputs = node_outputs.get(conn.source_node_id, {})
                    if conn.source_pin in source_outputs:
                        input_vars[pin_name] = source_outputs[conn.source_pin]
                    else:
                        input_vars[pin_name] = self._get_default_value(pin, generator)
                else:
                    input_vars[pin_name] = self._get_default_value(pin, generator)

            # Generate output variable names
            output_vars: Dict[str, str] = {}
            for pin_name in node.outputs.keys():
                output_vars[pin_name] = self._generate_unique_var(f"{node.name}_{pin_name}".replace(" ", "_"))

            # Generate node code
            code = node.generate_code(input_vars, output_vars)
            if code:
                # Add indentation
                for line in code.split("\n"):
                    if line.strip():
                        lines.append(f"    {line}")

            # Store output variables
            node_outputs[node.id] = output_vars

        return "\n".join(lines) + "\n"

    def _get_default_value(self, pin, generator: ShaderGenerator) -> str:
        """Get shader code for a pin's default value."""
        if pin.default_value is None:
            data_type = generator.generate_type(pin.data_type)
            if "float4" in data_type or "vec4" in data_type:
                return f"{data_type}(0, 0, 0, 1)"
            elif "float3" in data_type or "vec3" in data_type:
                return f"{data_type}(0, 0, 0)"
            elif "float2" in data_type or "vec2" in data_type:
                return f"{data_type}(0, 0)"
            return "0.0"

        if isinstance(pin.default_value, (tuple, list)):
            values = ", ".join(str(v) for v in pin.default_value)
            data_type = generator.generate_type(pin.data_type)
            return f"{data_type}({values})"
        return str(pin.default_value)

    def _generate_vertex_shader(
        self,
        generator: ShaderGenerator,
        params: List[ShaderParameter]
    ) -> str:
        """Generate a basic vertex shader."""
        if generator.language == ShaderLanguage.HLSL:
            return """// Generated HLSL vertex shader
cbuffer TransformBuffer : register(b1) {
    float4x4 worldMatrix;
    float4x4 viewProjectionMatrix;
    float3 cameraPosition;
};

""" + generator.generate_structs() + """
PSInput VSMain(VSInput input) {
    PSInput output;
    float4 worldPos = mul(float4(input.position, 1.0), worldMatrix);
    output.position = mul(worldPos, viewProjectionMatrix);
    output.worldPos = worldPos.xyz;
    output.normal = mul(input.normal, (float3x3)worldMatrix);
    output.uv = input.uv;
    output.color = input.color;
    output.viewDir = normalize(cameraPosition - worldPos.xyz);
    output.screenPos = output.position;
    output.tangent = mul(input.tangent.xyz, (float3x3)worldMatrix);
    output.bitangent = cross(output.normal, output.tangent) * input.tangent.w;
    return output;
}
"""
        elif generator.language == ShaderLanguage.GLSL:
            return """// Generated GLSL vertex shader
#version 450 core

layout(std140, binding = 1) uniform TransformBuffer {
    mat4 worldMatrix;
    mat4 viewProjectionMatrix;
    vec3 cameraPosition;
};

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_color;
layout(location = 4) in vec4 a_tangent;

layout(location = 0) out vec3 v_worldPos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out vec4 v_color;
layout(location = 4) out vec3 v_viewDir;

void main() {
    vec4 worldPos = worldMatrix * vec4(a_position, 1.0);
    gl_Position = viewProjectionMatrix * worldPos;
    v_worldPos = worldPos.xyz;
    v_normal = mat3(worldMatrix) * a_normal;
    v_uv = a_uv;
    v_color = a_color;
    v_viewDir = normalize(cameraPosition - worldPos.xyz);
}
"""
        else:  # Metal
            return """// Generated Metal vertex shader
#include <metal_stdlib>
using namespace metal;

struct TransformBuffer {
    float4x4 worldMatrix;
    float4x4 viewProjectionMatrix;
    float3 cameraPosition;
};

""" + generator.generate_structs() + """
vertex FragmentInput vertexMain(
    VertexInput input [[stage_in]],
    constant TransformBuffer& transforms [[buffer(1)]]
) {
    FragmentInput output;
    float4 worldPos = transforms.worldMatrix * float4(input.position, 1.0);
    output.position = transforms.viewProjectionMatrix * worldPos;
    output.worldPos = worldPos.xyz;
    output.normal = (transforms.worldMatrix * float4(input.normal, 0.0)).xyz;
    output.uv = input.uv;
    output.color = input.color;
    output.viewDir = normalize(transforms.cameraPosition - worldPos.xyz);
    output.screenPos = output.position;
    return output;
}
"""

    def _generate_helper_functions(self, language: ShaderLanguage) -> str:
        """Generate common helper functions."""
        if language == ShaderLanguage.HLSL:
            return """
#define PI 3.14159265359

float3 ComputePBR(float3 albedo, float3 normal, float metallic, float roughness, float3 F0) {
    // Simplified PBR - full implementation would include lighting
    return albedo;
}

"""
        elif language == ShaderLanguage.GLSL:
            return """
#define PI 3.14159265359

vec3 ComputePBR(vec3 albedo, vec3 normal, float metallic, float roughness, vec3 F0) {
    return albedo;
}

"""
        else:
            return """
constant float PI = 3.14159265359;

float3 ComputePBR(float3 albedo, float3 normal, float metallic, float roughness, float3 F0) {
    return albedo;
}

"""

    def _optimize_shader(self, shader: str) -> str:
        """Apply basic optimizations to shader code."""
        # Remove empty lines
        lines = [line for line in shader.split("\n") if line.strip() or line == ""]

        # Remove duplicate empty lines
        result = []
        prev_empty = False
        for line in lines:
            is_empty = not line.strip()
            if is_empty and prev_empty:
                continue
            result.append(line)
            prev_empty = is_empty

        return "\n".join(result)

    def get_supported_languages(self) -> List[ShaderLanguage]:
        """Get list of supported shader languages."""
        return list(self._generators.keys())
