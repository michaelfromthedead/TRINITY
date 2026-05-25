"""Reusable shader snippets and material function library.

This module provides common shader functions that can be reused
across multiple materials:
- Fresnel calculations
- Normal blending
- Parallax mapping
- Color space conversions
- Utility functions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

from engine.rendering.materials.material_system import (
    MaterialFunction,
    MaterialParameter,
    ParameterType,
)

__all__ = [
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
]


class MaterialFunctionLibrary:
    """Singleton library of reusable material functions.

    Provides access to common shader functions like Fresnel,
    normal blending, parallax mapping, etc.
    """

    _instance: Optional[MaterialFunctionLibrary] = None

    def __new__(cls) -> MaterialFunctionLibrary:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._functions: Dict[str, MaterialFunction] = {}
            cls._instance._initialize_builtin_functions()
        return cls._instance

    @classmethod
    def get_instance(cls) -> MaterialFunctionLibrary:
        """Get the singleton instance."""
        return cls()

    def register(self, func: MaterialFunction) -> None:
        """Register a material function."""
        self._functions[func.name] = func

    def get(self, name: str) -> Optional[MaterialFunction]:
        """Get a function by name."""
        return self._functions.get(name)

    def get_all(self) -> List[MaterialFunction]:
        """Get all registered functions."""
        return list(self._functions.values())

    def get_by_category(self, category: str) -> List[MaterialFunction]:
        """Get functions by category tag."""
        return [
            f for f in self._functions.values()
            if category in f.description.lower()
        ]

    def _initialize_builtin_functions(self) -> None:
        """Initialize built-in material functions."""
        # Register all built-in functions
        self.register(create_fresnel_function())
        self.register(create_fresnel_schlick_function())
        self.register(create_normal_blend_function())
        self.register(create_normal_blend_rnm_function())
        self.register(create_parallax_function())
        self.register(create_parallax_occlusion_function())
        self.register(create_triplanar_function())
        self.register(create_detail_normal_function())
        self.register(create_height_blend_function())
        self.register(create_srgb_to_linear_function())
        self.register(create_linear_to_srgb_function())
        self.register(create_luminance_function())
        self.register(create_saturation_function())
        self.register(create_contrast_function())
        self.register(create_noise_function())
        self.register(create_voronoi_function())
        self.register(create_gradient_noise_function())
        self.register(create_checkerboard_function())
        self.register(create_radial_gradient_function())
        self.register(create_box_mask_function())
        self.register(create_sphere_mask_function())
        self.register(create_blend_overlay_function())
        self.register(create_blend_soft_light_function())


def create_fresnel_function() -> MaterialFunction:
    """Create basic Fresnel effect function."""
    code = """
// Fresnel effect using Schlick approximation
float Fresnel(vec3 viewDir, vec3 normal, float power) {
    float NdotV = max(dot(normal, viewDir), 0.0);
    return pow(1.0 - NdotV, power);
}
"""
    func = MaterialFunction(
        name="Fresnel",
        code=code.strip(),
        description="Fresnel effect using Schlick approximation. Category: Lighting",
    )
    func.add_input(MaterialParameter(
        name="viewDir",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="View direction vector",
    ))
    func.add_input(MaterialParameter(
        name="normal",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Surface normal vector",
    ))
    func.add_input(MaterialParameter(
        name="power",
        param_type=ParameterType.FLOAT,
        default_value=5.0,
        min_value=0.0,
        max_value=10.0,
        description="Fresnel power/exponent",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.FLOAT,
        default_value=None,
        description="Fresnel term",
    ))
    return func


def create_fresnel_schlick_function() -> MaterialFunction:
    """Create Fresnel-Schlick with F0 function."""
    code = """
// Fresnel-Schlick approximation with F0 reflectance
vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (vec3(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Fresnel-Schlick with roughness for IBL
vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
"""
    func = MaterialFunction(
        name="FresnelSchlick",
        code=code.strip(),
        description="Fresnel-Schlick approximation with F0 reflectance. Category: Lighting",
    )
    func.add_input(MaterialParameter(
        name="cosTheta",
        param_type=ParameterType.FLOAT,
        default_value=None,
        description="Dot product of view and half vector",
    ))
    func.add_input(MaterialParameter(
        name="F0",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Base reflectance at normal incidence",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Fresnel reflectance",
    ))
    return func


def create_normal_blend_function() -> MaterialFunction:
    """Create normal blending function (whiteout blend)."""
    code = """
// Blend two normal maps using whiteout blending
vec3 NormalBlend(vec3 n1, vec3 n2) {
    vec3 t = n1 * vec3(2.0, 2.0, 2.0) + vec3(-1.0, -1.0, 0.0);
    vec3 u = n2 * vec3(-2.0, -2.0, 2.0) + vec3(1.0, 1.0, -1.0);
    return normalize(t * dot(t, u) - u * t.z);
}
"""
    func = MaterialFunction(
        name="NormalBlend",
        code=code.strip(),
        description="Blend two normal maps using whiteout blending. Category: Normals",
    )
    func.add_input(MaterialParameter(
        name="n1",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="First normal map",
    ))
    func.add_input(MaterialParameter(
        name="n2",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Second normal map",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Blended normal",
    ))
    return func


def create_normal_blend_rnm_function() -> MaterialFunction:
    """Create Reoriented Normal Mapping blend function."""
    code = """
// Blend normals using Reoriented Normal Mapping (RNM)
vec3 NormalBlendRNM(vec3 n1, vec3 n2) {
    n1 = n1 * 2.0 - 1.0;
    n2 = n2 * 2.0 - 1.0;
    n1.z += 1.0;
    n2.xy = -n2.xy;
    return normalize(n1 * dot(n1, n2) / n1.z - n2) * 0.5 + 0.5;
}
"""
    func = MaterialFunction(
        name="NormalBlendRNM",
        code=code.strip(),
        description="Blend normals using Reoriented Normal Mapping. Category: Normals",
    )
    func.add_input(MaterialParameter(
        name="n1",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Base normal map",
    ))
    func.add_input(MaterialParameter(
        name="n2",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Detail normal map",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Blended normal",
    ))
    return func


def create_parallax_function() -> MaterialFunction:
    """Create basic parallax offset function."""
    code = """
// Simple parallax offset mapping
vec2 ParallaxOffset(vec2 uv, vec3 viewDir, float height, float scale) {
    float h = height * scale - scale * 0.5;
    vec3 v = normalize(viewDir);
    return uv + v.xy * h / v.z;
}
"""
    func = MaterialFunction(
        name="ParallaxOffset",
        code=code.strip(),
        description="Simple parallax offset mapping. Category: UV",
    )
    func.add_input(MaterialParameter(
        name="uv",
        param_type=ParameterType.VEC2,
        default_value=None,
        description="Input UV coordinates",
    ))
    func.add_input(MaterialParameter(
        name="viewDir",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="View direction in tangent space",
    ))
    func.add_input(MaterialParameter(
        name="height",
        param_type=ParameterType.FLOAT,
        default_value=0.5,
        description="Height map sample",
    ))
    func.add_input(MaterialParameter(
        name="scale",
        param_type=ParameterType.FLOAT,
        default_value=0.05,
        min_value=0.0,
        max_value=0.5,
        description="Parallax scale",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.VEC2,
        default_value=None,
        description="Offset UV coordinates",
    ))
    return func


def create_parallax_occlusion_function() -> MaterialFunction:
    """Create parallax occlusion mapping function."""
    code = """
// Parallax Occlusion Mapping (POM)
vec2 ParallaxOcclusionMapping(
    sampler2D heightMap,
    vec2 uv,
    vec3 viewDir,
    float heightScale,
    float minLayers,
    float maxLayers
) {
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), viewDir)));
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;
    vec2 P = viewDir.xy / viewDir.z * heightScale;
    vec2 deltaUV = P / numLayers;

    vec2 currentUV = uv;
    float currentHeight = texture(heightMap, currentUV).r;

    while (currentLayerDepth < currentHeight) {
        currentUV -= deltaUV;
        currentHeight = texture(heightMap, currentUV).r;
        currentLayerDepth += layerDepth;
    }

    vec2 prevUV = currentUV + deltaUV;
    float afterDepth = currentHeight - currentLayerDepth;
    float beforeDepth = texture(heightMap, prevUV).r - currentLayerDepth + layerDepth;
    float weight = afterDepth / (afterDepth - beforeDepth);

    return mix(currentUV, prevUV, weight);
}
"""
    func = MaterialFunction(
        name="ParallaxOcclusionMapping",
        code=code.strip(),
        description="Parallax Occlusion Mapping (POM) with self-shadowing. Category: UV",
    )
    func.add_input(MaterialParameter(
        name="uv",
        param_type=ParameterType.VEC2,
        default_value=None,
        description="Input UV coordinates",
    ))
    func.add_input(MaterialParameter(
        name="viewDir",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="View direction in tangent space",
    ))
    func.add_input(MaterialParameter(
        name="heightScale",
        param_type=ParameterType.FLOAT,
        default_value=0.1,
        description="Height scale",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.VEC2,
        default_value=None,
        description="Offset UV coordinates",
    ))
    return func


def create_triplanar_function() -> MaterialFunction:
    """Create triplanar projection function."""
    code = """
// Triplanar texture projection
vec4 TriplanarSample(
    sampler2D tex,
    vec3 worldPos,
    vec3 worldNormal,
    float tiling,
    float blendSharpness
) {
    vec3 blending = pow(abs(worldNormal), vec3(blendSharpness));
    blending = blending / (blending.x + blending.y + blending.z);

    vec4 xaxis = texture(tex, worldPos.yz * tiling);
    vec4 yaxis = texture(tex, worldPos.xz * tiling);
    vec4 zaxis = texture(tex, worldPos.xy * tiling);

    return xaxis * blending.x + yaxis * blending.y + zaxis * blending.z;
}
"""
    func = MaterialFunction(
        name="TriplanarSample",
        code=code.strip(),
        description="Triplanar texture projection for seamless texturing. Category: UV",
    )
    func.add_input(MaterialParameter(
        name="worldPos",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="World space position",
    ))
    func.add_input(MaterialParameter(
        name="worldNormal",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="World space normal",
    ))
    func.add_input(MaterialParameter(
        name="tiling",
        param_type=ParameterType.FLOAT,
        default_value=1.0,
        description="Texture tiling scale",
    ))
    func.add_input(MaterialParameter(
        name="blendSharpness",
        param_type=ParameterType.FLOAT,
        default_value=4.0,
        description="Blend sharpness",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.VEC4,
        default_value=None,
        description="Sampled color",
    ))
    return func


def create_detail_normal_function() -> MaterialFunction:
    """Create detail normal blending function."""
    code = """
// Blend detail normal with base normal
vec3 DetailNormal(vec3 baseNormal, vec3 detailNormal, float strength) {
    vec3 t = baseNormal * vec3(2.0, 2.0, 2.0) + vec3(-1.0, -1.0, 0.0);
    vec3 u = detailNormal * vec3(-2.0, -2.0, 2.0) + vec3(1.0, 1.0, -1.0);
    vec3 blended = normalize(t * dot(t, u) - u * t.z);
    return mix(baseNormal * 2.0 - 1.0, blended, strength) * 0.5 + 0.5;
}
"""
    func = MaterialFunction(
        name="DetailNormal",
        code=code.strip(),
        description="Blend detail normal with base normal. Category: Normals",
    )
    func.add_input(MaterialParameter(
        name="baseNormal",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Base normal map",
    ))
    func.add_input(MaterialParameter(
        name="detailNormal",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Detail normal map",
    ))
    func.add_input(MaterialParameter(
        name="strength",
        param_type=ParameterType.FLOAT,
        default_value=1.0,
        min_value=0.0,
        max_value=2.0,
        description="Blend strength",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Blended normal",
    ))
    return func


def create_height_blend_function() -> MaterialFunction:
    """Create height-based texture blending function."""
    code = """
// Height-based texture blending for terrain
float HeightBlend(float height1, float height2, float factor, float contrast) {
    float h1 = height1 + (1.0 - factor);
    float h2 = height2 + factor;
    float depth = contrast;
    float max_h = max(h1, h2) - depth;
    float b1 = max(h1 - max_h, 0.0);
    float b2 = max(h2 - max_h, 0.0);
    return b2 / (b1 + b2);
}
"""
    func = MaterialFunction(
        name="HeightBlend",
        code=code.strip(),
        description="Height-based texture blending for terrain. Category: Blending",
    )
    func.add_input(MaterialParameter(
        name="height1",
        param_type=ParameterType.FLOAT,
        default_value=None,
        description="Height map value for texture 1",
    ))
    func.add_input(MaterialParameter(
        name="height2",
        param_type=ParameterType.FLOAT,
        default_value=None,
        description="Height map value for texture 2",
    ))
    func.add_input(MaterialParameter(
        name="factor",
        param_type=ParameterType.FLOAT,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        description="Blend factor",
    ))
    func.add_input(MaterialParameter(
        name="contrast",
        param_type=ParameterType.FLOAT,
        default_value=0.2,
        min_value=0.001,
        max_value=1.0,
        description="Blend contrast",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.FLOAT,
        default_value=None,
        description="Blend weight for texture 2",
    ))
    return func


def create_srgb_to_linear_function() -> MaterialFunction:
    """Create sRGB to linear color space conversion."""
    code = """
// Convert sRGB to linear color space
vec3 SRGBToLinear(vec3 srgb) {
    return pow(srgb, vec3(2.2));
}

vec4 SRGBToLinear4(vec4 srgba) {
    return vec4(pow(srgba.rgb, vec3(2.2)), srgba.a);
}
"""
    func = MaterialFunction(
        name="SRGBToLinear",
        code=code.strip(),
        description="Convert sRGB to linear color space. Category: Color",
    )
    func.add_input(MaterialParameter(
        name="srgb",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="sRGB color",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Linear color",
    ))
    return func


def create_linear_to_srgb_function() -> MaterialFunction:
    """Create linear to sRGB color space conversion."""
    code = """
// Convert linear to sRGB color space
vec3 LinearToSRGB(vec3 linear) {
    return pow(linear, vec3(1.0 / 2.2));
}

vec4 LinearToSRGB4(vec4 linear) {
    return vec4(pow(linear.rgb, vec3(1.0 / 2.2)), linear.a);
}
"""
    func = MaterialFunction(
        name="LinearToSRGB",
        code=code.strip(),
        description="Convert linear to sRGB color space. Category: Color",
    )
    func.add_input(MaterialParameter(
        name="linear",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Linear color",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="sRGB color",
    ))
    return func


def create_luminance_function() -> MaterialFunction:
    """Create luminance calculation function."""
    code = """
// Calculate luminance of a color
float Luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}
"""
    func = MaterialFunction(
        name="Luminance",
        code=code.strip(),
        description="Calculate luminance of a color (Rec. 709). Category: Color",
    )
    func.add_input(MaterialParameter(
        name="color",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Input color",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.FLOAT,
        default_value=None,
        description="Luminance value",
    ))
    return func


def create_saturation_function() -> MaterialFunction:
    """Create saturation adjustment function."""
    code = """
// Adjust color saturation
vec3 AdjustSaturation(vec3 color, float saturation) {
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(luma), color, saturation);
}
"""
    func = MaterialFunction(
        name="AdjustSaturation",
        code=code.strip(),
        description="Adjust color saturation. Category: Color",
    )
    func.add_input(MaterialParameter(
        name="color",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Input color",
    ))
    func.add_input(MaterialParameter(
        name="saturation",
        param_type=ParameterType.FLOAT,
        default_value=1.0,
        min_value=0.0,
        max_value=3.0,
        description="Saturation multiplier",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Adjusted color",
    ))
    return func


def create_contrast_function() -> MaterialFunction:
    """Create contrast adjustment function."""
    code = """
// Adjust color contrast
vec3 AdjustContrast(vec3 color, float contrast) {
    return (color - 0.5) * contrast + 0.5;
}
"""
    func = MaterialFunction(
        name="AdjustContrast",
        code=code.strip(),
        description="Adjust color contrast. Category: Color",
    )
    func.add_input(MaterialParameter(
        name="color",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Input color",
    ))
    func.add_input(MaterialParameter(
        name="contrast",
        param_type=ParameterType.FLOAT,
        default_value=1.0,
        min_value=0.0,
        max_value=3.0,
        description="Contrast multiplier",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Adjusted color",
    ))
    return func


def create_noise_function() -> MaterialFunction:
    """Create simple noise function."""
    code = """
// Simple hash-based pseudo-random noise
float Hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Value noise
float ValueNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = Hash(i);
    float b = Hash(i + vec2(1.0, 0.0));
    float c = Hash(i + vec2(0.0, 1.0));
    float d = Hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
"""
    func = MaterialFunction(
        name="ValueNoise",
        code=code.strip(),
        description="Simple value noise function. Category: Procedural",
    )
    func.add_input(MaterialParameter(
        name="p",
        param_type=ParameterType.VEC2,
        default_value=None,
        description="Input coordinates",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.FLOAT,
        default_value=None,
        description="Noise value [0, 1]",
    ))
    return func


def create_voronoi_function() -> MaterialFunction:
    """Create Voronoi/Worley noise function."""
    code = """
// Voronoi/Worley noise
vec2 Voronoi(vec2 p) {
    vec2 n = floor(p);
    vec2 f = fract(p);

    float minDist = 8.0;
    vec2 minPoint;

    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = Hash2(n + g);
            vec2 r = g + o - f;
            float d = dot(r, r);
            if (d < minDist) {
                minDist = d;
                minPoint = n + g + o;
            }
        }
    }

    return vec2(sqrt(minDist), Hash(minPoint));
}

vec2 Hash2(vec2 p) {
    return fract(sin(vec2(dot(p, vec2(127.1, 311.7)),
                          dot(p, vec2(269.5, 183.3)))) * 43758.5453);
}
"""
    func = MaterialFunction(
        name="Voronoi",
        code=code.strip(),
        description="Voronoi/Worley noise function. Category: Procedural",
    )
    func.add_input(MaterialParameter(
        name="p",
        param_type=ParameterType.VEC2,
        default_value=None,
        description="Input coordinates",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.VEC2,
        default_value=None,
        description="(distance, cell ID)",
    ))
    return func


def create_gradient_noise_function() -> MaterialFunction:
    """Create gradient/Perlin noise function."""
    code = """
// Gradient/Perlin noise
float GradientNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    // Quintic interpolation
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    return mix(mix(dot(GradientHash(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0)),
                   dot(GradientHash(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0)), u.x),
               mix(dot(GradientHash(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0)),
                   dot(GradientHash(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0)), u.x), u.y);
}

vec2 GradientHash(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}
"""
    func = MaterialFunction(
        name="GradientNoise",
        code=code.strip(),
        description="Gradient/Perlin noise function. Category: Procedural",
    )
    func.add_input(MaterialParameter(
        name="p",
        param_type=ParameterType.VEC2,
        default_value=None,
        description="Input coordinates",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.FLOAT,
        default_value=None,
        description="Noise value [-1, 1]",
    ))
    return func


def create_checkerboard_function() -> MaterialFunction:
    """Create checkerboard pattern function."""
    code = """
// Checkerboard pattern
float Checkerboard(vec2 uv, float scale) {
    vec2 pos = floor(uv * scale);
    return mod(pos.x + pos.y, 2.0);
}
"""
    func = MaterialFunction(
        name="Checkerboard",
        code=code.strip(),
        description="Checkerboard pattern generator. Category: Procedural",
    )
    func.add_input(MaterialParameter(
        name="uv",
        param_type=ParameterType.VEC2,
        default_value=None,
        description="UV coordinates",
    ))
    func.add_input(MaterialParameter(
        name="scale",
        param_type=ParameterType.FLOAT,
        default_value=8.0,
        description="Pattern scale",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.FLOAT,
        default_value=None,
        description="Pattern value (0 or 1)",
    ))
    return func


def create_radial_gradient_function() -> MaterialFunction:
    """Create radial gradient function."""
    code = """
// Radial gradient
float RadialGradient(vec2 uv, vec2 center, float radius) {
    return clamp(1.0 - length(uv - center) / radius, 0.0, 1.0);
}
"""
    func = MaterialFunction(
        name="RadialGradient",
        code=code.strip(),
        description="Radial gradient pattern. Category: Procedural",
    )
    func.add_input(MaterialParameter(
        name="uv",
        param_type=ParameterType.VEC2,
        default_value=None,
        description="UV coordinates",
    ))
    func.add_input(MaterialParameter(
        name="center",
        param_type=ParameterType.VEC2,
        default_value=None,
        description="Gradient center",
    ))
    func.add_input(MaterialParameter(
        name="radius",
        param_type=ParameterType.FLOAT,
        default_value=0.5,
        description="Gradient radius",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.FLOAT,
        default_value=None,
        description="Gradient value [0, 1]",
    ))
    return func


def create_box_mask_function() -> MaterialFunction:
    """Create box mask function."""
    code = """
// Box mask with smooth edges
float BoxMask(vec3 pos, vec3 boxMin, vec3 boxMax, float falloff) {
    vec3 d = max(boxMin - pos, pos - boxMax);
    return 1.0 - clamp(max(d.x, max(d.y, d.z)) / falloff, 0.0, 1.0);
}
"""
    func = MaterialFunction(
        name="BoxMask",
        code=code.strip(),
        description="3D box mask with smooth falloff. Category: Masks",
    )
    func.add_input(MaterialParameter(
        name="pos",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="World position",
    ))
    func.add_input(MaterialParameter(
        name="boxMin",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Box minimum corner",
    ))
    func.add_input(MaterialParameter(
        name="boxMax",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Box maximum corner",
    ))
    func.add_input(MaterialParameter(
        name="falloff",
        param_type=ParameterType.FLOAT,
        default_value=1.0,
        description="Edge falloff distance",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.FLOAT,
        default_value=None,
        description="Mask value [0, 1]",
    ))
    return func


def create_sphere_mask_function() -> MaterialFunction:
    """Create sphere mask function."""
    code = """
// Sphere mask with smooth edges
float SphereMask(vec3 pos, vec3 center, float radius, float hardness) {
    float d = distance(pos, center);
    return 1.0 - clamp((d - radius * (1.0 - hardness)) / (radius * hardness), 0.0, 1.0);
}
"""
    func = MaterialFunction(
        name="SphereMask",
        code=code.strip(),
        description="3D sphere mask with smooth falloff. Category: Masks",
    )
    func.add_input(MaterialParameter(
        name="pos",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="World position",
    ))
    func.add_input(MaterialParameter(
        name="center",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Sphere center",
    ))
    func.add_input(MaterialParameter(
        name="radius",
        param_type=ParameterType.FLOAT,
        default_value=100.0,
        description="Sphere radius",
    ))
    func.add_input(MaterialParameter(
        name="hardness",
        param_type=ParameterType.FLOAT,
        default_value=0.5,
        min_value=0.001,
        max_value=1.0,
        description="Edge hardness",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.FLOAT,
        default_value=None,
        description="Mask value [0, 1]",
    ))
    return func


def create_blend_overlay_function() -> MaterialFunction:
    """Create overlay blend mode function."""
    code = """
// Overlay blend mode
vec3 BlendOverlay(vec3 base, vec3 blend) {
    return mix(
        2.0 * base * blend,
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend),
        step(0.5, base)
    );
}
"""
    func = MaterialFunction(
        name="BlendOverlay",
        code=code.strip(),
        description="Overlay blend mode. Category: Blending",
    )
    func.add_input(MaterialParameter(
        name="base",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Base color",
    ))
    func.add_input(MaterialParameter(
        name="blend",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Blend color",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Blended color",
    ))
    return func


def create_blend_soft_light_function() -> MaterialFunction:
    """Create soft light blend mode function."""
    code = """
// Soft light blend mode
vec3 BlendSoftLight(vec3 base, vec3 blend) {
    return mix(
        2.0 * base * blend + base * base * (1.0 - 2.0 * blend),
        sqrt(base) * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend),
        step(0.5, blend)
    );
}
"""
    func = MaterialFunction(
        name="BlendSoftLight",
        code=code.strip(),
        description="Soft light blend mode. Category: Blending",
    )
    func.add_input(MaterialParameter(
        name="base",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Base color",
    ))
    func.add_input(MaterialParameter(
        name="blend",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Blend color",
    ))
    func.add_output(MaterialParameter(
        name="result",
        param_type=ParameterType.VEC3,
        default_value=None,
        description="Blended color",
    ))
    return func
