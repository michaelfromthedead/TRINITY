"""AST to WGSL compiler for Material DSL.

This module provides the MaterialCompiler class that wraps the PythonToWGSL
translation and adds PBR template assembly.
"""

import textwrap
import ast
import inspect
from typing import Any, Optional, TYPE_CHECKING

from trinity.materials.dsl import (
    PythonToWGSLTranslator,
    WGSLTranslationError,
    Material,
    SurfaceContext,
    SurfaceOutput,
)

if TYPE_CHECKING:
    from trinity.materials.variants import VariantConfig


class MaterialCompiler:
    """Compiles Material subclasses to complete WGSL shader modules.

    This compiler:
    1. Extracts the surface() method source from a Material class
    2. Translates Python AST to WGSL using PythonToWGSLTranslator
    3. Wraps the translated body in a PBR template with:
       - Struct definitions (PBRInput, PBRParams, PBROutput)
       - Texture and sampler bindings
       - Vertex shader entry point
       - Fragment shader entry point with BRDF evaluation

    Example::

        class GoldMaterial(Material, metaclass=MaterialMeta):
            @surface
            def surface(self, ctx: SurfaceContext, out: SurfaceOutput) -> None:
                out.base_color = Vec3(0.83, 0.69, 0.22)
                out.metallic = 0.9
                out.roughness = 0.3

        compiler = MaterialCompiler()
        wgsl = compiler.compile(GoldMaterial)
    """

    # Python type to WGSL type mapping
    TYPE_MAP = {
        float: "f32",
        int: "i32",
        bool: "bool",
        str: "str",
    }

    # PBR struct definitions - no format placeholders, used directly
    PBR_STRUCTS = """\
/// PBR shader input from vertex stage (vertex shader output / fragment shader input)
struct PBRInput {{
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) vertex_color: vec4<f32>,
}}

/// Material parameters from surface shader
struct PBRParams {{
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    normal: vec3<f32>,
    emissive: vec3<f32>,
    ambient_occlusion: f32,
    alpha: f32,
    specular: f32,
    subsurface: f32,
    subsurface_color: vec3<f32>,
    clearcoat: f32,
    clearcoat_roughness: f32,
    anisotropy: f32,
    anisotropy_direction: vec2<f32>,
    sheen: f32,
    sheen_color: vec3<f32>,
    transmission: f32,
    ior: f32,
}}

/// Fragment shader output
struct PBROutput {{
    @location(0) color: vec4<f32>,
}}

/// Global uniforms
struct Uniforms {{
    view_projection: mat4x4<f32>,
    camera_position: vec3<f32>,
    time: f32,
    light_count: u32,
}}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
"""

    # Vertex shader template with standard PBR vertex transformation
    VERTEX_TEMPLATE = """\
/// Vertex input from mesh
struct VertexInput {{
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) color: vec4<f32>,
}}

/// Model uniforms for per-object transform
struct ModelUniforms {{
    model: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
}}

@group(1) @binding(100) var<uniform> model: ModelUniforms;

@vertex
fn vs_main(in: VertexInput) -> PBRInput {{
    var out: PBRInput;

    let world_pos = model.model * vec4<f32>(in.position, 1.0);
    out.position = uniforms.view_projection * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal = normalize(model.normal_matrix * in.normal);
    out.world_tangent = vec4<f32>(normalize(model.normal_matrix * in.tangent.xyz), in.tangent.w);
    out.uv = in.uv;
    out.vertex_color = in.color;

    return out;
}}
"""

    # BRDF function declarations (placeholders for full implementation in T-MAT-3.2)
    BRDF_FUNCTIONS = """\
// ============================================================================
// BRDF Function Declarations
// Full implementations will be provided in T-MAT-3.2 (Cook-Torrance BRDF)
// ============================================================================

/// Fresnel-Schlick approximation
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {{
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}}

/// GGX/Trowbridge-Reitz Normal Distribution Function
fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {{
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;

    let num = a2;
    let denom = (NdotH2 * (a2 - 1.0) + 1.0);
    let denom2 = 3.14159265359 * denom * denom;

    return num / denom2;
}}

/// Smith's Geometry Function (Schlick-GGX)
fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {{
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}}

/// Smith's method for geometry obstruction
fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {{
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx2 = geometry_schlick_ggx(NdotV, roughness);
    let ggx1 = geometry_schlick_ggx(NdotL, roughness);
    return ggx1 * ggx2;
}}

/// Cook-Torrance BRDF evaluation
/// Returns specular and diffuse contribution for a single light
fn evaluate_brdf(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32
) -> vec3<f32> {{
    let H = normalize(V + L);

    // Fresnel reflectance at normal incidence
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);

    // Cook-Torrance BRDF components
    let D = distribution_ggx(N, H, roughness);
    let G = geometry_smith(N, V, L, roughness);
    let F = fresnel_schlick(max(dot(H, V), 0.0), f0);

    // Specular contribution
    let numerator = D * G * F;
    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.0);
    let denominator = 4.0 * NdotV * NdotL + 0.0001;
    let specular = numerator / denominator;

    // Diffuse contribution (energy conserving)
    let kS = F;
    let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);
    let diffuse = kD * base_color / 3.14159265359;

    return (diffuse + specular) * NdotL;
}}
"""

    # Light structure and light loop scaffolding
    LIGHT_LOOP = """\
// ============================================================================
// Light Loop Scaffolding
// Full implementation in T-MAT-3.3 (Light loop and shading)
// ============================================================================

/// Light types
const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;

/// Light data structure
struct Light {{
    position_or_direction: vec3<f32>,
    light_type: u32,
    color: vec3<f32>,
    intensity: f32,
    // For spot lights
    spot_direction: vec3<f32>,
    spot_angle_cos: f32,
    // Attenuation
    range: f32,
    _padding: vec3<f32>,
}}

/// Light buffer (max 16 lights for forward rendering)
const MAX_LIGHTS: u32 = 16u;

@group(2) @binding(0) var<storage, read> lights: array<Light, 16>;

/// Shadow sampling placeholder (returns 1.0 = fully lit)
/// Full implementation in T-MAT-3.3
fn sample_shadow(world_pos: vec3<f32>, light_index: u32) -> f32 {{
    return 1.0;
}}

/// Calculate light attenuation for point/spot lights
fn calculate_attenuation(light: Light, world_pos: vec3<f32>) -> f32 {{
    if (light.light_type == LIGHT_TYPE_DIRECTIONAL) {{
        return 1.0;
    }}

    let distance = length(light.position_or_direction - world_pos);
    let attenuation = 1.0 / (1.0 + distance * distance / (light.range * light.range));

    // Spot light cone attenuation
    if (light.light_type == LIGHT_TYPE_SPOT) {{
        let light_dir = normalize(light.position_or_direction - world_pos);
        let spot_cos = dot(-light_dir, normalize(light.spot_direction));
        let spot_atten = saturate((spot_cos - light.spot_angle_cos) / (1.0 - light.spot_angle_cos));
        return attenuation * spot_atten * spot_atten;
    }}

    return attenuation;
}}

/// Get light direction from light to surface point
fn get_light_direction(light: Light, world_pos: vec3<f32>) -> vec3<f32> {{
    if (light.light_type == LIGHT_TYPE_DIRECTIONAL) {{
        return normalize(-light.position_or_direction);
    }}
    return normalize(light.position_or_direction - world_pos);
}}

/// Evaluate lighting for all active lights
fn evaluate_lighting(
    world_pos: vec3<f32>,
    N: vec3<f32>,
    V: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32
) -> vec3<f32> {{
    var Lo = vec3<f32>(0.0);

    // Light loop - iterate over active lights
    for (var i = 0u; i < min(uniforms.light_count, MAX_LIGHTS); i = i + 1u) {{
        let light = lights[i];
        let L = get_light_direction(light, world_pos);
        let radiance = light.color * light.intensity * calculate_attenuation(light, world_pos);

        // Shadow factor
        let shadow = sample_shadow(world_pos, i);

        // BRDF evaluation
        let brdf = evaluate_brdf(N, V, L, base_color, metallic, roughness);

        Lo = Lo + brdf * radiance * shadow;
    }}

    // Ambient term (simplified IBL placeholder)
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let ambient = vec3<f32>(0.03) * base_color * ao;

    return ambient + Lo;
}}
"""

    # Fragment shader template with placeholder for surface body
    # Note: We use double braces {{ }} to escape literal braces in the template
    # The DSL translator generates code that writes to 'out.*', so we use 'out' as var name
    FRAGMENT_TEMPLATE = """
@fragment
fn fs_main(input: PBRInput) -> PBROutput {{
    // Initialize PBR parameters with defaults
    // Variable name 'out' matches DSL translator output
    var out: PBRParams;
    out.base_color = vec3<f32>(1.0, 1.0, 1.0);
    out.metallic = 0.0;
    out.roughness = 0.5;
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    out.emissive = vec3<f32>(0.0, 0.0, 0.0);
    out.ambient_occlusion = 1.0;
    out.alpha = 1.0;
    out.specular = 0.5;
    out.subsurface = 0.0;
    out.subsurface_color = vec3<f32>(1.0, 1.0, 1.0);
    out.clearcoat = 0.0;
    out.clearcoat_roughness = 0.03;
    out.anisotropy = 0.0;
    out.anisotropy_direction = vec2<f32>(1.0, 0.0);
    out.sheen = 0.0;
    out.sheen_color = vec3<f32>(1.0, 1.0, 1.0);
    out.transmission = 0.0;
    out.ior = 1.5;

    // === Surface shader body (translated from DSL) ===
{surface_body}
    // === End surface shader ===

    // Apply blend mode handling (alpha test for MASKED, alpha output for TRANSLUCENT)
    // ALPHA_TEST_ENABLED and ALPHA_BLEND_ENABLED are variant consts (default false)
    let blended_output = apply_blend_mode(vec4<f32>(out.base_color, out.alpha), 0.5);

    // Compute view vectors
    // Combine world normal with surface normal perturbation (normal mapping)
    let N = normalize(input.world_normal + out.normal - vec3<f32>(0.0, 0.0, 1.0));
    let V = normalize(uniforms.camera_position - input.world_position);

    // Evaluate PBR lighting via light loop
    let lit_color = evaluate_lighting(
        input.world_position,
        N,
        V,
        blended_output.rgb,
        out.metallic,
        out.roughness,
        out.ambient_occlusion
    );

    // Add emissive contribution
    let final_color = lit_color + out.emissive;

    return PBROutput(vec4<f32>(final_color, blended_output.a));
}}
"""

    def __init__(
        self,
        include_pbr_template: bool = True,
        variant_config: "VariantConfig | None" = None,
    ):
        """Initialize the compiler.

        Args:
            include_pbr_template: If True, wrap output in full PBR template.
                If False, only return the translated surface body.
            variant_config: Optional VariantConfig for injecting variant consts.
                If None, no variant consts are injected.
        """
        self.include_pbr_template = include_pbr_template
        self.variant_config = variant_config
        self.translator = PythonToWGSLTranslator()

    def compile(self, material_class: type) -> str:
        """Compile a Material subclass to WGSL surface function body.

        Args:
            material_class: A class with a surface() method to compile.

        Returns:
            WGSL shader code. If include_pbr_template is True, returns a
            complete shader module. Otherwise, returns just the surface body.

        Raises:
            WGSLTranslationError: If the surface method cannot be translated.
            ValueError: If the class has no surface method.
        """
        # Find surface method
        surface_method = None
        if hasattr(material_class, "_surface_method"):
            surface_method = material_class._surface_method
        elif hasattr(material_class, "surface"):
            surface_method = material_class.surface

        if surface_method is None:
            raise ValueError(f"{material_class.__name__} has no surface() method")

        # Check if already compiled
        used_builtins: set[str] = set()
        if hasattr(material_class, "_wgsl_source") and material_class._wgsl_source:
            body = material_class._wgsl_source
            # Try to recover used builtins from stored attribute
            if hasattr(material_class, "_used_builtins"):
                used_builtins = material_class._used_builtins
        else:
            body, used_builtins = self._compile_method(surface_method)

        if not self.include_pbr_template:
            return body

        # Wrap in PBR template with builtins
        return self._assemble_shader(material_class, body, used_builtins)

    def _compile_method(self, method) -> tuple[str, set[str]]:
        """Compile a single method to WGSL.

        Args:
            method: The method to compile.

        Returns:
            Tuple of (WGSL code for the method body, set of used builtins).
        """
        # Get source code
        source = textwrap.dedent(inspect.getsource(method))
        tree = ast.parse(source)

        # Translate to WGSL
        wgsl = self.translator.translate(tree)
        used_builtins = self.translator.get_used_builtins()

        return wgsl, used_builtins

    def _assemble_shader(
        self, material_class: type, surface_body: str, used_builtins: set[str] = None
    ) -> str:
        """Assemble a complete shader module from surface body.

        This assembles a full PBR shader including:
        - Struct definitions (PBRInput, PBRParams, PBROutput, Uniforms)
        - Vertex shader entry point (vs_main)
        - Custom builtin helper functions (noise, color conversion, etc.)
        - BRDF function declarations
        - Light loop scaffolding
        - Fragment shader entry point (fs_main) with surface body

        Args:
            material_class: The material class (for texture bindings).
            surface_body: The translated surface shader body.
            used_builtins: Set of custom builtin function names used.

        Returns:
            Complete WGSL shader module ready for naga validation.
        """
        # Collect texture bindings
        texture_bindings = ""
        if hasattr(material_class, "_texture_bindings"):
            binding_idx = 1  # 0 is uniforms
            for name, descriptor in material_class._texture_bindings.items():
                texture_bindings += self._emit_texture_binding(
                    name, descriptor, binding_idx
                )
                binding_idx += 2  # texture + sampler

        # Collect builtin helper functions if any are used
        builtin_helpers = ""
        if used_builtins:
            try:
                from trinity.materials.builtins import get_required_builtins
                builtin_helpers = get_required_builtins(used_builtins)
            except ImportError:
                pass  # builtins module not available

        # Indent surface body for insertion into fragment main
        indented_body = textwrap.indent(surface_body, "    ")

        # Build builtin section if helpers are needed
        builtin_section = ""
        if builtin_helpers:
            builtin_section = f"""
// ============================================================================
// Builtin Helper Functions (noise, color conversion, etc.)
// ============================================================================
{builtin_helpers}
"""

        # Build variant const section and blend handling if variant_config is set
        variant_section = ""
        blend_handling_section = ""
        if self.variant_config is not None:
            variant_section = f"""
{self.variant_config.generate_const_declarations()}
"""
            # Include blend mode handling code (apply_blend_mode function)
            blend_handling_section = f"""
// ============================================================================
// Blend Mode Handling
// ============================================================================
{self.variant_config.generate_blend_handling_code()}
"""
        else:
            # Provide default blend handling when no variant config
            # This ensures the shader always has the apply_blend_mode function
            blend_handling_section = """
// ============================================================================
// Blend Mode Handling (defaults)
// ============================================================================

// Default blend mode consts (when no variant config specified)
const ALPHA_TEST_ENABLED: bool = false;
const ALPHA_BLEND_ENABLED: bool = false;

/// Apply blend mode-specific operations (const-gated)
fn apply_blend_mode(color: vec4<f32>, alpha_threshold: f32) -> vec4<f32> {
    // Masked blend mode: alpha test with discard
    if ALPHA_TEST_ENABLED {
        if (color.a < alpha_threshold) {
            discard;
        }
        // Return fully opaque for masked materials
        return vec4<f32>(color.rgb, 1.0);
    }

    // Translucent/Additive/Modulate: preserve alpha for blending
    if ALPHA_BLEND_ENABLED {
        return color;
    }

    // Opaque: force alpha to 1.0
    return vec4<f32>(color.rgb, 1.0);
}
"""

        # Assemble complete shader module with all components
        shader = f"""\
// SPDX-License-Identifier: MIT
// Generated by TRINITY MaterialCompiler
// Material: {material_class.__name__}
{variant_section}
// ============================================================================
// PBR Struct Definitions
// ============================================================================
{self.PBR_STRUCTS}

// ============================================================================
// Vertex Shader
// ============================================================================
{self.VERTEX_TEMPLATE}

// ============================================================================
// Material Texture Bindings
// ============================================================================
{texture_bindings}
{builtin_section}
// ============================================================================
// BRDF Functions
// ============================================================================
{self.BRDF_FUNCTIONS}

// ============================================================================
// Light Loop
// ============================================================================
{self.LIGHT_LOOP}
{blend_handling_section}
// ============================================================================
// Fragment Shader
// ============================================================================
{self.FRAGMENT_TEMPLATE.format(surface_body=indented_body)}
"""
        return shader

    def _emit_texture_binding(
        self, name: str, descriptor: Any, binding_idx: int
    ) -> str:
        """Emit WGSL bindings for a texture descriptor.

        Args:
            name: Texture variable name
            descriptor: Texture descriptor object
            binding_idx: Starting binding index

        Returns:
            WGSL binding declarations
        """
        texture_type = "texture_2d<f32>"
        if hasattr(descriptor, "_is_cube") and descriptor._is_cube:
            texture_type = "texture_cube<f32>"

        return f"""
@group(1) @binding({binding_idx}) var {name}_texture: {texture_type};
@group(1) @binding({binding_idx + 1}) var {name}_sampler: sampler;
"""

    def compile_to_spirv(self, material_class: type) -> bytes:
        """Compile a Material to SPIR-V bytecode (via naga).

        Args:
            material_class: The material class to compile.

        Returns:
            SPIR-V bytecode as bytes.

        Raises:
            ImportError: If naga-py is not available.
            RuntimeError: If SPIR-V compilation fails.
        """
        wgsl = self.compile(material_class)

        try:
            import naga
        except ImportError:
            raise ImportError(
                "naga-py is required for SPIR-V compilation. "
                "Install with: pip install naga"
            )

        try:
            module = naga.parse_wgsl(wgsl)
            return naga.to_spirv(module)
        except Exception as e:
            raise RuntimeError(f"SPIR-V compilation failed: {e}")

    def validate_wgsl(self, wgsl: str) -> tuple[bool, Optional[str]]:
        """Validate WGSL code using naga.

        Args:
            wgsl: WGSL shader source code.

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
        """
        try:
            import naga
        except ImportError:
            # Without naga, we can't validate - assume valid
            return True, None

        try:
            naga.parse_wgsl(wgsl)
            return True, None
        except Exception as e:
            return False, str(e)

    def compile_with_variants(
        self,
        material_class: type,
        variant_config: "VariantConfig",
    ) -> str:
        """Compile a Material with specific variant configuration.

        This method temporarily sets the variant_config, compiles the material,
        and returns the WGSL with variant const declarations injected.

        Args:
            material_class: A class with a surface() method to compile.
            variant_config: VariantConfig specifying domain, blend, and quality.

        Returns:
            Complete WGSL shader with variant consts.

        Example::

            from trinity.materials.variants import VariantConfig, QualityTier

            config = VariantConfig(quality=QualityTier.LOW)
            wgsl = compiler.compile_with_variants(MyMaterial, config)
        """
        old_config = self.variant_config
        try:
            self.variant_config = variant_config
            return self.compile(material_class)
        finally:
            self.variant_config = old_config

    def compile_all_variants(
        self,
        material_class: type,
    ) -> dict[int, str]:
        """Compile a Material for all 75 variant combinations.

        Generates shaders for all domain x blend x quality combinations.
        Useful for pre-compilation and caching.

        Args:
            material_class: A class with a surface() method to compile.

        Returns:
            Dict mapping variant key (int) to WGSL source (str).
            Use VariantConfig.get_variant_key() to look up specific variants.

        Example::

            all_shaders = compiler.compile_all_variants(MyMaterial)
            # Get specific variant
            config = VariantConfig(quality=QualityTier.LOW)
            wgsl = all_shaders[config.get_variant_key()]
        """
        from trinity.materials.variants import generate_all_variant_combinations

        results = {}
        for config in generate_all_variant_combinations():
            wgsl = self.compile_with_variants(material_class, config)
            results[config.get_variant_key()] = wgsl
        return results
