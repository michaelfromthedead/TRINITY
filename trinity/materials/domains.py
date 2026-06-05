"""Domain-specific shader templates for material variants.

This module provides WGSL shader templates for each MaterialDomain type.
Each domain has distinct rendering requirements and output structures:

- SURFACE: Full PBR shading with lighting, shadows, and advanced effects
- DEFERRED_DECAL: G-buffer modification without direct lighting evaluation
- VOLUME: Single-scattering volumetric rendering
- POST_PROCESS: Fullscreen effects (tonemapping, color grading, etc.)
- UI: Unlit vertex-color multiplied by texture

The templates integrate with VariantConfig to generate domain-gated shader code
that is eliminated at compile time via naga dead-code elimination when the
const bool guards evaluate to false.

Task: T-MAT-2.2 Domain Variants
Gap: S3-G3 (CRITICAL)
Dependency: T-MAT-2.1 (variant const system)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Type

from trinity.materials.variants import MaterialDomain, VariantConfig


class DomainCapability(Enum):
    """Feature capabilities available per domain."""
    LIGHTING = "lighting"                 # Full PBR lighting evaluation
    SHADOWS = "shadows"                   # Shadow sampling
    GBUFFER_OUTPUT = "gbuffer_output"     # G-buffer writes for deferred
    VOLUMETRIC = "volumetric"             # Volume rendering (density/scattering)
    FULLSCREEN = "fullscreen"             # Fullscreen quad rendering
    VERTEX_COLOR = "vertex_color"         # Vertex color multiplication
    DEPTH_WRITE = "depth_write"           # Depth buffer writes
    NORMAL_MAPPING = "normal_mapping"     # Tangent-space normal perturbation
    ENVIRONMENT_MAP = "environment_map"  # IBL/environment sampling
    EMISSIVE = "emissive"                 # Emissive contribution


# Domain capability mappings
DOMAIN_CAPABILITIES: Dict[MaterialDomain, set[DomainCapability]] = {
    MaterialDomain.SURFACE: {
        DomainCapability.LIGHTING,
        DomainCapability.SHADOWS,
        DomainCapability.DEPTH_WRITE,
        DomainCapability.NORMAL_MAPPING,
        DomainCapability.ENVIRONMENT_MAP,
        DomainCapability.EMISSIVE,
    },
    MaterialDomain.DEFERRED_DECAL: {
        DomainCapability.GBUFFER_OUTPUT,
        DomainCapability.NORMAL_MAPPING,
    },
    MaterialDomain.VOLUME: {
        DomainCapability.VOLUMETRIC,
        DomainCapability.EMISSIVE,
    },
    MaterialDomain.POST_PROCESS: {
        DomainCapability.FULLSCREEN,
    },
    MaterialDomain.UI: {
        DomainCapability.VERTEX_COLOR,
    },
}


def domain_has_capability(domain: MaterialDomain, capability: DomainCapability) -> bool:
    """Check if a domain supports a specific capability.

    Args:
        domain: The material domain to check.
        capability: The capability to query.

    Returns:
        True if the domain supports the capability.
    """
    return capability in DOMAIN_CAPABILITIES.get(domain, set())


@dataclass(slots=True)
class DomainOutputFormat:
    """Describes the output format for a domain's fragment shader.

    Attributes:
        primary_color: Whether primary color (location 0) is output
        normal: Whether normal (location 1) is output (G-buffer)
        material: Whether material properties (location 2) are output (G-buffer)
        color_type: The WGSL type for color output ("vec4<f32>")
    """
    primary_color: bool = True
    normal: bool = False
    material: bool = False
    color_type: str = "vec4<f32>"

    def generate_output_struct(self, struct_name: str = "FragmentOutput") -> str:
        """Generate WGSL struct for fragment output.

        Args:
            struct_name: Name of the output struct.

        Returns:
            WGSL struct definition.
        """
        lines = [f"struct {struct_name} {{"]

        if self.primary_color:
            lines.append(f"    @location(0) color: {self.color_type},")
        if self.normal:
            lines.append("    @location(1) normal: vec4<f32>,")
        if self.material:
            lines.append("    @location(2) material: vec4<f32>,")

        lines.append("}")
        return "\n".join(lines)


# Output formats per domain
DOMAIN_OUTPUT_FORMATS: Dict[MaterialDomain, DomainOutputFormat] = {
    MaterialDomain.SURFACE: DomainOutputFormat(
        primary_color=True,
        normal=False,
        material=False,
    ),
    MaterialDomain.DEFERRED_DECAL: DomainOutputFormat(
        primary_color=True,
        normal=True,
        material=True,
    ),
    MaterialDomain.VOLUME: DomainOutputFormat(
        primary_color=True,
        normal=False,
        material=False,
    ),
    MaterialDomain.POST_PROCESS: DomainOutputFormat(
        primary_color=True,
        normal=False,
        material=False,
    ),
    MaterialDomain.UI: DomainOutputFormat(
        primary_color=True,
        normal=False,
        material=False,
    ),
}


class DomainShaderTemplate:
    """Domain-specific shader templates for material variants.

    Each template provides the core domain-specific shading logic that
    is gated behind DOMAIN_* const bools for dead-code elimination.
    """

    # =========================================================================
    # SURFACE Domain: Full PBR shading with lighting
    # =========================================================================

    SURFACE_FRAGMENT = '''\
/// Evaluate surface domain shading (full PBR with lighting).
/// This is the primary shading path for opaque and translucent objects.
fn evaluate_surface_domain(params: PBRParams, input: PBRInput) -> vec4<f32> {
    // Transform tangent-space normal to world space
    let T = normalize(input.world_tangent.xyz);
    let B = cross(input.world_normal, T) * input.world_tangent.w;
    let N_perturbed = params.normal;
    let N = normalize(
        T * N_perturbed.x +
        B * N_perturbed.y +
        input.world_normal * N_perturbed.z
    );
    let V = normalize(input.world_view);

    var color = vec3<f32>(0.0);

    // Lighting evaluation (quality-gated in parent)
    if LIGHTING_ENABLED {
        // Direct lighting loop
        for (var i = 0u; i < min(input.light_count, MAX_LIGHTS); i = i + 1u) {
            let light_contrib = evaluate_direct_light(params, N, V, i);

            // Shadow attenuation
            var shadow_atten = 1.0;
            if SHADOWS_ENABLED {
                shadow_atten = sample_shadow_for_light(input.world_position, i);
            }

            color = color + light_contrib * shadow_atten;
        }

        // Ambient/IBL contribution
        if QUALITY_HIGH {
            color = color + evaluate_ibl(params, N, V);
        } else {
            // Simplified ambient for lower quality
            color = color + vec3<f32>(0.03) * params.base_color * params.occlusion;
        }
    } else {
        // Unlit fallback
        color = params.base_color;
    }

    // Emissive contribution (always evaluated for SURFACE)
    color = color + params.emissive;

    // Apply ambient occlusion
    color = color * params.occlusion;

    return vec4<f32>(color, params.alpha);
}
'''

    # =========================================================================
    # DEFERRED_DECAL Domain: G-buffer modification
    # =========================================================================

    DEFERRED_DECAL_FRAGMENT = '''\
/// Decal output structure for G-buffer modification.
struct DecalOutput {
    /// Base color with alpha for blending
    color: vec4<f32>,
    /// World-space normal (packed)
    normal: vec4<f32>,
    /// Metallic-roughness packed
    material: vec4<f32>,
    /// Blend factor for decal application
    blend: f32,
}

/// Evaluate deferred decal domain (G-buffer modification only).
/// Decals modify existing G-buffer data without direct lighting.
fn evaluate_deferred_decal_domain(params: PBRParams, input: PBRInput) -> DecalOutput {
    var output: DecalOutput;

    // Transform normal to world space for G-buffer storage
    let T = normalize(input.world_tangent.xyz);
    let B = cross(input.world_normal, T) * input.world_tangent.w;
    let N_perturbed = params.normal;
    let world_normal = normalize(
        T * N_perturbed.x +
        B * N_perturbed.y +
        input.world_normal * N_perturbed.z
    );

    // Pack base color with alpha
    output.color = vec4<f32>(params.base_color, params.alpha);

    // Pack world normal (octahedral encoding could be used here)
    output.normal = vec4<f32>(world_normal * 0.5 + 0.5, 1.0);

    // Pack material properties: metallic, roughness, specular, AO
    output.material = vec4<f32>(
        params.metallic,
        params.roughness,
        params.specular,
        params.occlusion
    );

    // Blend factor based on alpha and projection angle
    let angle_fade = abs(dot(input.world_normal, vec3<f32>(0.0, 1.0, 0.0)));
    output.blend = params.alpha * angle_fade;

    return output;
}

/// Convert decal output to fragment output format.
fn decal_to_fragment_output(decal: DecalOutput) -> vec4<f32> {
    // For deferred rendering, output color with blend-weighted alpha
    return vec4<f32>(decal.color.rgb, decal.blend);
}
'''

    # =========================================================================
    # VOLUME Domain: Volumetric single-scattering
    # =========================================================================

    VOLUME_FRAGMENT = '''\
/// Volume rendering parameters derived from PBR params.
struct VolumeParams {
    /// Absorption coefficient (how much light is absorbed per unit distance)
    absorption: vec3<f32>,
    /// Scattering coefficient (how much light scatters per unit distance)
    scattering: vec3<f32>,
    /// Phase function asymmetry parameter (-1 to 1, 0 = isotropic)
    phase_g: f32,
    /// Density at this sample point
    density: f32,
    /// Emissive contribution (self-illumination)
    emission: vec3<f32>,
}

/// Convert PBR params to volume rendering params.
fn pbr_to_volume_params(params: PBRParams) -> VolumeParams {
    var vol: VolumeParams;

    // Use base_color as scattering albedo
    // Higher metallic = more absorption, less scattering
    let albedo = params.base_color * (1.0 - params.metallic * 0.8);

    // Derive absorption from inverse of albedo
    vol.absorption = (vec3<f32>(1.0) - albedo) * params.roughness;
    vol.scattering = albedo * (1.0 - params.roughness * 0.5);

    // Use specular as phase function asymmetry
    vol.phase_g = params.specular * 2.0 - 1.0;

    // Alpha controls density
    vol.density = params.alpha;

    // Direct emissive passthrough
    vol.emission = params.emissive;

    return vol;
}

/// Henyey-Greenstein phase function.
fn phase_hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265359 * pow(denom, 1.5));
}

/// Evaluate volume domain shading (single-scattering approximation).
/// For full volumetric rendering, raymarching would be done in a separate pass.
fn evaluate_volume_domain(params: PBRParams, input: PBRInput) -> vec4<f32> {
    let vol = pbr_to_volume_params(params);

    // For single-sample evaluation (no raymarching here)
    // This provides the contribution for one sample point in a volume

    var in_scatter = vec3<f32>(0.0);

    if LIGHTING_ENABLED {
        // Accumulate inscattering from lights
        for (var i = 0u; i < min(input.light_count, MAX_LIGHTS); i = i + 1u) {
            let light_dir = get_light_direction_for_volume(input.world_position, i);
            let light_color = get_light_color_for_volume(i);
            let cos_theta = dot(normalize(-input.world_view), light_dir);
            let phase = phase_hg(cos_theta, vol.phase_g);
            in_scatter = in_scatter + light_color * vol.scattering * phase;
        }
    }

    // Add emissive contribution
    let scatter_color = in_scatter + vol.emission;

    // Transmittance approximation for this sample
    // (Full Beer-Lambert would require path length)
    let extinction = vol.absorption + vol.scattering;
    let approx_transmittance = exp(-length(extinction) * vol.density);

    // Output: scattered light with density-based alpha
    return vec4<f32>(scatter_color * vol.density, vol.density * (1.0 - approx_transmittance));
}
'''

    # =========================================================================
    # POST_PROCESS Domain: Fullscreen effects
    # =========================================================================

    POST_PROCESS_FRAGMENT = '''\
/// Post-process input from scene render target.
struct PostProcessInput {
    /// Scene color before post-processing
    color: vec3<f32>,
    /// Scene depth (linear)
    depth: f32,
    /// UV coordinates (0-1 screen space)
    uv: vec2<f32>,
    /// Time for animated effects
    time: f32,
}

/// ACES filmic tonemapping operator.
fn tonemap_aces(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

/// Reinhard tonemapping operator.
fn tonemap_reinhard(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3<f32>(1.0));
}

/// sRGB OETF (linear to sRGB gamma).
fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let low = linear * 12.92;
    let high = 1.055 * pow(linear, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(low, high, linear > cutoff);
}

/// Evaluate post-process domain (fullscreen tonemapping/effects).
/// PBRParams are repurposed: base_color = scene color, alpha = exposure.
fn evaluate_postprocess_domain(params: PBRParams, input: PBRInput) -> vec4<f32> {
    var color = params.base_color;

    // Apply exposure (stored in alpha, default 1.0)
    let exposure = max(params.alpha, 0.001);
    color = color * exposure;

    // Tonemapping (quality-dependent)
    if QUALITY_HIGH {
        color = tonemap_aces(color);
    } else {
        color = tonemap_reinhard(color);
    }

    // Apply gamma correction for final output
    color = linear_to_srgb(color);

    // Vignette effect (subtle)
    if QUALITY_HIGH {
        let vignette_uv = input.uv * 2.0 - 1.0;
        let vignette = 1.0 - dot(vignette_uv, vignette_uv) * 0.25;
        color = color * vignette;
    }

    return vec4<f32>(color, 1.0);
}
'''

    # =========================================================================
    # UI Domain: Unlit vertex-color
    # =========================================================================

    UI_FRAGMENT = '''\
/// Evaluate UI domain shading (unlit vertex-color multiply).
/// UI rendering is simple: vertex color * texture color, no lighting.
fn evaluate_ui_domain(params: PBRParams, input: PBRInput) -> vec4<f32> {
    // UI uses vertex color multiplication
    let tex_color = vec4<f32>(params.base_color, params.alpha);
    let vert_color = input.vertex_color;

    // Simple multiply blend
    var result = tex_color * vert_color;

    // Premultiplied alpha output for UI compositing
    result.r = result.r * result.a;
    result.g = result.g * result.a;
    result.b = result.b * result.a;

    return result;
}

/// UI with gamma-correct blending (for sRGB framebuffers).
fn evaluate_ui_domain_srgb(params: PBRParams, input: PBRInput) -> vec4<f32> {
    let tex_color = vec4<f32>(params.base_color, params.alpha);
    let vert_color = input.vertex_color;

    // Blend in linear space, output in sRGB
    var linear_result = tex_color * vert_color;

    // Convert to sRGB for framebuffer
    let srgb_color = linear_to_srgb(linear_result.rgb);

    return vec4<f32>(srgb_color, linear_result.a);
}
'''

    # =========================================================================
    # Template accessor methods
    # =========================================================================

    @classmethod
    def get_for_domain(cls, domain: MaterialDomain) -> str:
        """Get the shader template for a specific domain.

        Args:
            domain: The MaterialDomain to get template for.

        Returns:
            WGSL shader code for the domain's evaluation function.
        """
        mapping = {
            MaterialDomain.SURFACE: cls.SURFACE_FRAGMENT,
            MaterialDomain.DEFERRED_DECAL: cls.DEFERRED_DECAL_FRAGMENT,
            MaterialDomain.VOLUME: cls.VOLUME_FRAGMENT,
            MaterialDomain.POST_PROCESS: cls.POST_PROCESS_FRAGMENT,
            MaterialDomain.UI: cls.UI_FRAGMENT,
        }
        return mapping.get(domain, cls.SURFACE_FRAGMENT)

    @classmethod
    def get_all_templates(cls) -> Dict[MaterialDomain, str]:
        """Get all domain templates.

        Returns:
            Dictionary mapping each domain to its shader template.
        """
        return {domain: cls.get_for_domain(domain) for domain in MaterialDomain}

    @classmethod
    def get_domain_function_name(cls, domain: MaterialDomain) -> str:
        """Get the primary evaluation function name for a domain.

        Args:
            domain: The MaterialDomain.

        Returns:
            Function name string (e.g., "evaluate_surface_domain").
        """
        mapping = {
            MaterialDomain.SURFACE: "evaluate_surface_domain",
            MaterialDomain.DEFERRED_DECAL: "evaluate_deferred_decal_domain",
            MaterialDomain.VOLUME: "evaluate_volume_domain",
            MaterialDomain.POST_PROCESS: "evaluate_postprocess_domain",
            MaterialDomain.UI: "evaluate_ui_domain",
        }
        return mapping.get(domain, "evaluate_surface_domain")


class DomainVariantGenerator:
    """Generates domain-specific shader code for material variants.

    Integrates with VariantConfig to produce complete domain-gated
    shader code that leverages naga dead-code elimination.
    """

    def __init__(self, config: VariantConfig):
        """Initialize with a variant configuration.

        Args:
            config: VariantConfig specifying domain, blend, quality.
        """
        self.config = config

    def generate_domain_code(self) -> str:
        """Generate complete domain-specific shader code.

        Returns:
            WGSL code block with domain template and dispatch function.
        """
        lines: List[str] = []

        # Header
        lines.append("// =============================================================================")
        lines.append("// Domain-Specific Shader Templates")
        lines.append(f"// Active Domain: {self.config.domain.name}")
        lines.append("// Generated by TRINITY DomainVariantGenerator")
        lines.append("// =============================================================================")
        lines.append("")

        # Include helper functions based on capabilities
        lines.append(self._generate_helper_stubs())
        lines.append("")

        # Include the active domain template
        template = DomainShaderTemplate.get_for_domain(self.config.domain)
        lines.append(template)
        lines.append("")

        # Generate domain dispatch function
        lines.append(self._generate_dispatch_function())

        return "\n".join(lines)

    def generate_all_domain_code(self) -> str:
        """Generate code for ALL domains (for variants with multiple paths).

        This generates all domain templates so the const-gated dispatch
        can select the correct path at compile time.

        Returns:
            WGSL code with all domain templates and unified dispatch.
        """
        lines: List[str] = []

        # Header
        lines.append("// =============================================================================")
        lines.append("// All Domain-Specific Shader Templates")
        lines.append("// Unused paths eliminated by naga dead-code elimination")
        lines.append("// =============================================================================")
        lines.append("")

        # Helper stubs
        lines.append(self._generate_helper_stubs())
        lines.append("")

        # All domain templates
        for domain in MaterialDomain:
            lines.append(f"// --- {domain.name} Domain ---")
            lines.append(DomainShaderTemplate.get_for_domain(domain))
            lines.append("")

        # Unified dispatch
        lines.append(self._generate_unified_dispatch())

        return "\n".join(lines)

    def _generate_helper_stubs(self) -> str:
        """Generate stub declarations for helper functions.

        These stubs allow domain templates to compile; actual implementations
        are provided by the lighting and BRDF modules.
        """
        return '''\
// Helper function stubs (implementations provided by lighting module)
fn evaluate_direct_light(params: PBRParams, N: vec3<f32>, V: vec3<f32>, light_idx: u32) -> vec3<f32> {
    // Stub: actual implementation in lighting.wgsl
    return vec3<f32>(0.0);
}

fn evaluate_ibl(params: PBRParams, N: vec3<f32>, V: vec3<f32>) -> vec3<f32> {
    // Stub: actual implementation in ibl.wgsl
    return vec3<f32>(0.03) * params.base_color;
}

fn sample_shadow_for_light(world_pos: vec3<f32>, light_idx: u32) -> f32 {
    // Stub: actual implementation in shadows.wgsl
    return 1.0;
}

fn get_light_direction_for_volume(world_pos: vec3<f32>, light_idx: u32) -> vec3<f32> {
    // Stub: actual implementation in lighting.wgsl
    return vec3<f32>(0.0, 1.0, 0.0);
}

fn get_light_color_for_volume(light_idx: u32) -> vec3<f32> {
    // Stub: actual implementation in lighting.wgsl
    return vec3<f32>(1.0);
}
'''

    def _generate_dispatch_function(self) -> str:
        """Generate domain dispatch for active domain only.

        Returns:
            WGSL dispatch function that evaluates the active domain.
        """
        func_name = DomainShaderTemplate.get_domain_function_name(self.config.domain)

        return f'''\
/// Dispatch to active domain evaluation.
/// This variant is compiled for DOMAIN_{self.config.domain.name}.
fn evaluate_domain(params: PBRParams, input: PBRInput) -> vec4<f32> {{
    return {func_name}(params, input);
}}
'''

    def _generate_unified_dispatch(self) -> str:
        """Generate unified dispatch across all domains.

        Returns:
            WGSL dispatch function with const-gated domain selection.
        """
        return '''\
/// Unified domain dispatch (const-gated for dead-code elimination).
/// Only the active domain path will survive naga compilation.
fn evaluate_domain(params: PBRParams, input: PBRInput) -> vec4<f32> {
    if DOMAIN_SURFACE {
        return evaluate_surface_domain(params, input);
    }

    if DOMAIN_DEFERRED_DECAL {
        let decal = evaluate_deferred_decal_domain(params, input);
        return decal_to_fragment_output(decal);
    }

    if DOMAIN_VOLUME {
        return evaluate_volume_domain(params, input);
    }

    if DOMAIN_POST_PROCESS {
        return evaluate_postprocess_domain(params, input);
    }

    if DOMAIN_UI {
        return evaluate_ui_domain(params, input);
    }

    // Fallback (should be eliminated by dead-code optimization)
    return vec4<f32>(params.base_color, params.alpha);
}
'''

    def get_output_format(self) -> DomainOutputFormat:
        """Get the output format for the active domain.

        Returns:
            DomainOutputFormat describing fragment output structure.
        """
        return DOMAIN_OUTPUT_FORMATS.get(
            self.config.domain,
            DomainOutputFormat()
        )

    def get_capabilities(self) -> set[DomainCapability]:
        """Get capabilities enabled for the active domain.

        Returns:
            Set of DomainCapability values for this domain.
        """
        return DOMAIN_CAPABILITIES.get(
            self.config.domain,
            set()
        )


def get_domain_shader_info(domain: MaterialDomain) -> dict:
    """Get comprehensive information about a domain's shader requirements.

    Args:
        domain: The MaterialDomain to query.

    Returns:
        Dictionary with capabilities, output format, function name, etc.
    """
    return {
        "domain": domain,
        "capabilities": DOMAIN_CAPABILITIES.get(domain, set()),
        "output_format": DOMAIN_OUTPUT_FORMATS.get(domain, DomainOutputFormat()),
        "function_name": DomainShaderTemplate.get_domain_function_name(domain),
        "template_length": len(DomainShaderTemplate.get_for_domain(domain)),
        "has_lighting": DomainCapability.LIGHTING in DOMAIN_CAPABILITIES.get(domain, set()),
        "has_gbuffer": DomainCapability.GBUFFER_OUTPUT in DOMAIN_CAPABILITIES.get(domain, set()),
    }


__all__ = [
    "DomainCapability",
    "DomainOutputFormat",
    "DomainShaderTemplate",
    "DomainVariantGenerator",
    "DOMAIN_CAPABILITIES",
    "DOMAIN_OUTPUT_FORMATS",
    "domain_has_capability",
    "get_domain_shader_info",
]
