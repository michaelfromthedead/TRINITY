"""Variant const system for material shader permutations.

This module implements a const-boolean variant selection system that enables
naga dead-code elimination to remove inactive shader branches. The system
supports three variant dimensions:

1. Domain variants: SURFACE, DEFERRED_DECAL, VOLUME, POST_PROCESS, UI
2. Blend mode variants: OPAQUE, MASKED, TRANSLUCENT, ADDITIVE, MODULATE
3. Quality tier variants: LOW, MEDIUM, HIGH

Each variant is represented as WGSL const bools that gate optional features
like lighting loops, shadow sampling, and advanced shading. naga's dead-code
elimination removes branches guarded by false const bools at compile time.

Example::

    config = VariantConfig(
        domain=MaterialDomain.SURFACE,
        blend=BlendMode.OPAQUE,
        quality=QualityTier.HIGH,
    )

    const_decls = config.generate_const_declarations()
    # Produces:
    # const DOMAIN_SURFACE: bool = true;
    # const DOMAIN_DEFERRED_DECAL: bool = false;
    # ...
    # const BLEND_OPAQUE: bool = true;
    # ...
    # const QUALITY_HIGH: bool = true;
    # ...

Task: T-MAT-2.1 Variant const system
Gap: S3-G3 (CRITICAL)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class MaterialDomain(Enum):
    """Material pipeline domain types.

    Each domain produces different shader output structures and
    enables/disables different rendering features.
    """
    SURFACE = "surface"           # Full PBR surface shading
    DEFERRED_DECAL = "deferred_decal"  # G-buffer modification only
    VOLUME = "volume"             # Single-scattering volumetric
    POST_PROCESS = "post_process" # Fullscreen post-processing
    UI = "ui"                     # Unlit vertex-color UI


class BlendMode(Enum):
    """Material blend modes.

    Controls alpha blending, depth write behavior, and lighting evaluation.
    """
    OPAQUE = "opaque"           # No transparency, full depth write
    MASKED = "masked"           # Alpha test with discard
    TRANSLUCENT = "translucent" # Alpha blend, transmissive lighting
    ADDITIVE = "additive"       # Additive blending
    MODULATE = "modulate"       # Multiplicative blending


class QualityTier(Enum):
    """Quality tier for scalable rendering.

    Controls feature complexity: light count, shadow quality, advanced shading.
    """
    LOW = "low"       # 1 light, no shadows, basic shading
    MEDIUM = "medium" # 4 lights, basic shadows, limited advanced
    HIGH = "high"     # Unlimited lights, PCSS shadows, full advanced


@dataclass(slots=True)
class VariantConfig:
    """Configuration for a material shader variant.

    Encapsulates the three variant dimensions (domain, blend, quality) and
    generates WGSL const declarations that enable naga dead-code elimination.

    Attributes:
        domain: Material domain (SURFACE, VOLUME, etc.)
        blend: Blend mode (OPAQUE, MASKED, etc.)
        quality: Quality tier (LOW, MEDIUM, HIGH)
        feature_flags: Additional feature toggles (e.g., "ENABLE_SSS")
    """
    domain: MaterialDomain = MaterialDomain.SURFACE
    blend: BlendMode = BlendMode.OPAQUE
    quality: QualityTier = QualityTier.HIGH
    feature_flags: Set[str] = field(default_factory=set)

    # Quality tier feature limits
    QUALITY_CONFIG = {
        QualityTier.LOW: {
            "max_lights": 1,
            "shadows_enabled": False,
            "shadow_quality": "none",
            "advanced_shading": False,
            "subsurface_enabled": False,
            "clearcoat_enabled": False,
            "anisotropy_enabled": False,
            "sheen_enabled": False,
            "transmission_enabled": False,
            "iridescence_enabled": False,
        },
        QualityTier.MEDIUM: {
            "max_lights": 4,
            "shadows_enabled": True,
            "shadow_quality": "basic",
            "advanced_shading": True,
            "subsurface_enabled": False,
            "clearcoat_enabled": True,
            "anisotropy_enabled": False,
            "sheen_enabled": False,
            "transmission_enabled": True,
            "iridescence_enabled": False,
        },
        QualityTier.HIGH: {
            "max_lights": 16,
            "shadows_enabled": True,
            "shadow_quality": "pcss",
            "advanced_shading": True,
            "subsurface_enabled": True,
            "clearcoat_enabled": True,
            "anisotropy_enabled": True,
            "sheen_enabled": True,
            "transmission_enabled": True,
            "iridescence_enabled": True,
        },
    }

    def generate_const_declarations(self) -> str:
        """Generate WGSL const bool declarations for this variant.

        Returns:
            WGSL code block with const declarations for domain, blend,
            quality, and feature flags. All domain, blend, and quality
            consts are declared, with exactly one true per category.
        """
        lines: List[str] = []

        # Header comment
        lines.append("// =============================================================================")
        lines.append("// Variant Const Declarations")
        lines.append("// Generated by TRINITY VariantConfig")
        lines.append("// =============================================================================")
        lines.append("")

        # Domain variants
        lines.append("// Domain variants (exactly one is true)")
        for domain in MaterialDomain:
            const_name = f"DOMAIN_{domain.name}"
            value = "true" if domain == self.domain else "false"
            lines.append(f"const {const_name}: bool = {value};")
        lines.append("")

        # Blend mode variants
        lines.append("// Blend mode variants (exactly one is true)")
        for blend in BlendMode:
            const_name = f"BLEND_{blend.name}"
            value = "true" if blend == self.blend else "false"
            lines.append(f"const {const_name}: bool = {value};")
        lines.append("")

        # Quality tier variants
        lines.append("// Quality tier variants (exactly one is true)")
        for quality in QualityTier:
            const_name = f"QUALITY_{quality.name}"
            value = "true" if quality == self.quality else "false"
            lines.append(f"const {const_name}: bool = {value};")
        lines.append("")

        # Quality-derived feature consts
        quality_cfg = self.QUALITY_CONFIG[self.quality]
        lines.append("// Quality-derived feature constants")
        lines.append(f"const MAX_LIGHTS: u32 = {quality_cfg['max_lights']}u;")
        lines.append(f"const SHADOWS_ENABLED: bool = {str(quality_cfg['shadows_enabled']).lower()};")
        lines.append(f"const ADVANCED_SHADING_ENABLED: bool = {str(quality_cfg['advanced_shading']).lower()};")
        lines.append(f"const SUBSURFACE_ENABLED: bool = {str(quality_cfg['subsurface_enabled']).lower()};")
        lines.append(f"const CLEARCOAT_ENABLED: bool = {str(quality_cfg['clearcoat_enabled']).lower()};")
        lines.append(f"const ANISOTROPY_ENABLED: bool = {str(quality_cfg['anisotropy_enabled']).lower()};")
        lines.append(f"const SHEEN_ENABLED: bool = {str(quality_cfg['sheen_enabled']).lower()};")
        lines.append(f"const TRANSMISSION_ENABLED: bool = {str(quality_cfg['transmission_enabled']).lower()};")
        lines.append(f"const IRIDESCENCE_ENABLED: bool = {str(quality_cfg['iridescence_enabled']).lower()};")
        lines.append("")

        # Domain-derived feature consts
        lines.append("// Domain-derived feature constants")
        lines.append(f"const LIGHTING_ENABLED: bool = {str(self._lighting_enabled()).lower()};")
        lines.append(f"const DEPTH_WRITE_ENABLED: bool = {str(self._depth_write_enabled()).lower()};")
        lines.append(f"const PBR_ENABLED: bool = {str(self._pbr_enabled()).lower()};")
        lines.append("")

        # Blend mode feature consts
        lines.append("// Blend-derived feature constants")
        lines.append(f"const ALPHA_TEST_ENABLED: bool = {str(self.blend == BlendMode.MASKED).lower()};")
        lines.append(f"const ALPHA_BLEND_ENABLED: bool = {str(self.blend in (BlendMode.TRANSLUCENT, BlendMode.ADDITIVE, BlendMode.MODULATE)).lower()};")
        lines.append("")

        # Custom feature flags
        if self.feature_flags:
            lines.append("// Custom feature flags")
            for flag in sorted(self.feature_flags):
                # Sanitize flag name to valid WGSL identifier
                safe_name = flag.upper().replace("-", "_").replace(" ", "_")
                lines.append(f"const {safe_name}: bool = true;")
            lines.append("")

        return "\n".join(lines)

    def generate_gated_lighting_code(self) -> str:
        """Generate quality-gated lighting evaluation code.

        Returns:
            WGSL code with const-gated lighting paths for each quality tier.
            naga dead-code elimination removes inactive branches.
        """
        return """\
/// Evaluate lighting based on quality tier (const-gated)
fn evaluate_lighting_gated(
    world_pos: vec3<f32>,
    N: vec3<f32>,
    V: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ao: f32
) -> vec3<f32> {
    var Lo = vec3<f32>(0.0);

    if QUALITY_HIGH {
        // High quality: Full light loop with shadows
        for (var i = 0u; i < min(uniforms.light_count, MAX_LIGHTS); i = i + 1u) {
            let light = lights[i];
            let L = get_light_direction(light, world_pos);
            let radiance = light.color * light.intensity * calculate_attenuation(light, world_pos);

            // Full shadow sampling
            let shadow = sample_shadow_pcss(world_pos, i);

            // Full BRDF evaluation
            let brdf = evaluate_brdf_full(N, V, L, base_color, metallic, roughness);

            Lo = Lo + brdf * radiance * shadow;
        }
    } else if QUALITY_MEDIUM {
        // Medium quality: Limited lights, basic shadows
        for (var i = 0u; i < min(uniforms.light_count, MAX_LIGHTS); i = i + 1u) {
            let light = lights[i];
            let L = get_light_direction(light, world_pos);
            let radiance = light.color * light.intensity * calculate_attenuation(light, world_pos);

            // Basic shadow sampling
            var shadow = 1.0;
            if SHADOWS_ENABLED {
                shadow = sample_shadow_basic(world_pos, i);
            }

            // Standard BRDF
            let brdf = evaluate_brdf(N, V, L, base_color, metallic, roughness);

            Lo = Lo + brdf * radiance * shadow;
        }
    } else {
        // Low quality: Single light, no shadows
        if (uniforms.light_count > 0u) {
            let light = lights[0];
            let L = get_light_direction(light, world_pos);
            let radiance = light.color * light.intensity;

            // Simple diffuse-only BRDF
            let NdotL = max(dot(N, L), 0.0);
            let brdf = base_color / 3.14159265359 * NdotL;

            Lo = Lo + brdf * radiance;
        }
    }

    // Ambient term (scales with quality)
    var ambient: vec3<f32>;
    if QUALITY_HIGH {
        ambient = evaluate_ambient_ibl(N, V, base_color, metallic, roughness, ao);
    } else if QUALITY_MEDIUM {
        ambient = vec3<f32>(0.03) * base_color * ao;
    } else {
        ambient = vec3<f32>(0.05) * base_color;
    }

    return ambient + Lo;
}
"""

    def generate_gated_features_code(self) -> str:
        """Generate const-gated advanced shading feature code.

        Returns:
            WGSL code for optional features (SSS, clearcoat, etc.) that
            are gated behind const bools based on quality tier.
        """
        return """\
/// Apply advanced shading features based on quality tier (const-gated)
fn apply_advanced_shading(
    base_result: vec3<f32>,
    params: PBRParams,
    N: vec3<f32>,
    V: vec3<f32>
) -> vec3<f32> {
    var result = base_result;

    if !ADVANCED_SHADING_ENABLED {
        return result;
    }

    // Subsurface scattering (HIGH quality only)
    if SUBSURFACE_ENABLED {
        if (params.subsurface > 0.0) {
            let sss = evaluate_subsurface(params.subsurface, params.subsurface_color, N, V);
            result = result + sss;
        }
    }

    // Clear coat
    if CLEARCOAT_ENABLED {
        if (params.clearcoat > 0.0) {
            let coat = evaluate_clearcoat(params.clearcoat, params.clearcoat_roughness, N, V);
            result = result + coat;
        }
    }

    // Anisotropy (HIGH quality only)
    if ANISOTROPY_ENABLED {
        if (params.anisotropy != 0.0) {
            // Anisotropic modification applied during BRDF evaluation
            // This is a placeholder for the anisotropy contribution
        }
    }

    // Sheen (HIGH quality only)
    if SHEEN_ENABLED {
        if (params.sheen > 0.0) {
            let sheen_lobe = evaluate_sheen(params.sheen, params.sheen_color, N, V);
            result = result + sheen_lobe;
        }
    }

    // Transmission
    if TRANSMISSION_ENABLED {
        if (params.transmission > 0.0) {
            let transmitted = evaluate_transmission(params.transmission, params.ior, N, V);
            result = mix(result, transmitted, params.transmission);
        }
    }

    // Iridescence (HIGH quality only)
    if IRIDESCENCE_ENABLED {
        // Iridescence modifies base_color during BRDF evaluation
        // This is handled in the main shading path
    }

    return result;
}
"""

    def generate_blend_handling_code(self) -> str:
        """Generate const-gated blend mode handling code.

        Returns:
            WGSL code for blend mode-specific behavior (alpha test, etc.)
        """
        return """\
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

    def generate_domain_handling_code(self) -> str:
        """Generate const-gated domain-specific code.

        Returns:
            WGSL code for domain-specific shading paths.
        """
        return """\
/// Domain-specific fragment output (const-gated)
fn compute_domain_output(input: PBRInput, params: PBRParams) -> vec4<f32> {
    if DOMAIN_SURFACE {
        // Full PBR surface shading
        let N = normalize(input.world_normal + params.normal - vec3<f32>(0.0, 0.0, 1.0));
        let V = normalize(uniforms.camera_position - input.world_position);

        var result = evaluate_lighting_gated(
            input.world_position,
            N,
            V,
            params.base_color,
            params.metallic,
            params.roughness,
            params.ambient_occlusion
        );

        // Apply advanced shading features
        result = apply_advanced_shading(result, params, N, V);

        // Add emissive
        result = result + params.emissive;

        return vec4<f32>(result, params.alpha);
    }

    if DOMAIN_DEFERRED_DECAL {
        // Deferred decal: output to G-buffer channels only
        // No lighting evaluation, just normal + color modification
        return vec4<f32>(params.base_color, params.alpha);
    }

    if DOMAIN_VOLUME {
        // Volume: single-scattering approximation
        // Simplified output for volumetric rendering
        return vec4<f32>(params.base_color, params.alpha);
    }

    if DOMAIN_POST_PROCESS {
        // Post-process: fullscreen effect, no PBR
        return vec4<f32>(params.base_color, 1.0);
    }

    if DOMAIN_UI {
        // UI: unlit, vertex-color only
        return vec4<f32>(params.base_color * input.vertex_color.rgb, params.alpha * input.vertex_color.a);
    }

    // Fallback
    return vec4<f32>(params.base_color, params.alpha);
}
"""

    def _lighting_enabled(self) -> bool:
        """Check if lighting is enabled for this domain."""
        return self.domain in (MaterialDomain.SURFACE,)

    def _depth_write_enabled(self) -> bool:
        """Check if depth write is enabled for this blend mode."""
        return self.blend not in (BlendMode.TRANSLUCENT, BlendMode.ADDITIVE, BlendMode.MODULATE)

    def _pbr_enabled(self) -> bool:
        """Check if PBR shading is enabled for this domain."""
        return self.domain == MaterialDomain.SURFACE

    def get_variant_key(self) -> int:
        """Compute a unique hash key for this variant configuration.

        Returns:
            Integer hash combining domain, blend, quality, and feature flags.
        """
        # Build a canonical string representation
        parts = [
            f"domain:{self.domain.value}",
            f"blend:{self.blend.value}",
            f"quality:{self.quality.value}",
        ]
        for flag in sorted(self.feature_flags):
            parts.append(f"flag:{flag}")

        return hash("|".join(parts))

    def copy_with(
        self,
        domain: Optional[MaterialDomain] = None,
        blend: Optional[BlendMode] = None,
        quality: Optional[QualityTier] = None,
        feature_flags: Optional[Set[str]] = None,
    ) -> VariantConfig:
        """Create a copy with optionally modified fields.

        Args:
            domain: New domain value, or None to keep current
            blend: New blend mode, or None to keep current
            quality: New quality tier, or None to keep current
            feature_flags: New feature flags, or None to keep current

        Returns:
            New VariantConfig with specified modifications.
        """
        return VariantConfig(
            domain=domain if domain is not None else self.domain,
            blend=blend if blend is not None else self.blend,
            quality=quality if quality is not None else self.quality,
            feature_flags=feature_flags if feature_flags is not None else self.feature_flags.copy(),
        )


class VariantCompiler:
    """Compiles material shaders with variant const declarations.

    Integrates with MaterialCompiler to inject variant consts and
    quality-gated code into the shader output.
    """

    def __init__(self, config: VariantConfig):
        """Initialize with a variant configuration.

        Args:
            config: VariantConfig specifying domain, blend, and quality.
        """
        self.config = config

    def inject_variant_consts(self, wgsl: str) -> str:
        """Inject variant const declarations into shader source.

        Args:
            wgsl: Original WGSL shader source.

        Returns:
            WGSL with variant consts injected after the header.
        """
        const_decls = self.config.generate_const_declarations()

        # Find insertion point after header comment
        lines = wgsl.split("\n")
        insert_idx = 0

        for i, line in enumerate(lines):
            # Skip header comments
            if line.startswith("//") or line.strip() == "":
                insert_idx = i + 1
            else:
                break

        # Insert const declarations
        lines.insert(insert_idx, "")
        lines.insert(insert_idx + 1, const_decls)

        return "\n".join(lines)

    def inject_gated_functions(self, wgsl: str) -> str:
        """Inject quality-gated function implementations.

        Args:
            wgsl: WGSL shader source with variant consts.

        Returns:
            WGSL with gated functions injected before fragment main.
        """
        gated_code = "\n".join([
            "// =============================================================================",
            "// Quality-Gated Functions",
            "// =============================================================================",
            "",
            self.config.generate_gated_lighting_code(),
            self.config.generate_gated_features_code(),
            self.config.generate_blend_handling_code(),
            self.config.generate_domain_handling_code(),
        ])

        # Find @fragment decorator
        lines = wgsl.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("@fragment"):
                # Insert before @fragment
                lines.insert(i, "")
                lines.insert(i + 1, gated_code)
                lines.insert(i + 2, "")
                break

        return "\n".join(lines)


def generate_all_variant_combinations() -> List[VariantConfig]:
    """Generate all possible variant configurations.

    Returns:
        List of VariantConfig for all domain x blend x quality combinations.
        Total: 5 domains x 5 blends x 3 qualities = 75 variants.
    """
    configs = []
    for domain in MaterialDomain:
        for blend in BlendMode:
            for quality in QualityTier:
                configs.append(VariantConfig(
                    domain=domain,
                    blend=blend,
                    quality=quality,
                ))
    return configs


def get_variant_for_material_system(
    domain_str: str,
    blend_str: str,
    quality_str: str = "high",
) -> VariantConfig:
    """Create VariantConfig from string enum values.

    This helper is for integration with the engine's MaterialSystem which
    uses string-based enum values.

    Args:
        domain_str: Domain value string (e.g., "surface")
        blend_str: Blend mode value string (e.g., "opaque")
        quality_str: Quality tier value string (e.g., "high")

    Returns:
        Corresponding VariantConfig.

    Raises:
        ValueError: If any string doesn't match a valid enum value.
    """
    # Map strings to enums
    domain_map = {d.value: d for d in MaterialDomain}
    blend_map = {b.value: b for b in BlendMode}
    quality_map = {q.value: q for q in QualityTier}

    if domain_str not in domain_map:
        raise ValueError(f"Invalid domain: {domain_str}")
    if blend_str not in blend_map:
        raise ValueError(f"Invalid blend mode: {blend_str}")
    if quality_str not in quality_map:
        raise ValueError(f"Invalid quality tier: {quality_str}")

    return VariantConfig(
        domain=domain_map[domain_str],
        blend=blend_map[blend_str],
        quality=quality_map[quality_str],
    )


__all__ = [
    "MaterialDomain",
    "BlendMode",
    "QualityTier",
    "VariantConfig",
    "VariantCompiler",
    "generate_all_variant_combinations",
    "get_variant_for_material_system",
]
