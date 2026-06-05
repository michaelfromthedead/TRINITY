"""Quality tier variant logic for LOW/MEDIUM/HIGH rendering quality.

This module implements quality-specific shader features that complement the
variant const system. Each quality tier has:

1. QualityFeatures: Feature set configuration (lights, shadows, effects)
2. QualityShaderCode: Tier-specific WGSL implementations

The quality system enables scalable rendering:
- LOW: Single light, no shadows, basic Lambert diffuse
- MEDIUM: 4 lights, basic shadow mapping, standard PBR
- HIGH: 16 lights, PCSS shadows, full advanced shading

Task: T-MAT-2.4 Quality Tier Variants
Gap: S3-G3, S3-G9 (CRITICAL + HIGH)
Dependency: T-MAT-2.1 (variant const system)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from trinity.materials.variants import QualityTier


@dataclass(frozen=True, slots=True)
class QualityFeatures:
    """Feature set enabled at each quality tier.

    This dataclass encapsulates the rendering capabilities available
    at each quality level. It provides a structured way to query
    quality-dependent features for both runtime decisions and shader
    generation.

    Attributes:
        max_lights: Maximum number of lights to evaluate.
        shadow_quality: Shadow algorithm ("none", "basic", "pcss").
        subsurface: Whether subsurface scattering is enabled.
        clearcoat: Whether clearcoat layer is enabled.
        anisotropy: Whether anisotropic reflections are enabled.
        screen_space_reflections: Whether SSR is enabled.
        ambient_occlusion: AO algorithm ("none", "ssao", "hbao").
        iridescence: Whether thin-film iridescence is enabled.
        sheen: Whether sheen (fabric) shading is enabled.
        transmission: Whether light transmission is enabled.
    """

    max_lights: int
    shadow_quality: str  # "none", "basic", "pcss"
    subsurface: bool
    clearcoat: bool
    anisotropy: bool
    screen_space_reflections: bool
    ambient_occlusion: str  # "none", "ssao", "hbao"
    iridescence: bool = False
    sheen: bool = False
    transmission: bool = False

    @classmethod
    def for_tier(cls, tier: QualityTier) -> "QualityFeatures":
        """Create QualityFeatures for a specific quality tier.

        Args:
            tier: The quality tier (LOW, MEDIUM, HIGH).

        Returns:
            QualityFeatures configured for the specified tier.

        Example::

            features = QualityFeatures.for_tier(QualityTier.HIGH)
            if features.subsurface:
                # Enable subsurface scattering pass
                pass
        """
        if tier == QualityTier.LOW:
            return cls(
                max_lights=1,
                shadow_quality="none",
                subsurface=False,
                clearcoat=False,
                anisotropy=False,
                screen_space_reflections=False,
                ambient_occlusion="none",
                iridescence=False,
                sheen=False,
                transmission=False,
            )
        elif tier == QualityTier.MEDIUM:
            return cls(
                max_lights=4,
                shadow_quality="basic",
                subsurface=False,
                clearcoat=True,
                anisotropy=False,
                screen_space_reflections=False,
                ambient_occlusion="ssao",
                iridescence=False,
                sheen=False,
                transmission=True,
            )
        else:  # HIGH
            return cls(
                max_lights=16,
                shadow_quality="pcss",
                subsurface=True,
                clearcoat=True,
                anisotropy=True,
                screen_space_reflections=True,
                ambient_occlusion="hbao",
                iridescence=True,
                sheen=True,
                transmission=True,
            )

    def to_dict(self) -> Dict[str, any]:
        """Convert features to dictionary for serialization.

        Returns:
            Dictionary with all feature settings.
        """
        return {
            "max_lights": self.max_lights,
            "shadow_quality": self.shadow_quality,
            "subsurface": self.subsurface,
            "clearcoat": self.clearcoat,
            "anisotropy": self.anisotropy,
            "screen_space_reflections": self.screen_space_reflections,
            "ambient_occlusion": self.ambient_occlusion,
            "iridescence": self.iridescence,
            "sheen": self.sheen,
            "transmission": self.transmission,
        }

    @property
    def has_shadows(self) -> bool:
        """Check if shadows are enabled for this tier."""
        return self.shadow_quality != "none"

    @property
    def has_advanced_shading(self) -> bool:
        """Check if any advanced shading features are enabled."""
        return self.subsurface or self.clearcoat or self.anisotropy or self.sheen

    @property
    def complexity_score(self) -> int:
        """Calculate a complexity score for this feature set.

        Higher scores indicate more GPU-intensive rendering.

        Returns:
            Integer complexity score (roughly correlates with shader ops).
        """
        score = self.max_lights * 10  # Base lighting cost
        if self.shadow_quality == "basic":
            score += 50
        elif self.shadow_quality == "pcss":
            score += 150
        if self.subsurface:
            score += 80
        if self.clearcoat:
            score += 30
        if self.anisotropy:
            score += 40
        if self.screen_space_reflections:
            score += 200
        if self.ambient_occlusion == "ssao":
            score += 60
        elif self.ambient_occlusion == "hbao":
            score += 100
        if self.iridescence:
            score += 50
        if self.sheen:
            score += 25
        if self.transmission:
            score += 70
        return score


class QualityShaderCode:
    """Quality-specific shader implementations.

    This class provides WGSL code snippets that vary by quality tier.
    The code is designed to work with the variant const system, where
    naga dead-code elimination removes unused paths at compile time.

    The implementations prioritize:
    - LOW: Simplicity and performance
    - MEDIUM: Balance of quality and performance
    - HIGH: Maximum visual fidelity

    Example::

        tier = QualityTier.HIGH
        shadow_code = QualityShaderCode.get_shadow_code(tier)
        brdf_code = QualityShaderCode.get_brdf_code(tier)
    """

    # =========================================================================
    # Shadow Sampling Implementations
    # =========================================================================

    SHADOW_NONE = """\
/// No shadow sampling - always returns fully lit (LOW quality)
fn sample_shadow(light_idx: u32, world_pos: vec3<f32>) -> f32 {
    return 1.0;  // No shadows at LOW quality
}
"""

    SHADOW_BASIC = """\
/// Basic shadow map sampling with bias (MEDIUM quality)
const SHADOW_BIAS: f32 = 0.005;

fn sample_shadow(light_idx: u32, world_pos: vec3<f32>) -> f32 {
    let shadow_coord = light_matrices[light_idx] * vec4<f32>(world_pos, 1.0);
    let proj_coord = shadow_coord.xyz / shadow_coord.w;

    // Transform to [0,1] range for texture lookup
    let uv = proj_coord.xy * 0.5 + 0.5;

    // Skip if outside shadow map bounds
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }

    let depth = textureSample(shadow_maps, shadow_sampler, uv, light_idx).r;
    return select(0.0, 1.0, proj_coord.z - SHADOW_BIAS < depth);
}
"""

    SHADOW_PCSS = """\
/// PCSS (Percentage-Closer Soft Shadows) implementation (HIGH quality)
const SHADOW_BIAS: f32 = 0.002;
const LIGHT_SIZE: f32 = 0.02;  // World-space light source size
const BLOCKER_SEARCH_SAMPLES: i32 = 16;
const PCF_SAMPLES: i32 = 32;

/// Poisson disk sampling points for blocker search
const POISSON_DISK: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>(0.94558609, -0.76890725),
    vec2<f32>(-0.094184101, -0.92938870),
    vec2<f32>(0.34495938, 0.29387760),
    vec2<f32>(-0.91588581, 0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543, 0.27676845),
    vec2<f32>(0.97484398, 0.75648379),
    vec2<f32>(0.44323325, -0.97511554),
    vec2<f32>(0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>(0.79197514, 0.19090188),
    vec2<f32>(-0.24188840, 0.99706507),
    vec2<f32>(-0.81409955, 0.91437590),
    vec2<f32>(0.19984126, 0.78641367),
    vec2<f32>(0.14383161, -0.14100790)
);

/// Find average blocker depth
fn find_blocker(shadow_coord: vec4<f32>, light_idx: u32) -> f32 {
    let proj_coord = shadow_coord.xyz / shadow_coord.w;
    let uv = proj_coord.xy * 0.5 + 0.5;
    let receiver_depth = proj_coord.z;

    var blocker_sum = 0.0;
    var blocker_count = 0.0;
    let search_radius = LIGHT_SIZE * receiver_depth;

    for (var i = 0; i < BLOCKER_SEARCH_SAMPLES; i = i + 1) {
        let offset = POISSON_DISK[i] * search_radius;
        let sample_uv = uv + offset;
        let depth = textureSample(shadow_maps, shadow_sampler, sample_uv, light_idx).r;

        if (depth < receiver_depth - SHADOW_BIAS) {
            blocker_sum = blocker_sum + depth;
            blocker_count = blocker_count + 1.0;
        }
    }

    if (blocker_count > 0.0) {
        return blocker_sum / blocker_count;
    }
    return -1.0;  // No blockers found
}

/// Estimate penumbra size based on blocker distance
fn estimate_penumbra(blocker_depth: f32, receiver_depth: f32) -> f32 {
    if (blocker_depth < 0.0) {
        return 0.0;  // No blocker, no shadow
    }
    return LIGHT_SIZE * (receiver_depth - blocker_depth) / blocker_depth;
}

/// PCF filter with variable kernel size
fn pcf_filter(shadow_coord: vec4<f32>, light_idx: u32, penumbra: f32) -> f32 {
    let proj_coord = shadow_coord.xyz / shadow_coord.w;
    let uv = proj_coord.xy * 0.5 + 0.5;
    let receiver_depth = proj_coord.z;

    var shadow = 0.0;
    let filter_radius = max(penumbra, 0.001);  // Minimum radius for hard shadows

    for (var i = 0; i < PCF_SAMPLES; i = i + 1) {
        let offset = POISSON_DISK[i % 16] * filter_radius;
        let sample_uv = uv + offset;
        let depth = textureSample(shadow_maps, shadow_sampler, sample_uv, light_idx).r;
        shadow = shadow + select(0.0, 1.0, receiver_depth - SHADOW_BIAS < depth);
    }

    return shadow / f32(PCF_SAMPLES);
}

fn sample_shadow(light_idx: u32, world_pos: vec3<f32>) -> f32 {
    let shadow_coord = light_matrices[light_idx] * vec4<f32>(world_pos, 1.0);
    let proj_coord = shadow_coord.xyz / shadow_coord.w;

    // Skip if outside shadow map bounds
    let uv = proj_coord.xy * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }

    // PCSS: find blocker, estimate penumbra, PCF with variable kernel
    let blocker = find_blocker(shadow_coord, light_idx);
    if (blocker < 0.0) {
        return 1.0;  // No blocker found - fully lit
    }
    let penumbra = estimate_penumbra(blocker, proj_coord.z);
    return pcf_filter(shadow_coord, light_idx, penumbra);
}
"""

    # =========================================================================
    # BRDF Implementations
    # =========================================================================

    BRDF_LOW = """\
/// Simplified diffuse-only BRDF (LOW quality)
fn evaluate_brdf_simple(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32
) -> vec3<f32> {
    // Simple Lambert diffuse - no specular for LOW quality
    let NdotL = max(dot(N, L), 0.0);
    return base_color / 3.14159265359 * NdotL;
}
"""

    BRDF_MEDIUM = """\
/// Standard Cook-Torrance BRDF (MEDIUM quality)
fn evaluate_brdf_standard(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32
) -> vec3<f32> {
    let H = normalize(V + L);
    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.0);
    let NdotH = max(dot(N, H), 0.0);
    let HdotV = max(dot(H, V), 0.0);

    // Fresnel reflectance at normal incidence
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);

    // Fresnel-Schlick
    let F = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - HdotV, 5.0);

    // GGX Distribution
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    let D = a2 / (3.14159265359 * denom * denom);

    // Smith's geometry function
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let G1_V = NdotV / (NdotV * (1.0 - k) + k);
    let G1_L = NdotL / (NdotL * (1.0 - k) + k);
    let G = G1_V * G1_L;

    // Specular
    let specular = D * G * F / (4.0 * NdotV * NdotL + 0.0001);

    // Diffuse (energy conserving)
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let diffuse = kD * base_color / 3.14159265359;

    return (diffuse + specular) * NdotL;
}
"""

    BRDF_HIGH = """\
/// Full multi-lobe BRDF with advanced features (HIGH quality)

/// Fresnel-Schlick with roughness
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(1.0 - cos_theta, 5.0);
}

/// GGX with anisotropic distribution
fn distribution_ggx_aniso(
    N: vec3<f32>,
    H: vec3<f32>,
    T: vec3<f32>,
    B: vec3<f32>,
    ax: f32,
    ay: f32
) -> f32 {
    let NdotH = dot(N, H);
    let TdotH = dot(T, H);
    let BdotH = dot(B, H);

    let d = (TdotH * TdotH) / (ax * ax) + (BdotH * BdotH) / (ay * ay) + NdotH * NdotH;
    return 1.0 / (3.14159265359 * ax * ay * d * d);
}

/// Smith's geometry with anisotropic roughness
fn geometry_smith_aniso(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    T: vec3<f32>,
    B: vec3<f32>,
    ax: f32,
    ay: f32
) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let TdotV = dot(T, V);
    let BdotV = dot(B, V);
    let TdotL = dot(T, L);
    let BdotL = dot(B, L);

    let lambda_V = NdotL * sqrt(ax * ax * TdotV * TdotV + ay * ay * BdotV * BdotV + NdotV * NdotV);
    let lambda_L = NdotV * sqrt(ax * ax * TdotL * TdotL + ay * ay * BdotL * BdotL + NdotL * NdotL);

    return 0.5 / (lambda_V + lambda_L + 0.0001);
}

/// Charlie sheen distribution (for fabric)
fn distribution_charlie(NdotH: f32, roughness: f32) -> f32 {
    let invAlpha = 1.0 / roughness;
    let cos2h = NdotH * NdotH;
    let sin2h = max(1.0 - cos2h, 0.0078125);
    return (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / (2.0 * 3.14159265359);
}

/// Sheen BRDF lobe
fn evaluate_sheen(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    sheen_color: vec3<f32>,
    sheen_roughness: f32
) -> vec3<f32> {
    let H = normalize(V + L);
    let NdotH = max(dot(N, H), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.0);

    let D = distribution_charlie(NdotH, sheen_roughness);
    // Simplified visibility for sheen
    let V_sheen = 1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV) + 0.0001);

    return sheen_color * D * V_sheen * NdotL;
}

/// Clearcoat BRDF lobe (GGX with fixed IOR 1.5)
fn evaluate_clearcoat(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    clearcoat: f32,
    clearcoat_roughness: f32
) -> vec3<f32> {
    let H = normalize(V + L);
    let NdotH = max(dot(N, H), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.0);
    let HdotV = max(dot(H, V), 0.0);

    // Fixed F0 for polyurethane clearcoat (IOR 1.5)
    let F0 = 0.04;
    let F = F0 + (1.0 - F0) * pow(1.0 - HdotV, 5.0);

    // GGX distribution
    let a = clearcoat_roughness * clearcoat_roughness;
    let a2 = a * a;
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    let D = a2 / (3.14159265359 * denom * denom);

    // Kelemen visibility
    let V_cc = 0.25 / (HdotV * HdotV + 0.0001);

    return vec3<f32>(clearcoat * F * D * V_cc * NdotL);
}

fn evaluate_brdf_full(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32
) -> vec3<f32> {
    let H = normalize(V + L);
    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.0);
    let NdotH = max(dot(N, H), 0.0);
    let HdotV = max(dot(H, V), 0.0);

    // Fresnel reflectance at normal incidence
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);

    // Fresnel with roughness consideration
    let F = fresnel_schlick_roughness(HdotV, f0, roughness);

    // GGX Distribution (isotropic path for standard materials)
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    let D = a2 / (3.14159265359 * denom * denom);

    // Smith's geometry function with height-correlated masking
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let G1_V = NdotV / (NdotV * (1.0 - k) + k);
    let G1_L = NdotL / (NdotL * (1.0 - k) + k);
    let G = G1_V * G1_L;

    // Specular BRDF
    let specular = D * G * F / (4.0 * NdotV * NdotL + 0.0001);

    // Diffuse with energy conservation
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let diffuse = kD * base_color / 3.14159265359;

    return (diffuse + specular) * NdotL;
}
"""

    # =========================================================================
    # Ambient Occlusion Implementations
    # =========================================================================

    AO_NONE = """\
/// No ambient occlusion (LOW quality)
fn sample_ambient_occlusion(uv: vec2<f32>, world_pos: vec3<f32>, N: vec3<f32>) -> f32 {
    return 1.0;  // No AO at LOW quality
}
"""

    AO_SSAO = """\
/// Screen-space ambient occlusion (MEDIUM quality)
const SSAO_SAMPLES: i32 = 16;
const SSAO_RADIUS: f32 = 0.5;
const SSAO_BIAS: f32 = 0.025;

fn sample_ambient_occlusion(uv: vec2<f32>, world_pos: vec3<f32>, N: vec3<f32>) -> f32 {
    var occlusion = 0.0;

    // Hemisphere sampling kernel
    for (var i = 0; i < SSAO_SAMPLES; i = i + 1) {
        // Generate sample in tangent space
        let sample_offset = get_ssao_kernel(i) * SSAO_RADIUS;

        // Transform to world space
        let sample_pos = world_pos + N * sample_offset.z + sample_offset.xy;

        // Project to screen space
        let proj_pos = uniforms.view_projection * vec4<f32>(sample_pos, 1.0);
        let sample_uv = proj_pos.xy / proj_pos.w * 0.5 + 0.5;

        // Sample depth
        let sample_depth = textureSample(depth_texture, depth_sampler, sample_uv).r;
        let range_check = smoothstep(0.0, 1.0, SSAO_RADIUS / abs(proj_pos.z - sample_depth));

        occlusion = occlusion + select(0.0, 1.0, sample_depth >= proj_pos.z + SSAO_BIAS) * range_check;
    }

    return 1.0 - (occlusion / f32(SSAO_SAMPLES));
}

fn get_ssao_kernel(idx: i32) -> vec3<f32> {
    // Pre-computed hemisphere kernel (simplified)
    let phi = f32(idx) * 2.399963229728653;  // Golden angle
    let r = sqrt(f32(idx) / f32(SSAO_SAMPLES));
    return vec3<f32>(cos(phi) * r, sin(phi) * r, 1.0 - r * r);
}
"""

    AO_HBAO = """\
/// Horizon-based ambient occlusion (HIGH quality)
const HBAO_DIRECTIONS: i32 = 8;
const HBAO_STEPS: i32 = 4;
const HBAO_RADIUS: f32 = 0.5;
const HBAO_ANGLE_BIAS: f32 = 0.1;
const HBAO_INTENSITY: f32 = 1.5;

fn sample_ambient_occlusion(uv: vec2<f32>, world_pos: vec3<f32>, N: vec3<f32>) -> f32 {
    var occlusion = 0.0;
    let step_size = HBAO_RADIUS / f32(HBAO_STEPS);

    // March in multiple directions around the pixel
    for (var dir = 0; dir < HBAO_DIRECTIONS; dir = dir + 1) {
        let angle = f32(dir) * 3.14159265359 * 2.0 / f32(HBAO_DIRECTIONS);
        let direction = vec2<f32>(cos(angle), sin(angle));

        var max_horizon = -1.0;

        // Step along direction
        for (var step = 1; step <= HBAO_STEPS; step = step + 1) {
            let offset = direction * step_size * f32(step);
            let sample_uv = uv + offset;

            // Get sample position
            let sample_depth = textureSample(depth_texture, depth_sampler, sample_uv).r;
            let sample_pos = reconstruct_world_position(sample_uv, sample_depth);

            // Calculate horizon angle
            let diff = sample_pos - world_pos;
            let dist = length(diff);

            if (dist > 0.001 && dist < HBAO_RADIUS) {
                let horizon_angle = dot(normalize(diff), N);
                max_horizon = max(max_horizon, horizon_angle);
            }
        }

        // Accumulate occlusion based on horizon
        let horizon = acos(clamp(max_horizon, -1.0, 1.0));
        let ao = 1.0 - sin(horizon) * sin(horizon);
        occlusion = occlusion + ao;
    }

    occlusion = occlusion / f32(HBAO_DIRECTIONS);
    return pow(1.0 - occlusion * HBAO_INTENSITY, 2.0);
}

fn reconstruct_world_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let world = uniforms.inverse_view_projection * ndc;
    return world.xyz / world.w;
}
"""

    # =========================================================================
    # Subsurface Scattering
    # =========================================================================

    SUBSURFACE_NONE = """\
/// No subsurface scattering (LOW/MEDIUM quality)
fn evaluate_subsurface(
    subsurface: f32,
    subsurface_color: vec3<f32>,
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>
) -> vec3<f32> {
    return vec3<f32>(0.0);
}
"""

    SUBSURFACE_APPROX = """\
/// Subsurface scattering approximation (HIGH quality)
fn evaluate_subsurface(
    subsurface: f32,
    subsurface_color: vec3<f32>,
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>
) -> vec3<f32> {
    // Wrap lighting approximation for subsurface
    let NdotL_wrap = (dot(N, L) + subsurface) / (1.0 + subsurface);
    let wrap_diffuse = max(NdotL_wrap, 0.0);

    // View-dependent subsurface term
    let H_sss = normalize(L + N * subsurface);
    let VdotH_sss = pow(saturate(dot(V, -H_sss)), 3.0);

    // Combine forward and back scattering
    let forward_scatter = wrap_diffuse;
    let back_scatter = VdotH_sss * subsurface;

    return subsurface_color * (forward_scatter + back_scatter) * subsurface;
}
"""

    @classmethod
    def get_shadow_code(cls, tier: QualityTier) -> str:
        """Get shadow sampling implementation for a quality tier.

        Args:
            tier: Quality tier (LOW, MEDIUM, HIGH).

        Returns:
            WGSL code string for shadow sampling.
        """
        if tier == QualityTier.LOW:
            return cls.SHADOW_NONE
        elif tier == QualityTier.MEDIUM:
            return cls.SHADOW_BASIC
        return cls.SHADOW_PCSS

    @classmethod
    def get_brdf_code(cls, tier: QualityTier) -> str:
        """Get BRDF implementation for a quality tier.

        Args:
            tier: Quality tier (LOW, MEDIUM, HIGH).

        Returns:
            WGSL code string for BRDF evaluation.
        """
        if tier == QualityTier.LOW:
            return cls.BRDF_LOW
        elif tier == QualityTier.MEDIUM:
            return cls.BRDF_MEDIUM
        return cls.BRDF_HIGH

    @classmethod
    def get_ao_code(cls, tier: QualityTier) -> str:
        """Get ambient occlusion implementation for a quality tier.

        Args:
            tier: Quality tier (LOW, MEDIUM, HIGH).

        Returns:
            WGSL code string for ambient occlusion sampling.
        """
        features = QualityFeatures.for_tier(tier)
        if features.ambient_occlusion == "none":
            return cls.AO_NONE
        elif features.ambient_occlusion == "ssao":
            return cls.AO_SSAO
        return cls.AO_HBAO

    @classmethod
    def get_subsurface_code(cls, tier: QualityTier) -> str:
        """Get subsurface scattering implementation for a quality tier.

        Args:
            tier: Quality tier (LOW, MEDIUM, HIGH).

        Returns:
            WGSL code string for subsurface scattering.
        """
        features = QualityFeatures.for_tier(tier)
        if features.subsurface:
            return cls.SUBSURFACE_APPROX
        return cls.SUBSURFACE_NONE

    @classmethod
    def get_all_quality_code(cls, tier: QualityTier) -> str:
        """Get all quality-specific shader code for a tier.

        Combines shadow, BRDF, AO, and subsurface code into a single block.

        Args:
            tier: Quality tier (LOW, MEDIUM, HIGH).

        Returns:
            Complete WGSL code block with all quality-specific implementations.
        """
        header = f"""\
// =============================================================================
// Quality Tier: {tier.name}
// Generated quality-specific shader implementations
// =============================================================================

"""
        sections = [
            "// --- Shadow Sampling ---\n" + cls.get_shadow_code(tier),
            "// --- BRDF ---\n" + cls.get_brdf_code(tier),
            "// --- Ambient Occlusion ---\n" + cls.get_ao_code(tier),
            "// --- Subsurface Scattering ---\n" + cls.get_subsurface_code(tier),
        ]

        return header + "\n\n".join(sections)

    @classmethod
    def estimate_instruction_count(cls, tier: QualityTier) -> Dict[str, int]:
        """Estimate shader instruction counts for each quality tier.

        Provides rough ALU/TEX instruction estimates useful for
        performance budgeting.

        Args:
            tier: Quality tier (LOW, MEDIUM, HIGH).

        Returns:
            Dictionary with instruction estimates by category.
        """
        if tier == QualityTier.LOW:
            return {
                "shadow": 0,
                "brdf": 15,
                "ao": 0,
                "subsurface": 0,
                "total": 15,
            }
        elif tier == QualityTier.MEDIUM:
            return {
                "shadow": 25,
                "brdf": 45,
                "ao": 80,
                "subsurface": 0,
                "total": 150,
            }
        return {
            "shadow": 180,
            "brdf": 120,
            "ao": 200,
            "subsurface": 35,
            "total": 535,
        }


def get_quality_config_for_device(
    gpu_tier: str,
    vram_mb: int,
    target_fps: int = 60,
) -> QualityTier:
    """Recommend a quality tier based on device capabilities.

    Heuristic-based quality selection considering GPU tier and VRAM.

    Args:
        gpu_tier: GPU performance tier ("low", "mid", "high", "ultra").
        vram_mb: Available VRAM in megabytes.
        target_fps: Target frame rate.

    Returns:
        Recommended QualityTier.
    """
    # Map GPU tier to base quality
    tier_map = {
        "low": QualityTier.LOW,
        "mid": QualityTier.MEDIUM,
        "high": QualityTier.HIGH,
        "ultra": QualityTier.HIGH,
    }

    base_tier = tier_map.get(gpu_tier.lower(), QualityTier.MEDIUM)

    # Downgrade if VRAM is limited
    if vram_mb < 2048 and base_tier == QualityTier.HIGH:
        base_tier = QualityTier.MEDIUM
    if vram_mb < 1024 and base_tier == QualityTier.MEDIUM:
        base_tier = QualityTier.LOW

    # Downgrade for high FPS targets (e.g., VR 90Hz)
    if target_fps > 60 and base_tier == QualityTier.HIGH:
        base_tier = QualityTier.MEDIUM

    return base_tier


__all__ = [
    "QualityFeatures",
    "QualityShaderCode",
    "get_quality_config_for_device",
]
