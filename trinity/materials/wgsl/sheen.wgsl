// Sheen BRDF Functions for TRINITY Material System
// T-MAT-4.4: Microfiber Retro-Reflection Sheen Lobe
//
// This module provides sheen BRDF functions for fabric/cloth rendering:
//   - Charlie/Ashikhmin sheen distribution (D_Charlie)
//   - Neubelt visibility term (V_Neubelt)
//   - Combined sheen evaluation with tinting
//
// Sheen models the soft retro-reflective properties of fabrics where
// microfibers create a characteristic brightening at grazing angles.
// Unlike the Fresnel effect, sheen is strongest at grazing and weakens
// toward normal incidence.
//
// References:
//   - Estevez & Kulla "Production Friendly Microfacet Sheen BRDF"
//   - Neubelt & Pettineo "Crafting a Next-Gen Material Pipeline for The Order: 1886"
//   - Conty & Kulla "Revisiting Physically Based Shading at Imageworks"

// Mathematical constants (shared with brdf.wgsl if included together)
#ifndef PI
const PI: f32 = 3.14159265359;
#endif
#ifndef INV_PI
const INV_PI: f32 = 0.31830988618;
#endif
#ifndef EPSILON
const EPSILON: f32 = 0.0001;
#endif

// ============================================================================
// Quality Tier Gating
// ============================================================================

/// Quality gate for sheen evaluation.
/// Set to false to compile out all sheen code (naga dead-code elimination).
/// This is overridden by the variant system for LOW/MEDIUM quality tiers.
const QUALITY_SHEEN_ENABLED: bool = true;

// ============================================================================
// Sheen Parameter Struct
// ============================================================================

/// Sheen material parameters.
/// Encapsulates all parameters needed for sheen evaluation.
struct SheenParams {
    /// Sheen intensity: 0.0 = no sheen, 1.0 = full sheen
    intensity: f32,
    /// Sheen tint color (linear RGB)
    /// Typically set to match fabric color or left white
    color: vec3<f32>,
    /// Sheen roughness: controls width of the sheen lobe
    /// Lower values = sharper sheen, higher = softer/wider
    /// Typical range: 0.1 - 0.6
    roughness: f32,
}

/// Create default sheen parameters.
/// Returns white sheen with medium intensity and roughness.
fn sheen_params_default() -> SheenParams {
    var params: SheenParams;
    params.intensity = 0.5;
    params.color = vec3<f32>(1.0, 1.0, 1.0);
    params.roughness = 0.3;
    return params;
}

// ============================================================================
// Sheen Distribution Function (NDF)
// ============================================================================

/// Charlie/Ashikhmin sheen distribution function.
/// Models the distribution of microfiber orientations for fabric surfaces.
///
/// The Charlie distribution uses sin(theta)^(1/roughness - 1) to create
/// a distribution that peaks at grazing angles rather than normal incidence.
/// This models the way microfibers on fabric surfaces tend to orient
/// perpendicular to the surface.
///
/// @param NoH: Dot product of surface normal and half-vector, clamped to [0,1]
/// @param roughness: Sheen roughness in [0,1], controls lobe width
/// @returns: NDF value (probability density of microfiber orientation)
fn D_Charlie(NoH: f32, roughness: f32) -> f32 {
    // Clamp roughness to avoid division by zero
    let alpha = max(roughness * roughness, 0.0001);

    // Charlie distribution: (1 + (1/alpha - 1) * (1 - NoH^2))^(-1/(2*alpha))
    // Simplified form using sin(theta) = sqrt(1 - cos^2(theta))
    let sin2_theta = 1.0 - NoH * NoH;

    // Inverted Ashikhmin distribution for sheen
    // D = (2 + 1/alpha) * (1 - NoH^2)^(1/(2*alpha)) / (2*PI)
    let inv_alpha = 1.0 / alpha;
    let power = inv_alpha * 0.5;

    // Use sin^n where n = 1/alpha
    // For small roughness, this creates a narrow peak at grazing angles
    let D = (2.0 + inv_alpha) * pow(max(sin2_theta, 0.0), power) * INV_PI * 0.5;

    return D;
}

/// Simplified Charlie distribution using the exponential form.
/// This variant is slightly cheaper but less accurate.
///
/// @param NoH: Dot product of surface normal and half-vector
/// @param roughness: Sheen roughness
/// @returns: Simplified NDF value
fn D_Charlie_Simple(NoH: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let sin2_theta = 1.0 - NoH * NoH;

    // Simplified: sin^(2/alpha) * normalization
    let D = pow(sin2_theta, 1.0 / alpha) * INV_PI;

    return D;
}

// ============================================================================
// Sheen Visibility Function (GSF)
// ============================================================================

/// Neubelt visibility term for sheen.
/// A simplified visibility function designed specifically for sheen BRDFs.
///
/// The Neubelt visibility term avoids the complexity of the Smith masking
/// function while providing a reasonable approximation for sheen materials.
/// It produces V = 1 / (4 * (NoL + NoV - NoL * NoV))
///
/// This term:
/// - Goes to 0.25 at normal incidence (NoL=NoV=1)
/// - Increases toward grazing angles
/// - Is bounded and well-behaved across all angles
///
/// @param NoV: Dot product of surface normal and view direction
/// @param NoL: Dot product of surface normal and light direction
/// @returns: Visibility term (includes 1/(4*NoL*NoV) normalization)
fn V_Neubelt(NoV: f32, NoL: f32) -> f32 {
    // Neubelt visibility: 1 / (4 * (NoL + NoV - NoL*NoV))
    // This simplifies the denominator compared to Smith G2
    let denom = NoL + NoV - NoL * NoV;
    return 1.0 / (4.0 * denom + EPSILON);
}

/// Alternative Ashikhmin visibility term.
/// Even simpler than Neubelt, just uses 1/(4*NoL*NoV).
///
/// @param NoV: Dot product of surface normal and view direction
/// @param NoL: Dot product of surface normal and light direction
/// @returns: Simple visibility term
fn V_Ashikhmin(NoV: f32, NoL: f32) -> f32 {
    return 1.0 / (4.0 * (NoL + NoV + EPSILON));
}

// ============================================================================
// Combined Sheen Evaluation
// ============================================================================

/// Evaluate the complete sheen BRDF contribution.
/// Combines the Charlie distribution and Neubelt visibility with tinting.
///
/// Sheen does NOT use Fresnel - it's a diffuse-like scattering effect
/// that occurs due to microfiber geometry, not index of refraction.
///
/// @param params: Sheen material parameters
/// @param N: Surface normal (normalized)
/// @param V: View direction (normalized, pointing toward camera)
/// @param L: Light direction (normalized, pointing toward light)
/// @returns: Sheen BRDF contribution (RGB)
fn evaluate_sheen(
    params: SheenParams,
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>
) -> vec3<f32> {
    // Early exit if sheen is disabled at compile time
    if !QUALITY_SHEEN_ENABLED {
        return vec3<f32>(0.0);
    }

    // Early exit if sheen intensity is zero
    if params.intensity < EPSILON {
        return vec3<f32>(0.0);
    }

    // Compute half-vector
    let H = normalize(V + L);

    // Compute dot products (clamped to avoid negative values)
    let NoL = max(dot(N, L), 0.0);
    let NoV = max(dot(N, V), 0.0);
    let NoH = max(dot(N, H), 0.0);

    // Early exit for grazing angles where both NoL and NoV are near zero
    if NoL < EPSILON && NoV < EPSILON {
        return vec3<f32>(0.0);
    }

    // Evaluate sheen BRDF: D * V * color * intensity
    // Note: No Fresnel term for sheen
    let D = D_Charlie(NoH, params.roughness);
    let Vis = V_Neubelt(NoV, NoL);

    // Sheen contribution with color tinting
    let sheen = D * Vis * params.color * params.intensity;

    return sheen;
}

/// Evaluate sheen with NoL factor already applied.
/// Use this when accumulating light contributions.
///
/// @param params: Sheen material parameters
/// @param N: Surface normal (normalized)
/// @param V: View direction (normalized)
/// @param L: Light direction (normalized)
/// @returns: Sheen BRDF value multiplied by NoL
fn evaluate_sheen_with_NoL(
    params: SheenParams,
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>
) -> vec3<f32> {
    let sheen = evaluate_sheen(params, N, V, L);
    let NoL = max(dot(N, L), 0.0);
    return sheen * NoL;
}

// ============================================================================
// Integration with PBR BRDF
// ============================================================================

/// Evaluate complete PBR BRDF with sheen lobe.
/// This combines diffuse + specular + sheen for full fabric rendering.
///
/// The sheen lobe is additive to the standard BRDF:
///   final = diffuse + specular + sheen
///
/// For energy conservation, the sheen contribution should ideally
/// reduce the diffuse component, but this simplified version treats
/// sheen as a purely additive term.
///
/// @param pbr_params: PBR material parameters (base_color, roughness, metallic)
/// @param sheen_params: Sheen parameters (intensity, color, roughness)
/// @param N: Surface normal (normalized)
/// @param V: View direction (normalized)
/// @param L: Light direction (normalized)
/// @param diffuse: Evaluated diffuse BRDF contribution
/// @param specular: Evaluated specular BRDF contribution
/// @returns: Combined BRDF contribution (RGB)
fn combine_brdf_with_sheen(
    sheen_params: SheenParams,
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    diffuse: vec3<f32>,
    specular: vec3<f32>
) -> vec3<f32> {
    // Evaluate sheen contribution
    let sheen = evaluate_sheen(sheen_params, N, V, L);

    // Combine all lobes
    // For more accurate energy conservation, scale diffuse by (1 - sheen_intensity)
    // but this adds complexity and the visual difference is subtle
    return diffuse + specular + sheen;
}

/// Get sheen contribution at a specific angle for debugging/visualization.
/// Returns just the sheen value without combining with other BRDF lobes.
///
/// @param intensity: Sheen intensity [0,1]
/// @param color: Sheen color (linear RGB)
/// @param roughness: Sheen roughness [0,1]
/// @param NoV: Dot product of normal and view direction
/// @param NoL: Dot product of normal and light direction
/// @param NoH: Dot product of normal and half-vector
/// @returns: Sheen contribution (RGB)
fn sheen_contribution(
    intensity: f32,
    color: vec3<f32>,
    roughness: f32,
    NoV: f32,
    NoL: f32,
    NoH: f32
) -> vec3<f32> {
    if !QUALITY_SHEEN_ENABLED {
        return vec3<f32>(0.0);
    }

    if intensity < EPSILON {
        return vec3<f32>(0.0);
    }

    let D = D_Charlie(NoH, roughness);
    let Vis = V_Neubelt(NoV, NoL);

    return D * Vis * color * intensity;
}
