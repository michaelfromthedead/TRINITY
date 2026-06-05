// Clear Coat BRDF Functions for TRINITY Material System
// T-MAT-4.2: Dual-Layer Clear Coat Implementation
//
// This module provides clear coat layer BRDF functions for PBR rendering:
//   - Fixed IOR 1.5 clear coat layer (F0 = 0.04)
//   - Independent roughness for clear coat
//   - GGX distribution and Smith geometry for clear coat
//   - Fresnel-weighted layer combination
//
// The clear coat model simulates automotive paint, lacquered wood, and
// other materials with a transparent protective layer over a base material.
//
// References:
//   - Karis "Real Shading in Unreal Engine 4"
//   - Enterprise PBR Shading Model (MaterialX)
//   - Filament Material Model Documentation

// Mathematical constants (may be imported from brdf.wgsl in production)
const CC_PI: f32 = 3.14159265359;
const CC_EPSILON: f32 = 0.0001;

// Clear coat layer F0 for IOR 1.5 (polyurethane/lacquer)
// Computed as: ((n-1)/(n+1))^2 = ((1.5-1)/(1.5+1))^2 = 0.04
const CLEAR_COAT_F0: f32 = 0.04;

// ============================================================================
// Clear Coat Parameters
// ============================================================================

/// Parameters controlling the clear coat layer.
/// intensity: Blend weight for the clear coat (0 = none, 1 = full)
/// roughness: Surface roughness of the clear coat layer in [0,1]
struct ClearCoatParams {
    intensity: f32,
    roughness: f32,
}

// ============================================================================
// Quality Tier Gating
// ============================================================================

/// Quality tier control for clear coat. When false, clear coat is disabled
/// and evaluate_clear_coat returns zero. This const is dead-code eliminated
/// by naga compiler in LOW quality variants.
const QUALITY_CLEAR_COAT_ENABLED: bool = true;

// ============================================================================
// Clear Coat Fresnel Function
// ============================================================================

/// Schlick Fresnel approximation for clear coat layer.
/// Uses fixed F0 = 0.04 for IOR 1.5 (polyurethane clear coat).
///
/// @param VoH: Dot product of view direction and half-vector
/// @returns: Fresnel reflectance for clear coat (scalar)
fn F_ClearCoat(VoH: f32) -> f32 {
    let Fc = pow(1.0 - VoH, 5.0);
    return CLEAR_COAT_F0 + (1.0 - CLEAR_COAT_F0) * Fc;
}

// ============================================================================
// Clear Coat NDF (Normal Distribution Function)
// ============================================================================

/// GGX/Trowbridge-Reitz NDF for clear coat layer.
/// Uses independent roughness from the base layer.
///
/// @param NoH: Dot product of surface normal and half-vector
/// @param cc_roughness: Clear coat roughness in [0,1]
/// @returns: NDF value for clear coat layer
fn D_ClearCoat(NoH: f32, cc_roughness: f32) -> f32 {
    // Apply Disney roughness remapping: a = roughness^2
    let a = cc_roughness * cc_roughness;
    let a2 = a * a;
    let NoH2 = NoH * NoH;

    // GGX formula
    let denom = NoH2 * (a2 - 1.0) + 1.0;
    return a2 / (CC_PI * denom * denom + CC_EPSILON);
}

// ============================================================================
// Clear Coat Geometry Function
// ============================================================================

/// Smith-GGX geometry function for clear coat layer (single direction).
/// Uses Kelemen approximation which is more suitable for the thin clear coat.
///
/// @param VoH: Dot product of view and half-vector
/// @returns: Geometry term for clear coat (Kelemen approximation)
fn G_ClearCoat_Kelemen(VoH: f32) -> f32 {
    // Kelemen visibility function: V = 1 / (4 * VoH^2)
    // This is a simplified form suitable for clear coat layers
    return 0.25 / (VoH * VoH + CC_EPSILON);
}

/// Smith-GGX geometry function for clear coat (full form).
/// Height-correlated form for more accurate clear coat shadowing.
///
/// @param NoV: Dot product of normal and view direction
/// @param NoL: Dot product of normal and light direction
/// @param cc_roughness: Clear coat roughness in [0,1]
/// @returns: Combined geometry term for clear coat
fn G_ClearCoat(NoV: f32, NoL: f32, cc_roughness: f32) -> f32 {
    let a = cc_roughness * cc_roughness;
    let a2 = a * a;

    // Height-correlated Smith G2
    let GGXV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    let GGXL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);

    return 0.5 / (GGXV + GGXL + CC_EPSILON);
}

// ============================================================================
// Clear Coat Specular Evaluation
// ============================================================================

/// Evaluate the clear coat specular BRDF contribution.
/// Returns both the clear coat color and the Fresnel term for layer blending.
///
/// @param N: Surface normal (normalized)
/// @param V: View direction (normalized, pointing toward camera)
/// @param L: Light direction (normalized, pointing toward light)
/// @param cc_params: Clear coat parameters (intensity, roughness)
/// @returns: Clear coat specular contribution (grayscale as vec3)
fn evaluate_clear_coat(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    cc_params: ClearCoatParams
) -> vec3<f32> {
    // Quality gate: return zero if clear coat is disabled
    if !QUALITY_CLEAR_COAT_ENABLED {
        return vec3<f32>(0.0);
    }

    // Skip if clear coat intensity is zero
    if cc_params.intensity < CC_EPSILON {
        return vec3<f32>(0.0);
    }

    // Compute half-vector
    let H = normalize(V + L);

    // Compute dot products (clamped to avoid negative values)
    let NoL = max(dot(N, L), 0.0);
    let NoV = max(dot(N, V), 0.0);
    let NoH = max(dot(N, H), 0.0);
    let VoH = max(dot(V, H), 0.0);

    // Early exit for grazing angles
    if NoL < CC_EPSILON || NoV < CC_EPSILON {
        return vec3<f32>(0.0);
    }

    // Evaluate clear coat BRDF terms
    let D = D_ClearCoat(NoH, cc_params.roughness);
    let G = G_ClearCoat_Kelemen(VoH);  // Use Kelemen for thin coat
    let F = F_ClearCoat(VoH);

    // Clear coat BRDF: D * G * F * intensity
    let cc_brdf = D * G * F * cc_params.intensity;

    // Return as grayscale (clear coat is achromatic)
    return vec3<f32>(cc_brdf);
}

/// Evaluate clear coat and return Fresnel term for layer blending.
/// Use this when you need both the clear coat contribution and the
/// Fresnel factor to attenuate the base layer.
///
/// @param N: Surface normal (normalized)
/// @param V: View direction (normalized)
/// @param L: Light direction (normalized)
/// @param cc_params: Clear coat parameters
/// @returns: vec4 where xyz = clear coat specular, w = Fresnel factor (Fc)
fn evaluate_clear_coat_with_fresnel(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    cc_params: ClearCoatParams
) -> vec4<f32> {
    // Quality gate
    if !QUALITY_CLEAR_COAT_ENABLED {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Skip if clear coat intensity is zero
    if cc_params.intensity < CC_EPSILON {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Compute half-vector
    let H = normalize(V + L);

    // Compute dot products
    let NoL = max(dot(N, L), 0.0);
    let NoV = max(dot(N, V), 0.0);
    let NoH = max(dot(N, H), 0.0);
    let VoH = max(dot(V, H), 0.0);

    // Early exit for grazing angles
    if NoL < CC_EPSILON || NoV < CC_EPSILON {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Evaluate clear coat BRDF terms
    let D = D_ClearCoat(NoH, cc_params.roughness);
    let G = G_ClearCoat_Kelemen(VoH);
    let F = F_ClearCoat(VoH);

    // Clear coat BRDF contribution
    let cc_brdf = D * G * F * cc_params.intensity;

    // Return clear coat RGB (grayscale) and Fresnel factor in alpha
    // The Fresnel factor is scaled by intensity for proper blending
    return vec4<f32>(cc_brdf, cc_brdf, cc_brdf, F * cc_params.intensity);
}

// ============================================================================
// Layer Combination Functions
// ============================================================================

/// Combine clear coat layer with base material BRDF.
/// Uses Fresnel-weighted blending: final = coat + (1 - Fc * intensity) * base
///
/// This physically models that light reflected by the clear coat is not
/// available for the base layer.
///
/// @param base_brdf: Base material BRDF value
/// @param cc_brdf: Clear coat BRDF value (xyz)
/// @param Fc: Clear coat Fresnel factor (from evaluate_clear_coat_with_fresnel.w)
/// @param cc_intensity: Clear coat intensity
/// @returns: Combined BRDF value
fn combine_clear_coat(
    base_brdf: vec3<f32>,
    cc_brdf: vec3<f32>,
    Fc: f32,
    cc_intensity: f32
) -> vec3<f32> {
    // Quality gate
    if !QUALITY_CLEAR_COAT_ENABLED {
        return base_brdf;
    }

    // Fresnel-weighted combination
    // Light that passes through clear coat: (1 - Fc * intensity)
    // This light is available for the base layer
    let base_attenuation = 1.0 - Fc * cc_intensity;

    return cc_brdf + base_brdf * base_attenuation;
}

/// Simplified layer combination using pre-computed clear coat.
/// Use when you've already computed both layers separately.
///
/// @param base_brdf: Base material BRDF value
/// @param cc_result: Result from evaluate_clear_coat_with_fresnel (xyz=brdf, w=Fc)
/// @returns: Combined BRDF value
fn combine_clear_coat_simple(
    base_brdf: vec3<f32>,
    cc_result: vec4<f32>
) -> vec3<f32> {
    // Quality gate
    if !QUALITY_CLEAR_COAT_ENABLED {
        return base_brdf;
    }

    // cc_result.w contains Fc * intensity
    let base_attenuation = 1.0 - cc_result.w;

    return cc_result.xyz + base_brdf * base_attenuation;
}

// ============================================================================
// Convenience Functions for Full PBR Integration
// ============================================================================

/// Evaluate complete clear coat contribution for a light.
/// Includes NoL factor for direct lighting integration.
///
/// @param N: Surface normal
/// @param V: View direction
/// @param L: Light direction
/// @param cc_params: Clear coat parameters
/// @returns: Clear coat contribution with NoL factor
fn evaluate_clear_coat_direct(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    cc_params: ClearCoatParams
) -> vec3<f32> {
    // Quality gate
    if !QUALITY_CLEAR_COAT_ENABLED {
        return vec3<f32>(0.0);
    }

    let NoL = max(dot(N, L), 0.0);
    let cc = evaluate_clear_coat(N, V, L, cc_params);
    return cc * NoL;
}

/// Get the clear coat attenuation factor for the base layer.
/// Use this to attenuate the base BRDF before computing it.
///
/// @param N: Surface normal
/// @param V: View direction
/// @param L: Light direction
/// @param cc_params: Clear coat parameters
/// @returns: Attenuation factor in [0,1] for base layer
fn get_clear_coat_attenuation(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    cc_params: ClearCoatParams
) -> f32 {
    // Quality gate
    if !QUALITY_CLEAR_COAT_ENABLED {
        return 1.0;
    }

    if cc_params.intensity < CC_EPSILON {
        return 1.0;
    }

    let H = normalize(V + L);
    let VoH = max(dot(V, H), 0.0);
    let Fc = F_ClearCoat(VoH);

    return 1.0 - Fc * cc_params.intensity;
}
