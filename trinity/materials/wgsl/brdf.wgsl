// BRDF Functions for TRINITY Material System
// T-MAT-3.2: Cook-Torrance BRDF Implementation
//
// This module provides the core BRDF functions for PBR rendering:
//   - GGX Normal Distribution Function (Trowbridge-Reitz)
//   - Smith-GGX Geometry Function (height-correlated)
//   - Schlick Fresnel approximation
//   - Cook-Torrance specular BRDF
//   - Lambertian diffuse BRDF
//
// References:
//   - Walter et al. "Microfacet Models for Refraction through Rough Surfaces"
//   - Heitz "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
//   - Schlick "An Inexpensive BRDF Model for Physically-based Rendering"

// Mathematical constants
const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;
const EPSILON: f32 = 0.0001;

// ============================================================================
// Normal Distribution Functions (NDF)
// ============================================================================

/// GGX/Trowbridge-Reitz Normal Distribution Function.
/// Models the distribution of microfacet orientations on a rough surface.
///
/// @param NoH: Dot product of surface normal and half-vector, clamped to [0,1]
/// @param roughness: Surface roughness in [0,1], where 0=mirror, 1=diffuse
/// @returns: NDF value (probability density of microfacet orientation)
fn D_GGX(NoH: f32, roughness: f32) -> f32 {
    // Remap roughness to alpha (Disney/Unreal convention)
    let a = roughness * roughness;
    let a2 = a * a;
    let NoH2 = NoH * NoH;

    // GGX formula: a^2 / (PI * ((NoH^2 * (a^2 - 1) + 1)^2))
    let denom = NoH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + EPSILON);
}

// ============================================================================
// Geometry Functions (GSF)
// ============================================================================

/// Schlick-GGX geometry function for a single direction.
/// Models self-shadowing/masking of microfacets.
///
/// @param NoX: Dot product with normal (either NoV or NoL)
/// @param roughness: Surface roughness in [0,1]
/// @returns: Geometry term for single direction
fn G1_SchlickGGX(NoX: f32, roughness: f32) -> f32 {
    // Remapped roughness for direct lighting (Disney/Unreal)
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;

    return NoX / (NoX * (1.0 - k) + k + EPSILON);
}

/// Smith-GGX Geometry Function (height-correlated form).
/// Combines masking (view) and shadowing (light) into a single term.
/// Uses the height-correlated form which is more physically accurate.
///
/// @param NoV: Dot product of surface normal and view direction
/// @param NoL: Dot product of surface normal and light direction
/// @param roughness: Surface roughness in [0,1]
/// @returns: Combined geometry term divided by (4 * NoV * NoL)
fn G_Smith_GGX(NoV: f32, NoL: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;

    // Height-correlated Smith G2 (Heitz 2014)
    // This form already includes the 1/(4*NoV*NoL) denominator
    let GGXV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    let GGXL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);

    return 0.5 / (GGXV + GGXL + EPSILON);
}

/// Smith geometry function using Schlick approximation.
/// Simpler/faster but less accurate than height-correlated form.
///
/// @param NoV: Dot product of surface normal and view direction
/// @param NoL: Dot product of surface normal and light direction
/// @param roughness: Surface roughness in [0,1]
/// @returns: Geometry term (NOT divided by denominator)
fn G_Smith_Schlick(NoV: f32, NoL: f32, roughness: f32) -> f32 {
    let ggxV = G1_SchlickGGX(NoV, roughness);
    let ggxL = G1_SchlickGGX(NoL, roughness);
    return ggxV * ggxL;
}

// ============================================================================
// Fresnel Functions
// ============================================================================

/// Schlick Fresnel approximation.
/// Models the increase in reflectance at grazing angles.
///
/// @param VoH: Dot product of view direction and half-vector
/// @param F0: Reflectance at normal incidence (base specular color)
/// @returns: Fresnel reflectance as RGB
fn F_Schlick(VoH: f32, F0: vec3<f32>) -> vec3<f32> {
    let Fc = pow(1.0 - VoH, 5.0);
    return F0 + (vec3<f32>(1.0) - F0) * Fc;
}

/// Schlick Fresnel with roughness factor for image-based lighting.
/// Accounts for surface roughness in the Fresnel term for IBL.
///
/// @param VoH: Dot product of view direction and half-vector
/// @param F0: Reflectance at normal incidence
/// @param roughness: Surface roughness in [0,1]
/// @returns: Roughness-adjusted Fresnel reflectance
fn F_Schlick_Roughness(VoH: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let Fc = pow(1.0 - VoH, 5.0);
    return F0 + (max(vec3<f32>(1.0 - roughness), F0) - F0) * Fc;
}

/// Scalar Fresnel for dielectrics with fixed F0.
/// Optimized version for non-metallic materials.
///
/// @param VoH: Dot product of view direction and half-vector
/// @param f0: Scalar reflectance at normal incidence (typically 0.04)
/// @returns: Scalar Fresnel reflectance
fn F_Schlick_Scalar(VoH: f32, f0: f32) -> f32 {
    let Fc = pow(1.0 - VoH, 5.0);
    return f0 + (1.0 - f0) * Fc;
}

// ============================================================================
// Specular BRDF
// ============================================================================

/// Cook-Torrance specular microfacet BRDF.
/// Combines NDF, geometry, and Fresnel terms for physically-based specular.
///
/// @param N: Surface normal (normalized)
/// @param V: View direction (normalized, pointing toward camera)
/// @param L: Light direction (normalized, pointing toward light)
/// @param roughness: Surface roughness in [0,1]
/// @param F0: Specular reflectance at normal incidence
/// @returns: Specular BRDF contribution (RGB)
fn BRDF_Specular(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    roughness: f32,
    F0: vec3<f32>
) -> vec3<f32> {
    // Compute half-vector
    let H = normalize(V + L);

    // Compute dot products (clamped to avoid negative values)
    let NoL = max(dot(N, L), 0.0);
    let NoV = max(dot(N, V), 0.0);
    let NoH = max(dot(N, H), 0.0);
    let VoH = max(dot(V, H), 0.0);

    // Early exit for grazing angles
    if NoL < EPSILON || NoV < EPSILON {
        return vec3<f32>(0.0);
    }

    // Evaluate BRDF terms
    let D = D_GGX(NoH, roughness);
    let G = G_Smith_GGX(NoV, NoL, roughness);
    let F = F_Schlick(VoH, F0);

    // Cook-Torrance: D * G * F (G already includes denominator)
    return D * G * F;
}

// ============================================================================
// Diffuse BRDF
// ============================================================================

/// Lambertian diffuse BRDF.
/// Simple energy-conserving diffuse model.
///
/// @param base_color: Surface albedo (linear RGB)
/// @returns: Diffuse BRDF contribution
fn BRDF_Diffuse(base_color: vec3<f32>) -> vec3<f32> {
    return base_color * INV_PI;
}

/// Disney diffuse BRDF (Burley 2012).
/// More accurate diffuse model with roughness-dependent response.
///
/// @param base_color: Surface albedo (linear RGB)
/// @param NoV: Dot product of normal and view direction
/// @param NoL: Dot product of normal and light direction
/// @param VoH: Dot product of view and half-vector
/// @param roughness: Surface roughness in [0,1]
/// @returns: Diffuse BRDF contribution
fn BRDF_Diffuse_Disney(
    base_color: vec3<f32>,
    NoV: f32,
    NoL: f32,
    VoH: f32,
    roughness: f32
) -> vec3<f32> {
    let FD90 = 0.5 + 2.0 * VoH * VoH * roughness;
    let FdV = 1.0 + (FD90 - 1.0) * pow(1.0 - NoV, 5.0);
    let FdL = 1.0 + (FD90 - 1.0) * pow(1.0 - NoL, 5.0);
    return base_color * INV_PI * FdV * FdL;
}

// ============================================================================
// Combined PBR BRDF
// ============================================================================

/// Compute F0 (specular reflectance at normal incidence) from material properties.
/// For dielectrics, uses a fixed 4% reflectance.
/// For metals, uses the base color as F0.
///
/// @param base_color: Surface base color
/// @param metallic: Metallic factor in [0,1]
/// @returns: F0 specular color
fn compute_F0(base_color: vec3<f32>, metallic: f32) -> vec3<f32> {
    let dielectric_F0 = vec3<f32>(0.04);
    return mix(dielectric_F0, base_color, metallic);
}

/// Evaluate the complete PBR BRDF for a single light.
/// Combines specular (Cook-Torrance) and diffuse (Lambertian) contributions.
///
/// @param params: PBR material parameters
/// @param N: Surface normal (normalized)
/// @param V: View direction (normalized)
/// @param L: Light direction (normalized)
/// @returns: BRDF value multiplied by NoL
fn evaluate_brdf(
    params: PBRParams,
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>
) -> vec3<f32> {
    // Compute F0 from metallic
    let F0 = compute_F0(params.base_color, params.metallic);

    // Specular contribution
    let specular = BRDF_Specular(N, V, L, params.roughness, F0);

    // Diffuse contribution (reduced by metallic factor)
    // Metals have no diffuse component
    let diffuse = BRDF_Diffuse(params.base_color) * (1.0 - params.metallic);

    // Combine with NoL factor
    let NoL = max(dot(N, L), 0.0);
    return (diffuse + specular) * NoL;
}

/// Evaluate PBR BRDF with energy conservation.
/// Accounts for the fact that energy reflected as specular cannot be diffuse.
///
/// @param params: PBR material parameters
/// @param N: Surface normal (normalized)
/// @param V: View direction (normalized)
/// @param L: Light direction (normalized)
/// @returns: Energy-conserving BRDF value multiplied by NoL
fn evaluate_brdf_energy_conserving(
    params: PBRParams,
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>
) -> vec3<f32> {
    let H = normalize(V + L);
    let NoL = max(dot(N, L), 0.0);
    let NoV = max(dot(N, V), 0.0);
    let VoH = max(dot(V, H), 0.0);

    if NoL < EPSILON {
        return vec3<f32>(0.0);
    }

    // Compute F0 from metallic
    let F0 = compute_F0(params.base_color, params.metallic);

    // Fresnel term (needed for energy conservation)
    let F = F_Schlick(VoH, F0);

    // Specular contribution
    let specular = BRDF_Specular(N, V, L, params.roughness, F0);

    // Diffuse contribution with energy conservation
    // (1 - F) is the amount of light not reflected as specular
    let kD = (vec3<f32>(1.0) - F) * (1.0 - params.metallic);
    let diffuse = BRDF_Diffuse(params.base_color) * kD;

    return (diffuse + specular) * NoL;
}
