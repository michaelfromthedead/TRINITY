// Anisotropic BRDF Functions for TRINITY Material System
// T-MAT-4.3: Anisotropic GGX Implementation
//
// This module provides anisotropic BRDF functions for directionally stretched
// highlights on surfaces like brushed metal, hair, and fabric:
//   - Anisotropic GGX Normal Distribution Function
//   - Tangent rotation for anisotropy direction
//   - Anisotropic alpha computation from roughness + strength
//   - Complete anisotropic BRDF evaluation
//
// References:
//   - Burley "Physically-Based Shading at Disney" (2012)
//   - Kulla & Conty "Revisiting Physically Based Shading at Imageworks" (2017)
//   - Heitz "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"

// Mathematical constants (shared with brdf.wgsl)
const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;
const EPSILON: f32 = 0.0001;

// Quality tier gating for anisotropy
// When false, anisotropy code paths are eliminated by naga dead-code elimination
const QUALITY_ANISOTROPY_ENABLED: bool = true;

// ============================================================================
// Anisotropy Parameters
// ============================================================================

/// Anisotropy parameters for directional roughness.
/// Used to configure anisotropic BRDF evaluation.
struct AnisotropyParams {
    /// Anisotropy strength in [0, 1].
    /// 0 = isotropic (same as standard GGX)
    /// 1 = maximum anisotropy (fully stretched highlights)
    strength: f32,
    /// Anisotropy direction in radians.
    /// Rotates the tangent basis for the anisotropic effect.
    /// 0 = along tangent, PI/2 = along bitangent
    direction: f32,
}

// ============================================================================
// Alpha Computation
// ============================================================================

/// Compute anisotropic alpha values from roughness and anisotropy strength.
/// Returns (alpha_x, alpha_y) where:
///   alpha_x = roughness * (1 + anisotropy)
///   alpha_y = roughness * (1 - anisotropy)
///
/// This creates directionally different roughness values that produce
/// stretched specular highlights along the tangent or bitangent direction.
///
/// @param roughness: Base surface roughness in [0, 1]
/// @param anisotropy: Anisotropy strength in [0, 1]
/// @returns: vec2 containing (alpha_x, alpha_y)
fn compute_aniso_alphas(roughness: f32, anisotropy: f32) -> vec2<f32> {
    // Remap roughness to alpha (Disney convention: a = roughness^2)
    let a = roughness * roughness;

    // Clamp anisotropy to valid range
    let aniso = clamp(anisotropy, 0.0, 1.0);

    // Compute directional alphas
    // alpha_x stretched along tangent, alpha_y along bitangent
    let alpha_x = a * (1.0 + aniso);
    let alpha_y = a * (1.0 - aniso);

    // Ensure minimum alpha to prevent division issues
    return vec2<f32>(
        max(alpha_x, EPSILON),
        max(alpha_y, EPSILON)
    );
}

// ============================================================================
// Tangent Rotation
// ============================================================================

/// Rotate tangent vector by anisotropy direction angle.
/// Creates a new tangent basis rotated in the tangent plane.
///
/// @param tangent: Original tangent vector (normalized)
/// @param bitangent: Original bitangent vector (normalized)
/// @param angle: Rotation angle in radians
/// @returns: Rotated tangent vector
fn rotate_tangent(tangent: vec3<f32>, bitangent: vec3<f32>, angle: f32) -> vec3<f32> {
    let cos_angle = cos(angle);
    let sin_angle = sin(angle);
    return tangent * cos_angle + bitangent * sin_angle;
}

/// Rotate bitangent vector by anisotropy direction angle.
/// Creates a new bitangent basis rotated in the tangent plane.
///
/// @param tangent: Original tangent vector (normalized)
/// @param bitangent: Original bitangent vector (normalized)
/// @param angle: Rotation angle in radians
/// @returns: Rotated bitangent vector
fn rotate_bitangent(tangent: vec3<f32>, bitangent: vec3<f32>, angle: f32) -> vec3<f32> {
    let cos_angle = cos(angle);
    let sin_angle = sin(angle);
    return -tangent * sin_angle + bitangent * cos_angle;
}

// ============================================================================
// Anisotropic Normal Distribution Function
// ============================================================================

/// Anisotropic GGX/Trowbridge-Reitz Normal Distribution Function.
/// Models elliptically stretched microfacet orientations on anisotropic surfaces.
///
/// The formula is:
///   D = 1 / (PI * alpha_x * alpha_y * denom^2)
/// where:
///   denom = (ToH/alpha_x)^2 + (BoH/alpha_y)^2 + NoH^2
///
/// @param NoH: Dot product of surface normal and half-vector
/// @param ToH: Dot product of tangent and half-vector
/// @param BoH: Dot product of bitangent and half-vector
/// @param alpha_x: Roughness along tangent direction
/// @param alpha_y: Roughness along bitangent direction
/// @returns: Anisotropic NDF value
fn D_GGX_Anisotropic(
    NoH: f32,
    ToH: f32,
    BoH: f32,
    alpha_x: f32,
    alpha_y: f32
) -> f32 {
    // Ensure valid inputs
    let ax = max(alpha_x, EPSILON);
    let ay = max(alpha_y, EPSILON);

    // Squared terms for the anisotropic distribution
    let ax2 = ax * ax;
    let ay2 = ay * ay;

    // Compute denominator terms
    let term_x = ToH * ToH / ax2;
    let term_y = BoH * BoH / ay2;
    let term_n = NoH * NoH;

    let denom = term_x + term_y + term_n;

    // Anisotropic GGX formula
    return 1.0 / (PI * ax * ay * denom * denom + EPSILON);
}

// ============================================================================
// Anisotropic Geometry Function
// ============================================================================

/// Anisotropic Smith-GGX geometry function (G1 term).
/// Models directionally-dependent self-shadowing/masking.
///
/// @param NoV: Dot product of normal and view/light direction
/// @param ToV: Dot product of tangent and view/light direction
/// @param BoV: Dot product of bitangent and view/light direction
/// @param alpha_x: Roughness along tangent direction
/// @param alpha_y: Roughness along bitangent direction
/// @returns: G1 geometry term
fn G1_GGX_Anisotropic(
    NoV: f32,
    ToV: f32,
    BoV: f32,
    alpha_x: f32,
    alpha_y: f32
) -> f32 {
    // Compute anisotropic roughness term
    let ax2 = alpha_x * alpha_x;
    let ay2 = alpha_y * alpha_y;

    let projected_roughness = sqrt(ToV * ToV * ax2 + BoV * BoV * ay2);

    // Lambda function for Smith G1
    let lambda = (-1.0 + sqrt(1.0 + projected_roughness * projected_roughness / (NoV * NoV + EPSILON))) / 2.0;

    return 1.0 / (1.0 + lambda + EPSILON);
}

/// Anisotropic Smith-GGX combined geometry function.
/// Height-correlated form combining view and light masking/shadowing.
///
/// @param NoV: Dot product of normal and view direction
/// @param NoL: Dot product of normal and light direction
/// @param ToV: Dot product of tangent and view direction
/// @param BoV: Dot product of bitangent and view direction
/// @param ToL: Dot product of tangent and light direction
/// @param BoL: Dot product of bitangent and light direction
/// @param alpha_x: Roughness along tangent direction
/// @param alpha_y: Roughness along bitangent direction
/// @returns: Combined geometry term (includes 1/4*NoV*NoL factor)
fn G_Smith_GGX_Anisotropic(
    NoV: f32,
    NoL: f32,
    ToV: f32,
    BoV: f32,
    ToL: f32,
    BoL: f32,
    alpha_x: f32,
    alpha_y: f32
) -> f32 {
    let ax2 = alpha_x * alpha_x;
    let ay2 = alpha_y * alpha_y;

    // Compute projected roughness for view direction
    let lambdaV = NoL * sqrt(ax2 * ToV * ToV + ay2 * BoV * BoV + NoV * NoV);
    // Compute projected roughness for light direction
    let lambdaL = NoV * sqrt(ax2 * ToL * ToL + ay2 * BoL * BoL + NoL * NoL);

    return 0.5 / (lambdaV + lambdaL + EPSILON);
}

// ============================================================================
// Complete Anisotropic BRDF
// ============================================================================

/// Evaluate complete anisotropic specular BRDF.
/// Combines anisotropic NDF, geometry, and Fresnel terms.
///
/// @param N: Surface normal (normalized)
/// @param V: View direction (normalized, toward camera)
/// @param L: Light direction (normalized, toward light)
/// @param T: Tangent vector (normalized)
/// @param B: Bitangent vector (normalized)
/// @param roughness: Base surface roughness
/// @param anisotropy_params: Anisotropy strength and direction
/// @param F0: Specular reflectance at normal incidence
/// @returns: Anisotropic specular BRDF contribution (RGB)
fn evaluate_aniso_brdf(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    T: vec3<f32>,
    B: vec3<f32>,
    roughness: f32,
    anisotropy_params: AnisotropyParams,
    F0: vec3<f32>
) -> vec3<f32> {
    // Quality tier gating
    if !QUALITY_ANISOTROPY_ENABLED {
        return vec3<f32>(0.0);
    }

    // Early exit for grazing angles
    let NoL = max(dot(N, L), 0.0);
    let NoV = max(dot(N, V), 0.0);

    if NoL < EPSILON || NoV < EPSILON {
        return vec3<f32>(0.0);
    }

    // Compute half-vector
    let H = normalize(V + L);
    let NoH = max(dot(N, H), 0.0);
    let VoH = max(dot(V, H), 0.0);

    // Rotate tangent basis by anisotropy direction
    let rotated_T = rotate_tangent(T, B, anisotropy_params.direction);
    let rotated_B = rotate_bitangent(T, B, anisotropy_params.direction);

    // Compute dot products with rotated basis
    let ToH = dot(rotated_T, H);
    let BoH = dot(rotated_B, H);
    let ToV = dot(rotated_T, V);
    let BoV = dot(rotated_B, V);
    let ToL = dot(rotated_T, L);
    let BoL = dot(rotated_B, L);

    // Compute anisotropic alphas
    let alphas = compute_aniso_alphas(roughness, anisotropy_params.strength);
    let alpha_x = alphas.x;
    let alpha_y = alphas.y;

    // Evaluate BRDF terms
    let D = D_GGX_Anisotropic(NoH, ToH, BoH, alpha_x, alpha_y);
    let G = G_Smith_GGX_Anisotropic(NoV, NoL, ToV, BoV, ToL, BoL, alpha_x, alpha_y);

    // Schlick Fresnel
    let Fc = pow(1.0 - VoH, 5.0);
    let F = F0 + (vec3<f32>(1.0) - F0) * Fc;

    // Cook-Torrance: D * G * F (G already includes 1/4*NoV*NoL)
    return D * G * F;
}

/// Evaluate anisotropic BRDF with PBR material parameters.
/// Convenience function that extracts anisotropy from PBRParams.
///
/// @param params: PBR material parameters (requires anisotropy field)
/// @param N: Surface normal (normalized)
/// @param V: View direction (normalized)
/// @param L: Light direction (normalized)
/// @param T: Tangent vector (normalized)
/// @param B: Bitangent vector (normalized)
/// @returns: BRDF value multiplied by NoL
fn evaluate_aniso_brdf_pbr(
    params: PBRParams,
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    T: vec3<f32>,
    B: vec3<f32>
) -> vec3<f32> {
    // Quality tier gating
    if !QUALITY_ANISOTROPY_ENABLED {
        return vec3<f32>(0.0);
    }

    // If anisotropy is zero, skip expensive computation
    if abs(params.anisotropy) < EPSILON {
        return vec3<f32>(0.0);
    }

    // Compute F0 from metallic
    let dielectric_F0 = vec3<f32>(0.04);
    let F0 = mix(dielectric_F0, params.base_color, params.metallic);

    // Create anisotropy params from material
    // Direction defaults to 0 (along tangent)
    var aniso_params: AnisotropyParams;
    aniso_params.strength = abs(params.anisotropy);
    aniso_params.direction = 0.0;  // Could be extended with a direction parameter

    // Evaluate anisotropic BRDF
    let specular = evaluate_aniso_brdf(N, V, L, T, B, params.roughness, aniso_params, F0);

    // Combine with NoL
    let NoL = max(dot(N, L), 0.0);
    return specular * NoL;
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Create default anisotropy parameters (isotropic).
fn anisotropy_params_default() -> AnisotropyParams {
    var params: AnisotropyParams;
    params.strength = 0.0;
    params.direction = 0.0;
    return params;
}

/// Create anisotropy parameters from strength and direction.
fn anisotropy_params_create(strength: f32, direction: f32) -> AnisotropyParams {
    var params: AnisotropyParams;
    params.strength = clamp(strength, 0.0, 1.0);
    params.direction = direction;
    return params;
}

/// Check if anisotropy is enabled and has non-zero effect.
fn has_anisotropy(strength: f32) -> bool {
    return QUALITY_ANISOTROPY_ENABLED && abs(strength) > EPSILON;
}
