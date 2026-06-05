// Transmission BRDF Functions for TRINITY Material System
// T-MAT-4.5: Transmission Implementation (Thin Glass, Liquids)
//
// This module provides transmission functions for PBR rendering:
//   - Snell's law refraction with total internal reflection handling
//   - Screen-space refraction with roughness-based blur
//   - Beer-Lambert absorption for colored transmission
//   - Fresnel-weighted blend between transmission and reflection
//
// The transmission model simulates:
//   - Thin glass (windows, bottles)
//   - Liquids (water, colored drinks)
//   - Thin-walled containers
//   - Ice and crystals
//
// References:
//   - KHR_materials_transmission glTF extension
//   - Filament Material Model Documentation
//   - Enterprise PBR Shading Model (MaterialX)

// Mathematical constants (may be imported from brdf.wgsl in production)
const TRANS_PI: f32 = 3.14159265359;
const TRANS_EPSILON: f32 = 0.0001;

// Default IOR for glass (crown glass)
const DEFAULT_IOR: f32 = 1.5;

// Air IOR (surrounding medium)
const AIR_IOR: f32 = 1.0;

// ============================================================================
// Transmission Parameters
// ============================================================================

/// Parameters controlling transmission behavior.
/// factor: Transmission amount (0 = opaque, 1 = fully transmissive)
/// ior: Index of refraction (1.0 = air, 1.5 = glass, 1.33 = water)
/// roughness: Transmitted ray roughness for blur effect
/// attenuation_color: Color absorbed over attenuation_distance
/// attenuation_distance: Distance for full attenuation (in world units)
struct TransmissionParams {
    factor: f32,
    ior: f32,
    roughness: f32,
    attenuation_color: vec3<f32>,
    attenuation_distance: f32,
}

// ============================================================================
// Quality Tier Gating
// ============================================================================

/// Quality tier control for transmission. When false, transmission is disabled
/// and evaluate_transmission returns zero. This const is dead-code eliminated
/// by naga compiler in LOW quality variants.
const QUALITY_TRANSMISSION_ENABLED: bool = true;

// ============================================================================
// Fresnel Functions for Transmission
// ============================================================================

/// Schlick Fresnel approximation for dielectric transmission.
/// Computes reflectance at the interface; transmission = 1 - reflectance.
///
/// @param cos_theta: Cosine of incidence angle (dot of view and normal)
/// @param ior: Index of refraction of transmissive medium
/// @returns: Fresnel reflectance (scalar)
fn F_Transmission(cos_theta: f32, ior: f32) -> f32 {
    // Compute F0 from IOR: ((n1 - n2) / (n1 + n2))^2
    let f0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
    let Fc = pow(1.0 - cos_theta, 5.0);
    return f0 + (1.0 - f0) * Fc;
}

/// Compute F0 (reflectance at normal incidence) from IOR.
///
/// @param ior: Index of refraction
/// @returns: F0 value
fn ior_to_f0(ior: f32) -> f32 {
    return pow((1.0 - ior) / (1.0 + ior), 2.0);
}

// ============================================================================
// Refraction Functions (Snell's Law)
// ============================================================================

/// Compute refracted direction using Snell's law.
/// Returns (0, 0, 0) if total internal reflection occurs.
///
/// Snell's law: n1 * sin(theta_i) = n2 * sin(theta_t)
/// Rewritten: eta = n1/n2, cos_t = sqrt(1 - eta^2 * (1 - cos_i^2))
///
/// @param incident: Incident direction (normalized, pointing away from surface)
/// @param normal: Surface normal (normalized)
/// @param eta: Ratio of IORs (IOR_from / IOR_to)
/// @returns: Refracted direction (normalized) or zero vector for TIR
fn refract_direction(incident: vec3<f32>, normal: vec3<f32>, eta: f32) -> vec3<f32> {
    let cos_i = dot(normal, incident);
    let sin2_i = 1.0 - cos_i * cos_i;
    let sin2_t = eta * eta * sin2_i;

    // Total internal reflection check
    // If sin2_t >= 1.0, no transmitted ray exists
    if sin2_t >= 1.0 {
        return vec3<f32>(0.0);
    }

    let cos_t = sqrt(1.0 - sin2_t);

    // Refracted direction: -eta * incident + (eta * cos_i - cos_t) * normal
    // Note: incident points away from surface, we want refracted pointing into medium
    return -eta * incident + (eta * cos_i - cos_t) * normal;
}

/// Check if total internal reflection occurs.
///
/// @param cos_i: Cosine of incidence angle
/// @param eta: Ratio of IORs (IOR_from / IOR_to)
/// @returns: true if TIR occurs
fn is_total_internal_reflection(cos_i: f32, eta: f32) -> bool {
    let sin2_i = 1.0 - cos_i * cos_i;
    let sin2_t = eta * eta * sin2_i;
    return sin2_t >= 1.0;
}

/// Compute refraction UV offset for screen-space refraction.
/// Projects the refracted direction difference onto screen space.
///
/// @param refracted_dir: Refracted direction (normalized)
/// @param view_dir: Original view direction (normalized)
/// @param normal: Surface normal
/// @param ior: Index of refraction
/// @returns: UV offset for screen-space sampling
fn compute_refraction_uv_offset(
    refracted_dir: vec3<f32>,
    view_dir: vec3<f32>,
    normal: vec3<f32>,
    ior: f32
) -> vec2<f32> {
    // Compute the tangent-space offset of the refraction
    // This is a simplified approximation for screen-space refraction
    let offset_3d = refracted_dir - view_dir;

    // Project onto tangent plane (perpendicular to view)
    // For a proper implementation, this would use the projection matrix
    let tangent = normalize(cross(normal, vec3<f32>(0.0, 1.0, 0.0)));
    let bitangent = cross(normal, tangent);

    // Scale by IOR-based distortion factor
    let distortion = (ior - 1.0) * 0.1;

    return vec2<f32>(
        dot(offset_3d, tangent) * distortion,
        dot(offset_3d, bitangent) * distortion
    );
}

// ============================================================================
// Beer-Lambert Absorption
// ============================================================================

/// Apply Beer-Lambert absorption for colored transmission.
/// Simulates light absorption as it travels through a medium.
///
/// Formula: transmittance = exp(-absorption * distance / attenuation_distance)
/// For each color channel: transmittance_c = attenuation_color_c ^ (distance / attenuation_distance)
///
/// @param transmitted_color: Color of light entering the medium
/// @param distance: Distance traveled through the medium
/// @param attenuation_color: Color absorbed over attenuation_distance
/// @param attenuation_distance: Distance for full attenuation
/// @returns: Attenuated color after absorption
fn apply_beer_law(
    transmitted_color: vec3<f32>,
    distance: f32,
    attenuation_color: vec3<f32>,
    attenuation_distance: f32
) -> vec3<f32> {
    // Handle infinite attenuation distance (no absorption)
    if attenuation_distance <= 0.0 || attenuation_distance > 1e10 {
        return transmitted_color;
    }

    // Handle zero or negative distance
    if distance <= 0.0 {
        return transmitted_color;
    }

    // Beer-Lambert: transmittance = color ^ (distance / attenuation_distance)
    // This uses the "attenuation_color" as the color remaining after traveling attenuation_distance
    let t = distance / attenuation_distance;

    // Compute per-channel transmittance
    // For white attenuation (1,1,1), no absorption occurs
    // For colored attenuation, channels absorb differently
    let transmittance = vec3<f32>(
        pow(attenuation_color.x, t),
        pow(attenuation_color.y, t),
        pow(attenuation_color.z, t)
    );

    return transmitted_color * transmittance;
}

/// Compute Beer-Lambert transmittance factor only.
///
/// @param distance: Distance traveled through medium
/// @param attenuation_color: Color absorbed over attenuation_distance
/// @param attenuation_distance: Distance for full attenuation
/// @returns: Transmittance factor per channel
fn compute_beer_transmittance(
    distance: f32,
    attenuation_color: vec3<f32>,
    attenuation_distance: f32
) -> vec3<f32> {
    if attenuation_distance <= 0.0 || attenuation_distance > 1e10 {
        return vec3<f32>(1.0);
    }

    if distance <= 0.0 {
        return vec3<f32>(1.0);
    }

    let t = distance / attenuation_distance;
    return vec3<f32>(
        pow(attenuation_color.x, t),
        pow(attenuation_color.y, t),
        pow(attenuation_color.z, t)
    );
}

// ============================================================================
// Screen-Space Refraction Sampling
// ============================================================================

/// Sample transmission from screen-space with roughness blur.
/// Uses mip-based blur approximation for rough transmission.
///
/// Note: This is a placeholder that returns the input color.
/// In a full implementation, this would sample from a render target
/// with appropriate mip level based on roughness.
///
/// @param uv: Current fragment UV
/// @param refraction_offset: UV offset from refraction
/// @param roughness: Transmission roughness for blur
/// @param base_color: Fallback/background color
/// @returns: Sampled and blurred transmission color
fn sample_transmission(
    uv: vec2<f32>,
    refraction_offset: vec2<f32>,
    roughness: f32,
    base_color: vec3<f32>
) -> vec3<f32> {
    // Compute target UV with refraction offset
    let target_uv = uv + refraction_offset;

    // Clamp UV to valid range [0, 1]
    let clamped_uv = clamp(target_uv, vec2<f32>(0.0), vec2<f32>(1.0));

    // In a full implementation:
    // 1. Sample from scene color buffer at clamped_uv
    // 2. Use mip level based on roughness: mip = roughness * max_mip_levels
    // 3. Apply cone tracing or screen-space blur

    // Placeholder: blend base color with slight offset effect
    // Real implementation would sample actual scene behind the surface
    let uv_visible = step(vec2<f32>(0.0), target_uv) * step(target_uv, vec2<f32>(1.0));
    let visibility = uv_visible.x * uv_visible.y;

    // Simulate roughness blur by lerping toward base color
    // Higher roughness = more blur = closer to uniform color
    let blur_factor = roughness * roughness;

    return mix(base_color, base_color * 0.9, blur_factor * (1.0 - visibility));
}

/// Compute mip level for roughness-based blur.
///
/// @param roughness: Transmission roughness
/// @param max_mip: Maximum mip level available
/// @returns: Mip level for texture sampling
fn compute_transmission_mip(roughness: f32, max_mip: f32) -> f32 {
    // Perceptual roughness to mip level mapping
    // Square roughness gives better perceptual distribution
    return roughness * roughness * max_mip;
}

// ============================================================================
// Transmission Evaluation
// ============================================================================

/// Evaluate transmission contribution for a surface point.
/// Combines refraction, Beer-Lambert absorption, and Fresnel weighting.
///
/// @param N: Surface normal (normalized)
/// @param V: View direction (normalized, pointing toward camera)
/// @param params: Transmission parameters
/// @param thickness: Material thickness for absorption calculation
/// @param background_color: Color of scene behind the surface
/// @returns: Final transmission contribution
fn evaluate_transmission(
    N: vec3<f32>,
    V: vec3<f32>,
    params: TransmissionParams,
    thickness: f32,
    background_color: vec3<f32>
) -> vec3<f32> {
    // Quality gate
    if !QUALITY_TRANSMISSION_ENABLED {
        return vec3<f32>(0.0);
    }

    // Skip if transmission factor is zero
    if params.factor < TRANS_EPSILON {
        return vec3<f32>(0.0);
    }

    // Compute view-normal angle
    let NoV = max(dot(N, V), 0.0);

    // Compute eta (IOR ratio) - assuming air outside
    let eta = AIR_IOR / params.ior;

    // Check for total internal reflection
    if is_total_internal_reflection(NoV, eta) {
        // TIR: no transmission, all reflection
        return vec3<f32>(0.0);
    }

    // Compute refracted direction
    let refracted = refract_direction(V, N, eta);

    // Compute Fresnel (how much is reflected vs transmitted)
    let F = F_Transmission(NoV, params.ior);
    let transmission_factor = (1.0 - F) * params.factor;

    // Apply Beer-Lambert absorption based on thickness
    let absorption_distance = thickness > 0.0 ? thickness : 0.001;
    let attenuated_color = apply_beer_law(
        background_color,
        absorption_distance,
        params.attenuation_color,
        params.attenuation_distance
    );

    // Apply transmission factor
    return attenuated_color * transmission_factor;
}

/// Evaluate transmission with screen-space refraction.
/// For use in forward rendering with access to scene color buffer.
///
/// @param N: Surface normal (normalized)
/// @param V: View direction (normalized)
/// @param uv: Fragment UV coordinates
/// @param params: Transmission parameters
/// @param thickness: Material thickness
/// @param background_color: Fallback background color
/// @returns: Final transmission contribution with screen-space effects
fn evaluate_transmission_screen_space(
    N: vec3<f32>,
    V: vec3<f32>,
    uv: vec2<f32>,
    params: TransmissionParams,
    thickness: f32,
    background_color: vec3<f32>
) -> vec3<f32> {
    // Quality gate
    if !QUALITY_TRANSMISSION_ENABLED {
        return vec3<f32>(0.0);
    }

    // Skip if transmission factor is zero
    if params.factor < TRANS_EPSILON {
        return vec3<f32>(0.0);
    }

    // Compute view-normal angle
    let NoV = max(dot(N, V), 0.0);

    // Compute eta (IOR ratio)
    let eta = AIR_IOR / params.ior;

    // Check for total internal reflection
    if is_total_internal_reflection(NoV, eta) {
        return vec3<f32>(0.0);
    }

    // Compute refracted direction
    let refracted = refract_direction(V, N, eta);

    // Skip if refraction failed (shouldn't happen after TIR check)
    if length(refracted) < TRANS_EPSILON {
        return vec3<f32>(0.0);
    }

    // Compute UV offset from refraction
    let uv_offset = compute_refraction_uv_offset(refracted, V, N, params.ior);

    // Sample scene with roughness blur
    let sampled_color = sample_transmission(
        uv,
        uv_offset,
        params.roughness,
        background_color
    );

    // Apply Beer-Lambert absorption
    let absorption_distance = thickness > 0.0 ? thickness : 0.001;
    let attenuated_color = apply_beer_law(
        sampled_color,
        absorption_distance,
        params.attenuation_color,
        params.attenuation_distance
    );

    // Compute Fresnel
    let F = F_Transmission(NoV, params.ior);
    let transmission_factor = (1.0 - F) * params.factor;

    return attenuated_color * transmission_factor;
}

/// Evaluate transmission and return both transmission color and Fresnel factor.
/// Use for proper blending with reflection.
///
/// @param N: Surface normal (normalized)
/// @param V: View direction (normalized)
/// @param params: Transmission parameters
/// @param thickness: Material thickness
/// @param background_color: Background color
/// @returns: vec4 where xyz = transmission color, w = transmission factor (1 - F)
fn evaluate_transmission_with_fresnel(
    N: vec3<f32>,
    V: vec3<f32>,
    params: TransmissionParams,
    thickness: f32,
    background_color: vec3<f32>
) -> vec4<f32> {
    // Quality gate
    if !QUALITY_TRANSMISSION_ENABLED {
        return vec4<f32>(0.0);
    }

    // Skip if transmission factor is zero
    if params.factor < TRANS_EPSILON {
        return vec4<f32>(0.0);
    }

    // Compute view-normal angle
    let NoV = max(dot(N, V), 0.0);

    // Compute eta
    let eta = AIR_IOR / params.ior;

    // Check for TIR
    if is_total_internal_reflection(NoV, eta) {
        return vec4<f32>(0.0);
    }

    // Compute Fresnel
    let F = F_Transmission(NoV, params.ior);
    let T = (1.0 - F) * params.factor;

    // Apply absorption
    let absorption_distance = thickness > 0.0 ? thickness : 0.001;
    let attenuated_color = apply_beer_law(
        background_color,
        absorption_distance,
        params.attenuation_color,
        params.attenuation_distance
    );

    let transmission_color = attenuated_color * T;

    return vec4<f32>(transmission_color, T);
}

// ============================================================================
// Layer Combination Functions
// ============================================================================

/// Combine transmission with reflection/base BRDF.
/// Uses Fresnel-weighted blending: final = F * reflection + (1 - F) * transmission
///
/// @param reflection: Reflected/specular component
/// @param transmission: Transmitted component
/// @param F: Fresnel reflectance
/// @param transmission_factor: Overall transmission factor
/// @returns: Combined color
fn combine_transmission_reflection(
    reflection: vec3<f32>,
    transmission: vec3<f32>,
    F: f32,
    transmission_factor: f32
) -> vec3<f32> {
    // Quality gate
    if !QUALITY_TRANSMISSION_ENABLED {
        return reflection;
    }

    // Fresnel-weighted combination
    // F portion is reflected, (1-F) * factor is transmitted
    let T = (1.0 - F) * transmission_factor;

    return reflection * F + transmission * T;
}

/// Combine transmission result with base BRDF.
/// Simplified version using pre-computed transmission.
///
/// @param base_brdf: Base material BRDF (diffuse + specular)
/// @param transmission_result: Result from evaluate_transmission_with_fresnel
/// @returns: Combined color
fn combine_transmission_simple(
    base_brdf: vec3<f32>,
    transmission_result: vec4<f32>
) -> vec3<f32> {
    // Quality gate
    if !QUALITY_TRANSMISSION_ENABLED {
        return base_brdf;
    }

    // transmission_result.w contains T = (1-F) * factor
    // Base BRDF contribution should be attenuated by (1 - T)
    let base_factor = 1.0 - transmission_result.w;

    return base_brdf * base_factor + transmission_result.xyz;
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create default transmission parameters for glass (IOR 1.5).
fn transmission_params_glass() -> TransmissionParams {
    var params: TransmissionParams;
    params.factor = 1.0;
    params.ior = 1.5;
    params.roughness = 0.0;
    params.attenuation_color = vec3<f32>(1.0, 1.0, 1.0);
    params.attenuation_distance = 1e10; // Infinite (no absorption)
    return params;
}

/// Create transmission parameters for water (IOR 1.33).
fn transmission_params_water() -> TransmissionParams {
    var params: TransmissionParams;
    params.factor = 1.0;
    params.ior = 1.33;
    params.roughness = 0.0;
    params.attenuation_color = vec3<f32>(0.85, 0.95, 1.0); // Slight blue tint
    params.attenuation_distance = 10.0;
    return params;
}

/// Create transmission parameters for colored glass.
fn transmission_params_colored_glass(color: vec3<f32>, absorption_distance: f32) -> TransmissionParams {
    var params: TransmissionParams;
    params.factor = 1.0;
    params.ior = 1.5;
    params.roughness = 0.0;
    params.attenuation_color = color;
    params.attenuation_distance = absorption_distance;
    return params;
}

/// Get the critical angle for total internal reflection (in radians).
///
/// @param ior: Index of refraction of denser medium
/// @returns: Critical angle in radians
fn get_critical_angle(ior: f32) -> f32 {
    // sin(critical_angle) = n2/n1 = 1/ior (assuming air outside)
    // critical_angle = asin(1/ior)
    return asin(1.0 / max(ior, 1.0001));
}

/// Check if viewing angle exceeds critical angle.
///
/// @param NoV: Dot product of normal and view direction
/// @param ior: Index of refraction
/// @returns: true if angle exceeds critical angle (TIR would occur)
fn exceeds_critical_angle(NoV: f32, ior: f32) -> bool {
    let critical_cos = sqrt(1.0 - 1.0 / (ior * ior));
    return NoV < critical_cos;
}
