// Lighting Functions for TRINITY Material System
// T-MAT-3.3: Light Loop and Shading
//
// This module provides lighting infrastructure for PBR rendering:
//   - Light struct with type enum (Directional, Point, Spot)
//   - Light evaluation functions for each type
//   - Light accumulation loop for N active lights
//   - Shadow sampling placeholder
//   - Ambient occlusion and emissive terms
//
// References:
//   - Karis, B. "Real Shading in Unreal Engine 4"
//   - Lagarde, S. & de Rousiers, C. "Moving Frostbite to PBR"

// Mathematical constants (shared with brdf.wgsl)
const PI: f32 = 3.14159265359;
const EPSILON: f32 = 0.0001;

// Maximum number of lights supported per fragment
const MAX_LIGHTS: u32 = 8u;

// ============================================================================
// Light Type Enumeration
// ============================================================================

/// Light type constants for the type field in Light struct.
/// Directional: Infinite distance light (sun), no attenuation
/// Point: Omnidirectional light with distance attenuation
/// Spot: Cone-shaped light with distance and angular attenuation
const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;

// ============================================================================
// Light Structures
// ============================================================================

/// Unified light structure for all light types.
/// The interpretation of fields depends on the light type.
struct Light {
    /// Light type: 0=Directional, 1=Point, 2=Spot
    light_type: u32,
    /// Intensity multiplier for the light
    intensity: f32,
    /// Range/radius for Point and Spot lights (ignored for Directional)
    range: f32,
    /// Padding for 16-byte alignment
    _pad0: f32,

    /// Position in world space (Point/Spot only)
    /// For Directional lights, this field is unused
    position: vec3<f32>,
    /// Cosine of inner cone angle for Spot lights
    /// Defines where falloff begins (1.0 = no inner cone)
    cos_inner_angle: f32,

    /// Direction of the light (normalized)
    /// For Directional: direction light is traveling (toward surface)
    /// For Spot: direction the spotlight is pointing
    direction: vec3<f32>,
    /// Cosine of outer cone angle for Spot lights
    /// Defines edge of light cone (0.0 = 90 degrees)
    cos_outer_angle: f32,

    /// Light color in linear RGB
    color: vec3<f32>,
    /// Padding for 16-byte alignment
    _pad1: f32,
}

/// Uniform buffer for light array.
/// Contains all active lights for the current frame.
struct LightBuffer {
    /// Number of active lights (0 to MAX_LIGHTS)
    count: u32,
    /// Padding for 16-byte alignment
    _pad: vec3<u32>,
    /// Array of lights
    lights: array<Light, 8>,
}

/// Result of light evaluation containing direction and radiance.
struct LightSample {
    /// Direction from surface point to light source (normalized)
    direction: vec3<f32>,
    /// Incoming radiance from the light (color * intensity * attenuation)
    radiance: vec3<f32>,
    /// Distance to light (for shadow calculations)
    distance: f32,
    /// Shadow factor (1.0 = fully lit, 0.0 = fully shadowed)
    shadow: f32,
}

// ============================================================================
// Attenuation Functions
// ============================================================================

/// Inverse-square attenuation with smooth falloff at range boundary.
/// Uses the UE4/Frostbite windowed falloff function.
///
/// @param distance: Distance from light to surface point
/// @param range: Maximum range of the light
/// @returns: Attenuation factor in [0, 1]
fn attenuation_point(distance: f32, range: f32) -> f32 {
    // Avoid division by zero
    let d = max(distance, EPSILON);
    let r = max(range, EPSILON);

    // Physical inverse-square falloff
    let distance_ratio = d / r;
    let distance_ratio_sq = distance_ratio * distance_ratio;
    let distance_ratio_4 = distance_ratio_sq * distance_ratio_sq;

    // Smooth falloff near range boundary (Frostbite/UE4 style)
    // saturate(1 - (d/r)^4)^2 * (1/d^2)
    let falloff = max(1.0 - distance_ratio_4, 0.0);
    let falloff_sq = falloff * falloff;

    // Inverse-square with range normalization
    // Multiply by (1/range^2) to normalize intensity at range
    let attenuation = falloff_sq / (d * d + EPSILON);

    return attenuation;
}

/// Angular attenuation for spotlight cone.
/// Smooth falloff between inner and outer cone angles.
///
/// @param cos_angle: Cosine of angle between light direction and surface direction
/// @param cos_inner: Cosine of inner cone angle (where falloff begins)
/// @param cos_outer: Cosine of outer cone angle (edge of light cone)
/// @returns: Angular attenuation factor in [0, 1]
fn attenuation_spot_angle(cos_angle: f32, cos_inner: f32, cos_outer: f32) -> f32 {
    // Clamp to prevent invalid values
    let inner = max(cos_inner, cos_outer + EPSILON);

    // Smooth interpolation from inner to outer cone
    let t = saturate((cos_angle - cos_outer) / (inner - cos_outer + EPSILON));

    // Quadratic falloff for smoother edge
    return t * t;
}

// ============================================================================
// Light Evaluation Functions
// ============================================================================

/// Evaluate a directional light.
/// Directional lights have no attenuation and represent infinitely distant sources.
///
/// @param light: The directional light to evaluate
/// @param world_position: World space position of the surface point
/// @returns: LightSample with direction and radiance
fn evaluate_directional_light(light: Light, world_position: vec3<f32>) -> LightSample {
    var sample: LightSample;

    // Direction is opposite of light travel direction (toward the light)
    sample.direction = -normalize(light.direction);

    // No attenuation for directional lights
    sample.radiance = light.color * light.intensity;

    // Infinite distance (used for shadow calculations)
    sample.distance = 1e10;

    // Shadow placeholder
    sample.shadow = 1.0;

    return sample;
}

/// Evaluate a point light.
/// Point lights emit uniformly in all directions with distance attenuation.
///
/// @param light: The point light to evaluate
/// @param world_position: World space position of the surface point
/// @returns: LightSample with direction and radiance
fn evaluate_point_light(light: Light, world_position: vec3<f32>) -> LightSample {
    var sample: LightSample;

    // Vector from surface to light
    let to_light = light.position - world_position;
    let distance = length(to_light);

    // Normalized direction toward light
    sample.direction = to_light / max(distance, EPSILON);

    // Distance attenuation (inverse-square with smooth falloff)
    let attenuation = attenuation_point(distance, light.range);

    // Final radiance
    sample.radiance = light.color * light.intensity * attenuation;

    // Store distance for shadow calculations
    sample.distance = distance;

    // Shadow placeholder
    sample.shadow = 1.0;

    return sample;
}

/// Evaluate a spot light.
/// Spot lights emit in a cone with both distance and angular attenuation.
///
/// @param light: The spot light to evaluate
/// @param world_position: World space position of the surface point
/// @returns: LightSample with direction and radiance
fn evaluate_spot_light(light: Light, world_position: vec3<f32>) -> LightSample {
    var sample: LightSample;

    // Vector from surface to light
    let to_light = light.position - world_position;
    let distance = length(to_light);

    // Normalized direction toward light
    sample.direction = to_light / max(distance, EPSILON);

    // Distance attenuation
    let dist_atten = attenuation_point(distance, light.range);

    // Angular attenuation
    // Negative dot because light.direction points away from light
    let cos_angle = dot(-sample.direction, normalize(light.direction));
    let angle_atten = attenuation_spot_angle(cos_angle, light.cos_inner_angle, light.cos_outer_angle);

    // Combined attenuation
    let attenuation = dist_atten * angle_atten;

    // Final radiance
    sample.radiance = light.color * light.intensity * attenuation;

    // Store distance for shadow calculations
    sample.distance = distance;

    // Shadow placeholder
    sample.shadow = 1.0;

    return sample;
}

/// Evaluate a light of any type.
/// Dispatches to the appropriate evaluation function based on light type.
///
/// @param light: The light to evaluate
/// @param world_position: World space position of the surface point
/// @returns: LightSample with direction and radiance
fn evaluate_light(light: Light, world_position: vec3<f32>) -> LightSample {
    switch light.light_type {
        case LIGHT_TYPE_DIRECTIONAL: {
            return evaluate_directional_light(light, world_position);
        }
        case LIGHT_TYPE_POINT: {
            return evaluate_point_light(light, world_position);
        }
        case LIGHT_TYPE_SPOT: {
            return evaluate_spot_light(light, world_position);
        }
        default: {
            // Unknown light type, return zero contribution
            var sample: LightSample;
            sample.direction = vec3<f32>(0.0, 1.0, 0.0);
            sample.radiance = vec3<f32>(0.0);
            sample.distance = 0.0;
            sample.shadow = 0.0;
            return sample;
        }
    }
}

// ============================================================================
// Shadow Sampling (Placeholder)
// ============================================================================

/// Sample shadow for a light at the given world position.
/// This is a placeholder implementation that always returns fully lit.
///
/// @param light_index: Index of the light in the light buffer
/// @param world_position: World space position of the surface point
/// @param light_sample: The evaluated light sample
/// @returns: Shadow factor (1.0 = fully lit, 0.0 = fully shadowed)
fn sample_shadow(light_index: u32, world_position: vec3<f32>, light_sample: LightSample) -> f32 {
    // Placeholder: always return fully lit
    // TODO: Implement shadow mapping in T-MAT-3.4
    return 1.0;
}

// ============================================================================
// Lighting Accumulation
// ============================================================================

/// Result of accumulated lighting calculation.
struct LightingResult {
    /// Accumulated diffuse lighting contribution
    diffuse: vec3<f32>,
    /// Accumulated specular lighting contribution
    specular: vec3<f32>,
}

/// Accumulate lighting from all active lights.
/// Evaluates BRDF for each light and sums contributions.
///
/// @param params: PBR material parameters
/// @param N: Surface normal (world space, normalized)
/// @param V: View direction (world space, normalized, pointing toward camera)
/// @param world_position: World space position of the surface point
/// @param lights: Array of lights to evaluate
/// @param light_count: Number of active lights (0 to MAX_LIGHTS)
/// @returns: Accumulated diffuse and specular lighting
fn accumulate_lighting(
    params: PBRParams,
    N: vec3<f32>,
    V: vec3<f32>,
    world_position: vec3<f32>,
    lights: array<Light, 8>,
    light_count: u32
) -> LightingResult {
    var result: LightingResult;
    result.diffuse = vec3<f32>(0.0);
    result.specular = vec3<f32>(0.0);

    // Compute F0 once for all lights
    let F0 = compute_F0(params.base_color, params.metallic);

    // Clamp light count to maximum
    let count = min(light_count, MAX_LIGHTS);

    // Loop over all active lights
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let light = lights[i];

        // Evaluate light contribution
        var light_sample = evaluate_light(light, world_position);

        // Sample shadow
        light_sample.shadow = sample_shadow(i, world_position, light_sample);

        // Light direction (L points toward light)
        let L = light_sample.direction;

        // Skip if light is behind surface
        let NoL = max(dot(N, L), 0.0);
        if NoL <= 0.0 {
            continue;
        }

        // Compute half-vector
        let H = normalize(V + L);
        let NoV = max(dot(N, V), 0.0);
        let NoH = max(dot(N, H), 0.0);
        let VoH = max(dot(V, H), 0.0);

        // Skip if view is behind surface
        if NoV <= 0.0 {
            continue;
        }

        // Evaluate BRDF terms
        let D = D_GGX(NoH, params.roughness);
        let G = G_Smith_GGX(NoV, NoL, params.roughness);
        let F = F_Schlick(VoH, F0);

        // Specular BRDF (D * G already includes 1/(4*NoV*NoL))
        let specular_brdf = D * G * F;

        // Diffuse BRDF with energy conservation
        // (1 - F) is the amount not reflected as specular
        let kD = (vec3<f32>(1.0) - F) * (1.0 - params.metallic);
        let diffuse_brdf = kD * params.base_color / PI;

        // Final light contribution
        let radiance = light_sample.radiance * light_sample.shadow * NoL;

        result.diffuse = result.diffuse + diffuse_brdf * radiance;
        result.specular = result.specular + specular_brdf * radiance;
    }

    return result;
}

// ============================================================================
// Final Shading Composition
// ============================================================================

/// Compose final shading with ambient occlusion and emissive.
/// Applies AO to indirect lighting and adds emissive term.
///
/// @param lighting: Accumulated lighting result from accumulate_lighting
/// @param params: PBR material parameters (for occlusion and emissive)
/// @param ambient: Ambient/indirect lighting contribution
/// @returns: Final composed color in linear RGB
fn compose_final_shading(
    lighting: LightingResult,
    params: PBRParams,
    ambient: vec3<f32>
) -> vec3<f32> {
    // Apply ambient occlusion to indirect/ambient lighting
    let ambient_occluded = ambient * params.occlusion;

    // Combine direct lighting (diffuse + specular) with ambient
    let direct = lighting.diffuse + lighting.specular;
    let lit = direct + ambient_occluded;

    // Add emissive (not affected by lighting or AO)
    let final_color = lit + params.emissive;

    return final_color;
}

/// Complete lighting evaluation for a surface point.
/// Convenience function that combines light accumulation and final composition.
///
/// @param params: PBR material parameters
/// @param N: Surface normal (world space, normalized)
/// @param V: View direction (world space, normalized, pointing toward camera)
/// @param world_position: World space position of the surface point
/// @param lights: Array of lights to evaluate
/// @param light_count: Number of active lights
/// @param ambient: Ambient/indirect lighting contribution
/// @returns: Final lit color in linear RGB
fn evaluate_all_lighting(
    params: PBRParams,
    N: vec3<f32>,
    V: vec3<f32>,
    world_position: vec3<f32>,
    lights: array<Light, 8>,
    light_count: u32,
    ambient: vec3<f32>
) -> vec3<f32> {
    // Accumulate direct lighting from all lights
    let lighting = accumulate_lighting(params, N, V, world_position, lights, light_count);

    // Compose final shading with AO and emissive
    return compose_final_shading(lighting, params, ambient);
}

// ============================================================================
// Helper Functions for Light Creation
// ============================================================================

/// Create a directional light.
///
/// @param direction: Direction the light is traveling (will be normalized)
/// @param color: Light color in linear RGB
/// @param intensity: Light intensity multiplier
/// @returns: Configured directional light
fn create_directional_light(direction: vec3<f32>, color: vec3<f32>, intensity: f32) -> Light {
    var light: Light;
    light.light_type = LIGHT_TYPE_DIRECTIONAL;
    light.intensity = intensity;
    light.range = 0.0;  // Unused for directional
    light.position = vec3<f32>(0.0);  // Unused for directional
    light.direction = normalize(direction);
    light.color = color;
    light.cos_inner_angle = 1.0;  // Unused for directional
    light.cos_outer_angle = 1.0;  // Unused for directional
    return light;
}

/// Create a point light.
///
/// @param position: World space position of the light
/// @param color: Light color in linear RGB
/// @param intensity: Light intensity multiplier
/// @param range: Maximum range of the light
/// @returns: Configured point light
fn create_point_light(position: vec3<f32>, color: vec3<f32>, intensity: f32, range: f32) -> Light {
    var light: Light;
    light.light_type = LIGHT_TYPE_POINT;
    light.intensity = intensity;
    light.range = range;
    light.position = position;
    light.direction = vec3<f32>(0.0, -1.0, 0.0);  // Unused for point
    light.color = color;
    light.cos_inner_angle = 1.0;  // Unused for point
    light.cos_outer_angle = -1.0;  // Unused for point
    return light;
}

/// Create a spot light.
///
/// @param position: World space position of the light
/// @param direction: Direction the spotlight is pointing (will be normalized)
/// @param color: Light color in linear RGB
/// @param intensity: Light intensity multiplier
/// @param range: Maximum range of the light
/// @param inner_angle: Inner cone angle in radians (no falloff inside)
/// @param outer_angle: Outer cone angle in radians (edge of light cone)
/// @returns: Configured spot light
fn create_spot_light(
    position: vec3<f32>,
    direction: vec3<f32>,
    color: vec3<f32>,
    intensity: f32,
    range: f32,
    inner_angle: f32,
    outer_angle: f32
) -> Light {
    var light: Light;
    light.light_type = LIGHT_TYPE_SPOT;
    light.intensity = intensity;
    light.range = range;
    light.position = position;
    light.direction = normalize(direction);
    light.color = color;
    light.cos_inner_angle = cos(inner_angle);
    light.cos_outer_angle = cos(outer_angle);
    return light;
}
