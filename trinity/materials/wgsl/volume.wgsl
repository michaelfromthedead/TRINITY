// =============================================================================
// Volume Rendering Functions for TRINITY Material System
// =============================================================================
// Task: T-MAT-5.4 Volume Domain Implementation
// Gap: S5-G4
// Dependency: T-MAT-3.4 (pipeline integration)
//
// This module provides volume rendering functions for fog, clouds, and smoke:
//   - Ray-AABB intersection for volume bounds
//   - Ray marching with adaptive step size
//   - Single-scattering integration (in-scattered light)
//   - Henyey-Greenstein phase function
//   - Beer's law transmittance
//
// References:
//   - Pharr, Humphreys "Physically Based Rendering" Ch. 11-12
//   - Fong et al. "Production Volume Rendering" (SIGGRAPH 2017)
//   - Henyey & Greenstein "Diffuse radiation in the galaxy" (1941)
// =============================================================================

// Mathematical constants
const PI: f32 = 3.14159265359;
const INV_4PI: f32 = 0.07957747154;
const EPSILON: f32 = 0.0001;

// Volume rendering constants
const MAX_MARCH_STEPS: u32 = 128u;
const MIN_STEP_SIZE: f32 = 0.001;
const MAX_STEP_SIZE: f32 = 0.5;
const TRANSMITTANCE_THRESHOLD: f32 = 0.001;

// =============================================================================
// Volume Parameters Structure
// =============================================================================

/// Volume rendering parameters for fog, clouds, and smoke effects.
/// Describes the optical properties of a participating medium.
struct VolumeParams {
    /// Density scale multiplier for the volume
    density_scale: f32,
    /// Absorption coefficient (how much light is absorbed per unit distance)
    /// Higher values = darker, more opaque medium
    absorption: vec3<f32>,
    /// Scattering coefficient (how much light scatters per unit distance)
    /// Higher values = brighter medium due to in-scattered light
    scattering: vec3<f32>,
    /// Phase function asymmetry parameter (Henyey-Greenstein g)
    /// Range: [-1, 1]. 0 = isotropic, >0 = forward scattering, <0 = back scattering
    phase_g: f32,
    /// Emissive contribution (self-illumination of the volume)
    emission: vec3<f32>,
    /// Maximum ray march distance
    max_distance: f32,
}

/// Create default volume parameters for homogeneous fog.
fn default_volume_params() -> VolumeParams {
    var params: VolumeParams;
    params.density_scale = 1.0;
    params.absorption = vec3<f32>(0.01, 0.01, 0.01);
    params.scattering = vec3<f32>(0.1, 0.1, 0.1);
    params.phase_g = 0.0;  // Isotropic
    params.emission = vec3<f32>(0.0);
    params.max_distance = 100.0;
    return params;
}

// =============================================================================
// Ray-AABB Intersection
// =============================================================================

/// Result of ray-AABB intersection test.
struct RayAABBResult {
    /// Whether the ray intersects the AABB
    hit: bool,
    /// Distance to near intersection (entry point)
    t_near: f32,
    /// Distance to far intersection (exit point)
    t_far: f32,
}

/// Compute ray-AABB intersection using the slab method.
/// Returns entry and exit distances along the ray.
///
/// @param ray_origin: Ray origin point
/// @param ray_dir: Normalized ray direction
/// @param aabb_min: Minimum corner of axis-aligned bounding box
/// @param aabb_max: Maximum corner of axis-aligned bounding box
/// @returns: RayAABBResult with hit status and t_near/t_far
fn ray_aabb_intersect(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>
) -> RayAABBResult {
    var result: RayAABBResult;
    result.hit = false;
    result.t_near = 0.0;
    result.t_far = 0.0;

    // Compute inverse direction for efficiency
    let inv_dir = 1.0 / ray_dir;

    // Compute intersection with each slab
    let t1 = (aabb_min - ray_origin) * inv_dir;
    let t2 = (aabb_max - ray_origin) * inv_dir;

    // Handle negative directions
    let t_min = min(t1, t2);
    let t_max = max(t1, t2);

    // Find the largest entry and smallest exit
    result.t_near = max(max(t_min.x, t_min.y), t_min.z);
    result.t_far = min(min(t_max.x, t_max.y), t_max.z);

    // Clamp t_near to avoid intersections behind the ray
    result.t_near = max(result.t_near, 0.0);

    // Check for valid intersection
    result.hit = result.t_far >= result.t_near && result.t_far > 0.0;

    return result;
}

// =============================================================================
// Density Sampling
// =============================================================================

/// Sample density from a 3D texture at world position.
/// Assumes texture coordinates are normalized to volume bounds.
///
/// @param pos: World-space position
/// @param volume_min: Minimum corner of volume bounds
/// @param volume_max: Maximum corner of volume bounds
/// @param density_texture: 3D density texture
/// @param density_sampler: Texture sampler
/// @returns: Density value at position (0 = empty, 1 = full density)
fn sample_density_texture(
    pos: vec3<f32>,
    volume_min: vec3<f32>,
    volume_max: vec3<f32>,
    density_texture: texture_3d<f32>,
    density_sampler: sampler
) -> f32 {
    // Compute normalized UVW coordinates
    let uvw = (pos - volume_min) / (volume_max - volume_min);

    // Check bounds
    if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
        return 0.0;
    }

    // Sample 3D texture (R channel contains density)
    return textureSample(density_texture, density_sampler, uvw).r;
}

/// Sample procedural density using exponential height fog.
/// Common for atmospheric effects and ground fog.
///
/// @param pos: World-space position
/// @param base_height: Height at which fog is densest
/// @param falloff: Exponential falloff rate (higher = thinner fog at height)
/// @param density_scale: Overall density multiplier
/// @returns: Density value at position
fn sample_density_exponential_fog(
    pos: vec3<f32>,
    base_height: f32,
    falloff: f32,
    density_scale: f32
) -> f32 {
    let height_above_base = max(pos.y - base_height, 0.0);
    return density_scale * exp(-falloff * height_above_base);
}

/// Sample procedural density using distance-based fog.
/// Fog that gets denser further from a reference point.
///
/// @param pos: World-space position
/// @param fog_start: Distance at which fog begins
/// @param fog_end: Distance at which fog reaches full density
/// @param reference_point: Point from which distance is measured (usually camera)
/// @returns: Density value at position
fn sample_density_distance_fog(
    pos: vec3<f32>,
    fog_start: f32,
    fog_end: f32,
    reference_point: vec3<f32>
) -> f32 {
    let dist = distance(pos, reference_point);
    return saturate((dist - fog_start) / (fog_end - fog_start));
}

// =============================================================================
// Phase Functions
// =============================================================================

/// Henyey-Greenstein phase function for anisotropic scattering.
/// Models the angular distribution of scattered light.
///
/// Formula: p(cos_theta) = (1 - g^2) / (4 * PI * (1 + g^2 - 2*g*cos_theta)^1.5)
///
/// @param cos_theta: Cosine of angle between incident and scattered directions
/// @param g: Asymmetry parameter in [-1, 1]
///           g > 0: Forward scattering (common for fog, clouds)
///           g = 0: Isotropic scattering
///           g < 0: Back scattering
/// @returns: Phase function value (probability density)
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;

    // Handle near-isotropic case to avoid numerical issues
    if abs(g) < EPSILON {
        return INV_4PI;
    }

    return (1.0 - g2) * INV_4PI / pow(denom, 1.5);
}

/// Two-lobe Henyey-Greenstein phase function.
/// Blends forward and backward scattering lobes for more realistic clouds.
///
/// @param cos_theta: Cosine of scattering angle
/// @param g_forward: Asymmetry for forward lobe (typically 0.7-0.9)
/// @param g_backward: Asymmetry for backward lobe (typically -0.2 to -0.5)
/// @param blend: Blend factor (0 = all backward, 1 = all forward)
/// @returns: Phase function value
fn henyey_greenstein_two_lobe(
    cos_theta: f32,
    g_forward: f32,
    g_backward: f32,
    blend: f32
) -> f32 {
    let forward = henyey_greenstein(cos_theta, g_forward);
    let backward = henyey_greenstein(cos_theta, g_backward);
    return mix(backward, forward, blend);
}

/// Isotropic phase function (uniform scattering in all directions).
/// Simplest model, used as baseline or for dense media.
///
/// @returns: Constant phase function value (1 / 4*PI)
fn phase_isotropic() -> f32 {
    return INV_4PI;
}

/// Rayleigh phase function for small particle scattering.
/// Used for atmospheric scattering (blue sky effect).
///
/// @param cos_theta: Cosine of scattering angle
/// @returns: Phase function value
fn phase_rayleigh(cos_theta: f32) -> f32 {
    // Rayleigh: (3 / 16*PI) * (1 + cos^2(theta))
    return 0.05968310365 * (1.0 + cos_theta * cos_theta);
}

// =============================================================================
// Transmittance (Beer's Law)
// =============================================================================

/// Compute transmittance using Beer-Lambert law.
/// T = exp(-integral(extinction * density * ds))
///
/// @param extinction: Extinction coefficient (absorption + scattering)
/// @param optical_depth: Integrated density * distance along path
/// @returns: Transmittance in [0, 1] (1 = fully transparent, 0 = fully opaque)
fn beer_lambert_transmittance(
    extinction: vec3<f32>,
    optical_depth: f32
) -> vec3<f32> {
    return exp(-extinction * optical_depth);
}

/// Compute transmittance for a homogeneous medium over a distance.
///
/// @param extinction: Extinction coefficient (absorption + scattering)
/// @param density: Constant density of the medium
/// @param distance: Travel distance through the medium
/// @returns: Transmittance
fn transmittance_homogeneous(
    extinction: vec3<f32>,
    density: f32,
    distance: f32
) -> vec3<f32> {
    return beer_lambert_transmittance(extinction, density * distance);
}

// =============================================================================
// In-Scattering Integration
// =============================================================================

/// Compute single-scattering contribution from a directional light.
/// Approximates light scattered toward the viewer at a sample point.
///
/// @param sample_pos: World position of the volume sample
/// @param view_dir: Direction toward camera (normalized)
/// @param light_dir: Direction toward light (normalized)
/// @param light_color: Light color and intensity
/// @param params: Volume parameters
/// @param density: Local density at sample position
/// @returns: In-scattered radiance contribution
fn integrate_inscattered_directional(
    sample_pos: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    light_color: vec3<f32>,
    params: VolumeParams,
    density: f32
) -> vec3<f32> {
    // Compute scattering angle
    let cos_theta = dot(-view_dir, light_dir);

    // Evaluate phase function
    let phase = henyey_greenstein(cos_theta, params.phase_g);

    // Scattering coefficient (how much light scatters per unit distance)
    // Multiplied by density for heterogeneous media
    let sigma_s = params.scattering * density * params.density_scale;

    // In-scattered radiance: L_scat = sigma_s * phase * L_light
    return sigma_s * phase * light_color;
}

/// Compute single-scattering contribution from a point light.
/// Includes distance attenuation.
///
/// @param sample_pos: World position of the volume sample
/// @param view_dir: Direction toward camera (normalized)
/// @param light_pos: World position of point light
/// @param light_color: Light color
/// @param light_intensity: Light intensity
/// @param light_range: Maximum light range (for attenuation)
/// @param params: Volume parameters
/// @param density: Local density at sample position
/// @returns: In-scattered radiance contribution
fn integrate_inscattered_point(
    sample_pos: vec3<f32>,
    view_dir: vec3<f32>,
    light_pos: vec3<f32>,
    light_color: vec3<f32>,
    light_intensity: f32,
    light_range: f32,
    params: VolumeParams,
    density: f32
) -> vec3<f32> {
    // Direction and distance to light
    let to_light = light_pos - sample_pos;
    let dist_to_light = length(to_light);
    let light_dir = to_light / max(dist_to_light, EPSILON);

    // Distance attenuation (inverse square with range falloff)
    let attenuation = light_intensity / (dist_to_light * dist_to_light + 1.0);
    let range_factor = saturate(1.0 - dist_to_light / light_range);

    // Scattering angle
    let cos_theta = dot(-view_dir, light_dir);

    // Phase function
    let phase = henyey_greenstein(cos_theta, params.phase_g);

    // Scattering coefficient
    let sigma_s = params.scattering * density * params.density_scale;

    return sigma_s * phase * light_color * attenuation * range_factor;
}

// =============================================================================
// Ray Marching
// =============================================================================

/// Result of volume ray marching.
struct VolumeResult {
    /// Accumulated in-scattered color
    color: vec3<f32>,
    /// Final transmittance (how much of the background is visible)
    transmittance: vec3<f32>,
    /// Number of steps taken
    steps: u32,
}

/// Compute adaptive step size based on local density.
/// Uses smaller steps in dense regions for accuracy.
///
/// @param density: Local density at current position
/// @param base_step: Base step size for empty regions
/// @returns: Adapted step size
fn adaptive_step_size(density: f32, base_step: f32) -> f32 {
    // Smaller steps when density is high
    let density_factor = 1.0 / (1.0 + density * 10.0);
    return clamp(base_step * density_factor, MIN_STEP_SIZE, MAX_STEP_SIZE);
}

/// March a ray through a homogeneous volume (constant density).
/// Simplified version for uniform fog effects.
///
/// @param ray_origin: Ray origin point
/// @param ray_dir: Normalized ray direction
/// @param t_near: Entry distance into volume
/// @param t_far: Exit distance from volume
/// @param density: Constant density of the volume
/// @param params: Volume parameters
/// @param light_dir: Direction toward primary light
/// @param light_color: Primary light color
/// @returns: VolumeResult with accumulated color and transmittance
fn march_volume_homogeneous(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    t_near: f32,
    t_far: f32,
    density: f32,
    params: VolumeParams,
    light_dir: vec3<f32>,
    light_color: vec3<f32>
) -> VolumeResult {
    var result: VolumeResult;
    result.color = vec3<f32>(0.0);
    result.transmittance = vec3<f32>(1.0);
    result.steps = 0u;

    // Path length through volume
    let path_length = min(t_far - t_near, params.max_distance);

    // Extinction coefficient
    let extinction = params.absorption + params.scattering;

    // Transmittance through entire volume
    result.transmittance = transmittance_homogeneous(
        extinction * params.density_scale,
        density,
        path_length
    );

    // Single-scattering at midpoint (approximation for homogeneous media)
    let mid_point = ray_origin + ray_dir * (t_near + path_length * 0.5);
    let view_dir = -ray_dir;

    // In-scattered contribution
    let inscattered = integrate_inscattered_directional(
        mid_point,
        view_dir,
        light_dir,
        light_color,
        params,
        density
    );

    // Integrate in-scattering: L = inscattered * (1 - T) / extinction
    // This is the closed-form solution for constant density
    let avg_extinction = (extinction.x + extinction.y + extinction.z) / 3.0;
    let one_minus_T = vec3<f32>(1.0) - result.transmittance;

    if avg_extinction > EPSILON {
        result.color = inscattered * one_minus_T / (extinction * params.density_scale * density + EPSILON);
    }

    // Add emission
    result.color = result.color + params.emission * path_length * density;

    result.steps = 1u;
    return result;
}

/// March a ray through a heterogeneous volume using numerical integration.
/// Full ray marching with adaptive step size for variable density.
///
/// @param ray_origin: Ray origin point
/// @param ray_dir: Normalized ray direction
/// @param t_near: Entry distance into volume
/// @param t_far: Exit distance from volume
/// @param params: Volume parameters
/// @param light_dir: Direction toward primary light
/// @param light_color: Primary light color
/// @param sample_density_fn: Function to sample density at a position
/// @returns: VolumeResult with accumulated color and transmittance
fn march_volume(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    t_near: f32,
    t_far: f32,
    params: VolumeParams,
    light_dir: vec3<f32>,
    light_color: vec3<f32>,
    base_density: f32
) -> VolumeResult {
    var result: VolumeResult;
    result.color = vec3<f32>(0.0);
    result.transmittance = vec3<f32>(1.0);
    result.steps = 0u;

    // Extinction coefficient
    let extinction = params.absorption + params.scattering;

    // Clamp march distance
    let max_t = min(t_far, t_near + params.max_distance);

    // Initial step size
    let base_step = (max_t - t_near) / f32(MAX_MARCH_STEPS);
    var t = t_near;

    let view_dir = -ray_dir;

    // Ray marching loop
    for (var i = 0u; i < MAX_MARCH_STEPS; i = i + 1u) {
        if t >= max_t {
            break;
        }

        // Early termination if transmittance is negligible
        let avg_T = (result.transmittance.x + result.transmittance.y + result.transmittance.z) / 3.0;
        if avg_T < TRANSMITTANCE_THRESHOLD {
            result.transmittance = vec3<f32>(0.0);
            break;
        }

        // Current sample position
        let pos = ray_origin + ray_dir * t;

        // Sample density (using base_density as a placeholder for procedural/texture)
        // In real usage, this would be sample_density_texture or sample_density_exponential_fog
        let density = base_density;

        // Adaptive step size
        let step = adaptive_step_size(density, base_step);
        let actual_step = min(step, max_t - t);

        // Optical depth for this segment
        let optical_depth = density * params.density_scale * actual_step;

        // Transmittance for this segment
        let segment_T = beer_lambert_transmittance(extinction, optical_depth);

        // In-scattered light at this sample
        let inscattered = integrate_inscattered_directional(
            pos,
            view_dir,
            light_dir,
            light_color,
            params,
            density
        );

        // Accumulate color: L_i = inscattered * T * (1 - segment_T)
        // This is the standard volume rendering equation
        let one_minus_segment_T = vec3<f32>(1.0) - segment_T;
        result.color = result.color + result.transmittance * inscattered * actual_step;

        // Add emission contribution
        result.color = result.color + result.transmittance * params.emission * density * actual_step;

        // Update total transmittance
        result.transmittance = result.transmittance * segment_T;

        // Advance ray
        t = t + actual_step;
        result.steps = result.steps + 1u;
    }

    return result;
}

// =============================================================================
// Volume Domain Evaluation
// =============================================================================

/// Evaluate volume rendering for a fragment.
/// Main entry point for volume domain shading.
///
/// @param ray_origin: Camera/view position
/// @param ray_dir: Normalized view ray direction
/// @param aabb_min: Volume bounds minimum
/// @param aabb_max: Volume bounds maximum
/// @param params: Volume parameters
/// @param lights: Array of light contributions
/// @param light_count: Number of active lights
/// @returns: RGBA color with premultiplied alpha
fn evaluate_volume(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    params: VolumeParams,
    light_dir: vec3<f32>,
    light_color: vec3<f32>,
    density: f32
) -> vec4<f32> {
    // Ray-AABB intersection
    let isect = ray_aabb_intersect(ray_origin, ray_dir, aabb_min, aabb_max);

    if !isect.hit {
        // No intersection with volume bounds
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // March through volume
    let vol_result = march_volume(
        ray_origin,
        ray_dir,
        isect.t_near,
        isect.t_far,
        params,
        light_dir,
        light_color,
        density
    );

    // Compute alpha from transmittance
    // Alpha = 1 - average(T) represents how much the volume obscures background
    let avg_T = (vol_result.transmittance.x + vol_result.transmittance.y + vol_result.transmittance.z) / 3.0;
    let alpha = 1.0 - avg_T;

    // Premultiply alpha for correct compositing
    return vec4<f32>(vol_result.color * alpha, alpha);
}

/// Evaluate homogeneous fog volume (optimized path).
/// Use this for simple distance-based or height-based fog.
///
/// @param ray_origin: Camera position
/// @param ray_dir: View ray direction
/// @param max_distance: Maximum fog distance
/// @param params: Volume parameters
/// @param light_dir: Primary light direction
/// @param light_color: Primary light color
/// @param density: Fog density
/// @returns: RGBA fog color with alpha
fn evaluate_homogeneous_fog(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    max_distance: f32,
    params: VolumeParams,
    light_dir: vec3<f32>,
    light_color: vec3<f32>,
    density: f32
) -> vec4<f32> {
    let vol_result = march_volume_homogeneous(
        ray_origin,
        ray_dir,
        0.0,
        max_distance,
        density,
        params,
        light_dir,
        light_color
    );

    let avg_T = (vol_result.transmittance.x + vol_result.transmittance.y + vol_result.transmittance.z) / 3.0;
    let alpha = 1.0 - avg_T;

    return vec4<f32>(vol_result.color, alpha);
}
