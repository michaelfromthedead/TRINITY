// SPDX-License-Identifier: MIT
//
// shadow_csm.wgsl — Cascaded Shadow Map sampling (T-BRG-6.4).
//
// Provides shadow sampling functions for PBR fragment shaders.
// Selects the appropriate CSM cascade based on view-space depth,
// applies PCF filtering with configurable kernel size and depth biases.
//
// Imported/consumed by pbr.frag.wgsl during the directional light pass.

// ── Data structures ──

struct CascadeData {
    light_view_proj: mat4x4<f32>,
    split_depth: f32,
    shadow_map_index: u32,
    _pad0: f32,
    _pad1: f32,
}

// ── Constants ──

const SHADOW_MAP_RESOLUTION: f32 = 2048.0;

// ── Cascade selection ──

// T-LIT-4.4: Cascade selection result with blend information for soft transitions.
struct CascadeSelection {
    primary_idx: u32,      // Primary cascade index
    secondary_idx: u32,    // Secondary cascade for blending (if in blend zone)
    blend_factor: f32,     // 0.0 = use primary only, 1.0 = use secondary only
}

/// Selects the cascade index based on view-space depth.
/// Cascade split_depths are ordered from nearest (index 0) to farthest (index 3).
fn select_cascade(
    world_pos: vec3<f32>,
    view_matrix: mat4x4<f32>,
    cascades: array<CascadeData, 4>,
) -> u32 {
    let view_pos = view_matrix * vec4<f32>(world_pos, 1.0);
    let view_depth = abs(view_pos.z);

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        if view_depth < cascades[i].split_depth {
            return i;
        }
    }
    return 3u;
}

/// T-LIT-4.4: Selects cascade with blend information for soft transitions.
/// Returns primary cascade index and blend factor towards next cascade.
///
/// Parameters:
/// - world_pos: World-space position
/// - view_matrix: Camera view matrix
/// - cascades: Array of 4 cascade descriptors
/// - blend_range: Distance over which to blend between cascades (default 2.0)
///
/// Returns: CascadeSelection with primary/secondary indices and blend factor.
fn select_cascade_blended(
    world_pos: vec3<f32>,
    view_matrix: mat4x4<f32>,
    cascades: array<CascadeData, 4>,
    blend_range: f32,
) -> CascadeSelection {
    let view_pos = view_matrix * vec4<f32>(world_pos, 1.0);
    let view_depth = abs(view_pos.z);

    var result: CascadeSelection;
    result.blend_factor = 0.0;
    result.secondary_idx = 3u;

    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let split_depth = cascades[i].split_depth;
        if view_depth < split_depth {
            result.primary_idx = i;

            // Check if within blend range of cascade boundary
            if blend_range > 0.0 && i < 3u {
                let blend_start = split_depth - blend_range;
                if view_depth > blend_start {
                    // Linear interpolation within blend zone
                    result.blend_factor = (view_depth - blend_start) / blend_range;
                    result.secondary_idx = i + 1u;
                }
            }
            return result;
        }
    }
    result.primary_idx = 3u;
    return result;
}

// ── Biased shadow UV computation ──

/// Transforms world position to shadow UV with depth bias applied.
/// Returns (shadow_uv, biased_depth) where shadow_uv.z is the cascade index.
fn compute_shadow_uv(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    cascade: CascadeData,
    bias_params: vec4<f32>, // (constant_bias, slope_bias, normal_bias, 0)
) -> vec4<f32> {
    let light_clip = cascade.light_view_proj * vec4<f32>(world_pos, 1.0);

    if light_clip.w <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0); // sentinel: behind light
    }

    var ndc = light_clip.xyz / light_clip.w;

    // Clamp to valid NDC range.
    if any(ndc.xy < vec2<f32>(-1.0)) || any(ndc.xy > vec2<f32>(1.0)) {
        return vec4<f32>(0.0, 0.0, 0.0, -2.0); // sentinel: outside frustum
    }
    if ndc.z < 0.0 || ndc.z > 1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, -2.0);
    }

    // Apply depth biases.
    let cos_theta = clamp(dot(normal, light_dir), 0.0, 1.0);
    let slope_bias = bias_params.y * tan(acos(min(cos_theta, 0.9999)));
    let constant_bias = bias_params.x;
    let normal_bias = bias_params.z * (1.0 - cos_theta);

    ndc.z = ndc.z - constant_bias - slope_bias - normal_bias;
    ndc.z = clamp(ndc.z, 0.0, 1.0);

    // Transform to UV space.
    let uv = vec2<f32>(ndc.xy * 0.5 + 0.5);

    return vec4<f32>(uv, ndc.z, f32(cascade.shadow_map_index));
}

// ── PCF shadow sampling ──

/// Samples the shadow map array with Percentage Closer Filtering.
/// Returns shadow factor in [0, 1] where 1 = fully lit, 0 = fully shadowed.
fn pcf_sample(
    shadow_uv: vec4<f32>, // xy=UV, z=depth, w=array_layer
    shadow_maps: texture_depth_2d_array,
    shadow_sampler: sampler_comparison,
    pcf_radius: u32,
) -> f32 {
    let texel_size = 1.0 / SHADOW_MAP_RESOLUTION;
    let layer = i32(shadow_uv.w);
    let depth = shadow_uv.z;

    let kernel_size = 2 * pcf_radius + 1u;
    var shadow: f32 = 0.0;

    for (var x: u32 = 0u; x < kernel_size; x = x + 1u) {
        for (var y: u32 = 0u; y < kernel_size; y = y + 1u) {
            let offset = vec2<f32>(
                f32(x) - f32(pcf_radius),
                f32(y) - f32(pcf_radius),
            ) * texel_size;

            let sample_uv = shadow_uv.xy + offset;
            shadow = shadow + textureSampleCompare(
                shadow_maps, shadow_sampler,
                sample_uv, layer, depth
            );
        }
    }

    let sample_count = f32(kernel_size * kernel_size);
    return shadow / sample_count;
}

// ── Main shadow factor entry point ──

/// Computes the shadow factor for a given world-space position.
///
/// Parameters:
/// - world_pos: world-space surface position
/// - normal: world-space surface normal (for normal bias)
/// - light_dir: direction TO the light (normalized)
/// - view_matrix: camera view matrix (for cascade selection)
/// - cascades: array of 4 cascade descriptors
/// - shadow_maps: texture_depth_2d_array with one layer per cascade
/// - shadow_sampler: sampler_comparison for depth comparison
/// - bias_params: (constant_bias, slope_bias, normal_bias, 0)
/// - pcf_radius: PCF kernel radius (1 = 3x3, 2 = 5x5, etc.)
///
/// Returns shadow factor in [0, 1] (1 = fully lit).
fn compute_shadow_factor(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    view_matrix: mat4x4<f32>,
    cascades: array<CascadeData, 4>,
    shadow_maps: texture_depth_2d_array,
    shadow_sampler: sampler_comparison,
    bias_params: vec4<f32>,
    pcf_radius: u32,
) -> f32 {
    let cascade_idx = select_cascade(world_pos, view_matrix, cascades);
    let cascade = cascades[cascade_idx];

    let shadow_uv = compute_shadow_uv(world_pos, normal, light_dir, cascade, bias_params);

    // Check for sentinel values.
    if shadow_uv.w < 0.0 {
        return 1.0; // outside all cascades — fully lit
    }

    return pcf_sample(shadow_uv, shadow_maps, shadow_sampler, pcf_radius);
}

/// T-LIT-4.4: Computes blended shadow factor with soft cascade transitions.
///
/// This version smoothly blends between cascades when the fragment is near
/// a cascade boundary, eliminating visible seams.
///
/// Parameters:
/// - world_pos: world-space surface position
/// - normal: world-space surface normal (for normal bias)
/// - light_dir: direction TO the light (normalized)
/// - view_matrix: camera view matrix (for cascade selection)
/// - cascades: array of 4 cascade descriptors
/// - shadow_maps: texture_depth_2d_array with one layer per cascade
/// - shadow_sampler: sampler_comparison for depth comparison
/// - bias_params: (constant_bias, slope_bias, normal_bias, 0)
/// - pcf_radius: PCF kernel radius (1 = 3x3, 2 = 5x5, etc.)
/// - blend_range: distance in world units over which to blend (default 2.0)
///
/// Returns shadow factor in [0, 1] (1 = fully lit).
fn compute_shadow_factor_blended(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    view_matrix: mat4x4<f32>,
    cascades: array<CascadeData, 4>,
    shadow_maps: texture_depth_2d_array,
    shadow_sampler: sampler_comparison,
    bias_params: vec4<f32>,
    pcf_radius: u32,
    blend_range: f32,
) -> f32 {
    let selection = select_cascade_blended(world_pos, view_matrix, cascades, blend_range);

    // Sample primary cascade.
    let primary_cascade = cascades[selection.primary_idx];
    let primary_uv = compute_shadow_uv(world_pos, normal, light_dir, primary_cascade, bias_params);

    // Check for sentinel values.
    if primary_uv.w < 0.0 {
        return 1.0; // outside all cascades — fully lit
    }

    let primary_shadow = pcf_sample(primary_uv, shadow_maps, shadow_sampler, pcf_radius);

    // If not in blend zone, return primary result only (fast path).
    if selection.blend_factor <= 0.0 {
        return primary_shadow;
    }

    // Sample secondary cascade for blending.
    let secondary_cascade = cascades[selection.secondary_idx];
    let secondary_uv = compute_shadow_uv(world_pos, normal, light_dir, secondary_cascade, bias_params);

    // If secondary is invalid, use primary only.
    if secondary_uv.w < 0.0 {
        return primary_shadow;
    }

    let secondary_shadow = pcf_sample(secondary_uv, shadow_maps, shadow_sampler, pcf_radius);

    // Linear interpolation between cascades for smooth transition.
    return mix(primary_shadow, secondary_shadow, selection.blend_factor);
}
