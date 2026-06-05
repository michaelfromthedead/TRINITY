// SPDX-License-Identifier: MIT
//
// shadow_common.wgsl — Shared shadow mapping utilities (T-LIT-6.3).
//
// Provides common types, bias computation functions, and transform helpers
// used by all shadow filtering modules (PCF, PCSS, ESM, VSM).
//
// This module is designed to be imported by:
// - shadow_pcf.wgsl    (percentage closer filtering)
// - shadow_pcss.wgsl   (percentage closer soft shadows)
// - shadow_esm.wgsl    (exponential shadow maps)
// - shadow_vsm.wgsl    (variance shadow maps)
// - pbr.frag.wgsl      (main PBR fragment shader)

// ============================================================================
// Constants
// ============================================================================

const SHADOW_EPSILON: f32 = 0.00001;
const SHADOW_MAX_BIAS: f32 = 0.01;
const SHADOW_MIN_BIAS: f32 = 0.0001;

// ============================================================================
// Data Structures
// ============================================================================

/// Per-tile shadow atlas information.
/// Used for both tiled shadow atlases and CSM cascade selection.
struct ShadowTileInfo {
    /// Atlas UV offset for this tile (bottom-left corner in UV space).
    uv_offset: vec2<f32>,
    /// Atlas UV scale for this tile (tile size in UV space).
    uv_scale: vec2<f32>,
    /// World-to-light clip space transformation matrix.
    light_space_matrix: mat4x4<f32>,
    /// For CSM: which cascade this tile represents (0-3).
    /// For point/spot lights: light index.
    cascade_index: u32,
    /// PCF kernel size in texels (e.g., 3.0 for 3x3 kernel).
    filter_size: f32,
    /// Constant depth bias applied to all fragments.
    bias_constant: f32,
    /// Slope-scaled bias multiplier (multiplied by surface slope).
    bias_slope: f32,
}

/// Global shadow configuration parameters.
/// Shared across all shadow sampling operations.
struct ShadowConfig {
    /// Shadow atlas dimensions in pixels (width, height).
    atlas_size: vec2<f32>,
    /// Inverse atlas size: 1.0 / atlas_size (for texel calculations).
    texel_size: vec2<f32>,
    /// PCF sample radius in texels (1 = 3x3, 2 = 5x5, etc.).
    pcf_radius: f32,
    /// PCSS blocker search radius in texels.
    pcss_blocker_search_radius: f32,
    /// Virtual light size for PCSS penumbra calculation (world units).
    pcss_light_size: f32,
    /// ESM exponential constant c (typically 32-128, higher = sharper).
    esm_exponent: f32,
}

/// Shadow sample result with debug information.
struct ShadowSampleResult {
    /// Shadow factor in [0, 1]: 0 = fully shadowed, 1 = fully lit.
    factor: f32,
    /// Number of samples taken (for adaptive sampling).
    sample_count: u32,
    /// Average blocker depth (for PCSS).
    avg_blocker_depth: f32,
    /// Penumbra size (for PCSS/soft shadows).
    penumbra_size: f32,
}

// ============================================================================
// Bias Computation Functions
// ============================================================================

/// Computes slope-scaled depth bias based on surface angle to light.
///
/// The bias increases as the surface becomes more grazing relative to the
/// light direction, reducing shadow acne on sloped surfaces.
///
/// Parameters:
/// - normal: World-space surface normal (normalized).
/// - light_dir: Direction TO the light (normalized).
/// - tile: Shadow tile information containing bias parameters.
///
/// Returns: Combined constant and slope-scaled bias value.
fn compute_shadow_bias(
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    tile: ShadowTileInfo
) -> f32 {
    // Cosine of angle between normal and light direction.
    let cos_theta = max(dot(normal, light_dir), SHADOW_EPSILON);

    // Slope factor: tan(theta) = sqrt(1 - cos^2) / cos
    // Clamped to avoid extreme values at grazing angles.
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    let slope = sin_theta / cos_theta;

    // Combine constant bias with slope-scaled bias.
    let total_bias = tile.bias_constant + tile.bias_slope * slope;

    // Clamp to reasonable range to prevent over-biasing.
    return clamp(total_bias, SHADOW_MIN_BIAS, SHADOW_MAX_BIAS);
}

/// Computes adaptive bias based on receiver depth and light distance.
///
/// Useful for perspective shadow maps where bias should vary with depth.
///
/// Parameters:
/// - normal: World-space surface normal (normalized).
/// - light_dir: Direction TO the light (normalized).
/// - receiver_depth: Fragment depth in light space [0, 1].
/// - tile: Shadow tile information.
///
/// Returns: Depth-adaptive bias value.
fn compute_adaptive_bias(
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    receiver_depth: f32,
    tile: ShadowTileInfo
) -> f32 {
    let base_bias = compute_shadow_bias(normal, light_dir, tile);

    // Scale bias based on depth (farther = more bias needed).
    // Using quadratic scaling for perspective projections.
    let depth_scale = 1.0 + receiver_depth * receiver_depth;

    return base_bias * depth_scale;
}

/// Applies normal offset bias by moving the sample point along the normal.
///
/// This technique is particularly effective for reducing peter-panning
/// and shadow acne on curved surfaces.
///
/// Parameters:
/// - world_pos: World-space position to offset.
/// - normal: World-space surface normal (normalized).
/// - light_dir: Direction TO the light (normalized).
/// - texel_size: Shadow map texel size in world units.
///
/// Returns: Offset world position for shadow sampling.
fn apply_normal_offset(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    texel_size: f32
) -> vec3<f32> {
    // Scale offset based on angle between normal and light.
    // Maximum offset when surface is edge-on to light.
    let cos_theta = dot(normal, light_dir);
    let offset_scale = clamp(1.0 - cos_theta, 0.0, 1.0);

    // Offset along normal by scaled texel size.
    return world_pos + normal * offset_scale * texel_size;
}

/// Computes receiver plane depth bias for improved shadow quality.
///
/// This bias considers the orientation of the receiver surface in light space,
/// providing more accurate bias for surfaces at various angles.
///
/// Parameters:
/// - shadow_coord: Shadow map coordinates (xy = UV, z = depth).
/// - texel_size: Shadow map texel size.
/// - normal: World-space surface normal.
/// - light_view_proj: Light's view-projection matrix.
///
/// Returns: Receiver plane bias value.
fn compute_receiver_plane_bias(
    shadow_coord: vec3<f32>,
    texel_size: vec2<f32>,
    normal: vec3<f32>,
    light_view_proj: mat4x4<f32>
) -> f32 {
    // Transform normal to light space.
    let light_normal = normalize((light_view_proj * vec4<f32>(normal, 0.0)).xyz);

    // Compute depth gradient in shadow map space.
    let dx = light_normal.x / max(abs(light_normal.z), SHADOW_EPSILON);
    let dy = light_normal.y / max(abs(light_normal.z), SHADOW_EPSILON);

    // Bias based on gradient and texel size.
    return (abs(dx) * texel_size.x + abs(dy) * texel_size.y) * 0.5;
}

// ============================================================================
// Transform Helpers
// ============================================================================

/// Transforms world position to shadow atlas UV coordinates.
///
/// Parameters:
/// - world_pos: World-space position to transform.
/// - tile: Shadow tile information with transform matrix and atlas offset.
///
/// Returns: vec3 where xy = atlas UV coordinates, z = depth in light space [0, 1].
fn world_to_shadow_uv(
    world_pos: vec3<f32>,
    tile: ShadowTileInfo
) -> vec3<f32> {
    // Transform to light clip space.
    let light_clip = tile.light_space_matrix * vec4<f32>(world_pos, 1.0);

    // Perspective divide.
    let ndc = light_clip.xyz / light_clip.w;

    // Transform from NDC [-1, 1] to UV [0, 1].
    let uv = ndc.xy * 0.5 + 0.5;

    // Apply atlas tile transform to get final atlas UV.
    let atlas_uv = uv * tile.uv_scale + tile.uv_offset;

    return vec3<f32>(atlas_uv, ndc.z);
}

/// Transforms world position to shadow UV with bias applied.
///
/// Combines transformation and bias computation for convenience.
///
/// Parameters:
/// - world_pos: World-space position.
/// - normal: World-space surface normal.
/// - light_dir: Direction TO the light.
/// - tile: Shadow tile information.
///
/// Returns: vec3 where xy = atlas UV, z = biased depth.
fn world_to_shadow_uv_biased(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    light_dir: vec3<f32>,
    tile: ShadowTileInfo
) -> vec3<f32> {
    let shadow_coord = world_to_shadow_uv(world_pos, tile);
    let bias = compute_shadow_bias(normal, light_dir, tile);

    return vec3<f32>(shadow_coord.xy, shadow_coord.z - bias);
}

/// Checks if shadow UV coordinates are valid (inside atlas tile bounds).
///
/// Parameters:
/// - uv: Atlas UV coordinates to check.
/// - tile: Shadow tile information with bounds.
///
/// Returns: true if UV is within the tile bounds, false otherwise.
fn is_valid_shadow_uv(uv: vec2<f32>, tile: ShadowTileInfo) -> bool {
    // Transform back to local tile UV space.
    let local_uv = (uv - tile.uv_offset) / tile.uv_scale;

    // Check bounds with small epsilon for floating-point tolerance.
    let epsilon = vec2<f32>(SHADOW_EPSILON);
    return all(local_uv >= -epsilon) && all(local_uv <= vec2<f32>(1.0) + epsilon);
}

/// Checks if a depth value is valid for shadow comparison.
///
/// Parameters:
/// - depth: Depth value in light space [0, 1].
///
/// Returns: true if depth is within valid range.
fn is_valid_shadow_depth(depth: f32) -> bool {
    return depth >= 0.0 && depth <= 1.0;
}

/// Computes the texel size in world units at a given depth.
///
/// Useful for calculating normal offset bias in world space.
///
/// Parameters:
/// - light_space_matrix: Light's view-projection matrix.
/// - depth: Depth in light space [0, 1].
/// - config: Shadow configuration with atlas size.
///
/// Returns: Approximate texel size in world units.
fn compute_world_texel_size(
    light_space_matrix: mat4x4<f32>,
    depth: f32,
    config: ShadowConfig
) -> f32 {
    // Approximate world-space size of one texel at given depth.
    // For orthographic projections, this is constant.
    // For perspective, it varies with depth.

    // Extract scale from light matrix (simplified approximation).
    let scale_x = length(vec3<f32>(light_space_matrix[0][0], light_space_matrix[1][0], light_space_matrix[2][0]));
    let scale_y = length(vec3<f32>(light_space_matrix[0][1], light_space_matrix[1][1], light_space_matrix[2][1]));

    let avg_scale = (scale_x + scale_y) * 0.5;
    let avg_texel_size = (config.texel_size.x + config.texel_size.y) * 0.5;

    return avg_texel_size / max(avg_scale, SHADOW_EPSILON);
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Converts depth from [0, 1] range to linear depth.
///
/// Parameters:
/// - depth: Non-linear depth value [0, 1].
/// - near: Near plane distance.
/// - far: Far plane distance.
///
/// Returns: Linear depth value.
fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - depth * (far - near));
}

/// Computes shadow fade factor based on distance from camera.
///
/// Used to smoothly fade shadows at the edge of shadow rendering distance.
///
/// Parameters:
/// - distance: Distance from camera to fragment.
/// - fade_start: Distance where fade begins.
/// - fade_end: Distance where shadows fully disappear.
///
/// Returns: Fade factor in [0, 1]: 1 = full shadow, 0 = no shadow.
fn compute_shadow_fade(distance: f32, fade_start: f32, fade_end: f32) -> f32 {
    return 1.0 - saturate((distance - fade_start) / (fade_end - fade_start));
}

/// Computes cascade blend factor for smooth CSM transitions.
///
/// Parameters:
/// - view_depth: View-space depth of fragment.
/// - cascade_split: Split depth for current cascade.
/// - blend_range: Range over which to blend between cascades.
///
/// Returns: Blend factor in [0, 1] for mixing with next cascade.
fn compute_cascade_blend(view_depth: f32, cascade_split: f32, blend_range: f32) -> f32 {
    let blend_start = cascade_split - blend_range;
    return saturate((view_depth - blend_start) / blend_range);
}

/// Generates a 2D Poisson disk sample offset.
///
/// For use in PCF/PCSS sampling to reduce banding artifacts.
///
/// Parameters:
/// - index: Sample index.
/// - seed: Random seed (e.g., based on screen position).
///
/// Returns: 2D offset in [-1, 1] range.
fn poisson_disk_sample(index: u32, seed: f32) -> vec2<f32> {
    // Pre-computed 16-sample Poisson disk.
    // Using let for array as const arrays need explicit type in WGSL.
    let samples = array<vec2<f32>, 16>(
        vec2<f32>(-0.94201624, -0.39906216),
        vec2<f32>( 0.94558609, -0.76890725),
        vec2<f32>(-0.09418410, -0.92938870),
        vec2<f32>( 0.34495938,  0.29387760),
        vec2<f32>(-0.91588581,  0.45771432),
        vec2<f32>(-0.81544232, -0.87912464),
        vec2<f32>(-0.38277543,  0.27676845),
        vec2<f32>( 0.97484398,  0.75648379),
        vec2<f32>( 0.44323325, -0.97511554),
        vec2<f32>( 0.53742981, -0.47373420),
        vec2<f32>(-0.26496911, -0.41893023),
        vec2<f32>( 0.79197514,  0.19090188),
        vec2<f32>(-0.24188840,  0.99706507),
        vec2<f32>(-0.81409955,  0.91437590),
        vec2<f32>( 0.19984126,  0.78641367),
        vec2<f32>( 0.14383161, -0.14100790)
    );

    let idx = index % 16u;
    let sample = samples[idx];

    // Rotate sample based on seed for randomization.
    let angle = seed * 6.283185307;
    let cos_a = cos(angle);
    let sin_a = sin(angle);

    return vec2<f32>(
        sample.x * cos_a - sample.y * sin_a,
        sample.x * sin_a + sample.y * cos_a
    );
}

/// Generates a pseudo-random value from screen coordinates.
///
/// Parameters:
/// - screen_pos: Screen-space position (e.g., from @builtin(position)).
///
/// Returns: Pseudo-random value in [0, 1].
fn random_from_screen(screen_pos: vec2<f32>) -> f32 {
    // Simple hash function for randomization.
    let dot_val = dot(screen_pos, vec2<f32>(12.9898, 78.233));
    return fract(sin(dot_val) * 43758.5453);
}

/// Interleaved gradient noise for dithered shadow sampling.
///
/// Provides temporally stable noise pattern that works well with TAA.
///
/// Parameters:
/// - screen_pos: Screen-space position.
///
/// Returns: Noise value in [0, 1].
fn interleaved_gradient_noise(screen_pos: vec2<f32>) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(screen_pos, magic.xy)));
}
