// SPDX-License-Identifier: MIT
//
// ssr_fade.wgsl - SSR Fade Functions Library (T-GIR-P4.3).
//
// This file provides fade functions for screen-space reflections to handle:
// - Screen edge fade: Reduces reflection strength near screen edges
// - Distance fade: Reduces reflection strength for distant hits
// - Roughness fade: Reduces reflection strength for rough surfaces
//
// All fade functions use smooth Hermite interpolation (smoothstep) for
// natural-looking falloff. The functions can be combined multiplicatively
// for the final fade factor.
//
// This file can be included in other shaders or used standalone for
// testing fade curve behavior.

// ---------------------------------------------------------------------------
// Fade Configuration Struct
// ---------------------------------------------------------------------------

/// Fade parameters for SSR falloff control.
/// Matches the Rust SSRFadeConfig struct exactly.
struct SSRFadeConfig {
    // Screen edge fade
    edge_fade_start: f32,       // Start fade at this distance from center (0.8)
    edge_fade_end: f32,         // Complete fade at this distance (1.0)

    // Distance fade
    distance_fade_start: f32,   // Start fade at this world distance (50m)
    distance_fade_end: f32,     // Complete fade at this distance (100m)

    // Roughness fade
    roughness_fade_start: f32,  // Start fade at this roughness (0.5)
    roughness_fade_end: f32,    // Complete fade at this roughness (0.8)

    // Padding for GPU alignment
    _pad: vec2<f32>,
}

// ---------------------------------------------------------------------------
// Smoothstep Function
// ---------------------------------------------------------------------------

/// Hermite interpolation (smoothstep).
///
/// Returns:
/// - 0.0 when x <= edge0
/// - 1.0 when x >= edge1
/// - Smooth S-curve interpolation between
///
/// The smoothstep function has the following properties:
/// - C1 continuous (derivative is continuous)
/// - Derivative is 0 at edge0 and edge1 (no sudden changes)
/// - More natural-looking than linear interpolation
///
/// Mathematical formula:
///   t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
///   result = t * t * (3 - 2 * t)
fn smoothstep_fade(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

/// Smoother version with quintic polynomial (smootherstep).
///
/// C2 continuous - second derivative is also continuous.
/// Even smoother transitions but slightly more expensive.
///
/// Mathematical formula:
///   t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
///   result = t * t * t * (t * (t * 6 - 15) + 10)
fn smootherstep_fade(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// ---------------------------------------------------------------------------
// Edge Fade Functions
// ---------------------------------------------------------------------------

/// Compute edge fade based on screen UV coordinates.
///
/// Returns 1.0 at screen center, smoothly fading to 0.0 at screen edges.
/// This handles the common case where SSR rays exit the screen and we
/// need to gracefully fall back to other reflection techniques.
///
/// Arguments:
/// - screen_uv: Screen-space UV coordinates (0.0-1.0, 0.0-1.0)
/// - config: Fade configuration parameters
///
/// Returns:
/// - Fade factor in range [0.0, 1.0]
fn compute_edge_fade(screen_uv: vec2<f32>, config: SSRFadeConfig) -> f32 {
    // Compute distance from screen center
    // Center is at (0.5, 0.5), corners are at (0, 0), (1, 0), (0, 1), (1, 1)
    let centered_uv = screen_uv - vec2<f32>(0.5);

    // Distance in normalized coordinates
    // 0.0 at center, ~0.707 at corners
    let distance_from_center = length(centered_uv) * 2.0;

    // Apply smooth fade
    // 1.0 when inside edge_fade_start, 0.0 when beyond edge_fade_end
    return 1.0 - smoothstep_fade(
        config.edge_fade_start,
        config.edge_fade_end,
        distance_from_center
    );
}

/// Compute separate horizontal and vertical edge fades.
///
/// More accurate for wide-aspect displays where corners fade too early
/// with the circular distance calculation.
fn compute_edge_fade_rectangular(screen_uv: vec2<f32>, config: SSRFadeConfig) -> f32 {
    // Compute distance from edges along each axis independently
    let centered_uv = abs(screen_uv - vec2<f32>(0.5)) * 2.0;

    // Fade each axis separately
    let fade_x = 1.0 - smoothstep_fade(config.edge_fade_start, config.edge_fade_end, centered_uv.x);
    let fade_y = 1.0 - smoothstep_fade(config.edge_fade_start, config.edge_fade_end, centered_uv.y);

    // Combine - minimum or multiply both work
    return fade_x * fade_y;
}

/// Compute edge fade with anisotropic control.
///
/// Allows different fade rates for horizontal and vertical edges.
/// Useful when rays tend to exit horizontally (e.g., side-scrolling views).
fn compute_edge_fade_anisotropic(
    screen_uv: vec2<f32>,
    h_start: f32,
    h_end: f32,
    v_start: f32,
    v_end: f32
) -> f32 {
    let centered_uv = abs(screen_uv - vec2<f32>(0.5)) * 2.0;

    let fade_x = 1.0 - smoothstep_fade(h_start, h_end, centered_uv.x);
    let fade_y = 1.0 - smoothstep_fade(v_start, v_end, centered_uv.y);

    return fade_x * fade_y;
}

// ---------------------------------------------------------------------------
// Distance Fade Functions
// ---------------------------------------------------------------------------

/// Compute distance fade based on hit distance.
///
/// Returns 1.0 for close hits, smoothly fading to 0.0 for distant hits.
/// Prevents SSR from showing reflections from very distant objects which
/// are typically less accurate.
///
/// Arguments:
/// - hit_distance: World-space distance to reflection hit point
/// - config: Fade configuration parameters
///
/// Returns:
/// - Fade factor in range [0.0, 1.0]
fn compute_distance_fade(hit_distance: f32, config: SSRFadeConfig) -> f32 {
    return 1.0 - smoothstep_fade(
        config.distance_fade_start,
        config.distance_fade_end,
        hit_distance
    );
}

/// Compute distance fade with exponential falloff.
///
/// More physically-based as light attenuation follows inverse-square law.
fn compute_distance_fade_exponential(hit_distance: f32, half_distance: f32) -> f32 {
    // Exponential falloff: fade = 2^(-distance / half_distance)
    return exp2(-hit_distance / half_distance);
}

/// Compute distance fade with linear falloff.
///
/// Simple linear interpolation, less natural but predictable.
fn compute_distance_fade_linear(hit_distance: f32, max_distance: f32) -> f32 {
    return clamp(1.0 - hit_distance / max_distance, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Roughness Fade Functions
// ---------------------------------------------------------------------------

/// Compute roughness fade based on surface roughness.
///
/// Returns 1.0 for smooth (mirror-like) surfaces, fading to 0.0 for rough surfaces.
/// SSR is most accurate for smooth surfaces; rough surfaces need many samples
/// or should use other techniques like reflection probes.
///
/// Arguments:
/// - roughness: Surface roughness (0.0 = mirror, 1.0 = fully rough)
/// - config: Fade configuration parameters
///
/// Returns:
/// - Fade factor in range [0.0, 1.0]
fn compute_roughness_fade(roughness: f32, config: SSRFadeConfig) -> f32 {
    return 1.0 - smoothstep_fade(
        config.roughness_fade_start,
        config.roughness_fade_end,
        roughness
    );
}

/// Compute roughness fade with perceptual roughness input.
///
/// Many engines use perceptual roughness (linear) which needs to be
/// converted to actual roughness (squared) for accurate fading.
fn compute_roughness_fade_perceptual(perceptual_roughness: f32, config: SSRFadeConfig) -> f32 {
    // Convert perceptual to actual roughness (inverse of what's typically stored)
    let roughness = perceptual_roughness * perceptual_roughness;
    return compute_roughness_fade(roughness, config);
}

/// Compute roughness fade with GGX lobe consideration.
///
/// For rough surfaces, the reflection lobe is wider and SSR single-ray
/// sampling is less representative. This provides a more aggressive
/// fade for physically-based workflows.
fn compute_roughness_fade_ggx(roughness: f32, viewing_angle: f32) -> f32 {
    // GGX lobe width increases with roughness and grazing angles
    // Fade more aggressively at grazing angles for rough surfaces
    let angle_factor = 1.0 - viewing_angle; // 0 = perpendicular, 1 = grazing
    let adjusted_roughness = roughness + roughness * angle_factor * 0.5;

    return 1.0 - smoothstep_fade(0.3, 0.7, adjusted_roughness);
}

// ---------------------------------------------------------------------------
// Combined Fade Functions
// ---------------------------------------------------------------------------

/// Compute combined fade factor from all sources.
///
/// Multiplies edge, distance, and roughness fades together.
/// The final fade is conservative - any source causing fade will reduce output.
///
/// Arguments:
/// - screen_uv: Screen-space UV coordinates
/// - hit_distance: World-space distance to hit point
/// - roughness: Surface roughness value
/// - config: Fade configuration parameters
///
/// Returns:
/// - Combined fade factor in range [0.0, 1.0]
fn compute_combined_fade(
    screen_uv: vec2<f32>,
    hit_distance: f32,
    roughness: f32,
    config: SSRFadeConfig
) -> f32 {
    let edge_fade = compute_edge_fade(screen_uv, config);
    let distance_fade = compute_distance_fade(hit_distance, config);
    let roughness_fade = compute_roughness_fade(roughness, config);

    return edge_fade * distance_fade * roughness_fade;
}

/// Compute combined fade with weights for each component.
///
/// Allows prioritizing certain fade sources over others.
fn compute_combined_fade_weighted(
    screen_uv: vec2<f32>,
    hit_distance: f32,
    roughness: f32,
    config: SSRFadeConfig,
    edge_weight: f32,
    distance_weight: f32,
    roughness_weight: f32
) -> f32 {
    // Compute individual fades
    let edge_fade = compute_edge_fade(screen_uv, config);
    let distance_fade = compute_distance_fade(hit_distance, config);
    let roughness_fade = compute_roughness_fade(roughness, config);

    // Weighted geometric mean
    let weighted_sum =
        edge_weight * log(max(edge_fade, 0.0001)) +
        distance_weight * log(max(distance_fade, 0.0001)) +
        roughness_weight * log(max(roughness_fade, 0.0001));

    let total_weight = edge_weight + distance_weight + roughness_weight;

    return exp(weighted_sum / total_weight);
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Apply fade to reflection color.
///
/// Combines the reflection color with the fade factor, optionally
/// blending with a fallback color.
fn apply_fade_to_color(
    reflection_color: vec3<f32>,
    fade: f32,
    fallback_color: vec3<f32>
) -> vec3<f32> {
    return mix(fallback_color, reflection_color, fade);
}

/// Apply fade with alpha channel for compositing.
///
/// Returns RGBA where alpha represents the reflection confidence/visibility.
fn apply_fade_with_alpha(reflection_color: vec3<f32>, fade: f32) -> vec4<f32> {
    return vec4<f32>(reflection_color, fade);
}

/// Remap fade value with contrast adjustment.
///
/// Allows sharpening or softening the fade transition.
/// contrast > 1.0 sharpens, < 1.0 softens.
fn remap_fade_contrast(fade: f32, contrast: f32) -> f32 {
    // Shift to [-0.5, 0.5], apply power, shift back
    let shifted = fade - 0.5;
    let sign = select(-1.0, 1.0, shifted >= 0.0);
    let powered = pow(abs(shifted) * 2.0, contrast) * 0.5;
    return sign * powered + 0.5;
}

// ---------------------------------------------------------------------------
// Debug Visualization
// ---------------------------------------------------------------------------

/// Generate a color visualization of the fade value.
///
/// Blue (cold) = 0.0, Red (hot) = 1.0
/// Useful for debugging fade patterns.
fn visualize_fade(fade: f32) -> vec3<f32> {
    // Blue -> Cyan -> Green -> Yellow -> Red
    var color: vec3<f32>;

    if (fade < 0.25) {
        color = mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0), fade * 4.0);
    } else if (fade < 0.5) {
        color = mix(vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), (fade - 0.25) * 4.0);
    } else if (fade < 0.75) {
        color = mix(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), (fade - 0.5) * 4.0);
    } else {
        color = mix(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), (fade - 0.75) * 4.0);
    }

    return color;
}

/// Generate grayscale visualization of fade value.
fn visualize_fade_grayscale(fade: f32) -> vec3<f32> {
    return vec3<f32>(fade);
}
