// SPDX-License-Identifier: MIT
//
// ssr_ray_march_linear.comp.wgsl - Linear SSR Ray Marching Shader (T-GIR-P4.3).
//
// This shader implements a simple linear ray marching fallback for screen-space
// reflections. It steps through screen-space at a fixed stride, testing depth
// at each step against the depth buffer. When an intersection is detected,
// binary refinement is used to find the exact hit point.
//
// This is simpler but potentially slower than HiZ-accelerated marching.
// Use when HiZ isn't available or for quality comparison.
//
// Algorithm:
//   1. For each pixel, reconstruct view-space position and normal
//   2. Compute reflected view-space ray direction
//   3. Project ray to screen-space and march at fixed stride
//   4. At each step, compare ray depth to scene depth
//   5. When intersection detected, binary refine for accuracy
//   6. Apply fade functions (edge, distance, roughness)
//   7. Sample scene color at hit point and output
//
// Workgroup size: 8x8 threads.

// Include fade functions from ssr_fade.wgsl
// NOTE: In WGSL, we inline the fade functions rather than #include

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 8u;
const PI: f32 = 3.14159265359;
const BINARY_REFINE_ITERATIONS: u32 = 4u;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct SSRLinearConfig {
    max_steps: u32,         // Maximum ray marching steps
    stride_pixels: f32,     // Step size in pixels
    thickness: f32,         // Depth comparison threshold
    jitter: f32,            // Temporal jitter factor
}

struct SSRFadeConfig {
    edge_fade_start: f32,       // Start screen edge fade (0.8)
    edge_fade_end: f32,         // End screen edge fade (1.0)
    distance_fade_start: f32,   // Start distance fade (50m)
    distance_fade_end: f32,     // End distance fade (100m)
    roughness_fade_start: f32,  // Start roughness fade (0.5)
    roughness_fade_end: f32,    // End roughness fade (0.8)
    _pad: vec2<f32>,            // Padding for alignment
}

struct SSRLinearUniforms {
    linear_config: SSRLinearConfig,
    fade_config: SSRFadeConfig,
    screen_size: vec2<u32>,     // Screen dimensions
    frame_index: u32,           // Frame index for jitter
    _pad: u32,
}

// Camera matrices (expected from a separate uniform buffer or push constants)
struct CameraData {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    near_plane: f32,
    far_plane: f32,
    _pad: vec2<f32>,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> uniforms: SSRLinearUniforms;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var normal_texture: texture_2d<f32>;
@group(0) @binding(3) var color_texture: texture_2d<f32>;
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var linear_sampler: sampler;

// Camera data - typically in a separate bind group
@group(1) @binding(0) var<uniform> camera: CameraData;

// ---------------------------------------------------------------------------
// Hit Result
// ---------------------------------------------------------------------------

struct HitResult {
    hit: bool,
    screen_uv: vec2<f32>,
    hit_distance: f32,
    steps: u32,
}

// ---------------------------------------------------------------------------
// Fade Functions (from ssr_fade.wgsl, inlined)
// ---------------------------------------------------------------------------

/// Hermite interpolation (smoothstep).
fn smoothstep_custom(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

/// Compute edge fade based on screen UV.
/// Returns 1.0 at screen center, fading to 0.0 at edges.
fn compute_edge_fade(screen_uv: vec2<f32>) -> f32 {
    // Distance from center (0.0 at center, ~0.707 at corners)
    let centered_uv = screen_uv - vec2<f32>(0.5);
    let distance_from_center = length(centered_uv) * 2.0;

    // Map to [0, 1] where 0 = edge_fade_end, 1 = edge_fade_start
    return 1.0 - smoothstep_custom(
        uniforms.fade_config.edge_fade_start,
        uniforms.fade_config.edge_fade_end,
        distance_from_center
    );
}

/// Compute distance fade based on hit distance.
/// Returns 1.0 for close hits, fading to 0.0 for distant hits.
fn compute_distance_fade(hit_distance: f32) -> f32 {
    return 1.0 - smoothstep_custom(
        uniforms.fade_config.distance_fade_start,
        uniforms.fade_config.distance_fade_end,
        hit_distance
    );
}

/// Compute roughness fade based on surface roughness.
/// Returns 1.0 for smooth surfaces, fading to 0.0 for rough surfaces.
fn compute_roughness_fade(roughness: f32) -> f32 {
    return 1.0 - smoothstep_custom(
        uniforms.fade_config.roughness_fade_start,
        uniforms.fade_config.roughness_fade_end,
        roughness
    );
}

/// Compute combined fade factor from all sources.
fn compute_combined_fade(screen_uv: vec2<f32>, hit_distance: f32, roughness: f32) -> f32 {
    let edge = compute_edge_fade(screen_uv);
    let distance = compute_distance_fade(hit_distance);
    let rough = compute_roughness_fade(roughness);
    return edge * distance * rough;
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Hash function for temporal jitter.
fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

/// Sample depth at screen UV coordinates.
fn sample_depth(uv: vec2<f32>) -> f32 {
    let coords = vec2<i32>(uv * vec2<f32>(uniforms.screen_size));
    return textureLoad(depth_texture, coords, 0);
}

/// Sample normal at screen UV coordinates (world-space).
fn sample_normal(uv: vec2<f32>) -> vec3<f32> {
    let coords = vec2<i32>(uv * vec2<f32>(uniforms.screen_size));
    return textureLoad(normal_texture, coords, 0).xyz * 2.0 - 1.0;
}

/// Convert screen UV and depth to view-space position.
fn reconstruct_view_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // Convert UV to NDC
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);

    // Transform to view space
    let view_pos = camera.inv_projection * ndc;
    return view_pos.xyz / view_pos.w;
}

/// Convert view-space position to screen UV.
fn project_to_screen(view_pos: vec3<f32>) -> vec3<f32> {
    let clip_pos = camera.projection * vec4<f32>(view_pos, 1.0);
    let ndc = clip_pos.xyz / clip_pos.w;

    return vec3<f32>(
        ndc.x * 0.5 + 0.5,
        1.0 - (ndc.y * 0.5 + 0.5),
        ndc.z
    );
}

/// Check if UV is within screen bounds.
fn is_valid_uv(uv: vec2<f32>) -> bool {
    return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

// ---------------------------------------------------------------------------
// Binary Refinement
// ---------------------------------------------------------------------------

/// Binary refinement to find exact hit point.
/// Given a range [t_min, t_max] where intersection occurs, narrows down the hit.
fn binary_refine(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    t_min: f32,
    t_max: f32
) -> HitResult {
    var lo = t_min;
    var hi = t_max;

    for (var i = 0u; i < BINARY_REFINE_ITERATIONS; i++) {
        let mid = (lo + hi) * 0.5;
        let pos = ray_origin + ray_dir * mid;
        let screen_pos = project_to_screen(pos);

        if (!is_valid_uv(screen_pos.xy)) {
            hi = mid;
            continue;
        }

        let scene_depth = sample_depth(screen_pos.xy);
        let ray_depth = screen_pos.z;

        // Check intersection (reversed-Z: larger = closer)
        if (ray_depth > scene_depth) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    // Final position
    let final_t = (lo + hi) * 0.5;
    let final_pos = ray_origin + ray_dir * final_t;
    let final_screen = project_to_screen(final_pos);

    return HitResult(
        true,
        final_screen.xy,
        final_t,
        BINARY_REFINE_ITERATIONS
    );
}

// ---------------------------------------------------------------------------
// Linear Ray Marching
// ---------------------------------------------------------------------------

/// Perform linear ray marching in view space.
fn linear_trace(ray_origin: vec3<f32>, ray_dir: vec3<f32>, jitter_offset: f32) -> HitResult {
    let screen_size_f = vec2<f32>(uniforms.screen_size);
    let stride = uniforms.linear_config.stride_pixels / screen_size_f.x;
    let thickness = uniforms.linear_config.thickness;
    let max_steps = uniforms.linear_config.max_steps;

    // Start position with jitter
    var t = stride * jitter_offset;
    var prev_t = 0.0;

    for (var i = 0u; i < max_steps; i++) {
        // Current position along ray
        let pos = ray_origin + ray_dir * t;

        // Project to screen space
        let screen_pos = project_to_screen(pos);

        // Bounds check - ray exited screen
        if (!is_valid_uv(screen_pos.xy)) {
            return HitResult(false, vec2<f32>(0.0), -1.0, i);
        }

        // Sample scene depth at this screen position
        let scene_depth = sample_depth(screen_pos.xy);
        let ray_depth = screen_pos.z;

        // Check for intersection (reversed-Z convention)
        // Ray is behind surface if ray_depth > scene_depth
        // Intersection if ray crosses surface within thickness
        if (ray_depth > scene_depth && ray_depth < scene_depth + thickness) {
            // Found intersection - binary refine for accuracy
            return binary_refine(ray_origin, ray_dir, prev_t, t);
        }

        // Advance ray
        prev_t = t;
        t += stride;
    }

    // No hit found
    return HitResult(false, vec2<f32>(0.0), -1.0, max_steps);
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn ssr_linear_march(@builtin(global_invocation_id) gid: vec3<u32>) {
    let screen_size = uniforms.screen_size;

    // Early-out if beyond screen bounds
    if (gid.x >= screen_size.x || gid.y >= screen_size.y) {
        return;
    }

    // Screen UV for this pixel
    let pixel_uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(screen_size);

    // Sample depth and normal
    let depth = sample_depth(pixel_uv);
    let normal_ws = normalize(sample_normal(pixel_uv));

    // Skip sky pixels (depth at far plane)
    if (depth <= 0.0001) {
        textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    // Reconstruct view-space position
    let view_pos = reconstruct_view_position(pixel_uv, depth);

    // Transform normal to view space
    let normal_vs = normalize((camera.view * vec4<f32>(normal_ws, 0.0)).xyz);

    // Compute view direction (from surface to camera, in view space camera is at origin)
    let view_dir = normalize(-view_pos);

    // Compute reflected ray direction in view space
    let reflect_dir = reflect(-view_dir, normal_vs);

    // Skip back-facing reflections
    if (reflect_dir.z > 0.0) {
        textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    // Temporal jitter for TAA
    let jitter = hash(vec2<f32>(gid.xy) + vec2<f32>(f32(uniforms.frame_index) * 0.1));
    let jitter_offset = jitter * uniforms.linear_config.jitter;

    // Perform linear ray march
    let hit = linear_trace(view_pos, reflect_dir, jitter_offset);

    // Output result
    if (hit.hit) {
        // Sample color at hit point using linear filtering
        let color = textureSampleLevel(color_texture, linear_sampler, hit.screen_uv, 0.0).rgb;

        // Get roughness from GBuffer (could be in alpha of normal texture, simplified here)
        let roughness = 0.0; // TODO: Sample from GBuffer roughness

        // Apply fade functions
        let fade = compute_combined_fade(hit.screen_uv, hit.hit_distance, roughness);

        // Output color with fade applied
        // Alpha channel stores technique mask (SSR = 1)
        textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(color * fade, 1.0));
    } else {
        // Miss - output black with zero alpha
        textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(0.0));
    }
}

// ---------------------------------------------------------------------------
// Debug Entry Points
// ---------------------------------------------------------------------------

/// Debug visualization of ray march step count.
@compute @workgroup_size(8, 8, 1)
fn ssr_linear_debug_steps(@builtin(global_invocation_id) gid: vec3<u32>) {
    let screen_size = uniforms.screen_size;

    if (gid.x >= screen_size.x || gid.y >= screen_size.y) {
        return;
    }

    let pixel_uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(screen_size);
    let depth = sample_depth(pixel_uv);

    if (depth <= 0.0001) {
        textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    let view_pos = reconstruct_view_position(pixel_uv, depth);
    let normal_ws = normalize(sample_normal(pixel_uv));
    let normal_vs = normalize((camera.view * vec4<f32>(normal_ws, 0.0)).xyz);
    let view_dir = normalize(-view_pos);
    let reflect_dir = reflect(-view_dir, normal_vs);

    if (reflect_dir.z > 0.0) {
        textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    let hit = linear_trace(view_pos, reflect_dir, 0.0);

    // Visualize step count as heat map
    let max_steps = uniforms.linear_config.max_steps;
    let step_ratio = f32(hit.steps) / f32(max_steps);

    // Blue -> Green -> Yellow -> Red heat map
    var color: vec3<f32>;
    if (step_ratio < 0.33) {
        color = mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), step_ratio * 3.0);
    } else if (step_ratio < 0.66) {
        color = mix(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0), (step_ratio - 0.33) * 3.0);
    } else {
        color = mix(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), (step_ratio - 0.66) * 3.0);
    }

    textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}

/// Debug visualization of fade factors.
@compute @workgroup_size(8, 8, 1)
fn ssr_linear_debug_fade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let screen_size = uniforms.screen_size;

    if (gid.x >= screen_size.x || gid.y >= screen_size.y) {
        return;
    }

    let pixel_uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(screen_size);

    // Visualize edge fade
    let edge_fade = compute_edge_fade(pixel_uv);

    textureStore(output_texture, vec2<i32>(gid.xy), vec4<f32>(edge_fade, edge_fade, edge_fade, 1.0));
}
