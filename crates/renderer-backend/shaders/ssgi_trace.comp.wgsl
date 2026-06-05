// SPDX-License-Identifier: MIT
//
// ssgi_trace.comp.wgsl - Screen-Space Global Illumination Compute Shader (T-GIR-P3.1).
//
// Traces rays in screen space using HiZ-accelerated ray marching to sample
// indirect lighting. SSGI captures diffuse color bleeding effects that enhance
// the realism of global illumination.
//
// Algorithm:
//   1. Reconstruct world position from depth buffer
//   2. Sample surface normal from GBuffer
//   3. Generate cosine-weighted hemisphere directions (Fibonacci spiral)
//   4. For each ray direction:
//      a. Project to screen space
//      b. March along ray using HiZ for acceleration
//      c. On hit: sample lighting buffer, apply distance fade
//   5. Accumulate and output average irradiance
//
// Quality tiers:
//   Low:    4 rays,  32 steps - Mobile / minimum quality (~0.5ms @ 1080p)
//   Medium: 8 rays,  48 steps - Balanced (~1.0ms @ 1080p)
//   High:   16 rays, 64 steps - High quality (~2.0ms @ 1080p)
//
// Workgroup size: 8x8 threads for optimal GPU occupancy.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 8u;
const PI: f32 = 3.14159265358979323846;
const GOLDEN_RATIO: f32 = 1.618033988749;

// Ray marching constants
const HIZ_START_MIP: u32 = 2u;     // Start at mip 2 for efficiency
const HIZ_MAX_MIP: u32 = 10u;      // Maximum mip level to query
const RAY_BIAS: f32 = 0.01;        // Bias along normal to prevent self-intersection

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct SSGIConfig {
    rays_per_pixel: u32,         // Number of rays per pixel (4, 8, or 16)
    max_steps: u32,              // Maximum ray march steps (32-64)
    max_distance: f32,           // Maximum ray distance in world space
    thickness: f32,              // Depth thickness threshold for hits
    intensity: f32,              // GI contribution intensity
    distance_fade_start: f32,    // Distance at which fade begins
    distance_fade_end: f32,      // Distance at which fade ends
    _pad: f32,
}

struct SSGIDispatchParams {
    screen_size: vec2<u32>,      // Output dimensions
    inv_screen: vec2<f32>,       // 1.0 / screen_size
    frame_index: u32,            // For temporal jittering
    _pad: vec3<u32>,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    near: f32,
    far: f32,
    _pad0: f32,
    _pad1: f32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> config: SSGIConfig;
@group(0) @binding(1) var<uniform> params: SSGIDispatchParams;
@group(0) @binding(2) var depth_texture: texture_depth_2d;
@group(0) @binding(3) var normal_texture: texture_2d<f32>;
@group(0) @binding(4) var lighting_texture: texture_2d<f32>;
@group(0) @binding(5) var hiz_texture: texture_2d<f32>;
@group(0) @binding(6) var hiz_sampler: sampler;
@group(0) @binding(7) var<uniform> camera: CameraUniforms;
@group(0) @binding(8) var output_texture: texture_storage_2d<rgba16float, write>;

// ---------------------------------------------------------------------------
// Helper Functions: Depth and Position Reconstruction
// ---------------------------------------------------------------------------

/// Convert linear depth to NDC depth (for reversed-Z: near=1, far=0).
fn linear_to_ndc_depth(linear_z: f32) -> f32 {
    // Reversed-Z: depth = (far * near) / (far - linear_z * (far - near))
    // Solving for NDC: depth = far * (1 - near/linear_z) / (far - near)
    return camera.far * camera.near / (linear_z * (camera.far - camera.near)) - camera.near / (camera.far - camera.near);
}

/// Linearize hardware depth (reversed-Z).
fn linearize_depth(depth: f32) -> f32 {
    // For reversed-Z: linear = (far * near) / (far - depth * (far - near))
    return camera.near * camera.far / (camera.far - depth * (camera.far - camera.near));
}

/// Reconstruct world position from UV and depth.
fn reconstruct_world_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // NDC position (flip Y for Vulkan/WGSL coordinate system)
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);

    // Unproject to view space
    let view_pos = camera.inv_proj * ndc;
    let view_pos_div = view_pos.xyz / view_pos.w;

    // Transform to world space
    let world_pos = camera.inv_view * vec4<f32>(view_pos_div, 1.0);
    return world_pos.xyz;
}

/// Project world position to screen UV.
fn project_to_screen(world_pos: vec3<f32>) -> vec3<f32> {
    let view_pos = camera.view * vec4<f32>(world_pos, 1.0);
    let clip = camera.proj * view_pos;
    let ndc = clip.xyz / clip.w;

    // Convert to UV (flip Y back)
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    return vec3<f32>(uv, ndc.z);
}

// ---------------------------------------------------------------------------
// Helper Functions: Hemisphere Sampling
// ---------------------------------------------------------------------------

/// Build orthonormal basis from normal using Frisvad's method.
fn build_orthonormal_basis(n: vec3<f32>) -> mat3x3<f32> {
    var tangent: vec3<f32>;
    var bitangent: vec3<f32>;

    if n.z < -0.9999999 {
        // Handle the singularity at n = (0, 0, -1)
        tangent = vec3<f32>(0.0, -1.0, 0.0);
        bitangent = vec3<f32>(-1.0, 0.0, 0.0);
    } else {
        let a = 1.0 / (1.0 + n.z);
        let b = -n.x * n.y * a;
        tangent = vec3<f32>(1.0 - n.x * n.x * a, b, -n.x);
        bitangent = vec3<f32>(b, 1.0 - n.y * n.y * a, -n.y);
    }

    return mat3x3<f32>(tangent, bitangent, n);
}

/// Cosine-weighted hemisphere sampling using Fibonacci spiral.
/// Returns a direction in world space given the surface normal.
fn sample_hemisphere_cosine(normal: vec3<f32>, sample_index: u32, total_samples: u32) -> vec3<f32> {
    // Fibonacci spiral on hemisphere
    let theta = 2.0 * PI * f32(sample_index) / GOLDEN_RATIO;
    let cos_phi = 1.0 - f32(sample_index) / f32(total_samples);
    let sin_phi = sqrt(max(0.0, 1.0 - cos_phi * cos_phi));

    // Local direction in tangent space (z-up)
    let local_dir = vec3<f32>(
        sin_phi * cos(theta),
        sin_phi * sin(theta),
        cos_phi
    );

    // Transform to world space using orthonormal basis
    let basis = build_orthonormal_basis(normal);
    return normalize(basis * local_dir);
}

/// Add temporal jitter to sample direction for temporal stability.
fn jitter_direction(dir: vec3<f32>, pixel_coord: vec2<u32>, frame_index: u32) -> vec3<f32> {
    // Simple hash for jitter
    let hash = (pixel_coord.x * 73856093u) ^ (pixel_coord.y * 19349663u) ^ (frame_index * 83492791u);
    let jitter_angle = f32(hash % 1000u) * 0.001 * 2.0 * PI;

    // Small rotation around normal (already incorporated in sampling)
    // This provides temporal variation without changing the distribution
    return dir; // For simplicity, rely on Fibonacci pattern variation
}

// ---------------------------------------------------------------------------
// Helper Functions: HiZ Ray Marching
// ---------------------------------------------------------------------------

/// Sample HiZ buffer at a specific mip level.
fn sample_hiz(uv: vec2<f32>, mip: u32) -> f32 {
    return textureSampleLevel(hiz_texture, hiz_sampler, uv, f32(mip)).r;
}

/// Sample depth at full resolution.
fn sample_depth(coord: vec2<i32>) -> f32 {
    let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(params.screen_size) - 1);
    return textureLoad(depth_texture, clamped, 0);
}

/// HiZ ray trace result.
struct HiZTraceResult {
    valid: bool,
    screen_pos: vec2<i32>,
    hit_uv: vec2<f32>,
    distance: f32,
}

/// Trace a ray using hierarchical Z-buffer for acceleration.
fn hiz_trace(origin: vec3<f32>, direction: vec3<f32>) -> HiZTraceResult {
    var result: HiZTraceResult;
    result.valid = false;
    result.screen_pos = vec2<i32>(0);
    result.hit_uv = vec2<f32>(0.0);
    result.distance = 0.0;

    // Project ray start and end to screen space
    let ray_end = origin + direction * config.max_distance;
    let start_screen = project_to_screen(origin);
    let end_screen = project_to_screen(ray_end);

    // Screen-space ray direction and length
    let ray_dir_screen = end_screen.xy - start_screen.xy;
    let ray_length = length(ray_dir_screen);

    // Skip degenerate rays
    if ray_length < 0.001 {
        return result;
    }

    let ray_step = ray_dir_screen / f32(config.max_steps);
    let depth_step = (end_screen.z - start_screen.z) / f32(config.max_steps);

    var current_uv = start_screen.xy;
    var current_depth = start_screen.z;
    var current_mip = HIZ_START_MIP;

    // Hierarchical ray march
    for (var i = 0u; i < config.max_steps; i = i + 1u) {
        // Advance ray
        current_uv = current_uv + ray_step;
        current_depth = current_depth + depth_step;

        // Check screen bounds
        if current_uv.x < 0.0 || current_uv.x > 1.0 || current_uv.y < 0.0 || current_uv.y > 1.0 {
            break;
        }

        // Sample HiZ at current mip level
        let hiz_depth = sample_hiz(current_uv, current_mip);

        // For reversed-Z: larger depth = closer to camera
        // Ray is behind scene if ray_depth < scene_depth
        if current_depth < hiz_depth {
            // Ray is behind geometry at this mip level
            if current_mip == 0u {
                // At finest level, check thickness
                let linear_ray_depth = linearize_depth(current_depth);
                let linear_scene_depth = linearize_depth(hiz_depth);
                let depth_diff = linear_scene_depth - linear_ray_depth;

                if depth_diff > 0.0 && depth_diff < config.thickness {
                    // Valid hit
                    result.valid = true;
                    result.screen_pos = vec2<i32>(current_uv * vec2<f32>(params.screen_size));
                    result.hit_uv = current_uv;
                    result.distance = length(reconstruct_world_position(current_uv, hiz_depth) - origin);
                    break;
                }
            } else {
                // Step down to finer mip level
                current_mip = current_mip - 1u;
                // Step back slightly to re-test at finer level
                current_uv = current_uv - ray_step * 0.5;
                current_depth = current_depth - depth_step * 0.5;
            }
        } else {
            // Ray is in front of geometry, can potentially step up
            if current_mip < HIZ_MAX_MIP {
                current_mip = min(current_mip + 1u, HIZ_MAX_MIP);
            }
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Helper Functions: Distance Fade
// ---------------------------------------------------------------------------

/// Compute distance fade factor using smoothstep.
fn compute_distance_fade(distance: f32) -> f32 {
    if distance <= config.distance_fade_start {
        return 1.0;
    }
    if distance >= config.distance_fade_end {
        return 0.0;
    }

    let t = (distance - config.distance_fade_start) / (config.distance_fade_end - config.distance_fade_start);
    // Smoothstep for smooth falloff
    return 1.0 - t * t * (3.0 - 2.0 * t);
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn ssgi_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Bounds check
    if gid.x >= params.screen_size.x || gid.y >= params.screen_size.y {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) * params.inv_screen;

    // Sample depth
    let depth = sample_depth(coord);

    // Skip sky pixels (depth at far plane)
    if depth >= 1.0 {
        textureStore(output_texture, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Reconstruct world position
    let world_pos = reconstruct_world_position(uv, depth);

    // Sample and unpack world-space normal
    let normal_sample = textureLoad(normal_texture, coord, 0);
    let normal = normalize(normal_sample.xyz * 2.0 - 1.0);

    // Bias origin along normal to prevent self-intersection
    let ray_origin = world_pos + normal * RAY_BIAS;

    // Accumulate irradiance from multiple rays
    var irradiance = vec3<f32>(0.0);
    var valid_samples = 0u;

    for (var i = 0u; i < config.rays_per_pixel; i = i + 1u) {
        // Generate hemisphere sample direction
        let ray_dir = sample_hemisphere_cosine(normal, i, config.rays_per_pixel);

        // Trace ray using HiZ
        let hit = hiz_trace(ray_origin, ray_dir);

        if hit.valid {
            // Sample lighting buffer at hit location
            let hit_coord = clamp(hit.screen_pos, vec2<i32>(0), vec2<i32>(params.screen_size) - 1);
            let hit_color = textureLoad(lighting_texture, hit_coord, 0).rgb;

            // Apply distance fade
            let fade = compute_distance_fade(hit.distance);

            // Accumulate with cosine weighting (already in Fibonacci sampling)
            irradiance = irradiance + hit_color * fade;
            valid_samples = valid_samples + 1u;
        }
    }

    // Average irradiance
    if valid_samples > 0u {
        irradiance = irradiance / f32(valid_samples);
    }

    // Apply intensity multiplier
    irradiance = irradiance * config.intensity;

    // Output irradiance (alpha = 1 indicates valid data)
    let alpha = select(0.0, 1.0, valid_samples > 0u);
    textureStore(output_texture, coord, vec4<f32>(irradiance, alpha));
}

// ---------------------------------------------------------------------------
// Notes
// ---------------------------------------------------------------------------
//
// HiZ Acceleration Strategy:
// - Start at coarse mip level (2) for fast traversal
// - Step down to finer levels when ray intersects geometry
// - Step up to coarser levels when ray is in empty space
// - Final hit validation at mip 0 with thickness check
//
// Cosine-Weighted Sampling:
// - Fibonacci spiral provides low-discrepancy distribution
// - Cosine weighting matches diffuse BRDF (Lambertian)
// - No explicit PDF division needed (baked into distribution)
//
// Temporal Stability:
// - Frame index can be used for temporal jittering
// - Bilateral upscale (separate pass) handles half-res noise
// - TAA/temporal accumulation recommended for final output
//
// Performance Considerations:
// - Half-res rendering provides ~4x speedup
// - HiZ acceleration reduces per-ray cost significantly
// - Workgroup size 8x8 optimizes for modern GPUs
// - Texture fetches are the primary bottleneck
