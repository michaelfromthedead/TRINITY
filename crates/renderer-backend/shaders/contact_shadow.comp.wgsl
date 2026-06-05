// SPDX-License-Identifier: MIT
//
// contact_shadow.comp.wgsl - Screen-Space Contact Shadow Compute Shader (T-LIT-8.1).
//
// Performs screen-space ray marching to generate high-detail contact shadows
// that complement cascaded shadow maps. Contact shadows add fine detail for
// small-scale occlusion that CSMs cannot capture due to resolution limits.
//
// Algorithm:
//   1. Reconstruct view-space position from depth buffer
//   2. Bias ray start along surface normal to prevent self-shadowing
//   3. March ray toward light in screen space
//   4. Compare ray depth against scene depth at each step
//   5. Detect occlusion when ray is behind scene geometry within thickness threshold
//
// Quality tiers (via config.step_count):
//   Low:    8 steps  - Mobile / minimum quality
//   Medium: 16 steps - Balanced performance/quality
//   High:   32 steps - High quality desktop
//   Ultra:  64 steps - Maximum quality
//
// Workgroup size: 8x8 threads for optimal occupancy on modern GPUs.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 8u;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct ContactShadowConfig {
    step_count: u32,      // Number of ray march steps
    max_distance: f32,    // Maximum ray distance in world space
    thickness: f32,       // Occlusion thickness threshold
    normal_bias: f32,     // Normal offset to prevent self-shadowing
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

struct LightUniforms {
    direction: vec3<f32>,  // Light direction (normalized, world space)
    _pad0: f32,
    color: vec3<f32>,
    intensity: f32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> config: ContactShadowConfig;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var normal_texture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> camera: CameraUniforms;
@group(0) @binding(4) var<uniform> light: LightUniforms;
@group(0) @binding(5) var output_texture: texture_storage_2d<r8unorm, write>;

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

// Reconstruct view-space position from depth
fn reconstruct_view_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // NDC position
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);

    // Unproject to view space
    let view_pos = camera.inv_proj * ndc;
    return view_pos.xyz / view_pos.w;
}

// Project view-space position to screen UV
fn project_to_screen(view_pos: vec3<f32>) -> vec2<f32> {
    let clip = camera.proj * vec4<f32>(view_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    return ndc.xy * 0.5 + 0.5;
}

// Linear depth from hardware depth
fn linear_depth(depth: f32) -> f32 {
    return camera.near * camera.far / (camera.far - depth * (camera.far - camera.near));
}

// ---------------------------------------------------------------------------
// Contact Shadow Ray March
// ---------------------------------------------------------------------------

fn trace_contact_shadow(origin: vec3<f32>, normal: vec3<f32>, dims: vec2<u32>) -> f32 {
    // Light direction in view space
    let light_dir_world = normalize(-light.direction);
    let light_dir_view = (camera.view * vec4<f32>(light_dir_world, 0.0)).xyz;

    // Apply normal bias to prevent self-shadowing
    let biased_origin = origin + normal * config.normal_bias;

    // Ray end point
    let ray_end = biased_origin + light_dir_view * config.max_distance;

    // Project both points to screen space
    let start_uv = project_to_screen(biased_origin);
    let end_uv = project_to_screen(ray_end);

    // Screen-space ray direction and length
    let ray_dir = end_uv - start_uv;
    let ray_length = length(ray_dir);

    // Skip if ray is too short
    if ray_length < 0.001 {
        return 1.0;
    }

    let ray_step = ray_dir / f32(config.step_count);

    // View-space depth interpolation
    let start_depth = biased_origin.z;
    let end_depth = ray_end.z;
    let depth_step = (end_depth - start_depth) / f32(config.step_count);

    var current_uv = start_uv;
    var current_depth = start_depth;
    var occlusion = 0.0;

    // Ray march
    for (var i = 1u; i <= config.step_count; i = i + 1u) {
        current_uv = start_uv + ray_step * f32(i);
        current_depth = start_depth + depth_step * f32(i);

        // Check bounds
        if current_uv.x < 0.0 || current_uv.x > 1.0 ||
           current_uv.y < 0.0 || current_uv.y > 1.0 {
            break;
        }

        // Sample depth buffer
        let sample_coord = vec2<i32>(current_uv * vec2<f32>(dims));
        let sampled_depth = textureLoad(depth_texture, sample_coord, 0);

        // Reconstruct view-space Z from sampled depth
        let sampled_view_z = linear_depth(sampled_depth);

        // Check for occlusion (scene is closer than ray)
        let depth_diff = -current_depth - sampled_view_z;

        if depth_diff > 0.0 && depth_diff < config.thickness {
            // Smooth falloff based on step position
            let step_factor = 1.0 - f32(i) / f32(config.step_count);
            occlusion = max(occlusion, step_factor);
        }
    }

    // Return shadow factor (1 = lit, 0 = shadowed)
    return 1.0 - occlusion;
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(output_texture);

    // Bounds check
    if global_id.x >= dims.x || global_id.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(global_id.xy);
    let uv = (vec2<f32>(global_id.xy) + 0.5) / vec2<f32>(dims);

    // Sample depth
    let depth = textureLoad(depth_texture, coord, 0);

    // Skip sky pixels (depth at far plane)
    if depth >= 1.0 {
        textureStore(output_texture, coord, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    // Reconstruct view-space position
    let view_pos = reconstruct_view_position(uv, depth);

    // Sample and unpack normal (assuming view-space normals in RGB)
    let normal_sample = textureLoad(normal_texture, coord, 0);
    let normal = normalize(normal_sample.xyz * 2.0 - 1.0);

    // Trace contact shadow
    let shadow = trace_contact_shadow(view_pos, normal, dims);

    // Store result
    textureStore(output_texture, coord, vec4<f32>(shadow, 0.0, 0.0, 0.0));
}
