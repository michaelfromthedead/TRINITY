// SPDX-License-Identifier: MIT
//
// mesh_particle.frag.wgsl - Mesh Particle Fragment Shader (T-GPU-6.2)
//
// Applies simple lighting and texturing to mesh particles.
// Uses a directional light with ambient for basic illumination.

// ============================================================================
// Data Structures
// ============================================================================

/// Fragment input from vertex shader.
struct FragmentInput {
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
}

// ============================================================================
// Bindings
// ============================================================================

@group(1) @binding(0) var base_texture: texture_2d<f32>;
@group(1) @binding(1) var base_sampler: sampler;

// ============================================================================
// Lighting Constants
// ============================================================================

// Simple directional light parameters
const LIGHT_DIR: vec3<f32> = vec3<f32>(0.5, 1.0, 0.3);
const AMBIENT_INTENSITY: f32 = 0.2;

// ============================================================================
// Fragment Shader Entry Point
// ============================================================================

@fragment
fn fs_mesh_particle(in: FragmentInput) -> @location(0) vec4<f32> {
    // Sample texture
    let tex_color = textureSample(base_texture, base_sampler, in.uv);

    // Early alpha discard for performance
    if tex_color.a < 0.01 {
        discard;
    }

    // Simple N dot L lighting
    let normal = normalize(in.world_normal);
    let light_dir = normalize(LIGHT_DIR);
    let ndotl = max(dot(normal, light_dir), 0.0);

    // Combine diffuse and ambient
    let lighting = ndotl + AMBIENT_INTENSITY;

    // Apply lighting, particle color modulation, and texture
    let final_rgb = tex_color.rgb * in.color.rgb * lighting;
    let final_alpha = tex_color.a * in.color.a;

    return vec4<f32>(final_rgb, final_alpha);
}
