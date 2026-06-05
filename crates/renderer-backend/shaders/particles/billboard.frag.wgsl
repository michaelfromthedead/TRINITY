// SPDX-License-Identifier: MIT
//
// billboard.frag.wgsl - Billboard Particle Fragment Shader (T-GPU-6.1)
//
// Samples particle texture and applies color modulation.
// Supports both alpha blending and additive blending modes.
//
// The fragment color is computed as:
//   final_color = texture_color * particle_color
//
// For soft particles (optional), alpha is modulated based on depth difference
// to prevent hard edges where particles intersect geometry.

// ============================================================================
// Data Structures
// ============================================================================

/// Fragment shader input from vertex shader.
struct FragmentInput {
    /// Texture coordinates for particle texture.
    @location(0) uv: vec2<f32>,
    /// Particle color (interpolated from emitter color_start/color_end).
    @location(1) color: vec4<f32>,
    /// Normalized age (0=just spawned, 1=about to die).
    @location(2) age_ratio: f32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(1) @binding(0) var particle_texture: texture_2d<f32>;
@group(1) @binding(1) var particle_sampler: sampler;

// ============================================================================
// Helper Functions
// ============================================================================

/// Apply soft particle fade based on age.
/// Creates a smooth fade-in/fade-out over the particle's lifetime.
fn calculate_lifetime_alpha(age_ratio: f32) -> f32 {
    // Fade in during first 10% of lifetime
    let fade_in = smoothstep(0.0, 0.1, age_ratio);
    // Fade out during last 20% of lifetime
    let fade_out = 1.0 - smoothstep(0.8, 1.0, age_ratio);
    return fade_in * fade_out;
}

/// Apply radial falloff for soft-edged particles.
/// Creates a smooth circular gradient from center to edges.
fn calculate_radial_falloff(uv: vec2<f32>) -> f32 {
    // Map UV [0,1] to [-1,1] centered
    let centered = uv * 2.0 - 1.0;
    // Distance from center
    let dist = length(centered);
    // Smooth falloff (1 at center, 0 at edges)
    return 1.0 - smoothstep(0.7, 1.0, dist);
}

// ============================================================================
// Fragment Shader Entry Point
// ============================================================================

@fragment
fn fs_billboard(input: FragmentInput) -> @location(0) vec4<f32> {
    // Sample particle texture
    let tex_color = textureSample(particle_texture, particle_sampler, input.uv);

    // Apply particle color modulation
    var final_color = tex_color * input.color;

    // Apply lifetime-based alpha fade
    let lifetime_alpha = calculate_lifetime_alpha(input.age_ratio);
    final_color.a *= lifetime_alpha;

    // Early discard for nearly transparent pixels
    if final_color.a < 0.001 {
        discard;
    }

    return final_color;
}

/// Alternative fragment shader for additive blending.
/// Used for fire, sparks, and glowing effects.
@fragment
fn fs_billboard_additive(input: FragmentInput) -> @location(0) vec4<f32> {
    // Sample particle texture
    let tex_color = textureSample(particle_texture, particle_sampler, input.uv);

    // Apply particle color modulation
    var final_color = tex_color * input.color;

    // Apply lifetime-based intensity fade
    let lifetime_alpha = calculate_lifetime_alpha(input.age_ratio);
    final_color = final_color * lifetime_alpha;

    // For additive blending, alpha channel controls intensity
    // RGB values are added to framebuffer
    return final_color;
}

/// Fragment shader with radial soft edges.
/// Creates circular particles without requiring a texture.
@fragment
fn fs_billboard_soft(input: FragmentInput) -> @location(0) vec4<f32> {
    // Sample particle texture
    let tex_color = textureSample(particle_texture, particle_sampler, input.uv);

    // Apply radial falloff for soft edges
    let radial_alpha = calculate_radial_falloff(input.uv);

    // Apply particle color modulation
    var final_color = tex_color * input.color;

    // Apply both radial and lifetime alpha
    let lifetime_alpha = calculate_lifetime_alpha(input.age_ratio);
    final_color.a *= radial_alpha * lifetime_alpha;

    // Early discard for nearly transparent pixels
    if final_color.a < 0.001 {
        discard;
    }

    return final_color;
}
