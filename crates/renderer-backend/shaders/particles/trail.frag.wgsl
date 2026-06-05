// SPDX-License-Identifier: MIT
//
// trail.frag.wgsl - Trail Ribbon Fragment Shader (T-GPU-6.3)
//
// Samples trail texture and applies color modulation with alpha fade.
// Supports both textured and solid-color trails.
//
// Features:
// - Texture sampling with linear filtering
// - Color modulation from vertex color
// - Alpha blending with age-based fade
// - Optional soft edge falloff across ribbon width

// ============================================================================
// Data Structures
// ============================================================================

/// Fragment shader input (from vertex shader).
struct FragmentInput {
    /// Texture coordinates (u=along trail, v=across ribbon).
    @location(0) uv: vec2<f32>,
    /// Interpolated vertex color.
    @location(1) color: vec4<f32>,
    /// Alpha factor from age-based fade.
    @location(2) alpha: f32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(1) @binding(0) var trail_texture: texture_2d<f32>;
@group(1) @binding(1) var trail_sampler: sampler;

// ============================================================================
// Fragment Shader Entry Point
// ============================================================================

/// Trail ribbon fragment shader.
///
/// Samples the trail texture and combines with vertex color and fade alpha.
/// The output is premultiplied alpha for correct blending.
@fragment
fn fs_trail(in: FragmentInput) -> @location(0) vec4<f32> {
    // Sample texture
    let tex_color = textureSample(trail_texture, trail_sampler, in.uv);

    // Combine texture with vertex color
    let rgb = tex_color.rgb * in.color.rgb;

    // Combine all alpha sources:
    // - Texture alpha
    // - Vertex color alpha
    // - Age-based fade alpha
    let final_alpha = tex_color.a * in.color.a * in.alpha;

    // Apply soft edge falloff at ribbon edges (optional, based on v coordinate)
    // This creates a smoother appearance by fading edges
    let edge_distance = abs(in.uv.y - 0.5) * 2.0;  // 0 at center, 1 at edges
    let edge_falloff = 1.0 - edge_distance * edge_distance * 0.3;  // Subtle quadratic falloff

    // Final color with premultiplied alpha
    return vec4<f32>(rgb * final_alpha * edge_falloff, final_alpha * edge_falloff);
}

// ============================================================================
// Alternative Entry Points
// ============================================================================

/// Solid color trail (no texture sampling).
/// Use when trail_texture is not bound or for performance.
@fragment
fn fs_trail_solid(in: FragmentInput) -> @location(0) vec4<f32> {
    let final_alpha = in.color.a * in.alpha;

    // Edge falloff
    let edge_distance = abs(in.uv.y - 0.5) * 2.0;
    let edge_falloff = 1.0 - edge_distance * edge_distance * 0.3;

    return vec4<f32>(in.color.rgb * final_alpha * edge_falloff, final_alpha * edge_falloff);
}

/// Additive blending trail (for glow effects).
/// Output is not premultiplied; blending should be (ONE, ONE).
@fragment
fn fs_trail_additive(in: FragmentInput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(trail_texture, trail_sampler, in.uv);

    // Combine with fade
    let intensity = tex_color.a * in.color.a * in.alpha;

    // Edge falloff
    let edge_distance = abs(in.uv.y - 0.5) * 2.0;
    let edge_falloff = 1.0 - edge_distance * edge_distance * 0.3;

    // Additive: RGB scaled by intensity, alpha ignored by blend mode
    return vec4<f32>(tex_color.rgb * in.color.rgb * intensity * edge_falloff, 1.0);
}
