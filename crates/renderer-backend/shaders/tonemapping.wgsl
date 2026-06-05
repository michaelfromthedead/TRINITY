// SPDX-License-Identifier: MIT
//
// tonemapping.wgsl - HDR Tonemapping Compute Shader (T-WGPU-P3.10.5).
//
// Maps HDR (high dynamic range) colors to LDR (low dynamic range) display
// using the ACES filmic tonemapping curve with exposure adjustment and
// gamma correction.
//
// Pipeline:
//   1. Apply exposure adjustment (multiply by 2^exposure)
//   2. Apply ACES filmic tonemapping curve
//   3. Apply gamma correction (sRGB encoding)
//
// ACES (Academy Color Encoding System):
//   - Industry standard for film and HDR content
//   - Produces pleasing results with realistic highlight rolloff
//   - Maintains color saturation better than Reinhard
//
// Workgroup size: 8x8 threads for optimal GPU occupancy.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 8u;

// ACES filmic curve parameters (RRT + ODT simplified fit)
// Based on Krzysztof Narkowicz's approximation
const ACES_A: f32 = 2.51;
const ACES_B: f32 = 0.03;
const ACES_C: f32 = 2.43;
const ACES_D: f32 = 0.59;
const ACES_E: f32 = 0.14;

// Gamma correction (sRGB)
const GAMMA: f32 = 2.2;
const INV_GAMMA: f32 = 0.4545454545; // 1.0 / 2.2

// Tonemapping mode constants
const MODE_ACES: u32 = 0u;
const MODE_REINHARD: u32 = 1u;
const MODE_UNCHARTED2: u32 = 2u;
const MODE_ACES_FITTED: u32 = 3u;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct TonemapUniforms {
    src_dims: vec2<u32>,     // Source texture dimensions
    dst_dims: vec2<u32>,     // Destination texture dimensions
    exposure: f32,           // Exposure adjustment in stops (EV)
    gamma: f32,              // Gamma correction value (typically 2.2)
    mode: u32,               // Tonemapping curve selection
    white_point: f32,        // White point for some curves (default 4.0)
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uniforms: TonemapUniforms;

// ---------------------------------------------------------------------------
// Tonemapping Curves
// ---------------------------------------------------------------------------

/// ACES filmic tonemapping (Krzysztof Narkowicz approximation).
///
/// Input: Linear HDR color (after exposure adjustment)
/// Output: Mapped color in [0, 1] range
///
/// Formula: (x * (a*x + b)) / (x * (c*x + d) + e)
fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    // ACES input transform (approximate RRT)
    let aces = color * 0.6; // Scale factor

    return saturate(
        (aces * (ACES_A * aces + ACES_B)) /
        (aces * (ACES_C * aces + ACES_D) + ACES_E)
    );
}

/// ACES fitted (Stephen Hill / Epic Games).
/// More accurate fit to the full ACES pipeline.
fn aces_fitted(color: vec3<f32>) -> vec3<f32> {
    // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
    let m1 = mat3x3<f32>(
        vec3<f32>(0.59719, 0.07600, 0.02840),
        vec3<f32>(0.35458, 0.90834, 0.13383),
        vec3<f32>(0.04823, 0.01566, 0.83777)
    );

    // ODT_SAT => XYZ => D60_2_D65 => sRGB
    let m2 = mat3x3<f32>(
        vec3<f32>( 1.60475, -0.10208, -0.00327),
        vec3<f32>(-0.53108,  1.10813, -0.07276),
        vec3<f32>(-0.07367, -0.00605,  1.07602)
    );

    let v = m1 * color;
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return saturate(m2 * (a / b));
}

/// Reinhard tonemapping (simple).
///
/// Formula: color / (1 + color)
/// Extended: color * (1 + color/white^2) / (1 + color)
fn reinhard_tonemap(color: vec3<f32>) -> vec3<f32> {
    return color / (vec3<f32>(1.0) + color);
}

/// Extended Reinhard with white point.
fn reinhard_extended(color: vec3<f32>, white: f32) -> vec3<f32> {
    let white_sq = white * white;
    let numerator = color * (vec3<f32>(1.0) + color / white_sq);
    let denominator = vec3<f32>(1.0) + color;
    return numerator / denominator;
}

/// Uncharted 2 (John Hable's filmic curve).
fn uncharted2_partial(x: vec3<f32>) -> vec3<f32> {
    let A = 0.15;  // Shoulder strength
    let B = 0.50;  // Linear strength
    let C = 0.10;  // Linear angle
    let D = 0.20;  // Toe strength
    let E = 0.02;  // Toe numerator
    let F = 0.30;  // Toe denominator

    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

fn uncharted2_tonemap(color: vec3<f32>) -> vec3<f32> {
    let exposure_bias = 2.0;
    let curr = uncharted2_partial(color * exposure_bias);

    let W = 11.2; // Linear white point
    let white_scale = vec3<f32>(1.0) / uncharted2_partial(vec3<f32>(W));

    return saturate(curr * white_scale);
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Apply exposure adjustment.
/// exposure is in stops (EV), so multiply by 2^exposure.
fn apply_exposure(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    return color * pow(2.0, exposure);
}

/// Apply gamma correction (linear to sRGB).
fn gamma_correct(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / gamma));
}

/// Apply sRGB transfer function (more accurate than simple gamma).
fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let cutoff = color < vec3<f32>(0.0031308);
    let higher = vec3<f32>(1.055) * pow(color, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    let lower = color * vec3<f32>(12.92);
    return select(higher, lower, cutoff);
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

/// HDR tonemapping kernel.
///
/// Applies exposure adjustment, tonemapping curve, and gamma correction
/// to convert HDR input to LDR output suitable for display.
@compute @workgroup_size(8, 8, 1)
fn tonemap(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Early-out for pixels outside destination bounds
    if (gid.x >= uniforms.dst_dims.x || gid.y >= uniforms.dst_dims.y) {
        return;
    }

    // Load HDR color
    let coord = vec2<i32>(gid.xy);
    let hdr = textureLoad(src_texture, coord, 0);

    // Apply exposure adjustment
    var color = apply_exposure(hdr.rgb, uniforms.exposure);

    // Apply selected tonemapping curve
    switch (uniforms.mode) {
        case MODE_REINHARD: {
            color = reinhard_extended(color, uniforms.white_point);
        }
        case MODE_UNCHARTED2: {
            color = uncharted2_tonemap(color);
        }
        case MODE_ACES_FITTED: {
            color = aces_fitted(color);
        }
        default: {
            // MODE_ACES: Default ACES filmic
            color = aces_tonemap(color);
        }
    }

    // Apply gamma correction
    color = gamma_correct(color, uniforms.gamma);

    // Write LDR output with original alpha
    textureStore(dst_texture, coord, vec4<f32>(color, hdr.a));
}

// ---------------------------------------------------------------------------
// Alternative Entry Points
// ---------------------------------------------------------------------------

/// Tonemapping without gamma (for further processing).
@compute @workgroup_size(8, 8, 1)
fn tonemap_linear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uniforms.dst_dims.x || gid.y >= uniforms.dst_dims.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let hdr = textureLoad(src_texture, coord, 0);

    var color = apply_exposure(hdr.rgb, uniforms.exposure);
    color = aces_tonemap(color);

    // No gamma correction - output remains linear
    textureStore(dst_texture, coord, vec4<f32>(color, hdr.a));
}

/// Quick ACES only (fastest path for common case).
@compute @workgroup_size(8, 8, 1)
fn tonemap_aces_fast(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uniforms.dst_dims.x || gid.y >= uniforms.dst_dims.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let hdr = textureLoad(src_texture, coord, 0);

    // Fixed exposure of 0 (no adjustment)
    let color = aces_tonemap(hdr.rgb);

    // Fixed gamma of 2.2
    let ldr = pow(max(color, vec3<f32>(0.0)), vec3<f32>(INV_GAMMA));

    textureStore(dst_texture, coord, vec4<f32>(ldr, hdr.a));
}

// ---------------------------------------------------------------------------
// Notes on Implementation
// ---------------------------------------------------------------------------
//
// ACES Pipeline:
//   The full ACES pipeline includes:
//   1. Input Device Transform (IDT) - converts camera RGB to ACES
//   2. Look Modification Transform (LMT) - artistic adjustments
//   3. Reference Rendering Transform (RRT) - scene-referred to display-referred
//   4. Output Device Transform (ODT) - converts to display colorspace
//
//   This shader implements a simplified RRT + ODT approximation.
//
// Exposure:
//   - Exposure value (EV) in stops
//   - Each stop doubles/halves the brightness
//   - EV = 0 means no adjustment
//   - EV = 1 means 2x brighter
//   - EV = -1 means 2x darker
//
// Gamma Correction:
//   - Converts linear light values to perceptually uniform encoding
//   - Standard sRGB gamma is approximately 2.2
//   - For accurate sRGB, use linear_to_srgb() instead of simple gamma
//
// Performance:
//   - ACES is ~10 ALU ops per pixel
//   - ACES fitted is ~25 ALU ops (matrix multiplications)
//   - Uncharted 2 is ~15 ALU ops
//   - Reinhard is ~3 ALU ops (simplest)
//
// Choosing a Curve:
//   - ACES: Best overall, industry standard
//   - ACES fitted: More accurate color, higher cost
//   - Uncharted 2: Good filmic look, used in games
//   - Reinhard: Simple, can look washed out
