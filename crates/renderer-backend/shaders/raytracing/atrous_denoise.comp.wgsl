// SPDX-License-Identifier: MIT
//
// atrous_denoise.comp.wgsl - A Trous Wavelet Spatial Denoiser (T-RT-P1.9).
//
// Implements edge-aware spatial denoising using the A Trous wavelet transform.
// This is a key component of modern ray tracing denoisers, typically run in
// multiple iterations with increasing step sizes (1, 2, 4, 8, ...) to filter
// at different spatial frequencies.
//
// Algorithm:
//   1. Sample center pixel (color, depth, normal)
//   2. Apply 5-tap separable A Trous kernel at current step_size
//   3. Compute edge-stopping weights based on:
//      - Depth similarity (preserve depth discontinuities)
//      - Normal similarity (preserve geometric edges)
//      - Luminance similarity (preserve color edges)
//   4. Accumulate weighted samples and normalize
//
// The A Trous ("with holes") wavelet inserts zeros between kernel samples,
// effectively filtering at power-of-two scales without downsampling.
//
// References:
//   - Dammertz et al., "Edge-Avoiding A-Trous Wavelet Transform for fast
//     Global Illumination Filtering" (HPG 2010)
//   - NVIDIA SVGF (Spatiotemporal Variance-Guided Filtering)
//
// Workgroup size: 8x8 threads for optimal GPU occupancy.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 8u;

// A Trous 5-tap kernel (B3 spline approximation)
// [1/16, 4/16, 6/16, 4/16, 1/16] = [0.0625, 0.25, 0.375, 0.25, 0.0625]
const KERNEL_WEIGHTS: array<f32, 5> = array<f32, 5>(
    0.0625,  // 1/16
    0.25,    // 4/16
    0.375,   // 6/16
    0.25,    // 4/16
    0.0625   // 1/16
);

// Kernel offsets (centered at 0)
const KERNEL_OFFSETS: array<i32, 5> = array<i32, 5>(-2, -1, 0, 1, 2);

// Minimum weight to avoid division by zero
const MIN_WEIGHT: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

/// Denoising parameters for the current A Trous iteration.
///
/// # Fields
///
/// - `step_size`: Wavelet step size (1, 2, 4, 8, ...). Each iteration doubles
///   the effective filter radius without increasing sample count.
/// - `sigma_color`: Color/luminance similarity weight. Higher values allow
///   more color variation before edge stopping kicks in.
/// - `sigma_depth`: Depth similarity weight. Controls sensitivity to depth
///   discontinuities. Lower values preserve depth edges more aggressively.
/// - `sigma_normal`: Normal similarity weight. Controls sensitivity to
///   surface orientation changes.
/// - `width`: Input texture width in pixels.
/// - `height`: Input texture height in pixels.
struct DenoiseParams {
    step_size: u32,      // 1, 2, 4, 8 for each iteration
    sigma_color: f32,    // color similarity weight (e.g., 4.0)
    sigma_depth: f32,    // depth edge stopping (e.g., 1.0)
    sigma_normal: f32,   // normal edge stopping (e.g., 128.0)
    width: u32,
    height: u32,
    _padding: vec2<u32>,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

/// Noisy input color buffer (result from ray tracing or previous A Trous pass)
@group(0) @binding(0) var input_texture: texture_2d<f32>;

/// Depth buffer for edge-stopping
@group(0) @binding(1) var depth_texture: texture_depth_2d;

/// World-space or view-space normal buffer for edge-stopping
@group(0) @binding(2) var normal_texture: texture_2d<f32>;

/// Denoised output (RGBA16Float for HDR support)
@group(0) @binding(3) var output_texture: texture_storage_2d<rgba16float, write>;

/// Denoising parameters
@group(0) @binding(4) var<uniform> params: DenoiseParams;

// ---------------------------------------------------------------------------
// Edge-Stopping Weight Functions
// ---------------------------------------------------------------------------

/// Compute luminance of an RGB color using Rec. 709 coefficients.
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/// Compute depth-based edge-stopping weight.
///
/// Uses exponential falloff based on depth difference. The sigma parameter
/// controls the sensitivity - lower values create sharper depth edges.
///
/// # Arguments
/// - `center_depth`: Depth at the center pixel
/// - `sample_depth`: Depth at the sample pixel
/// - `sigma`: Depth similarity parameter
///
/// # Returns
/// Weight in [0, 1] where 1 = identical depth, 0 = large depth difference
fn depth_weight(center_depth: f32, sample_depth: f32, sigma: f32) -> f32 {
    // Gradient-based depth difference (linearized)
    // Using absolute difference normalized by sigma
    let depth_diff = abs(center_depth - sample_depth);

    // Exponential falloff: exp(-diff^2 / sigma^2)
    // More robust: use linear falloff with clamping for better stability
    let normalized = depth_diff / max(sigma, MIN_WEIGHT);
    return exp(-normalized * normalized);
}

/// Compute normal-based edge-stopping weight.
///
/// Uses the dot product between normals to measure surface similarity.
/// Sharp geometric edges (very different normals) get low weights.
///
/// # Arguments
/// - `center_normal`: Normal at the center pixel (assumed normalized)
/// - `sample_normal`: Normal at the sample pixel (assumed normalized)
/// - `sigma`: Normal similarity parameter (higher = more tolerant)
///
/// # Returns
/// Weight in [0, 1] where 1 = parallel normals, 0 = perpendicular or opposite
fn normal_weight(center_normal: vec3<f32>, sample_normal: vec3<f32>, sigma: f32) -> f32 {
    // Dot product gives cos(angle) between normals
    let dot_val = max(dot(center_normal, sample_normal), 0.0);

    // Use power function for sharper falloff
    // Higher sigma = slower falloff = more blurring across normals
    let exponent = sigma;
    return pow(dot_val, exponent);
}

/// Compute luminance-based edge-stopping weight.
///
/// Prevents blurring across color edges by comparing luminance values.
/// This preserves texture detail and sharp color boundaries.
///
/// # Arguments
/// - `center_lum`: Luminance at the center pixel
/// - `sample_lum`: Luminance at the sample pixel
/// - `sigma`: Luminance similarity parameter
///
/// # Returns
/// Weight in [0, 1] where 1 = same luminance, 0 = large luminance difference
fn luminance_weight(center_lum: f32, sample_lum: f32, sigma: f32) -> f32 {
    let lum_diff = abs(center_lum - sample_lum);
    let normalized = lum_diff / max(sigma, MIN_WEIGHT);
    return exp(-normalized * normalized);
}

// ---------------------------------------------------------------------------
// Texture Sampling Helpers
// ---------------------------------------------------------------------------

/// Sample color at the given pixel coordinates, clamping to texture bounds.
fn sample_color(coord: vec2<i32>) -> vec4<f32> {
    let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(i32(params.width) - 1, i32(params.height) - 1));
    return textureLoad(input_texture, clamped, 0);
}

/// Sample depth at the given pixel coordinates, clamping to texture bounds.
fn sample_depth(coord: vec2<i32>) -> f32 {
    let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(i32(params.width) - 1, i32(params.height) - 1));
    return textureLoad(depth_texture, clamped, 0);
}

/// Sample and decode normal at the given pixel coordinates.
/// Assumes normals are stored as [0,1] range and need decoding to [-1,1].
fn sample_normal(coord: vec2<i32>) -> vec3<f32> {
    let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(i32(params.width) - 1, i32(params.height) - 1));
    let raw = textureLoad(normal_texture, clamped, 0);

    // Decode from [0,1] to [-1,1] range
    let decoded = raw.xyz * 2.0 - 1.0;

    // Normalize to handle any quantization errors
    let len_sq = dot(decoded, decoded);
    if len_sq > MIN_WEIGHT {
        return decoded / sqrt(len_sq);
    }
    return vec3<f32>(0.0, 0.0, 1.0); // Default up normal
}

// ---------------------------------------------------------------------------
// Main Compute Entry Point
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Bounds check - early exit for out-of-bounds threads
    if global_id.x >= params.width || global_id.y >= params.height {
        return;
    }

    let center_coord = vec2<i32>(global_id.xy);

    // Load center pixel data
    let center_color = sample_color(center_coord);
    let center_depth = sample_depth(center_coord);
    let center_normal = sample_normal(center_coord);
    let center_lum = luminance(center_color.rgb);

    // Skip sky pixels (depth at far plane) - pass through unchanged
    if center_depth >= 1.0 {
        textureStore(output_texture, center_coord, center_color);
        return;
    }

    // A Trous filtering with 5x5 separable kernel
    // Step size determines the spacing between samples
    let step = i32(params.step_size);

    // Accumulate weighted samples
    var sum_color = vec4<f32>(0.0);
    var sum_weight = 0.0;

    // Apply 2D separable kernel
    for (var j = 0; j < 5; j++) {
        for (var i = 0; i < 5; i++) {
            // Calculate sample offset with A Trous spacing
            let offset = vec2<i32>(
                KERNEL_OFFSETS[i] * step,
                KERNEL_OFFSETS[j] * step
            );
            let sample_coord = center_coord + offset;

            // Sample at offset location
            let sample_color = sample_color(sample_coord);
            let sample_depth = sample_depth(sample_coord);
            let sample_normal = sample_normal(sample_coord);
            let sample_lum = luminance(sample_color.rgb);

            // Compute spatial kernel weight (separable 2D)
            let kernel_w = KERNEL_WEIGHTS[i] * KERNEL_WEIGHTS[j];

            // Compute edge-stopping weights
            let d_weight = depth_weight(center_depth, sample_depth, params.sigma_depth);
            let n_weight = normal_weight(center_normal, sample_normal, params.sigma_normal);
            let l_weight = luminance_weight(center_lum, sample_lum, params.sigma_color);

            // Combined weight: spatial * edge-stopping factors
            let w = kernel_w * d_weight * n_weight * l_weight;

            // Accumulate
            sum_color += sample_color * w;
            sum_weight += w;
        }
    }

    // Normalize by total weight (fallback to center if all weights are ~0)
    var result: vec4<f32>;
    if sum_weight > MIN_WEIGHT {
        result = sum_color / sum_weight;
    } else {
        result = center_color;
    }

    // Preserve alpha channel from input
    result.a = center_color.a;

    // Write filtered result
    textureStore(output_texture, center_coord, result);
}
