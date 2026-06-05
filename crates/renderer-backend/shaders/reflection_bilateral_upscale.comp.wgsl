// SPDX-License-Identifier: MIT
//
// reflection_bilateral_upscale.comp.wgsl - Bilateral Upscale for Reflection Buffer (T-GIR-P1.5).
//
// Upscales half-resolution reflection data to full resolution using edge-aware
// bilateral filtering. The filter preserves sharp reflection boundaries by
// weighting samples based on depth and normal similarity.
//
// Algorithm:
//   1. For each full-res output pixel, find corresponding half-res location
//   2. Sample the 4 nearest half-res pixels (bilinear footprint)
//   3. Compute bilateral weights based on:
//      - Depth similarity (closer depths = higher weight)
//      - Normal similarity (similar normals = higher weight)
//   4. Blend samples using normalized bilateral weights
//   5. Apply edge sharpening to preserve reflection boundaries
//
// Workgroup size: 8x8 threads for optimal GPU occupancy.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 8u;

// Bilateral filter parameters
const DEPTH_WEIGHT_SCALE: f32 = 10.0;
const NORMAL_WEIGHT_SCALE: f32 = 4.0;
const MIN_WEIGHT: f32 = 0.001;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct BilateralUpscaleUniforms {
    src_dims: vec2<u32>,       // Half-res source dimensions
    dst_dims: vec2<u32>,       // Full-res destination dimensions
    depth_threshold: f32,      // Depth similarity threshold
    normal_threshold: f32,     // Normal similarity threshold (dot product)
    edge_sharpness: f32,       // Edge preservation factor
    _pad: f32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> uniforms: BilateralUpscaleUniforms;

// Half-res reflection source (RGBA32Float: RGB color + roughness/hit_distance packed)
@group(0) @binding(1) var src_reflection: texture_2d<f32>;

// Full-res depth buffer for edge detection
@group(0) @binding(2) var depth_texture: texture_depth_2d;

// Full-res world-space normals for edge detection
@group(0) @binding(3) var normal_texture: texture_2d<f32>;

// Full-res output (RGBA16Float)
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba16float, write>;

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Convert full-res UV to half-res pixel coordinates.
fn full_to_half_uv(full_uv: vec2<f32>) -> vec2<f32> {
    return full_uv * vec2<f32>(uniforms.src_dims);
}

/// Sample depth at full-res coordinates.
fn sample_depth(coord: vec2<i32>) -> f32 {
    let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(uniforms.dst_dims) - 1);
    return textureLoad(depth_texture, clamped, 0);
}

/// Sample normal at full-res coordinates (assumed to be in [-1, 1] range).
fn sample_normal(coord: vec2<i32>) -> vec3<f32> {
    let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(uniforms.dst_dims) - 1);
    let raw = textureLoad(normal_texture, clamped, 0);
    // Unpack from [0,1] to [-1,1] if stored as unsigned
    return normalize(raw.xyz * 2.0 - 1.0);
}

/// Sample half-res reflection buffer.
fn sample_reflection(coord: vec2<i32>) -> vec4<f32> {
    let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(uniforms.src_dims) - 1);
    return textureLoad(src_reflection, clamped, 0);
}

/// Compute depth-based bilateral weight.
fn depth_weight(center_depth: f32, sample_depth: f32) -> f32 {
    let diff = abs(center_depth - sample_depth);
    let normalized_diff = diff / uniforms.depth_threshold;
    return exp(-normalized_diff * normalized_diff * DEPTH_WEIGHT_SCALE);
}

/// Compute normal-based bilateral weight.
fn normal_weight(center_normal: vec3<f32>, sample_normal: vec3<f32>) -> f32 {
    let dot_val = max(dot(center_normal, sample_normal), 0.0);
    // Use threshold to create soft falloff
    let normalized = (dot_val - uniforms.normal_threshold) / (1.0 - uniforms.normal_threshold);
    let clamped = max(normalized, 0.0);
    return pow(clamped, NORMAL_WEIGHT_SCALE);
}

/// Compute combined bilateral weight.
fn bilateral_weight(
    center_depth: f32,
    center_normal: vec3<f32>,
    sample_depth: f32,
    sample_normal: vec3<f32>,
    spatial_weight: f32
) -> f32 {
    let d_weight = depth_weight(center_depth, sample_depth);
    let n_weight = normal_weight(center_normal, sample_normal);
    return max(d_weight * n_weight * spatial_weight, MIN_WEIGHT);
}

/// Get depth at half-res coordinate (sample from center of corresponding full-res region).
fn get_half_res_depth(half_coord: vec2<i32>) -> f32 {
    // Map half-res coord to full-res (center of 2x2 block)
    let full_coord = half_coord * 2 + 1;
    return sample_depth(full_coord);
}

/// Get normal at half-res coordinate.
fn get_half_res_normal(half_coord: vec2<i32>) -> vec3<f32> {
    let full_coord = half_coord * 2 + 1;
    return sample_normal(full_coord);
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Bounds check
    if global_id.x >= uniforms.dst_dims.x || global_id.y >= uniforms.dst_dims.y {
        return;
    }

    let full_coord = vec2<i32>(global_id.xy);
    let full_uv = (vec2<f32>(global_id.xy) + 0.5) / vec2<f32>(uniforms.dst_dims);

    // Get center pixel properties for bilateral comparison
    let center_depth = sample_depth(full_coord);
    let center_normal = sample_normal(full_coord);

    // Skip sky pixels (depth at far plane)
    if center_depth >= 1.0 {
        textureStore(output_texture, full_coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Map to half-res coordinates
    let half_uv = full_to_half_uv(full_uv);
    let half_base = vec2<i32>(floor(half_uv - 0.5));
    let fract_uv = fract(half_uv - 0.5);

    // Bilinear weights
    let w00 = (1.0 - fract_uv.x) * (1.0 - fract_uv.y);
    let w10 = fract_uv.x * (1.0 - fract_uv.y);
    let w01 = (1.0 - fract_uv.x) * fract_uv.y;
    let w11 = fract_uv.x * fract_uv.y;

    // Sample 4 nearest half-res pixels
    let coord00 = half_base;
    let coord10 = half_base + vec2<i32>(1, 0);
    let coord01 = half_base + vec2<i32>(0, 1);
    let coord11 = half_base + vec2<i32>(1, 1);

    let sample00 = sample_reflection(coord00);
    let sample10 = sample_reflection(coord10);
    let sample01 = sample_reflection(coord01);
    let sample11 = sample_reflection(coord11);

    // Get depth and normal at each half-res sample location
    let depth00 = get_half_res_depth(coord00);
    let depth10 = get_half_res_depth(coord10);
    let depth01 = get_half_res_depth(coord01);
    let depth11 = get_half_res_depth(coord11);

    let normal00 = get_half_res_normal(coord00);
    let normal10 = get_half_res_normal(coord10);
    let normal01 = get_half_res_normal(coord01);
    let normal11 = get_half_res_normal(coord11);

    // Compute bilateral weights
    let bw00 = bilateral_weight(center_depth, center_normal, depth00, normal00, w00);
    let bw10 = bilateral_weight(center_depth, center_normal, depth10, normal10, w10);
    let bw01 = bilateral_weight(center_depth, center_normal, depth01, normal01, w01);
    let bw11 = bilateral_weight(center_depth, center_normal, depth11, normal11, w11);

    let total_weight = bw00 + bw10 + bw01 + bw11;

    // Normalize and blend
    var result: vec4<f32>;
    if total_weight > MIN_WEIGHT {
        result = (sample00 * bw00 + sample10 * bw10 + sample01 * bw01 + sample11 * bw11) / total_weight;
    } else {
        // Fallback to simple bilinear if all weights are too low
        result = sample00 * w00 + sample10 * w10 + sample01 * w01 + sample11 * w11;
    }

    // Apply edge sharpening
    // Find the sample most similar to center and boost its contribution
    var max_weight = bw00;
    var max_sample = sample00;
    if bw10 > max_weight {
        max_weight = bw10;
        max_sample = sample10;
    }
    if bw01 > max_weight {
        max_weight = bw01;
        max_sample = sample01;
    }
    if bw11 > max_weight {
        max_weight = bw11;
        max_sample = sample11;
    }

    // Sharpen by blending toward the best-matching sample
    let sharpness_factor = saturate(uniforms.edge_sharpness * (1.0 - max_weight / total_weight));
    result = mix(result, max_sample, sharpness_factor);

    // Output reflection color (RGB) with alpha = 1
    textureStore(output_texture, full_coord, vec4<f32>(result.rgb, 1.0));
}
