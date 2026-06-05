// SPDX-License-Identifier: MIT
//
// downsample.wgsl - 2x Downsampling Compute Shader (T-WGPU-P3.10.5).
//
// Implements 2x downsampling using a box filter (average of 4 pixels).
// Used for mip generation, bloom downsampling, and resolution reduction.
//
// Algorithm:
//   1. For each output pixel, sample 2x2 block from source
//   2. Average the 4 samples using box filter
//   3. Write averaged result to destination
//
// Filter modes:
//   - Box filter (mode 0): Simple average, fastest
//   - Bilinear filter (mode 1): Hardware sampler, better quality
//   - Karis average (mode 2): Weighted by luminance, reduces fireflies
//
// Performance notes:
//   - Simple 4-tap average is fastest for most use cases
//   - Karis average preferred for HDR bloom to avoid fireflies
//   - Bilinear filter useful when exact pixel boundaries aren't critical
//
// Workgroup size: 8x8 threads for optimal GPU occupancy.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 8u;

// Filter mode constants
const FILTER_BOX: u32 = 0u;
const FILTER_BILINEAR: u32 = 1u;
const FILTER_KARIS: u32 = 2u;

// Luminance coefficients (Rec. 709)
const LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct DownsampleUniforms {
    src_dims: vec2<u32>,    // Source texture dimensions
    dst_dims: vec2<u32>,    // Destination texture dimensions (src / 2)
    filter_mode: u32,       // 0 = box, 1 = bilinear, 2 = karis
    mip_level: u32,         // Current mip level being generated
    _pad0: u32,
    _pad1: u32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> uniforms: DownsampleUniforms;
@group(0) @binding(3) var linear_sampler: sampler;

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Clamp coordinates to valid source texture bounds.
fn clamp_coord(coord: vec2<i32>, max_coord: vec2<i32>) -> vec2<i32> {
    return clamp(coord, vec2<i32>(0), max_coord);
}

/// Load a single texel from the source texture.
fn load_texel(coord: vec2<i32>) -> vec4<f32> {
    let max_coord = vec2<i32>(uniforms.src_dims) - vec2<i32>(1);
    let clamped = clamp_coord(coord, max_coord);
    return textureLoad(src_texture, clamped, 0);
}

/// Compute luminance of a color.
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, LUMA_WEIGHTS);
}

/// Box filter: Simple average of 2x2 block.
/// Fastest option, good for most use cases.
fn box_filter_2x2(base_coord: vec2<i32>) -> vec4<f32> {
    let c00 = load_texel(base_coord + vec2<i32>(0, 0));
    let c10 = load_texel(base_coord + vec2<i32>(1, 0));
    let c01 = load_texel(base_coord + vec2<i32>(0, 1));
    let c11 = load_texel(base_coord + vec2<i32>(1, 1));

    return (c00 + c10 + c01 + c11) * 0.25;
}

/// Bilinear filter: Uses hardware sampler for smooth interpolation.
fn bilinear_filter(output_coord: vec2<u32>) -> vec4<f32> {
    // Calculate normalized UV coordinates centered on the 2x2 block
    let uv = (vec2<f32>(output_coord) + vec2<f32>(0.5)) / vec2<f32>(uniforms.dst_dims);
    return textureSampleLevel(src_texture, linear_sampler, uv, 0.0);
}

/// Karis average: Luminance-weighted average to reduce fireflies in HDR bloom.
///
/// Based on Brian Karis's technique from "Tone Mapping" (2013).
/// Weights each sample by 1 / (1 + luminance) to reduce the contribution
/// of extremely bright pixels that cause bloom artifacts.
fn karis_average_2x2(base_coord: vec2<i32>) -> vec4<f32> {
    let c00 = load_texel(base_coord + vec2<i32>(0, 0));
    let c10 = load_texel(base_coord + vec2<i32>(1, 0));
    let c01 = load_texel(base_coord + vec2<i32>(0, 1));
    let c11 = load_texel(base_coord + vec2<i32>(1, 1));

    // Compute luminance-based weights
    let w00 = 1.0 / (1.0 + luminance(c00.rgb));
    let w10 = 1.0 / (1.0 + luminance(c10.rgb));
    let w01 = 1.0 / (1.0 + luminance(c01.rgb));
    let w11 = 1.0 / (1.0 + luminance(c11.rgb));

    let total_weight = w00 + w10 + w01 + w11;

    return (c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11) / total_weight;
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

/// Downsample kernel (2x reduction).
///
/// Each thread processes one output texel, reading from a 2x2 block in the
/// source texture and writing the filtered result to the destination.
@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Early-out if beyond destination texture bounds
    if (gid.x >= uniforms.dst_dims.x || gid.y >= uniforms.dst_dims.y) {
        return;
    }

    var color: vec4<f32>;

    switch (uniforms.filter_mode) {
        case FILTER_BILINEAR: {
            color = bilinear_filter(gid.xy);
        }
        case FILTER_KARIS: {
            // Karis average for first mip only (where fireflies are worst)
            // Fall back to box filter for subsequent mips
            if (uniforms.mip_level == 0u) {
                let base_coord = vec2<i32>(gid.xy) * 2;
                color = karis_average_2x2(base_coord);
            } else {
                let base_coord = vec2<i32>(gid.xy) * 2;
                color = box_filter_2x2(base_coord);
            }
        }
        default: {
            // FILTER_BOX (default): 2x2 box filter average
            let base_coord = vec2<i32>(gid.xy) * 2;
            color = box_filter_2x2(base_coord);
        }
    }

    // Write to the destination texture
    textureStore(dst_texture, vec2<i32>(gid.xy), color);
}

// ---------------------------------------------------------------------------
// Alternate Entry Point: Mip Chain Generation
// ---------------------------------------------------------------------------

/// Generate multiple mip levels in a single pass (up to 4 levels).
/// Useful for reducing dispatch overhead when generating full mip chains.
///
/// Note: This entry point requires 4 separate output textures bound.
/// Use the standard `downsample` entry point for single-level generation.
@compute @workgroup_size(8, 8, 1)
fn downsample_mip_chain(@builtin(global_invocation_id) gid: vec3<u32>) {
    // This is a simplified version that generates one mip at a time.
    // For full mip chain generation, use a multi-pass approach or
    // bind multiple output textures.

    // For now, delegate to the standard downsample
    if (gid.x >= uniforms.dst_dims.x || gid.y >= uniforms.dst_dims.y) {
        return;
    }

    let base_coord = vec2<i32>(gid.xy) * 2;
    let color = box_filter_2x2(base_coord);
    textureStore(dst_texture, vec2<i32>(gid.xy), color);
}

// ---------------------------------------------------------------------------
// Notes on Implementation
// ---------------------------------------------------------------------------
//
// NPOT (Non-Power-Of-Two) Handling:
//   - For odd-sized source textures, the last row/column is duplicated
//   - This maintains correct filtering without visible artifacts
//   - The clamp_coord function handles boundary cases
//
// Mip Chain Generation:
//   - For power-of-two textures: dst_dims = src_dims / 2
//   - For NPOT: dst_dims = floor((src_dims + 1) / 2)
//   - Continue until either dimension reaches 1
//
// HDR Bloom Pipeline:
//   1. Threshold pass (extract bright pixels)
//   2. Downsample chain with Karis average (first level) + box filter (rest)
//   3. Upsample chain with additive blending
//   4. Composite onto scene
//
// Performance:
//   - Box filter: 4 texture loads per output pixel
//   - Bilinear filter: 1 sampled texture read (hardware accelerated)
//   - Karis average: 4 texture loads + 4 luminance calculations
//   - Workgroup size 8x8 = 64 threads balances occupancy and cache usage
