// SPDX-License-Identifier: MIT
//
// mip_generate.comp.wgsl - Mipmap Generation Compute Shader (T-WGPU-P2.3.4).
//
// Generates a single mip level by downsampling the previous level. Supports
// both box filter (2x2 average) and bilinear filter modes for different
// quality/performance tradeoffs.
//
// Algorithm:
//   1. For each output texel at (gid.x, gid.y):
//   2. Compute the corresponding 2x2 block in the source mip
//   3. Sample all 4 texels from the source mip
//   4. Apply the selected filter (average for box, weighted for bilinear)
//   5. Write to the destination mip
//
// Filter modes:
//   - Box filter (mode = 0): Simple 2x2 average, fastest
//   - Bilinear filter (mode = 1): Weighted average with smooth gradients
//
// Edge handling:
//   - For odd-sized source textures, coordinates are clamped to valid range
//   - Minimum output dimension is 1x1
//
// Workgroup size: 8x8 threads for optimal occupancy on modern GPUs.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 8u;

// Filter mode constants
const FILTER_BOX: u32 = 0u;
const FILTER_BILINEAR: u32 = 1u;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct MipUniforms {
    src_size: vec2<u32>,    // Source mip dimensions (width, height)
    dst_size: vec2<u32>,    // Destination mip dimensions (width, height)
    filter_mode: u32,       // 0 = box, 1 = bilinear
    _pad0: u32,             // Padding for alignment
    _pad1: u32,             // Padding for alignment
    _pad2: u32,             // Padding for alignment
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uniforms: MipUniforms;
@group(0) @binding(3) var linear_sampler: sampler;

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Clamp coordinates to valid source texture bounds.
/// Handles edge cases for odd-sized textures.
fn clamp_coord(coord: vec2<i32>, max_coord: vec2<i32>) -> vec2<i32> {
    return clamp(coord, vec2<i32>(0), max_coord);
}

/// Sample a single texel from the source texture using textureLoad.
/// Uses exact texel fetching (no filtering).
fn load_texel(coord: vec2<i32>) -> vec4<f32> {
    let max_coord = vec2<i32>(uniforms.src_size) - vec2<i32>(1);
    let clamped = clamp_coord(coord, max_coord);
    return textureLoad(src_texture, clamped, 0);
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

/// Bilinear filter: Weighted average for smoother gradients.
/// Uses hardware texture sampler for better quality.
fn bilinear_filter(output_coord: vec2<u32>) -> vec4<f32> {
    // Calculate normalized UV coordinates centered on the 2x2 block
    let uv = (vec2<f32>(output_coord) + vec2<f32>(0.5)) / vec2<f32>(uniforms.dst_size);
    return textureSampleLevel(src_texture, linear_sampler, uv, 0.0);
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

/// Mip downsample kernel.
///
/// Each thread processes one output texel, reading from the source mip
/// and writing to the destination mip using the selected filter.
@compute @workgroup_size(8, 8, 1)
fn mip_downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Early-out if beyond destination texture bounds
    if (gid.x >= uniforms.dst_size.x || gid.y >= uniforms.dst_size.y) {
        return;
    }

    var color: vec4<f32>;

    if (uniforms.filter_mode == FILTER_BILINEAR) {
        // Use bilinear sampling for smoother results
        color = bilinear_filter(gid.xy);
    } else {
        // Default: box filter (2x2 average)
        let base_coord = vec2<i32>(gid.xy) * 2;
        color = box_filter_2x2(base_coord);
    }

    // Write to the destination mip level
    textureStore(dst_texture, vec2<i32>(gid.xy), color);
}

// ---------------------------------------------------------------------------
// Format-Specific Entry Points
// ---------------------------------------------------------------------------

// Note: For different texture formats, you may need additional entry points
// with matching storage texture formats. The main entry point above uses
// rgba8unorm as the most common case.
//
// For other formats, create bind group layouts with the appropriate
// texture_storage_2d format:
// - rgba16float for HDR textures
// - r32float for single-channel textures
// - rgba32float for high-precision data

// ---------------------------------------------------------------------------
// Notes on Implementation
// ---------------------------------------------------------------------------
//
// NPOT (Non-Power-Of-Two) Handling:
// - For odd-sized textures, the last row/column is duplicated when sampling
// - This maintains correct filtering without artifacts
// - The clamp_coord function handles boundary cases
//
// Performance Considerations:
// - Box filter: ~4 texture loads per output texel
// - Bilinear filter: 1 sampled texture read (hardware accelerated)
// - Workgroup size of 8x8 = 64 threads balances occupancy and cache usage
//
// Memory Layout:
// - Source texture must have TEXTURE_BINDING usage
// - Destination texture must have STORAGE_BINDING usage
// - Both textures should be in the same memory pool for best performance
