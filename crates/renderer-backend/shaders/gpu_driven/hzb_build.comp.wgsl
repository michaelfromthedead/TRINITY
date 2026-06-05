// SPDX-License-Identifier: MIT
//
// hzb_build.comp.wgsl - Hierarchical-Z Buffer Construction (T-GPU-4.1)
//
// Builds HZB mip chain from depth buffer using max reduction.
// Each dispatch builds one mip level by downsampling 2x2 blocks from the previous level.
// For occlusion culling, we store the MAX depth (furthest geometry) per tile
// to enable conservative visibility tests.
//
// Algorithm:
//   1. Each thread processes one output texel
//   2. Sample 2x2 block from source (previous mip or depth buffer)
//   3. Compute max of all 4 samples
//   4. Write result to destination mip level
//
// Edge handling:
//   - Non-power-of-2 textures are handled by clamping coordinates
//   - This ensures the max propagates correctly to coarser mips
//
// Depth conventions (TRINITY uses reversed-Z):
//   - Near plane = 1.0, Far plane = 0.0
//   - MAX depth = furthest geometry (most conservative for occlusion)
//   - During culling: if instance.z > hzb.z, the instance is in front
//
// Workgroup size: 8x8 = 64 threads for optimal occupancy.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 8u;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct HZBBuildParams {
    // Source mip dimensions (width, height)
    src_width: u32,
    src_height: u32,
    // Destination mip dimensions (width, height)
    dst_width: u32,
    dst_height: u32,
    // Current mip level being generated (0 = generating mip 0 from depth buffer)
    current_mip: u32,
    // Total number of mip levels in the HZB chain
    num_mips: u32,
    // Flags: bit 0 = use min instead of max (for standard Z)
    flags: u32,
    // Padding for 16-byte alignment
    _pad0: u32,
}

// Flag constants
const FLAG_USE_MIN: u32 = 1u;  // Use min instead of max (for standard Z depth)

// ---------------------------------------------------------------------------
// Bindings - Mip 0 Generation (from depth buffer)
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> params: HZBBuildParams;

// Source depth buffer (for mip 0 generation)
@group(0) @binding(1) var src_depth: texture_2d<f32>;

// Destination storage texture (current mip being written)
@group(0) @binding(2) var dst_mip: texture_storage_2d<r32float, write>;

// Source mip texture (for mip 1+ generation, reads from previous mip)
// This is only used when current_mip > 0
@group(0) @binding(3) var src_mip: texture_2d<f32>;

// ---------------------------------------------------------------------------
// Shared Memory for Workgroup Reduction
// ---------------------------------------------------------------------------

// Shared memory for hierarchical reduction within workgroup
// Each thread stores its computed depth value here
var<workgroup> shared_depth: array<f32, 64>;

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Clamp coordinates to valid texture bounds
fn clamp_coord(coord: vec2<i32>, max_coord: vec2<i32>) -> vec2<i32> {
    return clamp(coord, vec2<i32>(0), max_coord);
}

/// Sample depth from the source depth buffer (mip 0 source)
fn sample_depth_buffer(coord: vec2<i32>) -> f32 {
    let max_coord = vec2<i32>(i32(params.src_width) - 1, i32(params.src_height) - 1);
    let clamped = clamp_coord(coord, max_coord);
    return textureLoad(src_depth, clamped, 0).r;
}

/// Sample depth from a source mip level (mip 1+ source)
fn sample_src_mip(coord: vec2<i32>) -> f32 {
    let max_coord = vec2<i32>(i32(params.src_width) - 1, i32(params.src_height) - 1);
    let clamped = clamp_coord(coord, max_coord);
    return textureLoad(src_mip, clamped, 0).r;
}

/// Compute reduction of 4 depth values (max for reversed-Z, min for standard)
fn reduce_depth_4(d0: f32, d1: f32, d2: f32, d3: f32) -> f32 {
    if ((params.flags & FLAG_USE_MIN) != 0u) {
        // Standard Z: min = furthest geometry
        return min(min(d0, d1), min(d2, d3));
    } else {
        // Reversed-Z (default): max = furthest geometry
        return max(max(d0, d1), max(d2, d3));
    }
}

/// Sample and reduce a 2x2 block from the depth buffer
fn reduce_depth_2x2_from_buffer(base_coord: vec2<i32>) -> f32 {
    let d00 = sample_depth_buffer(base_coord + vec2<i32>(0, 0));
    let d10 = sample_depth_buffer(base_coord + vec2<i32>(1, 0));
    let d01 = sample_depth_buffer(base_coord + vec2<i32>(0, 1));
    let d11 = sample_depth_buffer(base_coord + vec2<i32>(1, 1));
    return reduce_depth_4(d00, d10, d01, d11);
}

/// Sample and reduce a 2x2 block from the source mip
fn reduce_depth_2x2_from_mip(base_coord: vec2<i32>) -> f32 {
    let d00 = sample_src_mip(base_coord + vec2<i32>(0, 0));
    let d10 = sample_src_mip(base_coord + vec2<i32>(1, 0));
    let d01 = sample_src_mip(base_coord + vec2<i32>(0, 1));
    let d11 = sample_src_mip(base_coord + vec2<i32>(1, 1));
    return reduce_depth_4(d00, d10, d01, d11);
}

// ---------------------------------------------------------------------------
// Main Entry Point: Build Single Mip Level
// ---------------------------------------------------------------------------

/// Build a single HZB mip level.
///
/// When current_mip == 0: reads from depth buffer, writes to mip 0
/// When current_mip > 0:  reads from mip (current_mip - 1), writes to current_mip
///
/// This shader is dispatched once per mip level to build the full chain.
@compute @workgroup_size(8, 8, 1)
fn build_hzb_mip(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32
) {
    // Early-out if beyond destination texture bounds
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
        return;
    }

    // Calculate the base coordinate in the source texture/mip
    // Each output texel corresponds to a 2x2 block in the source
    let base_coord = vec2<i32>(gid.xy) * 2;

    // Compute max depth of the 2x2 block
    var z_max: f32;
    if (params.current_mip == 0u) {
        // Mip 0: sample from depth buffer
        z_max = reduce_depth_2x2_from_buffer(base_coord);
    } else {
        // Mip 1+: sample from previous mip level
        z_max = reduce_depth_2x2_from_mip(base_coord);
    }

    // Write to the destination mip level
    textureStore(dst_mip, vec2<i32>(gid.xy), vec4<f32>(z_max, 0.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Alternative Entry Point: Build Mip 0 Only (from depth buffer)
// ---------------------------------------------------------------------------

/// Build HZB mip 0 directly from depth buffer.
/// Specialized version that always reads from the depth buffer.
@compute @workgroup_size(8, 8, 1)
fn build_hzb_mip0(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    // Early-out if beyond destination texture bounds
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
        return;
    }

    // Calculate the base coordinate in the depth buffer
    let base_coord = vec2<i32>(gid.xy) * 2;

    // Compute max depth of the 2x2 block
    let z_max = reduce_depth_2x2_from_buffer(base_coord);

    // Write to mip 0
    textureStore(dst_mip, vec2<i32>(gid.xy), vec4<f32>(z_max, 0.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Alternative Entry Point: Build Subsequent Mips (from previous mip)
// ---------------------------------------------------------------------------

/// Build HZB mip N from mip N-1.
/// Specialized version that always reads from the source mip texture.
@compute @workgroup_size(8, 8, 1)
fn build_hzb_mip_chain(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    // Early-out if beyond destination texture bounds
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
        return;
    }

    // Calculate the base coordinate in the source mip
    let base_coord = vec2<i32>(gid.xy) * 2;

    // Compute max depth of the 2x2 block from source mip
    let z_max = reduce_depth_2x2_from_mip(base_coord);

    // Write to destination mip
    textureStore(dst_mip, vec2<i32>(gid.xy), vec4<f32>(z_max, 0.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Notes on Usage
// ---------------------------------------------------------------------------
//
// To build the complete HZB chain:
//
// 1. First dispatch: build_hzb_mip0
//    - Input: depth buffer (texture_2d)
//    - Output: HZB mip 0 (texture_storage_2d)
//    - params.src_width/height = depth buffer size
//    - params.dst_width/height = mip 0 size (depth size / 2)
//    - params.current_mip = 0
//
// 2. Subsequent dispatches: build_hzb_mip_chain
//    - Input: HZB mip N-1 (texture_2d view of previous mip)
//    - Output: HZB mip N (texture_storage_2d view of current mip)
//    - params.src_width/height = mip N-1 size
//    - params.dst_width/height = mip N size (previous / 2)
//    - params.current_mip = N
//
// The generic build_hzb_mip entry point can be used for all mips by
// checking current_mip to determine the source.
//
// Dispatch sizes:
//   workgroups_x = (dst_width + 7) / 8
//   workgroups_y = (dst_height + 7) / 8
//   workgroups_z = 1
