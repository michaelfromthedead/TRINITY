// SPDX-License-Identifier: MIT
//
// hiz_downsample.wgsl - HiZ Downsample Compute Shader (T-WGPU-P6.4.2).
//
// Generates a single mip level of the Hierarchical-Z pyramid by performing
// a 2x2 max reduction on the source mip level. Each output texel stores the
// maximum depth value from its corresponding 2x2 block in the source.
//
// Algorithm:
//   1. For each output texel at (gid.x, gid.y):
//   2. Compute the corresponding 2x2 block in the source mip
//   3. Load all 4 texels from the source mip using textureLoad
//   4. Take the maximum depth value (reverse-Z: max = closest)
//   5. Write to the destination mip
//
// Depth Convention (Reverse-Z):
//   TRINITY uses reversed-Z (near=1.0, far=0.0):
//   - Larger Z values are closer to the camera
//   - MAX depth = closest geometry (conservative for culling)
//   - In occlusion tests: if object.z < hiz.z, object is occluded
//
// Edge Handling:
//   - For odd-sized source textures, coordinates are clamped to valid range
//   - This ensures no out-of-bounds reads while maintaining correct propagation
//
// Per-Mip Dispatch:
//   Each dispatch generates exactly one mip level from the previous level.
//   The caller iterates through mip levels, dispatching once per level.
//
// Workgroup Size: 8x8 threads (64 total) for optimal GPU occupancy.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Workgroup size (8x8x1 = 64 threads per workgroup).
const WORKGROUP_SIZE: u32 = 8u;

// ---------------------------------------------------------------------------
// Bindings: Group 0 - Textures
// ---------------------------------------------------------------------------

/// Source mip level texture (read-only).
/// Contains depth values from the previous mip level.
@group(0) @binding(0) var src_texture: texture_2d<f32>;

/// Linear sampler (optional, not used in this shader but kept for layout compatibility).
@group(0) @binding(1) var src_sampler: sampler;

/// Destination mip level storage texture (write-only).
/// Stores the max-reduced depth values.
@group(0) @binding(2) var dst_texture: texture_storage_2d<r32float, write>;

// ---------------------------------------------------------------------------
// Bindings: Group 1 - Parameters
// ---------------------------------------------------------------------------

/// Downsample parameters uniform buffer.
struct DownsampleParams {
    /// Source mip dimensions (width, height).
    src_size: vec2<u32>,
    /// Destination mip dimensions (width, height).
    dst_size: vec2<u32>,
    /// Current mip level being generated (for debugging/reference).
    mip_level: u32,
    /// Padding for 16-byte alignment.
    _padding: u32,
}

@group(1) @binding(0) var<uniform> params: DownsampleParams;

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Clamp coordinates to valid source texture bounds.
/// Handles edge cases for odd-sized textures.
fn clamp_src_coord(coord: vec2<u32>) -> vec2<i32> {
    let max_x = max(i32(params.src_size.x) - 1, 0);
    let max_y = max(i32(params.src_size.y) - 1, 0);
    return vec2<i32>(
        min(i32(coord.x), max_x),
        min(i32(coord.y), max_y)
    );
}

/// Load a single depth texel from the source texture.
/// Uses textureLoad for exact texel fetching (no filtering).
fn load_depth(coord: vec2<u32>) -> f32 {
    let clamped = clamp_src_coord(coord);
    return textureLoad(src_texture, clamped, 0).r;
}

/// Compute the maximum depth of a 2x2 block.
/// This is the core HiZ operation - max for reverse-Z gives conservative depth.
///
/// For reverse-Z (near=1.0, far=0.0):
///   - MAX depth = closest geometry (visible front)
///   - Conservative for occlusion: if object < max, it's behind everything
fn max_depth_2x2(base_coord: vec2<u32>) -> f32 {
    // Sample the 2x2 block from source mip
    let d00 = load_depth(base_coord + vec2<u32>(0u, 0u));
    let d10 = load_depth(base_coord + vec2<u32>(1u, 0u));
    let d01 = load_depth(base_coord + vec2<u32>(0u, 1u));
    let d11 = load_depth(base_coord + vec2<u32>(1u, 1u));

    // Return the maximum (closest to camera in reverse-Z)
    return max(max(d00, d10), max(d01, d11));
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

/// HiZ downsample kernel.
///
/// Each thread processes one output texel, reading a 2x2 block from the
/// source mip and writing the maximum depth to the destination mip.
///
/// For reverse-Z depth (TRINITY default):
///   - max(depths) gives the closest geometry
///   - Conservative for occlusion culling
@compute @workgroup_size(8, 8, 1)
fn hiz_downsample(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_coord = global_id.xy;

    // Bounds check: early-out if beyond destination texture bounds
    if (dst_coord.x >= params.dst_size.x || dst_coord.y >= params.dst_size.y) {
        return;
    }

    // Source coordinates: 2x2 block corresponding to this output texel
    let src_coord = dst_coord * 2u;

    // Max reduction of the 2x2 block (reverse-Z: max = closest)
    let max_depth = max_depth_2x2(src_coord);

    // Write to destination mip
    textureStore(dst_texture, vec2<i32>(dst_coord), vec4<f32>(max_depth, 0.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Alternative Entry Point: Min Reduction (Standard Z)
// ---------------------------------------------------------------------------

/// HiZ downsample using min reduction (for standard Z depth).
/// Use this if not using reverse-Z (near=0.0, far=1.0).
@compute @workgroup_size(8, 8, 1)
fn hiz_downsample_min(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_coord = global_id.xy;

    if (dst_coord.x >= params.dst_size.x || dst_coord.y >= params.dst_size.y) {
        return;
    }

    let src_coord = dst_coord * 2u;

    // Min reduction of the 2x2 block
    let d00 = load_depth(src_coord + vec2<u32>(0u, 0u));
    let d10 = load_depth(src_coord + vec2<u32>(1u, 0u));
    let d01 = load_depth(src_coord + vec2<u32>(0u, 1u));
    let d11 = load_depth(src_coord + vec2<u32>(1u, 1u));

    let min_depth = min(min(d00, d10), min(d01, d11));

    textureStore(dst_texture, vec2<i32>(dst_coord), vec4<f32>(min_depth, 0.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Notes on Usage
// ---------------------------------------------------------------------------
//
// Dispatch calculation:
//   workgroups_x = (dst_width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
//   workgroups_y = (dst_height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
//   workgroups_z = 1
//
// Mip chain generation:
//   for mip in 1..num_mips {
//       bind(src_mip_view[mip-1], dst_mip_view[mip], params[mip])
//       dispatch(workgroups_x[mip], workgroups_y[mip], 1)
//   }
//
// Memory layout:
//   - DownsampleParams: 24 bytes (6 u32 fields with padding)
//   - Must be 16-byte aligned for uniform buffer binding
//
// Performance considerations:
//   - textureLoad is used over textureSample for exact texel access
//   - Workgroup size 8x8 balances occupancy and cache efficiency
//   - Per-mip dispatch minimizes memory bandwidth
