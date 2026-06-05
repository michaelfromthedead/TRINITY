// SPDX-License-Identifier: MIT
//
// hiz_generate.comp.wgsl - Hierarchical Z-Buffer (HiZ) Mip Generation Shader (T-GIR-P4.1).
//
// Generates a single mip level of the hierarchical depth buffer by downsampling
// the previous mip level. Each output texel stores the maximum depth of its
// corresponding 2x2 block from the source mip level.
//
// HiZ buffers enable efficient screen-space ray marching by allowing early-out
// tests at coarse mip levels. During ray marching, if the ray's depth is less
// than the maximum depth at a coarse level, the ray cannot intersect geometry
// within that region.
//
// Algorithm:
//   1. For each output texel at (gid.x, gid.y):
//   2. Compute the corresponding 2x2 block in the source mip
//   3. Sample all 4 texels from the source mip
//   4. Take the maximum depth value
//   5. Write to the destination mip
//
// Edge handling:
//   - For odd-sized source textures, the shader clamps coordinates to valid range
//   - This ensures no out-of-bounds reads while maintaining correct max propagation
//
// Workgroup size: 8x8 threads for optimal occupancy on modern GPUs.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 8u;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct HiZUniforms {
    src_mip_level: u32,     // Source mip level index (for reference)
    dst_mip_level: u32,     // Destination mip level index (for reference)
    src_size: vec2<u32>,    // Source mip dimensions (width, height)
    dst_size: vec2<u32>,    // Destination mip dimensions (width, height)
    _pad: vec2<u32>,        // Padding for alignment
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var src_depth: texture_2d<f32>;
@group(0) @binding(1) var dst_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> uniforms: HiZUniforms;

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Clamp coordinates to valid source texture bounds.
/// Handles edge cases for odd-sized textures.
fn clamp_coord(coord: vec2<i32>, max_coord: vec2<i32>) -> vec2<i32> {
    return clamp(coord, vec2<i32>(0), max_coord);
}

/// Sample a single texel from the source depth texture.
/// Uses textureLoad for exact texel fetching (no filtering).
fn sample_depth(coord: vec2<i32>) -> f32 {
    let max_coord = vec2<i32>(uniforms.src_size) - vec2<i32>(1);
    let clamped = clamp_coord(coord, max_coord);
    return textureLoad(src_depth, clamped, 0).r;
}

/// Compute the maximum depth of a 2x2 block.
/// This is the core HiZ operation - conservative depth for occlusion testing.
fn max_depth_2x2(base_coord: vec2<i32>) -> f32 {
    // Sample the 2x2 block
    let z00 = sample_depth(base_coord + vec2<i32>(0, 0));
    let z10 = sample_depth(base_coord + vec2<i32>(1, 0));
    let z01 = sample_depth(base_coord + vec2<i32>(0, 1));
    let z11 = sample_depth(base_coord + vec2<i32>(1, 1));

    // Return the maximum (furthest depth)
    // Using reversed-Z, max depth means "furthest from camera"
    return max(max(z00, z10), max(z01, z11));
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

/// HiZ downsample kernel.
///
/// Each thread processes one output texel, reading a 2x2 block from the
/// source mip and writing the maximum depth to the destination mip.
@compute @workgroup_size(8, 8, 1)
fn hiz_downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Early-out if beyond destination texture bounds
    if (gid.x >= uniforms.dst_size.x || gid.y >= uniforms.dst_size.y) {
        return;
    }

    // Calculate the base coordinate in the source mip
    // Each output texel corresponds to a 2x2 block in the source
    let base_coord = vec2<i32>(gid.xy) * 2;

    // Compute max depth of the 2x2 block
    let z_max = max_depth_2x2(base_coord);

    // Write to the destination mip level
    textureStore(dst_depth, vec2<i32>(gid.xy), vec4<f32>(z_max, 0.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Alternative Entry Points (Future Expansion)
// ---------------------------------------------------------------------------

/// HiZ downsample with min/max for two-sided culling (not currently used).
/// Stores both min and max depth for tighter culling bounds.
/// Would require RG32Float format instead of R32Float.
// @compute @workgroup_size(8, 8, 1)
// fn hiz_downsample_minmax(@builtin(global_invocation_id) gid: vec3<u32>) {
//     if (gid.x >= uniforms.dst_size.x || gid.y >= uniforms.dst_size.y) {
//         return;
//     }
//
//     let base_coord = vec2<i32>(gid.xy) * 2;
//
//     let z00 = sample_depth(base_coord + vec2<i32>(0, 0));
//     let z10 = sample_depth(base_coord + vec2<i32>(1, 0));
//     let z01 = sample_depth(base_coord + vec2<i32>(0, 1));
//     let z11 = sample_depth(base_coord + vec2<i32>(1, 1));
//
//     let z_max = max(max(z00, z10), max(z01, z11));
//     let z_min = min(min(z00, z10), min(z01, z11));
//
//     // Would write to RG32Float texture: (min, max)
//     // textureStore(dst_depth, vec2<i32>(gid.xy), vec4<f32>(z_min, z_max, 0.0, 0.0));
// }

// ---------------------------------------------------------------------------
// Notes on Depth Conventions
// ---------------------------------------------------------------------------
//
// TRINITY uses reversed-Z (near=1.0, far=0.0) for better depth precision:
// - Larger Z values are closer to the camera
// - Smaller Z values are further from the camera
// - HiZ stores max(depth) to find the "furthest" geometry in each tile
// - During ray marching: if ray.z < hiz.z, the ray is behind all geometry
//
// For standard Z (near=0.0, far=1.0), you would swap max/min in the
// conservative test.
