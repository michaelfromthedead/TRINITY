// SPDX-License-Identifier: MIT
//
// blur_horizontal.wgsl - Horizontal Gaussian Blur Compute Shader (T-WGPU-P3.10.5).
//
// Implements separable 9-tap Gaussian blur (horizontal pass) with shared memory
// tile optimization for coalesced memory access and reduced global memory bandwidth.
//
// Algorithm:
//   1. Each workgroup loads a horizontal tile + halo into shared memory
//   2. Barrier synchronization ensures all data is loaded
//   3. Each thread computes 9-tap weighted average from shared memory
//   4. Result is written to output texture
//
// Kernel weights: Gaussian sigma ~1.5, normalized 9-tap kernel
//   [0.0276, 0.0663, 0.1238, 0.1801, 0.2042, 0.1801, 0.1238, 0.0663, 0.0276]
//
// Performance notes:
//   - Shared memory tile reduces global memory reads by ~4.5x vs naive approach
//   - Horizontal pass processes consecutive pixels for coalesced access
//   - Workgroup size 128x1 optimizes for horizontal memory layout
//
// Workgroup size: 128x1 threads, each processing one output pixel.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE_X: u32 = 128u;
const WORKGROUP_SIZE_Y: u32 = 1u;

// Kernel radius (4 pixels on each side = 9-tap kernel)
const KERNEL_RADIUS: i32 = 4;

// Tile size includes halo on both sides
const TILE_SIZE: u32 = WORKGROUP_SIZE_X + 8u; // 128 + 2*4 = 136

// 9-tap Gaussian kernel weights (sigma ~1.5, sum = 1.0)
const KERNEL_WEIGHT_0: f32 = 0.2042;  // center
const KERNEL_WEIGHT_1: f32 = 0.1801;  // +/- 1
const KERNEL_WEIGHT_2: f32 = 0.1238;  // +/- 2
const KERNEL_WEIGHT_3: f32 = 0.0663;  // +/- 3
const KERNEL_WEIGHT_4: f32 = 0.0276;  // +/- 4

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct BlurUniforms {
    src_dims: vec2<u32>,    // Source texture dimensions
    dst_dims: vec2<u32>,    // Destination texture dimensions (same for blur)
    blur_scale: f32,        // Scale factor for blur radius (1.0 = standard)
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> uniforms: BlurUniforms;

// ---------------------------------------------------------------------------
// Shared Memory
// ---------------------------------------------------------------------------

// Shared memory tile for horizontal strip with halo
// Each thread loads one pixel, plus threads at edges load halo pixels
var<workgroup> tile: array<vec4<f32>, 136>; // TILE_SIZE

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Clamp texture coordinates to valid bounds.
fn clamp_coord(coord: vec2<i32>, dims: vec2<u32>) -> vec2<i32> {
    return clamp(coord, vec2<i32>(0), vec2<i32>(dims) - 1);
}

/// Load a texel with edge clamping.
fn load_texel(coord: vec2<i32>) -> vec4<f32> {
    let clamped = clamp_coord(coord, uniforms.src_dims);
    return textureLoad(src_texture, clamped, 0);
}

/// Apply 9-tap Gaussian kernel using values from shared memory.
fn apply_kernel(center_idx: u32) -> vec4<f32> {
    var result = tile[center_idx] * KERNEL_WEIGHT_0;

    result += tile[center_idx - 1u] * KERNEL_WEIGHT_1;
    result += tile[center_idx + 1u] * KERNEL_WEIGHT_1;

    result += tile[center_idx - 2u] * KERNEL_WEIGHT_2;
    result += tile[center_idx + 2u] * KERNEL_WEIGHT_2;

    result += tile[center_idx - 3u] * KERNEL_WEIGHT_3;
    result += tile[center_idx + 3u] * KERNEL_WEIGHT_3;

    result += tile[center_idx - 4u] * KERNEL_WEIGHT_4;
    result += tile[center_idx + 4u] * KERNEL_WEIGHT_4;

    return result;
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

/// Horizontal Gaussian blur kernel.
///
/// Each workgroup processes a horizontal strip of 128 pixels. Threads cooperate
/// to load the tile (including halo) into shared memory, then each thread
/// computes the blurred value for its output pixel.
@compute @workgroup_size(128, 1, 1)
fn blur_horizontal(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_x = lid.x;
    let global_y = i32(gid.y);

    // Base X coordinate for this workgroup's tile (including halo offset)
    let tile_base_x = i32(wid.x * WORKGROUP_SIZE_X) - KERNEL_RADIUS;

    // ── Phase 1: Load tile into shared memory ──────────────────────────────

    // Each thread loads one primary pixel
    let load_x = tile_base_x + i32(local_x);
    tile[local_x] = load_texel(vec2<i32>(load_x, global_y));

    // Threads at the end of the workgroup load the right halo
    // We need 8 extra pixels (KERNEL_RADIUS * 2)
    if (local_x < 8u) {
        let halo_x = tile_base_x + i32(WORKGROUP_SIZE_X) + i32(local_x);
        tile[WORKGROUP_SIZE_X + local_x] = load_texel(vec2<i32>(halo_x, global_y));
    }

    workgroupBarrier();

    // ── Phase 2: Compute blurred output ────────────────────────────────────

    // Early-out for pixels outside destination bounds
    if (gid.x >= uniforms.dst_dims.x || gid.y >= uniforms.dst_dims.y) {
        return;
    }

    // Index into shared memory for this thread's center pixel
    // Offset by KERNEL_RADIUS because tile includes left halo
    let center_idx = local_x + u32(KERNEL_RADIUS);

    // Apply Gaussian kernel
    let blurred = apply_kernel(center_idx);

    // Write to output
    textureStore(dst_texture, vec2<i32>(gid.xy), blurred);
}

// ---------------------------------------------------------------------------
// Notes on Implementation
// ---------------------------------------------------------------------------
//
// Shared Memory Layout:
//   tile[0..3]           = left halo (4 pixels)
//   tile[4..131]         = primary data (128 pixels)
//   tile[132..135]       = right halo (4 pixels)
//   Total: 136 pixels = 136 * 16 bytes = 2176 bytes
//
// Memory Bandwidth Optimization:
//   - Without shared memory: 128 threads * 9 reads = 1152 global reads
//   - With shared memory: 136 global reads + 128 * 9 shared reads
//   - Shared memory has ~10x bandwidth of global memory
//   - Net improvement: ~4-5x reduction in global memory traffic
//
// Edge Handling:
//   - Coordinates are clamped to texture bounds
//   - Border pixels use replicated edge values (clamp-to-edge)
//
// Precision:
//   - All intermediate calculations in full f32 precision
//   - Output format (rgba16float) preserves HDR range
