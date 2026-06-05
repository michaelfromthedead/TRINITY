// SPDX-License-Identifier: MIT
//
// histogram.wgsl - Luminance Histogram Compute Shader (T-WGPU-P3.10.5).
//
// Computes a 256-bin luminance histogram from an HDR image using a two-phase
// approach: workgroup-local histogram accumulation followed by global merge.
//
// Algorithm:
//   1. Initialize shared memory histogram bins to zero
//   2. Each thread processes multiple pixels (grid-stride loop)
//   3. Compute luminance and map to bin index [0, 255]
//   4. Atomically increment workgroup-local histogram
//   5. After barrier, atomically merge local histogram into global
//
// Luminance mapping:
//   - Input is HDR (can be > 1.0)
//   - Log-space mapping for better distribution across bins
//   - bin = clamp(log2(luminance + epsilon) * scale + offset, 0, 255)
//
// Performance notes:
//   - Local histogram in shared memory reduces atomic contention
//   - Grid-stride loop allows handling arbitrary image sizes
//   - 256 threads per workgroup = 1 thread per bin for merge phase
//
// Workgroup size: 256x1 threads (matches 256 histogram bins).

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 256u;
const NUM_BINS: u32 = 256u;

// Luminance coefficients (Rec. 709)
const LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Log-space mapping constants
// Maps luminance range [MIN_LUMINANCE, MAX_LUMINANCE] to bins [0, 255]
const MIN_LOG_LUMINANCE: f32 = -10.0;  // log2(~0.001)
const MAX_LOG_LUMINANCE: f32 = 4.0;    // log2(16) for HDR
const LOG_RANGE: f32 = 14.0;           // MAX - MIN
const EPSILON: f32 = 0.00001;          // Avoid log(0)

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct HistogramUniforms {
    src_dims: vec2<u32>,     // Source texture dimensions
    num_pixels: u32,         // Total pixel count (width * height)
    min_luminance: f32,      // Minimum log luminance for mapping
    max_luminance: f32,      // Maximum log luminance for mapping
    _pad0: f32,
    _pad1: u32,
    _pad2: u32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, 256>;
@group(0) @binding(2) var<uniform> uniforms: HistogramUniforms;

// ---------------------------------------------------------------------------
// Shared Memory
// ---------------------------------------------------------------------------

// Workgroup-local histogram (256 bins)
// Using atomics for concurrent access within workgroup
var<workgroup> local_histogram: array<atomic<u32>, 256>;

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Compute luminance of a color.
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, LUMA_WEIGHTS);
}

/// Map luminance to histogram bin index [0, 255].
/// Uses log-space mapping for better distribution of HDR values.
fn luminance_to_bin(lum: f32) -> u32 {
    // Handle zero/negative luminance
    if (lum <= 0.0) {
        return 0u;
    }

    // Log-space mapping
    let log_lum = log2(lum + EPSILON);

    // Normalize to [0, 1] range using uniform parameters
    let range = uniforms.max_luminance - uniforms.min_luminance;
    let normalized = (log_lum - uniforms.min_luminance) / range;

    // Map to bin index [0, 255]
    let bin_f = clamp(normalized * 255.0, 0.0, 255.0);
    return u32(bin_f);
}

/// Convert 1D index to 2D texture coordinate.
fn index_to_coord(idx: u32) -> vec2<i32> {
    let x = idx % uniforms.src_dims.x;
    let y = idx / uniforms.src_dims.x;
    return vec2<i32>(i32(x), i32(y));
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------

/// Luminance histogram kernel.
///
/// Computes a 256-bin histogram using a two-phase approach:
/// 1. Accumulate into workgroup-local histogram (shared memory)
/// 2. Merge local histogram into global histogram (atomic operations)
@compute @workgroup_size(256, 1, 1)
fn compute_histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let local_idx = lid.x;

    // ── Phase 0: Initialize local histogram ────────────────────────────────

    // Each thread initializes one bin to zero
    atomicStore(&local_histogram[local_idx], 0u);
    workgroupBarrier();

    // ── Phase 1: Accumulate pixels into local histogram ────────────────────

    // Grid-stride loop: each thread processes multiple pixels
    let total_workgroups = num_wg.x;
    let threads_per_image = total_workgroups * WORKGROUP_SIZE;

    var pixel_idx = gid.x;
    while (pixel_idx < uniforms.num_pixels) {
        let coord = index_to_coord(pixel_idx);

        // Load pixel color
        let color = textureLoad(src_texture, coord, 0);

        // Compute luminance and bin index
        let lum = luminance(color.rgb);
        let bin = luminance_to_bin(lum);

        // Atomically increment local histogram bin
        atomicAdd(&local_histogram[bin], 1u);

        // Move to next pixel (grid-stride)
        pixel_idx += threads_per_image;
    }

    workgroupBarrier();

    // ── Phase 2: Merge local histogram into global ─────────────────────────

    // Each thread handles one bin
    let local_count = atomicLoad(&local_histogram[local_idx]);
    if (local_count > 0u) {
        atomicAdd(&histogram[local_idx], local_count);
    }
}

// ---------------------------------------------------------------------------
// Clear Entry Point
// ---------------------------------------------------------------------------

/// Clear the histogram buffer to zero.
/// Must be dispatched with 1 workgroup before computing histogram.
@compute @workgroup_size(256, 1, 1)
fn clear_histogram(@builtin(local_invocation_id) lid: vec3<u32>) {
    atomicStore(&histogram[lid.x], 0u);
}

// ---------------------------------------------------------------------------
// Notes on Implementation
// ---------------------------------------------------------------------------
//
// Atomic Operations:
//   - Local histogram uses atomicAdd for intra-workgroup synchronization
//   - Global histogram uses atomicAdd for inter-workgroup merging
//   - This two-level approach reduces contention vs. direct global atomics
//
// Memory Layout:
//   - Global histogram: array<atomic<u32>, 256> = 1024 bytes
//   - Local histogram: same size, but in shared memory (~10x faster)
//
// Grid-Stride Loop:
//   - Allows arbitrary image sizes without matching workgroup count
//   - Each thread processes pixels at indices: gid.x, gid.x + N, gid.x + 2N, ...
//   - N = total threads across all workgroups
//
// Log-Space Binning:
//   - HDR images have high dynamic range (0.001 to 100+)
//   - Linear binning would cluster most pixels in low bins
//   - Log-space spreads values more evenly across 256 bins
//
// Applications:
//   1. Auto-exposure: Find median luminance for exposure adjustment
//   2. Tone mapping: Histogram equalization for contrast enhancement
//   3. HDR analysis: Identify clipping, measure dynamic range
//   4. Debugging: Visualize luminance distribution
//
// Usage Example:
//   1. Dispatch clear_histogram with 1 workgroup
//   2. Dispatch compute_histogram with ceil(num_pixels / 256) workgroups
//   3. Read back histogram buffer for CPU processing
//      OR use in subsequent GPU passes (e.g., cumulative sum for percentiles)
