// SPDX-License-Identifier: MIT
//
// stream_compact.wgsl - Stream compaction using prefix scan (T-WGPU-P3.10.3).
//
// Stream compaction is a fundamental parallel primitive that filters an array
// based on predicates, producing a densely-packed output array containing only
// the elements that pass the predicate test.
//
// Algorithm:
// 1. Evaluate predicate for each element (produces 0/1 array)
// 2. Exclusive prefix scan of predicates gives output indices
// 3. Scatter: if predicate[i] == 1, write input[i] to output[scan_result[i]]
// 4. Total count = last_predicate + last_scan_result
//
// This shader implements step 3 (scatter). The prefix scan is performed by
// the PrefixScanPipeline from prefix_scan.rs before this shader is dispatched.
//
// Workgroup size: 256 threads
// Each thread handles one element.
//
// Use cases:
// - GPU-driven rendering: culling invisible objects
// - Particle systems: filtering dead particles
// - Physics simulation: active constraint filtering
// - Sparse matrix operations: non-zero element extraction
// - Ray tracing: hit/miss stream filtering

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

override WORKGROUP_SIZE: u32 = 256u;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct CompactParams {
    /// Total number of input elements.
    input_size: u32,
    /// Offset for multi-pass compaction (for very large arrays).
    block_offset: u32,
    /// Stride for input data (1 = contiguous, >1 = strided access).
    input_stride: u32,
    /// Stride for output data (1 = contiguous, >1 = strided access).
    output_stride: u32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

// Input data array (read-only)
@group(0) @binding(0) var<storage, read> input_data: array<u32>;

// Exclusive prefix scan result of predicates
// scan_result[i] = number of 1s in predicates[0..i)
@group(0) @binding(1) var<storage, read> scan_result: array<u32>;

// Predicate array (0 = exclude, non-zero = include)
@group(0) @binding(2) var<storage, read> predicates: array<u32>;

// Output compacted array (write-only)
@group(0) @binding(3) var<storage, read_write> output_data: array<u32>;

// Compaction parameters
@group(0) @binding(4) var<uniform> params: CompactParams;

// ---------------------------------------------------------------------------
// Count Binding (separate bind group for atomic count output)
// ---------------------------------------------------------------------------

// For count_elements kernel: atomic output count
@group(1) @binding(0) var<storage, read_write> output_count: atomic<u32>;

// ---------------------------------------------------------------------------
// Scatter Kernel
// ---------------------------------------------------------------------------

/// Scatter elements to their compacted positions.
///
/// Each thread:
/// 1. Checks its predicate value
/// 2. If predicate != 0, writes input[gid] to output[scan_result[gid]]
///
/// The prefix scan guarantees that scan_result[i] gives the correct
/// output index for element i (exclusive scan means index = count of
/// preceding elements that passed the predicate).
///
/// @param gid - Global invocation ID (element index)
@compute @workgroup_size(256, 1, 1)
fn scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x + params.block_offset;

    // Bounds check
    if (idx >= params.input_size) {
        return;
    }

    // Check predicate
    let predicate = predicates[idx];

    if (predicate != 0u) {
        // Read input element (with stride support)
        let input_idx = idx * params.input_stride;
        let value = input_data[input_idx];

        // Write to compacted position (with stride support)
        let output_idx = scan_result[idx] * params.output_stride;
        output_data[output_idx] = value;
    }
}

// ---------------------------------------------------------------------------
// Scatter Vec4 Kernel (for 4-component data)
// ---------------------------------------------------------------------------

/// Scatter 4-component elements to their compacted positions.
///
/// Optimized for common case of vec4<u32> data (e.g., RGBA, XYZW).
/// Each element is 4 u32s, allowing coalesced memory access.
///
/// @param gid - Global invocation ID (element index, not component index)
@compute @workgroup_size(256, 1, 1)
fn scatter_vec4(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x + params.block_offset;

    // Bounds check
    if (idx >= params.input_size) {
        return;
    }

    // Check predicate
    let predicate = predicates[idx];

    if (predicate != 0u) {
        // Read 4 components
        let input_base = idx * 4u;
        let v0 = input_data[input_base];
        let v1 = input_data[input_base + 1u];
        let v2 = input_data[input_base + 2u];
        let v3 = input_data[input_base + 3u];

        // Write 4 components to compacted position
        let output_base = scan_result[idx] * 4u;
        output_data[output_base] = v0;
        output_data[output_base + 1u] = v1;
        output_data[output_base + 2u] = v2;
        output_data[output_base + 3u] = v3;
    }
}

// ---------------------------------------------------------------------------
// Count Elements Kernel
// ---------------------------------------------------------------------------

/// Count the total number of elements passing the predicate.
///
/// This kernel computes: count = predicates[n-1] + scan_result[n-1]
/// where n = input_size.
///
/// Only the first thread does the work; others exit early.
/// This is launched with a single workgroup of 1 thread.
///
/// Alternative: Use a separate readback of the last elements.
/// This kernel approach avoids an extra buffer copy.
@compute @workgroup_size(1, 1, 1)
fn count_elements(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // Only first thread computes count
    if (gid.x != 0u) {
        return;
    }

    if (params.input_size == 0u) {
        atomicStore(&output_count, 0u);
        return;
    }

    // Count = last predicate + last scan result
    // For exclusive scan: scan_result[n-1] = sum of predicates[0..n-1]
    // Total = scan_result[n-1] + predicates[n-1]
    let last_idx = params.input_size - 1u;
    let last_predicate = predicates[last_idx];
    let last_scan = scan_result[last_idx];

    // Predicate is 0 or 1 (or non-zero treated as 1)
    var pred_val: u32 = 0u;
    if (last_predicate != 0u) {
        pred_val = 1u;
    }

    atomicStore(&output_count, last_scan + pred_val);
}

// ---------------------------------------------------------------------------
// Evaluate Predicate Kernels
// ---------------------------------------------------------------------------

/// Generate predicates by comparing input values against a threshold.
///
/// Predicate[i] = 1 if input[i] > threshold, else 0.
/// Useful for filtering by value (e.g., keeping particles above certain energy).
///
/// Parameters:
/// - params.input_stride encodes the threshold (reusing field)
///
/// Note: In production, you'd want separate uniform buffers for different
/// predicate types. This is a simplified example.
@compute @workgroup_size(256, 1, 1)
fn evaluate_predicate_greater_than(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x + params.block_offset;

    if (idx >= params.input_size) {
        return;
    }

    // Threshold is stored in input_stride field (for simplicity)
    let threshold = params.input_stride;
    let value = input_data[idx];

    // Write predicate (reusing output_data as predicate buffer temporarily)
    // In practice, predicates buffer would be separate
    if (value > threshold) {
        output_data[idx] = 1u;
    } else {
        output_data[idx] = 0u;
    }
}

/// Generate predicates for non-zero values.
///
/// Predicate[i] = 1 if input[i] != 0, else 0.
/// Common use case: filtering out zero/null entries.
@compute @workgroup_size(256, 1, 1)
fn evaluate_predicate_nonzero(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x + params.block_offset;

    if (idx >= params.input_size) {
        return;
    }

    let value = input_data[idx];

    // Write predicate
    if (value != 0u) {
        output_data[idx] = 1u;
    } else {
        output_data[idx] = 0u;
    }
}

// ---------------------------------------------------------------------------
// Fused Scatter with Predicate Generation (Optimization)
// ---------------------------------------------------------------------------

/// Fused kernel: evaluate predicate and scatter in one pass.
///
/// This is an optimization for cases where predicates are simple and
/// can be evaluated on-the-fly without storing intermediate predicate array.
///
/// Requires the input to be scanned BEFORE this kernel runs, meaning
/// you still need a separate predicate buffer for the scan. But after
/// scan completes, this kernel can do both predicate check and scatter
/// without re-reading predicates.
///
/// Note: This is mainly useful when the predicate buffer is the bottleneck.
/// In most cases, the separate scatter kernel is clearer.
@compute @workgroup_size(256, 1, 1)
fn scatter_fused_nonzero(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x + params.block_offset;

    if (idx >= params.input_size) {
        return;
    }

    // Re-evaluate predicate (non-zero check)
    let value = input_data[idx];

    if (value != 0u) {
        // Write to compacted position
        let output_idx = scan_result[idx];
        output_data[output_idx] = value;
    }
}

// ---------------------------------------------------------------------------
// Multi-Element Scatter (for structs larger than u32)
// ---------------------------------------------------------------------------

/// Scatter elements with arbitrary element size.
///
/// Each "element" consists of `params.input_stride` u32s.
/// This supports compaction of structs/records larger than a single u32.
///
/// Memory layout:
/// - input_data[i * stride + 0..stride] = element i components
/// - Same for output_data
///
/// @param gid - Global invocation ID (element index)
@compute @workgroup_size(256, 1, 1)
fn scatter_multi_element(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x + params.block_offset;

    if (idx >= params.input_size) {
        return;
    }

    let predicate = predicates[idx];

    if (predicate != 0u) {
        let element_size = params.input_stride;
        let input_base = idx * element_size;
        let output_base = scan_result[idx] * element_size;

        // Copy all components of the element
        for (var i: u32 = 0u; i < element_size; i = i + 1u) {
            output_data[output_base + i] = input_data[input_base + i];
        }
    }
}
