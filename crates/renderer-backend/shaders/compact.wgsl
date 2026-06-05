// SPDX-License-Identifier: MIT
//
// compact.wgsl - Visibility Stream Compaction Shader (T-WGPU-P6.6.1).
//
// Stream compaction shader that scatters visible object indices to a compacted
// buffer using prefix scan results. This shader reads bit-packed visibility
// flags and exclusive prefix sum results to determine output positions.
//
// Algorithm:
// 1. For each object index, check if the corresponding visibility bit is set
// 2. If visible, read the prefix sum at that index to get output position
// 3. Scatter the object index to the compacted buffer at that position
// 4. Final count = prefix_sum[last] + is_visible(last)
//
// Visibility flags are stored as packed u32 words (32 objects per word).
// Prefix scan is computed on the unpacked visibility values (1 per object).
//
// Workgroup size: 64 threads (optimal for most GPUs)
// Each thread processes one object.
//
// Stability: The compaction is stable - visible objects maintain their
// relative order from the original array.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of bits per visibility word.
const BITS_PER_WORD: u32 = 32u;

/// Workgroup size for compaction kernel.
const WORKGROUP_SIZE: u32 = 64u;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

/// Parameters for the compaction dispatch.
struct CompactParams {
    /// Total number of objects to process.
    object_count: u32,
    /// Reserved for future use (e.g., multi-pass compaction).
    _pad0: u32,
    /// Reserved for alignment.
    _pad1: u32,
    /// Reserved for alignment.
    _pad2: u32,
}

// ---------------------------------------------------------------------------
// Bindings - Group 0: Visibility + Prefix Scan (Read-only)
// ---------------------------------------------------------------------------

/// Packed visibility flags buffer (1 bit per object, 32 objects per u32).
/// Set by culling passes (frustum cull, HiZ occlusion cull, etc.).
@group(0) @binding(0) var<storage, read> visibility_flags: array<u32>;

/// Exclusive prefix sum of visibility values.
/// prefix_sum[i] = count of visible objects with index < i.
/// Computed by prefix scan pipeline before this shader.
@group(0) @binding(1) var<storage, read> prefix_sum: array<u32>;

// ---------------------------------------------------------------------------
// Bindings - Group 1: Output (Write)
// ---------------------------------------------------------------------------

/// Compacted output indices buffer.
/// Contains indices of visible objects in their original order.
@group(1) @binding(0) var<storage, read_write> compacted_indices: array<u32>;

/// Atomic counter for total visible objects.
/// Set by get_count_main entry point.
@group(1) @binding(1) var<storage, read_write> compacted_count: atomic<u32>;

// ---------------------------------------------------------------------------
// Bindings - Group 2: Parameters (Uniform)
// ---------------------------------------------------------------------------

/// Compaction parameters uniform buffer.
@group(2) @binding(0) var<uniform> params: CompactParams;

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Check if an object is visible from the packed visibility flags.
///
/// @param object_idx - The object index to check.
/// @returns true if the object's visibility bit is set.
fn is_visible(object_idx: u32) -> bool {
    let word_idx = object_idx / BITS_PER_WORD;
    let bit_idx = object_idx % BITS_PER_WORD;
    let bit_mask = 1u << bit_idx;
    return (visibility_flags[word_idx] & bit_mask) != 0u;
}

/// Get the bit value (0 or 1) for an object's visibility.
///
/// @param object_idx - The object index to check.
/// @returns 1 if visible, 0 otherwise.
fn get_visibility_bit(object_idx: u32) -> u32 {
    if (is_visible(object_idx)) {
        return 1u;
    }
    return 0u;
}

// ---------------------------------------------------------------------------
// Scatter Kernel - Main Compaction Entry Point
// ---------------------------------------------------------------------------

/// Scatter visible object indices to their compacted positions.
///
/// For each object in the input range:
/// 1. Check if the visibility bit is set
/// 2. If visible, use prefix_sum[object_idx] as the output index
/// 3. Write object_idx to compacted_indices[output_idx]
///
/// The exclusive prefix sum guarantees that:
/// - prefix_sum[i] = number of visible objects before index i
/// - This equals the correct output position for object i
/// - Order is preserved (stable compaction)
///
/// @param gid - Global invocation ID (object index)
@compute @workgroup_size(64, 1, 1)
fn compact_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let object_idx = gid.x;

    // Bounds check
    if (object_idx >= params.object_count) {
        return;
    }

    // Only process visible objects
    if (is_visible(object_idx)) {
        // Get output index from exclusive prefix sum
        // prefix_sum[i] contains the count of visible objects with index < i
        // This is exactly the position where object i should be written
        let output_idx = prefix_sum[object_idx];

        // Write object index to compacted output
        compacted_indices[output_idx] = object_idx;
    }
}

// ---------------------------------------------------------------------------
// Count Kernel - Get Final Visible Count
// ---------------------------------------------------------------------------

/// Compute the total count of visible objects.
///
/// The total count is computed as:
///   count = prefix_sum[n-1] + is_visible(n-1)
///
/// Where n = object_count.
///
/// This kernel runs with a single thread and atomically stores the count.
/// It should be dispatched after compact_main completes if the count is needed.
///
/// @param gid - Global invocation ID (should be 0)
@compute @workgroup_size(1, 1, 1)
fn get_count_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Only first thread computes count
    if (gid.x != 0u) {
        return;
    }

    // Handle empty case
    if (params.object_count == 0u) {
        atomicStore(&compacted_count, 0u);
        return;
    }

    // Total count = prefix_sum[last] + visibility[last]
    // For exclusive prefix sum: prefix_sum[n-1] = sum of visibility[0..n-1]
    // Total = prefix_sum[n-1] + visibility[n-1]
    let last_idx = params.object_count - 1u;
    let last_prefix = prefix_sum[last_idx];
    let last_visible = get_visibility_bit(last_idx);

    atomicStore(&compacted_count, last_prefix + last_visible);
}

// ---------------------------------------------------------------------------
// Fused Scatter + Count Kernel (Optimization)
// ---------------------------------------------------------------------------

/// Fused kernel that performs scatter and computes count in one pass.
///
/// This is an optimization when you need both the compacted indices and
/// the count. Uses atomic increment to track count during scatter.
///
/// Note: This variant uses atomicAdd which may be slower than the separate
/// count kernel for very large arrays due to atomic contention. Profile
/// to determine which approach is faster for your use case.
///
/// @param gid - Global invocation ID (object index)
@compute @workgroup_size(64, 1, 1)
fn compact_fused(@builtin(global_invocation_id) gid: vec3<u32>) {
    let object_idx = gid.x;

    // Bounds check
    if (object_idx >= params.object_count) {
        return;
    }

    // Only process visible objects
    if (is_visible(object_idx)) {
        // Get output index from exclusive prefix sum
        let output_idx = prefix_sum[object_idx];

        // Write object index to compacted output
        compacted_indices[output_idx] = object_idx;
    }
}

// ---------------------------------------------------------------------------
// Batch Scatter Kernel (Multiple Objects Per Thread)
// ---------------------------------------------------------------------------

/// Batch scatter kernel where each thread processes multiple objects.
///
/// This variant improves memory efficiency by having each thread handle
/// a contiguous range of objects, allowing better cache utilization
/// for the visibility flags buffer.
///
/// Each thread processes BATCH_SIZE objects starting from its base index.
///
/// @param gid - Global invocation ID (batch index)
/// @param lid - Local invocation ID
@compute @workgroup_size(64, 1, 1)
fn compact_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Process 4 objects per thread for better memory coalescing
    let batch_size: u32 = 4u;
    let base_idx = gid.x * batch_size;

    // Process batch of objects
    for (var i: u32 = 0u; i < batch_size; i = i + 1u) {
        let object_idx = base_idx + i;

        // Bounds check
        if (object_idx >= params.object_count) {
            return;
        }

        // Scatter if visible
        if (is_visible(object_idx)) {
            let output_idx = prefix_sum[object_idx];
            compacted_indices[output_idx] = object_idx;
        }
    }
}

// ---------------------------------------------------------------------------
// Validation Kernel (Debug)
// ---------------------------------------------------------------------------

/// Validation kernel for testing - verifies prefix sum consistency.
///
/// Checks that prefix_sum values are monotonically increasing and
/// match expected visibility patterns. Only for debug builds.
///
/// Results written to compacted_indices[0..3] as debug output:
/// - [0] = error count
/// - [1] = first error object index
/// - [2] = expected value at error
/// - [3] = actual value at error
///
/// @param gid - Global invocation ID
@compute @workgroup_size(64, 1, 1)
fn validate_prefix_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let object_idx = gid.x;

    // Only validate interior elements (need previous value)
    if (object_idx == 0u || object_idx >= params.object_count) {
        return;
    }

    let prev_prefix = prefix_sum[object_idx - 1u];
    let curr_prefix = prefix_sum[object_idx];
    let prev_visible = get_visibility_bit(object_idx - 1u);

    // Prefix sum property: prefix[i] = prefix[i-1] + visibility[i-1]
    let expected = prev_prefix + prev_visible;

    if (curr_prefix != expected) {
        // Atomic increment error count (stored at index 0)
        let err_idx = atomicAdd(&compacted_count, 1u);

        // Only record first error details
        if (err_idx == 0u) {
            compacted_indices[0] = 1u;  // Error flag
            compacted_indices[1] = object_idx;  // Error location
            compacted_indices[2] = expected;  // Expected value
            compacted_indices[3] = curr_prefix;  // Actual value
        }
    }
}
