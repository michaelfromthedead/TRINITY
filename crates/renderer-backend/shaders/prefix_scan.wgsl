// SPDX-License-Identifier: MIT
//
// prefix_scan.wgsl - Blelloch parallel prefix scan (exclusive scan) shader (T-WGPU-P3.10.2).
//
// Implements the Blelloch parallel exclusive prefix sum algorithm for wgpu 25.x.
// The algorithm has two phases:
//
// 1. UP-SWEEP (Reduce): Build partial sums tree from leaves to root
//    - Each level d (0 to log2(n)-1), stride = 2^(d+1)
//    - For k = 0 to n-1 by stride: data[k + stride - 1] += data[k + stride/2 - 1]
//
// 2. DOWN-SWEEP (Distribute): Distribute sums from root to leaves
//    - Set last element to 0 (identity for exclusive scan)
//    - Each level d (log2(n)-1 to 0), stride = 2^(d+1)
//    - For k = 0 to n-1 by stride:
//        temp = data[k + stride/2 - 1]
//        data[k + stride/2 - 1] = data[k + stride - 1]
//        data[k + stride - 1] += temp
//
// For arrays larger than one workgroup can handle, we use multi-block scan:
// - Each workgroup scans its block and outputs block sum
// - Recursively scan block sums
// - Add block prefix to all elements in each block
//
// Workgroup size: 256 threads
// Each workgroup processes 512 elements (2 per thread).
//
// References:
// - Blelloch, G. (1990). "Prefix Sums and Their Applications"
// - Harris, M. et al. (2007). "Parallel Prefix Sum (Scan) with CUDA"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

override WORKGROUP_SIZE: u32 = 256u;

// Double the workgroup size for shared memory (each thread handles 2 elements)
const SHARED_SIZE: u32 = 512u;

// Number of levels in the tree = log2(SHARED_SIZE) = 9 for 512 elements
const LOG2_SHARED_SIZE: u32 = 9u;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct ScanParams {
    input_size: u32,     // Total number of elements to scan
    block_offset: u32,   // Starting offset for this dispatch (for multi-pass)
    is_inclusive: u32,   // 0 = exclusive, 1 = inclusive (for final add pass)
    _pad: u32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

// Primary data buffer - both input and output for in-place scan
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

// Block sums output - one sum per workgroup for multi-block coordination
@group(0) @binding(1) var<storage, read_write> block_sums: array<u32>;

// Scan parameters
@group(0) @binding(2) var<uniform> params: ScanParams;

// ---------------------------------------------------------------------------
// Shared Memory
// ---------------------------------------------------------------------------

// Shared memory for workgroup-local prefix scan
// Size = 2 * WORKGROUP_SIZE = 512 elements
var<workgroup> temp: array<u32, 512>;

// ---------------------------------------------------------------------------
// Up-Sweep Phase (Build Reduction Tree)
// ---------------------------------------------------------------------------

/// Up-sweep phase of Blelloch scan.
///
/// Builds the reduction tree by computing partial sums from leaves to root.
/// After up-sweep, temp[n-1] contains the total sum of all elements.
///
/// Each workgroup:
/// 1. Loads 2 elements per thread from global memory
/// 2. Performs log2(n) up-sweep steps with workgroupBarrier synchronization
/// 3. Stores the block sum (total) to block_sums array
///
/// @param gid - Global invocation ID
/// @param lid - Local invocation ID within workgroup
/// @param wid - Workgroup ID
@compute @workgroup_size(256, 1, 1)
fn up_sweep(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_idx = lid.x;
    let workgroup_idx = wid.x;

    // Calculate global indices for this thread's two elements
    let global_base = (params.block_offset + workgroup_idx * SHARED_SIZE);
    let idx0 = global_base + local_idx;
    let idx1 = global_base + local_idx + WORKGROUP_SIZE;

    // Load 2 elements per thread from global memory (with bounds checking)
    if (idx0 < params.input_size) {
        temp[local_idx] = data[idx0];
    } else {
        temp[local_idx] = 0u;
    }

    if (idx1 < params.input_size) {
        temp[local_idx + WORKGROUP_SIZE] = data[idx1];
    } else {
        temp[local_idx + WORKGROUP_SIZE] = 0u;
    }

    workgroupBarrier();

    // Up-sweep: build partial sums tree
    // d = 0: stride = 2, pairs (0,1), (2,3), ...
    // d = 1: stride = 4, pairs (1,3), (5,7), ...
    // ...
    // d = 8: stride = 512, single pair (255, 511)
    var offset: u32 = 1u;

    for (var d: u32 = LOG2_SHARED_SIZE - 1u; d > 0u; d = d - 1u) {
        workgroupBarrier();

        if (local_idx < (1u << d)) {
            let ai = offset * (2u * local_idx + 1u) - 1u;
            let bi = offset * (2u * local_idx + 2u) - 1u;
            temp[bi] = temp[bi] + temp[ai];
        }

        offset = offset << 1u;
    }

    // Final up-sweep step (d = 0)
    workgroupBarrier();
    if (local_idx == 0u) {
        let ai = offset - 1u;
        let bi = offset * 2u - 1u;
        temp[bi] = temp[bi] + temp[ai];
    }

    workgroupBarrier();

    // Store block sum (last element contains total sum)
    if (local_idx == 0u) {
        block_sums[workgroup_idx] = temp[SHARED_SIZE - 1u];
    }
}

// ---------------------------------------------------------------------------
// Down-Sweep Phase (Distribute Prefix Sums)
// ---------------------------------------------------------------------------

/// Down-sweep phase of Blelloch scan.
///
/// Distributes prefix sums from root to leaves, converting the reduction tree
/// into an exclusive prefix sum.
///
/// Assumes up_sweep has already been called and block_sums contains the
/// partial sums from each workgroup.
///
/// @param gid - Global invocation ID
/// @param lid - Local invocation ID within workgroup
/// @param wid - Workgroup ID
@compute @workgroup_size(256, 1, 1)
fn down_sweep(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_idx = lid.x;
    let workgroup_idx = wid.x;

    // Calculate global indices for this thread's two elements
    let global_base = (params.block_offset + workgroup_idx * SHARED_SIZE);
    let idx0 = global_base + local_idx;
    let idx1 = global_base + local_idx + WORKGROUP_SIZE;

    // Load 2 elements per thread from global memory (with bounds checking)
    if (idx0 < params.input_size) {
        temp[local_idx] = data[idx0];
    } else {
        temp[local_idx] = 0u;
    }

    if (idx1 < params.input_size) {
        temp[local_idx + WORKGROUP_SIZE] = data[idx1];
    } else {
        temp[local_idx + WORKGROUP_SIZE] = 0u;
    }

    workgroupBarrier();

    // Up-sweep: build partial sums tree (same as up_sweep kernel)
    var offset: u32 = 1u;

    for (var d: u32 = LOG2_SHARED_SIZE - 1u; d > 0u; d = d - 1u) {
        workgroupBarrier();

        if (local_idx < (1u << d)) {
            let ai = offset * (2u * local_idx + 1u) - 1u;
            let bi = offset * (2u * local_idx + 2u) - 1u;
            temp[bi] = temp[bi] + temp[ai];
        }

        offset = offset << 1u;
    }

    // Final up-sweep step
    workgroupBarrier();
    if (local_idx == 0u) {
        let ai = offset - 1u;
        let bi = offset * 2u - 1u;
        temp[bi] = temp[bi] + temp[ai];
    }

    workgroupBarrier();

    // Clear last element for exclusive scan
    if (local_idx == 0u) {
        temp[SHARED_SIZE - 1u] = 0u;
    }

    // Down-sweep: distribute prefix sums
    // Reverse the up-sweep process
    for (var d: u32 = 0u; d < LOG2_SHARED_SIZE; d = d + 1u) {
        offset = offset >> 1u;
        workgroupBarrier();

        if (local_idx < (1u << d)) {
            let ai = offset * (2u * local_idx + 1u) - 1u;
            let bi = offset * (2u * local_idx + 2u) - 1u;

            let t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = temp[bi] + t;
        }
    }

    workgroupBarrier();

    // Write results back to global memory
    if (idx0 < params.input_size) {
        data[idx0] = temp[local_idx];
    }
    if (idx1 < params.input_size) {
        data[idx1] = temp[local_idx + WORKGROUP_SIZE];
    }
}

// ---------------------------------------------------------------------------
// Add Block Sums (Multi-Block Coordination)
// ---------------------------------------------------------------------------

/// Add block prefix sums to all elements in each block.
///
/// After scanning block sums, each element needs to have its block's prefix
/// added to complete the global scan.
///
/// block_sums[i] = exclusive prefix sum of all blocks before block i
///
/// @param gid - Global invocation ID
/// @param lid - Local invocation ID within workgroup
/// @param wid - Workgroup ID
@compute @workgroup_size(256, 1, 1)
fn add_block_sums(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_idx = lid.x;
    let workgroup_idx = wid.x;

    // Skip first block (no prefix to add)
    if (workgroup_idx == 0u) {
        return;
    }

    // Load block prefix sum into shared memory for efficiency
    // All threads in workgroup share the same prefix
    var block_prefix: u32 = 0u;
    if (local_idx == 0u) {
        block_prefix = block_sums[workgroup_idx];
    }

    // Broadcast block prefix to all threads
    temp[0] = block_prefix;
    workgroupBarrier();
    block_prefix = temp[0];

    // Calculate global indices for this thread's two elements
    let global_base = (params.block_offset + workgroup_idx * SHARED_SIZE);
    let idx0 = global_base + local_idx;
    let idx1 = global_base + local_idx + WORKGROUP_SIZE;

    // Add block prefix to both elements
    if (idx0 < params.input_size) {
        data[idx0] = data[idx0] + block_prefix;
    }
    if (idx1 < params.input_size) {
        data[idx1] = data[idx1] + block_prefix;
    }
}

// ---------------------------------------------------------------------------
// Single-Pass Scan (for small arrays that fit in one workgroup)
// ---------------------------------------------------------------------------

/// Single-pass scan for arrays <= 512 elements.
///
/// Performs complete exclusive prefix scan in one kernel invocation.
/// More efficient than multi-pass for small arrays.
///
/// @param gid - Global invocation ID
/// @param lid - Local invocation ID within workgroup
/// @param wid - Workgroup ID
@compute @workgroup_size(256, 1, 1)
fn scan_single_block(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_idx = lid.x;

    // Load 2 elements per thread
    let idx0 = local_idx;
    let idx1 = local_idx + WORKGROUP_SIZE;

    if (idx0 < params.input_size) {
        temp[local_idx] = data[idx0];
    } else {
        temp[local_idx] = 0u;
    }

    if (idx1 < params.input_size) {
        temp[local_idx + WORKGROUP_SIZE] = data[idx1];
    } else {
        temp[local_idx + WORKGROUP_SIZE] = 0u;
    }

    workgroupBarrier();

    // Up-sweep
    var offset: u32 = 1u;

    for (var d: u32 = LOG2_SHARED_SIZE - 1u; d > 0u; d = d - 1u) {
        workgroupBarrier();

        if (local_idx < (1u << d)) {
            let ai = offset * (2u * local_idx + 1u) - 1u;
            let bi = offset * (2u * local_idx + 2u) - 1u;
            temp[bi] = temp[bi] + temp[ai];
        }

        offset = offset << 1u;
    }

    // Final up-sweep step
    workgroupBarrier();
    if (local_idx == 0u) {
        let ai = offset - 1u;
        let bi = offset * 2u - 1u;
        temp[bi] = temp[bi] + temp[ai];

        // Store total sum in block_sums[0] for reference
        block_sums[0] = temp[SHARED_SIZE - 1u];

        // Clear for exclusive scan
        temp[SHARED_SIZE - 1u] = 0u;
    }

    workgroupBarrier();

    // Down-sweep
    for (var d: u32 = 0u; d < LOG2_SHARED_SIZE; d = d + 1u) {
        offset = offset >> 1u;
        workgroupBarrier();

        if (local_idx < (1u << d)) {
            let ai = offset * (2u * local_idx + 1u) - 1u;
            let bi = offset * (2u * local_idx + 2u) - 1u;

            let t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = temp[bi] + t;
        }
    }

    workgroupBarrier();

    // Write results
    if (idx0 < params.input_size) {
        data[idx0] = temp[local_idx];
    }
    if (idx1 < params.input_size) {
        data[idx1] = temp[local_idx + WORKGROUP_SIZE];
    }
}
