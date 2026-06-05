// SPDX-License-Identifier: MIT
//
// prefix_sum.wgsl - Parallel Prefix Sum (Exclusive Scan) using Blelloch algorithm
//
// This module provides workgroup-level prefix sum operations for use in GPU-driven
// rendering tasks such as particle compaction and visibility buffer compaction.
//
// Algorithm: Blelloch Parallel Scan
// - Work complexity: O(n)
// - Step complexity: O(log n)
// - Uses shared memory for workgroup-local operations
//
// Reference: Guy E. Blelloch, "Prefix Sums and Their Applications", 1990

// ============================================================================
// Constants
// ============================================================================

/// Workgroup size for prefix sum operations.
/// Must be a power of two for the Blelloch algorithm.
const WORKGROUP_SIZE: u32 = 256u;

/// Shared memory size (2x workgroup size to avoid bank conflicts).
/// Bank conflict avoidance is achieved by padding stride access patterns.
const SHARED_SIZE: u32 = 512u;

// ============================================================================
// Shared Memory
// ============================================================================

/// Workgroup-local shared memory for scan operations.
/// Sized at 2x workgroup size to support bank conflict avoidance through
/// stride padding when accessing memory.
var<workgroup> shared_data: array<u32, 512>;

// ============================================================================
// Bank Conflict Avoidance
// ============================================================================

/// Compute index with bank conflict avoidance offset.
/// For 32-bank architectures, we add offset for every 32 consecutive elements.
/// This prevents multiple threads from accessing the same memory bank.
fn conflict_free_index(idx: u32) -> u32 {
    // Add 1 padding element per 32 elements to avoid bank conflicts
    return idx + (idx >> 5u);
}

// ============================================================================
// Workgroup-Level Exclusive Prefix Sum
// ============================================================================

/// Performs an exclusive prefix sum (scan) within a workgroup using the
/// Blelloch algorithm.
///
/// # Arguments
/// * `lid` - Local invocation ID (thread index within workgroup, 0..WORKGROUP_SIZE-1)
/// * `value` - The value to contribute to the scan
///
/// # Returns
/// The exclusive prefix sum at this thread's position. For thread i, this is
/// the sum of all values at threads 0..(i-1). Thread 0 always returns 0.
///
/// # Algorithm
/// The Blelloch scan consists of two phases:
/// 1. Up-sweep (reduce): Build a balanced binary tree of partial sums
/// 2. Down-sweep: Traverse the tree to compute final prefix sums
///
/// # Example
/// Input:  [3, 1, 7, 0, 4, 1, 6, 3]
/// Output: [0, 3, 4, 11, 11, 15, 16, 22]
fn workgroup_prefix_sum(lid: u32, value: u32) -> u32 {
    // Store input value in shared memory with conflict-free indexing
    let ai = conflict_free_index(lid);
    shared_data[ai] = value;
    workgroupBarrier();

    // ========================================================================
    // Up-sweep (Reduce) Phase
    // ========================================================================
    // Build a balanced binary tree of partial sums.
    // At each level d, thread i adds element (2*i+1)*offset-1 to (2*i+2)*offset-1

    var offset: u32 = 1u;
    for (var d: u32 = WORKGROUP_SIZE >> 1u; d > 0u; d >>= 1u) {
        workgroupBarrier();

        if (lid < d) {
            let ai_idx = offset * (2u * lid + 1u) - 1u;
            let bi_idx = offset * (2u * lid + 2u) - 1u;
            let ai_cf = conflict_free_index(ai_idx);
            let bi_cf = conflict_free_index(bi_idx);
            shared_data[bi_cf] = shared_data[bi_cf] + shared_data[ai_cf];
        }

        offset <<= 1u;
    }

    // ========================================================================
    // Clear last element for exclusive scan
    // ========================================================================
    // The last element now contains the total sum. For an exclusive scan,
    // we set it to 0 (identity element for addition).

    if (lid == 0u) {
        let last_idx = conflict_free_index(WORKGROUP_SIZE - 1u);
        shared_data[last_idx] = 0u;
    }
    workgroupBarrier();

    // ========================================================================
    // Down-sweep Phase
    // ========================================================================
    // Traverse the tree from root to leaves, propagating prefix sums.
    // At each level, we swap and add to compute the final exclusive scan.

    for (var d: u32 = 1u; d < WORKGROUP_SIZE; d <<= 1u) {
        offset >>= 1u;
        workgroupBarrier();

        if (lid < d) {
            let ai_idx = offset * (2u * lid + 1u) - 1u;
            let bi_idx = offset * (2u * lid + 2u) - 1u;
            let ai_cf = conflict_free_index(ai_idx);
            let bi_cf = conflict_free_index(bi_idx);

            let tmp = shared_data[ai_cf];
            shared_data[ai_cf] = shared_data[bi_cf];
            shared_data[bi_cf] = shared_data[bi_cf] + tmp;
        }
    }
    workgroupBarrier();

    // Return the exclusive prefix sum for this thread
    return shared_data[ai];
}

// ============================================================================
// Workgroup-Level Inclusive Prefix Sum
// ============================================================================

/// Performs an inclusive prefix sum within a workgroup.
///
/// # Arguments
/// * `lid` - Local invocation ID (thread index within workgroup)
/// * `value` - The value to contribute to the scan
///
/// # Returns
/// The inclusive prefix sum at this thread's position. For thread i, this is
/// the sum of all values at threads 0..i (inclusive).
///
/// # Example
/// Input:  [3, 1, 7, 0, 4, 1, 6, 3]
/// Output: [3, 4, 11, 11, 15, 16, 22, 25]
fn workgroup_prefix_sum_inclusive(lid: u32, value: u32) -> u32 {
    let exclusive = workgroup_prefix_sum(lid, value);
    return exclusive + value;
}

// ============================================================================
// Workgroup Total Sum
// ============================================================================

/// Returns the total sum of all values in the workgroup.
/// Must be called after workgroup_prefix_sum to get valid results.
///
/// # Arguments
/// * `lid` - Local invocation ID
/// * `value` - The original value contributed by this thread
///
/// # Returns
/// The sum of all values across the workgroup.
fn workgroup_total_sum(lid: u32, value: u32) -> u32 {
    // The last thread's inclusive sum equals the total
    let exclusive = workgroup_prefix_sum(lid, value);

    // Broadcast from last thread
    var total: u32;
    if (lid == WORKGROUP_SIZE - 1u) {
        shared_data[0] = exclusive + value;
    }
    workgroupBarrier();

    return shared_data[0];
}
