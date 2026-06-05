// SPDX-License-Identifier: MIT
//
// gpu_sort.comp.wgsl - GPU Radix Sort (T-GPU-2.1).
//
// High-performance radix sort for 32-bit keys with optional 32-bit payloads.
// Based on the Onesweep algorithm with per-workgroup histogram spine.
//
// Algorithm Overview:
// - 4-bit radix (16 buckets) with 8 passes for 32-bit keys
// - Five-phase approach per pass:
//   0. Clear spine and global histogram
//   1. Build per-workgroup histogram in spine + global histogram
//   2. Compute global digit prefix sums
//   3. Compute per-workgroup offsets in spine
//   4. Scatter to final positions
//
// Use Cases:
// - Visibility buffer sorting by material/mesh for batching
// - Particle depth sorting (back-to-front transparency)
// - Indirect draw command sorting for GPU-driven rendering
//
// Performance Target: <0.5ms for 100K keys on modern GPUs.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 256u;
const RADIX_BITS: u32 = 4u;
const RADIX_SIZE: u32 = 16u;    // 2^4 = 16 buckets
const NUM_PASSES: u32 = 8u;      // 32 bits / 4 bits = 8 passes

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

/// Per-pass parameters for radix sort.
struct SortParams {
    /// Total number of elements to sort.
    num_elements: u32,
    /// Current pass index (0-7 for 4-bit radix).
    current_pass: u32,
    /// Number of workgroups.
    num_workgroups: u32,
    /// Padding for 16-byte alignment.
    padding: u32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> params: SortParams;
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(3) var<storage, read> values_in: array<u32>;
@group(0) @binding(4) var<storage, read_write> values_out: array<u32>;
// Spine buffer: array of size [num_workgroups * RADIX_SIZE]
// spine[wg * 16 + digit] stores per-workgroup histogram counts
// After spine_offsets: stores global offset for (wg, digit) pair
@group(0) @binding(5) var<storage, read_write> spine: array<atomic<u32>>;
// Global histogram: 16 entries for digit totals and prefix sums
@group(0) @binding(6) var<storage, read_write> global_histogram: array<atomic<u32>>;

// ---------------------------------------------------------------------------
// Workgroup Shared Memory
// ---------------------------------------------------------------------------

/// Local histogram for each radix digit (16 buckets).
var<workgroup> local_histogram: array<atomic<u32>, 16>;

/// Local prefix sums for scatter destinations.
var<workgroup> local_prefix: array<u32, 16>;

/// Scratch space for workgroup-local sorting.
var<workgroup> local_keys: array<u32, 256>;
var<workgroup> local_values: array<u32, 256>;

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Extract the radix digit from a key for the given pass.
fn extract_digit(key: u32, pass_idx: u32) -> u32 {
    return (key >> (pass_idx * RADIX_BITS)) & (RADIX_SIZE - 1u);
}

// ---------------------------------------------------------------------------
// Phase 0: Clear Buffers
// ---------------------------------------------------------------------------

/// Zero out the spine buffer and global histogram before each radix pass.
@compute @workgroup_size(256)
fn clear_buffers(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    // Clear spine: size = num_workgroups * RADIX_SIZE
    let spine_size = params.num_workgroups * RADIX_SIZE;
    if (gid.x < spine_size) {
        atomicStore(&spine[gid.x], 0u);
    }
    // Clear global histogram (only need 16 entries, done by first 16 threads)
    if (gid.x < RADIX_SIZE) {
        atomicStore(&global_histogram[gid.x], 0u);
    }
}

// ---------------------------------------------------------------------------
// Phase 1: Histogram Pass
// ---------------------------------------------------------------------------

/// Build per-workgroup histogram in spine AND global histogram totals.
@compute @workgroup_size(256)
fn histogram_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Initialize local histogram
    if (lid.x < RADIX_SIZE) {
        atomicStore(&local_histogram[lid.x], 0u);
    }
    workgroupBarrier();

    // Each thread processes one element
    let idx = gid.x;
    if (idx < params.num_elements) {
        let key = keys_in[idx];
        let digit = extract_digit(key, params.current_pass);
        atomicAdd(&local_histogram[digit], 1u);
    }
    workgroupBarrier();

    // Store to spine AND add to global histogram
    if (lid.x < RADIX_SIZE) {
        let local_count = atomicLoad(&local_histogram[lid.x]);
        // Store in spine
        let spine_idx = wid.x * RADIX_SIZE + lid.x;
        atomicStore(&spine[spine_idx], local_count);
        // Add to global histogram
        if (local_count > 0u) {
            atomicAdd(&global_histogram[lid.x], local_count);
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2: Global Prefix Sum
// ---------------------------------------------------------------------------

/// Compute exclusive prefix sum of global histogram.
/// After this, global_histogram[digit] = starting position for all elements with this digit.
@compute @workgroup_size(16)
fn global_prefix_sum(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    // Load counts into shared memory
    let count = atomicLoad(&global_histogram[lid.x]);
    local_prefix[lid.x] = count;
    workgroupBarrier();

    // Hillis-Steele inclusive scan
    // Step 1: offset 1
    var read_val: u32 = 0u;
    if (lid.x >= 1u) {
        read_val = local_prefix[lid.x - 1u];
    }
    workgroupBarrier();
    if (lid.x >= 1u) {
        local_prefix[lid.x] = local_prefix[lid.x] + read_val;
    }
    workgroupBarrier();

    // Step 2: offset 2
    read_val = 0u;
    if (lid.x >= 2u) {
        read_val = local_prefix[lid.x - 2u];
    }
    workgroupBarrier();
    if (lid.x >= 2u) {
        local_prefix[lid.x] = local_prefix[lid.x] + read_val;
    }
    workgroupBarrier();

    // Step 3: offset 4
    read_val = 0u;
    if (lid.x >= 4u) {
        read_val = local_prefix[lid.x - 4u];
    }
    workgroupBarrier();
    if (lid.x >= 4u) {
        local_prefix[lid.x] = local_prefix[lid.x] + read_val;
    }
    workgroupBarrier();

    // Step 4: offset 8
    read_val = 0u;
    if (lid.x >= 8u) {
        read_val = local_prefix[lid.x - 8u];
    }
    workgroupBarrier();
    if (lid.x >= 8u) {
        local_prefix[lid.x] = local_prefix[lid.x] + read_val;
    }
    workgroupBarrier();

    // Convert inclusive to exclusive prefix sum
    var prefix: u32;
    if (lid.x == 0u) {
        prefix = 0u;
    } else {
        prefix = local_prefix[lid.x - 1u];
    }

    // Store back as exclusive prefix sum
    atomicStore(&global_histogram[lid.x], prefix);
}

// ---------------------------------------------------------------------------
// Phase 3: Spine Offsets
// ---------------------------------------------------------------------------

/// Compute per-workgroup offsets in spine.
/// spine[wg * 16 + digit] = global_histogram[digit] + sum of counts for this digit in workgroups < wg
@compute @workgroup_size(16)
fn spine_offsets(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let digit = lid.x;
    if (digit >= RADIX_SIZE) {
        return;
    }

    // Start with global prefix for this digit
    var running_offset: u32 = atomicLoad(&global_histogram[digit]);

    // For each workgroup, compute its offset and update spine
    for (var wg: u32 = 0u; wg < params.num_workgroups; wg = wg + 1u) {
        let spine_idx = wg * RADIX_SIZE + digit;
        let count = atomicLoad(&spine[spine_idx]);
        atomicStore(&spine[spine_idx], running_offset);
        running_offset = running_offset + count;
    }
}

// ---------------------------------------------------------------------------
// Phase 4: Scatter Pass
// ---------------------------------------------------------------------------

/// Shared memory for local rank computation in scatter.
var<workgroup> local_digits: array<u32, 256>;

/// Scatter elements to their sorted positions.
@compute @workgroup_size(256)
fn scatter_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let idx = gid.x;

    // Load element and compute digit
    var key: u32 = 0xFFFFFFFFu;
    var value: u32 = 0u;
    var digit: u32 = 0u;
    var valid: bool = false;

    if (idx < params.num_elements) {
        key = keys_in[idx];
        value = values_in[idx];
        digit = extract_digit(key, params.current_pass);
        valid = true;
    }

    // Store digit in shared memory for local rank computation
    local_digits[lid.x] = digit;
    workgroupBarrier();

    // Load workgroup base offsets from spine
    if (lid.x < RADIX_SIZE) {
        let spine_idx = wid.x * RADIX_SIZE + lid.x;
        local_prefix[lid.x] = atomicLoad(&spine[spine_idx]);
    }
    workgroupBarrier();

    // Compute local rank within this workgroup (stable - count elements before this one)
    var local_rank: u32 = 0u;
    let workgroup_start = wid.x * WORKGROUP_SIZE;
    for (var i: u32 = 0u; i < lid.x; i = i + 1u) {
        let global_idx = workgroup_start + i;
        if (global_idx < params.num_elements && local_digits[i] == digit) {
            local_rank = local_rank + 1u;
        }
    }
    workgroupBarrier();

    // Final destination: workgroup base offset + local rank
    if (valid) {
        let base_offset = local_prefix[digit];
        let dest = base_offset + local_rank;
        keys_out[dest] = key;
        values_out[dest] = value;
    }
}

// ---------------------------------------------------------------------------
// Combined Pass (for small arrays) - Bitonic Sort
// ---------------------------------------------------------------------------

/// Single-pass sort for arrays that fit within a workgroup.
/// Uses shared memory bitonic sort for arrays <= 256 elements.
@compute @workgroup_size(256)
fn sort_small(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    // Load into shared memory
    var key: u32 = 0xFFFFFFFFu;  // Max value for padding
    var value: u32 = 0u;

    if (gid.x < params.num_elements) {
        key = keys_in[gid.x];
        value = values_in[gid.x];
    }

    local_keys[lid.x] = key;
    local_values[lid.x] = value;
    workgroupBarrier();

    // Bitonic sort - 8 stages for 256 elements
    for (var k = 2u; k <= 256u; k = k * 2u) {
        for (var j = k / 2u; j > 0u; j = j / 2u) {
            // Read both elements for comparison
            let my_key = local_keys[lid.x];
            let my_val = local_values[lid.x];
            let ixj = lid.x ^ j;
            let partner_key = local_keys[ixj];
            let partner_val = local_values[ixj];

            // Determine sort direction: ascending in first half of each k-block
            let ascending = ((lid.x & k) == 0u);

            var new_key: u32;
            var new_val: u32;

            if (lid.x < ixj) {
                let take_partner = (ascending && my_key > partner_key) ||
                                   (!ascending && my_key < partner_key);
                if (take_partner) {
                    new_key = partner_key;
                    new_val = partner_val;
                } else {
                    new_key = my_key;
                    new_val = my_val;
                }
            } else {
                let take_partner = (ascending && partner_key > my_key) ||
                                   (!ascending && partner_key < my_key);
                if (take_partner) {
                    new_key = partner_key;
                    new_val = partner_val;
                } else {
                    new_key = my_key;
                    new_val = my_val;
                }
            }

            workgroupBarrier();
            local_keys[lid.x] = new_key;
            local_values[lid.x] = new_val;
            workgroupBarrier();
        }
    }

    // Write back sorted results
    if (gid.x < params.num_elements) {
        keys_out[gid.x] = local_keys[lid.x];
        values_out[gid.x] = local_values[lid.x];
    }
}
