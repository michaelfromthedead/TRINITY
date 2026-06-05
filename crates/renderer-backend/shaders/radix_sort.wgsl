// SPDX-License-Identifier: MIT
//
// radix_sort.wgsl - GPU Radix Sort Compute Shader (T-WGPU-P3.10.4).
//
// Implements a parallel radix sort algorithm using histogram + prefix scan + scatter
// for sorting 32-bit key-value pairs on the GPU.
//
// Algorithm Overview:
//   Radix sort processes keys 4 bits at a time (one digit), requiring 8 passes
//   for 32-bit integers. Each pass consists of:
//
//   1. HISTOGRAM: Count occurrences of each digit (0-15) per workgroup
//   2. PREFIX SUM: Exclusive scan of histogram for scatter offsets
//   3. SCATTER: Write keys/values to sorted positions based on offsets
//
// Key properties:
//   - Stable sort: equal keys maintain relative order
//   - LSB-first: process least significant digit first (required for stability)
//   - 16 buckets per pass (4 bits = 2^4 = 16 values)
//   - 8 passes total for 32-bit keys (32 / 4 = 8)
//
// Performance notes:
//   - Workgroup-local histograms reduce atomic contention
//   - Coalesced memory access patterns for both read and write
//   - Uses prefix scan infrastructure for computing global offsets
//
// References:
//   - Merrill & Grimshaw (2011). "High Performance and Scalable Radix Sorting"
//   - Blelloch (1990). "Prefix Sums and Their Applications"
//
// Workgroup size: 256 threads.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 256u;
const RADIX_BITS: u32 = 4u;           // 4 bits per pass
const RADIX_BUCKETS: u32 = 16u;       // 2^4 = 16 buckets
const RADIX_MASK: u32 = 0xFu;         // Mask for lowest 4 bits
const TOTAL_PASSES: u32 = 8u;         // 32 bits / 4 bits = 8 passes

// Elements processed per thread (for efficiency)
const ELEMENTS_PER_THREAD: u32 = 4u;
const ELEMENTS_PER_WORKGROUP: u32 = 1024u; // WORKGROUP_SIZE * ELEMENTS_PER_THREAD

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

struct RadixSortParams {
    input_size: u32,       // Total number of key-value pairs
    pass_number: u32,      // Current radix pass (0-7)
    num_workgroups: u32,   // Total number of workgroups dispatched
    _pad: u32,
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

// Input keys buffer (read-only for histogram/scatter-read)
@group(0) @binding(0) var<storage, read> keys_in: array<u32>;

// Output keys buffer (write-only for scatter-write)
@group(0) @binding(1) var<storage, read_write> keys_out: array<u32>;

// Input values buffer (read-only)
@group(0) @binding(2) var<storage, read> values_in: array<u32>;

// Output values buffer (write-only)
@group(0) @binding(3) var<storage, read_write> values_out: array<u32>;

// Global histogram (atomic for accumulation across workgroups)
// Layout: [bucket0_wg0, bucket0_wg1, ..., bucket0_wgN, bucket1_wg0, ...]
// Total size: RADIX_BUCKETS * num_workgroups
@group(0) @binding(4) var<storage, read_write> global_histogram: array<atomic<u32>>;

// Prefix sums of global histogram (computed by separate prefix scan pass)
// Same layout as global_histogram, but after exclusive scan
@group(0) @binding(5) var<storage, read> global_offsets: array<u32>;

// Parameters uniform
@group(0) @binding(6) var<uniform> params: RadixSortParams;

// ---------------------------------------------------------------------------
// Shared Memory
// ---------------------------------------------------------------------------

// Workgroup-local histogram (16 buckets)
var<workgroup> local_histogram: array<atomic<u32>, 16>;

// Shared storage for scatter (stores digit of each element in workgroup)
var<workgroup> shared_digits: array<u32, 1024>;

// Local prefix sums for scatter within workgroup
var<workgroup> local_prefix: array<u32, 16>;

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Extract the 4-bit digit from a key for the current pass.
fn extract_digit(key: u32, pass: u32) -> u32 {
    let shift = pass * RADIX_BITS;
    return (key >> shift) & RADIX_MASK;
}

// ---------------------------------------------------------------------------
// Pass 1: Histogram
// ---------------------------------------------------------------------------

/// Compute per-workgroup histogram of digit frequencies.
///
/// Each workgroup counts how many keys have each digit value (0-15)
/// for the current radix pass. Results are stored in global_histogram
/// in a layout suitable for subsequent prefix scan.
///
/// Output layout: global_histogram[bucket * num_workgroups + workgroup_id]
@compute @workgroup_size(256, 1, 1)
fn histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_idx = lid.x;
    let workgroup_idx = wid.x;

    // Initialize local histogram to zero
    if (local_idx < RADIX_BUCKETS) {
        atomicStore(&local_histogram[local_idx], 0u);
    }
    workgroupBarrier();

    // Each thread processes multiple elements
    let base_idx = workgroup_idx * ELEMENTS_PER_WORKGROUP + local_idx;

    for (var i: u32 = 0u; i < ELEMENTS_PER_THREAD; i = i + 1u) {
        let idx = base_idx + i * WORKGROUP_SIZE;

        if (idx < params.input_size) {
            let key = keys_in[idx];
            let digit = extract_digit(key, params.pass_number);

            // Atomically increment local histogram
            atomicAdd(&local_histogram[digit], 1u);
        }
    }

    workgroupBarrier();

    // Write local histogram to global memory
    // Layout: histogram is stored column-major for efficient prefix scan
    // global_histogram[bucket][workgroup] = global_histogram[bucket * num_workgroups + workgroup]
    if (local_idx < RADIX_BUCKETS) {
        let count = atomicLoad(&local_histogram[local_idx]);
        let global_idx = local_idx * params.num_workgroups + workgroup_idx;
        atomicStore(&global_histogram[global_idx], count);
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Scatter
// ---------------------------------------------------------------------------

/// Scatter keys and values to their sorted positions.
///
/// Uses the precomputed prefix sums (global_offsets) to determine the
/// destination index for each key-value pair.
///
/// For efficiency, this kernel:
/// 1. Loads elements into shared memory
/// 2. Computes local offsets within the workgroup
/// 3. Writes to global memory using global + local offsets
@compute @workgroup_size(256, 1, 1)
fn scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_idx = lid.x;
    let workgroup_idx = wid.x;

    // Load global offsets for this workgroup into shared memory
    if (local_idx < RADIX_BUCKETS) {
        let global_idx = local_idx * params.num_workgroups + workgroup_idx;
        local_prefix[local_idx] = global_offsets[global_idx];
    }
    workgroupBarrier();

    // Process elements
    let base_idx = workgroup_idx * ELEMENTS_PER_WORKGROUP + local_idx;

    for (var i: u32 = 0u; i < ELEMENTS_PER_THREAD; i = i + 1u) {
        let idx = base_idx + i * WORKGROUP_SIZE;

        if (idx < params.input_size) {
            let key = keys_in[idx];
            let value = values_in[idx];
            let digit = extract_digit(key, params.pass_number);

            // Get the global base offset for this digit in this workgroup
            let global_base = local_prefix[digit];

            // Compute local offset within workgroup using atomics
            // This gives us the position within elements of this digit
            let local_offset = atomicAdd(&local_histogram[digit], 1u);

            // Write to sorted position
            let dst_idx = global_base + local_offset;
            keys_out[dst_idx] = key;
            values_out[dst_idx] = value;
        }
    }
}

/// Combined scatter kernel that first initializes local counters.
/// Use this instead of scatter when not following immediately after histogram.
@compute @workgroup_size(256, 1, 1)
fn scatter_init(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_idx = lid.x;
    let workgroup_idx = wid.x;

    // Initialize local histogram counters to zero for local offset computation
    if (local_idx < RADIX_BUCKETS) {
        atomicStore(&local_histogram[local_idx], 0u);
    }

    // Load global offsets for this workgroup into shared memory
    if (local_idx < RADIX_BUCKETS) {
        let global_idx = local_idx * params.num_workgroups + workgroup_idx;
        local_prefix[local_idx] = global_offsets[global_idx];
    }
    workgroupBarrier();

    // Process elements
    let base_idx = workgroup_idx * ELEMENTS_PER_WORKGROUP + local_idx;

    for (var i: u32 = 0u; i < ELEMENTS_PER_THREAD; i = i + 1u) {
        let idx = base_idx + i * WORKGROUP_SIZE;

        if (idx < params.input_size) {
            let key = keys_in[idx];
            let value = values_in[idx];
            let digit = extract_digit(key, params.pass_number);

            // Get the global base offset for this digit in this workgroup
            let global_base = local_prefix[digit];

            // Compute local offset within workgroup using atomics
            let local_offset = atomicAdd(&local_histogram[digit], 1u);

            // Write to sorted position
            let dst_idx = global_base + local_offset;
            keys_out[dst_idx] = key;
            values_out[dst_idx] = value;
        }
    }
}

// ---------------------------------------------------------------------------
// Keys-Only Variants
// ---------------------------------------------------------------------------

/// Scatter for keys only (no values).
/// More efficient when sorting indices or when values aren't needed.
@compute @workgroup_size(256, 1, 1)
fn scatter_keys_only(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_idx = lid.x;
    let workgroup_idx = wid.x;

    // Initialize local histogram counters to zero for local offset computation
    if (local_idx < RADIX_BUCKETS) {
        atomicStore(&local_histogram[local_idx], 0u);
    }

    // Load global offsets for this workgroup into shared memory
    if (local_idx < RADIX_BUCKETS) {
        let global_idx = local_idx * params.num_workgroups + workgroup_idx;
        local_prefix[local_idx] = global_offsets[global_idx];
    }
    workgroupBarrier();

    // Process elements
    let base_idx = workgroup_idx * ELEMENTS_PER_WORKGROUP + local_idx;

    for (var i: u32 = 0u; i < ELEMENTS_PER_THREAD; i = i + 1u) {
        let idx = base_idx + i * WORKGROUP_SIZE;

        if (idx < params.input_size) {
            let key = keys_in[idx];
            let digit = extract_digit(key, params.pass_number);

            // Get the global base offset for this digit in this workgroup
            let global_base = local_prefix[digit];

            // Compute local offset within workgroup using atomics
            let local_offset = atomicAdd(&local_histogram[digit], 1u);

            // Write to sorted position
            let dst_idx = global_base + local_offset;
            keys_out[dst_idx] = key;
        }
    }
}

// ---------------------------------------------------------------------------
// Utility Entry Points
// ---------------------------------------------------------------------------

/// Clear the global histogram buffer.
/// Dispatch with ceil(RADIX_BUCKETS * num_workgroups / WORKGROUP_SIZE) workgroups.
@compute @workgroup_size(256, 1, 1)
fn clear_histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let total_buckets = RADIX_BUCKETS * params.num_workgroups;

    if (idx < total_buckets) {
        atomicStore(&global_histogram[idx], 0u);
    }
}

/// Copy keys from input to output (used for initialization or odd passes).
@compute @workgroup_size(256, 1, 1)
fn copy_keys(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;

    if (idx < params.input_size) {
        keys_out[idx] = keys_in[idx];
    }
}

/// Copy key-value pairs from input to output.
@compute @workgroup_size(256, 1, 1)
fn copy_pairs(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;

    if (idx < params.input_size) {
        keys_out[idx] = keys_in[idx];
        values_out[idx] = values_in[idx];
    }
}

// ---------------------------------------------------------------------------
// Notes on Implementation
// ---------------------------------------------------------------------------
//
// Memory Layout for Global Histogram:
//   The histogram is stored in column-major order for efficient prefix scan:
//
//   global_histogram[bucket][workgroup] = global_histogram[bucket * num_workgroups + workgroup]
//
//   This layout allows the prefix scan to process all workgroup contributions
//   for one bucket contiguously, which is needed for correct scatter offsets.
//
//   After prefix scan, global_offsets contains the exclusive prefix sum:
//   - global_offsets[0][0] = 0 (first element goes to position 0)
//   - global_offsets[0][1] = count of digit 0 in workgroup 0
//   - global_offsets[0][2] = count of digit 0 in workgroups 0+1
//   - ...
//   - global_offsets[1][0] = total count of digit 0 across all workgroups
//   - etc.
//
// Double Buffering:
//   Radix sort alternates between two buffers (ping-pong):
//   - Even passes: read from buffer A, write to buffer B
//   - Odd passes: read from buffer B, write to buffer A
//
//   After 8 passes, the sorted result is in the starting buffer.
//   The Rust code manages this by swapping buffer bindings each pass.
//
// Stability:
//   Radix sort is stable because:
//   1. We process from LSB to MSB
//   2. Within each workgroup, elements with same digit maintain order via atomicAdd
//   3. Workgroups are processed in order (global_offsets are cumulative)
//
// Performance Considerations:
//   - ELEMENTS_PER_THREAD = 4 improves memory bandwidth utilization
//   - Local histogram in shared memory reduces global atomic contention
//   - Column-major histogram layout enables efficient prefix scan
//   - Workgroup size of 256 matches warp/wavefront sizes on most GPUs
//
// Comparison with Bitonic Sort:
//   - Radix sort: O(k*n) where k = number of passes (8 for 32-bit)
//   - Bitonic sort: O(n * log^2(n))
//   - Radix is faster for n > ~8192 elements with 32-bit keys
//   - Bitonic is better for small arrays or arbitrary comparison functions
