// SPDX-License-Identifier: MIT
//
// gpu_compact.comp.wgsl - GPU Stream Compaction for TRINITY Engine
//
// Removes elements where alive_flags[i] == 0 using parallel prefix sum.
// Used for particle compaction (T-GPU-2.2) and visibility buffer compaction (T-GPU-2.3).
//
// Algorithm: Three-phase parallel stream compaction
// 1. compact_scan: Per-block prefix sum, store block totals
// 2. scan_block_sums: Exclusive scan of block totals
// 3. compact_scatter: Scatter alive elements to compacted positions
//
// Performance: O(n) work, O(log n) steps per phase

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const SHARED_SIZE: u32 = 512u;

// ============================================================================
// Shared Memory (duplicated here since WGSL lacks #include)
// ============================================================================

var<workgroup> shared_data: array<u32, 512>;

/// Bank conflict avoidance index computation
fn conflict_free_index(idx: u32) -> u32 {
    return idx + (idx >> 5u);
}

/// Workgroup-level exclusive prefix sum (Blelloch algorithm)
fn workgroup_prefix_sum(lid: u32, value: u32) -> u32 {
    let ai = conflict_free_index(lid);
    shared_data[ai] = value;
    workgroupBarrier();

    // Up-sweep (reduce) phase
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

    // Clear last element for exclusive scan
    if (lid == 0u) {
        let last_idx = conflict_free_index(WORKGROUP_SIZE - 1u);
        shared_data[last_idx] = 0u;
    }
    workgroupBarrier();

    // Down-sweep phase
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

    return shared_data[ai];
}

// ============================================================================
// Uniform Buffer - Compaction Parameters
// ============================================================================

struct CompactParams {
    /// Total number of elements to process
    num_elements: u32,
    /// Number of workgroups (blocks) in dispatch
    num_blocks: u32,
    /// Reserved for future use
    _padding: vec2<u32>,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> params: CompactParams;

/// Input: alive flags array (0 = dead, 1 = alive)
@group(0) @binding(1) var<storage, read> alive_flags: array<u32>;

/// Input: source data to compact
@group(0) @binding(2) var<storage, read> input_data: array<u32>;

/// Output: compacted data (only alive elements)
@group(0) @binding(3) var<storage, read_write> output_data: array<u32>;

/// Intermediate: per-block prefix sums (exclusive sum of each block's total)
@group(0) @binding(4) var<storage, read_write> block_sums: array<u32>;

/// Output: final count of alive elements (single atomic u32)
@group(0) @binding(5) var<storage, read_write> output_count: atomic<u32>;

// ============================================================================
// Phase 1: Per-Block Prefix Sum
// ============================================================================
//
// Each workgroup computes the prefix sum of its alive_flags slice.
// The last thread in each workgroup stores the block's total count
// to block_sums for Phase 2.

@compute @workgroup_size(256)
fn compact_scan(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let idx = gid.x;

    // Load alive flag (0 for out-of-bounds elements)
    var flag: u32 = 0u;
    if (idx < params.num_elements) {
        flag = alive_flags[idx];
    }

    // Compute workgroup-local prefix sum
    let prefix = workgroup_prefix_sum(lid.x, flag);

    // Last thread in workgroup stores block total
    // Block total = prefix[last] + flag[last] (inclusive sum of last element)
    if (lid.x == WORKGROUP_SIZE - 1u) {
        block_sums[wid.x] = prefix + flag;
    }
}

// ============================================================================
// Phase 2: Scan Block Sums
// ============================================================================
//
// Single-workgroup kernel that scans the block_sums array.
// Converts block totals into global offsets for each block.
// Also computes and stores the total alive count.
//
// Note: For >256 blocks, this would need to be hierarchical.
// Current implementation handles up to 256 blocks (256*256 = 65536 elements).
// For larger arrays, use multi-level hierarchical scan.

@compute @workgroup_size(256)
fn scan_block_sums(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    // Load block sum (0 for inactive threads)
    var sum: u32 = 0u;
    if (lid.x < params.num_blocks) {
        sum = block_sums[lid.x];
    }

    // Exclusive scan of block sums
    let prefix = workgroup_prefix_sum(lid.x, sum);

    // Store prefix back to block_sums (now contains global offset for each block)
    if (lid.x < params.num_blocks) {
        block_sums[lid.x] = prefix;
    }

    // Last active thread stores total count
    // For the last block, total = prefix + sum
    if (lid.x == params.num_blocks - 1u) {
        atomicStore(&output_count, prefix + sum);
    }
}

// ============================================================================
// Phase 2b: Scan Block Sums (Hierarchical for Large Arrays)
// ============================================================================
//
// For arrays with >256 blocks, we need hierarchical scanning.
// This kernel handles the general case with arbitrary block counts.

@compute @workgroup_size(256)
fn scan_block_sums_hierarchical(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let idx = gid.x;

    // Load block sum
    var sum: u32 = 0u;
    if (idx < params.num_blocks) {
        sum = block_sums[idx];
    }

    // Compute workgroup-local prefix sum
    let prefix = workgroup_prefix_sum(lid.x, sum);

    // Store prefix back
    if (idx < params.num_blocks) {
        block_sums[idx] = prefix;
    }

    // For hierarchical scan, we'd store workgroup totals to another buffer
    // and repeat. For simplicity, current impl handles single-level.
}

// ============================================================================
// Phase 3: Scatter Alive Elements
// ============================================================================
//
// Each thread that has an alive element computes its final output position
// (local prefix + block offset) and writes to output_data.

@compute @workgroup_size(256)
fn compact_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let idx = gid.x;

    // Early exit for out-of-bounds threads
    if (idx >= params.num_elements) {
        return;
    }

    // Load alive flag
    let flag = alive_flags[idx];

    // Skip dead elements
    if (flag == 0u) {
        return;
    }

    // Recompute local prefix sum (must match Phase 1)
    // This is necessary because we can't store all prefix sums from Phase 1
    let local_prefix = workgroup_prefix_sum(lid.x, flag);

    // Add block offset from Phase 2
    let block_offset = block_sums[wid.x];
    let global_prefix = local_prefix + block_offset;

    // Scatter alive element to compacted position
    output_data[global_prefix] = input_data[idx];
}

// ============================================================================
// Single-Pass Variant (Small Arrays)
// ============================================================================
//
// For arrays that fit in a single workgroup (<=256 elements),
// we can compact in a single kernel dispatch.

@compute @workgroup_size(256)
fn compact_single_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = gid.x;

    // Load alive flag
    var flag: u32 = 0u;
    if (idx < params.num_elements) {
        flag = alive_flags[idx];
    }

    // Compute prefix sum
    let prefix = workgroup_prefix_sum(lid.x, flag);

    // Scatter alive elements
    if (idx < params.num_elements && flag != 0u) {
        output_data[prefix] = input_data[idx];
    }

    // Last thread stores total count
    if (lid.x == WORKGROUP_SIZE - 1u) {
        // Total = last prefix + last flag
        atomicStore(&output_count, prefix + flag);
    }
}

// ============================================================================
// Specialized Variants
// ============================================================================

// --- Particle Compaction Variant ---
// Compacts particle indices based on lifetime > 0

struct Particle {
    position: vec3<f32>,
    age: f32,
    velocity: vec3<f32>,
    lifetime: f32,
    size: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(1) @binding(0) var<storage, read> particles: array<Particle>;
@group(1) @binding(1) var<storage, read_write> compacted_particles: array<Particle>;

@compute @workgroup_size(256)
fn compact_particles_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= params.num_elements) {
        return;
    }

    let flag = alive_flags[idx];
    if (flag == 0u) {
        return;
    }

    let local_prefix = workgroup_prefix_sum(lid.x, flag);
    let block_offset = block_sums[wid.x];
    let global_prefix = local_prefix + block_offset;

    // Copy entire particle struct
    compacted_particles[global_prefix] = particles[idx];
}

// --- Visibility Buffer Compaction Variant ---
// Compacts instance indices for GPU-driven rendering

struct InstanceData {
    transform: mat4x4<f32>,
    mesh_id: u32,
    material_id: u32,
    flags: u32,
    _pad: u32,
}

@group(2) @binding(0) var<storage, read> instances: array<InstanceData>;
@group(2) @binding(1) var<storage, read_write> compacted_instances: array<InstanceData>;
@group(2) @binding(2) var<storage, read_write> draw_indirect: array<u32>;

@compute @workgroup_size(256)
fn compact_instances_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let idx = gid.x;
    if (idx >= params.num_elements) {
        return;
    }

    let flag = alive_flags[idx];
    if (flag == 0u) {
        return;
    }

    let local_prefix = workgroup_prefix_sum(lid.x, flag);
    let block_offset = block_sums[wid.x];
    let global_prefix = local_prefix + block_offset;

    // Copy instance data
    compacted_instances[global_prefix] = instances[idx];

    // Update draw indirect buffer (instance count) - only one thread does this
    if (lid.x == WORKGROUP_SIZE - 1u && wid.x == params.num_blocks - 1u) {
        // Store final instance count to draw_indirect[1] (instance_count field)
        draw_indirect[1] = atomicLoad(&output_count);
    }
}
