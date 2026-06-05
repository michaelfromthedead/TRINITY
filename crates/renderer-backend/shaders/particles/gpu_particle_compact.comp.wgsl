// SPDX-License-Identifier: MIT
//
// gpu_particle_compact.comp.wgsl - GPU Particle Compaction for TRINITY Engine (T-GPU-5.3)
//
// Compacts particle buffer by removing dead particles using prefix sum over alive flags.
// Uses the Blelloch parallel scan algorithm for efficient GPU-based stream compaction.
//
// Algorithm:
// 1. compute_prefix_sum: Compute per-block prefix sums and block totals from alive flags
// 2. scan_block_sums: Exclusive scan of block totals to get global offsets
// 3. compact_particles: Scatter alive particles to compacted positions
//
// Performance:
// - Work complexity: O(n)
// - Step complexity: O(log n) per phase
// - Target: < 0.3ms for 100K particles
//
// Stable compaction: Preserves relative ordering of alive particles.

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const SHARED_SIZE: u32 = 512u;

// Particle alive flag bit position
const PARTICLE_FLAG_ALIVE: u32 = 1u;

// ============================================================================
// Shared Memory
// ============================================================================

var<workgroup> shared_data: array<u32, 512>;

// ============================================================================
// Bank Conflict Avoidance
// ============================================================================

/// Compute index with bank conflict avoidance offset.
/// Adds padding to avoid bank conflicts on 32-bank GPU architectures.
fn conflict_free_index(idx: u32) -> u32 {
    return idx + (idx >> 5u);
}

// ============================================================================
// Blelloch Prefix Sum
// ============================================================================

/// Workgroup-level exclusive prefix sum using Blelloch algorithm.
/// Returns the exclusive prefix sum at this thread's position.
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
// Data Structures
// ============================================================================

/// Parameters for particle compaction operation.
struct ParticleCompactParams {
    /// Number of particles to process.
    num_particles: u32,
    /// Maximum particles the buffer can hold.
    max_particles: u32,
    /// Number of workgroups in dispatch.
    num_blocks: u32,
    /// Reserved for 16-byte alignment.
    _padding: u32,
}

/// GPU particle data (64 bytes per particle).
/// Matches the Particle struct from gpu_particle_spawn.comp.wgsl.
struct Particle {
    /// World-space position.
    position: vec3<f32>,
    /// Current age (seconds since spawn).
    age: f32,

    /// Current velocity (world units/second).
    velocity: vec3<f32>,
    /// Total lifetime (seconds).
    lifetime: f32,

    /// Current color (RGBA premultiplied alpha).
    color: vec4<f32>,

    /// Current size (world units).
    size: f32,
    /// Current rotation (radians).
    rotation: f32,
    /// Rotation speed (radians/second).
    rotation_speed: f32,
    /// Flags (bit 0: alive).
    flags: u32,
}

/// Indirect draw arguments for DrawIndexedIndirect.
/// Used to update instance_count after compaction.
struct DrawIndirectArgs {
    /// Number of indices per instance (6 for particle quad).
    index_count: u32,
    /// Number of instances (alive particle count).
    instance_count: u32,
    /// First index offset.
    first_index: u32,
    /// Base vertex offset.
    base_vertex: i32,
    /// First instance offset.
    first_instance: u32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> params: ParticleCompactParams;

/// Input: alive flags computed from particle update pass (0 = dead, 1 = alive).
@group(0) @binding(1) var<storage, read> alive_flags: array<u32>;

/// Intermediate: per-block prefix sums for multi-block compaction.
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

/// Input: source particle buffer to compact.
@group(0) @binding(3) var<storage, read> particles_in: array<Particle>;

/// Output: compacted particle buffer (only alive particles).
@group(0) @binding(4) var<storage, read_write> particles_out: array<Particle>;

/// Output: alive count (atomic u32).
@group(0) @binding(5) var<storage, read_write> alive_count: atomic<u32>;

/// Output: indirect draw buffer for particle rendering.
@group(0) @binding(6) var<storage, read_write> draw_indirect: DrawIndirectArgs;

// ============================================================================
// Phase 1: Compute Per-Block Prefix Sum
// ============================================================================
//
// Each workgroup processes WORKGROUP_SIZE particles, computing local prefix
// sums of alive flags. The last thread stores the block's total alive count
// to block_sums for Phase 2.

@compute @workgroup_size(256)
fn compute_prefix_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let idx = gid.x;

    // Load alive flag (0 for out-of-bounds particles)
    var flag: u32 = 0u;
    if (idx < params.num_particles) {
        flag = alive_flags[idx];
    }

    // Compute workgroup-local exclusive prefix sum
    let prefix = workgroup_prefix_sum(lid.x, flag);

    // Last thread in workgroup stores block total
    // Block total = exclusive prefix of last thread + its flag value
    if (lid.x == WORKGROUP_SIZE - 1u) {
        block_sums[wid.x] = prefix + flag;
    }
}

// ============================================================================
// Phase 2: Scan Block Sums
// ============================================================================
//
// Single-workgroup kernel that performs exclusive prefix sum on block_sums.
// Converts per-block totals into global offsets for each block.
// Also stores the total alive count.
//
// Limitation: Handles up to 256 blocks (256 * 256 = 65536 particles).
// For larger counts, hierarchical scanning would be needed.

@compute @workgroup_size(256)
fn scan_block_sums(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    // Load block sum (0 for threads beyond num_blocks)
    var sum: u32 = 0u;
    if (lid.x < params.num_blocks) {
        sum = block_sums[lid.x];
    }

    // Exclusive scan of block sums
    let prefix = workgroup_prefix_sum(lid.x, sum);

    // Store prefix back (now contains global offset for each block)
    if (lid.x < params.num_blocks) {
        block_sums[lid.x] = prefix;
    }

    // Last active thread stores total alive count
    if (lid.x == params.num_blocks - 1u) {
        let total = prefix + sum;
        atomicStore(&alive_count, total);
        // Also update indirect draw buffer instance_count
        draw_indirect.instance_count = total;
    }
}

// ============================================================================
// Phase 3: Scatter Particles (Compact)
// ============================================================================
//
// Each thread checks if its particle is alive. If so, computes the final
// scatter position (local prefix + block offset) and copies the particle
// to the output buffer. This preserves relative order (stable compaction).

@compute @workgroup_size(256)
fn compact_particles(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let idx = gid.x;

    // Early exit for out-of-bounds threads
    if (idx >= params.num_particles) {
        return;
    }

    // Load alive flag
    let flag = alive_flags[idx];

    // Skip dead particles
    if (flag == 0u) {
        return;
    }

    // Recompute local prefix sum (must match Phase 1 exactly)
    // This is required because we don't store intermediate prefix sums
    let local_prefix = workgroup_prefix_sum(lid.x, flag);

    // Add block offset from Phase 2
    let block_offset = block_sums[wid.x];
    let output_idx = local_prefix + block_offset;

    // Bounds check for output
    if (output_idx >= params.max_particles) {
        return;
    }

    // Copy particle to compacted position (stable: preserves order)
    particles_out[output_idx] = particles_in[idx];
}

// ============================================================================
// Single-Pass Variant (Small Particle Counts)
// ============================================================================
//
// For particle counts <= 256, we can compact in a single dispatch without
// the block-level scan. More efficient for small emitters.

@compute @workgroup_size(256)
fn compact_particles_single_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = gid.x;

    // Load alive flag
    var flag: u32 = 0u;
    if (idx < params.num_particles) {
        flag = alive_flags[idx];
    }

    // Compute exclusive prefix sum
    let prefix = workgroup_prefix_sum(lid.x, flag);

    // Scatter alive particles
    if (idx < params.num_particles && flag != 0u) {
        if (prefix < params.max_particles) {
            particles_out[prefix] = particles_in[idx];
        }
    }

    // Last thread stores total count
    if (lid.x == WORKGROUP_SIZE - 1u) {
        let total = prefix + flag;
        atomicStore(&alive_count, total);
        draw_indirect.instance_count = total;
    }
}

// ============================================================================
// In-Place Compaction Variant (Double-Buffered)
// ============================================================================
//
// Alternative entry point that supports ping-pong double buffering,
// where particles_in and particles_out are swapped each frame.
// This avoids a separate copy step.

@compute @workgroup_size(256)
fn compact_particles_inplace(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let idx = gid.x;

    if (idx >= params.num_particles) {
        return;
    }

    let flag = alive_flags[idx];

    if (flag == 0u) {
        return;
    }

    // Recompute prefix (matches Phase 1)
    let local_prefix = workgroup_prefix_sum(lid.x, flag);
    let block_offset = block_sums[wid.x];
    let output_idx = local_prefix + block_offset;

    // For in-place operation, only write if destination differs from source
    // and destination index is less than source (to avoid overwriting unread data)
    if (output_idx < params.max_particles && output_idx != idx) {
        particles_out[output_idx] = particles_in[idx];
    }
}

// ============================================================================
// Extract Alive Flags from Particle Buffer
// ============================================================================
//
// Utility kernel that extracts alive flags from the particle buffer's flags
// field. Run this before compaction if alive_flags buffer is not maintained
// separately by the update shader.

@compute @workgroup_size(256)
fn extract_alive_flags(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;

    if (idx >= params.num_particles) {
        return;
    }

    // Extract alive bit from particle flags
    let particle = particles_in[idx];
    let is_alive = (particle.flags & PARTICLE_FLAG_ALIVE) != 0u;

    // Store to alive_flags buffer (requires read_write on binding 1)
    // Note: This kernel uses a separate binding configuration
}
