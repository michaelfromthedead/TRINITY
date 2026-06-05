// SPDX-License-Identifier: MIT
//
// gpu_gen_draw_args.comp.wgsl - Draw Argument Generation (T-GPU-3.3)
//
// Generates IndirectDrawIndexedArgs from a sorted visibility buffer.
// Detects batch boundaries where material/mesh ID changes.
//
// Algorithm:
// - Input: Sorted visibility buffer (sorted by batch_key = material_id << 16 | mesh_id)
// - Output: Array of IndirectDrawIndexedArgs for indirect rendering
// - Each batch boundary starts a new draw command
//
// The shader scans the sorted visibility buffer, detects where batch keys change,
// and emits one draw command per unique batch. Instance count is determined by
// counting consecutive entries with the same batch key.
//
// Performance Target: <0.1ms for 100K instances

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// Maximum LOD levels supported
const MAX_LOD_LEVELS: u32 = 8u;

// ============================================================================
// Structures
// ============================================================================

/// Parameters for draw argument generation.
struct DrawArgsParams {
    /// Number of visible instances in the visibility buffer.
    num_visible: u32,
    /// Maximum number of draw commands to generate.
    max_draws: u32,
    /// Reserved for future use.
    _pad: vec2<u32>,
}

/// Visibility buffer entry (must be pre-sorted by batch_key).
struct VisibilityEntry {
    /// Original instance ID for transform lookup.
    instance_id: u32,
    /// Batch key: (material_id << 16) | mesh_id
    /// Used for draw call batching - all instances with same key are drawn together.
    batch_key: u32,
    /// LOD level selected during culling (0 = highest detail).
    lod_level: u32,
    /// Padding for alignment.
    _pad: u32,
}

/// Standard indirect draw indexed args (20 bytes, matches wgpu).
/// Must match Rust IndirectDrawIndexedArgs exactly.
struct IndirectDrawIndexedArgs {
    /// Number of indices to draw per instance.
    index_count: u32,
    /// Number of instances to draw in this batch.
    instance_count: u32,
    /// Offset into the index buffer.
    first_index: u32,
    /// Vertex offset added to each index.
    base_vertex: i32,
    /// First instance index (offset into visibility buffer for instance data).
    first_instance: u32,
}

/// Mesh metadata for looking up index counts per mesh/LOD.
struct MeshMetadata {
    /// Number of indices in the mesh.
    index_count: u32,
    /// Offset into the global index buffer.
    first_index: u32,
    /// Base vertex offset for this mesh.
    base_vertex: i32,
    /// Padding for vec4 alignment.
    _pad: u32,
    /// Index offset per LOD level (relative to first_index).
    /// lod_offsets[0] = 0 (full detail), lod_offsets[1] = offset to LOD1, etc.
    lod_index_offsets: array<u32, 8>,
    /// Index count per LOD level.
    /// Allows different triangle counts per LOD.
    lod_index_counts: array<u32, 8>,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> params: DrawArgsParams;

/// Sorted visibility buffer (sorted by batch_key).
@group(0) @binding(1) var<storage, read> visibility: array<VisibilityEntry>;

/// Mesh metadata table for index/vertex info per mesh.
@group(0) @binding(2) var<storage, read> mesh_metadata: array<MeshMetadata>;

/// Output: Generated draw commands.
@group(0) @binding(3) var<storage, read_write> draw_args: array<IndirectDrawIndexedArgs>;

/// Output: Number of draw commands generated (atomic).
@group(0) @binding(4) var<storage, read_write> draw_count: atomic<u32>;

// ============================================================================
// Workgroup Shared Memory
// ============================================================================

/// Flags for batch boundary detection (1 = start of new batch).
var<workgroup> batch_boundary: array<u32, 256>;

/// Running instance counts per thread position.
var<workgroup> instance_counts: array<u32, 256>;

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract mesh_id from batch_key.
fn get_mesh_id(batch_key: u32) -> u32 {
    return batch_key & 0xFFFFu;
}

/// Extract material_id from batch_key.
fn get_material_id(batch_key: u32) -> u32 {
    return batch_key >> 16u;
}

/// Check if this index starts a new batch.
fn is_batch_start(idx: u32) -> bool {
    if (idx == 0u) {
        return true;
    }
    if (idx >= params.num_visible) {
        return false;
    }
    return visibility[idx].batch_key != visibility[idx - 1u].batch_key;
}

/// Count consecutive instances with same batch_key starting at idx.
fn count_batch_instances(start_idx: u32) -> u32 {
    if (start_idx >= params.num_visible) {
        return 0u;
    }

    let start_key = visibility[start_idx].batch_key;
    var count: u32 = 1u;
    var i: u32 = start_idx + 1u;

    // Linear scan to count instances (bounded by visibility buffer size)
    while (i < params.num_visible && visibility[i].batch_key == start_key) {
        count = count + 1u;
        i = i + 1u;
    }

    return count;
}

// ============================================================================
// Main Shader: Detect Batch Boundaries and Generate Draw Args
// ============================================================================
//
// Each thread checks if its position is a batch boundary.
// If so, it atomically claims a draw slot and writes the draw command.

@compute @workgroup_size(256)
fn gen_draw_args(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = gid.x;

    // Early exit for out-of-bounds threads
    if (idx >= params.num_visible) {
        return;
    }

    // Check if this is a batch boundary
    if (!is_batch_start(idx)) {
        return;
    }

    // Count instances in this batch
    let instance_count = count_batch_instances(idx);

    // Atomically claim a draw command slot
    let draw_idx = atomicAdd(&draw_count, 1u);

    // Check we haven't exceeded max draws
    if (draw_idx >= params.max_draws) {
        // We've exceeded the limit - decrement and exit
        // Note: This is a rare case, should be avoided by proper max_draws sizing
        atomicSub(&draw_count, 1u);
        return;
    }

    // Get mesh info from the first instance in batch
    let entry = visibility[idx];
    let mesh_id = get_mesh_id(entry.batch_key);
    let lod = min(entry.lod_level, MAX_LOD_LEVELS - 1u);

    // Look up mesh metadata
    let mesh = mesh_metadata[mesh_id];

    // Calculate index count and offset based on LOD
    var index_count = mesh.index_count;
    var first_index = mesh.first_index;

    // Use LOD-specific values if available
    if (lod > 0u && mesh.lod_index_counts[lod] > 0u) {
        index_count = mesh.lod_index_counts[lod];
        first_index = mesh.first_index + mesh.lod_index_offsets[lod];
    } else if (mesh.lod_index_counts[0] > 0u) {
        // LOD 0 has explicit count
        index_count = mesh.lod_index_counts[0];
    }

    // Write the draw command
    draw_args[draw_idx] = IndirectDrawIndexedArgs(
        index_count,
        instance_count,
        first_index,
        mesh.base_vertex,
        idx  // first_instance = offset into visibility buffer
    );
}

// ============================================================================
// Alternative: Single-Pass for Small Visibility Buffers
// ============================================================================
//
// When visibility buffer fits in a single workgroup, we can use shared memory
// for more efficient boundary detection.

@compute @workgroup_size(256)
fn gen_draw_args_single_workgroup(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = gid.x;

    // Load batch key into shared memory (pad with sentinel)
    var batch_key: u32 = 0xFFFFFFFFu;
    if (idx < params.num_visible) {
        batch_key = visibility[idx].batch_key;
    }

    // Detect boundary: different from previous
    var is_boundary: u32 = 0u;
    if (idx < params.num_visible) {
        if (idx == 0u) {
            is_boundary = 1u;
        } else if (visibility[idx - 1u].batch_key != batch_key) {
            is_boundary = 1u;
        }
    }

    batch_boundary[lid.x] = is_boundary;
    workgroupBarrier();

    // Count instances in this batch using shared memory
    // Only boundary threads need to count
    if (is_boundary == 1u && idx < params.num_visible) {
        var count: u32 = 1u;
        var i: u32 = lid.x + 1u;
        while (i < 256u && gid.x + (i - lid.x) < params.num_visible) {
            if (batch_boundary[i] == 1u) {
                break;
            }
            count = count + 1u;
            i = i + 1u;
        }
        instance_counts[lid.x] = count;
    }
    workgroupBarrier();

    // Write draw commands for boundary threads
    if (is_boundary == 1u && idx < params.num_visible) {
        let draw_idx = atomicAdd(&draw_count, 1u);

        if (draw_idx < params.max_draws) {
            let entry = visibility[idx];
            let mesh_id = get_mesh_id(entry.batch_key);
            let lod = min(entry.lod_level, MAX_LOD_LEVELS - 1u);
            let mesh = mesh_metadata[mesh_id];

            var index_count = mesh.index_count;
            var first_index = mesh.first_index;

            if (lod > 0u && mesh.lod_index_counts[lod] > 0u) {
                index_count = mesh.lod_index_counts[lod];
                first_index = mesh.first_index + mesh.lod_index_offsets[lod];
            } else if (mesh.lod_index_counts[0] > 0u) {
                index_count = mesh.lod_index_counts[0];
            }

            draw_args[draw_idx] = IndirectDrawIndexedArgs(
                index_count,
                instance_counts[lid.x],
                first_index,
                mesh.base_vertex,
                idx
            );
        }
    }
}

// ============================================================================
// Clear Draw Count
// ============================================================================
//
// Reset draw count to 0 before generating new draw args.

@compute @workgroup_size(1)
fn clear_draw_count() {
    atomicStore(&draw_count, 0u);
}
