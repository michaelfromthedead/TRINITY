// SPDX-License-Identifier: MIT
//
// build_indirect.wgsl - Indirect Buffer Generation from Compacted Visible Objects (T-WGPU-P6.6.2)
//
// Generates DrawIndexedIndirectArgs from a compacted list of visible object indices.
// This shader runs after stream compaction (T-WGPU-P6.6.1) has produced a densely-packed
// buffer of visible object indices.
//
// Algorithm:
// 1. Read object index from compacted_indices[thread_idx]
// 2. Look up object data to get mesh_id
// 3. Look up LOD level for this object
// 4. Select appropriate mesh data based on LOD
// 5. Atomic increment draw count to get output slot
// 6. Write DrawIndexedIndirectArgs to output buffer
//
// Key Features:
// - LOD-aware mesh selection (supports 4 LOD levels)
// - Per-object LOD lookup via LodEntry buffer
// - Atomic draw count for GPU-side command counting
// - Object index passed via first_instance for shader access
//
// Memory Layout:
//   BuildIndirectParams: 16 bytes (visible_count + padding)
//   ObjectData:          144 bytes (transform, AABB, mesh_id, etc.)
//   MeshData:            48 bytes (index_count, first_index, base_vertex, LOD offsets)
//   LodEntry:            8 bytes (level + blend_factor)
//   DrawIndexedIndirectArgs: 20 bytes (standard indirect draw args)
//
// Performance:
//   - Workgroup size: 64 threads (optimal for GPU occupancy)
//   - One thread per visible object
//   - O(1) per object, O(n) total where n = visible count
//   - Atomic contention on draw_count (acceptable for most scenes)

// ============================================================================
// Constants
// ============================================================================

/// Workgroup size (64 threads for optimal GPU occupancy)
const WORKGROUP_SIZE: u32 = 64u;

/// Number of LOD levels supported (0 = highest detail, 3 = lowest)
const NUM_LOD_LEVELS: u32 = 4u;

/// Maximum LOD level index
const MAX_LOD_LEVEL: u32 = 3u;

// ============================================================================
// Structures
// ============================================================================

/// Parameters for indirect buffer generation.
///
/// Memory layout: 16 bytes (std140 aligned).
///
/// | Offset | Field         | Size | Description                      |
/// |--------|---------------|------|----------------------------------|
/// | 0      | visible_count | 4    | Number of visible objects        |
/// | 4      | max_draws     | 4    | Maximum draw commands to output  |
/// | 8      | _pad0         | 4    | Reserved for alignment           |
/// | 12     | _pad1         | 4    | Reserved for alignment           |
struct BuildIndirectParams {
    /// Number of visible objects in compacted_indices buffer.
    visible_count: u32,
    /// Maximum number of draw commands to generate.
    max_draws: u32,
    /// Reserved for future use.
    _pad0: u32,
    /// Reserved for alignment.
    _pad1: u32,
}

/// Per-object data for GPU-driven rendering.
///
/// Memory layout: 144 bytes (matches Rust ObjectData struct).
/// We only access mesh_index field, but must match full layout.
struct ObjectData {
    /// World transform matrix (column-major, 4x4).
    transform: mat4x4<f32>,
    /// AABB minimum corner.
    aabb_min: vec3<f32>,
    /// Padding for vec4 alignment.
    _pad0: f32,
    /// AABB maximum corner.
    aabb_max: vec3<f32>,
    /// Padding for vec4 alignment.
    _pad1: f32,
    /// Index into mesh buffer.
    mesh_index: u32,
    /// Index into material buffer.
    material_index: u32,
    /// LOD switch distances (squared).
    /// Uses array<f32, 4> instead of vec4<f32> to avoid 16-byte alignment requirement.
    lod_distances: array<f32, 4>,
    /// Object flags bitfield.
    flags: u32,
    /// Padding for 144-byte alignment.
    _padding: array<u32, 5>,
}

/// Mesh data with LOD support.
///
/// Memory layout: 48 bytes.
///
/// | Offset | Field            | Size | Description                     |
/// |--------|------------------|------|---------------------------------|
/// | 0      | index_count      | 4    | Number of indices (LOD 0)       |
/// | 4      | first_index      | 4    | Offset into global index buffer |
/// | 8      | base_vertex      | 4    | Vertex offset (signed)          |
/// | 12     | _pad             | 4    | Padding for vec4 alignment      |
/// | 16     | lod_index_counts | 16   | Index count per LOD level       |
/// | 32     | lod_first_index  | 16   | First index offset per LOD      |
struct MeshData {
    /// Number of indices in base mesh (LOD 0).
    index_count: u32,
    /// Offset into the global index buffer.
    first_index: u32,
    /// Vertex offset to add to each index (signed).
    base_vertex: i32,
    /// Padding for vec4 alignment.
    _pad: u32,
    /// Index count per LOD level (0-3).
    /// lod_index_counts[0] = LOD 0 (highest detail)
    /// lod_index_counts[3] = LOD 3 (lowest detail)
    lod_index_counts: vec4<u32>,
    /// First index offset per LOD level (relative to first_index).
    /// lod_first_index[0] = 0 (LOD 0 starts at first_index)
    /// lod_first_index[1] = offset to LOD 1 data
    lod_first_index: vec4<u32>,
}

/// Per-object LOD selection result.
///
/// Memory layout: 8 bytes (matches Rust LodEntry struct).
struct LodEntry {
    /// Selected LOD level (0 = highest detail, 3 = lowest).
    level: u32,
    /// Blend factor for smooth LOD transitions (0.0-1.0).
    blend_factor: f32,
}

/// Standard indirect draw indexed args (20 bytes).
///
/// Must match wgpu DrawIndexedIndirectArgs layout exactly.
/// This struct is read directly by the GPU for indirect rendering.
struct DrawIndexedIndirectArgs {
    /// Number of indices to draw.
    index_count: u32,
    /// Number of instances to draw.
    instance_count: u32,
    /// Offset into the index buffer.
    first_index: u32,
    /// Vertex offset added to each index (signed).
    base_vertex: i32,
    /// First instance index - used to pass object ID to shaders.
    first_instance: u32,
}

// ============================================================================
// Bindings - Group 0: Input Buffers (Read-Only)
// ============================================================================

/// Compacted indices buffer containing visible object indices.
/// Output from stream compaction (T-WGPU-P6.6.1).
@group(0) @binding(0) var<storage, read> compacted_indices: array<u32>;

/// Per-object data buffer containing transforms, mesh indices, etc.
@group(0) @binding(1) var<storage, read> object_data: array<ObjectData>;

/// Mesh data buffer containing index/vertex info per mesh.
@group(0) @binding(2) var<storage, read> mesh_data: array<MeshData>;

/// LOD selection buffer containing per-object LOD level.
/// Output from LOD selection pass (T-WGPU-P6.5.2).
@group(0) @binding(3) var<storage, read> lod_buffer: array<LodEntry>;

// ============================================================================
// Bindings - Group 1: Output Buffers (Read-Write)
// ============================================================================

/// Output indirect draw commands buffer.
@group(1) @binding(0) var<storage, read_write> indirect_commands: array<DrawIndexedIndirectArgs>;

/// Atomic counter for number of draw commands generated.
@group(1) @binding(1) var<storage, read_write> draw_count: atomic<u32>;

// ============================================================================
// Bindings - Group 2: Parameters (Uniform)
// ============================================================================

/// Build parameters uniform buffer.
@group(2) @binding(0) var<uniform> params: BuildIndirectParams;

// ============================================================================
// Helper Functions
// ============================================================================

/// Clamp LOD level to valid range.
fn clamp_lod(level: u32) -> u32 {
    return min(level, MAX_LOD_LEVEL);
}

/// Get index count for a mesh at a specific LOD level.
///
/// If LOD-specific count is 0, falls back to base mesh index_count.
fn get_lod_index_count(mesh: MeshData, lod: u32) -> u32 {
    let lod_count = mesh.lod_index_counts[lod];
    if (lod_count > 0u) {
        return lod_count;
    }
    // Fallback to base mesh if LOD not available
    return mesh.index_count;
}

/// Get first index offset for a mesh at a specific LOD level.
///
/// Returns absolute index buffer offset.
fn get_lod_first_index(mesh: MeshData, lod: u32) -> u32 {
    return mesh.first_index + mesh.lod_first_index[lod];
}

// ============================================================================
// Main Entry Point: Build Indirect Commands
// ============================================================================
//
// Each thread processes one visible object from the compacted indices buffer.
// The shader generates one DrawIndexedIndirectArgs per visible object.
//
// Workflow:
// 1. Read object index from compacted buffer
// 2. Look up object data to get mesh_id
// 3. Look up LOD level for this object
// 4. Get mesh data with LOD-appropriate index count/offset
// 5. Atomic increment to claim output slot
// 6. Write indirect draw command

@compute @workgroup_size(64, 1, 1)
fn build_indirect_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let compact_idx = gid.x;

    // Early exit for out-of-bounds threads
    if (compact_idx >= params.visible_count) {
        return;
    }

    // Step 1: Get original object index from compacted buffer
    let object_idx = compacted_indices[compact_idx];

    // Step 2: Look up object data to get mesh index
    let object = object_data[object_idx];
    let mesh_id = object.mesh_index;

    // Step 3: Get LOD level for this object
    let lod_entry = lod_buffer[object_idx];
    let lod_level = clamp_lod(lod_entry.level);

    // Step 4: Look up mesh data and get LOD-specific values
    let mesh = mesh_data[mesh_id];
    let index_count = get_lod_index_count(mesh, lod_level);
    let first_index = get_lod_first_index(mesh, lod_level);

    // Step 5: Atomic increment to get output slot
    let output_idx = atomicAdd(&draw_count, 1u);

    // Check we haven't exceeded max draws
    if (output_idx >= params.max_draws) {
        // Exceeded limit - decrement and exit
        // This should be rare with proper max_draws sizing
        atomicSub(&draw_count, 1u);
        return;
    }

    // Step 6: Write indirect draw command
    // - instance_count = 1 (one draw per visible object)
    // - first_instance = object_idx (allows shader to access object data)
    indirect_commands[output_idx] = DrawIndexedIndirectArgs(
        index_count,
        1u,                 // instance_count
        first_index,
        mesh.base_vertex,
        object_idx          // first_instance = object ID for shader access
    );
}

// ============================================================================
// Alternative Entry Point: Batched Build (Multiple Objects Per Thread)
// ============================================================================
//
// Optimization for very large visible counts.
// Each thread processes BATCH_SIZE objects.
// Reduces thread launch overhead at cost of slightly uneven workload.

const BATCH_SIZE: u32 = 4u;

@compute @workgroup_size(64, 1, 1)
fn build_indirect_batched(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base_idx = gid.x * BATCH_SIZE;

    // Process batch of objects
    for (var i: u32 = 0u; i < BATCH_SIZE; i = i + 1u) {
        let compact_idx = base_idx + i;

        // Bounds check
        if (compact_idx >= params.visible_count) {
            return;
        }

        // Get object index from compacted buffer
        let object_idx = compacted_indices[compact_idx];

        // Look up object and mesh data
        let object = object_data[object_idx];
        let mesh_id = object.mesh_index;
        let lod_entry = lod_buffer[object_idx];
        let lod_level = clamp_lod(lod_entry.level);
        let mesh = mesh_data[mesh_id];

        // Get LOD-specific values
        let index_count = get_lod_index_count(mesh, lod_level);
        let first_index = get_lod_first_index(mesh, lod_level);

        // Atomic increment to get output slot
        let output_idx = atomicAdd(&draw_count, 1u);

        if (output_idx >= params.max_draws) {
            atomicSub(&draw_count, 1u);
            return;
        }

        // Write indirect draw command
        indirect_commands[output_idx] = DrawIndexedIndirectArgs(
            index_count,
            1u,
            first_index,
            mesh.base_vertex,
            object_idx
        );
    }
}

// ============================================================================
// Clear Draw Count
// ============================================================================
//
// Reset draw count to 0 before generating new draw commands.
// Should be called at the start of each frame before build_indirect_main.

@compute @workgroup_size(1, 1, 1)
fn clear_draw_count() {
    atomicStore(&draw_count, 0u);
}

// ============================================================================
// Debug Entry Point: Validation
// ============================================================================
//
// Validates that all object indices in compacted buffer are within bounds.
// Only for debug builds - writes error info to indirect_commands[0..3].
//
// Output format:
// - indirect_commands[0].index_count = error count
// - indirect_commands[0].instance_count = first invalid index
// - indirect_commands[0].first_index = expected max
// - indirect_commands[0].base_vertex = actual invalid value (as i32)

@compute @workgroup_size(64, 1, 1)
fn validate_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let compact_idx = gid.x;

    if (compact_idx >= params.visible_count) {
        return;
    }

    let object_idx = compacted_indices[compact_idx];

    // Validate object index is within reasonable bounds
    // Note: We don't know the exact object count here, but can check for
    // obviously invalid values (e.g., 0xFFFFFFFF sentinel)
    if (object_idx == 0xFFFFFFFFu) {
        // Invalid sentinel value found
        let err_idx = atomicAdd(&draw_count, 1u);
        if (err_idx == 0u) {
            // Record first error details
            indirect_commands[0] = DrawIndexedIndirectArgs(
                1u,                     // error count
                compact_idx,            // first error index
                params.visible_count,   // expected max
                i32(object_idx),        // invalid value
                0u
            );
        }
    }
}
