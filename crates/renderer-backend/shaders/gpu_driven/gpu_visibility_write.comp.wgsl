// SPDX-License-Identifier: MIT
//
// gpu_visibility_write.comp.wgsl - Visibility Buffer Write (T-GPU-3.5)
//
// Writes visibility data for instances that pass all culling stages.
// This shader processes the compacted visible instance list and writes
// entries to the visibility buffer for subsequent material shading.
//
// Pipeline position:
// - Input: Compacted visible instances from culling (frustum + distance)
// - Output: Visibility buffer entries ready for material pass
//
// The visibility buffer approach stores:
// - Instance ID: For transform/instance data lookup
// - Primitive ID: Triangle within mesh (set by rasterizer)
// - Barycentrics: Interpolation coordinates (set by rasterizer)
//
// This compute shader handles the instance-level data; the rasterizer
// fills in primitive/barycentric data during the visibility pass.
//
// Performance Target: <0.05ms for 100K instances

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

// Sentinel values for invalid entries
const INVALID_INSTANCE_ID: u32 = 0xFFFFFFFFu;
const INVALID_PRIMITIVE_ID: u32 = 0xFFFFFFFFu;

// ============================================================================
// Structures
// ============================================================================

/// Parameters for visibility buffer writing.
struct VisibilityWriteParams {
    /// Number of visible instances to process (from culling output).
    num_visible: u32,
    /// Offset into the visibility buffer to start writing.
    /// Allows multi-view or split-frame rendering.
    visibility_buffer_offset: u32,
    /// Reserved for future use (e.g., frame index, debug flags).
    _pad: vec2<u32>,
}

/// Compacted visible instance from culling output.
/// This is the input from frustum + distance culling stages.
struct VisibleInstance {
    /// Original instance index (for transform lookup).
    original_index: u32,
    /// LOD level selected during distance culling (0 = highest detail).
    lod_level: u32,
    /// Batch key: (material_id << 16) | mesh_id
    /// Used for draw call batching and sorting.
    batch_key: u32,
    /// Reserved for future use (e.g., flags, cluster ID).
    _pad: u32,
}

/// Visibility data written to the visibility buffer.
/// This is the output consumed by the material shading pass.
///
/// Note: primitive_id and barycentrics are initialized to invalid/zero here
/// and will be overwritten by the rasterizer during the visibility pass.
struct VisibilityData {
    /// Instance ID for transform/material lookup.
    instance_id: u32,
    /// Primitive (triangle) ID within the mesh.
    /// Set by rasterizer, initialized to INVALID here.
    primitive_id: u32,
    /// Barycentric coordinates (u, v). w = 1 - u - v.
    /// Set by rasterizer, initialized to zero here.
    barycentrics: vec2<f32>,
}

/// Per-pixel visibility encoding (packed format).
/// Used when memory bandwidth is critical.
struct PackedVisibility {
    /// Packed data: bits [31:12] = instance_id (20 bits)
    ///              bits [11:0]  = primitive_id (12 bits)
    instance_primitive: u32,
    /// Packed barycentrics: bits [31:16] = u (16 bits, unorm)
    ///                      bits [15:0]  = v (16 bits, unorm)
    barycentrics_packed: u32,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> params: VisibilityWriteParams;

/// Input: Compacted visible instances from culling.
@group(0) @binding(1) var<storage, read> visible_instances: array<VisibleInstance>;

/// Output: Visibility buffer for material shading.
@group(0) @binding(2) var<storage, read_write> visibility_buffer: array<VisibilityData>;

/// Output: Count of entries written (atomic).
@group(0) @binding(3) var<storage, read_write> write_count: atomic<u32>;

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

/// Pack instance and primitive IDs into 32 bits.
/// Instance: 20 bits (max 1M instances)
/// Primitive: 12 bits (max 4K triangles per draw)
fn pack_ids(instance_id: u32, primitive_id: u32) -> u32 {
    return ((instance_id & 0xFFFFFu) << 12u) | (primitive_id & 0xFFFu);
}

/// Unpack instance ID from packed format.
fn unpack_instance_id(packed: u32) -> u32 {
    return packed >> 12u;
}

/// Unpack primitive ID from packed format.
fn unpack_primitive_id(packed: u32) -> u32 {
    return packed & 0xFFFu;
}

/// Pack barycentrics into 32 bits (16-bit unorm each).
fn pack_barycentrics(uv: vec2<f32>) -> u32 {
    let u_packed = u32(saturate(uv.x) * 65535.0);
    let v_packed = u32(saturate(uv.y) * 65535.0);
    return (u_packed << 16u) | v_packed;
}

/// Unpack barycentrics from 32 bits.
fn unpack_barycentrics(packed: u32) -> vec2<f32> {
    let u_val = f32(packed >> 16u) / 65535.0;
    let v_val = f32(packed & 0xFFFFu) / 65535.0;
    return vec2<f32>(u_val, v_val);
}

// ============================================================================
// Main Shader: Write Visibility Buffer Entries
// ============================================================================

@compute @workgroup_size(256)
fn visibility_write(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;

    // Early exit for out-of-bounds threads
    if (idx >= params.num_visible) {
        return;
    }

    // Read the visible instance from culling output
    let visible = visible_instances[idx];

    // Calculate output position in visibility buffer
    let output_idx = params.visibility_buffer_offset + idx;

    // Write visibility data
    // Note: primitive_id and barycentrics are placeholders
    // They will be overwritten during the rasterization visibility pass
    visibility_buffer[output_idx] = VisibilityData(
        visible.original_index,
        INVALID_PRIMITIVE_ID,  // Set by rasterizer
        vec2<f32>(0.0, 0.0)    // Set by rasterizer
    );

    // Increment write count atomically
    atomicAdd(&write_count, 1u);
}

// ============================================================================
// Clear Entry Point
// ============================================================================
//
// Reset the visibility buffer region before rendering.
// Called once per frame/view to initialize entries to invalid state.

@compute @workgroup_size(256)
fn visibility_clear(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;

    if (idx >= params.num_visible) {
        return;
    }

    let output_idx = params.visibility_buffer_offset + idx;

    // Write invalid/cleared visibility data
    visibility_buffer[output_idx] = VisibilityData(
        INVALID_INSTANCE_ID,
        INVALID_PRIMITIVE_ID,
        vec2<f32>(0.0, 0.0)
    );
}

// ============================================================================
// Write Count Clear Entry Point
// ============================================================================

@compute @workgroup_size(1)
fn clear_write_count() {
    atomicStore(&write_count, 0u);
}
