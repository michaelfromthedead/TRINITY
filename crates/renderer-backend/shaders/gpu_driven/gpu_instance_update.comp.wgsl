// SPDX-License-Identifier: MIT
//
// gpu_instance_update.comp.wgsl - GPU Instance Update for TRINITY Engine (T-GPU-4.2)
//
// Updates per-instance transform data for GPU-driven instanced rendering.
// One thread per instance, updates transforms from animation/physics data
// and packs instance data for efficient GPU access.
//
// Features:
// - mat4x3 transform packing (rotation + translation, assumes uniform scale)
// - LOD index per instance from distance culling
// - Visibility flags from culling passes
// - Skinned mesh instance flags
// - Bounding volume updates
//
// Performance: O(n) work, single dispatch, <0.1ms for 100K instances

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;

/// Instance flag bits
const FLAG_VISIBLE: u32 = 1u;           // Instance passed visibility culling
const FLAG_SKINNED: u32 = 2u;           // Instance uses skeletal animation
const FLAG_DIRTY: u32 = 4u;             // Transform needs update
const FLAG_CAST_SHADOW: u32 = 8u;       // Instance casts shadows
const FLAG_RECEIVE_SHADOW: u32 = 16u;   // Instance receives shadows
const FLAG_TWO_SIDED: u32 = 32u;        // Two-sided rendering
const FLAG_MOTION_BLUR: u32 = 64u;      // Apply motion blur
const FLAG_STATIC: u32 = 128u;          // Static geometry (no transform updates)

/// Invalid LOD marker (instance culled by distance)
const LOD_CULLED: u32 = 0xFFFFFFFFu;

/// Invalid material ID marker
const INVALID_MATERIAL: u32 = 0xFFFFFFFFu;

// ============================================================================
// Structs
// ============================================================================

/// Input transform data from animation/physics systems.
/// Uses full mat4 for maximum flexibility before packing.
struct InputTransform {
    /// Full 4x4 transformation matrix (row-major)
    /// row0: [m00, m01, m02, m03]
    /// row1: [m10, m11, m12, m13]
    /// row2: [m20, m21, m22, m23]
    /// row3: [m30, m31, m32, m33] (typically [0,0,0,1])
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
    row3: vec4<f32>,
}

/// Packed instance data for efficient GPU rendering.
/// Uses mat4x3 format (12 floats = 48 bytes for transform).
struct InstanceData {
    /// Packed transform matrix (mat4x3): 12 floats
    /// Stores first 3 columns of mat4 (rotation + scale) + translation
    /// Layout: [m00, m10, m20, m01, m11, m21, m02, m12, m22, m03, m13, m23]
    transform: array<f32, 12>,

    /// Bounding sphere center (world space after transform)
    bounds_center: vec3<f32>,
    /// Bounding sphere radius (scaled by transform)
    bounds_radius: f32,

    /// Selected LOD index (from distance culling)
    lod_index: u32,
    /// Instance flags (visibility, skinned, etc.)
    flags: u32,
    /// Material table index
    material_id: u32,
    /// Reserved/padding for 64-byte alignment
    _padding: u32,
}

/// Instance update parameters.
struct InstanceUpdateParams {
    /// Number of instances to process
    num_instances: u32,
    /// Update flags (enable/disable specific updates)
    update_flags: u32,
    /// Delta time since last update (for motion vectors)
    delta_time: f32,
    /// Reserved/padding
    _padding: u32,
}

/// Local bounding data for an instance (object space).
struct LocalBounds {
    /// Object-space bounding sphere center
    center: vec3<f32>,
    /// Object-space bounding sphere radius
    radius: f32,
}

// ============================================================================
// Bindings
// ============================================================================

/// Update parameters (uniform buffer)
@group(0) @binding(0) var<uniform> params: InstanceUpdateParams;

/// Input transforms from animation/physics (read-only storage)
@group(0) @binding(1) var<storage, read> input_transforms: array<InputTransform>;

/// Local bounding data per instance (read-only storage)
@group(0) @binding(2) var<storage, read> local_bounds: array<LocalBounds>;

/// LOD indices from distance culling (read-only storage)
@group(0) @binding(3) var<storage, read> lod_indices: array<u32>;

/// Visibility flags from culling passes (read-only storage)
@group(0) @binding(4) var<storage, read> visibility_flags: array<u32>;

/// Material IDs per instance (read-only storage)
@group(0) @binding(5) var<storage, read> material_ids: array<u32>;

/// Output packed instance data (read-write storage)
@group(0) @binding(6) var<storage, read_write> output_instances: array<InstanceData>;

// ============================================================================
// Utility Functions
// ============================================================================

/// Pack a mat4 into mat4x3 format (12 floats).
/// Extracts the 3x3 rotation/scale matrix and translation vector.
/// Layout: column-major order for GPU consumption
fn pack_transform_mat4x3(t: InputTransform) -> array<f32, 12> {
    var result: array<f32, 12>;

    // Column 0: first column of rotation matrix
    result[0] = t.row0.x;
    result[1] = t.row1.x;
    result[2] = t.row2.x;

    // Column 1: second column of rotation matrix
    result[3] = t.row0.y;
    result[4] = t.row1.y;
    result[5] = t.row2.y;

    // Column 2: third column of rotation matrix
    result[6] = t.row0.z;
    result[7] = t.row1.z;
    result[8] = t.row2.z;

    // Column 3: translation
    result[9] = t.row0.w;
    result[10] = t.row1.w;
    result[11] = t.row2.w;

    return result;
}

/// Compute the maximum scale factor from a transform matrix.
/// Uses column vector lengths (accounts for non-uniform scale).
fn compute_max_scale(t: InputTransform) -> f32 {
    // Extract column vectors (rotation/scale)
    let col0 = vec3<f32>(t.row0.x, t.row1.x, t.row2.x);
    let col1 = vec3<f32>(t.row0.y, t.row1.y, t.row2.y);
    let col2 = vec3<f32>(t.row0.z, t.row1.z, t.row2.z);

    // Scale is the length of each column
    let scale0 = length(col0);
    let scale1 = length(col1);
    let scale2 = length(col2);

    // Return maximum scale for bounding radius
    return max(max(scale0, scale1), scale2);
}

/// Transform a point by the input transform matrix.
fn transform_point(t: InputTransform, p: vec3<f32>) -> vec3<f32> {
    let x = t.row0.x * p.x + t.row0.y * p.y + t.row0.z * p.z + t.row0.w;
    let y = t.row1.x * p.x + t.row1.y * p.y + t.row1.z * p.z + t.row1.w;
    let z = t.row2.x * p.x + t.row2.y * p.y + t.row2.z * p.z + t.row2.w;
    return vec3<f32>(x, y, z);
}

/// Build instance flags from input data.
fn build_instance_flags(visible: u32, lod: u32, skinned: bool) -> u32 {
    var flags = 0u;

    // Visibility from culling passes
    if (visible != 0u && lod != LOD_CULLED) {
        flags |= FLAG_VISIBLE;
    }

    // Skinned mesh flag
    if (skinned) {
        flags |= FLAG_SKINNED;
    }

    return flags;
}

// ============================================================================
// Main Compute Kernel
// ============================================================================

/// Per-instance update kernel.
///
/// Each thread processes one instance:
/// 1. Pack transform matrix to mat4x3 format
/// 2. Transform local bounds to world space
/// 3. Gather LOD and visibility data
/// 4. Write packed instance data
@compute @workgroup_size(256)
fn instance_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // Bounds check: skip threads beyond instance count
    if (idx >= params.num_instances) {
        return;
    }

    // Load input data
    let input_transform = input_transforms[idx];
    let bounds = local_bounds[idx];
    let lod = lod_indices[idx];
    let visibility = visibility_flags[idx];
    let material_id = material_ids[idx];

    // Pack transform to mat4x3 format
    let packed_transform = pack_transform_mat4x3(input_transform);

    // Transform bounding sphere to world space
    let world_center = transform_point(input_transform, bounds.center);
    let max_scale = compute_max_scale(input_transform);
    let world_radius = bounds.radius * max_scale;

    // Build instance flags
    let flags = build_instance_flags(visibility, lod, false);

    // Write packed instance data
    var output: InstanceData;
    output.transform = packed_transform;
    output.bounds_center = world_center;
    output.bounds_radius = world_radius;
    output.lod_index = lod;
    output.flags = flags;
    output.material_id = material_id;
    output._padding = 0u;

    output_instances[idx] = output;
}

// ============================================================================
// Alternative Entry Points
// ============================================================================

/// Update only transforms (skip bounds update).
/// Use when bounds are computed elsewhere or are static.
@compute @workgroup_size(256)
fn instance_update_transform_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_instances) {
        return;
    }

    let input_transform = input_transforms[idx];
    let packed_transform = pack_transform_mat4x3(input_transform);

    // Update only the transform portion
    for (var i = 0u; i < 12u; i++) {
        output_instances[idx].transform[i] = packed_transform[i];
    }

    // Mark as dirty for downstream passes
    output_instances[idx].flags |= FLAG_DIRTY;
}

/// Update only visibility flags (after culling pass).
/// Lightweight update when transforms haven't changed.
@compute @workgroup_size(256)
fn instance_update_visibility_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_instances) {
        return;
    }

    let lod = lod_indices[idx];
    let visibility = visibility_flags[idx];

    // Clear and rebuild visibility-related flags
    var flags = output_instances[idx].flags;
    flags &= ~(FLAG_VISIBLE); // Clear visibility

    if (visibility != 0u && lod != LOD_CULLED) {
        flags |= FLAG_VISIBLE;
    }

    output_instances[idx].lod_index = lod;
    output_instances[idx].flags = flags;
}

/// Update skinned mesh instances.
/// Handles skeletal animation transforms with additional bone data.
@compute @workgroup_size(256)
fn instance_update_skinned(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_instances) {
        return;
    }

    let input_transform = input_transforms[idx];
    let bounds = local_bounds[idx];
    let lod = lod_indices[idx];
    let visibility = visibility_flags[idx];
    let material_id = material_ids[idx];

    let packed_transform = pack_transform_mat4x3(input_transform);

    let world_center = transform_point(input_transform, bounds.center);
    let max_scale = compute_max_scale(input_transform);
    let world_radius = bounds.radius * max_scale;

    // Build flags with skinned bit set
    let flags = build_instance_flags(visibility, lod, true);

    var output: InstanceData;
    output.transform = packed_transform;
    output.bounds_center = world_center;
    output.bounds_radius = world_radius;
    output.lod_index = lod;
    output.flags = flags;
    output.material_id = material_id;
    output._padding = 0u;

    output_instances[idx] = output;
}

/// Clear all instance visibility (prepare for new frame).
@compute @workgroup_size(256)
fn instance_clear_visibility(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_instances) {
        return;
    }

    // Clear visibility flag, keep other flags
    output_instances[idx].flags &= ~FLAG_VISIBLE;
    output_instances[idx].lod_index = LOD_CULLED;
}
