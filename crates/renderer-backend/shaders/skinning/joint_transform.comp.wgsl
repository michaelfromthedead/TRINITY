// SPDX-License-Identifier: MIT
//
// joint_transform.comp.wgsl - GPU Joint Transform Compute for TRINITY Engine (T-GPU-7.1)
//
// Computes world-space skinning matrices from local joint transforms and skeleton hierarchy.
// One thread per joint, traverses parent chain to build final skinning matrix.
//
// Algorithm:
// 1. Load local joint transform (mat4x3)
// 2. Traverse parent chain to accumulate world transform
// 3. Multiply by inverse bind pose matrix
// 4. Output final skinning matrix (mat4x3) for vertex skinning
//
// Performance: O(n*d) where n = joints, d = average hierarchy depth
// Target: < 0.05ms for 256 joints with depth <= 16

// ============================================================================
// Constants
// ============================================================================

/// Compute shader workgroup size
const WORKGROUP_SIZE: u32 = 64u;

/// Maximum number of joints supported per skeleton
const MAX_JOINTS: u32 = 256u;

/// Maximum hierarchy depth for parent traversal (prevents infinite loops)
const MAX_HIERARCHY_DEPTH: u32 = 64u;

/// Invalid joint index (root joints have no parent)
const INVALID_PARENT: i32 = -1;

// ============================================================================
// Structs
// ============================================================================

/// Joint data containing local transform and hierarchy info.
/// Memory layout: 64 bytes, aligned to vec4
///
/// | Offset | Field           | Size |
/// |--------|-----------------|------|
/// | 0      | local_transform | 48   | (mat4x3: 3 columns of vec4)
/// | 48     | parent_index    | 4    |
/// | 52     | joint_index     | 4    |
/// | 56     | _pad0           | 4    |
/// | 60     | _pad1           | 4    |
struct JointData {
    /// Local transform relative to parent (mat4x3 stored as 3 vec4 columns)
    /// Column 0: vec4(m00, m10, m20, 0) - X axis + padding
    /// Column 1: vec4(m01, m11, m21, 0) - Y axis + padding
    /// Column 2: vec4(m02, m12, m22, 0) - Z axis + padding
    /// Translation is packed in w components: (col0.w, col1.w, col2.w)
    local_col0: vec4<f32>,
    local_col1: vec4<f32>,
    local_col2: vec4<f32>,

    /// Parent joint index (-1 for root joints)
    parent_index: i32,

    /// This joint's index in the skeleton
    joint_index: u32,

    /// Padding for alignment
    _pad0: u32,
    _pad1: u32,
}

/// Joint transform computation parameters
struct JointTransformParams {
    /// Number of joints to process
    num_joints: u32,

    /// Flags for computation options
    flags: u32,

    /// Padding for 16-byte alignment
    _pad0: u32,
    _pad1: u32,
}

/// Inverse bind pose matrix (mat4x3 stored as 3 vec4 columns)
/// Pre-computed inverse of the bind pose for skinning
struct InverseBindPose {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
}

/// Output skinning matrix (mat4x3 stored as 3 vec4 columns)
/// Final world-space transform * inverse bind pose
struct SkinningMatrix {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
}

// ============================================================================
// Bindings
// ============================================================================

/// Joint transform parameters (uniform buffer)
@group(0) @binding(0) var<uniform> params: JointTransformParams;

/// Joint data array (read-only storage buffer)
@group(0) @binding(1) var<storage, read> joints: array<JointData>;

/// Inverse bind pose matrices (read-only storage buffer)
@group(0) @binding(2) var<storage, read> inverse_bind_poses: array<InverseBindPose>;

/// Output skinning matrices (read-write storage buffer)
@group(0) @binding(3) var<storage, read_write> skinning_matrices: array<SkinningMatrix>;

// ============================================================================
// Matrix Operations
// ============================================================================

/// Extract mat4x3 from JointData columns.
/// Returns 3x4 matrix where each row is a column of the transform.
fn extract_mat4x3(col0: vec4<f32>, col1: vec4<f32>, col2: vec4<f32>) -> mat4x3<f32> {
    // mat4x3 has 4 columns of vec3
    // Column 0: X axis (rotation/scale)
    // Column 1: Y axis (rotation/scale)
    // Column 2: Z axis (rotation/scale)
    // Column 3: Translation
    return mat4x3<f32>(
        vec3<f32>(col0.x, col0.y, col0.z),  // X axis
        vec3<f32>(col1.x, col1.y, col1.z),  // Y axis
        vec3<f32>(col2.x, col2.y, col2.z),  // Z axis
        vec3<f32>(col0.w, col1.w, col2.w)   // Translation in w components
    );
}

/// Multiply two mat4x3 matrices (treating them as mat4x4 with implicit [0,0,0,1] row).
/// Result = A * B where both A and B are affine transforms.
fn mul_mat4x3(a: mat4x3<f32>, b: mat4x3<f32>) -> mat4x3<f32> {
    // For affine transforms stored as mat4x3, multiplication is:
    // result.col[i] = a.col[0] * b.col[i].x + a.col[1] * b.col[i].y + a.col[2] * b.col[i].z + (i==3 ? a.col[3] : 0)

    // Extract columns from B
    let b_col0 = vec4<f32>(b[0], 0.0);  // X axis of B
    let b_col1 = vec4<f32>(b[1], 0.0);  // Y axis of B
    let b_col2 = vec4<f32>(b[2], 0.0);  // Z axis of B
    let b_col3 = vec4<f32>(b[3], 1.0);  // Translation of B (with w=1)

    // Extract columns from A (as vec4 with w=0 for rotation/scale, w=1 for translation)
    let a_col0 = vec4<f32>(a[0], 0.0);
    let a_col1 = vec4<f32>(a[1], 0.0);
    let a_col2 = vec4<f32>(a[2], 0.0);
    let a_col3 = vec4<f32>(a[3], 1.0);

    // Compute each column of result
    // result_col[i] = A * B_col[i]
    let r_col0 = a_col0 * b_col0.x + a_col1 * b_col0.y + a_col2 * b_col0.z;
    let r_col1 = a_col0 * b_col1.x + a_col1 * b_col1.y + a_col2 * b_col1.z;
    let r_col2 = a_col0 * b_col2.x + a_col1 * b_col2.y + a_col2 * b_col2.z;
    let r_col3 = a_col0 * b_col3.x + a_col1 * b_col3.y + a_col2 * b_col3.z + a_col3;

    return mat4x3<f32>(
        r_col0.xyz,
        r_col1.xyz,
        r_col2.xyz,
        r_col3.xyz
    );
}

/// Create identity mat4x3.
fn identity_mat4x3() -> mat4x3<f32> {
    return mat4x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),  // X axis
        vec3<f32>(0.0, 1.0, 0.0),  // Y axis
        vec3<f32>(0.0, 0.0, 1.0),  // Z axis
        vec3<f32>(0.0, 0.0, 0.0)   // Translation
    );
}

// ============================================================================
// Hierarchy Traversal
// ============================================================================

/// Compute world-space transform by traversing parent chain.
/// Returns accumulated world transform from root to this joint.
fn compute_world_transform(joint_idx: u32) -> mat4x3<f32> {
    // Start with this joint's local transform
    let joint = joints[joint_idx];
    var world = extract_mat4x3(joint.local_col0, joint.local_col1, joint.local_col2);

    // Traverse parent chain
    var current_parent = joint.parent_index;
    var depth = 0u;

    while (current_parent >= 0 && depth < MAX_HIERARCHY_DEPTH) {
        let parent_idx = u32(current_parent);

        // Safety check: prevent out-of-bounds access
        if (parent_idx >= params.num_joints) {
            break;
        }

        // Get parent's local transform
        let parent = joints[parent_idx];
        let parent_local = extract_mat4x3(
            parent.local_col0,
            parent.local_col1,
            parent.local_col2
        );

        // Accumulate: world = parent_local * world
        // This gives us: root_transform * ... * parent_transform * this_transform
        world = mul_mat4x3(parent_local, world);

        // Move to next parent
        current_parent = parent.parent_index;
        depth++;
    }

    return world;
}

/// Alternative: Compute world transform assuming joints are sorted in topological order.
/// Reads world transform of parent directly from output buffer.
/// This is more efficient but requires joints to be processed in correct order.
fn compute_world_transform_topological(joint_idx: u32) -> mat4x3<f32> {
    let joint = joints[joint_idx];
    let local = extract_mat4x3(joint.local_col0, joint.local_col1, joint.local_col2);

    if (joint.parent_index < 0) {
        // Root joint: world = local
        return local;
    }

    let parent_idx = u32(joint.parent_index);

    // Safety check
    if (parent_idx >= params.num_joints) {
        return local;
    }

    // Read parent's world transform (already computed if sorted topologically)
    let parent_world = skinning_matrices[parent_idx];
    let parent_mat = mat4x3<f32>(
        parent_world.col0.xyz,
        parent_world.col1.xyz,
        parent_world.col2.xyz,
        vec3<f32>(parent_world.col0.w, parent_world.col1.w, parent_world.col2.w)
    );

    // world = parent_world * local
    return mul_mat4x3(parent_mat, local);
}

// ============================================================================
// Main Compute Kernels
// ============================================================================

/// Flag: Use topological order optimization (joints must be sorted)
const FLAG_TOPOLOGICAL_ORDER: u32 = 1u;

/// Flag: Skip inverse bind pose multiplication (output world transforms only)
const FLAG_SKIP_INVERSE_BIND: u32 = 2u;

/// Main joint transform kernel.
/// Computes skinning matrix = world_transform * inverse_bind_pose.
@compute @workgroup_size(64)
fn compute_joint_transforms(@builtin(global_invocation_id) gid: vec3<u32>) {
    let joint_idx = gid.x;

    // Bounds check
    if (joint_idx >= params.num_joints) {
        return;
    }

    // Compute world-space transform via parent chain traversal
    let world = compute_world_transform(joint_idx);

    // Get inverse bind pose
    let inv_bind = inverse_bind_poses[joint_idx];
    let inv_bind_mat = mat4x3<f32>(
        inv_bind.col0.xyz,
        inv_bind.col1.xyz,
        inv_bind.col2.xyz,
        vec3<f32>(inv_bind.col0.w, inv_bind.col1.w, inv_bind.col2.w)
    );

    // Final skinning matrix = world * inverse_bind_pose
    var skinning: mat4x3<f32>;
    if ((params.flags & FLAG_SKIP_INVERSE_BIND) != 0u) {
        skinning = world;
    } else {
        skinning = mul_mat4x3(world, inv_bind_mat);
    }

    // Store result
    skinning_matrices[joint_idx] = SkinningMatrix(
        vec4<f32>(skinning[0], skinning[3].x),  // X axis + tx
        vec4<f32>(skinning[1], skinning[3].y),  // Y axis + ty
        vec4<f32>(skinning[2], skinning[3].z)   // Z axis + tz
    );
}

/// Topological order kernel (more efficient, requires sorted joints).
/// Process joints in order from root to leaves.
@compute @workgroup_size(64)
fn compute_joint_transforms_topological(@builtin(global_invocation_id) gid: vec3<u32>) {
    let joint_idx = gid.x;

    if (joint_idx >= params.num_joints) {
        return;
    }

    // Use topological traversal (reads parent's already-computed world transform)
    let world = compute_world_transform_topological(joint_idx);

    // Get inverse bind pose
    let inv_bind = inverse_bind_poses[joint_idx];
    let inv_bind_mat = mat4x3<f32>(
        inv_bind.col0.xyz,
        inv_bind.col1.xyz,
        inv_bind.col2.xyz,
        vec3<f32>(inv_bind.col0.w, inv_bind.col1.w, inv_bind.col2.w)
    );

    // Final skinning matrix = world * inverse_bind_pose
    var skinning: mat4x3<f32>;
    if ((params.flags & FLAG_SKIP_INVERSE_BIND) != 0u) {
        skinning = world;
    } else {
        skinning = mul_mat4x3(world, inv_bind_mat);
    }

    // Store result
    skinning_matrices[joint_idx] = SkinningMatrix(
        vec4<f32>(skinning[0], skinning[3].x),
        vec4<f32>(skinning[1], skinning[3].y),
        vec4<f32>(skinning[2], skinning[3].z)
    );
}

/// World transform only kernel (skips inverse bind pose, useful for debugging).
@compute @workgroup_size(64)
fn compute_world_transforms_only(@builtin(global_invocation_id) gid: vec3<u32>) {
    let joint_idx = gid.x;

    if (joint_idx >= params.num_joints) {
        return;
    }

    let world = compute_world_transform(joint_idx);

    skinning_matrices[joint_idx] = SkinningMatrix(
        vec4<f32>(world[0], world[3].x),
        vec4<f32>(world[1], world[3].y),
        vec4<f32>(world[2], world[3].z)
    );
}
