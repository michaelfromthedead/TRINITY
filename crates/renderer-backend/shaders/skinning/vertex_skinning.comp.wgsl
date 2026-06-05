// SPDX-License-Identifier: MIT
//
// vertex_skinning.comp.wgsl - GPU Vertex Skinning for TRINITY Engine (T-GPU-7.2)
//
// Applies skeletal animation transforms to mesh vertices on the GPU.
// Reads source vertices and bone weights, blends joint transforms, and writes
// skinned vertices to a destination buffer.
//
// Algorithm:
// 1. Each thread processes one vertex
// 2. Read bone weights (up to 4 influences per vertex)
// 3. Read skinning matrices for each influencing joint
// 4. Compute weighted blend of transforms
// 5. Transform position, normal, and tangent
// 6. Write skinned vertex to output buffer
//
// Performance:
// - Workgroup size 256 for optimal GPU occupancy
// - One thread per vertex
// - Memory-coalesced reads from source buffer
// - Proper normal transformation (transpose of inverse for non-uniform scale)

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const MAX_BONE_INFLUENCES: u32 = 4u;

// ============================================================================
// Data Structures
// ============================================================================

/// Parameters controlling vertex skinning dispatch.
struct VertexSkinningParams {
    /// Number of vertices to process.
    vertex_count: u32,
    /// Number of joints in the skeleton.
    joint_count: u32,
    /// Flags (bit 0: use dual quaternion, bit 1: normalize weights).
    flags: u32,
    /// Padding for 16-byte alignment.
    _padding: u32,
}

/// Source vertex data (48 bytes, matches SkinnedVertex in Rust).
struct SourceVertex {
    /// Object-space position.
    position: vec3<f32>,
    /// Vertex normal (unit length).
    normal: vec3<f32>,
    /// Vertex tangent (xyz) and handedness (w).
    tangent: vec4<f32>,
    /// Texture coordinates.
    uv: vec2<f32>,
}

/// Bone weight data (16 bytes, matches BoneWeight in Rust).
struct BoneWeight {
    /// Joint indices (4 x u8 packed into u32).
    indices: u32,
    /// Blend weights for each joint (should sum to 1.0).
    weights: vec3<f32>,
}

/// Output skinned vertex (48 bytes).
struct SkinnedVertex {
    /// World-space or skinned position.
    position: vec3<f32>,
    /// Skinned normal (unit length).
    normal: vec3<f32>,
    /// Skinned tangent (xyz) and handedness (w).
    tangent: vec4<f32>,
    /// Texture coordinates (unchanged).
    uv: vec2<f32>,
}

/// 4x4 transformation matrix stored as 4 column vectors.
/// Matches the layout from joint_transform output.
struct JointMatrix {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
    col3: vec4<f32>,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> params: VertexSkinningParams;
@group(0) @binding(1) var<storage, read> src_vertices: array<SourceVertex>;
@group(0) @binding(2) var<storage, read> bone_weights: array<BoneWeight>;
@group(0) @binding(3) var<storage, read> joint_matrices: array<JointMatrix>;
@group(0) @binding(4) var<storage, read_write> dst_vertices: array<SkinnedVertex>;

// ============================================================================
// Matrix Utilities
// ============================================================================

/// Convert JointMatrix to mat4x4.
fn joint_to_mat4(j: JointMatrix) -> mat4x4<f32> {
    return mat4x4<f32>(j.col0, j.col1, j.col2, j.col3);
}

/// Extract the 3x3 rotation/scale portion of a 4x4 matrix.
fn extract_mat3(m: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        m[0].xyz,
        m[1].xyz,
        m[2].xyz
    );
}

/// Compute the cofactor matrix (transpose of adjugate) for normal transformation.
/// For non-uniform scale, normals must be transformed by the transpose of the inverse.
/// The cofactor matrix is equivalent to (M^-1)^T * det(M), but we don't need det
/// since we'll normalize the result anyway.
fn cofactor_mat3(m: mat3x3<f32>) -> mat3x3<f32> {
    // Cofactors for each element
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

    let c10 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = m[0][1] * m[2][0] - m[0][0] * m[2][1];

    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = m[0][2] * m[1][0] - m[0][0] * m[1][2];
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];

    // Return as column-major
    return mat3x3<f32>(
        vec3<f32>(c00, c10, c20),
        vec3<f32>(c01, c11, c21),
        vec3<f32>(c02, c12, c22)
    );
}

/// Unpack 4 bone indices from a packed u32.
fn unpack_indices(packed: u32) -> vec4<u32> {
    return vec4<u32>(
        packed & 0xFFu,
        (packed >> 8u) & 0xFFu,
        (packed >> 16u) & 0xFFu,
        (packed >> 24u) & 0xFFu
    );
}

/// Compute the fourth weight to ensure weights sum to 1.0.
fn compute_weight4(weights_xyz: vec3<f32>) -> f32 {
    return max(0.0, 1.0 - weights_xyz.x - weights_xyz.y - weights_xyz.z);
}

// ============================================================================
// Main Entry Point
// ============================================================================

@compute @workgroup_size(256)
fn skin_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vertex_index = gid.x;

    // Bounds check
    if (vertex_index >= params.vertex_count) {
        return;
    }

    // Read source vertex and bone weights
    let src = src_vertices[vertex_index];
    let bw = bone_weights[vertex_index];

    // Unpack bone indices
    let indices = unpack_indices(bw.indices);

    // Get weights (fourth weight computed to sum to 1.0)
    let w0 = bw.weights.x;
    let w1 = bw.weights.y;
    let w2 = bw.weights.z;
    let w3 = compute_weight4(bw.weights);

    // Initialize blended matrix to zero
    var blended_mat = mat4x4<f32>(
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        vec4<f32>(0.0)
    );

    // Blend joint transforms
    // Only blend influences with non-zero weight
    if (w0 > 0.0 && indices.x < params.joint_count) {
        let m0 = joint_to_mat4(joint_matrices[indices.x]);
        blended_mat = blended_mat + m0 * w0;
    }

    if (w1 > 0.0 && indices.y < params.joint_count) {
        let m1 = joint_to_mat4(joint_matrices[indices.y]);
        blended_mat = blended_mat + m1 * w1;
    }

    if (w2 > 0.0 && indices.z < params.joint_count) {
        let m2 = joint_to_mat4(joint_matrices[indices.z]);
        blended_mat = blended_mat + m2 * w2;
    }

    if (w3 > 0.0 && indices.w < params.joint_count) {
        let m3 = joint_to_mat4(joint_matrices[indices.w]);
        blended_mat = blended_mat + m3 * w3;
    }

    // Transform position (using full 4x4 with translation)
    let pos_homogeneous = blended_mat * vec4<f32>(src.position, 1.0);
    let skinned_position = pos_homogeneous.xyz;

    // Transform normal using cofactor matrix (correct for non-uniform scale)
    let rotation_scale = extract_mat3(blended_mat);
    let normal_matrix = cofactor_mat3(rotation_scale);
    let skinned_normal = normalize(normal_matrix * src.normal);

    // Transform tangent (just rotation/scale, not translation)
    let skinned_tangent_xyz = normalize(rotation_scale * src.tangent.xyz);
    let skinned_tangent = vec4<f32>(skinned_tangent_xyz, src.tangent.w);

    // Write output
    dst_vertices[vertex_index] = SkinnedVertex(
        skinned_position,
        skinned_normal,
        skinned_tangent,
        src.uv
    );
}

// ============================================================================
// Variant: Single Bone Influence (Optimized)
// ============================================================================
// For meshes where most vertices have only one bone influence.

@compute @workgroup_size(256)
fn skin_vertices_single(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vertex_index = gid.x;

    if (vertex_index >= params.vertex_count) {
        return;
    }

    let src = src_vertices[vertex_index];
    let bw = bone_weights[vertex_index];

    // Use only first bone index
    let bone_index = bw.indices & 0xFFu;

    if (bone_index >= params.joint_count) {
        // No valid bone - output unchanged
        dst_vertices[vertex_index] = SkinnedVertex(
            src.position,
            src.normal,
            src.tangent,
            src.uv
        );
        return;
    }

    let joint_mat = joint_to_mat4(joint_matrices[bone_index]);

    // Transform position
    let pos_homogeneous = joint_mat * vec4<f32>(src.position, 1.0);
    let skinned_position = pos_homogeneous.xyz;

    // Transform normal
    let rotation_scale = extract_mat3(joint_mat);
    let normal_matrix = cofactor_mat3(rotation_scale);
    let skinned_normal = normalize(normal_matrix * src.normal);

    // Transform tangent
    let skinned_tangent_xyz = normalize(rotation_scale * src.tangent.xyz);

    dst_vertices[vertex_index] = SkinnedVertex(
        skinned_position,
        skinned_normal,
        vec4<f32>(skinned_tangent_xyz, src.tangent.w),
        src.uv
    );
}

// ============================================================================
// Variant: Dual Quaternion Blending
// ============================================================================
// Better for characters to avoid volume collapse at twisted joints.
// Not implemented in this version - flag reserved for future use.
