// SPDX-License-Identifier: MIT
//
// dualquat_skinning.comp.wgsl - Dual Quaternion Skinning for TRINITY Engine (T-GPU-7.3)
//
// Implements dual quaternion blending for skeletal animation, providing artifact-free
// skinning without the volume loss ("candy wrapper") effect of linear blend skinning.
//
// Dual quaternion representation:
//   DQ = (q_r, q_d) where:
//     q_r = rotation quaternion (real part)
//     q_d = 0.5 * translation_quat * q_r (dual part)
//
// Algorithm:
// 1. Convert joint mat4 transforms to dual quaternions (or use pre-converted)
// 2. For each vertex, blend dual quaternions with bone weights
// 3. Handle antipodal quaternions (flip sign if dot(q1.real, q2.real) < 0)
// 4. Normalize the blended dual quaternion
// 5. Apply to vertex position and normal
//
// Performance:
// - Workgroup size 256 for optimal GPU occupancy
// - One thread per vertex
// - Memory-coalesced reads/writes

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE: u32 = 256u;
const MAX_WEIGHTS_PER_VERTEX: u32 = 4u;

// ============================================================================
// Data Structures
// ============================================================================

/// Dual quaternion representation (32 bytes).
/// Real part encodes rotation, dual part encodes translation.
struct DualQuat {
    /// Rotation quaternion (x, y, z, w).
    real: vec4<f32>,
    /// Dual part = 0.5 * t * real where t = (tx, ty, tz, 0).
    dual: vec4<f32>,
}

/// Skinning parameters for compute dispatch.
struct DualQuatSkinningParams {
    /// Number of vertices to process.
    vertex_count: u32,
    /// Number of joints in the skeleton.
    joint_count: u32,
    /// Stride between vertices in the input buffer (bytes / 4 = floats).
    vertex_stride: u32,
    /// Offset to position in vertex (in floats).
    position_offset: u32,
}

/// Input vertex with position, normal, and skinning data.
struct SkinnedVertex {
    /// World-space position.
    position: vec3<f32>,
    /// Vertex normal.
    normal: vec3<f32>,
    /// Bone indices (up to 4 bones).
    bone_indices: vec4<u32>,
    /// Bone weights (sum should be 1.0).
    bone_weights: vec4<f32>,
}

/// Output skinned vertex data.
struct OutputVertex {
    /// Transformed position.
    position: vec3<f32>,
    /// Transformed normal.
    normal: vec3<f32>,
}

// ============================================================================
// Bindings
// ============================================================================

@group(0) @binding(0) var<uniform> params: DualQuatSkinningParams;
@group(0) @binding(1) var<storage, read> joint_dualquats: array<DualQuat>;
@group(0) @binding(2) var<storage, read> vertices_in: array<SkinnedVertex>;
@group(0) @binding(3) var<storage, read_write> vertices_out: array<OutputVertex>;

// Alternate binding set for mat4 to dual quat conversion
@group(1) @binding(0) var<storage, read> joint_matrices: array<mat4x4<f32>>;
@group(1) @binding(1) var<storage, read_write> joint_dualquats_out: array<DualQuat>;

// ============================================================================
// Quaternion Operations
// ============================================================================

/// Quaternion multiplication: q1 * q2
fn quat_mul(q1: vec4<f32>, q2: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

/// Quaternion conjugate.
fn quat_conjugate(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

/// Normalize a quaternion.
fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
    let len = length(q);
    if len > 0.0 {
        return q / len;
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

/// Rotate a vector by a quaternion.
fn quat_rotate_vector(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    // v' = q * v * q^(-1)
    // Optimized formula: v' = v + 2 * q.w * (q.xyz x v) + 2 * (q.xyz x (q.xyz x v))
    let qv = q.xyz;
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    return v + 2.0 * (q.w * uv + uuv);
}

// ============================================================================
// Dual Quaternion Operations
// ============================================================================

/// Create a dual quaternion from rotation quaternion and translation.
fn dualquat_from_rotation_translation(rotation: vec4<f32>, translation: vec3<f32>) -> DualQuat {
    // dual = 0.5 * t * real where t = (tx, ty, tz, 0)
    let t_quat = vec4<f32>(translation, 0.0);
    let dual = 0.5 * quat_mul(t_quat, rotation);

    return DualQuat(rotation, dual);
}

/// Convert a 4x4 transformation matrix to a dual quaternion.
fn mat4_to_dualquat(m: mat4x4<f32>) -> DualQuat {
    // Extract rotation as quaternion using Shepperd's method
    let trace = m[0][0] + m[1][1] + m[2][2];
    var q: vec4<f32>;

    if trace > 0.0 {
        let s = sqrt(trace + 1.0) * 2.0; // s = 4 * qw
        q = vec4<f32>(
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            0.25 * s
        );
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0; // s = 4 * qx
        q = vec4<f32>(
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s
        );
    } else if m[1][1] > m[2][2] {
        let s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0; // s = 4 * qy
        q = vec4<f32>(
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s
        );
    } else {
        let s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0; // s = 4 * qz
        q = vec4<f32>(
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
            (m[1][0] - m[0][1]) / s
        );
    }

    // Normalize the rotation quaternion
    q = quat_normalize(q);

    // Extract translation from the last column
    let translation = vec3<f32>(m[3][0], m[3][1], m[3][2]);

    return dualquat_from_rotation_translation(q, translation);
}

/// Normalize a dual quaternion.
fn dualquat_normalize(dq: DualQuat) -> DualQuat {
    let len = length(dq.real);
    if len > 0.0 {
        let inv_len = 1.0 / len;
        return DualQuat(
            dq.real * inv_len,
            dq.dual * inv_len
        );
    }
    return DualQuat(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(0.0));
}

/// Add two dual quaternions.
fn dualquat_add(a: DualQuat, b: DualQuat) -> DualQuat {
    return DualQuat(a.real + b.real, a.dual + b.dual);
}

/// Scale a dual quaternion by a scalar.
fn dualquat_scale(dq: DualQuat, s: f32) -> DualQuat {
    return DualQuat(dq.real * s, dq.dual * s);
}

/// Blend dual quaternions with antipodal quaternion handling.
/// When dot(q1.real, q2.real) < 0, the quaternions represent the same rotation
/// but are on opposite hemispheres. We flip q2 to ensure shortest path interpolation.
fn dualquat_blend(
    dq1: DualQuat, w1: f32,
    dq2: DualQuat, w2: f32,
    dq3: DualQuat, w3: f32,
    dq4: DualQuat, w4: f32
) -> DualQuat {
    // Start with first dual quaternion (reference for hemisphere)
    var result = dualquat_scale(dq1, w1);

    // Blend second quaternion with antipodal check
    if w2 > 0.0 {
        let d2 = select(-1.0, 1.0, dot(dq1.real, dq2.real) >= 0.0);
        result = dualquat_add(result, dualquat_scale(DualQuat(dq2.real * d2, dq2.dual * d2), w2));
    }

    // Blend third quaternion with antipodal check
    if w3 > 0.0 {
        let d3 = select(-1.0, 1.0, dot(dq1.real, dq3.real) >= 0.0);
        result = dualquat_add(result, dualquat_scale(DualQuat(dq3.real * d3, dq3.dual * d3), w3));
    }

    // Blend fourth quaternion with antipodal check
    if w4 > 0.0 {
        let d4 = select(-1.0, 1.0, dot(dq1.real, dq4.real) >= 0.0);
        result = dualquat_add(result, dualquat_scale(DualQuat(dq4.real * d4, dq4.dual * d4), w4));
    }

    return dualquat_normalize(result);
}

/// Extract translation from a dual quaternion.
fn dualquat_get_translation(dq: DualQuat) -> vec3<f32> {
    // t = 2 * dual * conjugate(real)
    let t_quat = 2.0 * quat_mul(dq.dual, quat_conjugate(dq.real));
    return t_quat.xyz;
}

/// Transform a point by a dual quaternion.
fn dualquat_transform_point(dq: DualQuat, point: vec3<f32>) -> vec3<f32> {
    // First rotate the point, then add translation
    let rotated = quat_rotate_vector(dq.real, point);
    let translation = dualquat_get_translation(dq);
    return rotated + translation;
}

/// Transform a normal by a dual quaternion (rotation only, no translation).
fn dualquat_transform_normal(dq: DualQuat, normal: vec3<f32>) -> vec3<f32> {
    return quat_rotate_vector(dq.real, normal);
}

// ============================================================================
// Compute Shaders
// ============================================================================

/// Main dual quaternion skinning compute shader.
/// Transforms vertices using pre-computed joint dual quaternions.
@compute @workgroup_size(256)
fn dq_skinning(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vertex_idx = global_id.x;

    if vertex_idx >= params.vertex_count {
        return;
    }

    // Read input vertex
    let vertex = vertices_in[vertex_idx];

    // Get joint dual quaternions for this vertex's bones
    let dq0 = joint_dualquats[vertex.bone_indices.x];
    let dq1 = joint_dualquats[vertex.bone_indices.y];
    let dq2 = joint_dualquats[vertex.bone_indices.z];
    let dq3 = joint_dualquats[vertex.bone_indices.w];

    // Blend dual quaternions with weights (handles antipodal quaternions)
    let blended_dq = dualquat_blend(
        dq0, vertex.bone_weights.x,
        dq1, vertex.bone_weights.y,
        dq2, vertex.bone_weights.z,
        dq3, vertex.bone_weights.w
    );

    // Transform position and normal
    let transformed_position = dualquat_transform_point(blended_dq, vertex.position);
    let transformed_normal = normalize(dualquat_transform_normal(blended_dq, vertex.normal));

    // Write output
    vertices_out[vertex_idx] = OutputVertex(transformed_position, transformed_normal);
}

/// Convert joint matrices to dual quaternions.
/// Run this once per frame when joint matrices change.
@compute @workgroup_size(256)
fn dq_convert_joints(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let joint_idx = global_id.x;

    if joint_idx >= params.joint_count {
        return;
    }

    // Read joint matrix
    let joint_matrix = joint_matrices[joint_idx];

    // Convert to dual quaternion
    let dq = mat4_to_dualquat(joint_matrix);

    // Write output
    joint_dualquats_out[joint_idx] = dq;
}
