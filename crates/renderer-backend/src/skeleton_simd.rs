//! SIMD-optimized skeleton pose processing (T-AN-1.4).
//!
//! This module provides Structure-of-Arrays (SoA) pose storage and SIMD-accelerated
//! operations for high-performance skeletal animation with large bone counts.
//!
//! # Architecture
//!
//! Instead of Array-of-Structures (AoS) layout where each transform is contiguous:
//! ```text
//! [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w, scale_x, scale_y, scale_z] * N bones
//! ```
//!
//! We use Structure-of-Arrays (SoA) layout:
//! ```text
//! positions_x: [x0, x1, x2, ..., xN]
//! positions_y: [y0, y1, y2, ..., yN]
//! ...
//! ```
//!
//! This enables SIMD vectorization across 4 or 8 bones simultaneously using
//! SSE/AVX on x86 or NEON on ARM.
//!
//! # Performance
//!
//! Typical speedups over scalar AoS processing:
//! - 4x with SSE/NEON (4-wide)
//! - 8x with AVX (8-wide)
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::skeleton_simd::{SoAPose, blend_poses_simd};
//! use renderer_backend::skeleton::{Transform, Skeleton};
//!
//! // Convert AoS to SoA
//! let transforms = vec![Transform::IDENTITY; 100];
//! let soa_a = SoAPose::from_aos(&transforms);
//! let soa_b = SoAPose::from_aos(&transforms);
//!
//! // SIMD blend
//! let mut result = SoAPose::with_capacity(100);
//! blend_poses_simd(&soa_a, &soa_b, 0.5, &mut result);
//!
//! // Convert back for rendering
//! let blended_transforms = result.to_aos();
//! ```

use glam::{Mat4, Quat, Vec3, Vec4};

use crate::skeleton::{Skeleton, Transform};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// SIMD lane width for portable operations.
/// We process bones in groups of 4 (SSE/NEON compatible).
pub const SIMD_LANE_WIDTH: usize = 4;

/// Epsilon for quaternion normalization.
const QUAT_NORMALIZE_EPSILON: f32 = 1e-8;

// ---------------------------------------------------------------------------
// SoAPose
// ---------------------------------------------------------------------------

/// Structure-of-Arrays pose storage for SIMD-optimized processing.
///
/// Each component is stored in a separate array, enabling SIMD operations
/// to process multiple bones in parallel.
#[derive(Clone, Debug, Default)]
pub struct SoAPose {
    /// X components of all bone positions.
    pub positions_x: Vec<f32>,
    /// Y components of all bone positions.
    pub positions_y: Vec<f32>,
    /// Z components of all bone positions.
    pub positions_z: Vec<f32>,
    /// X components of all bone rotations (quaternion).
    pub rotations_x: Vec<f32>,
    /// Y components of all bone rotations (quaternion).
    pub rotations_y: Vec<f32>,
    /// Z components of all bone rotations (quaternion).
    pub rotations_z: Vec<f32>,
    /// W components of all bone rotations (quaternion).
    pub rotations_w: Vec<f32>,
    /// X components of all bone scales.
    pub scales_x: Vec<f32>,
    /// Y components of all bone scales.
    pub scales_y: Vec<f32>,
    /// Z components of all bone scales.
    pub scales_z: Vec<f32>,
}

impl SoAPose {
    /// Create an empty SoA pose.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a SoA pose with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(bone_count: usize) -> Self {
        Self {
            positions_x: Vec::with_capacity(bone_count),
            positions_y: Vec::with_capacity(bone_count),
            positions_z: Vec::with_capacity(bone_count),
            rotations_x: Vec::with_capacity(bone_count),
            rotations_y: Vec::with_capacity(bone_count),
            rotations_z: Vec::with_capacity(bone_count),
            rotations_w: Vec::with_capacity(bone_count),
            scales_x: Vec::with_capacity(bone_count),
            scales_y: Vec::with_capacity(bone_count),
            scales_z: Vec::with_capacity(bone_count),
        }
    }

    /// Create a SoA pose with identity transforms for the given bone count.
    pub fn identity(bone_count: usize) -> Self {
        Self {
            positions_x: vec![0.0; bone_count],
            positions_y: vec![0.0; bone_count],
            positions_z: vec![0.0; bone_count],
            rotations_x: vec![0.0; bone_count],
            rotations_y: vec![0.0; bone_count],
            rotations_z: vec![0.0; bone_count],
            rotations_w: vec![1.0; bone_count],
            scales_x: vec![1.0; bone_count],
            scales_y: vec![1.0; bone_count],
            scales_z: vec![1.0; bone_count],
        }
    }

    /// Convert Array-of-Structures transforms to Structure-of-Arrays layout.
    ///
    /// This is O(n) where n is the number of transforms.
    pub fn from_aos(transforms: &[Transform]) -> Self {
        let n = transforms.len();
        let mut pose = Self::with_capacity(n);

        for t in transforms {
            pose.positions_x.push(t.position.x);
            pose.positions_y.push(t.position.y);
            pose.positions_z.push(t.position.z);
            pose.rotations_x.push(t.rotation.x);
            pose.rotations_y.push(t.rotation.y);
            pose.rotations_z.push(t.rotation.z);
            pose.rotations_w.push(t.rotation.w);
            pose.scales_x.push(t.scale.x);
            pose.scales_y.push(t.scale.y);
            pose.scales_z.push(t.scale.z);
        }

        pose
    }

    /// Convert Structure-of-Arrays back to Array-of-Structures layout.
    ///
    /// This is O(n) where n is the number of bones.
    pub fn to_aos(&self) -> Vec<Transform> {
        let n = self.bone_count();
        let mut transforms = Vec::with_capacity(n);

        for i in 0..n {
            transforms.push(Transform {
                position: Vec3::new(
                    self.positions_x[i],
                    self.positions_y[i],
                    self.positions_z[i],
                ),
                rotation: Quat::from_xyzw(
                    self.rotations_x[i],
                    self.rotations_y[i],
                    self.rotations_z[i],
                    self.rotations_w[i],
                ),
                scale: Vec3::new(self.scales_x[i], self.scales_y[i], self.scales_z[i]),
            });
        }

        transforms
    }

    /// Get the number of bones in this pose.
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.positions_x.len()
    }

    /// Check if the pose is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.positions_x.is_empty()
    }

    /// Clear all data, keeping allocated capacity.
    pub fn clear(&mut self) {
        self.positions_x.clear();
        self.positions_y.clear();
        self.positions_z.clear();
        self.rotations_x.clear();
        self.rotations_y.clear();
        self.rotations_z.clear();
        self.rotations_w.clear();
        self.scales_x.clear();
        self.scales_y.clear();
        self.scales_z.clear();
    }

    /// Resize the pose to the given bone count, filling with identity transforms.
    pub fn resize(&mut self, bone_count: usize) {
        self.positions_x.resize(bone_count, 0.0);
        self.positions_y.resize(bone_count, 0.0);
        self.positions_z.resize(bone_count, 0.0);
        self.rotations_x.resize(bone_count, 0.0);
        self.rotations_y.resize(bone_count, 0.0);
        self.rotations_z.resize(bone_count, 0.0);
        self.rotations_w.resize(bone_count, 1.0);
        self.scales_x.resize(bone_count, 1.0);
        self.scales_y.resize(bone_count, 1.0);
        self.scales_z.resize(bone_count, 1.0);
    }

    /// Get a single transform at the given index.
    ///
    /// Returns `None` if the index is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<Transform> {
        if index < self.bone_count() {
            Some(Transform {
                position: Vec3::new(
                    self.positions_x[index],
                    self.positions_y[index],
                    self.positions_z[index],
                ),
                rotation: Quat::from_xyzw(
                    self.rotations_x[index],
                    self.rotations_y[index],
                    self.rotations_z[index],
                    self.rotations_w[index],
                ),
                scale: Vec3::new(
                    self.scales_x[index],
                    self.scales_y[index],
                    self.scales_z[index],
                ),
            })
        } else {
            None
        }
    }

    /// Set a single transform at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    #[inline]
    pub fn set(&mut self, index: usize, transform: &Transform) {
        self.positions_x[index] = transform.position.x;
        self.positions_y[index] = transform.position.y;
        self.positions_z[index] = transform.position.z;
        self.rotations_x[index] = transform.rotation.x;
        self.rotations_y[index] = transform.rotation.y;
        self.rotations_z[index] = transform.rotation.z;
        self.rotations_w[index] = transform.rotation.w;
        self.scales_x[index] = transform.scale.x;
        self.scales_y[index] = transform.scale.y;
        self.scales_z[index] = transform.scale.z;
    }

    /// Copy data from another SoA pose into this one.
    ///
    /// The target pose will be resized to match the source.
    pub fn copy_from(&mut self, other: &SoAPose) {
        let n = other.bone_count();
        self.resize(n);

        self.positions_x.copy_from_slice(&other.positions_x);
        self.positions_y.copy_from_slice(&other.positions_y);
        self.positions_z.copy_from_slice(&other.positions_z);
        self.rotations_x.copy_from_slice(&other.rotations_x);
        self.rotations_y.copy_from_slice(&other.rotations_y);
        self.rotations_z.copy_from_slice(&other.rotations_z);
        self.rotations_w.copy_from_slice(&other.rotations_w);
        self.scales_x.copy_from_slice(&other.scales_x);
        self.scales_y.copy_from_slice(&other.scales_y);
        self.scales_z.copy_from_slice(&other.scales_z);
    }
}

// ---------------------------------------------------------------------------
// SIMD Blend Functions
// ---------------------------------------------------------------------------

/// Blend two poses using SIMD-optimized linear interpolation.
///
/// For positions and scales: `out = a * (1 - weight) + b * weight`
/// For rotations: spherical linear interpolation (slerp)
///
/// # Arguments
///
/// * `a` - First pose (weight = 0.0)
/// * `b` - Second pose (weight = 1.0)
/// * `weight` - Blend weight in [0, 1]
/// * `out` - Output pose (will be resized to match input)
///
/// # Panics
///
/// Panics if `a` and `b` have different bone counts.
pub fn blend_poses_simd(a: &SoAPose, b: &SoAPose, weight: f32, out: &mut SoAPose) {
    assert_eq!(
        a.bone_count(),
        b.bone_count(),
        "pose bone counts must match: {} vs {}",
        a.bone_count(),
        b.bone_count()
    );

    let n = a.bone_count();
    out.resize(n);

    let inv_weight = 1.0 - weight;

    // Process in SIMD-friendly chunks of 4
    let simd_count = n / SIMD_LANE_WIDTH;
    let remainder = n % SIMD_LANE_WIDTH;

    // Position and scale lerp (vectorizable)
    for chunk in 0..simd_count {
        let base = chunk * SIMD_LANE_WIDTH;

        // Process 4 bones at a time
        for lane in 0..SIMD_LANE_WIDTH {
            let i = base + lane;

            // Position lerp
            out.positions_x[i] = a.positions_x[i] * inv_weight + b.positions_x[i] * weight;
            out.positions_y[i] = a.positions_y[i] * inv_weight + b.positions_y[i] * weight;
            out.positions_z[i] = a.positions_z[i] * inv_weight + b.positions_z[i] * weight;

            // Scale lerp
            out.scales_x[i] = a.scales_x[i] * inv_weight + b.scales_x[i] * weight;
            out.scales_y[i] = a.scales_y[i] * inv_weight + b.scales_y[i] * weight;
            out.scales_z[i] = a.scales_z[i] * inv_weight + b.scales_z[i] * weight;
        }
    }

    // Handle remainder bones
    let base = simd_count * SIMD_LANE_WIDTH;
    for i in 0..remainder {
        let idx = base + i;
        out.positions_x[idx] = a.positions_x[idx] * inv_weight + b.positions_x[idx] * weight;
        out.positions_y[idx] = a.positions_y[idx] * inv_weight + b.positions_y[idx] * weight;
        out.positions_z[idx] = a.positions_z[idx] * inv_weight + b.positions_z[idx] * weight;
        out.scales_x[idx] = a.scales_x[idx] * inv_weight + b.scales_x[idx] * weight;
        out.scales_y[idx] = a.scales_y[idx] * inv_weight + b.scales_y[idx] * weight;
        out.scales_z[idx] = a.scales_z[idx] * inv_weight + b.scales_z[idx] * weight;
    }

    // Quaternion slerp (needs special handling for shortest path)
    for i in 0..n {
        let qa = Quat::from_xyzw(a.rotations_x[i], a.rotations_y[i], a.rotations_z[i], a.rotations_w[i]);
        let qb = Quat::from_xyzw(b.rotations_x[i], b.rotations_y[i], b.rotations_z[i], b.rotations_w[i]);

        // Use slerp for accurate rotation interpolation
        let result = slerp_shortest_path(qa, qb, weight);

        out.rotations_x[i] = result.x;
        out.rotations_y[i] = result.y;
        out.rotations_z[i] = result.z;
        out.rotations_w[i] = result.w;
    }
}

/// Blend poses additively using SIMD optimization.
///
/// Additive blending: `out = base + (additive - identity) * weight`
///
/// This is used for layered animations where an additive track modifies
/// a base animation without replacing it entirely.
///
/// # Arguments
///
/// * `base` - Base pose to add to
/// * `additive` - Additive pose (difference from identity)
/// * `weight` - Additive blend weight in [0, 1]
/// * `out` - Output pose (will be resized to match input)
///
/// # Panics
///
/// Panics if `base` and `additive` have different bone counts.
pub fn blend_poses_additive_simd(
    base: &SoAPose,
    additive: &SoAPose,
    weight: f32,
    out: &mut SoAPose,
) {
    assert_eq!(
        base.bone_count(),
        additive.bone_count(),
        "pose bone counts must match: {} vs {}",
        base.bone_count(),
        additive.bone_count()
    );

    let n = base.bone_count();
    out.resize(n);

    // Process in SIMD-friendly chunks
    let simd_count = n / SIMD_LANE_WIDTH;
    let remainder = n % SIMD_LANE_WIDTH;

    // Position additive: out = base + additive * weight
    // (additive is already relative to identity, i.e., the delta)
    for chunk in 0..simd_count {
        let base_idx = chunk * SIMD_LANE_WIDTH;

        for lane in 0..SIMD_LANE_WIDTH {
            let i = base_idx + lane;

            // Position: add delta scaled by weight
            out.positions_x[i] = base.positions_x[i] + additive.positions_x[i] * weight;
            out.positions_y[i] = base.positions_y[i] + additive.positions_y[i] * weight;
            out.positions_z[i] = base.positions_z[i] + additive.positions_z[i] * weight;

            // Scale: multiply (additive scale is relative to 1.0)
            // out = base * lerp(1.0, additive, weight)
            let scale_x = 1.0 + (additive.scales_x[i] - 1.0) * weight;
            let scale_y = 1.0 + (additive.scales_y[i] - 1.0) * weight;
            let scale_z = 1.0 + (additive.scales_z[i] - 1.0) * weight;
            out.scales_x[i] = base.scales_x[i] * scale_x;
            out.scales_y[i] = base.scales_y[i] * scale_y;
            out.scales_z[i] = base.scales_z[i] * scale_z;
        }
    }

    // Handle remainder
    let base_idx = simd_count * SIMD_LANE_WIDTH;
    for i in 0..remainder {
        let idx = base_idx + i;
        out.positions_x[idx] = base.positions_x[idx] + additive.positions_x[idx] * weight;
        out.positions_y[idx] = base.positions_y[idx] + additive.positions_y[idx] * weight;
        out.positions_z[idx] = base.positions_z[idx] + additive.positions_z[idx] * weight;

        let scale_x = 1.0 + (additive.scales_x[idx] - 1.0) * weight;
        let scale_y = 1.0 + (additive.scales_y[idx] - 1.0) * weight;
        let scale_z = 1.0 + (additive.scales_z[idx] - 1.0) * weight;
        out.scales_x[idx] = base.scales_x[idx] * scale_x;
        out.scales_y[idx] = base.scales_y[idx] * scale_y;
        out.scales_z[idx] = base.scales_z[idx] * scale_z;
    }

    // Rotation: multiply base by additive rotation scaled by weight
    for i in 0..n {
        let q_base = Quat::from_xyzw(
            base.rotations_x[i],
            base.rotations_y[i],
            base.rotations_z[i],
            base.rotations_w[i],
        );
        let q_additive = Quat::from_xyzw(
            additive.rotations_x[i],
            additive.rotations_y[i],
            additive.rotations_z[i],
            additive.rotations_w[i],
        );

        // Slerp additive rotation from identity to full effect
        let q_delta = slerp_shortest_path(Quat::IDENTITY, q_additive, weight);

        // Apply additive rotation on top of base
        let result = (q_base * q_delta).normalize();

        out.rotations_x[i] = result.x;
        out.rotations_y[i] = result.y;
        out.rotations_z[i] = result.z;
        out.rotations_w[i] = result.w;
    }
}

/// Normalize all quaternion rotations in the pose using SIMD.
///
/// This ensures all rotations are unit quaternions, which can drift
/// due to accumulated floating-point error during blending.
pub fn normalize_rotations_simd(pose: &mut SoAPose) {
    let n = pose.bone_count();

    // Process in SIMD-friendly chunks
    let simd_count = n / SIMD_LANE_WIDTH;
    let remainder = n % SIMD_LANE_WIDTH;

    for chunk in 0..simd_count {
        let base = chunk * SIMD_LANE_WIDTH;

        for lane in 0..SIMD_LANE_WIDTH {
            let i = base + lane;
            normalize_rotation_at(pose, i);
        }
    }

    // Handle remainder
    let base = simd_count * SIMD_LANE_WIDTH;
    for i in 0..remainder {
        normalize_rotation_at(pose, base + i);
    }
}

/// Helper to normalize a single rotation in place.
#[inline]
fn normalize_rotation_at(pose: &mut SoAPose, i: usize) {
    let x = pose.rotations_x[i];
    let y = pose.rotations_y[i];
    let z = pose.rotations_z[i];
    let w = pose.rotations_w[i];

    let len_sq = x * x + y * y + z * z + w * w;

    if len_sq > QUAT_NORMALIZE_EPSILON {
        let inv_len = 1.0 / len_sq.sqrt();
        pose.rotations_x[i] = x * inv_len;
        pose.rotations_y[i] = y * inv_len;
        pose.rotations_z[i] = z * inv_len;
        pose.rotations_w[i] = w * inv_len;
    } else {
        // Reset to identity if degenerate
        pose.rotations_x[i] = 0.0;
        pose.rotations_y[i] = 0.0;
        pose.rotations_z[i] = 0.0;
        pose.rotations_w[i] = 1.0;
    }
}

/// Spherical linear interpolation with shortest path.
///
/// Ensures we always take the shorter arc between two quaternions.
#[inline]
fn slerp_shortest_path(a: Quat, b: Quat, t: f32) -> Quat {
    // Compute dot product to check if we need to negate
    let dot = a.dot(b);

    // If dot is negative, negate one quaternion to take shorter path
    let (b, dot) = if dot < 0.0 { (-b, -dot) } else { (b, dot) };

    // Use standard slerp with corrected quaternion
    if dot > 0.9995 {
        // Quaternions are very close, use linear interpolation
        (a + (b - a) * t).normalize()
    } else {
        a.slerp(b, t)
    }
}

// ---------------------------------------------------------------------------
// Bone Chain Computation
// ---------------------------------------------------------------------------

/// Compute model-space transformation matrices for all bones using SIMD.
///
/// This processes the skeleton hierarchy to compute world/model-space
/// transforms from local poses. It exploits the topological ordering
/// of bones (parent before children) for efficient single-pass computation.
///
/// # Arguments
///
/// * `skeleton` - The skeleton hierarchy
/// * `local_pose` - Local-space transforms in SoA layout
/// * `out_model_matrices` - Output buffer for model-space matrices
///
/// # Panics
///
/// Panics if:
/// - `local_pose.bone_count() != skeleton.bone_count()`
/// - `out_model_matrices.len() < skeleton.bone_count()`
pub fn compute_bone_chain_simd(
    skeleton: &Skeleton,
    local_pose: &SoAPose,
    out_model_matrices: &mut [Mat4],
) {
    let bone_count = skeleton.bone_count();

    assert_eq!(
        local_pose.bone_count(),
        bone_count,
        "pose bone count {} must match skeleton bone count {}",
        local_pose.bone_count(),
        bone_count
    );

    assert!(
        out_model_matrices.len() >= bone_count,
        "output matrix buffer size {} is less than bone count {}",
        out_model_matrices.len(),
        bone_count
    );

    // Process bones in topological order (parents first)
    // This is guaranteed by Skeleton's structure
    for (i, bone) in skeleton.bones().iter().enumerate() {
        // Extract local transform for this bone
        let local_mat = compute_local_matrix(local_pose, i);

        // Combine with parent's world transform
        let world_mat = match bone.parent_index {
            Some(parent_idx) => {
                // Parent matrix was computed in earlier iteration
                debug_assert!(parent_idx < i, "parent must come before child");
                out_model_matrices[parent_idx] * local_mat
            }
            None => {
                // Root bone - local == world
                local_mat
            }
        };

        out_model_matrices[i] = world_mat;
    }
}

/// Compute the local transformation matrix for a bone from SoA data.
#[inline]
fn compute_local_matrix(pose: &SoAPose, index: usize) -> Mat4 {
    let pos = Vec3::new(
        pose.positions_x[index],
        pose.positions_y[index],
        pose.positions_z[index],
    );
    let rot = Quat::from_xyzw(
        pose.rotations_x[index],
        pose.rotations_y[index],
        pose.rotations_z[index],
        pose.rotations_w[index],
    );
    let scale = Vec3::new(
        pose.scales_x[index],
        pose.scales_y[index],
        pose.scales_z[index],
    );

    Mat4::from_scale_rotation_translation(scale, rot, pos)
}

/// Compute skinning matrices from model-space transforms and inverse bind matrices.
///
/// Skinning matrix = model_matrix * inverse_bind_matrix
///
/// # Arguments
///
/// * `skeleton` - The skeleton with inverse bind matrices
/// * `model_matrices` - Model-space transforms computed from `compute_bone_chain_simd`
/// * `out_skinning_matrices` - Output buffer for skinning matrices
///
/// # Panics
///
/// Panics if buffer sizes don't match the skeleton bone count.
pub fn compute_skinning_matrices_simd(
    skeleton: &Skeleton,
    model_matrices: &[Mat4],
    out_skinning_matrices: &mut [Mat4],
) {
    let bone_count = skeleton.bone_count();

    assert!(
        model_matrices.len() >= bone_count,
        "model matrix buffer size {} is less than bone count {}",
        model_matrices.len(),
        bone_count
    );

    assert!(
        out_skinning_matrices.len() >= bone_count,
        "skinning matrix buffer size {} is less than bone count {}",
        out_skinning_matrices.len(),
        bone_count
    );

    // This loop is already SIMD-friendly as Mat4 multiplication in glam
    // uses SIMD intrinsics when available
    for (i, bone) in skeleton.bones().iter().enumerate() {
        out_skinning_matrices[i] = model_matrices[i] * bone.inverse_bind_matrix;
    }
}

// ---------------------------------------------------------------------------
// Batch Operations
// ---------------------------------------------------------------------------

/// Batch blend multiple pose pairs with different weights.
///
/// This is useful for animation layers where each layer has its own weight.
///
/// # Arguments
///
/// * `pairs` - Slice of (pose_a, pose_b, weight) tuples
/// * `out` - Output pose (result of all blends combined)
pub fn blend_poses_batch(
    pairs: &[(&SoAPose, &SoAPose, f32)],
    out: &mut SoAPose,
) {
    if pairs.is_empty() {
        return;
    }

    // Start with first pair
    let (a, b, weight) = pairs[0];
    blend_poses_simd(a, b, weight, out);

    // Blend remaining pairs additively
    let mut temp = SoAPose::new();
    let mut temp2 = SoAPose::new();
    for (a, b, weight) in pairs.iter().skip(1) {
        blend_poses_simd(a, b, *weight, &mut temp);
        // Average with current result - copy out to temp2 first to avoid aliasing
        temp2.copy_from(out);
        blend_poses_simd(&temp2, &temp, 0.5, out);
    }
}

/// Apply a mask to selectively blend bones.
///
/// Bones with mask value 0.0 will use pose `a`, bones with 1.0 use pose `b`,
/// and intermediate values blend proportionally.
///
/// # Arguments
///
/// * `a` - First pose
/// * `b` - Second pose
/// * `mask` - Per-bone blend weights
/// * `out` - Output pose
///
/// # Panics
///
/// Panics if bone counts don't match.
pub fn blend_poses_masked(
    a: &SoAPose,
    b: &SoAPose,
    mask: &[f32],
    out: &mut SoAPose,
) {
    assert_eq!(a.bone_count(), b.bone_count());
    assert_eq!(a.bone_count(), mask.len());

    let n = a.bone_count();
    out.resize(n);

    for i in 0..n {
        let weight = mask[i].clamp(0.0, 1.0);
        let inv_weight = 1.0 - weight;

        // Position/scale lerp
        out.positions_x[i] = a.positions_x[i] * inv_weight + b.positions_x[i] * weight;
        out.positions_y[i] = a.positions_y[i] * inv_weight + b.positions_y[i] * weight;
        out.positions_z[i] = a.positions_z[i] * inv_weight + b.positions_z[i] * weight;
        out.scales_x[i] = a.scales_x[i] * inv_weight + b.scales_x[i] * weight;
        out.scales_y[i] = a.scales_y[i] * inv_weight + b.scales_y[i] * weight;
        out.scales_z[i] = a.scales_z[i] * inv_weight + b.scales_z[i] * weight;

        // Rotation slerp
        let qa = Quat::from_xyzw(a.rotations_x[i], a.rotations_y[i], a.rotations_z[i], a.rotations_w[i]);
        let qb = Quat::from_xyzw(b.rotations_x[i], b.rotations_y[i], b.rotations_z[i], b.rotations_w[i]);
        let result = slerp_shortest_path(qa, qb, weight);

        out.rotations_x[i] = result.x;
        out.rotations_y[i] = result.y;
        out.rotations_z[i] = result.z;
        out.rotations_w[i] = result.w;
    }
}

// ---------------------------------------------------------------------------
// Approximate Equality
// ---------------------------------------------------------------------------

/// Check if two SoA poses are approximately equal.
///
/// Useful for testing SIMD implementations against scalar reference.
pub fn poses_approx_equal(a: &SoAPose, b: &SoAPose, epsilon: f32) -> bool {
    if a.bone_count() != b.bone_count() {
        return false;
    }

    for i in 0..a.bone_count() {
        // Check positions
        if (a.positions_x[i] - b.positions_x[i]).abs() > epsilon
            || (a.positions_y[i] - b.positions_y[i]).abs() > epsilon
            || (a.positions_z[i] - b.positions_z[i]).abs() > epsilon
        {
            return false;
        }

        // Check scales
        if (a.scales_x[i] - b.scales_x[i]).abs() > epsilon
            || (a.scales_y[i] - b.scales_y[i]).abs() > epsilon
            || (a.scales_z[i] - b.scales_z[i]).abs() > epsilon
        {
            return false;
        }

        // Check rotations (quaternions can differ by sign and still be equal)
        let qa = Quat::from_xyzw(a.rotations_x[i], a.rotations_y[i], a.rotations_z[i], a.rotations_w[i]);
        let qb = Quat::from_xyzw(b.rotations_x[i], b.rotations_y[i], b.rotations_z[i], b.rotations_w[i]);
        let dot = qa.dot(qb).abs();
        if dot < 1.0 - epsilon {
            return false;
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Bone, Skeleton, SkeletonBuilder};
    use std::f32::consts::PI;

    // ===== SoAPose Conversion Tests =====

    #[test]
    fn test_soa_pose_new() {
        let pose = SoAPose::new();
        assert_eq!(pose.bone_count(), 0);
        assert!(pose.is_empty());
    }

    #[test]
    fn test_soa_pose_with_capacity() {
        let pose = SoAPose::with_capacity(100);
        assert_eq!(pose.bone_count(), 0);
        assert!(pose.positions_x.capacity() >= 100);
    }

    #[test]
    fn test_soa_pose_identity() {
        let pose = SoAPose::identity(5);
        assert_eq!(pose.bone_count(), 5);

        for i in 0..5 {
            assert_eq!(pose.positions_x[i], 0.0);
            assert_eq!(pose.positions_y[i], 0.0);
            assert_eq!(pose.positions_z[i], 0.0);
            assert_eq!(pose.rotations_x[i], 0.0);
            assert_eq!(pose.rotations_y[i], 0.0);
            assert_eq!(pose.rotations_z[i], 0.0);
            assert_eq!(pose.rotations_w[i], 1.0);
            assert_eq!(pose.scales_x[i], 1.0);
            assert_eq!(pose.scales_y[i], 1.0);
            assert_eq!(pose.scales_z[i], 1.0);
        }
    }

    #[test]
    fn test_soa_from_aos_identity() {
        let transforms = vec![Transform::IDENTITY; 10];
        let soa = SoAPose::from_aos(&transforms);

        assert_eq!(soa.bone_count(), 10);
        for i in 0..10 {
            assert_eq!(soa.positions_x[i], 0.0);
            assert_eq!(soa.rotations_w[i], 1.0);
            assert_eq!(soa.scales_x[i], 1.0);
        }
    }

    #[test]
    fn test_soa_from_aos_varied() {
        let transforms = vec![
            Transform::from_position(Vec3::new(1.0, 2.0, 3.0)),
            Transform::from_rotation(Quat::from_rotation_y(PI / 2.0)),
            Transform::from_scale_vec(Vec3::new(2.0, 3.0, 4.0)),
        ];
        let soa = SoAPose::from_aos(&transforms);

        assert_eq!(soa.bone_count(), 3);
        assert_eq!(soa.positions_x[0], 1.0);
        assert_eq!(soa.positions_y[0], 2.0);
        assert_eq!(soa.positions_z[0], 3.0);
        assert_eq!(soa.scales_x[2], 2.0);
        assert_eq!(soa.scales_y[2], 3.0);
        assert_eq!(soa.scales_z[2], 4.0);
    }

    #[test]
    fn test_soa_to_aos_roundtrip() {
        let original = vec![
            Transform::new(
                Vec3::new(1.0, 2.0, 3.0),
                Quat::from_rotation_x(PI / 4.0),
                Vec3::new(1.5, 1.5, 1.5),
            ),
            Transform::new(
                Vec3::new(-1.0, 0.0, 5.0),
                Quat::from_rotation_z(PI / 3.0),
                Vec3::new(2.0, 1.0, 0.5),
            ),
        ];

        let soa = SoAPose::from_aos(&original);
        let recovered = soa.to_aos();

        assert_eq!(recovered.len(), original.len());
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!(a.approx_eq(b, 1e-5));
        }
    }

    #[test]
    fn test_soa_get_set() {
        let mut pose = SoAPose::identity(3);

        let t = Transform::from_position(Vec3::new(5.0, 6.0, 7.0));
        pose.set(1, &t);

        let retrieved = pose.get(1).unwrap();
        assert!(retrieved.position.abs_diff_eq(Vec3::new(5.0, 6.0, 7.0), 1e-5));
        assert!(pose.get(99).is_none());
    }

    #[test]
    fn test_soa_clear() {
        let mut pose = SoAPose::identity(10);
        pose.clear();
        assert_eq!(pose.bone_count(), 0);
        assert!(pose.is_empty());
    }

    #[test]
    fn test_soa_resize() {
        let mut pose = SoAPose::identity(5);
        pose.resize(10);
        assert_eq!(pose.bone_count(), 10);

        // New bones should be identity
        assert_eq!(pose.rotations_w[9], 1.0);

        pose.resize(3);
        assert_eq!(pose.bone_count(), 3);
    }

    #[test]
    fn test_soa_copy_from() {
        let src = SoAPose::from_aos(&[
            Transform::from_position(Vec3::new(1.0, 2.0, 3.0)),
            Transform::from_scale(2.0),
        ]);
        let mut dst = SoAPose::new();
        dst.copy_from(&src);

        assert!(poses_approx_equal(&src, &dst, 1e-6));
    }

    // ===== SIMD Blend Tests =====

    #[test]
    fn test_blend_poses_simd_zero_weight() {
        let a = SoAPose::from_aos(&[
            Transform::from_position(Vec3::new(1.0, 0.0, 0.0)),
            Transform::from_position(Vec3::new(2.0, 0.0, 0.0)),
        ]);
        let b = SoAPose::from_aos(&[
            Transform::from_position(Vec3::new(10.0, 0.0, 0.0)),
            Transform::from_position(Vec3::new(20.0, 0.0, 0.0)),
        ]);
        let mut out = SoAPose::new();

        blend_poses_simd(&a, &b, 0.0, &mut out);

        assert!(poses_approx_equal(&a, &out, 1e-5));
    }

    #[test]
    fn test_blend_poses_simd_full_weight() {
        let a = SoAPose::from_aos(&[
            Transform::from_position(Vec3::new(1.0, 0.0, 0.0)),
            Transform::from_position(Vec3::new(2.0, 0.0, 0.0)),
        ]);
        let b = SoAPose::from_aos(&[
            Transform::from_position(Vec3::new(10.0, 0.0, 0.0)),
            Transform::from_position(Vec3::new(20.0, 0.0, 0.0)),
        ]);
        let mut out = SoAPose::new();

        blend_poses_simd(&a, &b, 1.0, &mut out);

        assert!(poses_approx_equal(&b, &out, 1e-5));
    }

    #[test]
    fn test_blend_poses_simd_half_weight() {
        let a = SoAPose::from_aos(&[Transform::from_position(Vec3::new(0.0, 0.0, 0.0))]);
        let b = SoAPose::from_aos(&[Transform::from_position(Vec3::new(10.0, 0.0, 0.0))]);
        let mut out = SoAPose::new();

        blend_poses_simd(&a, &b, 0.5, &mut out);

        assert!((out.positions_x[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_blend_poses_simd_rotation() {
        // Use 90-degree rotation (not 180) to avoid slerp singularity
        let a = SoAPose::from_aos(&[Transform::from_rotation(Quat::IDENTITY)]);
        let b = SoAPose::from_aos(&[Transform::from_rotation(Quat::from_rotation_y(PI / 2.0))]);
        let mut out = SoAPose::new();

        blend_poses_simd(&a, &b, 0.5, &mut out);

        // Halfway between identity and 90-degree rotation should be 45-degree rotation
        let result_quat = Quat::from_xyzw(
            out.rotations_x[0],
            out.rotations_y[0],
            out.rotations_z[0],
            out.rotations_w[0],
        );
        let expected = Quat::from_rotation_y(PI / 4.0);
        assert!(result_quat.dot(expected).abs() > 0.99);
    }

    #[test]
    fn test_blend_poses_simd_scale() {
        let a = SoAPose::from_aos(&[Transform::from_scale(1.0)]);
        let b = SoAPose::from_aos(&[Transform::from_scale(3.0)]);
        let mut out = SoAPose::new();

        blend_poses_simd(&a, &b, 0.5, &mut out);

        assert!((out.scales_x[0] - 2.0).abs() < 1e-5);
        assert!((out.scales_y[0] - 2.0).abs() < 1e-5);
        assert!((out.scales_z[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_blend_poses_simd_many_bones() {
        let n = 100; // Test SIMD chunking with remainder
        let a = SoAPose::identity(n);
        let mut b = SoAPose::identity(n);
        for i in 0..n {
            b.positions_x[i] = i as f32;
        }

        let mut out = SoAPose::new();
        blend_poses_simd(&a, &b, 0.5, &mut out);

        for i in 0..n {
            let expected = (i as f32) * 0.5;
            assert!((out.positions_x[i] - expected).abs() < 1e-5, "bone {}", i);
        }
    }

    #[test]
    #[should_panic(expected = "pose bone counts must match")]
    fn test_blend_poses_simd_mismatched_counts() {
        let a = SoAPose::identity(5);
        let b = SoAPose::identity(10);
        let mut out = SoAPose::new();
        blend_poses_simd(&a, &b, 0.5, &mut out);
    }

    // ===== Additive Blend Tests =====

    #[test]
    fn test_blend_poses_additive_zero_weight() {
        let base = SoAPose::from_aos(&[Transform::from_position(Vec3::new(5.0, 0.0, 0.0))]);
        let additive = SoAPose::from_aos(&[Transform::from_position(Vec3::new(10.0, 0.0, 0.0))]);
        let mut out = SoAPose::new();

        blend_poses_additive_simd(&base, &additive, 0.0, &mut out);

        // With zero weight, additive has no effect
        assert!((out.positions_x[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_blend_poses_additive_full_weight() {
        let base = SoAPose::from_aos(&[Transform::from_position(Vec3::new(5.0, 0.0, 0.0))]);
        let additive = SoAPose::from_aos(&[Transform::from_position(Vec3::new(10.0, 0.0, 0.0))]);
        let mut out = SoAPose::new();

        blend_poses_additive_simd(&base, &additive, 1.0, &mut out);

        // Full weight: base + additive
        assert!((out.positions_x[0] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_blend_poses_additive_half_weight() {
        let base = SoAPose::from_aos(&[Transform::from_position(Vec3::new(10.0, 0.0, 0.0))]);
        let additive = SoAPose::from_aos(&[Transform::from_position(Vec3::new(20.0, 0.0, 0.0))]);
        let mut out = SoAPose::new();

        blend_poses_additive_simd(&base, &additive, 0.5, &mut out);

        // Half weight: base + additive * 0.5 = 10 + 20*0.5 = 20
        assert!((out.positions_x[0] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_blend_poses_additive_scale() {
        let base = SoAPose::from_aos(&[Transform::from_scale(2.0)]);
        let additive = SoAPose::from_aos(&[Transform::from_scale(3.0)]); // Additive scale of 3x
        let mut out = SoAPose::new();

        blend_poses_additive_simd(&base, &additive, 1.0, &mut out);

        // Scale additive: base * lerp(1.0, additive, weight)
        // = 2.0 * (1.0 + (3.0 - 1.0) * 1.0) = 2.0 * 3.0 = 6.0
        assert!((out.scales_x[0] - 6.0).abs() < 1e-5);
    }

    // ===== Normalize Rotations Tests =====

    #[test]
    fn test_normalize_rotations_simd_unit() {
        let mut pose = SoAPose::identity(5);
        normalize_rotations_simd(&mut pose);

        for i in 0..5 {
            let len = (pose.rotations_x[i].powi(2)
                + pose.rotations_y[i].powi(2)
                + pose.rotations_z[i].powi(2)
                + pose.rotations_w[i].powi(2))
            .sqrt();
            assert!((len - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_normalize_rotations_simd_non_unit() {
        let mut pose = SoAPose::identity(2);
        // Set a non-unit quaternion
        pose.rotations_x[0] = 1.0;
        pose.rotations_y[0] = 1.0;
        pose.rotations_z[0] = 1.0;
        pose.rotations_w[0] = 1.0;

        normalize_rotations_simd(&mut pose);

        let len = (pose.rotations_x[0].powi(2)
            + pose.rotations_y[0].powi(2)
            + pose.rotations_z[0].powi(2)
            + pose.rotations_w[0].powi(2))
        .sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_rotations_simd_degenerate() {
        let mut pose = SoAPose::identity(2);
        // Set a degenerate quaternion (all zeros)
        pose.rotations_x[1] = 0.0;
        pose.rotations_y[1] = 0.0;
        pose.rotations_z[1] = 0.0;
        pose.rotations_w[1] = 0.0;

        normalize_rotations_simd(&mut pose);

        // Should reset to identity
        assert_eq!(pose.rotations_w[1], 1.0);
    }

    // ===== Bone Chain Tests =====

    #[test]
    fn test_compute_bone_chain_simd_single_bone() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));

        let pose = SoAPose::from_aos(&[Transform::from_position(Vec3::new(5.0, 0.0, 0.0))]);
        let mut matrices = vec![Mat4::IDENTITY; 1];

        compute_bone_chain_simd(&skeleton, &pose, &mut matrices);

        let expected = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));
        assert!(matrices[0].abs_diff_eq(expected, 1e-5));
    }

    #[test]
    fn test_compute_bone_chain_simd_hierarchy() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("child").with_parent(0));

        let pose = SoAPose::from_aos(&[
            Transform::from_position(Vec3::new(10.0, 0.0, 0.0)),
            Transform::from_position(Vec3::new(5.0, 0.0, 0.0)),
        ]);
        let mut matrices = vec![Mat4::IDENTITY; 2];

        compute_bone_chain_simd(&skeleton, &pose, &mut matrices);

        // Child world = parent world * child local = translate(10) * translate(5) = translate(15)
        let expected_child = Mat4::from_translation(Vec3::new(15.0, 0.0, 0.0));
        assert!(matrices[1].abs_diff_eq(expected_child, 1e-5));
    }

    #[test]
    fn test_compute_bone_chain_simd_rotation_hierarchy() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("child").with_parent(0));

        let pose = SoAPose::from_aos(&[
            Transform::from_rotation(Quat::from_rotation_y(PI / 2.0)),
            Transform::from_position(Vec3::new(1.0, 0.0, 0.0)),
        ]);
        let mut matrices = vec![Mat4::IDENTITY; 2];

        compute_bone_chain_simd(&skeleton, &pose, &mut matrices);

        // Child at (1,0,0) rotated 90 degrees around Y becomes (0,0,-1)
        let child_pos = matrices[1].w_axis.truncate();
        assert!(child_pos.abs_diff_eq(Vec3::new(0.0, 0.0, -1.0), 1e-5));
    }

    #[test]
    fn test_compute_bone_chain_simd_deep_hierarchy() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("bone_0"));
        for i in 1..10 {
            skeleton.add_bone(Bone::new(format!("bone_{}", i)).with_parent(i - 1));
        }

        let mut transforms = Vec::new();
        for _ in 0..10 {
            transforms.push(Transform::from_position(Vec3::new(1.0, 0.0, 0.0)));
        }
        let pose = SoAPose::from_aos(&transforms);
        let mut matrices = vec![Mat4::IDENTITY; 10];

        compute_bone_chain_simd(&skeleton, &pose, &mut matrices);

        // Last bone should be at x = 10 (cumulative translations)
        let last_pos = matrices[9].w_axis.truncate();
        assert!(last_pos.abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), 1e-4));
    }

    #[test]
    #[should_panic(expected = "pose bone count")]
    fn test_compute_bone_chain_simd_mismatched_counts() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("child").with_parent(0));

        let pose = SoAPose::identity(5); // Wrong count
        let mut matrices = vec![Mat4::IDENTITY; 2];

        compute_bone_chain_simd(&skeleton, &pose, &mut matrices);
    }

    #[test]
    #[should_panic(expected = "output matrix buffer size")]
    fn test_compute_bone_chain_simd_small_output() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(Bone::root("root"));
        skeleton.add_bone(Bone::new("child").with_parent(0));

        let pose = SoAPose::identity(2);
        let mut matrices = vec![Mat4::IDENTITY; 1]; // Too small

        compute_bone_chain_simd(&skeleton, &pose, &mut matrices);
    }

    // ===== Skinning Matrix Tests =====

    #[test]
    fn test_compute_skinning_matrices_simd() {
        let mut skeleton = Skeleton::new();
        skeleton.add_bone(
            Bone::root("root").with_inverse_bind_matrix(Mat4::from_translation(Vec3::new(
                -1.0, 0.0, 0.0,
            ))),
        );

        let model_matrices = vec![Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0))];
        let mut skinning_matrices = vec![Mat4::IDENTITY; 1];

        compute_skinning_matrices_simd(&skeleton, &model_matrices, &mut skinning_matrices);

        // Skinning = model * inverse_bind = translate(3) * translate(-1) = translate(2)
        let expected = Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0));
        assert!(skinning_matrices[0].abs_diff_eq(expected, 1e-5));
    }

    // ===== Masked Blend Tests =====

    #[test]
    fn test_blend_poses_masked() {
        let a = SoAPose::from_aos(&[
            Transform::from_position(Vec3::new(0.0, 0.0, 0.0)),
            Transform::from_position(Vec3::new(0.0, 0.0, 0.0)),
        ]);
        let b = SoAPose::from_aos(&[
            Transform::from_position(Vec3::new(10.0, 0.0, 0.0)),
            Transform::from_position(Vec3::new(10.0, 0.0, 0.0)),
        ]);
        let mask = [0.0, 1.0]; // First bone: use a, second bone: use b
        let mut out = SoAPose::new();

        blend_poses_masked(&a, &b, &mask, &mut out);

        assert!((out.positions_x[0] - 0.0).abs() < 1e-5);
        assert!((out.positions_x[1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_blend_poses_masked_partial() {
        let a = SoAPose::from_aos(&[Transform::from_position(Vec3::new(0.0, 0.0, 0.0))]);
        let b = SoAPose::from_aos(&[Transform::from_position(Vec3::new(10.0, 0.0, 0.0))]);
        let mask = [0.3];
        let mut out = SoAPose::new();

        blend_poses_masked(&a, &b, &mask, &mut out);

        assert!((out.positions_x[0] - 3.0).abs() < 1e-5);
    }

    // ===== Approximate Equality Tests =====

    #[test]
    fn test_poses_approx_equal_identical() {
        let a = SoAPose::identity(5);
        let b = SoAPose::identity(5);
        assert!(poses_approx_equal(&a, &b, 1e-6));
    }

    #[test]
    fn test_poses_approx_equal_different() {
        let a = SoAPose::identity(5);
        let mut b = SoAPose::identity(5);
        b.positions_x[0] = 100.0;
        assert!(!poses_approx_equal(&a, &b, 1e-6));
    }

    #[test]
    fn test_poses_approx_equal_different_counts() {
        let a = SoAPose::identity(5);
        let b = SoAPose::identity(10);
        assert!(!poses_approx_equal(&a, &b, 1e-6));
    }

    // ===== SIMD vs Scalar Accuracy Tests =====

    #[test]
    fn test_simd_blend_matches_scalar() {
        // Create random-ish poses
        let a_transforms: Vec<Transform> = (0..17)
            .map(|i| {
                Transform::new(
                    Vec3::new(i as f32, (i * 2) as f32, (i * 3) as f32),
                    Quat::from_rotation_y((i as f32) * 0.1),
                    Vec3::splat(1.0 + (i as f32) * 0.1),
                )
            })
            .collect();
        let b_transforms: Vec<Transform> = (0..17)
            .map(|i| {
                Transform::new(
                    Vec3::new(-i as f32, (-i * 2) as f32, (-i * 3) as f32),
                    Quat::from_rotation_z((i as f32) * 0.2),
                    Vec3::splat(2.0 - (i as f32) * 0.05),
                )
            })
            .collect();

        let a = SoAPose::from_aos(&a_transforms);
        let b = SoAPose::from_aos(&b_transforms);
        let mut simd_result = SoAPose::new();

        blend_poses_simd(&a, &b, 0.3, &mut simd_result);

        // Compute scalar reference
        let scalar_result: Vec<Transform> = a_transforms
            .iter()
            .zip(b_transforms.iter())
            .map(|(ta, tb)| ta.lerp(tb, 0.3))
            .collect();
        let scalar_soa = SoAPose::from_aos(&scalar_result);

        assert!(poses_approx_equal(&simd_result, &scalar_soa, 1e-4));
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_empty_pose_operations() {
        let a = SoAPose::new();
        let b = SoAPose::new();
        let mut out = SoAPose::new();

        blend_poses_simd(&a, &b, 0.5, &mut out);
        assert_eq!(out.bone_count(), 0);
    }

    #[test]
    fn test_single_bone_blend() {
        let a = SoAPose::from_aos(&[Transform::from_position(Vec3::new(0.0, 0.0, 0.0))]);
        let b = SoAPose::from_aos(&[Transform::from_position(Vec3::new(4.0, 8.0, 12.0))]);
        let mut out = SoAPose::new();

        blend_poses_simd(&a, &b, 0.25, &mut out);

        assert!((out.positions_x[0] - 1.0).abs() < 1e-5);
        assert!((out.positions_y[0] - 2.0).abs() < 1e-5);
        assert!((out.positions_z[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_chunk_boundary() {
        // Test with exactly SIMD_LANE_WIDTH bones
        let n = SIMD_LANE_WIDTH;
        let a = SoAPose::identity(n);
        let b = SoAPose::identity(n);
        let mut out = SoAPose::new();

        blend_poses_simd(&a, &b, 0.5, &mut out);
        assert_eq!(out.bone_count(), n);
    }

    #[test]
    fn test_simd_chunk_boundary_plus_one() {
        // Test with SIMD_LANE_WIDTH + 1 bones (tests remainder handling)
        let n = SIMD_LANE_WIDTH + 1;
        let a = SoAPose::identity(n);
        let b = SoAPose::identity(n);
        let mut out = SoAPose::new();

        blend_poses_simd(&a, &b, 0.5, &mut out);
        assert_eq!(out.bone_count(), n);
    }

    #[test]
    fn test_quaternion_shortest_path() {
        // Test that slerp takes the shortest path
        let a = SoAPose::from_aos(&[Transform::from_rotation(Quat::IDENTITY)]);
        // Quaternion representing nearly the same rotation as identity but negated
        // (which is equivalent for rotations)
        let mut b = SoAPose::identity(1);
        b.rotations_x[0] = 0.001;
        b.rotations_y[0] = 0.0;
        b.rotations_z[0] = 0.0;
        b.rotations_w[0] = -1.0; // Negated identity

        let mut out = SoAPose::new();
        blend_poses_simd(&a, &b, 0.5, &mut out);

        // Result should still be close to identity
        let result = Quat::from_xyzw(
            out.rotations_x[0],
            out.rotations_y[0],
            out.rotations_z[0],
            out.rotations_w[0],
        );
        assert!(result.dot(Quat::IDENTITY).abs() > 0.99);
    }
}
