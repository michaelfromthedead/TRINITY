//! Pose representation for skeletal animation in TRINITY Engine (T-AN-1.2).
//!
//! This module provides pose types for animation blending and playback:
//!
//! - `PoseType` enum for different pose semantics
//! - `Pose` struct with SoA (Structure of Arrays) storage for SIMD-friendly blending
//! - `PoseBuffer` for animation playback with precomputed world transforms
//! - Blend utilities: lerp, slerp, nlerp for position/rotation/scale interpolation
//!
//! # Memory Layout
//!
//! Poses use SoA layout for cache-friendly SIMD operations:
//! - `positions: Vec<Vec3>` - one position per bone
//! - `rotations: Vec<Quat>` - one rotation per bone
//! - `scales: Vec<Vec3>` - one scale per bone
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::pose::{Pose, PoseType, PoseBuffer};
//! use renderer_backend::skeleton::{Skeleton, Transform};
//! use glam::Vec3;
//!
//! // Create a pose from a skeleton
//! let skeleton = create_skeleton(); // hypothetical
//! let bind_pose = Pose::from_skeleton(&skeleton, PoseType::Bind);
//!
//! // Create a current pose and modify it
//! let mut current = Pose::from_skeleton(&skeleton, PoseType::Current);
//! let transform = Transform::from_position(Vec3::new(0.0, 1.0, 0.0));
//! current.set_transform(0, transform);
//!
//! // Blend between poses
//! let blended = bind_pose.blend(&current, 0.5);
//!
//! // Create a pose buffer for playback
//! let buffer = PoseBuffer::new(skeleton.bone_count());
//! ```

use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::skeleton::{Skeleton, Transform};

// ---------------------------------------------------------------------------
// Blend Utilities
// ---------------------------------------------------------------------------

/// Linearly interpolate between two Vec3 values.
///
/// # Arguments
///
/// * `a` - Start value
/// * `b` - End value
/// * `t` - Interpolation factor (0.0 = a, 1.0 = b)
///
/// # Returns
///
/// Interpolated Vec3 value.
#[inline]
pub fn lerp_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

/// Spherical linear interpolation between two quaternions (shortest path).
///
/// This uses true slerp which maintains constant angular velocity.
/// For most animation blending, consider `nlerp_quat` which is faster.
///
/// # Arguments
///
/// * `a` - Start rotation (should be normalized)
/// * `b` - End rotation (should be normalized)
/// * `t` - Interpolation factor (0.0 = a, 1.0 = b)
///
/// # Returns
///
/// Interpolated quaternion (normalized).
#[inline]
pub fn slerp_quat(a: Quat, b: Quat, t: f32) -> Quat {
    // Ensure shortest path by checking dot product
    let dot = a.dot(b);
    let b_adjusted = if dot < 0.0 { -b } else { b };
    a.slerp(b_adjusted, t)
}

/// Normalized linear interpolation between two quaternions (fast approximate).
///
/// This is faster than slerp and sufficient for most animation blending
/// where the rotation difference is small. The result is normalized.
///
/// # Arguments
///
/// * `a` - Start rotation (should be normalized)
/// * `b` - End rotation (should be normalized)
/// * `t` - Interpolation factor (0.0 = a, 1.0 = b)
///
/// # Returns
///
/// Interpolated quaternion (normalized).
#[inline]
pub fn nlerp_quat(a: Quat, b: Quat, t: f32) -> Quat {
    // Ensure shortest path by checking dot product
    let dot = a.dot(b);
    let b_adjusted = if dot < 0.0 { -b } else { b };

    // Linear interpolation then normalize
    let result = Quat::from_xyzw(
        a.x + (b_adjusted.x - a.x) * t,
        a.y + (b_adjusted.y - a.y) * t,
        a.z + (b_adjusted.z - a.z) * t,
        a.w + (b_adjusted.w - a.w) * t,
    );

    result.normalize()
}

// ---------------------------------------------------------------------------
// PoseType
// ---------------------------------------------------------------------------

/// Type of pose, indicating its semantic meaning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PoseType {
    /// Bind pose: the default skeleton pose used for skinning reference.
    /// Vertices are transformed relative to this pose.
    Bind,

    /// Reference pose: a canonical pose for motion matching or retargeting.
    /// Often a T-pose or A-pose.
    Reference,

    /// Current pose: the animated pose at the current frame.
    /// This is what gets rendered.
    #[default]
    Current,

    /// Additive pose: a delta pose to be added to another pose.
    /// Positions/rotations are relative differences, not absolute values.
    Additive,
}

impl PoseType {
    /// Check if this is an additive pose type.
    #[inline]
    pub fn is_additive(&self) -> bool {
        matches!(self, PoseType::Additive)
    }

    /// Get a human-readable name for this pose type.
    pub fn name(&self) -> &'static str {
        match self {
            PoseType::Bind => "Bind",
            PoseType::Reference => "Reference",
            PoseType::Current => "Current",
            PoseType::Additive => "Additive",
        }
    }
}

// ---------------------------------------------------------------------------
// Pose
// ---------------------------------------------------------------------------

/// A skeletal pose using Structure of Arrays (SoA) layout.
///
/// Stores per-bone transforms as separate arrays for SIMD-friendly operations.
/// All arrays must have the same length (one element per bone).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    /// The semantic type of this pose.
    pub pose_type: PoseType,

    /// Position for each bone in local space.
    pub positions: Vec<Vec3>,

    /// Rotation for each bone in local space (unit quaternions).
    pub rotations: Vec<Quat>,

    /// Scale for each bone in local space.
    pub scales: Vec<Vec3>,
}

impl Pose {
    /// Create a new pose with the given bone count and pose type.
    ///
    /// For additive poses, positions and scales are zero, rotations are identity.
    /// For other poses, positions are zero, rotations are identity, scales are one.
    ///
    /// # Arguments
    ///
    /// * `bone_count` - Number of bones in the pose
    /// * `pose_type` - Semantic type of the pose
    pub fn new(bone_count: usize, pose_type: PoseType) -> Self {
        if pose_type.is_additive() {
            // Additive poses: zero/identity deltas
            Self {
                pose_type,
                positions: vec![Vec3::ZERO; bone_count],
                rotations: vec![Quat::IDENTITY; bone_count],
                scales: vec![Vec3::ZERO; bone_count], // Additive scales are deltas
            }
        } else {
            // Regular poses: identity transforms
            Self {
                pose_type,
                positions: vec![Vec3::ZERO; bone_count],
                rotations: vec![Quat::IDENTITY; bone_count],
                scales: vec![Vec3::ONE; bone_count],
            }
        }
    }

    /// Create a pose from a skeleton's bind pose.
    ///
    /// Copies the local transforms from each bone in the skeleton.
    ///
    /// # Arguments
    ///
    /// * `skeleton` - The skeleton to copy transforms from
    /// * `pose_type` - Semantic type for the new pose
    pub fn from_skeleton(skeleton: &Skeleton, pose_type: PoseType) -> Self {
        let bone_count = skeleton.bone_count();
        let mut positions = Vec::with_capacity(bone_count);
        let mut rotations = Vec::with_capacity(bone_count);
        let mut scales = Vec::with_capacity(bone_count);

        for bone in skeleton.bones() {
            positions.push(bone.local_transform.position);
            rotations.push(bone.local_transform.rotation);
            scales.push(bone.local_transform.scale);
        }

        Self {
            pose_type,
            positions,
            rotations,
            scales,
        }
    }

    /// Get the number of bones in this pose.
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.positions.len()
    }

    /// Check if this pose is empty (no bones).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Get the transform for a specific bone.
    ///
    /// # Arguments
    ///
    /// * `bone_index` - Index of the bone
    ///
    /// # Panics
    ///
    /// Panics if `bone_index` is out of bounds.
    #[inline]
    pub fn get_transform(&self, bone_index: usize) -> Transform {
        Transform {
            position: self.positions[bone_index],
            rotation: self.rotations[bone_index],
            scale: self.scales[bone_index],
        }
    }

    /// Try to get the transform for a specific bone.
    ///
    /// Returns `None` if the bone index is out of bounds.
    #[inline]
    pub fn try_get_transform(&self, bone_index: usize) -> Option<Transform> {
        if bone_index < self.bone_count() {
            Some(self.get_transform(bone_index))
        } else {
            None
        }
    }

    /// Set the transform for a specific bone.
    ///
    /// # Arguments
    ///
    /// * `bone_index` - Index of the bone
    /// * `transform` - The new transform
    ///
    /// # Panics
    ///
    /// Panics if `bone_index` is out of bounds.
    #[inline]
    pub fn set_transform(&mut self, bone_index: usize, transform: Transform) {
        self.positions[bone_index] = transform.position;
        self.rotations[bone_index] = transform.rotation;
        self.scales[bone_index] = transform.scale;
    }

    /// Try to set the transform for a specific bone.
    ///
    /// Returns `false` if the bone index is out of bounds.
    #[inline]
    pub fn try_set_transform(&mut self, bone_index: usize, transform: Transform) -> bool {
        if bone_index < self.bone_count() {
            self.set_transform(bone_index, transform);
            true
        } else {
            false
        }
    }

    /// Blend this pose with another pose.
    ///
    /// Uses linear interpolation for positions and scales,
    /// spherical interpolation (slerp) for rotations.
    ///
    /// # Arguments
    ///
    /// * `other` - The pose to blend towards
    /// * `weight` - Blend weight (0.0 = self, 1.0 = other)
    ///
    /// # Returns
    ///
    /// A new pose with blended transforms.
    ///
    /// # Panics
    ///
    /// Panics if the poses have different bone counts.
    pub fn blend(&self, other: &Pose, weight: f32) -> Pose {
        assert_eq!(
            self.bone_count(),
            other.bone_count(),
            "cannot blend poses with different bone counts: {} vs {}",
            self.bone_count(),
            other.bone_count()
        );

        let bone_count = self.bone_count();
        let mut positions = Vec::with_capacity(bone_count);
        let mut rotations = Vec::with_capacity(bone_count);
        let mut scales = Vec::with_capacity(bone_count);

        // Clamp weight to valid range
        let t = weight.clamp(0.0, 1.0);

        for i in 0..bone_count {
            positions.push(lerp_vec3(self.positions[i], other.positions[i], t));
            rotations.push(slerp_quat(self.rotations[i], other.rotations[i], t));
            scales.push(lerp_vec3(self.scales[i], other.scales[i], t));
        }

        Pose {
            pose_type: PoseType::Current, // Blended poses become "current"
            positions,
            rotations,
            scales,
        }
    }

    /// Blend this pose with another using fast nlerp for rotations.
    ///
    /// Faster than `blend()` but with slightly less accurate rotation interpolation.
    /// Good for real-time animation where the rotation difference is small.
    ///
    /// # Arguments
    ///
    /// * `other` - The pose to blend towards
    /// * `weight` - Blend weight (0.0 = self, 1.0 = other)
    ///
    /// # Returns
    ///
    /// A new pose with blended transforms.
    pub fn blend_fast(&self, other: &Pose, weight: f32) -> Pose {
        assert_eq!(
            self.bone_count(),
            other.bone_count(),
            "cannot blend poses with different bone counts"
        );

        let bone_count = self.bone_count();
        let mut positions = Vec::with_capacity(bone_count);
        let mut rotations = Vec::with_capacity(bone_count);
        let mut scales = Vec::with_capacity(bone_count);

        let t = weight.clamp(0.0, 1.0);

        for i in 0..bone_count {
            positions.push(lerp_vec3(self.positions[i], other.positions[i], t));
            rotations.push(nlerp_quat(self.rotations[i], other.rotations[i], t));
            scales.push(lerp_vec3(self.scales[i], other.scales[i], t));
        }

        Pose {
            pose_type: PoseType::Current,
            positions,
            rotations,
            scales,
        }
    }

    /// Apply an additive pose to this pose.
    ///
    /// Additive blending:
    /// - Positions: self.position + additive.position * weight
    /// - Rotations: self.rotation * slerp(identity, additive.rotation, weight)
    /// - Scales: self.scale + additive.scale * weight
    ///
    /// # Arguments
    ///
    /// * `additive` - The additive pose to apply
    /// * `weight` - Blend weight (0.0 = no effect, 1.0 = full effect)
    ///
    /// # Returns
    ///
    /// A new pose with the additive applied.
    ///
    /// # Panics
    ///
    /// Panics if the poses have different bone counts.
    pub fn blend_additive(&self, additive: &Pose, weight: f32) -> Pose {
        assert_eq!(
            self.bone_count(),
            additive.bone_count(),
            "cannot blend additive pose with different bone count"
        );

        let bone_count = self.bone_count();
        let mut positions = Vec::with_capacity(bone_count);
        let mut rotations = Vec::with_capacity(bone_count);
        let mut scales = Vec::with_capacity(bone_count);

        // Clamp weight to valid range (can be negative for inverse additive)
        let t = weight.clamp(-1.0, 1.0);

        for i in 0..bone_count {
            // Position: add weighted delta
            positions.push(self.positions[i] + additive.positions[i] * t);

            // Rotation: multiply by weighted rotation
            let additive_rot = slerp_quat(Quat::IDENTITY, additive.rotations[i], t.abs());
            let additive_rot = if t < 0.0 { additive_rot.inverse() } else { additive_rot };
            rotations.push((self.rotations[i] * additive_rot).normalize());

            // Scale: add weighted delta
            scales.push(self.scales[i] + additive.scales[i] * t);
        }

        Pose {
            pose_type: PoseType::Current,
            positions,
            rotations,
            scales,
        }
    }

    /// Copy transforms from another pose into this pose.
    ///
    /// # Arguments
    ///
    /// * `other` - The pose to copy from
    ///
    /// # Panics
    ///
    /// Panics if the poses have different bone counts.
    pub fn copy_from(&mut self, other: &Pose) {
        assert_eq!(
            self.bone_count(),
            other.bone_count(),
            "cannot copy from pose with different bone count"
        );

        self.positions.copy_from_slice(&other.positions);
        self.rotations.copy_from_slice(&other.rotations);
        self.scales.copy_from_slice(&other.scales);
    }

    /// Reset all transforms to identity.
    ///
    /// For additive poses: positions and scales become zero, rotations become identity.
    /// For other poses: positions become zero, rotations become identity, scales become one.
    pub fn reset_to_identity(&mut self) {
        if self.pose_type.is_additive() {
            for pos in &mut self.positions {
                *pos = Vec3::ZERO;
            }
            for rot in &mut self.rotations {
                *rot = Quat::IDENTITY;
            }
            for scale in &mut self.scales {
                *scale = Vec3::ZERO;
            }
        } else {
            for pos in &mut self.positions {
                *pos = Vec3::ZERO;
            }
            for rot in &mut self.rotations {
                *rot = Quat::IDENTITY;
            }
            for scale in &mut self.scales {
                *scale = Vec3::ONE;
            }
        }
    }

    /// Get transforms as a slice for iteration.
    ///
    /// This creates a temporary vector, so use `get_transform` for single access.
    pub fn transforms(&self) -> Vec<Transform> {
        (0..self.bone_count())
            .map(|i| self.get_transform(i))
            .collect()
    }

    /// Resize the pose to a new bone count.
    ///
    /// New bones are initialized to identity transforms.
    pub fn resize(&mut self, new_bone_count: usize) {
        let default_scale = if self.pose_type.is_additive() {
            Vec3::ZERO
        } else {
            Vec3::ONE
        };

        self.positions.resize(new_bone_count, Vec3::ZERO);
        self.rotations.resize(new_bone_count, Quat::IDENTITY);
        self.scales.resize(new_bone_count, default_scale);
    }

    /// Check if all transforms are approximately identity.
    pub fn is_identity(&self, epsilon: f32) -> bool {
        let expected_scale = if self.pose_type.is_additive() {
            Vec3::ZERO
        } else {
            Vec3::ONE
        };

        for i in 0..self.bone_count() {
            if !self.positions[i].abs_diff_eq(Vec3::ZERO, epsilon) {
                return false;
            }
            if self.rotations[i].dot(Quat::IDENTITY).abs() < 1.0 - epsilon {
                return false;
            }
            if !self.scales[i].abs_diff_eq(expected_scale, epsilon) {
                return false;
            }
        }
        true
    }
}

impl Default for Pose {
    fn default() -> Self {
        Self::new(0, PoseType::Current)
    }
}

// ---------------------------------------------------------------------------
// PoseBuffer
// ---------------------------------------------------------------------------

/// Buffer for animation playback with precomputed world transforms.
///
/// Stores both the local pose and the computed model-space (world) transforms.
/// The model_pose matrices are ready for GPU upload.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseBuffer {
    /// Local-space pose (per-bone transforms relative to parent).
    pub local_pose: Pose,

    /// Model-space (world) transforms for each bone.
    /// These are computed from the local pose and skeleton hierarchy.
    pub model_pose: Vec<Mat4>,
}

impl PoseBuffer {
    /// Create a new pose buffer with the given bone count.
    ///
    /// Both local_pose and model_pose are initialized to identity.
    pub fn new(bone_count: usize) -> Self {
        Self {
            local_pose: Pose::new(bone_count, PoseType::Current),
            model_pose: vec![Mat4::IDENTITY; bone_count],
        }
    }

    /// Create a pose buffer from a skeleton's bind pose.
    ///
    /// Local pose is copied from skeleton, model pose is computed.
    pub fn from_skeleton(skeleton: &Skeleton) -> Self {
        let local_pose = Pose::from_skeleton(skeleton, PoseType::Current);
        let model_pose = skeleton.compute_world_transforms(&local_pose.transforms());

        Self {
            local_pose,
            model_pose,
        }
    }

    /// Get the number of bones in this buffer.
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.local_pose.bone_count()
    }

    /// Update model-space transforms from the local pose and skeleton hierarchy.
    ///
    /// Call this after modifying the local_pose to recompute world transforms.
    pub fn update_model_pose(&mut self, skeleton: &Skeleton) {
        assert_eq!(
            self.bone_count(),
            skeleton.bone_count(),
            "pose buffer bone count must match skeleton"
        );

        let transforms = self.local_pose.transforms();
        let world = skeleton.compute_world_transforms(&transforms);
        self.model_pose.copy_from_slice(&world);
    }

    /// Copy another pose into the local pose and recompute model transforms.
    pub fn set_pose(&mut self, pose: &Pose, skeleton: &Skeleton) {
        self.local_pose.copy_from(pose);
        self.update_model_pose(skeleton);
    }

    /// Get the model-space transform for a bone.
    #[inline]
    pub fn get_model_transform(&self, bone_index: usize) -> Mat4 {
        self.model_pose[bone_index]
    }

    /// Resize the buffer to a new bone count.
    pub fn resize(&mut self, new_bone_count: usize) {
        self.local_pose.resize(new_bone_count);
        self.model_pose.resize(new_bone_count, Mat4::IDENTITY);
    }

    /// Reset to identity pose.
    pub fn reset(&mut self) {
        self.local_pose.reset_to_identity();
        for mat in &mut self.model_pose {
            *mat = Mat4::IDENTITY;
        }
    }
}

impl Default for PoseBuffer {
    fn default() -> Self {
        Self::new(0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Bone, SkeletonBuilder};
    use std::f32::consts::PI;

    // ===== Blend Utility Tests =====

    #[test]
    fn test_lerp_vec3_zero() {
        let a = Vec3::ZERO;
        let b = Vec3::new(10.0, 20.0, 30.0);

        assert!(lerp_vec3(a, b, 0.0).abs_diff_eq(a, 1e-6));
        assert!(lerp_vec3(a, b, 1.0).abs_diff_eq(b, 1e-6));
        assert!(lerp_vec3(a, b, 0.5).abs_diff_eq(Vec3::new(5.0, 10.0, 15.0), 1e-6));
    }

    #[test]
    fn test_lerp_vec3_negative() {
        let a = Vec3::new(-5.0, -5.0, -5.0);
        let b = Vec3::new(5.0, 5.0, 5.0);

        let mid = lerp_vec3(a, b, 0.5);
        assert!(mid.abs_diff_eq(Vec3::ZERO, 1e-6));
    }

    #[test]
    fn test_lerp_vec3_extrapolate() {
        let a = Vec3::ZERO;
        let b = Vec3::new(10.0, 0.0, 0.0);

        // t > 1 extrapolates
        let result = lerp_vec3(a, b, 2.0);
        assert!(result.abs_diff_eq(Vec3::new(20.0, 0.0, 0.0), 1e-6));

        // t < 0 extrapolates backward
        let result = lerp_vec3(a, b, -0.5);
        assert!(result.abs_diff_eq(Vec3::new(-5.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_slerp_quat_identity() {
        let a = Quat::IDENTITY;
        let b = Quat::from_rotation_y(PI / 2.0);

        let result = slerp_quat(a, b, 0.0);
        assert!(result.abs_diff_eq(a, 1e-5));

        let result = slerp_quat(a, b, 1.0);
        assert!(result.abs_diff_eq(b, 1e-5));
    }

    #[test]
    fn test_slerp_quat_midpoint() {
        let a = Quat::IDENTITY;
        let b = Quat::from_rotation_y(PI / 2.0);

        let mid = slerp_quat(a, b, 0.5);
        let expected = Quat::from_rotation_y(PI / 4.0);
        assert!(mid.abs_diff_eq(expected, 1e-5));
    }

    #[test]
    fn test_slerp_quat_shortest_path() {
        // Two quaternions that represent the same rotation but are antipodal
        let a = Quat::IDENTITY;
        let b = Quat::from_rotation_y(PI * 1.5); // 270 degrees = -90 degrees

        // Should take the short path
        let result = slerp_quat(a, b, 0.5);
        // The short path from 0 to -90 is -45 degrees
        assert!(result.is_normalized());
    }

    #[test]
    fn test_nlerp_quat_identity() {
        let a = Quat::IDENTITY;
        let b = Quat::from_rotation_y(PI / 2.0);

        let result = nlerp_quat(a, b, 0.0);
        assert!(result.abs_diff_eq(a, 1e-5));

        let result = nlerp_quat(a, b, 1.0);
        assert!(result.abs_diff_eq(b, 1e-5));
    }

    #[test]
    fn test_nlerp_quat_normalized() {
        let a = Quat::IDENTITY;
        let b = Quat::from_rotation_y(PI / 2.0);

        // Result should always be normalized
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let result = nlerp_quat(a, b, t);
            assert!((result.length() - 1.0).abs() < 1e-5, "nlerp result not normalized at t={}", t);
        }
    }

    #[test]
    fn test_nlerp_quat_shortest_path() {
        let a = Quat::IDENTITY;
        let b = -Quat::IDENTITY; // Antipodal, same rotation

        // Should not flip around the sphere
        let result = nlerp_quat(a, b, 0.5);
        assert!(result.abs_diff_eq(Quat::IDENTITY, 1e-5) || result.abs_diff_eq(-Quat::IDENTITY, 1e-5));
    }

    // ===== PoseType Tests =====

    #[test]
    fn test_pose_type_default() {
        assert_eq!(PoseType::default(), PoseType::Current);
    }

    #[test]
    fn test_pose_type_is_additive() {
        assert!(!PoseType::Bind.is_additive());
        assert!(!PoseType::Reference.is_additive());
        assert!(!PoseType::Current.is_additive());
        assert!(PoseType::Additive.is_additive());
    }

    #[test]
    fn test_pose_type_name() {
        assert_eq!(PoseType::Bind.name(), "Bind");
        assert_eq!(PoseType::Reference.name(), "Reference");
        assert_eq!(PoseType::Current.name(), "Current");
        assert_eq!(PoseType::Additive.name(), "Additive");
    }

    #[test]
    fn test_pose_type_equality() {
        assert_eq!(PoseType::Bind, PoseType::Bind);
        assert_ne!(PoseType::Bind, PoseType::Current);
    }

    // ===== Pose Creation Tests =====

    #[test]
    fn test_pose_new_empty() {
        let pose = Pose::new(0, PoseType::Current);
        assert_eq!(pose.bone_count(), 0);
        assert!(pose.is_empty());
    }

    #[test]
    fn test_pose_new_current() {
        let pose = Pose::new(5, PoseType::Current);
        assert_eq!(pose.bone_count(), 5);
        assert_eq!(pose.pose_type, PoseType::Current);

        // All should be identity
        for i in 0..5 {
            let t = pose.get_transform(i);
            assert!(t.position.abs_diff_eq(Vec3::ZERO, 1e-6));
            assert!(t.rotation.abs_diff_eq(Quat::IDENTITY, 1e-6));
            assert!(t.scale.abs_diff_eq(Vec3::ONE, 1e-6));
        }
    }

    #[test]
    fn test_pose_new_additive() {
        let pose = Pose::new(3, PoseType::Additive);
        assert_eq!(pose.bone_count(), 3);
        assert_eq!(pose.pose_type, PoseType::Additive);

        // Additive poses have zero scales (deltas)
        for i in 0..3 {
            let t = pose.get_transform(i);
            assert!(t.position.abs_diff_eq(Vec3::ZERO, 1e-6));
            assert!(t.rotation.abs_diff_eq(Quat::IDENTITY, 1e-6));
            assert!(t.scale.abs_diff_eq(Vec3::ZERO, 1e-6)); // Zero delta
        }
    }

    #[test]
    fn test_pose_from_skeleton() {
        let skeleton = SkeletonBuilder::new()
            .root("root")
            .child_at("arm", "root", Vec3::new(1.0, 0.0, 0.0))
            .build_unchecked();

        let pose = Pose::from_skeleton(&skeleton, PoseType::Bind);

        assert_eq!(pose.bone_count(), 2);
        assert_eq!(pose.pose_type, PoseType::Bind);

        // Check that transforms were copied
        let root_t = pose.get_transform(0);
        assert!(root_t.position.abs_diff_eq(Vec3::ZERO, 1e-6));

        let arm_t = pose.get_transform(1);
        assert!(arm_t.position.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_pose_default() {
        let pose = Pose::default();
        assert_eq!(pose.bone_count(), 0);
        assert_eq!(pose.pose_type, PoseType::Current);
    }

    // ===== Pose Transform Access Tests =====

    #[test]
    fn test_pose_get_transform() {
        let mut pose = Pose::new(2, PoseType::Current);
        pose.positions[0] = Vec3::new(1.0, 2.0, 3.0);
        pose.rotations[0] = Quat::from_rotation_y(PI / 4.0);
        pose.scales[0] = Vec3::new(2.0, 2.0, 2.0);

        let t = pose.get_transform(0);
        assert!(t.position.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-6));
        assert!(t.rotation.abs_diff_eq(Quat::from_rotation_y(PI / 4.0), 1e-5));
        assert!(t.scale.abs_diff_eq(Vec3::new(2.0, 2.0, 2.0), 1e-6));
    }

    #[test]
    fn test_pose_try_get_transform() {
        let pose = Pose::new(2, PoseType::Current);

        assert!(pose.try_get_transform(0).is_some());
        assert!(pose.try_get_transform(1).is_some());
        assert!(pose.try_get_transform(2).is_none());
        assert!(pose.try_get_transform(100).is_none());
    }

    #[test]
    fn test_pose_set_transform() {
        let mut pose = Pose::new(2, PoseType::Current);

        let t = Transform::new(
            Vec3::new(5.0, 6.0, 7.0),
            Quat::from_rotation_z(PI / 3.0),
            Vec3::new(1.5, 1.5, 1.5),
        );

        pose.set_transform(1, t);

        assert!(pose.positions[1].abs_diff_eq(Vec3::new(5.0, 6.0, 7.0), 1e-6));
        assert!(pose.rotations[1].abs_diff_eq(Quat::from_rotation_z(PI / 3.0), 1e-5));
        assert!(pose.scales[1].abs_diff_eq(Vec3::new(1.5, 1.5, 1.5), 1e-6));
    }

    #[test]
    fn test_pose_try_set_transform() {
        let mut pose = Pose::new(2, PoseType::Current);
        let t = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));

        assert!(pose.try_set_transform(0, t));
        assert!(pose.try_set_transform(1, t));
        assert!(!pose.try_set_transform(2, t));
        assert!(!pose.try_set_transform(100, t));
    }

    #[test]
    #[should_panic]
    fn test_pose_get_transform_out_of_bounds() {
        let pose = Pose::new(2, PoseType::Current);
        let _ = pose.get_transform(5);
    }

    #[test]
    #[should_panic]
    fn test_pose_set_transform_out_of_bounds() {
        let mut pose = Pose::new(2, PoseType::Current);
        pose.set_transform(5, Transform::IDENTITY);
    }

    // ===== Pose Blend Tests =====

    #[test]
    fn test_pose_blend_weight_zero() {
        let a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);
        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let result = a.blend(&b, 0.0);

        // Should be exactly pose a
        assert!(result.positions[0].abs_diff_eq(Vec3::ZERO, 1e-6));
    }

    #[test]
    fn test_pose_blend_weight_one() {
        let a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);
        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let result = a.blend(&b, 1.0);

        // Should be exactly pose b
        assert!(result.positions[0].abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_pose_blend_midpoint() {
        let mut a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);

        a.positions[0] = Vec3::ZERO;
        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        a.rotations[0] = Quat::IDENTITY;
        b.rotations[0] = Quat::from_rotation_y(PI / 2.0);

        a.scales[0] = Vec3::ONE;
        b.scales[0] = Vec3::new(3.0, 3.0, 3.0);

        let result = a.blend(&b, 0.5);

        assert!(result.positions[0].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-6));
        assert!(result.rotations[0].abs_diff_eq(Quat::from_rotation_y(PI / 4.0), 1e-5));
        assert!(result.scales[0].abs_diff_eq(Vec3::new(2.0, 2.0, 2.0), 1e-6));
    }

    #[test]
    fn test_pose_blend_clamps_weight() {
        let a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);
        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        // Weight > 1 should clamp to 1
        let result = a.blend(&b, 2.0);
        assert!(result.positions[0].abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), 1e-6));

        // Weight < 0 should clamp to 0
        let result = a.blend(&b, -1.0);
        assert!(result.positions[0].abs_diff_eq(Vec3::ZERO, 1e-6));
    }

    #[test]
    fn test_pose_blend_result_type() {
        let a = Pose::new(2, PoseType::Bind);
        let b = Pose::new(2, PoseType::Reference);

        let result = a.blend(&b, 0.5);
        assert_eq!(result.pose_type, PoseType::Current);
    }

    #[test]
    #[should_panic(expected = "cannot blend poses with different bone counts")]
    fn test_pose_blend_different_sizes() {
        let a = Pose::new(2, PoseType::Current);
        let b = Pose::new(3, PoseType::Current);
        let _ = a.blend(&b, 0.5);
    }

    #[test]
    fn test_pose_blend_fast() {
        let mut a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);

        a.rotations[0] = Quat::IDENTITY;
        b.rotations[0] = Quat::from_rotation_y(PI / 2.0);

        let result = a.blend_fast(&b, 0.5);

        // nlerp gives slightly different result than slerp
        assert!(result.rotations[0].is_normalized());
    }

    // ===== Pose Additive Blend Tests =====

    #[test]
    fn test_pose_blend_additive_zero_weight() {
        let base = Pose::new(2, PoseType::Current);
        let mut additive = Pose::new(2, PoseType::Additive);
        additive.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let result = base.blend_additive(&additive, 0.0);

        // No effect
        assert!(result.positions[0].abs_diff_eq(Vec3::ZERO, 1e-6));
    }

    #[test]
    fn test_pose_blend_additive_full_weight() {
        let mut base = Pose::new(2, PoseType::Current);
        base.positions[0] = Vec3::new(5.0, 0.0, 0.0);

        let mut additive = Pose::new(2, PoseType::Additive);
        additive.positions[0] = Vec3::new(3.0, 0.0, 0.0);

        let result = base.blend_additive(&additive, 1.0);

        // Base + additive
        assert!(result.positions[0].abs_diff_eq(Vec3::new(8.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn test_pose_blend_additive_rotation() {
        let base = Pose::new(2, PoseType::Current);
        let mut additive = Pose::new(2, PoseType::Additive);
        additive.rotations[0] = Quat::from_rotation_y(PI / 2.0);

        let result = base.blend_additive(&additive, 1.0);

        // Should apply the full rotation
        assert!(result.rotations[0].abs_diff_eq(Quat::from_rotation_y(PI / 2.0), 1e-5));
    }

    #[test]
    fn test_pose_blend_additive_negative_weight() {
        let mut base = Pose::new(2, PoseType::Current);
        base.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let mut additive = Pose::new(2, PoseType::Additive);
        additive.positions[0] = Vec3::new(3.0, 0.0, 0.0);

        let result = base.blend_additive(&additive, -1.0);

        // Base - additive
        assert!(result.positions[0].abs_diff_eq(Vec3::new(7.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    #[should_panic(expected = "cannot blend additive pose")]
    fn test_pose_blend_additive_different_sizes() {
        let a = Pose::new(2, PoseType::Current);
        let b = Pose::new(3, PoseType::Additive);
        let _ = a.blend_additive(&b, 0.5);
    }

    // ===== Pose Copy/Reset Tests =====

    #[test]
    fn test_pose_copy_from() {
        let mut a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);

        b.positions[0] = Vec3::new(1.0, 2.0, 3.0);
        b.rotations[0] = Quat::from_rotation_x(PI / 6.0);
        b.scales[0] = Vec3::new(2.0, 2.0, 2.0);

        a.copy_from(&b);

        assert!(a.positions[0].abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-6));
        assert!(a.rotations[0].abs_diff_eq(Quat::from_rotation_x(PI / 6.0), 1e-5));
        assert!(a.scales[0].abs_diff_eq(Vec3::new(2.0, 2.0, 2.0), 1e-6));
    }

    #[test]
    #[should_panic(expected = "cannot copy from pose")]
    fn test_pose_copy_from_different_sizes() {
        let mut a = Pose::new(2, PoseType::Current);
        let b = Pose::new(3, PoseType::Current);
        a.copy_from(&b);
    }

    #[test]
    fn test_pose_reset_to_identity_current() {
        let mut pose = Pose::new(2, PoseType::Current);
        pose.positions[0] = Vec3::new(1.0, 2.0, 3.0);
        pose.rotations[0] = Quat::from_rotation_y(PI / 4.0);
        pose.scales[0] = Vec3::new(2.0, 2.0, 2.0);

        pose.reset_to_identity();

        assert!(pose.positions[0].abs_diff_eq(Vec3::ZERO, 1e-6));
        assert!(pose.rotations[0].abs_diff_eq(Quat::IDENTITY, 1e-6));
        assert!(pose.scales[0].abs_diff_eq(Vec3::ONE, 1e-6));
    }

    #[test]
    fn test_pose_reset_to_identity_additive() {
        let mut pose = Pose::new(2, PoseType::Additive);
        pose.positions[0] = Vec3::new(1.0, 2.0, 3.0);
        pose.rotations[0] = Quat::from_rotation_y(PI / 4.0);
        pose.scales[0] = Vec3::new(0.5, 0.5, 0.5);

        pose.reset_to_identity();

        assert!(pose.positions[0].abs_diff_eq(Vec3::ZERO, 1e-6));
        assert!(pose.rotations[0].abs_diff_eq(Quat::IDENTITY, 1e-6));
        assert!(pose.scales[0].abs_diff_eq(Vec3::ZERO, 1e-6)); // Zero for additive
    }

    // ===== Pose Utility Tests =====

    #[test]
    fn test_pose_transforms() {
        let mut pose = Pose::new(2, PoseType::Current);
        pose.positions[0] = Vec3::new(1.0, 0.0, 0.0);
        pose.positions[1] = Vec3::new(0.0, 2.0, 0.0);

        let transforms = pose.transforms();

        assert_eq!(transforms.len(), 2);
        assert!(transforms[0].position.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-6));
        assert!(transforms[1].position.abs_diff_eq(Vec3::new(0.0, 2.0, 0.0), 1e-6));
    }

    #[test]
    fn test_pose_resize_grow() {
        let mut pose = Pose::new(2, PoseType::Current);
        pose.positions[0] = Vec3::new(1.0, 0.0, 0.0);

        pose.resize(4);

        assert_eq!(pose.bone_count(), 4);
        // Original data preserved
        assert!(pose.positions[0].abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-6));
        // New bones are identity
        assert!(pose.positions[2].abs_diff_eq(Vec3::ZERO, 1e-6));
        assert!(pose.scales[2].abs_diff_eq(Vec3::ONE, 1e-6));
    }

    #[test]
    fn test_pose_resize_shrink() {
        let mut pose = Pose::new(4, PoseType::Current);
        pose.resize(2);
        assert_eq!(pose.bone_count(), 2);
    }

    #[test]
    fn test_pose_resize_additive() {
        let mut pose = Pose::new(1, PoseType::Additive);
        pose.resize(3);

        // New additive scales should be zero
        assert!(pose.scales[2].abs_diff_eq(Vec3::ZERO, 1e-6));
    }

    #[test]
    fn test_pose_is_identity() {
        let pose = Pose::new(2, PoseType::Current);
        assert!(pose.is_identity(1e-5));

        let mut modified = Pose::new(2, PoseType::Current);
        modified.positions[0] = Vec3::new(0.1, 0.0, 0.0);
        assert!(!modified.is_identity(1e-5));
    }

    #[test]
    fn test_pose_is_identity_additive() {
        let pose = Pose::new(2, PoseType::Additive);
        assert!(pose.is_identity(1e-5));
    }

    // ===== PoseBuffer Tests =====

    #[test]
    fn test_pose_buffer_new() {
        let buffer = PoseBuffer::new(3);

        assert_eq!(buffer.bone_count(), 3);
        assert_eq!(buffer.local_pose.bone_count(), 3);
        assert_eq!(buffer.model_pose.len(), 3);

        // Model pose should be identity
        assert!(buffer.model_pose[0].abs_diff_eq(Mat4::IDENTITY, 1e-6));
    }

    #[test]
    fn test_pose_buffer_from_skeleton() {
        let skeleton = SkeletonBuilder::new()
            .root("root")
            .child_at("arm", "root", Vec3::new(1.0, 0.0, 0.0))
            .build_unchecked();

        let buffer = PoseBuffer::from_skeleton(&skeleton);

        assert_eq!(buffer.bone_count(), 2);

        // Model pose should be computed
        let arm_pos = buffer.model_pose[1].w_axis.truncate();
        assert!(arm_pos.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_pose_buffer_update_model_pose() {
        let skeleton = SkeletonBuilder::new()
            .root("root")
            .child_at("arm", "root", Vec3::new(1.0, 0.0, 0.0))
            .build_unchecked();

        let mut buffer = PoseBuffer::new(2);
        buffer.local_pose = Pose::from_skeleton(&skeleton, PoseType::Current);

        // Modify local pose
        buffer.local_pose.positions[0] = Vec3::new(5.0, 0.0, 0.0);

        buffer.update_model_pose(&skeleton);

        // Root should be at (5, 0, 0)
        let root_pos = buffer.model_pose[0].w_axis.truncate();
        assert!(root_pos.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-5));

        // Arm should be at (6, 0, 0) = parent(5,0,0) + local(1,0,0)
        let arm_pos = buffer.model_pose[1].w_axis.truncate();
        assert!(arm_pos.abs_diff_eq(Vec3::new(6.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_pose_buffer_set_pose() {
        let skeleton = SkeletonBuilder::new()
            .root("root")
            .child("arm", "root")
            .build_unchecked();

        let mut buffer = PoseBuffer::from_skeleton(&skeleton);

        let mut new_pose = Pose::new(2, PoseType::Current);
        new_pose.positions[0] = Vec3::new(3.0, 0.0, 0.0);

        buffer.set_pose(&new_pose, &skeleton);

        // Should update both local and model
        assert!(buffer.local_pose.positions[0].abs_diff_eq(Vec3::new(3.0, 0.0, 0.0), 1e-6));
        let root_pos = buffer.model_pose[0].w_axis.truncate();
        assert!(root_pos.abs_diff_eq(Vec3::new(3.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_pose_buffer_get_model_transform() {
        let buffer = PoseBuffer::new(2);
        let mat = buffer.get_model_transform(0);
        assert!(mat.abs_diff_eq(Mat4::IDENTITY, 1e-6));
    }

    #[test]
    fn test_pose_buffer_resize() {
        let mut buffer = PoseBuffer::new(2);
        buffer.resize(4);

        assert_eq!(buffer.bone_count(), 4);
        assert_eq!(buffer.local_pose.bone_count(), 4);
        assert_eq!(buffer.model_pose.len(), 4);
    }

    #[test]
    fn test_pose_buffer_reset() {
        let mut buffer = PoseBuffer::new(2);
        buffer.local_pose.positions[0] = Vec3::new(1.0, 0.0, 0.0);
        buffer.model_pose[0] = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));

        buffer.reset();

        assert!(buffer.local_pose.positions[0].abs_diff_eq(Vec3::ZERO, 1e-6));
        assert!(buffer.model_pose[0].abs_diff_eq(Mat4::IDENTITY, 1e-6));
    }

    #[test]
    fn test_pose_buffer_default() {
        let buffer = PoseBuffer::default();
        assert_eq!(buffer.bone_count(), 0);
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_blend_with_rotated_skeleton() {
        let mut skeleton = crate::skeleton::Skeleton::new();
        skeleton.add_bone(
            Bone::root("root")
                .with_local_transform(Transform::from_rotation(Quat::from_rotation_y(PI / 2.0)))
        );
        skeleton.add_bone(
            Bone::new("child")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(1.0, 0.0, 0.0)))
        );

        let pose1 = Pose::from_skeleton(&skeleton, PoseType::Current);
        let pose2 = Pose::from_skeleton(&skeleton, PoseType::Current);

        let blended = pose1.blend(&pose2, 0.5);

        // Should be same as original since both identical
        assert!(blended.rotations[0].abs_diff_eq(pose1.rotations[0], 1e-5));
    }

    #[test]
    fn test_pose_equality() {
        let a = Pose::new(2, PoseType::Current);
        let b = Pose::new(2, PoseType::Current);

        assert_eq!(a, b);

        let mut c = Pose::new(2, PoseType::Current);
        c.positions[0] = Vec3::new(1.0, 0.0, 0.0);

        assert_ne!(a, c);
    }

    #[test]
    fn test_pose_serialization() {
        let mut pose = Pose::new(2, PoseType::Bind);
        pose.positions[0] = Vec3::new(1.0, 2.0, 3.0);
        pose.rotations[0] = Quat::from_rotation_y(PI / 4.0);

        let json = serde_json::to_string(&pose).unwrap();
        let recovered: Pose = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered.pose_type, PoseType::Bind);
        assert!(recovered.positions[0].abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-5));
    }

    #[test]
    fn test_large_skeleton_blend() {
        // Test with a realistic bone count
        let bone_count = 100;
        let mut a = Pose::new(bone_count, PoseType::Current);
        let mut b = Pose::new(bone_count, PoseType::Current);

        for i in 0..bone_count {
            a.positions[i] = Vec3::new(i as f32, 0.0, 0.0);
            b.positions[i] = Vec3::new(0.0, i as f32, 0.0);
        }

        let result = a.blend(&b, 0.5);

        // Spot check a few bones
        assert!(result.positions[50].abs_diff_eq(Vec3::new(25.0, 25.0, 0.0), 1e-5));
    }
}
