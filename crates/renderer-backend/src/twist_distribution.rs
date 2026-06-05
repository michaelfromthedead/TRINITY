//! Twist distribution for skeletal animation in TRINITY Engine (T-AN-7.7).
//!
//! This module provides twist bone spreading for forearm, spine, and neck chains.
//! It prevents candy-wrapper deformation artifacts by distributing twist rotation
//! across multiple bones rather than concentrating it at a single joint.
//!
//! # Problem
//!
//! When a forearm rotates 180 degrees around its length axis (e.g., flipping the hand),
//! a single-bone setup causes the mesh to collapse into a "candy wrapper" shape.
//! This module distributes that twist across helper bones.
//!
//! # Algorithm
//!
//! 1. Extract the twist component from the source bone using swing-twist decomposition
//! 2. Scale the twist based on configurable weights
//! 3. Apply weighted twist to each bone in the chain
//! 4. Preserve the original swing component (bend direction)
//!
//! # Chain Types
//!
//! - **Forearm twist**: Wrist rotation distributed to upper arm
//! - **Spine twist**: Pelvis rotation distributed through spine bones to chest
//! - **Neck twist**: Head rotation distributed through neck to shoulders
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::twist_distribution::{
//!     TwistChain, TwistParams, TwistFalloff,
//!     distribute_twist, create_forearm_twist_chain,
//! };
//! use renderer_backend::pose::Pose;
//! use renderer_backend::skeleton::Skeleton;
//!
//! // Create a forearm twist chain
//! let chain = create_forearm_twist_chain(1, 2, 3); // upper_arm, forearm, hand
//!
//! let params = TwistParams {
//!     enabled: true,
//!     falloff: TwistFalloff::Linear,
//!     max_twist: std::f32::consts::PI, // 180 degrees max
//! };
//!
//! // Apply twist distribution to a pose
//! distribute_twist(&chain, &skeleton, &mut pose, &params);
//! ```

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use crate::pose::Pose;
use crate::skeleton::Skeleton;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum weight threshold to avoid degenerate distribution.
pub const MIN_WEIGHT: f32 = 1e-6;

/// Maximum number of bones in a twist chain.
pub const MAX_TWIST_CHAIN_LENGTH: usize = 32;

/// Epsilon for quaternion operations.
pub const QUAT_EPSILON: f32 = 1e-7;

/// Default maximum twist angle in radians (180 degrees).
pub const DEFAULT_MAX_TWIST: f32 = PI;

/// Epsilon for axis normalization.
pub const AXIS_EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// TwistFalloff
// ---------------------------------------------------------------------------

/// Weight distribution pattern across bones in a twist chain.
///
/// Controls how the total twist angle is divided among bones.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum TwistFalloff {
    /// Equal distribution: each bone receives (1 / N) of the total twist.
    /// Results in smooth, even deformation across the chain.
    #[default]
    Linear,

    /// Exponential decay: bones closer to the source receive more twist.
    /// The `decay` parameter controls how quickly the weight falls off.
    /// - decay < 1.0: slower falloff (more even)
    /// - decay = 1.0: standard exponential
    /// - decay > 1.0: faster falloff (more concentrated at source)
    Exponential { decay: f32 },

    /// Custom per-bone weights provided in TwistChain.
    /// Use this when you need precise artistic control.
    Custom,
}

impl TwistFalloff {
    /// Create an exponential falloff with the given decay rate.
    #[inline]
    pub fn exponential(decay: f32) -> Self {
        Self::Exponential { decay: decay.max(0.1) }
    }

    /// Get a human-readable name for this falloff type.
    pub fn name(&self) -> &'static str {
        match self {
            TwistFalloff::Linear => "Linear",
            TwistFalloff::Exponential { .. } => "Exponential",
            TwistFalloff::Custom => "Custom",
        }
    }

    /// Check if this falloff uses custom weights.
    #[inline]
    pub fn is_custom(&self) -> bool {
        matches!(self, TwistFalloff::Custom)
    }
}

// ---------------------------------------------------------------------------
// TwistParams
// ---------------------------------------------------------------------------

/// Parameters controlling twist distribution behavior.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwistParams {
    /// Whether twist distribution is enabled.
    pub enabled: bool,

    /// Weight distribution pattern across bones.
    pub falloff: TwistFalloff,

    /// Maximum twist angle in radians. Twist beyond this is clamped.
    /// Use `PI` for 180 degrees, `2*PI` for full rotation.
    pub max_twist: f32,
}

impl Default for TwistParams {
    fn default() -> Self {
        Self {
            enabled: true,
            falloff: TwistFalloff::Linear,
            max_twist: DEFAULT_MAX_TWIST,
        }
    }
}

impl TwistParams {
    /// Create parameters with linear falloff.
    pub fn linear() -> Self {
        Self::default()
    }

    /// Create parameters with exponential falloff.
    pub fn exponential(decay: f32) -> Self {
        Self {
            enabled: true,
            falloff: TwistFalloff::exponential(decay),
            max_twist: DEFAULT_MAX_TWIST,
        }
    }

    /// Create parameters with custom weights (from TwistChain).
    pub fn custom() -> Self {
        Self {
            enabled: true,
            falloff: TwistFalloff::Custom,
            max_twist: DEFAULT_MAX_TWIST,
        }
    }

    /// Set the maximum twist angle.
    pub fn with_max_twist(mut self, max_twist: f32) -> Self {
        self.max_twist = max_twist.abs();
        self
    }

    /// Disable twist distribution.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// TwistChain
// ---------------------------------------------------------------------------

/// A chain of bones for twist distribution.
///
/// Defines which bones receive distributed twist and how much each one gets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwistChain {
    /// Bone indices in the chain (ordered from source outward).
    /// For forearm twist: [upper_arm_twist1, upper_arm_twist2, forearm]
    pub bones: Vec<usize>,

    /// Per-bone twist weights. Must sum to approximately 1.0.
    /// For linear falloff, this can be empty (computed dynamically).
    pub weights: Vec<f32>,

    /// Local twist axis for decomposition (usually X or Y axis in bone space).
    /// This should point along the bone's length.
    pub twist_axis: Vec3,

    /// Index of the bone that provides the twist rotation.
    /// For forearm: the hand/wrist bone.
    pub source_bone: usize,
}

impl TwistChain {
    /// Create a new twist chain.
    ///
    /// # Arguments
    ///
    /// * `bones` - Bone indices in the chain
    /// * `source_bone` - Index of the bone providing the twist
    /// * `twist_axis` - Local axis to decompose twist around (normalized)
    pub fn new(bones: Vec<usize>, source_bone: usize, twist_axis: Vec3) -> Self {
        let bone_count = bones.len();
        Self {
            bones,
            weights: Vec::new(), // Computed dynamically unless Custom falloff
            twist_axis: twist_axis.normalize_or_zero(),
            source_bone,
        }
    }

    /// Create a chain with custom per-bone weights.
    ///
    /// Weights will be normalized to sum to 1.0.
    pub fn with_custom_weights(
        bones: Vec<usize>,
        weights: Vec<f32>,
        source_bone: usize,
        twist_axis: Vec3,
    ) -> Self {
        let mut chain = Self::new(bones, source_bone, twist_axis);
        chain.set_weights(weights);
        chain
    }

    /// Set custom weights (normalizes automatically).
    pub fn set_weights(&mut self, weights: Vec<f32>) {
        self.weights = weights;
        self.normalize_weights();
    }

    /// Get the number of bones in the chain.
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Check if the chain is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bones.is_empty()
    }

    /// Normalize weights to sum to 1.0.
    pub fn normalize_weights(&mut self) {
        if self.weights.is_empty() {
            return;
        }

        let sum: f32 = self.weights.iter().map(|w| w.abs()).sum();
        if sum > MIN_WEIGHT {
            for w in &mut self.weights {
                *w /= sum;
            }
        } else {
            // Fallback to uniform weights
            let uniform = 1.0 / self.weights.len() as f32;
            for w in &mut self.weights {
                *w = uniform;
            }
        }
    }

    /// Compute weights based on falloff type.
    ///
    /// Returns a vector of weights (one per bone) that sum to 1.0.
    pub fn compute_weights(&self, falloff: TwistFalloff) -> Vec<f32> {
        let n = self.bone_count();
        if n == 0 {
            return Vec::new();
        }

        match falloff {
            TwistFalloff::Linear => {
                // Equal distribution
                vec![1.0 / n as f32; n]
            }
            TwistFalloff::Exponential { decay } => {
                // Exponential decay: w_i = decay^i / sum(decay^j)
                let mut weights = Vec::with_capacity(n);
                let mut sum = 0.0;

                for i in 0..n {
                    let w = decay.powi(i as i32);
                    weights.push(w);
                    sum += w;
                }

                // Normalize
                if sum > MIN_WEIGHT {
                    for w in &mut weights {
                        *w /= sum;
                    }
                }

                weights
            }
            TwistFalloff::Custom => {
                // Use stored weights (already normalized)
                if self.weights.len() == n {
                    self.weights.clone()
                } else {
                    // Fallback to linear if weights don't match
                    vec![1.0 / n as f32; n]
                }
            }
        }
    }

    /// Validate the chain against a skeleton.
    ///
    /// Returns `true` if all bone indices are valid.
    pub fn validate(&self, skeleton: &Skeleton) -> bool {
        let bone_count = skeleton.bone_count();

        // Check source bone
        if self.source_bone >= bone_count {
            return false;
        }

        // Check all bones in chain
        for &bone in &self.bones {
            if bone >= bone_count {
                return false;
            }
        }

        // Check twist axis is valid
        if self.twist_axis.length_squared() < AXIS_EPSILON {
            return false;
        }

        true
    }
}

// ---------------------------------------------------------------------------
// Swing-Twist Decomposition
// ---------------------------------------------------------------------------

/// Decompose a quaternion into swing and twist components around an axis.
///
/// Given a rotation Q and a twist axis T, decomposes Q into:
/// - Twist: rotation around T
/// - Swing: rotation that aligns the axis to its final direction
///
/// Such that Q = Swing * Twist
///
/// # Arguments
///
/// * `rotation` - The quaternion to decompose
/// * `twist_axis` - The axis to decompose around (should be normalized)
///
/// # Returns
///
/// A tuple of (swing, twist) quaternions.
///
/// # Algorithm
///
/// 1. Project the rotation's axis onto the twist axis
/// 2. The twist component is the rotation around this projected axis
/// 3. The swing is the remaining rotation (Q * twist.inverse())
#[inline]
pub fn swing_twist_decompose(rotation: Quat, twist_axis: Vec3) -> (Quat, Quat) {
    // Get the rotation axis and angle
    let (axis, _angle) = rotation.to_axis_angle();

    // Handle identity rotation
    if axis.length_squared() < QUAT_EPSILON {
        return (Quat::IDENTITY, Quat::IDENTITY);
    }

    // Project the quaternion's vector part onto the twist axis
    // The quaternion is [x, y, z, w] where [x, y, z] is the axis*sin(angle/2)
    let rotation_vector = Vec3::new(rotation.x, rotation.y, rotation.z);

    // Project onto twist axis
    let twist_axis_normalized = twist_axis.normalize_or_zero();
    let projection = rotation_vector.dot(twist_axis_normalized);

    // Build the twist quaternion from the projected component
    let twist_vector = twist_axis_normalized * projection;
    let twist = Quat::from_xyzw(twist_vector.x, twist_vector.y, twist_vector.z, rotation.w);

    // Normalize the twist quaternion
    let twist_len_sq = twist.length_squared();
    let twist = if twist_len_sq > QUAT_EPSILON {
        twist.normalize()
    } else {
        Quat::IDENTITY
    };

    // Compute swing as rotation * inverse(twist)
    // swing = rotation * twist^-1
    let swing = rotation * twist.conjugate();
    let swing = swing.normalize();

    (swing, twist)
}

/// Extract just the twist component around an axis.
///
/// This is a convenience function when only the twist is needed.
#[inline]
pub fn extract_twist(rotation: Quat, twist_axis: Vec3) -> Quat {
    swing_twist_decompose(rotation, twist_axis).1
}

/// Extract just the swing component around an axis.
///
/// This is a convenience function when only the swing is needed.
#[inline]
pub fn extract_swing(rotation: Quat, twist_axis: Vec3) -> Quat {
    swing_twist_decompose(rotation, twist_axis).0
}

/// Get the twist angle in radians from a quaternion around an axis.
///
/// Returns a value in the range [-PI, PI].
#[inline]
pub fn get_twist_angle(rotation: Quat, twist_axis: Vec3) -> f32 {
    let twist = extract_twist(rotation, twist_axis);
    let (axis, angle) = twist.to_axis_angle();

    // Determine sign based on axis alignment with twist_axis
    let sign = if axis.dot(twist_axis) >= 0.0 { 1.0 } else { -1.0 };
    sign * angle
}

/// Create a twist quaternion from an angle around an axis.
#[inline]
pub fn twist_from_angle(angle: f32, twist_axis: Vec3) -> Quat {
    Quat::from_axis_angle(twist_axis.normalize_or_zero(), angle)
}

/// Clamp a twist angle to a maximum magnitude.
#[inline]
pub fn clamp_twist_angle(angle: f32, max_angle: f32) -> f32 {
    angle.clamp(-max_angle, max_angle)
}

// ---------------------------------------------------------------------------
// Twist Distribution Functions
// ---------------------------------------------------------------------------

/// Distribute twist from source bone across a chain.
///
/// This modifies the pose in-place, spreading the twist component
/// from the source bone across all bones in the chain.
///
/// # Arguments
///
/// * `chain` - The twist chain configuration
/// * `skeleton` - The skeleton (for validation and bone lengths)
/// * `pose` - The pose to modify (in-place)
/// * `params` - Distribution parameters
///
/// # Returns
///
/// `true` if twist was applied, `false` if skipped (disabled or invalid).
pub fn distribute_twist(
    chain: &TwistChain,
    skeleton: &Skeleton,
    pose: &mut Pose,
    params: &TwistParams,
) -> bool {
    // Early exit if disabled
    if !params.enabled {
        return false;
    }

    // Validate chain
    if chain.is_empty() || !chain.validate(skeleton) {
        return false;
    }

    // Check source bone is in pose
    if chain.source_bone >= pose.bone_count() {
        return false;
    }

    // Get the source rotation
    let source_rotation = pose.rotations[chain.source_bone];

    // Extract the twist component
    let twist = extract_twist(source_rotation, chain.twist_axis);

    // Get twist angle and clamp
    let twist_angle = get_twist_angle(source_rotation, chain.twist_axis);
    let clamped_angle = clamp_twist_angle(twist_angle, params.max_twist);

    // Compute weights based on falloff
    let weights = chain.compute_weights(params.falloff);

    // Apply weighted twist to each bone in the chain
    for (i, &bone_index) in chain.bones.iter().enumerate() {
        if bone_index >= pose.bone_count() {
            continue;
        }

        let weight = weights.get(i).copied().unwrap_or(0.0);
        if weight.abs() < MIN_WEIGHT {
            continue;
        }

        // Create partial twist for this bone
        let partial_angle = clamped_angle * weight;
        let partial_twist = twist_from_angle(partial_angle, chain.twist_axis);

        // Get current rotation, decompose into swing and twist
        let current = pose.rotations[bone_index];
        let (swing, _old_twist) = swing_twist_decompose(current, chain.twist_axis);

        // Apply new twist: new_rotation = swing * partial_twist
        let new_rotation = (swing * partial_twist).normalize();
        pose.rotations[bone_index] = new_rotation;
    }

    // Remove twist from source bone (it's now distributed)
    let source_current = pose.rotations[chain.source_bone];
    let (source_swing, _source_twist) = swing_twist_decompose(source_current, chain.twist_axis);
    pose.rotations[chain.source_bone] = source_swing.normalize();

    true
}

/// Distribute twist without modifying the source bone.
///
/// Use this when you want to add helper bone twist without changing
/// the original bone's rotation.
pub fn distribute_twist_additive(
    chain: &TwistChain,
    skeleton: &Skeleton,
    pose: &mut Pose,
    params: &TwistParams,
) -> bool {
    // Early exit if disabled
    if !params.enabled {
        return false;
    }

    // Validate chain
    if chain.is_empty() || !chain.validate(skeleton) {
        return false;
    }

    // Check source bone is in pose
    if chain.source_bone >= pose.bone_count() {
        return false;
    }

    // Get the source rotation
    let source_rotation = pose.rotations[chain.source_bone];

    // Get twist angle and clamp
    let twist_angle = get_twist_angle(source_rotation, chain.twist_axis);
    let clamped_angle = clamp_twist_angle(twist_angle, params.max_twist);

    // Compute weights based on falloff
    let weights = chain.compute_weights(params.falloff);

    // Apply weighted twist to each bone in the chain
    for (i, &bone_index) in chain.bones.iter().enumerate() {
        if bone_index >= pose.bone_count() {
            continue;
        }

        let weight = weights.get(i).copied().unwrap_or(0.0);
        if weight.abs() < MIN_WEIGHT {
            continue;
        }

        // Create partial twist for this bone
        let partial_angle = clamped_angle * weight;
        let partial_twist = twist_from_angle(partial_angle, chain.twist_axis);

        // Add twist to current rotation
        let current = pose.rotations[bone_index];
        let new_rotation = (current * partial_twist).normalize();
        pose.rotations[bone_index] = new_rotation;
    }

    true
}

// ---------------------------------------------------------------------------
// Chain Factory Functions
// ---------------------------------------------------------------------------

/// Create a forearm twist chain with standard weights.
///
/// Standard forearm setup distributes twist from wrist toward upper arm:
/// - Upper arm receives 0% twist
/// - Forearm receives 50% twist
/// - Upper arm twist helpers (if present) receive remaining 50%
///
/// # Arguments
///
/// * `upper_arm` - Index of upper arm bone
/// * `forearm` - Index of forearm bone
/// * `hand` - Index of hand/wrist bone (twist source)
///
/// # Returns
///
/// A TwistChain configured for forearm twist distribution.
pub fn create_forearm_twist_chain(upper_arm: usize, forearm: usize, hand: usize) -> TwistChain {
    TwistChain {
        bones: vec![forearm, upper_arm],
        weights: vec![0.6, 0.4], // More twist on forearm
        twist_axis: Vec3::X, // Typical forearm twist axis
        source_bone: hand,
    }
}

/// Create a forearm twist chain with helper bones.
///
/// For setups with dedicated twist helper bones between forearm and upper arm.
///
/// # Arguments
///
/// * `twist_bones` - Indices of twist helper bones (ordered source to upper arm)
/// * `hand` - Index of hand/wrist bone (twist source)
/// * `twist_axis` - Local twist axis
pub fn create_forearm_twist_chain_with_helpers(
    twist_bones: &[usize],
    hand: usize,
    twist_axis: Vec3,
) -> TwistChain {
    TwistChain::new(twist_bones.to_vec(), hand, twist_axis)
}

/// Create a spine twist chain.
///
/// Distributes twist from pelvis rotation through spine bones to chest.
///
/// # Arguments
///
/// * `pelvis` - Index of pelvis bone (twist source)
/// * `spine_bones` - Indices of spine bones (ordered pelvis to chest)
/// * `chest` - Index of chest bone
///
/// # Returns
///
/// A TwistChain configured for spine twist distribution.
pub fn create_spine_twist_chain(pelvis: usize, spine_bones: &[usize], chest: usize) -> TwistChain {
    let mut bones = spine_bones.to_vec();
    bones.push(chest);

    TwistChain::new(bones, pelvis, Vec3::Y) // Spine typically twists around Y (up)
}

/// Create a neck twist chain.
///
/// Distributes twist from head rotation through neck bones.
///
/// # Arguments
///
/// * `head` - Index of head bone (twist source)
/// * `neck_bones` - Indices of neck bones (ordered head to shoulders)
///
/// # Returns
///
/// A TwistChain configured for neck twist distribution.
pub fn create_neck_twist_chain(head: usize, neck_bones: &[usize]) -> TwistChain {
    TwistChain::new(neck_bones.to_vec(), head, Vec3::Y)
}

/// Create a upper leg twist chain (for hip rotation).
///
/// # Arguments
///
/// * `thigh` - Index of thigh bone
/// * `hip` - Index of hip bone (twist source)
pub fn create_upper_leg_twist_chain(thigh: usize, hip: usize) -> TwistChain {
    TwistChain {
        bones: vec![thigh],
        weights: vec![1.0],
        twist_axis: Vec3::Y, // Leg typically twists around Y (bone length)
        source_bone: hip,
    }
}

/// Create a lower leg twist chain (for ankle rotation).
///
/// # Arguments
///
/// * `shin` - Index of shin bone
/// * `foot` - Index of foot bone (twist source)
pub fn create_lower_leg_twist_chain(shin: usize, foot: usize) -> TwistChain {
    TwistChain {
        bones: vec![shin],
        weights: vec![1.0],
        twist_axis: Vec3::Y,
        source_bone: foot,
    }
}

// ---------------------------------------------------------------------------
// Twist Chain Manager
// ---------------------------------------------------------------------------

/// Manager for multiple twist chains on a skeleton.
///
/// Handles common setups like forearm twist on both arms,
/// spine twist, and provides batch update functionality.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TwistChainManager {
    /// All twist chains to process.
    pub chains: Vec<TwistChain>,

    /// Shared parameters for all chains (can override per-chain if needed).
    pub params: TwistParams,
}

impl TwistChainManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a twist chain.
    pub fn add_chain(&mut self, chain: TwistChain) {
        self.chains.push(chain);
    }

    /// Get the number of chains.
    #[inline]
    pub fn chain_count(&self) -> usize {
        self.chains.len()
    }

    /// Apply all twist chains to a pose.
    ///
    /// Returns the number of chains successfully applied.
    pub fn apply_all(&self, skeleton: &Skeleton, pose: &mut Pose) -> usize {
        let mut applied = 0;
        for chain in &self.chains {
            if distribute_twist(chain, skeleton, pose, &self.params) {
                applied += 1;
            }
        }
        applied
    }

    /// Apply all twist chains additively (don't modify source bones).
    pub fn apply_all_additive(&self, skeleton: &Skeleton, pose: &mut Pose) -> usize {
        let mut applied = 0;
        for chain in &self.chains {
            if distribute_twist_additive(chain, skeleton, pose, &self.params) {
                applied += 1;
            }
        }
        applied
    }

    /// Create a standard humanoid twist setup.
    ///
    /// Includes:
    /// - Left and right forearm twist
    /// - Spine twist
    /// - Neck twist
    ///
    /// # Arguments
    ///
    /// * `left_arm` - (upper_arm, forearm, hand) bone indices
    /// * `right_arm` - (upper_arm, forearm, hand) bone indices
    /// * `spine` - (pelvis, spine_bones, chest) configuration
    /// * `neck` - (head, neck_bones) configuration
    pub fn create_humanoid(
        left_arm: (usize, usize, usize),
        right_arm: (usize, usize, usize),
        spine: Option<(usize, &[usize], usize)>,
        neck: Option<(usize, &[usize])>,
    ) -> Self {
        let mut manager = Self::new();

        // Left forearm twist
        manager.add_chain(create_forearm_twist_chain(
            left_arm.0,
            left_arm.1,
            left_arm.2,
        ));

        // Right forearm twist (mirror twist axis)
        let mut right_chain = create_forearm_twist_chain(
            right_arm.0,
            right_arm.1,
            right_arm.2,
        );
        right_chain.twist_axis = -Vec3::X; // Mirror for right side
        manager.add_chain(right_chain);

        // Spine twist
        if let Some((pelvis, spine_bones, chest)) = spine {
            manager.add_chain(create_spine_twist_chain(pelvis, spine_bones, chest));
        }

        // Neck twist
        if let Some((head, neck_bones)) = neck {
            manager.add_chain(create_neck_twist_chain(head, neck_bones));
        }

        manager
    }

    /// Enable or disable all chains.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.params.enabled = enabled;
    }

    /// Set the falloff for all chains.
    pub fn set_falloff(&mut self, falloff: TwistFalloff) {
        self.params.falloff = falloff;
    }

    /// Validate all chains against a skeleton.
    pub fn validate(&self, skeleton: &Skeleton) -> bool {
        self.chains.iter().all(|c| c.validate(skeleton))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Bone, Skeleton, Transform};
    use glam::Vec3;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    // Helper to create a test skeleton with given bone count
    fn create_test_skeleton(bone_count: usize) -> Skeleton {
        let mut skeleton = Skeleton::new();

        // Root bone
        skeleton.add_bone(Bone {
            name: "root".to_string(),
            parent_index: None,
            local_transform: Transform::IDENTITY,
            inverse_bind_matrix: glam::Mat4::IDENTITY,
        });

        // Add remaining bones as children of root
        for i in 1..bone_count {
            skeleton.add_bone(Bone {
                name: format!("bone_{}", i),
                parent_index: Some(0),
                local_transform: Transform::from_position(Vec3::new(0.0, 1.0 * i as f32, 0.0)),
                inverse_bind_matrix: glam::Mat4::IDENTITY,
            });
        }

        skeleton
    }

    // Helper to create a linked skeleton (proper parent chain)
    fn create_linked_skeleton(bone_count: usize) -> Skeleton {
        let mut skeleton = Skeleton::new();

        // Root bone
        skeleton.add_bone(Bone {
            name: "root".to_string(),
            parent_index: None,
            local_transform: Transform::IDENTITY,
            inverse_bind_matrix: glam::Mat4::IDENTITY,
        });

        // Add bones as a chain
        for i in 1..bone_count {
            skeleton.add_bone(Bone {
                name: format!("bone_{}", i),
                parent_index: Some(i - 1),
                local_transform: Transform::from_position(Vec3::new(0.0, 1.0, 0.0)),
                inverse_bind_matrix: glam::Mat4::IDENTITY,
            });
        }

        skeleton
    }

    // ---------------------------------------------------------------------------
    // Swing-Twist Decomposition Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_swing_twist_decompose_identity() {
        let (swing, twist) = swing_twist_decompose(Quat::IDENTITY, Vec3::X);

        assert!(swing.abs_diff_eq(Quat::IDENTITY, 1e-5));
        assert!(twist.abs_diff_eq(Quat::IDENTITY, 1e-5));
    }

    #[test]
    fn test_swing_twist_decompose_pure_twist() {
        // Rotation purely around X axis (twist axis)
        let angle = FRAC_PI_4;
        let rotation = Quat::from_axis_angle(Vec3::X, angle);

        let (swing, twist) = swing_twist_decompose(rotation, Vec3::X);

        // Swing should be identity, twist should be the full rotation
        assert!(swing.abs_diff_eq(Quat::IDENTITY, 1e-4), "swing = {:?}", swing);

        let (twist_axis, twist_angle) = twist.to_axis_angle();
        assert!((twist_angle.abs() - angle.abs()).abs() < 1e-4,
            "expected angle {}, got {}", angle, twist_angle);
    }

    #[test]
    fn test_swing_twist_decompose_pure_swing() {
        // Rotation around Y axis (perpendicular to twist axis X)
        let angle = FRAC_PI_4;
        let rotation = Quat::from_axis_angle(Vec3::Y, angle);

        let (swing, twist) = swing_twist_decompose(rotation, Vec3::X);

        // Twist should be identity, swing should be the full rotation
        assert!(twist.abs_diff_eq(Quat::IDENTITY, 1e-4), "twist = {:?}", twist);

        // Swing * Twist should equal original
        let reconstructed = (swing * twist).normalize();
        assert!(reconstructed.abs_diff_eq(rotation, 1e-4),
            "reconstructed {:?} != original {:?}", reconstructed, rotation);
    }

    #[test]
    fn test_swing_twist_decompose_combined() {
        // Combined rotation: twist around X, then swing around Y
        let twist_angle = FRAC_PI_4;
        let swing_angle = PI / 6.0;

        let twist_rot = Quat::from_axis_angle(Vec3::X, twist_angle);
        let swing_rot = Quat::from_axis_angle(Vec3::Y, swing_angle);
        let combined = (swing_rot * twist_rot).normalize();

        let (swing, twist) = swing_twist_decompose(combined, Vec3::X);

        // Reconstruction should match original
        let reconstructed = (swing * twist).normalize();
        assert!(reconstructed.abs_diff_eq(combined, 1e-3),
            "reconstructed {:?} != original {:?}", reconstructed, combined);
    }

    #[test]
    fn test_swing_twist_decompose_180_degrees() {
        // 180 degree twist
        let rotation = Quat::from_axis_angle(Vec3::X, PI);

        let (swing, twist) = swing_twist_decompose(rotation, Vec3::X);

        assert!(swing.abs_diff_eq(Quat::IDENTITY, 1e-4), "swing = {:?}", swing);

        let (_axis, twist_angle) = twist.to_axis_angle();
        assert!((twist_angle.abs() - PI).abs() < 1e-3,
            "expected angle PI, got {}", twist_angle);
    }

    #[test]
    fn test_swing_twist_decompose_negative_angle() {
        // Negative twist
        let angle = -FRAC_PI_4;
        let rotation = Quat::from_axis_angle(Vec3::X, angle);

        let (swing, twist) = swing_twist_decompose(rotation, Vec3::X);

        assert!(swing.abs_diff_eq(Quat::IDENTITY, 1e-4));

        // Reconstruction
        let reconstructed = (swing * twist).normalize();
        assert!(reconstructed.abs_diff_eq(rotation, 1e-4));
    }

    #[test]
    fn test_extract_twist() {
        let angle = FRAC_PI_4;
        let rotation = Quat::from_axis_angle(Vec3::X, angle);

        let twist = extract_twist(rotation, Vec3::X);

        let (_axis, twist_angle) = twist.to_axis_angle();
        assert!((twist_angle.abs() - angle.abs()).abs() < 1e-4);
    }

    #[test]
    fn test_get_twist_angle() {
        let angle = FRAC_PI_4;
        let rotation = Quat::from_axis_angle(Vec3::X, angle);

        let extracted = get_twist_angle(rotation, Vec3::X);

        assert!((extracted - angle).abs() < 1e-4,
            "expected {}, got {}", angle, extracted);
    }

    #[test]
    fn test_get_twist_angle_negative() {
        let angle = -FRAC_PI_2;
        let rotation = Quat::from_axis_angle(Vec3::X, angle);

        let extracted = get_twist_angle(rotation, Vec3::X);

        assert!((extracted - angle).abs() < 1e-4,
            "expected {}, got {}", angle, extracted);
    }

    #[test]
    fn test_twist_from_angle() {
        let angle = FRAC_PI_4;
        let twist = twist_from_angle(angle, Vec3::X);

        let expected = Quat::from_axis_angle(Vec3::X, angle);
        assert!(twist.abs_diff_eq(expected, 1e-5));
    }

    #[test]
    fn test_clamp_twist_angle() {
        assert_eq!(clamp_twist_angle(0.5, 1.0), 0.5);
        assert_eq!(clamp_twist_angle(1.5, 1.0), 1.0);
        assert_eq!(clamp_twist_angle(-1.5, 1.0), -1.0);
        assert_eq!(clamp_twist_angle(0.0, 1.0), 0.0);
    }

    // ---------------------------------------------------------------------------
    // TwistFalloff Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_twist_falloff_linear_name() {
        assert_eq!(TwistFalloff::Linear.name(), "Linear");
    }

    #[test]
    fn test_twist_falloff_exponential_name() {
        assert_eq!(TwistFalloff::exponential(0.5).name(), "Exponential");
    }

    #[test]
    fn test_twist_falloff_custom_name() {
        assert_eq!(TwistFalloff::Custom.name(), "Custom");
    }

    #[test]
    fn test_twist_falloff_is_custom() {
        assert!(!TwistFalloff::Linear.is_custom());
        assert!(!TwistFalloff::Exponential { decay: 0.5 }.is_custom());
        assert!(TwistFalloff::Custom.is_custom());
    }

    // ---------------------------------------------------------------------------
    // TwistParams Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_twist_params_default() {
        let params = TwistParams::default();
        assert!(params.enabled);
        assert_eq!(params.falloff, TwistFalloff::Linear);
        assert!((params.max_twist - PI).abs() < 1e-5);
    }

    #[test]
    fn test_twist_params_linear() {
        let params = TwistParams::linear();
        assert!(params.enabled);
        assert_eq!(params.falloff, TwistFalloff::Linear);
    }

    #[test]
    fn test_twist_params_exponential() {
        let params = TwistParams::exponential(0.7);
        assert!(params.enabled);
        assert!(matches!(params.falloff, TwistFalloff::Exponential { decay } if (decay - 0.7).abs() < 1e-5));
    }

    #[test]
    fn test_twist_params_custom() {
        let params = TwistParams::custom();
        assert!(params.enabled);
        assert_eq!(params.falloff, TwistFalloff::Custom);
    }

    #[test]
    fn test_twist_params_with_max_twist() {
        let params = TwistParams::linear().with_max_twist(FRAC_PI_2);
        assert!((params.max_twist - FRAC_PI_2).abs() < 1e-5);
    }

    #[test]
    fn test_twist_params_disabled() {
        let params = TwistParams::disabled();
        assert!(!params.enabled);
    }

    // ---------------------------------------------------------------------------
    // TwistChain Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_twist_chain_new() {
        let chain = TwistChain::new(vec![1, 2, 3], 4, Vec3::X);

        assert_eq!(chain.bone_count(), 3);
        assert_eq!(chain.source_bone, 4);
        assert!(chain.twist_axis.abs_diff_eq(Vec3::X, 1e-5));
        assert!(chain.weights.is_empty());
    }

    #[test]
    fn test_twist_chain_with_custom_weights() {
        let chain = TwistChain::with_custom_weights(
            vec![1, 2],
            vec![0.6, 0.4],
            3,
            Vec3::Y,
        );

        assert_eq!(chain.weights.len(), 2);
        assert!((chain.weights[0] - 0.6).abs() < 1e-5);
        assert!((chain.weights[1] - 0.4).abs() < 1e-5);
    }

    #[test]
    fn test_twist_chain_normalize_weights() {
        let mut chain = TwistChain::new(vec![1, 2], 3, Vec3::X);
        chain.weights = vec![2.0, 2.0];
        chain.normalize_weights();

        assert!((chain.weights[0] - 0.5).abs() < 1e-5);
        assert!((chain.weights[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_twist_chain_normalize_weights_zero() {
        let mut chain = TwistChain::new(vec![1, 2], 3, Vec3::X);
        chain.weights = vec![0.0, 0.0];
        chain.normalize_weights();

        // Should fall back to uniform
        assert!((chain.weights[0] - 0.5).abs() < 1e-5);
        assert!((chain.weights[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_twist_chain_compute_weights_linear() {
        let chain = TwistChain::new(vec![1, 2, 3], 4, Vec3::X);
        let weights = chain.compute_weights(TwistFalloff::Linear);

        assert_eq!(weights.len(), 3);
        for w in &weights {
            assert!((*w - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_twist_chain_compute_weights_exponential() {
        let chain = TwistChain::new(vec![1, 2, 3], 4, Vec3::X);
        let weights = chain.compute_weights(TwistFalloff::exponential(0.5));

        assert_eq!(weights.len(), 3);
        // First bone should have highest weight
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);

        // Sum should be 1.0
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_twist_chain_compute_weights_custom() {
        let chain = TwistChain::with_custom_weights(
            vec![1, 2],
            vec![0.7, 0.3],
            3,
            Vec3::X,
        );
        let weights = chain.compute_weights(TwistFalloff::Custom);

        assert_eq!(weights.len(), 2);
        assert!((weights[0] - 0.7).abs() < 1e-5);
        assert!((weights[1] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_twist_chain_validate_valid() {
        let skeleton = create_test_skeleton(5);
        let chain = TwistChain::new(vec![1, 2], 3, Vec3::X);

        assert!(chain.validate(&skeleton));
    }

    #[test]
    fn test_twist_chain_validate_invalid_source() {
        let skeleton = create_test_skeleton(5);
        let chain = TwistChain::new(vec![1, 2], 10, Vec3::X); // Invalid source

        assert!(!chain.validate(&skeleton));
    }

    #[test]
    fn test_twist_chain_validate_invalid_bone() {
        let skeleton = create_test_skeleton(5);
        let chain = TwistChain::new(vec![1, 10], 3, Vec3::X); // Invalid bone

        assert!(!chain.validate(&skeleton));
    }

    #[test]
    fn test_twist_chain_validate_zero_axis() {
        let skeleton = create_test_skeleton(5);
        let chain = TwistChain::new(vec![1, 2], 3, Vec3::ZERO);

        assert!(!chain.validate(&skeleton));
    }

    #[test]
    fn test_twist_chain_is_empty() {
        let chain_empty = TwistChain::new(vec![], 0, Vec3::X);
        let chain_full = TwistChain::new(vec![1, 2], 0, Vec3::X);

        assert!(chain_empty.is_empty());
        assert!(!chain_full.is_empty());
    }

    // ---------------------------------------------------------------------------
    // Distribute Twist Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_distribute_twist_disabled() {
        let skeleton = create_test_skeleton(5);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = TwistChain::new(vec![1, 2], 3, Vec3::X);
        let params = TwistParams::disabled();

        let applied = distribute_twist(&chain, &skeleton, &mut pose, &params);

        assert!(!applied);
    }

    #[test]
    fn test_distribute_twist_empty_chain() {
        let skeleton = create_test_skeleton(5);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = TwistChain::new(vec![], 3, Vec3::X);
        let params = TwistParams::default();

        let applied = distribute_twist(&chain, &skeleton, &mut pose, &params);

        assert!(!applied);
    }

    #[test]
    fn test_distribute_twist_invalid_chain() {
        let skeleton = create_test_skeleton(5);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = TwistChain::new(vec![1, 100], 3, Vec3::X); // Invalid bone
        let params = TwistParams::default();

        let applied = distribute_twist(&chain, &skeleton, &mut pose, &params);

        assert!(!applied);
    }

    #[test]
    fn test_distribute_twist_basic() {
        let skeleton = create_test_skeleton(5);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Set a twist on the source bone
        let twist_angle = FRAC_PI_4;
        pose.rotations[3] = Quat::from_axis_angle(Vec3::X, twist_angle);

        let chain = TwistChain::new(vec![1, 2], 3, Vec3::X);
        let params = TwistParams::linear();

        let applied = distribute_twist(&chain, &skeleton, &mut pose, &params);

        assert!(applied);

        // Source bone should have twist removed
        let source_twist = get_twist_angle(pose.rotations[3], Vec3::X);
        assert!(source_twist.abs() < 1e-3, "source twist should be ~0, got {}", source_twist);

        // Bones 1 and 2 should have partial twist
        let bone1_twist = get_twist_angle(pose.rotations[1], Vec3::X);
        let bone2_twist = get_twist_angle(pose.rotations[2], Vec3::X);

        assert!(bone1_twist.abs() > 0.01, "bone 1 should have some twist");
        assert!(bone2_twist.abs() > 0.01, "bone 2 should have some twist");
    }

    #[test]
    fn test_distribute_twist_linear_weights() {
        let skeleton = create_test_skeleton(5);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let twist_angle = PI / 3.0; // 60 degrees
        pose.rotations[3] = Quat::from_axis_angle(Vec3::X, twist_angle);

        let chain = TwistChain::new(vec![1, 2], 3, Vec3::X);
        let params = TwistParams::linear();

        distribute_twist(&chain, &skeleton, &mut pose, &params);

        // With linear falloff, each bone gets half
        let bone1_twist = get_twist_angle(pose.rotations[1], Vec3::X);
        let bone2_twist = get_twist_angle(pose.rotations[2], Vec3::X);

        let expected_per_bone = twist_angle / 2.0;
        assert!((bone1_twist - expected_per_bone).abs() < 0.1,
            "bone1 twist {}, expected {}", bone1_twist, expected_per_bone);
        assert!((bone2_twist - expected_per_bone).abs() < 0.1,
            "bone2 twist {}, expected {}", bone2_twist, expected_per_bone);
    }

    #[test]
    fn test_distribute_twist_exponential_weights() {
        let skeleton = create_test_skeleton(5);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let twist_angle = FRAC_PI_2;
        pose.rotations[3] = Quat::from_axis_angle(Vec3::X, twist_angle);

        let chain = TwistChain::new(vec![1, 2], 3, Vec3::X);
        let params = TwistParams::exponential(0.5);

        distribute_twist(&chain, &skeleton, &mut pose, &params);

        let bone1_twist = get_twist_angle(pose.rotations[1], Vec3::X);
        let bone2_twist = get_twist_angle(pose.rotations[2], Vec3::X);

        // First bone should have more twist than second
        assert!(bone1_twist.abs() > bone2_twist.abs(),
            "bone1 twist {} should be > bone2 twist {}", bone1_twist, bone2_twist);
    }

    #[test]
    fn test_distribute_twist_max_clamp() {
        let skeleton = create_test_skeleton(5);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Large twist that exceeds max
        let twist_angle = PI * 1.5;
        pose.rotations[3] = Quat::from_axis_angle(Vec3::X, twist_angle);

        let chain = TwistChain::new(vec![1], 3, Vec3::X);
        let params = TwistParams::linear().with_max_twist(FRAC_PI_2); // Max 90 degrees

        distribute_twist(&chain, &skeleton, &mut pose, &params);

        let bone1_twist = get_twist_angle(pose.rotations[1], Vec3::X);

        // Should be clamped to max
        assert!(bone1_twist.abs() <= FRAC_PI_2 + 0.01,
            "twist {} should be <= max {}", bone1_twist, FRAC_PI_2);
    }

    #[test]
    fn test_distribute_twist_zero_twist() {
        let skeleton = create_test_skeleton(5);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // No twist on source
        pose.rotations[3] = Quat::IDENTITY;

        let chain = TwistChain::new(vec![1, 2], 3, Vec3::X);
        let params = TwistParams::default();

        let original_bone1 = pose.rotations[1];
        let original_bone2 = pose.rotations[2];

        distribute_twist(&chain, &skeleton, &mut pose, &params);

        // Bones should be essentially unchanged
        assert!(pose.rotations[1].abs_diff_eq(original_bone1, 1e-4));
        assert!(pose.rotations[2].abs_diff_eq(original_bone2, 1e-4));
    }

    #[test]
    fn test_distribute_twist_preserves_swing() {
        let skeleton = create_test_skeleton(5);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Source has both swing and twist
        let swing = Quat::from_axis_angle(Vec3::Y, FRAC_PI_4);
        let twist = Quat::from_axis_angle(Vec3::X, FRAC_PI_4);
        pose.rotations[3] = (swing * twist).normalize();

        let chain = TwistChain::new(vec![1], 3, Vec3::X);
        let params = TwistParams::default();

        distribute_twist(&chain, &skeleton, &mut pose, &params);

        // Source should retain swing but lose twist
        let (remaining_swing, remaining_twist) = swing_twist_decompose(pose.rotations[3], Vec3::X);

        assert!(remaining_swing.abs_diff_eq(swing, 0.1),
            "swing {:?} should be preserved, got {:?}", swing, remaining_swing);
        assert!(remaining_twist.abs_diff_eq(Quat::IDENTITY, 0.1),
            "twist should be removed, got {:?}", remaining_twist);
    }

    #[test]
    fn test_distribute_twist_additive() {
        let skeleton = create_test_skeleton(5);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let twist_angle = FRAC_PI_4;
        pose.rotations[3] = Quat::from_axis_angle(Vec3::X, twist_angle);

        let original_source = pose.rotations[3];

        let chain = TwistChain::new(vec![1, 2], 3, Vec3::X);
        let params = TwistParams::linear();

        let applied = distribute_twist_additive(&chain, &skeleton, &mut pose, &params);

        assert!(applied);

        // Source should be unchanged in additive mode
        assert!(pose.rotations[3].abs_diff_eq(original_source, 1e-5));

        // Bones should still have twist applied
        let bone1_twist = get_twist_angle(pose.rotations[1], Vec3::X);
        assert!(bone1_twist.abs() > 0.01);
    }

    // ---------------------------------------------------------------------------
    // Chain Factory Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_create_forearm_twist_chain() {
        let chain = create_forearm_twist_chain(1, 2, 3);

        assert_eq!(chain.bones, vec![2, 1]); // forearm, upper_arm
        assert_eq!(chain.source_bone, 3);
        assert!(chain.twist_axis.abs_diff_eq(Vec3::X, 1e-5));
        assert_eq!(chain.weights.len(), 2);
    }

    #[test]
    fn test_create_forearm_twist_chain_with_helpers() {
        let chain = create_forearm_twist_chain_with_helpers(&[10, 11, 12], 13, Vec3::Y);

        assert_eq!(chain.bones, vec![10, 11, 12]);
        assert_eq!(chain.source_bone, 13);
        assert!(chain.twist_axis.abs_diff_eq(Vec3::Y, 1e-5));
    }

    #[test]
    fn test_create_spine_twist_chain() {
        let chain = create_spine_twist_chain(0, &[1, 2, 3], 4);

        assert_eq!(chain.bones, vec![1, 2, 3, 4]);
        assert_eq!(chain.source_bone, 0);
        assert!(chain.twist_axis.abs_diff_eq(Vec3::Y, 1e-5));
    }

    #[test]
    fn test_create_neck_twist_chain() {
        let chain = create_neck_twist_chain(10, &[8, 9]);

        assert_eq!(chain.bones, vec![8, 9]);
        assert_eq!(chain.source_bone, 10);
    }

    #[test]
    fn test_create_upper_leg_twist_chain() {
        let chain = create_upper_leg_twist_chain(5, 4);

        assert_eq!(chain.bones, vec![5]);
        assert_eq!(chain.source_bone, 4);
    }

    #[test]
    fn test_create_lower_leg_twist_chain() {
        let chain = create_lower_leg_twist_chain(6, 7);

        assert_eq!(chain.bones, vec![6]);
        assert_eq!(chain.source_bone, 7);
    }

    // ---------------------------------------------------------------------------
    // TwistChainManager Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_twist_chain_manager_new() {
        let manager = TwistChainManager::new();

        assert_eq!(manager.chain_count(), 0);
        assert!(manager.params.enabled);
    }

    #[test]
    fn test_twist_chain_manager_add_chain() {
        let mut manager = TwistChainManager::new();
        manager.add_chain(create_forearm_twist_chain(1, 2, 3));

        assert_eq!(manager.chain_count(), 1);
    }

    #[test]
    fn test_twist_chain_manager_apply_all() {
        let skeleton = create_test_skeleton(10);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Add twist to two source bones
        pose.rotations[3] = Quat::from_axis_angle(Vec3::X, FRAC_PI_4);
        pose.rotations[6] = Quat::from_axis_angle(Vec3::X, FRAC_PI_4);

        let mut manager = TwistChainManager::new();
        manager.add_chain(TwistChain::new(vec![1, 2], 3, Vec3::X));
        manager.add_chain(TwistChain::new(vec![4, 5], 6, Vec3::X));

        let applied = manager.apply_all(&skeleton, &mut pose);

        assert_eq!(applied, 2);
    }

    #[test]
    fn test_twist_chain_manager_apply_all_additive() {
        let skeleton = create_test_skeleton(10);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        pose.rotations[3] = Quat::from_axis_angle(Vec3::X, FRAC_PI_4);
        let original_source = pose.rotations[3];

        let mut manager = TwistChainManager::new();
        manager.add_chain(TwistChain::new(vec![1, 2], 3, Vec3::X));

        let applied = manager.apply_all_additive(&skeleton, &mut pose);

        assert_eq!(applied, 1);
        // Source should be unchanged
        assert!(pose.rotations[3].abs_diff_eq(original_source, 1e-5));
    }

    #[test]
    fn test_twist_chain_manager_create_humanoid() {
        let manager = TwistChainManager::create_humanoid(
            (1, 2, 3),  // Left arm
            (4, 5, 6),  // Right arm
            Some((0, &[7, 8], 9)),  // Spine
            Some((10, &[11, 12])),  // Neck
        );

        assert_eq!(manager.chain_count(), 4); // 2 arms + spine + neck
    }

    #[test]
    fn test_twist_chain_manager_create_humanoid_minimal() {
        let manager = TwistChainManager::create_humanoid(
            (1, 2, 3),
            (4, 5, 6),
            None,  // No spine
            None,  // No neck
        );

        assert_eq!(manager.chain_count(), 2); // Just arms
    }

    #[test]
    fn test_twist_chain_manager_set_enabled() {
        let mut manager = TwistChainManager::new();
        assert!(manager.params.enabled);

        manager.set_enabled(false);
        assert!(!manager.params.enabled);

        manager.set_enabled(true);
        assert!(manager.params.enabled);
    }

    #[test]
    fn test_twist_chain_manager_set_falloff() {
        let mut manager = TwistChainManager::new();

        manager.set_falloff(TwistFalloff::exponential(0.7));
        assert!(matches!(manager.params.falloff, TwistFalloff::Exponential { .. }));
    }

    #[test]
    fn test_twist_chain_manager_validate() {
        let skeleton = create_test_skeleton(10);

        let mut manager = TwistChainManager::new();
        manager.add_chain(TwistChain::new(vec![1, 2], 3, Vec3::X));

        assert!(manager.validate(&skeleton));

        // Add invalid chain
        manager.add_chain(TwistChain::new(vec![100], 3, Vec3::X));

        assert!(!manager.validate(&skeleton));
    }

    // ---------------------------------------------------------------------------
    // Edge Case Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_distribute_twist_single_bone_chain() {
        let skeleton = create_test_skeleton(5);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        pose.rotations[2] = Quat::from_axis_angle(Vec3::X, FRAC_PI_4);

        let chain = TwistChain::new(vec![1], 2, Vec3::X);
        let params = TwistParams::linear();

        let applied = distribute_twist(&chain, &skeleton, &mut pose, &params);

        assert!(applied);

        // Single bone gets all the twist
        let bone_twist = get_twist_angle(pose.rotations[1], Vec3::X);
        assert!((bone_twist - FRAC_PI_4).abs() < 0.1);
    }

    #[test]
    fn test_distribute_twist_degenerate_axis_y() {
        let skeleton = create_test_skeleton(5);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Twist around Y axis
        pose.rotations[3] = Quat::from_axis_angle(Vec3::Y, FRAC_PI_4);

        let chain = TwistChain::new(vec![1, 2], 3, Vec3::Y);
        let params = TwistParams::linear();

        let applied = distribute_twist(&chain, &skeleton, &mut pose, &params);

        assert!(applied);
    }

    #[test]
    fn test_distribute_twist_degenerate_axis_z() {
        let skeleton = create_test_skeleton(5);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        pose.rotations[3] = Quat::from_axis_angle(Vec3::Z, FRAC_PI_4);

        let chain = TwistChain::new(vec![1, 2], 3, Vec3::Z);
        let params = TwistParams::linear();

        let applied = distribute_twist(&chain, &skeleton, &mut pose, &params);

        assert!(applied);
    }

    #[test]
    fn test_swing_twist_roundtrip() {
        // Test that swing * twist = original for various rotations
        let test_rotations = [
            Quat::from_axis_angle(Vec3::X, 0.5),
            Quat::from_axis_angle(Vec3::Y, 0.5),
            Quat::from_axis_angle(Vec3::Z, 0.5),
            Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), 0.7),
            Quat::from_axis_angle(Vec3::new(1.0, 1.0, 1.0).normalize(), 1.0),
        ];

        for rotation in &test_rotations {
            let (swing, twist) = swing_twist_decompose(*rotation, Vec3::X);
            let reconstructed = (swing * twist).normalize();

            assert!(reconstructed.abs_diff_eq(*rotation, 1e-3),
                "Failed roundtrip for {:?}: got {:?}", rotation, reconstructed);
        }
    }

    #[test]
    fn test_twist_chain_many_bones() {
        let skeleton = create_test_skeleton(20);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        pose.rotations[15] = Quat::from_axis_angle(Vec3::X, FRAC_PI_2);

        let chain = TwistChain::new((1..15).collect(), 15, Vec3::X);
        let params = TwistParams::linear();

        let applied = distribute_twist(&chain, &skeleton, &mut pose, &params);

        assert!(applied);

        // Each bone should have twist / 14
        let expected_per_bone = FRAC_PI_2 / 14.0;
        for i in 1..15 {
            let twist = get_twist_angle(pose.rotations[i], Vec3::X);
            assert!((twist - expected_per_bone).abs() < 0.1,
                "bone {} twist {} != expected {}", i, twist, expected_per_bone);
        }
    }

    #[test]
    fn test_weight_sum_is_one() {
        let chain = TwistChain::new(vec![1, 2, 3, 4, 5], 6, Vec3::X);

        // Test various falloffs
        let falloffs = [
            TwistFalloff::Linear,
            TwistFalloff::exponential(0.3),
            TwistFalloff::exponential(0.5),
            TwistFalloff::exponential(0.8),
            TwistFalloff::exponential(1.0),
            TwistFalloff::exponential(1.5),
        ];

        for falloff in &falloffs {
            let weights = chain.compute_weights(*falloff);
            let sum: f32 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4,
                "weights for {:?} sum to {}, expected 1.0", falloff, sum);
        }
    }

    #[test]
    fn test_custom_weights_override() {
        let chain = TwistChain::with_custom_weights(
            vec![1, 2, 3],
            vec![0.5, 0.3, 0.2],
            4,
            Vec3::X,
        );

        let weights = chain.compute_weights(TwistFalloff::Custom);

        assert!((weights[0] - 0.5).abs() < 1e-5);
        assert!((weights[1] - 0.3).abs() < 1e-5);
        assert!((weights[2] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_180_degree_twist_distribution() {
        let skeleton = create_test_skeleton(5);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // 180 degree twist (common in forearm supination/pronation)
        pose.rotations[3] = Quat::from_axis_angle(Vec3::X, PI);

        let chain = TwistChain::new(vec![1, 2], 3, Vec3::X);
        let params = TwistParams::linear();

        let applied = distribute_twist(&chain, &skeleton, &mut pose, &params);

        assert!(applied);

        // Each bone should have ~90 degrees
        let bone1_twist = get_twist_angle(pose.rotations[1], Vec3::X);
        let bone2_twist = get_twist_angle(pose.rotations[2], Vec3::X);

        let expected = PI / 2.0;
        assert!((bone1_twist.abs() - expected).abs() < 0.2,
            "bone1 twist {} should be ~{}", bone1_twist, expected);
        assert!((bone2_twist.abs() - expected).abs() < 0.2,
            "bone2 twist {} should be ~{}", bone2_twist, expected);
    }

    #[test]
    fn test_source_bone_not_in_chain() {
        let skeleton = create_test_skeleton(10);
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        // Source bone (8) is not in the chain bones (1, 2, 3)
        pose.rotations[8] = Quat::from_axis_angle(Vec3::X, FRAC_PI_4);

        let chain = TwistChain::new(vec![1, 2, 3], 8, Vec3::X);
        let params = TwistParams::linear();

        let applied = distribute_twist(&chain, &skeleton, &mut pose, &params);

        assert!(applied);

        // Source twist should be removed
        let source_twist = get_twist_angle(pose.rotations[8], Vec3::X);
        assert!(source_twist.abs() < 0.01);
    }
}
