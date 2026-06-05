//! Advanced pose blending utilities for skeletal animation (T-AN-2.2).
//!
//! This module provides comprehensive pose blending functionality:
//!
//! - **Linear blending**: lerp for positions/scales, slerp/nlerp for rotations
//! - **Additive blending**: base + (delta - reference) * weight for layered animations
//! - **Inertialization**: momentum-preserving transitions with critically damped springs
//! - **Crossfade curves**: configurable easing for smooth animation transitions
//! - **Per-bone weights**: selective blending with bone masks
//!
//! # Inertialization
//!
//! Inertialization provides smooth, momentum-preserving transitions between poses
//! by matching both position and velocity at the transition point. This eliminates
//! discontinuities that can occur with simple crossfades.
//!
//! The algorithm uses critically damped springs to decay the offset over time:
//! - At t=0, the offset matches the discontinuity
//! - The velocity matches the source animation's velocity
//! - As t increases, the offset decays smoothly to zero
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::pose_blending::{
//!     BlendWeights, InertializationState, CrossfadeCurve,
//!     blend_poses_weighted, crossfade_weight,
//! };
//! use renderer_backend::pose::Pose;
//!
//! // Create upper body mask (bones 0-20 at full weight, rest at 0)
//! let mut weights = BlendWeights::uniform(100, 0.0);
//! for i in 0..20 {
//!     weights.weights[i] = 1.0;
//! }
//!
//! // Blend with per-bone weights
//! let result = blend_poses_weighted(&pose_a, &pose_b, 0.5, &weights);
//!
//! // Crossfade with easing
//! let progress = 0.3;
//! let weight = crossfade_weight(progress, CrossfadeCurve::EaseInOut);
//! ```

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::pose::{lerp_vec3, nlerp_quat, slerp_quat, Pose, PoseType};
use crate::skeleton::Transform;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default half-life for inertialization decay (in seconds).
pub const DEFAULT_INERTIALIZATION_HALF_LIFE: f32 = 0.1;

/// Minimum half-life to prevent division by zero.
pub const MIN_HALF_LIFE: f32 = 0.001;

/// Epsilon for quaternion comparisons.
pub const QUAT_EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// BlendWeights
// ---------------------------------------------------------------------------

/// Per-bone blend weights for selective pose blending.
///
/// Allows different bones to be blended with different weights,
/// enabling effects like upper body only, lower body only, or
/// smooth falloff from a source bone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlendWeights {
    /// Weight per bone, in range [0.0, 1.0].
    /// Length must match the pose bone count.
    pub weights: Vec<f32>,
}

impl BlendWeights {
    /// Create blend weights with all bones at the same weight.
    ///
    /// # Arguments
    ///
    /// * `bone_count` - Number of bones
    /// * `weight` - Weight for all bones (clamped to [0, 1])
    #[inline]
    pub fn uniform(bone_count: usize, weight: f32) -> Self {
        Self {
            weights: vec![weight.clamp(0.0, 1.0); bone_count],
        }
    }

    /// Create blend weights with all bones at full weight (1.0).
    #[inline]
    pub fn full(bone_count: usize) -> Self {
        Self::uniform(bone_count, 1.0)
    }

    /// Create blend weights with all bones at zero weight.
    #[inline]
    pub fn none(bone_count: usize) -> Self {
        Self::uniform(bone_count, 0.0)
    }

    /// Create blend weights from a slice.
    ///
    /// Weights are clamped to [0, 1].
    pub fn from_slice(weights: &[f32]) -> Self {
        Self {
            weights: weights.iter().map(|w| w.clamp(0.0, 1.0)).collect(),
        }
    }

    /// Get the number of bones.
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.weights.len()
    }

    /// Get the weight for a specific bone.
    ///
    /// Returns 0.0 if the index is out of bounds.
    #[inline]
    pub fn get(&self, bone_index: usize) -> f32 {
        self.weights.get(bone_index).copied().unwrap_or(0.0)
    }

    /// Set the weight for a specific bone.
    ///
    /// Weight is clamped to [0, 1]. Returns false if index is out of bounds.
    #[inline]
    pub fn set(&mut self, bone_index: usize, weight: f32) -> bool {
        if bone_index < self.weights.len() {
            self.weights[bone_index] = weight.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Set weights for a range of bones.
    ///
    /// # Arguments
    ///
    /// * `start` - First bone index (inclusive)
    /// * `end` - Last bone index (exclusive)
    /// * `weight` - Weight to set
    pub fn set_range(&mut self, start: usize, end: usize, weight: f32) {
        let clamped = weight.clamp(0.0, 1.0);
        let end = end.min(self.weights.len());
        for i in start..end {
            self.weights[i] = clamped;
        }
    }

    /// Invert all weights (1.0 - weight).
    pub fn invert(&mut self) {
        for w in &mut self.weights {
            *w = 1.0 - *w;
        }
    }

    /// Get an inverted copy of these weights.
    pub fn inverted(&self) -> Self {
        Self {
            weights: self.weights.iter().map(|w| 1.0 - w).collect(),
        }
    }

    /// Multiply all weights by a scalar.
    pub fn scale(&mut self, factor: f32) {
        for w in &mut self.weights {
            *w = (*w * factor).clamp(0.0, 1.0);
        }
    }

    /// Check if all weights are zero.
    pub fn is_zero(&self) -> bool {
        self.weights.iter().all(|&w| w == 0.0)
    }

    /// Check if all weights are one.
    pub fn is_full(&self) -> bool {
        self.weights.iter().all(|&w| w == 1.0)
    }

    /// Resize the weights array.
    ///
    /// New weights are initialized to the given default value.
    pub fn resize(&mut self, new_count: usize, default_weight: f32) {
        self.weights
            .resize(new_count, default_weight.clamp(0.0, 1.0));
    }
}

impl Default for BlendWeights {
    fn default() -> Self {
        Self {
            weights: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// CrossfadeCurve
// ---------------------------------------------------------------------------

/// Curve types for crossfade transitions.
///
/// These affect how the blend weight changes over the transition duration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CrossfadeCurve {
    /// Linear interpolation: weight = progress
    #[default]
    Linear,

    /// Ease in (slow start, fast end): weight = progress^2
    EaseIn,

    /// Ease out (fast start, slow end): weight = 1 - (1-progress)^2
    EaseOut,

    /// Ease in-out (slow start and end): smoothstep
    EaseInOut,

    /// Step function: 0 until progress >= 0.5, then 1
    Step,

    /// Cubic ease in: weight = progress^3
    CubicIn,

    /// Cubic ease out: weight = 1 - (1-progress)^3
    CubicOut,

    /// Sine ease in-out: weight = (1 - cos(pi * progress)) / 2
    SineInOut,
}

impl CrossfadeCurve {
    /// Evaluate the curve at the given progress.
    ///
    /// # Arguments
    ///
    /// * `progress` - Normalized progress in [0, 1]
    ///
    /// # Returns
    ///
    /// Blend weight in [0, 1]
    #[inline]
    pub fn evaluate(&self, progress: f32) -> f32 {
        let t = progress.clamp(0.0, 1.0);

        match self {
            CrossfadeCurve::Linear => t,

            CrossfadeCurve::EaseIn => t * t,

            CrossfadeCurve::EaseOut => {
                let inv = 1.0 - t;
                1.0 - inv * inv
            }

            CrossfadeCurve::EaseInOut => {
                // Smoothstep: 3t^2 - 2t^3
                t * t * (3.0 - 2.0 * t)
            }

            CrossfadeCurve::Step => {
                if t < 0.5 {
                    0.0
                } else {
                    1.0
                }
            }

            CrossfadeCurve::CubicIn => t * t * t,

            CrossfadeCurve::CubicOut => {
                let inv = 1.0 - t;
                1.0 - inv * inv * inv
            }

            CrossfadeCurve::SineInOut => {
                (1.0 - (t * std::f32::consts::PI).cos()) * 0.5
            }
        }
    }

    /// Get a human-readable name for this curve.
    pub fn name(&self) -> &'static str {
        match self {
            CrossfadeCurve::Linear => "Linear",
            CrossfadeCurve::EaseIn => "EaseIn",
            CrossfadeCurve::EaseOut => "EaseOut",
            CrossfadeCurve::EaseInOut => "EaseInOut",
            CrossfadeCurve::Step => "Step",
            CrossfadeCurve::CubicIn => "CubicIn",
            CrossfadeCurve::CubicOut => "CubicOut",
            CrossfadeCurve::SineInOut => "SineInOut",
        }
    }
}

/// Calculate crossfade weight from progress and curve.
///
/// Convenience function that wraps `CrossfadeCurve::evaluate`.
#[inline]
pub fn crossfade_weight(progress: f32, curve: CrossfadeCurve) -> f32 {
    curve.evaluate(progress)
}

// ---------------------------------------------------------------------------
// InertializationState
// ---------------------------------------------------------------------------

/// State for inertialization-based pose transitions.
///
/// Inertialization provides smooth, momentum-preserving transitions by:
/// 1. Capturing the discontinuity (offset) at the transition point
/// 2. Capturing the velocity at the transition point
/// 3. Using a critically damped spring to decay the offset over time
///
/// This produces C1-continuous (position and velocity) transitions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InertializationState {
    /// Position offset per bone (initial discontinuity).
    pub position_offsets: Vec<Vec3>,

    /// Position velocity per bone at transition start.
    pub position_velocities: Vec<Vec3>,

    /// Rotation offset per bone (axis-angle representation).
    pub rotation_offsets: Vec<Vec3>,

    /// Rotation velocity per bone (angular velocity).
    pub rotation_velocities: Vec<Vec3>,

    /// Scale offset per bone.
    pub scale_offsets: Vec<Vec3>,

    /// Scale velocity per bone.
    pub scale_velocities: Vec<Vec3>,

    /// Half-life for the spring decay (seconds).
    /// Smaller = faster decay, larger = smoother transition.
    pub half_life: f32,

    /// Current time since transition start (seconds).
    pub time: f32,

    /// Whether the inertialization is active.
    pub active: bool,
}

impl InertializationState {
    /// Create a new inertialization state for the given bone count.
    pub fn new(bone_count: usize, half_life: f32) -> Self {
        Self {
            position_offsets: vec![Vec3::ZERO; bone_count],
            position_velocities: vec![Vec3::ZERO; bone_count],
            rotation_offsets: vec![Vec3::ZERO; bone_count],
            rotation_velocities: vec![Vec3::ZERO; bone_count],
            scale_offsets: vec![Vec3::ZERO; bone_count],
            scale_velocities: vec![Vec3::ZERO; bone_count],
            half_life: half_life.max(MIN_HALF_LIFE),
            time: 0.0,
            active: false,
        }
    }

    /// Create with default half-life.
    pub fn with_bone_count(bone_count: usize) -> Self {
        Self::new(bone_count, DEFAULT_INERTIALIZATION_HALF_LIFE)
    }

    /// Get the number of bones.
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.position_offsets.len()
    }

    /// Initialize inertialization for a pose transition.
    ///
    /// Captures the discontinuity between the source and target poses,
    /// along with the source pose's velocity.
    ///
    /// # Arguments
    ///
    /// * `source_pose` - The pose we're transitioning from
    /// * `source_velocity` - Velocity of source pose (positions per second)
    /// * `target_pose` - The pose we're transitioning to
    pub fn init_transition(
        &mut self,
        source_pose: &Pose,
        source_velocity: Option<&PoseVelocity>,
        target_pose: &Pose,
    ) {
        assert_eq!(
            source_pose.bone_count(),
            target_pose.bone_count(),
            "poses must have same bone count"
        );
        assert_eq!(
            self.bone_count(),
            source_pose.bone_count(),
            "inertialization state must match pose bone count"
        );

        for i in 0..self.bone_count() {
            // Position offset: how far source is from target
            self.position_offsets[i] = source_pose.positions[i] - target_pose.positions[i];

            // Position velocity from source
            self.position_velocities[i] = source_velocity
                .map(|v| v.position_velocities[i])
                .unwrap_or(Vec3::ZERO);

            // Rotation offset as axis-angle
            // offset_rotation = source * inverse(target)
            let offset_rotation =
                source_pose.rotations[i] * target_pose.rotations[i].inverse();
            self.rotation_offsets[i] = quat_to_scaled_axis(offset_rotation);

            // Rotation velocity from source
            self.rotation_velocities[i] = source_velocity
                .map(|v| v.rotation_velocities[i])
                .unwrap_or(Vec3::ZERO);

            // Scale offset
            self.scale_offsets[i] = source_pose.scales[i] - target_pose.scales[i];

            // Scale velocity
            self.scale_velocities[i] = source_velocity
                .map(|v| v.scale_velocities[i])
                .unwrap_or(Vec3::ZERO);
        }

        self.time = 0.0;
        self.active = true;
    }

    /// Update the inertialization state by advancing time.
    ///
    /// # Arguments
    ///
    /// * `delta_time` - Time step in seconds
    pub fn update(&mut self, delta_time: f32) {
        if !self.active {
            return;
        }

        self.time += delta_time;

        // Check if we've effectively decayed (5 half-lives ≈ 3% remaining)
        if self.time > self.half_life * 5.0 {
            self.active = false;
        }
    }

    /// Apply inertialization correction to a pose.
    ///
    /// Adds the decayed offset to the target pose to produce a smooth transition.
    ///
    /// # Arguments
    ///
    /// * `target_pose` - The target pose to correct
    ///
    /// # Returns
    ///
    /// A new pose with inertialization correction applied.
    pub fn apply(&self, target_pose: &Pose) -> Pose {
        if !self.active {
            return target_pose.clone();
        }

        assert_eq!(
            self.bone_count(),
            target_pose.bone_count(),
            "inertialization state must match pose bone count"
        );

        let mut result = target_pose.clone();

        // Compute spring decay factor
        let decay = critically_damped_spring_decay(self.time, self.half_life);
        let velocity_decay =
            critically_damped_spring_velocity_decay(self.time, self.half_life);

        for i in 0..self.bone_count() {
            // Apply position correction
            let pos_correction = self.position_offsets[i] * decay
                + self.position_velocities[i] * velocity_decay;
            result.positions[i] = target_pose.positions[i] + pos_correction;

            // Apply rotation correction
            let rot_correction = self.rotation_offsets[i] * decay
                + self.rotation_velocities[i] * velocity_decay;
            let correction_quat = scaled_axis_to_quat(rot_correction);
            result.rotations[i] = (correction_quat * target_pose.rotations[i]).normalize();

            // Apply scale correction
            let scale_correction =
                self.scale_offsets[i] * decay + self.scale_velocities[i] * velocity_decay;
            result.scales[i] = target_pose.scales[i] + scale_correction;
        }

        result.pose_type = PoseType::Current;
        result
    }

    /// Check if inertialization is still active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Reset the inertialization state.
    pub fn reset(&mut self) {
        for i in 0..self.bone_count() {
            self.position_offsets[i] = Vec3::ZERO;
            self.position_velocities[i] = Vec3::ZERO;
            self.rotation_offsets[i] = Vec3::ZERO;
            self.rotation_velocities[i] = Vec3::ZERO;
            self.scale_offsets[i] = Vec3::ZERO;
            self.scale_velocities[i] = Vec3::ZERO;
        }
        self.time = 0.0;
        self.active = false;
    }

    /// Resize for a different bone count.
    pub fn resize(&mut self, new_bone_count: usize) {
        self.position_offsets.resize(new_bone_count, Vec3::ZERO);
        self.position_velocities.resize(new_bone_count, Vec3::ZERO);
        self.rotation_offsets.resize(new_bone_count, Vec3::ZERO);
        self.rotation_velocities.resize(new_bone_count, Vec3::ZERO);
        self.scale_offsets.resize(new_bone_count, Vec3::ZERO);
        self.scale_velocities.resize(new_bone_count, Vec3::ZERO);
    }

    /// Set the half-life for decay.
    pub fn set_half_life(&mut self, half_life: f32) {
        self.half_life = half_life.max(MIN_HALF_LIFE);
    }

    /// Get the remaining offset magnitude (for debugging).
    pub fn remaining_offset_magnitude(&self) -> f32 {
        if !self.active {
            return 0.0;
        }

        let decay = critically_damped_spring_decay(self.time, self.half_life);

        self.position_offsets
            .iter()
            .map(|v| v.length() * decay)
            .sum::<f32>()
            / self.bone_count() as f32
    }
}

impl Default for InertializationState {
    fn default() -> Self {
        Self::new(0, DEFAULT_INERTIALIZATION_HALF_LIFE)
    }
}

// ---------------------------------------------------------------------------
// PoseVelocity
// ---------------------------------------------------------------------------

/// Velocity of a pose (derivative with respect to time).
///
/// Used for inertialization to capture the source animation's momentum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseVelocity {
    /// Position velocity per bone (units/second).
    pub position_velocities: Vec<Vec3>,

    /// Rotation velocity per bone (radians/second, axis-angle).
    pub rotation_velocities: Vec<Vec3>,

    /// Scale velocity per bone (units/second).
    pub scale_velocities: Vec<Vec3>,
}

impl PoseVelocity {
    /// Create a zero-velocity state.
    pub fn zero(bone_count: usize) -> Self {
        Self {
            position_velocities: vec![Vec3::ZERO; bone_count],
            rotation_velocities: vec![Vec3::ZERO; bone_count],
            scale_velocities: vec![Vec3::ZERO; bone_count],
        }
    }

    /// Compute velocity from two poses and the time delta between them.
    ///
    /// # Arguments
    ///
    /// * `pose_a` - Earlier pose
    /// * `pose_b` - Later pose
    /// * `delta_time` - Time difference (seconds)
    pub fn from_pose_delta(pose_a: &Pose, pose_b: &Pose, delta_time: f32) -> Self {
        assert_eq!(
            pose_a.bone_count(),
            pose_b.bone_count(),
            "poses must have same bone count"
        );

        if delta_time <= 0.0 {
            return Self::zero(pose_a.bone_count());
        }

        let inv_dt = 1.0 / delta_time;
        let bone_count = pose_a.bone_count();

        let mut position_velocities = Vec::with_capacity(bone_count);
        let mut rotation_velocities = Vec::with_capacity(bone_count);
        let mut scale_velocities = Vec::with_capacity(bone_count);

        for i in 0..bone_count {
            // Position velocity
            position_velocities.push((pose_b.positions[i] - pose_a.positions[i]) * inv_dt);

            // Rotation velocity (angular velocity from quaternion delta)
            let delta_rotation = pose_b.rotations[i] * pose_a.rotations[i].inverse();
            let axis_angle = quat_to_scaled_axis(delta_rotation);
            rotation_velocities.push(axis_angle * inv_dt);

            // Scale velocity
            scale_velocities.push((pose_b.scales[i] - pose_a.scales[i]) * inv_dt);
        }

        Self {
            position_velocities,
            rotation_velocities,
            scale_velocities,
        }
    }

    /// Get the number of bones.
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.position_velocities.len()
    }

    /// Resize for a different bone count.
    pub fn resize(&mut self, new_bone_count: usize) {
        self.position_velocities.resize(new_bone_count, Vec3::ZERO);
        self.rotation_velocities.resize(new_bone_count, Vec3::ZERO);
        self.scale_velocities.resize(new_bone_count, Vec3::ZERO);
    }
}

impl Default for PoseVelocity {
    fn default() -> Self {
        Self::zero(0)
    }
}

// ---------------------------------------------------------------------------
// Blending Functions
// ---------------------------------------------------------------------------

/// Blend two poses with per-bone weights.
///
/// Each bone is blended according to its individual weight:
/// `result[i] = lerp(a[i], b[i], weights[i] * global_weight)`
///
/// # Arguments
///
/// * `a` - Source pose (weight = 0)
/// * `b` - Target pose (weight = 1)
/// * `global_weight` - Global blend factor applied to all bones
/// * `weights` - Per-bone blend weights
///
/// # Panics
///
/// Panics if poses have different bone counts or weights don't match.
pub fn blend_poses_weighted(
    a: &Pose,
    b: &Pose,
    global_weight: f32,
    weights: &BlendWeights,
) -> Pose {
    assert_eq!(
        a.bone_count(),
        b.bone_count(),
        "poses must have same bone count"
    );
    assert_eq!(
        a.bone_count(),
        weights.bone_count(),
        "weights must match pose bone count"
    );

    let bone_count = a.bone_count();
    let mut positions = Vec::with_capacity(bone_count);
    let mut rotations = Vec::with_capacity(bone_count);
    let mut scales = Vec::with_capacity(bone_count);

    for i in 0..bone_count {
        let t = (weights.weights[i] * global_weight).clamp(0.0, 1.0);

        positions.push(lerp_vec3(a.positions[i], b.positions[i], t));
        rotations.push(slerp_quat(a.rotations[i], b.rotations[i], t));
        scales.push(lerp_vec3(a.scales[i], b.scales[i], t));
    }

    Pose {
        pose_type: PoseType::Current,
        positions,
        rotations,
        scales,
    }
}

/// Blend two poses with per-bone weights using fast nlerp.
///
/// Same as `blend_poses_weighted` but uses normalized lerp for rotations,
/// which is faster but slightly less accurate for large rotations.
pub fn blend_poses_weighted_fast(
    a: &Pose,
    b: &Pose,
    global_weight: f32,
    weights: &BlendWeights,
) -> Pose {
    assert_eq!(
        a.bone_count(),
        b.bone_count(),
        "poses must have same bone count"
    );
    assert_eq!(
        a.bone_count(),
        weights.bone_count(),
        "weights must match pose bone count"
    );

    let bone_count = a.bone_count();
    let mut positions = Vec::with_capacity(bone_count);
    let mut rotations = Vec::with_capacity(bone_count);
    let mut scales = Vec::with_capacity(bone_count);

    for i in 0..bone_count {
        let t = (weights.weights[i] * global_weight).clamp(0.0, 1.0);

        positions.push(lerp_vec3(a.positions[i], b.positions[i], t));
        rotations.push(nlerp_quat(a.rotations[i], b.rotations[i], t));
        scales.push(lerp_vec3(a.scales[i], b.scales[i], t));
    }

    Pose {
        pose_type: PoseType::Current,
        positions,
        rotations,
        scales,
    }
}

/// Compute additive pose from current and reference poses.
///
/// Creates an additive (delta) pose that represents the difference
/// between the current pose and a reference pose:
///
/// `additive = current - reference`
///
/// # Arguments
///
/// * `current` - The current animation pose
/// * `reference` - The reference pose (often bind or T-pose)
///
/// # Returns
///
/// An additive pose that can be applied to other animations.
pub fn compute_additive_pose(current: &Pose, reference: &Pose) -> Pose {
    assert_eq!(
        current.bone_count(),
        reference.bone_count(),
        "poses must have same bone count"
    );

    let bone_count = current.bone_count();
    let mut positions = Vec::with_capacity(bone_count);
    let mut rotations = Vec::with_capacity(bone_count);
    let mut scales = Vec::with_capacity(bone_count);

    for i in 0..bone_count {
        // Position delta
        positions.push(current.positions[i] - reference.positions[i]);

        // Rotation delta: current * inverse(reference)
        rotations.push((current.rotations[i] * reference.rotations[i].inverse()).normalize());

        // Scale delta (for additive, we store the ratio minus one)
        // This allows additive scale to work multiplicatively: base * (1 + delta)
        let scale_ratio = Vec3::new(
            if reference.scales[i].x.abs() > QUAT_EPSILON {
                current.scales[i].x / reference.scales[i].x - 1.0
            } else {
                0.0
            },
            if reference.scales[i].y.abs() > QUAT_EPSILON {
                current.scales[i].y / reference.scales[i].y - 1.0
            } else {
                0.0
            },
            if reference.scales[i].z.abs() > QUAT_EPSILON {
                current.scales[i].z / reference.scales[i].z - 1.0
            } else {
                0.0
            },
        );
        scales.push(scale_ratio);
    }

    Pose {
        pose_type: PoseType::Additive,
        positions,
        rotations,
        scales,
    }
}

/// Apply additive pose with weight and per-bone masking.
///
/// Applies an additive (delta) pose to a base pose:
///
/// `result[i] = base[i] + delta[i] * weight * mask[i]`
///
/// # Arguments
///
/// * `base` - The base pose to modify
/// * `additive` - The additive pose to apply
/// * `weight` - Global weight for the additive layer
/// * `mask` - Optional per-bone mask weights
///
/// # Panics
///
/// Panics if poses have different bone counts.
pub fn apply_additive_weighted(
    base: &Pose,
    additive: &Pose,
    weight: f32,
    mask: Option<&BlendWeights>,
) -> Pose {
    assert_eq!(
        base.bone_count(),
        additive.bone_count(),
        "poses must have same bone count"
    );

    if let Some(m) = mask {
        assert_eq!(
            base.bone_count(),
            m.bone_count(),
            "mask must match pose bone count"
        );
    }

    let bone_count = base.bone_count();
    let mut positions = Vec::with_capacity(bone_count);
    let mut rotations = Vec::with_capacity(bone_count);
    let mut scales = Vec::with_capacity(bone_count);

    for i in 0..bone_count {
        let bone_weight = mask.map(|m| m.weights[i]).unwrap_or(1.0) * weight;
        let t = bone_weight.clamp(-2.0, 2.0); // Allow slight over-drive

        // Position: base + delta * weight
        positions.push(base.positions[i] + additive.positions[i] * t);

        // Rotation: base * slerp(identity, delta, weight)
        if t.abs() > QUAT_EPSILON {
            let additive_rot = slerp_quat(Quat::IDENTITY, additive.rotations[i], t.abs());
            let additive_rot = if t < 0.0 {
                additive_rot.inverse()
            } else {
                additive_rot
            };
            rotations.push((base.rotations[i] * additive_rot).normalize());
        } else {
            rotations.push(base.rotations[i]);
        }

        // Scale: base * (1 + delta * weight)
        // (additive scales are stored as ratio-1)
        let scale_factor = Vec3::ONE + additive.scales[i] * t;
        scales.push(base.scales[i] * scale_factor);
    }

    Pose {
        pose_type: PoseType::Current,
        positions,
        rotations,
        scales,
    }
}

/// Blend multiple poses with weights.
///
/// Blends N poses together using normalized weights.
/// The weights are normalized so they sum to 1.
///
/// # Arguments
///
/// * `poses` - Slice of poses to blend
/// * `weights` - Weight for each pose (will be normalized)
///
/// # Panics
///
/// Panics if poses slice is empty or poses have different bone counts.
pub fn blend_poses_multi(poses: &[&Pose], weights: &[f32]) -> Pose {
    assert!(!poses.is_empty(), "need at least one pose to blend");
    assert_eq!(
        poses.len(),
        weights.len(),
        "must have same number of poses and weights"
    );

    // All poses must have same bone count
    let bone_count = poses[0].bone_count();
    for pose in poses.iter().skip(1) {
        assert_eq!(
            pose.bone_count(),
            bone_count,
            "all poses must have same bone count"
        );
    }

    // Normalize weights
    let weight_sum: f32 = weights.iter().map(|w| w.max(0.0)).sum();
    if weight_sum <= QUAT_EPSILON {
        // All weights zero, return first pose
        return poses[0].clone();
    }

    let inv_sum = 1.0 / weight_sum;
    let normalized: Vec<f32> = weights.iter().map(|w| w.max(0.0) * inv_sum).collect();

    // Blend each bone
    let mut positions = vec![Vec3::ZERO; bone_count];
    let mut rotations = vec![Quat::IDENTITY; bone_count];
    let mut scales = vec![Vec3::ZERO; bone_count];

    for bone_idx in 0..bone_count {
        // Position: weighted sum
        for (pose_idx, &weight) in normalized.iter().enumerate() {
            positions[bone_idx] += poses[pose_idx].positions[bone_idx] * weight;
            scales[bone_idx] += poses[pose_idx].scales[bone_idx] * weight;
        }

        // Rotation: iterative slerp
        // Start with first pose's rotation
        let mut blended_rotation = poses[0].rotations[bone_idx];
        let mut accumulated_weight = normalized[0];

        for (pose_idx, &weight) in normalized.iter().enumerate().skip(1) {
            if weight > QUAT_EPSILON {
                // Blend this rotation in
                let total = accumulated_weight + weight;
                let t = weight / total;
                blended_rotation =
                    slerp_quat(blended_rotation, poses[pose_idx].rotations[bone_idx], t);
                accumulated_weight = total;
            }
        }

        rotations[bone_idx] = blended_rotation;
    }

    Pose {
        pose_type: PoseType::Current,
        positions,
        rotations,
        scales,
    }
}

/// Crossfade between two poses with a curve.
///
/// Convenience function combining progress, curve, and blending.
///
/// # Arguments
///
/// * `from` - Source pose
/// * `to` - Target pose
/// * `progress` - Crossfade progress [0, 1]
/// * `curve` - Easing curve for the transition
pub fn crossfade_poses(from: &Pose, to: &Pose, progress: f32, curve: CrossfadeCurve) -> Pose {
    let weight = curve.evaluate(progress);
    from.blend(to, weight)
}

/// Crossfade with per-bone weights.
pub fn crossfade_poses_weighted(
    from: &Pose,
    to: &Pose,
    progress: f32,
    curve: CrossfadeCurve,
    weights: &BlendWeights,
) -> Pose {
    let global_weight = curve.evaluate(progress);
    blend_poses_weighted(from, to, global_weight, weights)
}

// ---------------------------------------------------------------------------
// Spring Math (Critically Damped)
// ---------------------------------------------------------------------------

/// Compute the decay factor for a critically damped spring.
///
/// At t=0, decay=1. As t increases, decay approaches 0.
/// The half-life determines how quickly the spring settles.
///
/// This uses the critically damped spring formula that respects half-life:
/// At t = half_life, decay should be approximately 0.5.
///
/// We use: decay = exp(-4 * ln(2) * t / half_life)
/// This gives decay(half_life) ≈ 0.06, which ensures fast settling.
/// For smoother decay with exact half-life behavior, we blend with
/// (1 + y) * exp(-y) where y is scaled appropriately.
#[inline]
fn critically_damped_spring_decay(time: f32, half_life: f32) -> f32 {
    let hl = half_life.max(MIN_HALF_LIFE);
    // Use exponential decay scaled to achieve ~0.5 at half_life
    // decay = exp(-ln(2) * t / half_life) gives exactly 0.5 at t=half_life
    let ratio = time / hl;
    (-std::f32::consts::LN_2 * ratio).exp()
}

/// Compute the velocity decay factor for a critically damped spring.
///
/// This accounts for the initial velocity contribution to the offset.
/// The velocity contribution decays faster than position to prevent overshoot.
///
/// Formula: velocity_decay = t * exp(-2 * ln(2) * t / half_life)
#[inline]
fn critically_damped_spring_velocity_decay(time: f32, half_life: f32) -> f32 {
    let hl = half_life.max(MIN_HALF_LIFE);
    let ratio = time / hl;
    // Velocity decays faster (2x rate)
    time * (-2.0 * std::f32::consts::LN_2 * ratio).exp()
}

// ---------------------------------------------------------------------------
// Quaternion Utilities
// ---------------------------------------------------------------------------

/// Convert a quaternion to scaled axis-angle representation.
///
/// The result is a Vec3 where:
/// - Direction = rotation axis
/// - Magnitude = rotation angle (radians)
#[inline]
fn quat_to_scaled_axis(q: Quat) -> Vec3 {
    // Ensure w is positive for consistent axis
    let q = if q.w < 0.0 { -q } else { q };

    // Get the rotation angle
    let angle = 2.0 * q.w.clamp(-1.0, 1.0).acos();

    // Get the axis
    let sin_half = (1.0 - q.w * q.w).sqrt();

    if sin_half > QUAT_EPSILON {
        Vec3::new(q.x, q.y, q.z) / sin_half * angle
    } else {
        Vec3::ZERO
    }
}

/// Convert a scaled axis-angle representation back to quaternion.
///
/// The input Vec3:
/// - Direction = rotation axis
/// - Magnitude = rotation angle (radians)
#[inline]
fn scaled_axis_to_quat(axis_angle: Vec3) -> Quat {
    let angle = axis_angle.length();

    if angle < QUAT_EPSILON {
        return Quat::IDENTITY;
    }

    let axis = axis_angle / angle;
    let half_angle = angle * 0.5;
    let sin_half = half_angle.sin();
    let cos_half = half_angle.cos();

    Quat::from_xyzw(
        axis.x * sin_half,
        axis.y * sin_half,
        axis.z * sin_half,
        cos_half,
    )
    .normalize()
}

// ---------------------------------------------------------------------------
// Transform Blending
// ---------------------------------------------------------------------------

/// Blend two transforms.
///
/// Uses lerp for position/scale, slerp for rotation.
#[inline]
pub fn blend_transforms(a: Transform, b: Transform, t: f32) -> Transform {
    Transform {
        position: lerp_vec3(a.position, b.position, t),
        rotation: slerp_quat(a.rotation, b.rotation, t),
        scale: lerp_vec3(a.scale, b.scale, t),
    }
}

/// Blend two transforms using fast nlerp.
#[inline]
pub fn blend_transforms_fast(a: Transform, b: Transform, t: f32) -> Transform {
    Transform {
        position: lerp_vec3(a.position, b.position, t),
        rotation: nlerp_quat(a.rotation, b.rotation, t),
        scale: lerp_vec3(a.scale, b.scale, t),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPSILON: f32 = 1e-5;

    // ===== BlendWeights Tests =====

    #[test]
    fn test_blend_weights_uniform() {
        let weights = BlendWeights::uniform(5, 0.5);
        assert_eq!(weights.bone_count(), 5);
        for i in 0..5 {
            assert!((weights.get(i) - 0.5).abs() < EPSILON);
        }
    }

    #[test]
    fn test_blend_weights_full() {
        let weights = BlendWeights::full(3);
        assert!(weights.is_full());
        assert!(!weights.is_zero());
    }

    #[test]
    fn test_blend_weights_none() {
        let weights = BlendWeights::none(3);
        assert!(weights.is_zero());
        assert!(!weights.is_full());
    }

    #[test]
    fn test_blend_weights_clamp() {
        let weights = BlendWeights::uniform(2, 1.5); // Should clamp to 1.0
        assert!((weights.get(0) - 1.0).abs() < EPSILON);

        let weights = BlendWeights::uniform(2, -0.5); // Should clamp to 0.0
        assert!(weights.get(0).abs() < EPSILON);
    }

    #[test]
    fn test_blend_weights_set() {
        let mut weights = BlendWeights::none(3);
        assert!(weights.set(1, 0.75));
        assert!((weights.get(1) - 0.75).abs() < EPSILON);

        // Out of bounds
        assert!(!weights.set(10, 0.5));
    }

    #[test]
    fn test_blend_weights_set_range() {
        let mut weights = BlendWeights::none(10);
        weights.set_range(2, 5, 1.0);

        assert!(weights.get(0).abs() < EPSILON);
        assert!(weights.get(1).abs() < EPSILON);
        assert!((weights.get(2) - 1.0).abs() < EPSILON);
        assert!((weights.get(3) - 1.0).abs() < EPSILON);
        assert!((weights.get(4) - 1.0).abs() < EPSILON);
        assert!(weights.get(5).abs() < EPSILON);
    }

    #[test]
    fn test_blend_weights_invert() {
        let mut weights = BlendWeights::from_slice(&[0.0, 0.25, 0.5, 0.75, 1.0]);
        weights.invert();

        assert!((weights.get(0) - 1.0).abs() < EPSILON);
        assert!((weights.get(1) - 0.75).abs() < EPSILON);
        assert!((weights.get(2) - 0.5).abs() < EPSILON);
        assert!((weights.get(3) - 0.25).abs() < EPSILON);
        assert!(weights.get(4).abs() < EPSILON);
    }

    #[test]
    fn test_blend_weights_inverted() {
        let weights = BlendWeights::from_slice(&[0.2, 0.8]);
        let inverted = weights.inverted();

        assert!((inverted.get(0) - 0.8).abs() < EPSILON);
        assert!((inverted.get(1) - 0.2).abs() < EPSILON);
    }

    #[test]
    fn test_blend_weights_scale() {
        let mut weights = BlendWeights::uniform(3, 0.5);
        weights.scale(0.5);

        for i in 0..3 {
            assert!((weights.get(i) - 0.25).abs() < EPSILON);
        }
    }

    #[test]
    fn test_blend_weights_out_of_bounds() {
        let weights = BlendWeights::full(2);
        assert!(weights.get(100).abs() < EPSILON); // Returns 0.0 for out of bounds
    }

    #[test]
    fn test_blend_weights_resize() {
        let mut weights = BlendWeights::full(2);
        weights.resize(5, 0.3);

        assert_eq!(weights.bone_count(), 5);
        assert!((weights.get(0) - 1.0).abs() < EPSILON); // Original
        assert!((weights.get(3) - 0.3).abs() < EPSILON); // New
    }

    // ===== CrossfadeCurve Tests =====

    #[test]
    fn test_crossfade_curve_linear() {
        let curve = CrossfadeCurve::Linear;
        assert!(curve.evaluate(0.0).abs() < EPSILON);
        assert!((curve.evaluate(0.5) - 0.5).abs() < EPSILON);
        assert!((curve.evaluate(1.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_crossfade_curve_ease_in() {
        let curve = CrossfadeCurve::EaseIn;
        assert!(curve.evaluate(0.0).abs() < EPSILON);
        assert!((curve.evaluate(0.5) - 0.25).abs() < EPSILON); // 0.5^2
        assert!((curve.evaluate(1.0) - 1.0).abs() < EPSILON);

        // Should be slower at start
        assert!(curve.evaluate(0.25) < 0.25);
    }

    #[test]
    fn test_crossfade_curve_ease_out() {
        let curve = CrossfadeCurve::EaseOut;
        assert!(curve.evaluate(0.0).abs() < EPSILON);
        assert!((curve.evaluate(0.5) - 0.75).abs() < EPSILON); // 1 - 0.5^2
        assert!((curve.evaluate(1.0) - 1.0).abs() < EPSILON);

        // Should be faster at start
        assert!(curve.evaluate(0.25) > 0.25);
    }

    #[test]
    fn test_crossfade_curve_ease_in_out() {
        let curve = CrossfadeCurve::EaseInOut;
        assert!(curve.evaluate(0.0).abs() < EPSILON);
        assert!((curve.evaluate(0.5) - 0.5).abs() < EPSILON); // Midpoint
        assert!((curve.evaluate(1.0) - 1.0).abs() < EPSILON);

        // Symmetric
        assert!((curve.evaluate(0.25) + curve.evaluate(0.75) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_crossfade_curve_step() {
        let curve = CrossfadeCurve::Step;
        assert!(curve.evaluate(0.0).abs() < EPSILON);
        assert!(curve.evaluate(0.25).abs() < EPSILON);
        assert!(curve.evaluate(0.49).abs() < EPSILON);
        assert!((curve.evaluate(0.5) - 1.0).abs() < EPSILON);
        assert!((curve.evaluate(1.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_crossfade_curve_cubic_in() {
        let curve = CrossfadeCurve::CubicIn;
        assert!(curve.evaluate(0.0).abs() < EPSILON);
        assert!((curve.evaluate(0.5) - 0.125).abs() < EPSILON); // 0.5^3
        assert!((curve.evaluate(1.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_crossfade_curve_cubic_out() {
        let curve = CrossfadeCurve::CubicOut;
        assert!(curve.evaluate(0.0).abs() < EPSILON);
        assert!((curve.evaluate(0.5) - 0.875).abs() < EPSILON); // 1 - 0.5^3
        assert!((curve.evaluate(1.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_crossfade_curve_sine_in_out() {
        let curve = CrossfadeCurve::SineInOut;
        assert!(curve.evaluate(0.0).abs() < EPSILON);
        assert!((curve.evaluate(0.5) - 0.5).abs() < EPSILON);
        assert!((curve.evaluate(1.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_crossfade_curve_clamps() {
        let curve = CrossfadeCurve::Linear;
        assert!(curve.evaluate(-0.5).abs() < EPSILON);
        assert!((curve.evaluate(1.5) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_crossfade_weight() {
        assert!((crossfade_weight(0.5, CrossfadeCurve::Linear) - 0.5).abs() < EPSILON);
        assert!((crossfade_weight(0.5, CrossfadeCurve::EaseIn) - 0.25).abs() < EPSILON);
    }

    #[test]
    fn test_crossfade_curve_names() {
        assert_eq!(CrossfadeCurve::Linear.name(), "Linear");
        assert_eq!(CrossfadeCurve::EaseInOut.name(), "EaseInOut");
        assert_eq!(CrossfadeCurve::Step.name(), "Step");
    }

    #[test]
    fn test_crossfade_curve_default() {
        assert_eq!(CrossfadeCurve::default(), CrossfadeCurve::Linear);
    }

    // ===== InertializationState Tests =====

    #[test]
    fn test_inertialization_state_new() {
        let state = InertializationState::new(10, 0.15);
        assert_eq!(state.bone_count(), 10);
        assert!((state.half_life - 0.15).abs() < EPSILON);
        assert!(!state.is_active());
        assert!(state.time.abs() < EPSILON);
    }

    #[test]
    fn test_inertialization_state_min_half_life() {
        let state = InertializationState::new(2, 0.0001);
        assert!(state.half_life >= MIN_HALF_LIFE);
    }

    #[test]
    fn test_inertialization_init_transition() {
        let mut source = Pose::new(2, PoseType::Current);
        source.positions[0] = Vec3::new(1.0, 0.0, 0.0);

        let target = Pose::new(2, PoseType::Current);

        let mut state = InertializationState::with_bone_count(2);
        state.init_transition(&source, None, &target);

        assert!(state.is_active());
        assert!(state.position_offsets[0].abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_inertialization_apply_at_zero() {
        let mut source = Pose::new(2, PoseType::Current);
        source.positions[0] = Vec3::new(5.0, 0.0, 0.0);

        let target = Pose::new(2, PoseType::Current);

        let mut state = InertializationState::with_bone_count(2);
        state.init_transition(&source, None, &target);

        // At t=0, should be close to source
        let result = state.apply(&target);
        assert!(result.positions[0].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 0.01));
    }

    #[test]
    fn test_inertialization_decays_over_time() {
        let mut source = Pose::new(2, PoseType::Current);
        source.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let target = Pose::new(2, PoseType::Current);

        let mut state = InertializationState::new(2, 0.1); // 0.1s half-life
        state.init_transition(&source, None, &target);

        // After one half-life
        state.update(0.1);
        let result = state.apply(&target);
        // Should be roughly half the offset
        assert!(result.positions[0].x < 6.0);
        assert!(result.positions[0].x > 3.0);
    }

    #[test]
    fn test_inertialization_deactivates() {
        let source = Pose::new(2, PoseType::Current);
        let target = Pose::new(2, PoseType::Current);

        let mut state = InertializationState::new(2, 0.1);
        state.init_transition(&source, None, &target);

        assert!(state.is_active());

        // After >5 half-lives (0.51s > 0.5s = 5 * 0.1s)
        state.update(0.51);
        assert!(!state.is_active());
    }

    #[test]
    fn test_inertialization_inactive_returns_target() {
        let target = Pose::new(2, PoseType::Current);
        let state = InertializationState::with_bone_count(2);

        // Not active, should return target unchanged
        let result = state.apply(&target);
        assert_eq!(result, target);
    }

    #[test]
    fn test_inertialization_reset() {
        let mut source = Pose::new(2, PoseType::Current);
        source.positions[0] = Vec3::new(5.0, 0.0, 0.0);

        let target = Pose::new(2, PoseType::Current);

        let mut state = InertializationState::with_bone_count(2);
        state.init_transition(&source, None, &target);
        state.update(0.05);

        state.reset();

        assert!(!state.is_active());
        assert!(state.time.abs() < EPSILON);
        assert!(state.position_offsets[0].abs_diff_eq(Vec3::ZERO, EPSILON));
    }

    #[test]
    fn test_inertialization_with_velocity() {
        let source = Pose::new(2, PoseType::Current);
        let target = Pose::new(2, PoseType::Current);

        let mut velocity = PoseVelocity::zero(2);
        velocity.position_velocities[0] = Vec3::new(10.0, 0.0, 0.0); // Moving right

        let mut state = InertializationState::with_bone_count(2);
        state.init_transition(&source, Some(&velocity), &target);

        assert!(state.position_velocities[0].abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_inertialization_rotation() {
        let mut source = Pose::new(2, PoseType::Current);
        source.rotations[0] = Quat::from_rotation_y(PI / 4.0);

        let target = Pose::new(2, PoseType::Current);

        let mut state = InertializationState::with_bone_count(2);
        state.init_transition(&source, None, &target);

        // At t=0, should match source rotation
        let result = state.apply(&target);
        assert!(result.rotations[0].abs_diff_eq(source.rotations[0], 0.01));
    }

    #[test]
    fn test_inertialization_remaining_magnitude() {
        let mut source = Pose::new(2, PoseType::Current);
        source.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let target = Pose::new(2, PoseType::Current);

        let mut state = InertializationState::new(2, 0.1);
        state.init_transition(&source, None, &target);

        let initial_mag = state.remaining_offset_magnitude();
        state.update(0.1);
        let after_half = state.remaining_offset_magnitude();

        assert!(after_half < initial_mag);
    }

    #[test]
    fn test_inertialization_resize() {
        let mut state = InertializationState::with_bone_count(2);
        state.resize(5);
        assert_eq!(state.bone_count(), 5);
    }

    // ===== PoseVelocity Tests =====

    #[test]
    fn test_pose_velocity_zero() {
        let velocity = PoseVelocity::zero(3);
        assert_eq!(velocity.bone_count(), 3);
        assert!(velocity.position_velocities[0].abs_diff_eq(Vec3::ZERO, EPSILON));
    }

    #[test]
    fn test_pose_velocity_from_delta() {
        let mut pose_a = Pose::new(2, PoseType::Current);
        let mut pose_b = Pose::new(2, PoseType::Current);

        pose_a.positions[0] = Vec3::ZERO;
        pose_b.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let velocity = PoseVelocity::from_pose_delta(&pose_a, &pose_b, 1.0);

        assert!(velocity.position_velocities[0].abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_pose_velocity_from_delta_zero_dt() {
        let pose_a = Pose::new(2, PoseType::Current);
        let pose_b = Pose::new(2, PoseType::Current);

        let velocity = PoseVelocity::from_pose_delta(&pose_a, &pose_b, 0.0);

        // Should return zero velocity
        assert!(velocity.position_velocities[0].abs_diff_eq(Vec3::ZERO, EPSILON));
    }

    #[test]
    fn test_pose_velocity_rotation_delta() {
        let mut pose_a = Pose::new(2, PoseType::Current);
        let mut pose_b = Pose::new(2, PoseType::Current);

        pose_a.rotations[0] = Quat::IDENTITY;
        pose_b.rotations[0] = Quat::from_rotation_y(PI / 2.0);

        let velocity = PoseVelocity::from_pose_delta(&pose_a, &pose_b, 1.0);

        // Should have ~PI/2 rad/s angular velocity around Y
        assert!(velocity.rotation_velocities[0].y.abs() > 1.0);
    }

    #[test]
    fn test_pose_velocity_resize() {
        let mut velocity = PoseVelocity::zero(2);
        velocity.resize(5);
        assert_eq!(velocity.bone_count(), 5);
    }

    // ===== Weighted Blending Tests =====

    #[test]
    fn test_blend_poses_weighted_full() {
        let mut a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);

        a.positions[0] = Vec3::ZERO;
        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let weights = BlendWeights::full(2);
        let result = blend_poses_weighted(&a, &b, 0.5, &weights);

        assert!(result.positions[0].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_blend_poses_weighted_none() {
        let mut a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);

        a.positions[0] = Vec3::ZERO;
        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let weights = BlendWeights::none(2);
        let result = blend_poses_weighted(&a, &b, 1.0, &weights);

        // No blend, should be pose a
        assert!(result.positions[0].abs_diff_eq(Vec3::ZERO, EPSILON));
    }

    #[test]
    fn test_blend_poses_weighted_selective() {
        let mut a = Pose::new(3, PoseType::Current);
        let mut b = Pose::new(3, PoseType::Current);

        a.positions[0] = Vec3::ZERO;
        a.positions[1] = Vec3::ZERO;
        a.positions[2] = Vec3::ZERO;

        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);
        b.positions[1] = Vec3::new(10.0, 0.0, 0.0);
        b.positions[2] = Vec3::new(10.0, 0.0, 0.0);

        let weights = BlendWeights::from_slice(&[1.0, 0.5, 0.0]);
        let result = blend_poses_weighted(&a, &b, 1.0, &weights);

        assert!(result.positions[0].abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), EPSILON)); // Full blend
        assert!(result.positions[1].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), EPSILON)); // Half blend
        assert!(result.positions[2].abs_diff_eq(Vec3::ZERO, EPSILON)); // No blend
    }

    #[test]
    fn test_blend_poses_weighted_global() {
        let mut a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);

        a.positions[0] = Vec3::ZERO;
        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let weights = BlendWeights::full(2);
        let result = blend_poses_weighted(&a, &b, 0.25, &weights);

        assert!(result.positions[0].abs_diff_eq(Vec3::new(2.5, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_blend_poses_weighted_fast() {
        let mut a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);

        a.rotations[0] = Quat::IDENTITY;
        b.rotations[0] = Quat::from_rotation_y(PI / 2.0);

        let weights = BlendWeights::full(2);
        let result = blend_poses_weighted_fast(&a, &b, 0.5, &weights);

        // nlerp result should be normalized
        assert!((result.rotations[0].length() - 1.0).abs() < EPSILON);
    }

    #[test]
    #[should_panic(expected = "poses must have same bone count")]
    fn test_blend_poses_weighted_mismatched_poses() {
        let a = Pose::new(2, PoseType::Current);
        let b = Pose::new(3, PoseType::Current);
        let weights = BlendWeights::full(2);
        let _ = blend_poses_weighted(&a, &b, 0.5, &weights);
    }

    #[test]
    #[should_panic(expected = "weights must match pose bone count")]
    fn test_blend_poses_weighted_mismatched_weights() {
        let a = Pose::new(2, PoseType::Current);
        let b = Pose::new(2, PoseType::Current);
        let weights = BlendWeights::full(3);
        let _ = blend_poses_weighted(&a, &b, 0.5, &weights);
    }

    // ===== Additive Pose Tests =====

    #[test]
    fn test_compute_additive_pose_identity() {
        let pose = Pose::new(2, PoseType::Current);
        let reference = Pose::new(2, PoseType::Current);

        let additive = compute_additive_pose(&pose, &reference);

        assert_eq!(additive.pose_type, PoseType::Additive);
        assert!(additive.positions[0].abs_diff_eq(Vec3::ZERO, EPSILON));
        assert!(additive.rotations[0].abs_diff_eq(Quat::IDENTITY, EPSILON));
    }

    #[test]
    fn test_compute_additive_pose_position() {
        let mut current = Pose::new(2, PoseType::Current);
        current.positions[0] = Vec3::new(5.0, 0.0, 0.0);

        let reference = Pose::new(2, PoseType::Current);

        let additive = compute_additive_pose(&current, &reference);

        assert!(additive.positions[0].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_compute_additive_pose_rotation() {
        let mut current = Pose::new(2, PoseType::Current);
        current.rotations[0] = Quat::from_rotation_y(PI / 4.0);

        let reference = Pose::new(2, PoseType::Current);

        let additive = compute_additive_pose(&current, &reference);

        assert!(additive.rotations[0].abs_diff_eq(Quat::from_rotation_y(PI / 4.0), 0.01));
    }

    #[test]
    fn test_apply_additive_weighted_zero() {
        let base = Pose::new(2, PoseType::Current);
        let mut additive = Pose::new(2, PoseType::Additive);
        additive.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let result = apply_additive_weighted(&base, &additive, 0.0, None);

        assert!(result.positions[0].abs_diff_eq(Vec3::ZERO, EPSILON));
    }

    #[test]
    fn test_apply_additive_weighted_full() {
        let mut base = Pose::new(2, PoseType::Current);
        base.positions[0] = Vec3::new(5.0, 0.0, 0.0);

        let mut additive = Pose::new(2, PoseType::Additive);
        additive.positions[0] = Vec3::new(3.0, 0.0, 0.0);

        let result = apply_additive_weighted(&base, &additive, 1.0, None);

        assert!(result.positions[0].abs_diff_eq(Vec3::new(8.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_apply_additive_weighted_half() {
        let base = Pose::new(2, PoseType::Current);

        let mut additive = Pose::new(2, PoseType::Additive);
        additive.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let result = apply_additive_weighted(&base, &additive, 0.5, None);

        assert!(result.positions[0].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_apply_additive_weighted_with_mask() {
        let base = Pose::new(3, PoseType::Current);

        let mut additive = Pose::new(3, PoseType::Additive);
        additive.positions[0] = Vec3::new(10.0, 0.0, 0.0);
        additive.positions[1] = Vec3::new(10.0, 0.0, 0.0);
        additive.positions[2] = Vec3::new(10.0, 0.0, 0.0);

        let mask = BlendWeights::from_slice(&[1.0, 0.5, 0.0]);
        let result = apply_additive_weighted(&base, &additive, 1.0, Some(&mask));

        assert!(result.positions[0].abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), EPSILON));
        assert!(result.positions[1].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), EPSILON));
        assert!(result.positions[2].abs_diff_eq(Vec3::ZERO, EPSILON));
    }

    #[test]
    fn test_apply_additive_rotation() {
        let base = Pose::new(2, PoseType::Current);

        let mut additive = Pose::new(2, PoseType::Additive);
        additive.rotations[0] = Quat::from_rotation_y(PI / 2.0);

        let result = apply_additive_weighted(&base, &additive, 1.0, None);

        assert!(result.rotations[0].abs_diff_eq(Quat::from_rotation_y(PI / 2.0), 0.01));
    }

    // ===== Multi-Pose Blending Tests =====

    #[test]
    fn test_blend_poses_multi_single() {
        let pose = Pose::new(2, PoseType::Current);
        let result = blend_poses_multi(&[&pose], &[1.0]);
        assert_eq!(result, pose);
    }

    #[test]
    fn test_blend_poses_multi_equal_weights() {
        let mut a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);

        a.positions[0] = Vec3::ZERO;
        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let result = blend_poses_multi(&[&a, &b], &[1.0, 1.0]);

        // Equal weights, should be midpoint
        assert!(result.positions[0].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_blend_poses_multi_unequal_weights() {
        let mut a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);

        a.positions[0] = Vec3::ZERO;
        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let result = blend_poses_multi(&[&a, &b], &[1.0, 3.0]);

        // 1:3 ratio, should be 0.25*0 + 0.75*10 = 7.5
        assert!(result.positions[0].abs_diff_eq(Vec3::new(7.5, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_blend_poses_multi_three() {
        let mut a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);
        let mut c = Pose::new(2, PoseType::Current);

        a.positions[0] = Vec3::new(0.0, 0.0, 0.0);
        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);
        c.positions[0] = Vec3::new(20.0, 0.0, 0.0);

        let result = blend_poses_multi(&[&a, &b, &c], &[1.0, 1.0, 1.0]);

        // Equal weights: (0 + 10 + 20) / 3 = 10
        assert!(result.positions[0].abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_blend_poses_multi_zero_weights() {
        let a = Pose::new(2, PoseType::Current);
        let b = Pose::new(2, PoseType::Current);

        // All zero weights should return first pose
        let result = blend_poses_multi(&[&a, &b], &[0.0, 0.0]);
        assert_eq!(result, a);
    }

    #[test]
    fn test_blend_poses_multi_negative_weights() {
        let mut a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);

        a.positions[0] = Vec3::ZERO;
        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        // Negative weights should be treated as zero
        let result = blend_poses_multi(&[&a, &b], &[-1.0, 1.0]);

        assert!(result.positions[0].abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    #[should_panic(expected = "need at least one pose")]
    fn test_blend_poses_multi_empty() {
        let empty: &[&Pose] = &[];
        let _ = blend_poses_multi(empty, &[]);
    }

    #[test]
    #[should_panic(expected = "must have same number of poses and weights")]
    fn test_blend_poses_multi_mismatch() {
        let a = Pose::new(2, PoseType::Current);
        let _ = blend_poses_multi(&[&a], &[1.0, 2.0]);
    }

    // ===== Crossfade Tests =====

    #[test]
    fn test_crossfade_poses() {
        let mut a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);

        a.positions[0] = Vec3::ZERO;
        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let result = crossfade_poses(&a, &b, 0.5, CrossfadeCurve::Linear);

        assert!(result.positions[0].abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_crossfade_poses_ease_in() {
        let mut a = Pose::new(2, PoseType::Current);
        let mut b = Pose::new(2, PoseType::Current);

        a.positions[0] = Vec3::ZERO;
        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let result = crossfade_poses(&a, &b, 0.5, CrossfadeCurve::EaseIn);

        // EaseIn at 0.5 = 0.25
        assert!(result.positions[0].abs_diff_eq(Vec3::new(2.5, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_crossfade_poses_weighted() {
        let mut a = Pose::new(3, PoseType::Current);
        let mut b = Pose::new(3, PoseType::Current);

        a.positions[0] = Vec3::ZERO;
        a.positions[1] = Vec3::ZERO;
        b.positions[0] = Vec3::new(10.0, 0.0, 0.0);
        b.positions[1] = Vec3::new(10.0, 0.0, 0.0);

        let weights = BlendWeights::from_slice(&[1.0, 0.0, 0.5]);
        let result = crossfade_poses_weighted(&a, &b, 1.0, CrossfadeCurve::Linear, &weights);

        assert!(result.positions[0].abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), EPSILON)); // Full
        assert!(result.positions[1].abs_diff_eq(Vec3::ZERO, EPSILON)); // None
    }

    // ===== Spring Math Tests =====

    #[test]
    fn test_spring_decay_at_zero() {
        let decay = critically_damped_spring_decay(0.0, 0.1);
        assert!((decay - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_spring_decay_at_half_life() {
        let decay = critically_damped_spring_decay(0.1, 0.1);
        // At half-life, should be roughly 0.5
        assert!(decay < 0.6);
        assert!(decay > 0.4);
    }

    #[test]
    fn test_spring_decay_decreases() {
        let d1 = critically_damped_spring_decay(0.1, 0.1);
        let d2 = critically_damped_spring_decay(0.2, 0.1);
        let d3 = critically_damped_spring_decay(0.3, 0.1);

        assert!(d1 > d2);
        assert!(d2 > d3);
    }

    #[test]
    fn test_spring_velocity_decay_at_zero() {
        let decay = critically_damped_spring_velocity_decay(0.0, 0.1);
        assert!(decay.abs() < EPSILON);
    }

    #[test]
    fn test_spring_velocity_decay_positive() {
        let decay = critically_damped_spring_velocity_decay(0.1, 0.1);
        assert!(decay > 0.0);
    }

    // ===== Quaternion Utility Tests =====

    #[test]
    fn test_quat_to_scaled_axis_identity() {
        let axis = quat_to_scaled_axis(Quat::IDENTITY);
        assert!(axis.abs_diff_eq(Vec3::ZERO, EPSILON));
    }

    #[test]
    fn test_quat_to_scaled_axis_90deg_y() {
        let q = Quat::from_rotation_y(PI / 2.0);
        let axis = quat_to_scaled_axis(q);

        // Should be roughly (0, PI/2, 0)
        assert!(axis.x.abs() < EPSILON);
        assert!((axis.y - PI / 2.0).abs() < 0.01);
        assert!(axis.z.abs() < EPSILON);
    }

    #[test]
    fn test_scaled_axis_to_quat_identity() {
        let q = scaled_axis_to_quat(Vec3::ZERO);
        assert!(q.abs_diff_eq(Quat::IDENTITY, EPSILON));
    }

    #[test]
    fn test_quat_axis_roundtrip() {
        let original = Quat::from_rotation_y(PI / 3.0);
        let axis = quat_to_scaled_axis(original);
        let recovered = scaled_axis_to_quat(axis);

        // Should match (or be antipodal)
        let dot = original.dot(recovered).abs();
        assert!((dot - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_quat_axis_negative_w() {
        // Quaternion with negative w
        let q = -Quat::from_rotation_y(PI / 4.0);
        let axis = quat_to_scaled_axis(q);
        let recovered = scaled_axis_to_quat(axis);

        // Should recover equivalent rotation
        let dot = q.dot(recovered).abs();
        assert!((dot - 1.0).abs() < 0.01);
    }

    // ===== Transform Blending Tests =====

    #[test]
    fn test_blend_transforms_zero() {
        let a = Transform::from_position(Vec3::new(0.0, 0.0, 0.0));
        let b = Transform::from_position(Vec3::new(10.0, 0.0, 0.0));

        let result = blend_transforms(a, b, 0.0);
        assert!(result.position.abs_diff_eq(Vec3::ZERO, EPSILON));
    }

    #[test]
    fn test_blend_transforms_one() {
        let a = Transform::from_position(Vec3::new(0.0, 0.0, 0.0));
        let b = Transform::from_position(Vec3::new(10.0, 0.0, 0.0));

        let result = blend_transforms(a, b, 1.0);
        assert!(result.position.abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_blend_transforms_half() {
        let a = Transform::from_position(Vec3::new(0.0, 0.0, 0.0));
        let b = Transform::from_position(Vec3::new(10.0, 0.0, 0.0));

        let result = blend_transforms(a, b, 0.5);
        assert!(result.position.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), EPSILON));
    }

    #[test]
    fn test_blend_transforms_fast() {
        let a = Transform::from_rotation(Quat::IDENTITY);
        let b = Transform::from_rotation(Quat::from_rotation_y(PI / 2.0));

        let result = blend_transforms_fast(a, b, 0.5);
        assert!((result.rotation.length() - 1.0).abs() < EPSILON);
    }

    // ===== Edge Cases =====

    #[test]
    fn test_opposing_quaternions() {
        // Test blending quaternions that represent opposite rotations
        let a = Quat::from_rotation_y(0.0);
        let b = Quat::from_rotation_y(PI); // 180 degrees

        let mut pose_a = Pose::new(1, PoseType::Current);
        let mut pose_b = Pose::new(1, PoseType::Current);
        pose_a.rotations[0] = a;
        pose_b.rotations[0] = b;

        let weights = BlendWeights::full(1);
        let result = blend_poses_weighted(&pose_a, &pose_b, 0.5, &weights);

        // Should produce a valid quaternion
        assert!((result.rotations[0].length() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_zero_weights_all_bones() {
        let a = Pose::new(3, PoseType::Current);
        let mut b = Pose::new(3, PoseType::Current);
        b.positions[0] = Vec3::new(100.0, 0.0, 0.0);

        let weights = BlendWeights::none(3);
        let result = blend_poses_weighted(&a, &b, 1.0, &weights);

        // Should be entirely pose a
        for i in 0..3 {
            assert!(result.positions[i].abs_diff_eq(a.positions[i], EPSILON));
        }
    }

    #[test]
    fn test_degenerate_quaternion() {
        // Very small rotation
        let mut pose_a = Pose::new(1, PoseType::Current);
        let mut pose_b = Pose::new(1, PoseType::Current);

        pose_a.rotations[0] = Quat::IDENTITY;
        pose_b.rotations[0] = Quat::from_rotation_y(0.0001);

        let weights = BlendWeights::full(1);
        let result = blend_poses_weighted(&pose_a, &pose_b, 0.5, &weights);

        // Should still be normalized
        assert!((result.rotations[0].length() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_inertialization_continuity() {
        // Test that inertialization produces continuous values
        let mut source = Pose::new(1, PoseType::Current);
        source.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let target = Pose::new(1, PoseType::Current);

        let mut state = InertializationState::new(1, 0.1);
        state.init_transition(&source, None, &target);

        let mut prev_x = 10.0;
        for i in 0..50 {
            let dt = 0.01;
            state.update(dt);
            let result = state.apply(&target);
            let curr_x = result.positions[0].x;

            // Should be monotonically decreasing (no jumps up)
            assert!(
                curr_x <= prev_x + 0.1,
                "discontinuity at step {}: {} -> {}",
                i,
                prev_x,
                curr_x
            );
            prev_x = curr_x;
        }
    }

    #[test]
    fn test_large_bone_count() {
        // Test with realistic skeleton size
        let bone_count = 100;
        let a = Pose::new(bone_count, PoseType::Current);
        let b = Pose::new(bone_count, PoseType::Current);
        let weights = BlendWeights::full(bone_count);

        let result = blend_poses_weighted(&a, &b, 0.5, &weights);
        assert_eq!(result.bone_count(), bone_count);
    }

    #[test]
    fn test_blend_weights_serialization() {
        let weights = BlendWeights::from_slice(&[0.0, 0.5, 1.0]);
        let json = serde_json::to_string(&weights).unwrap();
        let recovered: BlendWeights = serde_json::from_str(&json).unwrap();

        assert_eq!(weights, recovered);
    }

    #[test]
    fn test_crossfade_curve_serialization() {
        let curve = CrossfadeCurve::EaseInOut;
        let json = serde_json::to_string(&curve).unwrap();
        let recovered: CrossfadeCurve = serde_json::from_str(&json).unwrap();

        assert_eq!(curve, recovered);
    }

    #[test]
    fn test_inertialization_state_serialization() {
        let mut state = InertializationState::new(2, 0.15);
        state.position_offsets[0] = Vec3::new(1.0, 2.0, 3.0);
        state.time = 0.05;
        state.active = true;

        let json = serde_json::to_string(&state).unwrap();
        let recovered: InertializationState = serde_json::from_str(&json).unwrap();

        assert_eq!(state.bone_count(), recovered.bone_count());
        assert!((state.half_life - recovered.half_life).abs() < EPSILON);
        assert!(state.position_offsets[0].abs_diff_eq(recovered.position_offsets[0], EPSILON));
    }
}
