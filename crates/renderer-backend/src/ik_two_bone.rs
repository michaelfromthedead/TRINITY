//! Two-bone inverse kinematics solver for TRINITY Engine (T-AN-4.1).
//!
//! This module provides an analytical O(1) two-bone IK solver using the law of cosines.
//! It supports:
//!
//! - Law of cosines angle calculation for mid joint
//! - Pole vector / swivel direction for joint orientation
//! - Joint angle constraints (min/max clamping, soft limits)
//! - Singularity handling (extended, contracted, unreachable poses)
//!
//! # Algorithm
//!
//! Two-bone IK solves for rotations of two bones (e.g., upper arm + forearm,
//! thigh + shin) to reach a target position. The algorithm:
//!
//! 1. Calculate the distance from root to target
//! 2. Use law of cosines to find the mid joint angle: cos(theta) = (a^2 + b^2 - c^2) / (2ab)
//! 3. Apply pole vector to determine joint orientation plane
//! 4. Clamp angles to constraints
//! 5. Compute final rotations for root and mid bones
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::ik_two_bone::{TwoBoneIkChain, TwoBoneIkParams, solve_two_bone_ik};
//! use renderer_backend::skeleton::{Skeleton, Transform};
//! use renderer_backend::pose::Pose;
//! use glam::Vec3;
//!
//! // Define the IK chain (e.g., shoulder -> elbow -> wrist)
//! let chain = TwoBoneIkChain {
//!     root: 1,  // shoulder bone index
//!     mid: 2,   // elbow bone index
//!     end: 3,   // wrist bone index
//! };
//!
//! // Set up IK parameters
//! let params = TwoBoneIkParams {
//!     target_position: Vec3::new(0.5, 0.0, 0.5),
//!     pole_vector: Vec3::NEG_Z,  // elbow points backward
//!     min_angle: 0.1,            // ~5 degrees minimum bend
//!     max_angle: 2.8,            // ~160 degrees maximum bend
//!     soft_limit_ratio: 0.0,     // hard limits
//! };
//!
//! // Solve IK
//! let result = solve_two_bone_ik(&chain, &skeleton, &pose, &params);
//! if result.success {
//!     // Apply result.root_rotation to shoulder
//!     // Apply result.mid_rotation to elbow
//! }
//! ```

use glam::{Quat, Vec3};
use std::f32::consts::PI;

use crate::pose::Pose;
use crate::skeleton::Skeleton;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum distance threshold to avoid division by zero.
const EPSILON: f32 = 1e-6;

/// Default minimum angle for joint constraints (prevents hyperextension).
pub const DEFAULT_MIN_ANGLE: f32 = 0.01; // ~0.5 degrees

/// Default maximum angle for joint constraints.
pub const DEFAULT_MAX_ANGLE: f32 = PI - 0.01; // ~179.5 degrees

// ---------------------------------------------------------------------------
// TwoBoneIkChain
// ---------------------------------------------------------------------------

/// Defines a two-bone IK chain in a skeleton.
///
/// The chain consists of three bones in sequence:
/// - `root`: The first bone (e.g., upper arm, thigh)
/// - `mid`: The joint bone (e.g., forearm, shin) - this is where the bend happens
/// - `end`: The effector bone (e.g., hand, foot) - this reaches the target
///
/// The bones must form a parent-child chain: root -> mid -> end
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoBoneIkChain {
    /// Index of the root bone (e.g., shoulder, hip).
    pub root: usize,

    /// Index of the mid/joint bone (e.g., elbow, knee).
    pub mid: usize,

    /// Index of the end/effector bone (e.g., wrist, ankle).
    pub end: usize,
}

impl TwoBoneIkChain {
    /// Create a new two-bone IK chain.
    ///
    /// # Arguments
    ///
    /// * `root` - Index of the root bone
    /// * `mid` - Index of the mid/joint bone
    /// * `end` - Index of the end/effector bone
    #[inline]
    pub fn new(root: usize, mid: usize, end: usize) -> Self {
        Self { root, mid, end }
    }

    /// Validate that the chain forms a valid parent-child hierarchy.
    ///
    /// Returns `true` if:
    /// - All indices are valid bone indices
    /// - mid's parent is root
    /// - end's parent is mid
    pub fn validate(&self, skeleton: &Skeleton) -> bool {
        // Check bone indices are valid
        let bone_count = skeleton.bone_count();
        if self.root >= bone_count || self.mid >= bone_count || self.end >= bone_count {
            return false;
        }

        // Check mid's parent is root
        if skeleton.parent(self.mid) != Some(self.root) {
            return false;
        }

        // Check end's parent is mid
        if skeleton.parent(self.end) != Some(self.mid) {
            return false;
        }

        true
    }
}

// ---------------------------------------------------------------------------
// SoftIkConfig (T-IK-3.5)
// ---------------------------------------------------------------------------

/// Configuration for soft IK distance handling.
///
/// Soft IK provides smooth falloff when targets are beyond the chain's reach,
/// instead of hard-clamping to the maximum reach distance. This prevents
/// sudden stops and creates more natural-looking animation.
///
/// # Formula
///
/// When the target distance `d` exceeds the soft start distance `d_start`:
///
/// ```text
/// d_soft = d_start + (d_max - d_start) * (1 - e^(-k * overshoot))
/// ```
///
/// Where:
/// - `d_start` = chain_length * (1 - soft_start_ratio)
/// - `d_max` = chain_length (maximum reach)
/// - `k` = falloff_rate
/// - `overshoot` = (d - d_start) / (d_max - d_start)
///
/// # Example
///
/// ```ignore
/// let soft_config = SoftIkConfig {
///     enabled: true,
///     soft_start_ratio: 0.1,  // Start softening at 90% reach
///     falloff_rate: 3.0,      // Exponential falloff rate
/// };
/// let params = TwoBoneIkParams::with_target(target)
///     .with_soft_ik(soft_config);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftIkConfig {
    /// Enable soft IK for unreachable targets.
    pub enabled: bool,

    /// Ratio of chain length at which soft falloff begins.
    ///
    /// Value between 0.0 and 1.0. At `soft_start_ratio = 0.1`, soft falloff
    /// starts when the target is at 90% of the chain's maximum reach.
    ///
    /// Lower values = earlier falloff (smoother but less accurate).
    /// Higher values = later falloff (more accurate but more sudden).
    pub soft_start_ratio: f32,

    /// Exponential falloff rate.
    ///
    /// Higher values = faster approach to maximum reach (more responsive).
    /// Lower values = slower approach (smoother but laggier).
    ///
    /// Typical values: 2.0 (smooth) to 5.0 (responsive).
    pub falloff_rate: f32,
}

impl Default for SoftIkConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            soft_start_ratio: 0.1,
            falloff_rate: 3.0,
        }
    }
}

impl SoftIkConfig {
    /// Create an enabled soft IK config with default parameters.
    #[inline]
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Create a soft IK config with custom parameters.
    #[inline]
    pub fn new(soft_start_ratio: f32, falloff_rate: f32) -> Self {
        Self {
            enabled: true,
            soft_start_ratio: soft_start_ratio.clamp(0.0, 0.5),
            falloff_rate: falloff_rate.max(0.1),
        }
    }

    /// Calculate the soft IK distance for an unreachable target.
    ///
    /// # Arguments
    ///
    /// * `target_dist` - Actual distance from root to target
    /// * `chain_length` - Maximum reach of the IK chain (upper + lower bone length)
    ///
    /// # Returns
    ///
    /// The effective distance to use for IK solving. This will be less than
    /// `chain_length` when soft IK is active, providing a smooth falloff.
    #[inline]
    pub fn apply(&self, target_dist: f32, chain_length: f32) -> f32 {
        if !self.enabled {
            return target_dist.min(chain_length);
        }

        let soft_start = chain_length * (1.0 - self.soft_start_ratio);

        if target_dist <= soft_start {
            // Within normal reach - no soft IK needed
            return target_dist;
        }

        // Calculate soft falloff using exponential decay
        // d_soft = d_start + (d_max - d_start) * (1 - e^(-k * overshoot))
        let soft_zone = chain_length - soft_start;
        if soft_zone <= EPSILON {
            return chain_length;
        }

        let overshoot = (target_dist - soft_start) / soft_zone;
        let falloff = 1.0 - (-self.falloff_rate * overshoot).exp();

        soft_start + soft_zone * falloff
    }

    /// Check if the target distance is in the soft IK zone.
    #[inline]
    pub fn is_in_soft_zone(&self, target_dist: f32, chain_length: f32) -> bool {
        if !self.enabled {
            return false;
        }
        let soft_start = chain_length * (1.0 - self.soft_start_ratio);
        target_dist > soft_start
    }
}

// ---------------------------------------------------------------------------
// TwoBoneIkParams
// ---------------------------------------------------------------------------

/// Parameters for two-bone IK solving.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoBoneIkParams {
    /// Target position in world space that the end effector should reach.
    pub target_position: Vec3,

    /// Pole vector / swivel direction hint for joint orientation.
    ///
    /// This vector indicates the preferred direction for the mid joint to bend.
    /// For example, for an elbow, this might point backward (-Z) to keep the
    /// elbow pointing away from the body.
    ///
    /// The pole vector is projected onto the plane perpendicular to the root-to-target
    /// line to determine the final joint plane.
    pub pole_vector: Vec3,

    /// Minimum angle at the mid joint (radians).
    ///
    /// Prevents the joint from fully extending (angle = 0 or PI).
    /// Default is approximately 0.5 degrees.
    pub min_angle: f32,

    /// Maximum angle at the mid joint (radians).
    ///
    /// Limits how much the joint can bend.
    /// Default is approximately 179.5 degrees.
    pub max_angle: f32,

    /// Soft limit ratio (0.0 = hard limits, 1.0 = full soft limits).
    ///
    /// When > 0, angles near the limits are smoothly blended toward the limit
    /// rather than hard clamped. This prevents popping at constraint boundaries.
    ///
    /// The soft limit zone is `soft_limit_ratio * (max_angle - min_angle)`.
    pub soft_limit_ratio: f32,

    /// Soft IK configuration for unreachable targets.
    ///
    /// When enabled, targets beyond the chain's reach are handled with exponential
    /// falloff instead of hard clamping, resulting in smoother motion.
    pub soft_ik: SoftIkConfig,

    /// Additional twist angle around the root-to-target axis (radians).
    ///
    /// This rotates the IK plane around the line from root to target, allowing
    /// control over the joint's orientation independent of the pole vector.
    /// Positive values rotate counter-clockwise when looking from root to target.
    ///
    /// Use cases:
    /// - Fine-tuning elbow/knee direction
    /// - Animating twist during motion
    /// - Correcting pole vector artifacts at extreme angles
    pub twist_angle: f32,
}

impl Default for TwoBoneIkParams {
    fn default() -> Self {
        Self {
            target_position: Vec3::ZERO,
            pole_vector: Vec3::NEG_Z, // Default: joint bends backward
            min_angle: DEFAULT_MIN_ANGLE,
            max_angle: DEFAULT_MAX_ANGLE,
            soft_limit_ratio: 0.0,
            soft_ik: SoftIkConfig::default(),
            twist_angle: 0.0,
        }
    }
}

impl TwoBoneIkParams {
    /// Create parameters with just a target position (default pole vector and limits).
    #[inline]
    pub fn with_target(target: Vec3) -> Self {
        Self {
            target_position: target,
            ..Default::default()
        }
    }

    /// Create parameters with target and pole vector.
    #[inline]
    pub fn with_target_and_pole(target: Vec3, pole: Vec3) -> Self {
        Self {
            target_position: target,
            pole_vector: pole,
            ..Default::default()
        }
    }

    /// Set angle constraints.
    #[inline]
    pub fn with_constraints(mut self, min_angle: f32, max_angle: f32) -> Self {
        self.min_angle = min_angle;
        self.max_angle = max_angle;
        self
    }

    /// Set soft limit ratio.
    #[inline]
    pub fn with_soft_limits(mut self, ratio: f32) -> Self {
        self.soft_limit_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Enable soft IK with default parameters.
    ///
    /// Soft IK provides smooth falloff when targets are beyond the chain's reach,
    /// instead of hard-clamping to the maximum reach distance.
    #[inline]
    pub fn with_soft_ik_enabled(mut self) -> Self {
        self.soft_ik = SoftIkConfig::enabled();
        self
    }

    /// Set soft IK configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The soft IK configuration to use
    #[inline]
    pub fn with_soft_ik(mut self, config: SoftIkConfig) -> Self {
        self.soft_ik = config;
        self
    }

    /// Set soft IK with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `soft_start_ratio` - Ratio at which soft falloff begins (0.0 to 0.5)
    /// * `falloff_rate` - Exponential falloff rate (typically 2.0 to 5.0)
    #[inline]
    pub fn with_soft_ik_params(mut self, soft_start_ratio: f32, falloff_rate: f32) -> Self {
        self.soft_ik = SoftIkConfig::new(soft_start_ratio, falloff_rate);
        self
    }

    /// Set twist angle for additional rotation around the root-to-target axis.
    ///
    /// # Arguments
    ///
    /// * `angle` - Twist angle in radians (positive = counter-clockwise from root to target)
    #[inline]
    pub fn with_twist(mut self, angle: f32) -> Self {
        self.twist_angle = angle;
        self
    }

    /// Set twist angle in degrees for convenience.
    #[inline]
    pub fn with_twist_degrees(mut self, degrees: f32) -> Self {
        self.twist_angle = degrees.to_radians();
        self
    }
}

// ---------------------------------------------------------------------------
// TwoBoneIkResult
// ---------------------------------------------------------------------------

/// Result of a two-bone IK solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoBoneIkResult {
    /// Rotation to apply to the mid/joint bone (local space).
    ///
    /// This rotation bends the joint to the computed angle.
    pub mid_rotation: Quat,

    /// Rotation to apply to the root bone (local space).
    ///
    /// This rotation aims the chain toward the target.
    pub root_rotation: Quat,

    /// Whether the IK solve was fully successful.
    ///
    /// False if the target was unreachable (too far or too close).
    /// Even when false, the result contains the best possible approximation.
    pub success: bool,

    /// Distance from the end effector to the target after solving.
    ///
    /// Zero (or very small) if successful, positive if target was unreachable.
    pub distance_to_target: f32,

    /// The computed angle at the mid joint (radians).
    ///
    /// This is the angle between the two bones, after constraint clamping.
    pub mid_angle: f32,

    /// Whether the result was clamped due to angle constraints.
    pub was_clamped: bool,
}

impl TwoBoneIkResult {
    /// Create a failed result with identity rotations.
    #[inline]
    pub fn failed(distance_to_target: f32) -> Self {
        Self {
            mid_rotation: Quat::IDENTITY,
            root_rotation: Quat::IDENTITY,
            success: false,
            distance_to_target,
            mid_angle: PI,
            was_clamped: false,
        }
    }
}

impl Default for TwoBoneIkResult {
    fn default() -> Self {
        Self {
            mid_rotation: Quat::IDENTITY,
            root_rotation: Quat::IDENTITY,
            success: true,
            distance_to_target: 0.0,
            mid_angle: PI,
            was_clamped: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Solve Functions
// ---------------------------------------------------------------------------

/// Solve two-bone IK for a given chain, skeleton, and pose.
///
/// This is the main entry point for two-bone IK. It:
/// 1. Extracts bone positions from the skeleton and pose
/// 2. Computes bone lengths
/// 3. Calls the analytical solver
/// 4. Returns rotations to apply to root and mid bones
///
/// # Arguments
///
/// * `chain` - The IK chain definition
/// * `skeleton` - The skeleton containing bone hierarchy
/// * `pose` - The current pose with local transforms
/// * `params` - IK solving parameters
///
/// # Returns
///
/// A `TwoBoneIkResult` containing the computed rotations and success status.
///
/// # Panics
///
/// Panics if any bone index in the chain is out of bounds.
pub fn solve_two_bone_ik(
    chain: &TwoBoneIkChain,
    skeleton: &Skeleton,
    pose: &Pose,
    params: &TwoBoneIkParams,
) -> TwoBoneIkResult {
    // Get world transforms for the chain
    let transforms = pose.transforms();
    let world_transforms = skeleton.compute_world_transforms(&transforms);

    // Extract world positions
    let root_pos = world_transforms[chain.root].w_axis.truncate();
    let mid_pos = world_transforms[chain.mid].w_axis.truncate();
    let end_pos = world_transforms[chain.end].w_axis.truncate();

    // Compute bone lengths
    let upper_length = (mid_pos - root_pos).length();
    let lower_length = (end_pos - mid_pos).length();

    // Get current root rotation (world space)
    let root_world_rot = Quat::from_mat4(&world_transforms[chain.root]);

    // Get local rotations
    let root_local_rot = pose.rotations[chain.root];
    let mid_local_rot = pose.rotations[chain.mid];

    // Solve in world space
    let world_result = solve_two_bone_ik_world(
        root_pos,
        upper_length,
        lower_length,
        root_world_rot,
        params,
    );

    // Convert world rotations to local space
    let result = if world_result.success || world_result.distance_to_target < upper_length + lower_length {
        // Get parent rotation for root bone
        let root_parent_rot = if let Some(parent_idx) = skeleton.parent(chain.root) {
            Quat::from_mat4(&world_transforms[parent_idx])
        } else {
            Quat::IDENTITY
        };

        // Root local rotation: parent_inv * world_root_rot
        let new_root_world_rot = world_result.root_rotation * root_world_rot;
        let new_root_local_rot = root_parent_rot.inverse() * new_root_world_rot;

        // For mid joint, we apply a rotation around the bend axis
        // The mid rotation is relative to the root's new orientation
        let new_mid_local_rot = world_result.mid_rotation * mid_local_rot;

        TwoBoneIkResult {
            root_rotation: new_root_local_rot,
            mid_rotation: new_mid_local_rot,
            success: world_result.success,
            distance_to_target: world_result.distance_to_target,
            mid_angle: world_result.mid_angle,
            was_clamped: world_result.was_clamped,
        }
    } else {
        TwoBoneIkResult {
            root_rotation: root_local_rot,
            mid_rotation: mid_local_rot,
            success: false,
            distance_to_target: world_result.distance_to_target,
            mid_angle: world_result.mid_angle,
            was_clamped: world_result.was_clamped,
        }
    };

    result
}

/// Solve two-bone IK in world space given positions and lengths.
///
/// This is the core analytical solver. It operates purely on geometric data
/// without needing skeleton/pose structures.
///
/// # Arguments
///
/// * `root_pos` - World position of the root joint
/// * `upper_length` - Length of the upper bone (root to mid)
/// * `lower_length` - Length of the lower bone (mid to end)
/// * `current_root_rot` - Current world rotation of the root bone
/// * `params` - IK solving parameters
///
/// # Returns
///
/// A `TwoBoneIkResult` with world-space rotations.
pub fn solve_two_bone_ik_world(
    root_pos: Vec3,
    upper_length: f32,
    lower_length: f32,
    current_root_rot: Quat,
    params: &TwoBoneIkParams,
) -> TwoBoneIkResult {
    let target = params.target_position;
    let pole = params.pole_vector;

    // Handle degenerate bone lengths
    if upper_length < EPSILON || lower_length < EPSILON {
        // Cannot solve with zero-length bones - return identity with failure
        let target_dist = (target - root_pos).length();
        return TwoBoneIkResult {
            mid_rotation: Quat::IDENTITY,
            root_rotation: Quat::IDENTITY,
            success: false,
            distance_to_target: target_dist,
            mid_angle: PI,
            was_clamped: false,
        };
    }

    // Vector from root to target
    let root_to_target = target - root_pos;
    let target_dist = root_to_target.length();

    // Handle singularities
    if target_dist < EPSILON {
        // Target is at root - undefined direction
        return TwoBoneIkResult::failed(0.0);
    }

    let chain_length = upper_length + lower_length;
    let min_reach = (upper_length - lower_length).abs();

    // Check reachability with soft IK support (T-IK-3.5)
    let (actual_dist, clamped, success) = if target_dist > chain_length {
        // Target too far - apply soft IK if enabled, otherwise extend fully
        if params.soft_ik.enabled {
            let soft_dist = params.soft_ik.apply(target_dist, chain_length);
            // Soft IK: we get a smooth falloff, but success depends on whether
            // the target is actually reachable (within chain_length)
            (soft_dist, true, false)
        } else {
            (chain_length, true, false)
        }
    } else if params.soft_ik.is_in_soft_zone(target_dist, chain_length) {
        // In soft zone but still reachable - apply soft IK for smooth approach
        let soft_dist = params.soft_ik.apply(target_dist, chain_length);
        (soft_dist, false, true)
    } else if target_dist < min_reach {
        // Target too close - contract fully
        (min_reach, true, false)
    } else {
        (target_dist, false, true)
    };

    // Law of cosines to find the angle at the mid joint
    // c^2 = a^2 + b^2 - 2ab*cos(C)
    // cos(C) = (a^2 + b^2 - c^2) / (2ab)
    let a = upper_length;
    let b = lower_length;
    let c = actual_dist;

    let cos_angle = ((a * a + b * b - c * c) / (2.0 * a * b)).clamp(-1.0, 1.0);
    let mut mid_angle = cos_angle.acos();

    // Apply angle constraints
    let was_clamped = clamped || mid_angle < params.min_angle || mid_angle > params.max_angle;
    mid_angle = apply_angle_constraints(mid_angle, params.min_angle, params.max_angle, params.soft_limit_ratio);

    // Recalculate actual distance based on constrained angle
    let constrained_dist = if was_clamped && !clamped {
        // Recalculate distance from constrained angle using law of cosines
        let new_c_sq = a * a + b * b - 2.0 * a * b * mid_angle.cos();
        new_c_sq.max(0.0).sqrt()
    } else {
        actual_dist
    };

    // Build the IK plane
    // The plane contains: root position, target position, and pole vector hint
    let target_dir = root_to_target.normalize();

    // Project pole vector onto plane perpendicular to target direction
    let pole_on_plane = pole - target_dir * pole.dot(target_dir);
    let mut pole_dir = if pole_on_plane.length_squared() > EPSILON * EPSILON {
        pole_on_plane.normalize()
    } else {
        // Pole vector is parallel to target - pick an arbitrary perpendicular
        // This is a singular position - handle gracefully (T-IK-3.6)
        get_perpendicular(target_dir)
    };

    // Apply twist angle around the target axis (T-IK-3.6)
    // This rotates the IK plane, allowing fine control over joint orientation
    if params.twist_angle.abs() > EPSILON {
        let twist_rotation = Quat::from_axis_angle(target_dir, params.twist_angle);
        pole_dir = twist_rotation * pole_dir;
    }

    // Build rotation from current orientation to target orientation
    // The IK plane is defined by target_dir (forward) and pole_dir (up hint for the joint)

    // Current forward direction (assuming Y-forward convention)
    let current_forward = current_root_rot * Vec3::Y;

    // Rotation to aim at target
    let aim_rotation = rotation_between_vectors(current_forward, target_dir);

    // Calculate where the mid joint should be
    // Using law of cosines: angle at root
    let cos_root_angle = ((a * a + constrained_dist * constrained_dist - b * b) / (2.0 * a * constrained_dist))
        .clamp(-1.0, 1.0);
    let root_angle = cos_root_angle.acos();

    // The mid joint position in the IK plane
    // Rotate the target direction by root_angle around the pole axis
    let bend_axis = target_dir.cross(pole_dir).normalize_or_zero();
    let bend_rotation = if bend_axis.length_squared() > EPSILON * EPSILON {
        Quat::from_axis_angle(bend_axis, -root_angle)
    } else {
        Quat::IDENTITY
    };

    // Combine rotations: first aim at target, then apply bend around the pole axis
    // The root rotation aims at the target and incorporates the IK plane orientation
    let ik_plane_up = pole_dir;
    let ik_plane_right = target_dir.cross(ik_plane_up).normalize_or_zero();

    // Build a rotation that aligns with the IK plane
    // This incorporates both the aim direction and the twist-affected plane orientation
    let root_rotation = if ik_plane_right.length_squared() > EPSILON * EPSILON {
        // Build orientation matrix from IK plane axes and convert to quaternion
        let forward = target_dir;
        let up = ik_plane_up;
        let right = ik_plane_right;
        // Construct rotation from these basis vectors
        let rot_matrix = glam::Mat3::from_cols(right, forward, up);
        Quat::from_mat3(&rot_matrix)
    } else {
        aim_rotation
    };

    // Mid joint rotation: bend by the computed angle
    // The bend axis is perpendicular to the IK plane (the "right" axis)
    // This ensures the twist affects the bend direction
    let mid_bend_angle = PI - mid_angle; // Convert from internal angle to bend amount
    let mid_rotation_axis = if ik_plane_right.length_squared() > EPSILON * EPSILON {
        // Transform the bend axis to be relative to the root rotation
        root_rotation.inverse() * ik_plane_right
    } else {
        Vec3::X
    };
    let mid_rotation = Quat::from_axis_angle(mid_rotation_axis.normalize_or_zero(), mid_bend_angle);

    // Calculate final distance to target
    let final_distance = if success {
        0.0
    } else {
        (target_dist - constrained_dist).abs()
    };

    TwoBoneIkResult {
        root_rotation,
        mid_rotation,
        success,
        distance_to_target: final_distance,
        mid_angle,
        was_clamped,
    }
}

/// Solve two-bone IK with direct position inputs (no skeleton/pose).
///
/// This is a convenience function for cases where you have raw positions
/// rather than skeleton data.
///
/// # Arguments
///
/// * `root_pos` - World position of the root joint
/// * `mid_pos` - World position of the mid joint
/// * `end_pos` - World position of the end effector
/// * `target_pos` - Target position for the end effector
/// * `pole_vector` - Direction hint for joint bending
/// * `min_angle` - Minimum joint angle (radians)
/// * `max_angle` - Maximum joint angle (radians)
///
/// # Returns
///
/// A tuple of (root_rotation, mid_rotation, success, mid_angle)
pub fn solve_two_bone_ik_positions(
    root_pos: Vec3,
    mid_pos: Vec3,
    end_pos: Vec3,
    target_pos: Vec3,
    pole_vector: Vec3,
    min_angle: f32,
    max_angle: f32,
) -> (Quat, Quat, bool, f32) {
    let upper_length = (mid_pos - root_pos).length();
    let lower_length = (end_pos - mid_pos).length();

    // Current bone directions
    let current_upper_dir = (mid_pos - root_pos).normalize_or_zero();
    let current_root_rot = if current_upper_dir.length_squared() > EPSILON * EPSILON {
        Quat::from_rotation_arc(Vec3::Y, current_upper_dir)
    } else {
        Quat::IDENTITY
    };

    let params = TwoBoneIkParams {
        target_position: target_pos,
        pole_vector,
        min_angle,
        max_angle,
        soft_limit_ratio: 0.0,
        soft_ik: SoftIkConfig::default(),
        twist_angle: 0.0,
    };

    let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, current_root_rot, &params);

    (result.root_rotation, result.mid_rotation, result.success, result.mid_angle)
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Apply angle constraints with optional soft limits.
///
/// Hard limits clamp the angle directly.
/// Soft limits smoothly blend the angle toward the limit as it approaches.
fn apply_angle_constraints(angle: f32, min: f32, max: f32, soft_ratio: f32) -> f32 {
    if soft_ratio <= EPSILON {
        // Hard limits
        return angle.clamp(min, max);
    }

    let range = max - min;
    let soft_zone = range * soft_ratio;

    // Apply soft limit at minimum
    let angle = if angle < min + soft_zone {
        let t = ((angle - min) / soft_zone).clamp(0.0, 1.0);
        min + soft_zone * smooth_step(t)
    } else {
        angle
    };

    // Apply soft limit at maximum
    if angle > max - soft_zone {
        let t = ((max - angle) / soft_zone).clamp(0.0, 1.0);
        max - soft_zone * smooth_step(t)
    } else {
        angle
    }
}

/// Smooth step function for soft limits (0 to 1, with smooth derivatives at boundaries).
#[inline]
fn smooth_step(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Get an arbitrary vector perpendicular to the input.
fn get_perpendicular(v: Vec3) -> Vec3 {
    // Choose the axis least parallel to v
    let abs_v = v.abs();
    let axis = if abs_v.x <= abs_v.y && abs_v.x <= abs_v.z {
        Vec3::X
    } else if abs_v.y <= abs_v.z {
        Vec3::Y
    } else {
        Vec3::Z
    };

    v.cross(axis).normalize()
}

/// Compute the rotation that transforms vector `from` to vector `to`.
fn rotation_between_vectors(from: Vec3, to: Vec3) -> Quat {
    let from = from.normalize_or_zero();
    let to = to.normalize_or_zero();

    let dot = from.dot(to);

    if dot > 0.99999 {
        // Vectors are nearly parallel
        return Quat::IDENTITY;
    }

    if dot < -0.99999 {
        // Vectors are nearly opposite - rotate 180 degrees around any perpendicular
        let perp = get_perpendicular(from);
        return Quat::from_axis_angle(perp, PI);
    }

    let cross = from.cross(to);
    let s = ((1.0 + dot) * 2.0).sqrt();
    let inv_s = 1.0 / s;

    Quat::from_xyzw(cross.x * inv_s, cross.y * inv_s, cross.z * inv_s, s * 0.5).normalize()
}

/// Compute the angle between two vectors (radians).
#[inline]
pub fn angle_between(a: Vec3, b: Vec3) -> f32 {
    let dot = a.normalize_or_zero().dot(b.normalize_or_zero()).clamp(-1.0, 1.0);
    dot.acos()
}

/// Calculate bone lengths from a skeleton and pose.
pub fn calculate_bone_lengths(
    chain: &TwoBoneIkChain,
    skeleton: &Skeleton,
    pose: &Pose,
) -> (f32, f32) {
    let transforms = pose.transforms();
    let world_transforms = skeleton.compute_world_transforms(&transforms);

    let root_pos = world_transforms[chain.root].w_axis.truncate();
    let mid_pos = world_transforms[chain.mid].w_axis.truncate();
    let end_pos = world_transforms[chain.end].w_axis.truncate();

    let upper_length = (mid_pos - root_pos).length();
    let lower_length = (end_pos - mid_pos).length();

    (upper_length, lower_length)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Bone, SkeletonBuilder};

    const TEST_EPSILON: f32 = 1e-4;

    // ===== Helper Functions =====

    fn create_arm_skeleton() -> Skeleton {
        // Create a simple arm: shoulder -> elbow -> wrist
        // Upper arm: 2 units, Lower arm: 2 units
        SkeletonBuilder::new()
            .root("shoulder")
            .child_at("elbow", "shoulder", Vec3::new(0.0, 2.0, 0.0))
            .child_at("wrist", "elbow", Vec3::new(0.0, 2.0, 0.0))
            .build_unchecked()
    }

    fn create_asymmetric_skeleton() -> Skeleton {
        // Upper: 3 units, Lower: 2 units
        SkeletonBuilder::new()
            .root("root")
            .child_at("mid", "root", Vec3::new(0.0, 3.0, 0.0))
            .child_at("end", "mid", Vec3::new(0.0, 2.0, 0.0))
            .build_unchecked()
    }

    fn create_arm_chain() -> TwoBoneIkChain {
        TwoBoneIkChain::new(0, 1, 2)
    }

    // ===== TwoBoneIkChain Tests =====

    #[test]
    fn test_chain_new() {
        let chain = TwoBoneIkChain::new(0, 1, 2);
        assert_eq!(chain.root, 0);
        assert_eq!(chain.mid, 1);
        assert_eq!(chain.end, 2);
    }

    #[test]
    fn test_chain_validate_valid() {
        let skeleton = create_arm_skeleton();
        let chain = create_arm_chain();
        assert!(chain.validate(&skeleton));
    }

    #[test]
    fn test_chain_validate_invalid_indices() {
        let skeleton = create_arm_skeleton();
        let chain = TwoBoneIkChain::new(0, 1, 10); // Index 10 doesn't exist
        assert!(!chain.validate(&skeleton));
    }

    #[test]
    fn test_chain_validate_wrong_parent_order() {
        let skeleton = create_arm_skeleton();
        let chain = TwoBoneIkChain::new(0, 2, 1); // Wrong order
        assert!(!chain.validate(&skeleton));
    }

    #[test]
    fn test_chain_equality() {
        let a = TwoBoneIkChain::new(0, 1, 2);
        let b = TwoBoneIkChain::new(0, 1, 2);
        let c = TwoBoneIkChain::new(1, 2, 3);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ===== TwoBoneIkParams Tests =====

    #[test]
    fn test_params_default() {
        let params = TwoBoneIkParams::default();
        assert_eq!(params.target_position, Vec3::ZERO);
        assert_eq!(params.pole_vector, Vec3::NEG_Z);
        assert!((params.min_angle - DEFAULT_MIN_ANGLE).abs() < TEST_EPSILON);
        assert!((params.max_angle - DEFAULT_MAX_ANGLE).abs() < TEST_EPSILON);
        assert!((params.soft_limit_ratio - 0.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_params_with_target() {
        let target = Vec3::new(1.0, 2.0, 3.0);
        let params = TwoBoneIkParams::with_target(target);
        assert!(params.target_position.abs_diff_eq(target, TEST_EPSILON));
    }

    #[test]
    fn test_params_with_target_and_pole() {
        let target = Vec3::new(1.0, 0.0, 0.0);
        let pole = Vec3::Y;
        let params = TwoBoneIkParams::with_target_and_pole(target, pole);
        assert!(params.target_position.abs_diff_eq(target, TEST_EPSILON));
        assert!(params.pole_vector.abs_diff_eq(pole, TEST_EPSILON));
    }

    #[test]
    fn test_params_with_constraints() {
        let params = TwoBoneIkParams::default().with_constraints(0.5, 2.5);
        assert!((params.min_angle - 0.5).abs() < TEST_EPSILON);
        assert!((params.max_angle - 2.5).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_params_with_soft_limits() {
        let params = TwoBoneIkParams::default().with_soft_limits(0.3);
        assert!((params.soft_limit_ratio - 0.3).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_params_soft_limits_clamped() {
        let params = TwoBoneIkParams::default().with_soft_limits(2.0);
        assert!((params.soft_limit_ratio - 1.0).abs() < TEST_EPSILON);

        let params = TwoBoneIkParams::default().with_soft_limits(-0.5);
        assert!((params.soft_limit_ratio - 0.0).abs() < TEST_EPSILON);
    }

    // ===== SoftIkConfig Tests (T-IK-3.5) =====

    #[test]
    fn test_soft_ik_config_default_disabled() {
        let config = SoftIkConfig::default();
        assert!(!config.enabled);
        assert!((config.soft_start_ratio - 0.1).abs() < TEST_EPSILON);
        assert!((config.falloff_rate - 3.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_soft_ik_config_enabled() {
        let config = SoftIkConfig::enabled();
        assert!(config.enabled);
        assert!((config.soft_start_ratio - 0.1).abs() < TEST_EPSILON);
        assert!((config.falloff_rate - 3.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_soft_ik_config_new() {
        let config = SoftIkConfig::new(0.2, 5.0);
        assert!(config.enabled);
        assert!((config.soft_start_ratio - 0.2).abs() < TEST_EPSILON);
        assert!((config.falloff_rate - 5.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_soft_ik_config_new_clamped() {
        // soft_start_ratio should be clamped to 0.0..0.5
        let config = SoftIkConfig::new(0.8, 5.0);
        assert!((config.soft_start_ratio - 0.5).abs() < TEST_EPSILON);

        // falloff_rate should be at least 0.1
        let config = SoftIkConfig::new(0.1, 0.01);
        assert!((config.falloff_rate - 0.1).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_soft_ik_apply_within_reach() {
        let config = SoftIkConfig::enabled();
        let chain_length = 4.0;

        // Target at 3.0 - well within reach (soft start at 3.6)
        let result = config.apply(3.0, chain_length);
        assert!((result - 3.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_soft_ik_apply_at_soft_start() {
        let config = SoftIkConfig::new(0.1, 3.0);
        let chain_length = 4.0;
        let soft_start = chain_length * 0.9; // 3.6

        // Target exactly at soft start - should return nearly the same
        let result = config.apply(soft_start, chain_length);
        assert!((result - soft_start).abs() < 0.1);
    }

    #[test]
    fn test_soft_ik_apply_beyond_reach() {
        let config = SoftIkConfig::new(0.1, 3.0);
        let chain_length = 4.0;

        // Target at 5.0 - beyond reach
        let result = config.apply(5.0, chain_length);

        // Result should be less than chain_length but approaching it
        assert!(result < chain_length);
        assert!(result > chain_length * 0.9);
    }

    #[test]
    fn test_soft_ik_apply_far_beyond_reach() {
        let config = SoftIkConfig::new(0.1, 3.0);
        let chain_length = 4.0;

        // Target at 10.0 - far beyond reach
        let result = config.apply(10.0, chain_length);

        // Result should asymptotically approach chain_length (may equal it for far targets)
        assert!(result <= chain_length + TEST_EPSILON);
        assert!(result > chain_length * 0.98); // Very close to max
    }

    #[test]
    fn test_soft_ik_apply_disabled() {
        let config = SoftIkConfig::default(); // disabled
        let chain_length = 4.0;

        // When disabled, should clamp to chain_length
        let result = config.apply(5.0, chain_length);
        assert!((result - chain_length).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_soft_ik_is_in_soft_zone() {
        let config = SoftIkConfig::new(0.1, 3.0);
        let chain_length = 4.0;

        // Below soft start (3.6) - not in zone
        assert!(!config.is_in_soft_zone(3.0, chain_length));

        // At soft start - not quite in zone (boundary)
        assert!(!config.is_in_soft_zone(3.6, chain_length));

        // Above soft start - in zone
        assert!(config.is_in_soft_zone(3.7, chain_length));
        assert!(config.is_in_soft_zone(5.0, chain_length));
    }

    #[test]
    fn test_soft_ik_is_in_soft_zone_disabled() {
        let config = SoftIkConfig::default(); // disabled

        // When disabled, never in soft zone
        assert!(!config.is_in_soft_zone(5.0, 4.0));
    }

    #[test]
    fn test_soft_ik_exponential_falloff_formula() {
        // Verify the exponential falloff formula:
        // d_soft = d_start + (d_max - d_start) * (1 - e^(-k * overshoot))
        let config = SoftIkConfig::new(0.2, 2.0);
        let chain_length = 5.0;
        let soft_start = chain_length * 0.8; // 4.0
        let soft_zone = chain_length - soft_start; // 1.0

        // At overshoot = 1.0 (target at 5.0), d_soft should be:
        // 4.0 + 1.0 * (1 - e^(-2.0 * 1.0)) = 4.0 + 1.0 * (1 - 0.1353) ≈ 4.865
        let result = config.apply(5.0, chain_length);
        let expected = soft_start + soft_zone * (1.0 - (-2.0_f32).exp());
        assert!((result - expected).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_soft_ik_monotonically_increasing() {
        // Soft IK should always increase (or stay same) as target moves away
        let config = SoftIkConfig::new(0.1, 3.0);
        let chain_length = 4.0;

        let mut prev = 0.0;
        for i in 0..20 {
            let target_dist = i as f32 * 0.5;
            let result = config.apply(target_dist, chain_length);
            assert!(result >= prev, "Soft IK should be monotonically increasing");
            prev = result;
        }
    }

    #[test]
    fn test_params_with_soft_ik_enabled() {
        let params = TwoBoneIkParams::default().with_soft_ik_enabled();
        assert!(params.soft_ik.enabled);
    }

    #[test]
    fn test_params_with_soft_ik_config() {
        let config = SoftIkConfig::new(0.15, 4.0);
        let params = TwoBoneIkParams::default().with_soft_ik(config);
        assert!(params.soft_ik.enabled);
        assert!((params.soft_ik.soft_start_ratio - 0.15).abs() < TEST_EPSILON);
        assert!((params.soft_ik.falloff_rate - 4.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_params_with_soft_ik_params() {
        let params = TwoBoneIkParams::default().with_soft_ik_params(0.2, 5.0);
        assert!(params.soft_ik.enabled);
        assert!((params.soft_ik.soft_start_ratio - 0.2).abs() < TEST_EPSILON);
        assert!((params.soft_ik.falloff_rate - 5.0).abs() < TEST_EPSILON);
    }

    // ===== Soft IK Solve Integration Tests =====

    #[test]
    fn test_solve_with_soft_ik_unreachable() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let chain_length = 4.0;

        // Target at 6.0 - beyond reach
        let params = TwoBoneIkParams::with_target(Vec3::new(0.0, 6.0, 0.0))
            .with_soft_ik_enabled();

        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        // Should not fully extend to chain_length due to soft IK
        assert!(!result.success); // Still unreachable
        assert!(result.was_clamped);
        // The solve should use soft IK distance (less than chain_length)
    }

    #[test]
    fn test_solve_with_soft_ik_in_soft_zone() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let chain_length = 4.0;

        // Target at 3.8 - in soft zone but reachable (soft start at 3.6 with 0.1 ratio)
        let params = TwoBoneIkParams::with_target(Vec3::new(0.0, 3.8, 0.0))
            .with_soft_ik_enabled();

        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        // Should succeed but with soft IK applied
        assert!(result.success);
    }

    #[test]
    fn test_solve_without_soft_ik_unreachable() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;

        // Target at 6.0 - beyond reach, no soft IK
        let params = TwoBoneIkParams::with_target(Vec3::new(0.0, 6.0, 0.0));

        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        assert!(!result.success);
        assert!(result.was_clamped);
    }

    #[test]
    fn test_soft_ik_smooth_transition() {
        // Test that soft IK provides smoother transition than hard clamping
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;

        let config = SoftIkConfig::new(0.1, 3.0);

        // Sample multiple target distances and verify smooth progression
        let mut prev_angle = 0.0;
        for i in 0..10 {
            let target_dist = 3.5 + i as f32 * 0.1; // 3.5 to 4.4
            let params = TwoBoneIkParams::with_target(Vec3::new(0.0, target_dist, 0.0))
                .with_soft_ik(config);

            let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

            // Angle should change smoothly
            if i > 0 {
                let angle_change = (result.mid_angle - prev_angle).abs();
                // Changes should be gradual
                assert!(angle_change < 0.3, "Angle change should be smooth: {}", angle_change);
            }
            prev_angle = result.mid_angle;
        }
    }

    // ===== TwoBoneIkResult Tests =====

    #[test]
    fn test_result_default() {
        let result = TwoBoneIkResult::default();
        assert!(result.success);
        assert!((result.distance_to_target - 0.0).abs() < TEST_EPSILON);
        assert!(result.mid_rotation.abs_diff_eq(Quat::IDENTITY, TEST_EPSILON));
        assert!(result.root_rotation.abs_diff_eq(Quat::IDENTITY, TEST_EPSILON));
    }

    #[test]
    fn test_result_failed() {
        let result = TwoBoneIkResult::failed(5.0);
        assert!(!result.success);
        assert!((result.distance_to_target - 5.0).abs() < TEST_EPSILON);
        assert!(result.mid_rotation.abs_diff_eq(Quat::IDENTITY, TEST_EPSILON));
        assert!(result.root_rotation.abs_diff_eq(Quat::IDENTITY, TEST_EPSILON));
    }

    // ===== Basic Solve Tests =====

    #[test]
    fn test_solve_target_in_range() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let current_rot = Quat::IDENTITY;

        // Target at 3 units - within reach (2+2=4)
        let params = TwoBoneIkParams::with_target(Vec3::new(0.0, 3.0, 0.0));
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, current_rot, &params);

        assert!(result.success);
        assert!(result.distance_to_target < TEST_EPSILON);
    }

    #[test]
    fn test_solve_target_at_max_reach() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let current_rot = Quat::IDENTITY;

        // Target exactly at max reach (4 units)
        let params = TwoBoneIkParams::with_target(Vec3::new(0.0, 4.0, 0.0));
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, current_rot, &params);

        // Should succeed (just barely reachable)
        assert!(result.success);
        // Mid angle should be near PI (fully extended)
        assert!((result.mid_angle - PI).abs() < 0.1);
    }

    #[test]
    fn test_solve_target_unreachable_far() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let current_rot = Quat::IDENTITY;

        // Target at 6 units - beyond reach (2+2=4)
        let params = TwoBoneIkParams::with_target(Vec3::new(0.0, 6.0, 0.0));
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, current_rot, &params);

        assert!(!result.success);
        assert!(result.distance_to_target > 0.0);
        // Should extend fully
        assert!((result.mid_angle - PI).abs() < 0.1);
    }

    #[test]
    fn test_solve_target_unreachable_close() {
        let root_pos = Vec3::ZERO;
        let upper_length = 3.0;
        let lower_length = 2.0;
        let current_rot = Quat::IDENTITY;

        // Target at 0.5 units - too close (min reach = |3-2| = 1)
        let params = TwoBoneIkParams::with_target(Vec3::new(0.0, 0.5, 0.0));
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, current_rot, &params);

        assert!(!result.success);
    }

    #[test]
    fn test_solve_target_at_root() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let current_rot = Quat::IDENTITY;

        // Target exactly at root - singular
        let params = TwoBoneIkParams::with_target(Vec3::ZERO);
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, current_rot, &params);

        assert!(!result.success);
    }

    #[test]
    fn test_solve_preserves_success_for_valid_target() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;

        // Various valid targets
        for target in [
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-2.0, 2.0, 0.0),
        ] {
            let params = TwoBoneIkParams::with_target(target);
            let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

            // All these targets should be reachable
            let dist = target.length();
            if dist <= upper_length + lower_length && dist >= (upper_length - lower_length).abs() {
                assert!(result.success, "Target {:?} should be reachable", target);
            }
        }
    }

    // ===== Pole Vector Tests =====

    #[test]
    fn test_pole_vector_affects_orientation() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let target = Vec3::new(0.0, 3.0, 0.0);

        // Solve with different pole vectors
        let params_back = TwoBoneIkParams::with_target_and_pole(target, Vec3::NEG_Z);
        let result_back = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params_back);

        let params_front = TwoBoneIkParams::with_target_and_pole(target, Vec3::Z);
        let result_front = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params_front);

        // Both should succeed
        assert!(result_back.success);
        assert!(result_front.success);

        // Mid angles should be the same (geometry is the same)
        assert!((result_back.mid_angle - result_front.mid_angle).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_pole_vector_parallel_to_target() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;

        // Pole vector parallel to target direction - should still work
        let target = Vec3::new(0.0, 3.0, 0.0);
        let pole = Vec3::Y; // Same direction as target
        let params = TwoBoneIkParams::with_target_and_pole(target, pole);
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        // Should still succeed (fallback to perpendicular)
        assert!(result.success);
    }

    // ===== Twist Control Tests (T-IK-3.6) =====

    #[test]
    fn test_twist_zero_no_effect() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let target = Vec3::new(0.0, 3.0, 0.0);

        let params_no_twist = TwoBoneIkParams::with_target(target);
        let params_zero_twist = TwoBoneIkParams::with_target(target).with_twist(0.0);

        let result1 = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params_no_twist);
        let result2 = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params_zero_twist);

        assert!(result1.root_rotation.abs_diff_eq(result2.root_rotation, TEST_EPSILON));
        assert!(result1.mid_rotation.abs_diff_eq(result2.mid_rotation, TEST_EPSILON));
    }

    #[test]
    fn test_twist_90_degrees() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let target = Vec3::new(1.0, 2.5, 0.5);

        let params_no_twist = TwoBoneIkParams::with_target(target);
        let params_twist_90 = TwoBoneIkParams::with_target(target).with_twist_degrees(90.0);

        let result1 = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params_no_twist);
        let result2 = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params_twist_90);

        // Both should succeed
        assert!(result1.success);
        assert!(result2.success);

        // Mid angles should be the same (geometry unchanged)
        assert!((result1.mid_angle - result2.mid_angle).abs() < 0.1);

        // Twist is stored in params and affects the solver
        assert!((params_twist_90.twist_angle - PI / 2.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_twist_180_degrees() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let target = Vec3::new(0.0, 3.0, 0.0);

        let params_twist_180 = TwoBoneIkParams::with_target(target).with_twist_degrees(180.0);
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params_twist_180);

        // Should succeed even with 180 degree twist
        assert!(result.success);
    }

    #[test]
    fn test_twist_negative() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let target = Vec3::new(1.0, 2.5, 0.5);

        let params_pos = TwoBoneIkParams::with_target(target).with_twist_degrees(45.0);
        let params_neg = TwoBoneIkParams::with_target(target).with_twist_degrees(-45.0);

        let result_pos = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params_pos);
        let result_neg = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params_neg);

        // Both should succeed
        assert!(result_pos.success);
        assert!(result_neg.success);

        // Twist values are stored correctly
        assert!((params_pos.twist_angle - PI / 4.0).abs() < 0.001);
        assert!((params_neg.twist_angle - (-PI / 4.0)).abs() < 0.001);

        // Mid angles should be the same (geometry unchanged by twist)
        assert!((result_pos.mid_angle - result_neg.mid_angle).abs() < 0.1);
    }

    #[test]
    fn test_twist_with_pole_vector() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let target = Vec3::new(0.0, 3.0, 0.0);

        // Twist should work in combination with pole vector
        let params = TwoBoneIkParams::with_target_and_pole(target, Vec3::NEG_Z)
            .with_twist_degrees(30.0);

        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);
        assert!(result.success);
    }

    #[test]
    fn test_twist_builder_methods() {
        let params1 = TwoBoneIkParams::default().with_twist(PI / 4.0);
        assert!((params1.twist_angle - PI / 4.0).abs() < TEST_EPSILON);

        let params2 = TwoBoneIkParams::default().with_twist_degrees(45.0);
        assert!((params2.twist_angle - PI / 4.0).abs() < 0.001);
    }

    #[test]
    fn test_twist_continuous_rotation() {
        // Test that twist provides smooth continuous control
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let target = Vec3::new(1.0, 2.0, 0.5);

        let mut prev_rotation = Quat::IDENTITY;
        for i in 0..12 {
            let twist = (i as f32) * 30.0; // 0, 30, 60, ..., 330 degrees
            let params = TwoBoneIkParams::with_target(target).with_twist_degrees(twist);
            let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

            assert!(result.success, "Twist {} degrees should succeed", twist);

            if i > 0 {
                // Changes should be gradual
                let angle_diff = result.root_rotation.angle_between(prev_rotation);
                assert!(angle_diff < 1.0, "Twist changes should be smooth: {} rad at {} deg", angle_diff, twist);
            }
            prev_rotation = result.root_rotation;
        }
    }

    // ===== Angle Constraint Tests =====

    #[test]
    fn test_angle_constraints_min() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;

        // Target at max reach would require angle = PI, but we limit to 2.5
        let target = Vec3::new(0.0, 4.0, 0.0);
        let mut params = TwoBoneIkParams::with_target(target);
        params.min_angle = 0.1;
        params.max_angle = 2.5;

        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        // Angle should be clamped to max
        assert!(result.mid_angle <= params.max_angle + TEST_EPSILON);
        assert!(result.was_clamped);
    }

    #[test]
    fn test_angle_constraints_max() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;

        // Target very close - would require small angle
        let target = Vec3::new(0.0, 0.1, 0.0);
        let mut params = TwoBoneIkParams::with_target(target);
        params.min_angle = 0.5;
        params.max_angle = PI - 0.01;

        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        // Should handle gracefully
        assert!(result.mid_angle >= params.min_angle - TEST_EPSILON);
    }

    #[test]
    fn test_angle_constraints_no_clamp_in_range() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;

        // Target that results in angle well within constraints
        let target = Vec3::new(0.0, 2.0, 0.0);
        let mut params = TwoBoneIkParams::with_target(target);
        params.min_angle = 0.1;
        params.max_angle = PI - 0.1;

        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        assert!(result.success);
        assert!(!result.was_clamped);
    }

    // ===== Soft Limit Tests =====

    #[test]
    fn test_soft_limits_smooth() {
        // Soft limits should produce smooth transitions
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;

        let mut params = TwoBoneIkParams::with_target(Vec3::new(0.0, 3.9, 0.0));
        params.min_angle = 0.1;
        params.max_angle = 2.5;

        // Without soft limits
        params.soft_limit_ratio = 0.0;
        let result_hard = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        // With soft limits
        params.soft_limit_ratio = 0.3;
        let result_soft = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        // Both should constrain, but soft should be less aggressive
        assert!(result_hard.mid_angle <= params.max_angle + TEST_EPSILON);
        assert!(result_soft.mid_angle <= params.max_angle + TEST_EPSILON);
    }

    // ===== Singularity Tests =====

    #[test]
    fn test_singularity_aligned_bones() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;

        // Target that would make bones perfectly aligned (fully extended)
        let target = Vec3::new(0.0, 4.0, 0.0);
        let params = TwoBoneIkParams::with_target(target);
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        // Should handle gracefully
        assert!(result.mid_rotation.is_normalized());
        assert!(result.root_rotation.is_normalized());
    }

    #[test]
    fn test_singularity_zero_length_upper() {
        let root_pos = Vec3::ZERO;
        let upper_length = 0.0;
        let lower_length = 2.0;

        let target = Vec3::new(0.0, 1.0, 0.0);
        let params = TwoBoneIkParams::with_target(target);
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        // Should fail gracefully for degenerate case
        assert!(!result.success);
        assert!(result.mid_rotation.is_normalized());
        assert!(result.root_rotation.is_normalized());
    }

    #[test]
    fn test_singularity_zero_length_lower() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 0.0;

        let target = Vec3::new(0.0, 1.0, 0.0);
        let params = TwoBoneIkParams::with_target(target);
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        // Should fail gracefully for degenerate case
        assert!(!result.success);
        assert!(result.mid_rotation.is_normalized());
        assert!(result.root_rotation.is_normalized());
    }

    #[test]
    fn test_singularity_both_zero_length() {
        let root_pos = Vec3::ZERO;
        let upper_length = 0.0;
        let lower_length = 0.0;

        let target = Vec3::new(0.0, 1.0, 0.0);
        let params = TwoBoneIkParams::with_target(target);
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        // Should fail gracefully for degenerate case
        assert!(!result.success);
        assert!(result.mid_rotation.is_normalized());
        assert!(result.root_rotation.is_normalized());
    }

    #[test]
    fn test_singularity_very_small_target_distance() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;

        let target = Vec3::new(0.0, 0.0001, 0.0);
        let params = TwoBoneIkParams::with_target(target);
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        // Should handle gracefully
        assert!(result.mid_rotation.is_normalized());
        assert!(result.root_rotation.is_normalized());
    }

    // ===== Limb Configuration Tests =====

    #[test]
    fn test_arm_configuration() {
        // Typical arm: upper arm 0.3m, forearm 0.25m
        let root_pos = Vec3::ZERO;
        let upper_length = 0.3;
        let lower_length = 0.25;

        // Reach forward and down
        let target = Vec3::new(0.3, -0.2, 0.3);
        let params = TwoBoneIkParams::with_target_and_pole(target, Vec3::NEG_Z);
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        // Should be reachable
        let max_reach = upper_length + lower_length;
        let target_dist = target.length();
        if target_dist <= max_reach {
            assert!(result.success);
        }
    }

    #[test]
    fn test_leg_configuration() {
        // Typical leg: thigh 0.45m, shin 0.4m
        let root_pos = Vec3::ZERO;
        let upper_length = 0.45;
        let lower_length = 0.4;

        // Step forward
        let target = Vec3::new(0.3, -0.7, 0.0);
        let params = TwoBoneIkParams::with_target_and_pole(target, Vec3::Z); // Knee forward
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        let max_reach = upper_length + lower_length;
        let target_dist = target.length();
        if target_dist <= max_reach {
            assert!(result.success);
        }
    }

    // ===== Full Solve with Skeleton Tests =====

    #[test]
    fn test_full_solve_basic() {
        let skeleton = create_arm_skeleton();
        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = create_arm_chain();

        // Target within reach
        let params = TwoBoneIkParams::with_target(Vec3::new(0.0, 3.0, 1.0));
        let result = solve_two_bone_ik(&chain, &skeleton, &pose, &params);

        assert!(result.success);
        assert!(result.root_rotation.is_normalized());
        assert!(result.mid_rotation.is_normalized());
    }

    #[test]
    fn test_full_solve_extended() {
        let skeleton = create_arm_skeleton();
        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = create_arm_chain();

        // Target at max reach
        let params = TwoBoneIkParams::with_target(Vec3::new(0.0, 4.0, 0.0));
        let result = solve_two_bone_ik(&chain, &skeleton, &pose, &params);

        assert!(result.success);
        // Should be near fully extended
        assert!((result.mid_angle - PI).abs() < 0.1);
    }

    #[test]
    fn test_full_solve_contracted() {
        let skeleton = create_asymmetric_skeleton();
        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = create_arm_chain();

        // Target at min reach (|3-2| = 1)
        let params = TwoBoneIkParams::with_target(Vec3::new(0.0, 1.0, 0.0));
        let result = solve_two_bone_ik(&chain, &skeleton, &pose, &params);

        // Should succeed (just barely reachable)
        assert!(result.success);
    }

    #[test]
    fn test_full_solve_unreachable() {
        let skeleton = create_arm_skeleton();
        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = create_arm_chain();

        // Target way out of reach
        let params = TwoBoneIkParams::with_target(Vec3::new(0.0, 10.0, 0.0));
        let result = solve_two_bone_ik(&chain, &skeleton, &pose, &params);

        assert!(!result.success);
        assert!(result.distance_to_target > 0.0);
    }

    // ===== Position-based Solve Tests =====

    #[test]
    fn test_solve_positions_basic() {
        let root = Vec3::ZERO;
        let mid = Vec3::new(0.0, 2.0, 0.0);
        let end = Vec3::new(0.0, 4.0, 0.0);
        let target = Vec3::new(0.0, 3.0, 1.0);

        let (root_rot, mid_rot, success, _) = solve_two_bone_ik_positions(
            root, mid, end, target, Vec3::NEG_Z, 0.1, PI - 0.1
        );

        assert!(success);
        assert!(root_rot.is_normalized());
        assert!(mid_rot.is_normalized());
    }

    // ===== Bone Length Calculation Tests =====

    #[test]
    fn test_calculate_bone_lengths() {
        let skeleton = create_arm_skeleton();
        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = create_arm_chain();

        let (upper, lower) = calculate_bone_lengths(&chain, &skeleton, &pose);

        assert!((upper - 2.0).abs() < TEST_EPSILON);
        assert!((lower - 2.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_calculate_bone_lengths_asymmetric() {
        let skeleton = create_asymmetric_skeleton();
        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = create_arm_chain();

        let (upper, lower) = calculate_bone_lengths(&chain, &skeleton, &pose);

        assert!((upper - 3.0).abs() < TEST_EPSILON);
        assert!((lower - 2.0).abs() < TEST_EPSILON);
    }

    // ===== Helper Function Tests =====

    #[test]
    fn test_angle_between() {
        // Same direction
        let angle = angle_between(Vec3::X, Vec3::X);
        assert!(angle.abs() < TEST_EPSILON);

        // Opposite directions
        let angle = angle_between(Vec3::X, Vec3::NEG_X);
        assert!((angle - PI).abs() < TEST_EPSILON);

        // Perpendicular
        let angle = angle_between(Vec3::X, Vec3::Y);
        assert!((angle - PI / 2.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_get_perpendicular() {
        for v in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(1.0, 2.0, 3.0).normalize()] {
            let perp = get_perpendicular(v);
            let dot = v.dot(perp);
            assert!(dot.abs() < TEST_EPSILON, "Perpendicular vector should be orthogonal");
            assert!((perp.length() - 1.0).abs() < TEST_EPSILON, "Perpendicular should be normalized");
        }
    }

    #[test]
    fn test_rotation_between_vectors() {
        let from = Vec3::X;
        let to = Vec3::Y;
        let rot = rotation_between_vectors(from, to);

        let result = rot * from;
        assert!(result.abs_diff_eq(to, TEST_EPSILON));
    }

    #[test]
    fn test_rotation_between_vectors_same() {
        let v = Vec3::new(1.0, 2.0, 3.0).normalize();
        let rot = rotation_between_vectors(v, v);
        assert!(rot.abs_diff_eq(Quat::IDENTITY, TEST_EPSILON));
    }

    #[test]
    fn test_rotation_between_vectors_opposite() {
        let from = Vec3::X;
        let to = Vec3::NEG_X;
        let rot = rotation_between_vectors(from, to);

        let result = rot * from;
        assert!(result.abs_diff_eq(to, TEST_EPSILON));
    }

    #[test]
    fn test_apply_angle_constraints_no_clamp() {
        let angle = apply_angle_constraints(1.5, 0.5, 2.5, 0.0);
        assert!((angle - 1.5).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_apply_angle_constraints_clamp_min() {
        let angle = apply_angle_constraints(0.3, 0.5, 2.5, 0.0);
        assert!((angle - 0.5).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_apply_angle_constraints_clamp_max() {
        let angle = apply_angle_constraints(3.0, 0.5, 2.5, 0.0);
        assert!((angle - 2.5).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_smooth_step() {
        assert!((smooth_step(0.0) - 0.0).abs() < TEST_EPSILON);
        assert!((smooth_step(1.0) - 1.0).abs() < TEST_EPSILON);
        assert!((smooth_step(0.5) - 0.5).abs() < TEST_EPSILON);
    }

    // ===== Edge Cases =====

    #[test]
    fn test_negative_target_positions() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;

        for target in [
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(0.0, -3.0, 0.0),
            Vec3::new(-2.0, 2.0, -1.0),
        ] {
            let params = TwoBoneIkParams::with_target(target);
            let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

            let dist = target.length();
            if dist <= upper_length + lower_length && dist >= (upper_length - lower_length).abs() && dist > EPSILON {
                assert!(result.success, "Target {:?} should be reachable", target);
            }
        }
    }

    #[test]
    fn test_non_origin_root() {
        let root_pos = Vec3::new(5.0, 10.0, -3.0);
        let upper_length = 2.0;
        let lower_length = 2.0;

        // Target relative to root
        let target = root_pos + Vec3::new(0.0, 3.0, 0.0);
        let params = TwoBoneIkParams::with_target(target);
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        assert!(result.success);
    }

    #[test]
    fn test_rotated_initial_orientation() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let initial_rot = Quat::from_rotation_z(PI / 4.0);

        let target = Vec3::new(0.0, 3.0, 0.0);
        let params = TwoBoneIkParams::with_target(target);
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, initial_rot, &params);

        assert!(result.success);
        assert!(result.root_rotation.is_normalized());
    }

    #[test]
    fn test_equal_bone_lengths() {
        let root_pos = Vec3::ZERO;
        let length = 2.0;

        let target = Vec3::new(0.0, 3.0, 0.0);
        let params = TwoBoneIkParams::with_target(target);
        let result = solve_two_bone_ik_world(root_pos, length, length, Quat::IDENTITY, &params);

        assert!(result.success);
    }

    #[test]
    fn test_unequal_bone_lengths() {
        let root_pos = Vec3::ZERO;

        // Very unequal lengths
        let upper = 4.0;
        let lower = 1.0;

        let target = Vec3::new(0.0, 4.0, 0.0);
        let params = TwoBoneIkParams::with_target(target);
        let result = solve_two_bone_ik_world(root_pos, upper, lower, Quat::IDENTITY, &params);

        assert!(result.success);
    }

    #[test]
    fn test_multiple_solves_consistent() {
        let root_pos = Vec3::ZERO;
        let upper_length = 2.0;
        let lower_length = 2.0;
        let target = Vec3::new(1.0, 2.0, 1.0);
        let params = TwoBoneIkParams::with_target(target);

        // Solve multiple times - results should be identical
        let result1 = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);
        let result2 = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        assert!(result1.mid_rotation.abs_diff_eq(result2.mid_rotation, TEST_EPSILON));
        assert!(result1.root_rotation.abs_diff_eq(result2.root_rotation, TEST_EPSILON));
        assert_eq!(result1.success, result2.success);
    }

    // ===== Law of Cosines Verification =====

    #[test]
    fn test_law_of_cosines_90_degrees() {
        // 3-4-5 right triangle: angle opposite to 5 should be 90 degrees
        let root_pos = Vec3::ZERO;
        let upper_length = 3.0;
        let lower_length = 4.0;

        // Target at distance 5 (pythagorean triple)
        let target = Vec3::new(0.0, 5.0, 0.0);
        let params = TwoBoneIkParams::with_target(target);
        let result = solve_two_bone_ik_world(root_pos, upper_length, lower_length, Quat::IDENTITY, &params);

        // The internal angle at mid should be 90 degrees (PI/2)
        // Using law of cosines: cos(C) = (a^2 + b^2 - c^2) / (2ab)
        // cos(C) = (9 + 16 - 25) / (2*3*4) = 0 / 24 = 0 => C = 90 degrees
        assert!((result.mid_angle - PI / 2.0).abs() < 0.01);
    }

    #[test]
    fn test_law_of_cosines_60_degrees() {
        // Equilateral setup: all sides equal, all angles 60 degrees
        let root_pos = Vec3::ZERO;
        let length = 2.0;

        // Target at same distance as bone lengths
        let target = Vec3::new(0.0, length, 0.0);
        let params = TwoBoneIkParams::with_target(target);
        let result = solve_two_bone_ik_world(root_pos, length, length, Quat::IDENTITY, &params);

        // Internal angle should be 60 degrees (PI/3)
        // cos(60) = (4 + 4 - 4) / (2*2*2) = 4/8 = 0.5
        assert!((result.mid_angle - PI / 3.0).abs() < 0.01);
    }
}
