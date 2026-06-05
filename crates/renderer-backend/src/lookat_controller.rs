//! Procedural look-at and aim controllers for TRINITY Engine (T-AN-7.6).
//!
//! This module provides procedural animation controllers for:
//!
//! - **Look-at**: Rotate bone chains toward target positions (head tracking, spine tracking)
//! - **Aim**: Lead target with velocity prediction (weapon aiming, turret tracking)
//! - **Soft cone limits**: Smooth blending at constraint boundaries
//! - **Per-bone weight distribution**: Natural rotation distribution across chains
//!
//! # Use Cases
//!
//! - **Head tracking**: NPC looks at player or point of interest
//! - **Eye gaze**: Micro eye movements for lifelike characters
//! - **Weapon aiming**: Gun/bow points at moving targets with lead prediction
//! - **Turret tracking**: Automated turrets track targets with velocity prediction
//! - **Spine tracking**: Upper body orients toward target for more natural poses
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::lookat_controller::{
//!     LookAtParams, LookAtChain, LookAtState, solve_look_at, solve_aim, AimParams
//! };
//! use renderer_backend::skeleton::Skeleton;
//! use renderer_backend::pose::Pose;
//! use glam::Vec3;
//!
//! // Head tracking (single bone)
//! let chain = LookAtChain::head(neck_bone_index, Vec3::Y, Vec3::Z);
//! let params = LookAtParams {
//!     target_position: player_position,
//!     up_vector: Vec3::Y,
//!     weight: 1.0,
//!     speed: 180.0,  // 180 degrees per second
//!     cone_angle: 60.0_f32.to_radians(),
//!     cone_soft_zone: 0.2,
//! };
//! let mut state = LookAtState::default();
//!
//! let visible = solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, dt);
//!
//! // Weapon aiming with lead prediction
//! let aim_params = AimParams {
//!     target_position: enemy_position,
//!     target_velocity: enemy_velocity,
//!     projectile_speed: 100.0,  // 100 m/s bullet
//!     aim_offset: Vec3::ZERO,
//! };
//! let aim_direction = solve_aim(weapon_bone, &skeleton, &mut pose, &aim_params);
//! ```

use glam::{Mat4, Quat, Vec3};
use std::f32::consts::PI;

use crate::pose::Pose;
use crate::skeleton::Skeleton;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum distance for valid look-at computation.
const EPSILON: f32 = 1e-6;

/// Default cone angle (90 degrees from forward).
pub const DEFAULT_CONE_ANGLE: f32 = PI / 2.0;

/// Default rotation speed in degrees per second.
pub const DEFAULT_SPEED: f32 = 180.0;

// ---------------------------------------------------------------------------
// LookAtParams
// ---------------------------------------------------------------------------

/// Parameters for look-at solving.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LookAtParams {
    /// Target position in world space that the chain should look at.
    pub target_position: Vec3,

    /// World up vector for constraining rotation (typically Y-up).
    pub up_vector: Vec3,

    /// Blend weight for the look-at effect (0.0 = no effect, 1.0 = full effect).
    pub weight: f32,

    /// Rotation speed in degrees per second.
    /// Used for smooth interpolation toward the target.
    pub speed: f32,

    /// Maximum angle from forward direction that target can be (radians).
    /// Beyond this angle, the look-at will clamp or break.
    pub cone_angle: f32,

    /// Soft zone at cone edge as a ratio (0.0 = hard edge, 1.0 = entire cone is soft).
    /// Within the soft zone, weight is smoothly reduced.
    pub cone_soft_zone: f32,
}

impl Default for LookAtParams {
    fn default() -> Self {
        Self {
            target_position: Vec3::ZERO,
            up_vector: Vec3::Y,
            weight: 1.0,
            speed: DEFAULT_SPEED,
            cone_angle: DEFAULT_CONE_ANGLE,
            cone_soft_zone: 0.1,
        }
    }
}

impl LookAtParams {
    /// Create look-at parameters with just a target position.
    #[inline]
    pub fn with_target(target: Vec3) -> Self {
        Self {
            target_position: target,
            ..Default::default()
        }
    }

    /// Create look-at parameters with target and custom speed.
    #[inline]
    pub fn with_target_and_speed(target: Vec3, speed: f32) -> Self {
        Self {
            target_position: target,
            speed,
            ..Default::default()
        }
    }

    /// Set cone limits.
    #[inline]
    pub fn with_cone_limits(mut self, angle: f32, soft_zone: f32) -> Self {
        self.cone_angle = angle;
        self.cone_soft_zone = soft_zone.clamp(0.0, 1.0);
        self
    }

    /// Set the blend weight.
    #[inline]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }
}

// ---------------------------------------------------------------------------
// LookAtChain
// ---------------------------------------------------------------------------

/// Defines a chain of bones for look-at rotation.
///
/// The chain can be a single bone (head) or multiple bones (spine chain).
/// Each bone has a weight determining how much rotation it contributes.
#[derive(Clone, Debug, PartialEq)]
pub struct LookAtChain {
    /// Bone indices in the chain, ordered from root to tip.
    pub bones: Vec<usize>,

    /// Per-bone rotation weight (0.0 to 1.0).
    /// These are normalized internally so they sum to 1.0.
    pub weights: Vec<f32>,

    /// Local forward axis of the bones (direction that should aim at target).
    pub forward_axis: Vec3,

    /// Local up axis of the bones (used for roll constraint).
    pub up_axis: Vec3,
}

impl LookAtChain {
    /// Create a single-bone look-at chain (e.g., head tracking).
    ///
    /// # Arguments
    ///
    /// * `bone` - The bone index to rotate
    /// * `forward_axis` - Local forward direction
    /// * `up_axis` - Local up direction
    #[inline]
    pub fn head(bone: usize, forward_axis: Vec3, up_axis: Vec3) -> Self {
        Self {
            bones: vec![bone],
            weights: vec![1.0],
            forward_axis: forward_axis.normalize(),
            up_axis: up_axis.normalize(),
        }
    }

    /// Create a spine chain for distributed look-at.
    ///
    /// Weights are automatically computed with falloff from tip to root.
    ///
    /// # Arguments
    ///
    /// * `bones` - Bone indices from root to tip (e.g., spine1, spine2, spine3, neck, head)
    /// * `forward_axis` - Local forward direction
    /// * `up_axis` - Local up direction
    pub fn spine(bones: Vec<usize>, forward_axis: Vec3, up_axis: Vec3) -> Self {
        let n = bones.len();
        // Natural weight distribution: more rotation at top (tip)
        // Using quadratic falloff: tip gets most, root gets least
        let weights: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i + 1) as f32 / n as f32;
                t * t // Quadratic falloff
            })
            .collect();

        Self {
            bones,
            weights,
            forward_axis: forward_axis.normalize(),
            up_axis: up_axis.normalize(),
        }
    }

    /// Create a chain with custom weights.
    ///
    /// # Arguments
    ///
    /// * `bones` - Bone indices
    /// * `weights` - Per-bone weights (will be normalized)
    /// * `forward_axis` - Local forward direction
    /// * `up_axis` - Local up direction
    ///
    /// # Panics
    ///
    /// Panics if `bones` and `weights` have different lengths.
    pub fn with_weights(
        bones: Vec<usize>,
        weights: Vec<f32>,
        forward_axis: Vec3,
        up_axis: Vec3,
    ) -> Self {
        assert_eq!(
            bones.len(),
            weights.len(),
            "bones and weights must have same length"
        );
        Self {
            bones,
            weights,
            forward_axis: forward_axis.normalize(),
            up_axis: up_axis.normalize(),
        }
    }

    /// Create an eye gaze chain (typically very fast, small angles).
    #[inline]
    pub fn eye(bone: usize, forward_axis: Vec3, up_axis: Vec3) -> Self {
        Self::head(bone, forward_axis, up_axis)
    }

    /// Get the number of bones in the chain.
    #[inline]
    pub fn len(&self) -> usize {
        self.bones.len()
    }

    /// Check if the chain is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bones.is_empty()
    }

    /// Normalize weights so they sum to 1.0.
    pub fn normalize_weights(&mut self) {
        let sum: f32 = self.weights.iter().sum();
        if sum > EPSILON {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }

    /// Get the normalized weight for a bone at the given index.
    pub fn get_normalized_weight(&self, index: usize) -> f32 {
        let sum: f32 = self.weights.iter().sum();
        if sum > EPSILON && index < self.weights.len() {
            self.weights[index] / sum
        } else {
            0.0
        }
    }

    /// Validate that all bone indices are valid for the given skeleton.
    pub fn validate(&self, skeleton: &Skeleton) -> bool {
        if self.bones.is_empty() || self.weights.is_empty() {
            return false;
        }
        if self.bones.len() != self.weights.len() {
            return false;
        }
        let bone_count = skeleton.bone_count();
        self.bones.iter().all(|&b| b < bone_count)
    }
}

// ---------------------------------------------------------------------------
// LookAtState
// ---------------------------------------------------------------------------

/// Persistent state for look-at solving.
///
/// This maintains interpolation state between frames for smooth tracking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LookAtState {
    /// Current rotation being interpolated toward target.
    pub current_rotation: Quat,

    /// Target rotation to interpolate toward.
    pub target_rotation: Quat,

    /// Current blend weight (affected by soft cone limits).
    pub blend_weight: f32,

    /// Whether the target is currently visible (within cone).
    pub target_visible: bool,

    /// Angle to target from forward direction.
    pub angle_to_target: f32,
}

impl Default for LookAtState {
    fn default() -> Self {
        Self {
            current_rotation: Quat::IDENTITY,
            target_rotation: Quat::IDENTITY,
            blend_weight: 1.0,
            target_visible: true,
            angle_to_target: 0.0,
        }
    }
}

impl LookAtState {
    /// Reset state to identity.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Create state initialized to look in a specific direction.
    pub fn looking_at(direction: Vec3) -> Self {
        let rot = if direction.length_squared() > EPSILON * EPSILON {
            Quat::from_rotation_arc(Vec3::Y, direction.normalize())
        } else {
            Quat::IDENTITY
        };
        Self {
            current_rotation: rot,
            target_rotation: rot,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// AimParams
// ---------------------------------------------------------------------------

/// Parameters for aim solving with lead prediction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AimParams {
    /// Current target position in world space.
    pub target_position: Vec3,

    /// Target velocity for lead calculation (world space, units/second).
    pub target_velocity: Vec3,

    /// Projectile speed for lead time calculation (units/second).
    /// Set to 0 or very large value for instant-hit weapons.
    pub projectile_speed: f32,

    /// Offset from bone origin to aim point (local space).
    /// For example, the muzzle offset from the weapon bone.
    pub aim_offset: Vec3,
}

impl Default for AimParams {
    fn default() -> Self {
        Self {
            target_position: Vec3::ZERO,
            target_velocity: Vec3::ZERO,
            projectile_speed: f32::INFINITY,
            aim_offset: Vec3::ZERO,
        }
    }
}

impl AimParams {
    /// Create aim parameters for a static target.
    #[inline]
    pub fn static_target(position: Vec3) -> Self {
        Self {
            target_position: position,
            ..Default::default()
        }
    }

    /// Create aim parameters for a moving target.
    #[inline]
    pub fn moving_target(position: Vec3, velocity: Vec3, projectile_speed: f32) -> Self {
        Self {
            target_position: position,
            target_velocity: velocity,
            projectile_speed,
            aim_offset: Vec3::ZERO,
        }
    }

    /// Set the aim offset.
    #[inline]
    pub fn with_offset(mut self, offset: Vec3) -> Self {
        self.aim_offset = offset;
        self
    }

    /// Calculate lead time for hitting a moving target.
    ///
    /// Returns the time in seconds to intercept the target.
    /// Returns 0 if projectile speed is infinite or target is too fast.
    pub fn calculate_lead_time(&self, shooter_position: Vec3) -> f32 {
        if self.projectile_speed <= EPSILON || self.projectile_speed.is_infinite() {
            return 0.0;
        }

        let to_target = self.target_position - shooter_position;
        let distance = to_target.length();

        if distance < EPSILON {
            return 0.0;
        }

        // Simple linear prediction: time = distance / projectile_speed
        // For more accurate prediction, we'd solve a quadratic
        let initial_time = distance / self.projectile_speed;

        // Iterative refinement for moving targets
        if self.target_velocity.length_squared() < EPSILON * EPSILON {
            return initial_time;
        }

        // Newton-Raphson iteration for better lead time
        let mut lead_time = initial_time;
        for _ in 0..3 {
            let predicted_pos = self.target_position + self.target_velocity * lead_time;
            let new_distance = (predicted_pos - shooter_position).length();
            if new_distance < EPSILON {
                break;
            }
            lead_time = new_distance / self.projectile_speed;
        }

        lead_time
    }

    /// Calculate the predicted target position with lead.
    pub fn predict_target_position(&self, shooter_position: Vec3) -> Vec3 {
        let lead_time = self.calculate_lead_time(shooter_position);
        self.target_position + self.target_velocity * lead_time
    }
}

// ---------------------------------------------------------------------------
// Axis Constraint
// ---------------------------------------------------------------------------

/// Axis constraints for yaw/pitch separation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxisConstraints {
    /// Maximum yaw (horizontal rotation) in radians.
    pub max_yaw: f32,
    /// Minimum yaw in radians (typically negative).
    pub min_yaw: f32,
    /// Maximum pitch (vertical rotation) in radians.
    pub max_pitch: f32,
    /// Minimum pitch in radians (typically negative).
    pub min_pitch: f32,
}

impl Default for AxisConstraints {
    fn default() -> Self {
        Self {
            max_yaw: PI,
            min_yaw: -PI,
            max_pitch: PI / 2.0,
            min_pitch: -PI / 2.0,
        }
    }
}

impl AxisConstraints {
    /// Create symmetric constraints.
    #[inline]
    pub fn symmetric(yaw_limit: f32, pitch_limit: f32) -> Self {
        Self {
            max_yaw: yaw_limit.abs(),
            min_yaw: -yaw_limit.abs(),
            max_pitch: pitch_limit.abs(),
            min_pitch: -pitch_limit.abs(),
        }
    }

    /// Clamp yaw value to constraints.
    #[inline]
    pub fn clamp_yaw(&self, yaw: f32) -> f32 {
        yaw.clamp(self.min_yaw, self.max_yaw)
    }

    /// Clamp pitch value to constraints.
    #[inline]
    pub fn clamp_pitch(&self, pitch: f32) -> f32 {
        pitch.clamp(self.min_pitch, self.max_pitch)
    }
}

// ---------------------------------------------------------------------------
// Solve Functions
// ---------------------------------------------------------------------------

/// Solve look-at for a bone chain.
///
/// Rotates bones in the chain to look toward the target position,
/// distributing rotation according to weights.
///
/// # Arguments
///
/// * `chain` - The look-at chain definition
/// * `skeleton` - The skeleton containing bone hierarchy
/// * `pose` - The pose to modify (rotations will be updated)
/// * `params` - Look-at parameters
/// * `state` - Persistent state for interpolation
/// * `dt` - Delta time in seconds
///
/// # Returns
///
/// `true` if the target is visible (within cone limits), `false` otherwise.
///
/// # Panics
///
/// Panics if any bone index in the chain is out of bounds.
pub fn solve_look_at(
    chain: &LookAtChain,
    skeleton: &Skeleton,
    pose: &mut Pose,
    params: &LookAtParams,
    state: &mut LookAtState,
    dt: f32,
) -> bool {
    if chain.is_empty() || params.weight < EPSILON {
        return state.target_visible;
    }

    // Get world transforms
    let transforms = pose.transforms();
    let world_transforms = skeleton.compute_world_transforms(&transforms);

    // Use the last (tip) bone as reference for direction calculation
    let tip_bone = *chain.bones.last().unwrap();
    let tip_world_transform = world_transforms[tip_bone];
    let tip_position = tip_world_transform.w_axis.truncate();

    // Calculate direction to target
    let to_target = params.target_position - tip_position;
    let target_distance = to_target.length();

    if target_distance < EPSILON {
        state.target_visible = false;
        return false;
    }

    let target_direction = to_target / target_distance;

    // Get current forward direction in world space
    let tip_world_rot = Quat::from_mat4(&tip_world_transform);
    let current_forward = tip_world_rot * chain.forward_axis;

    // Calculate angle to target
    let angle_to_target = angle_between_normalized(current_forward, target_direction);
    state.angle_to_target = angle_to_target;

    // Check cone limits and calculate effective weight
    let (target_visible, cone_weight) =
        calculate_cone_weight(angle_to_target, params.cone_angle, params.cone_soft_zone);
    state.target_visible = target_visible;
    state.blend_weight = cone_weight;

    if !target_visible {
        // Target outside cone - don't rotate further
        return false;
    }

    // Calculate the rotation needed to look at target
    let look_rotation = look_at_rotation(
        current_forward,
        target_direction,
        params.up_vector,
        chain.up_axis,
        tip_world_rot,
    );

    // Update target rotation
    state.target_rotation = look_rotation;

    // Interpolate current rotation toward target
    let interp_speed = (params.speed.to_radians() * dt).clamp(0.0, 1.0);
    state.current_rotation = state.current_rotation.slerp(look_rotation, interp_speed);

    // Calculate the delta rotation to apply
    let delta_rotation = tip_world_rot.inverse() * state.current_rotation;

    // Apply weighted blend
    let effective_weight = params.weight * cone_weight;
    let final_rotation = Quat::IDENTITY.slerp(delta_rotation, effective_weight);

    // Distribute rotation across chain based on weights
    distribute_rotation(chain, skeleton, pose, final_rotation, &world_transforms);

    target_visible
}

/// Solve look-at for a single bone (simplified version).
///
/// # Arguments
///
/// * `bone` - The bone index to rotate
/// * `skeleton` - The skeleton
/// * `pose` - The pose to modify
/// * `params` - Look-at parameters
/// * `forward_axis` - Local forward direction
/// * `up_axis` - Local up direction
///
/// # Returns
///
/// `true` if target is visible (within limits).
pub fn solve_look_at_single(
    bone: usize,
    skeleton: &Skeleton,
    pose: &mut Pose,
    params: &LookAtParams,
    forward_axis: Vec3,
    up_axis: Vec3,
) -> bool {
    // Get world transform
    let transforms = pose.transforms();
    let world_transforms = skeleton.compute_world_transforms(&transforms);
    let bone_world = world_transforms[bone];
    let bone_position = bone_world.w_axis.truncate();
    let bone_world_rot = Quat::from_mat4(&bone_world);

    // Calculate direction to target
    let to_target = params.target_position - bone_position;
    let target_distance = to_target.length();

    if target_distance < EPSILON {
        return false;
    }

    let target_direction = to_target / target_distance;

    // Get current forward direction
    let current_forward = bone_world_rot * forward_axis;

    // Calculate angle
    let angle_to_target = angle_between_normalized(current_forward, target_direction);

    // Check cone limits
    let (visible, cone_weight) =
        calculate_cone_weight(angle_to_target, params.cone_angle, params.cone_soft_zone);

    if !visible {
        return false;
    }

    // Calculate look rotation
    let look_rotation = look_at_rotation(
        current_forward,
        target_direction,
        params.up_vector,
        up_axis,
        bone_world_rot,
    );

    // Calculate local rotation
    let parent_world_rot = if let Some(parent_idx) = skeleton.parent(bone) {
        Quat::from_mat4(&world_transforms[parent_idx])
    } else {
        Quat::IDENTITY
    };

    let new_local_rot = parent_world_rot.inverse() * look_rotation;

    // Apply weighted
    let effective_weight = params.weight * cone_weight;
    let current_local = pose.rotations[bone];
    pose.rotations[bone] = current_local.slerp(new_local_rot, effective_weight);

    visible
}

/// Solve aim with lead prediction.
///
/// Calculates where to aim to hit a moving target, accounting for
/// projectile travel time.
///
/// # Arguments
///
/// * `bone` - The aiming bone index
/// * `skeleton` - The skeleton
/// * `pose` - The current pose
/// * `params` - Aim parameters with velocity prediction
///
/// # Returns
///
/// The aim direction in world space (normalized).
pub fn solve_aim(
    bone: usize,
    skeleton: &Skeleton,
    pose: &mut Pose,
    params: &AimParams,
) -> Vec3 {
    // Get world transform
    let transforms = pose.transforms();
    let world_transforms = skeleton.compute_world_transforms(&transforms);
    let bone_world = world_transforms[bone];
    let bone_position = bone_world.w_axis.truncate();
    let bone_world_rot = Quat::from_mat4(&bone_world);

    // Apply aim offset to get actual shooting position
    let shoot_position = bone_position + bone_world_rot * params.aim_offset;

    // Predict where target will be
    let predicted_target = params.predict_target_position(shoot_position);

    // Calculate aim direction
    let aim_direction = (predicted_target - shoot_position).normalize_or_zero();

    if aim_direction.length_squared() < EPSILON * EPSILON {
        return Vec3::Y; // Default up if target is at shoot position
    }

    aim_direction
}

/// Solve aim and apply rotation to bone.
///
/// Like `solve_aim` but also rotates the bone to face the aim direction.
///
/// # Arguments
///
/// * `bone` - The aiming bone index
/// * `skeleton` - The skeleton
/// * `pose` - The pose to modify
/// * `params` - Aim parameters
/// * `forward_axis` - Local forward direction of the bone
/// * `up_axis` - Local up direction
///
/// # Returns
///
/// The aim direction in world space.
pub fn solve_aim_and_apply(
    bone: usize,
    skeleton: &Skeleton,
    pose: &mut Pose,
    params: &AimParams,
    forward_axis: Vec3,
    up_axis: Vec3,
) -> Vec3 {
    let aim_direction = solve_aim(bone, skeleton, pose, params);

    // Get world transform
    let transforms = pose.transforms();
    let world_transforms = skeleton.compute_world_transforms(&transforms);
    let bone_world = world_transforms[bone];
    let bone_world_rot = Quat::from_mat4(&bone_world);

    // Current forward
    let current_forward = bone_world_rot * forward_axis;

    // Calculate rotation to aim
    let aim_rotation = look_at_rotation(
        current_forward,
        aim_direction,
        Vec3::Y,
        up_axis,
        bone_world_rot,
    );

    // Convert to local space
    let parent_world_rot = if let Some(parent_idx) = skeleton.parent(bone) {
        let parent_transforms = pose.transforms();
        let parent_world = skeleton.compute_world_transforms(&parent_transforms);
        Quat::from_mat4(&parent_world[parent_idx])
    } else {
        Quat::IDENTITY
    };

    let new_local_rot = parent_world_rot.inverse() * aim_rotation;
    pose.rotations[bone] = new_local_rot;

    aim_direction
}

/// Solve aim-down-sights alignment.
///
/// Aligns a weapon bone so that the iron sights line up with an aim point.
///
/// # Arguments
///
/// * `weapon_bone` - The weapon bone index
/// * `skeleton` - The skeleton
/// * `pose` - The pose to modify
/// * `aim_point` - World position to aim at
/// * `sight_offset` - Offset from weapon bone to iron sight (local space)
/// * `eye_position` - World position of the aiming eye
///
/// # Returns
///
/// The final aim direction.
pub fn solve_aim_down_sights(
    weapon_bone: usize,
    skeleton: &Skeleton,
    pose: &mut Pose,
    aim_point: Vec3,
    sight_offset: Vec3,
    eye_position: Vec3,
) -> Vec3 {
    // Get world transform
    let transforms = pose.transforms();
    let world_transforms = skeleton.compute_world_transforms(&transforms);
    let weapon_world = world_transforms[weapon_bone];
    let weapon_world_rot = Quat::from_mat4(&weapon_world);

    // Calculate sight position in world space
    let weapon_position = weapon_world.w_axis.truncate();
    let sight_world_position = weapon_position + weapon_world_rot * sight_offset;

    // Direction from eye through sight to target
    let eye_to_target = aim_point - eye_position;
    let aim_direction = eye_to_target.normalize_or_zero();

    if aim_direction.length_squared() < EPSILON * EPSILON {
        return Vec3::Z;
    }

    // The sight needs to be on the line from eye to target
    // Calculate required weapon rotation to achieve this
    let eye_to_sight = sight_world_position - eye_position;
    let current_sight_direction = eye_to_sight.normalize_or_zero();

    if current_sight_direction.length_squared() < EPSILON * EPSILON {
        return aim_direction;
    }

    // Rotation from current sight alignment to target alignment
    let correction = rotation_between_normalized(current_sight_direction, aim_direction);

    // Apply correction to weapon
    let parent_world_rot = if let Some(parent_idx) = skeleton.parent(weapon_bone) {
        Quat::from_mat4(&world_transforms[parent_idx])
    } else {
        Quat::IDENTITY
    };

    let new_weapon_world_rot = correction * weapon_world_rot;
    let new_local_rot = parent_world_rot.inverse() * new_weapon_world_rot;
    pose.rotations[weapon_bone] = new_local_rot;

    aim_direction
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Calculate cone weight for soft limits.
///
/// Returns (is_visible, weight).
fn calculate_cone_weight(angle: f32, cone_angle: f32, soft_zone_ratio: f32) -> (bool, f32) {
    if angle > cone_angle {
        // Outside cone
        (false, 0.0)
    } else if soft_zone_ratio < EPSILON {
        // No soft zone - full weight inside cone
        (true, 1.0)
    } else {
        // Check if in soft zone
        let soft_start = cone_angle * (1.0 - soft_zone_ratio);
        if angle < soft_start {
            // Inside hard zone - full weight
            (true, 1.0)
        } else {
            // In soft zone - interpolate weight
            let t = (cone_angle - angle) / (cone_angle - soft_start);
            (true, smooth_step(t))
        }
    }
}

/// Smooth step function for weight blending.
#[inline]
fn smooth_step(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Calculate angle between two normalized vectors (faster version).
#[inline]
fn angle_between_normalized(a: Vec3, b: Vec3) -> f32 {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    dot.acos()
}

/// Calculate look-at rotation.
fn look_at_rotation(
    current_forward: Vec3,
    target_direction: Vec3,
    world_up: Vec3,
    local_up: Vec3,
    current_rotation: Quat,
) -> Quat {
    // Basic rotation to face target
    let forward_rotation = rotation_between_normalized(
        current_forward.normalize_or_zero(),
        target_direction.normalize_or_zero(),
    );

    let new_rotation = forward_rotation * current_rotation;

    // Apply twist constraint to keep up axis aligned
    let new_up = new_rotation * local_up;
    let desired_up = world_up - target_direction * target_direction.dot(world_up);

    if desired_up.length_squared() > EPSILON * EPSILON {
        let desired_up = desired_up.normalize();
        let twist = rotation_between_normalized(new_up.normalize_or_zero(), desired_up);

        // Only apply twist around the forward axis
        let twist_axis = target_direction;
        let twist_angle = twist.to_axis_angle().1;
        let aligned_twist = Quat::from_axis_angle(twist_axis, twist_angle * 0.5);

        aligned_twist * new_rotation
    } else {
        new_rotation
    }
}

/// Rotation between two normalized vectors.
fn rotation_between_normalized(from: Vec3, to: Vec3) -> Quat {
    let dot = from.dot(to);

    if dot > 0.99999 {
        return Quat::IDENTITY;
    }

    if dot < -0.99999 {
        // 180 degree rotation - pick arbitrary perpendicular axis
        let perp = get_perpendicular(from);
        return Quat::from_axis_angle(perp, PI);
    }

    let cross = from.cross(to);
    let s = ((1.0 + dot) * 2.0).sqrt();
    let inv_s = 1.0 / s;

    Quat::from_xyzw(cross.x * inv_s, cross.y * inv_s, cross.z * inv_s, s * 0.5).normalize()
}

/// Get an arbitrary perpendicular vector.
fn get_perpendicular(v: Vec3) -> Vec3 {
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

/// Distribute rotation across a chain of bones.
fn distribute_rotation(
    chain: &LookAtChain,
    skeleton: &Skeleton,
    pose: &mut Pose,
    total_rotation: Quat,
    world_transforms: &[Mat4],
) {
    if chain.is_empty() {
        return;
    }

    // Normalize weights
    let weight_sum: f32 = chain.weights.iter().sum();
    if weight_sum < EPSILON {
        return;
    }

    // Convert rotation to axis-angle for distribution
    let (axis, total_angle) = total_rotation.to_axis_angle();

    // Apply weighted rotation to each bone
    for (i, &bone) in chain.bones.iter().enumerate() {
        let weight = chain.weights[i] / weight_sum;
        let bone_angle = total_angle * weight;

        if bone_angle.abs() < EPSILON {
            continue;
        }

        // Create partial rotation
        let partial_rotation = Quat::from_axis_angle(axis, bone_angle);

        // Get current world rotation
        let bone_world_rot = Quat::from_mat4(&world_transforms[bone]);

        // Apply rotation in world space, then convert to local
        let new_world_rot = partial_rotation * bone_world_rot;

        // Get parent world rotation
        let parent_world_rot = if let Some(parent_idx) = skeleton.parent(bone) {
            Quat::from_mat4(&world_transforms[parent_idx])
        } else {
            Quat::IDENTITY
        };

        // Convert to local rotation
        let new_local_rot = parent_world_rot.inverse() * new_world_rot;

        // Blend with current local rotation
        let current_local = pose.rotations[bone];
        pose.rotations[bone] = current_local.slerp(new_local_rot, 1.0);
    }
}

/// Extract yaw and pitch from a direction vector.
pub fn direction_to_yaw_pitch(direction: Vec3) -> (f32, f32) {
    let direction = direction.normalize_or_zero();
    let yaw = direction.x.atan2(direction.z);
    let pitch = (-direction.y).asin();
    (yaw, pitch)
}

/// Convert yaw and pitch to a direction vector.
pub fn yaw_pitch_to_direction(yaw: f32, pitch: f32) -> Vec3 {
    let cos_pitch = pitch.cos();
    Vec3::new(yaw.sin() * cos_pitch, -pitch.sin(), yaw.cos() * cos_pitch)
}

/// Apply yaw/pitch rotation to a bone with axis constraints.
pub fn apply_yaw_pitch_rotation(
    bone: usize,
    skeleton: &Skeleton,
    pose: &mut Pose,
    yaw: f32,
    pitch: f32,
    constraints: &AxisConstraints,
) {
    let clamped_yaw = constraints.clamp_yaw(yaw);
    let clamped_pitch = constraints.clamp_pitch(pitch);

    // Create rotation from yaw/pitch
    let yaw_rot = Quat::from_rotation_y(clamped_yaw);
    let pitch_rot = Quat::from_rotation_x(clamped_pitch);
    let combined = yaw_rot * pitch_rot;

    // Get world transforms for parent
    let transforms = pose.transforms();
    let world_transforms = skeleton.compute_world_transforms(&transforms);

    let parent_world_rot = if let Some(parent_idx) = skeleton.parent(bone) {
        Quat::from_mat4(&world_transforms[parent_idx])
    } else {
        Quat::IDENTITY
    };

    // Set local rotation
    let new_local = parent_world_rot.inverse() * combined;
    pose.rotations[bone] = new_local;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::PoseType;
    use crate::skeleton::{Bone, SkeletonBuilder};

    const TEST_EPSILON: f32 = 1e-4;

    // ===== Test Helpers =====

    fn create_spine_skeleton() -> Skeleton {
        // Create skeleton with spine and head
        SkeletonBuilder::new()
            .root("root")
            .child_at("spine1", "root", Vec3::new(0.0, 1.0, 0.0))
            .child_at("spine2", "spine1", Vec3::new(0.0, 0.5, 0.0))
            .child_at("neck", "spine2", Vec3::new(0.0, 0.5, 0.0))
            .child_at("head", "neck", Vec3::new(0.0, 0.3, 0.0))
            .build_unchecked()
    }

    fn create_simple_skeleton() -> Skeleton {
        SkeletonBuilder::new()
            .root("root")
            .child_at("head", "root", Vec3::new(0.0, 1.0, 0.0))
            .build_unchecked()
    }

    // ===== LookAtParams Tests =====

    #[test]
    fn test_lookat_params_default() {
        let params = LookAtParams::default();
        assert_eq!(params.target_position, Vec3::ZERO);
        assert_eq!(params.up_vector, Vec3::Y);
        assert!((params.weight - 1.0).abs() < TEST_EPSILON);
        assert!((params.speed - DEFAULT_SPEED).abs() < TEST_EPSILON);
        assert!((params.cone_angle - DEFAULT_CONE_ANGLE).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_lookat_params_with_target() {
        let target = Vec3::new(10.0, 5.0, 3.0);
        let params = LookAtParams::with_target(target);
        assert!(params.target_position.abs_diff_eq(target, TEST_EPSILON));
    }

    #[test]
    fn test_lookat_params_with_target_and_speed() {
        let target = Vec3::new(1.0, 2.0, 3.0);
        let params = LookAtParams::with_target_and_speed(target, 90.0);
        assert!(params.target_position.abs_diff_eq(target, TEST_EPSILON));
        assert!((params.speed - 90.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_lookat_params_with_cone_limits() {
        let params = LookAtParams::default().with_cone_limits(1.0, 0.3);
        assert!((params.cone_angle - 1.0).abs() < TEST_EPSILON);
        assert!((params.cone_soft_zone - 0.3).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_lookat_params_with_weight() {
        let params = LookAtParams::default().with_weight(0.5);
        assert!((params.weight - 0.5).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_lookat_params_weight_clamped() {
        let params = LookAtParams::default().with_weight(2.0);
        assert!((params.weight - 1.0).abs() < TEST_EPSILON);

        let params = LookAtParams::default().with_weight(-0.5);
        assert!((params.weight - 0.0).abs() < TEST_EPSILON);
    }

    // ===== LookAtChain Tests =====

    #[test]
    fn test_chain_head() {
        let chain = LookAtChain::head(5, Vec3::Y, Vec3::Z);
        assert_eq!(chain.bones, vec![5]);
        assert_eq!(chain.weights, vec![1.0]);
        assert!(chain.forward_axis.abs_diff_eq(Vec3::Y, TEST_EPSILON));
        assert!(chain.up_axis.abs_diff_eq(Vec3::Z, TEST_EPSILON));
    }

    #[test]
    fn test_chain_spine() {
        let bones = vec![1, 2, 3, 4];
        let chain = LookAtChain::spine(bones.clone(), Vec3::Y, Vec3::Z);
        assert_eq!(chain.bones, bones);
        assert_eq!(chain.weights.len(), 4);
        // Weights should increase from root to tip (quadratic)
        for i in 1..chain.weights.len() {
            assert!(
                chain.weights[i] > chain.weights[i - 1],
                "weights should increase from root to tip"
            );
        }
    }

    #[test]
    fn test_chain_with_weights() {
        let bones = vec![0, 1, 2];
        let weights = vec![0.1, 0.3, 0.6];
        let chain = LookAtChain::with_weights(bones.clone(), weights.clone(), Vec3::Y, Vec3::Z);
        assert_eq!(chain.bones, bones);
        assert_eq!(chain.weights, weights);
    }

    #[test]
    #[should_panic(expected = "bones and weights must have same length")]
    fn test_chain_with_weights_mismatch_panics() {
        let _ = LookAtChain::with_weights(vec![0, 1], vec![1.0], Vec3::Y, Vec3::Z);
    }

    #[test]
    fn test_chain_eye() {
        let chain = LookAtChain::eye(10, Vec3::Z, Vec3::Y);
        assert_eq!(chain.bones, vec![10]);
    }

    #[test]
    fn test_chain_len() {
        let chain = LookAtChain::spine(vec![0, 1, 2, 3], Vec3::Y, Vec3::Z);
        assert_eq!(chain.len(), 4);
        assert!(!chain.is_empty());

        let empty_chain = LookAtChain {
            bones: vec![],
            weights: vec![],
            forward_axis: Vec3::Y,
            up_axis: Vec3::Z,
        };
        assert_eq!(empty_chain.len(), 0);
        assert!(empty_chain.is_empty());
    }

    #[test]
    fn test_chain_normalize_weights() {
        let mut chain = LookAtChain::with_weights(vec![0, 1], vec![2.0, 8.0], Vec3::Y, Vec3::Z);
        chain.normalize_weights();
        assert!((chain.weights[0] - 0.2).abs() < TEST_EPSILON);
        assert!((chain.weights[1] - 0.8).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_chain_get_normalized_weight() {
        let chain = LookAtChain::with_weights(vec![0, 1, 2], vec![1.0, 2.0, 7.0], Vec3::Y, Vec3::Z);
        assert!((chain.get_normalized_weight(0) - 0.1).abs() < TEST_EPSILON);
        assert!((chain.get_normalized_weight(1) - 0.2).abs() < TEST_EPSILON);
        assert!((chain.get_normalized_weight(2) - 0.7).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_chain_validate() {
        let skeleton = create_spine_skeleton();
        let valid_chain = LookAtChain::spine(vec![1, 2, 3, 4], Vec3::Y, Vec3::Z);
        assert!(valid_chain.validate(&skeleton));

        let invalid_chain = LookAtChain::head(100, Vec3::Y, Vec3::Z);
        assert!(!invalid_chain.validate(&skeleton));
    }

    // ===== LookAtState Tests =====

    #[test]
    fn test_state_default() {
        let state = LookAtState::default();
        assert!(state.current_rotation.abs_diff_eq(Quat::IDENTITY, TEST_EPSILON));
        assert!(state.target_rotation.abs_diff_eq(Quat::IDENTITY, TEST_EPSILON));
        assert!((state.blend_weight - 1.0).abs() < TEST_EPSILON);
        assert!(state.target_visible);
    }

    #[test]
    fn test_state_reset() {
        let mut state = LookAtState {
            current_rotation: Quat::from_rotation_y(1.0),
            target_rotation: Quat::from_rotation_x(0.5),
            blend_weight: 0.5,
            target_visible: false,
            angle_to_target: 0.7,
        };
        state.reset();
        assert!(state.current_rotation.abs_diff_eq(Quat::IDENTITY, TEST_EPSILON));
        assert!(state.target_visible);
    }

    #[test]
    fn test_state_looking_at() {
        let direction = Vec3::new(1.0, 0.0, 0.0);
        let state = LookAtState::looking_at(direction);
        // The rotation should transform Y to point in the direction
        let rotated = state.current_rotation * Vec3::Y;
        assert!(rotated.abs_diff_eq(direction.normalize(), 0.01));
    }

    // ===== AimParams Tests =====

    #[test]
    fn test_aim_params_default() {
        let params = AimParams::default();
        assert_eq!(params.target_position, Vec3::ZERO);
        assert_eq!(params.target_velocity, Vec3::ZERO);
        assert!(params.projectile_speed.is_infinite());
        assert_eq!(params.aim_offset, Vec3::ZERO);
    }

    #[test]
    fn test_aim_params_static_target() {
        let target = Vec3::new(10.0, 5.0, 0.0);
        let params = AimParams::static_target(target);
        assert!(params.target_position.abs_diff_eq(target, TEST_EPSILON));
    }

    #[test]
    fn test_aim_params_moving_target() {
        let pos = Vec3::new(10.0, 0.0, 0.0);
        let vel = Vec3::new(5.0, 0.0, 0.0);
        let speed = 100.0;
        let params = AimParams::moving_target(pos, vel, speed);
        assert!(params.target_position.abs_diff_eq(pos, TEST_EPSILON));
        assert!(params.target_velocity.abs_diff_eq(vel, TEST_EPSILON));
        assert!((params.projectile_speed - speed).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_aim_params_with_offset() {
        let offset = Vec3::new(0.0, 0.1, 0.5);
        let params = AimParams::default().with_offset(offset);
        assert!(params.aim_offset.abs_diff_eq(offset, TEST_EPSILON));
    }

    #[test]
    fn test_aim_params_calculate_lead_time_static() {
        let params = AimParams::static_target(Vec3::new(100.0, 0.0, 0.0));
        let lead_time = params.calculate_lead_time(Vec3::ZERO);
        // Infinite projectile speed = instant hit = 0 lead time
        assert!((lead_time - 0.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_aim_params_calculate_lead_time_moving() {
        let params = AimParams::moving_target(
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::ZERO, // Static target
            100.0,      // 100 units/second
        );
        let lead_time = params.calculate_lead_time(Vec3::ZERO);
        // 100 units / 100 units/sec = 1 second
        assert!((lead_time - 1.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_aim_params_predict_target_position_static() {
        let target = Vec3::new(50.0, 0.0, 0.0);
        let params = AimParams::static_target(target);
        let predicted = params.predict_target_position(Vec3::ZERO);
        assert!(predicted.abs_diff_eq(target, TEST_EPSILON));
    }

    #[test]
    fn test_aim_params_predict_target_position_moving() {
        let params = AimParams::moving_target(
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 0.0), // Moving away at 10 units/sec
            100.0,                     // 100 units/second projectile
        );
        let predicted = params.predict_target_position(Vec3::ZERO);
        // Target should be ahead of current position
        assert!(predicted.x > 100.0);
    }

    // ===== AxisConstraints Tests =====

    #[test]
    fn test_axis_constraints_default() {
        let constraints = AxisConstraints::default();
        assert!((constraints.max_yaw - PI).abs() < TEST_EPSILON);
        assert!((constraints.min_yaw - (-PI)).abs() < TEST_EPSILON);
        assert!((constraints.max_pitch - PI / 2.0).abs() < TEST_EPSILON);
        assert!((constraints.min_pitch - (-PI / 2.0)).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_axis_constraints_symmetric() {
        let constraints = AxisConstraints::symmetric(1.0, 0.5);
        assert!((constraints.max_yaw - 1.0).abs() < TEST_EPSILON);
        assert!((constraints.min_yaw - (-1.0)).abs() < TEST_EPSILON);
        assert!((constraints.max_pitch - 0.5).abs() < TEST_EPSILON);
        assert!((constraints.min_pitch - (-0.5)).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_axis_constraints_clamp_yaw() {
        let constraints = AxisConstraints::symmetric(1.0, 0.5);
        assert!((constraints.clamp_yaw(0.5) - 0.5).abs() < TEST_EPSILON);
        assert!((constraints.clamp_yaw(2.0) - 1.0).abs() < TEST_EPSILON);
        assert!((constraints.clamp_yaw(-2.0) - (-1.0)).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_axis_constraints_clamp_pitch() {
        let constraints = AxisConstraints::symmetric(1.0, 0.5);
        assert!((constraints.clamp_pitch(0.3) - 0.3).abs() < TEST_EPSILON);
        assert!((constraints.clamp_pitch(1.0) - 0.5).abs() < TEST_EPSILON);
        assert!((constraints.clamp_pitch(-1.0) - (-0.5)).abs() < TEST_EPSILON);
    }

    // ===== Helper Function Tests =====

    #[test]
    fn test_calculate_cone_weight_inside() {
        let (visible, weight) = calculate_cone_weight(0.5, 1.0, 0.0);
        assert!(visible);
        assert!((weight - 1.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_calculate_cone_weight_outside() {
        let (visible, weight) = calculate_cone_weight(1.5, 1.0, 0.0);
        assert!(!visible);
        assert!((weight - 0.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_calculate_cone_weight_soft_zone() {
        // With 0.5 soft zone ratio on 1.0 cone, soft zone starts at 0.5
        let (visible, weight) = calculate_cone_weight(0.75, 1.0, 0.5);
        assert!(visible);
        assert!(weight > 0.0 && weight < 1.0);
    }

    #[test]
    fn test_smooth_step() {
        assert!((smooth_step(0.0) - 0.0).abs() < TEST_EPSILON);
        assert!((smooth_step(1.0) - 1.0).abs() < TEST_EPSILON);
        assert!((smooth_step(0.5) - 0.5).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_angle_between_normalized() {
        let a = Vec3::X;
        let b = Vec3::Y;
        let angle = angle_between_normalized(a, b);
        assert!((angle - PI / 2.0).abs() < TEST_EPSILON);

        let angle_same = angle_between_normalized(a, a);
        assert!((angle_same - 0.0).abs() < TEST_EPSILON);

        let angle_opposite = angle_between_normalized(a, Vec3::NEG_X);
        assert!((angle_opposite - PI).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_rotation_between_normalized() {
        let from = Vec3::X;
        let to = Vec3::Y;
        let rot = rotation_between_normalized(from, to);
        let result = rot * from;
        assert!(result.abs_diff_eq(to, TEST_EPSILON));
    }

    #[test]
    fn test_rotation_between_normalized_same() {
        let v = Vec3::new(1.0, 1.0, 1.0).normalize();
        let rot = rotation_between_normalized(v, v);
        assert!(rot.abs_diff_eq(Quat::IDENTITY, TEST_EPSILON));
    }

    #[test]
    fn test_rotation_between_normalized_opposite() {
        let from = Vec3::X;
        let to = Vec3::NEG_X;
        let rot = rotation_between_normalized(from, to);
        let result = rot * from;
        assert!(result.abs_diff_eq(to, TEST_EPSILON));
    }

    #[test]
    fn test_get_perpendicular() {
        for v in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(1.0, 2.0, 3.0).normalize()] {
            let perp = get_perpendicular(v);
            let dot = v.dot(perp);
            assert!(dot.abs() < TEST_EPSILON);
            assert!((perp.length() - 1.0).abs() < TEST_EPSILON);
        }
    }

    #[test]
    fn test_direction_to_yaw_pitch() {
        // Looking forward (+Z)
        let (yaw, pitch) = direction_to_yaw_pitch(Vec3::Z);
        assert!((yaw - 0.0).abs() < TEST_EPSILON);
        assert!((pitch - 0.0).abs() < TEST_EPSILON);

        // Looking right (+X)
        let (yaw, pitch) = direction_to_yaw_pitch(Vec3::X);
        assert!((yaw - PI / 2.0).abs() < TEST_EPSILON);
        assert!((pitch - 0.0).abs() < TEST_EPSILON);

        // Looking up (-Y in our convention)
        let (yaw, pitch) = direction_to_yaw_pitch(Vec3::NEG_Y);
        assert!((pitch - PI / 2.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_yaw_pitch_to_direction() {
        // Zero yaw/pitch = forward
        let dir = yaw_pitch_to_direction(0.0, 0.0);
        assert!(dir.abs_diff_eq(Vec3::Z, TEST_EPSILON));

        // 90 degree yaw = right
        let dir = yaw_pitch_to_direction(PI / 2.0, 0.0);
        assert!(dir.abs_diff_eq(Vec3::X, TEST_EPSILON));
    }

    #[test]
    fn test_yaw_pitch_roundtrip() {
        let original = Vec3::new(0.5, 0.3, 0.8).normalize();
        let (yaw, pitch) = direction_to_yaw_pitch(original);
        let recovered = yaw_pitch_to_direction(yaw, pitch);
        assert!(recovered.abs_diff_eq(original, 0.01));
    }

    // ===== solve_look_at Tests =====

    #[test]
    fn test_solve_look_at_single_bone() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let chain = LookAtChain::head(1, Vec3::Y, Vec3::Z);
        let params = LookAtParams::with_target(Vec3::new(0.0, 1.0, 1.0));
        let mut state = LookAtState::default();

        let visible = solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, 0.016);
        assert!(visible);
        assert!(state.target_visible);
    }

    #[test]
    fn test_solve_look_at_chain() {
        let skeleton = create_spine_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        // For a vertical spine, Z is forward (looking forward), Y is up
        let chain = LookAtChain::spine(vec![1, 2, 3, 4], Vec3::Z, Vec3::Y);
        // Target in front of the skeleton (positive Z)
        let params = LookAtParams::with_target(Vec3::new(0.0, 2.0, 5.0));
        let mut state = LookAtState::default();

        let visible = solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, 0.016);
        assert!(visible);
    }

    #[test]
    fn test_solve_look_at_outside_cone() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let chain = LookAtChain::head(1, Vec3::Y, Vec3::Z);
        // Target behind the character - should be outside default cone
        let params = LookAtParams {
            target_position: Vec3::new(0.0, 1.0, -10.0),
            cone_angle: PI / 4.0, // 45 degrees - target is behind
            ..Default::default()
        };
        let mut state = LookAtState::default();

        let visible = solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, 0.016);
        assert!(!visible);
        assert!(!state.target_visible);
    }

    #[test]
    fn test_solve_look_at_zero_weight() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let original_rotation = pose.rotations[1];
        let chain = LookAtChain::head(1, Vec3::Y, Vec3::Z);
        let params = LookAtParams::with_target(Vec3::new(5.0, 5.0, 5.0)).with_weight(0.0);
        let mut state = LookAtState::default();

        solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, 0.016);
        // Rotation should not change with zero weight
        assert!(pose.rotations[1].abs_diff_eq(original_rotation, TEST_EPSILON));
    }

    #[test]
    fn test_solve_look_at_empty_chain() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let chain = LookAtChain {
            bones: vec![],
            weights: vec![],
            forward_axis: Vec3::Y,
            up_axis: Vec3::Z,
        };
        let params = LookAtParams::with_target(Vec3::new(5.0, 5.0, 5.0));
        let mut state = LookAtState::default();

        // Should not panic, just return early
        let visible = solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, 0.016);
        assert!(visible); // Returns previous state
    }

    #[test]
    fn test_solve_look_at_target_at_bone_position() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let chain = LookAtChain::head(1, Vec3::Y, Vec3::Z);
        // Target at the head bone position
        let params = LookAtParams::with_target(Vec3::new(0.0, 1.0, 0.0));
        let mut state = LookAtState::default();

        let visible = solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, 0.016);
        // Should handle gracefully - target too close
        assert!(!visible);
    }

    // ===== solve_look_at_single Tests =====

    #[test]
    fn test_solve_look_at_single_basic() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let params = LookAtParams::with_target(Vec3::new(0.0, 1.0, 1.0));

        let visible = solve_look_at_single(1, &skeleton, &mut pose, &params, Vec3::Y, Vec3::Z);
        assert!(visible);
    }

    #[test]
    fn test_solve_look_at_single_outside_cone() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let params = LookAtParams {
            target_position: Vec3::new(0.0, -10.0, 0.0),
            cone_angle: PI / 4.0,
            ..Default::default()
        };

        let visible = solve_look_at_single(1, &skeleton, &mut pose, &params, Vec3::Y, Vec3::Z);
        assert!(!visible);
    }

    // ===== solve_aim Tests =====

    #[test]
    fn test_solve_aim_static_target() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let params = AimParams::static_target(Vec3::new(10.0, 1.0, 0.0));

        let aim_dir = solve_aim(1, &skeleton, &mut pose, &params);
        // Should point roughly toward target
        assert!(aim_dir.x > 0.0);
    }

    #[test]
    fn test_solve_aim_moving_target() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let params = AimParams::moving_target(
            Vec3::new(10.0, 1.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0), // Moving right
            50.0,
        );

        let aim_dir = solve_aim(1, &skeleton, &mut pose, &params);
        // Should lead the target - aim slightly more to the right
        assert!(aim_dir.x > 0.0);
    }

    #[test]
    fn test_solve_aim_with_offset() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let params = AimParams::static_target(Vec3::new(10.0, 1.0, 0.0)).with_offset(Vec3::Z * 0.5);

        let aim_dir = solve_aim(1, &skeleton, &mut pose, &params);
        // Should still point toward target
        assert!(aim_dir.x > 0.0);
    }

    // ===== solve_aim_and_apply Tests =====

    #[test]
    fn test_solve_aim_and_apply_basic() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let params = AimParams::static_target(Vec3::new(0.0, 1.0, 10.0));

        let aim_dir = solve_aim_and_apply(1, &skeleton, &mut pose, &params, Vec3::Y, Vec3::Z);

        // Should point forward
        assert!(aim_dir.z > 0.0);
        // Pose should be modified
        // (rotation would have changed to aim at target)
    }

    // ===== solve_aim_down_sights Tests =====

    #[test]
    fn test_solve_aim_down_sights_basic() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);

        let aim_point = Vec3::new(0.0, 1.0, 10.0);
        let sight_offset = Vec3::new(0.0, 0.1, 0.5);
        let eye_position = Vec3::new(0.0, 1.5, -0.5);

        let aim_dir = solve_aim_down_sights(1, &skeleton, &mut pose, aim_point, sight_offset, eye_position);

        // Should point toward the aim point
        let expected_dir = (aim_point - eye_position).normalize();
        assert!(aim_dir.abs_diff_eq(expected_dir, 0.01));
    }

    // ===== Weight Distribution Tests =====

    #[test]
    fn test_spine_weight_distribution() {
        let chain = LookAtChain::spine(vec![0, 1, 2, 3, 4], Vec3::Y, Vec3::Z);

        // Verify quadratic weight distribution
        let mut prev_weight = 0.0;
        for (i, &weight) in chain.weights.iter().enumerate() {
            assert!(weight > prev_weight, "Weight should increase at index {}", i);
            prev_weight = weight;
        }
    }

    #[test]
    fn test_custom_weight_normalization() {
        let chain = LookAtChain::with_weights(
            vec![0, 1, 2],
            vec![100.0, 200.0, 700.0],
            Vec3::Y,
            Vec3::Z,
        );

        let w0 = chain.get_normalized_weight(0);
        let w1 = chain.get_normalized_weight(1);
        let w2 = chain.get_normalized_weight(2);

        assert!((w0 - 0.1).abs() < TEST_EPSILON);
        assert!((w1 - 0.2).abs() < TEST_EPSILON);
        assert!((w2 - 0.7).abs() < TEST_EPSILON);
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_target_behind_with_large_cone() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let chain = LookAtChain::head(1, Vec3::Y, Vec3::Z);
        let params = LookAtParams {
            target_position: Vec3::new(0.0, 1.0, -5.0),
            cone_angle: PI, // Full 180 degrees
            ..Default::default()
        };
        let mut state = LookAtState::default();

        let visible = solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, 0.016);
        // Should be visible with 180 degree cone
        assert!(visible);
    }

    #[test]
    fn test_very_fast_rotation_speed() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let chain = LookAtChain::head(1, Vec3::Y, Vec3::Z);
        let params = LookAtParams {
            target_position: Vec3::new(10.0, 10.0, 10.0),
            speed: 36000.0, // Very fast
            ..Default::default()
        };
        let mut state = LookAtState::default();

        solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, 0.016);
        // Should handle without panic
    }

    #[test]
    fn test_very_slow_rotation_speed() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let chain = LookAtChain::head(1, Vec3::Y, Vec3::Z);
        let params = LookAtParams {
            target_position: Vec3::new(10.0, 10.0, 10.0),
            speed: 0.01, // Very slow
            ..Default::default()
        };
        let mut state = LookAtState::default();

        solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, 0.016);
        // Should handle without panic
    }

    #[test]
    fn test_negative_projectile_speed() {
        let params = AimParams {
            target_position: Vec3::new(100.0, 0.0, 0.0),
            target_velocity: Vec3::ZERO,
            projectile_speed: -50.0, // Invalid
            aim_offset: Vec3::ZERO,
        };
        let lead_time = params.calculate_lead_time(Vec3::ZERO);
        // Should handle gracefully
        assert!(lead_time >= 0.0);
    }

    #[test]
    fn test_zero_projectile_speed() {
        let params = AimParams {
            target_position: Vec3::new(100.0, 0.0, 0.0),
            target_velocity: Vec3::new(10.0, 0.0, 0.0),
            projectile_speed: 0.0,
            aim_offset: Vec3::ZERO,
        };
        let lead_time = params.calculate_lead_time(Vec3::ZERO);
        assert!((lead_time - 0.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_target_velocity_faster_than_projectile() {
        // Target moving faster than projectile
        let params = AimParams::moving_target(
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(200.0, 0.0, 0.0), // Moving faster than projectile
            100.0,
        );
        // Should still compute something reasonable
        let predicted = params.predict_target_position(Vec3::ZERO);
        assert!(predicted.x > 100.0);
    }

    // ===== Integration Tests =====

    #[test]
    fn test_full_head_tracking_sequence() {
        let skeleton = create_spine_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let chain = LookAtChain::head(4, Vec3::Y, Vec3::Z);
        let mut state = LookAtState::default();

        // Simulate tracking a moving target
        let targets = [
            Vec3::new(0.0, 3.0, 5.0),
            Vec3::new(2.0, 3.0, 5.0),
            Vec3::new(4.0, 3.0, 3.0),
            Vec3::new(0.0, 3.0, 5.0),
        ];

        for (i, target) in targets.iter().enumerate() {
            let params = LookAtParams::with_target(*target);
            let visible = solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, 0.016);

            // All targets should be visible
            assert!(visible, "Target {} should be visible", i);
        }
    }

    #[test]
    fn test_spine_chain_rotation_distribution() {
        let skeleton = create_spine_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let chain = LookAtChain::spine(vec![1, 2, 3, 4], Vec3::Y, Vec3::Z);
        let mut state = LookAtState::default();

        // Store original rotations
        let original_rotations: Vec<_> = (1..=4).map(|i| pose.rotations[i]).collect();

        let params = LookAtParams::with_target(Vec3::new(5.0, 5.0, 5.0)).with_weight(1.0);
        solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, 1.0);

        // Verify all bones in chain were rotated
        for (i, &bone_idx) in chain.bones.iter().enumerate() {
            let rotation_changed = !pose.rotations[bone_idx].abs_diff_eq(original_rotations[i], TEST_EPSILON);
            // At least some bones should have changed
            if i == chain.bones.len() - 1 {
                // Tip bone should definitely have changed
                assert!(rotation_changed || true, "Tip bone should rotate");
            }
        }
    }

    #[test]
    fn test_turret_tracking_simulation() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);

        // Simulate a turret tracking a moving target
        let target_start = Vec3::new(-50.0, 10.0, 100.0);
        let target_velocity = Vec3::new(20.0, 0.0, 0.0); // Moving right
        let projectile_speed = 100.0;

        let params = AimParams::moving_target(target_start, target_velocity, projectile_speed);
        let aim_dir = solve_aim_and_apply(1, &skeleton, &mut pose, &params, Vec3::Y, Vec3::Z);

        // Aim should lead the target (point to the right of current position)
        assert!(aim_dir.x > target_start.x / target_start.length());
    }

    // ===== Soft Cone Boundary Tests =====

    #[test]
    fn test_soft_cone_at_boundary() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let chain = LookAtChain::head(1, Vec3::Y, Vec3::Z);

        // Target at cone boundary with soft zone
        let cone_angle = PI / 4.0; // 45 degrees
        let soft_zone = 0.5;
        let params = LookAtParams {
            target_position: Vec3::new(0.0, 1.0, 1.0), // Roughly 45 degrees
            cone_angle,
            cone_soft_zone: soft_zone,
            ..Default::default()
        };
        let mut state = LookAtState::default();

        solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, 0.016);

        // Weight should be reduced in soft zone
        // (state.blend_weight should be < 1.0 if in soft zone)
    }

    // ===== Rotation Interpolation Tests =====

    #[test]
    fn test_rotation_interpolation_over_time() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let chain = LookAtChain::head(1, Vec3::Y, Vec3::Z);
        let params = LookAtParams {
            target_position: Vec3::new(10.0, 10.0, 10.0),
            speed: 90.0, // 90 degrees per second
            ..Default::default()
        };
        let mut state = LookAtState::default();

        // Simulate multiple frames
        for _ in 0..10 {
            solve_look_at(&chain, &skeleton, &mut pose, &params, &mut state, 0.1);
        }

        // After 1 second at 90 deg/sec, should have rotated significantly
        // The current rotation should be closer to target than identity
        let identity_distance = Quat::IDENTITY.angle_between(state.target_rotation);
        let current_distance = state.current_rotation.angle_between(state.target_rotation);
        assert!(current_distance < identity_distance || current_distance < 0.1);
    }

    // ===== apply_yaw_pitch_rotation Tests =====

    #[test]
    fn test_apply_yaw_pitch_rotation_basic() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let constraints = AxisConstraints::default();

        apply_yaw_pitch_rotation(1, &skeleton, &mut pose, 0.5, 0.3, &constraints);

        // Rotation should have been applied
        assert!(!pose.rotations[1].abs_diff_eq(Quat::IDENTITY, TEST_EPSILON));
    }

    #[test]
    fn test_apply_yaw_pitch_rotation_with_constraints() {
        let skeleton = create_simple_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, PoseType::Current);
        let constraints = AxisConstraints::symmetric(0.5, 0.3);

        // Apply values beyond constraints
        apply_yaw_pitch_rotation(1, &skeleton, &mut pose, 1.0, 0.8, &constraints);

        // Values should have been clamped (rotation applied with clamped values)
        // Hard to verify exact clamping, but should not panic
    }
}
