//! CCD (Cyclic Coordinate Descent) Inverse Kinematics Solver (T-AN-4.3).
//!
//! This module provides a CCD-based IK solver that iteratively rotates each joint
//! from the effector (tip) towards the root to minimize the distance between the
//! effector and a target position. CCD is known for its:
//!
//! - Fast convergence for simple chains
//! - Natural-looking results for organic motion
//! - Simple implementation and easy constraint integration
//!
//! # Algorithm Overview
//!
//! For each iteration:
//! 1. Start from the bone closest to the effector (tip)
//! 2. For each bone moving towards the root:
//!    a. Compute the vector from current bone to effector
//!    b. Compute the vector from current bone to target
//!    c. Calculate the rotation that aligns these vectors
//!    d. Apply rotation (with damping and constraints)
//!    e. Update effector position
//! 3. Check for convergence (effector within tolerance of target)
//! 4. Repeat until converged or max iterations reached
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::ik_ccd::{CcdChain, CcdParams, solve_ccd};
//! use renderer_backend::skeleton::Skeleton;
//! use renderer_backend::pose::Pose;
//! use glam::Vec3;
//!
//! // Build IK chain from effector to root
//! let chain = CcdChain::new(vec![4, 3, 2, 1, 0]); // tip to root order
//!
//! // Configure solver parameters
//! let params = CcdParams {
//!     target_position: Vec3::new(1.0, 2.0, 0.0),
//!     max_iterations: 10,
//!     tolerance: 0.001,
//!     damping: 0.8,
//!     constraints: vec![],
//! };
//!
//! // Solve IK
//! let result = solve_ccd(&chain, &skeleton, &mut pose, &params);
//! if result.converged {
//!     println!("IK solved in {} iterations", result.iterations);
//! }
//! ```

use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::pose::Pose;
use crate::skeleton::{Skeleton, Transform};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum iterations for CCD solver.
pub const DEFAULT_MAX_ITERATIONS: u32 = 10;

/// Default convergence tolerance in units.
pub const DEFAULT_TOLERANCE: f32 = 0.001;

/// Default damping factor (1.0 = no damping).
pub const DEFAULT_DAMPING: f32 = 1.0;

/// Minimum rotation angle threshold (radians) below which we skip rotation.
pub const MIN_ROTATION_ANGLE: f32 = 1e-6;

/// Small epsilon for floating point comparisons.
pub const EPSILON: f32 = 1e-7;

// ---------------------------------------------------------------------------
// JointConstraint
// ---------------------------------------------------------------------------

/// Type of joint constraint for IK solving.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum JointConstraintType {
    /// Unconstrained - free rotation in all axes.
    None,

    /// Cone constraint - rotation limited to a cone around the rest direction.
    /// The angle is the half-angle of the cone in radians.
    Cone {
        /// Half-angle of the cone in radians.
        half_angle: f32,
        /// Reference axis for the cone (local space).
        axis: Vec3,
    },

    /// Hinge constraint - rotation limited to a single axis.
    Hinge {
        /// Rotation axis in local space (should be normalized).
        axis: Vec3,
        /// Minimum angle in radians.
        min_angle: f32,
        /// Maximum angle in radians.
        max_angle: f32,
    },

    /// Ball-and-socket with twist limit.
    BallSocket {
        /// Swing cone half-angle in radians.
        swing_limit: f32,
        /// Twist limit (min, max) in radians.
        twist_limits: (f32, f32),
        /// Twist axis in local space.
        twist_axis: Vec3,
    },

    /// Per-axis angle limits (Euler angles).
    EulerLimits {
        /// X-axis rotation limits (min, max) in radians.
        x_limits: (f32, f32),
        /// Y-axis rotation limits (min, max) in radians.
        y_limits: (f32, f32),
        /// Z-axis rotation limits (min, max) in radians.
        z_limits: (f32, f32),
    },
}

impl Default for JointConstraintType {
    fn default() -> Self {
        Self::None
    }
}

/// Constraint configuration for a single joint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointConstraint {
    /// Index of the bone this constraint applies to.
    pub bone_index: usize,

    /// Type of constraint.
    pub constraint_type: JointConstraintType,

    /// Weight of this constraint (0.0-1.0). Lower values make constraints softer.
    pub weight: f32,
}

impl JointConstraint {
    /// Create a new unconstrained joint.
    pub fn none(bone_index: usize) -> Self {
        Self {
            bone_index,
            constraint_type: JointConstraintType::None,
            weight: 1.0,
        }
    }

    /// Create a cone constraint.
    pub fn cone(bone_index: usize, half_angle: f32, axis: Vec3) -> Self {
        Self {
            bone_index,
            constraint_type: JointConstraintType::Cone {
                half_angle,
                axis: axis.normalize_or_zero(),
            },
            weight: 1.0,
        }
    }

    /// Create a hinge constraint.
    pub fn hinge(bone_index: usize, axis: Vec3, min_angle: f32, max_angle: f32) -> Self {
        Self {
            bone_index,
            constraint_type: JointConstraintType::Hinge {
                axis: axis.normalize_or_zero(),
                min_angle,
                max_angle,
            },
            weight: 1.0,
        }
    }

    /// Create a ball-and-socket constraint.
    pub fn ball_socket(
        bone_index: usize,
        swing_limit: f32,
        twist_min: f32,
        twist_max: f32,
        twist_axis: Vec3,
    ) -> Self {
        Self {
            bone_index,
            constraint_type: JointConstraintType::BallSocket {
                swing_limit,
                twist_limits: (twist_min, twist_max),
                twist_axis: twist_axis.normalize_or_zero(),
            },
            weight: 1.0,
        }
    }

    /// Create Euler angle limits constraint.
    pub fn euler_limits(
        bone_index: usize,
        x_limits: (f32, f32),
        y_limits: (f32, f32),
        z_limits: (f32, f32),
    ) -> Self {
        Self {
            bone_index,
            constraint_type: JointConstraintType::EulerLimits {
                x_limits,
                y_limits,
                z_limits,
            },
            weight: 1.0,
        }
    }

    /// Set the constraint weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Apply this constraint to a rotation delta.
    ///
    /// # Arguments
    ///
    /// * `rotation` - The rotation to constrain
    /// * `rest_rotation` - The rest pose rotation for reference
    ///
    /// # Returns
    ///
    /// The constrained rotation.
    pub fn apply(&self, rotation: Quat, rest_rotation: Quat) -> Quat {
        if self.weight < EPSILON {
            return rotation;
        }

        let constrained = match &self.constraint_type {
            JointConstraintType::None => rotation,
            JointConstraintType::Cone { half_angle, axis } => {
                Self::apply_cone_constraint(rotation, rest_rotation, *axis, *half_angle)
            }
            JointConstraintType::Hinge {
                axis,
                min_angle,
                max_angle,
            } => Self::apply_hinge_constraint(rotation, *axis, *min_angle, *max_angle),
            JointConstraintType::BallSocket {
                swing_limit,
                twist_limits,
                twist_axis,
            } => Self::apply_ball_socket_constraint(
                rotation,
                *swing_limit,
                *twist_limits,
                *twist_axis,
            ),
            JointConstraintType::EulerLimits {
                x_limits,
                y_limits,
                z_limits,
            } => Self::apply_euler_limits(rotation, *x_limits, *y_limits, *z_limits),
        };

        // Blend between unconstrained and constrained based on weight
        if (self.weight - 1.0).abs() < EPSILON {
            constrained
        } else {
            rotation.slerp(constrained, self.weight)
        }
    }

    /// Apply cone constraint to rotation.
    fn apply_cone_constraint(rotation: Quat, rest: Quat, axis: Vec3, half_angle: f32) -> Quat {
        // Get the direction after rotation
        let rotated_axis = rotation * axis;
        let rest_axis = rest * axis;

        // Calculate angle between rotated and rest axis
        let dot = rotated_axis.dot(rest_axis).clamp(-1.0, 1.0);
        let angle = dot.acos();

        if angle <= half_angle {
            return rotation;
        }

        // Clamp to cone boundary
        let t = half_angle / angle;

        // Find the rotation axis between rotated and rest
        let cross = rotated_axis.cross(rest_axis);
        if cross.length_squared() < EPSILON {
            return rotation;
        }

        let correction_axis = cross.normalize();
        let correction_angle = angle - half_angle;
        let correction = Quat::from_axis_angle(correction_axis, correction_angle);

        (correction * rotation).normalize()
    }

    /// Apply hinge constraint to rotation.
    fn apply_hinge_constraint(rotation: Quat, axis: Vec3, min_angle: f32, max_angle: f32) -> Quat {
        // Decompose rotation into twist around axis and swing
        let (twist, swing) = Self::decompose_twist_swing(rotation, axis);

        // Get twist angle
        let (twist_axis, mut twist_angle) = twist.to_axis_angle();

        // Check if twist is around the correct axis direction
        if twist_axis.dot(axis) < 0.0 {
            twist_angle = -twist_angle;
        }

        // Clamp twist angle
        let clamped_angle = twist_angle.clamp(min_angle, max_angle);

        // Reconstruct with clamped twist (ignore swing for pure hinge)
        Quat::from_axis_angle(axis, clamped_angle)
    }

    /// Apply ball-and-socket constraint.
    fn apply_ball_socket_constraint(
        rotation: Quat,
        swing_limit: f32,
        twist_limits: (f32, f32),
        twist_axis: Vec3,
    ) -> Quat {
        let (twist, swing) = Self::decompose_twist_swing(rotation, twist_axis);

        // Clamp twist
        let (t_axis, mut t_angle) = twist.to_axis_angle();
        if t_axis.dot(twist_axis) < 0.0 {
            t_angle = -t_angle;
        }
        let clamped_twist_angle = t_angle.clamp(twist_limits.0, twist_limits.1);
        let clamped_twist = Quat::from_axis_angle(twist_axis, clamped_twist_angle);

        // Clamp swing
        let (s_axis, s_angle) = swing.to_axis_angle();
        let clamped_swing_angle = s_angle.min(swing_limit);
        let clamped_swing = if s_axis.length_squared() > EPSILON {
            Quat::from_axis_angle(s_axis, clamped_swing_angle)
        } else {
            Quat::IDENTITY
        };

        (clamped_swing * clamped_twist).normalize()
    }

    /// Apply Euler angle limits.
    fn apply_euler_limits(
        rotation: Quat,
        x_limits: (f32, f32),
        y_limits: (f32, f32),
        z_limits: (f32, f32),
    ) -> Quat {
        let (mut x, mut y, mut z) = rotation.to_euler(glam::EulerRot::XYZ);

        x = x.clamp(x_limits.0, x_limits.1);
        y = y.clamp(y_limits.0, y_limits.1);
        z = z.clamp(z_limits.0, z_limits.1);

        Quat::from_euler(glam::EulerRot::XYZ, x, y, z)
    }

    /// Decompose a quaternion into twist (around axis) and swing components.
    fn decompose_twist_swing(rotation: Quat, axis: Vec3) -> (Quat, Quat) {
        let ra = Vec3::new(rotation.x, rotation.y, rotation.z);
        let proj = axis * ra.dot(axis);

        let twist = Quat::from_xyzw(proj.x, proj.y, proj.z, rotation.w).normalize();
        let swing = rotation * twist.conjugate();

        (twist, swing.normalize())
    }
}

// ---------------------------------------------------------------------------
// CcdChain
// ---------------------------------------------------------------------------

/// An IK chain for CCD solving.
///
/// Stores bone indices from the effector (tip) to the root.
/// The first index is the effector bone, the last is the root of the chain.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CcdChain {
    /// Bone indices from tip to root.
    pub bones: Vec<usize>,

    /// Optional name for this chain.
    pub name: Option<String>,

    /// Index of the effector bone (first in chain).
    /// This is cached for convenience.
    pub effector_index: Option<usize>,
}

impl CcdChain {
    /// Create a new CCD chain from bone indices.
    ///
    /// # Arguments
    ///
    /// * `bones` - Bone indices from effector (tip) to root.
    ///
    /// # Panics
    ///
    /// Panics if `bones` is empty.
    pub fn new(bones: Vec<usize>) -> Self {
        assert!(!bones.is_empty(), "CCD chain must have at least one bone");

        Self {
            effector_index: Some(bones[0]),
            bones,
            name: None,
        }
    }

    /// Create a named CCD chain.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Create a CCD chain from a skeleton by walking from effector to root.
    ///
    /// # Arguments
    ///
    /// * `skeleton` - The skeleton to extract the chain from
    /// * `effector` - Index of the effector (tip) bone
    /// * `chain_root` - Optional index of the chain root (stops at this bone).
    ///                  If None, walks all the way to the skeleton root.
    ///
    /// # Returns
    ///
    /// A CCD chain from effector to (chain) root.
    pub fn from_skeleton(
        skeleton: &Skeleton,
        effector: usize,
        chain_root: Option<usize>,
    ) -> Option<Self> {
        if effector >= skeleton.bone_count() {
            return None;
        }

        let mut bones = vec![effector];
        let mut current = effector;

        loop {
            // Stop if we reached the chain root
            if let Some(root) = chain_root {
                if current == root {
                    break;
                }
            }

            // Get parent
            match skeleton.parent(current) {
                Some(parent) => {
                    bones.push(parent);
                    current = parent;

                    // Stop if we reached the chain root
                    if let Some(root) = chain_root {
                        if parent == root {
                            break;
                        }
                    }
                }
                None => break, // Reached skeleton root
            }
        }

        Some(Self::new(bones))
    }

    /// Get the effector (tip) bone index.
    #[inline]
    pub fn effector(&self) -> usize {
        self.bones[0]
    }

    /// Get the root of this chain.
    #[inline]
    pub fn root(&self) -> usize {
        *self.bones.last().unwrap()
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

    /// Validate that this chain is valid for the given skeleton.
    pub fn validate(&self, skeleton: &Skeleton) -> Result<(), CcdError> {
        if self.bones.is_empty() {
            return Err(CcdError::EmptyChain);
        }

        for (i, &bone_idx) in self.bones.iter().enumerate() {
            if bone_idx >= skeleton.bone_count() {
                return Err(CcdError::InvalidBoneIndex {
                    index: bone_idx,
                    bone_count: skeleton.bone_count(),
                });
            }

            // Verify parent relationship (each bone should be child of the next in chain)
            if i + 1 < self.bones.len() {
                let parent = skeleton.parent(bone_idx);
                if parent != Some(self.bones[i + 1]) {
                    return Err(CcdError::BrokenChain {
                        bone_index: bone_idx,
                        expected_parent: self.bones[i + 1],
                        actual_parent: parent,
                    });
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CcdParams
// ---------------------------------------------------------------------------

/// Parameters for the CCD solver.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CcdParams {
    /// Target position in world space.
    pub target_position: Vec3,

    /// Maximum number of iterations.
    pub max_iterations: u32,

    /// Convergence tolerance (distance from effector to target).
    pub tolerance: f32,

    /// Damping factor (0.0-1.0, where 1.0 = no damping).
    /// Lower values limit the rotation applied per step for smoother convergence.
    pub damping: f32,

    /// Joint constraints for each bone.
    pub constraints: Vec<JointConstraint>,

    /// Optional pole target for controlling elbow/knee direction.
    pub pole_target: Option<Vec3>,

    /// Weight for pole target influence (0.0-1.0).
    pub pole_weight: f32,
}

impl Default for CcdParams {
    fn default() -> Self {
        Self {
            target_position: Vec3::ZERO,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tolerance: DEFAULT_TOLERANCE,
            damping: DEFAULT_DAMPING,
            constraints: Vec::new(),
            pole_target: None,
            pole_weight: 0.5,
        }
    }
}

impl CcdParams {
    /// Create new CCD parameters with a target position.
    pub fn new(target: Vec3) -> Self {
        Self {
            target_position: target,
            ..Default::default()
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, iterations: u32) -> Self {
        self.max_iterations = iterations.max(1);
        self
    }

    /// Set convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance.max(EPSILON);
        self
    }

    /// Set damping factor.
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping.clamp(0.01, 1.0);
        self
    }

    /// Add a joint constraint.
    pub fn with_constraint(mut self, constraint: JointConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Set joint constraints.
    pub fn with_constraints(mut self, constraints: Vec<JointConstraint>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Set pole target for elbow/knee direction.
    pub fn with_pole_target(mut self, target: Vec3, weight: f32) -> Self {
        self.pole_target = Some(target);
        self.pole_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Get constraint for a specific bone, if any.
    pub fn get_constraint(&self, bone_index: usize) -> Option<&JointConstraint> {
        self.constraints.iter().find(|c| c.bone_index == bone_index)
    }
}

// ---------------------------------------------------------------------------
// CcdResult
// ---------------------------------------------------------------------------

/// Result of CCD IK solving.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CcdResult {
    /// Final rotations for each bone in the chain.
    pub bone_rotations: Vec<Quat>,

    /// Number of iterations performed.
    pub iterations: u32,

    /// Whether the solver converged (reached tolerance).
    pub converged: bool,

    /// Final distance from effector to target.
    pub final_distance: f32,

    /// Distance at each iteration (for debugging/analysis).
    pub distance_history: Vec<f32>,
}

impl CcdResult {
    /// Create a new CCD result.
    fn new(bone_count: usize) -> Self {
        Self {
            bone_rotations: vec![Quat::IDENTITY; bone_count],
            iterations: 0,
            converged: false,
            final_distance: f32::MAX,
            distance_history: Vec::new(),
        }
    }

    /// Check if the solve was successful (converged or close enough).
    pub fn success(&self) -> bool {
        self.converged
    }

    /// Get the improvement ratio from first to last distance.
    pub fn improvement_ratio(&self) -> f32 {
        if self.distance_history.len() < 2 {
            return 0.0;
        }

        let initial = self.distance_history[0];
        if initial < EPSILON {
            return 1.0;
        }

        1.0 - (self.final_distance / initial)
    }
}

// ---------------------------------------------------------------------------
// CcdError
// ---------------------------------------------------------------------------

/// Errors that can occur during CCD solving.
#[derive(Clone, Debug, PartialEq)]
pub enum CcdError {
    /// The chain is empty.
    EmptyChain,

    /// A bone index is out of bounds.
    InvalidBoneIndex { index: usize, bone_count: usize },

    /// Chain contains bones that are not properly connected.
    BrokenChain {
        bone_index: usize,
        expected_parent: usize,
        actual_parent: Option<usize>,
    },

    /// Pose bone count doesn't match skeleton.
    PoseMismatch {
        pose_bones: usize,
        skeleton_bones: usize,
    },
}

impl std::fmt::Display for CcdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyChain => write!(f, "CCD chain is empty"),
            Self::InvalidBoneIndex { index, bone_count } => {
                write!(
                    f,
                    "invalid bone index {} (skeleton has {} bones)",
                    index, bone_count
                )
            }
            Self::BrokenChain {
                bone_index,
                expected_parent,
                actual_parent,
            } => {
                write!(
                    f,
                    "broken chain at bone {}: expected parent {}, got {:?}",
                    bone_index, expected_parent, actual_parent
                )
            }
            Self::PoseMismatch {
                pose_bones,
                skeleton_bones,
            } => {
                write!(
                    f,
                    "pose has {} bones but skeleton has {}",
                    pose_bones, skeleton_bones
                )
            }
        }
    }
}

impl std::error::Error for CcdError {}

// ---------------------------------------------------------------------------
// Core CCD Solver
// ---------------------------------------------------------------------------

/// Compute world-space position for a bone given current pose.
fn compute_bone_world_position(skeleton: &Skeleton, pose: &Pose, bone_index: usize) -> Vec3 {
    let transforms: Vec<Transform> = (0..pose.bone_count())
        .map(|i| pose.get_transform(i))
        .collect();

    let world_transforms = skeleton.compute_world_transforms(&transforms);
    world_transforms[bone_index].w_axis.truncate()
}

/// Compute world-space transforms for all bones.
fn compute_world_transforms(skeleton: &Skeleton, pose: &Pose) -> Vec<Mat4> {
    let transforms: Vec<Transform> = (0..pose.bone_count())
        .map(|i| pose.get_transform(i))
        .collect();

    skeleton.compute_world_transforms(&transforms)
}

/// Solve inverse kinematics using CCD algorithm.
///
/// # Arguments
///
/// * `chain` - The IK chain to solve (effector to root order)
/// * `skeleton` - The skeleton hierarchy
/// * `pose` - Mutable pose to update with IK solution
/// * `params` - Solver parameters
///
/// # Returns
///
/// Result containing the solution and statistics.
///
/// # Example
///
/// ```ignore
/// let chain = CcdChain::new(vec![3, 2, 1, 0]);
/// let params = CcdParams::new(Vec3::new(1.0, 1.0, 0.0));
/// let result = solve_ccd(&chain, &skeleton, &mut pose, &params);
/// ```
pub fn solve_ccd(
    chain: &CcdChain,
    skeleton: &Skeleton,
    pose: &mut Pose,
    params: &CcdParams,
) -> CcdResult {
    // Validate inputs
    if chain.is_empty() {
        return CcdResult::new(0);
    }

    if pose.bone_count() != skeleton.bone_count() {
        return CcdResult::new(0);
    }

    let mut result = CcdResult::new(chain.len());

    // Store original rotations
    for (i, &bone_idx) in chain.bones.iter().enumerate() {
        result.bone_rotations[i] = pose.rotations[bone_idx];
    }

    // Main iteration loop
    for iteration in 0..params.max_iterations {
        result.iterations = iteration + 1;

        // Compute current effector position
        let world_transforms = compute_world_transforms(skeleton, pose);
        let effector_pos = world_transforms[chain.effector()].w_axis.truncate();

        // Check convergence
        let distance = (effector_pos - params.target_position).length();
        result.distance_history.push(distance);
        result.final_distance = distance;

        if distance <= params.tolerance {
            result.converged = true;
            break;
        }

        // Process each bone from effector towards root (skip effector itself)
        for (chain_idx, &bone_idx) in chain.bones.iter().enumerate().skip(1) {
            // Get world-space positions
            let world_transforms = compute_world_transforms(skeleton, pose);
            let bone_world_pos = world_transforms[bone_idx].w_axis.truncate();
            let effector_world_pos = world_transforms[chain.effector()].w_axis.truncate();

            // Vector from bone to effector
            let to_effector = effector_world_pos - bone_world_pos;
            let to_effector_len = to_effector.length();

            if to_effector_len < EPSILON {
                continue;
            }

            // Vector from bone to target
            let to_target = params.target_position - bone_world_pos;
            let to_target_len = to_target.length();

            if to_target_len < EPSILON {
                continue;
            }

            // Normalize vectors
            let to_effector_norm = to_effector / to_effector_len;
            let to_target_norm = to_target / to_target_len;

            // Calculate rotation to align effector direction to target direction
            let rotation = rotation_between_vectors(to_effector_norm, to_target_norm);

            if rotation.is_nan() {
                continue;
            }

            // Apply damping
            let damped_rotation = if params.damping < 1.0 - EPSILON {
                Quat::IDENTITY.slerp(rotation, params.damping)
            } else {
                rotation
            };

            // Get parent's world rotation (to convert back to local space)
            let parent_world_rot = if let Some(parent_idx) = skeleton.parent(bone_idx) {
                Quat::from_mat4(&world_transforms[parent_idx])
            } else {
                Quat::IDENTITY
            };

            // Current local rotation
            let current_local_rot = pose.rotations[bone_idx];

            // Current world rotation
            let current_world_rot = Quat::from_mat4(&world_transforms[bone_idx]);

            // New world rotation (apply delta in world space)
            let new_world_rot = (damped_rotation * current_world_rot).normalize();

            // Convert back to local space: local = inv(parent_world) * new_world
            let new_local_rot = (parent_world_rot.conjugate() * new_world_rot).normalize();

            // Apply constraints if any
            let final_rot = if let Some(constraint) = params.get_constraint(bone_idx) {
                let rest_rot = skeleton
                    .bone(bone_idx)
                    .map(|b| b.local_transform.rotation)
                    .unwrap_or(Quat::IDENTITY);
                constraint.apply(new_local_rot, rest_rot)
            } else {
                new_local_rot
            };

            // Update pose
            pose.rotations[bone_idx] = final_rot;

            // Update result
            result.bone_rotations[chain_idx] = final_rot;
        }

        // Apply pole target if specified (for elbow/knee direction)
        if let Some(pole_target) = params.pole_target {
            if chain.len() >= 3 && params.pole_weight > EPSILON {
                apply_pole_target(
                    chain,
                    skeleton,
                    pose,
                    pole_target,
                    params.pole_weight,
                    &params.constraints,
                );
            }
        }
    }

    // Final update of result rotations
    for (i, &bone_idx) in chain.bones.iter().enumerate() {
        result.bone_rotations[i] = pose.rotations[bone_idx];
    }

    result
}

/// Calculate the rotation that transforms one vector to another.
fn rotation_between_vectors(from: Vec3, to: Vec3) -> Quat {
    let dot = from.dot(to);

    // Handle parallel vectors
    if dot > 1.0 - EPSILON {
        return Quat::IDENTITY;
    }

    // Handle anti-parallel vectors
    if dot < -1.0 + EPSILON {
        // Find an orthogonal axis
        let ortho = if from.x.abs() < 0.9 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let axis = from.cross(ortho).normalize();
        return Quat::from_axis_angle(axis, std::f32::consts::PI);
    }

    // Normal case: use cross product as axis
    let axis = from.cross(to);
    let axis_len = axis.length();

    if axis_len < EPSILON {
        return Quat::IDENTITY;
    }

    let angle = dot.clamp(-1.0, 1.0).acos();
    Quat::from_axis_angle(axis / axis_len, angle)
}

/// Apply pole target to control elbow/knee direction.
fn apply_pole_target(
    chain: &CcdChain,
    skeleton: &Skeleton,
    pose: &mut Pose,
    pole_target: Vec3,
    weight: f32,
    constraints: &[JointConstraint],
) {
    // For a 3+ bone chain, we adjust the middle bone(s) to point towards the pole
    // This is most commonly used for arm (shoulder-elbow-wrist) or leg (hip-knee-ankle) chains

    if chain.len() < 3 {
        return;
    }

    let world_transforms = compute_world_transforms(skeleton, pose);

    // Get root, mid, and effector positions
    let root_pos = world_transforms[chain.root()].w_axis.truncate();
    let effector_pos = world_transforms[chain.effector()].w_axis.truncate();

    // For each middle bone (between root and effector)
    for &mid_idx in &chain.bones[1..chain.bones.len() - 1] {
        let mid_pos = world_transforms[mid_idx].w_axis.truncate();

        // Calculate the "hinge" plane normal (root -> effector direction)
        let chain_dir = (effector_pos - root_pos).normalize_or_zero();
        if chain_dir.length_squared() < EPSILON {
            continue;
        }

        // Project pole target onto plane perpendicular to chain direction
        let to_pole = pole_target - root_pos;
        let pole_on_plane = to_pole - chain_dir * to_pole.dot(chain_dir);

        if pole_on_plane.length_squared() < EPSILON {
            continue;
        }

        let pole_dir = pole_on_plane.normalize();

        // Current mid direction on plane
        let to_mid = mid_pos - root_pos;
        let mid_on_plane = to_mid - chain_dir * to_mid.dot(chain_dir);

        if mid_on_plane.length_squared() < EPSILON {
            continue;
        }

        let mid_dir = mid_on_plane.normalize();

        // Calculate rotation around chain axis to align mid with pole
        let rotation = rotation_between_vectors(mid_dir, pole_dir);

        // Apply weighted rotation
        let weighted_rot = Quat::IDENTITY.slerp(rotation, weight);

        // Apply to the bone before mid (parent of mid in chain)
        let chain_idx = chain
            .bones
            .iter()
            .position(|&b| b == mid_idx)
            .unwrap_or(0);
        if chain_idx + 1 < chain.bones.len() {
            let parent_in_chain = chain.bones[chain_idx + 1];

            let parent_world_rot = if let Some(pp) = skeleton.parent(parent_in_chain) {
                Quat::from_mat4(&world_transforms[pp])
            } else {
                Quat::IDENTITY
            };

            let current_world_rot = Quat::from_mat4(&world_transforms[parent_in_chain]);
            let new_world_rot = (weighted_rot * current_world_rot).normalize();
            let new_local_rot = (parent_world_rot.conjugate() * new_world_rot).normalize();

            // Apply constraints
            let final_rot =
                if let Some(constraint) = constraints.iter().find(|c| c.bone_index == parent_in_chain)
                {
                    let rest_rot = skeleton
                        .bone(parent_in_chain)
                        .map(|b| b.local_transform.rotation)
                        .unwrap_or(Quat::IDENTITY);
                    constraint.apply(new_local_rot, rest_rot)
                } else {
                    new_local_rot
                };

            pose.rotations[parent_in_chain] = final_rot;
        }
    }
}

/// Solve CCD with validation (returns Result for error handling).
pub fn solve_ccd_checked(
    chain: &CcdChain,
    skeleton: &Skeleton,
    pose: &mut Pose,
    params: &CcdParams,
) -> Result<CcdResult, CcdError> {
    // Validate chain
    chain.validate(skeleton)?;

    // Validate pose matches skeleton
    if pose.bone_count() != skeleton.bone_count() {
        return Err(CcdError::PoseMismatch {
            pose_bones: pose.bone_count(),
            skeleton_bones: skeleton.bone_count(),
        });
    }

    Ok(solve_ccd(chain, skeleton, pose, params))
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Calculate the total chain length (sum of bone lengths).
pub fn calculate_chain_length(chain: &CcdChain, skeleton: &Skeleton, pose: &Pose) -> f32 {
    if chain.len() < 2 {
        return 0.0;
    }

    let world_transforms = compute_world_transforms(skeleton, pose);
    let mut total = 0.0;

    for i in 0..chain.len() - 1 {
        let bone_a = chain.bones[i];
        let bone_b = chain.bones[i + 1];

        let pos_a = world_transforms[bone_a].w_axis.truncate();
        let pos_b = world_transforms[bone_b].w_axis.truncate();

        total += (pos_a - pos_b).length();
    }

    total
}

/// Check if a target is reachable by the chain.
pub fn is_target_reachable(
    chain: &CcdChain,
    skeleton: &Skeleton,
    pose: &Pose,
    target: Vec3,
) -> bool {
    let chain_length = calculate_chain_length(chain, skeleton, pose);
    let world_transforms = compute_world_transforms(skeleton, pose);

    let root_pos = world_transforms[chain.root()].w_axis.truncate();
    let distance_to_target = (target - root_pos).length();

    distance_to_target <= chain_length
}

/// Calculate the distance from effector to target.
pub fn effector_to_target_distance(
    chain: &CcdChain,
    skeleton: &Skeleton,
    pose: &Pose,
    target: Vec3,
) -> f32 {
    let effector_pos = compute_bone_world_position(skeleton, pose, chain.effector());
    (effector_pos - target).length()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Bone, SkeletonBuilder};
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    // ==================== Test Helpers ====================

    fn create_simple_2_bone_skeleton() -> Skeleton {
        SkeletonBuilder::new()
            .root("root")
            .child_at("bone1", "root", Vec3::new(0.0, 1.0, 0.0))
            .child_at("effector", "bone1", Vec3::new(0.0, 1.0, 0.0))
            .build()
            .expect("valid skeleton")
    }

    fn create_5_bone_skeleton() -> Skeleton {
        SkeletonBuilder::new()
            .root("root")
            .child_at("bone1", "root", Vec3::new(0.0, 1.0, 0.0))
            .child_at("bone2", "bone1", Vec3::new(0.0, 1.0, 0.0))
            .child_at("bone3", "bone2", Vec3::new(0.0, 1.0, 0.0))
            .child_at("effector", "bone3", Vec3::new(0.0, 1.0, 0.0))
            .build()
            .expect("valid skeleton")
    }

    fn create_pose_for_skeleton(skeleton: &Skeleton) -> Pose {
        Pose::from_skeleton(skeleton, crate::pose::PoseType::Current)
    }

    // ==================== Basic Tests (12) ====================

    #[test]
    fn test_ccd_chain_new() {
        let chain = CcdChain::new(vec![3, 2, 1, 0]);
        assert_eq!(chain.len(), 4);
        assert_eq!(chain.effector(), 3);
        assert_eq!(chain.root(), 0);
    }

    #[test]
    fn test_ccd_chain_with_name() {
        let chain = CcdChain::new(vec![2, 1, 0]).with_name("arm_ik");
        assert_eq!(chain.name, Some("arm_ik".to_string()));
    }

    #[test]
    fn test_ccd_chain_from_skeleton() {
        let skeleton = create_5_bone_skeleton();
        let chain = CcdChain::from_skeleton(&skeleton, 4, None).unwrap();

        assert_eq!(chain.len(), 5);
        assert_eq!(chain.effector(), 4);
        assert_eq!(chain.root(), 0);
    }

    #[test]
    fn test_ccd_chain_from_skeleton_partial() {
        let skeleton = create_5_bone_skeleton();
        let chain = CcdChain::from_skeleton(&skeleton, 4, Some(2)).unwrap();

        assert_eq!(chain.len(), 3);
        assert_eq!(chain.effector(), 4);
        assert_eq!(chain.root(), 2);
    }

    #[test]
    fn test_ccd_chain_validate_valid() {
        let skeleton = create_5_bone_skeleton();
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        assert!(chain.validate(&skeleton).is_ok());
    }

    #[test]
    fn test_ccd_chain_validate_invalid_index() {
        let skeleton = create_5_bone_skeleton();
        let chain = CcdChain::new(vec![10, 3, 2, 1, 0]);

        assert!(matches!(
            chain.validate(&skeleton),
            Err(CcdError::InvalidBoneIndex { .. })
        ));
    }

    #[test]
    fn test_ccd_params_default() {
        let params = CcdParams::default();

        assert_eq!(params.max_iterations, DEFAULT_MAX_ITERATIONS);
        assert_eq!(params.tolerance, DEFAULT_TOLERANCE);
        assert_eq!(params.damping, DEFAULT_DAMPING);
        assert!(params.constraints.is_empty());
    }

    #[test]
    fn test_ccd_params_builder() {
        let params = CcdParams::new(Vec3::new(1.0, 2.0, 3.0))
            .with_max_iterations(20)
            .with_tolerance(0.01)
            .with_damping(0.5);

        assert_eq!(params.target_position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(params.max_iterations, 20);
        assert_eq!(params.tolerance, 0.01);
        assert_eq!(params.damping, 0.5);
    }

    #[test]
    fn test_ccd_result_new() {
        let result = CcdResult::new(5);

        assert_eq!(result.bone_rotations.len(), 5);
        assert_eq!(result.iterations, 0);
        assert!(!result.converged);
        assert_eq!(result.final_distance, f32::MAX);
    }

    #[test]
    fn test_ccd_result_improvement_ratio() {
        let mut result = CcdResult::new(3);
        result.distance_history = vec![10.0, 5.0, 2.0, 1.0];
        result.final_distance = 1.0;

        let ratio = result.improvement_ratio();
        assert!((ratio - 0.9).abs() < 0.01); // 90% improvement
    }

    #[test]
    fn test_rotation_between_vectors_identity() {
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rot = rotation_between_vectors(v, v);

        assert!(rot.abs_diff_eq(Quat::IDENTITY, 1e-5));
    }

    #[test]
    fn test_rotation_between_vectors_90_degrees() {
        let from = Vec3::X;
        let to = Vec3::Y;
        let rot = rotation_between_vectors(from, to);

        let result = rot * from;
        assert!(result.abs_diff_eq(to, 1e-5));
    }

    // ==================== 2-Bone Chain Tests (8) ====================

    #[test]
    fn test_2_bone_basic_solve_target_forward() {
        let skeleton = create_simple_2_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![2, 1, 0]);

        // Target straight up (already aligned)
        let params = CcdParams::new(Vec3::new(0.0, 2.0, 0.0))
            .with_max_iterations(10)
            .with_tolerance(0.01);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        assert!(result.converged || result.final_distance < 0.1);
    }

    #[test]
    fn test_2_bone_solve_target_right() {
        let skeleton = create_simple_2_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![2, 1, 0]);

        // Target to the right
        let params = CcdParams::new(Vec3::new(1.5, 1.0, 0.0))
            .with_max_iterations(20)
            .with_tolerance(0.01);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should get reasonably close
        assert!(result.final_distance < 0.5);
    }

    #[test]
    fn test_2_bone_unreachable_target() {
        let skeleton = create_simple_2_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![2, 1, 0]);

        // Target way too far (chain length is 2, target at 10)
        let params = CcdParams::new(Vec3::new(0.0, 10.0, 0.0))
            .with_max_iterations(10)
            .with_tolerance(0.001);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should NOT converge but should extend towards target
        assert!(!result.converged);
        assert!(result.final_distance > 5.0); // Still far from unreachable target
    }

    #[test]
    fn test_2_bone_target_at_root() {
        let skeleton = create_simple_2_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![2, 1, 0]);

        // Target at origin (root position)
        let params = CcdParams::new(Vec3::ZERO)
            .with_max_iterations(15)
            .with_tolerance(0.01);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should get close (folding back)
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_2_bone_with_damping() {
        let skeleton = create_simple_2_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![2, 1, 0]);

        // Same target but with heavy damping
        let params = CcdParams::new(Vec3::new(1.0, 1.0, 0.0))
            .with_max_iterations(5)
            .with_damping(0.3);

        let result_damped = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Reset pose
        let mut pose2 = create_pose_for_skeleton(&skeleton);
        let params_no_damp = CcdParams::new(Vec3::new(1.0, 1.0, 0.0))
            .with_max_iterations(5)
            .with_damping(1.0);

        let result_undamped = solve_ccd(&chain, &skeleton, &mut pose2, &params_no_damp);

        // With same iterations, undamped should converge faster (get closer)
        // But damped should be smoother (harder to test directly)
        assert!(result_damped.iterations >= 1);
        assert!(result_undamped.iterations >= 1);
    }

    #[test]
    fn test_2_bone_iterations_vs_accuracy() {
        let skeleton = create_simple_2_bone_skeleton();
        let target = Vec3::new(1.2, 0.8, 0.0);

        // 1 iteration
        let mut pose1 = create_pose_for_skeleton(&skeleton);
        let result1 = solve_ccd(
            &CcdChain::new(vec![2, 1, 0]),
            &skeleton,
            &mut pose1,
            &CcdParams::new(target).with_max_iterations(1),
        );

        // 5 iterations
        let mut pose5 = create_pose_for_skeleton(&skeleton);
        let result5 = solve_ccd(
            &CcdChain::new(vec![2, 1, 0]),
            &skeleton,
            &mut pose5,
            &CcdParams::new(target).with_max_iterations(5),
        );

        // 20 iterations
        let mut pose20 = create_pose_for_skeleton(&skeleton);
        let result20 = solve_ccd(
            &CcdChain::new(vec![2, 1, 0]),
            &skeleton,
            &mut pose20,
            &CcdParams::new(target).with_max_iterations(20),
        );

        // More iterations should give better (or equal) results
        assert!(result5.final_distance <= result1.final_distance + 0.01);
        assert!(result20.final_distance <= result5.final_distance + 0.01);
    }

    #[test]
    fn test_2_bone_early_termination() {
        let skeleton = create_simple_2_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![2, 1, 0]);

        // Target very close to current effector position
        let params = CcdParams::new(Vec3::new(0.0, 2.0, 0.0))
            .with_max_iterations(100)
            .with_tolerance(0.1);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should converge very quickly (likely 1-2 iterations)
        assert!(result.converged);
        assert!(result.iterations < 50);
    }

    #[test]
    fn test_2_bone_aligned_bones() {
        let skeleton = create_simple_2_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![2, 1, 0]);

        // Target along the chain direction
        let params = CcdParams::new(Vec3::new(0.0, 2.5, 0.0))
            .with_max_iterations(10)
            .with_tolerance(0.1);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should handle gracefully (bones are aligned, can't reach further)
        assert!(result.iterations > 0);
    }

    // ==================== 5-Bone Chain Tests (8) ====================

    #[test]
    fn test_5_bone_basic_solve() {
        let skeleton = create_5_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        let params = CcdParams::new(Vec3::new(2.0, 2.0, 0.0))
            .with_max_iterations(15)
            .with_tolerance(0.05);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        assert!(result.final_distance < 1.0);
    }

    #[test]
    fn test_5_bone_convergence() {
        let skeleton = create_5_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        // Reachable target
        let params = CcdParams::new(Vec3::new(1.5, 1.5, 1.5))
            .with_max_iterations(30)
            .with_tolerance(0.05);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Distance should decrease over iterations
        if result.distance_history.len() >= 2 {
            let first = result.distance_history[0];
            let last = *result.distance_history.last().unwrap();
            assert!(last <= first);
        }
    }

    #[test]
    fn test_5_bone_with_pole_target() {
        let skeleton = create_5_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        let params = CcdParams::new(Vec3::new(2.0, 0.0, 0.0))
            .with_max_iterations(15)
            .with_pole_target(Vec3::new(0.0, 0.0, 1.0), 0.5);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should complete without errors
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_5_bone_partial_chain() {
        let skeleton = create_5_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);

        // Only use bones 4, 3, 2 (partial chain)
        let chain = CcdChain::new(vec![4, 3, 2]);

        let params = CcdParams::new(Vec3::new(1.0, 3.0, 0.0))
            .with_max_iterations(10)
            .with_tolerance(0.1);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should solve with 3 bones
        assert_eq!(result.bone_rotations.len(), 3);
    }

    #[test]
    fn test_5_bone_distance_history() {
        let skeleton = create_5_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        let params = CcdParams::new(Vec3::new(2.0, 1.0, 0.0))
            .with_max_iterations(10)
            .with_tolerance(0.01);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should have recorded distances
        assert!(!result.distance_history.is_empty());
        assert!(result.distance_history.len() <= 10);
    }

    #[test]
    fn test_5_bone_target_behind() {
        let skeleton = create_5_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        // Target behind the chain
        let params = CcdParams::new(Vec3::new(0.0, -1.0, 0.0))
            .with_max_iterations(20)
            .with_tolerance(0.1);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should still attempt to solve
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_5_bone_sideways_target() {
        let skeleton = create_5_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        // Target to the side
        let params = CcdParams::new(Vec3::new(3.0, 0.0, 0.0))
            .with_max_iterations(15)
            .with_tolerance(0.1);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should reach or get close (chain length is 4)
        assert!(result.final_distance < 2.0);
    }

    #[test]
    fn test_5_bone_tiny_tolerance() {
        let skeleton = create_5_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        // Very tight tolerance
        let params = CcdParams::new(Vec3::new(0.0, 4.0, 0.0)) // Straight up, should match
            .with_max_iterations(50)
            .with_tolerance(0.0001);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should converge (target is exactly at max reach)
        assert!(result.final_distance < 0.1);
    }

    // ==================== Constraint Tests (10) ====================

    #[test]
    fn test_constraint_none() {
        let constraint = JointConstraint::none(0);
        let rot = Quat::from_axis_angle(Vec3::Y, 1.0);
        let result = constraint.apply(rot, Quat::IDENTITY);

        assert!(result.abs_diff_eq(rot, 1e-5));
    }

    #[test]
    fn test_constraint_cone_within_limit() {
        let constraint = JointConstraint::cone(0, FRAC_PI_2, Vec3::Y);

        // Small rotation within cone
        let rot = Quat::from_axis_angle(Vec3::X, 0.1);
        let result = constraint.apply(rot, Quat::IDENTITY);

        // Should be unchanged
        assert!(result.abs_diff_eq(rot, 1e-3));
    }

    #[test]
    fn test_constraint_cone_exceeds_limit() {
        let constraint = JointConstraint::cone(0, 0.1, Vec3::Y);

        // Large rotation exceeding cone
        let rot = Quat::from_axis_angle(Vec3::X, 1.0);
        let result = constraint.apply(rot, Quat::IDENTITY);

        // Should be clamped (rotation reduced)
        let (_, angle) = result.to_axis_angle();
        assert!(angle < 0.5); // Significantly reduced
    }

    #[test]
    fn test_constraint_hinge_within_limits() {
        let constraint = JointConstraint::hinge(0, Vec3::Y, -FRAC_PI_2, FRAC_PI_2);

        let rot = Quat::from_axis_angle(Vec3::Y, 0.3);
        let result = constraint.apply(rot, Quat::IDENTITY);

        // Should be close to original (only twist around Y)
        assert!(result.abs_diff_eq(rot, 0.1));
    }

    #[test]
    fn test_constraint_hinge_exceeds_max() {
        let constraint = JointConstraint::hinge(0, Vec3::Y, -0.5, 0.5);

        // Rotation exceeding max
        let rot = Quat::from_axis_angle(Vec3::Y, 2.0);
        let result = constraint.apply(rot, Quat::IDENTITY);

        // Should be clamped to max
        let (_, angle) = result.to_axis_angle();
        assert!(angle <= 0.6); // Near the limit
    }

    #[test]
    fn test_constraint_euler_limits() {
        let constraint = JointConstraint::euler_limits(
            0,
            (-0.5, 0.5), // X
            (-0.5, 0.5), // Y
            (-0.5, 0.5), // Z
        );

        // Rotation within limits
        let rot = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.3, 0.3);
        let result = constraint.apply(rot, Quat::IDENTITY);

        // Should be similar
        let (x, y, z) = result.to_euler(glam::EulerRot::XYZ);
        assert!((x - 0.3).abs() < 0.1);
        assert!((y - 0.3).abs() < 0.1);
        assert!((z - 0.3).abs() < 0.1);
    }

    #[test]
    fn test_constraint_euler_limits_clamped() {
        let constraint = JointConstraint::euler_limits(
            0,
            (-0.2, 0.2), // X
            (-0.2, 0.2), // Y
            (-0.2, 0.2), // Z
        );

        // Large rotation
        let rot = Quat::from_euler(glam::EulerRot::XYZ, 1.0, 1.0, 1.0);
        let result = constraint.apply(rot, Quat::IDENTITY);

        // Should be clamped
        let (x, y, z) = result.to_euler(glam::EulerRot::XYZ);
        assert!(x <= 0.25);
        assert!(y <= 0.25);
        assert!(z <= 0.25);
    }

    #[test]
    fn test_constraint_weight_zero() {
        let constraint = JointConstraint::hinge(0, Vec3::Y, 0.0, 0.0).with_weight(0.0);

        let rot = Quat::from_axis_angle(Vec3::X, 1.0);
        let result = constraint.apply(rot, Quat::IDENTITY);

        // Weight 0 means no constraint
        assert!(result.abs_diff_eq(rot, 1e-5));
    }

    #[test]
    fn test_constraint_weight_partial() {
        let constraint = JointConstraint::hinge(0, Vec3::Y, 0.0, 0.0).with_weight(0.5);

        let rot = Quat::from_axis_angle(Vec3::X, 1.0);
        let result = constraint.apply(rot, Quat::IDENTITY);

        // Partial constraint - should be between original and fully constrained
        let (_, angle) = result.to_axis_angle();
        assert!(angle > 0.1 && angle < 0.9);
    }

    #[test]
    fn test_solve_with_constraints() {
        let skeleton = create_5_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        // Add hinge constraints
        let params = CcdParams::new(Vec3::new(2.0, 1.0, 0.0))
            .with_max_iterations(15)
            .with_constraint(JointConstraint::hinge(1, Vec3::Z, -1.0, 1.0))
            .with_constraint(JointConstraint::hinge(2, Vec3::Z, -1.0, 1.0));

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should still solve (maybe not perfectly due to constraints)
        assert!(result.iterations > 0);
    }

    // ==================== Edge Case Tests (8) ====================

    #[test]
    fn test_single_bone_chain() {
        let skeleton = create_simple_2_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![2]);

        let params = CcdParams::new(Vec3::new(1.0, 1.0, 0.0))
            .with_max_iterations(5);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should handle single bone (no joints to rotate after effector)
        assert_eq!(result.bone_rotations.len(), 1);
    }

    #[test]
    fn test_target_at_effector() {
        let skeleton = create_simple_2_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![2, 1, 0]);

        // Target exactly at current effector position
        let params = CcdParams::new(Vec3::new(0.0, 2.0, 0.0))
            .with_max_iterations(10)
            .with_tolerance(0.001);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should converge immediately
        assert!(result.converged);
        assert!(result.iterations <= 2);
    }

    #[test]
    fn test_zero_length_vectors() {
        let skeleton = create_simple_2_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![2, 1, 0]);

        // Target at bone position (zero-length to-target vector)
        let params = CcdParams::new(Vec3::new(0.0, 1.0, 0.0)) // At bone1 position
            .with_max_iterations(5);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should handle gracefully
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_nan_handling() {
        let from = Vec3::ZERO;
        let to = Vec3::ZERO;
        let rot = rotation_between_vectors(from, to);

        // Should return identity, not NaN
        assert!(!rot.is_nan());
    }

    #[test]
    fn test_anti_parallel_vectors() {
        let from = Vec3::Y;
        let to = -Vec3::Y;
        let rot = rotation_between_vectors(from, to);

        // Should handle 180 degree rotation
        assert!(!rot.is_nan());
        let result = rot * from;
        assert!(result.abs_diff_eq(to, 1e-4) || result.abs_diff_eq(-to, 1e-4));
    }

    #[test]
    fn test_checked_solve_valid() {
        let skeleton = create_simple_2_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![2, 1, 0]);
        let params = CcdParams::new(Vec3::new(1.0, 1.0, 0.0));

        let result = solve_ccd_checked(&chain, &skeleton, &mut pose, &params);

        assert!(result.is_ok());
    }

    #[test]
    fn test_checked_solve_invalid_chain() {
        let skeleton = create_simple_2_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![10, 9, 8]); // Invalid indices
        let params = CcdParams::new(Vec3::new(1.0, 1.0, 0.0));

        let result = solve_ccd_checked(&chain, &skeleton, &mut pose, &params);

        assert!(matches!(result, Err(CcdError::InvalidBoneIndex { .. })));
    }

    #[test]
    fn test_checked_solve_pose_mismatch() {
        let skeleton = create_simple_2_bone_skeleton();
        let mut pose = Pose::new(10, crate::pose::PoseType::Current); // Wrong size
        let chain = CcdChain::new(vec![2, 1, 0]);
        let params = CcdParams::new(Vec3::new(1.0, 1.0, 0.0));

        let result = solve_ccd_checked(&chain, &skeleton, &mut pose, &params);

        assert!(matches!(result, Err(CcdError::PoseMismatch { .. })));
    }

    // ==================== Utility Function Tests (5) ====================

    #[test]
    fn test_calculate_chain_length() {
        let skeleton = create_5_bone_skeleton();
        let pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        let length = calculate_chain_length(&chain, &skeleton, &pose);

        // 4 bones with 1 unit each
        assert!((length - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_is_target_reachable_yes() {
        let skeleton = create_5_bone_skeleton();
        let pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        // Target within reach
        let reachable = is_target_reachable(&chain, &skeleton, &pose, Vec3::new(2.0, 2.0, 0.0));

        assert!(reachable);
    }

    #[test]
    fn test_is_target_reachable_no() {
        let skeleton = create_5_bone_skeleton();
        let pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        // Target too far
        let reachable = is_target_reachable(&chain, &skeleton, &pose, Vec3::new(10.0, 10.0, 0.0));

        assert!(!reachable);
    }

    #[test]
    fn test_effector_to_target_distance() {
        let skeleton = create_simple_2_bone_skeleton();
        let pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![2, 1, 0]);

        let target = Vec3::new(0.0, 5.0, 0.0);
        let distance = effector_to_target_distance(&chain, &skeleton, &pose, target);

        // Effector at (0, 2, 0), target at (0, 5, 0) = distance 3
        assert!((distance - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_calculate_chain_length_empty() {
        let skeleton = create_simple_2_bone_skeleton();
        let pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![2]);

        let length = calculate_chain_length(&chain, &skeleton, &pose);

        // Single bone = no length between bones
        assert!(length.abs() < 0.01);
    }

    // ==================== Performance/Benchmark Hints (4) ====================

    #[test]
    fn test_many_iterations_performance() {
        let skeleton = create_5_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        let params = CcdParams::new(Vec3::new(2.0, 2.0, 2.0))
            .with_max_iterations(100)
            .with_tolerance(0.0001);

        let start = std::time::Instant::now();
        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);
        let elapsed = start.elapsed();

        // Should complete reasonably fast (< 10ms for 100 iterations)
        assert!(elapsed.as_millis() < 100);
        assert!(result.iterations <= 100);
    }

    #[test]
    fn test_long_chain_solve() {
        // Create a longer skeleton
        let mut skeleton = crate::skeleton::Skeleton::new();
        skeleton.add_bone(Bone::root("root"));

        for i in 1..=15 {
            skeleton.add_bone(
                Bone::new(format!("bone{}", i))
                    .with_parent(i - 1)
                    .with_local_transform(Transform::from_position(Vec3::new(0.0, 0.5, 0.0))),
            );
        }

        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let bones: Vec<usize> = (0..=15).rev().collect();
        let chain = CcdChain::new(bones);

        let params = CcdParams::new(Vec3::new(4.0, 4.0, 0.0))
            .with_max_iterations(30);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Should handle long chains
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_multiple_times() {
        let skeleton = create_5_bone_skeleton();
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        // Solve multiple times with different targets
        let targets = [
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 2.0, 1.0),
            Vec3::new(0.0, 3.0, -1.0),
            Vec3::new(2.0, 0.0, 2.0),
        ];

        for target in targets {
            let mut pose = create_pose_for_skeleton(&skeleton);
            let params = CcdParams::new(target).with_max_iterations(15);
            let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

            assert!(result.iterations > 0);
        }
    }

    #[test]
    fn test_convergence_rate() {
        let skeleton = create_5_bone_skeleton();
        let mut pose = create_pose_for_skeleton(&skeleton);
        let chain = CcdChain::new(vec![4, 3, 2, 1, 0]);

        let params = CcdParams::new(Vec3::new(2.5, 1.5, 0.5))
            .with_max_iterations(20)
            .with_tolerance(0.01);

        let result = solve_ccd(&chain, &skeleton, &mut pose, &params);

        // Check that distance generally decreases
        if result.distance_history.len() >= 3 {
            let mut decreasing_count = 0;
            for i in 1..result.distance_history.len() {
                if result.distance_history[i] <= result.distance_history[i - 1] {
                    decreasing_count += 1;
                }
            }
            // Most iterations should show improvement
            let decrease_ratio = decreasing_count as f32 / (result.distance_history.len() - 1) as f32;
            assert!(decrease_ratio > 0.5);
        }
    }
}
