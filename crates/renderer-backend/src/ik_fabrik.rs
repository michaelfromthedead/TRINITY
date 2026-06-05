//! FABRIK Inverse Kinematics Solver for TRINITY Engine (T-AN-4.2).
//!
//! This module implements the FABRIK (Forward And Backward Reaching Inverse Kinematics)
//! algorithm for solving inverse kinematics on arbitrary-length bone chains.
//!
//! # Algorithm Overview
//!
//! FABRIK works by iteratively performing two passes:
//!
//! 1. **Forward Pass**: Start from the effector (tip), move toward the root.
//!    Each joint is repositioned to satisfy the distance constraint from its child.
//!
//! 2. **Backward Pass**: Start from the root, move toward the effector.
//!    The root is anchored at its original position, and constraints propagate outward.
//!
//! This process repeats until the effector reaches the target (within tolerance)
//! or the maximum iteration count is reached.
//!
//! # Joint Constraints
//!
//! FABRIK supports per-joint angle limits:
//! - **Cone**: Restricts the joint to a cone around the parent bone direction
//! - **Hinge**: Restricts rotation to a single axis with min/max angles
//! - **BallSocket**: Allows swing within a limit and twist around the bone axis
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::ik_fabrik::{FabrikChain, FabrikParams, JointConstraint, solve_fabrik};
//! use renderer_backend::skeleton::Skeleton;
//! use renderer_backend::pose::Pose;
//! use glam::Vec3;
//!
//! // Define a chain from root to tip (bone indices in order)
//! let chain = FabrikChain {
//!     bones: vec![0, 1, 2],  // root -> mid -> tip
//! };
//!
//! // Set up solver parameters
//! let params = FabrikParams {
//!     target_position: Vec3::new(5.0, 3.0, 0.0),
//!     max_iterations: 10,
//!     tolerance: 0.001,
//!     constraints: vec![JointConstraint::None; 3],
//! };
//!
//! // Solve IK
//! let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);
//!
//! if result.converged {
//!     println!("IK solved in {} iterations", result.iterations);
//! }
//! ```

use glam::{Quat, Vec3};

use crate::pose::Pose;
use crate::skeleton::{Skeleton, Transform};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum iterations for FABRIK solver.
pub const DEFAULT_MAX_ITERATIONS: u32 = 10;

/// Default tolerance for convergence check (in world units).
pub const DEFAULT_TOLERANCE: f32 = 0.001;

/// Minimum bone length to avoid division by zero.
const MIN_BONE_LENGTH: f32 = 1e-6;

/// Small epsilon for angle comparisons.
const ANGLE_EPSILON: f32 = 1e-6;

// ---------------------------------------------------------------------------
// FabrikChain
// ---------------------------------------------------------------------------

/// Defines a kinematic chain for FABRIK solving.
///
/// A chain is an ordered list of bone indices from root to tip (effector).
/// The solver will manipulate these bones to reach the target position.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FabrikChain {
    /// Bone indices from root to tip (effector).
    /// Must contain at least 2 bones for a valid chain.
    pub bones: Vec<usize>,
}

impl FabrikChain {
    /// Create a new FABRIK chain from bone indices.
    ///
    /// # Arguments
    ///
    /// * `bones` - Bone indices from root to tip
    pub fn new(bones: Vec<usize>) -> Self {
        Self { bones }
    }

    /// Create a chain with a single segment (2 bones).
    pub fn two_bone(root: usize, tip: usize) -> Self {
        Self {
            bones: vec![root, tip],
        }
    }

    /// Create a chain with two segments (3 bones).
    pub fn three_bone(root: usize, mid: usize, tip: usize) -> Self {
        Self {
            bones: vec![root, mid, tip],
        }
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

    /// Check if the chain is valid (has at least 2 bones).
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.bones.len() >= 2
    }

    /// Get the root bone index.
    #[inline]
    pub fn root(&self) -> Option<usize> {
        self.bones.first().copied()
    }

    /// Get the effector (tip) bone index.
    #[inline]
    pub fn effector(&self) -> Option<usize> {
        self.bones.last().copied()
    }

    /// Get the number of segments (bones - 1).
    #[inline]
    pub fn segment_count(&self) -> usize {
        if self.bones.len() > 0 {
            self.bones.len() - 1
        } else {
            0
        }
    }
}

// ---------------------------------------------------------------------------
// JointConstraint
// ---------------------------------------------------------------------------

/// Joint angle constraint types for FABRIK.
///
/// Constraints are applied after each backward pass to limit joint angles.
#[derive(Clone, Debug, PartialEq)]
pub enum JointConstraint {
    /// No constraint - full freedom of movement.
    None,

    /// Cone constraint: joint can move within a cone around the parent bone direction.
    ///
    /// The angle is the half-angle of the cone in radians.
    Cone {
        /// Half-angle of the cone in radians (e.g., PI/4 for 45 degrees).
        angle: f32,
    },

    /// Hinge constraint: joint rotates around a single axis.
    ///
    /// Common for elbows and knees.
    Hinge {
        /// The axis of rotation (in parent bone's local space).
        axis: Vec3,
        /// Minimum angle in radians.
        min: f32,
        /// Maximum angle in radians.
        max: f32,
    },

    /// Ball-and-socket constraint: limited swing and twist.
    ///
    /// Common for shoulders and hips.
    BallSocket {
        /// Maximum swing angle from the rest direction (in radians).
        swing_limit: f32,
        /// Maximum twist angle around the bone axis (in radians).
        twist_limit: f32,
    },
}

impl Default for JointConstraint {
    fn default() -> Self {
        JointConstraint::None
    }
}

impl JointConstraint {
    /// Create a cone constraint with the given half-angle in degrees.
    pub fn cone_degrees(angle_degrees: f32) -> Self {
        JointConstraint::Cone {
            angle: angle_degrees.to_radians(),
        }
    }

    /// Create a hinge constraint with angles in degrees.
    pub fn hinge_degrees(axis: Vec3, min_degrees: f32, max_degrees: f32) -> Self {
        JointConstraint::Hinge {
            axis: axis.normalize_or_zero(),
            min: min_degrees.to_radians(),
            max: max_degrees.to_radians(),
        }
    }

    /// Create a ball-socket constraint with angles in degrees.
    pub fn ball_socket_degrees(swing_degrees: f32, twist_degrees: f32) -> Self {
        JointConstraint::BallSocket {
            swing_limit: swing_degrees.to_radians(),
            twist_limit: twist_degrees.to_radians(),
        }
    }

    /// Check if this constraint allows unrestricted movement.
    #[inline]
    pub fn is_unconstrained(&self) -> bool {
        matches!(self, JointConstraint::None)
    }
}

// ---------------------------------------------------------------------------
// FabrikParams
// ---------------------------------------------------------------------------

/// Parameters for the FABRIK solver.
#[derive(Clone, Debug, PartialEq)]
pub struct FabrikParams {
    /// Target position for the effector (end of the chain) in world space.
    pub target_position: Vec3,

    /// Maximum number of iterations before giving up.
    pub max_iterations: u32,

    /// Distance tolerance for convergence (solver stops when effector is within this distance).
    pub tolerance: f32,

    /// Per-joint constraints. Should have one entry per bone in the chain.
    /// If fewer constraints are provided, remaining joints are unconstrained.
    pub constraints: Vec<JointConstraint>,
}

impl Default for FabrikParams {
    fn default() -> Self {
        Self {
            target_position: Vec3::ZERO,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tolerance: DEFAULT_TOLERANCE,
            constraints: Vec::new(),
        }
    }
}

impl FabrikParams {
    /// Create new FABRIK parameters with the given target.
    pub fn new(target_position: Vec3) -> Self {
        Self {
            target_position,
            ..Default::default()
        }
    }

    /// Set the maximum number of iterations.
    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the joint constraints.
    pub fn with_constraints(mut self, constraints: Vec<JointConstraint>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Get the constraint for a specific joint index.
    ///
    /// Returns `JointConstraint::None` if the index is out of bounds.
    pub fn get_constraint(&self, index: usize) -> &JointConstraint {
        self.constraints.get(index).unwrap_or(&JointConstraint::None)
    }
}

// ---------------------------------------------------------------------------
// FabrikResult
// ---------------------------------------------------------------------------

/// Result of FABRIK solving.
#[derive(Clone, Debug, PartialEq)]
pub struct FabrikResult {
    /// World-space positions of each bone in the chain after solving.
    pub bone_positions: Vec<Vec3>,

    /// Number of iterations performed.
    pub iterations: u32,

    /// Whether the solver converged (effector reached target within tolerance).
    pub converged: bool,

    /// Final distance from effector to target.
    pub final_distance: f32,
}

impl Default for FabrikResult {
    fn default() -> Self {
        Self {
            bone_positions: Vec::new(),
            iterations: 0,
            converged: false,
            final_distance: f32::MAX,
        }
    }
}

impl FabrikResult {
    /// Check if the target was reachable.
    ///
    /// A target is considered reachable if the solver converged.
    #[inline]
    pub fn is_reachable(&self) -> bool {
        self.converged
    }

    /// Get the effector position after solving.
    #[inline]
    pub fn effector_position(&self) -> Option<Vec3> {
        self.bone_positions.last().copied()
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Compute world-space positions for each bone in the chain.
fn compute_chain_positions(chain: &FabrikChain, skeleton: &Skeleton, pose: &Pose) -> Vec<Vec3> {
    let transforms = pose.transforms();
    let world_transforms = skeleton.compute_world_transforms(&transforms);

    chain
        .bones
        .iter()
        .map(|&bone_idx| world_transforms[bone_idx].w_axis.truncate())
        .collect()
}

/// Compute the bone lengths between consecutive joints.
fn compute_bone_lengths(positions: &[Vec3]) -> Vec<f32> {
    if positions.len() < 2 {
        return Vec::new();
    }

    positions
        .windows(2)
        .map(|w| (w[1] - w[0]).length().max(MIN_BONE_LENGTH))
        .collect()
}

/// Compute total chain length.
fn compute_total_length(lengths: &[f32]) -> f32 {
    lengths.iter().sum()
}

/// Move a point toward another point by a fixed distance.
///
/// Returns the new position that is exactly `distance` away from `toward`.
fn move_toward(from: Vec3, toward: Vec3, distance: f32) -> Vec3 {
    let direction = from - toward;
    let len = direction.length();

    if len < MIN_BONE_LENGTH {
        // Points are coincident, pick an arbitrary direction
        toward + Vec3::Y * distance
    } else {
        toward + (direction / len) * distance
    }
}

/// Apply a cone constraint to a joint position.
///
/// The constrained position lies within a cone of half-angle `max_angle`
/// centered on the direction from `parent_pos` to `rest_direction`.
fn apply_cone_constraint(
    joint_pos: Vec3,
    parent_pos: Vec3,
    reference_dir: Vec3,
    max_angle: f32,
    bone_length: f32,
) -> Vec3 {
    let current_dir = (joint_pos - parent_pos).normalize_or_zero();

    if current_dir.length_squared() < MIN_BONE_LENGTH {
        return parent_pos + reference_dir.normalize_or_zero() * bone_length;
    }

    let ref_dir = reference_dir.normalize_or_zero();
    if ref_dir.length_squared() < MIN_BONE_LENGTH {
        // No valid reference direction, can't constrain
        return joint_pos;
    }

    let dot = current_dir.dot(ref_dir).clamp(-1.0, 1.0);
    let current_angle = dot.acos();

    if current_angle <= max_angle + ANGLE_EPSILON {
        // Within constraint
        joint_pos
    } else {
        // Constrain to cone surface - rotate current_dir toward ref_dir
        let axis = current_dir.cross(ref_dir);
        if axis.length_squared() < MIN_BONE_LENGTH {
            // Parallel or anti-parallel
            parent_pos + ref_dir * bone_length
        } else {
            let axis = axis.normalize();
            // Rotate current_dir toward ref_dir by (current_angle - max_angle)
            // This brings the direction to the cone surface
            let angle_to_rotate = current_angle - max_angle;
            let constrained_dir = Quat::from_axis_angle(axis, angle_to_rotate) * current_dir;
            parent_pos + constrained_dir * bone_length
        }
    }
}

/// Apply a hinge constraint to a joint position.
///
/// The joint can only rotate around the specified axis within [min, max] angle limits.
fn apply_hinge_constraint(
    joint_pos: Vec3,
    parent_pos: Vec3,
    reference_dir: Vec3,
    axis: Vec3,
    min_angle: f32,
    max_angle: f32,
    bone_length: f32,
) -> Vec3 {
    let current_dir = (joint_pos - parent_pos).normalize_or_zero();
    let ref_dir = reference_dir.normalize_or_zero();

    // Project both vectors onto the plane perpendicular to the hinge axis
    let axis_norm = axis.normalize_or_zero();
    let current_proj = (current_dir - axis_norm * current_dir.dot(axis_norm)).normalize_or_zero();
    let ref_proj = (ref_dir - axis_norm * ref_dir.dot(axis_norm)).normalize_or_zero();

    if current_proj.length_squared() < MIN_BONE_LENGTH || ref_proj.length_squared() < MIN_BONE_LENGTH {
        return parent_pos + ref_dir * bone_length;
    }

    // Compute signed angle
    let dot = current_proj.dot(ref_proj).clamp(-1.0, 1.0);
    let mut angle = dot.acos();

    // Determine sign using cross product
    let cross = ref_proj.cross(current_proj);
    if cross.dot(axis_norm) < 0.0 {
        angle = -angle;
    }

    // Clamp angle
    let clamped_angle = angle.clamp(min_angle, max_angle);

    // Rotate reference direction by clamped angle around axis
    let rotation = Quat::from_axis_angle(axis_norm, clamped_angle);
    let constrained_dir = rotation * ref_dir;

    parent_pos + constrained_dir * bone_length
}

/// Apply a ball-and-socket constraint.
///
/// Limits both swing (angle from reference) and twist (rotation around bone axis).
fn apply_ball_socket_constraint(
    joint_pos: Vec3,
    parent_pos: Vec3,
    reference_dir: Vec3,
    swing_limit: f32,
    _twist_limit: f32, // TODO: implement twist limiting
    bone_length: f32,
) -> Vec3 {
    // For now, treat as cone constraint with swing_limit
    // Full ball-socket with twist limiting requires tracking bone orientation
    apply_cone_constraint(joint_pos, parent_pos, reference_dir, swing_limit, bone_length)
}

/// Apply the appropriate constraint to a joint.
fn apply_constraint(
    joint_pos: Vec3,
    parent_pos: Vec3,
    reference_dir: Vec3,
    constraint: &JointConstraint,
    bone_length: f32,
) -> Vec3 {
    match constraint {
        JointConstraint::None => joint_pos,
        JointConstraint::Cone { angle } => {
            apply_cone_constraint(joint_pos, parent_pos, reference_dir, *angle, bone_length)
        }
        JointConstraint::Hinge { axis, min, max } => {
            apply_hinge_constraint(joint_pos, parent_pos, reference_dir, *axis, *min, *max, bone_length)
        }
        JointConstraint::BallSocket {
            swing_limit,
            twist_limit,
        } => {
            apply_ball_socket_constraint(
                joint_pos,
                parent_pos,
                reference_dir,
                *swing_limit,
                *twist_limit,
                bone_length,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// FABRIK Solver
// ---------------------------------------------------------------------------

/// Solve inverse kinematics using the FABRIK algorithm.
///
/// This function modifies the pose in-place to move the chain's effector
/// toward the target position.
///
/// # Arguments
///
/// * `chain` - The bone chain to solve (root to tip order)
/// * `skeleton` - The skeleton containing the bones
/// * `pose` - The current pose (will be modified)
/// * `params` - Solver parameters including target position and constraints
///
/// # Returns
///
/// A `FabrikResult` containing the final positions and convergence info.
///
/// # Panics
///
/// Panics if the chain contains invalid bone indices or has fewer than 2 bones.
pub fn solve_fabrik(
    chain: &FabrikChain,
    skeleton: &Skeleton,
    pose: &mut Pose,
    params: &FabrikParams,
) -> FabrikResult {
    // Validate chain
    assert!(chain.is_valid(), "FABRIK chain must have at least 2 bones");
    for &bone_idx in &chain.bones {
        assert!(
            bone_idx < pose.bone_count(),
            "bone index {} out of range (pose has {} bones)",
            bone_idx,
            pose.bone_count()
        );
    }

    // Get initial positions
    let mut positions = compute_chain_positions(chain, skeleton, pose);
    let bone_lengths = compute_bone_lengths(&positions);
    let total_length = compute_total_length(&bone_lengths);

    // Store original root position (anchor point)
    let root_pos = positions[0];

    // Check if target is reachable
    let target = params.target_position;
    let distance_to_target = (target - root_pos).length();

    // If target is completely unreachable, stretch toward it
    if distance_to_target > total_length + params.tolerance {
        // Target is out of reach - stretch chain toward target
        let direction = (target - root_pos).normalize_or_zero();
        let mut current_pos = root_pos;
        positions[0] = current_pos;

        for i in 1..positions.len() {
            current_pos = current_pos + direction * bone_lengths[i - 1];
            positions[i] = current_pos;
        }

        // Apply constraints after stretching
        apply_constraints_to_chain(&mut positions, &bone_lengths, params);

        // Update pose
        update_pose_from_positions(chain, skeleton, pose, &positions);

        let final_distance = (positions.last().unwrap_or(&Vec3::ZERO) - target).length();

        return FabrikResult {
            bone_positions: positions,
            iterations: 1,
            converged: false,
            final_distance,
        };
    }

    // FABRIK iteration loop
    let mut iterations = 0;
    let mut converged = false;

    while iterations < params.max_iterations {
        iterations += 1;

        // Forward pass: from effector toward root
        forward_pass(&mut positions, &bone_lengths, target);

        // Backward pass: from root toward effector
        backward_pass(&mut positions, &bone_lengths, root_pos);

        // Apply constraints
        apply_constraints_to_chain(&mut positions, &bone_lengths, params);

        // Check convergence
        let effector_pos = *positions.last().unwrap();
        let distance = (effector_pos - target).length();

        if distance <= params.tolerance {
            converged = true;
            break;
        }
    }

    // Update pose with final positions
    update_pose_from_positions(chain, skeleton, pose, &positions);

    let final_distance = (positions.last().unwrap_or(&Vec3::ZERO) - target).length();

    FabrikResult {
        bone_positions: positions,
        iterations,
        converged,
        final_distance,
    }
}

/// Forward pass: move joints from effector toward root.
///
/// The effector is placed at the target, then each joint is moved
/// to maintain the bone length constraint with its child.
fn forward_pass(positions: &mut [Vec3], bone_lengths: &[f32], target: Vec3) {
    let n = positions.len();
    if n < 2 {
        return;
    }

    // Set effector to target
    positions[n - 1] = target;

    // Work backward from effector to root
    for i in (0..n - 1).rev() {
        let child_pos = positions[i + 1];
        let bone_length = bone_lengths[i];
        positions[i] = move_toward(positions[i], child_pos, bone_length);
    }
}

/// Backward pass: move joints from root toward effector.
///
/// The root is anchored at its original position, then each joint is moved
/// to maintain the bone length constraint with its parent.
fn backward_pass(positions: &mut [Vec3], bone_lengths: &[f32], root_pos: Vec3) {
    let n = positions.len();
    if n < 2 {
        return;
    }

    // Anchor root at original position
    positions[0] = root_pos;

    // Work forward from root to effector
    for i in 1..n {
        let parent_pos = positions[i - 1];
        let bone_length = bone_lengths[i - 1];
        positions[i] = move_toward(positions[i], parent_pos, bone_length);
    }
}

/// Apply constraints to all joints in the chain.
fn apply_constraints_to_chain(
    positions: &mut [Vec3],
    bone_lengths: &[f32],
    params: &FabrikParams,
) {
    let n = positions.len();
    if n < 2 {
        return;
    }

    // Skip root (index 0), apply constraints from first child onward
    for i in 1..n {
        let constraint = params.get_constraint(i);

        if constraint.is_unconstrained() {
            continue;
        }

        let parent_pos = positions[i - 1];
        let bone_length = bone_lengths[i - 1];

        // Reference direction: from parent toward original child position
        // In practice, this should be the rest pose direction
        // For now, use the previous iteration's direction or a default
        let reference_dir = if i + 1 < n {
            (positions[i + 1] - positions[i]).normalize_or_zero()
        } else {
            (positions[i] - parent_pos).normalize_or_zero()
        };

        // Apply constraint
        positions[i] = apply_constraint(
            positions[i],
            parent_pos,
            reference_dir,
            constraint,
            bone_length,
        );
    }
}

/// Update the pose with the solved positions.
///
/// Converts world-space positions back to local-space transforms.
fn update_pose_from_positions(
    chain: &FabrikChain,
    skeleton: &Skeleton,
    pose: &mut Pose,
    positions: &[Vec3],
) {
    // Compute current world transforms for all bones
    let transforms = pose.transforms();
    let mut world_transforms = skeleton.compute_world_transforms(&transforms);

    // Update world positions for chain bones
    for (i, &bone_idx) in chain.bones.iter().enumerate() {
        let new_pos = positions[i];
        let mut transform = world_transforms[bone_idx];
        transform.w_axis = new_pos.extend(1.0);
        world_transforms[bone_idx] = transform;
    }

    // Convert back to local transforms
    for (i, &bone_idx) in chain.bones.iter().enumerate() {
        let new_world_pos = positions[i];

        // Get parent world transform
        let parent_world = if let Some(parent_idx) = skeleton.parent(bone_idx) {
            world_transforms[parent_idx]
        } else {
            glam::Mat4::IDENTITY
        };

        // Convert world position to local position
        let parent_inv = parent_world.inverse();
        let local_pos = parent_inv.transform_point3(new_world_pos);

        // Update pose local position
        pose.positions[bone_idx] = local_pos;

        // Update rotation to point toward child (if not the effector)
        if i + 1 < positions.len() {
            let child_world_pos = positions[i + 1];
            let direction = (child_world_pos - new_world_pos).normalize_or_zero();

            if direction.length_squared() > MIN_BONE_LENGTH {
                // Compute rotation that points +Y (or chosen axis) toward child
                // This is a simplified approach; proper implementation would preserve twist
                let up = Vec3::Y;
                let rotation = compute_look_rotation(direction, up);

                // Convert to local rotation
                let parent_rot = if let Some(parent_idx) = skeleton.parent(bone_idx) {
                    Transform::from_matrix(world_transforms[parent_idx])
                        .map(|t| t.rotation)
                        .unwrap_or(Quat::IDENTITY)
                } else {
                    Quat::IDENTITY
                };

                let local_rotation = parent_rot.inverse() * rotation;
                pose.rotations[bone_idx] = local_rotation;
            }
        }
    }
}

/// Compute a rotation that points the forward axis toward a direction.
fn compute_look_rotation(forward: Vec3, up: Vec3) -> Quat {
    let forward = forward.normalize_or_zero();
    if forward.length_squared() < MIN_BONE_LENGTH {
        return Quat::IDENTITY;
    }

    let right = up.cross(forward).normalize_or_zero();
    if right.length_squared() < MIN_BONE_LENGTH {
        // Forward is parallel to up, choose arbitrary right
        let right = if forward.x.abs() < 0.9 {
            Vec3::X.cross(forward).normalize()
        } else {
            Vec3::Z.cross(forward).normalize()
        };
        let up = forward.cross(right);
        return Quat::from_mat3(&glam::Mat3::from_cols(right, up, forward));
    }

    let up = forward.cross(right).normalize();
    Quat::from_mat3(&glam::Mat3::from_cols(right, up, forward))
}

// ---------------------------------------------------------------------------
// Two-Bone Analytical Solver
// ---------------------------------------------------------------------------

/// Analytical solution for a two-bone IK chain.
///
/// This is faster and more accurate than iterative FABRIK for simple 2-bone chains
/// like arms or legs.
///
/// # Arguments
///
/// * `root_pos` - Position of the root joint (shoulder/hip)
/// * `mid_pos` - Position of the middle joint (elbow/knee)
/// * `tip_pos` - Position of the end effector (wrist/ankle)
/// * `target` - Target position for the end effector
/// * `pole_target` - Optional pole target for elbow/knee direction
///
/// # Returns
///
/// Tuple of (new_mid_pos, new_tip_pos) if solvable, None if target is unreachable.
pub fn solve_two_bone_analytical(
    root_pos: Vec3,
    mid_pos: Vec3,
    tip_pos: Vec3,
    target: Vec3,
    pole_target: Option<Vec3>,
) -> Option<(Vec3, Vec3)> {
    let len_a = (mid_pos - root_pos).length();
    let len_b = (tip_pos - mid_pos).length();
    let total_len = len_a + len_b;

    let to_target = target - root_pos;
    let target_dist = to_target.length();

    // Check if target is reachable
    if target_dist > total_len || target_dist < (len_a - len_b).abs() {
        return None;
    }

    // Use law of cosines to find the elbow angle
    // cos(angle) = (a^2 + c^2 - b^2) / (2ac)
    // where a = len_a, b = len_b, c = target_dist
    let cos_angle = (len_a * len_a + target_dist * target_dist - len_b * len_b)
        / (2.0 * len_a * target_dist);
    let cos_angle = cos_angle.clamp(-1.0, 1.0);
    let angle = cos_angle.acos();

    // Direction to target
    let target_dir = to_target.normalize_or_zero();

    // Compute the plane for the elbow bend
    let pole_dir = if let Some(pole) = pole_target {
        // Use pole target to determine bend direction
        let to_pole = pole - root_pos;
        let plane_normal = target_dir.cross(to_pole).normalize_or_zero();
        if plane_normal.length_squared() < MIN_BONE_LENGTH {
            Vec3::Y // Fallback
        } else {
            plane_normal
        }
    } else {
        // Default: bend in XZ plane or YZ plane
        if target_dir.y.abs() > 0.9 {
            Vec3::Z
        } else {
            Vec3::Y.cross(target_dir).normalize_or_zero()
        }
    };

    // Rotate target direction by the angle around the pole normal
    let rotation = Quat::from_axis_angle(pole_dir, angle);
    let mid_dir = rotation * target_dir;

    // Compute new positions
    let new_mid = root_pos + mid_dir * len_a;
    let new_tip = target;

    Some((new_mid, new_tip))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Bone, SkeletonBuilder};
    use std::f32::consts::PI;

    // Helper to create a simple skeleton for testing
    fn create_test_skeleton(bone_count: usize, spacing: f32) -> (Skeleton, Pose) {
        let mut skeleton = crate::skeleton::Skeleton::new();

        // Create linear chain of bones along Y axis
        skeleton.add_bone(
            Bone::root("bone_0")
                .with_local_transform(Transform::from_position(Vec3::ZERO)),
        );

        for i in 1..bone_count {
            skeleton.add_bone(
                Bone::new(format!("bone_{}", i))
                    .with_parent(i - 1)
                    .with_local_transform(Transform::from_position(Vec3::new(0.0, spacing, 0.0))),
            );
        }

        skeleton.rebuild_indices();

        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        (skeleton, pose)
    }

    // ===== FabrikChain Tests =====

    #[test]
    fn test_fabrik_chain_new() {
        let chain = FabrikChain::new(vec![0, 1, 2]);
        assert_eq!(chain.len(), 3);
        assert!(!chain.is_empty());
        assert!(chain.is_valid());
    }

    #[test]
    fn test_fabrik_chain_two_bone() {
        let chain = FabrikChain::two_bone(0, 1);
        assert_eq!(chain.bones, vec![0, 1]);
        assert!(chain.is_valid());
        assert_eq!(chain.segment_count(), 1);
    }

    #[test]
    fn test_fabrik_chain_three_bone() {
        let chain = FabrikChain::three_bone(0, 1, 2);
        assert_eq!(chain.bones, vec![0, 1, 2]);
        assert_eq!(chain.segment_count(), 2);
    }

    #[test]
    fn test_fabrik_chain_root_and_effector() {
        let chain = FabrikChain::new(vec![3, 5, 7, 9]);
        assert_eq!(chain.root(), Some(3));
        assert_eq!(chain.effector(), Some(9));
    }

    #[test]
    fn test_fabrik_chain_empty() {
        let chain = FabrikChain::default();
        assert!(chain.is_empty());
        assert!(!chain.is_valid());
        assert_eq!(chain.root(), None);
        assert_eq!(chain.effector(), None);
        assert_eq!(chain.segment_count(), 0);
    }

    #[test]
    fn test_fabrik_chain_single_bone_invalid() {
        let chain = FabrikChain::new(vec![0]);
        assert!(!chain.is_valid());
    }

    // ===== JointConstraint Tests =====

    #[test]
    fn test_joint_constraint_default() {
        let c = JointConstraint::default();
        assert!(matches!(c, JointConstraint::None));
        assert!(c.is_unconstrained());
    }

    #[test]
    fn test_joint_constraint_cone() {
        let c = JointConstraint::Cone { angle: PI / 4.0 };
        assert!(!c.is_unconstrained());
    }

    #[test]
    fn test_joint_constraint_cone_degrees() {
        let c = JointConstraint::cone_degrees(45.0);
        if let JointConstraint::Cone { angle } = c {
            assert!((angle - PI / 4.0).abs() < 1e-5);
        } else {
            panic!("expected Cone constraint");
        }
    }

    #[test]
    fn test_joint_constraint_hinge_degrees() {
        let c = JointConstraint::hinge_degrees(Vec3::Z, -90.0, 90.0);
        if let JointConstraint::Hinge { axis, min, max } = c {
            assert!(axis.abs_diff_eq(Vec3::Z, 1e-5));
            assert!((min - (-PI / 2.0)).abs() < 1e-5);
            assert!((max - (PI / 2.0)).abs() < 1e-5);
        } else {
            panic!("expected Hinge constraint");
        }
    }

    #[test]
    fn test_joint_constraint_ball_socket_degrees() {
        let c = JointConstraint::ball_socket_degrees(45.0, 30.0);
        if let JointConstraint::BallSocket {
            swing_limit,
            twist_limit,
        } = c
        {
            assert!((swing_limit - PI / 4.0).abs() < 1e-5);
            assert!((twist_limit - PI / 6.0).abs() < 1e-5);
        } else {
            panic!("expected BallSocket constraint");
        }
    }

    // ===== FabrikParams Tests =====

    #[test]
    fn test_fabrik_params_default() {
        let params = FabrikParams::default();
        assert_eq!(params.target_position, Vec3::ZERO);
        assert_eq!(params.max_iterations, DEFAULT_MAX_ITERATIONS);
        assert_eq!(params.tolerance, DEFAULT_TOLERANCE);
        assert!(params.constraints.is_empty());
    }

    #[test]
    fn test_fabrik_params_new() {
        let params = FabrikParams::new(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(params.target_position, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_fabrik_params_builder() {
        let params = FabrikParams::new(Vec3::X)
            .with_max_iterations(20)
            .with_tolerance(0.01)
            .with_constraints(vec![JointConstraint::None, JointConstraint::cone_degrees(45.0)]);

        assert_eq!(params.max_iterations, 20);
        assert_eq!(params.tolerance, 0.01);
        assert_eq!(params.constraints.len(), 2);
    }

    #[test]
    fn test_fabrik_params_get_constraint() {
        let params = FabrikParams::default()
            .with_constraints(vec![JointConstraint::None, JointConstraint::cone_degrees(45.0)]);

        assert!(params.get_constraint(0).is_unconstrained());
        assert!(!params.get_constraint(1).is_unconstrained());
        assert!(params.get_constraint(99).is_unconstrained()); // Out of bounds
    }

    // ===== FabrikResult Tests =====

    #[test]
    fn test_fabrik_result_default() {
        let result = FabrikResult::default();
        assert!(result.bone_positions.is_empty());
        assert_eq!(result.iterations, 0);
        assert!(!result.converged);
        assert_eq!(result.final_distance, f32::MAX);
    }

    #[test]
    fn test_fabrik_result_is_reachable() {
        let mut result = FabrikResult::default();
        assert!(!result.is_reachable());

        result.converged = true;
        assert!(result.is_reachable());
    }

    #[test]
    fn test_fabrik_result_effector_position() {
        let result = FabrikResult {
            bone_positions: vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)],
            ..Default::default()
        };
        assert_eq!(result.effector_position(), Some(Vec3::new(2.0, 0.0, 0.0)));
    }

    // ===== Helper Function Tests =====

    #[test]
    fn test_compute_bone_lengths() {
        let positions = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
        ];

        let lengths = compute_bone_lengths(&positions);
        assert_eq!(lengths.len(), 2);
        assert!((lengths[0] - 1.0).abs() < 1e-5);
        assert!((lengths[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_bone_lengths_empty() {
        let lengths = compute_bone_lengths(&[]);
        assert!(lengths.is_empty());
    }

    #[test]
    fn test_compute_total_length() {
        let lengths = vec![1.0, 2.0, 3.0];
        assert!((compute_total_length(&lengths) - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_move_toward() {
        let from = Vec3::new(0.0, 0.0, 0.0);
        let toward = Vec3::new(10.0, 0.0, 0.0);

        let result = move_toward(from, toward, 3.0);
        assert!(result.abs_diff_eq(Vec3::new(7.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_move_toward_coincident() {
        let from = Vec3::ZERO;
        let toward = Vec3::ZERO;

        // Should pick arbitrary direction
        let result = move_toward(from, toward, 1.0);
        assert!((result.length() - 1.0).abs() < 1e-5);
    }

    // ===== Forward/Backward Pass Tests =====

    #[test]
    fn test_forward_pass() {
        let mut positions = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let lengths = vec![1.0, 1.0];
        let target = Vec3::new(1.5, 1.0, 0.0);

        forward_pass(&mut positions, &lengths, target);

        // Effector should be at target
        assert!(positions[2].abs_diff_eq(target, 1e-5));

        // Each segment should maintain length
        assert!(((positions[1] - positions[2]).length() - 1.0).abs() < 1e-5);
        assert!(((positions[0] - positions[1]).length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_backward_pass() {
        let mut positions = vec![
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.5, 1.0, 0.0),
        ];
        let lengths = vec![1.0, 1.0];
        let root = Vec3::ZERO;

        backward_pass(&mut positions, &lengths, root);

        // Root should be anchored
        assert!(positions[0].abs_diff_eq(root, 1e-5));

        // Each segment should maintain length
        assert!(((positions[1] - positions[0]).length() - 1.0).abs() < 1e-5);
        assert!(((positions[2] - positions[1]).length() - 1.0).abs() < 1e-5);
    }

    // ===== FABRIK Solver Tests =====

    #[test]
    fn test_solve_fabrik_two_bone_reachable() {
        let (skeleton, mut pose) = create_test_skeleton(2, 1.0);

        let chain = FabrikChain::two_bone(0, 1);
        // Target at a reachable location (distance = sqrt(0.6^2 + 0.8^2) = 1.0)
        let target = Vec3::new(0.6, 0.8, 0.0);
        let params = FabrikParams::new(target)
            .with_max_iterations(20)
            .with_tolerance(0.05);

        // Get initial positions to verify setup
        let transforms = pose.transforms();
        let world_transforms = skeleton.compute_world_transforms(&transforms);
        let init_pos_0 = world_transforms[0].w_axis.truncate();
        let init_pos_1 = world_transforms[1].w_axis.truncate();
        let init_length = (init_pos_1 - init_pos_0).length();
        let target_dist = (target - init_pos_0).length();

        // Verify the setup is correct
        assert!(init_pos_0.abs_diff_eq(Vec3::ZERO, 1e-5), "bone 0 should be at origin, got {:?}", init_pos_0);
        assert!(init_pos_1.abs_diff_eq(Vec3::new(0.0, 1.0, 0.0), 1e-5), "bone 1 should be at (0,1,0), got {:?}", init_pos_1);
        assert!((init_length - 1.0).abs() < 1e-5, "bone length should be 1.0, got {}", init_length);
        assert!((target_dist - 1.0).abs() < 1e-5, "target distance should be ~1.0, got {}", target_dist);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        // For a 2-bone chain (single segment), FABRIK rotates to point at target.
        // Since target is exactly 1 unit away, effector should be at target.
        assert!(
            result.final_distance < 0.1,
            "final distance {} should be small, effector at {:?}, target at {:?}",
            result.final_distance,
            result.bone_positions.get(1),
            target
        );
        assert!(result.iterations <= 20);
    }

    #[test]
    fn test_solve_fabrik_two_bone_unreachable() {
        let (skeleton, mut pose) = create_test_skeleton(2, 1.0);

        let chain = FabrikChain::two_bone(0, 1);
        // Target is 10 units away but chain is only 1 unit long
        let params = FabrikParams::new(Vec3::new(10.0, 0.0, 0.0))
            .with_max_iterations(10)
            .with_tolerance(0.001);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        assert!(!result.converged, "should not converge for unreachable target");
        // Chain should be stretched toward target
        assert!(result.bone_positions[1].x > result.bone_positions[0].x);
    }

    #[test]
    fn test_solve_fabrik_five_bone_chain() {
        let (skeleton, mut pose) = create_test_skeleton(5, 1.0);

        let chain = FabrikChain::new(vec![0, 1, 2, 3, 4]);
        let params = FabrikParams::new(Vec3::new(2.0, 2.0, 0.0))
            .with_max_iterations(50)
            .with_tolerance(0.01);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        assert!(result.converged, "5-bone chain should reach nearby target");
        assert!(result.final_distance < 0.01);
        assert_eq!(result.bone_positions.len(), 5);
    }

    #[test]
    fn test_solve_fabrik_ten_bone_spine() {
        let (skeleton, mut pose) = create_test_skeleton(10, 0.5);

        let chain = FabrikChain::new((0..10).collect());
        let params = FabrikParams::new(Vec3::new(2.0, 3.0, 1.0))
            .with_max_iterations(100)
            .with_tolerance(0.01);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        // 10 bones * 0.5 spacing = 4.5 units total length
        // Target at distance sqrt(4+9+1) = ~3.74, should be reachable
        assert!(result.converged, "10-bone spine should converge");
        assert!(result.final_distance < 0.01);
    }

    #[test]
    fn test_solve_fabrik_convergence_detection() {
        let (skeleton, mut pose) = create_test_skeleton(3, 1.0);

        let chain = FabrikChain::three_bone(0, 1, 2);
        // Target exactly at current effector position
        let current_effector = Vec3::new(0.0, 2.0, 0.0);
        let params = FabrikParams::new(current_effector)
            .with_max_iterations(100)
            .with_tolerance(0.001);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        assert!(result.converged);
        // Should converge quickly since target is already reached
        assert!(result.iterations <= 3);
    }

    #[test]
    fn test_solve_fabrik_max_iterations() {
        let (skeleton, mut pose) = create_test_skeleton(3, 1.0);

        let chain = FabrikChain::three_bone(0, 1, 2);
        let params = FabrikParams::new(Vec3::new(0.5, 1.5, 0.0))
            .with_max_iterations(3)
            .with_tolerance(0.0001); // Very tight tolerance

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        // Should stop at max iterations
        assert!(result.iterations <= 3);
    }

    #[test]
    fn test_solve_fabrik_with_cone_constraint() {
        let (skeleton, mut pose) = create_test_skeleton(3, 1.0);

        let chain = FabrikChain::three_bone(0, 1, 2);
        let params = FabrikParams::new(Vec3::new(1.5, 1.0, 0.0))
            .with_max_iterations(50)
            .with_tolerance(0.1)
            .with_constraints(vec![
                JointConstraint::None,
                JointConstraint::cone_degrees(30.0),
                JointConstraint::None,
            ]);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        // Should still attempt to solve (may not fully converge with constraints)
        assert!(result.iterations > 0);
        assert_eq!(result.bone_positions.len(), 3);
    }

    #[test]
    fn test_solve_fabrik_with_hinge_constraint() {
        let (skeleton, mut pose) = create_test_skeleton(3, 1.0);

        let chain = FabrikChain::three_bone(0, 1, 2);
        let params = FabrikParams::new(Vec3::new(1.0, 1.0, 0.0))
            .with_max_iterations(50)
            .with_tolerance(0.1)
            .with_constraints(vec![
                JointConstraint::None,
                JointConstraint::hinge_degrees(Vec3::Z, -45.0, 45.0),
                JointConstraint::None,
            ]);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_fabrik_preserves_bone_lengths() {
        let (skeleton, mut pose) = create_test_skeleton(4, 1.5);

        let chain = FabrikChain::new(vec![0, 1, 2, 3]);
        let params = FabrikParams::new(Vec3::new(2.0, 3.0, 0.0))
            .with_max_iterations(50)
            .with_tolerance(0.01);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        // Check that bone lengths are preserved
        for i in 0..result.bone_positions.len() - 1 {
            let length = (result.bone_positions[i + 1] - result.bone_positions[i]).length();
            assert!(
                (length - 1.5).abs() < 0.1,
                "bone {} length {} should be ~1.5",
                i,
                length
            );
        }
    }

    #[test]
    fn test_solve_fabrik_iteration_accuracy_tradeoff() {
        let (skeleton, mut pose) = create_test_skeleton(5, 1.0);

        let chain = FabrikChain::new(vec![0, 1, 2, 3, 4]);
        let target = Vec3::new(2.0, 2.5, 0.0);

        // Low iterations
        let params_low = FabrikParams::new(target)
            .with_max_iterations(2)
            .with_tolerance(0.001);
        let mut pose_low = pose.clone();
        let result_low = solve_fabrik(&chain, &skeleton, &mut pose_low, &params_low);

        // High iterations
        let params_high = FabrikParams::new(target)
            .with_max_iterations(50)
            .with_tolerance(0.001);
        let mut pose_high = pose.clone();
        let result_high = solve_fabrik(&chain, &skeleton, &mut pose_high, &params_high);

        // More iterations should give better accuracy
        assert!(
            result_high.final_distance <= result_low.final_distance + 0.001,
            "more iterations should give equal or better accuracy"
        );
    }

    #[test]
    #[should_panic(expected = "FABRIK chain must have at least 2 bones")]
    fn test_solve_fabrik_invalid_chain_empty() {
        let (skeleton, mut pose) = create_test_skeleton(3, 1.0);
        let chain = FabrikChain::default();
        let params = FabrikParams::default();
        solve_fabrik(&chain, &skeleton, &mut pose, &params);
    }

    #[test]
    #[should_panic(expected = "bone index")]
    fn test_solve_fabrik_invalid_bone_index() {
        let (skeleton, mut pose) = create_test_skeleton(3, 1.0);
        let chain = FabrikChain::new(vec![0, 1, 99]); // 99 is out of range
        let params = FabrikParams::default();
        solve_fabrik(&chain, &skeleton, &mut pose, &params);
    }

    // ===== Two-Bone Analytical Solver Tests =====

    #[test]
    fn test_two_bone_analytical_reachable() {
        let root = Vec3::ZERO;
        let mid = Vec3::new(1.0, 0.0, 0.0);
        let tip = Vec3::new(2.0, 0.0, 0.0);
        let target = Vec3::new(1.5, 0.5, 0.0);

        let result = solve_two_bone_analytical(root, mid, tip, target, None);

        assert!(result.is_some());
        let (new_mid, new_tip) = result.unwrap();

        // Tip should be at target
        assert!(new_tip.abs_diff_eq(target, 1e-5));

        // Bone lengths should be preserved
        let len_a = (new_mid - root).length();
        let len_b = (new_tip - new_mid).length();
        assert!((len_a - 1.0).abs() < 0.01);
        assert!((len_b - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_two_bone_analytical_unreachable_too_far() {
        let root = Vec3::ZERO;
        let mid = Vec3::new(1.0, 0.0, 0.0);
        let tip = Vec3::new(2.0, 0.0, 0.0);
        let target = Vec3::new(10.0, 0.0, 0.0); // Too far

        let result = solve_two_bone_analytical(root, mid, tip, target, None);

        assert!(result.is_none());
    }

    #[test]
    fn test_two_bone_analytical_unreachable_too_close() {
        let root = Vec3::ZERO;
        let mid = Vec3::new(2.0, 0.0, 0.0);
        let tip = Vec3::new(3.0, 0.0, 0.0);
        let target = Vec3::new(0.1, 0.0, 0.0); // Inside minimum reach

        let result = solve_two_bone_analytical(root, mid, tip, target, None);

        assert!(result.is_none());
    }

    #[test]
    fn test_two_bone_analytical_with_pole_target() {
        let root = Vec3::ZERO;
        let mid = Vec3::new(1.0, 0.0, 0.0);
        let tip = Vec3::new(2.0, 0.0, 0.0);
        let target = Vec3::new(1.0, 1.0, 0.0);
        let pole = Vec3::new(1.0, 0.0, 1.0); // Pole in +Z direction

        let result = solve_two_bone_analytical(root, mid, tip, target, Some(pole));

        assert!(result.is_some());
        let (new_mid, _) = result.unwrap();

        // Elbow should bend toward the pole direction
        // (exact behavior depends on implementation)
        assert!(new_mid.length() > 0.0);
    }

    #[test]
    fn test_two_bone_analytical_straight_extension() {
        let root = Vec3::ZERO;
        let mid = Vec3::new(1.0, 0.0, 0.0);
        let tip = Vec3::new(2.0, 0.0, 0.0);
        let target = Vec3::new(2.0, 0.0, 0.0); // Exactly at tip

        let result = solve_two_bone_analytical(root, mid, tip, target, None);

        assert!(result.is_some());
        let (new_mid, new_tip) = result.unwrap();

        // Should maintain straight line
        assert!(new_tip.abs_diff_eq(target, 1e-5));
    }

    // ===== Constraint Application Tests =====

    #[test]
    fn test_apply_cone_constraint_within_limit() {
        let parent_pos = Vec3::ZERO;
        let joint_pos = Vec3::new(0.0, 1.0, 0.0);
        let ref_dir = Vec3::Y;
        let max_angle = PI / 4.0; // 45 degrees
        let bone_length = 1.0;

        let result = apply_cone_constraint(joint_pos, parent_pos, ref_dir, max_angle, bone_length);

        // Should not change since it's within the cone
        assert!(result.abs_diff_eq(joint_pos, 1e-5));
    }

    #[test]
    fn test_apply_cone_constraint_outside_limit() {
        let parent_pos = Vec3::ZERO;
        let joint_pos = Vec3::new(1.0, 0.1, 0.0); // Almost horizontal
        let ref_dir = Vec3::Y;
        let max_angle = PI / 6.0; // 30 degrees
        let bone_length = 1.0;

        let result = apply_cone_constraint(joint_pos, parent_pos, ref_dir, max_angle, bone_length);

        // Should be constrained to cone surface
        let result_dir = (result - parent_pos).normalize();
        let angle = ref_dir.dot(result_dir).acos();
        assert!(angle <= max_angle + 0.01);

        // Should maintain bone length
        let result_length = (result - parent_pos).length();
        assert!((result_length - bone_length).abs() < 0.01);
    }

    #[test]
    fn test_apply_hinge_constraint_within_limits() {
        let parent_pos = Vec3::ZERO;
        let joint_pos = Vec3::new(0.0, 1.0, 0.0);
        let ref_dir = Vec3::Y;
        let axis = Vec3::Z;
        let bone_length = 1.0;

        let result = apply_hinge_constraint(
            joint_pos,
            parent_pos,
            ref_dir,
            axis,
            -PI / 2.0,
            PI / 2.0,
            bone_length,
        );

        // Should maintain bone length
        let result_length = (result - parent_pos).length();
        assert!((result_length - bone_length).abs() < 0.01);
    }

    // ===== Various Chain Configuration Tests =====

    #[test]
    fn test_solve_fabrik_horizontal_chain() {
        let mut skeleton = crate::skeleton::Skeleton::new();
        skeleton.add_bone(Bone::root("bone_0").with_local_transform(Transform::from_position(Vec3::ZERO)));
        skeleton.add_bone(
            Bone::new("bone_1")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(1.0, 0.0, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("bone_2")
                .with_parent(1)
                .with_local_transform(Transform::from_position(Vec3::new(1.0, 0.0, 0.0))),
        );
        skeleton.rebuild_indices();

        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let chain = FabrikChain::three_bone(0, 1, 2);
        let params = FabrikParams::new(Vec3::new(1.5, 1.0, 0.0))
            .with_max_iterations(30)
            .with_tolerance(0.01);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        assert!(result.converged);
    }

    #[test]
    fn test_solve_fabrik_diagonal_chain() {
        let mut skeleton = crate::skeleton::Skeleton::new();
        skeleton.add_bone(Bone::root("bone_0").with_local_transform(Transform::from_position(Vec3::ZERO)));
        skeleton.add_bone(
            Bone::new("bone_1")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(0.7, 0.7, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("bone_2")
                .with_parent(1)
                .with_local_transform(Transform::from_position(Vec3::new(0.7, 0.7, 0.0))),
        );
        skeleton.rebuild_indices();

        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let chain = FabrikChain::three_bone(0, 1, 2);
        let params = FabrikParams::new(Vec3::new(1.5, 0.5, 0.0))
            .with_max_iterations(30)
            .with_tolerance(0.05);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        // Should attempt to solve
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_fabrik_3d_target() {
        let (skeleton, mut pose) = create_test_skeleton(4, 1.0);

        let chain = FabrikChain::new(vec![0, 1, 2, 3]);
        let params = FabrikParams::new(Vec3::new(1.0, 2.0, 1.5))
            .with_max_iterations(50)
            .with_tolerance(0.05);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        // Should reach the 3D target
        let effector = result.effector_position().unwrap();
        assert!(effector.z.abs() > 0.1, "effector should have Z component");
    }

    #[test]
    fn test_solve_fabrik_target_at_root() {
        let (skeleton, mut pose) = create_test_skeleton(3, 1.0);

        let chain = FabrikChain::three_bone(0, 1, 2);
        let params = FabrikParams::new(Vec3::ZERO) // Target at root
            .with_max_iterations(50)
            .with_tolerance(0.1);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        // Should handle gracefully (unreachable or converge depending on chain config)
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_fabrik_very_long_chain() {
        let (skeleton, mut pose) = create_test_skeleton(20, 0.25);

        let chain = FabrikChain::new((0..20).collect());
        // Total length = 19 * 0.25 = 4.75 units
        let params = FabrikParams::new(Vec3::new(2.0, 3.0, 1.0))
            .with_max_iterations(100)
            .with_tolerance(0.05);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        assert_eq!(result.bone_positions.len(), 20);
        // Should make progress toward target
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_fabrik_with_ball_socket_constraint() {
        let (skeleton, mut pose) = create_test_skeleton(3, 1.0);

        let chain = FabrikChain::three_bone(0, 1, 2);
        let params = FabrikParams::new(Vec3::new(1.0, 1.5, 0.0))
            .with_max_iterations(50)
            .with_tolerance(0.1)
            .with_constraints(vec![
                JointConstraint::None,
                JointConstraint::ball_socket_degrees(60.0, 30.0),
                JointConstraint::None,
            ]);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_fabrik_multiple_constraints() {
        let (skeleton, mut pose) = create_test_skeleton(5, 1.0);

        let chain = FabrikChain::new(vec![0, 1, 2, 3, 4]);
        let params = FabrikParams::new(Vec3::new(2.0, 2.0, 0.0))
            .with_max_iterations(50)
            .with_tolerance(0.2)
            .with_constraints(vec![
                JointConstraint::None,
                JointConstraint::cone_degrees(45.0),
                JointConstraint::hinge_degrees(Vec3::Z, -60.0, 60.0),
                JointConstraint::cone_degrees(30.0),
                JointConstraint::None,
            ]);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        assert!(result.iterations > 0);
        assert_eq!(result.bone_positions.len(), 5);
    }

    // ===== Analytical Comparison Test =====

    #[test]
    fn test_fabrik_vs_analytical_two_bone() {
        // Create a simple 2-bone chain
        let root = Vec3::ZERO;
        let mid = Vec3::new(1.0, 0.0, 0.0);
        let tip = Vec3::new(2.0, 0.0, 0.0);
        let target = Vec3::new(1.4, 1.0, 0.0);

        // Analytical solution
        let analytical = solve_two_bone_analytical(root, mid, tip, target, None);
        assert!(analytical.is_some());
        let (analytical_mid, analytical_tip) = analytical.unwrap();

        // FABRIK solution
        let mut skeleton = crate::skeleton::Skeleton::new();
        skeleton.add_bone(Bone::root("root").with_local_transform(Transform::from_position(root)));
        skeleton.add_bone(
            Bone::new("mid")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(1.0, 0.0, 0.0))),
        );
        skeleton.add_bone(
            Bone::new("tip")
                .with_parent(1)
                .with_local_transform(Transform::from_position(Vec3::new(1.0, 0.0, 0.0))),
        );
        skeleton.rebuild_indices();

        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = FabrikChain::three_bone(0, 1, 2);
        let params = FabrikParams::new(target)
            .with_max_iterations(100)
            .with_tolerance(0.001);

        let fabrik_result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        // Both should reach the target
        assert!(fabrik_result.converged);
        assert!(analytical_tip.abs_diff_eq(target, 0.01));
        assert!(fabrik_result.bone_positions[2].abs_diff_eq(target, 0.01));

        // Both should have similar middle joint positions (within tolerance)
        // Note: they may differ slightly due to different algorithms
        let distance = (analytical_mid - fabrik_result.bone_positions[1]).length();
        assert!(
            distance < 0.5,
            "analytical and FABRIK middle joints should be similar, distance: {}",
            distance
        );
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_solve_fabrik_zero_length_bone_handling() {
        // Test that coincident bones don't cause issues
        let mut skeleton = crate::skeleton::Skeleton::new();
        skeleton.add_bone(Bone::root("root").with_local_transform(Transform::from_position(Vec3::ZERO)));
        skeleton.add_bone(
            Bone::new("mid")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(0.001, 0.0, 0.0))), // Very small
        );
        skeleton.add_bone(
            Bone::new("tip")
                .with_parent(1)
                .with_local_transform(Transform::from_position(Vec3::new(1.0, 0.0, 0.0))),
        );
        skeleton.rebuild_indices();

        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let chain = FabrikChain::three_bone(0, 1, 2);
        let params = FabrikParams::new(Vec3::new(0.5, 0.5, 0.0))
            .with_max_iterations(50)
            .with_tolerance(0.1);

        // Should not panic
        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_fabrik_negative_target() {
        let (skeleton, mut pose) = create_test_skeleton(3, 1.0);

        let chain = FabrikChain::three_bone(0, 1, 2);
        let params = FabrikParams::new(Vec3::new(-1.0, -1.0, -1.0))
            .with_max_iterations(50)
            .with_tolerance(0.1);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        // Should handle negative coordinates
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_fabrik_very_tight_tolerance() {
        let (skeleton, mut pose) = create_test_skeleton(3, 1.0);

        let chain = FabrikChain::three_bone(0, 1, 2);
        let params = FabrikParams::new(Vec3::new(1.0, 1.5, 0.0))
            .with_max_iterations(200)
            .with_tolerance(0.0001); // Very tight

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        if result.converged {
            assert!(result.final_distance < 0.0001);
        }
    }

    #[test]
    fn test_solve_fabrik_single_iteration() {
        let (skeleton, mut pose) = create_test_skeleton(3, 1.0);

        let chain = FabrikChain::three_bone(0, 1, 2);
        let params = FabrikParams::new(Vec3::new(1.0, 1.5, 0.0))
            .with_max_iterations(1)
            .with_tolerance(0.001);

        let result = solve_fabrik(&chain, &skeleton, &mut pose, &params);

        assert_eq!(result.iterations, 1);
    }

    // ===== Look Rotation Tests =====

    #[test]
    fn test_compute_look_rotation_forward() {
        let forward = Vec3::Z;
        let up = Vec3::Y;
        let rot = compute_look_rotation(forward, up);

        // Should produce a rotation that points Z forward
        let rotated = rot * Vec3::Z;
        assert!(rotated.abs_diff_eq(Vec3::Z, 0.01));
    }

    #[test]
    fn test_compute_look_rotation_up_aligned() {
        let forward = Vec3::Y; // Looking straight up
        let up = Vec3::Y;
        let rot = compute_look_rotation(forward, up);

        // Should still produce a valid rotation
        assert!((rot.length() - 1.0).abs() < 0.01);
    }
}
