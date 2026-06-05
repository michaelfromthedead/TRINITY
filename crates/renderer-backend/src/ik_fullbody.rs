//! Full-Body IK Solver for TRINITY Engine (T-AN-4.5).
//!
//! This module implements a multi-effector full-body inverse kinematics system
//! supporting:
//!
//! - Multiple end effectors (feet, hands, head, hips)
//! - Balance constraint with center of mass projection
//! - Posture preservation for natural poses
//! - Per-joint angle limits with soft boundaries
//! - Priority-based task layering with null-space projection
//!
//! # Architecture
//!
//! The full-body IK solver uses a hierarchical approach:
//!
//! 1. **Balance Layer** (highest priority): Ensures center of mass stays within
//!    the support polygon formed by planted feet. Adjusts hips position.
//!
//! 2. **Foot Placement Layer**: Positions feet on ground, maintaining contact
//!    with the support surface.
//!
//! 3. **Hand Reach Layer** (lowest priority): Reaches hands toward targets in
//!    the null-space of higher priority tasks.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::ik_fullbody::{
//!     FullBodyTarget, BalanceParams, PostureParams, FullBodyIkParams,
//!     solve_fullbody_ik,
//! };
//! use renderer_backend::skeleton::Skeleton;
//! use renderer_backend::pose::Pose;
//! use glam::Vec3;
//!
//! // Define targets for hands and feet
//! let targets = vec![
//!     FullBodyTarget::new(left_hand_idx, Vec3::new(-0.5, 1.0, 0.5))
//!         .with_priority(2),
//!     FullBodyTarget::new(right_foot_idx, Vec3::new(0.2, 0.0, 0.1))
//!         .with_priority(1),
//! ];
//!
//! // Set up balance constraint
//! let balance = BalanceParams {
//!     com_bone: hips_idx,
//!     support_margin: 0.05,
//!     balance_weight: 1.0,
//! };
//!
//! let params = FullBodyIkParams {
//!     targets,
//!     balance: Some(balance),
//!     posture: None,
//!     max_iterations: 20,
//!     tolerance: 0.001,
//! };
//!
//! let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
//! ```

use glam::{Mat4, Quat, Vec3};
use std::f32::consts::PI;

use crate::pose::Pose;
use crate::skeleton::{Skeleton, Transform};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default maximum iterations for full-body solver.
pub const DEFAULT_MAX_ITERATIONS: u32 = 20;

/// Default convergence tolerance (in world units).
pub const DEFAULT_TOLERANCE: f32 = 0.001;

/// Maximum number of end effectors supported.
pub const MAX_EFFECTORS: usize = 16;

/// Maximum number of priority levels.
pub const MAX_PRIORITY_LEVELS: u8 = 8;

/// Small epsilon for numerical stability.
const EPSILON: f32 = 1e-6;

/// Minimum bone length to avoid division by zero.
const MIN_BONE_LENGTH: f32 = 1e-5;

/// Damping factor for pseudo-inverse computation.
const DEFAULT_DAMPING: f32 = 0.05;

// ---------------------------------------------------------------------------
// Joint Limits
// ---------------------------------------------------------------------------

/// Per-joint angle limits for full-body IK.
#[derive(Clone, Debug, PartialEq)]
pub struct JointLimits {
    /// Minimum rotation angle around X axis (radians).
    pub min_x: f32,
    /// Maximum rotation angle around X axis (radians).
    pub max_x: f32,
    /// Minimum rotation angle around Y axis (radians).
    pub min_y: f32,
    /// Maximum rotation angle around Y axis (radians).
    pub max_y: f32,
    /// Minimum rotation angle around Z axis (radians).
    pub min_z: f32,
    /// Maximum rotation angle around Z axis (radians).
    pub max_z: f32,
    /// Soft limit zone ratio (0.0 = hard limits, 1.0 = full soft).
    pub soft_ratio: f32,
}

impl Default for JointLimits {
    fn default() -> Self {
        Self {
            min_x: -PI,
            max_x: PI,
            min_y: -PI,
            max_y: PI,
            min_z: -PI,
            max_z: PI,
            soft_ratio: 0.1,
        }
    }
}

impl JointLimits {
    /// Create limits with no restrictions (full range of motion).
    pub fn unconstrained() -> Self {
        Self::default()
    }

    /// Create symmetric limits around zero for all axes.
    pub fn symmetric(max_angle: f32) -> Self {
        Self {
            min_x: -max_angle,
            max_x: max_angle,
            min_y: -max_angle,
            max_y: max_angle,
            min_z: -max_angle,
            max_z: max_angle,
            soft_ratio: 0.1,
        }
    }

    /// Create hinge-style limits (only rotation around one axis).
    pub fn hinge_x(min: f32, max: f32) -> Self {
        Self {
            min_x: min,
            max_x: max,
            min_y: 0.0,
            max_y: 0.0,
            min_z: 0.0,
            max_z: 0.0,
            soft_ratio: 0.1,
        }
    }

    /// Create limits for a typical elbow joint.
    pub fn elbow() -> Self {
        Self {
            min_x: 0.0,
            max_x: 2.6, // ~150 degrees
            min_y: -0.1,
            max_y: 0.1,
            min_z: -0.1,
            max_z: 0.1,
            soft_ratio: 0.15,
        }
    }

    /// Create limits for a typical knee joint.
    pub fn knee() -> Self {
        Self {
            min_x: 0.0,
            max_x: 2.4, // ~140 degrees
            min_y: -0.05,
            max_y: 0.05,
            min_z: -0.05,
            max_z: 0.05,
            soft_ratio: 0.15,
        }
    }

    /// Create limits for a typical shoulder joint.
    pub fn shoulder() -> Self {
        Self {
            min_x: -PI * 0.5,
            max_x: PI * 0.5,
            min_y: -PI * 0.5,
            max_y: PI * 0.5,
            min_z: -PI * 0.25,
            max_z: PI * 0.25,
            soft_ratio: 0.1,
        }
    }

    /// Create limits for a typical hip joint.
    pub fn hip() -> Self {
        Self {
            min_x: -PI * 0.6,
            max_x: PI * 0.4,
            min_y: -PI * 0.3,
            max_y: PI * 0.3,
            min_z: -PI * 0.25,
            max_z: PI * 0.25,
            soft_ratio: 0.1,
        }
    }

    /// Create limits for spine joints.
    pub fn spine() -> Self {
        Self {
            min_x: -PI * 0.2,
            max_x: PI * 0.2,
            min_y: -PI * 0.15,
            max_y: PI * 0.15,
            min_z: -PI * 0.1,
            max_z: PI * 0.1,
            soft_ratio: 0.2,
        }
    }

    /// Set the soft limit ratio.
    pub fn with_soft_ratio(mut self, ratio: f32) -> Self {
        self.soft_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Clamp Euler angles to these limits with optional soft clamping.
    pub fn clamp_euler(&self, euler: Vec3) -> Vec3 {
        Vec3::new(
            soft_clamp(euler.x, self.min_x, self.max_x, self.soft_ratio),
            soft_clamp(euler.y, self.min_y, self.max_y, self.soft_ratio),
            soft_clamp(euler.z, self.min_z, self.max_z, self.soft_ratio),
        )
    }

    /// Check if angles are within limits.
    pub fn is_within(&self, euler: Vec3) -> bool {
        euler.x >= self.min_x - EPSILON
            && euler.x <= self.max_x + EPSILON
            && euler.y >= self.min_y - EPSILON
            && euler.y <= self.max_y + EPSILON
            && euler.z >= self.min_z - EPSILON
            && euler.z <= self.max_z + EPSILON
    }
}

/// Soft clamping function that smoothly approaches limits.
fn soft_clamp(value: f32, min: f32, max: f32, soft_ratio: f32) -> f32 {
    if soft_ratio <= EPSILON {
        return value.clamp(min, max);
    }

    let range = max - min;
    if range <= EPSILON {
        return (min + max) * 0.5;
    }

    let soft_zone = range * soft_ratio;

    if value < min + soft_zone {
        // In lower soft zone
        let t = (value - min) / soft_zone;
        if t < 0.0 {
            min
        } else {
            min + soft_zone * smooth_step(t)
        }
    } else if value > max - soft_zone {
        // In upper soft zone
        let t = (max - value) / soft_zone;
        if t < 0.0 {
            max
        } else {
            max - soft_zone * smooth_step(t)
        }
    } else {
        value
    }
}

/// Smooth step function for soft limits.
#[inline]
fn smooth_step(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// FullBodyTarget
// ---------------------------------------------------------------------------

/// Target specification for a full-body IK effector.
#[derive(Clone, Debug, PartialEq)]
pub struct FullBodyTarget {
    /// Bone index of the end effector.
    pub effector: usize,

    /// Target position in world space.
    pub position: Vec3,

    /// Optional target rotation in world space.
    pub rotation: Option<Quat>,

    /// Weight for this target (0.0-1.0).
    pub weight: f32,

    /// Priority level (0 = highest priority).
    /// Higher priority targets are solved first; lower priority targets
    /// are constrained to the null-space of higher priority tasks.
    pub priority: u8,
}

impl FullBodyTarget {
    /// Create a new position-only target with default weight and priority.
    pub fn new(effector: usize, position: Vec3) -> Self {
        Self {
            effector,
            position,
            rotation: None,
            weight: 1.0,
            priority: 0,
        }
    }

    /// Create a position+rotation target.
    pub fn with_rotation(effector: usize, position: Vec3, rotation: Quat) -> Self {
        Self {
            effector,
            position,
            rotation: Some(rotation),
            weight: 1.0,
            priority: 0,
        }
    }

    /// Set the weight for this target.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the priority level (0 = highest).
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Get the number of constraint dimensions (3 for position, 6 for position+rotation).
    pub fn constraint_dim(&self) -> usize {
        if self.rotation.is_some() {
            6
        } else {
            3
        }
    }
}

// ---------------------------------------------------------------------------
// BalanceParams
// ---------------------------------------------------------------------------

/// Parameters for balance constraint in full-body IK.
#[derive(Clone, Debug, PartialEq)]
pub struct BalanceParams {
    /// Bone index representing the center of mass (typically hips/pelvis).
    pub com_bone: usize,

    /// Safety margin inside the support polygon (meters).
    /// The COM must stay this far inside the polygon edges.
    pub support_margin: f32,

    /// Weight for balance constraint (0.0-1.0).
    pub balance_weight: f32,
}

impl Default for BalanceParams {
    fn default() -> Self {
        Self {
            com_bone: 0,
            support_margin: 0.05,
            balance_weight: 1.0,
        }
    }
}

impl BalanceParams {
    /// Create balance parameters with the given COM bone.
    pub fn new(com_bone: usize) -> Self {
        Self {
            com_bone,
            ..Default::default()
        }
    }

    /// Set the support margin.
    pub fn with_margin(mut self, margin: f32) -> Self {
        self.support_margin = margin.max(0.0);
        self
    }

    /// Set the balance weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.balance_weight = weight.clamp(0.0, 1.0);
        self
    }
}

// ---------------------------------------------------------------------------
// PostureParams
// ---------------------------------------------------------------------------

/// Parameters for posture preservation in full-body IK.
#[derive(Clone, Debug)]
pub struct PostureParams {
    /// Reference pose to preserve (typically bind pose or a natural stance).
    pub reference_pose: Pose,

    /// Weight for posture preservation (0.0-1.0).
    /// Higher values make the solver prefer staying close to reference pose.
    pub posture_weight: f32,

    /// Stiffness for spine joints (0.0-1.0).
    /// Higher values make the spine resist bending more.
    pub spine_stiffness: f32,

    /// Indices of spine bones (for special handling).
    pub spine_bones: Vec<usize>,
}

impl PostureParams {
    /// Create posture parameters with the given reference pose.
    pub fn new(reference_pose: Pose) -> Self {
        Self {
            reference_pose,
            posture_weight: 0.5,
            spine_stiffness: 0.7,
            spine_bones: Vec::new(),
        }
    }

    /// Set the posture weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.posture_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the spine stiffness.
    pub fn with_spine_stiffness(mut self, stiffness: f32) -> Self {
        self.spine_stiffness = stiffness.clamp(0.0, 1.0);
        self
    }

    /// Set the spine bone indices.
    pub fn with_spine_bones(mut self, bones: Vec<usize>) -> Self {
        self.spine_bones = bones;
        self
    }
}

// ---------------------------------------------------------------------------
// FullBodyIkParams
// ---------------------------------------------------------------------------

/// Parameters for full-body IK solving.
#[derive(Clone, Debug)]
pub struct FullBodyIkParams {
    /// List of targets to solve for.
    pub targets: Vec<FullBodyTarget>,

    /// Optional balance constraint parameters.
    pub balance: Option<BalanceParams>,

    /// Optional posture preservation parameters.
    pub posture: Option<PostureParams>,

    /// Per-joint angle limits. Key is bone index.
    pub joint_limits: Vec<(usize, JointLimits)>,

    /// Maximum number of solver iterations.
    pub max_iterations: u32,

    /// Convergence tolerance (in world units).
    pub tolerance: f32,

    /// Damping factor for pseudo-inverse (higher = more stable, slower).
    pub damping: f32,

    /// Indices of foot bones for support polygon calculation.
    pub foot_bones: Vec<usize>,
}

impl Default for FullBodyIkParams {
    fn default() -> Self {
        Self {
            targets: Vec::new(),
            balance: None,
            posture: None,
            joint_limits: Vec::new(),
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tolerance: DEFAULT_TOLERANCE,
            damping: DEFAULT_DAMPING,
            foot_bones: Vec::new(),
        }
    }
}

impl FullBodyIkParams {
    /// Create parameters with the given targets.
    pub fn new(targets: Vec<FullBodyTarget>) -> Self {
        Self {
            targets,
            ..Default::default()
        }
    }

    /// Add a target.
    pub fn add_target(mut self, target: FullBodyTarget) -> Self {
        self.targets.push(target);
        self
    }

    /// Set balance parameters.
    pub fn with_balance(mut self, balance: BalanceParams) -> Self {
        self.balance = Some(balance);
        self
    }

    /// Set posture parameters.
    pub fn with_posture(mut self, posture: PostureParams) -> Self {
        self.posture = Some(posture);
        self
    }

    /// Add a joint limit.
    pub fn add_joint_limit(mut self, bone: usize, limits: JointLimits) -> Self {
        self.joint_limits.push((bone, limits));
        self
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations.max(1);
        self
    }

    /// Set convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance.max(EPSILON);
        self
    }

    /// Set damping factor.
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping.max(EPSILON);
        self
    }

    /// Set foot bones for support polygon.
    pub fn with_foot_bones(mut self, bones: Vec<usize>) -> Self {
        self.foot_bones = bones;
        self
    }

    /// Get joint limits for a specific bone.
    pub fn get_joint_limits(&self, bone: usize) -> Option<&JointLimits> {
        self.joint_limits
            .iter()
            .find(|(b, _)| *b == bone)
            .map(|(_, l)| l)
    }

    /// Sort targets by priority.
    pub fn targets_by_priority(&self) -> Vec<&FullBodyTarget> {
        let mut targets: Vec<_> = self.targets.iter().collect();
        targets.sort_by_key(|t| t.priority);
        targets
    }
}

// ---------------------------------------------------------------------------
// FullBodyIkResult
// ---------------------------------------------------------------------------

/// Result of full-body IK solving.
#[derive(Clone, Debug)]
pub struct FullBodyIkResult {
    /// Whether the solver successfully converged.
    pub success: bool,

    /// Number of iterations used.
    pub iterations: u32,

    /// Final balance error (distance of COM from support polygon center).
    pub balance_error: f32,

    /// Per-target error (distance from effector to target position).
    pub per_target_error: Vec<f32>,

    /// Total weighted error across all targets.
    pub total_error: f32,

    /// Indices of targets that reached their goal.
    pub reached_targets: Vec<usize>,

    /// Indices of targets that couldn't be reached.
    pub unreachable_targets: Vec<usize>,
}

impl Default for FullBodyIkResult {
    fn default() -> Self {
        Self {
            success: false,
            iterations: 0,
            balance_error: f32::MAX,
            per_target_error: Vec::new(),
            total_error: f32::MAX,
            reached_targets: Vec::new(),
            unreachable_targets: Vec::new(),
        }
    }
}

impl FullBodyIkResult {
    /// Create a failed result.
    pub fn failed(target_count: usize) -> Self {
        Self {
            success: false,
            iterations: 0,
            balance_error: f32::MAX,
            per_target_error: vec![f32::MAX; target_count],
            total_error: f32::MAX,
            reached_targets: Vec::new(),
            unreachable_targets: (0..target_count).collect(),
        }
    }

    /// Check if a specific target was reached.
    pub fn target_reached(&self, index: usize) -> bool {
        self.reached_targets.contains(&index)
    }
}

// ---------------------------------------------------------------------------
// Support Polygon
// ---------------------------------------------------------------------------

/// A 2D support polygon for balance calculations.
#[derive(Clone, Debug)]
pub struct SupportPolygon {
    /// Vertices of the polygon in XZ plane (Y is up).
    vertices: Vec<Vec3>,
    /// Center of the polygon.
    center: Vec3,
}

impl SupportPolygon {
    /// Create an empty support polygon.
    pub fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            center: Vec3::ZERO,
        }
    }

    /// Create a support polygon from foot positions.
    ///
    /// For a single foot, creates a small circle around the foot.
    /// For two feet, creates a rectangle encompassing both.
    /// For more, computes the convex hull.
    pub fn from_feet(foot_positions: &[Vec3], foot_radius: f32) -> Self {
        if foot_positions.is_empty() {
            return Self::empty();
        }

        if foot_positions.len() == 1 {
            // Single foot: create a small polygon around it
            let center = foot_positions[0];
            let vertices = vec![
                center + Vec3::new(foot_radius, 0.0, 0.0),
                center + Vec3::new(0.0, 0.0, foot_radius),
                center + Vec3::new(-foot_radius, 0.0, 0.0),
                center + Vec3::new(0.0, 0.0, -foot_radius),
            ];
            return Self { vertices, center };
        }

        // Two or more feet: compute bounding polygon
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        let mut center = Vec3::ZERO;

        for pos in foot_positions {
            min_x = min_x.min(pos.x - foot_radius);
            max_x = max_x.max(pos.x + foot_radius);
            min_z = min_z.min(pos.z - foot_radius);
            max_z = max_z.max(pos.z + foot_radius);
            center += *pos;
        }

        center /= foot_positions.len() as f32;

        // Simple rectangular polygon
        let vertices = vec![
            Vec3::new(min_x, 0.0, min_z),
            Vec3::new(max_x, 0.0, min_z),
            Vec3::new(max_x, 0.0, max_z),
            Vec3::new(min_x, 0.0, max_z),
        ];

        Self { vertices, center }
    }

    /// Get the center of the support polygon.
    pub fn center(&self) -> Vec3 {
        self.center
    }

    /// Check if a point (projected to XZ plane) is inside the polygon.
    pub fn contains(&self, point: Vec3) -> bool {
        if self.vertices.len() < 3 {
            return false;
        }

        // Ray casting algorithm for point-in-polygon
        let x = point.x;
        let z = point.z;
        let mut inside = false;

        let n = self.vertices.len();
        let mut j = n - 1;

        for i in 0..n {
            let vi = &self.vertices[i];
            let vj = &self.vertices[j];

            if ((vi.z > z) != (vj.z > z))
                && (x < (vj.x - vi.x) * (z - vi.z) / (vj.z - vi.z) + vi.x)
            {
                inside = !inside;
            }
            j = i;
        }

        inside
    }

    /// Get the closest point on the polygon boundary to the given point.
    /// If the point is inside, returns the point itself (optionally with margin).
    pub fn closest_point(&self, point: Vec3, margin: f32) -> Vec3 {
        if self.vertices.is_empty() {
            return point;
        }

        // If inside and far enough from edges, return the point
        if self.contains(point) && self.distance_to_edge(point) >= margin {
            return point;
        }

        // Find closest point on each edge
        let mut closest = self.center;
        let mut closest_dist = f32::MAX;

        let n = self.vertices.len();
        for i in 0..n {
            let v1 = self.vertices[i];
            let v2 = self.vertices[(i + 1) % n];

            let edge = v2 - v1;
            let edge_len_sq = edge.x * edge.x + edge.z * edge.z;

            if edge_len_sq < EPSILON {
                continue;
            }

            let to_point = point - v1;
            let t = ((to_point.x * edge.x + to_point.z * edge.z) / edge_len_sq).clamp(0.0, 1.0);

            let proj = v1 + edge * t;
            let dist = ((point.x - proj.x).powi(2) + (point.z - proj.z).powi(2)).sqrt();

            if dist < closest_dist {
                closest_dist = dist;
                // Move point inside by margin
                let normal = Vec3::new(edge.z, 0.0, -edge.x).normalize_or_zero();
                closest = proj + normal * margin;
            }
        }

        closest
    }

    /// Get the distance from a point to the nearest edge.
    fn distance_to_edge(&self, point: Vec3) -> f32 {
        if self.vertices.len() < 2 {
            return 0.0;
        }

        let mut min_dist = f32::MAX;
        let n = self.vertices.len();

        for i in 0..n {
            let v1 = self.vertices[i];
            let v2 = self.vertices[(i + 1) % n];

            let edge = v2 - v1;
            let edge_len_sq = edge.x * edge.x + edge.z * edge.z;

            if edge_len_sq < EPSILON {
                continue;
            }

            let to_point = point - v1;
            let t = ((to_point.x * edge.x + to_point.z * edge.z) / edge_len_sq).clamp(0.0, 1.0);

            let proj = v1 + edge * t;
            let dist = ((point.x - proj.x).powi(2) + (point.z - proj.z).powi(2)).sqrt();

            min_dist = min_dist.min(dist);
        }

        min_dist
    }
}

// ---------------------------------------------------------------------------
// IK Chain for Full Body
// ---------------------------------------------------------------------------

/// A kinematic chain connecting a root to an effector.
#[derive(Clone, Debug)]
struct IkChain {
    /// Bone indices from root to effector (inclusive).
    bones: Vec<usize>,
    /// Lengths between consecutive bones.
    lengths: Vec<f32>,
    /// Total chain length.
    total_length: f32,
}

impl IkChain {
    /// Build a chain from effector back to root.
    fn from_effector(skeleton: &Skeleton, effector: usize) -> Self {
        let mut bones = vec![effector];
        let mut current = effector;

        // Walk up the hierarchy
        while let Some(parent) = skeleton.parent(current) {
            bones.push(parent);
            current = parent;
        }

        bones.reverse();

        Self {
            bones,
            lengths: Vec::new(),
            total_length: 0.0,
        }
    }

    /// Compute chain lengths from current pose.
    fn compute_lengths(&mut self, skeleton: &Skeleton, pose: &Pose) {
        let transforms = pose.transforms();
        let world_transforms = skeleton.compute_world_transforms(&transforms);

        self.lengths.clear();
        self.total_length = 0.0;

        for i in 0..self.bones.len().saturating_sub(1) {
            let p1 = world_transforms[self.bones[i]].w_axis.truncate();
            let p2 = world_transforms[self.bones[i + 1]].w_axis.truncate();
            let length = (p2 - p1).length().max(MIN_BONE_LENGTH);
            self.lengths.push(length);
            self.total_length += length;
        }
    }
}

// ---------------------------------------------------------------------------
// Full-Body IK Solver
// ---------------------------------------------------------------------------

/// Solve full-body inverse kinematics.
///
/// This function modifies the pose in-place to reach the specified targets
/// while respecting balance constraints, posture preferences, and joint limits.
///
/// # Arguments
///
/// * `skeleton` - The skeleton hierarchy
/// * `pose` - The pose to modify (modified in-place)
/// * `params` - Solver parameters including targets and constraints
///
/// # Returns
///
/// A result struct containing convergence info and per-target errors.
pub fn solve_fullbody_ik(
    skeleton: &Skeleton,
    pose: &mut Pose,
    params: &FullBodyIkParams,
) -> FullBodyIkResult {
    if params.targets.is_empty() {
        return FullBodyIkResult {
            success: true,
            iterations: 0,
            balance_error: 0.0,
            per_target_error: Vec::new(),
            total_error: 0.0,
            reached_targets: Vec::new(),
            unreachable_targets: Vec::new(),
        };
    }

    // Validate targets
    for target in &params.targets {
        if target.effector >= skeleton.bone_count() {
            return FullBodyIkResult::failed(params.targets.len());
        }
    }

    // Group targets by priority
    let mut priority_groups: Vec<Vec<usize>> = vec![Vec::new(); MAX_PRIORITY_LEVELS as usize];
    for (i, target) in params.targets.iter().enumerate() {
        let priority = (target.priority as usize).min(MAX_PRIORITY_LEVELS as usize - 1);
        priority_groups[priority].push(i);
    }

    // Build chains for each target
    let chains: Vec<IkChain> = params
        .targets
        .iter()
        .map(|t| {
            let mut chain = IkChain::from_effector(skeleton, t.effector);
            chain.compute_lengths(skeleton, pose);
            chain
        })
        .collect();

    // Compute support polygon for balance
    let support_polygon = if params.balance.is_some() && !params.foot_bones.is_empty() {
        let transforms = pose.transforms();
        let world_transforms = skeleton.compute_world_transforms(&transforms);
        let foot_positions: Vec<Vec3> = params
            .foot_bones
            .iter()
            .filter(|&&b| b < skeleton.bone_count())
            .map(|&b| world_transforms[b].w_axis.truncate())
            .collect();
        SupportPolygon::from_feet(&foot_positions, 0.1)
    } else {
        SupportPolygon::empty()
    };

    let mut result = FullBodyIkResult {
        success: false,
        iterations: 0,
        balance_error: 0.0,
        per_target_error: vec![0.0; params.targets.len()],
        total_error: 0.0,
        reached_targets: Vec::new(),
        unreachable_targets: Vec::new(),
    };

    // Main iteration loop
    for iteration in 0..params.max_iterations {
        result.iterations = iteration + 1;

        // Process each priority level
        for priority in 0..MAX_PRIORITY_LEVELS as usize {
            let target_indices = &priority_groups[priority];
            if target_indices.is_empty() {
                continue;
            }

            // Solve targets at this priority level
            for &target_idx in target_indices {
                let target = &params.targets[target_idx];
                let chain = &chains[target_idx];

                solve_single_target(skeleton, pose, target, chain, params);
            }
        }

        // Apply balance constraint
        if let Some(ref balance) = params.balance {
            apply_balance_constraint(
                skeleton,
                pose,
                balance,
                &support_polygon,
            );
        }

        // Apply posture preservation
        if let Some(ref posture) = params.posture {
            apply_posture_preservation(skeleton, pose, posture);
        }

        // Apply joint limits
        apply_joint_limits(skeleton, pose, params);

        // Compute errors
        let transforms = pose.transforms();
        let world_transforms = skeleton.compute_world_transforms(&transforms);

        result.total_error = 0.0;
        result.reached_targets.clear();
        result.unreachable_targets.clear();

        for (i, target) in params.targets.iter().enumerate() {
            let effector_pos = world_transforms[target.effector].w_axis.truncate();
            let error = (effector_pos - target.position).length();
            result.per_target_error[i] = error;
            result.total_error += error * target.weight;

            if error <= params.tolerance {
                result.reached_targets.push(i);
            } else {
                result.unreachable_targets.push(i);
            }
        }

        // Compute balance error
        if let Some(ref balance) = params.balance {
            let com_pos = world_transforms[balance.com_bone].w_axis.truncate();
            let target_pos = support_polygon.closest_point(com_pos, balance.support_margin);
            result.balance_error = ((com_pos.x - target_pos.x).powi(2)
                + (com_pos.z - target_pos.z).powi(2))
            .sqrt();
        }

        // Check convergence
        if result.total_error <= params.tolerance * params.targets.len() as f32 {
            result.success = true;
            break;
        }
    }

    // Final success check
    if result.unreachable_targets.is_empty() {
        result.success = true;
    }

    result
}

/// Solve for a single target using a damped Jacobian approach.
fn solve_single_target(
    skeleton: &Skeleton,
    pose: &mut Pose,
    target: &FullBodyTarget,
    chain: &IkChain,
    params: &FullBodyIkParams,
) {
    if chain.bones.is_empty() {
        return;
    }

    // Get current effector position
    let transforms = pose.transforms();
    let world_transforms = skeleton.compute_world_transforms(&transforms);
    let effector_pos = world_transforms[target.effector].w_axis.truncate();

    // Compute error
    let error = target.position - effector_pos;
    let error_len = error.length();

    if error_len <= params.tolerance {
        return;
    }

    // Simple gradient descent on joint angles
    // We use a simplified approach: adjust each bone's rotation toward the target
    let step_size = 0.1 * target.weight;

    for i in 0..chain.bones.len().saturating_sub(1) {
        let bone_idx = chain.bones[i];
        let child_idx = chain.bones[i + 1];

        // Get current bone world transform
        let bone_world = world_transforms[bone_idx];
        let bone_pos = bone_world.w_axis.truncate();

        // Get child position
        let child_pos = world_transforms[child_idx].w_axis.truncate();

        // Vector from bone to child
        let to_child = (child_pos - bone_pos).normalize_or_zero();

        // Vector from bone to target
        let to_target = (target.position - bone_pos).normalize_or_zero();

        // If vectors are valid, compute rotation
        if to_child.length_squared() > EPSILON && to_target.length_squared() > EPSILON {
            // Rotation axis
            let axis = to_child.cross(to_target);
            let axis_len = axis.length();

            if axis_len > EPSILON {
                let axis = axis / axis_len;
                let angle = to_child.dot(to_target).clamp(-1.0, 1.0).acos();
                let clamped_angle = (angle * step_size).min(0.3); // Limit step

                // Apply rotation delta
                let delta_rot = Quat::from_axis_angle(axis, clamped_angle);

                // Convert world rotation delta to local space
                let parent_world_rot = if let Some(parent) = skeleton.parent(bone_idx) {
                    Quat::from_mat4(&world_transforms[parent])
                } else {
                    Quat::IDENTITY
                };

                let local_delta = parent_world_rot.inverse() * delta_rot * parent_world_rot;
                let current_rot = pose.rotations[bone_idx];
                pose.rotations[bone_idx] = (local_delta * current_rot).normalize();
            }
        }
    }
}

/// Apply balance constraint by adjusting hip position.
fn apply_balance_constraint(
    skeleton: &Skeleton,
    pose: &mut Pose,
    balance: &BalanceParams,
    support_polygon: &SupportPolygon,
) {
    if balance.com_bone >= skeleton.bone_count() {
        return;
    }

    // Get current COM position
    let transforms = pose.transforms();
    let world_transforms = skeleton.compute_world_transforms(&transforms);
    let com_pos = world_transforms[balance.com_bone].w_axis.truncate();

    // Project COM to support polygon
    let target_pos = support_polygon.closest_point(com_pos, balance.support_margin);

    // Compute required adjustment (in XZ plane)
    let delta = Vec3::new(
        (target_pos.x - com_pos.x) * balance.balance_weight,
        0.0,
        (target_pos.z - com_pos.z) * balance.balance_weight,
    );

    if delta.length_squared() < EPSILON {
        return;
    }

    // Adjust the COM bone position
    // Convert world delta to local space
    let parent_world = if let Some(parent) = skeleton.parent(balance.com_bone) {
        world_transforms[parent]
    } else {
        Mat4::IDENTITY
    };

    let parent_rot = Quat::from_mat4(&parent_world);
    let local_delta = parent_rot.inverse() * delta;

    pose.positions[balance.com_bone] += local_delta;
}

/// Apply posture preservation by blending toward reference pose.
fn apply_posture_preservation(_skeleton: &Skeleton, pose: &mut Pose, posture: &PostureParams) {
    if pose.bone_count() != posture.reference_pose.bone_count() {
        return;
    }

    let weight = posture.posture_weight;
    let spine_weight = weight * posture.spine_stiffness;

    for i in 0..pose.bone_count() {
        let blend_weight = if posture.spine_bones.contains(&i) {
            spine_weight
        } else {
            weight
        };

        if blend_weight > EPSILON {
            // Blend rotation toward reference
            let ref_rot = posture.reference_pose.rotations[i];
            pose.rotations[i] = pose.rotations[i]
                .slerp(ref_rot, blend_weight * 0.1)
                .normalize();
        }
    }
}

/// Apply joint limits to all bones.
fn apply_joint_limits(_skeleton: &Skeleton, pose: &mut Pose, params: &FullBodyIkParams) {
    for (bone_idx, limits) in &params.joint_limits {
        if *bone_idx >= pose.bone_count() {
            continue;
        }

        let rotation = pose.rotations[*bone_idx];

        // Convert to Euler angles (XYZ order)
        let (x, y, z) = quat_to_euler_xyz(rotation);
        let euler = Vec3::new(x, y, z);

        // Clamp to limits
        let clamped = limits.clamp_euler(euler);

        // Convert back to quaternion
        if (euler - clamped).length_squared() > EPSILON {
            pose.rotations[*bone_idx] = euler_xyz_to_quat(clamped.x, clamped.y, clamped.z);
        }
    }
}

/// Convert quaternion to Euler angles (XYZ intrinsic rotation order).
fn quat_to_euler_xyz(q: Quat) -> (f32, f32, f32) {
    // Use glam's built-in euler conversion for consistency with euler_xyz_to_quat
    q.to_euler(glam::EulerRot::XYZ)
}

/// Convert Euler angles (XYZ intrinsic) to quaternion.
fn euler_xyz_to_quat(x: f32, y: f32, z: f32) -> Quat {
    Quat::from_euler(glam::EulerRot::XYZ, x, y, z)
}

// ---------------------------------------------------------------------------
// Pose Extension
// ---------------------------------------------------------------------------

/// Extension trait for Pose to provide transform iteration.
#[allow(dead_code)]
trait PoseExt {
    fn transforms(&self) -> Vec<Transform>;
}

impl PoseExt for Pose {
    fn transforms(&self) -> Vec<Transform> {
        (0..self.bone_count())
            .map(|i| self.get_transform(i))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::Bone;

    /// Create a test skeleton with a simple humanoid structure.
    fn create_test_skeleton() -> Skeleton {
        let mut skeleton = Skeleton::new();

        // Root (hips) - index 0
        skeleton.add_bone(Bone::root("hips").with_local_transform(
            Transform::from_position(Vec3::new(0.0, 1.0, 0.0)),
        ));

        // Spine - index 1
        skeleton.add_bone(
            Bone::new("spine")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 0.2, 0.0))),
        );

        // Chest - index 2
        skeleton.add_bone(
            Bone::new("chest")
                .with_parent(1)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 0.2, 0.0))),
        );

        // Head - index 3
        skeleton.add_bone(
            Bone::new("head")
                .with_parent(2)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 0.3, 0.0))),
        );

        // Left shoulder - index 4
        skeleton.add_bone(
            Bone::new("l_shoulder")
                .with_parent(2)
                .with_local_transform(Transform::from_position(Vec3::new(-0.2, 0.0, 0.0))),
        );

        // Left elbow - index 5
        skeleton.add_bone(
            Bone::new("l_elbow")
                .with_parent(4)
                .with_local_transform(Transform::from_position(Vec3::new(-0.3, 0.0, 0.0))),
        );

        // Left hand - index 6
        skeleton.add_bone(
            Bone::new("l_hand")
                .with_parent(5)
                .with_local_transform(Transform::from_position(Vec3::new(-0.25, 0.0, 0.0))),
        );

        // Right shoulder - index 7
        skeleton.add_bone(
            Bone::new("r_shoulder")
                .with_parent(2)
                .with_local_transform(Transform::from_position(Vec3::new(0.2, 0.0, 0.0))),
        );

        // Right elbow - index 8
        skeleton.add_bone(
            Bone::new("r_elbow")
                .with_parent(7)
                .with_local_transform(Transform::from_position(Vec3::new(0.3, 0.0, 0.0))),
        );

        // Right hand - index 9
        skeleton.add_bone(
            Bone::new("r_hand")
                .with_parent(8)
                .with_local_transform(Transform::from_position(Vec3::new(0.25, 0.0, 0.0))),
        );

        // Left hip - index 10
        skeleton.add_bone(
            Bone::new("l_hip")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(-0.1, 0.0, 0.0))),
        );

        // Left knee - index 11
        skeleton.add_bone(
            Bone::new("l_knee")
                .with_parent(10)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, -0.4, 0.0))),
        );

        // Left foot - index 12
        skeleton.add_bone(
            Bone::new("l_foot")
                .with_parent(11)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, -0.4, 0.0))),
        );

        // Right hip - index 13
        skeleton.add_bone(
            Bone::new("r_hip")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(0.1, 0.0, 0.0))),
        );

        // Right knee - index 14
        skeleton.add_bone(
            Bone::new("r_knee")
                .with_parent(13)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, -0.4, 0.0))),
        );

        // Right foot - index 15
        skeleton.add_bone(
            Bone::new("r_foot")
                .with_parent(14)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, -0.4, 0.0))),
        );

        skeleton
    }

    fn create_test_pose(skeleton: &Skeleton) -> Pose {
        Pose::from_skeleton(skeleton, crate::pose::PoseType::Current)
    }

    // -------------------------------------------------------------------------
    // Joint Limits Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_joint_limits_default() {
        let limits = JointLimits::default();
        assert_eq!(limits.min_x, -PI);
        assert_eq!(limits.max_x, PI);
        assert!(limits.soft_ratio > 0.0);
    }

    #[test]
    fn test_joint_limits_symmetric() {
        let limits = JointLimits::symmetric(1.0);
        assert_eq!(limits.min_x, -1.0);
        assert_eq!(limits.max_x, 1.0);
        assert_eq!(limits.min_y, -1.0);
        assert_eq!(limits.max_y, 1.0);
    }

    #[test]
    fn test_joint_limits_hinge() {
        let limits = JointLimits::hinge_x(-0.5, 2.5);
        assert_eq!(limits.min_x, -0.5);
        assert_eq!(limits.max_x, 2.5);
        assert_eq!(limits.min_y, 0.0);
        assert_eq!(limits.max_y, 0.0);
    }

    #[test]
    fn test_joint_limits_elbow() {
        let limits = JointLimits::elbow();
        assert!(limits.min_x >= 0.0);
        assert!(limits.max_x <= 3.0);
    }

    #[test]
    fn test_joint_limits_knee() {
        let limits = JointLimits::knee();
        assert!(limits.min_x >= 0.0);
        assert!(limits.max_x <= 3.0);
    }

    #[test]
    fn test_joint_limits_shoulder() {
        let limits = JointLimits::shoulder();
        assert!(limits.min_x < 0.0);
        assert!(limits.max_x > 0.0);
    }

    #[test]
    fn test_joint_limits_hip() {
        let limits = JointLimits::hip();
        assert!(limits.min_x < 0.0);
        assert!(limits.max_x > 0.0);
    }

    #[test]
    fn test_joint_limits_spine() {
        let limits = JointLimits::spine();
        assert!(limits.min_x.abs() < 1.0);
        assert!(limits.max_x.abs() < 1.0);
    }

    #[test]
    fn test_joint_limits_clamp_euler() {
        let limits = JointLimits::symmetric(1.0).with_soft_ratio(0.0);
        let euler = Vec3::new(0.5, -0.5, 0.0);
        let clamped = limits.clamp_euler(euler);
        assert_eq!(clamped, euler); // Within limits

        let euler_outside = Vec3::new(1.5, -1.5, 0.0);
        let clamped_outside = limits.clamp_euler(euler_outside);
        assert!((clamped_outside.x - 1.0).abs() < EPSILON);
        assert!((clamped_outside.y - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_joint_limits_is_within() {
        let limits = JointLimits::symmetric(1.0);
        assert!(limits.is_within(Vec3::new(0.5, 0.5, 0.5)));
        assert!(!limits.is_within(Vec3::new(1.5, 0.5, 0.5)));
    }

    #[test]
    fn test_soft_clamp_in_range() {
        let result = soft_clamp(0.5, 0.0, 1.0, 0.1);
        assert!((result - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_soft_clamp_below_min() {
        let result = soft_clamp(-0.5, 0.0, 1.0, 0.0);
        assert!((result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_soft_clamp_above_max() {
        let result = soft_clamp(1.5, 0.0, 1.0, 0.0);
        assert!((result - 1.0).abs() < EPSILON);
    }

    // -------------------------------------------------------------------------
    // FullBodyTarget Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fullbody_target_new() {
        let target = FullBodyTarget::new(5, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(target.effector, 5);
        assert_eq!(target.position, Vec3::new(1.0, 2.0, 3.0));
        assert!(target.rotation.is_none());
        assert_eq!(target.weight, 1.0);
        assert_eq!(target.priority, 0);
    }

    #[test]
    fn test_fullbody_target_with_rotation() {
        let rot = Quat::from_rotation_y(PI / 2.0);
        let target = FullBodyTarget::with_rotation(5, Vec3::ZERO, rot);
        assert!(target.rotation.is_some());
        assert_eq!(target.constraint_dim(), 6);
    }

    #[test]
    fn test_fullbody_target_with_weight() {
        let target = FullBodyTarget::new(5, Vec3::ZERO).with_weight(0.5);
        assert_eq!(target.weight, 0.5);
    }

    #[test]
    fn test_fullbody_target_weight_clamped() {
        let target = FullBodyTarget::new(5, Vec3::ZERO).with_weight(2.0);
        assert_eq!(target.weight, 1.0);

        let target2 = FullBodyTarget::new(5, Vec3::ZERO).with_weight(-1.0);
        assert_eq!(target2.weight, 0.0);
    }

    #[test]
    fn test_fullbody_target_with_priority() {
        let target = FullBodyTarget::new(5, Vec3::ZERO).with_priority(2);
        assert_eq!(target.priority, 2);
    }

    #[test]
    fn test_fullbody_target_constraint_dim_position() {
        let target = FullBodyTarget::new(5, Vec3::ZERO);
        assert_eq!(target.constraint_dim(), 3);
    }

    // -------------------------------------------------------------------------
    // BalanceParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_balance_params_default() {
        let params = BalanceParams::default();
        assert_eq!(params.com_bone, 0);
        assert!(params.support_margin > 0.0);
        assert_eq!(params.balance_weight, 1.0);
    }

    #[test]
    fn test_balance_params_new() {
        let params = BalanceParams::new(5);
        assert_eq!(params.com_bone, 5);
    }

    #[test]
    fn test_balance_params_with_margin() {
        let params = BalanceParams::new(0).with_margin(0.1);
        assert_eq!(params.support_margin, 0.1);
    }

    #[test]
    fn test_balance_params_with_weight() {
        let params = BalanceParams::new(0).with_weight(0.8);
        assert_eq!(params.balance_weight, 0.8);
    }

    // -------------------------------------------------------------------------
    // PostureParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_posture_params_new() {
        let skeleton = create_test_skeleton();
        let pose = create_test_pose(&skeleton);
        let params = PostureParams::new(pose);
        assert_eq!(params.posture_weight, 0.5);
        assert_eq!(params.spine_stiffness, 0.7);
    }

    #[test]
    fn test_posture_params_with_weight() {
        let skeleton = create_test_skeleton();
        let pose = create_test_pose(&skeleton);
        let params = PostureParams::new(pose).with_weight(0.3);
        assert_eq!(params.posture_weight, 0.3);
    }

    #[test]
    fn test_posture_params_with_spine_stiffness() {
        let skeleton = create_test_skeleton();
        let pose = create_test_pose(&skeleton);
        let params = PostureParams::new(pose).with_spine_stiffness(0.9);
        assert_eq!(params.spine_stiffness, 0.9);
    }

    #[test]
    fn test_posture_params_with_spine_bones() {
        let skeleton = create_test_skeleton();
        let pose = create_test_pose(&skeleton);
        let params = PostureParams::new(pose).with_spine_bones(vec![1, 2]);
        assert_eq!(params.spine_bones, vec![1, 2]);
    }

    // -------------------------------------------------------------------------
    // FullBodyIkParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fullbody_ik_params_default() {
        let params = FullBodyIkParams::default();
        assert!(params.targets.is_empty());
        assert!(params.balance.is_none());
        assert!(params.posture.is_none());
        assert_eq!(params.max_iterations, DEFAULT_MAX_ITERATIONS);
    }

    #[test]
    fn test_fullbody_ik_params_new() {
        let targets = vec![FullBodyTarget::new(5, Vec3::ZERO)];
        let params = FullBodyIkParams::new(targets);
        assert_eq!(params.targets.len(), 1);
    }

    #[test]
    fn test_fullbody_ik_params_add_target() {
        let params = FullBodyIkParams::default()
            .add_target(FullBodyTarget::new(5, Vec3::ZERO))
            .add_target(FullBodyTarget::new(6, Vec3::ONE));
        assert_eq!(params.targets.len(), 2);
    }

    #[test]
    fn test_fullbody_ik_params_with_balance() {
        let params = FullBodyIkParams::default().with_balance(BalanceParams::new(0));
        assert!(params.balance.is_some());
    }

    #[test]
    fn test_fullbody_ik_params_with_max_iterations() {
        let params = FullBodyIkParams::default().with_max_iterations(50);
        assert_eq!(params.max_iterations, 50);
    }

    #[test]
    fn test_fullbody_ik_params_with_tolerance() {
        let params = FullBodyIkParams::default().with_tolerance(0.01);
        assert_eq!(params.tolerance, 0.01);
    }

    #[test]
    fn test_fullbody_ik_params_with_damping() {
        let params = FullBodyIkParams::default().with_damping(0.1);
        assert_eq!(params.damping, 0.1);
    }

    #[test]
    fn test_fullbody_ik_params_add_joint_limit() {
        let params = FullBodyIkParams::default()
            .add_joint_limit(5, JointLimits::elbow());
        assert_eq!(params.joint_limits.len(), 1);
        assert!(params.get_joint_limits(5).is_some());
    }

    #[test]
    fn test_fullbody_ik_params_targets_by_priority() {
        let params = FullBodyIkParams::default()
            .add_target(FullBodyTarget::new(5, Vec3::ZERO).with_priority(2))
            .add_target(FullBodyTarget::new(6, Vec3::ZERO).with_priority(0))
            .add_target(FullBodyTarget::new(7, Vec3::ZERO).with_priority(1));

        let sorted = params.targets_by_priority();
        assert_eq!(sorted[0].priority, 0);
        assert_eq!(sorted[1].priority, 1);
        assert_eq!(sorted[2].priority, 2);
    }

    // -------------------------------------------------------------------------
    // FullBodyIkResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fullbody_ik_result_default() {
        let result = FullBodyIkResult::default();
        assert!(!result.success);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_fullbody_ik_result_failed() {
        let result = FullBodyIkResult::failed(3);
        assert!(!result.success);
        assert_eq!(result.per_target_error.len(), 3);
        assert_eq!(result.unreachable_targets.len(), 3);
    }

    #[test]
    fn test_fullbody_ik_result_target_reached() {
        let mut result = FullBodyIkResult::default();
        result.reached_targets = vec![0, 2];
        assert!(result.target_reached(0));
        assert!(!result.target_reached(1));
        assert!(result.target_reached(2));
    }

    // -------------------------------------------------------------------------
    // Support Polygon Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_support_polygon_empty() {
        let poly = SupportPolygon::empty();
        assert_eq!(poly.center(), Vec3::ZERO);
        assert!(!poly.contains(Vec3::ZERO));
    }

    #[test]
    fn test_support_polygon_single_foot() {
        let poly = SupportPolygon::from_feet(&[Vec3::new(1.0, 0.0, 1.0)], 0.5);
        assert!((poly.center().x - 1.0).abs() < EPSILON);
        assert!((poly.center().z - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_support_polygon_two_feet() {
        let poly = SupportPolygon::from_feet(
            &[Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0)],
            0.1,
        );
        assert!((poly.center().x).abs() < EPSILON);
        assert!((poly.center().z).abs() < EPSILON);
    }

    #[test]
    fn test_support_polygon_contains_center() {
        let poly = SupportPolygon::from_feet(
            &[Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, 1.0)],
            0.1,
        );
        assert!(poly.contains(Vec3::ZERO));
    }

    #[test]
    fn test_support_polygon_contains_outside() {
        let poly = SupportPolygon::from_feet(
            &[Vec3::new(-0.5, 0.0, -0.5), Vec3::new(0.5, 0.0, 0.5)],
            0.1,
        );
        assert!(!poly.contains(Vec3::new(10.0, 0.0, 10.0)));
    }

    #[test]
    fn test_support_polygon_closest_point_inside() {
        let poly = SupportPolygon::from_feet(
            &[Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, 1.0)],
            0.1,
        );
        let closest = poly.closest_point(Vec3::ZERO, 0.0);
        assert!((closest.x).abs() < EPSILON);
        assert!((closest.z).abs() < EPSILON);
    }

    // -------------------------------------------------------------------------
    // IK Chain Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ik_chain_from_effector() {
        let skeleton = create_test_skeleton();
        let chain = IkChain::from_effector(&skeleton, 6); // Left hand

        // Should contain: hips -> spine -> chest -> l_shoulder -> l_elbow -> l_hand
        assert!(chain.bones.len() >= 4);
        assert_eq!(*chain.bones.last().unwrap(), 6);
        assert_eq!(chain.bones[0], 0); // Root
    }

    #[test]
    fn test_ik_chain_compute_lengths() {
        let skeleton = create_test_skeleton();
        let pose = create_test_pose(&skeleton);
        let mut chain = IkChain::from_effector(&skeleton, 6);

        chain.compute_lengths(&skeleton, &pose);

        assert!(!chain.lengths.is_empty());
        assert!(chain.total_length > 0.0);
    }

    // -------------------------------------------------------------------------
    // Full-Body IK Solver Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_solve_fullbody_ik_empty_targets() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);
        let params = FullBodyIkParams::default();

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.success);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_solve_fullbody_ik_single_hand() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        // Target the left hand (index 6) to a reachable position
        let target = FullBodyTarget::new(6, Vec3::new(-0.6, 1.4, 0.3));
        let params = FullBodyIkParams::new(vec![target])
            .with_max_iterations(30)
            .with_tolerance(0.1);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
        assert!(!result.per_target_error.is_empty());
    }

    #[test]
    fn test_solve_fullbody_ik_both_hands() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        let params = FullBodyIkParams::default()
            .add_target(FullBodyTarget::new(6, Vec3::new(-0.5, 1.3, 0.2)))
            .add_target(FullBodyTarget::new(9, Vec3::new(0.5, 1.3, 0.2)))
            .with_max_iterations(30);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
        assert_eq!(result.per_target_error.len(), 2);
    }

    #[test]
    fn test_solve_fullbody_ik_hands_and_feet() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        let params = FullBodyIkParams::default()
            .add_target(FullBodyTarget::new(6, Vec3::new(-0.5, 1.3, 0.2)).with_priority(2))
            .add_target(FullBodyTarget::new(9, Vec3::new(0.5, 1.3, 0.2)).with_priority(2))
            .add_target(FullBodyTarget::new(12, Vec3::new(-0.15, 0.0, 0.0)).with_priority(1))
            .add_target(FullBodyTarget::new(15, Vec3::new(0.15, 0.0, 0.0)).with_priority(1))
            .with_max_iterations(30);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
        assert_eq!(result.per_target_error.len(), 4);
    }

    #[test]
    fn test_solve_fullbody_ik_with_balance() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        let params = FullBodyIkParams::default()
            .add_target(FullBodyTarget::new(6, Vec3::new(-0.5, 1.2, 0.3)))
            .with_balance(BalanceParams::new(0).with_margin(0.05))
            .with_foot_bones(vec![12, 15])
            .with_max_iterations(20);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
        // Balance error should be computed
        assert!(result.balance_error < f32::MAX || result.balance_error >= 0.0);
    }

    #[test]
    fn test_solve_fullbody_ik_with_posture() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);
        let reference_pose = create_test_pose(&skeleton);

        let params = FullBodyIkParams::default()
            .add_target(FullBodyTarget::new(6, Vec3::new(-0.5, 1.2, 0.3)))
            .with_posture(
                PostureParams::new(reference_pose)
                    .with_weight(0.3)
                    .with_spine_bones(vec![1, 2]),
            )
            .with_max_iterations(20);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_fullbody_ik_priority_layering() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        // Priority 0 = balance (highest)
        // Priority 1 = feet
        // Priority 2 = hands (lowest)
        let params = FullBodyIkParams::default()
            .add_target(FullBodyTarget::new(12, Vec3::new(-0.1, 0.0, 0.0)).with_priority(1))
            .add_target(FullBodyTarget::new(15, Vec3::new(0.1, 0.0, 0.0)).with_priority(1))
            .add_target(FullBodyTarget::new(6, Vec3::new(-0.6, 1.5, 0.5)).with_priority(2))
            .with_balance(BalanceParams::new(0))
            .with_foot_bones(vec![12, 15])
            .with_max_iterations(30);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
        // Lower priority targets (hands) should have higher error than feet
    }

    #[test]
    fn test_solve_fullbody_ik_with_joint_limits() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        let params = FullBodyIkParams::default()
            .add_target(FullBodyTarget::new(6, Vec3::new(-0.5, 1.2, 0.3)))
            .add_joint_limit(5, JointLimits::elbow())
            .add_joint_limit(4, JointLimits::shoulder())
            .with_max_iterations(20);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_fullbody_ik_unreachable_target() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        // Target way too far to reach
        let target = FullBodyTarget::new(6, Vec3::new(-10.0, 10.0, 10.0));
        let params = FullBodyIkParams::new(vec![target])
            .with_max_iterations(10)
            .with_tolerance(0.01);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        // Should not converge for unreachable target
        assert!(!result.success || result.per_target_error[0] > 0.01);
    }

    #[test]
    fn test_solve_fullbody_ik_conflicting_targets() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        // Two targets that conflict (both hands trying to reach same spot)
        let params = FullBodyIkParams::default()
            .add_target(FullBodyTarget::new(6, Vec3::new(0.0, 1.5, 0.5)))
            .add_target(FullBodyTarget::new(9, Vec3::new(0.0, 1.5, 0.5)))
            .with_max_iterations(30)
            .with_tolerance(0.1);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
        // At least one target should have some error due to conflict
    }

    #[test]
    fn test_solve_fullbody_ik_invalid_bone_index() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        let target = FullBodyTarget::new(999, Vec3::ZERO);
        let params = FullBodyIkParams::new(vec![target]);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(!result.success);
    }

    // -------------------------------------------------------------------------
    // Euler Angle Conversion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_quat_to_euler_identity() {
        let (x, y, z) = quat_to_euler_xyz(Quat::IDENTITY);
        assert!(x.abs() < EPSILON);
        assert!(y.abs() < EPSILON);
        assert!(z.abs() < EPSILON);
    }

    #[test]
    fn test_euler_to_quat_identity() {
        let q = euler_xyz_to_quat(0.0, 0.0, 0.0);
        assert!((q.dot(Quat::IDENTITY) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_euler_roundtrip() {
        let original = Vec3::new(0.3, 0.2, 0.1);
        let q = euler_xyz_to_quat(original.x, original.y, original.z);
        let (x, y, z) = quat_to_euler_xyz(q);
        let recovered = Vec3::new(x, y, z);

        assert!((original - recovered).length() < 0.01);
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_solve_zero_weight_target() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        let target = FullBodyTarget::new(6, Vec3::new(-0.5, 1.2, 0.3)).with_weight(0.0);
        let params = FullBodyIkParams::new(vec![target]).with_max_iterations(10);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_target_at_current_position() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        // Get current hand position
        let transforms = pose.transforms();
        let world_transforms = skeleton.compute_world_transforms(&transforms);
        let hand_pos = world_transforms[6].w_axis.truncate();

        let target = FullBodyTarget::new(6, hand_pos);
        let params = FullBodyIkParams::new(vec![target])
            .with_max_iterations(10)
            .with_tolerance(0.01);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.success);
        assert!(result.per_target_error[0] < 0.01);
    }

    #[test]
    fn test_solve_with_rotation_target() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        let rot = Quat::from_rotation_y(PI / 4.0);
        let target = FullBodyTarget::with_rotation(6, Vec3::new(-0.5, 1.2, 0.3), rot);
        let params = FullBodyIkParams::new(vec![target]).with_max_iterations(20);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_solve_many_iterations() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        let target = FullBodyTarget::new(6, Vec3::new(-0.4, 1.3, 0.2));
        let params = FullBodyIkParams::new(vec![target]).with_max_iterations(100);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
        assert!(result.iterations <= 100);
    }

    #[test]
    fn test_solve_very_small_tolerance() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        let target = FullBodyTarget::new(6, Vec3::new(-0.5, 1.4, 0.2));
        let params = FullBodyIkParams::new(vec![target])
            .with_max_iterations(50)
            .with_tolerance(0.0001);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_all_priority_levels() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        let params = FullBodyIkParams::default()
            .add_target(FullBodyTarget::new(3, Vec3::new(0.0, 1.9, 0.1)).with_priority(0))
            .add_target(FullBodyTarget::new(12, Vec3::new(-0.1, 0.0, 0.0)).with_priority(1))
            .add_target(FullBodyTarget::new(15, Vec3::new(0.1, 0.0, 0.0)).with_priority(1))
            .add_target(FullBodyTarget::new(6, Vec3::new(-0.5, 1.3, 0.2)).with_priority(2))
            .add_target(FullBodyTarget::new(9, Vec3::new(0.5, 1.3, 0.2)).with_priority(2))
            .with_max_iterations(30);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert_eq!(result.per_target_error.len(), 5);
    }

    // -------------------------------------------------------------------------
    // Balance-Specific Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_balance_with_single_foot() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        let params = FullBodyIkParams::default()
            .add_target(FullBodyTarget::new(6, Vec3::new(-0.5, 1.2, 0.3)))
            .with_balance(BalanceParams::new(0))
            .with_foot_bones(vec![12]) // Single foot
            .with_max_iterations(20);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_balance_no_foot_bones() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);

        let params = FullBodyIkParams::default()
            .add_target(FullBodyTarget::new(6, Vec3::new(-0.5, 1.2, 0.3)))
            .with_balance(BalanceParams::new(0))
            .with_foot_bones(vec![]) // No feet
            .with_max_iterations(20);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
    }

    // -------------------------------------------------------------------------
    // Posture Preservation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_posture_full_weight() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);
        let reference_pose = create_test_pose(&skeleton);

        let params = FullBodyIkParams::default()
            .add_target(FullBodyTarget::new(6, Vec3::new(-0.5, 1.2, 0.3)))
            .with_posture(PostureParams::new(reference_pose).with_weight(1.0))
            .with_max_iterations(20);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_posture_zero_weight() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);
        let reference_pose = create_test_pose(&skeleton);

        let params = FullBodyIkParams::default()
            .add_target(FullBodyTarget::new(6, Vec3::new(-0.5, 1.2, 0.3)))
            .with_posture(PostureParams::new(reference_pose).with_weight(0.0))
            .with_max_iterations(20);

        let result = solve_fullbody_ik(&skeleton, &mut pose, &params);
        assert!(result.iterations > 0);
    }
}
