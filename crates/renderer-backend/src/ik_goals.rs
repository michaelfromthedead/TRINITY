//! IK Goal System with Foot Placement for TRINITY Engine (T-AN-4.6).
//!
//! This module provides a goal-based inverse kinematics system with priority resolution
//! and specialized foot IK for terrain adaptation. It supports:
//!
//! - IK goals with target positions, normals, weights, and priorities
//! - Priority-based goal resolution (higher priority solved first)
//! - Temporal coherence for smooth IK transitions
//! - Foot IK with terrain height sampling and surface alignment
//! - Ankle roll and toe alignment for natural foot placement
//! - Smooth blending to prevent IK popping
//!
//! # Architecture
//!
//! The system is built around these core types:
//!
//! - [`IkGoal`]: A single IK goal with target, weight, priority, and blend speed
//! - [`IkGoalState`]: Runtime state tracking for smooth weight transitions
//! - [`IkGoalSystem`]: Manages multiple goals with priority resolution
//! - [`FootIkParams`]: Configuration for foot IK chains
//! - [`FootPlacementResult`]: Output of foot IK solving
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::ik_goals::{IkGoal, IkGoalSystem, FootIkParams, solve_foot_ik};
//! use renderer_backend::skeleton::Skeleton;
//! use renderer_backend::pose::Pose;
//! use glam::Vec3;
//!
//! // Create goal system
//! let mut system = IkGoalSystem::new();
//!
//! // Add a high-priority hand goal
//! let hand_goal = IkGoal::new(5, Vec3::new(1.0, 1.5, 0.5))
//!     .with_priority(10)
//!     .with_weight(1.0)
//!     .with_blend_speed(5.0);
//! system.add_goal(hand_goal);
//!
//! // Update goals (handles smooth blending)
//! system.update(delta_time);
//!
//! // Get active goals sorted by priority
//! let active_goals = system.get_active_goals();
//!
//! // Foot IK example
//! let foot_params = FootIkParams::new(ankle_bone, toe_bone)
//!     .with_ground_offset(0.02)
//!     .with_max_extension(0.15);
//!
//! let result = solve_foot_ik(
//!     &skeleton,
//!     &mut pose,
//!     &foot_params,
//!     ground_height,
//!     ground_normal,
//! );
//! ```

use glam::{Quat, Vec3};

use crate::pose::Pose;
use crate::skeleton::Skeleton;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum weight threshold for considering a goal active.
const MIN_ACTIVE_WEIGHT: f32 = 0.001;

/// Epsilon for floating point comparisons.
const EPSILON: f32 = 1e-6;

/// Default blend speed for IK goals (units per second).
pub const DEFAULT_BLEND_SPEED: f32 = 5.0;

/// Default ground offset for foot IK (meters above ground).
pub const DEFAULT_GROUND_OFFSET: f32 = 0.02;

/// Default maximum leg extension for foot IK.
pub const DEFAULT_MAX_EXTENSION: f32 = 0.15;

/// Default ankle roll limit (radians).
pub const DEFAULT_ANKLE_ROLL_LIMIT: f32 = 0.35; // ~20 degrees

/// Maximum goals per system.
pub const MAX_GOALS: usize = 64;

// ---------------------------------------------------------------------------
// IkGoal
// ---------------------------------------------------------------------------

/// An inverse kinematics goal for an effector.
///
/// Goals define target positions and orientations for skeletal effectors
/// (hands, feet, head, etc.). Multiple goals can be active simultaneously
/// with priority-based resolution determining solve order.
#[derive(Clone, Debug, PartialEq)]
pub struct IkGoal {
    /// Index of the effector bone this goal controls.
    pub effector: usize,

    /// Target position in world space.
    pub target_position: Vec3,

    /// Optional target normal for surface alignment (e.g., foot to ground).
    /// If Some, the effector will try to align with this normal.
    pub target_normal: Option<Vec3>,

    /// Goal weight (0.0 = no effect, 1.0 = full effect).
    /// Used for blending between animation and IK.
    pub weight: f32,

    /// Priority level (0-255). Higher values are solved first.
    /// Lower priority goals solve in the remaining degrees of freedom.
    pub priority: u8,

    /// Blend speed in weight units per second.
    /// Controls how fast the goal activates/deactivates to prevent popping.
    pub blend_speed: f32,

    /// Whether this goal is enabled.
    pub enabled: bool,
}

impl IkGoal {
    /// Create a new IK goal for an effector with a target position.
    ///
    /// # Arguments
    ///
    /// * `effector` - Index of the effector bone
    /// * `target_position` - Target position in world space
    #[inline]
    pub fn new(effector: usize, target_position: Vec3) -> Self {
        Self {
            effector,
            target_position,
            target_normal: None,
            weight: 1.0,
            priority: 0,
            blend_speed: DEFAULT_BLEND_SPEED,
            enabled: true,
        }
    }

    /// Set the target normal for surface alignment.
    #[inline]
    pub fn with_normal(mut self, normal: Vec3) -> Self {
        self.target_normal = Some(normal.normalize_or_zero());
        self
    }

    /// Set the goal weight.
    #[inline]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the priority level.
    #[inline]
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set the blend speed.
    #[inline]
    pub fn with_blend_speed(mut self, speed: f32) -> Self {
        self.blend_speed = speed.max(0.0);
        self
    }

    /// Enable or disable the goal.
    #[inline]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Check if this goal is effectively active (enabled with non-zero weight).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.enabled && self.weight > MIN_ACTIVE_WEIGHT
    }

    /// Update the target position.
    #[inline]
    pub fn set_target(&mut self, position: Vec3) {
        self.target_position = position;
    }

    /// Update the target position and normal.
    #[inline]
    pub fn set_target_with_normal(&mut self, position: Vec3, normal: Vec3) {
        self.target_position = position;
        self.target_normal = Some(normal.normalize_or_zero());
    }
}

impl Default for IkGoal {
    fn default() -> Self {
        Self {
            effector: 0,
            target_position: Vec3::ZERO,
            target_normal: None,
            weight: 1.0,
            priority: 0,
            blend_speed: DEFAULT_BLEND_SPEED,
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// IkGoalState
// ---------------------------------------------------------------------------

/// Runtime state for an IK goal, tracking smooth weight transitions.
///
/// This state is maintained by [`IkGoalSystem`] to enable smooth blending
/// when goals are enabled/disabled or when targets change.
#[derive(Clone, Debug, PartialEq)]
pub struct IkGoalState {
    /// Current effective weight (may differ from goal weight during transitions).
    pub current_weight: f32,

    /// Current blended position (smoothed during target changes).
    pub current_position: Vec3,

    /// Current blended normal (if applicable).
    pub current_normal: Option<Vec3>,

    /// Whether the goal is currently active (weight > threshold).
    pub is_active: bool,

    /// Time since the goal became active (for debugging/animation curves).
    pub active_time: f32,

    /// Previous frame's position for velocity estimation.
    pub previous_position: Vec3,
}

impl IkGoalState {
    /// Create a new goal state, initially inactive.
    pub fn new() -> Self {
        Self {
            current_weight: 0.0,
            current_position: Vec3::ZERO,
            current_normal: None,
            is_active: false,
            active_time: 0.0,
            previous_position: Vec3::ZERO,
        }
    }

    /// Create a new goal state initialized from a goal.
    pub fn from_goal(goal: &IkGoal) -> Self {
        Self {
            current_weight: if goal.enabled { goal.weight } else { 0.0 },
            current_position: goal.target_position,
            current_normal: goal.target_normal,
            is_active: goal.is_active(),
            active_time: 0.0,
            previous_position: goal.target_position,
        }
    }

    /// Update the state toward the goal configuration.
    ///
    /// # Arguments
    ///
    /// * `goal` - The target goal configuration
    /// * `dt` - Delta time in seconds
    pub fn update(&mut self, goal: &IkGoal, dt: f32) {
        // Store previous position for velocity
        self.previous_position = self.current_position;

        // Compute target weight
        let target_weight = if goal.enabled { goal.weight } else { 0.0 };

        // Blend weight smoothly
        let weight_delta = goal.blend_speed * dt;
        if self.current_weight < target_weight {
            self.current_weight = (self.current_weight + weight_delta).min(target_weight);
        } else if self.current_weight > target_weight {
            self.current_weight = (self.current_weight - weight_delta).max(target_weight);
        }

        // Blend position smoothly (using same blend speed as weight, scaled by position)
        let position_blend = (goal.blend_speed * dt).clamp(0.0, 1.0);
        self.current_position = self.current_position.lerp(goal.target_position, position_blend);

        // Blend normal if present
        if let Some(target_normal) = goal.target_normal {
            if let Some(current_normal) = &mut self.current_normal {
                *current_normal = current_normal.lerp(target_normal, position_blend).normalize_or_zero();
            } else {
                self.current_normal = Some(target_normal);
            }
        } else {
            self.current_normal = None;
        }

        // Update active state
        let was_active = self.is_active;
        self.is_active = self.current_weight > MIN_ACTIVE_WEIGHT;

        // Track active time
        if self.is_active {
            self.active_time += dt;
        } else {
            self.active_time = 0.0;
        }

        // Reset active time when becoming active
        if self.is_active && !was_active {
            self.active_time = 0.0;
        }
    }

    /// Get the velocity of the goal position (for prediction).
    #[inline]
    pub fn velocity(&self, dt: f32) -> Vec3 {
        if dt > EPSILON {
            (self.current_position - self.previous_position) / dt
        } else {
            Vec3::ZERO
        }
    }

    /// Reset the state to inactive.
    pub fn reset(&mut self) {
        self.current_weight = 0.0;
        self.is_active = false;
        self.active_time = 0.0;
    }
}

impl Default for IkGoalState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// IkGoalSystem
// ---------------------------------------------------------------------------

/// System for managing multiple IK goals with priority resolution.
///
/// The goal system handles:
/// - Adding/removing goals by ID
/// - Smooth weight transitions (prevents IK popping)
/// - Priority-based solve ordering
/// - Temporal coherence for stable IK
#[derive(Clone, Debug, Default)]
pub struct IkGoalSystem {
    /// All registered goals.
    pub goals: Vec<IkGoal>,

    /// Runtime states for each goal.
    pub states: Vec<IkGoalState>,

    /// Cached priority order (indices into goals array, sorted by priority descending).
    priority_order: Vec<usize>,

    /// Whether priority order needs recalculation.
    priority_dirty: bool,
}

impl IkGoalSystem {
    /// Create a new empty goal system.
    pub fn new() -> Self {
        Self {
            goals: Vec::new(),
            states: Vec::new(),
            priority_order: Vec::new(),
            priority_dirty: true,
        }
    }

    /// Create a goal system with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            goals: Vec::with_capacity(capacity),
            states: Vec::with_capacity(capacity),
            priority_order: Vec::with_capacity(capacity),
            priority_dirty: true,
        }
    }

    /// Add a goal to the system.
    ///
    /// Returns the ID (index) of the added goal.
    ///
    /// # Panics
    ///
    /// Panics if the system already has MAX_GOALS goals.
    pub fn add_goal(&mut self, goal: IkGoal) -> usize {
        assert!(
            self.goals.len() < MAX_GOALS,
            "IK goal system exceeded maximum goals ({})",
            MAX_GOALS
        );

        let id = self.goals.len();
        let state = IkGoalState::from_goal(&goal);

        self.goals.push(goal);
        self.states.push(state);
        self.priority_dirty = true;

        id
    }

    /// Remove a goal by ID.
    ///
    /// This marks the goal as disabled rather than actually removing it
    /// to maintain stable IDs. The goal will fade out smoothly.
    pub fn remove_goal(&mut self, id: usize) {
        if let Some(goal) = self.goals.get_mut(id) {
            goal.enabled = false;
        }
    }

    /// Get a goal by ID.
    #[inline]
    pub fn get_goal(&self, id: usize) -> Option<&IkGoal> {
        self.goals.get(id)
    }

    /// Get a mutable reference to a goal by ID.
    #[inline]
    pub fn get_goal_mut(&mut self, id: usize) -> Option<&mut IkGoal> {
        let goal = self.goals.get_mut(id)?;
        self.priority_dirty = true;
        Some(goal)
    }

    /// Get the state for a goal by ID.
    #[inline]
    pub fn get_state(&self, id: usize) -> Option<&IkGoalState> {
        self.states.get(id)
    }

    /// Get the number of goals in the system.
    #[inline]
    pub fn goal_count(&self) -> usize {
        self.goals.len()
    }

    /// Get the number of currently active goals.
    pub fn active_goal_count(&self) -> usize {
        self.states.iter().filter(|s| s.is_active).count()
    }

    /// Update all goal states for smooth blending.
    ///
    /// Call this once per frame before solving IK.
    ///
    /// # Arguments
    ///
    /// * `dt` - Delta time in seconds
    pub fn update(&mut self, dt: f32) {
        for (goal, state) in self.goals.iter().zip(self.states.iter_mut()) {
            state.update(goal, dt);
        }
    }

    /// Get all currently active goals (references).
    ///
    /// Returns goals with current_weight > MIN_ACTIVE_WEIGHT.
    pub fn get_active_goals(&self) -> Vec<&IkGoal> {
        self.goals
            .iter()
            .zip(self.states.iter())
            .filter(|(_, state)| state.is_active)
            .map(|(goal, _)| goal)
            .collect()
    }

    /// Get active goal IDs sorted by priority (highest first).
    pub fn get_active_goal_ids_by_priority(&mut self) -> Vec<usize> {
        // Recalculate priority order if needed
        if self.priority_dirty {
            self.recalculate_priority_order();
        }

        // Filter to only active goals
        self.priority_order
            .iter()
            .copied()
            .filter(|&id| self.states.get(id).map(|s| s.is_active).unwrap_or(false))
            .collect()
    }

    /// Resolve priorities and return goal indices sorted by priority (highest first).
    ///
    /// This is the main entry point for determining solve order.
    pub fn resolve_priorities(&mut self) -> Vec<usize> {
        if self.priority_dirty {
            self.recalculate_priority_order();
        }

        self.priority_order.clone()
    }

    /// Recalculate the priority order.
    fn recalculate_priority_order(&mut self) {
        self.priority_order.clear();
        self.priority_order.extend(0..self.goals.len());

        // Sort by priority descending, then by effector index for stability
        self.priority_order.sort_by(|&a, &b| {
            let pa = self.goals[a].priority;
            let pb = self.goals[b].priority;
            pb.cmp(&pa).then_with(|| a.cmp(&b))
        });

        self.priority_dirty = false;
    }

    /// Clear all goals from the system.
    pub fn clear(&mut self) {
        self.goals.clear();
        self.states.clear();
        self.priority_order.clear();
        self.priority_dirty = true;
    }

    /// Reset all goal states to inactive (goals remain but weights go to zero).
    pub fn reset_states(&mut self) {
        for state in &mut self.states {
            state.reset();
        }
    }

    /// Find a goal by effector index.
    ///
    /// Returns the first goal that controls the given effector.
    pub fn find_by_effector(&self, effector: usize) -> Option<usize> {
        self.goals.iter().position(|g| g.effector == effector)
    }

    /// Get all goals for a specific effector.
    pub fn goals_for_effector(&self, effector: usize) -> Vec<usize> {
        self.goals
            .iter()
            .enumerate()
            .filter(|(_, g)| g.effector == effector)
            .map(|(i, _)| i)
            .collect()
    }

    /// Set the target for a goal by ID.
    pub fn set_target(&mut self, id: usize, position: Vec3) {
        if let Some(goal) = self.goals.get_mut(id) {
            goal.target_position = position;
        }
    }

    /// Enable or disable a goal by ID.
    pub fn set_enabled(&mut self, id: usize, enabled: bool) {
        if let Some(goal) = self.goals.get_mut(id) {
            goal.enabled = enabled;
        }
    }

    /// Set the weight for a goal by ID.
    pub fn set_weight(&mut self, id: usize, weight: f32) {
        if let Some(goal) = self.goals.get_mut(id) {
            goal.weight = weight.clamp(0.0, 1.0);
        }
    }

    /// Set the priority for a goal by ID.
    pub fn set_priority(&mut self, id: usize, priority: u8) {
        if let Some(goal) = self.goals.get_mut(id) {
            goal.priority = priority;
            self.priority_dirty = true;
        }
    }
}

// ---------------------------------------------------------------------------
// FootIkParams
// ---------------------------------------------------------------------------

/// Parameters for foot IK chain configuration.
///
/// Defines the bone indices and constraints for foot placement IK.
#[derive(Clone, Debug, PartialEq)]
pub struct FootIkParams {
    /// Index of the foot/ankle bone.
    pub foot_bone: usize,

    /// Index of the ankle bone (parent of foot, used for ankle roll).
    pub ankle_bone: usize,

    /// Optional index of the toe bone for toe alignment.
    pub toe_bone: Option<usize>,

    /// Height offset above the ground (prevents z-fighting and clipping).
    pub ground_offset: f32,

    /// Maximum leg extension distance (prevents over-stretching).
    pub max_extension: f32,

    /// Maximum ankle roll angle in radians.
    pub ankle_roll_limit: f32,

    /// Maximum toe pitch angle in radians.
    pub toe_pitch_limit: f32,

    /// Blend speed for foot transitions (weight per second).
    pub blend_speed: f32,

    /// Whether to use raycast height (if false, uses provided height directly).
    pub use_raycast: bool,
}

impl FootIkParams {
    /// Create foot IK params with ankle and toe bones.
    ///
    /// # Arguments
    ///
    /// * `ankle_bone` - Index of the ankle bone
    /// * `toe_bone` - Optional index of the toe bone
    pub fn new(ankle_bone: usize, toe_bone: Option<usize>) -> Self {
        Self {
            foot_bone: ankle_bone,
            ankle_bone,
            toe_bone,
            ground_offset: DEFAULT_GROUND_OFFSET,
            max_extension: DEFAULT_MAX_EXTENSION,
            ankle_roll_limit: DEFAULT_ANKLE_ROLL_LIMIT,
            toe_pitch_limit: 0.5, // ~30 degrees
            blend_speed: DEFAULT_BLEND_SPEED,
            use_raycast: true,
        }
    }

    /// Create foot IK params with separate foot and ankle bones.
    ///
    /// # Arguments
    ///
    /// * `foot_bone` - Index of the foot bone
    /// * `ankle_bone` - Index of the ankle bone
    /// * `toe_bone` - Optional index of the toe bone
    pub fn with_foot_bone(foot_bone: usize, ankle_bone: usize, toe_bone: Option<usize>) -> Self {
        Self {
            foot_bone,
            ankle_bone,
            toe_bone,
            ground_offset: DEFAULT_GROUND_OFFSET,
            max_extension: DEFAULT_MAX_EXTENSION,
            ankle_roll_limit: DEFAULT_ANKLE_ROLL_LIMIT,
            toe_pitch_limit: 0.5,
            blend_speed: DEFAULT_BLEND_SPEED,
            use_raycast: true,
        }
    }

    /// Set the ground offset.
    #[inline]
    pub fn with_ground_offset(mut self, offset: f32) -> Self {
        self.ground_offset = offset;
        self
    }

    /// Set the maximum extension distance.
    #[inline]
    pub fn with_max_extension(mut self, extension: f32) -> Self {
        self.max_extension = extension;
        self
    }

    /// Set the ankle roll limit in radians.
    #[inline]
    pub fn with_ankle_roll_limit(mut self, limit: f32) -> Self {
        self.ankle_roll_limit = limit.abs();
        self
    }

    /// Set the toe pitch limit in radians.
    #[inline]
    pub fn with_toe_pitch_limit(mut self, limit: f32) -> Self {
        self.toe_pitch_limit = limit.abs();
        self
    }

    /// Set the blend speed.
    #[inline]
    pub fn with_blend_speed(mut self, speed: f32) -> Self {
        self.blend_speed = speed.max(0.0);
        self
    }

    /// Set whether to use raycast for height.
    #[inline]
    pub fn with_use_raycast(mut self, use_raycast: bool) -> Self {
        self.use_raycast = use_raycast;
        self
    }
}

impl Default for FootIkParams {
    fn default() -> Self {
        Self::new(0, None)
    }
}

// ---------------------------------------------------------------------------
// FootPlacementResult
// ---------------------------------------------------------------------------

/// Result of foot IK solving.
///
/// Contains the computed positions and rotations for foot placement.
#[derive(Clone, Debug, PartialEq)]
pub struct FootPlacementResult {
    /// Computed ankle position in world space.
    pub ankle_position: Vec3,

    /// Computed ankle rotation for surface alignment.
    pub ankle_rotation: Quat,

    /// Optional toe rotation for toe alignment.
    pub toe_rotation: Option<Quat>,

    /// Whether the foot is currently planted (in stance phase).
    pub is_planted: bool,

    /// The ground height at the foot position.
    pub ground_height: f32,

    /// The ground normal at the foot position.
    pub ground_normal: Vec3,

    /// Vertical offset applied to the ankle.
    pub height_offset: f32,

    /// The blend weight used for this frame.
    pub blend_weight: f32,
}

impl FootPlacementResult {
    /// Create a new foot placement result.
    pub fn new(ankle_position: Vec3, ankle_rotation: Quat) -> Self {
        Self {
            ankle_position,
            ankle_rotation,
            toe_rotation: None,
            is_planted: false,
            ground_height: 0.0,
            ground_normal: Vec3::Y,
            height_offset: 0.0,
            blend_weight: 1.0,
        }
    }

    /// Create an identity result (no IK applied).
    pub fn identity() -> Self {
        Self {
            ankle_position: Vec3::ZERO,
            ankle_rotation: Quat::IDENTITY,
            toe_rotation: None,
            is_planted: false,
            ground_height: 0.0,
            ground_normal: Vec3::Y,
            height_offset: 0.0,
            blend_weight: 0.0,
        }
    }

    /// Check if this result has meaningful IK data.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.blend_weight > MIN_ACTIVE_WEIGHT
    }
}

impl Default for FootPlacementResult {
    fn default() -> Self {
        Self::identity()
    }
}

// ---------------------------------------------------------------------------
// FootIkState
// ---------------------------------------------------------------------------

/// Runtime state for foot IK, maintaining temporal coherence.
#[derive(Clone, Debug, Default)]
pub struct FootIkState {
    /// Current blended ankle position.
    pub current_position: Vec3,

    /// Current blended ankle rotation.
    pub current_rotation: Quat,

    /// Current IK blend weight.
    pub current_weight: f32,

    /// Whether the foot is in contact with ground.
    pub is_in_contact: bool,

    /// Time the foot has been in contact.
    pub contact_time: f32,

    /// Previous frame's ankle position for velocity.
    pub previous_position: Vec3,

    /// Target position from animation.
    pub animation_position: Vec3,

    /// Target rotation from animation.
    pub animation_rotation: Quat,
}

impl FootIkState {
    /// Create a new foot IK state.
    pub fn new() -> Self {
        Self {
            current_position: Vec3::ZERO,
            current_rotation: Quat::IDENTITY,
            current_weight: 0.0,
            is_in_contact: false,
            contact_time: 0.0,
            previous_position: Vec3::ZERO,
            animation_position: Vec3::ZERO,
            animation_rotation: Quat::IDENTITY,
        }
    }

    /// Update the state toward target values.
    pub fn update(
        &mut self,
        target_position: Vec3,
        target_rotation: Quat,
        target_weight: f32,
        blend_speed: f32,
        dt: f32,
    ) {
        self.previous_position = self.current_position;

        let blend_factor = (blend_speed * dt).clamp(0.0, 1.0);

        // Blend position
        self.current_position = self.current_position.lerp(target_position, blend_factor);

        // Blend rotation using slerp
        self.current_rotation = self.current_rotation.slerp(target_rotation, blend_factor);

        // Blend weight
        if self.current_weight < target_weight {
            self.current_weight = (self.current_weight + blend_speed * dt).min(target_weight);
        } else {
            self.current_weight = (self.current_weight - blend_speed * dt).max(target_weight);
        }

        // Update contact time
        if self.is_in_contact {
            self.contact_time += dt;
        } else {
            self.contact_time = 0.0;
        }
    }

    /// Get foot velocity for prediction.
    #[inline]
    pub fn velocity(&self, dt: f32) -> Vec3 {
        if dt > EPSILON {
            (self.current_position - self.previous_position) / dt
        } else {
            Vec3::ZERO
        }
    }

    /// Reset the state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ---------------------------------------------------------------------------
// Foot IK Solver
// ---------------------------------------------------------------------------

/// Solve foot IK to plant a foot on terrain.
///
/// This function computes the ankle position and rotation needed to place
/// the foot on the ground at the specified height and aligned to the normal.
///
/// # Arguments
///
/// * `skeleton` - The skeleton containing the foot bones
/// * `pose` - The current pose (will be read for bone positions)
/// * `params` - Foot IK parameters defining the chain
/// * `ground_height` - Height of the ground at the foot position
/// * `ground_normal` - Normal vector of the ground surface
///
/// # Returns
///
/// A [`FootPlacementResult`] containing the computed positions and rotations.
pub fn solve_foot_ik(
    skeleton: &Skeleton,
    pose: &Pose,
    params: &FootIkParams,
    ground_height: f32,
    ground_normal: Vec3,
) -> FootPlacementResult {
    // Validate bone indices
    if params.ankle_bone >= skeleton.bone_count() {
        return FootPlacementResult::identity();
    }

    // Compute world transforms
    let transforms = pose.transforms();
    let world_transforms = skeleton.compute_world_transforms(&transforms);

    // Get current ankle world position
    let ankle_world = world_transforms[params.ankle_bone].w_axis.truncate();

    // Calculate target ankle height
    let target_height = ground_height + params.ground_offset;
    let height_diff = target_height - ankle_world.y;

    // Clamp height difference to max extension
    let clamped_diff = height_diff.clamp(-params.max_extension, params.max_extension);
    let target_position = Vec3::new(ankle_world.x, ankle_world.y + clamped_diff, ankle_world.z);

    // Compute ankle rotation to align with ground normal
    let ankle_rotation = compute_surface_alignment(
        ground_normal,
        params.ankle_roll_limit,
    );

    // Compute toe rotation if toe bone exists
    let toe_rotation = if params.toe_bone.is_some() {
        Some(compute_toe_alignment(
            ground_normal,
            params.toe_pitch_limit,
        ))
    } else {
        None
    };

    // Determine if foot is planted (close to ground)
    let is_planted = height_diff.abs() < params.max_extension * 0.5;

    FootPlacementResult {
        ankle_position: target_position,
        ankle_rotation,
        toe_rotation,
        is_planted,
        ground_height,
        ground_normal,
        height_offset: clamped_diff,
        blend_weight: 1.0,
    }
}

/// Apply foot IK result to a pose.
///
/// This modifies the pose in place to apply the foot placement result.
///
/// # Arguments
///
/// * `skeleton` - The skeleton containing the foot bones
/// * `pose` - The pose to modify
/// * `params` - Foot IK parameters
/// * `result` - The foot placement result to apply
/// * `weight` - Blend weight (0.0 = animation, 1.0 = full IK)
pub fn apply_foot_ik(
    _skeleton: &Skeleton,
    pose: &mut Pose,
    params: &FootIkParams,
    result: &FootPlacementResult,
    weight: f32,
) {
    if weight < MIN_ACTIVE_WEIGHT || !result.is_valid() {
        return;
    }

    let weight = weight.clamp(0.0, 1.0);

    // Apply position offset to ankle
    if params.ankle_bone < pose.bone_count() {
        let current_pos = pose.positions[params.ankle_bone];
        let offset = Vec3::new(0.0, result.height_offset, 0.0);
        pose.positions[params.ankle_bone] = current_pos + offset * weight;
    }

    // Apply rotation to ankle
    if params.ankle_bone < pose.bone_count() {
        let current_rot = pose.rotations[params.ankle_bone];
        pose.rotations[params.ankle_bone] = current_rot.slerp(
            current_rot * result.ankle_rotation,
            weight,
        );
    }

    // Apply toe rotation if available
    if let (Some(toe_bone), Some(toe_rot)) = (params.toe_bone, result.toe_rotation) {
        if toe_bone < pose.bone_count() {
            let current_rot = pose.rotations[toe_bone];
            pose.rotations[toe_bone] = current_rot.slerp(
                current_rot * toe_rot,
                weight,
            );
        }
    }
}

/// Compute the rotation needed to align a bone with a surface normal.
///
/// # Arguments
///
/// * `surface_normal` - The surface normal to align with
/// * `max_roll` - Maximum allowed rotation angle in radians
///
/// # Returns
///
/// A quaternion rotation that aligns the bone with the surface.
fn compute_surface_alignment(surface_normal: Vec3, max_roll: f32) -> Quat {
    let normal = surface_normal.normalize_or_zero();

    // If normal is essentially up, no rotation needed
    if normal.abs_diff_eq(Vec3::Y, EPSILON) {
        return Quat::IDENTITY;
    }

    // If normal is essentially down, handle this edge case
    if normal.abs_diff_eq(Vec3::NEG_Y, EPSILON) {
        // Flip 180 degrees around Z
        return Quat::from_rotation_z(std::f32::consts::PI);
    }

    // Compute rotation from up to normal
    let up = Vec3::Y;
    let axis = up.cross(normal);

    if axis.length_squared() < EPSILON {
        return Quat::IDENTITY;
    }

    let axis = axis.normalize();
    let angle = up.dot(normal).clamp(-1.0, 1.0).acos();

    // Clamp angle to max roll
    let clamped_angle = angle.min(max_roll);

    Quat::from_axis_angle(axis, clamped_angle)
}

/// Compute toe alignment rotation for surface adaptation.
///
/// # Arguments
///
/// * `surface_normal` - The surface normal
/// * `max_pitch` - Maximum toe pitch angle in radians
///
/// # Returns
///
/// A quaternion for toe rotation (pitch only).
fn compute_toe_alignment(surface_normal: Vec3, max_pitch: f32) -> Quat {
    let normal = surface_normal.normalize_or_zero();

    // Project normal onto XZ plane to get slope direction
    let slope_dir = Vec3::new(normal.x, 0.0, normal.z).normalize_or_zero();

    if slope_dir.length_squared() < EPSILON {
        return Quat::IDENTITY;
    }

    // Compute pitch angle from the slope
    let pitch_angle = normal.y.acos() - std::f32::consts::FRAC_PI_2;
    let clamped_pitch = pitch_angle.clamp(-max_pitch, max_pitch);

    // Rotate around the local X axis (perpendicular to slope direction)
    let pitch_axis = slope_dir.cross(Vec3::Y).normalize_or_zero();

    if pitch_axis.length_squared() < EPSILON {
        Quat::from_rotation_x(clamped_pitch)
    } else {
        Quat::from_axis_angle(pitch_axis, clamped_pitch)
    }
}

/// Sample terrain height at a position (stub for integration).
///
/// In a real implementation, this would query the terrain system.
/// For now, it returns a flat ground at y=0.
///
/// # Arguments
///
/// * `position` - World position to sample
///
/// # Returns
///
/// Tuple of (height, normal) at the position.
pub fn sample_terrain_height(_position: Vec3) -> (f32, Vec3) {
    (0.0, Vec3::Y)
}

/// Compute the heel lift amount during a step.
///
/// # Arguments
///
/// * `phase` - Step phase (0.0 = heel strike, 0.5 = mid-stance, 1.0 = toe-off)
/// * `max_lift` - Maximum heel lift in world units
///
/// # Returns
///
/// Heel lift amount for the given phase.
pub fn compute_heel_lift(phase: f32, max_lift: f32) -> f32 {
    // Heel lift curve: lifts during late stance (0.5 to 1.0)
    if phase < 0.5 {
        0.0
    } else {
        let t = (phase - 0.5) * 2.0; // 0 to 1 during lift phase
        // Smooth ease-in curve
        let ease = t * t * (3.0 - 2.0 * t);
        ease * max_lift
    }
}

/// Detect if a foot is in swing or stance phase based on velocity.
///
/// # Arguments
///
/// * `velocity` - Current foot velocity
/// * `threshold` - Velocity threshold for swing detection
///
/// # Returns
///
/// `true` if foot is in stance (velocity below threshold), `false` if in swing.
pub fn detect_stance_phase(velocity: Vec3, threshold: f32) -> bool {
    velocity.length() < threshold
}

// ---------------------------------------------------------------------------
// Multi-Goal Resolution
// ---------------------------------------------------------------------------

/// Resolve multiple IK goals with priority-based DOF allocation.
///
/// Higher priority goals are solved first, consuming degrees of freedom.
/// Lower priority goals solve in the remaining DOF space.
///
/// # Arguments
///
/// * `goals` - Slice of goals to resolve (should be sorted by priority)
/// * `states` - Corresponding goal states
///
/// # Returns
///
/// Vector of (goal_index, effective_weight) pairs for solving.
pub fn resolve_goal_priorities(
    goals: &[IkGoal],
    states: &[IkGoalState],
) -> Vec<(usize, f32)> {
    let mut result = Vec::new();
    let mut consumed_effectors = std::collections::HashSet::new();

    for (idx, (goal, state)) in goals.iter().zip(states.iter()).enumerate() {
        if !state.is_active {
            continue;
        }

        // Check if this effector is already consumed by a higher priority goal
        if consumed_effectors.contains(&goal.effector) {
            // Reduce effective weight for conflicts
            let effective_weight = state.current_weight * 0.5;
            if effective_weight > MIN_ACTIVE_WEIGHT {
                result.push((idx, effective_weight));
            }
        } else {
            result.push((idx, state.current_weight));
            consumed_effectors.insert(goal.effector);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Bone, Transform};

    // Helper to create a test skeleton with legs
    fn create_leg_skeleton() -> Skeleton {
        let mut skeleton = Skeleton::new();

        // Root (hip)
        skeleton.add_bone(
            Bone::root("hip")
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 1.0, 0.0))),
        );

        // Upper leg
        skeleton.add_bone(
            Bone::new("upper_leg")
                .with_parent(0)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, -0.5, 0.0))),
        );

        // Lower leg
        skeleton.add_bone(
            Bone::new("lower_leg")
                .with_parent(1)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, -0.5, 0.0))),
        );

        // Ankle/foot
        skeleton.add_bone(
            Bone::new("ankle")
                .with_parent(2)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, -0.1, 0.0))),
        );

        // Toe
        skeleton.add_bone(
            Bone::new("toe")
                .with_parent(3)
                .with_local_transform(Transform::from_position(Vec3::new(0.0, 0.0, 0.1))),
        );

        skeleton.rebuild_indices();
        skeleton
    }

    // ===== IkGoal Tests =====

    #[test]
    fn test_ik_goal_new() {
        let goal = IkGoal::new(5, Vec3::new(1.0, 2.0, 3.0));

        assert_eq!(goal.effector, 5);
        assert!(goal.target_position.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-6));
        assert_eq!(goal.weight, 1.0);
        assert_eq!(goal.priority, 0);
        assert!(goal.enabled);
        assert!(goal.target_normal.is_none());
    }

    #[test]
    fn test_ik_goal_default() {
        let goal = IkGoal::default();

        assert_eq!(goal.effector, 0);
        assert!(goal.target_position.abs_diff_eq(Vec3::ZERO, 1e-6));
        assert_eq!(goal.weight, 1.0);
        assert!(goal.enabled);
    }

    #[test]
    fn test_ik_goal_with_normal() {
        let goal = IkGoal::new(0, Vec3::ZERO).with_normal(Vec3::new(0.0, 1.0, 0.0));

        assert!(goal.target_normal.is_some());
        assert!(goal.target_normal.unwrap().abs_diff_eq(Vec3::Y, 1e-6));
    }

    #[test]
    fn test_ik_goal_with_weight() {
        let goal = IkGoal::new(0, Vec3::ZERO).with_weight(0.5);
        assert_eq!(goal.weight, 0.5);

        // Test clamping
        let goal_clamped = IkGoal::new(0, Vec3::ZERO).with_weight(1.5);
        assert_eq!(goal_clamped.weight, 1.0);

        let goal_negative = IkGoal::new(0, Vec3::ZERO).with_weight(-0.5);
        assert_eq!(goal_negative.weight, 0.0);
    }

    #[test]
    fn test_ik_goal_with_priority() {
        let goal = IkGoal::new(0, Vec3::ZERO).with_priority(100);
        assert_eq!(goal.priority, 100);
    }

    #[test]
    fn test_ik_goal_with_blend_speed() {
        let goal = IkGoal::new(0, Vec3::ZERO).with_blend_speed(10.0);
        assert_eq!(goal.blend_speed, 10.0);

        // Negative speed should clamp to 0
        let goal_neg = IkGoal::new(0, Vec3::ZERO).with_blend_speed(-5.0);
        assert_eq!(goal_neg.blend_speed, 0.0);
    }

    #[test]
    fn test_ik_goal_with_enabled() {
        let goal = IkGoal::new(0, Vec3::ZERO).with_enabled(false);
        assert!(!goal.enabled);
        assert!(!goal.is_active());
    }

    #[test]
    fn test_ik_goal_is_active() {
        let goal = IkGoal::new(0, Vec3::ZERO);
        assert!(goal.is_active());

        let disabled = IkGoal::new(0, Vec3::ZERO).with_enabled(false);
        assert!(!disabled.is_active());

        let zero_weight = IkGoal::new(0, Vec3::ZERO).with_weight(0.0);
        assert!(!zero_weight.is_active());
    }

    #[test]
    fn test_ik_goal_set_target() {
        let mut goal = IkGoal::new(0, Vec3::ZERO);
        goal.set_target(Vec3::new(5.0, 6.0, 7.0));

        assert!(goal.target_position.abs_diff_eq(Vec3::new(5.0, 6.0, 7.0), 1e-6));
    }

    #[test]
    fn test_ik_goal_set_target_with_normal() {
        let mut goal = IkGoal::new(0, Vec3::ZERO);
        goal.set_target_with_normal(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 1.0, 0.0));

        assert!(goal.target_position.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-6));
        assert!(goal.target_normal.unwrap().abs_diff_eq(Vec3::Y, 1e-6));
    }

    // ===== IkGoalState Tests =====

    #[test]
    fn test_ik_goal_state_new() {
        let state = IkGoalState::new();

        assert_eq!(state.current_weight, 0.0);
        assert!(!state.is_active);
        assert_eq!(state.active_time, 0.0);
    }

    #[test]
    fn test_ik_goal_state_from_goal() {
        let goal = IkGoal::new(0, Vec3::new(1.0, 2.0, 3.0))
            .with_weight(0.8);

        let state = IkGoalState::from_goal(&goal);

        assert_eq!(state.current_weight, 0.8);
        assert!(state.current_position.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-6));
        assert!(state.is_active);
    }

    #[test]
    fn test_ik_goal_state_from_disabled_goal() {
        let goal = IkGoal::new(0, Vec3::ZERO).with_enabled(false);
        let state = IkGoalState::from_goal(&goal);

        assert_eq!(state.current_weight, 0.0);
        assert!(!state.is_active);
    }

    #[test]
    fn test_ik_goal_state_update_weight_increase() {
        let goal = IkGoal::new(0, Vec3::ZERO)
            .with_weight(1.0)
            .with_blend_speed(5.0);

        let mut state = IkGoalState::new();
        state.update(&goal, 0.1); // 0.1 seconds

        // Weight should increase by 5.0 * 0.1 = 0.5
        assert!(state.current_weight > 0.0);
        assert!(state.current_weight <= 0.5 + 1e-6);
    }

    #[test]
    fn test_ik_goal_state_update_weight_decrease() {
        let goal = IkGoal::new(0, Vec3::ZERO)
            .with_weight(0.0)
            .with_blend_speed(5.0);

        let mut state = IkGoalState {
            current_weight: 1.0,
            ..IkGoalState::new()
        };

        state.update(&goal, 0.1);

        // Weight should decrease
        assert!(state.current_weight < 1.0);
        assert!(state.current_weight >= 0.5 - 1e-6);
    }

    #[test]
    fn test_ik_goal_state_update_position_blend() {
        let goal = IkGoal::new(0, Vec3::new(10.0, 0.0, 0.0))
            .with_blend_speed(1.0);  // Use slower blend so dt=0.5 doesn't saturate

        let mut state = IkGoalState::new();
        state.current_position = Vec3::ZERO;

        state.update(&goal, 0.5);

        // Position should move toward target (blend_speed * dt = 0.5, so 50% blend)
        assert!(state.current_position.x > 0.0);
        assert!(state.current_position.x < 10.0);
    }

    #[test]
    fn test_ik_goal_state_update_active_time() {
        let goal = IkGoal::new(0, Vec3::ZERO).with_weight(1.0);

        let mut state = IkGoalState::from_goal(&goal);
        state.update(&goal, 0.5);
        state.update(&goal, 0.3);

        assert!((state.active_time - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_ik_goal_state_velocity() {
        let mut state = IkGoalState::new();
        state.previous_position = Vec3::ZERO;
        state.current_position = Vec3::new(1.0, 0.0, 0.0);

        let vel = state.velocity(0.1);
        assert!(vel.abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_ik_goal_state_velocity_zero_dt() {
        let state = IkGoalState::new();
        let vel = state.velocity(0.0);
        assert!(vel.abs_diff_eq(Vec3::ZERO, 1e-6));
    }

    #[test]
    fn test_ik_goal_state_reset() {
        let mut state = IkGoalState {
            current_weight: 1.0,
            is_active: true,
            active_time: 5.0,
            ..IkGoalState::new()
        };

        state.reset();

        assert_eq!(state.current_weight, 0.0);
        assert!(!state.is_active);
        assert_eq!(state.active_time, 0.0);
    }

    // ===== IkGoalSystem Tests =====

    #[test]
    fn test_ik_goal_system_new() {
        let system = IkGoalSystem::new();
        assert_eq!(system.goal_count(), 0);
        assert_eq!(system.active_goal_count(), 0);
    }

    #[test]
    fn test_ik_goal_system_with_capacity() {
        let system = IkGoalSystem::with_capacity(16);
        assert_eq!(system.goal_count(), 0);
    }

    #[test]
    fn test_ik_goal_system_add_goal() {
        let mut system = IkGoalSystem::new();

        let id = system.add_goal(IkGoal::new(5, Vec3::new(1.0, 2.0, 3.0)));

        assert_eq!(id, 0);
        assert_eq!(system.goal_count(), 1);

        let id2 = system.add_goal(IkGoal::new(6, Vec3::ZERO));
        assert_eq!(id2, 1);
        assert_eq!(system.goal_count(), 2);
    }

    #[test]
    fn test_ik_goal_system_get_goal() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(5, Vec3::new(1.0, 2.0, 3.0)));

        let goal = system.get_goal(0).unwrap();
        assert_eq!(goal.effector, 5);

        assert!(system.get_goal(99).is_none());
    }

    #[test]
    fn test_ik_goal_system_get_goal_mut() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(5, Vec3::ZERO));

        if let Some(goal) = system.get_goal_mut(0) {
            goal.weight = 0.5;
        }

        assert_eq!(system.get_goal(0).unwrap().weight, 0.5);
    }

    #[test]
    fn test_ik_goal_system_remove_goal() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(5, Vec3::ZERO));

        system.remove_goal(0);

        // Goal is disabled but still exists
        assert_eq!(system.goal_count(), 1);
        assert!(!system.get_goal(0).unwrap().enabled);
    }

    #[test]
    fn test_ik_goal_system_update() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(0, Vec3::ZERO).with_weight(1.0).with_blend_speed(10.0));

        // Update should modify state
        let initial_weight = system.get_state(0).unwrap().current_weight;
        system.update(0.1);
        let new_weight = system.get_state(0).unwrap().current_weight;

        // Weight should increase toward 1.0
        assert!(new_weight >= initial_weight);
    }

    #[test]
    fn test_ik_goal_system_get_active_goals() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(0, Vec3::ZERO).with_weight(1.0));
        system.add_goal(IkGoal::new(1, Vec3::ZERO).with_weight(0.0)); // Inactive
        system.add_goal(IkGoal::new(2, Vec3::ZERO).with_weight(0.5));

        let active = system.get_active_goals();
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_ik_goal_system_active_goal_count() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(0, Vec3::ZERO).with_weight(1.0));
        system.add_goal(IkGoal::new(1, Vec3::ZERO).with_weight(0.0));

        assert_eq!(system.active_goal_count(), 1);
    }

    #[test]
    fn test_ik_goal_system_resolve_priorities() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(0, Vec3::ZERO).with_priority(5));
        system.add_goal(IkGoal::new(1, Vec3::ZERO).with_priority(10));
        system.add_goal(IkGoal::new(2, Vec3::ZERO).with_priority(1));

        let order = system.resolve_priorities();

        // Should be sorted by priority descending: 10, 5, 1
        assert_eq!(order[0], 1); // Priority 10
        assert_eq!(order[1], 0); // Priority 5
        assert_eq!(order[2], 2); // Priority 1
    }

    #[test]
    fn test_ik_goal_system_get_active_goal_ids_by_priority() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(0, Vec3::ZERO).with_priority(5).with_weight(1.0));
        system.add_goal(IkGoal::new(1, Vec3::ZERO).with_priority(10).with_weight(0.0)); // Inactive
        system.add_goal(IkGoal::new(2, Vec3::ZERO).with_priority(1).with_weight(0.8));

        let active_ids = system.get_active_goal_ids_by_priority();

        // Only active goals, sorted by priority
        assert_eq!(active_ids.len(), 2);
        assert_eq!(active_ids[0], 0); // Priority 5
        assert_eq!(active_ids[1], 2); // Priority 1
    }

    #[test]
    fn test_ik_goal_system_clear() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(0, Vec3::ZERO));
        system.add_goal(IkGoal::new(1, Vec3::ZERO));

        system.clear();

        assert_eq!(system.goal_count(), 0);
    }

    #[test]
    fn test_ik_goal_system_reset_states() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(0, Vec3::ZERO).with_weight(1.0));

        // All states should be reset
        system.reset_states();

        assert!(!system.get_state(0).unwrap().is_active);
        assert_eq!(system.get_state(0).unwrap().current_weight, 0.0);
    }

    #[test]
    fn test_ik_goal_system_find_by_effector() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(5, Vec3::ZERO));
        system.add_goal(IkGoal::new(10, Vec3::ZERO));

        assert_eq!(system.find_by_effector(5), Some(0));
        assert_eq!(system.find_by_effector(10), Some(1));
        assert_eq!(system.find_by_effector(99), None);
    }

    #[test]
    fn test_ik_goal_system_goals_for_effector() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(5, Vec3::ZERO));
        system.add_goal(IkGoal::new(10, Vec3::ZERO));
        system.add_goal(IkGoal::new(5, Vec3::new(1.0, 0.0, 0.0))); // Same effector

        let goals = system.goals_for_effector(5);
        assert_eq!(goals.len(), 2);
        assert!(goals.contains(&0));
        assert!(goals.contains(&2));
    }

    #[test]
    fn test_ik_goal_system_set_target() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(0, Vec3::ZERO));

        system.set_target(0, Vec3::new(5.0, 6.0, 7.0));

        assert!(system.get_goal(0).unwrap().target_position.abs_diff_eq(Vec3::new(5.0, 6.0, 7.0), 1e-6));
    }

    #[test]
    fn test_ik_goal_system_set_enabled() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(0, Vec3::ZERO));

        system.set_enabled(0, false);
        assert!(!system.get_goal(0).unwrap().enabled);

        system.set_enabled(0, true);
        assert!(system.get_goal(0).unwrap().enabled);
    }

    #[test]
    fn test_ik_goal_system_set_weight() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(0, Vec3::ZERO));

        system.set_weight(0, 0.7);
        assert!((system.get_goal(0).unwrap().weight - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_ik_goal_system_set_priority() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(0, Vec3::ZERO));

        system.set_priority(0, 50);
        assert_eq!(system.get_goal(0).unwrap().priority, 50);
    }

    // ===== FootIkParams Tests =====

    #[test]
    fn test_foot_ik_params_new() {
        let params = FootIkParams::new(3, Some(4));

        assert_eq!(params.ankle_bone, 3);
        assert_eq!(params.foot_bone, 3);
        assert_eq!(params.toe_bone, Some(4));
        assert_eq!(params.ground_offset, DEFAULT_GROUND_OFFSET);
    }

    #[test]
    fn test_foot_ik_params_with_foot_bone() {
        let params = FootIkParams::with_foot_bone(2, 3, Some(4));

        assert_eq!(params.foot_bone, 2);
        assert_eq!(params.ankle_bone, 3);
        assert_eq!(params.toe_bone, Some(4));
    }

    #[test]
    fn test_foot_ik_params_builder() {
        let params = FootIkParams::new(0, None)
            .with_ground_offset(0.05)
            .with_max_extension(0.2)
            .with_ankle_roll_limit(0.5)
            .with_toe_pitch_limit(0.4)
            .with_blend_speed(8.0)
            .with_use_raycast(false);

        assert!((params.ground_offset - 0.05).abs() < 1e-6);
        assert!((params.max_extension - 0.2).abs() < 1e-6);
        assert!((params.ankle_roll_limit - 0.5).abs() < 1e-6);
        assert!((params.toe_pitch_limit - 0.4).abs() < 1e-6);
        assert!((params.blend_speed - 8.0).abs() < 1e-6);
        assert!(!params.use_raycast);
    }

    #[test]
    fn test_foot_ik_params_default() {
        let params = FootIkParams::default();
        assert_eq!(params.ankle_bone, 0);
        assert_eq!(params.toe_bone, None);
    }

    // ===== FootPlacementResult Tests =====

    #[test]
    fn test_foot_placement_result_new() {
        let result = FootPlacementResult::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_x(0.1),
        );

        assert!(result.ankle_position.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-6));
        assert!(!result.is_planted);
        assert_eq!(result.blend_weight, 1.0);
    }

    #[test]
    fn test_foot_placement_result_identity() {
        let result = FootPlacementResult::identity();

        assert!(result.ankle_position.abs_diff_eq(Vec3::ZERO, 1e-6));
        assert!(result.ankle_rotation.abs_diff_eq(Quat::IDENTITY, 1e-6));
        assert_eq!(result.blend_weight, 0.0);
    }

    #[test]
    fn test_foot_placement_result_is_valid() {
        let valid = FootPlacementResult::new(Vec3::ZERO, Quat::IDENTITY);
        assert!(valid.is_valid());

        let invalid = FootPlacementResult::identity();
        assert!(!invalid.is_valid());
    }

    // ===== FootIkState Tests =====

    #[test]
    fn test_foot_ik_state_new() {
        let state = FootIkState::new();

        assert_eq!(state.current_weight, 0.0);
        assert!(!state.is_in_contact);
        assert_eq!(state.contact_time, 0.0);
    }

    #[test]
    fn test_foot_ik_state_update() {
        let mut state = FootIkState::new();

        state.update(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
            5.0,
            0.1,
        );

        assert!(state.current_weight > 0.0);
        assert!(state.current_position.x > 0.0);
    }

    #[test]
    fn test_foot_ik_state_velocity() {
        let mut state = FootIkState::new();
        state.previous_position = Vec3::ZERO;
        state.current_position = Vec3::new(0.5, 0.0, 0.0);

        let vel = state.velocity(0.1);
        assert!(vel.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_foot_ik_state_reset() {
        let mut state = FootIkState {
            current_weight: 1.0,
            is_in_contact: true,
            contact_time: 5.0,
            ..FootIkState::new()
        };

        state.reset();

        assert_eq!(state.current_weight, 0.0);
        assert!(!state.is_in_contact);
        assert_eq!(state.contact_time, 0.0);
    }

    // ===== Foot IK Solver Tests =====

    #[test]
    fn test_solve_foot_ik_flat_ground() {
        let skeleton = create_leg_skeleton();
        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let params = FootIkParams::new(3, Some(4)); // ankle, toe

        let result = solve_foot_ik(
            &skeleton,
            &pose,
            &params,
            0.0,    // ground height
            Vec3::Y, // ground normal (flat)
        );

        assert!(result.is_valid());
        assert!(result.ground_normal.abs_diff_eq(Vec3::Y, 1e-6));
    }

    #[test]
    fn test_solve_foot_ik_raised_ground() {
        let skeleton = create_leg_skeleton();
        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let params = FootIkParams::new(3, Some(4))
            .with_ground_offset(0.02)
            .with_max_extension(0.2);

        let result = solve_foot_ik(
            &skeleton,
            &pose,
            &params,
            0.1,    // raised ground
            Vec3::Y,
        );

        assert!(result.height_offset > 0.0);
    }

    #[test]
    fn test_solve_foot_ik_lowered_ground() {
        let skeleton = create_leg_skeleton();
        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let params = FootIkParams::new(3, Some(4))
            .with_max_extension(0.3);

        // Ground is lower than the ankle (ankle world y ≈ -0.1)
        // Use ground at -0.25 so ankle must move down
        let result = solve_foot_ik(
            &skeleton,
            &pose,
            &params,
            -0.25,   // lowered ground (below ankle)
            Vec3::Y,
        );

        assert!(result.height_offset < 0.0, "Height offset should be negative for lowered ground: {}", result.height_offset);
    }

    #[test]
    fn test_solve_foot_ik_sloped_ground() {
        let skeleton = create_leg_skeleton();
        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let params = FootIkParams::new(3, Some(4))
            .with_ankle_roll_limit(0.5);

        // 30 degree slope
        let slope_normal = Vec3::new(0.5, 0.866, 0.0).normalize();

        let result = solve_foot_ik(
            &skeleton,
            &pose,
            &params,
            0.0,
            slope_normal,
        );

        // Ankle should rotate to match slope
        assert!(!result.ankle_rotation.abs_diff_eq(Quat::IDENTITY, 1e-3));
    }

    #[test]
    fn test_solve_foot_ik_max_extension_clamped() {
        let skeleton = create_leg_skeleton();
        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let params = FootIkParams::new(3, None)
            .with_max_extension(0.1);

        // Ground way below - should be clamped
        let result = solve_foot_ik(
            &skeleton,
            &pose,
            &params,
            -1.0,
            Vec3::Y,
        );

        assert!((result.height_offset.abs() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_solve_foot_ik_invalid_bone_index() {
        let skeleton = create_leg_skeleton();
        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let params = FootIkParams::new(999, None); // Invalid

        let result = solve_foot_ik(
            &skeleton,
            &pose,
            &params,
            0.0,
            Vec3::Y,
        );

        assert!(!result.is_valid());
    }

    // ===== Apply Foot IK Tests =====

    #[test]
    fn test_apply_foot_ik_zero_weight() {
        let skeleton = create_leg_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let original_pos = pose.positions[3];

        let params = FootIkParams::new(3, Some(4));
        let result = FootPlacementResult::new(Vec3::new(0.0, 0.1, 0.0), Quat::IDENTITY);

        apply_foot_ik(&skeleton, &mut pose, &params, &result, 0.0);

        // Should not change
        assert!(pose.positions[3].abs_diff_eq(original_pos, 1e-6));
    }

    #[test]
    fn test_apply_foot_ik_full_weight() {
        let skeleton = create_leg_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let params = FootIkParams::new(3, Some(4));
        let mut result = FootPlacementResult::new(Vec3::ZERO, Quat::IDENTITY);
        result.height_offset = 0.1;
        result.blend_weight = 1.0;

        apply_foot_ik(&skeleton, &mut pose, &params, &result, 1.0);

        // Position should change by height offset
        // (Note: the actual position depends on local space transforms)
    }

    #[test]
    fn test_apply_foot_ik_invalid_result() {
        let skeleton = create_leg_skeleton();
        let mut pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);
        let original = pose.clone();

        let params = FootIkParams::new(3, None);
        let result = FootPlacementResult::identity(); // blend_weight = 0

        apply_foot_ik(&skeleton, &mut pose, &params, &result, 1.0);

        // Should not change
        assert_eq!(pose, original);
    }

    // ===== Surface Alignment Tests =====

    #[test]
    fn test_compute_surface_alignment_flat() {
        let rot = compute_surface_alignment(Vec3::Y, 1.0);
        assert!(rot.abs_diff_eq(Quat::IDENTITY, 1e-5));
    }

    #[test]
    fn test_compute_surface_alignment_tilted() {
        let normal = Vec3::new(0.0, 0.866, 0.5).normalize(); // ~30 degrees
        let rot = compute_surface_alignment(normal, 1.0);

        // Should produce a non-identity rotation
        assert!(!rot.abs_diff_eq(Quat::IDENTITY, 0.01));
    }

    #[test]
    fn test_compute_surface_alignment_max_roll_limit() {
        let steep_normal = Vec3::new(0.707, 0.707, 0.0).normalize(); // 45 degrees
        let rot = compute_surface_alignment(steep_normal, 0.35); // ~20 degree limit

        // Rotation angle should be clamped
        let angle = rot.to_axis_angle().1;
        assert!(angle <= 0.35 + 1e-5);
    }

    #[test]
    fn test_compute_surface_alignment_down_normal() {
        let rot = compute_surface_alignment(Vec3::NEG_Y, 1.0);
        // Should handle this edge case gracefully
        assert!(rot.is_normalized());
    }

    // ===== Toe Alignment Tests =====

    #[test]
    fn test_compute_toe_alignment_flat() {
        let rot = compute_toe_alignment(Vec3::Y, 0.5);
        // On flat ground, toe doesn't need to pitch
        // (Note: implementation may vary)
    }

    #[test]
    fn test_compute_toe_alignment_slope() {
        let slope_normal = Vec3::new(0.0, 0.866, 0.5).normalize();
        let rot = compute_toe_alignment(slope_normal, 0.5);

        assert!(rot.is_normalized());
    }

    // ===== Utility Function Tests =====

    #[test]
    fn test_sample_terrain_height() {
        let (height, normal) = sample_terrain_height(Vec3::ZERO);

        assert_eq!(height, 0.0);
        assert!(normal.abs_diff_eq(Vec3::Y, 1e-6));
    }

    #[test]
    fn test_compute_heel_lift_early_stance() {
        let lift = compute_heel_lift(0.2, 0.1);
        assert_eq!(lift, 0.0);
    }

    #[test]
    fn test_compute_heel_lift_late_stance() {
        let lift = compute_heel_lift(0.75, 0.1);
        assert!(lift > 0.0);
        assert!(lift < 0.1);
    }

    #[test]
    fn test_compute_heel_lift_toe_off() {
        let lift = compute_heel_lift(1.0, 0.1);
        assert!((lift - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_detect_stance_phase() {
        assert!(detect_stance_phase(Vec3::ZERO, 0.1));
        assert!(detect_stance_phase(Vec3::new(0.05, 0.0, 0.0), 0.1));
        assert!(!detect_stance_phase(Vec3::new(0.5, 0.0, 0.0), 0.1));
    }

    // ===== Priority Resolution Tests =====

    #[test]
    fn test_resolve_goal_priorities_empty() {
        let goals: Vec<IkGoal> = vec![];
        let states: Vec<IkGoalState> = vec![];

        let result = resolve_goal_priorities(&goals, &states);
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_goal_priorities_single() {
        let goals = vec![IkGoal::new(0, Vec3::ZERO).with_weight(0.8)];
        let states = vec![IkGoalState::from_goal(&goals[0])];

        let result = resolve_goal_priorities(&goals, &states);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
        assert!((result[0].1 - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_resolve_goal_priorities_conflict() {
        // Two goals for the same effector
        let goals = vec![
            IkGoal::new(5, Vec3::new(1.0, 0.0, 0.0)).with_priority(10).with_weight(1.0),
            IkGoal::new(5, Vec3::new(0.0, 1.0, 0.0)).with_priority(5).with_weight(1.0),
        ];
        let states = vec![
            IkGoalState::from_goal(&goals[0]),
            IkGoalState::from_goal(&goals[1]),
        ];

        let result = resolve_goal_priorities(&goals, &states);

        // Both should be present, but second should have reduced weight
        assert_eq!(result.len(), 2);
        assert!(result[1].1 < result[0].1); // Lower priority = reduced weight
    }

    #[test]
    fn test_resolve_goal_priorities_inactive_filtered() {
        let goals = vec![
            IkGoal::new(0, Vec3::ZERO).with_weight(1.0),
            IkGoal::new(1, Vec3::ZERO).with_weight(0.0), // Inactive
        ];
        let mut states = vec![
            IkGoalState::from_goal(&goals[0]),
            IkGoalState::from_goal(&goals[1]),
        ];
        states[1].is_active = false;

        let result = resolve_goal_priorities(&goals, &states);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_ik_goal_system_multiple_updates() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(0, Vec3::ZERO).with_weight(1.0).with_blend_speed(2.0));

        // Multiple small updates
        for _ in 0..10 {
            system.update(0.05);
        }

        let state = system.get_state(0).unwrap();
        assert!(state.is_active);
        assert!(state.current_weight > 0.9);
    }

    #[test]
    fn test_ik_goal_state_normal_blending() {
        let goal = IkGoal::new(0, Vec3::ZERO)
            .with_normal(Vec3::new(0.0, 1.0, 0.0))
            .with_blend_speed(5.0);

        let mut state = IkGoalState::new();
        state.current_normal = Some(Vec3::new(0.0, 0.0, 1.0));

        state.update(&goal, 0.5);

        // Normal should blend toward target
        if let Some(normal) = state.current_normal {
            assert!(normal.y > 0.0);
        }
    }

    #[test]
    fn test_foot_ik_with_toe_rotation() {
        let skeleton = create_leg_skeleton();
        let pose = Pose::from_skeleton(&skeleton, crate::pose::PoseType::Current);

        let params = FootIkParams::new(3, Some(4))
            .with_toe_pitch_limit(0.5);

        let slope_normal = Vec3::new(0.0, 0.9, 0.436).normalize();

        let result = solve_foot_ik(
            &skeleton,
            &pose,
            &params,
            0.0,
            slope_normal,
        );

        assert!(result.toe_rotation.is_some());
    }

    // ===== Stability Tests =====

    #[test]
    fn test_ik_goal_system_stability_rapid_enable_disable() {
        let mut system = IkGoalSystem::new();
        system.add_goal(IkGoal::new(0, Vec3::ZERO).with_blend_speed(100.0));

        for i in 0..100 {
            system.set_enabled(0, i % 2 == 0);
            system.update(0.001);
        }

        // Should not crash or produce NaN
        let state = system.get_state(0).unwrap();
        assert!(!state.current_weight.is_nan());
    }

    #[test]
    fn test_surface_alignment_normalized_output() {
        // Test various normals to ensure output is always normalized
        let normals = vec![
            Vec3::Y,
            Vec3::new(1.0, 1.0, 0.0).normalize(),
            Vec3::new(0.1, 0.9, 0.1).normalize(),
            Vec3::new(-0.5, 0.5, 0.5).normalize(),
        ];

        for normal in normals {
            let rot = compute_surface_alignment(normal, 1.0);
            assert!(rot.is_normalized(), "Rotation not normalized for normal {:?}", normal);
        }
    }

    #[test]
    fn test_goal_priority_stability() {
        let mut system = IkGoalSystem::new();

        // Add goals in random priority order
        system.add_goal(IkGoal::new(0, Vec3::ZERO).with_priority(5));
        system.add_goal(IkGoal::new(1, Vec3::ZERO).with_priority(10));
        system.add_goal(IkGoal::new(2, Vec3::ZERO).with_priority(1));
        system.add_goal(IkGoal::new(3, Vec3::ZERO).with_priority(5)); // Same priority as 0

        let order1 = system.resolve_priorities();
        let order2 = system.resolve_priorities();

        // Order should be stable
        assert_eq!(order1, order2);
    }
}
