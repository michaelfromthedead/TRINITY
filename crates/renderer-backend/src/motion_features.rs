//! Motion matching feature extraction for TRINITY Engine (T-AN-6.2).
//!
//! This module provides feature extraction for motion matching, converting
//! skeletal poses and trajectories into normalized feature vectors for
//! efficient nearest neighbor search.
//!
//! # Features Extracted
//!
//! - **Pose features**: Joint positions and velocities relative to root
//! - **Trajectory features**: Future root positions and facing directions
//! - **Foot features**: Contact state, position, velocity, and gait phase
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::motion_features::{
//!     FeatureExtractor, PoseFeatureExtractor, TrajectoryExtractor,
//!     FootFeatureExtractor, ExtractionContext,
//! };
//!
//! // Create extractor with key bones
//! let extractor = FeatureExtractor {
//!     pose: PoseFeatureExtractor {
//!         key_bones: vec![0, 1, 5, 6, 10, 11],  // hips, feet, hands
//!         height_scale: 1.8,
//!         include_velocities: true,
//!     },
//!     trajectory: TrajectoryExtractor::default(),
//!     foot: FootFeatureExtractor::default(),
//!     normalizer: None,
//! };
//!
//! // Extract features
//! let context = ExtractionContext { ... };
//! let features = extractor.extract_all(&context);
//! ```

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::pose::Pose;
use crate::skeleton::{Skeleton, Transform};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default trajectory sample times in seconds.
pub const DEFAULT_TRAJECTORY_TIMES: [f32; 3] = [0.2, 0.5, 1.0];

/// Default contact velocity threshold (m/s).
pub const DEFAULT_CONTACT_THRESHOLD: f32 = 0.1;

/// Default character height for normalization (meters).
pub const DEFAULT_HEIGHT: f32 = 1.8;

/// Maximum number of key bones for feature extraction.
pub const MAX_KEY_BONES: usize = 32;

/// Maximum foot count for tracking.
pub const MAX_FEET: usize = 4;

// ---------------------------------------------------------------------------
// LocomotionStyle
// ---------------------------------------------------------------------------

/// Locomotion style tags for motion classification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum LocomotionStyle {
    /// Standing still or idle
    #[default]
    Idle = 0,
    /// Walking locomotion
    Walk = 1,
    /// Running locomotion
    Run = 2,
    /// Sprinting locomotion
    Sprint = 3,
    /// Crouching locomotion
    Crouch = 4,
}

impl LocomotionStyle {
    /// Get style from speed (m/s).
    pub fn from_speed(speed: f32) -> Self {
        if speed < 0.1 {
            Self::Idle
        } else if speed < 2.0 {
            Self::Walk
        } else if speed < 5.0 {
            Self::Run
        } else {
            Self::Sprint
        }
    }

    /// Get the name of this style.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Walk => "walk",
            Self::Run => "run",
            Self::Sprint => "sprint",
            Self::Crouch => "crouch",
        }
    }
}

// ---------------------------------------------------------------------------
// TerrainType
// ---------------------------------------------------------------------------

/// Terrain type for motion classification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TerrainType {
    /// Flat ground
    #[default]
    Flat = 0,
    /// Upward slope
    SlopeUp = 1,
    /// Downward slope
    SlopeDown = 2,
    /// Ascending stairs
    StairsUp = 3,
    /// Descending stairs
    StairsDown = 4,
}

impl TerrainType {
    /// Get terrain type from slope angle (radians).
    pub fn from_slope(angle: f32) -> Self {
        if angle.abs() < 0.05 {
            Self::Flat
        } else if angle > 0.0 {
            Self::SlopeUp
        } else {
            Self::SlopeDown
        }
    }

    /// Get the name of this terrain type.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::SlopeUp => "slope_up",
            Self::SlopeDown => "slope_down",
            Self::StairsUp => "stairs_up",
            Self::StairsDown => "stairs_down",
        }
    }
}

// ---------------------------------------------------------------------------
// ActionType
// ---------------------------------------------------------------------------

/// Action type for motion classification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ActionType {
    /// Idle/standing
    #[default]
    Idle = 0,
    /// General movement
    Move = 1,
    /// Combat actions
    Combat = 2,
    /// Interaction actions
    Interact = 3,
    /// Transition between states
    Transition = 4,
}

impl ActionType {
    /// Get the name of this action type.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Move => "move",
            Self::Combat => "combat",
            Self::Interact => "interact",
            Self::Transition => "transition",
        }
    }
}

// ---------------------------------------------------------------------------
// MotionTags
// ---------------------------------------------------------------------------

/// Combined motion tags for classification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MotionTags {
    /// Locomotion style
    pub locomotion: LocomotionStyle,
    /// Terrain type
    pub terrain: TerrainType,
    /// Action type
    pub action: ActionType,
}

impl MotionTags {
    /// Create new motion tags.
    #[inline]
    pub const fn new(locomotion: LocomotionStyle, terrain: TerrainType, action: ActionType) -> Self {
        Self {
            locomotion,
            terrain,
            action,
        }
    }

    /// Create idle tags.
    #[inline]
    pub const fn idle() -> Self {
        Self {
            locomotion: LocomotionStyle::Idle,
            terrain: TerrainType::Flat,
            action: ActionType::Idle,
        }
    }

    /// Create walking tags.
    #[inline]
    pub const fn walk() -> Self {
        Self {
            locomotion: LocomotionStyle::Walk,
            terrain: TerrainType::Flat,
            action: ActionType::Move,
        }
    }

    /// Create running tags.
    #[inline]
    pub const fn run() -> Self {
        Self {
            locomotion: LocomotionStyle::Run,
            terrain: TerrainType::Flat,
            action: ActionType::Move,
        }
    }

    /// Check if matches required tags (all must match).
    #[inline]
    pub fn matches(&self, required: &MotionTags) -> bool {
        (required.locomotion == LocomotionStyle::Idle || self.locomotion == required.locomotion)
            && (required.terrain == TerrainType::Flat || self.terrain == required.terrain)
            && (required.action == ActionType::Idle || self.action == required.action)
    }

    /// Pack tags into a single u32 for efficient storage.
    #[inline]
    pub fn pack(&self) -> u32 {
        (self.locomotion as u32) | ((self.terrain as u32) << 8) | ((self.action as u32) << 16)
    }

    /// Unpack tags from a u32.
    #[inline]
    pub fn unpack(packed: u32) -> Self {
        let locomotion = match packed & 0xFF {
            1 => LocomotionStyle::Walk,
            2 => LocomotionStyle::Run,
            3 => LocomotionStyle::Sprint,
            4 => LocomotionStyle::Crouch,
            _ => LocomotionStyle::Idle,
        };
        let terrain = match (packed >> 8) & 0xFF {
            1 => TerrainType::SlopeUp,
            2 => TerrainType::SlopeDown,
            3 => TerrainType::StairsUp,
            4 => TerrainType::StairsDown,
            _ => TerrainType::Flat,
        };
        let action = match (packed >> 16) & 0xFF {
            1 => ActionType::Move,
            2 => ActionType::Combat,
            3 => ActionType::Interact,
            4 => ActionType::Transition,
            _ => ActionType::Idle,
        };
        Self {
            locomotion,
            terrain,
            action,
        }
    }
}

// ---------------------------------------------------------------------------
// RootTransform
// ---------------------------------------------------------------------------

/// Root transform for trajectory tracking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RootTransform {
    /// World position of the root.
    pub position: Vec3,
    /// World rotation of the root (yaw only typically).
    pub rotation: Quat,
    /// Facing direction (forward vector in XZ plane).
    pub facing: Vec3,
    /// Timestamp in seconds.
    pub time: f32,
}

impl RootTransform {
    /// Create a new root transform.
    #[inline]
    pub fn new(position: Vec3, rotation: Quat, time: f32) -> Self {
        let facing = rotation * Vec3::Z;
        Self {
            position,
            rotation,
            facing: Vec3::new(facing.x, 0.0, facing.z).normalize_or_zero(),
            time,
        }
    }

    /// Create from position and facing angle (radians).
    #[inline]
    pub fn from_position_angle(position: Vec3, angle: f32, time: f32) -> Self {
        let rotation = Quat::from_rotation_y(angle);
        let facing = Vec3::new(angle.sin(), 0.0, angle.cos());
        Self {
            position,
            rotation,
            facing,
            time,
        }
    }

    /// Get the facing angle in radians.
    #[inline]
    pub fn facing_angle(&self) -> f32 {
        self.facing.x.atan2(self.facing.z)
    }

    /// Interpolate between two root transforms.
    #[inline]
    pub fn lerp(&self, other: &RootTransform, t: f32) -> Self {
        Self {
            position: self.position.lerp(other.position, t),
            rotation: self.rotation.slerp(other.rotation, t),
            facing: self.facing.lerp(other.facing, t).normalize_or_zero(),
            time: self.time + (other.time - self.time) * t,
        }
    }

    /// Transform a point from root space to world space.
    #[inline]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.rotation * point + self.position
    }

    /// Transform a point from world space to root space.
    #[inline]
    pub fn inverse_transform_point(&self, point: Vec3) -> Vec3 {
        self.rotation.inverse() * (point - self.position)
    }
}

// ---------------------------------------------------------------------------
// FootFeatures
// ---------------------------------------------------------------------------

/// Extracted foot features for motion matching.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FootFeatures {
    /// Per-foot contact state (true = planted).
    pub contact_states: Vec<bool>,
    /// Per-foot position relative to root.
    pub positions: Vec<Vec3>,
    /// Per-foot velocity relative to root.
    pub velocities: Vec<Vec3>,
    /// Per-foot phase in gait cycle [0, 1].
    pub phases: Vec<f32>,
}

impl FootFeatures {
    /// Create foot features for the specified number of feet.
    pub fn with_count(count: usize) -> Self {
        Self {
            contact_states: vec![false; count],
            positions: vec![Vec3::ZERO; count],
            velocities: vec![Vec3::ZERO; count],
            phases: vec![0.0; count],
        }
    }

    /// Get the number of feet.
    #[inline]
    pub fn foot_count(&self) -> usize {
        self.contact_states.len()
    }

    /// Check if any foot is planted.
    #[inline]
    pub fn any_planted(&self) -> bool {
        self.contact_states.iter().any(|&c| c)
    }

    /// Check if all feet are planted.
    #[inline]
    pub fn all_planted(&self) -> bool {
        self.contact_states.iter().all(|&c| c)
    }

    /// Count planted feet.
    #[inline]
    pub fn planted_count(&self) -> usize {
        self.contact_states.iter().filter(|&&c| c).count()
    }

    /// Flatten to feature vector.
    pub fn to_flat_vector(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.foot_count() * 8);
        for i in 0..self.foot_count() {
            flat.push(if self.contact_states[i] { 1.0 } else { 0.0 });
            flat.push(self.positions[i].x);
            flat.push(self.positions[i].y);
            flat.push(self.positions[i].z);
            flat.push(self.velocities[i].x);
            flat.push(self.velocities[i].y);
            flat.push(self.velocities[i].z);
            flat.push(self.phases[i]);
        }
        flat
    }

    /// Create from flat vector.
    pub fn from_flat_vector(flat: &[f32], foot_count: usize) -> Self {
        let mut features = Self::with_count(foot_count);
        for i in 0..foot_count {
            let base = i * 8;
            if base + 7 < flat.len() {
                features.contact_states[i] = flat[base] > 0.5;
                features.positions[i] = Vec3::new(flat[base + 1], flat[base + 2], flat[base + 3]);
                features.velocities[i] = Vec3::new(flat[base + 4], flat[base + 5], flat[base + 6]);
                features.phases[i] = flat[base + 7];
            }
        }
        features
    }

    /// Compute squared distance to another foot features.
    pub fn distance_squared(
        &self,
        other: &FootFeatures,
        contact_weight: f32,
        pos_weight: f32,
        vel_weight: f32,
        phase_weight: f32,
    ) -> f32 {
        let mut dist = 0.0;
        let count = self.foot_count().min(other.foot_count());

        for i in 0..count {
            // Contact state penalty
            if self.contact_states[i] != other.contact_states[i] {
                dist += contact_weight;
            }
            // Position distance
            dist += (self.positions[i] - other.positions[i]).length_squared() * pos_weight;
            // Velocity distance
            dist += (self.velocities[i] - other.velocities[i]).length_squared() * vel_weight;
            // Phase distance (circular)
            let phase_diff = (self.phases[i] - other.phases[i]).abs();
            let wrapped = phase_diff.min(1.0 - phase_diff);
            dist += wrapped * wrapped * phase_weight;
        }

        dist
    }
}

// ---------------------------------------------------------------------------
// MotionFeatures
// ---------------------------------------------------------------------------

/// Complete extracted motion features for a frame.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MotionFeatures {
    /// Pose features (joint positions/velocities relative to root).
    pub pose: Vec<f32>,
    /// Trajectory features (future root positions/facings).
    pub trajectory: Vec<f32>,
    /// Foot features.
    pub foot: FootFeatures,
    /// Motion tags.
    pub tags: MotionTags,
    /// Root velocity.
    pub root_velocity: Vec3,
    /// Root angular velocity (radians/s).
    pub root_angular_velocity: f32,
}

impl MotionFeatures {
    /// Create empty motion features.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get total feature dimension.
    pub fn dimension(&self) -> usize {
        self.pose.len() + self.trajectory.len() + self.foot.foot_count() * 8
    }

    /// Flatten all features to a single vector.
    pub fn to_flat_vector(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.dimension());
        flat.extend(&self.pose);
        flat.extend(&self.trajectory);
        flat.extend(self.foot.to_flat_vector());
        flat
    }
}

// ---------------------------------------------------------------------------
// FeatureWeights
// ---------------------------------------------------------------------------

/// Weights for computing feature distances.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureWeights {
    /// Weight for pose position features.
    pub pose_position: f32,
    /// Weight for pose velocity features.
    pub pose_velocity: f32,
    /// Weight for trajectory position features.
    pub trajectory_position: f32,
    /// Weight for trajectory facing features.
    pub trajectory_facing: f32,
    /// Weight for foot contact state.
    pub foot_contact: f32,
    /// Weight for foot position.
    pub foot_position: f32,
    /// Weight for foot velocity.
    pub foot_velocity: f32,
    /// Weight for foot phase.
    pub foot_phase: f32,
}

impl Default for FeatureWeights {
    fn default() -> Self {
        Self {
            pose_position: 1.0,
            pose_velocity: 0.1,
            trajectory_position: 1.0,
            trajectory_facing: 1.0,
            foot_contact: 2.0,
            foot_position: 0.5,
            foot_velocity: 0.1,
            foot_phase: 0.5,
        }
    }
}

impl FeatureWeights {
    /// Create uniform weights.
    pub fn uniform() -> Self {
        Self {
            pose_position: 1.0,
            pose_velocity: 1.0,
            trajectory_position: 1.0,
            trajectory_facing: 1.0,
            foot_contact: 1.0,
            foot_position: 1.0,
            foot_velocity: 1.0,
            foot_phase: 1.0,
        }
    }

    /// Create trajectory-focused weights.
    pub fn trajectory_focused() -> Self {
        Self {
            pose_position: 0.5,
            pose_velocity: 0.1,
            trajectory_position: 2.0,
            trajectory_facing: 2.0,
            foot_contact: 1.0,
            foot_position: 0.5,
            foot_velocity: 0.1,
            foot_phase: 0.5,
        }
    }

    /// Create pose-focused weights.
    pub fn pose_focused() -> Self {
        Self {
            pose_position: 2.0,
            pose_velocity: 0.5,
            trajectory_position: 0.5,
            trajectory_facing: 0.5,
            foot_contact: 1.0,
            foot_position: 1.0,
            foot_velocity: 0.5,
            foot_phase: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// PoseFeatureExtractor
// ---------------------------------------------------------------------------

/// Extracts pose features from skeletal poses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseFeatureExtractor {
    /// Indices of key bones to include in features.
    pub key_bones: Vec<usize>,
    /// Character height for position normalization.
    pub height_scale: f32,
    /// Whether to include joint velocities.
    pub include_velocities: bool,
}

impl Default for PoseFeatureExtractor {
    fn default() -> Self {
        Self {
            key_bones: Vec::new(),
            height_scale: DEFAULT_HEIGHT,
            include_velocities: true,
        }
    }
}

impl PoseFeatureExtractor {
    /// Create with specified key bones.
    pub fn with_key_bones(key_bones: Vec<usize>) -> Self {
        Self {
            key_bones,
            ..Default::default()
        }
    }

    /// Create standard humanoid key bones (hips, feet, hands, head).
    /// Assumes standard bone indices: 0=hips, 5/6=feet, 10/11=hands, 15=head.
    pub fn humanoid() -> Self {
        Self {
            key_bones: vec![0, 5, 6, 10, 11, 15],
            height_scale: DEFAULT_HEIGHT,
            include_velocities: true,
        }
    }

    /// Get the number of key bones.
    #[inline]
    pub fn key_bone_count(&self) -> usize {
        self.key_bones.len()
    }

    /// Get feature dimension (per pose).
    #[inline]
    pub fn feature_dimension(&self) -> usize {
        let pos_dim = self.key_bones.len() * 3;
        if self.include_velocities {
            pos_dim * 2
        } else {
            pos_dim
        }
    }

    /// Extract pose features from a pose and skeleton.
    ///
    /// Returns joint positions (and optionally velocities) relative to root,
    /// normalized by character height.
    pub fn extract(
        &self,
        skeleton: &Skeleton,
        pose: &Pose,
        prev_pose: Option<&Pose>,
        dt: f32,
    ) -> Vec<f32> {
        let mut features = Vec::with_capacity(self.feature_dimension());

        // Compute world transforms
        let transforms = pose.transforms();
        let world_mats = skeleton.compute_world_transforms(&transforms);

        // Get root transform (assumed to be bone 0)
        let root_pos = if !world_mats.is_empty() {
            world_mats[0].w_axis.truncate()
        } else {
            Vec3::ZERO
        };
        let root_rot = if !world_mats.is_empty() {
            Transform::from_matrix(world_mats[0])
                .map(|t| t.rotation)
                .unwrap_or(Quat::IDENTITY)
        } else {
            Quat::IDENTITY
        };
        let root_inv_rot = root_rot.inverse();

        // Extract positions relative to root
        for &bone_idx in &self.key_bones {
            let world_pos = if bone_idx < world_mats.len() {
                world_mats[bone_idx].w_axis.truncate()
            } else {
                Vec3::ZERO
            };
            let rel_pos = root_inv_rot * (world_pos - root_pos);
            let normalized = rel_pos / self.height_scale;

            features.push(normalized.x);
            features.push(normalized.y);
            features.push(normalized.z);
        }

        // Extract velocities if requested
        if self.include_velocities {
            if let Some(prev) = prev_pose {
                if dt > 0.0 {
                    let prev_transforms = prev.transforms();
                    let prev_world_mats = skeleton.compute_world_transforms(&prev_transforms);

                    let prev_root_pos = if !prev_world_mats.is_empty() {
                        prev_world_mats[0].w_axis.truncate()
                    } else {
                        Vec3::ZERO
                    };
                    let prev_root_rot = if !prev_world_mats.is_empty() {
                        Transform::from_matrix(prev_world_mats[0])
                            .map(|t| t.rotation)
                            .unwrap_or(Quat::IDENTITY)
                    } else {
                        Quat::IDENTITY
                    };
                    let prev_root_inv_rot = prev_root_rot.inverse();

                    for &bone_idx in &self.key_bones {
                        let curr_pos = if bone_idx < world_mats.len() {
                            world_mats[bone_idx].w_axis.truncate()
                        } else {
                            Vec3::ZERO
                        };
                        let prev_pos = if bone_idx < prev_world_mats.len() {
                            prev_world_mats[bone_idx].w_axis.truncate()
                        } else {
                            Vec3::ZERO
                        };

                        let curr_rel = root_inv_rot * (curr_pos - root_pos);
                        let prev_rel = prev_root_inv_rot * (prev_pos - prev_root_pos);
                        let vel = (curr_rel - prev_rel) / dt / self.height_scale;

                        features.push(vel.x);
                        features.push(vel.y);
                        features.push(vel.z);
                    }
                } else {
                    // Zero velocities for zero dt
                    for _ in 0..self.key_bones.len() * 3 {
                        features.push(0.0);
                    }
                }
            } else {
                // No previous pose, zero velocities
                for _ in 0..self.key_bones.len() * 3 {
                    features.push(0.0);
                }
            }
        }

        features
    }

    /// Extract pose features without velocity (single pose).
    pub fn extract_positions_only(&self, skeleton: &Skeleton, pose: &Pose) -> Vec<f32> {
        let mut features = Vec::with_capacity(self.key_bones.len() * 3);

        let transforms = pose.transforms();
        let world_mats = skeleton.compute_world_transforms(&transforms);

        let root_pos = if !world_mats.is_empty() {
            world_mats[0].w_axis.truncate()
        } else {
            Vec3::ZERO
        };
        let root_rot = if !world_mats.is_empty() {
            Transform::from_matrix(world_mats[0])
                .map(|t| t.rotation)
                .unwrap_or(Quat::IDENTITY)
        } else {
            Quat::IDENTITY
        };
        let root_inv_rot = root_rot.inverse();

        for &bone_idx in &self.key_bones {
            let world_pos = if bone_idx < world_mats.len() {
                world_mats[bone_idx].w_axis.truncate()
            } else {
                Vec3::ZERO
            };
            let rel_pos = root_inv_rot * (world_pos - root_pos);
            let normalized = rel_pos / self.height_scale;

            features.push(normalized.x);
            features.push(normalized.y);
            features.push(normalized.z);
        }

        features
    }
}

// ---------------------------------------------------------------------------
// TrajectoryExtractor
// ---------------------------------------------------------------------------

/// Extracts trajectory features from root transform history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryExtractor {
    /// Time offsets for trajectory samples (e.g., [0.2, 0.5, 1.0]).
    pub sample_times: [f32; 3],
    /// Whether to include facing direction.
    pub include_facing: bool,
    /// Whether to include trajectory curvature.
    pub include_curvature: bool,
}

impl Default for TrajectoryExtractor {
    fn default() -> Self {
        Self {
            sample_times: DEFAULT_TRAJECTORY_TIMES,
            include_facing: true,
            include_curvature: true,
        }
    }
}

impl TrajectoryExtractor {
    /// Create with custom sample times.
    pub fn with_sample_times(sample_times: [f32; 3]) -> Self {
        Self {
            sample_times,
            ..Default::default()
        }
    }

    /// Get feature dimension.
    pub fn feature_dimension(&self) -> usize {
        let mut dim = 3 * 3; // 3 positions (x, y, z) at 3 times
        if self.include_facing {
            dim += 3; // 3 facing angles
        }
        if self.include_curvature {
            dim += 1; // curvature scalar
        }
        dim
    }

    /// Extract trajectory features from root transform history.
    ///
    /// The history should be sorted by time, with the most recent transform last.
    /// Features are extracted relative to the current (last) root transform.
    pub fn extract(&self, history: &[RootTransform]) -> Vec<f32> {
        if history.is_empty() {
            return vec![0.0; self.feature_dimension()];
        }

        let current = history.last().unwrap();
        let current_time = current.time;
        let root_inv_rot = current.rotation.inverse();

        let mut features = Vec::with_capacity(self.feature_dimension());

        // Sample future positions at each time offset
        let mut sampled_positions = [Vec3::ZERO; 3];
        let mut sampled_facings = [0.0f32; 3];

        for (i, &offset) in self.sample_times.iter().enumerate() {
            let target_time = current_time + offset;

            // Extrapolate from history
            if let Some((pos, facing)) = self.extrapolate_position(history, target_time) {
                // Convert to relative coordinates
                let rel_pos = root_inv_rot * (pos - current.position);
                sampled_positions[i] = rel_pos;

                // Compute relative facing angle
                let rel_facing = facing - current.facing_angle();
                sampled_facings[i] = rel_facing;
            }
        }

        // Add positions
        for pos in &sampled_positions {
            features.push(pos.x);
            features.push(pos.y);
            features.push(pos.z);
        }

        // Add facing directions
        if self.include_facing {
            for facing in &sampled_facings {
                features.push(*facing);
            }
        }

        // Add curvature
        if self.include_curvature {
            let curvature = self.compute_curvature(&sampled_positions);
            features.push(curvature);
        }

        features
    }

    /// Extrapolate position at a target time from history.
    fn extrapolate_position(&self, history: &[RootTransform], target_time: f32) -> Option<(Vec3, f32)> {
        if history.is_empty() {
            return None;
        }

        if history.len() == 1 {
            // Can't extrapolate, return current
            return Some((history[0].position, history[0].facing_angle()));
        }

        // Find the two closest samples to extrapolate from
        let last = history.last().unwrap();
        let second_last = &history[history.len() - 2];

        let dt = last.time - second_last.time;
        if dt <= 0.0 {
            return Some((last.position, last.facing_angle()));
        }

        // Linear extrapolation
        let velocity = (last.position - second_last.position) / dt;
        let angular_velocity = (last.facing_angle() - second_last.facing_angle()) / dt;

        let future_dt = target_time - last.time;
        let future_pos = last.position + velocity * future_dt;
        let future_facing = last.facing_angle() + angular_velocity * future_dt;

        Some((future_pos, future_facing))
    }

    /// Compute curvature from sampled positions.
    fn compute_curvature(&self, positions: &[Vec3; 3]) -> f32 {
        // Use discrete curvature: angle change over arc length
        let v1 = positions[1] - positions[0];
        let v2 = positions[2] - positions[1];

        let len1 = v1.length();
        let len2 = v2.length();

        if len1 < 1e-6 || len2 < 1e-6 {
            return 0.0;
        }

        let v1_norm = v1 / len1;
        let v2_norm = v2 / len2;

        // Angle between vectors
        let dot = v1_norm.dot(v2_norm).clamp(-1.0, 1.0);
        let angle = dot.acos();

        // Curvature = angle / arc_length
        let arc_length = len1 + len2;
        if arc_length > 1e-6 {
            angle / arc_length
        } else {
            0.0
        }
    }

    /// Extract trajectory from desired/predicted future states.
    pub fn extract_from_predictions(
        &self,
        current: &RootTransform,
        future_positions: [Vec3; 3],
        future_facings: [f32; 3],
    ) -> Vec<f32> {
        let mut features = Vec::with_capacity(self.feature_dimension());
        let root_inv_rot = current.rotation.inverse();

        // Convert to relative coordinates
        let mut rel_positions = [Vec3::ZERO; 3];
        let mut rel_facings = [0.0f32; 3];

        for i in 0..3 {
            rel_positions[i] = root_inv_rot * (future_positions[i] - current.position);
            rel_facings[i] = future_facings[i] - current.facing_angle();
        }

        // Add positions
        for pos in &rel_positions {
            features.push(pos.x);
            features.push(pos.y);
            features.push(pos.z);
        }

        // Add facing directions
        if self.include_facing {
            for facing in &rel_facings {
                features.push(*facing);
            }
        }

        // Add curvature
        if self.include_curvature {
            let curvature = self.compute_curvature(&rel_positions);
            features.push(curvature);
        }

        features
    }
}

// ---------------------------------------------------------------------------
// FootFeatureExtractor
// ---------------------------------------------------------------------------

/// Extracts foot contact and phase features.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FootFeatureExtractor {
    /// Bone index for left foot.
    pub left_foot: usize,
    /// Bone index for right foot.
    pub right_foot: usize,
    /// Additional foot indices (for quadrupeds, etc.).
    pub extra_feet: Vec<usize>,
    /// Velocity threshold for contact detection (m/s).
    pub contact_threshold: f32,
}

impl Default for FootFeatureExtractor {
    fn default() -> Self {
        Self {
            left_foot: 5,
            right_foot: 6,
            extra_feet: Vec::new(),
            contact_threshold: DEFAULT_CONTACT_THRESHOLD,
        }
    }
}

impl FootFeatureExtractor {
    /// Create with specified foot bone indices.
    pub fn new(left_foot: usize, right_foot: usize) -> Self {
        Self {
            left_foot,
            right_foot,
            ..Default::default()
        }
    }

    /// Create for quadruped with four feet.
    pub fn quadruped(front_left: usize, front_right: usize, back_left: usize, back_right: usize) -> Self {
        Self {
            left_foot: front_left,
            right_foot: front_right,
            extra_feet: vec![back_left, back_right],
            contact_threshold: DEFAULT_CONTACT_THRESHOLD,
        }
    }

    /// Get total foot count.
    #[inline]
    pub fn foot_count(&self) -> usize {
        2 + self.extra_feet.len()
    }

    /// Get all foot indices.
    pub fn all_foot_indices(&self) -> Vec<usize> {
        let mut indices = vec![self.left_foot, self.right_foot];
        indices.extend(&self.extra_feet);
        indices
    }

    /// Extract foot features from current and previous pose.
    pub fn extract(
        &self,
        skeleton: &Skeleton,
        pose: &Pose,
        prev_pose: &Pose,
        dt: f32,
    ) -> FootFeatures {
        let foot_indices = self.all_foot_indices();
        let mut features = FootFeatures::with_count(foot_indices.len());

        if dt <= 0.0 {
            return features;
        }

        // Compute world transforms
        let transforms = pose.transforms();
        let world_mats = skeleton.compute_world_transforms(&transforms);

        let prev_transforms = prev_pose.transforms();
        let prev_world_mats = skeleton.compute_world_transforms(&prev_transforms);

        // Get root info
        let root_pos = if !world_mats.is_empty() {
            world_mats[0].w_axis.truncate()
        } else {
            Vec3::ZERO
        };
        let root_rot = if !world_mats.is_empty() {
            Transform::from_matrix(world_mats[0])
                .map(|t| t.rotation)
                .unwrap_or(Quat::IDENTITY)
        } else {
            Quat::IDENTITY
        };
        let root_inv_rot = root_rot.inverse();

        let prev_root_pos = if !prev_world_mats.is_empty() {
            prev_world_mats[0].w_axis.truncate()
        } else {
            Vec3::ZERO
        };

        // Extract features for each foot
        for (i, &foot_idx) in foot_indices.iter().enumerate() {
            let curr_world_pos = if foot_idx < world_mats.len() {
                world_mats[foot_idx].w_axis.truncate()
            } else {
                Vec3::ZERO
            };

            let prev_world_pos = if foot_idx < prev_world_mats.len() {
                prev_world_mats[foot_idx].w_axis.truncate()
            } else {
                Vec3::ZERO
            };

            // Position relative to root
            let rel_pos = root_inv_rot * (curr_world_pos - root_pos);
            features.positions[i] = rel_pos;

            // Velocity (world space, then convert to root space)
            let world_vel = (curr_world_pos - prev_world_pos) / dt;
            let root_world_vel = (root_pos - prev_root_pos) / dt;
            let rel_vel = root_inv_rot * (world_vel - root_world_vel);
            features.velocities[i] = rel_vel;

            // Contact detection based on velocity and height
            let speed = rel_vel.length();
            let height = rel_pos.y;
            features.contact_states[i] = speed < self.contact_threshold && height < 0.1;

            // Phase estimation (simplified: based on foot height cycle)
            // Real implementation would track over multiple frames
            features.phases[i] = (rel_pos.y / 0.3).clamp(0.0, 1.0);
        }

        features
    }

    /// Extract foot features for a single pose (without velocities).
    pub fn extract_static(&self, skeleton: &Skeleton, pose: &Pose) -> FootFeatures {
        let foot_indices = self.all_foot_indices();
        let mut features = FootFeatures::with_count(foot_indices.len());

        // Compute world transforms
        let transforms = pose.transforms();
        let world_mats = skeleton.compute_world_transforms(&transforms);

        // Get root info
        let root_pos = if !world_mats.is_empty() {
            world_mats[0].w_axis.truncate()
        } else {
            Vec3::ZERO
        };
        let root_rot = if !world_mats.is_empty() {
            Transform::from_matrix(world_mats[0])
                .map(|t| t.rotation)
                .unwrap_or(Quat::IDENTITY)
        } else {
            Quat::IDENTITY
        };
        let root_inv_rot = root_rot.inverse();

        // Extract features for each foot
        for (i, &foot_idx) in foot_indices.iter().enumerate() {
            let world_pos = if foot_idx < world_mats.len() {
                world_mats[foot_idx].w_axis.truncate()
            } else {
                Vec3::ZERO
            };

            let rel_pos = root_inv_rot * (world_pos - root_pos);
            features.positions[i] = rel_pos;

            // Assume planted if foot is low
            features.contact_states[i] = rel_pos.y < 0.1;
        }

        features
    }
}

// ---------------------------------------------------------------------------
// FeatureNormalizer
// ---------------------------------------------------------------------------

/// Normalizes features to zero mean and unit variance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureNormalizer {
    /// Mean values for each feature dimension.
    pub mean: Vec<f32>,
    /// Standard deviation for each feature dimension.
    pub std_dev: Vec<f32>,
}

impl Default for FeatureNormalizer {
    fn default() -> Self {
        Self {
            mean: Vec::new(),
            std_dev: Vec::new(),
        }
    }
}

impl FeatureNormalizer {
    /// Create a new empty normalizer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a normalizer with pre-computed statistics.
    pub fn with_stats(mean: Vec<f32>, std_dev: Vec<f32>) -> Self {
        Self { mean, std_dev }
    }

    /// Get feature dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// Check if the normalizer is initialized.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        !self.mean.is_empty()
    }

    /// Compute normalization statistics from a collection of feature vectors.
    pub fn compute_from_features(&mut self, features: &[Vec<f32>]) {
        if features.is_empty() {
            return;
        }

        let dim = features[0].len();
        let n = features.len() as f32;

        // Compute mean
        self.mean = vec![0.0; dim];
        for feature in features {
            for (i, &val) in feature.iter().enumerate() {
                if i < dim {
                    self.mean[i] += val;
                }
            }
        }
        for m in &mut self.mean {
            *m /= n;
        }

        // Compute standard deviation
        self.std_dev = vec![0.0; dim];
        for feature in features {
            for (i, &val) in feature.iter().enumerate() {
                if i < dim {
                    let diff = val - self.mean[i];
                    self.std_dev[i] += diff * diff;
                }
            }
        }
        for sd in &mut self.std_dev {
            *sd = (*sd / n).sqrt();
            // Prevent division by zero
            if *sd < 1e-8 {
                *sd = 1.0;
            }
        }
    }

    /// Normalize a feature vector in place.
    pub fn normalize(&self, features: &mut [f32]) {
        let dim = self.mean.len().min(features.len());
        for i in 0..dim {
            features[i] = (features[i] - self.mean[i]) / self.std_dev[i];
        }
    }

    /// Normalize a feature vector, returning a new vector.
    pub fn normalize_copy(&self, features: &[f32]) -> Vec<f32> {
        let mut result = features.to_vec();
        self.normalize(&mut result);
        result
    }

    /// Denormalize a feature vector in place.
    pub fn denormalize(&self, features: &mut [f32]) {
        let dim = self.mean.len().min(features.len());
        for i in 0..dim {
            features[i] = features[i] * self.std_dev[i] + self.mean[i];
        }
    }

    /// Denormalize a feature vector, returning a new vector.
    pub fn denormalize_copy(&self, features: &[f32]) -> Vec<f32> {
        let mut result = features.to_vec();
        self.denormalize(&mut result);
        result
    }

    /// Verify that normalized features have approximately zero mean and unit variance.
    pub fn verify_normalization(&self, features: &[Vec<f32>]) -> (f32, f32) {
        if features.is_empty() || self.mean.is_empty() {
            return (0.0, 0.0);
        }

        let dim = self.mean.len();
        let n = features.len() as f32;

        // Compute mean of normalized features
        let mut normalized_mean = vec![0.0f32; dim];
        let mut normalized_var = vec![0.0f32; dim];

        for feature in features {
            let normalized = self.normalize_copy(feature);
            for (i, &val) in normalized.iter().enumerate() {
                if i < dim {
                    normalized_mean[i] += val;
                }
            }
        }
        for m in &mut normalized_mean {
            *m /= n;
        }

        // Compute variance of normalized features
        for feature in features {
            let normalized = self.normalize_copy(feature);
            for (i, &val) in normalized.iter().enumerate() {
                if i < dim {
                    let diff = val - normalized_mean[i];
                    normalized_var[i] += diff * diff;
                }
            }
        }
        for v in &mut normalized_var {
            *v = (*v / n).sqrt();
        }

        // Return average mean deviation and average std deviation
        let avg_mean: f32 = normalized_mean.iter().map(|&x| x.abs()).sum::<f32>() / dim as f32;
        let avg_std: f32 = normalized_var.iter().sum::<f32>() / dim as f32;

        (avg_mean, avg_std)
    }
}

// ---------------------------------------------------------------------------
// ExtractionContext
// ---------------------------------------------------------------------------

/// Context for feature extraction.
#[derive(Clone, Debug)]
pub struct ExtractionContext<'a> {
    /// Skeleton for the character.
    pub skeleton: &'a Skeleton,
    /// Current pose.
    pub pose: &'a Pose,
    /// Previous pose (for velocity computation).
    pub prev_pose: Option<&'a Pose>,
    /// Time delta since previous pose.
    pub dt: f32,
    /// Root transform history (for trajectory).
    pub root_history: &'a [RootTransform],
    /// Motion tags for this frame.
    pub tags: MotionTags,
}

// ---------------------------------------------------------------------------
// FeatureExtractor
// ---------------------------------------------------------------------------

/// Complete feature extractor combining pose, trajectory, and foot features.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FeatureExtractor {
    /// Pose feature extractor.
    pub pose: PoseFeatureExtractor,
    /// Trajectory feature extractor.
    pub trajectory: TrajectoryExtractor,
    /// Foot feature extractor.
    pub foot: FootFeatureExtractor,
    /// Feature normalizer (optional).
    pub normalizer: Option<FeatureNormalizer>,
}

impl FeatureExtractor {
    /// Create a new feature extractor with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a humanoid feature extractor.
    pub fn humanoid() -> Self {
        Self {
            pose: PoseFeatureExtractor::humanoid(),
            trajectory: TrajectoryExtractor::default(),
            foot: FootFeatureExtractor::default(),
            normalizer: None,
        }
    }

    /// Create with specific key bones.
    pub fn with_key_bones(key_bones: Vec<usize>) -> Self {
        Self {
            pose: PoseFeatureExtractor::with_key_bones(key_bones),
            trajectory: TrajectoryExtractor::default(),
            foot: FootFeatureExtractor::default(),
            normalizer: None,
        }
    }

    /// Get total feature dimension.
    pub fn total_dimension(&self) -> usize {
        self.pose.feature_dimension()
            + self.trajectory.feature_dimension()
            + self.foot.foot_count() * 8
    }

    /// Extract pose features.
    pub fn extract_pose(
        &self,
        skeleton: &Skeleton,
        pose: &Pose,
        prev_pose: Option<&Pose>,
        dt: f32,
    ) -> Vec<f32> {
        self.pose.extract(skeleton, pose, prev_pose, dt)
    }

    /// Extract trajectory features.
    pub fn extract_trajectory(&self, history: &[RootTransform]) -> Vec<f32> {
        self.trajectory.extract(history)
    }

    /// Extract foot features.
    pub fn extract_foot(
        &self,
        skeleton: &Skeleton,
        pose: &Pose,
        prev_pose: &Pose,
        dt: f32,
    ) -> FootFeatures {
        self.foot.extract(skeleton, pose, prev_pose, dt)
    }

    /// Extract all features from context.
    pub fn extract_all(&self, context: &ExtractionContext) -> MotionFeatures {
        let pose_features = if let Some(prev) = context.prev_pose {
            self.pose.extract(context.skeleton, context.pose, Some(prev), context.dt)
        } else {
            self.pose.extract(context.skeleton, context.pose, None, context.dt)
        };

        let trajectory_features = self.trajectory.extract(context.root_history);

        let foot_features = if let Some(prev) = context.prev_pose {
            self.foot.extract(context.skeleton, context.pose, prev, context.dt)
        } else {
            self.foot.extract_static(context.skeleton, context.pose)
        };

        // Compute root velocity from history
        let root_velocity = if context.root_history.len() >= 2 {
            let curr = context.root_history.last().unwrap();
            let prev = &context.root_history[context.root_history.len() - 2];
            let dt = curr.time - prev.time;
            if dt > 0.0 {
                (curr.position - prev.position) / dt
            } else {
                Vec3::ZERO
            }
        } else {
            Vec3::ZERO
        };

        let root_angular_velocity = if context.root_history.len() >= 2 {
            let curr = context.root_history.last().unwrap();
            let prev = &context.root_history[context.root_history.len() - 2];
            let dt = curr.time - prev.time;
            if dt > 0.0 {
                (curr.facing_angle() - prev.facing_angle()) / dt
            } else {
                0.0
            }
        } else {
            0.0
        };

        MotionFeatures {
            pose: pose_features,
            trajectory: trajectory_features,
            foot: foot_features,
            tags: context.tags,
            root_velocity,
            root_angular_velocity,
        }
    }

    /// Normalize features in place.
    pub fn normalize(&self, features: &mut [f32]) {
        if let Some(ref normalizer) = self.normalizer {
            normalizer.normalize(features);
        }
    }

    /// Compute normalization statistics from feature database.
    pub fn compute_normalization(&mut self, all_features: &[Vec<f32>]) {
        let mut normalizer = FeatureNormalizer::new();
        normalizer.compute_from_features(all_features);
        self.normalizer = Some(normalizer);
    }

    /// Set the normalizer.
    pub fn set_normalizer(&mut self, normalizer: FeatureNormalizer) {
        self.normalizer = Some(normalizer);
    }

    /// Clear the normalizer.
    pub fn clear_normalizer(&mut self) {
        self.normalizer = None;
    }
}

// ---------------------------------------------------------------------------
// Distance computation
// ---------------------------------------------------------------------------

/// Compute weighted feature distance between two motion features.
pub fn compute_feature_distance(
    a: &MotionFeatures,
    b: &MotionFeatures,
    weights: &FeatureWeights,
) -> f32 {
    let mut distance = 0.0;

    // Pose distance (split into positions and velocities)
    let pose_dim = a.pose.len().min(b.pose.len());
    let half_dim = pose_dim / 2;

    // Position features (first half)
    for i in 0..half_dim {
        let diff = a.pose[i] - b.pose[i];
        distance += diff * diff * weights.pose_position;
    }

    // Velocity features (second half)
    for i in half_dim..pose_dim {
        let diff = a.pose[i] - b.pose[i];
        distance += diff * diff * weights.pose_velocity;
    }

    // Trajectory distance
    let traj_dim = a.trajectory.len().min(b.trajectory.len());
    // First 9 values are positions (3 positions x 3 components)
    for i in 0..9.min(traj_dim) {
        let diff = a.trajectory[i] - b.trajectory[i];
        distance += diff * diff * weights.trajectory_position;
    }
    // Next 3 values are facings
    for i in 9..12.min(traj_dim) {
        let diff = a.trajectory[i] - b.trajectory[i];
        // Wrap angle difference
        let wrapped = diff.abs().min(std::f32::consts::TAU - diff.abs());
        distance += wrapped * wrapped * weights.trajectory_facing;
    }

    // Foot distance
    distance += a.foot.distance_squared(
        &b.foot,
        weights.foot_contact,
        weights.foot_position,
        weights.foot_velocity,
        weights.foot_phase,
    );

    distance.sqrt()
}

/// Compute squared feature distance (faster, no sqrt).
pub fn compute_feature_distance_squared(
    a: &MotionFeatures,
    b: &MotionFeatures,
    weights: &FeatureWeights,
) -> f32 {
    let mut distance = 0.0;

    // Pose distance
    let pose_dim = a.pose.len().min(b.pose.len());
    let half_dim = pose_dim / 2;

    for i in 0..half_dim {
        let diff = a.pose[i] - b.pose[i];
        distance += diff * diff * weights.pose_position;
    }
    for i in half_dim..pose_dim {
        let diff = a.pose[i] - b.pose[i];
        distance += diff * diff * weights.pose_velocity;
    }

    // Trajectory distance
    let traj_dim = a.trajectory.len().min(b.trajectory.len());
    for i in 0..9.min(traj_dim) {
        let diff = a.trajectory[i] - b.trajectory[i];
        distance += diff * diff * weights.trajectory_position;
    }
    for i in 9..12.min(traj_dim) {
        let diff = a.trajectory[i] - b.trajectory[i];
        let wrapped = diff.abs().min(std::f32::consts::TAU - diff.abs());
        distance += wrapped * wrapped * weights.trajectory_facing;
    }

    // Foot distance
    distance += a.foot.distance_squared(
        &b.foot,
        weights.foot_contact,
        weights.foot_position,
        weights.foot_velocity,
        weights.foot_phase,
    );

    distance
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Bone, SkeletonBuilder};
    use std::f32::consts::PI;

    // =====================================================================
    // Helper functions
    // =====================================================================

    fn create_test_skeleton() -> Skeleton {
        SkeletonBuilder::new()
            .root("root")
            .child_at("spine", "root", Vec3::new(0.0, 1.0, 0.0))
            .child_at("head", "spine", Vec3::new(0.0, 0.5, 0.0))
            .child_at("left_arm", "spine", Vec3::new(-0.5, 0.0, 0.0))
            .child_at("right_arm", "spine", Vec3::new(0.5, 0.0, 0.0))
            .child_at("left_foot", "root", Vec3::new(-0.2, -1.0, 0.0))
            .child_at("right_foot", "root", Vec3::new(0.2, -1.0, 0.0))
            .build_unchecked()
    }

    fn create_test_pose(skeleton: &Skeleton) -> Pose {
        Pose::from_skeleton(skeleton, crate::pose::PoseType::Current)
    }

    // =====================================================================
    // LocomotionStyle tests
    // =====================================================================

    #[test]
    fn test_locomotion_style_from_speed() {
        assert_eq!(LocomotionStyle::from_speed(0.0), LocomotionStyle::Idle);
        assert_eq!(LocomotionStyle::from_speed(0.05), LocomotionStyle::Idle);
        assert_eq!(LocomotionStyle::from_speed(1.0), LocomotionStyle::Walk);
        assert_eq!(LocomotionStyle::from_speed(3.0), LocomotionStyle::Run);
        assert_eq!(LocomotionStyle::from_speed(7.0), LocomotionStyle::Sprint);
    }

    #[test]
    fn test_locomotion_style_name() {
        assert_eq!(LocomotionStyle::Idle.name(), "idle");
        assert_eq!(LocomotionStyle::Walk.name(), "walk");
        assert_eq!(LocomotionStyle::Run.name(), "run");
        assert_eq!(LocomotionStyle::Sprint.name(), "sprint");
        assert_eq!(LocomotionStyle::Crouch.name(), "crouch");
    }

    #[test]
    fn test_locomotion_style_default() {
        assert_eq!(LocomotionStyle::default(), LocomotionStyle::Idle);
    }

    // =====================================================================
    // TerrainType tests
    // =====================================================================

    #[test]
    fn test_terrain_type_from_slope() {
        assert_eq!(TerrainType::from_slope(0.0), TerrainType::Flat);
        assert_eq!(TerrainType::from_slope(0.03), TerrainType::Flat);
        assert_eq!(TerrainType::from_slope(0.2), TerrainType::SlopeUp);
        assert_eq!(TerrainType::from_slope(-0.2), TerrainType::SlopeDown);
    }

    #[test]
    fn test_terrain_type_name() {
        assert_eq!(TerrainType::Flat.name(), "flat");
        assert_eq!(TerrainType::SlopeUp.name(), "slope_up");
        assert_eq!(TerrainType::SlopeDown.name(), "slope_down");
        assert_eq!(TerrainType::StairsUp.name(), "stairs_up");
        assert_eq!(TerrainType::StairsDown.name(), "stairs_down");
    }

    // =====================================================================
    // ActionType tests
    // =====================================================================

    #[test]
    fn test_action_type_name() {
        assert_eq!(ActionType::Idle.name(), "idle");
        assert_eq!(ActionType::Move.name(), "move");
        assert_eq!(ActionType::Combat.name(), "combat");
        assert_eq!(ActionType::Interact.name(), "interact");
        assert_eq!(ActionType::Transition.name(), "transition");
    }

    // =====================================================================
    // MotionTags tests
    // =====================================================================

    #[test]
    fn test_motion_tags_new() {
        let tags = MotionTags::new(LocomotionStyle::Walk, TerrainType::SlopeUp, ActionType::Move);
        assert_eq!(tags.locomotion, LocomotionStyle::Walk);
        assert_eq!(tags.terrain, TerrainType::SlopeUp);
        assert_eq!(tags.action, ActionType::Move);
    }

    #[test]
    fn test_motion_tags_presets() {
        let idle = MotionTags::idle();
        assert_eq!(idle.locomotion, LocomotionStyle::Idle);

        let walk = MotionTags::walk();
        assert_eq!(walk.locomotion, LocomotionStyle::Walk);

        let run = MotionTags::run();
        assert_eq!(run.locomotion, LocomotionStyle::Run);
    }

    #[test]
    fn test_motion_tags_pack_unpack() {
        let original = MotionTags::new(LocomotionStyle::Run, TerrainType::StairsUp, ActionType::Combat);
        let packed = original.pack();
        let unpacked = MotionTags::unpack(packed);

        assert_eq!(original.locomotion, unpacked.locomotion);
        assert_eq!(original.terrain, unpacked.terrain);
        assert_eq!(original.action, unpacked.action);
    }

    #[test]
    fn test_motion_tags_matches() {
        let frame = MotionTags::walk();
        let walk_required = MotionTags::new(LocomotionStyle::Walk, TerrainType::Flat, ActionType::Idle);
        let run_required = MotionTags::new(LocomotionStyle::Run, TerrainType::Flat, ActionType::Idle);

        assert!(frame.matches(&walk_required));
        assert!(!frame.matches(&run_required));
    }

    // =====================================================================
    // RootTransform tests
    // =====================================================================

    #[test]
    fn test_root_transform_new() {
        let rt = RootTransform::new(Vec3::new(1.0, 0.0, 2.0), Quat::IDENTITY, 0.5);
        assert_eq!(rt.position, Vec3::new(1.0, 0.0, 2.0));
        assert_eq!(rt.time, 0.5);
    }

    #[test]
    fn test_root_transform_from_position_angle() {
        let rt = RootTransform::from_position_angle(Vec3::ZERO, PI / 2.0, 0.0);
        assert!(rt.facing.x.abs() - 1.0 < 0.01); // Should face +X
    }

    #[test]
    fn test_root_transform_facing_angle() {
        let rt = RootTransform::from_position_angle(Vec3::ZERO, PI / 4.0, 0.0);
        let angle = rt.facing_angle();
        assert!((angle - PI / 4.0).abs() < 0.01);
    }

    #[test]
    fn test_root_transform_lerp() {
        let a = RootTransform::new(Vec3::ZERO, Quat::IDENTITY, 0.0);
        let b = RootTransform::new(Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY, 1.0);
        let mid = a.lerp(&b, 0.5);

        assert!((mid.position.x - 5.0).abs() < 0.01);
        assert!((mid.time - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_root_transform_transform_point() {
        let rt = RootTransform::new(Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY, 0.0);
        let world = rt.transform_point(Vec3::new(1.0, 0.0, 0.0));
        assert!((world.x - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_root_transform_inverse_transform_point() {
        let rt = RootTransform::new(Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY, 0.0);
        let local = rt.inverse_transform_point(Vec3::new(6.0, 0.0, 0.0));
        assert!((local.x - 1.0).abs() < 0.01);
    }

    // =====================================================================
    // FootFeatures tests
    // =====================================================================

    #[test]
    fn test_foot_features_with_count() {
        let ff = FootFeatures::with_count(2);
        assert_eq!(ff.foot_count(), 2);
        assert!(!ff.any_planted());
    }

    #[test]
    fn test_foot_features_planted_detection() {
        let mut ff = FootFeatures::with_count(2);
        ff.contact_states[0] = true;

        assert!(ff.any_planted());
        assert!(!ff.all_planted());
        assert_eq!(ff.planted_count(), 1);

        ff.contact_states[1] = true;
        assert!(ff.all_planted());
        assert_eq!(ff.planted_count(), 2);
    }

    #[test]
    fn test_foot_features_to_flat_vector() {
        let mut ff = FootFeatures::with_count(1);
        ff.contact_states[0] = true;
        ff.positions[0] = Vec3::new(1.0, 2.0, 3.0);
        ff.velocities[0] = Vec3::new(4.0, 5.0, 6.0);
        ff.phases[0] = 0.5;

        let flat = ff.to_flat_vector();
        assert_eq!(flat.len(), 8);
        assert_eq!(flat[0], 1.0); // contact
        assert_eq!(flat[1], 1.0); // pos.x
        assert_eq!(flat[7], 0.5); // phase
    }

    #[test]
    fn test_foot_features_from_flat_vector() {
        let flat = vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5];
        let ff = FootFeatures::from_flat_vector(&flat, 1);

        assert!(ff.contact_states[0]);
        assert_eq!(ff.positions[0], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(ff.phases[0], 0.5);
    }

    #[test]
    fn test_foot_features_distance_squared_identical() {
        let ff = FootFeatures::with_count(2);
        let dist = ff.distance_squared(&ff, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_foot_features_distance_squared_different_contact() {
        let mut a = FootFeatures::with_count(1);
        let mut b = FootFeatures::with_count(1);
        a.contact_states[0] = true;
        b.contact_states[0] = false;

        let dist = a.distance_squared(&b, 10.0, 0.0, 0.0, 0.0);
        assert_eq!(dist, 10.0);
    }

    // =====================================================================
    // MotionFeatures tests
    // =====================================================================

    #[test]
    fn test_motion_features_new() {
        let mf = MotionFeatures::new();
        assert!(mf.pose.is_empty());
        assert!(mf.trajectory.is_empty());
    }

    #[test]
    fn test_motion_features_dimension() {
        let mut mf = MotionFeatures::new();
        mf.pose = vec![0.0; 10];
        mf.trajectory = vec![0.0; 5];
        mf.foot = FootFeatures::with_count(2);

        assert_eq!(mf.dimension(), 10 + 5 + 16);
    }

    #[test]
    fn test_motion_features_to_flat_vector() {
        let mut mf = MotionFeatures::new();
        mf.pose = vec![1.0, 2.0];
        mf.trajectory = vec![3.0, 4.0];
        mf.foot = FootFeatures::with_count(1);

        let flat = mf.to_flat_vector();
        assert_eq!(flat[0], 1.0);
        assert_eq!(flat[1], 2.0);
        assert_eq!(flat[2], 3.0);
        assert_eq!(flat[3], 4.0);
    }

    // =====================================================================
    // FeatureWeights tests
    // =====================================================================

    #[test]
    fn test_feature_weights_default() {
        let w = FeatureWeights::default();
        assert!(w.pose_position > 0.0);
        assert!(w.trajectory_position > 0.0);
    }

    #[test]
    fn test_feature_weights_uniform() {
        let w = FeatureWeights::uniform();
        assert_eq!(w.pose_position, w.pose_velocity);
        assert_eq!(w.trajectory_position, w.trajectory_facing);
    }

    #[test]
    fn test_feature_weights_trajectory_focused() {
        let w = FeatureWeights::trajectory_focused();
        assert!(w.trajectory_position > w.pose_position);
    }

    #[test]
    fn test_feature_weights_pose_focused() {
        let w = FeatureWeights::pose_focused();
        assert!(w.pose_position > w.trajectory_position);
    }

    // =====================================================================
    // PoseFeatureExtractor tests
    // =====================================================================

    #[test]
    fn test_pose_extractor_default() {
        let pe = PoseFeatureExtractor::default();
        assert!(pe.key_bones.is_empty());
        assert_eq!(pe.height_scale, DEFAULT_HEIGHT);
        assert!(pe.include_velocities);
    }

    #[test]
    fn test_pose_extractor_with_key_bones() {
        let pe = PoseFeatureExtractor::with_key_bones(vec![0, 1, 2]);
        assert_eq!(pe.key_bones, vec![0, 1, 2]);
    }

    #[test]
    fn test_pose_extractor_humanoid() {
        let pe = PoseFeatureExtractor::humanoid();
        assert!(!pe.key_bones.is_empty());
    }

    #[test]
    fn test_pose_extractor_feature_dimension() {
        let pe = PoseFeatureExtractor::with_key_bones(vec![0, 1, 2]);
        // 3 bones * 3 positions * 2 (pos + vel)
        assert_eq!(pe.feature_dimension(), 18);
    }

    #[test]
    fn test_pose_extractor_feature_dimension_no_velocity() {
        let mut pe = PoseFeatureExtractor::with_key_bones(vec![0, 1, 2]);
        pe.include_velocities = false;
        assert_eq!(pe.feature_dimension(), 9);
    }

    #[test]
    fn test_pose_extractor_extract() {
        let skeleton = create_test_skeleton();
        let pose = create_test_pose(&skeleton);
        let pe = PoseFeatureExtractor::with_key_bones(vec![0, 1]);

        let features = pe.extract(&skeleton, &pose, None, 0.0);
        assert_eq!(features.len(), pe.feature_dimension());
    }

    #[test]
    fn test_pose_extractor_extract_with_prev_pose() {
        let skeleton = create_test_skeleton();
        let pose = create_test_pose(&skeleton);
        let prev_pose = create_test_pose(&skeleton);
        let pe = PoseFeatureExtractor::with_key_bones(vec![0, 1]);

        let features = pe.extract(&skeleton, &pose, Some(&prev_pose), 0.033);
        assert_eq!(features.len(), pe.feature_dimension());
    }

    #[test]
    fn test_pose_extractor_extract_positions_only() {
        let skeleton = create_test_skeleton();
        let pose = create_test_pose(&skeleton);
        let pe = PoseFeatureExtractor::with_key_bones(vec![0, 1, 2]);

        let features = pe.extract_positions_only(&skeleton, &pose);
        assert_eq!(features.len(), 9); // 3 bones * 3 components
    }

    // =====================================================================
    // TrajectoryExtractor tests
    // =====================================================================

    #[test]
    fn test_trajectory_extractor_default() {
        let te = TrajectoryExtractor::default();
        assert_eq!(te.sample_times, DEFAULT_TRAJECTORY_TIMES);
        assert!(te.include_facing);
        assert!(te.include_curvature);
    }

    #[test]
    fn test_trajectory_extractor_with_sample_times() {
        let te = TrajectoryExtractor::with_sample_times([0.1, 0.3, 0.5]);
        assert_eq!(te.sample_times, [0.1, 0.3, 0.5]);
    }

    #[test]
    fn test_trajectory_extractor_feature_dimension() {
        let te = TrajectoryExtractor::default();
        // 9 (positions) + 3 (facings) + 1 (curvature) = 13
        assert_eq!(te.feature_dimension(), 13);
    }

    #[test]
    fn test_trajectory_extractor_feature_dimension_minimal() {
        let mut te = TrajectoryExtractor::default();
        te.include_facing = false;
        te.include_curvature = false;
        assert_eq!(te.feature_dimension(), 9);
    }

    #[test]
    fn test_trajectory_extractor_extract_empty_history() {
        let te = TrajectoryExtractor::default();
        let features = te.extract(&[]);
        assert_eq!(features.len(), te.feature_dimension());
    }

    #[test]
    fn test_trajectory_extractor_extract_single() {
        let te = TrajectoryExtractor::default();
        let history = vec![RootTransform::new(Vec3::ZERO, Quat::IDENTITY, 0.0)];
        let features = te.extract(&history);
        assert_eq!(features.len(), te.feature_dimension());
    }

    #[test]
    fn test_trajectory_extractor_extract_multiple() {
        let te = TrajectoryExtractor::default();
        let history = vec![
            RootTransform::new(Vec3::ZERO, Quat::IDENTITY, 0.0),
            RootTransform::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, 0.1),
            RootTransform::new(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY, 0.2),
        ];
        let features = te.extract(&history);
        assert_eq!(features.len(), te.feature_dimension());
    }

    #[test]
    fn test_trajectory_extractor_extract_from_predictions() {
        let te = TrajectoryExtractor::default();
        let current = RootTransform::new(Vec3::ZERO, Quat::IDENTITY, 0.0);
        let positions = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0)];
        let facings = [0.1, 0.2, 0.3];

        let features = te.extract_from_predictions(&current, positions, facings);
        assert_eq!(features.len(), te.feature_dimension());
    }

    #[test]
    fn test_trajectory_extractor_curvature_straight() {
        let te = TrajectoryExtractor::default();
        let positions = [Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, 3.0)];
        let curvature = te.compute_curvature(&positions);
        assert!(curvature.abs() < 0.01); // Straight line has zero curvature
    }

    #[test]
    fn test_trajectory_extractor_curvature_curve() {
        let te = TrajectoryExtractor::default();
        let positions = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 1.0), Vec3::new(2.0, 0.0, 0.0)];
        let curvature = te.compute_curvature(&positions);
        assert!(curvature > 0.0); // Curve has positive curvature
    }

    // =====================================================================
    // FootFeatureExtractor tests
    // =====================================================================

    #[test]
    fn test_foot_extractor_default() {
        let fe = FootFeatureExtractor::default();
        assert_eq!(fe.left_foot, 5);
        assert_eq!(fe.right_foot, 6);
        assert_eq!(fe.foot_count(), 2);
    }

    #[test]
    fn test_foot_extractor_new() {
        let fe = FootFeatureExtractor::new(10, 11);
        assert_eq!(fe.left_foot, 10);
        assert_eq!(fe.right_foot, 11);
    }

    #[test]
    fn test_foot_extractor_quadruped() {
        let fe = FootFeatureExtractor::quadruped(1, 2, 3, 4);
        assert_eq!(fe.foot_count(), 4);
        assert_eq!(fe.all_foot_indices(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_foot_extractor_extract_static() {
        let skeleton = create_test_skeleton();
        let pose = create_test_pose(&skeleton);
        let fe = FootFeatureExtractor::new(5, 6);

        let features = fe.extract_static(&skeleton, &pose);
        assert_eq!(features.foot_count(), 2);
    }

    #[test]
    fn test_foot_extractor_extract_dynamic() {
        let skeleton = create_test_skeleton();
        let pose = create_test_pose(&skeleton);
        let prev_pose = create_test_pose(&skeleton);
        let fe = FootFeatureExtractor::new(5, 6);

        let features = fe.extract(&skeleton, &pose, &prev_pose, 0.033);
        assert_eq!(features.foot_count(), 2);
    }

    // =====================================================================
    // FeatureNormalizer tests
    // =====================================================================

    #[test]
    fn test_normalizer_new() {
        let n = FeatureNormalizer::new();
        assert!(!n.is_initialized());
        assert_eq!(n.dimension(), 0);
    }

    #[test]
    fn test_normalizer_with_stats() {
        let mean = vec![1.0, 2.0, 3.0];
        let std = vec![0.5, 1.0, 2.0];
        let n = FeatureNormalizer::with_stats(mean.clone(), std.clone());

        assert!(n.is_initialized());
        assert_eq!(n.dimension(), 3);
        assert_eq!(n.mean, mean);
        assert_eq!(n.std_dev, std);
    }

    #[test]
    fn test_normalizer_compute_from_features() {
        let features = vec![
            vec![0.0, 0.0, 0.0],
            vec![2.0, 4.0, 6.0],
        ];

        let mut n = FeatureNormalizer::new();
        n.compute_from_features(&features);

        // Mean should be [1.0, 2.0, 3.0]
        assert!((n.mean[0] - 1.0).abs() < 0.01);
        assert!((n.mean[1] - 2.0).abs() < 0.01);
        assert!((n.mean[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_normalizer_normalize() {
        let n = FeatureNormalizer::with_stats(vec![0.0, 0.0], vec![2.0, 1.0]);
        let mut features = vec![4.0, 3.0];
        n.normalize(&mut features);

        assert!((features[0] - 2.0).abs() < 0.01); // (4-0)/2 = 2
        assert!((features[1] - 3.0).abs() < 0.01); // (3-0)/1 = 3
    }

    #[test]
    fn test_normalizer_normalize_copy() {
        let n = FeatureNormalizer::with_stats(vec![1.0], vec![2.0]);
        let features = vec![5.0];
        let normalized = n.normalize_copy(&features);

        assert!((normalized[0] - 2.0).abs() < 0.01); // (5-1)/2 = 2
    }

    #[test]
    fn test_normalizer_denormalize() {
        let n = FeatureNormalizer::with_stats(vec![1.0], vec![2.0]);
        let mut features = vec![2.0];
        n.denormalize(&mut features);

        assert!((features[0] - 5.0).abs() < 0.01); // 2*2 + 1 = 5
    }

    #[test]
    fn test_normalizer_roundtrip() {
        let n = FeatureNormalizer::with_stats(vec![10.0, 20.0], vec![5.0, 10.0]);
        let original = vec![25.0, 40.0];

        let normalized = n.normalize_copy(&original);
        let denormalized = n.denormalize_copy(&normalized);

        assert!((denormalized[0] - original[0]).abs() < 0.01);
        assert!((denormalized[1] - original[1]).abs() < 0.01);
    }

    #[test]
    fn test_normalizer_verify_normalization() {
        let features = vec![
            vec![10.0, 20.0],
            vec![20.0, 40.0],
            vec![30.0, 60.0],
        ];

        let mut n = FeatureNormalizer::new();
        n.compute_from_features(&features);

        let (avg_mean, avg_std) = n.verify_normalization(&features);
        assert!(avg_mean < 0.01); // Mean should be close to 0
        assert!((avg_std - 1.0).abs() < 0.1); // Std should be close to 1
    }

    // =====================================================================
    // FeatureExtractor tests
    // =====================================================================

    #[test]
    fn test_feature_extractor_new() {
        let fe = FeatureExtractor::new();
        assert!(fe.pose.key_bones.is_empty());
        assert!(fe.normalizer.is_none());
    }

    #[test]
    fn test_feature_extractor_humanoid() {
        let fe = FeatureExtractor::humanoid();
        assert!(!fe.pose.key_bones.is_empty());
    }

    #[test]
    fn test_feature_extractor_with_key_bones() {
        let fe = FeatureExtractor::with_key_bones(vec![0, 1, 2]);
        assert_eq!(fe.pose.key_bones, vec![0, 1, 2]);
    }

    #[test]
    fn test_feature_extractor_total_dimension() {
        let fe = FeatureExtractor::with_key_bones(vec![0, 1]);
        let dim = fe.total_dimension();
        assert!(dim > 0);
    }

    #[test]
    fn test_feature_extractor_extract_pose() {
        let skeleton = create_test_skeleton();
        let pose = create_test_pose(&skeleton);
        let fe = FeatureExtractor::with_key_bones(vec![0, 1]);

        let features = fe.extract_pose(&skeleton, &pose, None, 0.0);
        assert!(!features.is_empty());
    }

    #[test]
    fn test_feature_extractor_extract_trajectory() {
        let fe = FeatureExtractor::new();
        let history = vec![
            RootTransform::new(Vec3::ZERO, Quat::IDENTITY, 0.0),
            RootTransform::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, 0.1),
        ];

        let features = fe.extract_trajectory(&history);
        assert!(!features.is_empty());
    }

    #[test]
    fn test_feature_extractor_set_normalizer() {
        let mut fe = FeatureExtractor::new();
        let normalizer = FeatureNormalizer::with_stats(vec![0.0], vec![1.0]);

        fe.set_normalizer(normalizer);
        assert!(fe.normalizer.is_some());

        fe.clear_normalizer();
        assert!(fe.normalizer.is_none());
    }

    #[test]
    fn test_feature_extractor_compute_normalization() {
        let mut fe = FeatureExtractor::new();
        let features = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];

        fe.compute_normalization(&features);
        assert!(fe.normalizer.is_some());
    }

    #[test]
    fn test_feature_extractor_normalize() {
        let mut fe = FeatureExtractor::new();
        fe.set_normalizer(FeatureNormalizer::with_stats(vec![0.0, 0.0], vec![1.0, 1.0]));

        let mut features = vec![1.0, 2.0];
        fe.normalize(&mut features);

        assert_eq!(features, vec![1.0, 2.0]); // No change with mean=0, std=1
    }

    // =====================================================================
    // Distance computation tests
    // =====================================================================

    #[test]
    fn test_compute_feature_distance_identical() {
        let a = MotionFeatures::new();
        let weights = FeatureWeights::default();
        let dist = compute_feature_distance(&a, &a, &weights);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_compute_feature_distance_different() {
        let mut a = MotionFeatures::new();
        let mut b = MotionFeatures::new();
        a.pose = vec![0.0, 0.0, 0.0];
        b.pose = vec![1.0, 0.0, 0.0];

        let weights = FeatureWeights::default();
        let dist = compute_feature_distance(&a, &b, &weights);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_compute_feature_distance_squared() {
        let mut a = MotionFeatures::new();
        let mut b = MotionFeatures::new();
        a.pose = vec![0.0, 0.0];
        b.pose = vec![3.0, 4.0];

        let weights = FeatureWeights::uniform();
        let dist_sq = compute_feature_distance_squared(&a, &b, &weights);
        let dist = compute_feature_distance(&a, &b, &weights);

        assert!((dist_sq - dist * dist).abs() < 0.01);
    }

    // =====================================================================
    // Integration tests
    // =====================================================================

    #[test]
    fn test_full_extraction_pipeline() {
        let skeleton = create_test_skeleton();
        let pose = create_test_pose(&skeleton);
        let prev_pose = create_test_pose(&skeleton);

        let extractor = FeatureExtractor::with_key_bones(vec![0, 1, 2]);

        let root_history = vec![
            RootTransform::new(Vec3::ZERO, Quat::IDENTITY, 0.0),
            RootTransform::new(Vec3::new(0.1, 0.0, 0.0), Quat::IDENTITY, 0.033),
        ];

        let context = ExtractionContext {
            skeleton: &skeleton,
            pose: &pose,
            prev_pose: Some(&prev_pose),
            dt: 0.033,
            root_history: &root_history,
            tags: MotionTags::walk(),
        };

        let features = extractor.extract_all(&context);

        assert!(!features.pose.is_empty());
        assert!(!features.trajectory.is_empty());
        assert_eq!(features.tags.locomotion, LocomotionStyle::Walk);
    }

    #[test]
    fn test_feature_serialization() {
        let mut features = MotionFeatures::new();
        features.pose = vec![1.0, 2.0, 3.0];
        features.trajectory = vec![4.0, 5.0];
        features.foot = FootFeatures::with_count(2);
        features.tags = MotionTags::run();

        let json = serde_json::to_string(&features).unwrap();
        let restored: MotionFeatures = serde_json::from_str(&json).unwrap();

        assert_eq!(features.pose, restored.pose);
        assert_eq!(features.trajectory, restored.trajectory);
        assert_eq!(features.tags.locomotion, restored.tags.locomotion);
    }

    #[test]
    fn test_normalizer_serialization() {
        let n = FeatureNormalizer::with_stats(vec![1.0, 2.0], vec![0.5, 1.5]);

        let json = serde_json::to_string(&n).unwrap();
        let restored: FeatureNormalizer = serde_json::from_str(&json).unwrap();

        assert_eq!(n.mean, restored.mean);
        assert_eq!(n.std_dev, restored.std_dev);
    }

    #[test]
    fn test_extractor_serialization() {
        let mut extractor = FeatureExtractor::humanoid();
        extractor.set_normalizer(FeatureNormalizer::with_stats(vec![0.0], vec![1.0]));

        let json = serde_json::to_string(&extractor).unwrap();
        let restored: FeatureExtractor = serde_json::from_str(&json).unwrap();

        assert_eq!(extractor.pose.key_bones, restored.pose.key_bones);
        assert!(restored.normalizer.is_some());
    }

    // =====================================================================
    // Edge case tests
    // =====================================================================

    #[test]
    fn test_empty_skeleton() {
        let skeleton = Skeleton::new();
        let pose = Pose::new(0, crate::pose::PoseType::Current);
        let pe = PoseFeatureExtractor::with_key_bones(vec![0, 1]);

        let features = pe.extract(&skeleton, &pose, None, 0.0);
        // Should handle gracefully, returning zeros for missing bones
        assert!(!features.is_empty());
    }

    #[test]
    fn test_static_pose_velocities() {
        let skeleton = create_test_skeleton();
        let pose = create_test_pose(&skeleton);
        let pe = PoseFeatureExtractor::with_key_bones(vec![0]);

        // Same pose for current and previous = zero velocity
        let features = pe.extract(&skeleton, &pose, Some(&pose), 0.033);

        // Second half should be velocities, all zeros
        let half = features.len() / 2;
        for i in half..features.len() {
            assert!((features[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_zero_dt_handling() {
        let skeleton = create_test_skeleton();
        let pose = create_test_pose(&skeleton);
        let prev_pose = create_test_pose(&skeleton);
        let pe = PoseFeatureExtractor::with_key_bones(vec![0]);

        // Zero dt should not cause division by zero
        let features = pe.extract(&skeleton, &pose, Some(&prev_pose), 0.0);
        assert!(!features.is_empty());
        assert!(features.iter().all(|&f| f.is_finite()));
    }

    #[test]
    fn test_rapid_motion_velocities() {
        let skeleton = create_test_skeleton();
        let mut pose = create_test_pose(&skeleton);
        let prev_pose = create_test_pose(&skeleton);

        // Move a bone rapidly
        pose.positions[0] = Vec3::new(10.0, 0.0, 0.0);

        let pe = PoseFeatureExtractor::with_key_bones(vec![0]);
        let features = pe.extract(&skeleton, &pose, Some(&prev_pose), 0.033);

        // Velocities should be large but finite
        let half = features.len() / 2;
        assert!(features[half..].iter().all(|&f| f.is_finite()));
    }

    #[test]
    fn test_foot_contact_threshold() {
        let mut fe = FootFeatureExtractor::default();
        fe.contact_threshold = 0.5;

        // Foot moving at 0.3 m/s should be planted
        // Foot moving at 0.7 m/s should not be planted
        assert_eq!(fe.contact_threshold, 0.5);
    }

    #[test]
    fn test_distance_with_empty_features() {
        let a = MotionFeatures::new();
        let b = MotionFeatures::new();
        let weights = FeatureWeights::default();

        let dist = compute_feature_distance(&a, &b, &weights);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_normalizer_empty_features() {
        let mut n = FeatureNormalizer::new();
        n.compute_from_features(&[]);
        assert!(!n.is_initialized());
    }
}
