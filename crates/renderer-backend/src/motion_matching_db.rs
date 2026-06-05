//! Motion matching database for data-driven animation (T-AN-6.1).
//!
//! This module provides a database of pre-processed animation frames for motion matching,
//! enabling natural, responsive character animation by finding the best matching frame
//! from a database given the current character state and desired trajectory.
//!
//! # Architecture
//!
//! ```text
//! MotionDatabase
//! ├── frames: Vec<MotionFrame>       # All pre-processed frames
//! │   ├── PoseFeature                # Joint positions/velocities relative to root
//! │   ├── TrajectoryFeature          # Future root positions at T+0.2, T+0.5, T+1.0s
//! │   ├── FootContact[]              # Per-foot contact state
//! │   └── tags: u32                  # Bitflags for locomotion style, terrain, action
//! ├── clips: Vec<ClipInfo>           # Source clip metadata
//! ├── kd_tree: KdTree                # ANN index for fast nearest neighbor search
//! └── feature_weights: FeatureWeights # Per-feature distance weights
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::motion_matching_db::{MotionDatabase, MotionFrame, QueryFeature};
//!
//! // Build database from animation clips
//! let mut builder = MotionDatabaseBuilder::new();
//! builder.add_clip("walk", &walk_clip, &skeleton);
//! builder.add_clip("run", &run_clip, &skeleton);
//! let db = builder.build();
//!
//! // Query for best matching frame
//! let query = QueryFeature {
//!     pose: current_pose_feature,
//!     trajectory: desired_trajectory,
//!     required_tags: LocomotionTags::WALK,
//! };
//! let (frame_index, distance) = db.find_best_match(&query);
//! ```

use std::fmt;
use std::io::{self, Write};

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of joints supported in pose features.
pub const MAX_POSE_JOINTS: usize = 128;

/// Number of future trajectory samples (T+0.2, T+0.5, T+1.0).
pub const TRAJECTORY_SAMPLES: usize = 3;

/// Maximum number of feet for contact tracking.
pub const MAX_FEET: usize = 4;

/// Default K for KD-tree searches.
pub const DEFAULT_K_NEIGHBORS: usize = 5;

/// Magic bytes for .mmdb file format.
pub const MMDB_MAGIC: [u8; 4] = [b'M', b'M', b'D', b'B'];

/// Current file format version.
pub const MMDB_VERSION: u32 = 1;

/// Trajectory time offsets in seconds.
pub const TRAJECTORY_TIMES: [f32; 3] = [0.2, 0.5, 1.0];

/// Maximum KD-tree depth for balanced construction.
pub const MAX_TREE_DEPTH: usize = 32;

// ---------------------------------------------------------------------------
// Locomotion Tags (bitflags)
// ---------------------------------------------------------------------------

bitflags::bitflags! {
    /// Locomotion style tags for filtering motion database queries.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct LocomotionTags: u32 {
        /// Idle/standing
        const IDLE = 1 << 0;
        /// Walking locomotion
        const WALK = 1 << 1;
        /// Running locomotion
        const RUN = 1 << 2;
        /// Sprinting locomotion
        const SPRINT = 1 << 3;
        /// Crouching locomotion
        const CROUCH = 1 << 4;
        /// Jumping action
        const JUMP = 1 << 5;
        /// Falling state
        const FALL = 1 << 6;
        /// Landing action
        const LAND = 1 << 7;
        /// Strafing movement
        const STRAFE = 1 << 8;
        /// Turning in place
        const TURN = 1 << 9;
        /// Climbing action
        const CLIMB = 1 << 10;
        /// Swimming locomotion
        const SWIM = 1 << 11;
        /// Combat stance
        const COMBAT = 1 << 12;
        /// Carrying/holding
        const CARRY = 1 << 13;
        /// Flat terrain
        const TERRAIN_FLAT = 1 << 16;
        /// Uphill terrain
        const TERRAIN_UPHILL = 1 << 17;
        /// Downhill terrain
        const TERRAIN_DOWNHILL = 1 << 18;
        /// Stairs terrain
        const TERRAIN_STAIRS = 1 << 19;
        /// Rough terrain
        const TERRAIN_ROUGH = 1 << 20;
        /// Start transition
        const TRANSITION_START = 1 << 24;
        /// Stop transition
        const TRANSITION_STOP = 1 << 25;
        /// Direction change transition
        const TRANSITION_TURN = 1 << 26;
    }
}

// Implement serde manually for LocomotionTags
impl Serialize for LocomotionTags {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.bits().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LocomotionTags {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bits = u32::deserialize(deserializer)?;
        Ok(LocomotionTags::from_bits_truncate(bits))
    }
}

impl LocomotionTags {
    /// Check if all required tags are present.
    #[inline]
    pub fn contains_all(&self, required: LocomotionTags) -> bool {
        self.contains(required)
    }

    /// Check if any of the given tags are present.
    #[inline]
    pub fn contains_any(&self, tags: LocomotionTags) -> bool {
        self.intersects(tags)
    }
}

// ---------------------------------------------------------------------------
// PoseFeature
// ---------------------------------------------------------------------------

/// Pose feature vector for motion matching queries.
///
/// Contains joint positions and velocities relative to the character's root,
/// enabling pose comparison independent of world position.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseFeature {
    /// Joint positions relative to root (in root space).
    pub joint_positions: Vec<Vec3>,

    /// Joint velocities relative to root (in root space).
    pub joint_velocities: Vec<Vec3>,
}

impl Default for PoseFeature {
    fn default() -> Self {
        Self {
            joint_positions: Vec::new(),
            joint_velocities: Vec::new(),
        }
    }
}

impl PoseFeature {
    /// Create a new empty pose feature.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a pose feature with specified joint count.
    pub fn with_joint_count(joint_count: usize) -> Self {
        Self {
            joint_positions: vec![Vec3::ZERO; joint_count],
            joint_velocities: vec![Vec3::ZERO; joint_count],
        }
    }

    /// Get the number of joints.
    #[inline]
    pub fn joint_count(&self) -> usize {
        self.joint_positions.len()
    }

    /// Check if the pose feature is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.joint_positions.is_empty()
    }

    /// Compute squared distance to another pose feature.
    ///
    /// Uses weighted sum of position and velocity distances.
    pub fn distance_squared(&self, other: &PoseFeature, pos_weight: f32, vel_weight: f32) -> f32 {
        let mut dist = 0.0;

        // Position distance
        let min_pos = self.joint_positions.len().min(other.joint_positions.len());
        for i in 0..min_pos {
            dist += (self.joint_positions[i] - other.joint_positions[i]).length_squared() * pos_weight;
        }

        // Velocity distance
        let min_vel = self.joint_velocities.len().min(other.joint_velocities.len());
        for i in 0..min_vel {
            dist += (self.joint_velocities[i] - other.joint_velocities[i]).length_squared() * vel_weight;
        }

        dist
    }

    /// Flatten to a vector of floats for KD-tree indexing.
    pub fn to_flat_vector(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.joint_positions.len() * 6);
        for pos in &self.joint_positions {
            flat.push(pos.x);
            flat.push(pos.y);
            flat.push(pos.z);
        }
        for vel in &self.joint_velocities {
            flat.push(vel.x);
            flat.push(vel.y);
            flat.push(vel.z);
        }
        flat
    }

    /// Create from a flat vector of floats.
    pub fn from_flat_vector(flat: &[f32], joint_count: usize) -> Self {
        let mut positions = Vec::with_capacity(joint_count);
        let mut velocities = Vec::with_capacity(joint_count);

        for i in 0..joint_count {
            let base = i * 3;
            if base + 2 < flat.len() {
                positions.push(Vec3::new(flat[base], flat[base + 1], flat[base + 2]));
            }
        }

        let vel_offset = joint_count * 3;
        for i in 0..joint_count {
            let base = vel_offset + i * 3;
            if base + 2 < flat.len() {
                velocities.push(Vec3::new(flat[base], flat[base + 1], flat[base + 2]));
            }
        }

        Self {
            joint_positions: positions,
            joint_velocities: velocities,
        }
    }
}

// ---------------------------------------------------------------------------
// TrajectoryFeature
// ---------------------------------------------------------------------------

/// Trajectory feature for motion matching queries.
///
/// Contains future root positions and facing directions at fixed time offsets,
/// enabling trajectory matching for responsive locomotion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct TrajectoryFeature {
    /// Future root positions at T+0.2, T+0.5, T+1.0 seconds (relative to current root).
    pub future_positions: [Vec3; TRAJECTORY_SAMPLES],

    /// Future facing angles (radians) at the same time offsets.
    pub future_facings: [f32; TRAJECTORY_SAMPLES],
}

// Safety: TrajectoryFeature is repr(C) and contains only Pod types
unsafe impl Zeroable for TrajectoryFeature {}
unsafe impl Pod for TrajectoryFeature {}

impl TrajectoryFeature {
    /// Create a new trajectory feature.
    #[inline]
    pub const fn new() -> Self {
        Self {
            future_positions: [Vec3::ZERO; TRAJECTORY_SAMPLES],
            future_facings: [0.0; TRAJECTORY_SAMPLES],
        }
    }

    /// Create a trajectory feature from positions and facings.
    #[inline]
    pub fn from_predictions(positions: [Vec3; TRAJECTORY_SAMPLES], facings: [f32; TRAJECTORY_SAMPLES]) -> Self {
        Self {
            future_positions: positions,
            future_facings: facings,
        }
    }

    /// Compute squared distance to another trajectory feature.
    pub fn distance_squared(&self, other: &TrajectoryFeature, pos_weight: f32, facing_weight: f32) -> f32 {
        let mut dist = 0.0;

        // Position distance
        for i in 0..TRAJECTORY_SAMPLES {
            dist += (self.future_positions[i] - other.future_positions[i]).length_squared() * pos_weight;
        }

        // Facing angle distance (wrapped)
        for i in 0..TRAJECTORY_SAMPLES {
            let angle_diff = (self.future_facings[i] - other.future_facings[i]).abs();
            let wrapped = angle_diff.min(std::f32::consts::TAU - angle_diff);
            dist += wrapped * wrapped * facing_weight;
        }

        dist
    }

    /// Flatten to a vector of floats.
    pub fn to_flat_vector(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(TRAJECTORY_SAMPLES * 4);
        for pos in &self.future_positions {
            flat.push(pos.x);
            flat.push(pos.y);
            flat.push(pos.z);
        }
        for facing in &self.future_facings {
            flat.push(*facing);
        }
        flat
    }

    /// Create from a flat vector of floats.
    pub fn from_flat_vector(flat: &[f32]) -> Self {
        let mut feature = Self::new();
        for i in 0..TRAJECTORY_SAMPLES {
            let base = i * 3;
            if base + 2 < flat.len() {
                feature.future_positions[i] = Vec3::new(flat[base], flat[base + 1], flat[base + 2]);
            }
        }
        let facing_offset = TRAJECTORY_SAMPLES * 3;
        for i in 0..TRAJECTORY_SAMPLES {
            if facing_offset + i < flat.len() {
                feature.future_facings[i] = flat[facing_offset + i];
            }
        }
        feature
    }

    /// Interpolate between two trajectory features.
    #[inline]
    pub fn lerp(&self, other: &TrajectoryFeature, t: f32) -> Self {
        let mut result = Self::new();
        for i in 0..TRAJECTORY_SAMPLES {
            result.future_positions[i] = self.future_positions[i].lerp(other.future_positions[i], t);
            result.future_facings[i] = self.future_facings[i] + (other.future_facings[i] - self.future_facings[i]) * t;
        }
        result
    }
}

// ---------------------------------------------------------------------------
// FootContact
// ---------------------------------------------------------------------------

/// Foot contact state for motion matching.
///
/// Tracks whether each foot is planted and its position/velocity
/// for accurate foot placement during transitions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct FootContact {
    /// Whether the foot is planted (touching ground).
    pub is_planted: bool,

    /// Foot position in root space.
    pub position: Vec3,

    /// Foot velocity in root space.
    pub velocity: Vec3,
}

impl FootContact {
    /// Create a new foot contact.
    #[inline]
    pub const fn new() -> Self {
        Self {
            is_planted: false,
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
        }
    }

    /// Create a planted foot contact.
    #[inline]
    pub fn planted(position: Vec3) -> Self {
        Self {
            is_planted: true,
            position,
            velocity: Vec3::ZERO,
        }
    }

    /// Create a moving foot contact.
    #[inline]
    pub fn moving(position: Vec3, velocity: Vec3) -> Self {
        Self {
            is_planted: false,
            position,
            velocity,
        }
    }

    /// Compute squared distance to another foot contact.
    pub fn distance_squared(&self, other: &FootContact, contact_weight: f32, pos_weight: f32, vel_weight: f32) -> f32 {
        let mut dist = 0.0;

        // Contact state penalty
        if self.is_planted != other.is_planted {
            dist += contact_weight;
        }

        // Position distance
        dist += (self.position - other.position).length_squared() * pos_weight;

        // Velocity distance
        dist += (self.velocity - other.velocity).length_squared() * vel_weight;

        dist
    }
}

// ---------------------------------------------------------------------------
// MotionFrame
// ---------------------------------------------------------------------------

/// A single frame from the motion database.
///
/// Contains all features needed for motion matching queries:
/// pose, trajectory, foot contacts, and tags.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionFrame {
    /// Index of the source animation clip.
    pub clip_index: u32,

    /// Time within the source clip (seconds).
    pub time: f32,

    /// Pose feature (joint positions/velocities).
    pub pose: PoseFeature,

    /// Trajectory feature (future root positions).
    pub trajectory: TrajectoryFeature,

    /// Foot contact states.
    pub foot_contacts: Vec<FootContact>,

    /// Locomotion tags (bitflags).
    pub tags: LocomotionTags,

    /// Root velocity at this frame (for inertial blending).
    pub root_velocity: Vec3,

    /// Root angular velocity at this frame.
    pub root_angular_velocity: f32,
}

impl Default for MotionFrame {
    fn default() -> Self {
        Self {
            clip_index: 0,
            time: 0.0,
            pose: PoseFeature::default(),
            trajectory: TrajectoryFeature::default(),
            foot_contacts: Vec::new(),
            tags: LocomotionTags::empty(),
            root_velocity: Vec3::ZERO,
            root_angular_velocity: 0.0,
        }
    }
}

impl MotionFrame {
    /// Create a new empty motion frame.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a motion frame with specified clip and time.
    pub fn from_clip(clip_index: u32, time: f32) -> Self {
        Self {
            clip_index,
            time,
            ..Default::default()
        }
    }

    /// Check if this frame matches the required tags.
    #[inline]
    pub fn matches_tags(&self, required: LocomotionTags, excluded: LocomotionTags) -> bool {
        self.tags.contains(required) && !self.tags.intersects(excluded)
    }
}

// ---------------------------------------------------------------------------
// ClipInfo
// ---------------------------------------------------------------------------

/// Metadata about a source animation clip.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClipInfo {
    /// Clip name/identifier.
    pub name: String,

    /// Clip duration in seconds.
    pub duration: f32,

    /// Starting frame index in the database.
    pub start_frame: u32,

    /// Number of frames from this clip.
    pub frame_count: u32,

    /// Sample rate (frames per second).
    pub sample_rate: f32,

    /// Default tags for frames from this clip.
    pub default_tags: LocomotionTags,

    /// Whether the clip loops seamlessly.
    pub is_looping: bool,
}

impl ClipInfo {
    /// Create a new clip info.
    pub fn new(name: impl Into<String>, duration: f32, sample_rate: f32) -> Self {
        Self {
            name: name.into(),
            duration,
            start_frame: 0,
            frame_count: 0,
            sample_rate,
            default_tags: LocomotionTags::empty(),
            is_looping: false,
        }
    }

    /// Get the end frame index (exclusive).
    #[inline]
    pub fn end_frame(&self) -> u32 {
        self.start_frame + self.frame_count
    }
}

// ---------------------------------------------------------------------------
// FeatureWeights
// ---------------------------------------------------------------------------

/// Weights for different feature components in distance computation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureWeights {
    /// Weight for joint positions.
    pub pose_position: f32,

    /// Weight for joint velocities.
    pub pose_velocity: f32,

    /// Weight for trajectory positions.
    pub trajectory_position: f32,

    /// Weight for trajectory facing directions.
    pub trajectory_facing: f32,

    /// Weight for foot contact state.
    pub foot_contact_state: f32,

    /// Weight for foot positions.
    pub foot_position: f32,

    /// Weight for foot velocities.
    pub foot_velocity: f32,
}

impl Default for FeatureWeights {
    fn default() -> Self {
        Self {
            pose_position: 1.0,
            pose_velocity: 0.1,
            trajectory_position: 1.0,
            trajectory_facing: 1.0,
            foot_contact_state: 2.0,
            foot_position: 0.5,
            foot_velocity: 0.1,
        }
    }
}

impl FeatureWeights {
    /// Create feature weights with equal emphasis on all components.
    pub fn uniform() -> Self {
        Self {
            pose_position: 1.0,
            pose_velocity: 1.0,
            trajectory_position: 1.0,
            trajectory_facing: 1.0,
            foot_contact_state: 1.0,
            foot_position: 1.0,
            foot_velocity: 1.0,
        }
    }

    /// Create weights emphasizing trajectory matching (for responsive locomotion).
    pub fn trajectory_focused() -> Self {
        Self {
            pose_position: 0.5,
            pose_velocity: 0.1,
            trajectory_position: 2.0,
            trajectory_facing: 2.0,
            foot_contact_state: 1.0,
            foot_position: 0.5,
            foot_velocity: 0.1,
        }
    }

    /// Create weights emphasizing pose matching (for smooth transitions).
    pub fn pose_focused() -> Self {
        Self {
            pose_position: 2.0,
            pose_velocity: 0.5,
            trajectory_position: 0.5,
            trajectory_facing: 0.5,
            foot_contact_state: 1.0,
            foot_position: 1.0,
            foot_velocity: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// QueryFeature
// ---------------------------------------------------------------------------

/// Query feature for motion matching searches.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QueryFeature {
    /// Current pose feature.
    pub pose: PoseFeature,

    /// Desired trajectory.
    pub trajectory: TrajectoryFeature,

    /// Current foot contacts.
    pub foot_contacts: Vec<FootContact>,

    /// Required tags (all must be present).
    pub required_tags: LocomotionTags,

    /// Excluded tags (none must be present).
    pub excluded_tags: LocomotionTags,
}

impl QueryFeature {
    /// Create a new query feature.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the required tags.
    #[inline]
    pub fn with_required_tags(mut self, tags: LocomotionTags) -> Self {
        self.required_tags = tags;
        self
    }

    /// Set the excluded tags.
    #[inline]
    pub fn with_excluded_tags(mut self, tags: LocomotionTags) -> Self {
        self.excluded_tags = tags;
        self
    }
}

// ---------------------------------------------------------------------------
// KdTreeNode
// ---------------------------------------------------------------------------

/// A node in the KD-tree for nearest neighbor search.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KdTreeNode {
    /// Frame index for leaf nodes, or split dimension for internal nodes.
    pub data: u32,

    /// Split value for internal nodes.
    pub split_value: f32,

    /// Left child index (0 = no child).
    pub left: u32,

    /// Right child index (0 = no child).
    pub right: u32,

    /// Whether this is a leaf node.
    pub is_leaf: bool,
}

impl Default for KdTreeNode {
    fn default() -> Self {
        Self {
            data: 0,
            split_value: 0.0,
            left: 0,
            right: 0,
            is_leaf: true,
        }
    }
}

impl KdTreeNode {
    /// Create a leaf node.
    #[inline]
    pub fn leaf(frame_index: u32) -> Self {
        Self {
            data: frame_index,
            split_value: 0.0,
            left: 0,
            right: 0,
            is_leaf: true,
        }
    }

    /// Create an internal node.
    #[inline]
    pub fn internal(split_dim: u32, split_value: f32, left: u32, right: u32) -> Self {
        Self {
            data: split_dim,
            split_value,
            left,
            right,
            is_leaf: false,
        }
    }
}

// ---------------------------------------------------------------------------
// KdTree
// ---------------------------------------------------------------------------

/// KD-tree for fast approximate nearest neighbor search.
///
/// Used for efficient motion matching queries in high-dimensional feature space.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct KdTree {
    /// Tree nodes.
    pub nodes: Vec<KdTreeNode>,

    /// Feature vectors for each frame.
    pub features: Vec<Vec<f32>>,

    /// Dimensionality of feature vectors.
    pub dimensions: usize,

    /// Frame indices (for leaf node mapping).
    pub frame_indices: Vec<u32>,
}

impl KdTree {
    /// Create a new empty KD-tree.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a KD-tree from motion frames.
    ///
    /// Extracts feature vectors from frames and constructs a balanced KD-tree.
    pub fn build(frames: &[MotionFrame], feature_dims: usize) -> Self {
        if frames.is_empty() {
            return Self::new();
        }

        // Extract feature vectors
        let features: Vec<Vec<f32>> = frames
            .iter()
            .map(|f| Self::extract_features(f, feature_dims))
            .collect();

        let frame_indices: Vec<u32> = (0..frames.len() as u32).collect();

        let mut tree = Self {
            nodes: Vec::with_capacity(frames.len() * 2),
            features,
            dimensions: feature_dims,
            frame_indices: frame_indices.clone(),
        };

        // Build tree recursively
        let indices: Vec<usize> = (0..frames.len()).collect();
        tree.build_recursive(&indices, 0, 0);

        tree
    }

    /// Extract feature vector from a motion frame.
    fn extract_features(frame: &MotionFrame, max_dims: usize) -> Vec<f32> {
        let mut features = Vec::with_capacity(max_dims);

        // Add pose features (positions then velocities)
        for pos in &frame.pose.joint_positions {
            features.push(pos.x);
            features.push(pos.y);
            features.push(pos.z);
            if features.len() >= max_dims {
                break;
            }
        }
        if features.len() < max_dims {
            for vel in &frame.pose.joint_velocities {
                features.push(vel.x);
                features.push(vel.y);
                features.push(vel.z);
                if features.len() >= max_dims {
                    break;
                }
            }
        }

        // Add trajectory features
        if features.len() < max_dims {
            for pos in &frame.trajectory.future_positions {
                features.push(pos.x);
                features.push(pos.y);
                features.push(pos.z);
                if features.len() >= max_dims {
                    break;
                }
            }
        }
        if features.len() < max_dims {
            for facing in &frame.trajectory.future_facings {
                features.push(*facing);
                if features.len() >= max_dims {
                    break;
                }
            }
        }

        // Pad to max_dims if needed
        while features.len() < max_dims {
            features.push(0.0);
        }

        features
    }

    /// Build the tree recursively.
    fn build_recursive(&mut self, indices: &[usize], depth: usize, _parent: usize) -> u32 {
        if indices.is_empty() {
            return 0;
        }

        if indices.len() == 1 || depth >= MAX_TREE_DEPTH {
            // Leaf node
            let node_idx = self.nodes.len() as u32;
            self.nodes.push(KdTreeNode::leaf(indices[0] as u32));
            return node_idx;
        }

        // Choose split dimension (cycle through dimensions)
        let split_dim = depth % self.dimensions;

        // Sort indices by feature value at split dimension
        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_by(|&a, &b| {
            let val_a = self.features.get(a).and_then(|f| f.get(split_dim)).copied().unwrap_or(0.0);
            let val_b = self.features.get(b).and_then(|f| f.get(split_dim)).copied().unwrap_or(0.0);
            val_a.partial_cmp(&val_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find median
        let median_idx = sorted_indices.len() / 2;
        let median_frame = sorted_indices[median_idx];
        let split_value = self.features.get(median_frame)
            .and_then(|f| f.get(split_dim))
            .copied()
            .unwrap_or(0.0);

        // Split into left and right
        let left_indices: Vec<usize> = sorted_indices[..median_idx].to_vec();
        let right_indices: Vec<usize> = sorted_indices[median_idx + 1..].to_vec();

        // Create internal node (reserve index)
        let node_idx = self.nodes.len() as u32;
        self.nodes.push(KdTreeNode::default()); // Placeholder

        // Build children
        let left_child = if !left_indices.is_empty() {
            self.build_recursive(&left_indices, depth + 1, node_idx as usize)
        } else {
            0
        };
        let right_child = if !right_indices.is_empty() {
            self.build_recursive(&right_indices, depth + 1, node_idx as usize)
        } else {
            0
        };

        // Update node
        self.nodes[node_idx as usize] = KdTreeNode::internal(
            split_dim as u32,
            split_value,
            left_child,
            right_child,
        );

        node_idx
    }

    /// Find k nearest neighbors to a query point.
    ///
    /// Returns vec of (frame_index, distance_squared).
    pub fn find_k_nearest(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        if self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut results: Vec<(u32, f32)> = Vec::with_capacity(k);
        let mut max_dist = f32::INFINITY;

        self.search_recursive(0, query, k, &mut results, &mut max_dist, 0);

        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Recursive k-NN search.
    fn search_recursive(
        &self,
        node_idx: u32,
        query: &[f32],
        k: usize,
        results: &mut Vec<(u32, f32)>,
        max_dist: &mut f32,
        depth: usize,
    ) {
        if node_idx == 0 && !self.nodes.is_empty() && depth > 0 {
            return;
        }

        let node = match self.nodes.get(node_idx as usize) {
            Some(n) => n,
            None => return,
        };

        if node.is_leaf {
            // Compute distance to this frame
            let frame_idx = node.data;
            if let Some(features) = self.features.get(frame_idx as usize) {
                let dist = Self::distance_squared(query, features);

                if results.len() < k {
                    results.push((frame_idx, dist));
                    if results.len() == k {
                        *max_dist = results.iter().map(|&(_, d)| d).fold(0.0f32, f32::max);
                    }
                } else if dist < *max_dist {
                    // Replace worst result
                    if let Some(worst_idx) = results.iter().position(|&(_, d)| d == *max_dist) {
                        results[worst_idx] = (frame_idx, dist);
                        *max_dist = results.iter().map(|&(_, d)| d).fold(0.0f32, f32::max);
                    }
                }
            }
            return;
        }

        // Internal node
        let split_dim = node.data as usize;
        let split_value = node.split_value;
        let query_value = query.get(split_dim).copied().unwrap_or(0.0);

        // Determine which child to search first
        let (first, second) = if query_value <= split_value {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        // Search first child
        if first != 0 || (first == 0 && !self.nodes.is_empty() && depth == 0) {
            self.search_recursive(first, query, k, results, max_dist, depth + 1);
        }

        // Check if we need to search second child
        let plane_dist = (query_value - split_value).powi(2);
        if plane_dist < *max_dist || results.len() < k {
            if second != 0 {
                self.search_recursive(second, query, k, results, max_dist, depth + 1);
            }
        }
    }

    /// Compute squared Euclidean distance between two vectors.
    #[inline]
    fn distance_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum()
    }

    /// Get the number of nodes in the tree.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the tree is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ---------------------------------------------------------------------------
// VpTreeNode
// ---------------------------------------------------------------------------

/// A node in the VP-tree (Vantage Point tree).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VpTreeNode {
    /// Frame index of the vantage point.
    pub vantage_point: u32,

    /// Radius threshold for splitting.
    pub radius: f32,

    /// Left child index (points inside radius).
    pub inside: u32,

    /// Right child index (points outside radius).
    pub outside: u32,

    /// Whether this is a leaf node.
    pub is_leaf: bool,
}

impl Default for VpTreeNode {
    fn default() -> Self {
        Self {
            vantage_point: 0,
            radius: 0.0,
            inside: 0,
            outside: 0,
            is_leaf: true,
        }
    }
}

// ---------------------------------------------------------------------------
// VpTree
// ---------------------------------------------------------------------------

/// VP-tree (Vantage Point tree) for high-dimensional nearest neighbor search.
///
/// Alternative to KD-tree that can be more efficient for very high-dimensional data.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VpTree {
    /// Tree nodes.
    pub nodes: Vec<VpTreeNode>,

    /// Feature vectors for each frame.
    pub features: Vec<Vec<f32>>,

    /// Frame indices.
    pub frame_indices: Vec<u32>,
}

impl VpTree {
    /// Create a new empty VP-tree.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a VP-tree from feature vectors.
    pub fn build(features: Vec<Vec<f32>>) -> Self {
        if features.is_empty() {
            return Self::new();
        }

        let frame_indices: Vec<u32> = (0..features.len() as u32).collect();
        let indices: Vec<usize> = (0..features.len()).collect();

        let mut tree = Self {
            nodes: Vec::with_capacity(features.len()),
            features,
            frame_indices,
        };

        tree.build_recursive(&indices);
        tree
    }

    /// Build the tree recursively.
    fn build_recursive(&mut self, indices: &[usize]) -> u32 {
        if indices.is_empty() {
            return u32::MAX;
        }

        if indices.len() == 1 {
            let node_idx = self.nodes.len() as u32;
            self.nodes.push(VpTreeNode {
                vantage_point: indices[0] as u32,
                radius: 0.0,
                inside: u32::MAX,
                outside: u32::MAX,
                is_leaf: true,
            });
            return node_idx;
        }

        // Choose vantage point (first element for simplicity)
        let vp_idx = indices[0];
        let vp_features = &self.features[vp_idx];

        // Compute distances to all other points
        let mut distances: Vec<(usize, f32)> = indices[1..]
            .iter()
            .map(|&i| {
                let dist = KdTree::distance_squared(vp_features, &self.features[i]).sqrt();
                (i, dist)
            })
            .collect();

        if distances.is_empty() {
            let node_idx = self.nodes.len() as u32;
            self.nodes.push(VpTreeNode {
                vantage_point: vp_idx as u32,
                radius: 0.0,
                inside: u32::MAX,
                outside: u32::MAX,
                is_leaf: true,
            });
            return node_idx;
        }

        // Find median distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let median_idx = distances.len() / 2;
        let radius = distances[median_idx].1;

        // Split into inside and outside
        let inside_indices: Vec<usize> = distances[..median_idx].iter().map(|&(i, _)| i).collect();
        let outside_indices: Vec<usize> = distances[median_idx..].iter().map(|&(i, _)| i).collect();

        // Create node
        let node_idx = self.nodes.len() as u32;
        self.nodes.push(VpTreeNode::default()); // Placeholder

        // Build children
        let inside_child = self.build_recursive(&inside_indices);
        let outside_child = self.build_recursive(&outside_indices);

        self.nodes[node_idx as usize] = VpTreeNode {
            vantage_point: vp_idx as u32,
            radius,
            inside: inside_child,
            outside: outside_child,
            is_leaf: false,
        };

        node_idx
    }

    /// Find k nearest neighbors.
    pub fn find_k_nearest(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        if self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }

        let mut results: Vec<(u32, f32)> = Vec::with_capacity(k);
        let mut max_dist = f32::INFINITY;

        self.search_recursive(0, query, k, &mut results, &mut max_dist);

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Recursive search.
    fn search_recursive(
        &self,
        node_idx: u32,
        query: &[f32],
        k: usize,
        results: &mut Vec<(u32, f32)>,
        max_dist: &mut f32,
    ) {
        if node_idx == u32::MAX {
            return;
        }

        let node = match self.nodes.get(node_idx as usize) {
            Some(n) => n,
            None => return,
        };

        let vp_features = match self.features.get(node.vantage_point as usize) {
            Some(f) => f,
            None => return,
        };

        let dist = KdTree::distance_squared(query, vp_features).sqrt();

        // Add vantage point to results if close enough
        if results.len() < k {
            results.push((node.vantage_point, dist));
            if results.len() == k {
                *max_dist = results.iter().map(|&(_, d)| d).fold(0.0f32, f32::max);
            }
        } else if dist < *max_dist {
            if let Some(worst_idx) = results.iter().position(|&(_, d)| d == *max_dist) {
                results[worst_idx] = (node.vantage_point, dist);
                *max_dist = results.iter().map(|&(_, d)| d).fold(0.0f32, f32::max);
            }
        }

        if node.is_leaf {
            return;
        }

        // Determine search order
        if dist < node.radius {
            // Query is inside, search inside first
            self.search_recursive(node.inside, query, k, results, max_dist);
            if dist + *max_dist >= node.radius {
                self.search_recursive(node.outside, query, k, results, max_dist);
            }
        } else {
            // Query is outside, search outside first
            self.search_recursive(node.outside, query, k, results, max_dist);
            if dist - *max_dist <= node.radius {
                self.search_recursive(node.inside, query, k, results, max_dist);
            }
        }
    }

    /// Get the number of nodes.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ---------------------------------------------------------------------------
// DistanceMetric
// ---------------------------------------------------------------------------

/// Distance metric for motion matching queries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Squared Euclidean distance (fastest).
    #[default]
    EuclideanSquared,

    /// Euclidean distance.
    Euclidean,

    /// Manhattan (L1) distance.
    Manhattan,

    /// Weighted Euclidean with feature weights.
    WeightedEuclidean,
}

impl DistanceMetric {
    /// Compute distance between two feature vectors.
    pub fn compute(&self, a: &[f32], b: &[f32], weights: Option<&[f32]>) -> f32 {
        match self {
            Self::EuclideanSquared => {
                a.iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| (x - y).powi(2))
                    .sum()
            }
            Self::Euclidean => {
                let sq: f32 = a.iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| (x - y).powi(2))
                    .sum();
                sq.sqrt()
            }
            Self::Manhattan => {
                a.iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| (x - y).abs())
                    .sum()
            }
            Self::WeightedEuclidean => {
                match weights {
                    Some(w) => {
                        let sq: f32 = a.iter()
                            .zip(b.iter())
                            .zip(w.iter())
                            .map(|((&x, &y), &w)| (x - y).powi(2) * w)
                            .sum();
                        sq.sqrt()
                    }
                    None => self.compute(a, b, None),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

/// Result from a motion matching search.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    /// Frame index in the database.
    pub frame_index: u32,

    /// Distance/cost to the query.
    pub distance: f32,

    /// Source clip index.
    pub clip_index: u32,

    /// Time within the source clip.
    pub clip_time: f32,
}

impl SearchResult {
    /// Create a new search result.
    #[inline]
    pub fn new(frame_index: u32, distance: f32, clip_index: u32, clip_time: f32) -> Self {
        Self {
            frame_index,
            distance,
            clip_index,
            clip_time,
        }
    }
}

// ---------------------------------------------------------------------------
// MotionDatabase
// ---------------------------------------------------------------------------

/// Database of pre-processed animation frames for motion matching.
///
/// Contains all frames from source animation clips, indexed for fast
/// nearest neighbor search.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MotionDatabase {
    /// All pre-processed frames.
    pub frames: Vec<MotionFrame>,

    /// Source clip metadata.
    pub clips: Vec<ClipInfo>,

    /// KD-tree for fast nearest neighbor search.
    pub kd_tree: KdTree,

    /// Feature weights for distance computation.
    pub feature_weights: FeatureWeights,

    /// Number of joints per pose.
    pub joint_count: usize,

    /// Number of feet tracked.
    pub foot_count: usize,

    /// Feature dimensionality for indexing.
    pub feature_dims: usize,

    /// Database name.
    pub name: String,

    /// Database version.
    pub version: u32,
}

impl MotionDatabase {
    /// Create a new empty motion database.
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            clips: Vec::new(),
            kd_tree: KdTree::new(),
            feature_weights: FeatureWeights::default(),
            joint_count: 0,
            foot_count: 2,
            feature_dims: 64,
            name: String::new(),
            version: MMDB_VERSION,
        }
    }

    /// Create a named motion database.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::new()
        }
    }

    /// Get the number of frames in the database.
    #[inline]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Get the number of clips in the database.
    #[inline]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Check if the database is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Find the best matching frame for a query.
    ///
    /// Returns the best match or None if the database is empty.
    pub fn find_best_match(&self, query: &QueryFeature) -> Option<SearchResult> {
        self.find_k_best_matches(query, 1).into_iter().next()
    }

    /// Find the k best matching frames for a query.
    pub fn find_k_best_matches(&self, query: &QueryFeature, k: usize) -> Vec<SearchResult> {
        if self.frames.is_empty() || k == 0 {
            return Vec::new();
        }

        // Convert query to feature vector
        let query_features = self.query_to_features(query);

        // Use KD-tree for initial candidates
        let candidates = self.kd_tree.find_k_nearest(&query_features, k * 2);

        // Filter by tags and compute full distance
        let mut results: Vec<SearchResult> = candidates
            .into_iter()
            .filter_map(|(frame_idx, _)| {
                let frame = self.frames.get(frame_idx as usize)?;

                // Check tags
                if !frame.matches_tags(query.required_tags, query.excluded_tags) {
                    return None;
                }

                // Compute full distance
                let distance = self.compute_distance(query, frame);

                Some(SearchResult::new(
                    frame_idx,
                    distance,
                    frame.clip_index,
                    frame.time,
                ))
            })
            .collect();

        // Sort by distance and take top k
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        results
    }

    /// Convert query to feature vector for KD-tree search.
    fn query_to_features(&self, query: &QueryFeature) -> Vec<f32> {
        let mut features = Vec::with_capacity(self.feature_dims);

        // Add pose features
        for pos in &query.pose.joint_positions {
            features.push(pos.x);
            features.push(pos.y);
            features.push(pos.z);
            if features.len() >= self.feature_dims {
                break;
            }
        }

        // Add trajectory features
        if features.len() < self.feature_dims {
            for pos in &query.trajectory.future_positions {
                features.push(pos.x);
                features.push(pos.y);
                features.push(pos.z);
                if features.len() >= self.feature_dims {
                    break;
                }
            }
        }

        // Pad to feature_dims
        while features.len() < self.feature_dims {
            features.push(0.0);
        }

        features
    }

    /// Compute full weighted distance between query and frame.
    fn compute_distance(&self, query: &QueryFeature, frame: &MotionFrame) -> f32 {
        let weights = &self.feature_weights;
        let mut distance = 0.0;

        // Pose distance
        distance += frame.pose.distance_squared(
            &query.pose,
            weights.pose_position,
            weights.pose_velocity,
        );

        // Trajectory distance
        distance += frame.trajectory.distance_squared(
            &query.trajectory,
            weights.trajectory_position,
            weights.trajectory_facing,
        );

        // Foot contact distance
        let min_feet = query.foot_contacts.len().min(frame.foot_contacts.len());
        for i in 0..min_feet {
            distance += frame.foot_contacts[i].distance_squared(
                &query.foot_contacts[i],
                weights.foot_contact_state,
                weights.foot_position,
                weights.foot_velocity,
            );
        }

        distance.sqrt()
    }

    /// Get frame at index.
    #[inline]
    pub fn get_frame(&self, index: usize) -> Option<&MotionFrame> {
        self.frames.get(index)
    }

    /// Get clip info at index.
    #[inline]
    pub fn get_clip(&self, index: usize) -> Option<&ClipInfo> {
        self.clips.get(index)
    }

    /// Rebuild the KD-tree index.
    pub fn rebuild_index(&mut self) {
        self.kd_tree = KdTree::build(&self.frames, self.feature_dims);
    }

    /// Serialize to binary format.
    pub fn serialize(&self) -> io::Result<Vec<u8>> {
        let mut buffer = Vec::new();

        // Write magic and version
        buffer.write_all(&MMDB_MAGIC)?;
        buffer.write_all(&self.version.to_le_bytes())?;

        // Serialize the rest with serde_json (for simplicity)
        let json = serde_json::to_vec(self)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        // Write length and data
        let len = json.len() as u64;
        buffer.write_all(&len.to_le_bytes())?;
        buffer.write_all(&json)?;

        Ok(buffer)
    }

    /// Deserialize from binary format.
    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        if data.len() < 16 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Data too short"));
        }

        // Check magic
        if &data[0..4] != &MMDB_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic bytes"));
        }

        // Read version
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version > MMDB_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported version: {}", version),
            ));
        }

        // Read length and data
        let len = u64::from_le_bytes([
            data[8], data[9], data[10], data[11],
            data[12], data[13], data[14], data[15],
        ]) as usize;

        if data.len() < 16 + len {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Truncated data"));
        }

        let db: Self = serde_json::from_slice(&data[16..16 + len])
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        Ok(db)
    }

    /// Save to file.
    pub fn save_to_file(&self, path: &std::path::Path) -> io::Result<()> {
        let data = self.serialize()?;
        std::fs::write(path, data)
    }

    /// Load from file.
    pub fn load_from_file(path: &std::path::Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        Self::deserialize(&data)
    }
}

// ---------------------------------------------------------------------------
// MotionDatabaseBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing motion databases from animation clips.
#[derive(Clone, Debug, Default)]
pub struct MotionDatabaseBuilder {
    /// Frames being accumulated.
    frames: Vec<MotionFrame>,

    /// Clips being accumulated.
    clips: Vec<ClipInfo>,

    /// Sample rate (frames per second).
    sample_rate: f32,

    /// Joints to include in features.
    joint_indices: Vec<usize>,

    /// Foot joint indices for contact detection.
    foot_joint_indices: Vec<usize>,

    /// Feature weights.
    feature_weights: FeatureWeights,

    /// Database name.
    name: String,
}

impl MotionDatabaseBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            sample_rate: 30.0,
            feature_weights: FeatureWeights::default(),
            ..Default::default()
        }
    }

    /// Set the database name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the sample rate.
    pub fn with_sample_rate(mut self, rate: f32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Set the joints to include in features.
    pub fn with_joint_indices(mut self, indices: Vec<usize>) -> Self {
        self.joint_indices = indices;
        self
    }

    /// Set the foot joint indices.
    pub fn with_foot_joints(mut self, indices: Vec<usize>) -> Self {
        self.foot_joint_indices = indices;
        self
    }

    /// Set feature weights.
    pub fn with_feature_weights(mut self, weights: FeatureWeights) -> Self {
        self.feature_weights = weights;
        self
    }

    /// Add frames from pre-computed data.
    pub fn add_frames(&mut self, clip_name: &str, duration: f32, frames: Vec<MotionFrame>, tags: LocomotionTags) {
        let start_frame = self.frames.len() as u32;
        let frame_count = frames.len() as u32;

        let clip_info = ClipInfo {
            name: clip_name.to_string(),
            duration,
            start_frame,
            frame_count,
            sample_rate: self.sample_rate,
            default_tags: tags,
            is_looping: false,
        };

        self.clips.push(clip_info);
        self.frames.extend(frames);
    }

    /// Add a single frame.
    pub fn add_frame(&mut self, frame: MotionFrame) {
        self.frames.push(frame);
    }

    /// Build the motion database.
    pub fn build(self) -> MotionDatabase {
        let joint_count = self.frames.first()
            .map(|f| f.pose.joint_count())
            .unwrap_or(0);

        let foot_count = self.frames.first()
            .map(|f| f.foot_contacts.len())
            .unwrap_or(2);

        // Compute feature dimensionality
        let feature_dims = (joint_count * 6 + TRAJECTORY_SAMPLES * 4).min(128);

        let mut db = MotionDatabase {
            frames: self.frames,
            clips: self.clips,
            kd_tree: KdTree::new(),
            feature_weights: self.feature_weights,
            joint_count,
            foot_count,
            feature_dims,
            name: self.name,
            version: MMDB_VERSION,
        };

        // Build KD-tree index
        db.rebuild_index();

        db
    }
}

// ---------------------------------------------------------------------------
// StreamingLoader
// ---------------------------------------------------------------------------

/// Streaming loader for large motion databases.
///
/// Loads chunks of the database on demand to reduce memory usage.
#[derive(Debug)]
pub struct StreamingLoader {
    /// Path to the database file.
    path: std::path::PathBuf,

    /// Header information.
    header: Option<StreamingHeader>,

    /// Currently loaded frame range.
    loaded_range: Option<(usize, usize)>,

    /// Loaded frames.
    frames: Vec<MotionFrame>,
}

/// Header for streaming database access.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingHeader {
    /// Total number of frames.
    pub total_frames: usize,

    /// Number of clips.
    pub clip_count: usize,

    /// Frame data offset in file.
    pub frame_offset: u64,

    /// Bytes per frame (approximate).
    pub bytes_per_frame: usize,
}

impl StreamingLoader {
    /// Create a new streaming loader.
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            path: path.into(),
            header: None,
            loaded_range: None,
            frames: Vec::new(),
        }
    }

    /// Open the database and read header.
    pub fn open(&mut self) -> io::Result<()> {
        let data = std::fs::read(&self.path)?;
        let db = MotionDatabase::deserialize(&data)?;

        self.header = Some(StreamingHeader {
            total_frames: db.frames.len(),
            clip_count: db.clips.len(),
            frame_offset: 16, // After magic/version/length
            bytes_per_frame: 512, // Approximate
        });

        Ok(())
    }

    /// Load a range of frames.
    pub fn load_range(&mut self, start: usize, end: usize) -> io::Result<()> {
        let data = std::fs::read(&self.path)?;
        let db = MotionDatabase::deserialize(&data)?;

        let actual_end = end.min(db.frames.len());
        let actual_start = start.min(actual_end);

        self.frames = db.frames[actual_start..actual_end].to_vec();
        self.loaded_range = Some((actual_start, actual_end));

        Ok(())
    }

    /// Get a loaded frame.
    pub fn get_frame(&self, global_index: usize) -> Option<&MotionFrame> {
        let (start, end) = self.loaded_range?;
        if global_index >= start && global_index < end {
            self.frames.get(global_index - start)
        } else {
            None
        }
    }

    /// Get the header.
    pub fn header(&self) -> Option<&StreamingHeader> {
        self.header.as_ref()
    }

    /// Get the loaded frame count.
    pub fn loaded_count(&self) -> usize {
        self.frames.len()
    }
}

// ---------------------------------------------------------------------------
// Display implementations
// ---------------------------------------------------------------------------

impl fmt::Display for MotionDatabase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MotionDatabase({} frames, {} clips, {} joints)",
            self.frame_count(),
            self.clip_count(),
            self.joint_count
        )
    }
}

impl fmt::Display for LocomotionTags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =====================================================================
    // PoseFeature tests
    // =====================================================================

    #[test]
    fn test_pose_feature_new() {
        let pose = PoseFeature::new();
        assert!(pose.is_empty());
        assert_eq!(pose.joint_count(), 0);
    }

    #[test]
    fn test_pose_feature_with_joint_count() {
        let pose = PoseFeature::with_joint_count(10);
        assert!(!pose.is_empty());
        assert_eq!(pose.joint_count(), 10);
        assert_eq!(pose.joint_positions.len(), 10);
        assert_eq!(pose.joint_velocities.len(), 10);
    }

    #[test]
    fn test_pose_feature_distance_squared_identical() {
        let pose = PoseFeature::with_joint_count(5);
        let dist = pose.distance_squared(&pose, 1.0, 1.0);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_pose_feature_distance_squared_different() {
        let mut pose_a = PoseFeature::with_joint_count(2);
        pose_a.joint_positions[0] = Vec3::new(1.0, 0.0, 0.0);

        let pose_b = PoseFeature::with_joint_count(2);
        let dist = pose_a.distance_squared(&pose_b, 1.0, 0.0);
        assert_eq!(dist, 1.0);
    }

    #[test]
    fn test_pose_feature_to_flat_vector() {
        let mut pose = PoseFeature::with_joint_count(1);
        pose.joint_positions[0] = Vec3::new(1.0, 2.0, 3.0);
        pose.joint_velocities[0] = Vec3::new(4.0, 5.0, 6.0);

        let flat = pose.to_flat_vector();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_pose_feature_from_flat_vector() {
        let flat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pose = PoseFeature::from_flat_vector(&flat, 1);

        assert_eq!(pose.joint_positions[0], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(pose.joint_velocities[0], Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn test_pose_feature_roundtrip() {
        let mut pose = PoseFeature::with_joint_count(3);
        pose.joint_positions[0] = Vec3::new(1.0, 2.0, 3.0);
        pose.joint_positions[1] = Vec3::new(4.0, 5.0, 6.0);
        pose.joint_velocities[2] = Vec3::new(7.0, 8.0, 9.0);

        let flat = pose.to_flat_vector();
        let restored = PoseFeature::from_flat_vector(&flat, 3);

        assert_eq!(pose.joint_positions, restored.joint_positions);
        assert_eq!(pose.joint_velocities, restored.joint_velocities);
    }

    // =====================================================================
    // TrajectoryFeature tests
    // =====================================================================

    #[test]
    fn test_trajectory_feature_new() {
        let traj = TrajectoryFeature::new();
        assert_eq!(traj.future_positions, [Vec3::ZERO; 3]);
        assert_eq!(traj.future_facings, [0.0; 3]);
    }

    #[test]
    fn test_trajectory_feature_from_predictions() {
        let positions = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let facings = [0.5, 1.0, 1.5];

        let traj = TrajectoryFeature::from_predictions(positions, facings);
        assert_eq!(traj.future_positions[0], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(traj.future_facings[2], 1.5);
    }

    #[test]
    fn test_trajectory_feature_distance_squared_identical() {
        let traj = TrajectoryFeature::from_predictions(
            [Vec3::X, Vec3::Y, Vec3::Z],
            [0.0, 1.0, 2.0],
        );
        let dist = traj.distance_squared(&traj, 1.0, 1.0);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_trajectory_feature_distance_squared_different() {
        let traj_a = TrajectoryFeature::new();
        let traj_b = TrajectoryFeature::from_predictions(
            [Vec3::X, Vec3::ZERO, Vec3::ZERO],
            [0.0, 0.0, 0.0],
        );
        let dist = traj_a.distance_squared(&traj_b, 1.0, 0.0);
        assert_eq!(dist, 1.0);
    }

    #[test]
    fn test_trajectory_feature_to_flat_vector() {
        let traj = TrajectoryFeature::from_predictions(
            [Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO, Vec3::ZERO],
            [0.5, 0.0, 0.0],
        );
        let flat = traj.to_flat_vector();
        assert_eq!(flat[0], 1.0);
        assert_eq!(flat[1], 2.0);
        assert_eq!(flat[2], 3.0);
        assert_eq!(flat[9], 0.5);
    }

    #[test]
    fn test_trajectory_feature_from_flat_vector() {
        let flat = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0];
        let traj = TrajectoryFeature::from_flat_vector(&flat);
        assert_eq!(traj.future_positions[0], Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(traj.future_facings[0], 0.5);
    }

    #[test]
    fn test_trajectory_feature_lerp() {
        let traj_a = TrajectoryFeature::new();
        let traj_b = TrajectoryFeature::from_predictions(
            [Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, Vec3::ZERO],
            [1.0, 0.0, 0.0],
        );

        let result = traj_a.lerp(&traj_b, 0.5);
        assert_eq!(result.future_positions[0], Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(result.future_facings[0], 0.5);
    }

    // =====================================================================
    // FootContact tests
    // =====================================================================

    #[test]
    fn test_foot_contact_new() {
        let contact = FootContact::new();
        assert!(!contact.is_planted);
        assert_eq!(contact.position, Vec3::ZERO);
        assert_eq!(contact.velocity, Vec3::ZERO);
    }

    #[test]
    fn test_foot_contact_planted() {
        let contact = FootContact::planted(Vec3::new(1.0, 0.0, 2.0));
        assert!(contact.is_planted);
        assert_eq!(contact.position, Vec3::new(1.0, 0.0, 2.0));
        assert_eq!(contact.velocity, Vec3::ZERO);
    }

    #[test]
    fn test_foot_contact_moving() {
        let contact = FootContact::moving(Vec3::X, Vec3::Y);
        assert!(!contact.is_planted);
        assert_eq!(contact.position, Vec3::X);
        assert_eq!(contact.velocity, Vec3::Y);
    }

    #[test]
    fn test_foot_contact_distance_same_state() {
        let a = FootContact::planted(Vec3::ZERO);
        let b = FootContact::planted(Vec3::X);
        let dist = a.distance_squared(&b, 10.0, 1.0, 1.0);
        assert_eq!(dist, 1.0); // Only position difference
    }

    #[test]
    fn test_foot_contact_distance_different_state() {
        let a = FootContact::planted(Vec3::ZERO);
        let b = FootContact::moving(Vec3::ZERO, Vec3::ZERO);
        let dist = a.distance_squared(&b, 10.0, 1.0, 1.0);
        assert_eq!(dist, 10.0); // State penalty
    }

    // =====================================================================
    // MotionFrame tests
    // =====================================================================

    #[test]
    fn test_motion_frame_new() {
        let frame = MotionFrame::new();
        assert_eq!(frame.clip_index, 0);
        assert_eq!(frame.time, 0.0);
        assert!(frame.pose.is_empty());
        assert!(frame.tags.is_empty());
    }

    #[test]
    fn test_motion_frame_from_clip() {
        let frame = MotionFrame::from_clip(5, 1.5);
        assert_eq!(frame.clip_index, 5);
        assert_eq!(frame.time, 1.5);
    }

    #[test]
    fn test_motion_frame_matches_tags() {
        let mut frame = MotionFrame::new();
        frame.tags = LocomotionTags::WALK | LocomotionTags::TERRAIN_FLAT;

        assert!(frame.matches_tags(LocomotionTags::WALK, LocomotionTags::empty()));
        assert!(frame.matches_tags(LocomotionTags::TERRAIN_FLAT, LocomotionTags::empty()));
        assert!(!frame.matches_tags(LocomotionTags::RUN, LocomotionTags::empty()));
        assert!(!frame.matches_tags(LocomotionTags::WALK, LocomotionTags::TERRAIN_FLAT));
    }

    #[test]
    fn test_motion_frame_matches_tags_empty_required() {
        let frame = MotionFrame::new();
        assert!(frame.matches_tags(LocomotionTags::empty(), LocomotionTags::empty()));
    }

    // =====================================================================
    // ClipInfo tests
    // =====================================================================

    #[test]
    fn test_clip_info_new() {
        let clip = ClipInfo::new("walk", 2.0, 30.0);
        assert_eq!(clip.name, "walk");
        assert_eq!(clip.duration, 2.0);
        assert_eq!(clip.sample_rate, 30.0);
        assert_eq!(clip.start_frame, 0);
        assert_eq!(clip.frame_count, 0);
    }

    #[test]
    fn test_clip_info_end_frame() {
        let mut clip = ClipInfo::new("test", 1.0, 30.0);
        clip.start_frame = 10;
        clip.frame_count = 30;
        assert_eq!(clip.end_frame(), 40);
    }

    // =====================================================================
    // FeatureWeights tests
    // =====================================================================

    #[test]
    fn test_feature_weights_default() {
        let weights = FeatureWeights::default();
        assert_eq!(weights.pose_position, 1.0);
        assert!(weights.pose_velocity < weights.pose_position);
    }

    #[test]
    fn test_feature_weights_uniform() {
        let weights = FeatureWeights::uniform();
        assert_eq!(weights.pose_position, weights.pose_velocity);
        assert_eq!(weights.trajectory_position, weights.trajectory_facing);
    }

    #[test]
    fn test_feature_weights_trajectory_focused() {
        let weights = FeatureWeights::trajectory_focused();
        assert!(weights.trajectory_position > weights.pose_position);
    }

    #[test]
    fn test_feature_weights_pose_focused() {
        let weights = FeatureWeights::pose_focused();
        assert!(weights.pose_position > weights.trajectory_position);
    }

    // =====================================================================
    // LocomotionTags tests
    // =====================================================================

    #[test]
    fn test_locomotion_tags_basic() {
        let tags = LocomotionTags::WALK | LocomotionTags::RUN;
        assert!(tags.contains(LocomotionTags::WALK));
        assert!(tags.contains(LocomotionTags::RUN));
        assert!(!tags.contains(LocomotionTags::SPRINT));
    }

    #[test]
    fn test_locomotion_tags_contains_all() {
        let tags = LocomotionTags::WALK | LocomotionTags::TERRAIN_FLAT;
        assert!(tags.contains_all(LocomotionTags::WALK));
        assert!(tags.contains_all(LocomotionTags::WALK | LocomotionTags::TERRAIN_FLAT));
        assert!(!tags.contains_all(LocomotionTags::WALK | LocomotionTags::RUN));
    }

    #[test]
    fn test_locomotion_tags_contains_any() {
        let tags = LocomotionTags::WALK | LocomotionTags::TERRAIN_FLAT;
        assert!(tags.contains_any(LocomotionTags::WALK));
        assert!(tags.contains_any(LocomotionTags::WALK | LocomotionTags::RUN));
        assert!(!tags.contains_any(LocomotionTags::RUN | LocomotionTags::SPRINT));
    }

    #[test]
    fn test_locomotion_tags_terrain() {
        let uphill = LocomotionTags::TERRAIN_UPHILL;
        let downhill = LocomotionTags::TERRAIN_DOWNHILL;
        assert!(!uphill.intersects(downhill));
    }

    #[test]
    fn test_locomotion_tags_transition() {
        let tags = LocomotionTags::TRANSITION_START | LocomotionTags::WALK;
        assert!(tags.contains(LocomotionTags::TRANSITION_START));
        assert!(tags.contains(LocomotionTags::WALK));
    }

    // =====================================================================
    // KdTree tests
    // =====================================================================

    #[test]
    fn test_kd_tree_new() {
        let tree = KdTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.node_count(), 0);
    }

    #[test]
    fn test_kd_tree_build_empty() {
        let tree = KdTree::build(&[], 10);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_kd_tree_build_single() {
        let frame = MotionFrame::new();
        let tree = KdTree::build(&[frame], 10);
        assert!(!tree.is_empty());
        assert_eq!(tree.node_count(), 1);
    }

    #[test]
    fn test_kd_tree_build_multiple() {
        let frames: Vec<MotionFrame> = (0..10)
            .map(|i| {
                let mut frame = MotionFrame::new();
                frame.pose = PoseFeature::with_joint_count(1);
                frame.pose.joint_positions[0] = Vec3::new(i as f32, 0.0, 0.0);
                frame
            })
            .collect();

        let tree = KdTree::build(&frames, 10);
        assert!(!tree.is_empty());
        assert!(tree.node_count() > 0);
    }

    #[test]
    fn test_kd_tree_find_k_nearest_empty() {
        let tree = KdTree::new();
        let results = tree.find_k_nearest(&[0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_kd_tree_find_k_nearest_single() {
        let frame = MotionFrame::new();
        let tree = KdTree::build(&[frame], 10);

        let results = tree.find_k_nearest(&vec![0.0; 10], 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_kd_tree_find_k_nearest_correctness() {
        let frames: Vec<MotionFrame> = (0..5)
            .map(|i| {
                let mut frame = MotionFrame::new();
                frame.pose = PoseFeature::with_joint_count(1);
                frame.pose.joint_positions[0] = Vec3::new(i as f32 * 10.0, 0.0, 0.0);
                frame
            })
            .collect();

        let tree = KdTree::build(&frames, 10);

        // Query close to frame 0
        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = tree.find_k_nearest(&query, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0); // Closest to origin
    }

    #[test]
    fn test_kd_tree_find_k_nearest_sorted() {
        let frames: Vec<MotionFrame> = (0..10)
            .map(|i| {
                let mut frame = MotionFrame::new();
                frame.pose = PoseFeature::with_joint_count(1);
                frame.pose.joint_positions[0] = Vec3::new(i as f32, 0.0, 0.0);
                frame
            })
            .collect();

        let tree = KdTree::build(&frames, 10);
        let query = vec![5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = tree.find_k_nearest(&query, 3);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    // =====================================================================
    // VpTree tests
    // =====================================================================

    #[test]
    fn test_vp_tree_new() {
        let tree = VpTree::new();
        assert!(tree.is_empty());
    }

    #[test]
    fn test_vp_tree_build_empty() {
        let tree = VpTree::build(Vec::new());
        assert!(tree.is_empty());
    }

    #[test]
    fn test_vp_tree_build_single() {
        let tree = VpTree::build(vec![vec![1.0, 2.0, 3.0]]);
        assert!(!tree.is_empty());
        assert_eq!(tree.node_count(), 1);
    }

    #[test]
    fn test_vp_tree_build_multiple() {
        let features: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32, 0.0, 0.0])
            .collect();
        let tree = VpTree::build(features);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_vp_tree_find_k_nearest_empty() {
        let tree = VpTree::new();
        let results = tree.find_k_nearest(&[0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_vp_tree_find_k_nearest() {
        let features: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32 * 10.0, 0.0, 0.0])
            .collect();
        let tree = VpTree::build(features);

        let results = tree.find_k_nearest(&[25.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
    }

    // =====================================================================
    // DistanceMetric tests
    // =====================================================================

    #[test]
    fn test_distance_metric_euclidean_squared() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        let dist = DistanceMetric::EuclideanSquared.compute(&a, &b, None);
        assert_eq!(dist, 25.0);
    }

    #[test]
    fn test_distance_metric_euclidean() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        let dist = DistanceMetric::Euclidean.compute(&a, &b, None);
        assert_eq!(dist, 5.0);
    }

    #[test]
    fn test_distance_metric_manhattan() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        let dist = DistanceMetric::Manhattan.compute(&a, &b, None);
        assert_eq!(dist, 7.0);
    }

    #[test]
    fn test_distance_metric_weighted() {
        let a = [0.0, 0.0];
        let b = [1.0, 1.0];
        let weights = [4.0, 1.0];
        let dist = DistanceMetric::WeightedEuclidean.compute(&a, &b, Some(&weights));
        assert!((dist - (5.0_f32).sqrt()).abs() < 0.001);
    }

    // =====================================================================
    // MotionDatabase tests
    // =====================================================================

    #[test]
    fn test_motion_database_new() {
        let db = MotionDatabase::new();
        assert!(db.is_empty());
        assert_eq!(db.frame_count(), 0);
        assert_eq!(db.clip_count(), 0);
    }

    #[test]
    fn test_motion_database_with_name() {
        let db = MotionDatabase::with_name("test_db");
        assert_eq!(db.name, "test_db");
    }

    #[test]
    fn test_motion_database_find_best_match_empty() {
        let db = MotionDatabase::new();
        let query = QueryFeature::new();
        let result = db.find_best_match(&query);
        assert!(result.is_none());
    }

    #[test]
    fn test_motion_database_find_k_best_matches_empty() {
        let db = MotionDatabase::new();
        let query = QueryFeature::new();
        let results = db.find_k_best_matches(&query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_motion_database_get_frame() {
        let mut db = MotionDatabase::new();
        db.frames.push(MotionFrame::from_clip(0, 0.5));

        assert!(db.get_frame(0).is_some());
        assert!(db.get_frame(1).is_none());
    }

    #[test]
    fn test_motion_database_get_clip() {
        let mut db = MotionDatabase::new();
        db.clips.push(ClipInfo::new("walk", 2.0, 30.0));

        assert!(db.get_clip(0).is_some());
        assert_eq!(db.get_clip(0).unwrap().name, "walk");
        assert!(db.get_clip(1).is_none());
    }

    #[test]
    fn test_motion_database_serialization_roundtrip() {
        let mut db = MotionDatabase::with_name("test");
        db.frames.push(MotionFrame::from_clip(0, 0.5));
        db.clips.push(ClipInfo::new("walk", 2.0, 30.0));

        let data = db.serialize().unwrap();
        let restored = MotionDatabase::deserialize(&data).unwrap();

        assert_eq!(restored.name, "test");
        assert_eq!(restored.frame_count(), 1);
        assert_eq!(restored.clip_count(), 1);
    }

    #[test]
    fn test_motion_database_serialization_invalid_magic() {
        let data = vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let result = MotionDatabase::deserialize(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_motion_database_serialization_too_short() {
        let data = vec![b'M', b'M', b'D', b'B'];
        let result = MotionDatabase::deserialize(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_motion_database_rebuild_index() {
        let mut db = MotionDatabase::new();
        db.feature_dims = 10;

        let mut frame = MotionFrame::new();
        frame.pose = PoseFeature::with_joint_count(1);
        db.frames.push(frame);

        db.rebuild_index();
        assert!(!db.kd_tree.is_empty());
    }

    // =====================================================================
    // MotionDatabaseBuilder tests
    // =====================================================================

    #[test]
    fn test_motion_database_builder_new() {
        let builder = MotionDatabaseBuilder::new();
        assert_eq!(builder.sample_rate, 30.0);
    }

    #[test]
    fn test_motion_database_builder_with_name() {
        let builder = MotionDatabaseBuilder::new().with_name("test");
        assert_eq!(builder.name, "test");
    }

    #[test]
    fn test_motion_database_builder_with_sample_rate() {
        let builder = MotionDatabaseBuilder::new().with_sample_rate(60.0);
        assert_eq!(builder.sample_rate, 60.0);
    }

    #[test]
    fn test_motion_database_builder_build_empty() {
        let db = MotionDatabaseBuilder::new().build();
        assert!(db.is_empty());
    }

    #[test]
    fn test_motion_database_builder_add_frame() {
        let mut builder = MotionDatabaseBuilder::new();
        builder.add_frame(MotionFrame::new());

        let db = builder.build();
        assert_eq!(db.frame_count(), 1);
    }

    #[test]
    fn test_motion_database_builder_add_frames() {
        let mut builder = MotionDatabaseBuilder::new();

        let frames: Vec<MotionFrame> = (0..10)
            .map(|i| MotionFrame::from_clip(0, i as f32 * 0.033))
            .collect();

        builder.add_frames("walk", 0.3, frames, LocomotionTags::WALK);

        let db = builder.build();
        assert_eq!(db.frame_count(), 10);
        assert_eq!(db.clip_count(), 1);
        assert_eq!(db.clips[0].name, "walk");
        assert_eq!(db.clips[0].default_tags, LocomotionTags::WALK);
    }

    #[test]
    fn test_motion_database_builder_multiple_clips() {
        let mut builder = MotionDatabaseBuilder::new();

        let walk_frames: Vec<MotionFrame> = (0..5)
            .map(|i| MotionFrame::from_clip(0, i as f32 * 0.1))
            .collect();
        let run_frames: Vec<MotionFrame> = (0..5)
            .map(|i| MotionFrame::from_clip(1, i as f32 * 0.05))
            .collect();

        builder.add_frames("walk", 0.5, walk_frames, LocomotionTags::WALK);
        builder.add_frames("run", 0.25, run_frames, LocomotionTags::RUN);

        let db = builder.build();
        assert_eq!(db.frame_count(), 10);
        assert_eq!(db.clip_count(), 2);
        assert_eq!(db.clips[0].start_frame, 0);
        assert_eq!(db.clips[1].start_frame, 5);
    }

    // =====================================================================
    // QueryFeature tests
    // =====================================================================

    #[test]
    fn test_query_feature_new() {
        let query = QueryFeature::new();
        assert!(query.pose.is_empty());
        assert!(query.required_tags.is_empty());
        assert!(query.excluded_tags.is_empty());
    }

    #[test]
    fn test_query_feature_with_required_tags() {
        let query = QueryFeature::new().with_required_tags(LocomotionTags::WALK);
        assert_eq!(query.required_tags, LocomotionTags::WALK);
    }

    #[test]
    fn test_query_feature_with_excluded_tags() {
        let query = QueryFeature::new().with_excluded_tags(LocomotionTags::RUN);
        assert_eq!(query.excluded_tags, LocomotionTags::RUN);
    }

    // =====================================================================
    // SearchResult tests
    // =====================================================================

    #[test]
    fn test_search_result_new() {
        let result = SearchResult::new(10, 0.5, 2, 1.5);
        assert_eq!(result.frame_index, 10);
        assert_eq!(result.distance, 0.5);
        assert_eq!(result.clip_index, 2);
        assert_eq!(result.clip_time, 1.5);
    }

    // =====================================================================
    // StreamingLoader tests
    // =====================================================================

    #[test]
    fn test_streaming_loader_new() {
        let loader = StreamingLoader::new("/tmp/test.mmdb");
        assert!(loader.header().is_none());
        assert_eq!(loader.loaded_count(), 0);
    }

    #[test]
    fn test_streaming_loader_get_frame_no_load() {
        let loader = StreamingLoader::new("/tmp/test.mmdb");
        assert!(loader.get_frame(0).is_none());
    }

    // =====================================================================
    // Integration tests
    // =====================================================================

    #[test]
    fn test_full_pipeline() {
        // Build a database
        let mut builder = MotionDatabaseBuilder::new()
            .with_name("locomotion")
            .with_sample_rate(30.0)
            .with_feature_weights(FeatureWeights::default());

        // Create walk frames
        let walk_frames: Vec<MotionFrame> = (0..30)
            .map(|i| {
                let mut frame = MotionFrame::from_clip(0, i as f32 / 30.0);
                frame.pose = PoseFeature::with_joint_count(5);
                frame.pose.joint_positions[0] = Vec3::new(0.0, 1.0, i as f32 * 0.1);
                frame.trajectory = TrajectoryFeature::from_predictions(
                    [
                        Vec3::new(0.0, 0.0, 0.2),
                        Vec3::new(0.0, 0.0, 0.5),
                        Vec3::new(0.0, 0.0, 1.0),
                    ],
                    [0.0, 0.0, 0.0],
                );
                frame.foot_contacts = vec![
                    FootContact::planted(Vec3::ZERO),
                    FootContact::moving(Vec3::Y, Vec3::X),
                ];
                frame.tags = LocomotionTags::WALK | LocomotionTags::TERRAIN_FLAT;
                frame
            })
            .collect();

        builder.add_frames("walk", 1.0, walk_frames, LocomotionTags::WALK | LocomotionTags::TERRAIN_FLAT);

        let db = builder.build();

        // Verify database
        assert_eq!(db.frame_count(), 30);
        assert_eq!(db.clip_count(), 1);
        assert!(!db.kd_tree.is_empty());

        // Create query
        let mut query = QueryFeature::new();
        query.pose = PoseFeature::with_joint_count(5);
        query.pose.joint_positions[0] = Vec3::new(0.0, 1.0, 0.5);
        query.trajectory = TrajectoryFeature::from_predictions(
            [
                Vec3::new(0.0, 0.0, 0.2),
                Vec3::new(0.0, 0.0, 0.5),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            [0.0, 0.0, 0.0],
        );
        query.foot_contacts = vec![
            FootContact::planted(Vec3::ZERO),
            FootContact::moving(Vec3::Y, Vec3::X),
        ];
        query.required_tags = LocomotionTags::WALK;

        // Find best match
        let result = db.find_best_match(&query);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.clip_index, 0);

        // Find k best matches
        let results = db.find_k_best_matches(&query, 5);
        assert!(results.len() <= 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_tag_filtering() {
        let mut builder = MotionDatabaseBuilder::new();

        // Add walk frames
        let walk_frames: Vec<MotionFrame> = (0..10)
            .map(|i| {
                let mut frame = MotionFrame::from_clip(0, i as f32 * 0.1);
                frame.tags = LocomotionTags::WALK;
                frame
            })
            .collect();

        // Add run frames
        let run_frames: Vec<MotionFrame> = (0..10)
            .map(|i| {
                let mut frame = MotionFrame::from_clip(1, i as f32 * 0.05);
                frame.tags = LocomotionTags::RUN;
                frame
            })
            .collect();

        builder.add_frames("walk", 1.0, walk_frames, LocomotionTags::WALK);
        builder.add_frames("run", 0.5, run_frames, LocomotionTags::RUN);

        let db = builder.build();

        // Query for walk only
        let query = QueryFeature::new().with_required_tags(LocomotionTags::WALK);
        let results = db.find_k_best_matches(&query, 20);

        for result in &results {
            let frame = db.get_frame(result.frame_index as usize).unwrap();
            assert!(frame.tags.contains(LocomotionTags::WALK));
        }

        // Query excluding run
        let query = QueryFeature::new().with_excluded_tags(LocomotionTags::RUN);
        let results = db.find_k_best_matches(&query, 20);

        for result in &results {
            let frame = db.get_frame(result.frame_index as usize).unwrap();
            assert!(!frame.tags.contains(LocomotionTags::RUN));
        }
    }

    #[test]
    fn test_large_database_performance() {
        let mut builder = MotionDatabaseBuilder::new();

        // Create 1000 frames
        let frames: Vec<MotionFrame> = (0..1000)
            .map(|i| {
                let mut frame = MotionFrame::from_clip(0, i as f32 * 0.001);
                frame.pose = PoseFeature::with_joint_count(10);
                frame.pose.joint_positions[0] = Vec3::new(
                    (i as f32 * 0.1).sin(),
                    (i as f32 * 0.1).cos(),
                    i as f32 * 0.01,
                );
                frame.tags = LocomotionTags::WALK;
                frame
            })
            .collect();

        builder.add_frames("large", 1.0, frames, LocomotionTags::WALK);
        let db = builder.build();

        // Query should be fast
        let query = QueryFeature::new();
        let start = std::time::Instant::now();
        let results = db.find_k_best_matches(&query, 10);
        let elapsed = start.elapsed();

        assert!(!results.is_empty());
        assert!(elapsed.as_millis() < 100); // Should be fast
    }

    #[test]
    fn test_display_impl() {
        let db = MotionDatabase::with_name("test");
        let display = format!("{}", db);
        assert!(display.contains("MotionDatabase"));
        assert!(display.contains("0 frames"));
    }

    #[test]
    fn test_kd_tree_node_leaf() {
        let node = KdTreeNode::leaf(42);
        assert!(node.is_leaf);
        assert_eq!(node.data, 42);
    }

    #[test]
    fn test_kd_tree_node_internal() {
        let node = KdTreeNode::internal(2, 5.0, 1, 3);
        assert!(!node.is_leaf);
        assert_eq!(node.data, 2);
        assert_eq!(node.split_value, 5.0);
        assert_eq!(node.left, 1);
        assert_eq!(node.right, 3);
    }
}
