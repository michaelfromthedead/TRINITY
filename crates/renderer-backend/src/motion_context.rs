//! Motion matching context system for TRINITY Engine (T-AN-6.5).
//!
//! This module provides the context management system for motion matching,
//! handling character state, trajectory requests, tag management, foot contact
//! tracking, and context interpolation for smooth, responsive animations.
//!
//! # Architecture
//!
//! ```text
//! MotionContext
//! ├── current_pose: PoseFeatures       # Current character pose features
//! ├── desired_trajectory: TrajectoryRequest
//! ├── motion_state: MotionState        # Current clip, frame, time
//! ├── active_tags: TagSet              # Active locomotion/terrain tags
//! ├── foot_contacts: FootContactTracker
//! └── update_policy: ContextUpdatePolicy
//!
//! ContextBuilder
//! ├── from_controller_input()          # Build from character controller
//! ├── from_navigation_path()           # Build from AI navigation
//! ├── from_animation_events()          # Build from animation events
//! └── combine()                        # Combine multiple context sources
//!
//! TagManager
//! ├── active_tags: TagSet
//! ├── priority_rules: Vec<TagPriority>
//! ├── exclusion_rules: Vec<TagExclusion>
//! └── terrain_overrides: HashMap
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::motion_context::{
//!     MotionContext, ContextBuilder, TrajectoryRequest, TagManager,
//! };
//!
//! // Build context from controller input
//! let context = ContextBuilder::new()
//!     .from_controller_input(&input, dt)
//!     .with_trajectory(trajectory)
//!     .with_tags(tags)
//!     .build();
//!
//! // Use context for motion search
//! let query = context.to_search_query();
//! let result = searcher.search(&query);
//! ```

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::motion_features::{FootFeatures, LocomotionStyle, MotionFeatures, MotionTags, TerrainType};
use crate::motion_matching_db::{LocomotionTags, TrajectoryFeature, TRAJECTORY_SAMPLES};
use crate::motion_search::{SearchQuery, TrajectoryFeatureCompat};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default trajectory sample times in seconds.
pub const TRAJECTORY_TIMES: [f32; 4] = [0.2, 0.4, 0.6, 1.0];

/// Default update interval (frames between searches).
pub const DEFAULT_UPDATE_INTERVAL: u32 = 4;

/// Default cost threshold for triggering new search.
pub const DEFAULT_SEARCH_THRESHOLD: f32 = 0.3;

/// Default cooldown after transition (seconds).
pub const DEFAULT_TRANSITION_COOLDOWN: f32 = 0.15;

/// Default hysteresis factor for preventing oscillation.
pub const DEFAULT_HYSTERESIS: f32 = 0.1;

/// Maximum trajectory samples.
pub const MAX_TRAJECTORY_SAMPLES: usize = 8;

/// Default interpolation smoothness (0-1).
pub const DEFAULT_INTERPOLATION_SMOOTHNESS: f32 = 0.2;

/// Minimum speed threshold for locomotion.
pub const MIN_SPEED_THRESHOLD: f32 = 0.05;

/// Contact height threshold for foot detection.
pub const CONTACT_HEIGHT_THRESHOLD: f32 = 0.1;

/// Contact velocity threshold for foot detection (m/s).
pub const CONTACT_VELOCITY_THRESHOLD: f32 = 0.15;

// ---------------------------------------------------------------------------
// TrajectoryPoint
// ---------------------------------------------------------------------------

/// A single point in a trajectory request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    /// Future position relative to current root.
    pub position: Vec3,

    /// Future facing direction (normalized XZ).
    pub facing: Vec3,

    /// Speed at this point (m/s).
    pub speed: f32,

    /// Time offset from current time (seconds).
    pub time_offset: f32,
}

impl TrajectoryPoint {
    /// Create a new trajectory point.
    #[inline]
    pub fn new(position: Vec3, facing: Vec3, speed: f32, time_offset: f32) -> Self {
        Self {
            position,
            facing: facing.normalize_or_zero(),
            speed,
            time_offset,
        }
    }

    /// Create from position only (facing derived from direction).
    pub fn from_position(position: Vec3, time_offset: f32) -> Self {
        let facing = if position.length_squared() > 1e-6 {
            Vec3::new(position.x, 0.0, position.z).normalize_or_zero()
        } else {
            Vec3::Z
        };
        let speed = position.length() / time_offset.max(0.001);

        Self {
            position,
            facing,
            speed,
            time_offset,
        }
    }

    /// Get the facing angle in radians.
    #[inline]
    pub fn facing_angle(&self) -> f32 {
        self.facing.x.atan2(self.facing.z)
    }

    /// Interpolate between two trajectory points.
    pub fn lerp(&self, other: &TrajectoryPoint, t: f32) -> Self {
        Self {
            position: self.position.lerp(other.position, t),
            facing: self.facing.lerp(other.facing, t).normalize_or_zero(),
            speed: self.speed + (other.speed - self.speed) * t,
            time_offset: self.time_offset + (other.time_offset - self.time_offset) * t,
        }
    }

    /// Compute distance to another point.
    #[inline]
    pub fn distance(&self, other: &TrajectoryPoint) -> f32 {
        let pos_dist = (self.position - other.position).length();
        let facing_diff = (self.facing_angle() - other.facing_angle()).abs();
        let facing_wrapped = facing_diff.min(std::f32::consts::TAU - facing_diff);
        pos_dist + facing_wrapped * 0.5
    }
}

// ---------------------------------------------------------------------------
// TrajectoryRequest
// ---------------------------------------------------------------------------

/// Desired future trajectory for motion matching.
///
/// Contains predicted future positions and facings at multiple time offsets,
/// along with speed profile and curvature hints.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryRequest {
    /// Future trajectory points.
    pub points: Vec<TrajectoryPoint>,

    /// Speed profile (average speed).
    pub speed_profile: f32,

    /// Curvature hint (-1 = sharp left, 0 = straight, 1 = sharp right).
    pub curvature: f32,

    /// Whether the trajectory involves stopping.
    pub stopping: bool,

    /// Whether the trajectory involves starting.
    pub starting: bool,

    /// Urgency factor (0 = relaxed, 1 = urgent).
    pub urgency: f32,
}

impl TrajectoryRequest {
    /// Create a new empty trajectory request.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a straight trajectory.
    pub fn straight(speed: f32, duration: f32) -> Self {
        let mut request = Self::new();
        request.speed_profile = speed;

        for &t in &TRAJECTORY_TIMES {
            if t <= duration {
                let distance = speed * t;
                request.points.push(TrajectoryPoint::new(
                    Vec3::new(0.0, 0.0, distance),
                    Vec3::Z,
                    speed,
                    t,
                ));
            }
        }

        request
    }

    /// Create a turning trajectory.
    pub fn turning(speed: f32, turn_rate: f32, duration: f32) -> Self {
        let mut request = Self::new();
        request.speed_profile = speed;
        request.curvature = turn_rate.signum() * (turn_rate.abs() / std::f32::consts::PI).min(1.0);

        for &t in &TRAJECTORY_TIMES {
            if t <= duration {
                let angle = turn_rate * t;
                let distance = speed * t;
                let x = distance * angle.sin();
                let z = distance * angle.cos();
                let facing = Vec3::new(angle.sin(), 0.0, angle.cos());

                request.points.push(TrajectoryPoint::new(
                    Vec3::new(x, 0.0, z),
                    facing,
                    speed,
                    t,
                ));
            }
        }

        request
    }

    /// Create a stopping trajectory.
    pub fn stopping(current_speed: f32, decel_time: f32) -> Self {
        let mut request = Self::new();
        request.stopping = true;
        request.speed_profile = current_speed * 0.5;

        for &t in &TRAJECTORY_TIMES {
            let progress = (t / decel_time).min(1.0);
            let speed = current_speed * (1.0 - progress);
            let distance = current_speed * t * (1.0 - progress * 0.5);

            request.points.push(TrajectoryPoint::new(
                Vec3::new(0.0, 0.0, distance),
                Vec3::Z,
                speed,
                t,
            ));
        }

        request
    }

    /// Create a starting trajectory.
    pub fn starting(target_speed: f32, accel_time: f32, direction: Vec3) -> Self {
        let mut request = Self::new();
        request.starting = true;
        request.speed_profile = target_speed * 0.5;

        let dir = direction.normalize_or_zero();

        for &t in &TRAJECTORY_TIMES {
            let progress = (t / accel_time).min(1.0);
            let speed = target_speed * progress;
            let distance = target_speed * t * progress * 0.5;

            request.points.push(TrajectoryPoint::new(
                dir * distance,
                dir,
                speed,
                t,
            ));
        }

        request
    }

    /// Get the number of trajectory points.
    #[inline]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Check if the trajectory is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get point at time offset (interpolated).
    pub fn sample_at(&self, time: f32) -> TrajectoryPoint {
        if self.points.is_empty() {
            return TrajectoryPoint::default();
        }

        if self.points.len() == 1 {
            return self.points[0];
        }

        // Find surrounding points
        for i in 0..self.points.len() - 1 {
            let p0 = &self.points[i];
            let p1 = &self.points[i + 1];

            if time >= p0.time_offset && time <= p1.time_offset {
                let t = (time - p0.time_offset) / (p1.time_offset - p0.time_offset);
                return p0.lerp(p1, t);
            }
        }

        // Beyond range, return last
        self.points.last().copied().unwrap_or_default()
    }

    /// Convert to TrajectoryFeature (for motion_matching_db).
    pub fn to_trajectory_feature(&self) -> TrajectoryFeature {
        let mut positions = [Vec3::ZERO; TRAJECTORY_SAMPLES];
        let mut facings = [0.0f32; TRAJECTORY_SAMPLES];

        for (i, &t) in [0.2, 0.5, 1.0].iter().enumerate() {
            let point = self.sample_at(t);
            positions[i] = point.position;
            facings[i] = point.facing_angle();
        }

        TrajectoryFeature::from_predictions(positions, facings)
    }

    /// Convert to TrajectoryFeatureCompat (for motion_search).
    pub fn to_search_trajectory(&self) -> TrajectoryFeatureCompat {
        let mut positions = [Vec3::ZERO; 3];
        let mut facings = [0.0f32; 3];

        for (i, &t) in [0.2, 0.5, 1.0].iter().enumerate() {
            let point = self.sample_at(t);
            positions[i] = point.position;
            facings[i] = point.facing_angle();
        }

        TrajectoryFeatureCompat {
            future_positions: positions,
            future_facings: facings,
        }
    }

    /// Compute distance to another trajectory request.
    pub fn distance(&self, other: &TrajectoryRequest) -> f32 {
        let mut total = 0.0;
        let mut count = 0;

        for (a, b) in self.points.iter().zip(other.points.iter()) {
            total += a.distance(b);
            count += 1;
        }

        if count > 0 {
            total / count as f32
        } else {
            0.0
        }
    }

    /// Interpolate between two trajectory requests.
    pub fn lerp(&self, other: &TrajectoryRequest, t: f32) -> Self {
        let mut result = Self::new();
        result.speed_profile = self.speed_profile + (other.speed_profile - self.speed_profile) * t;
        result.curvature = self.curvature + (other.curvature - self.curvature) * t;
        result.stopping = if t < 0.5 { self.stopping } else { other.stopping };
        result.starting = if t < 0.5 { self.starting } else { other.starting };
        result.urgency = self.urgency + (other.urgency - self.urgency) * t;

        let max_points = self.points.len().max(other.points.len());
        for i in 0..max_points {
            let a = self.points.get(i).copied().unwrap_or_default();
            let b = other.points.get(i).copied().unwrap_or_default();
            result.points.push(a.lerp(&b, t));
        }

        result
    }
}

// ---------------------------------------------------------------------------
// MotionState
// ---------------------------------------------------------------------------

/// Current motion playback state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MotionState {
    /// Current clip index in the database.
    pub clip_index: u32,

    /// Current frame index within the clip.
    pub frame_index: u32,

    /// Current time within the clip (seconds).
    pub time: f32,

    /// Playback speed multiplier.
    pub playback_speed: f32,

    /// Whether currently in a transition.
    pub in_transition: bool,

    /// Time remaining in transition.
    pub transition_remaining: f32,

    /// Root velocity (world space).
    pub root_velocity: Vec3,

    /// Root angular velocity (radians/s).
    pub root_angular_velocity: f32,
}

impl MotionState {
    /// Create a new motion state.
    pub fn new() -> Self {
        Self {
            playback_speed: 1.0,
            ..Default::default()
        }
    }

    /// Create from clip and time.
    pub fn from_clip(clip_index: u32, time: f32) -> Self {
        Self {
            clip_index,
            time,
            playback_speed: 1.0,
            ..Default::default()
        }
    }

    /// Update the state with elapsed time.
    pub fn advance(&mut self, dt: f32) {
        self.time += dt * self.playback_speed;

        if self.in_transition {
            self.transition_remaining = (self.transition_remaining - dt).max(0.0);
            if self.transition_remaining <= 0.0 {
                self.in_transition = false;
            }
        }
    }

    /// Start a transition.
    pub fn start_transition(&mut self, new_clip: u32, new_time: f32, duration: f32) {
        self.clip_index = new_clip;
        self.time = new_time;
        self.in_transition = true;
        self.transition_remaining = duration;
    }

    /// Get the speed from root velocity.
    #[inline]
    pub fn speed(&self) -> f32 {
        self.root_velocity.length()
    }
}

// ---------------------------------------------------------------------------
// TagSet
// ---------------------------------------------------------------------------

/// Set of active tags for motion matching.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TagSet {
    /// Locomotion tags (bitflags).
    pub locomotion: LocomotionTags,

    /// Motion tags (high-level).
    pub motion: MotionTags,

    /// Custom string tags.
    pub custom: HashSet<String>,
}

impl TagSet {
    /// Create a new empty tag set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from locomotion tags.
    pub fn from_locomotion(tags: LocomotionTags) -> Self {
        Self {
            locomotion: tags,
            motion: MotionTags::default(),
            custom: HashSet::new(),
        }
    }

    /// Create from motion tags.
    pub fn from_motion(tags: MotionTags) -> Self {
        Self {
            locomotion: LocomotionTags::empty(),
            motion: tags,
            custom: HashSet::new(),
        }
    }

    /// Add locomotion tags.
    pub fn with_locomotion(mut self, tags: LocomotionTags) -> Self {
        self.locomotion |= tags;
        self
    }

    /// Add a custom tag.
    pub fn with_custom(mut self, tag: impl Into<String>) -> Self {
        self.custom.insert(tag.into());
        self
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.locomotion.is_empty() && self.custom.is_empty()
    }

    /// Check if contains a locomotion tag.
    #[inline]
    pub fn has_locomotion(&self, tag: LocomotionTags) -> bool {
        self.locomotion.contains(tag)
    }

    /// Check if contains a custom tag.
    #[inline]
    pub fn has_custom(&self, tag: &str) -> bool {
        self.custom.contains(tag)
    }

    /// Merge with another tag set.
    pub fn merge(&mut self, other: &TagSet) {
        self.locomotion |= other.locomotion;
        self.motion = other.motion;
        for tag in &other.custom {
            self.custom.insert(tag.clone());
        }
    }

    /// Clear all tags.
    pub fn clear(&mut self) {
        self.locomotion = LocomotionTags::empty();
        self.motion = MotionTags::default();
        self.custom.clear();
    }
}

// ---------------------------------------------------------------------------
// TagPriority
// ---------------------------------------------------------------------------

/// Priority rule for tag selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TagPriority {
    /// Tag that takes priority.
    pub primary: LocomotionTags,

    /// Tags that are overridden.
    pub overrides: LocomotionTags,

    /// Priority level (higher = more important).
    pub level: i32,
}

impl TagPriority {
    /// Create a new priority rule.
    pub fn new(primary: LocomotionTags, overrides: LocomotionTags, level: i32) -> Self {
        Self {
            primary,
            overrides,
            level,
        }
    }
}

// ---------------------------------------------------------------------------
// TagExclusion
// ---------------------------------------------------------------------------

/// Exclusion rule preventing tag combinations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TagExclusion {
    /// Tags that are mutually exclusive.
    pub tags: LocomotionTags,

    /// Which tag to keep when conflict occurs.
    pub keep: LocomotionTags,
}

impl TagExclusion {
    /// Create a new exclusion rule.
    pub fn new(tags: LocomotionTags, keep: LocomotionTags) -> Self {
        Self { tags, keep }
    }
}

// ---------------------------------------------------------------------------
// TagManager
// ---------------------------------------------------------------------------

/// Manages active tags with priority and exclusion rules.
#[derive(Clone, Debug, Default)]
pub struct TagManager {
    /// Currently active tags.
    pub active: TagSet,

    /// Priority rules.
    pub priorities: Vec<TagPriority>,

    /// Exclusion rules.
    pub exclusions: Vec<TagExclusion>,

    /// Terrain overrides by region/zone.
    pub terrain_overrides: HashMap<u32, TerrainType>,

    /// Default terrain type.
    pub default_terrain: TerrainType,
}

impl TagManager {
    /// Create a new tag manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with standard locomotion rules.
    pub fn with_standard_rules() -> Self {
        let mut manager = Self::new();

        // Priority: COMBAT > SPRINT > RUN > WALK > IDLE
        manager.priorities.push(TagPriority::new(
            LocomotionTags::COMBAT,
            LocomotionTags::SPRINT | LocomotionTags::RUN | LocomotionTags::WALK,
            100,
        ));
        manager.priorities.push(TagPriority::new(
            LocomotionTags::SPRINT,
            LocomotionTags::RUN | LocomotionTags::WALK,
            80,
        ));
        manager.priorities.push(TagPriority::new(
            LocomotionTags::RUN,
            LocomotionTags::WALK,
            60,
        ));

        // Exclusions: can't be walking and running simultaneously
        manager.exclusions.push(TagExclusion::new(
            LocomotionTags::WALK | LocomotionTags::RUN,
            LocomotionTags::RUN,
        ));
        manager.exclusions.push(TagExclusion::new(
            LocomotionTags::RUN | LocomotionTags::SPRINT,
            LocomotionTags::SPRINT,
        ));

        manager
    }

    /// Add a priority rule.
    pub fn add_priority(&mut self, rule: TagPriority) {
        self.priorities.push(rule);
        self.priorities.sort_by(|a, b| b.level.cmp(&a.level));
    }

    /// Add an exclusion rule.
    pub fn add_exclusion(&mut self, rule: TagExclusion) {
        self.exclusions.push(rule);
    }

    /// Set terrain override for a zone.
    pub fn set_terrain_override(&mut self, zone: u32, terrain: TerrainType) {
        self.terrain_overrides.insert(zone, terrain);
    }

    /// Get terrain for a zone.
    pub fn get_terrain(&self, zone: Option<u32>) -> TerrainType {
        zone.and_then(|z| self.terrain_overrides.get(&z).copied())
            .unwrap_or(self.default_terrain)
    }

    /// Set active tags.
    pub fn set_tags(&mut self, tags: TagSet) {
        self.active = tags;
        self.apply_rules();
    }

    /// Add tags.
    pub fn add_tags(&mut self, tags: LocomotionTags) {
        self.active.locomotion |= tags;
        self.apply_rules();
    }

    /// Remove tags.
    pub fn remove_tags(&mut self, tags: LocomotionTags) {
        self.active.locomotion.remove(tags);
    }

    /// Apply priority and exclusion rules.
    pub fn apply_rules(&mut self) {
        // Apply priorities
        for rule in &self.priorities {
            if self.active.locomotion.contains(rule.primary) {
                self.active.locomotion.remove(rule.overrides);
            }
        }

        // Apply exclusions
        for rule in &self.exclusions {
            let conflict = self.active.locomotion & rule.tags;
            if conflict.bits().count_ones() > 1 {
                // Multiple conflicting tags, keep only the specified one
                self.active.locomotion.remove(rule.tags);
                self.active.locomotion.insert(rule.keep);
            }
        }
    }

    /// Get filtered tags for search.
    pub fn get_search_tags(&self) -> (LocomotionTags, LocomotionTags) {
        let required = self.active.locomotion;
        let excluded = LocomotionTags::empty();
        (required, excluded)
    }

    /// Update tags from locomotion style.
    pub fn update_from_style(&mut self, style: LocomotionStyle) {
        // Remove all locomotion state tags
        self.active.locomotion.remove(
            LocomotionTags::IDLE
                | LocomotionTags::WALK
                | LocomotionTags::RUN
                | LocomotionTags::SPRINT
                | LocomotionTags::CROUCH,
        );

        // Add appropriate tag
        let tag = match style {
            LocomotionStyle::Idle => LocomotionTags::IDLE,
            LocomotionStyle::Walk => LocomotionTags::WALK,
            LocomotionStyle::Run => LocomotionTags::RUN,
            LocomotionStyle::Sprint => LocomotionTags::SPRINT,
            LocomotionStyle::Crouch => LocomotionTags::CROUCH,
        };
        self.active.locomotion.insert(tag);
        self.active.motion.locomotion = style;
    }

    /// Update tags from terrain type.
    pub fn update_from_terrain(&mut self, terrain: TerrainType) {
        // Remove all terrain tags
        self.active.locomotion.remove(
            LocomotionTags::TERRAIN_FLAT
                | LocomotionTags::TERRAIN_UPHILL
                | LocomotionTags::TERRAIN_DOWNHILL
                | LocomotionTags::TERRAIN_STAIRS
                | LocomotionTags::TERRAIN_ROUGH,
        );

        // Add appropriate tag
        let tag = match terrain {
            TerrainType::Flat => LocomotionTags::TERRAIN_FLAT,
            TerrainType::SlopeUp => LocomotionTags::TERRAIN_UPHILL,
            TerrainType::SlopeDown => LocomotionTags::TERRAIN_DOWNHILL,
            TerrainType::StairsUp | TerrainType::StairsDown => LocomotionTags::TERRAIN_STAIRS,
        };
        self.active.locomotion.insert(tag);
        self.active.motion.terrain = terrain;
    }
}

// ---------------------------------------------------------------------------
// FootContactState
// ---------------------------------------------------------------------------

/// State of a single foot contact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FootContactState {
    /// Whether the foot is currently planted.
    pub planted: bool,

    /// Position relative to root.
    pub position: Vec3,

    /// Velocity relative to root.
    pub velocity: Vec3,

    /// Height above ground.
    pub height: f32,

    /// Phase in gait cycle (0-1).
    pub phase: f32,

    /// Time since last contact change.
    pub time_since_change: f32,
}

impl FootContactState {
    /// Create a new foot contact state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a planted foot.
    pub fn planted(position: Vec3) -> Self {
        Self {
            planted: true,
            position,
            velocity: Vec3::ZERO,
            height: 0.0,
            phase: 0.0,
            time_since_change: 0.0,
        }
    }

    /// Create a moving foot.
    pub fn moving(position: Vec3, velocity: Vec3) -> Self {
        Self {
            planted: false,
            position,
            velocity,
            height: position.y,
            phase: 0.5,
            time_since_change: 0.0,
        }
    }

    /// Update the contact state.
    pub fn update(&mut self, position: Vec3, velocity: Vec3, dt: f32) {
        let was_planted = self.planted;
        self.position = position;
        self.velocity = velocity;
        self.height = position.y;

        // Detect contact based on height and velocity
        let is_low = self.height < CONTACT_HEIGHT_THRESHOLD;
        let is_slow = velocity.length() < CONTACT_VELOCITY_THRESHOLD;
        let now_planted = is_low && is_slow;

        if now_planted != was_planted {
            self.time_since_change = 0.0;
        } else {
            self.time_since_change += dt;
        }

        self.planted = now_planted;
    }
}

// ---------------------------------------------------------------------------
// FootContactTracker
// ---------------------------------------------------------------------------

/// Tracks foot contact states for motion matching.
#[derive(Clone, Debug, Default)]
pub struct FootContactTracker {
    /// Contact states per foot.
    pub feet: Vec<FootContactState>,

    /// Bone indices for feet.
    pub foot_bones: Vec<usize>,

    /// Predicted upcoming contacts.
    pub predicted_contacts: Vec<f32>,

    /// Gait cycle phase (0-1).
    pub gait_phase: f32,

    /// Gait cycle frequency (Hz).
    pub gait_frequency: f32,

    /// Contact history for prediction.
    contact_history: Vec<(f32, Vec<bool>)>,

    /// Maximum history length.
    max_history: usize,
}

impl FootContactTracker {
    /// Create a new foot contact tracker.
    pub fn new(foot_count: usize) -> Self {
        Self {
            feet: vec![FootContactState::new(); foot_count],
            foot_bones: (0..foot_count).collect(),
            predicted_contacts: vec![0.0; foot_count],
            gait_phase: 0.0,
            gait_frequency: 2.0, // Default walking frequency
            contact_history: Vec::new(),
            max_history: 60,
        }
    }

    /// Create for bipedal character.
    pub fn bipedal(left_foot_bone: usize, right_foot_bone: usize) -> Self {
        let mut tracker = Self::new(2);
        tracker.foot_bones = vec![left_foot_bone, right_foot_bone];
        tracker
    }

    /// Create for quadruped character.
    pub fn quadruped(bones: [usize; 4]) -> Self {
        let mut tracker = Self::new(4);
        tracker.foot_bones = bones.to_vec();
        tracker
    }

    /// Get foot count.
    #[inline]
    pub fn foot_count(&self) -> usize {
        self.feet.len()
    }

    /// Update foot contacts from positions and velocities.
    pub fn update(&mut self, positions: &[Vec3], velocities: &[Vec3], dt: f32, current_time: f32) {
        let foot_count = self.feet.len().min(positions.len()).min(velocities.len());

        for i in 0..foot_count {
            self.feet[i].update(positions[i], velocities[i], dt);
        }

        // Update gait phase
        self.gait_phase += dt * self.gait_frequency;
        if self.gait_phase >= 1.0 {
            self.gait_phase -= 1.0;
        }

        // Update phase for each foot (offset by 180 degrees for alternating gait)
        for (i, foot) in self.feet.iter_mut().enumerate() {
            let offset = (i as f32 / foot_count as f32) * 0.5; // Offset each foot
            foot.phase = (self.gait_phase + offset) % 1.0;
        }

        // Record history
        let contacts: Vec<bool> = self.feet.iter().map(|f| f.planted).collect();
        self.contact_history.push((current_time, contacts));
        if self.contact_history.len() > self.max_history {
            self.contact_history.remove(0);
        }

        // Predict upcoming contacts
        self.predict_contacts();
    }

    /// Predict upcoming contact times for each foot.
    fn predict_contacts(&mut self) {
        // Simple prediction based on gait phase and frequency
        let period = 1.0 / self.gait_frequency;

        for (i, foot) in self.feet.iter().enumerate() {
            if foot.planted {
                // Predict time to lift-off
                let phase_to_liftoff = 0.5 - foot.phase.min(0.5);
                self.predicted_contacts[i] = phase_to_liftoff * period;
            } else {
                // Predict time to contact
                let phase_to_contact = if foot.phase < 0.5 {
                    0.5 - foot.phase
                } else {
                    1.0 - foot.phase
                };
                self.predicted_contacts[i] = phase_to_contact * period;
            }
        }
    }

    /// Get contact states for search query.
    pub fn to_foot_features(&self) -> FootFeatures {
        let mut features = FootFeatures::with_count(self.feet.len());

        for (i, foot) in self.feet.iter().enumerate() {
            features.contact_states[i] = foot.planted;
            features.positions[i] = foot.position;
            features.velocities[i] = foot.velocity;
            features.phases[i] = foot.phase;
        }

        features
    }

    /// Check if any foot is planted.
    #[inline]
    pub fn any_planted(&self) -> bool {
        self.feet.iter().any(|f| f.planted)
    }

    /// Check if all feet are planted.
    #[inline]
    pub fn all_planted(&self) -> bool {
        self.feet.iter().all(|f| f.planted)
    }

    /// Get planted foot count.
    #[inline]
    pub fn planted_count(&self) -> usize {
        self.feet.iter().filter(|f| f.planted).count()
    }

    /// Synchronize with motion from database.
    pub fn sync_with_motion(&mut self, foot_contacts: &[(bool, Vec3, Vec3)]) {
        for (i, (planted, pos, vel)) in foot_contacts.iter().enumerate() {
            if i < self.feet.len() {
                self.feet[i].planted = *planted;
                self.feet[i].position = *pos;
                self.feet[i].velocity = *vel;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ContextUpdatePolicy
// ---------------------------------------------------------------------------

/// Policy for when to update context and trigger searches.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextUpdatePolicy {
    /// Update frequency (every N frames).
    pub update_interval: u32,

    /// Cost threshold for triggering new search.
    pub search_threshold: f32,

    /// Hysteresis factor to prevent oscillation.
    pub hysteresis: f32,

    /// Cooldown after transition (seconds).
    pub transition_cooldown: f32,

    /// Whether to allow search during transitions.
    pub allow_search_in_transition: bool,

    /// Minimum time in current motion before switching.
    pub min_motion_time: f32,

    /// Maximum time without search.
    pub max_time_between_searches: f32,

    /// Current frame counter.
    frame_counter: u32,

    /// Time since last search.
    time_since_search: f32,

    /// Time since last transition.
    time_since_transition: f32,

    /// Last search cost.
    last_search_cost: f32,
}

impl Default for ContextUpdatePolicy {
    fn default() -> Self {
        Self {
            update_interval: DEFAULT_UPDATE_INTERVAL,
            search_threshold: DEFAULT_SEARCH_THRESHOLD,
            hysteresis: DEFAULT_HYSTERESIS,
            transition_cooldown: DEFAULT_TRANSITION_COOLDOWN,
            allow_search_in_transition: false,
            min_motion_time: 0.1,
            max_time_between_searches: 0.5,
            frame_counter: 0,
            time_since_search: 0.0,
            time_since_transition: 1.0,
            last_search_cost: 0.0,
        }
    }
}

impl ContextUpdatePolicy {
    /// Create a new update policy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a responsive policy (frequent updates).
    pub fn responsive() -> Self {
        Self {
            update_interval: 2,
            search_threshold: 0.2,
            hysteresis: 0.05,
            transition_cooldown: 0.1,
            max_time_between_searches: 0.25,
            ..Default::default()
        }
    }

    /// Create a conservative policy (fewer updates).
    pub fn conservative() -> Self {
        Self {
            update_interval: 8,
            search_threshold: 0.5,
            hysteresis: 0.15,
            transition_cooldown: 0.25,
            max_time_between_searches: 1.0,
            ..Default::default()
        }
    }

    /// Update timing counters.
    pub fn update(&mut self, dt: f32) {
        self.frame_counter += 1;
        self.time_since_search += dt;
        self.time_since_transition += dt;
    }

    /// Notify that a search was performed.
    pub fn on_search(&mut self, cost: f32) {
        self.time_since_search = 0.0;
        self.last_search_cost = cost;
    }

    /// Notify that a transition occurred.
    pub fn on_transition(&mut self) {
        self.time_since_transition = 0.0;
    }

    /// Check if should perform a search.
    pub fn should_search(&self, current_cost: f32, in_transition: bool) -> bool {
        // Respect transition cooldown
        if in_transition && !self.allow_search_in_transition {
            return false;
        }

        if self.time_since_transition < self.transition_cooldown {
            return false;
        }

        // Check frame interval
        if self.frame_counter % self.update_interval != 0 {
            return false;
        }

        // Force search if too long since last
        if self.time_since_search >= self.max_time_between_searches {
            return true;
        }

        // Check cost threshold with hysteresis
        let threshold = if current_cost > self.last_search_cost {
            self.search_threshold + self.hysteresis
        } else {
            self.search_threshold - self.hysteresis
        };

        current_cost >= threshold
    }

    /// Reset the policy.
    pub fn reset(&mut self) {
        self.frame_counter = 0;
        self.time_since_search = 0.0;
        self.time_since_transition = 1.0;
        self.last_search_cost = 0.0;
    }
}

// ---------------------------------------------------------------------------
// ContextInterpolator
// ---------------------------------------------------------------------------

/// Interpolates context changes over time for smooth trajectories.
#[derive(Clone, Debug)]
pub struct ContextInterpolator {
    /// Previous trajectory request.
    prev_trajectory: TrajectoryRequest,

    /// Current trajectory request.
    curr_trajectory: TrajectoryRequest,

    /// Target trajectory request.
    target_trajectory: TrajectoryRequest,

    /// Interpolation progress (0-1).
    progress: f32,

    /// Interpolation duration.
    duration: f32,

    /// Smoothness factor (0 = instant, 1 = very smooth).
    pub smoothness: f32,

    /// Whether interpolation is active.
    active: bool,
}

impl Default for ContextInterpolator {
    fn default() -> Self {
        Self {
            prev_trajectory: TrajectoryRequest::new(),
            curr_trajectory: TrajectoryRequest::new(),
            target_trajectory: TrajectoryRequest::new(),
            progress: 1.0,
            duration: 0.2,
            smoothness: DEFAULT_INTERPOLATION_SMOOTHNESS,
            active: false,
        }
    }
}

impl ContextInterpolator {
    /// Create a new context interpolator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with specific smoothness.
    pub fn with_smoothness(smoothness: f32) -> Self {
        Self {
            smoothness: smoothness.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Set a new target trajectory.
    pub fn set_target(&mut self, target: TrajectoryRequest) {
        self.prev_trajectory = self.curr_trajectory.clone();
        self.target_trajectory = target;
        self.progress = 0.0;
        self.duration = self.compute_duration();
        self.active = true;
    }

    /// Compute interpolation duration based on trajectory difference.
    fn compute_duration(&self) -> f32 {
        let distance = self.prev_trajectory.distance(&self.target_trajectory);
        let base_duration = 0.1 + distance * 0.1;
        base_duration * (1.0 + self.smoothness)
    }

    /// Update the interpolation.
    pub fn update(&mut self, dt: f32) {
        if !self.active {
            return;
        }

        self.progress += dt / self.duration;

        if self.progress >= 1.0 {
            self.progress = 1.0;
            self.curr_trajectory = self.target_trajectory.clone();
            self.active = false;
        } else {
            // Use smooth-step for more natural interpolation
            let t = self.smooth_step(self.progress);
            self.curr_trajectory = self.prev_trajectory.lerp(&self.target_trajectory, t);
        }
    }

    /// Smooth-step function for natural easing.
    #[inline]
    fn smooth_step(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// Get the current interpolated trajectory.
    #[inline]
    pub fn current(&self) -> &TrajectoryRequest {
        &self.curr_trajectory
    }

    /// Get the target trajectory.
    #[inline]
    pub fn target(&self) -> &TrajectoryRequest {
        &self.target_trajectory
    }

    /// Check if interpolation is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Get interpolation progress.
    #[inline]
    pub fn progress(&self) -> f32 {
        self.progress
    }

    /// Handle sudden direction changes.
    pub fn handle_direction_change(&mut self, new_trajectory: TrajectoryRequest, urgency: f32) {
        // For urgent changes, reduce duration
        let urgency_factor = 1.0 - urgency * 0.7;
        self.set_target(new_trajectory);
        self.duration *= urgency_factor;
    }

    /// Force immediate update (no interpolation).
    pub fn force_immediate(&mut self, trajectory: TrajectoryRequest) {
        self.prev_trajectory = trajectory.clone();
        self.curr_trajectory = trajectory.clone();
        self.target_trajectory = trajectory;
        self.progress = 1.0;
        self.active = false;
    }

    /// Reset the interpolator.
    pub fn reset(&mut self) {
        self.prev_trajectory = TrajectoryRequest::new();
        self.curr_trajectory = TrajectoryRequest::new();
        self.target_trajectory = TrajectoryRequest::new();
        self.progress = 1.0;
        self.active = false;
    }
}

// ---------------------------------------------------------------------------
// MotionContext
// ---------------------------------------------------------------------------

/// Complete context for motion matching queries.
///
/// Contains all information needed to find the best matching animation frame:
/// pose features, desired trajectory, motion state, tags, and foot contacts.
#[derive(Clone, Debug, Default)]
pub struct MotionContext {
    /// Current pose features.
    pub pose: MotionFeatures,

    /// Desired trajectory.
    pub trajectory: TrajectoryRequest,

    /// Current motion playback state.
    pub motion_state: MotionState,

    /// Active tags.
    pub tags: TagSet,

    /// Foot contact tracker.
    pub foot_tracker: FootContactTracker,

    /// Update policy.
    pub update_policy: ContextUpdatePolicy,

    /// Trajectory interpolator.
    pub interpolator: ContextInterpolator,

    /// Current time.
    pub time: f32,

    /// Root position (world space).
    pub root_position: Vec3,

    /// Root rotation.
    pub root_rotation: Quat,
}

impl MotionContext {
    /// Create a new empty motion context.
    pub fn new() -> Self {
        Self {
            foot_tracker: FootContactTracker::new(2),
            ..Default::default()
        }
    }

    /// Create with specified foot count.
    pub fn with_foot_count(foot_count: usize) -> Self {
        Self {
            foot_tracker: FootContactTracker::new(foot_count),
            ..Default::default()
        }
    }

    /// Update the context.
    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        self.motion_state.advance(dt);
        self.update_policy.update(dt);
        self.interpolator.update(dt);
    }

    /// Set the desired trajectory.
    pub fn set_trajectory(&mut self, trajectory: TrajectoryRequest) {
        self.interpolator.set_target(trajectory);
    }

    /// Set trajectory immediately (no interpolation).
    pub fn set_trajectory_immediate(&mut self, trajectory: TrajectoryRequest) {
        self.interpolator.force_immediate(trajectory);
    }

    /// Get the current (possibly interpolated) trajectory.
    pub fn current_trajectory(&self) -> &TrajectoryRequest {
        self.interpolator.current()
    }

    /// Check if should perform a motion search.
    pub fn should_search(&self, current_cost: f32) -> bool {
        self.update_policy.should_search(current_cost, self.motion_state.in_transition)
    }

    /// Notify that a search was performed.
    pub fn on_search(&mut self, cost: f32) {
        self.update_policy.on_search(cost);
    }

    /// Start a motion transition.
    pub fn start_transition(&mut self, new_clip: u32, new_time: f32, duration: f32) {
        self.motion_state.start_transition(new_clip, new_time, duration);
        self.update_policy.on_transition();
    }

    /// Convert to a search query.
    pub fn to_search_query(&self) -> SearchQuery {
        let trajectory = self.interpolator.current();

        SearchQuery {
            current_pose: self.pose.clone(),
            desired_trajectory: trajectory.to_search_trajectory(),
            current_clip: Some((self.motion_state.clip_index as usize, self.motion_state.time)),
            required_tags: self.tags.motion,
            root_velocity: self.motion_state.root_velocity,
            root_angular_velocity: self.motion_state.root_angular_velocity,
        }
    }

    /// Get motion tags for search.
    pub fn get_search_tags(&self) -> MotionTags {
        self.tags.motion
    }

    /// Check if in transition.
    #[inline]
    pub fn in_transition(&self) -> bool {
        self.motion_state.in_transition
    }

    /// Get current speed.
    #[inline]
    pub fn speed(&self) -> f32 {
        self.motion_state.speed()
    }
}

// ---------------------------------------------------------------------------
// ContextBuilder
// ---------------------------------------------------------------------------

/// Builder for creating motion contexts from various sources.
#[derive(Clone, Debug, Default)]
pub struct ContextBuilder {
    /// Pose features.
    pose: Option<MotionFeatures>,

    /// Trajectory request.
    trajectory: Option<TrajectoryRequest>,

    /// Motion state.
    motion_state: Option<MotionState>,

    /// Tags.
    tags: Option<TagSet>,

    /// Foot tracker.
    foot_tracker: Option<FootContactTracker>,

    /// Update policy.
    update_policy: Option<ContextUpdatePolicy>,

    /// Root position.
    root_position: Vec3,

    /// Root rotation.
    root_rotation: Quat,

    /// Current time.
    time: f32,
}

impl ContextBuilder {
    /// Create a new context builder.
    pub fn new() -> Self {
        Self {
            root_rotation: Quat::IDENTITY,
            ..Default::default()
        }
    }

    /// Build context from controller input.
    pub fn from_controller_input(
        mut self,
        move_input: Vec3,
        _look_direction: Vec3,
        speed: f32,
        _dt: f32,
    ) -> Self {
        // Determine locomotion style
        let style = LocomotionStyle::from_speed(speed);

        // Create trajectory from input
        let trajectory = if move_input.length_squared() > 0.01 {
            let direction = move_input.normalize_or_zero();
            let mut traj = TrajectoryRequest::straight(speed, 1.0);

            // Adjust for direction
            for point in &mut traj.points {
                point.facing = direction;
                point.position = direction * point.position.length();
            }

            traj
        } else {
            TrajectoryRequest::straight(0.0, 1.0)
        };

        self.trajectory = Some(trajectory);

        // Set tags based on style
        let mut tags = TagSet::new();
        tags.motion.locomotion = style;
        tags.locomotion = match style {
            LocomotionStyle::Idle => LocomotionTags::IDLE,
            LocomotionStyle::Walk => LocomotionTags::WALK,
            LocomotionStyle::Run => LocomotionTags::RUN,
            LocomotionStyle::Sprint => LocomotionTags::SPRINT,
            LocomotionStyle::Crouch => LocomotionTags::CROUCH,
        };
        self.tags = Some(tags);

        self
    }

    /// Build context from AI navigation path.
    pub fn from_navigation_path(
        mut self,
        path_points: &[Vec3],
        current_position: Vec3,
        desired_speed: f32,
    ) -> Self {
        if path_points.is_empty() {
            self.trajectory = Some(TrajectoryRequest::straight(0.0, 1.0));
            return self;
        }

        let mut trajectory = TrajectoryRequest::new();
        trajectory.speed_profile = desired_speed;

        // Sample path at trajectory times
        for &t in &TRAJECTORY_TIMES {
            let distance_along = desired_speed * t;
            let point = self.sample_path_at_distance(path_points, current_position, distance_along);
            trajectory.points.push(point);
        }

        // Compute curvature from path
        if trajectory.points.len() >= 2 {
            let dir1 = trajectory.points[0].facing;
            let dir2 = trajectory.points.last().unwrap().facing;
            let angle = dir1.x.atan2(dir1.z) - dir2.x.atan2(dir2.z);
            trajectory.curvature = (angle / std::f32::consts::PI).clamp(-1.0, 1.0);
        }

        self.trajectory = Some(trajectory);
        self.root_position = current_position;
        self
    }

    /// Sample path at a given distance.
    fn sample_path_at_distance(
        &self,
        path: &[Vec3],
        start: Vec3,
        distance: f32,
    ) -> TrajectoryPoint {
        if path.is_empty() {
            return TrajectoryPoint::default();
        }

        let mut remaining = distance;
        let mut current = start;

        for &next in path {
            let segment = next - current;
            let segment_len = segment.length();

            if remaining <= segment_len {
                let t = remaining / segment_len.max(0.001);
                let position = current + segment * t - start;
                let facing = segment.normalize_or_zero();
                return TrajectoryPoint::new(
                    position,
                    Vec3::new(facing.x, 0.0, facing.z),
                    distance / TRAJECTORY_TIMES[0],
                    distance / distance.max(0.001) * TRAJECTORY_TIMES[0],
                );
            }

            remaining -= segment_len;
            current = next;
        }

        // Beyond path end
        let last = path.last().copied().unwrap_or(start);
        TrajectoryPoint::new(
            last - start,
            Vec3::Z,
            0.0,
            TRAJECTORY_TIMES.last().copied().unwrap_or(1.0),
        )
    }

    /// Build context from animation events.
    pub fn from_animation_events(
        mut self,
        event_tags: &[String],
        _transition_targets: Option<&str>,
    ) -> Self {
        let mut tags = self.tags.unwrap_or_default();

        for tag in event_tags {
            tags.custom.insert(tag.clone());

            // Parse common event tags
            match tag.as_str() {
                "jump_start" => tags.locomotion.insert(LocomotionTags::JUMP),
                "land" => tags.locomotion.insert(LocomotionTags::LAND),
                "fall" => tags.locomotion.insert(LocomotionTags::FALL),
                "combat_enter" => tags.locomotion.insert(LocomotionTags::COMBAT),
                "combat_exit" => tags.locomotion.remove(LocomotionTags::COMBAT),
                _ => {}
            }
        }

        self.tags = Some(tags);
        self
    }

    /// Set pose features.
    pub fn with_pose(mut self, pose: MotionFeatures) -> Self {
        self.pose = Some(pose);
        self
    }

    /// Set trajectory.
    pub fn with_trajectory(mut self, trajectory: TrajectoryRequest) -> Self {
        self.trajectory = Some(trajectory);
        self
    }

    /// Set motion state.
    pub fn with_motion_state(mut self, state: MotionState) -> Self {
        self.motion_state = Some(state);
        self
    }

    /// Set tags.
    pub fn with_tags(mut self, tags: TagSet) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Set foot tracker.
    pub fn with_foot_tracker(mut self, tracker: FootContactTracker) -> Self {
        self.foot_tracker = Some(tracker);
        self
    }

    /// Set update policy.
    pub fn with_update_policy(mut self, policy: ContextUpdatePolicy) -> Self {
        self.update_policy = Some(policy);
        self
    }

    /// Set root transform.
    pub fn with_root_transform(mut self, position: Vec3, rotation: Quat) -> Self {
        self.root_position = position;
        self.root_rotation = rotation;
        self
    }

    /// Set time.
    pub fn with_time(mut self, time: f32) -> Self {
        self.time = time;
        self
    }

    /// Combine with another builder (merges settings).
    pub fn combine(mut self, other: ContextBuilder) -> Self {
        if other.pose.is_some() {
            self.pose = other.pose;
        }
        if other.trajectory.is_some() {
            self.trajectory = other.trajectory;
        }
        if other.motion_state.is_some() {
            self.motion_state = other.motion_state;
        }
        if let Some(other_tags) = other.tags {
            if let Some(ref mut self_tags) = self.tags {
                self_tags.merge(&other_tags);
            } else {
                self.tags = Some(other_tags);
            }
        }
        if other.foot_tracker.is_some() {
            self.foot_tracker = other.foot_tracker;
        }
        if other.update_policy.is_some() {
            self.update_policy = other.update_policy;
        }
        if other.root_position != Vec3::ZERO {
            self.root_position = other.root_position;
        }
        if other.root_rotation != Quat::IDENTITY {
            self.root_rotation = other.root_rotation;
        }
        if other.time > 0.0 {
            self.time = other.time;
        }
        self
    }

    /// Build the motion context.
    pub fn build(self) -> MotionContext {
        let mut context = MotionContext::new();

        if let Some(pose) = self.pose {
            context.pose = pose;
        }
        if let Some(trajectory) = self.trajectory {
            context.interpolator.force_immediate(trajectory);
        }
        if let Some(state) = self.motion_state {
            context.motion_state = state;
        }
        if let Some(tags) = self.tags {
            context.tags = tags;
        }
        if let Some(tracker) = self.foot_tracker {
            context.foot_tracker = tracker;
        }
        if let Some(policy) = self.update_policy {
            context.update_policy = policy;
        }

        context.root_position = self.root_position;
        context.root_rotation = self.root_rotation;
        context.time = self.time;

        context
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // =====================================================================
    // TrajectoryPoint tests
    // =====================================================================

    #[test]
    fn test_trajectory_point_new() {
        let point = TrajectoryPoint::new(
            Vec3::new(1.0, 0.0, 2.0),
            Vec3::Z,
            3.0,
            0.5,
        );
        assert_eq!(point.position, Vec3::new(1.0, 0.0, 2.0));
        assert_eq!(point.speed, 3.0);
        assert_eq!(point.time_offset, 0.5);
    }

    #[test]
    fn test_trajectory_point_from_position() {
        let point = TrajectoryPoint::from_position(Vec3::new(0.0, 0.0, 2.0), 1.0);
        assert_eq!(point.position, Vec3::new(0.0, 0.0, 2.0));
        assert!((point.speed - 2.0).abs() < 0.01);
        assert!((point.facing.z - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_trajectory_point_facing_angle() {
        let point = TrajectoryPoint::new(Vec3::ZERO, Vec3::X, 0.0, 0.0);
        assert!((point.facing_angle() - PI / 2.0).abs() < 0.01);
    }

    #[test]
    fn test_trajectory_point_lerp() {
        let a = TrajectoryPoint::new(Vec3::ZERO, Vec3::Z, 0.0, 0.0);
        let b = TrajectoryPoint::new(Vec3::new(10.0, 0.0, 0.0), Vec3::Z, 10.0, 1.0);
        let mid = a.lerp(&b, 0.5);

        assert!((mid.position.x - 5.0).abs() < 0.01);
        assert!((mid.speed - 5.0).abs() < 0.01);
        assert!((mid.time_offset - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_trajectory_point_distance() {
        let a = TrajectoryPoint::new(Vec3::ZERO, Vec3::Z, 0.0, 0.0);
        let b = TrajectoryPoint::new(Vec3::new(3.0, 0.0, 4.0), Vec3::Z, 0.0, 0.0);
        let dist = a.distance(&b);
        assert!((dist - 5.0).abs() < 0.01);
    }

    // =====================================================================
    // TrajectoryRequest tests
    // =====================================================================

    #[test]
    fn test_trajectory_request_new() {
        let req = TrajectoryRequest::new();
        assert!(req.is_empty());
        assert_eq!(req.point_count(), 0);
    }

    #[test]
    fn test_trajectory_request_straight() {
        let req = TrajectoryRequest::straight(2.0, 1.0);
        assert!(!req.is_empty());
        assert_eq!(req.speed_profile, 2.0);
        assert_eq!(req.curvature, 0.0);
    }

    #[test]
    fn test_trajectory_request_turning() {
        let req = TrajectoryRequest::turning(2.0, PI / 4.0, 1.0);
        assert!(!req.is_empty());
        assert!(req.curvature > 0.0);
    }

    #[test]
    fn test_trajectory_request_stopping() {
        let req = TrajectoryRequest::stopping(3.0, 0.5);
        assert!(req.stopping);
        assert!(!req.is_empty());
    }

    #[test]
    fn test_trajectory_request_starting() {
        let req = TrajectoryRequest::starting(3.0, 0.5, Vec3::Z);
        assert!(req.starting);
        assert!(!req.is_empty());
    }

    #[test]
    fn test_trajectory_request_sample_at() {
        let req = TrajectoryRequest::straight(2.0, 1.0);
        let point = req.sample_at(0.3);
        assert!(point.position.z > 0.0);
    }

    #[test]
    fn test_trajectory_request_to_trajectory_feature() {
        let req = TrajectoryRequest::straight(2.0, 1.0);
        let feature = req.to_trajectory_feature();
        assert!(feature.future_positions[0].z > 0.0);
    }

    #[test]
    fn test_trajectory_request_to_search_trajectory() {
        let req = TrajectoryRequest::straight(2.0, 1.0);
        let compat = req.to_search_trajectory();
        assert!(compat.future_positions[0].z > 0.0);
    }

    #[test]
    fn test_trajectory_request_distance() {
        let a = TrajectoryRequest::straight(2.0, 1.0);
        let b = TrajectoryRequest::straight(4.0, 1.0);
        let dist = a.distance(&b);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_trajectory_request_lerp() {
        let a = TrajectoryRequest::straight(2.0, 1.0);
        let b = TrajectoryRequest::straight(4.0, 1.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.speed_profile - 3.0).abs() < 0.01);
    }

    // =====================================================================
    // MotionState tests
    // =====================================================================

    #[test]
    fn test_motion_state_new() {
        let state = MotionState::new();
        assert_eq!(state.playback_speed, 1.0);
        assert!(!state.in_transition);
    }

    #[test]
    fn test_motion_state_from_clip() {
        let state = MotionState::from_clip(5, 1.5);
        assert_eq!(state.clip_index, 5);
        assert_eq!(state.time, 1.5);
    }

    #[test]
    fn test_motion_state_advance() {
        let mut state = MotionState::new();
        state.advance(0.1);
        assert!((state.time - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_motion_state_advance_with_speed() {
        let mut state = MotionState::new();
        state.playback_speed = 2.0;
        state.advance(0.1);
        assert!((state.time - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_motion_state_start_transition() {
        let mut state = MotionState::new();
        state.start_transition(3, 0.5, 0.2);

        assert_eq!(state.clip_index, 3);
        assert_eq!(state.time, 0.5);
        assert!(state.in_transition);
        assert!((state.transition_remaining - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_motion_state_transition_ends() {
        let mut state = MotionState::new();
        state.start_transition(1, 0.0, 0.1);

        state.advance(0.15);
        assert!(!state.in_transition);
    }

    #[test]
    fn test_motion_state_speed() {
        let mut state = MotionState::new();
        state.root_velocity = Vec3::new(3.0, 0.0, 4.0);
        assert!((state.speed() - 5.0).abs() < 0.001);
    }

    // =====================================================================
    // TagSet tests
    // =====================================================================

    #[test]
    fn test_tag_set_new() {
        let tags = TagSet::new();
        assert!(tags.is_empty());
    }

    #[test]
    fn test_tag_set_from_locomotion() {
        let tags = TagSet::from_locomotion(LocomotionTags::WALK);
        assert!(tags.has_locomotion(LocomotionTags::WALK));
    }

    #[test]
    fn test_tag_set_with_custom() {
        let tags = TagSet::new().with_custom("test_tag");
        assert!(tags.has_custom("test_tag"));
    }

    #[test]
    fn test_tag_set_merge() {
        let mut a = TagSet::from_locomotion(LocomotionTags::WALK);
        let b = TagSet::from_locomotion(LocomotionTags::RUN);
        a.merge(&b);
        assert!(a.has_locomotion(LocomotionTags::WALK));
        assert!(a.has_locomotion(LocomotionTags::RUN));
    }

    #[test]
    fn test_tag_set_clear() {
        let mut tags = TagSet::from_locomotion(LocomotionTags::WALK).with_custom("test");
        tags.clear();
        assert!(tags.is_empty());
    }

    // =====================================================================
    // TagManager tests
    // =====================================================================

    #[test]
    fn test_tag_manager_new() {
        let manager = TagManager::new();
        assert!(manager.active.is_empty());
    }

    #[test]
    fn test_tag_manager_with_standard_rules() {
        let manager = TagManager::with_standard_rules();
        assert!(!manager.priorities.is_empty());
        assert!(!manager.exclusions.is_empty());
    }

    #[test]
    fn test_tag_manager_add_tags() {
        let mut manager = TagManager::new();
        manager.add_tags(LocomotionTags::WALK);
        assert!(manager.active.has_locomotion(LocomotionTags::WALK));
    }

    #[test]
    fn test_tag_manager_remove_tags() {
        let mut manager = TagManager::new();
        manager.add_tags(LocomotionTags::WALK | LocomotionTags::RUN);
        manager.remove_tags(LocomotionTags::WALK);
        assert!(!manager.active.has_locomotion(LocomotionTags::WALK));
        assert!(manager.active.has_locomotion(LocomotionTags::RUN));
    }

    #[test]
    fn test_tag_manager_terrain_override() {
        let mut manager = TagManager::new();
        manager.set_terrain_override(1, TerrainType::SlopeUp);
        assert_eq!(manager.get_terrain(Some(1)), TerrainType::SlopeUp);
        assert_eq!(manager.get_terrain(Some(2)), TerrainType::Flat);
    }

    #[test]
    fn test_tag_manager_update_from_style() {
        let mut manager = TagManager::new();
        manager.update_from_style(LocomotionStyle::Run);
        assert!(manager.active.has_locomotion(LocomotionTags::RUN));
        assert!(!manager.active.has_locomotion(LocomotionTags::WALK));
    }

    #[test]
    fn test_tag_manager_update_from_terrain() {
        let mut manager = TagManager::new();
        manager.update_from_terrain(TerrainType::SlopeUp);
        assert!(manager.active.has_locomotion(LocomotionTags::TERRAIN_UPHILL));
    }

    // =====================================================================
    // FootContactState tests
    // =====================================================================

    #[test]
    fn test_foot_contact_state_new() {
        let state = FootContactState::new();
        assert!(!state.planted);
        assert_eq!(state.position, Vec3::ZERO);
    }

    #[test]
    fn test_foot_contact_state_planted() {
        let state = FootContactState::planted(Vec3::new(1.0, 0.0, 2.0));
        assert!(state.planted);
        assert_eq!(state.position, Vec3::new(1.0, 0.0, 2.0));
    }

    #[test]
    fn test_foot_contact_state_moving() {
        let state = FootContactState::moving(Vec3::Y, Vec3::X);
        assert!(!state.planted);
        assert_eq!(state.velocity, Vec3::X);
    }

    #[test]
    fn test_foot_contact_state_update_plants() {
        let mut state = FootContactState::moving(Vec3::Y * 0.5, Vec3::X);
        state.update(Vec3::new(0.0, 0.05, 0.0), Vec3::ZERO, 0.016);
        assert!(state.planted);
    }

    #[test]
    fn test_foot_contact_state_update_lifts() {
        let mut state = FootContactState::planted(Vec3::ZERO);
        state.update(Vec3::Y * 0.5, Vec3::new(0.0, 1.0, 0.0), 0.016);
        assert!(!state.planted);
    }

    // =====================================================================
    // FootContactTracker tests
    // =====================================================================

    #[test]
    fn test_foot_contact_tracker_new() {
        let tracker = FootContactTracker::new(2);
        assert_eq!(tracker.foot_count(), 2);
    }

    #[test]
    fn test_foot_contact_tracker_bipedal() {
        let tracker = FootContactTracker::bipedal(5, 6);
        assert_eq!(tracker.foot_count(), 2);
        assert_eq!(tracker.foot_bones, vec![5, 6]);
    }

    #[test]
    fn test_foot_contact_tracker_quadruped() {
        let tracker = FootContactTracker::quadruped([1, 2, 3, 4]);
        assert_eq!(tracker.foot_count(), 4);
    }

    #[test]
    fn test_foot_contact_tracker_update() {
        let mut tracker = FootContactTracker::new(2);
        let positions = vec![Vec3::new(0.0, 0.05, 0.0), Vec3::Y * 0.5];
        let velocities = vec![Vec3::ZERO, Vec3::Y];

        tracker.update(&positions, &velocities, 0.016, 0.0);

        assert!(tracker.feet[0].planted);
        assert!(!tracker.feet[1].planted);
    }

    #[test]
    fn test_foot_contact_tracker_any_planted() {
        let mut tracker = FootContactTracker::new(2);
        tracker.feet[0] = FootContactState::planted(Vec3::ZERO);
        tracker.feet[1] = FootContactState::moving(Vec3::Y, Vec3::ZERO);

        assert!(tracker.any_planted());
        assert!(!tracker.all_planted());
        assert_eq!(tracker.planted_count(), 1);
    }

    #[test]
    fn test_foot_contact_tracker_to_foot_features() {
        let mut tracker = FootContactTracker::new(2);
        tracker.feet[0] = FootContactState::planted(Vec3::X);
        let features = tracker.to_foot_features();

        assert!(features.contact_states[0]);
        assert_eq!(features.positions[0], Vec3::X);
    }

    // =====================================================================
    // ContextUpdatePolicy tests
    // =====================================================================

    #[test]
    fn test_context_update_policy_default() {
        let policy = ContextUpdatePolicy::default();
        assert_eq!(policy.update_interval, DEFAULT_UPDATE_INTERVAL);
    }

    #[test]
    fn test_context_update_policy_responsive() {
        let policy = ContextUpdatePolicy::responsive();
        assert!(policy.update_interval < DEFAULT_UPDATE_INTERVAL);
    }

    #[test]
    fn test_context_update_policy_conservative() {
        let policy = ContextUpdatePolicy::conservative();
        assert!(policy.update_interval > DEFAULT_UPDATE_INTERVAL);
    }

    #[test]
    fn test_context_update_policy_update() {
        let mut policy = ContextUpdatePolicy::new();
        policy.update(0.016);
        assert_eq!(policy.frame_counter, 1);
        assert!(policy.time_since_search > 0.0);
    }

    #[test]
    fn test_context_update_policy_on_search() {
        let mut policy = ContextUpdatePolicy::new();
        policy.time_since_search = 1.0;
        policy.on_search(0.5);
        assert_eq!(policy.time_since_search, 0.0);
        assert_eq!(policy.last_search_cost, 0.5);
    }

    #[test]
    fn test_context_update_policy_on_transition() {
        let mut policy = ContextUpdatePolicy::new();
        policy.time_since_transition = 1.0;
        policy.on_transition();
        assert_eq!(policy.time_since_transition, 0.0);
    }

    #[test]
    fn test_context_update_policy_should_search_interval() {
        let mut policy = ContextUpdatePolicy::new();
        policy.update_interval = 4;
        policy.frame_counter = 3;
        assert!(!policy.should_search(1.0, false));

        policy.frame_counter = 4;
        assert!(policy.should_search(1.0, false));
    }

    #[test]
    fn test_context_update_policy_should_search_cooldown() {
        let mut policy = ContextUpdatePolicy::new();
        policy.transition_cooldown = 0.2;
        policy.time_since_transition = 0.1;
        assert!(!policy.should_search(1.0, false));

        policy.time_since_transition = 0.3;
        policy.update_interval = 1;
        assert!(policy.should_search(1.0, false));
    }

    #[test]
    fn test_context_update_policy_should_search_max_time() {
        let mut policy = ContextUpdatePolicy::new();
        policy.update_interval = 1;
        policy.max_time_between_searches = 0.5;
        policy.time_since_search = 0.6;
        policy.time_since_transition = 1.0;
        assert!(policy.should_search(0.0, false));
    }

    #[test]
    fn test_context_update_policy_reset() {
        let mut policy = ContextUpdatePolicy::new();
        policy.frame_counter = 100;
        policy.time_since_search = 5.0;
        policy.reset();
        assert_eq!(policy.frame_counter, 0);
        assert_eq!(policy.time_since_search, 0.0);
    }

    // =====================================================================
    // ContextInterpolator tests
    // =====================================================================

    #[test]
    fn test_context_interpolator_new() {
        let interp = ContextInterpolator::new();
        assert!(!interp.is_active());
        assert_eq!(interp.progress(), 1.0);
    }

    #[test]
    fn test_context_interpolator_with_smoothness() {
        let interp = ContextInterpolator::with_smoothness(0.5);
        assert_eq!(interp.smoothness, 0.5);
    }

    #[test]
    fn test_context_interpolator_set_target() {
        let mut interp = ContextInterpolator::new();
        let target = TrajectoryRequest::straight(2.0, 1.0);
        interp.set_target(target);

        assert!(interp.is_active());
        assert_eq!(interp.progress(), 0.0);
    }

    #[test]
    fn test_context_interpolator_update() {
        let mut interp = ContextInterpolator::new();
        interp.set_target(TrajectoryRequest::straight(2.0, 1.0));
        interp.duration = 0.2;

        interp.update(0.1);
        assert!(interp.progress() > 0.0);
        assert!(interp.progress() < 1.0);
        assert!(interp.is_active());

        interp.update(0.2);
        assert_eq!(interp.progress(), 1.0);
        assert!(!interp.is_active());
    }

    #[test]
    fn test_context_interpolator_force_immediate() {
        let mut interp = ContextInterpolator::new();
        let traj = TrajectoryRequest::straight(3.0, 1.0);
        interp.force_immediate(traj.clone());

        assert!(!interp.is_active());
        assert_eq!(interp.current().speed_profile, 3.0);
    }

    #[test]
    fn test_context_interpolator_handle_direction_change() {
        let mut interp = ContextInterpolator::new();
        interp.smoothness = 0.5;

        let traj = TrajectoryRequest::turning(2.0, PI / 2.0, 1.0);
        interp.handle_direction_change(traj, 0.8);

        assert!(interp.is_active());
        assert!(interp.duration < 0.3); // Should be shortened due to urgency
    }

    #[test]
    fn test_context_interpolator_reset() {
        let mut interp = ContextInterpolator::new();
        interp.set_target(TrajectoryRequest::straight(2.0, 1.0));
        interp.reset();

        assert!(!interp.is_active());
        assert!(interp.current().is_empty());
    }

    // =====================================================================
    // MotionContext tests
    // =====================================================================

    #[test]
    fn test_motion_context_new() {
        let ctx = MotionContext::new();
        assert_eq!(ctx.foot_tracker.foot_count(), 2);
        assert_eq!(ctx.time, 0.0);
    }

    #[test]
    fn test_motion_context_with_foot_count() {
        let ctx = MotionContext::with_foot_count(4);
        assert_eq!(ctx.foot_tracker.foot_count(), 4);
    }

    #[test]
    fn test_motion_context_update() {
        let mut ctx = MotionContext::new();
        ctx.update(0.016);
        assert!(ctx.time > 0.0);
    }

    #[test]
    fn test_motion_context_set_trajectory() {
        let mut ctx = MotionContext::new();
        ctx.set_trajectory(TrajectoryRequest::straight(2.0, 1.0));
        assert!(ctx.interpolator.is_active());
    }

    #[test]
    fn test_motion_context_set_trajectory_immediate() {
        let mut ctx = MotionContext::new();
        ctx.set_trajectory_immediate(TrajectoryRequest::straight(2.0, 1.0));
        assert!(!ctx.interpolator.is_active());
        assert_eq!(ctx.current_trajectory().speed_profile, 2.0);
    }

    #[test]
    fn test_motion_context_should_search() {
        let mut ctx = MotionContext::new();
        ctx.update_policy.update_interval = 1;
        ctx.update_policy.time_since_transition = 1.0;
        ctx.update_policy.max_time_between_searches = 0.1;
        ctx.update_policy.time_since_search = 0.2;
        assert!(ctx.should_search(1.0));
    }

    #[test]
    fn test_motion_context_start_transition() {
        let mut ctx = MotionContext::new();
        ctx.start_transition(5, 0.3, 0.2);

        assert!(ctx.in_transition());
        assert_eq!(ctx.motion_state.clip_index, 5);
        assert_eq!(ctx.update_policy.time_since_transition, 0.0);
    }

    #[test]
    fn test_motion_context_to_search_query() {
        let mut ctx = MotionContext::new();
        ctx.set_trajectory_immediate(TrajectoryRequest::straight(2.0, 1.0));
        ctx.motion_state = MotionState::from_clip(3, 1.5);

        let query = ctx.to_search_query();
        assert_eq!(query.current_clip, Some((3, 1.5)));
    }

    #[test]
    fn test_motion_context_speed() {
        let mut ctx = MotionContext::new();
        ctx.motion_state.root_velocity = Vec3::new(3.0, 0.0, 4.0);
        assert!((ctx.speed() - 5.0).abs() < 0.001);
    }

    // =====================================================================
    // ContextBuilder tests
    // =====================================================================

    #[test]
    fn test_context_builder_new() {
        let builder = ContextBuilder::new();
        let ctx = builder.build();
        assert!(ctx.trajectory.is_empty() || ctx.interpolator.current().is_empty());
    }

    #[test]
    fn test_context_builder_from_controller_input() {
        let builder = ContextBuilder::new()
            .from_controller_input(Vec3::new(0.0, 0.0, 1.0), Vec3::Z, 2.0, 0.016);
        let ctx = builder.build();

        assert!(!ctx.interpolator.current().is_empty());
        assert!(ctx.tags.has_locomotion(LocomotionTags::WALK));
    }

    #[test]
    fn test_context_builder_from_navigation_path() {
        let path = vec![
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, 4.0),
            Vec3::new(0.0, 0.0, 6.0),
        ];
        let builder = ContextBuilder::new()
            .from_navigation_path(&path, Vec3::ZERO, 2.0);
        let ctx = builder.build();

        assert!(!ctx.interpolator.current().is_empty());
    }

    #[test]
    fn test_context_builder_from_animation_events() {
        let events = vec!["combat_enter".to_string(), "custom_event".to_string()];
        let builder = ContextBuilder::new()
            .from_animation_events(&events, None);
        let ctx = builder.build();

        assert!(ctx.tags.has_locomotion(LocomotionTags::COMBAT));
        assert!(ctx.tags.has_custom("custom_event"));
    }

    #[test]
    fn test_context_builder_with_pose() {
        let pose = MotionFeatures {
            pose: vec![1.0, 2.0, 3.0],
            ..Default::default()
        };
        let ctx = ContextBuilder::new().with_pose(pose).build();
        assert_eq!(ctx.pose.pose, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_context_builder_with_trajectory() {
        let traj = TrajectoryRequest::straight(3.0, 1.0);
        let ctx = ContextBuilder::new().with_trajectory(traj).build();
        assert_eq!(ctx.interpolator.current().speed_profile, 3.0);
    }

    #[test]
    fn test_context_builder_with_motion_state() {
        let state = MotionState::from_clip(5, 1.5);
        let ctx = ContextBuilder::new().with_motion_state(state).build();
        assert_eq!(ctx.motion_state.clip_index, 5);
    }

    #[test]
    fn test_context_builder_with_tags() {
        let tags = TagSet::from_locomotion(LocomotionTags::RUN);
        let ctx = ContextBuilder::new().with_tags(tags).build();
        assert!(ctx.tags.has_locomotion(LocomotionTags::RUN));
    }

    #[test]
    fn test_context_builder_with_foot_tracker() {
        let tracker = FootContactTracker::quadruped([1, 2, 3, 4]);
        let ctx = ContextBuilder::new().with_foot_tracker(tracker).build();
        assert_eq!(ctx.foot_tracker.foot_count(), 4);
    }

    #[test]
    fn test_context_builder_with_update_policy() {
        let policy = ContextUpdatePolicy::responsive();
        let ctx = ContextBuilder::new().with_update_policy(policy).build();
        assert!(ctx.update_policy.update_interval < DEFAULT_UPDATE_INTERVAL);
    }

    #[test]
    fn test_context_builder_with_root_transform() {
        let ctx = ContextBuilder::new()
            .with_root_transform(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY)
            .build();
        assert_eq!(ctx.root_position, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_context_builder_with_time() {
        let ctx = ContextBuilder::new().with_time(5.0).build();
        assert_eq!(ctx.time, 5.0);
    }

    #[test]
    fn test_context_builder_combine() {
        let a = ContextBuilder::new()
            .with_tags(TagSet::from_locomotion(LocomotionTags::WALK));
        let b = ContextBuilder::new()
            .with_trajectory(TrajectoryRequest::straight(2.0, 1.0));

        let ctx = a.combine(b).build();

        assert!(ctx.tags.has_locomotion(LocomotionTags::WALK));
        assert_eq!(ctx.interpolator.current().speed_profile, 2.0);
    }

    // =====================================================================
    // Integration tests
    // =====================================================================

    #[test]
    fn test_full_context_workflow() {
        // Build context from controller input
        let mut ctx = ContextBuilder::new()
            .from_controller_input(Vec3::Z, Vec3::Z, 2.0, 0.016)
            .with_foot_tracker(FootContactTracker::bipedal(5, 6))
            .with_update_policy(ContextUpdatePolicy::responsive())
            .build();

        // Simulate a few frames
        for _ in 0..10 {
            ctx.update(0.016);
        }

        // Check state
        assert!(ctx.time > 0.0);

        // Generate search query
        let query = ctx.to_search_query();
        assert!(!query.desired_trajectory.future_positions[0].is_nan());
    }

    #[test]
    fn test_trajectory_interpolation_workflow() {
        let mut ctx = MotionContext::new();

        // Set initial trajectory
        ctx.set_trajectory_immediate(TrajectoryRequest::straight(2.0, 1.0));

        // Change direction
        ctx.set_trajectory(TrajectoryRequest::turning(2.0, PI / 4.0, 1.0));

        // Interpolate over several frames
        for _ in 0..20 {
            ctx.update(0.016);
        }

        // Should have completed interpolation
        assert!(!ctx.interpolator.is_active());
    }

    #[test]
    fn test_tag_management_workflow() {
        let mut manager = TagManager::with_standard_rules();

        // Start walking
        manager.update_from_style(LocomotionStyle::Walk);
        assert!(manager.active.has_locomotion(LocomotionTags::WALK));

        // Switch to running
        manager.update_from_style(LocomotionStyle::Run);
        assert!(manager.active.has_locomotion(LocomotionTags::RUN));
        assert!(!manager.active.has_locomotion(LocomotionTags::WALK));

        // Add terrain
        manager.update_from_terrain(TerrainType::SlopeUp);
        assert!(manager.active.has_locomotion(LocomotionTags::TERRAIN_UPHILL));
    }

    #[test]
    fn test_foot_tracking_workflow() {
        let mut tracker = FootContactTracker::bipedal(5, 6);

        // Simulate walking gait
        for frame in 0..60 {
            let phase = (frame as f32 / 30.0) * std::f32::consts::TAU;

            // Left foot plants when right is up and vice versa
            let left_height = 0.1 * (phase).sin().max(0.0);
            let right_height = 0.1 * (phase + PI).sin().max(0.0);

            let positions = vec![
                Vec3::new(-0.1, left_height, 0.0),
                Vec3::new(0.1, right_height, 0.0),
            ];
            let velocities = vec![
                Vec3::new(0.0, left_height * 60.0, 0.0),
                Vec3::new(0.0, right_height * 60.0, 0.0),
            ];

            tracker.update(&positions, &velocities, 1.0 / 60.0, frame as f32 / 60.0);
        }

        // Should have alternating contacts
        assert!(tracker.any_planted());
    }

    // =====================================================================
    // Edge case tests
    // =====================================================================

    #[test]
    fn test_empty_navigation_path() {
        let ctx = ContextBuilder::new()
            .from_navigation_path(&[], Vec3::ZERO, 2.0)
            .build();
        assert!(ctx.interpolator.current().speed_profile == 0.0);
    }

    #[test]
    fn test_zero_speed_trajectory() {
        let traj = TrajectoryRequest::straight(0.0, 1.0);
        assert!(!traj.is_empty());
    }

    #[test]
    fn test_180_degree_turn() {
        let traj = TrajectoryRequest::turning(2.0, PI, 1.0);
        assert!(traj.curvature.abs() > 0.9);
    }

    #[test]
    fn test_very_short_transition_cooldown() {
        let mut policy = ContextUpdatePolicy::new();
        policy.transition_cooldown = 0.001;
        policy.time_since_transition = 0.002;
        policy.update_interval = 1;
        policy.max_time_between_searches = 10.0;

        assert!(policy.should_search(1.0, false));
    }

    #[test]
    fn test_interpolator_very_small_duration() {
        let mut interp = ContextInterpolator::new();
        interp.smoothness = 0.0;
        interp.set_target(TrajectoryRequest::straight(2.0, 1.0));

        // Should still work with small duration
        interp.update(0.001);
        assert!(interp.progress() > 0.0);
    }
}
