//! Motion matching runtime search for TRINITY Engine (T-AN-6.3).
//!
//! This module provides runtime motion matching search with configurable cost functions,
//! ANN-based nearest neighbor search using KD-trees, and performance budget enforcement.
//!
//! # Architecture
//!
//! ```text
//! MotionSearcher
//! ├── database: MotionDatabase      # Pre-processed motion frames
//! ├── config: SearchConfig          # Search parameters
//! │   ├── weights: SearchCostWeights
//! │   ├── cost_threshold: f32
//! │   ├── min_clip_time: f32
//! │   ├── stickiness: f32
//! │   └── budget_ms: f32
//! └── search methods
//!     ├── search()                  # Single best match
//!     ├── search_k_nearest()        # K best matches
//!     └── compute_cost()            # Cost calculation
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::motion_search::{
//!     MotionSearcher, SearchConfig, SearchQuery, SearchCostWeights,
//! };
//!
//! // Create searcher
//! let config = SearchConfig::default();
//! let searcher = MotionSearcher::new(database, config);
//!
//! // Search for best match
//! let query = SearchQuery {
//!     current_pose: features,
//!     desired_trajectory: trajectory,
//!     current_clip: Some((0, 0.5)),
//!     required_tags: MotionTags::walk(),
//! };
//! if let Some(result) = searcher.search(&query) {
//!     println!("Best match: frame {}, cost {}", result.frame_index, result.cost);
//! }
//! ```

use std::time::Instant;

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::motion_features::{
    compute_feature_distance_squared, FeatureWeights, FootFeatures, MotionFeatures, MotionTags,
};
use crate::motion_matching_db::{
    LocomotionTags, MotionDatabase, MotionFrame, TRAJECTORY_SAMPLES,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default search time budget in milliseconds.
pub const DEFAULT_BUDGET_MS: f32 = 2.0;

/// Default cost threshold for rejecting matches.
pub const DEFAULT_COST_THRESHOLD: f32 = 100.0;

/// Default minimum time in a clip before switching (seconds).
pub const DEFAULT_MIN_CLIP_TIME: f32 = 0.1;

/// Default stickiness bonus for staying in current clip.
pub const DEFAULT_STICKINESS: f32 = 0.05;

/// Maximum candidates to evaluate in exhaustive fallback.
pub const MAX_EXHAUSTIVE_CANDIDATES: usize = 1000;

/// Default K for approximate nearest neighbor search.
pub const DEFAULT_K_NEIGHBORS: usize = 32;

// ---------------------------------------------------------------------------
// SearchCostWeights
// ---------------------------------------------------------------------------

/// Weights for computing search cost.
///
/// Each weight controls the relative importance of different feature
/// components in the motion matching cost function.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchCostWeights {
    /// Weight for pose position features.
    pub pose_weight: f32,

    /// Weight for trajectory features.
    pub trajectory_weight: f32,

    /// Weight for velocity features.
    pub velocity_weight: f32,

    /// Weight for transition smoothness.
    pub transition_weight: f32,

    /// Penalty for tag mismatch.
    pub tag_mismatch_penalty: f32,

    /// Weight for foot contact features.
    pub foot_contact_weight: f32,

    /// Weight for foot position features.
    pub foot_position_weight: f32,

    /// Weight for foot velocity features.
    pub foot_velocity_weight: f32,
}

impl Default for SearchCostWeights {
    fn default() -> Self {
        Self {
            pose_weight: 1.0,
            trajectory_weight: 1.0,
            velocity_weight: 0.1,
            transition_weight: 0.5,
            tag_mismatch_penalty: 10.0,
            foot_contact_weight: 2.0,
            foot_position_weight: 0.5,
            foot_velocity_weight: 0.1,
        }
    }
}

impl SearchCostWeights {
    /// Create trajectory-focused weights.
    pub fn trajectory_focused() -> Self {
        Self {
            pose_weight: 0.5,
            trajectory_weight: 2.0,
            velocity_weight: 0.1,
            transition_weight: 0.3,
            tag_mismatch_penalty: 10.0,
            foot_contact_weight: 1.0,
            foot_position_weight: 0.3,
            foot_velocity_weight: 0.05,
        }
    }

    /// Create pose-focused weights.
    pub fn pose_focused() -> Self {
        Self {
            pose_weight: 2.0,
            trajectory_weight: 0.5,
            velocity_weight: 0.3,
            transition_weight: 0.8,
            tag_mismatch_penalty: 10.0,
            foot_contact_weight: 1.5,
            foot_position_weight: 0.8,
            foot_velocity_weight: 0.2,
        }
    }

    /// Create transition-focused weights (for smooth blending).
    pub fn transition_focused() -> Self {
        Self {
            pose_weight: 1.0,
            trajectory_weight: 0.5,
            velocity_weight: 0.5,
            transition_weight: 2.0,
            tag_mismatch_penalty: 5.0,
            foot_contact_weight: 3.0,
            foot_position_weight: 1.0,
            foot_velocity_weight: 0.5,
        }
    }

    /// Create uniform weights.
    pub fn uniform() -> Self {
        Self {
            pose_weight: 1.0,
            trajectory_weight: 1.0,
            velocity_weight: 1.0,
            transition_weight: 1.0,
            tag_mismatch_penalty: 1.0,
            foot_contact_weight: 1.0,
            foot_position_weight: 1.0,
            foot_velocity_weight: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// SearchConfig
// ---------------------------------------------------------------------------

/// Configuration for motion matching search.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Cost weights for different feature components.
    pub weights: SearchCostWeights,

    /// Maximum cost threshold (reject matches above this).
    pub cost_threshold: f32,

    /// Minimum time in current clip before switching (seconds).
    pub min_clip_time: f32,

    /// Stickiness bonus for staying in current clip.
    pub stickiness: f32,

    /// Search time budget in milliseconds.
    pub budget_ms: f32,

    /// Maximum candidates to evaluate.
    pub max_candidates: usize,

    /// K value for nearest neighbor search.
    pub k_neighbors: usize,

    /// Distance threshold for pruning candidates.
    pub distance_threshold: Option<f32>,

    /// Whether to use early termination.
    pub early_termination: bool,

    /// Minimum improvement required to accept a candidate.
    pub min_improvement: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            weights: SearchCostWeights::default(),
            cost_threshold: DEFAULT_COST_THRESHOLD,
            min_clip_time: DEFAULT_MIN_CLIP_TIME,
            stickiness: DEFAULT_STICKINESS,
            budget_ms: DEFAULT_BUDGET_MS,
            max_candidates: MAX_EXHAUSTIVE_CANDIDATES,
            k_neighbors: DEFAULT_K_NEIGHBORS,
            distance_threshold: None,
            early_termination: true,
            min_improvement: 0.01,
        }
    }
}

impl SearchConfig {
    /// Create a fast config with lower quality.
    pub fn fast() -> Self {
        Self {
            budget_ms: 0.5,
            max_candidates: 100,
            k_neighbors: 8,
            early_termination: true,
            ..Default::default()
        }
    }

    /// Create a quality-focused config.
    pub fn quality() -> Self {
        Self {
            budget_ms: 5.0,
            max_candidates: 2000,
            k_neighbors: 64,
            early_termination: false,
            ..Default::default()
        }
    }

    /// Create a balanced config.
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Set the weights.
    pub fn with_weights(mut self, weights: SearchCostWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Set the cost threshold.
    pub fn with_cost_threshold(mut self, threshold: f32) -> Self {
        self.cost_threshold = threshold;
        self
    }

    /// Set the minimum clip time.
    pub fn with_min_clip_time(mut self, time: f32) -> Self {
        self.min_clip_time = time;
        self
    }

    /// Set the stickiness.
    pub fn with_stickiness(mut self, stickiness: f32) -> Self {
        self.stickiness = stickiness;
        self
    }

    /// Set the time budget.
    pub fn with_budget_ms(mut self, budget_ms: f32) -> Self {
        self.budget_ms = budget_ms;
        self
    }
}

// ---------------------------------------------------------------------------
// TrajectoryFeatureCompat
// ---------------------------------------------------------------------------

/// Trajectory feature for search queries (motion_features compatible).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrajectoryFeatureCompat {
    /// Future positions at T+0.2, T+0.5, T+1.0 seconds.
    pub future_positions: [Vec3; 3],

    /// Future facing angles.
    pub future_facings: [f32; 3],
}

impl TrajectoryFeatureCompat {
    /// Create from flat vector.
    pub fn from_flat_vector(flat: &[f32]) -> Self {
        let mut result = Self::default();
        for i in 0..3 {
            let base = i * 3;
            if base + 2 < flat.len() {
                result.future_positions[i] = Vec3::new(flat[base], flat[base + 1], flat[base + 2]);
            }
        }
        for i in 0..3 {
            let idx = 9 + i;
            if idx < flat.len() {
                result.future_facings[i] = flat[idx];
            }
        }
        result
    }

    /// Convert to flat vector.
    pub fn to_flat_vector(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(12);
        for pos in &self.future_positions {
            flat.push(pos.x);
            flat.push(pos.y);
            flat.push(pos.z);
        }
        for &facing in &self.future_facings {
            flat.push(facing);
        }
        flat
    }

    /// Compute distance to another trajectory.
    pub fn distance_squared(&self, other: &TrajectoryFeatureCompat, pos_weight: f32, facing_weight: f32) -> f32 {
        let mut dist = 0.0;
        for i in 0..3 {
            dist += (self.future_positions[i] - other.future_positions[i]).length_squared() * pos_weight;
        }
        for i in 0..3 {
            let angle_diff = (self.future_facings[i] - other.future_facings[i]).abs();
            let wrapped = angle_diff.min(std::f32::consts::TAU - angle_diff);
            dist += wrapped * wrapped * facing_weight;
        }
        dist
    }
}

// ---------------------------------------------------------------------------
// SearchQuery
// ---------------------------------------------------------------------------

/// Query for motion matching search.
#[derive(Clone, Debug, Default)]
pub struct SearchQuery {
    /// Current pose features.
    pub current_pose: MotionFeatures,

    /// Desired future trajectory.
    pub desired_trajectory: TrajectoryFeatureCompat,

    /// Current clip and time (clip_id, time), if any.
    pub current_clip: Option<(usize, f32)>,

    /// Required motion tags.
    pub required_tags: MotionTags,

    /// Current root velocity (for transition cost).
    pub root_velocity: Vec3,

    /// Current root angular velocity.
    pub root_angular_velocity: f32,
}

impl SearchQuery {
    /// Create a new empty query.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the current pose.
    pub fn with_pose(mut self, pose: MotionFeatures) -> Self {
        self.current_pose = pose;
        self
    }

    /// Set the desired trajectory.
    pub fn with_trajectory(mut self, trajectory: TrajectoryFeatureCompat) -> Self {
        self.desired_trajectory = trajectory;
        self
    }

    /// Set the current clip.
    pub fn with_current_clip(mut self, clip_id: usize, time: f32) -> Self {
        self.current_clip = Some((clip_id, time));
        self
    }

    /// Set the required tags.
    pub fn with_required_tags(mut self, tags: MotionTags) -> Self {
        self.required_tags = tags;
        self
    }

    /// Set the root velocity.
    pub fn with_root_velocity(mut self, velocity: Vec3) -> Self {
        self.root_velocity = velocity;
        self
    }
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

/// Result from a motion matching search.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    /// Frame index in the database.
    pub frame_index: usize,

    /// Clip index.
    pub clip_index: usize,

    /// Time within the clip (seconds).
    pub time: f32,

    /// Total cost of this match.
    pub cost: f32,

    /// Number of candidates evaluated.
    pub candidates_evaluated: usize,

    /// Search time in milliseconds.
    pub search_time_ms: f32,

    /// Whether the budget was exceeded.
    pub budget_exceeded: bool,

    /// Individual cost components (for debugging).
    pub cost_breakdown: Option<CostBreakdown>,
}

impl SearchResult {
    /// Create a new search result.
    pub fn new(frame_index: usize, clip_index: usize, time: f32, cost: f32) -> Self {
        Self {
            frame_index,
            clip_index,
            time,
            cost,
            candidates_evaluated: 0,
            search_time_ms: 0.0,
            budget_exceeded: false,
            cost_breakdown: None,
        }
    }

    /// Check if this is a valid match.
    pub fn is_valid(&self, threshold: f32) -> bool {
        self.cost <= threshold && !self.budget_exceeded
    }
}

// ---------------------------------------------------------------------------
// CostBreakdown
// ---------------------------------------------------------------------------

/// Breakdown of individual cost components.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CostBreakdown {
    /// Pose cost component.
    pub pose_cost: f32,

    /// Trajectory cost component.
    pub trajectory_cost: f32,

    /// Velocity cost component.
    pub velocity_cost: f32,

    /// Transition cost component.
    pub transition_cost: f32,

    /// Foot contact cost component.
    pub foot_cost: f32,

    /// Tag mismatch penalty.
    pub tag_penalty: f32,

    /// Stickiness bonus applied.
    pub stickiness_bonus: f32,
}

impl CostBreakdown {
    /// Get the total cost.
    pub fn total(&self) -> f32 {
        self.pose_cost
            + self.trajectory_cost
            + self.velocity_cost
            + self.transition_cost
            + self.foot_cost
            + self.tag_penalty
            - self.stickiness_bonus
    }
}

// ---------------------------------------------------------------------------
// SearchStatistics
// ---------------------------------------------------------------------------

/// Statistics from a search operation.
#[derive(Clone, Debug, Default)]
pub struct SearchStatistics {
    /// Total candidates considered.
    pub candidates_total: usize,

    /// Candidates passing tag filter.
    pub candidates_after_tags: usize,

    /// Candidates passing distance threshold.
    pub candidates_after_pruning: usize,

    /// Final candidates evaluated.
    pub candidates_evaluated: usize,

    /// Search time in milliseconds.
    pub search_time_ms: f32,

    /// KD-tree search time.
    pub kd_tree_time_ms: f32,

    /// Cost evaluation time.
    pub cost_eval_time_ms: f32,

    /// Best cost found.
    pub best_cost: f32,

    /// Whether budget was exceeded.
    pub budget_exceeded: bool,
}

// ---------------------------------------------------------------------------
// MotionSearcher
// ---------------------------------------------------------------------------

/// Motion matching searcher for finding best animation frames.
///
/// Combines KD-tree based ANN search with configurable cost functions
/// and performance budgeting.
pub struct MotionSearcher {
    /// Motion database to search.
    pub database: MotionDatabase,

    /// Search configuration.
    pub config: SearchConfig,

    /// Last search statistics.
    last_stats: SearchStatistics,
}

impl MotionSearcher {
    /// Create a new motion searcher.
    pub fn new(database: MotionDatabase, config: SearchConfig) -> Self {
        Self {
            database,
            config,
            last_stats: SearchStatistics::default(),
        }
    }

    /// Create with default config.
    pub fn with_default_config(database: MotionDatabase) -> Self {
        Self::new(database, SearchConfig::default())
    }

    /// Set the search configuration.
    pub fn set_config(&mut self, config: SearchConfig) {
        self.config = config;
    }

    /// Get the current configuration.
    pub fn config(&self) -> &SearchConfig {
        &self.config
    }

    /// Get the last search statistics.
    pub fn last_statistics(&self) -> &SearchStatistics {
        &self.last_stats
    }

    /// Search for the best matching frame.
    ///
    /// Returns the best match or None if no valid match is found.
    pub fn search(&mut self, query: &SearchQuery) -> Option<SearchResult> {
        let results = self.search_k_nearest(query, 1);
        results.into_iter().next()
    }

    /// Search for the K best matching frames.
    ///
    /// Returns up to K results sorted by cost (lowest first).
    pub fn search_k_nearest(&mut self, query: &SearchQuery, k: usize) -> Vec<SearchResult> {
        let start_time = Instant::now();
        let budget_ns = (self.config.budget_ms * 1_000_000.0) as u128;

        let mut stats = SearchStatistics::default();

        if self.database.is_empty() || k == 0 {
            stats.search_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
            self.last_stats = stats;
            return Vec::new();
        }

        // Step 1: Get candidates from KD-tree
        let kd_start = Instant::now();
        let query_features = self.query_to_features(query);
        let kd_candidates = self.database.kd_tree.find_k_nearest(
            &query_features,
            self.config.k_neighbors.max(k * 4),
        );
        stats.kd_tree_time_ms = kd_start.elapsed().as_secs_f32() * 1000.0;
        stats.candidates_total = kd_candidates.len();

        // Step 2: Filter by tags
        let tag_filtered: Vec<(u32, f32)> = kd_candidates
            .into_iter()
            .filter(|&(frame_idx, _)| {
                if let Some(frame) = self.database.frames.get(frame_idx as usize) {
                    self.check_tags(query, frame)
                } else {
                    false
                }
            })
            .collect();
        stats.candidates_after_tags = tag_filtered.len();

        // Step 3: Apply distance threshold pruning if configured
        let pruned: Vec<(u32, f32)> = if let Some(threshold) = self.config.distance_threshold {
            tag_filtered
                .into_iter()
                .filter(|&(_, dist)| dist <= threshold * threshold)
                .collect()
        } else {
            tag_filtered
        };
        stats.candidates_after_pruning = pruned.len();

        // Step 4: Evaluate full cost for candidates
        let eval_start = Instant::now();
        let mut results: Vec<SearchResult> = Vec::with_capacity(k);
        let mut best_cost = f32::INFINITY;
        let mut candidates_evaluated = 0;

        for (frame_idx, _kd_dist) in pruned {
            // Check time budget
            if self.config.early_termination && start_time.elapsed().as_nanos() > budget_ns {
                stats.budget_exceeded = true;
                break;
            }

            let frame = match self.database.frames.get(frame_idx as usize) {
                Some(f) => f,
                None => continue,
            };

            // Check min clip time
            if !self.check_min_clip_time(query, frame) {
                continue;
            }

            // Compute full cost
            let (cost, breakdown) = self.compute_cost_with_breakdown(query, frame);
            candidates_evaluated += 1;

            // Apply cost threshold
            if cost > self.config.cost_threshold {
                continue;
            }

            // Early termination: if we have k results and this is worse than all, skip
            if self.config.early_termination
                && results.len() >= k
                && cost >= best_cost * (1.0 - self.config.min_improvement)
            {
                continue;
            }

            let mut result = SearchResult::new(
                frame_idx as usize,
                frame.clip_index as usize,
                frame.time,
                cost,
            );
            result.cost_breakdown = Some(breakdown);

            // Insert in sorted order
            let insert_pos = results
                .iter()
                .position(|r| r.cost > cost)
                .unwrap_or(results.len());
            results.insert(insert_pos, result);

            // Keep only top k
            if results.len() > k {
                results.pop();
            }

            // Update best cost
            if cost < best_cost {
                best_cost = cost;
            }
        }

        stats.cost_eval_time_ms = eval_start.elapsed().as_secs_f32() * 1000.0;
        stats.candidates_evaluated = candidates_evaluated;
        stats.best_cost = best_cost;
        stats.search_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

        // Update results with final stats
        let search_time_ms = stats.search_time_ms;
        let budget_exceeded = stats.budget_exceeded;
        for result in &mut results {
            result.candidates_evaluated = candidates_evaluated;
            result.search_time_ms = search_time_ms;
            result.budget_exceeded = budget_exceeded;
        }

        self.last_stats = stats;
        results
    }

    /// Compute the cost of matching a frame to a query.
    pub fn compute_cost(&self, query: &SearchQuery, frame: &MotionFrame) -> f32 {
        self.compute_cost_with_breakdown(query, frame).0
    }

    /// Compute cost with detailed breakdown.
    pub fn compute_cost_with_breakdown(&self, query: &SearchQuery, frame: &MotionFrame) -> (f32, CostBreakdown) {
        let weights = &self.config.weights;
        let mut breakdown = CostBreakdown::default();

        // Pose cost: compare joint positions
        breakdown.pose_cost = self.compute_pose_cost(query, frame) * weights.pose_weight;

        // Trajectory cost
        breakdown.trajectory_cost = self.compute_trajectory_cost(query, frame) * weights.trajectory_weight;

        // Velocity cost
        breakdown.velocity_cost = self.compute_velocity_cost(query, frame) * weights.velocity_weight;

        // Transition cost (smoothness of velocity change)
        breakdown.transition_cost = self.compute_transition_cost(query, frame) * weights.transition_weight;

        // Foot contact cost
        breakdown.foot_cost = self.compute_foot_cost(query, frame)
            * (weights.foot_contact_weight + weights.foot_position_weight + weights.foot_velocity_weight);

        // Tag mismatch penalty
        if !self.check_tags_partial(query, frame) {
            breakdown.tag_penalty = weights.tag_mismatch_penalty;
        }

        // Stickiness bonus
        if let Some((current_clip, _)) = query.current_clip {
            if frame.clip_index as usize == current_clip {
                breakdown.stickiness_bonus = self.config.stickiness;
            }
        }

        let total = breakdown.total().max(0.0);
        (total, breakdown)
    }

    /// Convert query to feature vector for KD-tree search.
    fn query_to_features(&self, query: &SearchQuery) -> Vec<f32> {
        let mut features = Vec::with_capacity(self.database.feature_dims);

        // Add pose features
        for &val in &query.current_pose.pose {
            features.push(val);
            if features.len() >= self.database.feature_dims {
                break;
            }
        }

        // Add trajectory features
        if features.len() < self.database.feature_dims {
            let traj_flat = query.desired_trajectory.to_flat_vector();
            for val in traj_flat {
                features.push(val);
                if features.len() >= self.database.feature_dims {
                    break;
                }
            }
        }

        // Pad to feature_dims
        while features.len() < self.database.feature_dims {
            features.push(0.0);
        }

        features
    }

    /// Check if frame tags match query requirements.
    fn check_tags(&self, query: &SearchQuery, frame: &MotionFrame) -> bool {
        // Convert MotionTags to LocomotionTags check
        // The query uses MotionTags, the frame uses LocomotionTags
        // For now, we do a simple check based on locomotion style

        match query.required_tags.locomotion {
            crate::motion_features::LocomotionStyle::Idle => {
                frame.tags.contains(LocomotionTags::IDLE) || frame.tags.is_empty()
            }
            crate::motion_features::LocomotionStyle::Walk => {
                frame.tags.contains(LocomotionTags::WALK) || frame.tags.is_empty()
            }
            crate::motion_features::LocomotionStyle::Run => {
                frame.tags.contains(LocomotionTags::RUN) || frame.tags.is_empty()
            }
            crate::motion_features::LocomotionStyle::Sprint => {
                frame.tags.contains(LocomotionTags::SPRINT) || frame.tags.is_empty()
            }
            crate::motion_features::LocomotionStyle::Crouch => {
                frame.tags.contains(LocomotionTags::CROUCH) || frame.tags.is_empty()
            }
        }
    }

    /// Check partial tag match (for cost calculation).
    fn check_tags_partial(&self, query: &SearchQuery, frame: &MotionFrame) -> bool {
        self.check_tags(query, frame)
    }

    /// Check minimum clip time requirement.
    fn check_min_clip_time(&self, query: &SearchQuery, frame: &MotionFrame) -> bool {
        if let Some((current_clip, current_time)) = query.current_clip {
            if frame.clip_index as usize == current_clip {
                // Same clip: check if we've been in it long enough to switch
                if current_time < self.config.min_clip_time {
                    return false;
                }
            }
        }
        true
    }

    /// Compute pose cost between query and frame.
    fn compute_pose_cost(&self, query: &SearchQuery, frame: &MotionFrame) -> f32 {
        let query_pose = &query.current_pose.pose;
        let frame_pose = &frame.pose;

        let min_len = query_pose.len().min(frame_pose.joint_positions.len() * 3);
        if min_len == 0 {
            return 0.0;
        }

        let mut dist = 0.0;
        for i in 0..min_len {
            let query_val = query_pose.get(i).copied().unwrap_or(0.0);
            let frame_val = if i < frame_pose.joint_positions.len() * 3 {
                let joint_idx = i / 3;
                let component = i % 3;
                match component {
                    0 => frame_pose.joint_positions[joint_idx].x,
                    1 => frame_pose.joint_positions[joint_idx].y,
                    _ => frame_pose.joint_positions[joint_idx].z,
                }
            } else {
                0.0
            };
            let diff = query_val - frame_val;
            dist += diff * diff;
        }

        dist.sqrt()
    }

    /// Compute trajectory cost between query and frame.
    fn compute_trajectory_cost(&self, query: &SearchQuery, frame: &MotionFrame) -> f32 {
        let mut dist = 0.0;

        // Position distance
        for i in 0..TRAJECTORY_SAMPLES {
            let query_pos = query.desired_trajectory.future_positions[i];
            let frame_pos = frame.trajectory.future_positions[i];
            dist += (query_pos - frame_pos).length_squared();
        }

        // Facing distance
        for i in 0..TRAJECTORY_SAMPLES {
            let query_facing = query.desired_trajectory.future_facings[i];
            let frame_facing = frame.trajectory.future_facings[i];
            let angle_diff = (query_facing - frame_facing).abs();
            let wrapped = angle_diff.min(std::f32::consts::TAU - angle_diff);
            dist += wrapped * wrapped;
        }

        dist.sqrt()
    }

    /// Compute velocity cost between query and frame.
    fn compute_velocity_cost(&self, query: &SearchQuery, frame: &MotionFrame) -> f32 {
        let vel_diff = query.root_velocity - frame.root_velocity;
        let angular_diff = query.root_angular_velocity - frame.root_angular_velocity;
        (vel_diff.length_squared() + angular_diff * angular_diff).sqrt()
    }

    /// Compute transition cost (how smooth will the transition be).
    fn compute_transition_cost(&self, query: &SearchQuery, frame: &MotionFrame) -> f32 {
        // Compare pose velocities if available
        let query_pose = &query.current_pose.pose;
        let half_len = query_pose.len() / 2;

        if half_len == 0 || frame.pose.joint_velocities.is_empty() {
            return 0.0;
        }

        let mut dist = 0.0;
        let vel_count = half_len.min(frame.pose.joint_velocities.len() * 3);

        for i in 0..vel_count {
            let query_vel = query_pose.get(half_len + i).copied().unwrap_or(0.0);
            let joint_idx = i / 3;
            let component = i % 3;
            let frame_vel = if joint_idx < frame.pose.joint_velocities.len() {
                match component {
                    0 => frame.pose.joint_velocities[joint_idx].x,
                    1 => frame.pose.joint_velocities[joint_idx].y,
                    _ => frame.pose.joint_velocities[joint_idx].z,
                }
            } else {
                0.0
            };
            let diff = query_vel - frame_vel;
            dist += diff * diff;
        }

        dist.sqrt()
    }

    /// Compute foot contact cost.
    fn compute_foot_cost(&self, query: &SearchQuery, frame: &MotionFrame) -> f32 {
        let query_foot = &query.current_pose.foot;
        let frame_foot = &frame.foot_contacts;

        let min_feet = query_foot.foot_count().min(frame_foot.len());
        if min_feet == 0 {
            return 0.0;
        }

        let mut dist = 0.0;
        let weights = &self.config.weights;

        for i in 0..min_feet {
            // Contact state mismatch
            let query_planted = query_foot.contact_states.get(i).copied().unwrap_or(false);
            let frame_planted = frame_foot[i].is_planted;
            if query_planted != frame_planted {
                dist += weights.foot_contact_weight;
            }

            // Position distance
            let query_pos = query_foot.positions.get(i).copied().unwrap_or(Vec3::ZERO);
            let frame_pos = frame_foot[i].position;
            dist += (query_pos - frame_pos).length_squared() * weights.foot_position_weight;

            // Velocity distance
            let query_vel = query_foot.velocities.get(i).copied().unwrap_or(Vec3::ZERO);
            let frame_vel = frame_foot[i].velocity;
            dist += (query_vel - frame_vel).length_squared() * weights.foot_velocity_weight;
        }

        dist.sqrt()
    }

    /// Get the database.
    pub fn database(&self) -> &MotionDatabase {
        &self.database
    }

    /// Get mutable database reference.
    pub fn database_mut(&mut self) -> &mut MotionDatabase {
        &mut self.database
    }

    /// Rebuild the search index.
    pub fn rebuild_index(&mut self) {
        self.database.rebuild_index();
    }
}

// ---------------------------------------------------------------------------
// Pruning Functions
// ---------------------------------------------------------------------------

/// Prune candidates by distance threshold.
///
/// Returns indices of candidates within the threshold distance.
pub fn prune_by_distance(
    candidates: &[usize],
    threshold: f32,
    features: &[MotionFeatures],
    query: &MotionFeatures,
) -> Vec<usize> {
    let threshold_sq = threshold * threshold;
    let weights = FeatureWeights::default();

    candidates
        .iter()
        .filter(|&&idx| {
            if let Some(frame_features) = features.get(idx) {
                let dist_sq = compute_feature_distance_squared(query, frame_features, &weights);
                dist_sq <= threshold_sq
            } else {
                false
            }
        })
        .copied()
        .collect()
}

/// Prune candidates by tag requirements.
pub fn prune_by_tags(
    candidates: &[usize],
    frames: &[MotionFrame],
    required_tags: LocomotionTags,
    excluded_tags: LocomotionTags,
) -> Vec<usize> {
    candidates
        .iter()
        .filter(|&&idx| {
            if let Some(frame) = frames.get(idx) {
                frame.matches_tags(required_tags, excluded_tags)
            } else {
                false
            }
        })
        .copied()
        .collect()
}

/// Prune candidates by clip constraints.
pub fn prune_by_clip_constraints(
    candidates: &[usize],
    frames: &[MotionFrame],
    current_clip: Option<(usize, f32)>,
    min_clip_time: f32,
) -> Vec<usize> {
    candidates
        .iter()
        .filter(|&&idx| {
            if let Some(frame) = frames.get(idx) {
                if let Some((clip_id, time)) = current_clip {
                    // Allow switching if we've been in the clip long enough
                    if frame.clip_index as usize == clip_id && time < min_clip_time {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        })
        .copied()
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion_matching_db::{FootContact, MotionDatabaseBuilder, PoseFeature, TrajectoryFeature};

    // =====================================================================
    // Helper functions
    // =====================================================================

    fn create_test_database(frame_count: usize) -> MotionDatabase {
        let mut builder = MotionDatabaseBuilder::new()
            .with_name("test_db")
            .with_sample_rate(30.0);

        let mut frames = Vec::with_capacity(frame_count);
        for i in 0..frame_count {
            let mut frame = MotionFrame::from_clip(0, i as f32 / 30.0);
            frame.pose = PoseFeature::with_joint_count(4);
            frame.pose.joint_positions[0] = Vec3::new(i as f32 * 0.1, 0.0, 0.0);
            frame.trajectory = TrajectoryFeature::from_predictions(
                [
                    Vec3::new(i as f32 * 0.2, 0.0, i as f32 * 0.1),
                    Vec3::new(i as f32 * 0.4, 0.0, i as f32 * 0.2),
                    Vec3::new(i as f32 * 0.8, 0.0, i as f32 * 0.4),
                ],
                [0.0, 0.0, 0.0],
            );
            frame.tags = LocomotionTags::WALK | LocomotionTags::TERRAIN_FLAT;
            frame.foot_contacts = vec![FootContact::planted(Vec3::ZERO), FootContact::new()];
            frames.push(frame);
        }

        builder.add_frames("walk", frame_count as f32 / 30.0, frames, LocomotionTags::WALK);
        builder.build()
    }

    fn create_test_query() -> SearchQuery {
        let mut query = SearchQuery::new();
        query.current_pose.pose = vec![0.0; 24]; // 4 joints * 3 components * 2 (pos + vel)
        query.desired_trajectory = TrajectoryFeatureCompat {
            future_positions: [Vec3::X, Vec3::new(2.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0)],
            future_facings: [0.0, 0.0, 0.0],
        };
        query.required_tags = MotionTags::walk();
        query
    }

    // =====================================================================
    // SearchCostWeights tests
    // =====================================================================

    #[test]
    fn test_cost_weights_default() {
        let weights = SearchCostWeights::default();
        assert_eq!(weights.pose_weight, 1.0);
        assert_eq!(weights.trajectory_weight, 1.0);
        assert!(weights.tag_mismatch_penalty > 0.0);
    }

    #[test]
    fn test_cost_weights_trajectory_focused() {
        let weights = SearchCostWeights::trajectory_focused();
        assert!(weights.trajectory_weight > weights.pose_weight);
    }

    #[test]
    fn test_cost_weights_pose_focused() {
        let weights = SearchCostWeights::pose_focused();
        assert!(weights.pose_weight > weights.trajectory_weight);
    }

    #[test]
    fn test_cost_weights_transition_focused() {
        let weights = SearchCostWeights::transition_focused();
        assert!(weights.transition_weight > weights.pose_weight);
        assert!(weights.foot_contact_weight > weights.foot_position_weight);
    }

    #[test]
    fn test_cost_weights_uniform() {
        let weights = SearchCostWeights::uniform();
        assert_eq!(weights.pose_weight, weights.trajectory_weight);
        assert_eq!(weights.velocity_weight, weights.transition_weight);
    }

    // =====================================================================
    // SearchConfig tests
    // =====================================================================

    #[test]
    fn test_search_config_default() {
        let config = SearchConfig::default();
        assert!(config.budget_ms > 0.0);
        assert!(config.max_candidates > 0);
        assert!(config.k_neighbors > 0);
    }

    #[test]
    fn test_search_config_fast() {
        let config = SearchConfig::fast();
        assert!(config.budget_ms < SearchConfig::quality().budget_ms);
        assert!(config.max_candidates < SearchConfig::quality().max_candidates);
    }

    #[test]
    fn test_search_config_quality() {
        let config = SearchConfig::quality();
        assert!(config.budget_ms > SearchConfig::fast().budget_ms);
        assert!(config.max_candidates > SearchConfig::fast().max_candidates);
    }

    #[test]
    fn test_search_config_with_weights() {
        let config = SearchConfig::default()
            .with_weights(SearchCostWeights::trajectory_focused());
        assert!(config.weights.trajectory_weight > config.weights.pose_weight);
    }

    #[test]
    fn test_search_config_with_cost_threshold() {
        let config = SearchConfig::default().with_cost_threshold(50.0);
        assert_eq!(config.cost_threshold, 50.0);
    }

    #[test]
    fn test_search_config_with_min_clip_time() {
        let config = SearchConfig::default().with_min_clip_time(0.5);
        assert_eq!(config.min_clip_time, 0.5);
    }

    #[test]
    fn test_search_config_with_stickiness() {
        let config = SearchConfig::default().with_stickiness(0.1);
        assert_eq!(config.stickiness, 0.1);
    }

    #[test]
    fn test_search_config_with_budget() {
        let config = SearchConfig::default().with_budget_ms(10.0);
        assert_eq!(config.budget_ms, 10.0);
    }

    // =====================================================================
    // TrajectoryFeatureCompat tests
    // =====================================================================

    #[test]
    fn test_trajectory_feature_compat_default() {
        let traj = TrajectoryFeatureCompat::default();
        assert_eq!(traj.future_positions, [Vec3::ZERO; 3]);
        assert_eq!(traj.future_facings, [0.0; 3]);
    }

    #[test]
    fn test_trajectory_feature_compat_to_flat_vector() {
        let traj = TrajectoryFeatureCompat {
            future_positions: [Vec3::X, Vec3::Y, Vec3::Z],
            future_facings: [1.0, 2.0, 3.0],
        };
        let flat = traj.to_flat_vector();
        assert_eq!(flat.len(), 12);
        assert_eq!(flat[0], 1.0); // X of first position
        assert_eq!(flat[9], 1.0); // First facing
    }

    #[test]
    fn test_trajectory_feature_compat_from_flat_vector() {
        let flat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 1.5];
        let traj = TrajectoryFeatureCompat::from_flat_vector(&flat);
        assert_eq!(traj.future_positions[0], Vec3::X);
        assert_eq!(traj.future_facings[0], 0.5);
    }

    #[test]
    fn test_trajectory_feature_compat_distance_squared() {
        let a = TrajectoryFeatureCompat::default();
        let b = TrajectoryFeatureCompat {
            future_positions: [Vec3::X, Vec3::ZERO, Vec3::ZERO],
            future_facings: [0.0, 0.0, 0.0],
        };
        let dist = a.distance_squared(&b, 1.0, 0.0);
        assert_eq!(dist, 1.0);
    }

    // =====================================================================
    // SearchQuery tests
    // =====================================================================

    #[test]
    fn test_search_query_new() {
        let query = SearchQuery::new();
        assert!(query.current_clip.is_none());
        assert_eq!(query.root_velocity, Vec3::ZERO);
    }

    #[test]
    fn test_search_query_with_pose() {
        let mut pose = MotionFeatures::default();
        pose.pose = vec![1.0, 2.0, 3.0];

        let query = SearchQuery::new().with_pose(pose);
        assert_eq!(query.current_pose.pose, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_search_query_with_trajectory() {
        let traj = TrajectoryFeatureCompat {
            future_positions: [Vec3::X, Vec3::Y, Vec3::Z],
            future_facings: [0.0, 0.0, 0.0],
        };
        let query = SearchQuery::new().with_trajectory(traj);
        assert_eq!(query.desired_trajectory.future_positions[0], Vec3::X);
    }

    #[test]
    fn test_search_query_with_current_clip() {
        let query = SearchQuery::new().with_current_clip(5, 0.3);
        assert_eq!(query.current_clip, Some((5, 0.3)));
    }

    #[test]
    fn test_search_query_with_required_tags() {
        let query = SearchQuery::new().with_required_tags(MotionTags::run());
        assert_eq!(query.required_tags.locomotion, crate::motion_features::LocomotionStyle::Run);
    }

    #[test]
    fn test_search_query_with_root_velocity() {
        let query = SearchQuery::new().with_root_velocity(Vec3::new(1.0, 0.0, 2.0));
        assert_eq!(query.root_velocity, Vec3::new(1.0, 0.0, 2.0));
    }

    // =====================================================================
    // SearchResult tests
    // =====================================================================

    #[test]
    fn test_search_result_new() {
        let result = SearchResult::new(10, 1, 0.5, 5.0);
        assert_eq!(result.frame_index, 10);
        assert_eq!(result.clip_index, 1);
        assert_eq!(result.time, 0.5);
        assert_eq!(result.cost, 5.0);
    }

    #[test]
    fn test_search_result_is_valid() {
        let mut result = SearchResult::new(0, 0, 0.0, 50.0);
        assert!(result.is_valid(100.0));
        assert!(!result.is_valid(40.0));

        result.budget_exceeded = true;
        assert!(!result.is_valid(100.0));
    }

    // =====================================================================
    // CostBreakdown tests
    // =====================================================================

    #[test]
    fn test_cost_breakdown_default() {
        let breakdown = CostBreakdown::default();
        assert_eq!(breakdown.total(), 0.0);
    }

    #[test]
    fn test_cost_breakdown_total() {
        let breakdown = CostBreakdown {
            pose_cost: 1.0,
            trajectory_cost: 2.0,
            velocity_cost: 0.5,
            transition_cost: 0.3,
            foot_cost: 0.2,
            tag_penalty: 0.0,
            stickiness_bonus: 0.5,
        };
        assert!((breakdown.total() - 3.5).abs() < 0.001);
    }

    #[test]
    fn test_cost_breakdown_with_tag_penalty() {
        let breakdown = CostBreakdown {
            pose_cost: 1.0,
            tag_penalty: 10.0,
            ..Default::default()
        };
        assert_eq!(breakdown.total(), 11.0);
    }

    #[test]
    fn test_cost_breakdown_stickiness_reduces_cost() {
        let breakdown = CostBreakdown {
            pose_cost: 1.0,
            stickiness_bonus: 0.5,
            ..Default::default()
        };
        assert_eq!(breakdown.total(), 0.5);
    }

    // =====================================================================
    // MotionSearcher basic tests
    // =====================================================================

    #[test]
    fn test_searcher_new() {
        let db = create_test_database(10);
        let config = SearchConfig::default();
        let searcher = MotionSearcher::new(db, config);
        assert!(!searcher.database.is_empty());
    }

    #[test]
    fn test_searcher_with_default_config() {
        let db = create_test_database(10);
        let searcher = MotionSearcher::with_default_config(db);
        assert_eq!(searcher.config.budget_ms, DEFAULT_BUDGET_MS);
    }

    #[test]
    fn test_searcher_set_config() {
        let db = create_test_database(10);
        let mut searcher = MotionSearcher::with_default_config(db);
        searcher.set_config(SearchConfig::fast());
        assert!(searcher.config.budget_ms < DEFAULT_BUDGET_MS);
    }

    #[test]
    fn test_searcher_empty_database() {
        let db = MotionDatabase::new();
        let mut searcher = MotionSearcher::with_default_config(db);
        let query = create_test_query();
        let result = searcher.search(&query);
        assert!(result.is_none());
    }

    #[test]
    fn test_searcher_search_finds_result() {
        let db = create_test_database(20);
        let mut searcher = MotionSearcher::with_default_config(db);
        let query = create_test_query();
        let result = searcher.search(&query);
        assert!(result.is_some());
    }

    #[test]
    fn test_searcher_search_k_nearest() {
        let db = create_test_database(50);
        let mut searcher = MotionSearcher::with_default_config(db);
        let query = create_test_query();
        let results = searcher.search_k_nearest(&query, 5);
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_searcher_search_k_nearest_sorted() {
        let db = create_test_database(50);
        let mut searcher = MotionSearcher::with_default_config(db);
        let query = create_test_query();
        let results = searcher.search_k_nearest(&query, 10);

        // Verify results are sorted by cost
        for i in 1..results.len() {
            assert!(results[i - 1].cost <= results[i].cost);
        }
    }

    #[test]
    fn test_searcher_compute_cost() {
        let db = create_test_database(10);
        let searcher = MotionSearcher::with_default_config(db);
        let query = create_test_query();
        let frame = &searcher.database.frames[0];
        let cost = searcher.compute_cost(&query, frame);
        assert!(cost >= 0.0);
        assert!(cost.is_finite());
    }

    // =====================================================================
    // Cost weight effect tests
    // =====================================================================

    #[test]
    fn test_cost_weight_affects_pose() {
        let db = create_test_database(10);
        let config1 = SearchConfig::default().with_weights(SearchCostWeights {
            pose_weight: 0.0,
            ..Default::default()
        });
        let config2 = SearchConfig::default().with_weights(SearchCostWeights {
            pose_weight: 10.0,
            ..Default::default()
        });

        let searcher1 = MotionSearcher::new(db.clone(), config1);
        let searcher2 = MotionSearcher::new(db, config2);
        let query = create_test_query();

        let cost1 = searcher1.compute_cost(&query, &searcher1.database.frames[5]);
        let cost2 = searcher2.compute_cost(&query, &searcher2.database.frames[5]);

        // Higher pose weight should result in higher cost
        assert!(cost2 >= cost1);
    }

    #[test]
    fn test_cost_weight_affects_trajectory() {
        let db = create_test_database(10);
        let config_low = SearchConfig::default().with_weights(SearchCostWeights {
            trajectory_weight: 0.1,
            ..Default::default()
        });
        let config_high = SearchConfig::default().with_weights(SearchCostWeights {
            trajectory_weight: 5.0,
            ..Default::default()
        });

        let searcher_low = MotionSearcher::new(db.clone(), config_low);
        let searcher_high = MotionSearcher::new(db, config_high);
        let query = create_test_query();

        let cost_low = searcher_low.compute_cost(&query, &searcher_low.database.frames[0]);
        let cost_high = searcher_high.compute_cost(&query, &searcher_high.database.frames[0]);

        assert!(cost_high >= cost_low);
    }

    // =====================================================================
    // Tag filtering tests
    // =====================================================================

    #[test]
    fn test_searcher_filters_by_tags() {
        let mut builder = MotionDatabaseBuilder::new();

        // Add walk frames
        let mut walk_frames: Vec<MotionFrame> = (0..10)
            .map(|i| {
                let mut frame = MotionFrame::from_clip(0, i as f32 / 30.0);
                frame.tags = LocomotionTags::WALK;
                frame.pose = PoseFeature::with_joint_count(4);
                frame.foot_contacts = vec![FootContact::new(), FootContact::new()];
                frame
            })
            .collect();
        builder.add_frames("walk", 10.0 / 30.0, walk_frames, LocomotionTags::WALK);

        // Add run frames
        let mut run_frames: Vec<MotionFrame> = (0..10)
            .map(|i| {
                let mut frame = MotionFrame::from_clip(1, i as f32 / 30.0);
                frame.tags = LocomotionTags::RUN;
                frame.pose = PoseFeature::with_joint_count(4);
                frame.foot_contacts = vec![FootContact::new(), FootContact::new()];
                frame
            })
            .collect();
        builder.add_frames("run", 10.0 / 30.0, run_frames, LocomotionTags::RUN);

        let db = builder.build();
        let mut searcher = MotionSearcher::with_default_config(db);

        // Query for run
        let mut query = SearchQuery::new();
        query.required_tags = MotionTags::run();

        let results = searcher.search_k_nearest(&query, 5);

        // All results should be from run clip (clip_index 1)
        for result in &results {
            assert_eq!(result.clip_index, 1, "Expected run clip (1), got {}", result.clip_index);
        }
    }

    // =====================================================================
    // Budget enforcement tests
    // =====================================================================

    #[test]
    fn test_searcher_respects_budget() {
        let db = create_test_database(1000);
        let config = SearchConfig::default()
            .with_budget_ms(0.01); // Very tight budget
        let mut searcher = MotionSearcher::new(db, config);
        let query = create_test_query();

        let _result = searcher.search(&query);
        // Either completes quickly or marks budget exceeded
        let stats = searcher.last_statistics();
        assert!(stats.search_time_ms < 1.0 || stats.budget_exceeded);
    }

    #[test]
    fn test_searcher_statistics_populated() {
        let db = create_test_database(50);
        let mut searcher = MotionSearcher::with_default_config(db);
        let query = create_test_query();

        let _result = searcher.search(&query);
        let stats = searcher.last_statistics();

        assert!(stats.candidates_total > 0 || searcher.database.is_empty());
        assert!(stats.search_time_ms >= 0.0);
    }

    // =====================================================================
    // Stickiness behavior tests
    // =====================================================================

    #[test]
    fn test_searcher_stickiness_bonus() {
        let db = create_test_database(20);
        let config = SearchConfig::default().with_stickiness(1.0);
        let searcher = MotionSearcher::new(db, config);

        let query_same_clip = SearchQuery::new().with_current_clip(0, 0.5);
        let query_diff_clip = SearchQuery::new().with_current_clip(99, 0.5);

        let frame = &searcher.database.frames[0]; // Clip index 0

        let (cost_same, breakdown_same) = searcher.compute_cost_with_breakdown(&query_same_clip, frame);
        let (cost_diff, breakdown_diff) = searcher.compute_cost_with_breakdown(&query_diff_clip, frame);

        // Stickiness bonus should reduce cost when in same clip
        assert!(breakdown_same.stickiness_bonus > 0.0);
        assert_eq!(breakdown_diff.stickiness_bonus, 0.0);
        assert!(cost_same < cost_diff);
    }

    // =====================================================================
    // Min clip time enforcement tests
    // =====================================================================

    #[test]
    fn test_searcher_min_clip_time() {
        let db = create_test_database(20);
        let config = SearchConfig::default().with_min_clip_time(1.0);
        let mut searcher = MotionSearcher::new(db, config);

        // Query from clip 0 with very short time (should prevent staying in clip 0)
        let query = SearchQuery::new().with_current_clip(0, 0.05);

        // We don't have multiple clips in our test database, but the logic should work
        let _result = searcher.search(&query);
        // Just ensure it doesn't crash with min_clip_time enforcement
    }

    // =====================================================================
    // K-nearest search tests
    // =====================================================================

    #[test]
    fn test_search_k_returns_correct_count() {
        let db = create_test_database(100);
        let mut searcher = MotionSearcher::with_default_config(db);
        let query = create_test_query();

        for k in [1, 5, 10, 20] {
            let results = searcher.search_k_nearest(&query, k);
            assert!(results.len() <= k);
        }
    }

    #[test]
    fn test_search_k_zero_returns_empty() {
        let db = create_test_database(50);
        let mut searcher = MotionSearcher::with_default_config(db);
        let query = create_test_query();

        let results = searcher.search_k_nearest(&query, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_k_larger_than_database() {
        let db = create_test_database(5);
        let mut searcher = MotionSearcher::with_default_config(db);
        let query = create_test_query();

        let results = searcher.search_k_nearest(&query, 100);
        assert!(results.len() <= 5);
    }

    // =====================================================================
    // Distance pruning tests
    // =====================================================================

    #[test]
    fn test_prune_by_distance_empty() {
        let candidates: Vec<usize> = vec![];
        let features: Vec<MotionFeatures> = vec![];
        let query = MotionFeatures::default();

        let pruned = prune_by_distance(&candidates, 1.0, &features, &query);
        assert!(pruned.is_empty());
    }

    #[test]
    fn test_prune_by_distance_all_pass() {
        let mut f1 = MotionFeatures::default();
        f1.pose = vec![0.0, 0.0, 0.0];
        let mut f2 = MotionFeatures::default();
        f2.pose = vec![0.1, 0.0, 0.0];

        let features = vec![f1.clone(), f2];
        let candidates = vec![0, 1];
        let query = MotionFeatures::default();

        let pruned = prune_by_distance(&candidates, 10.0, &features, &query);
        assert_eq!(pruned.len(), 2);
    }

    #[test]
    fn test_prune_by_distance_some_fail() {
        let mut f1 = MotionFeatures::default();
        f1.pose = vec![0.0, 0.0, 0.0];
        let mut f2 = MotionFeatures::default();
        f2.pose = vec![10.0, 0.0, 0.0]; // Far away

        let features = vec![f1, f2];
        let candidates = vec![0, 1];
        let mut query = MotionFeatures::default();
        query.pose = vec![0.0, 0.0, 0.0];

        let pruned = prune_by_distance(&candidates, 1.0, &features, &query);
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0], 0);
    }

    // =====================================================================
    // Tag pruning tests
    // =====================================================================

    #[test]
    fn test_prune_by_tags_empty() {
        let candidates: Vec<usize> = vec![];
        let frames: Vec<MotionFrame> = vec![];

        let pruned = prune_by_tags(
            &candidates,
            &frames,
            LocomotionTags::WALK,
            LocomotionTags::empty(),
        );
        assert!(pruned.is_empty());
    }

    #[test]
    fn test_prune_by_tags_filters_correctly() {
        let mut walk_frame = MotionFrame::new();
        walk_frame.tags = LocomotionTags::WALK;

        let mut run_frame = MotionFrame::new();
        run_frame.tags = LocomotionTags::RUN;

        let frames = vec![walk_frame, run_frame];
        let candidates = vec![0, 1];

        let pruned = prune_by_tags(
            &candidates,
            &frames,
            LocomotionTags::WALK,
            LocomotionTags::empty(),
        );
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0], 0);
    }

    #[test]
    fn test_prune_by_tags_excludes_correctly() {
        let mut frame1 = MotionFrame::new();
        frame1.tags = LocomotionTags::WALK | LocomotionTags::COMBAT;

        let mut frame2 = MotionFrame::new();
        frame2.tags = LocomotionTags::WALK;

        let frames = vec![frame1, frame2];
        let candidates = vec![0, 1];

        let pruned = prune_by_tags(
            &candidates,
            &frames,
            LocomotionTags::WALK,
            LocomotionTags::COMBAT,
        );
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0], 1);
    }

    // =====================================================================
    // Clip constraint pruning tests
    // =====================================================================

    #[test]
    fn test_prune_by_clip_constraints_no_current() {
        let frame = MotionFrame::from_clip(0, 0.5);
        let frames = vec![frame];
        let candidates = vec![0];

        let pruned = prune_by_clip_constraints(&candidates, &frames, None, 0.1);
        assert_eq!(pruned.len(), 1);
    }

    #[test]
    fn test_prune_by_clip_constraints_same_clip_short_time() {
        let frame = MotionFrame::from_clip(0, 0.5);
        let frames = vec![frame];
        let candidates = vec![0];

        // Current clip is 0 with time 0.05 (less than min_clip_time 0.1)
        let pruned = prune_by_clip_constraints(&candidates, &frames, Some((0, 0.05)), 0.1);
        assert!(pruned.is_empty());
    }

    #[test]
    fn test_prune_by_clip_constraints_same_clip_long_time() {
        let frame = MotionFrame::from_clip(0, 0.5);
        let frames = vec![frame];
        let candidates = vec![0];

        // Current clip is 0 with time 0.5 (more than min_clip_time 0.1)
        let pruned = prune_by_clip_constraints(&candidates, &frames, Some((0, 0.5)), 0.1);
        assert_eq!(pruned.len(), 1);
    }

    #[test]
    fn test_prune_by_clip_constraints_different_clip() {
        let frame = MotionFrame::from_clip(0, 0.5);
        let frames = vec![frame];
        let candidates = vec![0];

        // Current clip is 99 (different), should allow switching regardless of time
        let pruned = prune_by_clip_constraints(&candidates, &frames, Some((99, 0.01)), 1.0);
        assert_eq!(pruned.len(), 1);
    }

    // =====================================================================
    // Edge case tests
    // =====================================================================

    #[test]
    fn test_searcher_empty_query_pose() {
        let db = create_test_database(10);
        let mut searcher = MotionSearcher::with_default_config(db);
        let query = SearchQuery::new(); // Empty pose

        let result = searcher.search(&query);
        // Should still find a result
        assert!(result.is_some());
    }

    #[test]
    fn test_searcher_very_high_cost_threshold() {
        let db = create_test_database(10);
        let config = SearchConfig::default().with_cost_threshold(f32::INFINITY);
        let mut searcher = MotionSearcher::new(db, config);
        let query = create_test_query();

        let result = searcher.search(&query);
        assert!(result.is_some());
    }

    #[test]
    fn test_searcher_very_low_cost_threshold() {
        let db = create_test_database(10);
        let config = SearchConfig::default().with_cost_threshold(0.0);
        let mut searcher = MotionSearcher::new(db, config);
        let query = create_test_query();

        let results = searcher.search_k_nearest(&query, 10);
        // All results should be filtered out
        assert!(results.is_empty());
    }

    #[test]
    fn test_searcher_database_access() {
        let db = create_test_database(10);
        let mut searcher = MotionSearcher::with_default_config(db);

        assert_eq!(searcher.database().frame_count(), 10);

        searcher.database_mut().name = "modified".to_string();
        assert_eq!(searcher.database().name, "modified");
    }

    #[test]
    fn test_searcher_rebuild_index() {
        let db = create_test_database(10);
        let mut searcher = MotionSearcher::with_default_config(db);

        // Add a frame
        searcher.database_mut().frames.push(MotionFrame::from_clip(99, 0.0));

        // Rebuild index
        searcher.rebuild_index();

        // Should work with the new frame
        assert_eq!(searcher.database().frame_count(), 11);
    }

    // =====================================================================
    // Cost breakdown tests
    // =====================================================================

    #[test]
    fn test_cost_breakdown_components() {
        let db = create_test_database(10);
        let searcher = MotionSearcher::with_default_config(db);
        let query = create_test_query();
        let frame = &searcher.database.frames[5];

        let (cost, breakdown) = searcher.compute_cost_with_breakdown(&query, frame);

        // All components should be non-negative
        assert!(breakdown.pose_cost >= 0.0);
        assert!(breakdown.trajectory_cost >= 0.0);
        assert!(breakdown.velocity_cost >= 0.0);
        assert!(breakdown.transition_cost >= 0.0);
        assert!(breakdown.foot_cost >= 0.0);
        assert!(breakdown.tag_penalty >= 0.0);
        assert!(breakdown.stickiness_bonus >= 0.0);

        // Total should match
        assert!((cost - breakdown.total()).abs() < 0.001);
    }

    // =====================================================================
    // Performance tests
    // =====================================================================

    #[test]
    fn test_search_performance_large_database() {
        let db = create_test_database(500);
        let config = SearchConfig::balanced();
        let mut searcher = MotionSearcher::new(db, config);
        let query = create_test_query();

        let start = Instant::now();
        let _result = searcher.search(&query);
        let elapsed = start.elapsed();

        // Should complete in reasonable time
        assert!(elapsed.as_millis() < 100);
    }

    // =====================================================================
    // Integration tests
    // =====================================================================

    #[test]
    fn test_full_search_workflow() {
        // Build database
        let db = create_test_database(100);
        let config = SearchConfig::balanced()
            .with_weights(SearchCostWeights::trajectory_focused())
            .with_cost_threshold(500.0)
            .with_stickiness(0.1);

        let mut searcher = MotionSearcher::new(db, config);

        // Create query
        let query = SearchQuery::new()
            .with_pose(MotionFeatures {
                pose: vec![0.0; 24],
                trajectory: vec![0.0; 12],
                foot: FootFeatures::with_count(2),
                ..Default::default()
            })
            .with_trajectory(TrajectoryFeatureCompat {
                future_positions: [Vec3::X, Vec3::new(2.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0)],
                future_facings: [0.0, 0.0, 0.0],
            })
            .with_required_tags(MotionTags::walk())
            .with_current_clip(0, 0.5);

        // Search
        let results = searcher.search_k_nearest(&query, 5);

        // Verify results
        assert!(!results.is_empty());
        for (i, result) in results.iter().enumerate() {
            // Sorted by cost
            if i > 0 {
                assert!(result.cost >= results[i - 1].cost);
            }
            // Has breakdown
            assert!(result.cost_breakdown.is_some());
            // Within threshold
            assert!(result.cost <= 500.0);
        }

        // Check statistics
        let stats = searcher.last_statistics();
        assert!(stats.candidates_evaluated > 0);
        assert!(stats.search_time_ms > 0.0);
    }
}
