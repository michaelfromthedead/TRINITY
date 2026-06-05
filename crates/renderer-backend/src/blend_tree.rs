//! Blend tree system for parametric animation blending (T-AN-5.3).
//!
//! This module provides blend tree structures for animation blending:
//!
//! - **1D Blend Tree**: Single parameter controls blend between N clips
//! - **2D Blend Tree**: Two parameters with triangulation-based interpolation
//! - **Directional Blend Space**: Radial interpolation for direction+speed
//! - **Additive Blend Space**: Base pose + weighted additive overlays
//!
//! # Architecture
//!
//! ```text
//! BlendTree1D
//! +-- parameter: String (e.g., "speed")
//! +-- clips: Vec<(threshold, clip_index)>
//! +-- Output: Vec<(clip_index, weight)>
//!
//! BlendTree2D
//! +-- param_x, param_y: String (e.g., "velocity_x", "velocity_z")
//! +-- clips: Vec<(x, y, clip_index)>
//! +-- triangles: Vec<[usize; 3]> (Delaunay triangulation)
//! +-- Output: Vec<(clip_index, weight)> (barycentric weights)
//!
//! DirectionalBlendSpace
//! +-- direction_param, speed_param: String
//! +-- clips: Vec<(angle_radians, speed, clip_index)>
//! +-- Output: Vec<(clip_index, weight)> (radial interpolation)
//!
//! AdditiveBlendSpace
//! +-- base_clip: usize
//! +-- additive_clips: Vec<(threshold, clip_index)>
//! +-- additive_param: String
//! +-- bone_mask: Option<Vec<bool>>
//! +-- Output: (base, Vec<(clip_index, weight)>)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::blend_tree::{BlendTree1D, BlendTree2D, DirectionalBlendSpace};
//!
//! // 1D blend tree for locomotion speed
//! let mut locomotion = BlendTree1D::new("speed");
//! locomotion.add_clip(0.0, 0);  // idle at speed 0
//! locomotion.add_clip(2.0, 1);  // walk at speed 2
//! locomotion.add_clip(6.0, 2);  // run at speed 6
//!
//! // Evaluate at speed 4 -> blend between walk and run
//! let weights = locomotion.evaluate(4.0);
//! // weights = [(1, 0.5), (2, 0.5)]
//!
//! // 2D blend tree for directional movement
//! let mut directional = BlendTree2D::new("velocity_x", "velocity_z");
//! directional.add_clip(0.0, 0.0, 0);   // idle at center
//! directional.add_clip(1.0, 0.0, 1);   // strafe right
//! directional.add_clip(-1.0, 0.0, 2);  // strafe left
//! directional.add_clip(0.0, 1.0, 3);   // forward
//! directional.add_clip(0.0, -1.0, 4);  // backward
//! directional.triangulate();
//!
//! let weights = directional.evaluate(0.5, 0.5);  // diagonal blend
//! ```

use serde::{Deserialize, Serialize};
use std::f32::consts::{PI, TAU};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Epsilon for floating-point comparisons.
pub const BLEND_EPSILON: f32 = 1e-6;

/// Maximum clips per blend tree.
pub const MAX_BLEND_CLIPS: usize = 32;

/// Maximum triangles for 2D blend space.
pub const MAX_TRIANGLES: usize = 128;

/// Default angle tolerance for directional matching (radians).
pub const DEFAULT_ANGLE_TOLERANCE: f32 = PI / 4.0;

// ---------------------------------------------------------------------------
// BlendTreeError
// ---------------------------------------------------------------------------

/// Errors that can occur during blend tree operations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BlendTreeError {
    /// Not enough clips for operation.
    InsufficientClips { required: usize, actual: usize },

    /// Invalid parameter value.
    InvalidParameter { name: String, value: f32 },

    /// Triangulation failed.
    TriangulationFailed { reason: String },

    /// Clip index out of bounds.
    ClipIndexOutOfBounds { index: usize, max: usize },

    /// Duplicate threshold value.
    DuplicateThreshold { threshold: f32 },
}

impl std::fmt::Display for BlendTreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientClips { required, actual } => {
                write!(f, "insufficient clips: need {}, have {}", required, actual)
            }
            Self::InvalidParameter { name, value } => {
                write!(f, "invalid parameter '{}': {}", name, value)
            }
            Self::TriangulationFailed { reason } => {
                write!(f, "triangulation failed: {}", reason)
            }
            Self::ClipIndexOutOfBounds { index, max } => {
                write!(f, "clip index {} out of bounds (max {})", index, max)
            }
            Self::DuplicateThreshold { threshold } => {
                write!(f, "duplicate threshold: {}", threshold)
            }
        }
    }
}

impl std::error::Error for BlendTreeError {}

// ---------------------------------------------------------------------------
// BlendTree1D
// ---------------------------------------------------------------------------

/// 1D blend tree with single parameter control.
///
/// Clips are placed at threshold values along a single axis. The parameter
/// value determines which clips to blend based on linear interpolation
/// between the two nearest thresholds.
///
/// # Example
///
/// ```ignore
/// let mut tree = BlendTree1D::new("speed");
/// tree.add_clip(0.0, 0);  // idle
/// tree.add_clip(3.0, 1);  // walk
/// tree.add_clip(8.0, 2);  // run
///
/// // At speed=5, blend between walk (1) and run (2)
/// let weights = tree.evaluate(5.0);
/// assert_eq!(weights, vec![(1, 0.6), (2, 0.4)]);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlendTree1D {
    /// Name of the parameter that drives this blend tree.
    pub parameter: String,

    /// Clips at their threshold positions.
    /// Tuple: (threshold, clip_index)
    pub clips: Vec<(f32, usize)>,

    /// Whether clips are sorted by threshold.
    sorted: bool,
}

impl BlendTree1D {
    /// Create a new 1D blend tree.
    ///
    /// # Arguments
    ///
    /// * `parameter` - Name of the driving parameter
    pub fn new(parameter: impl Into<String>) -> Self {
        Self {
            parameter: parameter.into(),
            clips: Vec::new(),
            sorted: true,
        }
    }

    /// Create a 1D blend tree with pre-allocated capacity.
    pub fn with_capacity(parameter: impl Into<String>, capacity: usize) -> Self {
        Self {
            parameter: parameter.into(),
            clips: Vec::with_capacity(capacity),
            sorted: true,
        }
    }

    /// Add a clip at a threshold position.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Parameter value at which this clip is at full weight
    /// * `clip_index` - Index of the animation clip
    pub fn add_clip(&mut self, threshold: f32, clip_index: usize) {
        self.clips.push((threshold, clip_index));
        self.sorted = false;
    }

    /// Get the number of clips in this blend tree.
    #[inline]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Get the parameter name.
    #[inline]
    pub fn parameter_name(&self) -> &str {
        &self.parameter
    }

    /// Get the range of threshold values.
    ///
    /// Returns (min, max) or None if empty.
    pub fn threshold_range(&self) -> Option<(f32, f32)> {
        if self.clips.is_empty() {
            return None;
        }

        let mut min = f32::MAX;
        let mut max = f32::MIN;

        for (threshold, _) in &self.clips {
            min = min.min(*threshold);
            max = max.max(*threshold);
        }

        Some((min, max))
    }

    /// Sort clips by threshold (required for evaluation).
    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.clips
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            self.sorted = true;
        }
    }

    /// Evaluate the blend tree at a parameter value.
    ///
    /// Returns a list of (clip_index, weight) pairs. Weights sum to 1.0.
    ///
    /// # Arguments
    ///
    /// * `param_value` - Current parameter value
    ///
    /// # Returns
    ///
    /// Vector of (clip_index, weight) tuples for blending.
    pub fn evaluate(&mut self, param_value: f32) -> Vec<(usize, f32)> {
        if self.clips.is_empty() {
            return Vec::new();
        }

        self.ensure_sorted();

        // Single clip - return at full weight
        if self.clips.len() == 1 {
            return vec![(self.clips[0].1, 1.0)];
        }

        // Check if we're exactly at a threshold
        for (threshold, clip_idx) in &self.clips {
            if (param_value - *threshold).abs() < BLEND_EPSILON {
                return vec![(*clip_idx, 1.0)];
            }
        }

        // Find the two clips to blend between
        let (lower_idx, upper_idx) = self.find_bracket(param_value);

        let (lower_threshold, lower_clip) = self.clips[lower_idx];
        let (upper_threshold, upper_clip) = self.clips[upper_idx];

        // Same clip (at exact threshold or clamped)
        if lower_idx == upper_idx {
            return vec![(lower_clip, 1.0)];
        }

        // Calculate blend weight
        let range = upper_threshold - lower_threshold;
        if range.abs() < BLEND_EPSILON {
            // Thresholds are the same, use lower clip
            return vec![(lower_clip, 1.0)];
        }

        let t = ((param_value - lower_threshold) / range).clamp(0.0, 1.0);

        // Return both clips with their weights
        if lower_clip == upper_clip {
            vec![(lower_clip, 1.0)]
        } else {
            vec![(lower_clip, 1.0 - t), (upper_clip, t)]
        }
    }

    /// Find the bracket indices for a parameter value.
    ///
    /// Returns (lower_index, upper_index) where the parameter falls between.
    fn find_bracket(&self, param_value: f32) -> (usize, usize) {
        debug_assert!(self.sorted, "clips must be sorted before evaluation");
        debug_assert!(!self.clips.is_empty(), "clips must not be empty");

        // Below minimum
        if param_value <= self.clips[0].0 {
            return (0, 0);
        }

        // Above maximum
        let last = self.clips.len() - 1;
        if param_value >= self.clips[last].0 {
            return (last, last);
        }

        // Binary search for bracket
        for i in 0..last {
            if param_value >= self.clips[i].0 && param_value <= self.clips[i + 1].0 {
                return (i, i + 1);
            }
        }

        // Fallback (shouldn't reach here)
        (0, 0)
    }

    /// Validate the blend tree configuration.
    pub fn validate(&self) -> Result<(), BlendTreeError> {
        if self.clips.is_empty() {
            return Err(BlendTreeError::InsufficientClips {
                required: 1,
                actual: 0,
            });
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BlendTree2D
// ---------------------------------------------------------------------------

/// 2D blend tree with Delaunay triangulation.
///
/// Clips are placed at (x, y) positions in 2D space. The evaluation point
/// is projected onto the triangulation, and barycentric coordinates are
/// used to blend the three clips of the containing triangle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlendTree2D {
    /// Name of the X-axis parameter.
    pub param_x: String,

    /// Name of the Y-axis parameter.
    pub param_y: String,

    /// Clips at their 2D positions.
    /// Tuple: (x, y, clip_index)
    pub clips: Vec<(f32, f32, usize)>,

    /// Triangulation indices.
    /// Each triangle is defined by three clip indices.
    pub triangles: Vec<[usize; 3]>,

    /// Whether triangulation is up-to-date.
    triangulated: bool,
}

impl BlendTree2D {
    /// Create a new 2D blend tree.
    pub fn new(param_x: impl Into<String>, param_y: impl Into<String>) -> Self {
        Self {
            param_x: param_x.into(),
            param_y: param_y.into(),
            clips: Vec::new(),
            triangles: Vec::new(),
            triangulated: false,
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(
        param_x: impl Into<String>,
        param_y: impl Into<String>,
        capacity: usize,
    ) -> Self {
        Self {
            param_x: param_x.into(),
            param_y: param_y.into(),
            clips: Vec::with_capacity(capacity),
            triangles: Vec::new(),
            triangulated: false,
        }
    }

    /// Add a clip at a 2D position.
    pub fn add_clip(&mut self, x: f32, y: f32, clip_index: usize) {
        self.clips.push((x, y, clip_index));
        self.triangulated = false;
    }

    /// Get the number of clips.
    #[inline]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Check if triangulation is current.
    #[inline]
    pub fn is_triangulated(&self) -> bool {
        self.triangulated
    }

    /// Get the bounding box of clip positions.
    pub fn bounds(&self) -> Option<((f32, f32), (f32, f32))> {
        if self.clips.is_empty() {
            return None;
        }

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for (x, y, _) in &self.clips {
            min_x = min_x.min(*x);
            min_y = min_y.min(*y);
            max_x = max_x.max(*x);
            max_y = max_y.max(*y);
        }

        Some(((min_x, min_y), (max_x, max_y)))
    }

    /// Perform Delaunay triangulation of the clip positions.
    ///
    /// This must be called before evaluation if clips have been added.
    /// Uses the Bowyer-Watson algorithm for incremental triangulation.
    pub fn triangulate(&mut self) {
        self.triangles.clear();

        if self.clips.len() < 3 {
            // Not enough points for triangulation
            self.triangulated = true;
            return;
        }

        // Collect points
        let points: Vec<(f32, f32)> = self.clips.iter().map(|(x, y, _)| (*x, *y)).collect();

        // Perform Delaunay triangulation
        self.triangles = delaunay_triangulate(&points);
        self.triangulated = true;
    }

    /// Evaluate the blend tree at a 2D position.
    ///
    /// Returns weighted clips based on barycentric interpolation.
    pub fn evaluate(&mut self, x: f32, y: f32) -> Vec<(usize, f32)> {
        if self.clips.is_empty() {
            return Vec::new();
        }

        // Single clip
        if self.clips.len() == 1 {
            return vec![(self.clips[0].2, 1.0)];
        }

        // Two clips - linear interpolation
        if self.clips.len() == 2 {
            return self.evaluate_linear(x, y);
        }

        // Ensure triangulation is current
        if !self.triangulated {
            self.triangulate();
        }

        // Find containing triangle and compute barycentric weights
        if let Some((tri_idx, weights)) = self.find_containing_triangle(x, y) {
            let tri = self.triangles[tri_idx];
            let mut result = Vec::with_capacity(3);

            for i in 0..3 {
                let clip_idx = tri[i];
                let weight = weights[i];
                if weight > BLEND_EPSILON {
                    result.push((self.clips[clip_idx].2, weight));
                }
            }

            // Normalize weights
            let total: f32 = result.iter().map(|(_, w)| w).sum();
            if total > BLEND_EPSILON {
                for (_, w) in &mut result {
                    *w /= total;
                }
            }

            return result;
        }

        // Point outside triangulation - find nearest clip
        self.evaluate_nearest(x, y)
    }

    /// Linear interpolation for 2-clip case.
    fn evaluate_linear(&self, x: f32, y: f32) -> Vec<(usize, f32)> {
        let (x0, y0, clip0) = self.clips[0];
        let (x1, y1, clip1) = self.clips[1];

        // Project point onto line segment
        let dx = x1 - x0;
        let dy = y1 - y0;
        let len_sq = dx * dx + dy * dy;

        if len_sq < BLEND_EPSILON {
            return vec![(clip0, 1.0)];
        }

        let t = (((x - x0) * dx + (y - y0) * dy) / len_sq).clamp(0.0, 1.0);

        if clip0 == clip1 {
            vec![(clip0, 1.0)]
        } else {
            vec![(clip0, 1.0 - t), (clip1, t)]
        }
    }

    /// Find nearest clip for points outside triangulation.
    fn evaluate_nearest(&self, x: f32, y: f32) -> Vec<(usize, f32)> {
        let mut nearest_idx = 0;
        let mut nearest_dist_sq = f32::MAX;

        for (i, (cx, cy, _)) in self.clips.iter().enumerate() {
            let dx = x - cx;
            let dy = y - cy;
            let dist_sq = dx * dx + dy * dy;

            if dist_sq < nearest_dist_sq {
                nearest_dist_sq = dist_sq;
                nearest_idx = i;
            }
        }

        vec![(self.clips[nearest_idx].2, 1.0)]
    }

    /// Find the triangle containing a point and compute barycentric coords.
    fn find_containing_triangle(&self, x: f32, y: f32) -> Option<(usize, [f32; 3])> {
        for (tri_idx, tri) in self.triangles.iter().enumerate() {
            let (x0, y0, _) = self.clips[tri[0]];
            let (x1, y1, _) = self.clips[tri[1]];
            let (x2, y2, _) = self.clips[tri[2]];

            if let Some(weights) = barycentric_coords(x, y, x0, y0, x1, y1, x2, y2) {
                // Check if point is inside triangle (all weights >= 0)
                if weights[0] >= -BLEND_EPSILON
                    && weights[1] >= -BLEND_EPSILON
                    && weights[2] >= -BLEND_EPSILON
                {
                    return Some((tri_idx, weights));
                }
            }
        }

        None
    }

    /// Validate the blend tree.
    pub fn validate(&self) -> Result<(), BlendTreeError> {
        if self.clips.len() < 3 {
            return Err(BlendTreeError::InsufficientClips {
                required: 3,
                actual: self.clips.len(),
            });
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DirectionalBlendSpace
// ---------------------------------------------------------------------------

/// Directional blend space for direction + speed parameters.
///
/// Uses radial interpolation with proper wrap-around handling at 0/360 degrees.
/// Ideal for omnidirectional locomotion where both facing direction and
/// movement speed affect the animation blend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectionalBlendSpace {
    /// Name of the direction parameter (angle in radians).
    pub direction_param: String,

    /// Name of the speed parameter.
    pub speed_param: String,

    /// Clips at their (angle, speed) positions.
    /// Tuple: (angle_radians, speed, clip_index)
    pub clips: Vec<(f32, f32, usize)>,

    /// Angle tolerance for direction matching (radians).
    pub angle_tolerance: f32,
}

impl DirectionalBlendSpace {
    /// Create a new directional blend space.
    pub fn new(direction_param: impl Into<String>, speed_param: impl Into<String>) -> Self {
        Self {
            direction_param: direction_param.into(),
            speed_param: speed_param.into(),
            clips: Vec::new(),
            angle_tolerance: DEFAULT_ANGLE_TOLERANCE,
        }
    }

    /// Add a clip at an angle and speed.
    ///
    /// # Arguments
    ///
    /// * `angle_radians` - Direction angle in radians (0 = forward, PI/2 = right)
    /// * `speed` - Movement speed threshold
    /// * `clip_index` - Animation clip index
    pub fn add_clip(&mut self, angle_radians: f32, speed: f32, clip_index: usize) {
        // Normalize angle to [0, 2*PI)
        let normalized = normalize_angle(angle_radians);
        self.clips.push((normalized, speed, clip_index));
    }

    /// Set the angle tolerance for direction matching.
    pub fn set_angle_tolerance(&mut self, tolerance: f32) {
        self.angle_tolerance = tolerance.abs();
    }

    /// Get the number of clips.
    #[inline]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Evaluate the blend space at a direction and speed.
    ///
    /// # Arguments
    ///
    /// * `direction` - Direction angle in radians
    /// * `speed` - Movement speed
    ///
    /// # Returns
    ///
    /// Weighted clips for blending.
    pub fn evaluate(&self, direction: f32, speed: f32) -> Vec<(usize, f32)> {
        if self.clips.is_empty() {
            return Vec::new();
        }

        if self.clips.len() == 1 {
            return vec![(self.clips[0].2, 1.0)];
        }

        let direction = normalize_angle(direction);

        // Find clips that match direction (within tolerance) and bracket speed
        let matching_clips = self.find_direction_matches(direction);

        if matching_clips.is_empty() {
            // No direction matches - find nearest overall
            return self.find_nearest_clip(direction, speed);
        }

        // Blend by speed within direction matches
        self.blend_by_speed(&matching_clips, speed)
    }

    /// Find clips matching the given direction.
    fn find_direction_matches(&self, direction: f32) -> Vec<(usize, f32, f32)> {
        let mut matches = Vec::new();

        for (i, (angle, speed, _clip_idx)) in self.clips.iter().enumerate() {
            let diff = angle_difference(direction, *angle);
            if diff <= self.angle_tolerance {
                // Weight by how close the direction match is
                let weight = 1.0 - (diff / self.angle_tolerance);
                matches.push((i, *speed, weight));
            }
        }

        // Sort by speed for bracketing
        matches.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        matches
    }

    /// Blend clips by speed parameter.
    fn blend_by_speed(&self, matches: &[(usize, f32, f32)], speed: f32) -> Vec<(usize, f32)> {
        if matches.is_empty() {
            return Vec::new();
        }

        if matches.len() == 1 {
            let (idx, _, dir_weight) = matches[0];
            return vec![(self.clips[idx].2, dir_weight)];
        }

        // Check if all matches have the same speed - if so, blend by direction weights only
        let first_speed = matches[0].1;
        let all_same_speed = matches.iter().all(|(_, s, _)| (*s - first_speed).abs() < BLEND_EPSILON);

        if all_same_speed {
            // All clips at same speed: blend using direction weights
            let total_weight: f32 = matches.iter().map(|(_, _, w)| *w).sum();
            if total_weight < BLEND_EPSILON {
                return vec![(self.clips[matches[0].0].2, 1.0)];
            }
            return matches
                .iter()
                .map(|(idx, _, w)| (self.clips[*idx].2, *w / total_weight))
                .collect();
        }

        // Find speed bracket
        let (lower, upper) = self.find_speed_bracket(matches, speed);

        let (lower_idx, lower_speed, lower_dir_weight) = matches[lower];
        let (upper_idx, upper_speed, upper_dir_weight) = matches[upper];

        if lower == upper {
            return vec![(self.clips[lower_idx].2, lower_dir_weight)];
        }

        // Interpolate by speed
        let speed_range = upper_speed - lower_speed;
        let t = if speed_range.abs() < BLEND_EPSILON {
            0.5
        } else {
            ((speed - lower_speed) / speed_range).clamp(0.0, 1.0)
        };

        // Combine direction weights with speed interpolation
        let w_lower = (1.0 - t) * lower_dir_weight;
        let w_upper = t * upper_dir_weight;

        let total = w_lower + w_upper;
        if total < BLEND_EPSILON {
            return vec![(self.clips[lower_idx].2, 1.0)];
        }

        vec![
            (self.clips[lower_idx].2, w_lower / total),
            (self.clips[upper_idx].2, w_upper / total),
        ]
    }

    /// Find speed bracket indices.
    fn find_speed_bracket(&self, matches: &[(usize, f32, f32)], speed: f32) -> (usize, usize) {
        if speed <= matches[0].1 {
            return (0, 0);
        }

        let last = matches.len() - 1;
        if speed >= matches[last].1 {
            return (last, last);
        }

        for i in 0..last {
            if speed >= matches[i].1 && speed <= matches[i + 1].1 {
                return (i, i + 1);
            }
        }

        (0, 0)
    }

    /// Find the nearest clip when no direction match.
    fn find_nearest_clip(&self, direction: f32, speed: f32) -> Vec<(usize, f32)> {
        let mut nearest_idx = 0;
        let mut nearest_dist = f32::MAX;

        for (i, (angle, clip_speed, _)) in self.clips.iter().enumerate() {
            let angle_dist = angle_difference(direction, *angle);
            let speed_dist = (speed - clip_speed).abs();
            // Combined distance metric
            let dist = angle_dist + speed_dist * 0.1;

            if dist < nearest_dist {
                nearest_dist = dist;
                nearest_idx = i;
            }
        }

        vec![(self.clips[nearest_idx].2, 1.0)]
    }

    /// Validate the blend space.
    pub fn validate(&self) -> Result<(), BlendTreeError> {
        if self.clips.is_empty() {
            return Err(BlendTreeError::InsufficientClips {
                required: 1,
                actual: 0,
            });
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AdditiveBlendSpace
// ---------------------------------------------------------------------------

/// Additive blend space for layered animation.
///
/// Combines a base animation with additive overlays controlled by a parameter.
/// Supports per-bone masking to limit additive effects to specific body parts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdditiveBlendSpace {
    /// Base animation clip index (always plays at weight 1.0).
    pub base_clip: usize,

    /// Additive clips at their threshold values.
    /// Tuple: (threshold, clip_index)
    pub additive_clips: Vec<(f32, usize)>,

    /// Parameter controlling additive weight.
    pub additive_param: String,

    /// Optional per-bone mask.
    /// If Some, only bones with true are affected by additive.
    pub bone_mask: Option<Vec<bool>>,

    /// Maximum additive weight (usually 1.0).
    pub max_weight: f32,

    /// Whether clips are sorted by threshold.
    sorted: bool,
}

impl AdditiveBlendSpace {
    /// Create a new additive blend space.
    ///
    /// # Arguments
    ///
    /// * `base_clip` - Index of the base animation clip
    /// * `additive_param` - Name of the parameter controlling additive weight
    pub fn new(base_clip: usize, additive_param: impl Into<String>) -> Self {
        Self {
            base_clip,
            additive_clips: Vec::new(),
            additive_param: additive_param.into(),
            bone_mask: None,
            max_weight: 1.0,
            sorted: true,
        }
    }

    /// Add an additive clip at a threshold.
    pub fn add_additive_clip(&mut self, threshold: f32, clip_index: usize) {
        self.additive_clips.push((threshold, clip_index));
        self.sorted = false;
    }

    /// Set the bone mask for additive blending.
    ///
    /// # Arguments
    ///
    /// * `mask` - Vector of bool, one per bone. True = affected by additive.
    pub fn set_bone_mask(&mut self, mask: Vec<bool>) {
        self.bone_mask = Some(mask);
    }

    /// Clear the bone mask (apply additive to all bones).
    pub fn clear_bone_mask(&mut self) {
        self.bone_mask = None;
    }

    /// Set the maximum additive weight.
    pub fn set_max_weight(&mut self, weight: f32) {
        self.max_weight = weight.clamp(0.0, 2.0);
    }

    /// Get the number of additive clips.
    #[inline]
    pub fn additive_clip_count(&self) -> usize {
        self.additive_clips.len()
    }

    /// Check if a bone is affected by additive.
    pub fn is_bone_masked(&self, bone_index: usize) -> bool {
        match &self.bone_mask {
            Some(mask) => mask.get(bone_index).copied().unwrap_or(true),
            None => true,
        }
    }

    /// Sort clips by threshold.
    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.additive_clips
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            self.sorted = true;
        }
    }

    /// Evaluate the blend space.
    ///
    /// Returns (base_clip, additive_weights) where additive_weights is
    /// a list of (clip_index, weight) pairs.
    ///
    /// # Arguments
    ///
    /// * `param_value` - Current additive parameter value
    ///
    /// # Returns
    ///
    /// Tuple of (base_clip_index, Vec<(additive_clip_index, weight)>)
    pub fn evaluate(&mut self, param_value: f32) -> (usize, Vec<(usize, f32)>) {
        if self.additive_clips.is_empty() {
            return (self.base_clip, Vec::new());
        }

        self.ensure_sorted();

        // Find the additive clips to blend
        let (lower_idx, upper_idx) = self.find_bracket(param_value);

        let (lower_threshold, lower_clip) = self.additive_clips[lower_idx];
        let (upper_threshold, upper_clip) = self.additive_clips[upper_idx];

        // Calculate additive weight based on parameter position
        let additive_weight = self.calculate_additive_weight(param_value);

        // Same bracket - single additive clip
        if lower_idx == upper_idx {
            if additive_weight > BLEND_EPSILON {
                return (self.base_clip, vec![(lower_clip, additive_weight)]);
            } else {
                return (self.base_clip, Vec::new());
            }
        }

        // Blend between two additive clips
        let range = upper_threshold - lower_threshold;
        if range.abs() < BLEND_EPSILON {
            return (self.base_clip, vec![(lower_clip, additive_weight)]);
        }

        let t = ((param_value - lower_threshold) / range).clamp(0.0, 1.0);

        let w_lower = (1.0 - t) * additive_weight;
        let w_upper = t * additive_weight;

        let mut additives = Vec::new();
        if w_lower > BLEND_EPSILON {
            additives.push((lower_clip, w_lower));
        }
        if w_upper > BLEND_EPSILON && upper_clip != lower_clip {
            additives.push((upper_clip, w_upper));
        }

        (self.base_clip, additives)
    }

    /// Calculate the overall additive weight from parameter value.
    fn calculate_additive_weight(&self, param_value: f32) -> f32 {
        if self.additive_clips.is_empty() {
            return 0.0;
        }

        let min_threshold = self.additive_clips[0].0;
        let max_threshold = self.additive_clips.last().unwrap().0;

        // Single clip case: full weight when at or past the threshold
        if (max_threshold - min_threshold).abs() < BLEND_EPSILON {
            if param_value >= min_threshold {
                return self.max_weight;
            } else {
                return 0.0;
            }
        }

        // Weight ramps from 0 at min to max_weight at max
        if param_value <= min_threshold {
            return 0.0;
        }
        if param_value >= max_threshold {
            return self.max_weight;
        }

        let range = max_threshold - min_threshold;
        ((param_value - min_threshold) / range) * self.max_weight
    }

    /// Find bracket indices for parameter value.
    fn find_bracket(&self, param_value: f32) -> (usize, usize) {
        debug_assert!(self.sorted, "clips must be sorted");
        debug_assert!(!self.additive_clips.is_empty());

        if param_value <= self.additive_clips[0].0 {
            return (0, 0);
        }

        let last = self.additive_clips.len() - 1;
        if param_value >= self.additive_clips[last].0 {
            return (last, last);
        }

        for i in 0..last {
            if param_value >= self.additive_clips[i].0
                && param_value <= self.additive_clips[i + 1].0
            {
                return (i, i + 1);
            }
        }

        (0, 0)
    }

    /// Validate the blend space.
    pub fn validate(&self) -> Result<(), BlendTreeError> {
        // Base clip is required, additives are optional
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/// Normalize an angle to [0, 2*PI).
#[inline]
pub fn normalize_angle(angle: f32) -> f32 {
    let mut a = angle % TAU;
    if a < 0.0 {
        a += TAU;
    }
    a
}

/// Calculate the shortest angular difference between two angles.
#[inline]
pub fn angle_difference(a: f32, b: f32) -> f32 {
    let diff = (normalize_angle(a) - normalize_angle(b)).abs();
    if diff > PI {
        TAU - diff
    } else {
        diff
    }
}

/// Compute barycentric coordinates for a point in a triangle.
///
/// Returns None if the triangle is degenerate.
pub fn barycentric_coords(
    px: f32,
    py: f32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
) -> Option<[f32; 3]> {
    let v0x = x1 - x0;
    let v0y = y1 - y0;
    let v1x = x2 - x0;
    let v1y = y2 - y0;
    let v2x = px - x0;
    let v2y = py - y0;

    let dot00 = v0x * v0x + v0y * v0y;
    let dot01 = v0x * v1x + v0y * v1y;
    let dot02 = v0x * v2x + v0y * v2y;
    let dot11 = v1x * v1x + v1y * v1y;
    let dot12 = v1x * v2x + v1y * v2y;

    let denom = dot00 * dot11 - dot01 * dot01;
    if denom.abs() < BLEND_EPSILON {
        return None; // Degenerate triangle
    }

    let inv_denom = 1.0 / denom;
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
    let w = 1.0 - u - v;

    Some([w, u, v])
}

/// Perform Delaunay triangulation using Bowyer-Watson algorithm.
///
/// Returns a list of triangle index triples.
fn delaunay_triangulate(points: &[(f32, f32)]) -> Vec<[usize; 3]> {
    if points.len() < 3 {
        return Vec::new();
    }

    // Find bounding box
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for (x, y) in points {
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
    }

    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let delta_max = dx.max(dy);
    let mid_x = (min_x + max_x) / 2.0;
    let mid_y = (min_y + max_y) / 2.0;

    // Create super-triangle
    let p0 = (mid_x - 2.0 * delta_max, mid_y - delta_max);
    let p1 = (mid_x, mid_y + 2.0 * delta_max);
    let p2 = (mid_x + 2.0 * delta_max, mid_y - delta_max);

    // Extended points including super-triangle
    let n = points.len();
    let mut vertices: Vec<(f32, f32)> = points.to_vec();
    vertices.push(p0);
    vertices.push(p1);
    vertices.push(p2);

    // Start with super-triangle
    let mut triangles: Vec<[usize; 3]> = vec![[n, n + 1, n + 2]];

    // Insert each point
    for (i, &point) in points.iter().enumerate() {
        // Find triangles whose circumcircle contains the point
        let mut bad_triangles: Vec<usize> = Vec::new();

        for (ti, tri) in triangles.iter().enumerate() {
            let (cx, cy, r_sq) = circumcircle(
                vertices[tri[0]],
                vertices[tri[1]],
                vertices[tri[2]],
            );

            let dist_sq = (point.0 - cx).powi(2) + (point.1 - cy).powi(2);
            if dist_sq <= r_sq + BLEND_EPSILON {
                bad_triangles.push(ti);
            }
        }

        // Find the boundary polygon of bad triangles
        let mut polygon: Vec<[usize; 2]> = Vec::new();

        for &ti in &bad_triangles {
            let tri = triangles[ti];
            for j in 0..3 {
                let edge = [tri[j], tri[(j + 1) % 3]];
                // Check if this edge is shared with another bad triangle
                let mut is_shared = false;
                for &ti2 in &bad_triangles {
                    if ti2 == ti {
                        continue;
                    }
                    let tri2 = triangles[ti2];
                    for k in 0..3 {
                        let edge2 = [tri2[k], tri2[(k + 1) % 3]];
                        if (edge[0] == edge2[0] && edge[1] == edge2[1])
                            || (edge[0] == edge2[1] && edge[1] == edge2[0])
                        {
                            is_shared = true;
                            break;
                        }
                    }
                    if is_shared {
                        break;
                    }
                }
                if !is_shared {
                    polygon.push(edge);
                }
            }
        }

        // Remove bad triangles (in reverse order to preserve indices)
        bad_triangles.sort_by(|a, b| b.cmp(a));
        for ti in bad_triangles {
            triangles.swap_remove(ti);
        }

        // Create new triangles from polygon edges to new point
        for edge in polygon {
            triangles.push([edge[0], edge[1], i]);
        }
    }

    // Remove triangles that share vertices with super-triangle
    triangles.retain(|tri| {
        tri[0] < n && tri[1] < n && tri[2] < n
    });

    triangles
}

/// Compute circumcircle center and squared radius.
fn circumcircle(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32)) -> (f32, f32, f32) {
    let ax = p1.0 - p0.0;
    let ay = p1.1 - p0.1;
    let bx = p2.0 - p0.0;
    let by = p2.1 - p0.1;

    let d = 2.0 * (ax * by - ay * bx);
    if d.abs() < BLEND_EPSILON {
        // Degenerate case - return large circle
        let cx = (p0.0 + p1.0 + p2.0) / 3.0;
        let cy = (p0.1 + p1.1 + p2.1) / 3.0;
        return (cx, cy, f32::MAX);
    }

    let a_sq = ax * ax + ay * ay;
    let b_sq = bx * bx + by * by;

    let ux = (by * a_sq - ay * b_sq) / d;
    let uy = (ax * b_sq - bx * a_sq) / d;

    let cx = p0.0 + ux;
    let cy = p0.1 + uy;
    let r_sq = ux * ux + uy * uy;

    (cx, cy, r_sq)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // BlendTree1D Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_tree_1d_new() {
        let tree = BlendTree1D::new("speed");
        assert_eq!(tree.parameter, "speed");
        assert!(tree.clips.is_empty());
    }

    #[test]
    fn test_blend_tree_1d_add_clip() {
        let mut tree = BlendTree1D::new("speed");
        tree.add_clip(0.0, 0);
        tree.add_clip(5.0, 1);
        tree.add_clip(10.0, 2);

        assert_eq!(tree.clip_count(), 3);
    }

    #[test]
    fn test_blend_tree_1d_threshold_range() {
        let mut tree = BlendTree1D::new("speed");
        assert!(tree.threshold_range().is_none());

        tree.add_clip(5.0, 0);
        tree.add_clip(0.0, 1);
        tree.add_clip(10.0, 2);

        let (min, max) = tree.threshold_range().unwrap();
        assert!((min - 0.0).abs() < BLEND_EPSILON);
        assert!((max - 10.0).abs() < BLEND_EPSILON);
    }

    #[test]
    fn test_blend_tree_1d_single_clip() {
        let mut tree = BlendTree1D::new("speed");
        tree.add_clip(5.0, 0);

        let weights = tree.evaluate(0.0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0], (0, 1.0));

        let weights = tree.evaluate(10.0);
        assert_eq!(weights[0], (0, 1.0));
    }

    #[test]
    fn test_blend_tree_1d_at_threshold() {
        let mut tree = BlendTree1D::new("speed");
        tree.add_clip(0.0, 0);
        tree.add_clip(5.0, 1);
        tree.add_clip(10.0, 2);

        // Exactly at threshold
        let weights = tree.evaluate(5.0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0], (1, 1.0));
    }

    #[test]
    fn test_blend_tree_1d_between_clips() {
        let mut tree = BlendTree1D::new("speed");
        tree.add_clip(0.0, 0);
        tree.add_clip(10.0, 1);

        // Midpoint
        let weights = tree.evaluate(5.0);
        assert_eq!(weights.len(), 2);
        assert!((weights[0].1 - 0.5).abs() < 0.01);
        assert!((weights[1].1 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_blend_tree_1d_below_min() {
        let mut tree = BlendTree1D::new("speed");
        tree.add_clip(5.0, 0);
        tree.add_clip(10.0, 1);

        let weights = tree.evaluate(0.0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].0, 0);
    }

    #[test]
    fn test_blend_tree_1d_above_max() {
        let mut tree = BlendTree1D::new("speed");
        tree.add_clip(0.0, 0);
        tree.add_clip(5.0, 1);

        let weights = tree.evaluate(10.0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].0, 1);
    }

    #[test]
    fn test_blend_tree_1d_unsorted_clips() {
        let mut tree = BlendTree1D::new("speed");
        // Add out of order
        tree.add_clip(10.0, 2);
        tree.add_clip(0.0, 0);
        tree.add_clip(5.0, 1);

        // Should still work correctly
        let weights = tree.evaluate(2.5);
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].0, 0); // First clip
        assert_eq!(weights[1].0, 1); // Second clip
    }

    #[test]
    fn test_blend_tree_1d_weight_normalization() {
        let mut tree = BlendTree1D::new("speed");
        tree.add_clip(0.0, 0);
        tree.add_clip(10.0, 1);

        for param in [0.0, 2.5, 5.0, 7.5, 10.0] {
            let weights = tree.evaluate(param);
            let total: f32 = weights.iter().map(|(_, w)| w).sum();
            assert!((total - 1.0).abs() < 0.01, "weights should sum to 1.0");
        }
    }

    #[test]
    fn test_blend_tree_1d_empty() {
        let mut tree = BlendTree1D::new("speed");
        let weights = tree.evaluate(5.0);
        assert!(weights.is_empty());
    }

    #[test]
    fn test_blend_tree_1d_validate() {
        let tree = BlendTree1D::new("speed");
        assert!(tree.validate().is_err());

        let mut tree = BlendTree1D::new("speed");
        tree.add_clip(0.0, 0);
        assert!(tree.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // BlendTree2D Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_tree_2d_new() {
        let tree = BlendTree2D::new("vel_x", "vel_z");
        assert_eq!(tree.param_x, "vel_x");
        assert_eq!(tree.param_y, "vel_z");
        assert!(tree.clips.is_empty());
    }

    #[test]
    fn test_blend_tree_2d_add_clip() {
        let mut tree = BlendTree2D::new("vel_x", "vel_z");
        tree.add_clip(0.0, 0.0, 0);
        tree.add_clip(1.0, 0.0, 1);
        tree.add_clip(0.0, 1.0, 2);

        assert_eq!(tree.clip_count(), 3);
    }

    #[test]
    fn test_blend_tree_2d_bounds() {
        let mut tree = BlendTree2D::new("x", "y");
        assert!(tree.bounds().is_none());

        tree.add_clip(-1.0, -1.0, 0);
        tree.add_clip(1.0, 2.0, 1);

        let ((min_x, min_y), (max_x, max_y)) = tree.bounds().unwrap();
        assert!((min_x - (-1.0)).abs() < BLEND_EPSILON);
        assert!((min_y - (-1.0)).abs() < BLEND_EPSILON);
        assert!((max_x - 1.0).abs() < BLEND_EPSILON);
        assert!((max_y - 2.0).abs() < BLEND_EPSILON);
    }

    #[test]
    fn test_blend_tree_2d_triangulation() {
        let mut tree = BlendTree2D::new("x", "y");
        tree.add_clip(0.0, 0.0, 0);
        tree.add_clip(1.0, 0.0, 1);
        tree.add_clip(0.5, 1.0, 2);

        assert!(!tree.is_triangulated());
        tree.triangulate();
        assert!(tree.is_triangulated());
        assert_eq!(tree.triangles.len(), 1);
    }

    #[test]
    fn test_blend_tree_2d_triangulation_square() {
        let mut tree = BlendTree2D::new("x", "y");
        tree.add_clip(0.0, 0.0, 0);
        tree.add_clip(1.0, 0.0, 1);
        tree.add_clip(1.0, 1.0, 2);
        tree.add_clip(0.0, 1.0, 3);

        tree.triangulate();
        // A square should produce 2 triangles
        assert_eq!(tree.triangles.len(), 2);
    }

    #[test]
    fn test_blend_tree_2d_barycentric_center() {
        let weights = barycentric_coords(
            1.0 / 3.0, 1.0 / 3.0, // center of unit triangle
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
        );

        let w = weights.unwrap();
        // All three weights should be equal at center
        assert!((w[0] - 1.0 / 3.0).abs() < 0.01);
        assert!((w[1] - 1.0 / 3.0).abs() < 0.01);
        assert!((w[2] - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_blend_tree_2d_barycentric_vertex() {
        // Point at vertex 0
        let w = barycentric_coords(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0).unwrap();
        assert!((w[0] - 1.0).abs() < BLEND_EPSILON);
        assert!(w[1].abs() < BLEND_EPSILON);
        assert!(w[2].abs() < BLEND_EPSILON);

        // Point at vertex 1
        let w = barycentric_coords(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0).unwrap();
        assert!(w[0].abs() < BLEND_EPSILON);
        assert!((w[1] - 1.0).abs() < BLEND_EPSILON);
        assert!(w[2].abs() < BLEND_EPSILON);
    }

    #[test]
    fn test_blend_tree_2d_single_clip() {
        let mut tree = BlendTree2D::new("x", "y");
        tree.add_clip(0.0, 0.0, 5);

        let weights = tree.evaluate(1.0, 1.0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0], (5, 1.0));
    }

    #[test]
    fn test_blend_tree_2d_two_clips() {
        let mut tree = BlendTree2D::new("x", "y");
        tree.add_clip(0.0, 0.0, 0);
        tree.add_clip(2.0, 0.0, 1);

        // Midpoint
        let weights = tree.evaluate(1.0, 0.0);
        assert_eq!(weights.len(), 2);
        assert!((weights[0].1 - 0.5).abs() < 0.01);
        assert!((weights[1].1 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_blend_tree_2d_evaluate_inside() {
        let mut tree = BlendTree2D::new("x", "y");
        tree.add_clip(0.0, 0.0, 0);
        tree.add_clip(1.0, 0.0, 1);
        tree.add_clip(0.5, 1.0, 2);
        tree.triangulate();

        // Center of triangle
        let weights = tree.evaluate(0.5, 0.33);
        assert!(!weights.is_empty());

        let total: f32 = weights.iter().map(|(_, w)| w).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_blend_tree_2d_evaluate_outside() {
        let mut tree = BlendTree2D::new("x", "y");
        tree.add_clip(0.0, 0.0, 0);
        tree.add_clip(1.0, 0.0, 1);
        tree.add_clip(0.5, 1.0, 2);
        tree.triangulate();

        // Outside triangle - should return nearest
        let weights = tree.evaluate(10.0, 10.0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].0, 2); // Nearest to (0.5, 1.0)
    }

    #[test]
    fn test_blend_tree_2d_validate() {
        let tree = BlendTree2D::new("x", "y");
        assert!(tree.validate().is_err());

        let mut tree = BlendTree2D::new("x", "y");
        tree.add_clip(0.0, 0.0, 0);
        tree.add_clip(1.0, 0.0, 1);
        tree.add_clip(0.5, 1.0, 2);
        assert!(tree.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // DirectionalBlendSpace Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_directional_new() {
        let space = DirectionalBlendSpace::new("direction", "speed");
        assert_eq!(space.direction_param, "direction");
        assert_eq!(space.speed_param, "speed");
        assert!(space.clips.is_empty());
    }

    #[test]
    fn test_directional_add_clip() {
        let mut space = DirectionalBlendSpace::new("dir", "spd");
        space.add_clip(0.0, 0.0, 0);
        space.add_clip(PI / 2.0, 5.0, 1);
        space.add_clip(-PI, 10.0, 2);

        assert_eq!(space.clip_count(), 3);
    }

    #[test]
    fn test_directional_angle_normalization() {
        let mut space = DirectionalBlendSpace::new("dir", "spd");
        space.add_clip(-PI, 0.0, 0);
        space.add_clip(3.0 * PI, 0.0, 1);

        // Both should be normalized to [0, 2*PI)
        assert!(space.clips[0].0 >= 0.0 && space.clips[0].0 < TAU);
        assert!(space.clips[1].0 >= 0.0 && space.clips[1].0 < TAU);
    }

    #[test]
    fn test_directional_wrap_around() {
        // Test that angle difference handles wrap-around correctly
        let diff1 = angle_difference(0.1, TAU - 0.1);
        assert!((diff1 - 0.2).abs() < 0.01);

        let diff2 = angle_difference(TAU - 0.1, 0.1);
        assert!((diff2 - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_directional_evaluate_single() {
        let mut space = DirectionalBlendSpace::new("dir", "spd");
        space.add_clip(0.0, 5.0, 0);

        let weights = space.evaluate(0.0, 5.0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0], (0, 1.0));
    }

    #[test]
    fn test_directional_evaluate_direction_match() {
        let mut space = DirectionalBlendSpace::new("dir", "spd");
        space.set_angle_tolerance(PI / 4.0);
        space.add_clip(0.0, 0.0, 0);
        space.add_clip(0.0, 10.0, 1);

        // Same direction, different speeds
        let weights = space.evaluate(0.0, 5.0);
        assert_eq!(weights.len(), 2);
    }

    #[test]
    fn test_directional_evaluate_no_direction_match() {
        let mut space = DirectionalBlendSpace::new("dir", "spd");
        space.set_angle_tolerance(PI / 8.0);
        space.add_clip(0.0, 5.0, 0);
        space.add_clip(PI, 5.0, 1);

        // Direction doesn't match any clip
        let weights = space.evaluate(PI / 2.0, 5.0);
        // Should return nearest
        assert_eq!(weights.len(), 1);
    }

    #[test]
    fn test_directional_cardinal_directions() {
        let mut space = DirectionalBlendSpace::new("dir", "spd");
        space.set_angle_tolerance(PI / 4.0);

        // Cardinal directions
        space.add_clip(0.0, 5.0, 0);         // Forward
        space.add_clip(PI / 2.0, 5.0, 1);    // Right
        space.add_clip(PI, 5.0, 2);          // Back
        space.add_clip(3.0 * PI / 2.0, 5.0, 3); // Left

        // Test each cardinal direction
        let w = space.evaluate(0.0, 5.0);
        assert!(w.iter().any(|(idx, _)| *idx == 0));

        let w = space.evaluate(PI / 2.0, 5.0);
        assert!(w.iter().any(|(idx, _)| *idx == 1));
    }

    #[test]
    fn test_directional_validate() {
        let space = DirectionalBlendSpace::new("dir", "spd");
        assert!(space.validate().is_err());

        let mut space = DirectionalBlendSpace::new("dir", "spd");
        space.add_clip(0.0, 0.0, 0);
        assert!(space.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // AdditiveBlendSpace Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_additive_new() {
        let space = AdditiveBlendSpace::new(0, "intensity");
        assert_eq!(space.base_clip, 0);
        assert_eq!(space.additive_param, "intensity");
        assert!(space.additive_clips.is_empty());
    }

    #[test]
    fn test_additive_add_clip() {
        let mut space = AdditiveBlendSpace::new(0, "intensity");
        space.add_additive_clip(0.0, 1);
        space.add_additive_clip(1.0, 2);

        assert_eq!(space.additive_clip_count(), 2);
    }

    #[test]
    fn test_additive_bone_mask() {
        let mut space = AdditiveBlendSpace::new(0, "intensity");
        space.set_bone_mask(vec![true, true, false, false, true]);

        assert!(space.is_bone_masked(0));
        assert!(space.is_bone_masked(1));
        assert!(!space.is_bone_masked(2));
        assert!(!space.is_bone_masked(3));
        assert!(space.is_bone_masked(4));

        // Out of bounds should return true
        assert!(space.is_bone_masked(100));

        space.clear_bone_mask();
        assert!(space.is_bone_masked(2)); // Now all bones affected
    }

    #[test]
    fn test_additive_evaluate_no_additives() {
        let mut space = AdditiveBlendSpace::new(0, "intensity");

        let (base, additives) = space.evaluate(0.5);
        assert_eq!(base, 0);
        assert!(additives.is_empty());
    }

    #[test]
    fn test_additive_evaluate_at_min() {
        let mut space = AdditiveBlendSpace::new(0, "intensity");
        space.add_additive_clip(0.0, 1);
        space.add_additive_clip(1.0, 2);

        let (base, additives) = space.evaluate(0.0);
        assert_eq!(base, 0);
        // At minimum threshold, additive weight should be 0
        assert!(additives.is_empty());
    }

    #[test]
    fn test_additive_evaluate_at_max() {
        let mut space = AdditiveBlendSpace::new(0, "intensity");
        space.add_additive_clip(0.0, 1);
        space.add_additive_clip(1.0, 2);

        let (base, additives) = space.evaluate(1.0);
        assert_eq!(base, 0);
        assert_eq!(additives.len(), 1);
        assert_eq!(additives[0].0, 2);
        assert!((additives[0].1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_additive_evaluate_between() {
        let mut space = AdditiveBlendSpace::new(0, "intensity");
        space.add_additive_clip(0.0, 1);
        space.add_additive_clip(1.0, 2);

        let (base, additives) = space.evaluate(0.5);
        assert_eq!(base, 0);
        // Should have both additive clips blended
        assert_eq!(additives.len(), 2);

        // Total additive weight should be 0.5 (halfway through the range)
        let total: f32 = additives.iter().map(|(_, w)| w).sum();
        assert!((total - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_additive_max_weight() {
        let mut space = AdditiveBlendSpace::new(0, "intensity");
        space.set_max_weight(0.5);
        space.add_additive_clip(0.0, 1);
        space.add_additive_clip(1.0, 2);

        let (_, additives) = space.evaluate(1.0);
        let total: f32 = additives.iter().map(|(_, w)| w).sum();
        assert!((total - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_additive_single_clip() {
        let mut space = AdditiveBlendSpace::new(0, "intensity");
        space.add_additive_clip(0.5, 1);

        // Below threshold
        let (base, additives) = space.evaluate(0.0);
        assert_eq!(base, 0);
        assert!(additives.is_empty());

        // At threshold
        let (base, additives) = space.evaluate(0.5);
        assert_eq!(base, 0);
        // Single clip at its threshold
        assert_eq!(additives.len(), 1);
    }

    #[test]
    fn test_additive_validate() {
        let space = AdditiveBlendSpace::new(0, "intensity");
        // Valid even with no additive clips
        assert!(space.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Utility Function Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_angle() {
        assert!((normalize_angle(0.0) - 0.0).abs() < BLEND_EPSILON);
        assert!((normalize_angle(TAU) - 0.0).abs() < BLEND_EPSILON);
        assert!((normalize_angle(-PI) - PI).abs() < BLEND_EPSILON);
        assert!((normalize_angle(3.0 * PI) - PI).abs() < BLEND_EPSILON);
    }

    #[test]
    fn test_angle_difference_same() {
        assert!(angle_difference(0.0, 0.0).abs() < BLEND_EPSILON);
        assert!(angle_difference(PI, PI).abs() < BLEND_EPSILON);
    }

    #[test]
    fn test_angle_difference_opposite() {
        let diff = angle_difference(0.0, PI);
        assert!((diff - PI).abs() < BLEND_EPSILON);
    }

    #[test]
    fn test_angle_difference_wrap() {
        // Small difference across 0/2PI boundary
        let diff = angle_difference(0.1, TAU - 0.1);
        assert!((diff - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_barycentric_degenerate() {
        // Collinear points
        let w = barycentric_coords(0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0);
        assert!(w.is_none());
    }

    #[test]
    fn test_barycentric_edge() {
        // Point on edge between v0 and v1
        let w = barycentric_coords(0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0).unwrap();
        assert!((w[2]).abs() < BLEND_EPSILON); // v2 weight should be 0
        assert!((w[0] + w[1] - 1.0).abs() < BLEND_EPSILON);
    }

    #[test]
    fn test_delaunay_simple_triangle() {
        let points = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
        let triangles = delaunay_triangulate(&points);
        assert_eq!(triangles.len(), 1);
    }

    #[test]
    fn test_delaunay_square() {
        let points = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let triangles = delaunay_triangulate(&points);
        assert_eq!(triangles.len(), 2);
    }

    #[test]
    fn test_delaunay_pentagon() {
        let points: Vec<(f32, f32)> = (0..5)
            .map(|i| {
                let angle = (i as f32) * TAU / 5.0;
                (angle.cos(), angle.sin())
            })
            .collect();

        let triangles = delaunay_triangulate(&points);
        // Pentagon should produce 3 triangles
        assert_eq!(triangles.len(), 3);
    }

    #[test]
    fn test_delaunay_not_enough_points() {
        assert!(delaunay_triangulate(&[]).is_empty());
        assert!(delaunay_triangulate(&[(0.0, 0.0)]).is_empty());
        assert!(delaunay_triangulate(&[(0.0, 0.0), (1.0, 0.0)]).is_empty());
    }

    // -----------------------------------------------------------------------
    // Edge Case Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_edge_case_duplicate_threshold_1d() {
        let mut tree = BlendTree1D::new("speed");
        tree.add_clip(5.0, 0);
        tree.add_clip(5.0, 1); // Same threshold

        // Should not crash
        let weights = tree.evaluate(5.0);
        assert!(!weights.is_empty());
    }

    #[test]
    fn test_edge_case_same_clip_index() {
        let mut tree = BlendTree1D::new("speed");
        tree.add_clip(0.0, 0);
        tree.add_clip(10.0, 0); // Same clip at different thresholds

        let weights = tree.evaluate(5.0);
        // Should consolidate to single clip
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].0, 0);
    }

    #[test]
    fn test_edge_case_negative_param() {
        let mut tree = BlendTree1D::new("speed");
        tree.add_clip(-10.0, 0);
        tree.add_clip(0.0, 1);
        tree.add_clip(10.0, 2);

        let weights = tree.evaluate(-5.0);
        assert_eq!(weights.len(), 2);
    }

    #[test]
    fn test_edge_case_very_small_range() {
        let mut tree = BlendTree1D::new("speed");
        tree.add_clip(0.0, 0);
        tree.add_clip(0.0001, 1);

        // Should handle very small ranges
        let weights = tree.evaluate(0.00005);
        assert!(!weights.is_empty());
    }

    #[test]
    fn test_edge_case_collinear_2d() {
        let mut tree = BlendTree2D::new("x", "y");
        tree.add_clip(0.0, 0.0, 0);
        tree.add_clip(1.0, 0.0, 1);
        tree.add_clip(2.0, 0.0, 2);

        tree.triangulate();
        // Collinear points produce no triangles
        assert!(tree.triangles.is_empty());

        // Should still evaluate via nearest
        let weights = tree.evaluate(0.5, 0.0);
        assert!(!weights.is_empty());
    }

    #[test]
    fn test_edge_case_directional_zero_speed() {
        let mut space = DirectionalBlendSpace::new("dir", "spd");
        space.add_clip(0.0, 0.0, 0);
        space.add_clip(0.0, 10.0, 1);

        let weights = space.evaluate(0.0, 0.0);
        assert!(!weights.is_empty());
    }

    #[test]
    fn test_edge_case_additive_negative_param() {
        let mut space = AdditiveBlendSpace::new(0, "intensity");
        space.add_additive_clip(0.0, 1);
        space.add_additive_clip(1.0, 2);

        let (base, additives) = space.evaluate(-1.0);
        assert_eq!(base, 0);
        assert!(additives.is_empty());
    }

    // -----------------------------------------------------------------------
    // Integration Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_locomotion_blend_tree() {
        // Simulate a typical locomotion setup
        let mut tree = BlendTree1D::new("speed");
        tree.add_clip(0.0, 0);   // idle
        tree.add_clip(1.5, 1);   // walk
        tree.add_clip(3.0, 2);   // jog
        tree.add_clip(6.0, 3);   // run
        tree.add_clip(10.0, 4);  // sprint

        // Test various speeds
        let test_cases = [
            (0.0, vec![(0, 1.0)]),          // idle
            (1.5, vec![(1, 1.0)]),          // walk
            (4.5, vec![(2, 0.5), (3, 0.5)]), // jog/run blend
        ];

        for (speed, expected) in test_cases {
            let weights = tree.evaluate(speed);
            assert_eq!(weights.len(), expected.len(), "speed={}", speed);
            for (i, (clip, weight)) in expected.into_iter().enumerate() {
                assert_eq!(weights[i].0, clip, "speed={}, clip mismatch", speed);
                assert!((weights[i].1 - weight).abs() < 0.01, "speed={}, weight mismatch", speed);
            }
        }
    }

    #[test]
    fn test_directional_locomotion() {
        let mut space = DirectionalBlendSpace::new("direction", "speed");
        space.set_angle_tolerance(PI / 4.0);

        // 8-directional setup
        for i in 0..8 {
            let angle = (i as f32) * PI / 4.0;
            space.add_clip(angle, 5.0, i);
        }

        // Test cardinal directions
        let forward_weights = space.evaluate(0.0, 5.0);
        assert!(forward_weights.iter().any(|(idx, _)| *idx == 0));

        let right_weights = space.evaluate(PI / 2.0, 5.0);
        assert!(right_weights.iter().any(|(idx, _)| *idx == 2));
    }

    #[test]
    fn test_upper_body_additive() {
        let mut space = AdditiveBlendSpace::new(0, "aim_weight");

        // Upper body mask (bones 0-10)
        let mut mask = vec![false; 50];
        for i in 0..10 {
            mask[i] = true;
        }
        space.set_bone_mask(mask);

        space.add_additive_clip(0.0, 1); // aim down
        space.add_additive_clip(0.5, 2); // aim center
        space.add_additive_clip(1.0, 3); // aim up

        // Test aim blend
        let (base, additives) = space.evaluate(0.75);
        assert_eq!(base, 0);
        assert!(!additives.is_empty());

        // Check bone masking
        assert!(space.is_bone_masked(5));   // Upper body
        assert!(!space.is_bone_masked(30)); // Lower body
    }
}
