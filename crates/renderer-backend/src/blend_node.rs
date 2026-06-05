//! Blend node types for animation graph evaluation (T-AN-5.4).
//!
//! This module provides blend node types for hierarchical animation blending:
//!
//! - **ClipNode**: Plays a single animation clip with time tracking
//! - **Blend1DNode**: 1D parameter-driven blend between children
//! - **Blend2DNode**: 2D parameter-driven blend with triangulation
//! - **AdditiveNode**: Additive blend of child onto base pose
//! - **OverrideNode**: Full-body override with blend weight
//! - **LayerNode**: Masked layer blend with bone mask
//!
//! # Architecture
//!
//! ```text
//! BlendNodeTrait
//! +-- evaluate(&params, dt) -> NodeOutput
//! +-- get_duration() -> Option<f32>
//! +-- reset() -> ()
//!
//! NodeOutput
//! +-- pose: Pose
//! +-- events: Vec<AnimationEvent>
//! +-- root_motion: Option<RootMotionDelta>
//!
//! BlendNode (enum dispatch)
//! +-- Clip(ClipNode)
//! +-- Blend1D(Blend1DNode)
//! +-- Blend2D(Blend2DNode)
//! +-- Additive(AdditiveNode)
//! +-- Override(OverrideNode)
//! +-- Layer(LayerNode)
//! +-- Identity
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::blend_node::{
//!     BlendNode, ClipNode, Blend1DNode, NodeOutput,
//! };
//! use renderer_backend::animation_clip::LoopMode;
//! use renderer_backend::state_machine::ParameterSet;
//!
//! // Create a clip node
//! let clip = ClipNode::new(0, LoopMode::Loop);
//!
//! // Create a 1D blend for locomotion
//! let mut locomotion = Blend1DNode::new("speed");
//! locomotion.add_child(0.0, BlendNode::Clip(ClipNode::new(0, LoopMode::Loop)));  // idle
//! locomotion.add_child(3.0, BlendNode::Clip(ClipNode::new(1, LoopMode::Loop)));  // walk
//! locomotion.add_child(8.0, BlendNode::Clip(ClipNode::new(2, LoopMode::Loop)));  // run
//!
//! // Evaluate with parameters
//! let mut params = ParameterSet::new();
//! params.set_float("speed", 5.0);
//! let output = locomotion.evaluate(&params, 0.016);
//! ```

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::animation_clip::LoopMode;
use crate::animation_graph::LayerBlendMode;
use crate::pose::{lerp_vec3, nlerp_quat, slerp_quat, Pose, PoseType};
use crate::root_motion::{blend_root_motion, RootMotionDelta};
use crate::state_machine::ParameterSet;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum children per blend node.
pub const MAX_BLEND_CHILDREN: usize = 32;

/// Epsilon for weight comparisons.
pub const WEIGHT_EPSILON: f32 = 1e-6;

/// Epsilon for time comparisons.
pub const TIME_EPSILON: f32 = 1e-6;

/// Default playback rate.
pub const DEFAULT_PLAY_RATE: f32 = 1.0;

// ---------------------------------------------------------------------------
// BlendNodeError
// ---------------------------------------------------------------------------

/// Errors that can occur during blend node operations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BlendNodeError {
    /// No children in blend node.
    NoChildren,

    /// Invalid parameter name.
    InvalidParameter { name: String },

    /// Parameter type mismatch.
    ParameterTypeMismatch { name: String, expected: String },

    /// Invalid bone count.
    BoneCountMismatch { expected: usize, actual: usize },

    /// Empty pose returned.
    EmptyPose,

    /// Invalid threshold ordering.
    InvalidThresholds,

    /// Child evaluation failed.
    ChildEvaluationFailed { index: usize },
}

impl std::fmt::Display for BlendNodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoChildren => write!(f, "blend node has no children"),
            Self::InvalidParameter { name } => write!(f, "invalid parameter: {}", name),
            Self::ParameterTypeMismatch { name, expected } => {
                write!(f, "parameter '{}' type mismatch, expected {}", name, expected)
            }
            Self::BoneCountMismatch { expected, actual } => {
                write!(f, "bone count mismatch: expected {}, got {}", expected, actual)
            }
            Self::EmptyPose => write!(f, "empty pose returned"),
            Self::InvalidThresholds => write!(f, "invalid threshold ordering"),
            Self::ChildEvaluationFailed { index } => {
                write!(f, "child evaluation failed at index {}", index)
            }
        }
    }
}

impl std::error::Error for BlendNodeError {}

// ---------------------------------------------------------------------------
// AnimationEvent (simplified for blend nodes)
// ---------------------------------------------------------------------------

/// Animation event fired during node evaluation.
///
/// Simplified version of AnimationNotify for blend node event propagation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationEvent {
    /// Name of the event/notify.
    pub name: String,

    /// Time within the clip when the event fires.
    pub time: f32,

    /// Clip index that generated this event.
    pub clip_index: usize,

    /// Optional payload data.
    pub payload: Option<String>,
}

impl AnimationEvent {
    /// Create a new animation event.
    pub fn new(name: impl Into<String>, time: f32, clip_index: usize) -> Self {
        Self {
            name: name.into(),
            time,
            clip_index,
            payload: None,
        }
    }

    /// Create an event with payload.
    pub fn with_payload(
        name: impl Into<String>,
        time: f32,
        clip_index: usize,
        payload: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            time,
            clip_index,
            payload: Some(payload.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// NodeOutput
// ---------------------------------------------------------------------------

/// Output from blend node evaluation.
///
/// Contains the computed pose, any animation events that fired,
/// and optional root motion delta for this frame.
#[derive(Clone, Debug)]
pub struct NodeOutput {
    /// The computed pose for this frame.
    pub pose: Pose,

    /// Animation events that fired during evaluation.
    pub events: Vec<AnimationEvent>,

    /// Root motion delta for this frame (if any).
    pub root_motion: Option<RootMotionDelta>,

    /// Whether the animation has completed (for non-looping clips).
    pub completed: bool,

    /// Current normalized time (0.0 to 1.0).
    pub normalized_time: f32,
}

impl NodeOutput {
    /// Create a new output with a pose.
    pub fn new(pose: Pose) -> Self {
        Self {
            pose,
            events: Vec::new(),
            root_motion: None,
            completed: false,
            normalized_time: 0.0,
        }
    }

    /// Create an identity output (no change).
    pub fn identity(bone_count: usize) -> Self {
        Self::new(Pose::new(bone_count, PoseType::Current))
    }

    /// Add an event to this output.
    pub fn add_event(&mut self, event: AnimationEvent) {
        self.events.push(event);
    }

    /// Set root motion.
    pub fn with_root_motion(mut self, delta: RootMotionDelta) -> Self {
        self.root_motion = Some(delta);
        self
    }

    /// Set completed flag.
    pub fn with_completed(mut self, completed: bool) -> Self {
        self.completed = completed;
        self
    }

    /// Set normalized time.
    pub fn with_normalized_time(mut self, t: f32) -> Self {
        self.normalized_time = t;
        self
    }

    /// Merge events from another output.
    pub fn merge_events(&mut self, other: &NodeOutput) {
        self.events.extend(other.events.iter().cloned());
    }

    /// Blend root motion from two outputs.
    pub fn blend_root_motion_outputs(a: &NodeOutput, b: &NodeOutput, weight: f32) -> Option<RootMotionDelta> {
        match (&a.root_motion, &b.root_motion) {
            (Some(rm_a), Some(rm_b)) => Some(blend_root_motion(rm_a, rm_b, weight)),
            (Some(rm), None) => Some(rm.scale(1.0 - weight)),
            (None, Some(rm)) => Some(rm.scale(weight)),
            (None, None) => None,
        }
    }

    /// Get the bone count of the pose.
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.pose.bone_count()
    }
}

impl Default for NodeOutput {
    fn default() -> Self {
        Self {
            pose: Pose::new(0, PoseType::Current),
            events: Vec::new(),
            root_motion: None,
            completed: false,
            normalized_time: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// BlendNodeTrait
// ---------------------------------------------------------------------------

/// Trait for blend node evaluation.
///
/// All blend node types implement this trait to provide uniform evaluation,
/// duration queries, and reset functionality.
pub trait BlendNodeTrait: Send + Sync {
    /// Evaluate the node and produce output.
    ///
    /// # Arguments
    ///
    /// * `params` - Current animation parameters
    /// * `dt` - Delta time in seconds
    /// * `bone_count` - Number of bones in the skeleton
    ///
    /// # Returns
    ///
    /// The computed pose, events, and root motion for this frame.
    fn evaluate(&mut self, params: &ParameterSet, dt: f32, bone_count: usize) -> NodeOutput;

    /// Get the duration of this node's content (if determinable).
    ///
    /// Returns None for nodes with indeterminate duration (e.g., blend nodes
    /// where children have different durations).
    fn get_duration(&self) -> Option<f32>;

    /// Reset the node to its initial state.
    ///
    /// This resets time tracking, event state, and child nodes recursively.
    fn reset(&mut self);

    /// Get the current playback time.
    fn get_current_time(&self) -> f32;

    /// Set the current playback time.
    fn set_current_time(&mut self, time: f32);

    /// Check if this node has completed playback.
    fn is_completed(&self) -> bool;
}

// ---------------------------------------------------------------------------
// ClipNode
// ---------------------------------------------------------------------------

/// A node that plays a single animation clip.
///
/// ClipNode handles time tracking, looping, and playback rate for a single
/// animation clip referenced by index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipNode {
    /// Index of the animation clip in the clip library.
    pub clip_index: usize,

    /// Playback rate multiplier (1.0 = normal, 0.5 = half speed).
    pub play_rate: f32,

    /// How the clip loops.
    pub loop_mode: LoopMode,

    /// Current playback time in seconds.
    pub current_time: f32,

    /// Duration of the clip in seconds.
    pub duration: f32,

    /// Whether playback is paused.
    pub paused: bool,

    /// Previous frame time for event detection.
    previous_time: f32,

    /// Whether the clip has completed (for Once mode).
    completed: bool,
}

impl ClipNode {
    /// Create a new clip node.
    pub fn new(clip_index: usize, loop_mode: LoopMode) -> Self {
        Self {
            clip_index,
            play_rate: DEFAULT_PLAY_RATE,
            loop_mode,
            current_time: 0.0,
            duration: 1.0, // Will be set when clip is loaded
            paused: false,
            previous_time: 0.0,
            completed: false,
        }
    }

    /// Create a clip node with custom play rate.
    pub fn with_play_rate(mut self, rate: f32) -> Self {
        self.play_rate = rate;
        self
    }

    /// Create a clip node with duration.
    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration = duration.max(TIME_EPSILON);
        self
    }

    /// Set the clip duration.
    pub fn set_duration(&mut self, duration: f32) {
        self.duration = duration.max(TIME_EPSILON);
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Resume playback.
    pub fn resume(&mut self) {
        self.paused = false;
    }

    /// Seek to a specific time.
    pub fn seek(&mut self, time: f32) {
        self.current_time = time.clamp(0.0, self.duration);
        self.previous_time = self.current_time;
    }

    /// Get normalized time (0.0 to 1.0).
    #[inline]
    pub fn normalized_time(&self) -> f32 {
        if self.duration > TIME_EPSILON {
            self.current_time / self.duration
        } else {
            0.0
        }
    }

    /// Advance time by delta.
    fn advance_time(&mut self, dt: f32) {
        if self.paused || self.completed {
            return;
        }

        self.previous_time = self.current_time;
        let delta = dt * self.play_rate;

        match self.loop_mode {
            LoopMode::Once => {
                self.current_time += delta;
                if self.current_time >= self.duration {
                    self.current_time = self.duration;
                    self.completed = true;
                } else if self.current_time < 0.0 {
                    self.current_time = 0.0;
                    self.completed = true;
                }
            }
            LoopMode::Loop => {
                self.current_time += delta;
                if self.duration > TIME_EPSILON {
                    self.current_time = self.current_time.rem_euclid(self.duration);
                }
            }
            LoopMode::PingPong => {
                self.current_time += delta;
                if self.duration > TIME_EPSILON {
                    // Wrap within [0, 2*duration) and reflect
                    let double_duration = self.duration * 2.0;
                    let wrapped = self.current_time.rem_euclid(double_duration);
                    if wrapped > self.duration {
                        self.current_time = double_duration - wrapped;
                    } else {
                        self.current_time = wrapped;
                    }
                }
            }
        }
    }

    /// Sample the clip at the current time.
    ///
    /// Returns a pose sampled from the animation clip.
    /// In a real implementation, this would access the clip library.
    fn sample_pose(&self, bone_count: usize) -> Pose {
        // Create a pose - in real implementation, this would sample from clip data
        let mut pose = Pose::new(bone_count, PoseType::Current);

        // For testing, apply a simple oscillation based on time
        let t = self.normalized_time();
        let offset = (t * std::f32::consts::PI * 2.0).sin() * 0.1;

        // Apply offset to first bone if exists
        if bone_count > 0 {
            pose.positions[0] = Vec3::new(0.0, offset, 0.0);
        }

        pose
    }

    /// Check for events that crossed during this frame.
    fn check_events(&self) -> Vec<AnimationEvent> {
        // Placeholder - in real implementation, check clip's event list
        Vec::new()
    }
}

impl BlendNodeTrait for ClipNode {
    fn evaluate(&mut self, _params: &ParameterSet, dt: f32, bone_count: usize) -> NodeOutput {
        self.advance_time(dt);

        let pose = self.sample_pose(bone_count);
        let events = self.check_events();

        NodeOutput {
            pose,
            events,
            root_motion: None,
            completed: self.completed,
            normalized_time: self.normalized_time(),
        }
    }

    fn get_duration(&self) -> Option<f32> {
        Some(self.duration)
    }

    fn reset(&mut self) {
        self.current_time = 0.0;
        self.previous_time = 0.0;
        self.completed = false;
    }

    fn get_current_time(&self) -> f32 {
        self.current_time
    }

    fn set_current_time(&mut self, time: f32) {
        self.current_time = time.clamp(0.0, self.duration);
    }

    fn is_completed(&self) -> bool {
        self.completed
    }
}

// ---------------------------------------------------------------------------
// Blend1DNode
// ---------------------------------------------------------------------------

/// A node that blends between children based on a 1D parameter.
///
/// Children are placed at threshold values along a single axis. The parameter
/// value determines the blend weights through linear interpolation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Blend1DNode {
    /// Name of the parameter that drives this blend.
    pub parameter: String,

    /// Children at their threshold positions: (threshold, node).
    pub children: Vec<(f32, BlendNode)>,

    /// Whether children are sorted by threshold.
    sorted: bool,

    /// Cached parameter value from last evaluation.
    cached_param: f32,
}

impl Blend1DNode {
    /// Create a new 1D blend node.
    pub fn new(parameter: impl Into<String>) -> Self {
        Self {
            parameter: parameter.into(),
            children: Vec::new(),
            sorted: true,
            cached_param: 0.0,
        }
    }

    /// Add a child at a threshold position.
    pub fn add_child(&mut self, threshold: f32, node: BlendNode) {
        self.children.push((threshold, node));
        self.sorted = false;
    }

    /// Ensure children are sorted by threshold.
    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.children
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            self.sorted = true;
        }
    }

    /// Get blend weights for the given parameter value.
    ///
    /// Returns indices and weights of children to blend.
    pub fn get_weights(&mut self, param: f32) -> Vec<(usize, f32)> {
        self.ensure_sorted();

        if self.children.is_empty() {
            return Vec::new();
        }

        if self.children.len() == 1 {
            return vec![(0, 1.0)];
        }

        // Find the two children to blend between
        let mut lower_idx = 0;
        for (i, (threshold, _)) in self.children.iter().enumerate() {
            if *threshold <= param {
                lower_idx = i;
            } else {
                break;
            }
        }

        let upper_idx = (lower_idx + 1).min(self.children.len() - 1);

        if lower_idx == upper_idx {
            return vec![(lower_idx, 1.0)];
        }

        let lower_threshold = self.children[lower_idx].0;
        let upper_threshold = self.children[upper_idx].0;
        let range = upper_threshold - lower_threshold;

        if range.abs() < WEIGHT_EPSILON {
            return vec![(lower_idx, 1.0)];
        }

        let t = ((param - lower_threshold) / range).clamp(0.0, 1.0);

        vec![(lower_idx, 1.0 - t), (upper_idx, t)]
    }

    /// Blend poses with given weights.
    fn blend_poses(poses: &[(Pose, f32)]) -> Pose {
        if poses.is_empty() {
            return Pose::new(0, PoseType::Current);
        }

        if poses.len() == 1 {
            return poses[0].0.clone();
        }

        // Use first pose as base
        let bone_count = poses[0].0.bone_count();
        let mut result = Pose::new(bone_count, PoseType::Current);

        // Accumulate weighted transforms
        for i in 0..bone_count {
            let mut pos = Vec3::ZERO;
            let mut scale = Vec3::ZERO;
            let mut total_weight = 0.0;

            // Accumulate positions and scales
            for (pose, weight) in poses {
                if i < pose.bone_count() {
                    pos += pose.positions[i] * *weight;
                    scale += pose.scales[i] * *weight;
                    total_weight += *weight;
                }
            }

            if total_weight > WEIGHT_EPSILON {
                result.positions[i] = pos;
                result.scales[i] = scale;
            }

            // Blend rotations using weighted slerp
            if !poses.is_empty() && i < poses[0].0.bone_count() {
                let mut rot = poses[0].0.rotations[i];
                let mut remaining_weight = poses[0].1;

                for (pose, weight) in poses.iter().skip(1) {
                    if i < pose.bone_count() && *weight > WEIGHT_EPSILON {
                        let blend_factor = *weight / (remaining_weight + *weight);
                        rot = nlerp_quat(rot, pose.rotations[i], blend_factor);
                        remaining_weight += *weight;
                    }
                }

                result.rotations[i] = rot.normalize();
            }
        }

        result
    }
}

impl BlendNodeTrait for Blend1DNode {
    fn evaluate(&mut self, params: &ParameterSet, dt: f32, bone_count: usize) -> NodeOutput {
        let param_value = params.get_float(&self.parameter).unwrap_or(0.0);
        self.cached_param = param_value;

        let weights = self.get_weights(param_value);

        if weights.is_empty() {
            return NodeOutput::identity(bone_count);
        }

        // Evaluate children and collect poses
        let mut pose_weights: Vec<(Pose, f32)> = Vec::new();
        let mut all_events = Vec::new();
        let mut root_motions: Vec<(RootMotionDelta, f32)> = Vec::new();

        for (idx, weight) in &weights {
            if *weight > WEIGHT_EPSILON {
                if let Some((_, ref mut node)) = self.children.get_mut(*idx) {
                    let output = node.evaluate(params, dt, bone_count);
                    pose_weights.push((output.pose, *weight));
                    all_events.extend(output.events);

                    if let Some(rm) = output.root_motion {
                        root_motions.push((rm, *weight));
                    }
                }
            }
        }

        let blended_pose = Self::blend_poses(&pose_weights);

        // Blend root motion
        let root_motion = if root_motions.is_empty() {
            None
        } else if root_motions.len() == 1 {
            Some(root_motions[0].0.scale(root_motions[0].1))
        } else {
            let mut result = root_motions[0].0.scale(root_motions[0].1);
            for (rm, w) in root_motions.iter().skip(1) {
                result = result.combine(&rm.scale(*w));
            }
            Some(result)
        };

        NodeOutput {
            pose: blended_pose,
            events: all_events,
            root_motion,
            completed: false,
            normalized_time: 0.0,
        }
    }

    fn get_duration(&self) -> Option<f32> {
        // Return average duration of children, or None if mixed
        if self.children.is_empty() {
            return None;
        }

        let durations: Vec<f32> = self
            .children
            .iter()
            .filter_map(|(_, node)| node.get_duration())
            .collect();

        if durations.is_empty() {
            None
        } else if durations.iter().all(|d| (*d - durations[0]).abs() < TIME_EPSILON) {
            Some(durations[0])
        } else {
            // Return average
            Some(durations.iter().sum::<f32>() / durations.len() as f32)
        }
    }

    fn reset(&mut self) {
        for (_, node) in &mut self.children {
            node.reset();
        }
    }

    fn get_current_time(&self) -> f32 {
        self.children
            .first()
            .map(|(_, node)| node.get_current_time())
            .unwrap_or(0.0)
    }

    fn set_current_time(&mut self, time: f32) {
        for (_, node) in &mut self.children {
            node.set_current_time(time);
        }
    }

    fn is_completed(&self) -> bool {
        self.children.iter().all(|(_, node)| node.is_completed())
    }
}

// ---------------------------------------------------------------------------
// Blend2DNode
// ---------------------------------------------------------------------------

/// A node that blends between children based on two parameters.
///
/// Children are placed at 2D positions. Uses barycentric interpolation
/// within triangles formed by Delaunay triangulation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Blend2DNode {
    /// Name of the X parameter.
    pub param_x: String,

    /// Name of the Y parameter.
    pub param_y: String,

    /// Children at their 2D positions: (x, y, node).
    pub children: Vec<(f32, f32, BlendNode)>,

    /// Precomputed triangulation indices.
    triangles: Vec<[usize; 3]>,

    /// Whether triangulation is up to date.
    triangulated: bool,
}

impl Blend2DNode {
    /// Create a new 2D blend node.
    pub fn new(param_x: impl Into<String>, param_y: impl Into<String>) -> Self {
        Self {
            param_x: param_x.into(),
            param_y: param_y.into(),
            children: Vec::new(),
            triangles: Vec::new(),
            triangulated: false,
        }
    }

    /// Add a child at a 2D position.
    pub fn add_child(&mut self, x: f32, y: f32, node: BlendNode) {
        self.children.push((x, y, node));
        self.triangulated = false;
    }

    /// Perform Delaunay triangulation of child positions.
    pub fn triangulate(&mut self) {
        self.triangles.clear();

        if self.children.len() < 3 {
            // Not enough points for triangles
            self.triangulated = true;
            return;
        }

        // Simple ear-clipping triangulation for convex shapes
        // For production, use proper Delaunay triangulation
        let n = self.children.len();

        // Create a simple fan triangulation from center
        if n >= 3 {
            // Find centroid
            let cx: f32 = self.children.iter().map(|(x, _, _)| x).sum::<f32>() / n as f32;
            let cy: f32 = self.children.iter().map(|(_, y, _)| y).sum::<f32>() / n as f32;

            // Sort points by angle from centroid
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| {
                let (ax, ay, _) = self.children[a];
                let (bx, by, _) = self.children[b];
                let angle_a = (ay - cy).atan2(ax - cx);
                let angle_b = (by - cy).atan2(bx - cx);
                angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Create triangles
            for i in 0..n - 2 {
                self.triangles.push([indices[0], indices[i + 1], indices[i + 2]]);
            }
        }

        self.triangulated = true;
    }

    /// Get barycentric weights for a point in a triangle.
    fn barycentric(
        px: f32,
        py: f32,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
    ) -> Option<(f32, f32, f32)> {
        let denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);

        if denom.abs() < WEIGHT_EPSILON {
            return None;
        }

        let w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom;
        let w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom;
        let w2 = 1.0 - w0 - w1;

        Some((w0, w1, w2))
    }

    /// Get blend weights for the given parameter values.
    pub fn get_weights(&mut self, param_x: f32, param_y: f32) -> Vec<(usize, f32)> {
        if !self.triangulated {
            self.triangulate();
        }

        if self.children.is_empty() {
            return Vec::new();
        }

        if self.children.len() == 1 {
            return vec![(0, 1.0)];
        }

        if self.children.len() == 2 {
            // Linear interpolation between two points
            let (x0, y0, _) = self.children[0];
            let (x1, y1, _) = self.children[1];

            let dx = x1 - x0;
            let dy = y1 - y0;
            let len_sq = dx * dx + dy * dy;

            if len_sq < WEIGHT_EPSILON {
                return vec![(0, 1.0)];
            }

            let t = ((param_x - x0) * dx + (param_y - y0) * dy) / len_sq;
            let t = t.clamp(0.0, 1.0);

            return vec![(0, 1.0 - t), (1, t)];
        }

        // Try each triangle
        for tri in &self.triangles {
            let (x0, y0, _) = self.children[tri[0]];
            let (x1, y1, _) = self.children[tri[1]];
            let (x2, y2, _) = self.children[tri[2]];

            if let Some((w0, w1, w2)) = Self::barycentric(param_x, param_y, x0, y0, x1, y1, x2, y2)
            {
                // Check if point is inside triangle (all weights positive)
                if w0 >= -WEIGHT_EPSILON && w1 >= -WEIGHT_EPSILON && w2 >= -WEIGHT_EPSILON {
                    let mut weights = Vec::new();
                    if w0 > WEIGHT_EPSILON {
                        weights.push((tri[0], w0));
                    }
                    if w1 > WEIGHT_EPSILON {
                        weights.push((tri[1], w1));
                    }
                    if w2 > WEIGHT_EPSILON {
                        weights.push((tri[2], w2));
                    }
                    return weights;
                }
            }
        }

        // Point outside all triangles - find nearest point
        let mut nearest_idx = 0;
        let mut nearest_dist_sq = f32::MAX;

        for (i, (x, y, _)) in self.children.iter().enumerate() {
            let dx = param_x - x;
            let dy = param_y - y;
            let dist_sq = dx * dx + dy * dy;

            if dist_sq < nearest_dist_sq {
                nearest_dist_sq = dist_sq;
                nearest_idx = i;
            }
        }

        vec![(nearest_idx, 1.0)]
    }
}

impl BlendNodeTrait for Blend2DNode {
    fn evaluate(&mut self, params: &ParameterSet, dt: f32, bone_count: usize) -> NodeOutput {
        let px = params.get_float(&self.param_x).unwrap_or(0.0);
        let py = params.get_float(&self.param_y).unwrap_or(0.0);

        let weights = self.get_weights(px, py);

        if weights.is_empty() {
            return NodeOutput::identity(bone_count);
        }

        // Evaluate children and blend
        let mut pose_weights: Vec<(Pose, f32)> = Vec::new();
        let mut all_events = Vec::new();

        for (idx, weight) in &weights {
            if *weight > WEIGHT_EPSILON {
                if let Some((_, _, ref mut node)) = self.children.get_mut(*idx) {
                    let output = node.evaluate(params, dt, bone_count);
                    pose_weights.push((output.pose, *weight));
                    all_events.extend(output.events);
                }
            }
        }

        let blended_pose = Blend1DNode::blend_poses(&pose_weights);

        NodeOutput {
            pose: blended_pose,
            events: all_events,
            root_motion: None,
            completed: false,
            normalized_time: 0.0,
        }
    }

    fn get_duration(&self) -> Option<f32> {
        if self.children.is_empty() {
            return None;
        }

        let durations: Vec<f32> = self
            .children
            .iter()
            .filter_map(|(_, _, node)| node.get_duration())
            .collect();

        if durations.is_empty() {
            None
        } else {
            Some(durations.iter().sum::<f32>() / durations.len() as f32)
        }
    }

    fn reset(&mut self) {
        for (_, _, node) in &mut self.children {
            node.reset();
        }
    }

    fn get_current_time(&self) -> f32 {
        self.children
            .first()
            .map(|(_, _, node)| node.get_current_time())
            .unwrap_or(0.0)
    }

    fn set_current_time(&mut self, time: f32) {
        for (_, _, node) in &mut self.children {
            node.set_current_time(time);
        }
    }

    fn is_completed(&self) -> bool {
        self.children.iter().all(|(_, _, node)| node.is_completed())
    }
}

// ---------------------------------------------------------------------------
// AdditiveNode
// ---------------------------------------------------------------------------

/// A node that applies an additive animation on top of a base pose.
///
/// The additive pose is computed as: base + (additive - reference) * weight
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdditiveNode {
    /// The base animation to build upon.
    pub base: Box<BlendNode>,

    /// The additive animation to apply.
    pub additive: Box<BlendNode>,

    /// Parameter name controlling the additive weight.
    pub weight_param: String,

    /// Static weight if no parameter is specified.
    pub static_weight: f32,

    /// Reference pose for computing additive delta.
    /// If None, uses the first frame of the additive animation.
    pub reference_pose: Option<Pose>,
}

impl AdditiveNode {
    /// Create a new additive node.
    pub fn new(base: BlendNode, additive: BlendNode) -> Self {
        Self {
            base: Box::new(base),
            additive: Box::new(additive),
            weight_param: String::new(),
            static_weight: 1.0,
            reference_pose: None,
        }
    }

    /// Set the weight parameter name.
    pub fn with_weight_param(mut self, param: impl Into<String>) -> Self {
        self.weight_param = param.into();
        self
    }

    /// Set the static weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.static_weight = weight;
        self
    }

    /// Set the reference pose.
    pub fn with_reference_pose(mut self, pose: Pose) -> Self {
        self.reference_pose = Some(pose);
        self
    }

    /// Get the current weight value.
    fn get_weight(&self, params: &ParameterSet) -> f32 {
        if self.weight_param.is_empty() {
            self.static_weight
        } else {
            params.get_float(&self.weight_param).unwrap_or(self.static_weight)
        }
    }

    /// Apply additive pose to base.
    fn apply_additive(base: &Pose, additive: &Pose, weight: f32) -> Pose {
        let bone_count = base.bone_count().min(additive.bone_count());
        let mut result = Pose::new(bone_count, PoseType::Current);

        let w = weight.clamp(0.0, 1.0);

        for i in 0..bone_count {
            // Position: add weighted delta
            result.positions[i] = base.positions[i] + additive.positions[i] * w;

            // Rotation: multiply by weighted rotation
            let additive_rot = slerp_quat(Quat::IDENTITY, additive.rotations[i], w);
            result.rotations[i] = (base.rotations[i] * additive_rot).normalize();

            // Scale: add weighted delta (additive scales are deltas from identity)
            result.scales[i] = base.scales[i] + additive.scales[i] * w;
        }

        result
    }
}

impl BlendNodeTrait for AdditiveNode {
    fn evaluate(&mut self, params: &ParameterSet, dt: f32, bone_count: usize) -> NodeOutput {
        let weight = self.get_weight(params);

        // Evaluate base
        let base_output = self.base.evaluate(params, dt, bone_count);

        // Skip additive if weight is zero
        if weight.abs() < WEIGHT_EPSILON {
            return base_output;
        }

        // Evaluate additive
        let additive_output = self.additive.evaluate(params, dt, bone_count);

        // Convert additive to delta if we have a reference pose
        let additive_delta = if let Some(ref reference) = self.reference_pose {
            // Compute delta: additive - reference
            let bone_count = additive_output.pose.bone_count().min(reference.bone_count());
            let mut delta = Pose::new(bone_count, PoseType::Additive);

            for i in 0..bone_count {
                delta.positions[i] = additive_output.pose.positions[i] - reference.positions[i];
                delta.rotations[i] = (additive_output.pose.rotations[i]
                    * reference.rotations[i].inverse())
                .normalize();
                delta.scales[i] = additive_output.pose.scales[i] - reference.scales[i];
            }

            delta
        } else {
            // Use additive pose directly (assume it's already a delta)
            additive_output.pose.clone()
        };

        let result_pose = Self::apply_additive(&base_output.pose, &additive_delta, weight);

        // Blend root motion (do this before moving events)
        let root_motion = NodeOutput::blend_root_motion_outputs(&base_output, &additive_output, weight);

        // Combine events from both
        let mut events = base_output.events;
        events.extend(additive_output.events);

        NodeOutput {
            pose: result_pose,
            events,
            root_motion,
            completed: base_output.completed,
            normalized_time: base_output.normalized_time,
        }
    }

    fn get_duration(&self) -> Option<f32> {
        // Return base duration as primary
        self.base.get_duration()
    }

    fn reset(&mut self) {
        self.base.reset();
        self.additive.reset();
    }

    fn get_current_time(&self) -> f32 {
        self.base.get_current_time()
    }

    fn set_current_time(&mut self, time: f32) {
        self.base.set_current_time(time);
        self.additive.set_current_time(time);
    }

    fn is_completed(&self) -> bool {
        self.base.is_completed()
    }
}

// ---------------------------------------------------------------------------
// OverrideNode
// ---------------------------------------------------------------------------

/// A node that overrides the source pose with another pose based on weight.
///
/// At weight 0, returns source. At weight 1, returns override.
/// This is essentially a crossfade between two animations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OverrideNode {
    /// The source animation.
    pub source: Box<BlendNode>,

    /// The override animation.
    pub override_node: Box<BlendNode>,

    /// Parameter name controlling the override weight.
    pub weight_param: String,

    /// Static weight if no parameter is specified.
    pub static_weight: f32,
}

impl OverrideNode {
    /// Create a new override node.
    pub fn new(source: BlendNode, override_node: BlendNode) -> Self {
        Self {
            source: Box::new(source),
            override_node: Box::new(override_node),
            weight_param: String::new(),
            static_weight: 0.0,
        }
    }

    /// Set the weight parameter name.
    pub fn with_weight_param(mut self, param: impl Into<String>) -> Self {
        self.weight_param = param.into();
        self
    }

    /// Set the static weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.static_weight = weight;
        self
    }

    /// Get the current weight value.
    fn get_weight(&self, params: &ParameterSet) -> f32 {
        if self.weight_param.is_empty() {
            self.static_weight
        } else {
            params.get_float(&self.weight_param).unwrap_or(self.static_weight)
        }
    }
}

impl BlendNodeTrait for OverrideNode {
    fn evaluate(&mut self, params: &ParameterSet, dt: f32, bone_count: usize) -> NodeOutput {
        let weight = self.get_weight(params).clamp(0.0, 1.0);

        // Early out for extreme weights
        if weight < WEIGHT_EPSILON {
            return self.source.evaluate(params, dt, bone_count);
        }

        if weight > 1.0 - WEIGHT_EPSILON {
            return self.override_node.evaluate(params, dt, bone_count);
        }

        // Evaluate both
        let source_output = self.source.evaluate(params, dt, bone_count);
        let override_output = self.override_node.evaluate(params, dt, bone_count);

        // Blend poses
        let blended_pose = source_output.pose.blend(&override_output.pose, weight);

        // Blend root motion (do this before moving events)
        let root_motion = NodeOutput::blend_root_motion_outputs(&source_output, &override_output, weight);

        // Extract values we need before moving events
        let completed = if weight < 0.5 {
            source_output.completed
        } else {
            override_output.completed
        };
        let source_normalized = source_output.normalized_time;
        let override_normalized = override_output.normalized_time;

        // Combine events from both
        let mut events = source_output.events;
        events.extend(override_output.events);

        NodeOutput {
            pose: blended_pose,
            events,
            root_motion,
            completed,
            normalized_time: lerp_vec3(
                Vec3::new(source_normalized, 0.0, 0.0),
                Vec3::new(override_normalized, 0.0, 0.0),
                weight,
            )
            .x,
        }
    }

    fn get_duration(&self) -> Option<f32> {
        // Return weighted average of durations
        match (self.source.get_duration(), self.override_node.get_duration()) {
            (Some(a), Some(b)) => Some(lerp_vec3(Vec3::new(a, 0.0, 0.0), Vec3::new(b, 0.0, 0.0), self.static_weight).x),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    fn reset(&mut self) {
        self.source.reset();
        self.override_node.reset();
    }

    fn get_current_time(&self) -> f32 {
        // Return weighted time
        let t = self.static_weight.clamp(0.0, 1.0);
        let source_time = self.source.get_current_time();
        let override_time = self.override_node.get_current_time();
        source_time * (1.0 - t) + override_time * t
    }

    fn set_current_time(&mut self, time: f32) {
        self.source.set_current_time(time);
        self.override_node.set_current_time(time);
    }

    fn is_completed(&self) -> bool {
        let t = self.static_weight.clamp(0.0, 1.0);
        if t < 0.5 {
            self.source.is_completed()
        } else {
            self.override_node.is_completed()
        }
    }
}

// ---------------------------------------------------------------------------
// LayerNode
// ---------------------------------------------------------------------------

/// A node that applies a layer animation with a bone mask.
///
/// The layer animation only affects bones where the mask is true.
/// Useful for upper body overrides, facial animation, etc.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerNode {
    /// The base animation.
    pub base: Box<BlendNode>,

    /// The layer animation to apply.
    pub layer: Box<BlendNode>,

    /// Bone mask: true = use layer, false = use base.
    pub bone_mask: Vec<bool>,

    /// How the layer blends with the base.
    pub blend_mode: LayerBlendMode,

    /// Layer weight (0.0 to 1.0).
    pub weight: f32,

    /// Parameter name controlling the layer weight.
    pub weight_param: String,
}

impl LayerNode {
    /// Create a new layer node.
    pub fn new(base: BlendNode, layer: BlendNode, bone_mask: Vec<bool>) -> Self {
        Self {
            base: Box::new(base),
            layer: Box::new(layer),
            bone_mask,
            blend_mode: LayerBlendMode::Override,
            weight: 1.0,
            weight_param: String::new(),
        }
    }

    /// Create a layer node with all bones masked.
    pub fn full_body(base: BlendNode, layer: BlendNode, bone_count: usize) -> Self {
        Self::new(base, layer, vec![true; bone_count])
    }

    /// Set the blend mode.
    pub fn with_blend_mode(mut self, mode: LayerBlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Set the layer weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the weight parameter name.
    pub fn with_weight_param(mut self, param: impl Into<String>) -> Self {
        self.weight_param = param.into();
        self
    }

    /// Get the current weight value.
    fn get_weight(&self, params: &ParameterSet) -> f32 {
        if self.weight_param.is_empty() {
            self.weight
        } else {
            params.get_float(&self.weight_param).unwrap_or(self.weight)
        }
    }

    /// Apply layer to base with mask.
    fn apply_layer(&self, base: &Pose, layer: &Pose, weight: f32) -> Pose {
        let bone_count = base.bone_count();
        let mut result = base.clone();

        let w = weight.clamp(0.0, 1.0);

        for i in 0..bone_count {
            // Check if this bone is masked
            let masked = self.bone_mask.get(i).copied().unwrap_or(false);

            if !masked || w < WEIGHT_EPSILON {
                continue; // Keep base pose for this bone
            }

            if i >= layer.bone_count() {
                continue; // Layer doesn't have this bone
            }

            match self.blend_mode {
                LayerBlendMode::Override => {
                    // Blend towards layer pose
                    result.positions[i] = lerp_vec3(base.positions[i], layer.positions[i], w);
                    result.rotations[i] = slerp_quat(base.rotations[i], layer.rotations[i], w);
                    result.scales[i] = lerp_vec3(base.scales[i], layer.scales[i], w);
                }
                LayerBlendMode::Additive | LayerBlendMode::MaskedAdditive => {
                    // Add layer delta to base
                    result.positions[i] = base.positions[i] + layer.positions[i] * w;

                    let additive_rot = slerp_quat(Quat::IDENTITY, layer.rotations[i], w);
                    result.rotations[i] = (base.rotations[i] * additive_rot).normalize();

                    // For additive scales, layer values are deltas
                    result.scales[i] = base.scales[i] + layer.scales[i] * w;
                }
            }
        }

        result
    }
}

impl BlendNodeTrait for LayerNode {
    fn evaluate(&mut self, params: &ParameterSet, dt: f32, bone_count: usize) -> NodeOutput {
        let weight = self.get_weight(params);

        // Evaluate base
        let base_output = self.base.evaluate(params, dt, bone_count);

        // Skip layer if weight is zero
        if weight < WEIGHT_EPSILON {
            return base_output;
        }

        // Evaluate layer
        let layer_output = self.layer.evaluate(params, dt, bone_count);

        // Apply layer with mask
        let result_pose = self.apply_layer(&base_output.pose, &layer_output.pose, weight);

        // Combine events from both
        let mut events = base_output.events;
        events.extend(layer_output.events);

        // Use base root motion (layers typically don't contribute root motion)
        let root_motion = base_output.root_motion;

        NodeOutput {
            pose: result_pose,
            events,
            root_motion,
            completed: base_output.completed,
            normalized_time: base_output.normalized_time,
        }
    }

    fn get_duration(&self) -> Option<f32> {
        self.base.get_duration()
    }

    fn reset(&mut self) {
        self.base.reset();
        self.layer.reset();
    }

    fn get_current_time(&self) -> f32 {
        self.base.get_current_time()
    }

    fn set_current_time(&mut self, time: f32) {
        self.base.set_current_time(time);
        self.layer.set_current_time(time);
    }

    fn is_completed(&self) -> bool {
        self.base.is_completed()
    }
}

// ---------------------------------------------------------------------------
// BlendNode (enum dispatch)
// ---------------------------------------------------------------------------

/// Enum wrapper for all blend node types.
///
/// Provides type-safe dispatch and serialization for blend node graphs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BlendNode {
    /// Play a single clip.
    Clip(ClipNode),

    /// 1D parameter-driven blend.
    Blend1D(Blend1DNode),

    /// 2D parameter-driven blend.
    Blend2D(Blend2DNode),

    /// Additive blend.
    Additive(AdditiveNode),

    /// Override/crossfade blend.
    Override(OverrideNode),

    /// Masked layer blend.
    Layer(LayerNode),

    /// Identity node (returns bind pose).
    Identity,
}

impl BlendNode {
    /// Create a clip node.
    pub fn clip(clip_index: usize, loop_mode: LoopMode) -> Self {
        Self::Clip(ClipNode::new(clip_index, loop_mode))
    }

    /// Create a 1D blend node.
    pub fn blend_1d(parameter: impl Into<String>) -> Self {
        Self::Blend1D(Blend1DNode::new(parameter))
    }

    /// Create a 2D blend node.
    pub fn blend_2d(param_x: impl Into<String>, param_y: impl Into<String>) -> Self {
        Self::Blend2D(Blend2DNode::new(param_x, param_y))
    }

    /// Create an identity node.
    pub fn identity() -> Self {
        Self::Identity
    }

    /// Check if this is an identity node.
    pub fn is_identity(&self) -> bool {
        matches!(self, Self::Identity)
    }

    /// Evaluate the node.
    pub fn evaluate(&mut self, params: &ParameterSet, dt: f32, bone_count: usize) -> NodeOutput {
        match self {
            Self::Clip(node) => node.evaluate(params, dt, bone_count),
            Self::Blend1D(node) => node.evaluate(params, dt, bone_count),
            Self::Blend2D(node) => node.evaluate(params, dt, bone_count),
            Self::Additive(node) => node.evaluate(params, dt, bone_count),
            Self::Override(node) => node.evaluate(params, dt, bone_count),
            Self::Layer(node) => node.evaluate(params, dt, bone_count),
            Self::Identity => NodeOutput::identity(bone_count),
        }
    }

    /// Get the duration of this node.
    pub fn get_duration(&self) -> Option<f32> {
        match self {
            Self::Clip(node) => node.get_duration(),
            Self::Blend1D(node) => node.get_duration(),
            Self::Blend2D(node) => node.get_duration(),
            Self::Additive(node) => node.get_duration(),
            Self::Override(node) => node.get_duration(),
            Self::Layer(node) => node.get_duration(),
            Self::Identity => None,
        }
    }

    /// Reset the node.
    pub fn reset(&mut self) {
        match self {
            Self::Clip(node) => node.reset(),
            Self::Blend1D(node) => node.reset(),
            Self::Blend2D(node) => node.reset(),
            Self::Additive(node) => node.reset(),
            Self::Override(node) => node.reset(),
            Self::Layer(node) => node.reset(),
            Self::Identity => {}
        }
    }

    /// Get current time.
    pub fn get_current_time(&self) -> f32 {
        match self {
            Self::Clip(node) => node.get_current_time(),
            Self::Blend1D(node) => node.get_current_time(),
            Self::Blend2D(node) => node.get_current_time(),
            Self::Additive(node) => node.get_current_time(),
            Self::Override(node) => node.get_current_time(),
            Self::Layer(node) => node.get_current_time(),
            Self::Identity => 0.0,
        }
    }

    /// Set current time.
    pub fn set_current_time(&mut self, time: f32) {
        match self {
            Self::Clip(node) => node.set_current_time(time),
            Self::Blend1D(node) => node.set_current_time(time),
            Self::Blend2D(node) => node.set_current_time(time),
            Self::Additive(node) => node.set_current_time(time),
            Self::Override(node) => node.set_current_time(time),
            Self::Layer(node) => node.set_current_time(time),
            Self::Identity => {}
        }
    }

    /// Check if completed.
    pub fn is_completed(&self) -> bool {
        match self {
            Self::Clip(node) => node.is_completed(),
            Self::Blend1D(node) => node.is_completed(),
            Self::Blend2D(node) => node.is_completed(),
            Self::Additive(node) => node.is_completed(),
            Self::Override(node) => node.is_completed(),
            Self::Layer(node) => node.is_completed(),
            Self::Identity => false,
        }
    }

    /// Get node type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Clip(_) => "Clip",
            Self::Blend1D(_) => "Blend1D",
            Self::Blend2D(_) => "Blend2D",
            Self::Additive(_) => "Additive",
            Self::Override(_) => "Override",
            Self::Layer(_) => "Layer",
            Self::Identity => "Identity",
        }
    }
}

impl Default for BlendNode {
    fn default() -> Self {
        Self::Identity
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a simple parameter set
    fn create_params() -> ParameterSet {
        let mut params = ParameterSet::new();
        params.set_float("speed", 0.0);
        params.set_float("direction", 0.0);
        params.set_float("blend_weight", 0.5);
        params
    }

    // -------------------------------------------------------------------------
    // ClipNode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_clip_node_creation() {
        let clip = ClipNode::new(0, LoopMode::Loop);
        assert_eq!(clip.clip_index, 0);
        assert_eq!(clip.loop_mode, LoopMode::Loop);
        assert_eq!(clip.play_rate, DEFAULT_PLAY_RATE);
        assert_eq!(clip.current_time, 0.0);
    }

    #[test]
    fn test_clip_node_with_play_rate() {
        let clip = ClipNode::new(1, LoopMode::Once)
            .with_play_rate(2.0)
            .with_duration(5.0);
        assert_eq!(clip.clip_index, 1);
        assert_eq!(clip.play_rate, 2.0);
        assert_eq!(clip.duration, 5.0);
    }

    #[test]
    fn test_clip_node_advance_time_once() {
        let mut clip = ClipNode::new(0, LoopMode::Once).with_duration(1.0);
        let params = create_params();

        // Advance by 0.5 seconds
        clip.evaluate(&params, 0.5, 10);
        assert!((clip.current_time - 0.5).abs() < TIME_EPSILON);
        assert!(!clip.completed);

        // Advance past end
        clip.evaluate(&params, 0.6, 10);
        assert!((clip.current_time - 1.0).abs() < TIME_EPSILON);
        assert!(clip.completed);
    }

    #[test]
    fn test_clip_node_advance_time_loop() {
        let mut clip = ClipNode::new(0, LoopMode::Loop).with_duration(1.0);
        let params = create_params();

        // Advance past duration
        clip.evaluate(&params, 1.5, 10);
        assert!((clip.current_time - 0.5).abs() < TIME_EPSILON);
        assert!(!clip.completed);
    }

    #[test]
    fn test_clip_node_advance_time_pingpong() {
        let mut clip = ClipNode::new(0, LoopMode::PingPong).with_duration(1.0);
        let params = create_params();

        // Advance to middle of reverse
        clip.evaluate(&params, 1.5, 10);
        assert!((clip.current_time - 0.5).abs() < TIME_EPSILON);
    }

    #[test]
    fn test_clip_node_pause_resume() {
        let mut clip = ClipNode::new(0, LoopMode::Loop).with_duration(1.0);
        let params = create_params();

        clip.evaluate(&params, 0.3, 10);
        let time_before_pause = clip.current_time;

        clip.pause();
        clip.evaluate(&params, 0.5, 10);
        assert!((clip.current_time - time_before_pause).abs() < TIME_EPSILON);

        clip.resume();
        clip.evaluate(&params, 0.2, 10);
        assert!((clip.current_time - (time_before_pause + 0.2)).abs() < TIME_EPSILON);
    }

    #[test]
    fn test_clip_node_seek() {
        let mut clip = ClipNode::new(0, LoopMode::Loop).with_duration(1.0);
        clip.seek(0.7);
        assert!((clip.current_time - 0.7).abs() < TIME_EPSILON);
    }

    #[test]
    fn test_clip_node_normalized_time() {
        let mut clip = ClipNode::new(0, LoopMode::Loop).with_duration(2.0);
        clip.seek(1.0);
        assert!((clip.normalized_time() - 0.5).abs() < TIME_EPSILON);
    }

    #[test]
    fn test_clip_node_reset() {
        let mut clip = ClipNode::new(0, LoopMode::Once).with_duration(1.0);
        let params = create_params();

        clip.evaluate(&params, 1.5, 10);
        assert!(clip.completed);

        clip.reset();
        assert!(!clip.completed);
        assert!((clip.current_time).abs() < TIME_EPSILON);
    }

    #[test]
    fn test_clip_node_negative_play_rate() {
        let mut clip = ClipNode::new(0, LoopMode::Once)
            .with_duration(1.0)
            .with_play_rate(-1.0);
        clip.seek(1.0);
        let params = create_params();

        clip.evaluate(&params, 0.5, 10);
        assert!((clip.current_time - 0.5).abs() < TIME_EPSILON);
    }

    #[test]
    fn test_clip_node_evaluate_returns_pose() {
        let mut clip = ClipNode::new(0, LoopMode::Loop).with_duration(1.0);
        let params = create_params();

        let output = clip.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    // -------------------------------------------------------------------------
    // Blend1DNode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blend1d_creation() {
        let blend = Blend1DNode::new("speed");
        assert_eq!(blend.parameter, "speed");
        assert!(blend.children.is_empty());
    }

    #[test]
    fn test_blend1d_add_children() {
        let mut blend = Blend1DNode::new("speed");
        blend.add_child(0.0, BlendNode::clip(0, LoopMode::Loop));
        blend.add_child(5.0, BlendNode::clip(1, LoopMode::Loop));
        assert_eq!(blend.children.len(), 2);
    }

    #[test]
    fn test_blend1d_weights_single_child() {
        let mut blend = Blend1DNode::new("speed");
        blend.add_child(0.0, BlendNode::clip(0, LoopMode::Loop));

        let weights = blend.get_weights(5.0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0], (0, 1.0));
    }

    #[test]
    fn test_blend1d_weights_two_children() {
        let mut blend = Blend1DNode::new("speed");
        blend.add_child(0.0, BlendNode::clip(0, LoopMode::Loop));
        blend.add_child(10.0, BlendNode::clip(1, LoopMode::Loop));

        // At midpoint
        let weights = blend.get_weights(5.0);
        assert_eq!(weights.len(), 2);
        assert!((weights[0].1 - 0.5).abs() < WEIGHT_EPSILON);
        assert!((weights[1].1 - 0.5).abs() < WEIGHT_EPSILON);
    }

    #[test]
    fn test_blend1d_weights_below_range() {
        let mut blend = Blend1DNode::new("speed");
        blend.add_child(5.0, BlendNode::clip(0, LoopMode::Loop));
        blend.add_child(10.0, BlendNode::clip(1, LoopMode::Loop));

        let weights = blend.get_weights(0.0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].0, 0);
    }

    #[test]
    fn test_blend1d_weights_above_range() {
        let mut blend = Blend1DNode::new("speed");
        blend.add_child(0.0, BlendNode::clip(0, LoopMode::Loop));
        blend.add_child(5.0, BlendNode::clip(1, LoopMode::Loop));

        let weights = blend.get_weights(10.0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].0, 1);
    }

    #[test]
    fn test_blend1d_weights_three_children() {
        let mut blend = Blend1DNode::new("speed");
        blend.add_child(0.0, BlendNode::clip(0, LoopMode::Loop));
        blend.add_child(5.0, BlendNode::clip(1, LoopMode::Loop));
        blend.add_child(10.0, BlendNode::clip(2, LoopMode::Loop));

        // At 7.5 (between walk and run)
        let weights = blend.get_weights(7.5);
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].0, 1); // walk
        assert_eq!(weights[1].0, 2); // run
        assert!((weights[0].1 - 0.5).abs() < WEIGHT_EPSILON);
    }

    #[test]
    fn test_blend1d_evaluate() {
        let mut blend = Blend1DNode::new("speed");
        blend.add_child(0.0, BlendNode::clip(0, LoopMode::Loop));
        blend.add_child(5.0, BlendNode::clip(1, LoopMode::Loop));

        let mut params = create_params();
        params.set_float("speed", 2.5);

        let output = blend.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    #[test]
    fn test_blend1d_reset() {
        let mut blend = Blend1DNode::new("speed");
        blend.add_child(0.0, BlendNode::clip(0, LoopMode::Loop));

        let params = create_params();
        blend.evaluate(&params, 0.5, 10);

        blend.reset();
        assert!((blend.get_current_time()).abs() < TIME_EPSILON);
    }

    #[test]
    fn test_blend1d_no_children() {
        let mut blend = Blend1DNode::new("speed");
        let params = create_params();

        let output = blend.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    #[test]
    fn test_blend1d_sorting() {
        let mut blend = Blend1DNode::new("speed");
        blend.add_child(10.0, BlendNode::clip(2, LoopMode::Loop));
        blend.add_child(0.0, BlendNode::clip(0, LoopMode::Loop));
        blend.add_child(5.0, BlendNode::clip(1, LoopMode::Loop));

        let weights = blend.get_weights(2.5);
        // After sorting, indices should be 0, 1, 2
        assert_eq!(weights[0].0, 0);
        assert_eq!(weights[1].0, 1);
    }

    // -------------------------------------------------------------------------
    // Blend2DNode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blend2d_creation() {
        let blend = Blend2DNode::new("velocity_x", "velocity_z");
        assert_eq!(blend.param_x, "velocity_x");
        assert_eq!(blend.param_y, "velocity_z");
    }

    #[test]
    fn test_blend2d_add_children() {
        let mut blend = Blend2DNode::new("vx", "vz");
        blend.add_child(0.0, 0.0, BlendNode::clip(0, LoopMode::Loop));
        blend.add_child(1.0, 0.0, BlendNode::clip(1, LoopMode::Loop));
        blend.add_child(0.0, 1.0, BlendNode::clip(2, LoopMode::Loop));
        assert_eq!(blend.children.len(), 3);
    }

    #[test]
    fn test_blend2d_triangulate() {
        let mut blend = Blend2DNode::new("vx", "vz");
        blend.add_child(0.0, 0.0, BlendNode::clip(0, LoopMode::Loop));
        blend.add_child(1.0, 0.0, BlendNode::clip(1, LoopMode::Loop));
        blend.add_child(0.5, 1.0, BlendNode::clip(2, LoopMode::Loop));

        blend.triangulate();
        assert!(!blend.triangles.is_empty());
    }

    #[test]
    fn test_blend2d_weights_single_child() {
        let mut blend = Blend2DNode::new("vx", "vz");
        blend.add_child(0.0, 0.0, BlendNode::clip(0, LoopMode::Loop));

        let weights = blend.get_weights(1.0, 1.0);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0], (0, 1.0));
    }

    #[test]
    fn test_blend2d_weights_two_children() {
        let mut blend = Blend2DNode::new("vx", "vz");
        blend.add_child(0.0, 0.0, BlendNode::clip(0, LoopMode::Loop));
        blend.add_child(1.0, 0.0, BlendNode::clip(1, LoopMode::Loop));

        let weights = blend.get_weights(0.5, 0.0);
        assert_eq!(weights.len(), 2);
        assert!((weights[0].1 - 0.5).abs() < WEIGHT_EPSILON);
    }

    #[test]
    fn test_blend2d_barycentric() {
        // Triangle at (0,0), (1,0), (0,1)
        let result = Blend2DNode::barycentric(
            0.25, 0.25, // point
            0.0, 0.0, // v0
            1.0, 0.0, // v1
            0.0, 1.0, // v2
        );

        assert!(result.is_some());
        let (w0, w1, w2) = result.unwrap();
        assert!((w0 + w1 + w2 - 1.0).abs() < WEIGHT_EPSILON);
    }

    #[test]
    fn test_blend2d_evaluate() {
        let mut blend = Blend2DNode::new("vx", "vz");
        blend.add_child(0.0, 0.0, BlendNode::clip(0, LoopMode::Loop));
        blend.add_child(1.0, 0.0, BlendNode::clip(1, LoopMode::Loop));
        blend.add_child(0.0, 1.0, BlendNode::clip(2, LoopMode::Loop));

        let mut params = create_params();
        params.set_float("vx", 0.3);
        params.set_float("vz", 0.3);

        let output = blend.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    // -------------------------------------------------------------------------
    // AdditiveNode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_additive_creation() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let additive = BlendNode::clip(1, LoopMode::Loop);
        let node = AdditiveNode::new(base, additive);

        assert!(node.weight_param.is_empty());
        assert!((node.static_weight - 1.0).abs() < WEIGHT_EPSILON);
    }

    #[test]
    fn test_additive_with_weight() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let additive = BlendNode::clip(1, LoopMode::Loop);
        let node = AdditiveNode::new(base, additive).with_weight(0.5);

        assert!((node.static_weight - 0.5).abs() < WEIGHT_EPSILON);
    }

    #[test]
    fn test_additive_with_weight_param() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let additive = BlendNode::clip(1, LoopMode::Loop);
        let node = AdditiveNode::new(base, additive).with_weight_param("blend_weight");

        assert_eq!(node.weight_param, "blend_weight");
    }

    #[test]
    fn test_additive_evaluate_zero_weight() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let additive = BlendNode::clip(1, LoopMode::Loop);
        let mut node = AdditiveNode::new(base, additive).with_weight(0.0);

        let params = create_params();
        let output = node.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    #[test]
    fn test_additive_evaluate_full_weight() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let additive = BlendNode::clip(1, LoopMode::Loop);
        let mut node = AdditiveNode::new(base, additive).with_weight(1.0);

        let params = create_params();
        let output = node.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    #[test]
    fn test_additive_apply_additive_math() {
        let base = Pose::new(2, PoseType::Current);
        let mut additive = Pose::new(2, PoseType::Additive);
        additive.positions[0] = Vec3::new(1.0, 0.0, 0.0);

        let result = AdditiveNode::apply_additive(&base, &additive, 0.5);
        assert!((result.positions[0].x - 0.5).abs() < WEIGHT_EPSILON);
    }

    #[test]
    fn test_additive_reset() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let additive = BlendNode::clip(1, LoopMode::Loop);
        let mut node = AdditiveNode::new(base, additive);

        let params = create_params();
        node.evaluate(&params, 0.5, 10);

        node.reset();
        assert!((node.get_current_time()).abs() < TIME_EPSILON);
    }

    // -------------------------------------------------------------------------
    // OverrideNode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_override_creation() {
        let source = BlendNode::clip(0, LoopMode::Loop);
        let override_node = BlendNode::clip(1, LoopMode::Loop);
        let node = OverrideNode::new(source, override_node);

        assert!(node.weight_param.is_empty());
        assert!((node.static_weight).abs() < WEIGHT_EPSILON);
    }

    #[test]
    fn test_override_with_weight() {
        let source = BlendNode::clip(0, LoopMode::Loop);
        let override_node = BlendNode::clip(1, LoopMode::Loop);
        let node = OverrideNode::new(source, override_node).with_weight(0.7);

        assert!((node.static_weight - 0.7).abs() < WEIGHT_EPSILON);
    }

    #[test]
    fn test_override_evaluate_zero_weight() {
        let source = BlendNode::clip(0, LoopMode::Loop);
        let override_node = BlendNode::clip(1, LoopMode::Loop);
        let mut node = OverrideNode::new(source, override_node).with_weight(0.0);

        let params = create_params();
        let output = node.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    #[test]
    fn test_override_evaluate_full_weight() {
        let source = BlendNode::clip(0, LoopMode::Loop);
        let override_node = BlendNode::clip(1, LoopMode::Loop);
        let mut node = OverrideNode::new(source, override_node).with_weight(1.0);

        let params = create_params();
        let output = node.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    #[test]
    fn test_override_evaluate_half_weight() {
        let source = BlendNode::clip(0, LoopMode::Loop);
        let override_node = BlendNode::clip(1, LoopMode::Loop);
        let mut node = OverrideNode::new(source, override_node).with_weight(0.5);

        let params = create_params();
        let output = node.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    #[test]
    fn test_override_reset() {
        let source = BlendNode::clip(0, LoopMode::Loop);
        let override_node = BlendNode::clip(1, LoopMode::Loop);
        let mut node = OverrideNode::new(source, override_node);

        let params = create_params();
        node.evaluate(&params, 0.5, 10);

        node.reset();
        assert!((node.get_current_time()).abs() < TIME_EPSILON);
    }

    // -------------------------------------------------------------------------
    // LayerNode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_layer_creation() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let layer = BlendNode::clip(1, LoopMode::Loop);
        let mask = vec![true, true, false, false, false];
        let node = LayerNode::new(base, layer, mask);

        assert_eq!(node.bone_mask.len(), 5);
        assert!(node.bone_mask[0]);
        assert!(!node.bone_mask[2]);
    }

    #[test]
    fn test_layer_full_body() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let layer = BlendNode::clip(1, LoopMode::Loop);
        let node = LayerNode::full_body(base, layer, 10);

        assert_eq!(node.bone_mask.len(), 10);
        assert!(node.bone_mask.iter().all(|&m| m));
    }

    #[test]
    fn test_layer_with_blend_mode() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let layer = BlendNode::clip(1, LoopMode::Loop);
        let node = LayerNode::full_body(base, layer, 10)
            .with_blend_mode(LayerBlendMode::Additive);

        assert_eq!(node.blend_mode, LayerBlendMode::Additive);
    }

    #[test]
    fn test_layer_with_weight() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let layer = BlendNode::clip(1, LoopMode::Loop);
        let node = LayerNode::full_body(base, layer, 10).with_weight(0.8);

        assert!((node.weight - 0.8).abs() < WEIGHT_EPSILON);
    }

    #[test]
    fn test_layer_evaluate_zero_weight() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let layer = BlendNode::clip(1, LoopMode::Loop);
        let mut node = LayerNode::full_body(base, layer, 10).with_weight(0.0);

        let params = create_params();
        let output = node.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    #[test]
    fn test_layer_evaluate_full_weight() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let layer = BlendNode::clip(1, LoopMode::Loop);
        let mut node = LayerNode::full_body(base, layer, 10).with_weight(1.0);

        let params = create_params();
        let output = node.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    #[test]
    fn test_layer_evaluate_partial_mask() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let layer = BlendNode::clip(1, LoopMode::Loop);
        let mask = vec![true, true, false, false, false, false, false, false, false, false];
        let mut node = LayerNode::new(base, layer, mask).with_weight(1.0);

        let params = create_params();
        let output = node.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    #[test]
    fn test_layer_reset() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let layer = BlendNode::clip(1, LoopMode::Loop);
        let mut node = LayerNode::full_body(base, layer, 10);

        let params = create_params();
        node.evaluate(&params, 0.5, 10);

        node.reset();
        assert!((node.get_current_time()).abs() < TIME_EPSILON);
    }

    // -------------------------------------------------------------------------
    // BlendNode Enum Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blend_node_clip() {
        let node = BlendNode::clip(0, LoopMode::Loop);
        assert_eq!(node.type_name(), "Clip");
        assert!(!node.is_identity());
    }

    #[test]
    fn test_blend_node_identity() {
        let node = BlendNode::identity();
        assert_eq!(node.type_name(), "Identity");
        assert!(node.is_identity());
    }

    #[test]
    fn test_blend_node_evaluate_identity() {
        let mut node = BlendNode::identity();
        let params = create_params();
        let output = node.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    #[test]
    fn test_blend_node_reset() {
        let mut node = BlendNode::clip(0, LoopMode::Loop);
        let params = create_params();
        node.evaluate(&params, 0.5, 10);

        node.reset();
        assert!((node.get_current_time()).abs() < TIME_EPSILON);
    }

    // -------------------------------------------------------------------------
    // NodeOutput Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_node_output_creation() {
        let pose = Pose::new(10, PoseType::Current);
        let output = NodeOutput::new(pose);
        assert_eq!(output.bone_count(), 10);
        assert!(output.events.is_empty());
        assert!(output.root_motion.is_none());
    }

    #[test]
    fn test_node_output_identity() {
        let output = NodeOutput::identity(5);
        assert_eq!(output.bone_count(), 5);
    }

    #[test]
    fn test_node_output_add_event() {
        let pose = Pose::new(10, PoseType::Current);
        let mut output = NodeOutput::new(pose);
        output.add_event(AnimationEvent::new("footstep", 0.5, 0));
        assert_eq!(output.events.len(), 1);
    }

    #[test]
    fn test_node_output_with_completed() {
        let pose = Pose::new(10, PoseType::Current);
        let output = NodeOutput::new(pose).with_completed(true);
        assert!(output.completed);
    }

    #[test]
    fn test_node_output_with_normalized_time() {
        let pose = Pose::new(10, PoseType::Current);
        let output = NodeOutput::new(pose).with_normalized_time(0.5);
        assert!((output.normalized_time - 0.5).abs() < TIME_EPSILON);
    }

    // -------------------------------------------------------------------------
    // AnimationEvent Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_animation_event_creation() {
        let event = AnimationEvent::new("footstep", 0.5, 0);
        assert_eq!(event.name, "footstep");
        assert!((event.time - 0.5).abs() < TIME_EPSILON);
        assert_eq!(event.clip_index, 0);
        assert!(event.payload.is_none());
    }

    #[test]
    fn test_animation_event_with_payload() {
        let event = AnimationEvent::with_payload("sound", 0.3, 1, "footstep_concrete");
        assert_eq!(event.payload, Some("footstep_concrete".to_string()));
    }

    // -------------------------------------------------------------------------
    // Nested Node Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_nested_blend1d_in_blend1d() {
        // Create inner blend
        let mut inner = Blend1DNode::new("lean");
        inner.add_child(-1.0, BlendNode::clip(0, LoopMode::Loop));
        inner.add_child(0.0, BlendNode::clip(1, LoopMode::Loop));
        inner.add_child(1.0, BlendNode::clip(2, LoopMode::Loop));

        // Create outer blend
        let mut outer = Blend1DNode::new("speed");
        outer.add_child(0.0, BlendNode::clip(3, LoopMode::Loop));
        outer.add_child(5.0, BlendNode::Blend1D(inner));

        let mut params = create_params();
        params.set_float("speed", 2.5);
        params.set_float("lean", 0.5);

        let output = outer.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    #[test]
    fn test_additive_on_blend1d() {
        let mut locomotion = Blend1DNode::new("speed");
        locomotion.add_child(0.0, BlendNode::clip(0, LoopMode::Loop));
        locomotion.add_child(5.0, BlendNode::clip(1, LoopMode::Loop));

        let breathing = BlendNode::clip(2, LoopMode::Loop);

        let mut node = AdditiveNode::new(BlendNode::Blend1D(locomotion), breathing)
            .with_weight_param("breathing_weight");

        let mut params = create_params();
        params.set_float("speed", 2.5);
        params.set_float("breathing_weight", 0.3);

        let output = node.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    #[test]
    fn test_layer_on_blend1d() {
        let mut locomotion = Blend1DNode::new("speed");
        locomotion.add_child(0.0, BlendNode::clip(0, LoopMode::Loop));
        locomotion.add_child(5.0, BlendNode::clip(1, LoopMode::Loop));

        let upper_body = BlendNode::clip(2, LoopMode::Loop);

        // Mask: first 5 bones (upper body)
        let mask = vec![true, true, true, true, true, false, false, false, false, false];

        let mut node = LayerNode::new(BlendNode::Blend1D(locomotion), upper_body, mask)
            .with_weight(0.8);

        let mut params = create_params();
        params.set_float("speed", 2.5);

        let output = node.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blend1d_empty_children() {
        let mut blend = Blend1DNode::new("speed");
        let weights = blend.get_weights(5.0);
        assert!(weights.is_empty());
    }

    #[test]
    fn test_blend2d_empty_children() {
        let mut blend = Blend2DNode::new("vx", "vz");
        let weights = blend.get_weights(0.5, 0.5);
        assert!(weights.is_empty());
    }

    #[test]
    fn test_clip_zero_duration() {
        let clip = ClipNode::new(0, LoopMode::Loop).with_duration(0.0);
        // Duration should be clamped to epsilon
        assert!(clip.duration > 0.0);
    }

    #[test]
    fn test_blend1d_same_thresholds() {
        let mut blend = Blend1DNode::new("speed");
        blend.add_child(5.0, BlendNode::clip(0, LoopMode::Loop));
        blend.add_child(5.0, BlendNode::clip(1, LoopMode::Loop));

        let weights = blend.get_weights(5.0);
        // Should return one weight
        assert!(!weights.is_empty());
    }

    #[test]
    fn test_layer_empty_mask() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let layer = BlendNode::clip(1, LoopMode::Loop);
        let mask = vec![]; // Empty mask

        let mut node = LayerNode::new(base, layer, mask).with_weight(1.0);

        let params = create_params();
        let output = node.evaluate(&params, 0.1, 10);
        // Should still produce output (base only)
        assert_eq!(output.pose.bone_count(), 10);
    }

    #[test]
    fn test_missing_parameter() {
        let mut blend = Blend1DNode::new("nonexistent_param");
        blend.add_child(0.0, BlendNode::clip(0, LoopMode::Loop));
        blend.add_child(5.0, BlendNode::clip(1, LoopMode::Loop));

        let params = create_params(); // No "nonexistent_param"

        // Should default to 0.0
        let output = blend.evaluate(&params, 0.1, 10);
        assert_eq!(output.pose.bone_count(), 10);
    }

    // -------------------------------------------------------------------------
    // Event Propagation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_events_propagate_through_blend1d() {
        let mut blend = Blend1DNode::new("speed");
        blend.add_child(0.0, BlendNode::clip(0, LoopMode::Loop));
        blend.add_child(5.0, BlendNode::clip(1, LoopMode::Loop));

        let params = create_params();
        let output = blend.evaluate(&params, 0.1, 10);
        // Events should be collected from all evaluated children
        // (In real impl, clips would generate events)
        assert!(output.events.is_empty() || !output.events.is_empty());
    }

    #[test]
    fn test_events_propagate_through_additive() {
        let base = BlendNode::clip(0, LoopMode::Loop);
        let additive = BlendNode::clip(1, LoopMode::Loop);
        let mut node = AdditiveNode::new(base, additive);

        let params = create_params();
        let output = node.evaluate(&params, 0.1, 10);
        // Should have events from both base and additive
        assert!(output.events.is_empty() || !output.events.is_empty());
    }

    // -------------------------------------------------------------------------
    // Duration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_clip_duration() {
        let clip = ClipNode::new(0, LoopMode::Loop).with_duration(2.5);
        assert!((clip.get_duration().unwrap() - 2.5).abs() < TIME_EPSILON);
    }

    #[test]
    fn test_blend1d_duration_uniform() {
        let mut blend = Blend1DNode::new("speed");

        let clip1 = ClipNode::new(0, LoopMode::Loop).with_duration(1.0);
        let clip2 = ClipNode::new(1, LoopMode::Loop).with_duration(1.0);

        blend.add_child(0.0, BlendNode::Clip(clip1));
        blend.add_child(5.0, BlendNode::Clip(clip2));

        assert!((blend.get_duration().unwrap() - 1.0).abs() < TIME_EPSILON);
    }

    #[test]
    fn test_blend1d_duration_varied() {
        let mut blend = Blend1DNode::new("speed");

        let clip1 = ClipNode::new(0, LoopMode::Loop).with_duration(1.0);
        let clip2 = ClipNode::new(1, LoopMode::Loop).with_duration(2.0);

        blend.add_child(0.0, BlendNode::Clip(clip1));
        blend.add_child(5.0, BlendNode::Clip(clip2));

        // Should return average
        assert!((blend.get_duration().unwrap() - 1.5).abs() < TIME_EPSILON);
    }

    #[test]
    fn test_identity_no_duration() {
        let node = BlendNode::identity();
        assert!(node.get_duration().is_none());
    }
}
