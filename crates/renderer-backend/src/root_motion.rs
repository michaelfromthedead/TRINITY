//! Root motion extraction and application for skeletal animation (T-AN-2.3).
//!
//! This module provides root motion handling for character locomotion:
//!
//! - Extract root bone delta from animation clips (position and rotation)
//! - Multiple motion modes: animation-driven, physics-driven, and blended
//! - Component separation: horizontal (XZ), vertical (Y), and rotation
//! - Accumulation tracking with loop handling and reset capabilities
//!
//! # Architecture
//!
//! ```text
//! AnimationClip --> RootMotionExtractor --> RootMotionDelta
//!                                               |
//!                                               v
//!                                      RootMotionAccumulator
//!                                               |
//!                                               v
//!                                          Transform
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::root_motion::{
//!     RootMotionAccumulator, RootMotionConfig, RootMotionMode, RootMotionDelta,
//! };
//! use renderer_backend::pose::Pose;
//!
//! // Configure root motion extraction
//! let config = RootMotionConfig {
//!     mode: RootMotionMode::AnimationDriven,
//!     extract_horizontal: true,
//!     extract_vertical: false,
//!     extract_rotation: true,
//!     root_bone: 0,
//! };
//!
//! // Create accumulator
//! let mut accumulator = RootMotionAccumulator::new(config);
//!
//! // Extract delta from current pose
//! let delta = accumulator.extract_delta(&pose);
//!
//! // Apply accumulated motion to character transform
//! accumulator.apply_to_transform(&mut character_transform);
//! ```

use std::f32::consts::PI;

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::animation_clip::{AnimationClip, BoneTrack, Keyframe, Track};
use crate::pose::Pose;
use crate::skeleton::Transform;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Epsilon for floating point comparisons.
pub const ROOT_MOTION_EPSILON: f32 = 1e-6;

/// Maximum rotation delta per frame (radians) to prevent discontinuities.
pub const MAX_ROTATION_DELTA: f32 = PI;

// ---------------------------------------------------------------------------
// RootMotionMode
// ---------------------------------------------------------------------------

/// How root motion affects character movement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum RootMotionMode {
    /// Animation fully controls character movement.
    /// The root bone delta is extracted and applied to the character transform.
    #[default]
    AnimationDriven,

    /// Gameplay/physics code controls character movement.
    /// Root motion is ignored; animation plays in place.
    PhysicsDriven,

    /// Partial animation, partial physics control.
    /// The animation_weight determines the blend (0.0 = physics, 1.0 = animation).
    Blended {
        /// Weight of animation-driven motion (0.0 to 1.0).
        animation_weight: f32,
    },
}

impl RootMotionMode {
    /// Create a blended mode with the given animation weight.
    #[inline]
    pub fn blended(animation_weight: f32) -> Self {
        Self::Blended {
            animation_weight: animation_weight.clamp(0.0, 1.0),
        }
    }

    /// Get the effective animation weight for this mode.
    #[inline]
    pub fn animation_weight(&self) -> f32 {
        match self {
            Self::AnimationDriven => 1.0,
            Self::PhysicsDriven => 0.0,
            Self::Blended { animation_weight } => *animation_weight,
        }
    }

    /// Check if this mode uses any animation-driven motion.
    #[inline]
    pub fn uses_animation(&self) -> bool {
        !matches!(self, Self::PhysicsDriven)
    }

    /// Check if this mode uses any physics-driven motion.
    #[inline]
    pub fn uses_physics(&self) -> bool {
        !matches!(self, Self::AnimationDriven)
    }
}

// Implement Eq for RootMotionMode
// We need custom implementation because of the f32 in Blended
impl Eq for RootMotionMode {}

// ---------------------------------------------------------------------------
// RootMotionConfig
// ---------------------------------------------------------------------------

/// Configuration for root motion extraction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RootMotionConfig {
    /// How root motion affects character movement.
    pub mode: RootMotionMode,

    /// Extract horizontal motion (XZ plane).
    pub extract_horizontal: bool,

    /// Extract vertical motion (Y axis).
    pub extract_vertical: bool,

    /// Extract rotation (yaw by default, full rotation if extract_full_rotation is true).
    pub extract_rotation: bool,

    /// Extract full 3D rotation instead of just yaw.
    pub extract_full_rotation: bool,

    /// Index of the root bone in the skeleton.
    pub root_bone: usize,
}

impl Default for RootMotionConfig {
    fn default() -> Self {
        Self {
            mode: RootMotionMode::AnimationDriven,
            extract_horizontal: true,
            extract_vertical: true,
            extract_rotation: true,
            extract_full_rotation: false,
            root_bone: 0,
        }
    }
}

impl RootMotionConfig {
    /// Create a new config with default settings for the given root bone.
    #[inline]
    pub fn new(root_bone: usize) -> Self {
        Self {
            root_bone,
            ..Default::default()
        }
    }

    /// Configure for horizontal-only motion (typical for ground locomotion).
    #[inline]
    pub fn horizontal_only(root_bone: usize) -> Self {
        Self {
            mode: RootMotionMode::AnimationDriven,
            extract_horizontal: true,
            extract_vertical: false,
            extract_rotation: true,
            extract_full_rotation: false,
            root_bone,
        }
    }

    /// Configure for full 3D motion (flying, swimming).
    #[inline]
    pub fn full_3d(root_bone: usize) -> Self {
        Self {
            mode: RootMotionMode::AnimationDriven,
            extract_horizontal: true,
            extract_vertical: true,
            extract_rotation: true,
            extract_full_rotation: true,
            root_bone,
        }
    }

    /// Set the motion mode.
    #[inline]
    pub fn with_mode(mut self, mode: RootMotionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set horizontal extraction.
    #[inline]
    pub fn with_horizontal(mut self, extract: bool) -> Self {
        self.extract_horizontal = extract;
        self
    }

    /// Set vertical extraction.
    #[inline]
    pub fn with_vertical(mut self, extract: bool) -> Self {
        self.extract_vertical = extract;
        self
    }

    /// Set rotation extraction.
    #[inline]
    pub fn with_rotation(mut self, extract: bool) -> Self {
        self.extract_rotation = extract;
        self
    }

    /// Set full rotation extraction.
    #[inline]
    pub fn with_full_rotation(mut self, full: bool) -> Self {
        self.extract_full_rotation = full;
        self
    }
}

// ---------------------------------------------------------------------------
// RootMotionDelta
// ---------------------------------------------------------------------------

/// Delta motion extracted from a single frame or time range.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RootMotionDelta {
    /// Translation delta in local space.
    pub translation: Vec3,

    /// Rotation delta as a quaternion.
    pub rotation: Quat,
}

impl Default for RootMotionDelta {
    fn default() -> Self {
        Self::ZERO
    }
}

impl RootMotionDelta {
    /// Zero delta (no motion).
    pub const ZERO: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    /// Create a new delta with translation only.
    #[inline]
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            rotation: Quat::IDENTITY,
        }
    }

    /// Create a new delta with rotation only.
    #[inline]
    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation,
        }
    }

    /// Create a new delta with both translation and rotation.
    #[inline]
    pub fn new(translation: Vec3, rotation: Quat) -> Self {
        Self {
            translation,
            rotation,
        }
    }

    /// Check if this delta is approximately zero.
    #[inline]
    pub fn is_zero(&self, epsilon: f32) -> bool {
        self.translation.length_squared() < epsilon * epsilon
            && (self.rotation.dot(Quat::IDENTITY).abs() > 1.0 - epsilon)
    }

    /// Get the horizontal (XZ) component of the translation.
    #[inline]
    pub fn horizontal_translation(&self) -> Vec3 {
        Vec3::new(self.translation.x, 0.0, self.translation.z)
    }

    /// Get the vertical (Y) component of the translation.
    #[inline]
    pub fn vertical_translation(&self) -> Vec3 {
        Vec3::new(0.0, self.translation.y, 0.0)
    }

    /// Extract yaw rotation only (rotation around Y axis).
    #[inline]
    pub fn yaw_rotation(&self) -> Quat {
        // Decompose to euler angles and extract yaw
        let (yaw, _pitch, _roll) = self.rotation.to_euler(glam::EulerRot::YXZ);
        Quat::from_rotation_y(yaw)
    }

    /// Combine two deltas (self then other).
    #[inline]
    pub fn combine(&self, other: &RootMotionDelta) -> RootMotionDelta {
        // Rotate other's translation by self's rotation, then add
        let rotated_translation = self.rotation * other.translation;
        RootMotionDelta {
            translation: self.translation + rotated_translation,
            rotation: self.rotation * other.rotation,
        }
    }

    /// Scale the delta by a factor.
    #[inline]
    pub fn scale(&self, factor: f32) -> RootMotionDelta {
        RootMotionDelta {
            translation: self.translation * factor,
            rotation: Quat::IDENTITY.slerp(self.rotation, factor),
        }
    }

    /// Invert the delta.
    #[inline]
    pub fn inverse(&self) -> RootMotionDelta {
        let inv_rotation = self.rotation.inverse();
        RootMotionDelta {
            translation: inv_rotation * (-self.translation),
            rotation: inv_rotation,
        }
    }

    /// Get the magnitude of translation.
    #[inline]
    pub fn translation_magnitude(&self) -> f32 {
        self.translation.length()
    }

    /// Get the angle of rotation in radians.
    #[inline]
    pub fn rotation_angle(&self) -> f32 {
        self.rotation.to_axis_angle().1
    }
}

// ---------------------------------------------------------------------------
// RootMotionAccumulator
// ---------------------------------------------------------------------------

/// Accumulates root motion deltas over time.
///
/// Tracks total accumulated motion from the start of an animation or clip,
/// handling loop boundaries and providing reset capabilities.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RootMotionAccumulator {
    /// Configuration for motion extraction.
    pub config: RootMotionConfig,

    /// Total accumulated translation.
    pub accumulated_translation: Vec3,

    /// Total accumulated rotation.
    pub accumulated_rotation: Quat,

    /// Last known root transform for delta calculation.
    pub last_root_transform: Option<(Vec3, Quat)>,

    /// Current playback time (for loop detection).
    last_time: f32,

    /// Number of loops completed.
    loop_count: u32,
}

impl RootMotionAccumulator {
    /// Create a new accumulator with the given configuration.
    pub fn new(config: RootMotionConfig) -> Self {
        Self {
            config,
            accumulated_translation: Vec3::ZERO,
            accumulated_rotation: Quat::IDENTITY,
            last_root_transform: None,
            last_time: 0.0,
            loop_count: 0,
        }
    }

    /// Create an accumulator with default config for the given root bone.
    #[inline]
    pub fn for_root_bone(root_bone: usize) -> Self {
        Self::new(RootMotionConfig::new(root_bone))
    }

    /// Extract delta from the current pose and accumulate it.
    ///
    /// Returns the delta for this frame.
    pub fn extract_delta(&mut self, pose: &Pose) -> RootMotionDelta {
        let root_bone = self.config.root_bone;

        // Get current root transform
        let current_transform = pose.get_transform(root_bone);
        let current_pos = current_transform.position;
        let current_rot = current_transform.rotation;

        // Calculate delta from last frame
        let delta = if let Some((last_pos, last_rot)) = self.last_root_transform {
            let pos_delta = current_pos - last_pos;
            let rot_delta = last_rot.inverse() * current_rot;
            RootMotionDelta::new(pos_delta, rot_delta)
        } else {
            RootMotionDelta::ZERO
        };

        // Apply configuration filters
        let filtered_delta = self.filter_delta(&delta);

        // Apply mode weighting
        let weighted_delta = filtered_delta.scale(self.config.mode.animation_weight());

        // Accumulate
        self.accumulate(&weighted_delta);

        // Store current as last
        self.last_root_transform = Some((current_pos, current_rot));

        weighted_delta
    }

    /// Extract delta from a pose, optionally handling time for loop detection.
    pub fn extract_delta_with_time(&mut self, pose: &Pose, current_time: f32, duration: f32) -> RootMotionDelta {
        // Detect loop wrap-around
        if current_time < self.last_time && duration > 0.0 {
            self.loop_count += 1;
            // Reset last transform to handle discontinuity
            self.last_root_transform = None;
        }
        self.last_time = current_time;

        self.extract_delta(pose)
    }

    /// Filter a delta according to configuration.
    fn filter_delta(&self, delta: &RootMotionDelta) -> RootMotionDelta {
        let mut translation = delta.translation;
        let mut rotation = delta.rotation;

        // Filter translation components
        if !self.config.extract_horizontal {
            translation.x = 0.0;
            translation.z = 0.0;
        }
        if !self.config.extract_vertical {
            translation.y = 0.0;
        }

        // Filter rotation
        if !self.config.extract_rotation {
            rotation = Quat::IDENTITY;
        } else if !self.config.extract_full_rotation {
            // Extract yaw only
            rotation = delta.yaw_rotation();
        }

        RootMotionDelta::new(translation, rotation)
    }

    /// Accumulate a delta into the total.
    fn accumulate(&mut self, delta: &RootMotionDelta) {
        // Rotate translation by accumulated rotation before adding
        let rotated_translation = self.accumulated_rotation * delta.translation;
        self.accumulated_translation += rotated_translation;
        self.accumulated_rotation = self.accumulated_rotation * delta.rotation;

        // Normalize to prevent drift
        self.accumulated_rotation = self.accumulated_rotation.normalize();
    }

    /// Reset the accumulator to initial state.
    pub fn reset(&mut self) {
        self.accumulated_translation = Vec3::ZERO;
        self.accumulated_rotation = Quat::IDENTITY;
        self.last_root_transform = None;
        self.last_time = 0.0;
        self.loop_count = 0;
    }

    /// Reset accumulation but keep last transform (for clip transitions).
    pub fn reset_accumulation(&mut self) {
        self.accumulated_translation = Vec3::ZERO;
        self.accumulated_rotation = Quat::IDENTITY;
        self.loop_count = 0;
    }

    /// Apply accumulated motion to a transform.
    pub fn apply_to_transform(&self, transform: &mut Transform) {
        // Apply translation (in world space, rotated by current orientation)
        transform.position += transform.rotation * self.accumulated_translation;

        // Apply rotation
        transform.rotation = transform.rotation * self.accumulated_rotation;
        transform.rotation = transform.rotation.normalize();
    }

    /// Apply accumulated motion to position and rotation separately.
    pub fn apply_to_components(&self, position: &mut Vec3, rotation: &mut Quat) {
        *position += *rotation * self.accumulated_translation;
        *rotation = (*rotation * self.accumulated_rotation).normalize();
    }

    /// Get the accumulated motion as a delta.
    #[inline]
    pub fn get_accumulated(&self) -> RootMotionDelta {
        RootMotionDelta::new(self.accumulated_translation, self.accumulated_rotation)
    }

    /// Get the number of loops completed.
    #[inline]
    pub fn loop_count(&self) -> u32 {
        self.loop_count
    }

    /// Get the current motion mode.
    #[inline]
    pub fn mode(&self) -> RootMotionMode {
        self.config.mode
    }

    /// Set the motion mode.
    #[inline]
    pub fn set_mode(&mut self, mode: RootMotionMode) {
        self.config.mode = mode;
    }

    /// Check if there is any accumulated motion.
    #[inline]
    pub fn has_motion(&self) -> bool {
        !self.get_accumulated().is_zero(ROOT_MOTION_EPSILON)
    }

    /// Get horizontal accumulated translation.
    #[inline]
    pub fn horizontal_translation(&self) -> Vec3 {
        Vec3::new(self.accumulated_translation.x, 0.0, self.accumulated_translation.z)
    }

    /// Get vertical accumulated translation.
    #[inline]
    pub fn vertical_translation(&self) -> Vec3 {
        Vec3::new(0.0, self.accumulated_translation.y, 0.0)
    }
}

impl Default for RootMotionAccumulator {
    fn default() -> Self {
        Self::new(RootMotionConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Free Functions
// ---------------------------------------------------------------------------

/// Extract root motion from an animation clip over a time range.
///
/// This samples the clip at the start and end times and computes the delta.
pub fn extract_root_motion_from_clip(
    clip: &AnimationClip,
    root_bone: usize,
    start_time: f32,
    end_time: f32,
) -> RootMotionDelta {
    // Get the root bone track
    let bone_track = match clip.bone_tracks.get(root_bone) {
        Some(track) => track,
        None => return RootMotionDelta::ZERO,
    };

    // Sample at start and end
    let start_transform = bone_track.sample(start_time);
    let end_transform = bone_track.sample(end_time);

    // Compute delta
    let translation_delta = end_transform.position - start_transform.position;
    let rotation_delta = start_transform.rotation.inverse() * end_transform.rotation;

    RootMotionDelta::new(translation_delta, rotation_delta)
}

/// Extract root motion from a clip by bone name.
pub fn extract_root_motion_by_name(
    clip: &AnimationClip,
    root_bone_name: &str,
    start_time: f32,
    end_time: f32,
) -> RootMotionDelta {
    // Find the root bone track
    let bone_track = match clip.bone_track(root_bone_name) {
        Some(track) => track,
        None => return RootMotionDelta::ZERO,
    };

    // Sample at start and end
    let start_transform = bone_track.sample(start_time);
    let end_transform = bone_track.sample(end_time);

    // Compute delta
    let translation_delta = end_transform.position - start_transform.position;
    let rotation_delta = start_transform.rotation.inverse() * end_transform.rotation;

    RootMotionDelta::new(translation_delta, rotation_delta)
}

/// Extract total root motion for an entire clip.
pub fn extract_total_root_motion(clip: &AnimationClip, root_bone: usize) -> RootMotionDelta {
    extract_root_motion_from_clip(clip, root_bone, 0.0, clip.duration)
}

/// Extract root motion with horizontal/vertical separation.
pub fn extract_root_motion_separated(
    clip: &AnimationClip,
    root_bone: usize,
    start_time: f32,
    end_time: f32,
) -> (Vec3, Vec3, Quat) {
    let delta = extract_root_motion_from_clip(clip, root_bone, start_time, end_time);

    let horizontal = Vec3::new(delta.translation.x, 0.0, delta.translation.z);
    let vertical = Vec3::new(0.0, delta.translation.y, 0.0);

    (horizontal, vertical, delta.rotation)
}

/// Compute root motion velocity from a delta and time interval.
#[inline]
pub fn root_motion_velocity(delta: &RootMotionDelta, dt: f32) -> (Vec3, f32) {
    if dt <= ROOT_MOTION_EPSILON {
        return (Vec3::ZERO, 0.0);
    }

    let linear_velocity = delta.translation / dt;
    let angular_velocity = delta.rotation_angle() / dt;

    (linear_velocity, angular_velocity)
}

/// Blend two root motion deltas.
#[inline]
pub fn blend_root_motion(a: &RootMotionDelta, b: &RootMotionDelta, t: f32) -> RootMotionDelta {
    let t = t.clamp(0.0, 1.0);
    RootMotionDelta {
        translation: a.translation.lerp(b.translation, t),
        rotation: a.rotation.slerp(b.rotation, t),
    }
}

/// Create a root motion delta from velocity and time.
#[inline]
pub fn velocity_to_root_motion(linear_velocity: Vec3, angular_velocity: f32, dt: f32) -> RootMotionDelta {
    RootMotionDelta {
        translation: linear_velocity * dt,
        rotation: Quat::from_rotation_y(angular_velocity * dt),
    }
}

// ---------------------------------------------------------------------------
// RootMotionClipAnalyzer
// ---------------------------------------------------------------------------

/// Analyzer for extracting root motion statistics from animation clips.
#[derive(Clone, Debug, Default)]
pub struct RootMotionClipAnalyzer {
    /// Total translation over the clip.
    pub total_translation: Vec3,

    /// Total rotation angle (radians).
    pub total_rotation_angle: f32,

    /// Average speed (units per second).
    pub average_speed: f32,

    /// Average angular speed (radians per second).
    pub average_angular_speed: f32,

    /// Whether the clip loops seamlessly (start ~= end).
    pub is_seamless_loop: bool,

    /// Maximum speed reached during the clip.
    pub max_speed: f32,
}

impl RootMotionClipAnalyzer {
    /// Analyze a clip for root motion statistics.
    pub fn analyze(clip: &AnimationClip, root_bone: usize, sample_rate: f32) -> Self {
        if clip.duration <= 0.0 || sample_rate <= 0.0 {
            return Self::default();
        }

        let dt = 1.0 / sample_rate;
        let bone_track = match clip.bone_tracks.get(root_bone) {
            Some(track) => track,
            None => return Self::default(),
        };

        let mut total_translation = Vec3::ZERO;
        let mut total_rotation_angle = 0.0f32;
        let mut max_speed = 0.0f32;

        let mut time = 0.0;
        let mut last_transform = bone_track.sample(0.0);

        while time < clip.duration {
            time += dt;
            let current_transform = bone_track.sample(time.min(clip.duration));

            let delta_pos = current_transform.position - last_transform.position;
            let delta_rot = last_transform.rotation.inverse() * current_transform.rotation;

            total_translation += delta_pos;
            total_rotation_angle += delta_rot.to_axis_angle().1;

            let speed = delta_pos.length() / dt;
            max_speed = max_speed.max(speed);

            last_transform = current_transform;
        }

        // Check if the clip loops seamlessly
        let start_transform = bone_track.sample(0.0);
        let end_transform = bone_track.sample(clip.duration);
        let is_seamless_loop = start_transform.position.abs_diff_eq(end_transform.position, 0.01)
            && start_transform.rotation.dot(end_transform.rotation).abs() > 0.999;

        Self {
            total_translation,
            total_rotation_angle,
            average_speed: total_translation.length() / clip.duration,
            average_angular_speed: total_rotation_angle / clip.duration,
            is_seamless_loop,
            max_speed,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::PoseType;
    use std::f32::consts::FRAC_PI_2;
    use std::f32::consts::FRAC_PI_4;

    // ===== RootMotionMode Tests =====

    #[test]
    fn test_root_motion_mode_default() {
        let mode = RootMotionMode::default();
        assert_eq!(mode, RootMotionMode::AnimationDriven);
    }

    #[test]
    fn test_root_motion_mode_animation_weight() {
        assert_eq!(RootMotionMode::AnimationDriven.animation_weight(), 1.0);
        assert_eq!(RootMotionMode::PhysicsDriven.animation_weight(), 0.0);
        assert_eq!(RootMotionMode::blended(0.5).animation_weight(), 0.5);
        assert_eq!(RootMotionMode::blended(0.75).animation_weight(), 0.75);
    }

    #[test]
    fn test_root_motion_mode_blended_clamping() {
        // Should clamp to valid range
        assert_eq!(RootMotionMode::blended(-0.5).animation_weight(), 0.0);
        assert_eq!(RootMotionMode::blended(1.5).animation_weight(), 1.0);
    }

    #[test]
    fn test_root_motion_mode_uses_animation() {
        assert!(RootMotionMode::AnimationDriven.uses_animation());
        assert!(!RootMotionMode::PhysicsDriven.uses_animation());
        assert!(RootMotionMode::blended(0.5).uses_animation());
    }

    #[test]
    fn test_root_motion_mode_uses_physics() {
        assert!(!RootMotionMode::AnimationDriven.uses_physics());
        assert!(RootMotionMode::PhysicsDriven.uses_physics());
        assert!(RootMotionMode::blended(0.5).uses_physics());
    }

    #[test]
    fn test_root_motion_mode_equality() {
        assert_eq!(RootMotionMode::AnimationDriven, RootMotionMode::AnimationDriven);
        assert_ne!(RootMotionMode::AnimationDriven, RootMotionMode::PhysicsDriven);

        // Blended modes with same weight should be equal
        let a = RootMotionMode::blended(0.5);
        let b = RootMotionMode::blended(0.5);
        assert_eq!(a, b);
    }

    // ===== RootMotionConfig Tests =====

    #[test]
    fn test_root_motion_config_default() {
        let config = RootMotionConfig::default();
        assert_eq!(config.root_bone, 0);
        assert!(config.extract_horizontal);
        assert!(config.extract_vertical);
        assert!(config.extract_rotation);
        assert!(!config.extract_full_rotation);
        assert_eq!(config.mode, RootMotionMode::AnimationDriven);
    }

    #[test]
    fn test_root_motion_config_new() {
        let config = RootMotionConfig::new(5);
        assert_eq!(config.root_bone, 5);
    }

    #[test]
    fn test_root_motion_config_horizontal_only() {
        let config = RootMotionConfig::horizontal_only(3);
        assert_eq!(config.root_bone, 3);
        assert!(config.extract_horizontal);
        assert!(!config.extract_vertical);
        assert!(config.extract_rotation);
    }

    #[test]
    fn test_root_motion_config_full_3d() {
        let config = RootMotionConfig::full_3d(2);
        assert_eq!(config.root_bone, 2);
        assert!(config.extract_horizontal);
        assert!(config.extract_vertical);
        assert!(config.extract_rotation);
        assert!(config.extract_full_rotation);
    }

    #[test]
    fn test_root_motion_config_builder_pattern() {
        let config = RootMotionConfig::new(0)
            .with_mode(RootMotionMode::PhysicsDriven)
            .with_horizontal(false)
            .with_vertical(true)
            .with_rotation(false)
            .with_full_rotation(true);

        assert_eq!(config.mode, RootMotionMode::PhysicsDriven);
        assert!(!config.extract_horizontal);
        assert!(config.extract_vertical);
        assert!(!config.extract_rotation);
        assert!(config.extract_full_rotation);
    }

    // ===== RootMotionDelta Tests =====

    #[test]
    fn test_root_motion_delta_default() {
        let delta = RootMotionDelta::default();
        assert_eq!(delta.translation, Vec3::ZERO);
        assert_eq!(delta.rotation, Quat::IDENTITY);
    }

    #[test]
    fn test_root_motion_delta_zero() {
        let delta = RootMotionDelta::ZERO;
        assert!(delta.is_zero(1e-6));
    }

    #[test]
    fn test_root_motion_delta_from_translation() {
        let delta = RootMotionDelta::from_translation(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(delta.translation, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(delta.rotation, Quat::IDENTITY);
    }

    #[test]
    fn test_root_motion_delta_from_rotation() {
        let rot = Quat::from_rotation_y(FRAC_PI_2);
        let delta = RootMotionDelta::from_rotation(rot);
        assert_eq!(delta.translation, Vec3::ZERO);
        assert!(delta.rotation.abs_diff_eq(rot, 1e-5));
    }

    #[test]
    fn test_root_motion_delta_new() {
        let trans = Vec3::new(1.0, 0.0, 0.0);
        let rot = Quat::from_rotation_y(FRAC_PI_4);
        let delta = RootMotionDelta::new(trans, rot);
        assert_eq!(delta.translation, trans);
        assert!(delta.rotation.abs_diff_eq(rot, 1e-5));
    }

    #[test]
    fn test_root_motion_delta_is_zero() {
        let zero = RootMotionDelta::ZERO;
        assert!(zero.is_zero(1e-6));

        let small = RootMotionDelta::new(Vec3::new(1e-8, 0.0, 0.0), Quat::IDENTITY);
        assert!(small.is_zero(1e-5));

        let not_zero = RootMotionDelta::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY);
        assert!(!not_zero.is_zero(1e-6));
    }

    #[test]
    fn test_root_motion_delta_horizontal_translation() {
        let delta = RootMotionDelta::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let horizontal = delta.horizontal_translation();
        assert_eq!(horizontal, Vec3::new(1.0, 0.0, 3.0));
    }

    #[test]
    fn test_root_motion_delta_vertical_translation() {
        let delta = RootMotionDelta::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let vertical = delta.vertical_translation();
        assert_eq!(vertical, Vec3::new(0.0, 2.0, 0.0));
    }

    #[test]
    fn test_root_motion_delta_yaw_rotation() {
        // Pure yaw rotation should be preserved
        let yaw = Quat::from_rotation_y(FRAC_PI_4);
        let delta = RootMotionDelta::from_rotation(yaw);
        let extracted = delta.yaw_rotation();
        assert!(extracted.abs_diff_eq(yaw, 1e-4));
    }

    #[test]
    fn test_root_motion_delta_yaw_rotation_extracts_yaw_only() {
        // Combined rotation should extract only yaw
        let combined = Quat::from_rotation_y(FRAC_PI_4) * Quat::from_rotation_x(0.5);
        let delta = RootMotionDelta::from_rotation(combined);
        let yaw_only = delta.yaw_rotation();

        // Yaw rotation should only have Y component
        let expected = Quat::from_rotation_y(FRAC_PI_4);
        assert!(yaw_only.abs_diff_eq(expected, 0.1)); // Approximate due to euler decomposition
    }

    #[test]
    fn test_root_motion_delta_combine() {
        let delta1 = RootMotionDelta::new(Vec3::new(1.0, 0.0, 0.0), Quat::from_rotation_y(FRAC_PI_2));
        let delta2 = RootMotionDelta::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY);

        let combined = delta1.combine(&delta2);

        // delta2's translation should be rotated by delta1's rotation (90 deg around Y)
        // So (1,0,0) becomes (0,0,-1)
        assert!(combined.translation.abs_diff_eq(Vec3::new(1.0, 0.0, -1.0), 1e-5));
    }

    #[test]
    fn test_root_motion_delta_scale() {
        let delta = RootMotionDelta::new(Vec3::new(2.0, 0.0, 0.0), Quat::from_rotation_y(FRAC_PI_2));

        let scaled = delta.scale(0.5);
        assert!(scaled.translation.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5));
        // Rotation should be halfway
        let expected_rot = Quat::from_rotation_y(FRAC_PI_4);
        assert!(scaled.rotation.abs_diff_eq(expected_rot, 1e-4));
    }

    #[test]
    fn test_root_motion_delta_scale_zero() {
        let delta = RootMotionDelta::new(Vec3::new(2.0, 0.0, 0.0), Quat::from_rotation_y(FRAC_PI_2));
        let scaled = delta.scale(0.0);
        assert!(scaled.is_zero(1e-5));
    }

    #[test]
    fn test_root_motion_delta_inverse() {
        let delta = RootMotionDelta::new(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_y(FRAC_PI_4));
        let inverse = delta.inverse();
        let combined = delta.combine(&inverse);

        // Should return to identity
        assert!(combined.is_zero(1e-4));
    }

    #[test]
    fn test_root_motion_delta_translation_magnitude() {
        let delta = RootMotionDelta::from_translation(Vec3::new(3.0, 4.0, 0.0));
        assert!((delta.translation_magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_root_motion_delta_rotation_angle() {
        let delta = RootMotionDelta::from_rotation(Quat::from_rotation_y(FRAC_PI_2));
        assert!((delta.rotation_angle() - FRAC_PI_2).abs() < 1e-5);
    }

    // ===== RootMotionAccumulator Tests =====

    #[test]
    fn test_root_motion_accumulator_new() {
        let config = RootMotionConfig::new(0);
        let acc = RootMotionAccumulator::new(config);

        assert_eq!(acc.accumulated_translation, Vec3::ZERO);
        assert_eq!(acc.accumulated_rotation, Quat::IDENTITY);
        assert!(acc.last_root_transform.is_none());
        assert_eq!(acc.loop_count(), 0);
    }

    #[test]
    fn test_root_motion_accumulator_for_root_bone() {
        let acc = RootMotionAccumulator::for_root_bone(5);
        assert_eq!(acc.config.root_bone, 5);
    }

    #[test]
    fn test_root_motion_accumulator_default() {
        let acc = RootMotionAccumulator::default();
        assert_eq!(acc.config.root_bone, 0);
    }

    #[test]
    fn test_root_motion_accumulator_extract_delta_first_frame() {
        let config = RootMotionConfig::new(0);
        let mut acc = RootMotionAccumulator::new(config);

        let mut pose = Pose::new(1, PoseType::Current);
        pose.set_transform(0, Transform::from_position(Vec3::new(1.0, 0.0, 0.0)));

        let delta = acc.extract_delta(&pose);

        // First frame should return zero delta
        assert!(delta.is_zero(1e-5));
    }

    #[test]
    fn test_root_motion_accumulator_extract_delta_second_frame() {
        let config = RootMotionConfig::new(0);
        let mut acc = RootMotionAccumulator::new(config);

        // First frame
        let mut pose1 = Pose::new(1, PoseType::Current);
        pose1.set_transform(0, Transform::from_position(Vec3::ZERO));
        acc.extract_delta(&pose1);

        // Second frame
        let mut pose2 = Pose::new(1, PoseType::Current);
        pose2.set_transform(0, Transform::from_position(Vec3::new(1.0, 0.0, 0.0)));
        let delta = acc.extract_delta(&pose2);

        // Should have delta of (1, 0, 0)
        assert!(delta.translation.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_root_motion_accumulator_accumulation() {
        let config = RootMotionConfig::new(0);
        let mut acc = RootMotionAccumulator::new(config);

        // Frame 1: at origin
        let mut pose = Pose::new(1, PoseType::Current);
        pose.set_transform(0, Transform::IDENTITY);
        acc.extract_delta(&pose);

        // Frame 2: move +1 on X
        pose.set_transform(0, Transform::from_position(Vec3::new(1.0, 0.0, 0.0)));
        acc.extract_delta(&pose);

        // Frame 3: move another +1 on X
        pose.set_transform(0, Transform::from_position(Vec3::new(2.0, 0.0, 0.0)));
        acc.extract_delta(&pose);

        // Total accumulated should be (2, 0, 0)
        assert!(acc.accumulated_translation.abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_root_motion_accumulator_horizontal_only() {
        let config = RootMotionConfig::horizontal_only(0);
        let mut acc = RootMotionAccumulator::new(config);

        // Frame 1
        let mut pose = Pose::new(1, PoseType::Current);
        pose.set_transform(0, Transform::IDENTITY);
        acc.extract_delta(&pose);

        // Frame 2: move in all directions
        pose.set_transform(0, Transform::from_position(Vec3::new(1.0, 2.0, 3.0)));
        acc.extract_delta(&pose);

        // Should only have X and Z
        let accumulated = acc.get_accumulated();
        assert!(accumulated.translation.abs_diff_eq(Vec3::new(1.0, 0.0, 3.0), 1e-5));
    }

    #[test]
    fn test_root_motion_accumulator_rotation_extraction() {
        let config = RootMotionConfig::new(0);
        let mut acc = RootMotionAccumulator::new(config);

        // Frame 1
        let mut pose = Pose::new(1, PoseType::Current);
        pose.set_transform(0, Transform::IDENTITY);
        acc.extract_delta(&pose);

        // Frame 2: rotate 45 degrees
        pose.set_transform(0, Transform::from_rotation(Quat::from_rotation_y(FRAC_PI_4)));
        let delta = acc.extract_delta(&pose);

        assert!(delta.rotation.abs_diff_eq(Quat::from_rotation_y(FRAC_PI_4), 1e-4));
    }

    #[test]
    fn test_root_motion_accumulator_physics_driven_mode() {
        let config = RootMotionConfig::new(0).with_mode(RootMotionMode::PhysicsDriven);
        let mut acc = RootMotionAccumulator::new(config);

        // Frame 1
        let mut pose = Pose::new(1, PoseType::Current);
        pose.set_transform(0, Transform::IDENTITY);
        acc.extract_delta(&pose);

        // Frame 2: move
        pose.set_transform(0, Transform::from_position(Vec3::new(10.0, 0.0, 0.0)));
        let delta = acc.extract_delta(&pose);

        // Physics driven should return zero delta
        assert!(delta.is_zero(1e-5));
        assert!(acc.accumulated_translation.abs_diff_eq(Vec3::ZERO, 1e-5));
    }

    #[test]
    fn test_root_motion_accumulator_blended_mode() {
        let config = RootMotionConfig::new(0).with_mode(RootMotionMode::blended(0.5));
        let mut acc = RootMotionAccumulator::new(config);

        // Frame 1
        let mut pose = Pose::new(1, PoseType::Current);
        pose.set_transform(0, Transform::IDENTITY);
        acc.extract_delta(&pose);

        // Frame 2: move 2 units
        pose.set_transform(0, Transform::from_position(Vec3::new(2.0, 0.0, 0.0)));
        let delta = acc.extract_delta(&pose);

        // Should be half
        assert!(delta.translation.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_root_motion_accumulator_reset() {
        let config = RootMotionConfig::new(0);
        let mut acc = RootMotionAccumulator::new(config);

        // Accumulate some motion
        let mut pose = Pose::new(1, PoseType::Current);
        pose.set_transform(0, Transform::IDENTITY);
        acc.extract_delta(&pose);
        pose.set_transform(0, Transform::from_position(Vec3::new(5.0, 0.0, 0.0)));
        acc.extract_delta(&pose);

        // Reset
        acc.reset();

        assert_eq!(acc.accumulated_translation, Vec3::ZERO);
        assert_eq!(acc.accumulated_rotation, Quat::IDENTITY);
        assert!(acc.last_root_transform.is_none());
        assert_eq!(acc.loop_count(), 0);
    }

    #[test]
    fn test_root_motion_accumulator_reset_accumulation() {
        let config = RootMotionConfig::new(0);
        let mut acc = RootMotionAccumulator::new(config);

        // Accumulate some motion
        let mut pose = Pose::new(1, PoseType::Current);
        pose.set_transform(0, Transform::IDENTITY);
        acc.extract_delta(&pose);
        pose.set_transform(0, Transform::from_position(Vec3::new(5.0, 0.0, 0.0)));
        acc.extract_delta(&pose);

        // Reset accumulation only
        acc.reset_accumulation();

        assert_eq!(acc.accumulated_translation, Vec3::ZERO);
        assert!(acc.last_root_transform.is_some()); // Should keep last transform
    }

    #[test]
    fn test_root_motion_accumulator_apply_to_transform() {
        let config = RootMotionConfig::new(0);
        let mut acc = RootMotionAccumulator::new(config);

        // Simulate motion
        acc.accumulated_translation = Vec3::new(1.0, 0.0, 0.0);
        acc.accumulated_rotation = Quat::from_rotation_y(FRAC_PI_2);

        let mut transform = Transform::IDENTITY;
        acc.apply_to_transform(&mut transform);

        // Position should be updated
        assert!(transform.position.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5));
        // Rotation should be updated
        assert!(transform.rotation.abs_diff_eq(Quat::from_rotation_y(FRAC_PI_2), 1e-4));
    }

    #[test]
    fn test_root_motion_accumulator_apply_to_rotated_transform() {
        let config = RootMotionConfig::new(0);
        let mut acc = RootMotionAccumulator::new(config);

        // Accumulated motion in local space
        acc.accumulated_translation = Vec3::new(1.0, 0.0, 0.0);

        // Transform already rotated 90 degrees
        let mut transform = Transform::from_rotation(Quat::from_rotation_y(FRAC_PI_2));
        acc.apply_to_transform(&mut transform);

        // Local +X becomes world +Z when rotated 90 degrees around Y
        assert!(transform.position.abs_diff_eq(Vec3::new(0.0, 0.0, -1.0), 1e-5));
    }

    #[test]
    fn test_root_motion_accumulator_apply_to_components() {
        let config = RootMotionConfig::new(0);
        let mut acc = RootMotionAccumulator::new(config);

        acc.accumulated_translation = Vec3::new(1.0, 2.0, 3.0);
        acc.accumulated_rotation = Quat::from_rotation_y(FRAC_PI_4);

        let mut position = Vec3::ZERO;
        let mut rotation = Quat::IDENTITY;
        acc.apply_to_components(&mut position, &mut rotation);

        assert!(position.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-5));
        assert!(rotation.abs_diff_eq(Quat::from_rotation_y(FRAC_PI_4), 1e-4));
    }

    #[test]
    fn test_root_motion_accumulator_loop_detection() {
        let config = RootMotionConfig::new(0);
        let mut acc = RootMotionAccumulator::new(config);

        let pose = Pose::new(1, PoseType::Current);

        // Simulate time progression
        acc.extract_delta_with_time(&pose, 0.0, 1.0);
        acc.extract_delta_with_time(&pose, 0.5, 1.0);
        acc.extract_delta_with_time(&pose, 0.9, 1.0);

        assert_eq!(acc.loop_count(), 0);

        // Time wraps around
        acc.extract_delta_with_time(&pose, 0.1, 1.0);

        assert_eq!(acc.loop_count(), 1);
    }

    #[test]
    fn test_root_motion_accumulator_has_motion() {
        let mut acc = RootMotionAccumulator::default();
        assert!(!acc.has_motion());

        acc.accumulated_translation = Vec3::new(1.0, 0.0, 0.0);
        assert!(acc.has_motion());
    }

    #[test]
    fn test_root_motion_accumulator_horizontal_vertical_accessors() {
        let mut acc = RootMotionAccumulator::default();
        acc.accumulated_translation = Vec3::new(1.0, 2.0, 3.0);

        assert_eq!(acc.horizontal_translation(), Vec3::new(1.0, 0.0, 3.0));
        assert_eq!(acc.vertical_translation(), Vec3::new(0.0, 2.0, 0.0));
    }

    #[test]
    fn test_root_motion_accumulator_mode_accessors() {
        let mut acc = RootMotionAccumulator::default();
        assert_eq!(acc.mode(), RootMotionMode::AnimationDriven);

        acc.set_mode(RootMotionMode::PhysicsDriven);
        assert_eq!(acc.mode(), RootMotionMode::PhysicsDriven);
    }

    // ===== Free Function Tests =====

    #[test]
    fn test_extract_root_motion_from_clip() {
        let mut clip = AnimationClip::new("walk", 1.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(2.0, 0.0, 0.0)),
        ]);
        clip.add_bone_track(BoneTrack::new("root").with_position(pos_track));

        let delta = extract_root_motion_from_clip(&clip, 0, 0.0, 1.0);

        assert!(delta.translation.abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_extract_root_motion_from_clip_partial() {
        let mut clip = AnimationClip::new("walk", 1.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(2.0, 0.0, 0.0)),
        ]);
        clip.add_bone_track(BoneTrack::new("root").with_position(pos_track));

        let delta = extract_root_motion_from_clip(&clip, 0, 0.0, 0.5);

        assert!(delta.translation.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_extract_root_motion_from_clip_no_bone() {
        let clip = AnimationClip::new("empty", 1.0);
        let delta = extract_root_motion_from_clip(&clip, 0, 0.0, 1.0);
        assert!(delta.is_zero(1e-5));
    }

    #[test]
    fn test_extract_root_motion_by_name() {
        let mut clip = AnimationClip::new("walk", 1.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(3.0, 0.0, 0.0)),
        ]);
        clip.add_bone_track(BoneTrack::new("hips").with_position(pos_track));

        let delta = extract_root_motion_by_name(&clip, "hips", 0.0, 1.0);
        assert!(delta.translation.abs_diff_eq(Vec3::new(3.0, 0.0, 0.0), 1e-5));

        let delta_missing = extract_root_motion_by_name(&clip, "nonexistent", 0.0, 1.0);
        assert!(delta_missing.is_zero(1e-5));
    }

    #[test]
    fn test_extract_total_root_motion() {
        let mut clip = AnimationClip::new("walk", 2.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(2.0, Vec3::new(4.0, 1.0, 0.0)),
        ]);
        clip.add_bone_track(BoneTrack::new("root").with_position(pos_track));

        let delta = extract_total_root_motion(&clip, 0);
        assert!(delta.translation.abs_diff_eq(Vec3::new(4.0, 1.0, 0.0), 1e-5));
    }

    #[test]
    fn test_extract_root_motion_separated() {
        let mut clip = AnimationClip::new("walk", 1.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(1.0, 2.0, 3.0)),
        ]);
        let rot_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
            Keyframe::linear(1.0, Quat::from_rotation_y(FRAC_PI_2)),
        ]);
        clip.add_bone_track(
            BoneTrack::new("root")
                .with_position(pos_track)
                .with_rotation(rot_track),
        );

        let (horizontal, vertical, rotation) = extract_root_motion_separated(&clip, 0, 0.0, 1.0);

        assert!(horizontal.abs_diff_eq(Vec3::new(1.0, 0.0, 3.0), 1e-5));
        assert!(vertical.abs_diff_eq(Vec3::new(0.0, 2.0, 0.0), 1e-5));
        assert!(rotation.abs_diff_eq(Quat::from_rotation_y(FRAC_PI_2), 1e-4));
    }

    #[test]
    fn test_root_motion_velocity() {
        let delta = RootMotionDelta::new(Vec3::new(2.0, 0.0, 0.0), Quat::from_rotation_y(FRAC_PI_2));
        let (linear, angular) = root_motion_velocity(&delta, 0.5);

        assert!(linear.abs_diff_eq(Vec3::new(4.0, 0.0, 0.0), 1e-5));
        assert!((angular - PI).abs() < 1e-4); // 90 deg / 0.5s = 180 deg/s = PI rad/s
    }

    #[test]
    fn test_root_motion_velocity_zero_dt() {
        let delta = RootMotionDelta::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY);
        let (linear, angular) = root_motion_velocity(&delta, 0.0);

        assert_eq!(linear, Vec3::ZERO);
        assert_eq!(angular, 0.0);
    }

    #[test]
    fn test_blend_root_motion() {
        let a = RootMotionDelta::from_translation(Vec3::ZERO);
        let b = RootMotionDelta::from_translation(Vec3::new(10.0, 0.0, 0.0));

        let blended = blend_root_motion(&a, &b, 0.5);
        assert!(blended.translation.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-5));

        let blend_0 = blend_root_motion(&a, &b, 0.0);
        assert!(blend_0.translation.abs_diff_eq(Vec3::ZERO, 1e-5));

        let blend_1 = blend_root_motion(&a, &b, 1.0);
        assert!(blend_1.translation.abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_blend_root_motion_clamping() {
        let a = RootMotionDelta::from_translation(Vec3::ZERO);
        let b = RootMotionDelta::from_translation(Vec3::new(10.0, 0.0, 0.0));

        let blend_over = blend_root_motion(&a, &b, 2.0);
        assert!(blend_over.translation.abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), 1e-5));

        let blend_under = blend_root_motion(&a, &b, -1.0);
        assert!(blend_under.translation.abs_diff_eq(Vec3::ZERO, 1e-5));
    }

    #[test]
    fn test_velocity_to_root_motion() {
        let velocity = Vec3::new(2.0, 0.0, 0.0);
        let angular = FRAC_PI_2;
        let dt = 0.5;

        let delta = velocity_to_root_motion(velocity, angular, dt);

        assert!(delta.translation.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5));
        assert!(delta.rotation.abs_diff_eq(Quat::from_rotation_y(FRAC_PI_4), 1e-4));
    }

    // ===== RootMotionClipAnalyzer Tests =====

    #[test]
    fn test_root_motion_clip_analyzer_basic() {
        let mut clip = AnimationClip::new("walk", 1.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(2.0, 0.0, 0.0)),
        ]);
        clip.add_bone_track(BoneTrack::new("root").with_position(pos_track));

        let analysis = RootMotionClipAnalyzer::analyze(&clip, 0, 30.0);

        assert!(analysis.total_translation.abs_diff_eq(Vec3::new(2.0, 0.0, 0.0), 0.1));
        assert!((analysis.average_speed - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_root_motion_clip_analyzer_empty_clip() {
        let clip = AnimationClip::new("empty", 0.0);
        let analysis = RootMotionClipAnalyzer::analyze(&clip, 0, 30.0);

        assert_eq!(analysis.total_translation, Vec3::ZERO);
        assert_eq!(analysis.average_speed, 0.0);
    }

    #[test]
    fn test_root_motion_clip_analyzer_no_bone() {
        let clip = AnimationClip::new("test", 1.0);
        let analysis = RootMotionClipAnalyzer::analyze(&clip, 0, 30.0);

        assert_eq!(analysis.total_translation, Vec3::ZERO);
    }

    #[test]
    fn test_root_motion_clip_analyzer_seamless_loop() {
        let mut clip = AnimationClip::new("idle", 1.0);

        // Perfectly looping clip (same start and end)
        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(0.5, Vec3::new(0.0, 0.1, 0.0)),
            Keyframe::linear(1.0, Vec3::ZERO),
        ]);
        clip.add_bone_track(BoneTrack::new("root").with_position(pos_track));

        let analysis = RootMotionClipAnalyzer::analyze(&clip, 0, 30.0);
        assert!(analysis.is_seamless_loop);
    }

    #[test]
    fn test_root_motion_clip_analyzer_not_seamless() {
        let mut clip = AnimationClip::new("walk", 1.0);

        // Non-looping clip
        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(2.0, 0.0, 0.0)),
        ]);
        clip.add_bone_track(BoneTrack::new("root").with_position(pos_track));

        let analysis = RootMotionClipAnalyzer::analyze(&clip, 0, 30.0);
        assert!(!analysis.is_seamless_loop);
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_root_motion_delta_combine_identity() {
        let delta = RootMotionDelta::new(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_y(0.5));
        let identity = RootMotionDelta::ZERO;

        let combined = delta.combine(&identity);
        assert!(combined.translation.abs_diff_eq(delta.translation, 1e-5));
        assert!(combined.rotation.abs_diff_eq(delta.rotation, 1e-5));
    }

    #[test]
    fn test_root_motion_with_scale_in_transform() {
        let config = RootMotionConfig::new(0);
        let mut acc = RootMotionAccumulator::new(config);

        acc.accumulated_translation = Vec3::new(1.0, 0.0, 0.0);

        // Transform with scale
        let mut transform = Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::splat(2.0));
        acc.apply_to_transform(&mut transform);

        // Position should still be updated normally (scale doesn't affect position)
        assert!(transform.position.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_root_motion_accumulator_rotation_accumulation() {
        let config = RootMotionConfig::new(0);
        let mut acc = RootMotionAccumulator::new(config);

        // Frame 1
        let mut pose = Pose::new(1, PoseType::Current);
        pose.set_transform(0, Transform::IDENTITY);
        acc.extract_delta(&pose);

        // Frame 2: rotate 45 degrees
        pose.set_transform(0, Transform::from_rotation(Quat::from_rotation_y(FRAC_PI_4)));
        acc.extract_delta(&pose);

        // Frame 3: rotate another 45 degrees
        pose.set_transform(0, Transform::from_rotation(Quat::from_rotation_y(FRAC_PI_2)));
        acc.extract_delta(&pose);

        // Total should be 90 degrees
        assert!(acc.accumulated_rotation.abs_diff_eq(Quat::from_rotation_y(FRAC_PI_2), 1e-4));
    }

    #[test]
    fn test_root_motion_with_negative_translation() {
        let delta = RootMotionDelta::from_translation(Vec3::new(-1.0, -2.0, -3.0));
        assert!(delta.translation.abs_diff_eq(Vec3::new(-1.0, -2.0, -3.0), 1e-5));

        let inverse = delta.inverse();
        let combined = delta.combine(&inverse);
        assert!(combined.is_zero(1e-4));
    }

    #[test]
    fn test_root_motion_mode_serialization() {
        let modes = vec![
            RootMotionMode::AnimationDriven,
            RootMotionMode::PhysicsDriven,
            RootMotionMode::blended(0.7),
        ];

        for mode in modes {
            let json = serde_json::to_string(&mode).unwrap();
            let recovered: RootMotionMode = serde_json::from_str(&json).unwrap();
            assert_eq!(recovered.animation_weight(), mode.animation_weight());
        }
    }

    #[test]
    fn test_root_motion_config_serialization() {
        let config = RootMotionConfig::full_3d(5).with_mode(RootMotionMode::blended(0.3));

        let json = serde_json::to_string(&config).unwrap();
        let recovered: RootMotionConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered.root_bone, config.root_bone);
        assert_eq!(recovered.extract_horizontal, config.extract_horizontal);
        assert_eq!(recovered.extract_vertical, config.extract_vertical);
        assert_eq!(recovered.extract_rotation, config.extract_rotation);
        assert_eq!(recovered.extract_full_rotation, config.extract_full_rotation);
    }

    #[test]
    fn test_root_motion_delta_serialization() {
        let delta = RootMotionDelta::new(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_y(FRAC_PI_4),
        );

        let json = serde_json::to_string(&delta).unwrap();
        let recovered: RootMotionDelta = serde_json::from_str(&json).unwrap();

        assert!(recovered.translation.abs_diff_eq(delta.translation, 1e-5));
        assert!(recovered.rotation.abs_diff_eq(delta.rotation, 1e-4));
    }

    #[test]
    fn test_accumulator_rotation_normalization() {
        let config = RootMotionConfig::new(0);
        let mut acc = RootMotionAccumulator::new(config);

        // Simulate many small rotations that could cause quaternion drift
        let mut pose = Pose::new(1, PoseType::Current);
        let small_rotation = Quat::from_rotation_y(0.01);

        pose.set_transform(0, Transform::IDENTITY);
        acc.extract_delta(&pose);

        let mut current_rot = Quat::IDENTITY;
        for _ in 0..100 {
            current_rot = current_rot * small_rotation;
            pose.set_transform(0, Transform::from_rotation(current_rot));
            acc.extract_delta(&pose);
        }

        // Accumulated rotation should still be normalized
        assert!((acc.accumulated_rotation.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_extract_rotation_with_clip_rotation() {
        let mut clip = AnimationClip::new("turn", 1.0);

        let rot_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
            Keyframe::linear(1.0, Quat::from_rotation_y(PI)),
        ]);
        clip.add_bone_track(BoneTrack::new("root").with_rotation(rot_track));

        let delta = extract_root_motion_from_clip(&clip, 0, 0.0, 1.0);

        // Should have 180 degree rotation
        assert!((delta.rotation_angle() - PI).abs() < 1e-4);
    }

    #[test]
    fn test_root_motion_accumulator_yaw_only() {
        let config = RootMotionConfig::new(0)
            .with_rotation(true)
            .with_full_rotation(false);
        let mut acc = RootMotionAccumulator::new(config);

        // Frame 1
        let mut pose = Pose::new(1, PoseType::Current);
        pose.set_transform(0, Transform::IDENTITY);
        acc.extract_delta(&pose);

        // Frame 2: rotate with pitch and yaw
        let combined_rot = Quat::from_rotation_y(FRAC_PI_4) * Quat::from_rotation_x(0.5);
        pose.set_transform(0, Transform::from_rotation(combined_rot));
        let delta = acc.extract_delta(&pose);

        // Should only extract yaw component
        // The rotation should be approximately Y-only
        let (yaw, pitch, roll) = delta.rotation.to_euler(glam::EulerRot::YXZ);
        assert!(pitch.abs() < 0.1); // Pitch should be filtered
        assert!(roll.abs() < 0.1);  // Roll should be filtered
        assert!(yaw.abs() > 0.1);   // Yaw should be preserved
    }

    #[test]
    fn test_large_clip_analysis() {
        let mut clip = AnimationClip::new("long_walk", 10.0);

        // Create a track with many keyframes
        let mut keyframes = Vec::new();
        for i in 0..=100 {
            let t = i as f32 / 10.0;
            let x = t * 0.5; // Move 0.5 units per second
            keyframes.push(Keyframe::linear(t, Vec3::new(x, 0.0, 0.0)));
        }
        let pos_track = Track::from_keyframes(keyframes);
        clip.add_bone_track(BoneTrack::new("root").with_position(pos_track));

        let analysis = RootMotionClipAnalyzer::analyze(&clip, 0, 60.0);

        // Total translation should be about 5 units
        assert!((analysis.total_translation.x - 5.0).abs() < 0.5);
        // Average speed should be about 0.5 units/sec
        assert!((analysis.average_speed - 0.5).abs() < 0.1);
    }
}
