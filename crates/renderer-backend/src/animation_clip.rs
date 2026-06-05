//! Animation clip asset for keyframe-based skeletal animation (T-AN-1.3).
//!
//! This module provides the AnimationClip asset type that stores keyframed
//! animation data for skeletal meshes. It supports:
//!
//! - Keyframe interpolation (step, linear, cubic Hermite)
//! - Per-bone animation tracks for position, rotation, and scale
//! - Animation events for gameplay callbacks
//! - Float curve tracks for blend shapes and parameters
//! - Multiple loop modes (once, loop, ping-pong)
//!
//! # Architecture
//!
//! ```text
//! AnimationClip
//! ├── BoneTrack[]           # Per-bone animation channels
//! │   ├── Track<Vec3>       # Position keyframes
//! │   ├── Track<Quat>       # Rotation keyframes
//! │   └── Track<Vec3>       # Scale keyframes
//! ├── EventTrack[]          # Animation notifies/events
//! └── CurveTrack[]          # Float parameter curves
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::animation_clip::{AnimationClip, BoneTrack, Track, Keyframe, InterpolationMode, LoopMode};
//! use glam::{Vec3, Quat};
//!
//! // Create a simple walk animation
//! let mut clip = AnimationClip::new("walk", 1.0);
//! clip.looping = LoopMode::Loop;
//!
//! // Add hip bone animation
//! let mut hip_track = BoneTrack::new("hip");
//! hip_track.position = Some(Track::from_keyframes(vec![
//!     Keyframe::linear(0.0, Vec3::new(0.0, 1.0, 0.0)),
//!     Keyframe::linear(0.5, Vec3::new(0.0, 1.1, 0.0)),
//!     Keyframe::linear(1.0, Vec3::new(0.0, 1.0, 0.0)),
//! ]));
//! clip.add_bone_track(hip_track);
//!
//! // Sample at time 0.25
//! let pose = clip.sample(0.25);
//! ```

use std::collections::HashMap;
use std::fmt;

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::skeleton::Transform;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default frame rate for animation clips.
pub const DEFAULT_FRAME_RATE: f32 = 30.0;

/// Maximum number of bones per animation clip.
pub const MAX_ANIMATED_BONES: usize = 256;

/// Maximum number of events per animation clip.
pub const MAX_EVENTS_PER_CLIP: usize = 1024;

/// Maximum number of curve tracks per clip.
pub const MAX_CURVES_PER_CLIP: usize = 64;

// ---------------------------------------------------------------------------
// InterpolationMode
// ---------------------------------------------------------------------------

/// Interpolation method for keyframe values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InterpolationMode {
    /// Constant value until next keyframe (no interpolation).
    Step,

    /// Linear interpolation between keyframes.
    #[default]
    Linear,

    /// Cubic Hermite spline interpolation using tangents.
    Cubic,
}

impl InterpolationMode {
    /// Returns true if this mode requires tangent data.
    #[inline]
    pub fn requires_tangents(&self) -> bool {
        matches!(self, Self::Cubic)
    }
}

impl fmt::Display for InterpolationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Step => write!(f, "step"),
            Self::Linear => write!(f, "linear"),
            Self::Cubic => write!(f, "cubic"),
        }
    }
}

// ---------------------------------------------------------------------------
// LoopMode
// ---------------------------------------------------------------------------

/// Playback behavior when animation reaches its end.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoopMode {
    /// Play once and stop at the last frame.
    #[default]
    Once,

    /// Loop from start when reaching end.
    Loop,

    /// Alternate between forward and backward playback.
    PingPong,
}

impl LoopMode {
    /// Calculate the effective playback time for the given raw time.
    ///
    /// Returns (effective_time, is_reversed) where is_reversed indicates
    /// if we're playing backwards in ping-pong mode.
    #[inline]
    pub fn calculate_time(&self, time: f32, duration: f32) -> (f32, bool) {
        if duration <= 0.0 {
            return (0.0, false);
        }

        match self {
            Self::Once => {
                let clamped = time.clamp(0.0, duration);
                (clamped, false)
            }
            Self::Loop => {
                let wrapped = time.rem_euclid(duration);
                (wrapped, false)
            }
            Self::PingPong => {
                // Each "cycle" is 2 * duration (forward + backward)
                let cycle_time = time.rem_euclid(duration * 2.0);
                if cycle_time <= duration {
                    (cycle_time, false)
                } else {
                    // Reverse direction
                    let reversed_time = duration * 2.0 - cycle_time;
                    (reversed_time, true)
                }
            }
        }
    }

    /// Check if the animation is complete at the given time.
    #[inline]
    pub fn is_complete(&self, time: f32, duration: f32) -> bool {
        match self {
            Self::Once => time >= duration,
            Self::Loop | Self::PingPong => false, // Never complete
        }
    }
}

impl fmt::Display for LoopMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Once => write!(f, "once"),
            Self::Loop => write!(f, "loop"),
            Self::PingPong => write!(f, "ping-pong"),
        }
    }
}

// ---------------------------------------------------------------------------
// Keyframe<T>
// ---------------------------------------------------------------------------

/// A single keyframe in an animation track.
///
/// Keyframes store a time, value, interpolation mode, and optional tangents
/// for cubic interpolation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Keyframe<T> {
    /// Time of this keyframe in seconds from clip start.
    pub time: f32,

    /// Value at this keyframe.
    pub value: T,

    /// How to interpolate to the next keyframe.
    pub interpolation: InterpolationMode,

    /// Incoming tangent for cubic interpolation.
    pub in_tangent: Option<T>,

    /// Outgoing tangent for cubic interpolation.
    pub out_tangent: Option<T>,
}

impl<T: Clone> Keyframe<T> {
    /// Create a new keyframe with step interpolation.
    #[inline]
    pub fn step(time: f32, value: T) -> Self {
        Self {
            time,
            value,
            interpolation: InterpolationMode::Step,
            in_tangent: None,
            out_tangent: None,
        }
    }

    /// Create a new keyframe with linear interpolation.
    #[inline]
    pub fn linear(time: f32, value: T) -> Self {
        Self {
            time,
            value,
            interpolation: InterpolationMode::Linear,
            in_tangent: None,
            out_tangent: None,
        }
    }

    /// Create a new keyframe with cubic interpolation.
    #[inline]
    pub fn cubic(time: f32, value: T, in_tangent: T, out_tangent: T) -> Self {
        Self {
            time,
            value,
            interpolation: InterpolationMode::Cubic,
            in_tangent: Some(in_tangent),
            out_tangent: Some(out_tangent),
        }
    }

    /// Create a keyframe with explicit interpolation mode.
    #[inline]
    pub fn new(time: f32, value: T, interpolation: InterpolationMode) -> Self {
        Self {
            time,
            value,
            interpolation,
            in_tangent: None,
            out_tangent: None,
        }
    }

    /// Set tangents for cubic interpolation.
    #[inline]
    pub fn with_tangents(mut self, in_tangent: T, out_tangent: T) -> Self {
        self.in_tangent = Some(in_tangent);
        self.out_tangent = Some(out_tangent);
        self
    }
}

impl<T: Default + Clone> Default for Keyframe<T> {
    fn default() -> Self {
        Self {
            time: 0.0,
            value: T::default(),
            interpolation: InterpolationMode::Linear,
            in_tangent: None,
            out_tangent: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Track<T>
// ---------------------------------------------------------------------------

/// A track containing keyframes for a single animated property.
///
/// Tracks are sorted by time and support efficient binary search for sampling.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Track<T> {
    /// Keyframes sorted by time.
    pub keyframes: Vec<Keyframe<T>>,
}

impl<T: Clone> Track<T> {
    /// Create an empty track.
    #[inline]
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Create a track with the given keyframes.
    ///
    /// Keyframes will be sorted by time.
    pub fn from_keyframes(mut keyframes: Vec<Keyframe<T>>) -> Self {
        keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
        Self { keyframes }
    }

    /// Add a keyframe to the track, maintaining sorted order.
    pub fn add_keyframe(&mut self, keyframe: Keyframe<T>) {
        let pos = self
            .keyframes
            .binary_search_by(|k| k.time.partial_cmp(&keyframe.time).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(|pos| pos);
        self.keyframes.insert(pos, keyframe);
    }

    /// Remove a keyframe at the given index.
    #[inline]
    pub fn remove_keyframe(&mut self, index: usize) -> Option<Keyframe<T>> {
        if index < self.keyframes.len() {
            Some(self.keyframes.remove(index))
        } else {
            None
        }
    }

    /// Get the number of keyframes.
    #[inline]
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    /// Check if the track is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Get the duration of this track (time of last keyframe).
    #[inline]
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map(|k| k.time).unwrap_or(0.0)
    }

    /// Get the time range of this track.
    #[inline]
    pub fn time_range(&self) -> Option<(f32, f32)> {
        if self.keyframes.is_empty() {
            None
        } else {
            Some((
                self.keyframes.first().map(|k| k.time).unwrap_or(0.0),
                self.keyframes.last().map(|k| k.time).unwrap_or(0.0),
            ))
        }
    }

    /// Find the keyframe indices surrounding the given time.
    ///
    /// Returns (prev_index, next_index, normalized_t) where normalized_t
    /// is the interpolation factor between the two keyframes [0, 1].
    pub fn find_keyframes(&self, time: f32) -> Option<(usize, usize, f32)> {
        if self.keyframes.is_empty() {
            return None;
        }

        // Before first keyframe
        if time <= self.keyframes[0].time {
            return Some((0, 0, 0.0));
        }

        // After last keyframe
        let last_idx = self.keyframes.len() - 1;
        if time >= self.keyframes[last_idx].time {
            return Some((last_idx, last_idx, 0.0));
        }

        // Binary search for the keyframe pair
        let next_idx = self
            .keyframes
            .binary_search_by(|k| {
                if k.time <= time {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            })
            .unwrap_or_else(|pos| pos);

        let prev_idx = next_idx.saturating_sub(1);

        let prev_time = self.keyframes[prev_idx].time;
        let next_time = self.keyframes[next_idx].time;
        let range = next_time - prev_time;

        let t = if range > f32::EPSILON {
            (time - prev_time) / range
        } else {
            0.0
        };

        Some((prev_idx, next_idx, t))
    }
}

// ---------------------------------------------------------------------------
// Track<Vec3> sampling
// ---------------------------------------------------------------------------

impl Track<Vec3> {
    /// Sample the track at the given time.
    pub fn sample(&self, time: f32) -> Option<Vec3> {
        let (prev_idx, next_idx, t) = self.find_keyframes(time)?;

        let prev = &self.keyframes[prev_idx];
        let next = &self.keyframes[next_idx];

        Some(match prev.interpolation {
            InterpolationMode::Step => prev.value,
            InterpolationMode::Linear => prev.value.lerp(next.value, t),
            InterpolationMode::Cubic => {
                let in_tan = prev.out_tangent.unwrap_or(Vec3::ZERO);
                let out_tan = next.in_tangent.unwrap_or(Vec3::ZERO);
                hermite_vec3(prev.value, next.value, in_tan, out_tan, t)
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Track<Quat> sampling
// ---------------------------------------------------------------------------

impl Track<Quat> {
    /// Sample the track at the given time.
    pub fn sample(&self, time: f32) -> Option<Quat> {
        let (prev_idx, next_idx, t) = self.find_keyframes(time)?;

        let prev = &self.keyframes[prev_idx];
        let next = &self.keyframes[next_idx];

        Some(match prev.interpolation {
            InterpolationMode::Step => prev.value,
            InterpolationMode::Linear => prev.value.slerp(next.value, t),
            InterpolationMode::Cubic => {
                // For quaternions, we use spherical cubic interpolation (squad)
                // Simplified version using slerp for now
                prev.value.slerp(next.value, t)
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Track<f32> sampling
// ---------------------------------------------------------------------------

impl Track<f32> {
    /// Sample the track at the given time.
    pub fn sample(&self, time: f32) -> Option<f32> {
        let (prev_idx, next_idx, t) = self.find_keyframes(time)?;

        let prev = &self.keyframes[prev_idx];
        let next = &self.keyframes[next_idx];

        Some(match prev.interpolation {
            InterpolationMode::Step => prev.value,
            InterpolationMode::Linear => {
                prev.value + (next.value - prev.value) * t
            }
            InterpolationMode::Cubic => {
                let in_tan = prev.out_tangent.unwrap_or(0.0);
                let out_tan = next.in_tangent.unwrap_or(0.0);
                hermite_f32(prev.value, next.value, in_tan, out_tan, t)
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Hermite interpolation helpers
// ---------------------------------------------------------------------------

/// Cubic Hermite interpolation for Vec3.
fn hermite_vec3(p0: Vec3, p1: Vec3, m0: Vec3, m1: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;

    // Hermite basis functions
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11
}

/// Cubic Hermite interpolation for f32.
fn hermite_f32(p0: f32, p1: f32, m0: f32, m1: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;

    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11
}

// ---------------------------------------------------------------------------
// BoneTrack
// ---------------------------------------------------------------------------

/// Animation data for a single bone.
///
/// Each property (position, rotation, scale) is optional - bones that don't
/// change in a particular dimension can omit that track for efficiency.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BoneTrack {
    /// Name of the target bone.
    pub bone_name: String,

    /// Position animation track.
    pub position: Option<Track<Vec3>>,

    /// Rotation animation track.
    pub rotation: Option<Track<Quat>>,

    /// Scale animation track.
    pub scale: Option<Track<Vec3>>,
}

impl BoneTrack {
    /// Create a new bone track for the given bone.
    #[inline]
    pub fn new(bone_name: impl Into<String>) -> Self {
        Self {
            bone_name: bone_name.into(),
            position: None,
            rotation: None,
            scale: None,
        }
    }

    /// Set the position track.
    #[inline]
    pub fn with_position(mut self, track: Track<Vec3>) -> Self {
        self.position = Some(track);
        self
    }

    /// Set the rotation track.
    #[inline]
    pub fn with_rotation(mut self, track: Track<Quat>) -> Self {
        self.rotation = Some(track);
        self
    }

    /// Set the scale track.
    #[inline]
    pub fn with_scale(mut self, track: Track<Vec3>) -> Self {
        self.scale = Some(track);
        self
    }

    /// Sample all tracks at the given time.
    ///
    /// Returns a Transform with interpolated values. Properties without
    /// tracks return identity values.
    pub fn sample(&self, time: f32) -> Transform {
        let position = self
            .position
            .as_ref()
            .and_then(|t| t.sample(time))
            .unwrap_or(Vec3::ZERO);

        let rotation = self
            .rotation
            .as_ref()
            .and_then(|t| t.sample(time))
            .unwrap_or(Quat::IDENTITY);

        let scale = self
            .scale
            .as_ref()
            .and_then(|t| t.sample(time))
            .unwrap_or(Vec3::ONE);

        Transform::new(position, rotation, scale)
    }

    /// Check if this track has any animation data.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.position.as_ref().map_or(true, |t| t.is_empty())
            && self.rotation.as_ref().map_or(true, |t| t.is_empty())
            && self.scale.as_ref().map_or(true, |t| t.is_empty())
    }

    /// Get the duration of this bone track.
    pub fn duration(&self) -> f32 {
        let pos_dur = self.position.as_ref().map_or(0.0, |t| t.duration());
        let rot_dur = self.rotation.as_ref().map_or(0.0, |t| t.duration());
        let scale_dur = self.scale.as_ref().map_or(0.0, |t| t.duration());
        pos_dur.max(rot_dur).max(scale_dur)
    }

    /// Get total keyframe count across all tracks.
    pub fn keyframe_count(&self) -> usize {
        self.position.as_ref().map_or(0, |t| t.len())
            + self.rotation.as_ref().map_or(0, |t| t.len())
            + self.scale.as_ref().map_or(0, |t| t.len())
    }
}

// ---------------------------------------------------------------------------
// AnimationEvent
// ---------------------------------------------------------------------------

/// An event that fires at a specific time in an animation.
///
/// Events are used for gameplay callbacks like footstep sounds,
/// particle spawning, or state machine transitions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimationEvent {
    /// Name of the event.
    pub name: String,

    /// Time at which the event fires (seconds from clip start).
    pub time: f32,

    /// Optional string payload.
    pub string_param: Option<String>,

    /// Optional float payload.
    pub float_param: Option<f32>,

    /// Optional integer payload.
    pub int_param: Option<i32>,
}

impl AnimationEvent {
    /// Create a new animation event.
    #[inline]
    pub fn new(name: impl Into<String>, time: f32) -> Self {
        Self {
            name: name.into(),
            time,
            string_param: None,
            float_param: None,
            int_param: None,
        }
    }

    /// Set a string parameter.
    #[inline]
    pub fn with_string(mut self, value: impl Into<String>) -> Self {
        self.string_param = Some(value.into());
        self
    }

    /// Set a float parameter.
    #[inline]
    pub fn with_float(mut self, value: f32) -> Self {
        self.float_param = Some(value);
        self
    }

    /// Set an integer parameter.
    #[inline]
    pub fn with_int(mut self, value: i32) -> Self {
        self.int_param = Some(value);
        self
    }
}

impl Default for AnimationEvent {
    fn default() -> Self {
        Self {
            name: String::new(),
            time: 0.0,
            string_param: None,
            float_param: None,
            int_param: None,
        }
    }
}

// ---------------------------------------------------------------------------
// EventTrack
// ---------------------------------------------------------------------------

/// A track containing animation events.
///
/// Events are sorted by time for efficient range queries.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EventTrack {
    /// Track name/identifier.
    pub name: String,

    /// Events sorted by time.
    pub events: Vec<AnimationEvent>,
}

impl EventTrack {
    /// Create a new event track.
    #[inline]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            events: Vec::new(),
        }
    }

    /// Add an event to the track, maintaining sorted order.
    pub fn add_event(&mut self, event: AnimationEvent) {
        let pos = self
            .events
            .binary_search_by(|e| e.time.partial_cmp(&event.time).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(|pos| pos);
        self.events.insert(pos, event);
    }

    /// Get events in the given time range [start, end).
    pub fn events_in_range(&self, start: f32, end: f32) -> Vec<&AnimationEvent> {
        self.events
            .iter()
            .filter(|e| e.time >= start && e.time < end)
            .collect()
    }

    /// Get the number of events.
    #[inline]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if the track is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

// ---------------------------------------------------------------------------
// CurveTrack
// ---------------------------------------------------------------------------

/// A named float curve for animating parameters like blend shapes or material values.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CurveTrack {
    /// Name of the curve (e.g., "blendshape.smile", "material.alpha").
    pub name: String,

    /// Float keyframe track.
    pub track: Track<f32>,
}

impl CurveTrack {
    /// Create a new curve track.
    #[inline]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            track: Track::new(),
        }
    }

    /// Create a curve track with keyframes.
    pub fn with_keyframes(name: impl Into<String>, keyframes: Vec<Keyframe<f32>>) -> Self {
        Self {
            name: name.into(),
            track: Track::from_keyframes(keyframes),
        }
    }

    /// Sample the curve at the given time.
    #[inline]
    pub fn sample(&self, time: f32) -> Option<f32> {
        self.track.sample(time)
    }

    /// Get the duration of this curve.
    #[inline]
    pub fn duration(&self) -> f32 {
        self.track.duration()
    }
}

// ---------------------------------------------------------------------------
// Pose
// ---------------------------------------------------------------------------

/// A snapshot of bone transforms at a point in time.
///
/// This is the output of sampling an AnimationClip.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    /// Map from bone name to transform.
    pub bone_transforms: HashMap<String, Transform>,
}

impl Pose {
    /// Create an empty pose.
    #[inline]
    pub fn new() -> Self {
        Self {
            bone_transforms: HashMap::new(),
        }
    }

    /// Create a pose with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bone_transforms: HashMap::with_capacity(capacity),
        }
    }

    /// Set a bone transform.
    #[inline]
    pub fn set(&mut self, bone_name: impl Into<String>, transform: Transform) {
        self.bone_transforms.insert(bone_name.into(), transform);
    }

    /// Get a bone transform.
    #[inline]
    pub fn get(&self, bone_name: &str) -> Option<&Transform> {
        self.bone_transforms.get(bone_name)
    }

    /// Get the number of bones in the pose.
    #[inline]
    pub fn len(&self) -> usize {
        self.bone_transforms.len()
    }

    /// Check if the pose is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bone_transforms.is_empty()
    }

    /// Iterate over bone names and transforms.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Transform)> {
        self.bone_transforms.iter()
    }

    /// Blend two poses together.
    ///
    /// Returns a new pose where each bone transform is interpolated
    /// between self and other by factor t (0 = self, 1 = other).
    pub fn blend(&self, other: &Pose, t: f32) -> Pose {
        let mut result = Pose::with_capacity(self.len().max(other.len()));

        // Blend bones from self
        for (name, transform) in &self.bone_transforms {
            let blended = match other.bone_transforms.get(name) {
                Some(other_transform) => transform.lerp(other_transform, t),
                None => *transform,
            };
            result.bone_transforms.insert(name.clone(), blended);
        }

        // Add bones only in other
        for (name, transform) in &other.bone_transforms {
            if !self.bone_transforms.contains_key(name) {
                result.bone_transforms.insert(name.clone(), *transform);
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// AnimationClipError
// ---------------------------------------------------------------------------

/// Errors that can occur during animation clip operations.
#[derive(Clone, Debug, PartialEq)]
pub enum AnimationClipError {
    /// Bone track not found.
    BoneTrackNotFound { bone_name: String },

    /// Invalid time value.
    InvalidTime { time: f32 },

    /// Too many bone tracks.
    TooManyBoneTracks { count: usize, max: usize },

    /// Too many events.
    TooManyEvents { count: usize, max: usize },

    /// Too many curve tracks.
    TooManyCurveTracks { count: usize, max: usize },

    /// Empty animation clip.
    EmptyClip,
}

impl fmt::Display for AnimationClipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BoneTrackNotFound { bone_name } => {
                write!(f, "bone track '{}' not found", bone_name)
            }
            Self::InvalidTime { time } => {
                write!(f, "invalid time value: {}", time)
            }
            Self::TooManyBoneTracks { count, max } => {
                write!(f, "too many bone tracks: {} (max {})", count, max)
            }
            Self::TooManyEvents { count, max } => {
                write!(f, "too many events: {} (max {})", count, max)
            }
            Self::TooManyCurveTracks { count, max } => {
                write!(f, "too many curve tracks: {} (max {})", count, max)
            }
            Self::EmptyClip => write!(f, "animation clip has no data"),
        }
    }
}

impl std::error::Error for AnimationClipError {}

// ---------------------------------------------------------------------------
// AnimationClip
// ---------------------------------------------------------------------------

/// A complete animation clip containing keyframed animation data.
///
/// Animation clips store per-bone animation tracks, animation events,
/// and float curve tracks. They support various loop modes and can be
/// sampled at any time to produce a Pose.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AnimationClip {
    /// Name of the animation clip.
    pub name: String,

    /// Duration of the clip in seconds.
    pub duration: f32,

    /// Frame rate used for authoring (affects preview).
    pub frame_rate: f32,

    /// How the animation loops.
    pub looping: LoopMode,

    /// Per-bone animation tracks.
    pub bone_tracks: Vec<BoneTrack>,

    /// Animation event tracks.
    pub event_tracks: Vec<EventTrack>,

    /// Float curve tracks for parameters.
    pub curve_tracks: Vec<CurveTrack>,

    /// Map from bone name to track index for O(1) lookup.
    #[serde(skip)]
    bone_name_to_index: HashMap<String, usize>,
}

impl AnimationClip {
    /// Create a new animation clip.
    #[inline]
    pub fn new(name: impl Into<String>, duration: f32) -> Self {
        Self {
            name: name.into(),
            duration,
            frame_rate: DEFAULT_FRAME_RATE,
            looping: LoopMode::Once,
            bone_tracks: Vec::new(),
            event_tracks: Vec::new(),
            curve_tracks: Vec::new(),
            bone_name_to_index: HashMap::new(),
        }
    }

    /// Create a looping animation clip.
    #[inline]
    pub fn looping(name: impl Into<String>, duration: f32) -> Self {
        Self {
            looping: LoopMode::Loop,
            ..Self::new(name, duration)
        }
    }

    /// Set the loop mode.
    #[inline]
    pub fn with_loop_mode(mut self, mode: LoopMode) -> Self {
        self.looping = mode;
        self
    }

    /// Set the frame rate.
    #[inline]
    pub fn with_frame_rate(mut self, frame_rate: f32) -> Self {
        self.frame_rate = frame_rate;
        self
    }

    /// Add a bone track.
    pub fn add_bone_track(&mut self, track: BoneTrack) {
        let index = self.bone_tracks.len();
        self.bone_name_to_index.insert(track.bone_name.clone(), index);
        self.bone_tracks.push(track);
    }

    /// Add an event track.
    #[inline]
    pub fn add_event_track(&mut self, track: EventTrack) {
        self.event_tracks.push(track);
    }

    /// Add a curve track.
    #[inline]
    pub fn add_curve_track(&mut self, track: CurveTrack) {
        self.curve_tracks.push(track);
    }

    /// Get a bone track by name.
    pub fn bone_track(&self, bone_name: &str) -> Option<&BoneTrack> {
        self.bone_name_to_index
            .get(bone_name)
            .and_then(|&idx| self.bone_tracks.get(idx))
    }

    /// Get a mutable bone track by name.
    pub fn bone_track_mut(&mut self, bone_name: &str) -> Option<&mut BoneTrack> {
        self.bone_name_to_index
            .get(bone_name)
            .copied()
            .and_then(move |idx| self.bone_tracks.get_mut(idx))
    }

    /// Get the number of animated bones.
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.bone_tracks.len()
    }

    /// Sample the entire clip at the given time.
    ///
    /// Returns a Pose containing transforms for all animated bones.
    pub fn sample(&self, time: f32) -> Pose {
        let (effective_time, _reversed) = self.looping.calculate_time(time, self.duration);

        let mut pose = Pose::with_capacity(self.bone_tracks.len());

        for track in &self.bone_tracks {
            let transform = track.sample(effective_time);
            pose.set(track.bone_name.clone(), transform);
        }

        pose
    }

    /// Sample a specific bone at the given time.
    pub fn sample_bone(&self, bone_name: &str, time: f32) -> Option<Transform> {
        let (effective_time, _reversed) = self.looping.calculate_time(time, self.duration);

        self.bone_track(bone_name).map(|track| track.sample(effective_time))
    }

    /// Get all events that fire in the given time range.
    ///
    /// This handles looping correctly - if the animation loops during
    /// the range, events from multiple loops will be included.
    pub fn events_in_range(&self, start: f32, end: f32) -> Vec<&AnimationEvent> {
        let mut result = Vec::new();

        // Calculate effective times
        let (start_time, _) = self.looping.calculate_time(start, self.duration);
        let (end_time, _) = self.looping.calculate_time(end, self.duration);

        for track in &self.event_tracks {
            if start_time <= end_time {
                // Normal case - no wrap
                result.extend(track.events_in_range(start_time, end_time));
            } else {
                // Wrapped around - get events from end of animation and start
                result.extend(track.events_in_range(start_time, self.duration));
                result.extend(track.events_in_range(0.0, end_time));
            }
        }

        result
    }

    /// Sample a curve track at the given time.
    pub fn sample_curve(&self, curve_name: &str, time: f32) -> Option<f32> {
        let (effective_time, _) = self.looping.calculate_time(time, self.duration);

        self.curve_tracks
            .iter()
            .find(|c| c.name == curve_name)
            .and_then(|c| c.sample(effective_time))
    }

    /// Get total keyframe count across all tracks.
    pub fn total_keyframe_count(&self) -> usize {
        self.bone_tracks.iter().map(|t| t.keyframe_count()).sum::<usize>()
            + self.curve_tracks.iter().map(|c| c.track.len()).sum::<usize>()
    }

    /// Get total event count.
    pub fn total_event_count(&self) -> usize {
        self.event_tracks.iter().map(|t| t.len()).sum()
    }

    /// Validate the animation clip.
    pub fn validate(&self) -> Result<(), AnimationClipError> {
        if self.bone_tracks.len() > MAX_ANIMATED_BONES {
            return Err(AnimationClipError::TooManyBoneTracks {
                count: self.bone_tracks.len(),
                max: MAX_ANIMATED_BONES,
            });
        }

        let event_count = self.total_event_count();
        if event_count > MAX_EVENTS_PER_CLIP {
            return Err(AnimationClipError::TooManyEvents {
                count: event_count,
                max: MAX_EVENTS_PER_CLIP,
            });
        }

        if self.curve_tracks.len() > MAX_CURVES_PER_CLIP {
            return Err(AnimationClipError::TooManyCurveTracks {
                count: self.curve_tracks.len(),
                max: MAX_CURVES_PER_CLIP,
            });
        }

        Ok(())
    }

    /// Rebuild internal indices after deserialization.
    pub fn rebuild_indices(&mut self) {
        self.bone_name_to_index.clear();
        for (i, track) in self.bone_tracks.iter().enumerate() {
            self.bone_name_to_index.insert(track.bone_name.clone(), i);
        }
    }

    /// Compute duration from track data.
    pub fn compute_duration(&self) -> f32 {
        let bone_dur = self
            .bone_tracks
            .iter()
            .map(|t| t.duration())
            .fold(0.0f32, f32::max);

        let curve_dur = self
            .curve_tracks
            .iter()
            .map(|c| c.duration())
            .fold(0.0f32, f32::max);

        let event_dur = self
            .event_tracks
            .iter()
            .flat_map(|t| t.events.last())
            .map(|e| e.time)
            .fold(0.0f32, f32::max);

        bone_dur.max(curve_dur).max(event_dur)
    }

    /// Convert to frame number.
    #[inline]
    pub fn time_to_frame(&self, time: f32) -> u32 {
        (time * self.frame_rate).round() as u32
    }

    /// Convert from frame number.
    #[inline]
    pub fn frame_to_time(&self, frame: u32) -> f32 {
        frame as f32 / self.frame_rate
    }

    /// Get the number of frames in this clip.
    #[inline]
    pub fn frame_count(&self) -> u32 {
        (self.duration * self.frame_rate).ceil() as u32
    }

    /// Check if the animation is complete at the given time.
    #[inline]
    pub fn is_complete(&self, time: f32) -> bool {
        self.looping.is_complete(time, self.duration)
    }

    /// Create a JSON representation.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Create from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let mut clip: Self = serde_json::from_str(json)?;
        clip.rebuild_indices();
        Ok(clip)
    }
}

// ---------------------------------------------------------------------------
// AnimationClipBuilder
// ---------------------------------------------------------------------------

/// Builder for creating animation clips with a fluent API.
#[derive(Default)]
pub struct AnimationClipBuilder {
    clip: AnimationClip,
}

impl AnimationClipBuilder {
    /// Create a new builder.
    pub fn new(name: impl Into<String>, duration: f32) -> Self {
        Self {
            clip: AnimationClip::new(name, duration),
        }
    }

    /// Set the loop mode.
    pub fn loop_mode(mut self, mode: LoopMode) -> Self {
        self.clip.looping = mode;
        self
    }

    /// Set the frame rate.
    pub fn frame_rate(mut self, rate: f32) -> Self {
        self.clip.frame_rate = rate;
        self
    }

    /// Add a bone track.
    pub fn bone_track(mut self, track: BoneTrack) -> Self {
        self.clip.add_bone_track(track);
        self
    }

    /// Add an event track.
    pub fn event_track(mut self, track: EventTrack) -> Self {
        self.clip.add_event_track(track);
        self
    }

    /// Add a curve track.
    pub fn curve_track(mut self, track: CurveTrack) -> Self {
        self.clip.add_curve_track(track);
        self
    }

    /// Build the animation clip.
    pub fn build(self) -> Result<AnimationClip, AnimationClipError> {
        self.clip.validate()?;
        Ok(self.clip)
    }

    /// Build without validation.
    pub fn build_unchecked(self) -> AnimationClip {
        self.clip
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // ===== InterpolationMode Tests =====

    #[test]
    fn test_interpolation_mode_default() {
        let mode = InterpolationMode::default();
        assert_eq!(mode, InterpolationMode::Linear);
    }

    #[test]
    fn test_interpolation_mode_requires_tangents() {
        assert!(!InterpolationMode::Step.requires_tangents());
        assert!(!InterpolationMode::Linear.requires_tangents());
        assert!(InterpolationMode::Cubic.requires_tangents());
    }

    #[test]
    fn test_interpolation_mode_display() {
        assert_eq!(format!("{}", InterpolationMode::Step), "step");
        assert_eq!(format!("{}", InterpolationMode::Linear), "linear");
        assert_eq!(format!("{}", InterpolationMode::Cubic), "cubic");
    }

    // ===== LoopMode Tests =====

    #[test]
    fn test_loop_mode_default() {
        let mode = LoopMode::default();
        assert_eq!(mode, LoopMode::Once);
    }

    #[test]
    fn test_loop_mode_once_calculate_time() {
        let mode = LoopMode::Once;
        let duration = 2.0;

        // Before start
        assert_eq!(mode.calculate_time(-1.0, duration), (0.0, false));

        // Within range
        assert_eq!(mode.calculate_time(1.0, duration), (1.0, false));

        // After end
        assert_eq!(mode.calculate_time(3.0, duration), (2.0, false));
    }

    #[test]
    fn test_loop_mode_loop_calculate_time() {
        let mode = LoopMode::Loop;
        let duration = 2.0;

        // First loop
        let (t, rev) = mode.calculate_time(1.0, duration);
        assert!((t - 1.0).abs() < 1e-5);
        assert!(!rev);

        // Second loop
        let (t, rev) = mode.calculate_time(3.0, duration);
        assert!((t - 1.0).abs() < 1e-5);
        assert!(!rev);

        // At boundary
        let (t, _) = mode.calculate_time(2.0, duration);
        assert!((t - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_loop_mode_pingpong_calculate_time() {
        let mode = LoopMode::PingPong;
        let duration = 2.0;

        // Forward
        let (t, rev) = mode.calculate_time(1.0, duration);
        assert!((t - 1.0).abs() < 1e-5);
        assert!(!rev);

        // Backward
        let (t, rev) = mode.calculate_time(3.0, duration);
        assert!((t - 1.0).abs() < 1e-5);
        assert!(rev);

        // At peak
        let (t, _) = mode.calculate_time(2.0, duration);
        assert!((t - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_loop_mode_is_complete() {
        assert!(!LoopMode::Once.is_complete(0.5, 1.0));
        assert!(LoopMode::Once.is_complete(1.0, 1.0));
        assert!(LoopMode::Once.is_complete(1.5, 1.0));

        assert!(!LoopMode::Loop.is_complete(100.0, 1.0));
        assert!(!LoopMode::PingPong.is_complete(100.0, 1.0));
    }

    #[test]
    fn test_loop_mode_zero_duration() {
        assert_eq!(LoopMode::Once.calculate_time(1.0, 0.0), (0.0, false));
        assert_eq!(LoopMode::Loop.calculate_time(1.0, 0.0), (0.0, false));
        assert_eq!(LoopMode::PingPong.calculate_time(1.0, 0.0), (0.0, false));
    }

    #[test]
    fn test_loop_mode_display() {
        assert_eq!(format!("{}", LoopMode::Once), "once");
        assert_eq!(format!("{}", LoopMode::Loop), "loop");
        assert_eq!(format!("{}", LoopMode::PingPong), "ping-pong");
    }

    // ===== Keyframe Tests =====

    #[test]
    fn test_keyframe_step() {
        let kf = Keyframe::step(0.5, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(kf.time, 0.5);
        assert_eq!(kf.interpolation, InterpolationMode::Step);
        assert!(kf.in_tangent.is_none());
        assert!(kf.out_tangent.is_none());
    }

    #[test]
    fn test_keyframe_linear() {
        let kf = Keyframe::linear(1.0, Vec3::ONE);
        assert_eq!(kf.time, 1.0);
        assert_eq!(kf.interpolation, InterpolationMode::Linear);
    }

    #[test]
    fn test_keyframe_cubic() {
        let kf = Keyframe::cubic(0.0, Vec3::ZERO, Vec3::X, Vec3::Y);
        assert_eq!(kf.interpolation, InterpolationMode::Cubic);
        assert_eq!(kf.in_tangent, Some(Vec3::X));
        assert_eq!(kf.out_tangent, Some(Vec3::Y));
    }

    #[test]
    fn test_keyframe_with_tangents() {
        let kf = Keyframe::new(0.0, 1.0f32, InterpolationMode::Cubic)
            .with_tangents(0.5, 0.5);
        assert_eq!(kf.in_tangent, Some(0.5));
        assert_eq!(kf.out_tangent, Some(0.5));
    }

    #[test]
    fn test_keyframe_default() {
        let kf: Keyframe<f32> = Keyframe::default();
        assert_eq!(kf.time, 0.0);
        assert_eq!(kf.value, 0.0);
        assert_eq!(kf.interpolation, InterpolationMode::Linear);
    }

    // ===== Track Tests =====

    #[test]
    fn test_track_new() {
        let track: Track<f32> = Track::new();
        assert!(track.is_empty());
        assert_eq!(track.len(), 0);
    }

    #[test]
    fn test_track_from_keyframes() {
        let keyframes = vec![
            Keyframe::linear(1.0, 1.0f32),
            Keyframe::linear(0.0, 0.0f32),
            Keyframe::linear(0.5, 0.5f32),
        ];
        let track = Track::from_keyframes(keyframes);

        // Should be sorted by time
        assert_eq!(track.keyframes[0].time, 0.0);
        assert_eq!(track.keyframes[1].time, 0.5);
        assert_eq!(track.keyframes[2].time, 1.0);
    }

    #[test]
    fn test_track_add_keyframe() {
        let mut track: Track<f32> = Track::new();
        track.add_keyframe(Keyframe::linear(1.0, 1.0));
        track.add_keyframe(Keyframe::linear(0.0, 0.0));
        track.add_keyframe(Keyframe::linear(0.5, 0.5));

        assert_eq!(track.len(), 3);
        assert_eq!(track.keyframes[0].time, 0.0);
        assert_eq!(track.keyframes[1].time, 0.5);
        assert_eq!(track.keyframes[2].time, 1.0);
    }

    #[test]
    fn test_track_remove_keyframe() {
        let mut track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, 1.0f32),
            Keyframe::linear(1.0, 2.0f32),
        ]);

        let removed = track.remove_keyframe(0);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().value, 1.0);
        assert_eq!(track.len(), 1);

        assert!(track.remove_keyframe(100).is_none());
    }

    #[test]
    fn test_track_duration() {
        let track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, 0.0f32),
            Keyframe::linear(2.5, 1.0f32),
        ]);
        assert_eq!(track.duration(), 2.5);
    }

    #[test]
    fn test_track_time_range() {
        let track = Track::from_keyframes(vec![
            Keyframe::linear(0.5, 0.0f32),
            Keyframe::linear(2.5, 1.0f32),
        ]);
        assert_eq!(track.time_range(), Some((0.5, 2.5)));

        let empty: Track<f32> = Track::new();
        assert_eq!(empty.time_range(), None);
    }

    #[test]
    fn test_track_find_keyframes_empty() {
        let track: Track<f32> = Track::new();
        assert!(track.find_keyframes(0.5).is_none());
    }

    #[test]
    fn test_track_find_keyframes_single() {
        let track = Track::from_keyframes(vec![Keyframe::linear(1.0, 0.5f32)]);

        // Before
        let (prev, next, t) = track.find_keyframes(0.5).unwrap();
        assert_eq!(prev, 0);
        assert_eq!(next, 0);
        assert_eq!(t, 0.0);

        // After
        let (prev, next, t) = track.find_keyframes(2.0).unwrap();
        assert_eq!(prev, 0);
        assert_eq!(next, 0);
        assert_eq!(t, 0.0);
    }

    #[test]
    fn test_track_find_keyframes_interpolation() {
        let track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, 0.0f32),
            Keyframe::linear(1.0, 1.0f32),
        ]);

        let (prev, next, t) = track.find_keyframes(0.5).unwrap();
        assert_eq!(prev, 0);
        assert_eq!(next, 1);
        assert!((t - 0.5).abs() < 1e-5);

        let (prev, next, t) = track.find_keyframes(0.25).unwrap();
        assert_eq!(prev, 0);
        assert_eq!(next, 1);
        assert!((t - 0.25).abs() < 1e-5);
    }

    // ===== Track<Vec3> Sampling Tests =====

    #[test]
    fn test_track_vec3_sample_step() {
        let track = Track::from_keyframes(vec![
            Keyframe::step(0.0, Vec3::new(1.0, 0.0, 0.0)),
            Keyframe::step(1.0, Vec3::new(2.0, 0.0, 0.0)),
        ]);

        // Step interpolation should return previous value
        let v = track.sample(0.5).unwrap();
        assert!(v.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_track_vec3_sample_linear() {
        let track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(10.0, 0.0, 0.0)),
        ]);

        let v = track.sample(0.5).unwrap();
        assert!(v.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-5));

        let v = track.sample(0.0).unwrap();
        assert!(v.abs_diff_eq(Vec3::ZERO, 1e-5));

        let v = track.sample(1.0).unwrap();
        assert!(v.abs_diff_eq(Vec3::new(10.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_track_vec3_sample_cubic() {
        let track = Track::from_keyframes(vec![
            Keyframe::cubic(0.0, Vec3::ZERO, Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)),
            Keyframe::cubic(1.0, Vec3::new(10.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO),
        ]);

        // Cubic should give smooth interpolation
        let v = track.sample(0.5).unwrap();
        // With tangents, the curve should bulge slightly
        assert!(v.x > 4.0 && v.x < 6.0);
    }

    #[test]
    fn test_track_vec3_sample_empty() {
        let track: Track<Vec3> = Track::new();
        assert!(track.sample(0.5).is_none());
    }

    // ===== Track<Quat> Sampling Tests =====

    #[test]
    fn test_track_quat_sample_step() {
        let track = Track::from_keyframes(vec![
            Keyframe::step(0.0, Quat::IDENTITY),
            Keyframe::step(1.0, Quat::from_rotation_y(PI)),
        ]);

        let q = track.sample(0.5).unwrap();
        assert!(q.abs_diff_eq(Quat::IDENTITY, 1e-5));
    }

    #[test]
    fn test_track_quat_sample_linear() {
        let track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
            Keyframe::linear(1.0, Quat::from_rotation_y(PI / 2.0)),
        ]);

        let q = track.sample(0.5).unwrap();
        let expected = Quat::from_rotation_y(PI / 4.0);
        assert!(q.abs_diff_eq(expected, 1e-4));
    }

    #[test]
    fn test_track_quat_sample_empty() {
        let track: Track<Quat> = Track::new();
        assert!(track.sample(0.5).is_none());
    }

    // ===== Track<f32> Sampling Tests =====

    #[test]
    fn test_track_f32_sample_step() {
        let track = Track::from_keyframes(vec![
            Keyframe::step(0.0, 0.0f32),
            Keyframe::step(1.0, 1.0f32),
        ]);

        assert!((track.sample(0.5).unwrap() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_track_f32_sample_linear() {
        let track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, 0.0f32),
            Keyframe::linear(1.0, 10.0f32),
        ]);

        assert!((track.sample(0.5).unwrap() - 5.0).abs() < 1e-5);
        assert!((track.sample(0.25).unwrap() - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_track_f32_sample_cubic() {
        let track = Track::from_keyframes(vec![
            Keyframe::cubic(0.0, 0.0f32, 0.0, 5.0),
            Keyframe::cubic(1.0, 10.0f32, 5.0, 0.0),
        ]);

        let v = track.sample(0.5).unwrap();
        assert!(v > 4.0 && v < 6.0);
    }

    // ===== BoneTrack Tests =====

    #[test]
    fn test_bone_track_new() {
        let track = BoneTrack::new("hip");
        assert_eq!(track.bone_name, "hip");
        assert!(track.position.is_none());
        assert!(track.rotation.is_none());
        assert!(track.scale.is_none());
        assert!(track.is_empty());
    }

    #[test]
    fn test_bone_track_with_position() {
        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::ONE),
        ]);
        let track = BoneTrack::new("hip").with_position(pos_track);
        assert!(track.position.is_some());
        assert!(!track.is_empty());
    }

    #[test]
    fn test_bone_track_with_rotation() {
        let rot_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
            Keyframe::linear(1.0, Quat::from_rotation_y(PI)),
        ]);
        let track = BoneTrack::new("hip").with_rotation(rot_track);
        assert!(track.rotation.is_some());
    }

    #[test]
    fn test_bone_track_with_scale() {
        let scale_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ONE),
            Keyframe::linear(1.0, Vec3::splat(2.0)),
        ]);
        let track = BoneTrack::new("hip").with_scale(scale_track);
        assert!(track.scale.is_some());
    }

    #[test]
    fn test_bone_track_sample() {
        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(10.0, 0.0, 0.0)),
        ]);
        let rot_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
            Keyframe::linear(1.0, Quat::from_rotation_y(PI / 2.0)),
        ]);

        let track = BoneTrack::new("hip")
            .with_position(pos_track)
            .with_rotation(rot_track);

        let transform = track.sample(0.5);
        assert!(transform.position.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-5));
        assert!(transform.rotation.abs_diff_eq(Quat::from_rotation_y(PI / 4.0), 1e-4));
        assert_eq!(transform.scale, Vec3::ONE); // Default
    }

    #[test]
    fn test_bone_track_sample_empty() {
        let track = BoneTrack::new("hip");
        let transform = track.sample(0.5);
        assert_eq!(transform, Transform::IDENTITY);
    }

    #[test]
    fn test_bone_track_duration() {
        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(2.0, Vec3::ONE),
        ]);
        let rot_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
            Keyframe::linear(3.0, Quat::IDENTITY),
        ]);

        let track = BoneTrack::new("hip")
            .with_position(pos_track)
            .with_rotation(rot_track);

        assert_eq!(track.duration(), 3.0);
    }

    #[test]
    fn test_bone_track_keyframe_count() {
        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::ONE),
        ]);
        let rot_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Quat::IDENTITY),
        ]);

        let track = BoneTrack::new("hip")
            .with_position(pos_track)
            .with_rotation(rot_track);

        assert_eq!(track.keyframe_count(), 3);
    }

    // ===== AnimationEvent Tests =====

    #[test]
    fn test_animation_event_new() {
        let event = AnimationEvent::new("footstep", 0.5);
        assert_eq!(event.name, "footstep");
        assert_eq!(event.time, 0.5);
        assert!(event.string_param.is_none());
        assert!(event.float_param.is_none());
        assert!(event.int_param.is_none());
    }

    #[test]
    fn test_animation_event_with_params() {
        let event = AnimationEvent::new("attack", 1.0)
            .with_string("sword")
            .with_float(50.0)
            .with_int(1);

        assert_eq!(event.string_param, Some("sword".to_string()));
        assert_eq!(event.float_param, Some(50.0));
        assert_eq!(event.int_param, Some(1));
    }

    // ===== EventTrack Tests =====

    #[test]
    fn test_event_track_new() {
        let track = EventTrack::new("notifies");
        assert_eq!(track.name, "notifies");
        assert!(track.is_empty());
    }

    #[test]
    fn test_event_track_add_event() {
        let mut track = EventTrack::new("notifies");
        track.add_event(AnimationEvent::new("step2", 1.0));
        track.add_event(AnimationEvent::new("step1", 0.5));

        // Should be sorted by time
        assert_eq!(track.events[0].name, "step1");
        assert_eq!(track.events[1].name, "step2");
    }

    #[test]
    fn test_event_track_events_in_range() {
        let mut track = EventTrack::new("notifies");
        track.add_event(AnimationEvent::new("a", 0.25));
        track.add_event(AnimationEvent::new("b", 0.5));
        track.add_event(AnimationEvent::new("c", 0.75));
        track.add_event(AnimationEvent::new("d", 1.0));

        let events = track.events_in_range(0.3, 0.8);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].name, "b");
        assert_eq!(events[1].name, "c");
    }

    #[test]
    fn test_event_track_events_in_range_empty() {
        let mut track = EventTrack::new("notifies");
        track.add_event(AnimationEvent::new("a", 0.5));

        let events = track.events_in_range(0.6, 1.0);
        assert!(events.is_empty());
    }

    // ===== CurveTrack Tests =====

    #[test]
    fn test_curve_track_new() {
        let curve = CurveTrack::new("blend.smile");
        assert_eq!(curve.name, "blend.smile");
        assert!(curve.track.is_empty());
    }

    #[test]
    fn test_curve_track_with_keyframes() {
        let curve = CurveTrack::with_keyframes("alpha", vec![
            Keyframe::linear(0.0, 0.0),
            Keyframe::linear(1.0, 1.0),
        ]);

        assert_eq!(curve.track.len(), 2);
        assert!((curve.sample(0.5).unwrap() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_curve_track_duration() {
        let curve = CurveTrack::with_keyframes("alpha", vec![
            Keyframe::linear(0.0, 0.0),
            Keyframe::linear(2.5, 1.0),
        ]);

        assert_eq!(curve.duration(), 2.5);
    }

    // ===== Pose Tests =====

    #[test]
    fn test_pose_new() {
        let pose = Pose::new();
        assert!(pose.is_empty());
        assert_eq!(pose.len(), 0);
    }

    #[test]
    fn test_pose_set_get() {
        let mut pose = Pose::new();
        pose.set("hip", Transform::from_position(Vec3::new(0.0, 1.0, 0.0)));

        let t = pose.get("hip").unwrap();
        assert!(t.position.abs_diff_eq(Vec3::new(0.0, 1.0, 0.0), 1e-5));

        assert!(pose.get("nonexistent").is_none());
    }

    #[test]
    fn test_pose_iter() {
        let mut pose = Pose::new();
        pose.set("a", Transform::IDENTITY);
        pose.set("b", Transform::IDENTITY);

        let count = pose.iter().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_pose_blend() {
        let mut pose1 = Pose::new();
        pose1.set("hip", Transform::from_position(Vec3::ZERO));

        let mut pose2 = Pose::new();
        pose2.set("hip", Transform::from_position(Vec3::new(10.0, 0.0, 0.0)));

        let blended = pose1.blend(&pose2, 0.5);
        let t = blended.get("hip").unwrap();
        assert!(t.position.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_pose_blend_missing_bones() {
        let mut pose1 = Pose::new();
        pose1.set("a", Transform::from_position(Vec3::X));

        let mut pose2 = Pose::new();
        pose2.set("b", Transform::from_position(Vec3::Y));

        let blended = pose1.blend(&pose2, 0.5);

        // Should have both bones
        assert!(blended.get("a").is_some());
        assert!(blended.get("b").is_some());
    }

    // ===== AnimationClip Tests =====

    #[test]
    fn test_animation_clip_new() {
        let clip = AnimationClip::new("walk", 1.0);
        assert_eq!(clip.name, "walk");
        assert_eq!(clip.duration, 1.0);
        assert_eq!(clip.frame_rate, DEFAULT_FRAME_RATE);
        assert_eq!(clip.looping, LoopMode::Once);
    }

    #[test]
    fn test_animation_clip_looping() {
        let clip = AnimationClip::looping("run", 0.8);
        assert_eq!(clip.looping, LoopMode::Loop);
    }

    #[test]
    fn test_animation_clip_add_bone_track() {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.add_bone_track(BoneTrack::new("hip"));
        clip.add_bone_track(BoneTrack::new("spine"));

        assert_eq!(clip.bone_count(), 2);
        assert!(clip.bone_track("hip").is_some());
        assert!(clip.bone_track("spine").is_some());
    }

    #[test]
    fn test_animation_clip_sample() {
        let mut clip = AnimationClip::new("test", 1.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(10.0, 0.0, 0.0)),
        ]);
        clip.add_bone_track(BoneTrack::new("hip").with_position(pos_track));

        let pose = clip.sample(0.5);
        let t = pose.get("hip").unwrap();
        assert!(t.position.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_animation_clip_sample_bone() {
        let mut clip = AnimationClip::new("test", 1.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::ONE),
        ]);
        clip.add_bone_track(BoneTrack::new("hip").with_position(pos_track));

        let t = clip.sample_bone("hip", 0.5).unwrap();
        assert!(t.position.abs_diff_eq(Vec3::splat(0.5), 1e-5));

        assert!(clip.sample_bone("nonexistent", 0.5).is_none());
    }

    #[test]
    fn test_animation_clip_sample_looping() {
        let mut clip = AnimationClip::looping("test", 1.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::new(10.0, 0.0, 0.0)),
        ]);
        clip.add_bone_track(BoneTrack::new("hip").with_position(pos_track));

        // At time 1.5, should wrap to 0.5
        let pose = clip.sample(1.5);
        let t = pose.get("hip").unwrap();
        assert!(t.position.abs_diff_eq(Vec3::new(5.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn test_animation_clip_events_in_range() {
        let mut clip = AnimationClip::new("test", 1.0);

        let mut event_track = EventTrack::new("notifies");
        event_track.add_event(AnimationEvent::new("a", 0.25));
        event_track.add_event(AnimationEvent::new("b", 0.5));
        event_track.add_event(AnimationEvent::new("c", 0.75));
        clip.add_event_track(event_track);

        let events = clip.events_in_range(0.3, 0.8);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_animation_clip_sample_curve() {
        let mut clip = AnimationClip::new("test", 1.0);

        clip.add_curve_track(CurveTrack::with_keyframes("alpha", vec![
            Keyframe::linear(0.0, 0.0),
            Keyframe::linear(1.0, 1.0),
        ]));

        let v = clip.sample_curve("alpha", 0.5).unwrap();
        assert!((v - 0.5).abs() < 1e-5);

        assert!(clip.sample_curve("nonexistent", 0.5).is_none());
    }

    #[test]
    fn test_animation_clip_total_keyframe_count() {
        let mut clip = AnimationClip::new("test", 1.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::ONE),
        ]);
        clip.add_bone_track(BoneTrack::new("hip").with_position(pos_track));

        clip.add_curve_track(CurveTrack::with_keyframes("alpha", vec![
            Keyframe::linear(0.0, 0.0),
        ]));

        assert_eq!(clip.total_keyframe_count(), 3);
    }

    #[test]
    fn test_animation_clip_total_event_count() {
        let mut clip = AnimationClip::new("test", 1.0);

        let mut track1 = EventTrack::new("a");
        track1.add_event(AnimationEvent::new("e1", 0.0));
        track1.add_event(AnimationEvent::new("e2", 0.5));

        let mut track2 = EventTrack::new("b");
        track2.add_event(AnimationEvent::new("e3", 0.25));

        clip.add_event_track(track1);
        clip.add_event_track(track2);

        assert_eq!(clip.total_event_count(), 3);
    }

    #[test]
    fn test_animation_clip_validate() {
        let clip = AnimationClip::new("test", 1.0);
        assert!(clip.validate().is_ok());
    }

    #[test]
    fn test_animation_clip_compute_duration() {
        let mut clip = AnimationClip::new("test", 0.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(2.0, Vec3::ONE),
        ]);
        clip.add_bone_track(BoneTrack::new("hip").with_position(pos_track));

        clip.add_curve_track(CurveTrack::with_keyframes("alpha", vec![
            Keyframe::linear(0.0, 0.0),
            Keyframe::linear(3.0, 1.0),
        ]));

        let mut event_track = EventTrack::new("notifies");
        event_track.add_event(AnimationEvent::new("end", 2.5));
        clip.add_event_track(event_track);

        assert_eq!(clip.compute_duration(), 3.0);
    }

    #[test]
    fn test_animation_clip_time_frame_conversion() {
        let clip = AnimationClip::new("test", 1.0).with_frame_rate(30.0);

        assert_eq!(clip.time_to_frame(0.5), 15);
        assert!((clip.frame_to_time(15) - 0.5).abs() < 1e-5);
        assert_eq!(clip.frame_count(), 30);
    }

    #[test]
    fn test_animation_clip_is_complete() {
        let clip = AnimationClip::new("test", 1.0);
        assert!(!clip.is_complete(0.5));
        assert!(clip.is_complete(1.0));
        assert!(clip.is_complete(1.5));

        let looping_clip = AnimationClip::looping("test", 1.0);
        assert!(!looping_clip.is_complete(100.0));
    }

    #[test]
    fn test_animation_clip_json_roundtrip() {
        let mut original = AnimationClip::new("test", 1.0);

        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::ONE),
        ]);
        original.add_bone_track(BoneTrack::new("hip").with_position(pos_track));

        let json = original.to_json().unwrap();
        let recovered = AnimationClip::from_json(&json).unwrap();

        assert_eq!(recovered.name, original.name);
        assert_eq!(recovered.duration, original.duration);
        assert_eq!(recovered.bone_count(), original.bone_count());
        assert!(recovered.bone_track("hip").is_some());
    }

    // ===== AnimationClipBuilder Tests =====

    #[test]
    fn test_animation_clip_builder() {
        let pos_track = Track::from_keyframes(vec![
            Keyframe::linear(0.0, Vec3::ZERO),
            Keyframe::linear(1.0, Vec3::ONE),
        ]);

        let clip = AnimationClipBuilder::new("walk", 1.0)
            .loop_mode(LoopMode::Loop)
            .frame_rate(60.0)
            .bone_track(BoneTrack::new("hip").with_position(pos_track))
            .build()
            .unwrap();

        assert_eq!(clip.name, "walk");
        assert_eq!(clip.looping, LoopMode::Loop);
        assert_eq!(clip.frame_rate, 60.0);
        assert_eq!(clip.bone_count(), 1);
    }

    #[test]
    fn test_animation_clip_builder_unchecked() {
        let clip = AnimationClipBuilder::new("test", 1.0)
            .build_unchecked();

        assert_eq!(clip.name, "test");
    }

    // ===== Error Tests =====

    #[test]
    fn test_animation_clip_error_display() {
        let err = AnimationClipError::BoneTrackNotFound {
            bone_name: "hip".to_string(),
        };
        assert!(format!("{}", err).contains("hip"));

        let err = AnimationClipError::TooManyBoneTracks {
            count: 300,
            max: 256,
        };
        assert!(format!("{}", err).contains("300"));
        assert!(format!("{}", err).contains("256"));

        let err = AnimationClipError::EmptyClip;
        assert!(format!("{}", err).contains("no data"));
    }

    // ===== Hermite Interpolation Tests =====

    #[test]
    fn test_hermite_vec3_endpoints() {
        let p0 = Vec3::ZERO;
        let p1 = Vec3::ONE;
        let m0 = Vec3::ZERO;
        let m1 = Vec3::ZERO;

        // At t=0, should be p0
        let v = hermite_vec3(p0, p1, m0, m1, 0.0);
        assert!(v.abs_diff_eq(p0, 1e-5));

        // At t=1, should be p1
        let v = hermite_vec3(p0, p1, m0, m1, 1.0);
        assert!(v.abs_diff_eq(p1, 1e-5));
    }

    #[test]
    fn test_hermite_f32_endpoints() {
        // At t=0, should be p0
        assert!((hermite_f32(0.0, 1.0, 0.0, 0.0, 0.0) - 0.0).abs() < 1e-5);

        // At t=1, should be p1
        assert!((hermite_f32(0.0, 1.0, 0.0, 0.0, 1.0) - 1.0).abs() < 1e-5);
    }

    // ===== Multiple Bone Track Tests =====

    #[test]
    fn test_clip_multiple_bones() {
        let mut clip = AnimationClip::new("test", 1.0);

        for i in 0..5 {
            let pos_track = Track::from_keyframes(vec![
                Keyframe::linear(0.0, Vec3::splat(i as f32)),
                Keyframe::linear(1.0, Vec3::splat((i + 1) as f32)),
            ]);
            clip.add_bone_track(BoneTrack::new(format!("bone_{}", i)).with_position(pos_track));
        }

        assert_eq!(clip.bone_count(), 5);

        let pose = clip.sample(0.5);
        assert_eq!(pose.len(), 5);

        for i in 0..5 {
            let t = pose.get(&format!("bone_{}", i)).unwrap();
            let expected = i as f32 + 0.5;
            assert!(t.position.abs_diff_eq(Vec3::splat(expected), 1e-5));
        }
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_track_single_keyframe() {
        let track = Track::from_keyframes(vec![Keyframe::linear(0.5, Vec3::ONE)]);

        // Before
        let v = track.sample(0.0).unwrap();
        assert!(v.abs_diff_eq(Vec3::ONE, 1e-5));

        // At
        let v = track.sample(0.5).unwrap();
        assert!(v.abs_diff_eq(Vec3::ONE, 1e-5));

        // After
        let v = track.sample(1.0).unwrap();
        assert!(v.abs_diff_eq(Vec3::ONE, 1e-5));
    }

    #[test]
    fn test_clip_no_tracks_sample() {
        let clip = AnimationClip::new("empty", 1.0);
        let pose = clip.sample(0.5);
        assert!(pose.is_empty());
    }

    #[test]
    fn test_pingpong_multiple_cycles() {
        let mode = LoopMode::PingPong;
        let duration = 1.0;

        // Cycle 0: forward 0->1
        let (t, rev) = mode.calculate_time(0.5, duration);
        assert!((t - 0.5).abs() < 1e-5);
        assert!(!rev);

        // Cycle 0: backward 1->0
        let (t, rev) = mode.calculate_time(1.5, duration);
        assert!((t - 0.5).abs() < 1e-5);
        assert!(rev);

        // Cycle 1: forward 0->1
        let (t, rev) = mode.calculate_time(2.5, duration);
        assert!((t - 0.5).abs() < 1e-5);
        assert!(!rev);
    }

    #[test]
    fn test_bone_track_mut() {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.add_bone_track(BoneTrack::new("hip"));

        if let Some(track) = clip.bone_track_mut("hip") {
            track.position = Some(Track::from_keyframes(vec![
                Keyframe::linear(0.0, Vec3::ONE),
            ]));
        }

        let t = clip.sample_bone("hip", 0.0).unwrap();
        assert!(t.position.abs_diff_eq(Vec3::ONE, 1e-5));
    }

    #[test]
    fn test_rebuild_indices() {
        let mut clip = AnimationClip::new("test", 1.0);
        clip.bone_tracks.push(BoneTrack::new("hip"));
        clip.bone_tracks.push(BoneTrack::new("spine"));
        // Index map is empty

        clip.rebuild_indices();

        assert!(clip.bone_track("hip").is_some());
        assert!(clip.bone_track("spine").is_some());
    }
}
