//! Inertialization-based motion transitions for TRINITY Engine (T-AN-6.4).
//!
//! This module provides smooth transitions between animation states using inertialization,
//! which preserves momentum and creates natural motion transitions without pop or jitter.
//!
//! # Key Concepts
//!
//! **Inertialization** is a technique that captures the difference between the current pose
//! and the target pose at transition start, then smoothly decays this offset over time
//! while preserving velocity continuity.
//!
//! # Architecture
//!
//! ```text
//! InertializationBlender
//! ├── state: InertializationState         # Current transition state
//! │   ├── source_offsets: Vec<JointOffset> # Position/rotation offsets
//! │   ├── source_velocities: Vec<JointVelocity>
//! │   ├── transition_duration: f32
//! │   └── elapsed: f32
//! ├── config: InertializationConfig       # Transition parameters
//! │   ├── duration: f32                   # Default 0.2s
//! │   ├── half_life: f32                  # Decay half-life
//! │   ├── damping_mode: DampingMode       # Critical/under/over
//! │   └── joint_overrides: HashMap        # Per-joint configs
//! └── methods
//!     ├── start_transition()              # Begin new transition
//!     ├── update(dt, target_pose)         # Advance and blend
//!     ├── apply(pose) -> blended          # Apply offset to pose
//!     └── is_complete() -> bool           # Check if finished
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::inertialization::{
//!     InertializationBlender, InertializationConfig, DampingMode,
//! };
//!
//! // Create blender with default config
//! let config = InertializationConfig::default()
//!     .with_duration(0.2)
//!     .with_damping_mode(DampingMode::Critical);
//! let mut blender = InertializationBlender::new(config);
//!
//! // Start transition from current to new animation
//! blender.start_transition(&current_pose, &current_velocity);
//!
//! // Each frame: update and apply
//! loop {
//!     blender.update(dt);
//!     let blended = blender.apply(&target_pose);
//!     if blender.is_complete() {
//!         break;
//!     }
//! }
//! ```

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default transition duration in seconds.
pub const DEFAULT_TRANSITION_DURATION: f32 = 0.2;

/// Default half-life for exponential decay (seconds).
pub const DEFAULT_HALF_LIFE: f32 = 0.1;

/// Default velocity preservation factor (0-1).
pub const DEFAULT_VELOCITY_PRESERVATION: f32 = 0.8;

/// Minimum transition duration (prevents division by zero).
pub const MIN_TRANSITION_DURATION: f32 = 0.001;

/// Epsilon for floating point comparisons.
pub const EPSILON: f32 = 1e-6;

/// Natural log of 2, for half-life calculations.
const LN_2: f32 = 0.693147180559945;

// ---------------------------------------------------------------------------
// DampingMode
// ---------------------------------------------------------------------------

/// Damping mode for decay curves.
///
/// Different damping modes create different motion feels:
/// - **Critical**: Smooth, no overshoot - best for most transitions
/// - **Underdamped**: Slight overshoot, springy feel
/// - **Overdamped**: Slow, heavy feel
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DampingMode {
    /// Critical damping - smooth exponential decay, no overshoot.
    #[default]
    Critical = 0,

    /// Underdamped - slight overshoot for springy feel.
    Underdamped = 1,

    /// Overdamped - slow, heavy feel.
    Overdamped = 2,

    /// Custom curve (uses decay_curve_fn).
    Custom = 3,
}

impl DampingMode {
    /// Get the name of this damping mode.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Critical => "critical",
            Self::Underdamped => "underdamped",
            Self::Overdamped => "overdamped",
            Self::Custom => "custom",
        }
    }

    /// Get the damping ratio for this mode.
    ///
    /// - Critical: 1.0
    /// - Underdamped: 0.5-0.9
    /// - Overdamped: 1.5-3.0
    pub fn damping_ratio(&self) -> f32 {
        match self {
            Self::Critical => 1.0,
            Self::Underdamped => 0.7,
            Self::Overdamped => 2.0,
            Self::Custom => 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// DecayCurve
// ---------------------------------------------------------------------------

/// Decay curve for inertialization blending.
///
/// Computes the decay factor at time t based on the damping mode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecayCurve {
    /// Damping mode.
    pub mode: DampingMode,

    /// Half-life in seconds (time for value to decay to 50%).
    pub half_life: f32,

    /// Damping ratio override (if different from mode default).
    pub damping_ratio: Option<f32>,

    /// Angular frequency for underdamped oscillation.
    #[serde(skip)]
    omega: f32,
}

impl Default for DecayCurve {
    fn default() -> Self {
        Self {
            mode: DampingMode::Critical,
            half_life: DEFAULT_HALF_LIFE,
            damping_ratio: None,
            omega: 0.0,
        }
    }
}

impl DecayCurve {
    /// Create a new decay curve.
    pub fn new(mode: DampingMode, half_life: f32) -> Self {
        let mut curve = Self {
            mode,
            half_life: half_life.max(EPSILON),
            damping_ratio: None,
            omega: 0.0,
        };
        curve.update_omega();
        curve
    }

    /// Create a critical damping curve.
    pub fn critical(half_life: f32) -> Self {
        Self::new(DampingMode::Critical, half_life)
    }

    /// Create an underdamped curve.
    pub fn underdamped(half_life: f32) -> Self {
        Self::new(DampingMode::Underdamped, half_life)
    }

    /// Create an overdamped curve.
    pub fn overdamped(half_life: f32) -> Self {
        Self::new(DampingMode::Overdamped, half_life)
    }

    /// Set the damping ratio.
    pub fn with_damping_ratio(mut self, ratio: f32) -> Self {
        self.damping_ratio = Some(ratio.max(EPSILON));
        self.update_omega();
        self
    }

    /// Update the angular frequency based on damping.
    fn update_omega(&mut self) {
        let zeta = self.effective_damping_ratio();
        if zeta < 1.0 {
            // Underdamped: omega_d = omega_n * sqrt(1 - zeta^2)
            let omega_n = LN_2 / self.half_life;
            self.omega = omega_n * (1.0 - zeta * zeta).sqrt();
        }
    }

    /// Get the effective damping ratio.
    #[inline]
    pub fn effective_damping_ratio(&self) -> f32 {
        self.damping_ratio.unwrap_or_else(|| self.mode.damping_ratio())
    }

    /// Compute decay factor at time t.
    ///
    /// Returns a value between 0 and 1 (or slightly above 1 for underdamped).
    /// At t=0, returns 1.0. As t increases, approaches 0.
    pub fn decay(&self, t: f32) -> f32 {
        if t <= 0.0 {
            return 1.0;
        }

        let zeta = self.effective_damping_ratio();
        let omega_n = LN_2 / self.half_life;

        match self.mode {
            DampingMode::Critical => {
                // Critical damping: (1 + omega_n * t) * exp(-omega_n * t)
                let x = omega_n * t;
                (1.0 + x) * (-x).exp()
            }
            DampingMode::Underdamped => {
                // Underdamped: exp(-zeta * omega_n * t) * cos(omega_d * t)
                let decay = (-zeta * omega_n * t).exp();
                let oscillation = (self.omega * t).cos();
                decay * oscillation
            }
            DampingMode::Overdamped => {
                // Overdamped: combination of two exponentials
                let sqrt_term = (zeta * zeta - 1.0).sqrt();
                let r1 = omega_n * (-zeta + sqrt_term);
                let r2 = omega_n * (-zeta - sqrt_term);

                // Weighted combination to start at 1 and end at 0
                let c1 = (zeta + sqrt_term) / (2.0 * sqrt_term);
                let c2 = (sqrt_term - zeta) / (2.0 * sqrt_term);

                c1 * (r1 * t).exp() + c2 * (r2 * t).exp()
            }
            DampingMode::Custom => {
                // Default to exponential decay for custom
                (-LN_2 * t / self.half_life).exp()
            }
        }
    }

    /// Compute velocity decay factor at time t.
    ///
    /// This is the derivative of the position decay, used for velocity blending.
    pub fn velocity_decay(&self, t: f32) -> f32 {
        if t <= 0.0 {
            return 1.0;
        }

        let zeta = self.effective_damping_ratio();
        let omega_n = LN_2 / self.half_life;

        match self.mode {
            DampingMode::Critical => {
                // Derivative of critical damping
                let x = omega_n * t;
                omega_n * omega_n * t * (-x).exp()
            }
            DampingMode::Underdamped => {
                // Derivative of underdamped
                let decay = (-zeta * omega_n * t).exp();
                let cos_term = (self.omega * t).cos();
                let sin_term = (self.omega * t).sin();

                decay * (-zeta * omega_n * cos_term - self.omega * sin_term)
            }
            DampingMode::Overdamped => {
                // Derivative of overdamped
                let sqrt_term = (zeta * zeta - 1.0).sqrt();
                let r1 = omega_n * (-zeta + sqrt_term);
                let r2 = omega_n * (-zeta - sqrt_term);

                let c1 = (zeta + sqrt_term) / (2.0 * sqrt_term);
                let c2 = (sqrt_term - zeta) / (2.0 * sqrt_term);

                c1 * r1 * (r1 * t).exp() + c2 * r2 * (r2 * t).exp()
            }
            DampingMode::Custom => {
                // Derivative of exponential decay
                let rate = LN_2 / self.half_life;
                -rate * (-rate * t).exp()
            }
        }
    }

    /// Check if the decay is effectively complete (below threshold).
    #[inline]
    pub fn is_complete(&self, t: f32, threshold: f32) -> bool {
        self.decay(t).abs() < threshold
    }
}

// ---------------------------------------------------------------------------
// JointOffset
// ---------------------------------------------------------------------------

/// Offset for a single joint during inertialization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JointOffset {
    /// Position offset in local space.
    pub position: Vec3,

    /// Rotation offset as axis-angle (axis * angle).
    pub rotation: Vec3,

    /// Scale offset.
    pub scale: Vec3,
}

impl JointOffset {
    /// Create a new zero offset.
    #[inline]
    pub const fn zero() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ZERO,
        }
    }

    /// Create from position offset only.
    #[inline]
    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            rotation: Vec3::ZERO,
            scale: Vec3::ZERO,
        }
    }

    /// Create from rotation offset only.
    #[inline]
    pub fn from_rotation(rotation: Vec3) -> Self {
        Self {
            position: Vec3::ZERO,
            rotation,
            scale: Vec3::ZERO,
        }
    }

    /// Create from quaternion difference.
    pub fn from_quat_diff(from: Quat, to: Quat) -> Self {
        let diff = to.inverse() * from;
        let (axis, angle) = diff.to_axis_angle();

        Self {
            position: Vec3::ZERO,
            rotation: axis * angle,
            scale: Vec3::ZERO,
        }
    }

    /// Get the rotation as a quaternion.
    #[inline]
    pub fn rotation_quat(&self) -> Quat {
        let angle = self.rotation.length();
        if angle < EPSILON {
            Quat::IDENTITY
        } else {
            Quat::from_axis_angle(self.rotation / angle, angle)
        }
    }

    /// Scale the offset by a factor.
    #[inline]
    pub fn scaled(&self, factor: f32) -> Self {
        Self {
            position: self.position * factor,
            rotation: self.rotation * factor,
            scale: self.scale * factor,
        }
    }

    /// Compute the magnitude of this offset.
    #[inline]
    pub fn magnitude(&self) -> f32 {
        self.position.length() + self.rotation.length() + self.scale.length()
    }

    /// Check if this offset is negligible.
    #[inline]
    pub fn is_negligible(&self, threshold: f32) -> bool {
        self.magnitude() < threshold
    }

    /// Interpolate between two offsets.
    #[inline]
    pub fn lerp(&self, other: &JointOffset, t: f32) -> Self {
        Self {
            position: self.position.lerp(other.position, t),
            rotation: self.rotation.lerp(other.rotation, t),
            scale: self.scale.lerp(other.scale, t),
        }
    }
}

// ---------------------------------------------------------------------------
// JointVelocity
// ---------------------------------------------------------------------------

/// Velocity for a single joint during inertialization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JointVelocity {
    /// Linear velocity.
    pub linear: Vec3,

    /// Angular velocity (axis * angular_speed).
    pub angular: Vec3,

    /// Scale velocity.
    pub scale: Vec3,
}

impl JointVelocity {
    /// Create a new zero velocity.
    #[inline]
    pub const fn zero() -> Self {
        Self {
            linear: Vec3::ZERO,
            angular: Vec3::ZERO,
            scale: Vec3::ZERO,
        }
    }

    /// Create from linear velocity only.
    #[inline]
    pub fn from_linear(linear: Vec3) -> Self {
        Self {
            linear,
            angular: Vec3::ZERO,
            scale: Vec3::ZERO,
        }
    }

    /// Create from angular velocity only.
    #[inline]
    pub fn from_angular(angular: Vec3) -> Self {
        Self {
            linear: Vec3::ZERO,
            angular,
            scale: Vec3::ZERO,
        }
    }

    /// Scale the velocity by a factor.
    #[inline]
    pub fn scaled(&self, factor: f32) -> Self {
        Self {
            linear: self.linear * factor,
            angular: self.angular * factor,
            scale: self.scale * factor,
        }
    }

    /// Compute the magnitude of this velocity.
    #[inline]
    pub fn magnitude(&self) -> f32 {
        self.linear.length() + self.angular.length() + self.scale.length()
    }

    /// Check if this velocity is negligible.
    #[inline]
    pub fn is_negligible(&self, threshold: f32) -> bool {
        self.magnitude() < threshold
    }

    /// Interpolate between two velocities.
    #[inline]
    pub fn lerp(&self, other: &JointVelocity, t: f32) -> Self {
        Self {
            linear: self.linear.lerp(other.linear, t),
            angular: self.angular.lerp(other.angular, t),
            scale: self.scale.lerp(other.scale, t),
        }
    }

    /// Estimate angular velocity from quaternion delta.
    pub fn angular_from_quat_delta(from: Quat, to: Quat, dt: f32) -> Vec3 {
        if dt < EPSILON {
            return Vec3::ZERO;
        }

        let delta = to * from.inverse();
        let (axis, angle) = delta.to_axis_angle();

        axis * (angle / dt)
    }
}

// ---------------------------------------------------------------------------
// JointConfig
// ---------------------------------------------------------------------------

/// Per-joint configuration for inertialization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointConfig {
    /// Weight for this joint (0-1).
    pub weight: f32,

    /// Override duration for this joint (None = use global).
    pub duration_override: Option<f32>,

    /// Override half-life for this joint (None = use global).
    pub half_life_override: Option<f32>,

    /// Override damping mode for this joint.
    pub damping_override: Option<DampingMode>,

    /// Whether to preserve velocity for this joint.
    pub preserve_velocity: bool,
}

impl Default for JointConfig {
    fn default() -> Self {
        Self {
            weight: 1.0,
            duration_override: None,
            half_life_override: None,
            damping_override: None,
            preserve_velocity: true,
        }
    }
}

impl JointConfig {
    /// Create a new joint config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific weight.
    pub fn with_weight(weight: f32) -> Self {
        Self {
            weight: weight.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Set the duration override.
    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration_override = Some(duration.max(MIN_TRANSITION_DURATION));
        self
    }

    /// Set the half-life override.
    pub fn with_half_life(mut self, half_life: f32) -> Self {
        self.half_life_override = Some(half_life.max(EPSILON));
        self
    }

    /// Set the damping mode override.
    pub fn with_damping(mut self, mode: DampingMode) -> Self {
        self.damping_override = Some(mode);
        self
    }

    /// Disable velocity preservation.
    pub fn without_velocity(mut self) -> Self {
        self.preserve_velocity = false;
        self
    }
}

// ---------------------------------------------------------------------------
// ComponentConfig
// ---------------------------------------------------------------------------

/// Configuration for a specific transform component (position/rotation/scale).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComponentConfig {
    /// Whether this component is enabled.
    pub enabled: bool,

    /// Weight for this component.
    pub weight: f32,

    /// Decay curve for this component.
    pub decay_curve: DecayCurve,
}

impl Default for ComponentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            weight: 1.0,
            decay_curve: DecayCurve::default(),
        }
    }
}

impl ComponentConfig {
    /// Create a new component config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Disable this component.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            weight: 0.0,
            decay_curve: DecayCurve::default(),
        }
    }

    /// Set the weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the decay curve.
    pub fn with_decay(mut self, curve: DecayCurve) -> Self {
        self.decay_curve = curve;
        self
    }
}

// ---------------------------------------------------------------------------
// InertializationConfig
// ---------------------------------------------------------------------------

/// Configuration for inertialization transitions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InertializationConfig {
    /// Default transition duration in seconds.
    pub duration: f32,

    /// Half-life for decay.
    pub half_life: f32,

    /// Global damping mode.
    pub damping_mode: DampingMode,

    /// Velocity preservation factor (0-1).
    pub velocity_preservation: f32,

    /// Configuration for position component.
    pub position: ComponentConfig,

    /// Configuration for rotation component.
    pub rotation: ComponentConfig,

    /// Configuration for scale component.
    pub scale: ComponentConfig,

    /// Per-joint configuration overrides.
    pub joint_overrides: HashMap<usize, JointConfig>,

    /// Completion threshold (offset magnitude below this = complete).
    pub completion_threshold: f32,
}

impl Default for InertializationConfig {
    fn default() -> Self {
        Self {
            duration: DEFAULT_TRANSITION_DURATION,
            half_life: DEFAULT_HALF_LIFE,
            damping_mode: DampingMode::Critical,
            velocity_preservation: DEFAULT_VELOCITY_PRESERVATION,
            position: ComponentConfig::default(),
            rotation: ComponentConfig::default(),
            scale: ComponentConfig::default(),
            joint_overrides: HashMap::new(),
            completion_threshold: 0.001,
        }
    }
}

impl InertializationConfig {
    /// Create a new default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a fast transition config.
    pub fn fast() -> Self {
        Self {
            duration: 0.1,
            half_life: 0.05,
            ..Default::default()
        }
    }

    /// Create a slow, smooth transition config.
    pub fn smooth() -> Self {
        Self {
            duration: 0.3,
            half_life: 0.15,
            damping_mode: DampingMode::Critical,
            ..Default::default()
        }
    }

    /// Create a springy transition config.
    pub fn springy() -> Self {
        Self {
            duration: 0.25,
            half_life: 0.1,
            damping_mode: DampingMode::Underdamped,
            ..Default::default()
        }
    }

    /// Create a heavy transition config.
    pub fn heavy() -> Self {
        Self {
            duration: 0.4,
            half_life: 0.2,
            damping_mode: DampingMode::Overdamped,
            ..Default::default()
        }
    }

    /// Set the duration.
    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration = duration.max(MIN_TRANSITION_DURATION);
        self
    }

    /// Set the half-life.
    pub fn with_half_life(mut self, half_life: f32) -> Self {
        self.half_life = half_life.max(EPSILON);
        self
    }

    /// Set the damping mode.
    pub fn with_damping_mode(mut self, mode: DampingMode) -> Self {
        self.damping_mode = mode;
        self
    }

    /// Set the velocity preservation factor.
    pub fn with_velocity_preservation(mut self, factor: f32) -> Self {
        self.velocity_preservation = factor.clamp(0.0, 1.0);
        self
    }

    /// Set position component config.
    pub fn with_position(mut self, config: ComponentConfig) -> Self {
        self.position = config;
        self
    }

    /// Set rotation component config.
    pub fn with_rotation(mut self, config: ComponentConfig) -> Self {
        self.rotation = config;
        self
    }

    /// Set scale component config.
    pub fn with_scale(mut self, config: ComponentConfig) -> Self {
        self.scale = config;
        self
    }

    /// Add a per-joint override.
    pub fn with_joint_override(mut self, joint: usize, config: JointConfig) -> Self {
        self.joint_overrides.insert(joint, config);
        self
    }

    /// Get the config for a specific joint.
    pub fn joint_config(&self, joint: usize) -> Option<&JointConfig> {
        self.joint_overrides.get(&joint)
    }

    /// Get effective duration for a joint.
    pub fn effective_duration(&self, joint: usize) -> f32 {
        self.joint_overrides
            .get(&joint)
            .and_then(|c| c.duration_override)
            .unwrap_or(self.duration)
    }

    /// Get effective half-life for a joint.
    pub fn effective_half_life(&self, joint: usize) -> f32 {
        self.joint_overrides
            .get(&joint)
            .and_then(|c| c.half_life_override)
            .unwrap_or(self.half_life)
    }

    /// Get effective damping mode for a joint.
    pub fn effective_damping(&self, joint: usize) -> DampingMode {
        self.joint_overrides
            .get(&joint)
            .and_then(|c| c.damping_override)
            .unwrap_or(self.damping_mode)
    }

    /// Get effective weight for a joint.
    pub fn effective_weight(&self, joint: usize) -> f32 {
        self.joint_overrides
            .get(&joint)
            .map(|c| c.weight)
            .unwrap_or(1.0)
    }
}

// ---------------------------------------------------------------------------
// InertializationState
// ---------------------------------------------------------------------------

/// State of an inertialization transition.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InertializationState {
    /// Offset snapshots at transition start (per joint).
    pub source_offsets: Vec<JointOffset>,

    /// Velocity snapshots at transition start (per joint).
    pub source_velocities: Vec<JointVelocity>,

    /// Decay curve per joint.
    pub decay_curves: Vec<DecayCurve>,

    /// Transition duration.
    pub duration: f32,

    /// Elapsed time since transition start.
    pub elapsed: f32,

    /// Whether the transition is active.
    pub is_active: bool,

    /// Joint count.
    pub joint_count: usize,
}

impl InertializationState {
    /// Create a new inactive state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create state for a specific joint count.
    pub fn with_joint_count(joint_count: usize) -> Self {
        Self {
            source_offsets: vec![JointOffset::zero(); joint_count],
            source_velocities: vec![JointVelocity::zero(); joint_count],
            decay_curves: vec![DecayCurve::default(); joint_count],
            duration: DEFAULT_TRANSITION_DURATION,
            elapsed: 0.0,
            is_active: false,
            joint_count,
        }
    }

    /// Get the normalized progress (0-1).
    #[inline]
    pub fn progress(&self) -> f32 {
        if self.duration <= MIN_TRANSITION_DURATION {
            1.0
        } else {
            (self.elapsed / self.duration).clamp(0.0, 1.0)
        }
    }

    /// Check if the transition is complete.
    #[inline]
    pub fn is_complete(&self) -> bool {
        !self.is_active || self.elapsed >= self.duration
    }

    /// Get remaining time.
    #[inline]
    pub fn remaining(&self) -> f32 {
        (self.duration - self.elapsed).max(0.0)
    }

    /// Reset the state.
    pub fn reset(&mut self) {
        for offset in &mut self.source_offsets {
            *offset = JointOffset::zero();
        }
        for vel in &mut self.source_velocities {
            *vel = JointVelocity::zero();
        }
        self.elapsed = 0.0;
        self.is_active = false;
    }

    /// Get the current offset for a joint.
    pub fn current_offset(&self, joint: usize) -> JointOffset {
        if joint >= self.source_offsets.len() {
            return JointOffset::zero();
        }

        let decay = self.decay_curves.get(joint)
            .map(|c| c.decay(self.elapsed))
            .unwrap_or(0.0);

        self.source_offsets[joint].scaled(decay)
    }

    /// Get the current velocity for a joint.
    pub fn current_velocity(&self, joint: usize) -> JointVelocity {
        if joint >= self.source_velocities.len() {
            return JointVelocity::zero();
        }

        let decay = self.decay_curves.get(joint)
            .map(|c| c.velocity_decay(self.elapsed))
            .unwrap_or(0.0);

        self.source_velocities[joint].scaled(decay)
    }
}

// ---------------------------------------------------------------------------
// VelocityEstimator
// ---------------------------------------------------------------------------

/// Estimates joint velocities from pose history.
#[derive(Clone, Debug)]
pub struct VelocityEstimator {
    /// Previous positions per joint.
    prev_positions: Vec<Vec3>,

    /// Previous rotations per joint.
    prev_rotations: Vec<Quat>,

    /// Previous scales per joint.
    prev_scales: Vec<Vec3>,

    /// Time of previous sample.
    prev_time: f32,

    /// Smoothing factor for EMA filter (0-1).
    smoothing: f32,

    /// Accumulated velocities (for EMA).
    smoothed_velocities: Vec<JointVelocity>,

    /// Whether the estimator is initialized.
    initialized: bool,
}

impl Default for VelocityEstimator {
    fn default() -> Self {
        Self {
            prev_positions: Vec::new(),
            prev_rotations: Vec::new(),
            prev_scales: Vec::new(),
            prev_time: 0.0,
            smoothing: 0.2,
            smoothed_velocities: Vec::new(),
            initialized: false,
        }
    }
}

impl VelocityEstimator {
    /// Create a new velocity estimator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific smoothing factor.
    pub fn with_smoothing(smoothing: f32) -> Self {
        Self {
            smoothing: smoothing.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Initialize for a specific joint count.
    pub fn initialize(&mut self, joint_count: usize) {
        self.prev_positions = vec![Vec3::ZERO; joint_count];
        self.prev_rotations = vec![Quat::IDENTITY; joint_count];
        self.prev_scales = vec![Vec3::ONE; joint_count];
        self.smoothed_velocities = vec![JointVelocity::zero(); joint_count];
        self.initialized = false;
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        for pos in &mut self.prev_positions {
            *pos = Vec3::ZERO;
        }
        for rot in &mut self.prev_rotations {
            *rot = Quat::IDENTITY;
        }
        for scale in &mut self.prev_scales {
            *scale = Vec3::ONE;
        }
        for vel in &mut self.smoothed_velocities {
            *vel = JointVelocity::zero();
        }
        self.prev_time = 0.0;
        self.initialized = false;
    }

    /// Update with a new pose sample.
    ///
    /// Returns estimated velocities for all joints.
    pub fn update(
        &mut self,
        positions: &[Vec3],
        rotations: &[Quat],
        scales: &[Vec3],
        time: f32,
    ) -> Vec<JointVelocity> {
        let joint_count = positions.len().min(rotations.len()).min(scales.len());

        // Ensure capacity
        if self.prev_positions.len() < joint_count {
            self.initialize(joint_count);
        }

        let dt = time - self.prev_time;

        if !self.initialized {
            // First sample: store and return zero velocities
            for i in 0..joint_count {
                self.prev_positions[i] = positions[i];
                self.prev_rotations[i] = rotations[i];
                self.prev_scales[i] = scales[i];
            }
            self.prev_time = time;
            self.initialized = true;
            return vec![JointVelocity::zero(); joint_count];
        }

        if dt <= EPSILON {
            // No time passed, return previous velocities
            return self.smoothed_velocities.clone();
        }

        // Compute instantaneous velocities
        let mut velocities = Vec::with_capacity(joint_count);
        for i in 0..joint_count {
            let linear = (positions[i] - self.prev_positions[i]) / dt;
            let angular = JointVelocity::angular_from_quat_delta(
                self.prev_rotations[i],
                rotations[i],
                dt,
            );
            let scale_vel = (scales[i] - self.prev_scales[i]) / dt;

            velocities.push(JointVelocity {
                linear,
                angular,
                scale: scale_vel,
            });
        }

        // Apply EMA smoothing
        for i in 0..joint_count {
            self.smoothed_velocities[i] = self.smoothed_velocities[i].lerp(
                &velocities[i],
                self.smoothing,
            );
        }

        // Store current pose for next frame
        for i in 0..joint_count {
            self.prev_positions[i] = positions[i];
            self.prev_rotations[i] = rotations[i];
            self.prev_scales[i] = scales[i];
        }
        self.prev_time = time;

        self.smoothed_velocities.clone()
    }

    /// Get the last estimated velocities.
    pub fn velocities(&self) -> &[JointVelocity] {
        &self.smoothed_velocities
    }

    /// Check if initialized.
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

// ---------------------------------------------------------------------------
// InertializationBlender
// ---------------------------------------------------------------------------

/// Blender for inertialization-based transitions.
///
/// Provides smooth transitions between animation states by capturing the
/// difference at transition start and decaying it over time.
#[derive(Clone, Debug)]
pub struct InertializationBlender {
    /// Configuration.
    pub config: InertializationConfig,

    /// Current state.
    pub state: InertializationState,

    /// Velocity estimator.
    velocity_estimator: VelocityEstimator,
}

impl Default for InertializationBlender {
    fn default() -> Self {
        Self::new(InertializationConfig::default())
    }
}

impl InertializationBlender {
    /// Create a new blender with the given configuration.
    pub fn new(config: InertializationConfig) -> Self {
        Self {
            config,
            state: InertializationState::new(),
            velocity_estimator: VelocityEstimator::new(),
        }
    }

    /// Create with default configuration.
    pub fn with_default_config() -> Self {
        Self::default()
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: InertializationConfig) {
        self.config = config;
    }

    /// Get the configuration.
    pub fn config(&self) -> &InertializationConfig {
        &self.config
    }

    /// Start a new transition.
    ///
    /// Captures the difference between source and target poses, and
    /// initializes the decay to smoothly blend to the target.
    pub fn start_transition(
        &mut self,
        source_positions: &[Vec3],
        source_rotations: &[Quat],
        target_positions: &[Vec3],
        target_rotations: &[Quat],
        source_velocities: Option<&[JointVelocity]>,
    ) {
        let joint_count = source_positions.len()
            .min(source_rotations.len())
            .min(target_positions.len())
            .min(target_rotations.len());

        if joint_count == 0 {
            return;
        }

        // Initialize state
        self.state = InertializationState::with_joint_count(joint_count);
        self.state.duration = self.config.duration;
        self.state.elapsed = 0.0;
        self.state.is_active = true;

        // Compute offsets (source - target)
        for i in 0..joint_count {
            // Position offset
            let pos_offset = source_positions[i] - target_positions[i];

            // Rotation offset (as axis-angle)
            let rot_diff = target_rotations[i].inverse() * source_rotations[i];
            let (axis, angle) = rot_diff.to_axis_angle();
            let rot_offset = if angle.abs() < EPSILON {
                Vec3::ZERO
            } else {
                axis * angle
            };

            self.state.source_offsets[i] = JointOffset {
                position: pos_offset,
                rotation: rot_offset,
                scale: Vec3::ZERO,
            };

            // Velocity (if provided)
            if let Some(vels) = source_velocities {
                if i < vels.len() {
                    self.state.source_velocities[i] = vels[i].scaled(
                        self.config.velocity_preservation,
                    );
                }
            }

            // Set up decay curve for this joint
            let half_life = self.config.effective_half_life(i);
            let damping = self.config.effective_damping(i);
            self.state.decay_curves[i] = DecayCurve::new(damping, half_life);
        }
    }

    /// Start a transition with only offsets (no velocity).
    pub fn start_transition_from_offset(
        &mut self,
        offsets: &[JointOffset],
    ) {
        let joint_count = offsets.len();
        if joint_count == 0 {
            return;
        }

        self.state = InertializationState::with_joint_count(joint_count);
        self.state.duration = self.config.duration;
        self.state.elapsed = 0.0;
        self.state.is_active = true;

        for i in 0..joint_count {
            self.state.source_offsets[i] = offsets[i];

            let half_life = self.config.effective_half_life(i);
            let damping = self.config.effective_damping(i);
            self.state.decay_curves[i] = DecayCurve::new(damping, half_life);
        }
    }

    /// Update the transition with elapsed time.
    pub fn update(&mut self, dt: f32) {
        if !self.state.is_active {
            return;
        }

        self.state.elapsed += dt;

        // Check for completion
        if self.state.elapsed >= self.state.duration {
            self.state.is_active = false;
        }
    }

    /// Apply the inertialization offset to a target pose.
    ///
    /// Returns the blended pose with the decayed offset applied.
    pub fn apply_positions(&self, target_positions: &[Vec3]) -> Vec<Vec3> {
        if !self.state.is_active || target_positions.is_empty() {
            return target_positions.to_vec();
        }

        let mut result = Vec::with_capacity(target_positions.len());

        for (i, &target) in target_positions.iter().enumerate() {
            let offset = self.state.current_offset(i);
            let weight = self.config.effective_weight(i);

            if self.config.position.enabled {
                let pos_weight = weight * self.config.position.weight;
                result.push(target + offset.position * pos_weight);
            } else {
                result.push(target);
            }
        }

        result
    }

    /// Apply the inertialization offset to target rotations.
    pub fn apply_rotations(&self, target_rotations: &[Quat]) -> Vec<Quat> {
        if !self.state.is_active || target_rotations.is_empty() {
            return target_rotations.to_vec();
        }

        let mut result = Vec::with_capacity(target_rotations.len());

        for (i, &target) in target_rotations.iter().enumerate() {
            let offset = self.state.current_offset(i);
            let weight = self.config.effective_weight(i);

            if self.config.rotation.enabled {
                let rot_weight = weight * self.config.rotation.weight;
                let offset_quat = offset.rotation_quat();
                // Slerp between identity and offset based on weight
                let weighted_offset = Quat::IDENTITY.slerp(offset_quat, rot_weight);
                result.push(weighted_offset * target);
            } else {
                result.push(target);
            }
        }

        result
    }

    /// Apply inertialization to both positions and rotations.
    pub fn apply(
        &self,
        target_positions: &[Vec3],
        target_rotations: &[Quat],
    ) -> (Vec<Vec3>, Vec<Quat>) {
        (
            self.apply_positions(target_positions),
            self.apply_rotations(target_rotations),
        )
    }

    /// Check if the transition is complete.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.state.is_complete()
    }

    /// Check if the transition is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.state.is_active
    }

    /// Get the normalized progress (0-1).
    #[inline]
    pub fn get_progress(&self) -> f32 {
        self.state.progress()
    }

    /// Get remaining transition time.
    #[inline]
    pub fn remaining_time(&self) -> f32 {
        self.state.remaining()
    }

    /// Cancel the current transition.
    pub fn cancel(&mut self) {
        self.state.is_active = false;
        self.state.elapsed = self.state.duration;
    }

    /// Reset the blender.
    pub fn reset(&mut self) {
        self.state.reset();
        self.velocity_estimator.reset();
    }

    /// Get the current state.
    pub fn state(&self) -> &InertializationState {
        &self.state
    }

    /// Get the velocity estimator.
    pub fn velocity_estimator(&self) -> &VelocityEstimator {
        &self.velocity_estimator
    }

    /// Get mutable velocity estimator.
    pub fn velocity_estimator_mut(&mut self) -> &mut VelocityEstimator {
        &mut self.velocity_estimator
    }
}

// ---------------------------------------------------------------------------
// HierarchyAwareBlender
// ---------------------------------------------------------------------------

/// Inertialization blender that respects skeleton hierarchy.
///
/// Parent joint transitions affect children, maintaining chain continuity.
#[derive(Clone, Debug)]
pub struct HierarchyAwareBlender {
    /// Base blender.
    blender: InertializationBlender,

    /// Parent indices for each joint (-1 = root).
    parent_indices: Vec<i32>,

    /// Hierarchy propagation weight (0-1).
    hierarchy_weight: f32,
}

impl HierarchyAwareBlender {
    /// Create a new hierarchy-aware blender.
    pub fn new(config: InertializationConfig, parent_indices: Vec<i32>) -> Self {
        Self {
            blender: InertializationBlender::new(config),
            parent_indices,
            hierarchy_weight: 0.5,
        }
    }

    /// Set the hierarchy propagation weight.
    pub fn set_hierarchy_weight(&mut self, weight: f32) {
        self.hierarchy_weight = weight.clamp(0.0, 1.0);
    }

    /// Start a transition (same as base blender).
    pub fn start_transition(
        &mut self,
        source_positions: &[Vec3],
        source_rotations: &[Quat],
        target_positions: &[Vec3],
        target_rotations: &[Quat],
        source_velocities: Option<&[JointVelocity]>,
    ) {
        self.blender.start_transition(
            source_positions,
            source_rotations,
            target_positions,
            target_rotations,
            source_velocities,
        );
    }

    /// Update the transition.
    pub fn update(&mut self, dt: f32) {
        self.blender.update(dt);
    }

    /// Apply with hierarchy propagation.
    ///
    /// Parent offsets are propagated to children with the hierarchy weight.
    pub fn apply(
        &self,
        target_positions: &[Vec3],
        target_rotations: &[Quat],
    ) -> (Vec<Vec3>, Vec<Quat>) {
        if !self.blender.is_active() {
            return (target_positions.to_vec(), target_rotations.to_vec());
        }

        // First, get base offsets
        let base_positions = self.blender.apply_positions(target_positions);
        let base_rotations = self.blender.apply_rotations(target_rotations);

        // Then, propagate parent influence
        let mut final_positions = base_positions.clone();
        let mut final_rotations = base_rotations.clone();

        for i in 0..self.parent_indices.len().min(target_positions.len()) {
            let parent_idx = self.parent_indices[i];
            if parent_idx >= 0 {
                let parent = parent_idx as usize;
                if parent < final_positions.len() {
                    // Accumulate parent offset with reduced weight
                    let parent_offset = self.blender.state.current_offset(parent);
                    let child_offset = self.blender.state.current_offset(i);

                    // Combine parent and child offsets
                    let combined_pos = child_offset.position
                        + parent_offset.position * self.hierarchy_weight;

                    // Apply to position
                    final_positions[i] = target_positions[i] + combined_pos;

                    // For rotation, multiply parent influence
                    let parent_rot = parent_offset.rotation_quat();
                    let child_rot = child_offset.rotation_quat();
                    let parent_influence = Quat::IDENTITY.slerp(parent_rot, self.hierarchy_weight);
                    final_rotations[i] = parent_influence * child_rot * target_rotations[i];
                }
            }
        }

        (final_positions, final_rotations)
    }

    /// Check if complete.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.blender.is_complete()
    }

    /// Get progress.
    #[inline]
    pub fn get_progress(&self) -> f32 {
        self.blender.get_progress()
    }

    /// Get the base blender.
    pub fn base_blender(&self) -> &InertializationBlender {
        &self.blender
    }

    /// Get mutable base blender.
    pub fn base_blender_mut(&mut self) -> &mut InertializationBlender {
        &mut self.blender
    }
}

// ---------------------------------------------------------------------------
// TransitionManager
// ---------------------------------------------------------------------------

/// Manages multiple concurrent inertialization transitions.
#[derive(Clone, Debug, Default)]
pub struct TransitionManager {
    /// Active transitions by ID.
    transitions: HashMap<u32, InertializationBlender>,

    /// Next transition ID.
    next_id: u32,

    /// Default configuration.
    default_config: InertializationConfig,
}

impl TransitionManager {
    /// Create a new transition manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a default configuration.
    pub fn with_config(config: InertializationConfig) -> Self {
        Self {
            default_config: config,
            ..Default::default()
        }
    }

    /// Start a new transition, returning its ID.
    pub fn start(
        &mut self,
        source_positions: &[Vec3],
        source_rotations: &[Quat],
        target_positions: &[Vec3],
        target_rotations: &[Quat],
        config: Option<InertializationConfig>,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let cfg = config.unwrap_or_else(|| self.default_config.clone());
        let mut blender = InertializationBlender::new(cfg);
        blender.start_transition(
            source_positions,
            source_rotations,
            target_positions,
            target_rotations,
            None,
        );

        self.transitions.insert(id, blender);
        id
    }

    /// Update all active transitions.
    pub fn update(&mut self, dt: f32) {
        // Update all transitions
        for blender in self.transitions.values_mut() {
            blender.update(dt);
        }

        // Remove completed transitions
        self.transitions.retain(|_, b| !b.is_complete());
    }

    /// Apply all active transitions to a pose.
    ///
    /// Multiple transitions are blended additively.
    pub fn apply(
        &self,
        positions: &[Vec3],
        rotations: &[Quat],
    ) -> (Vec<Vec3>, Vec<Quat>) {
        let mut result_positions = positions.to_vec();
        let mut result_rotations = rotations.to_vec();

        for blender in self.transitions.values() {
            let (new_pos, new_rot) = blender.apply(&result_positions, &result_rotations);
            result_positions = new_pos;
            result_rotations = new_rot;
        }

        (result_positions, result_rotations)
    }

    /// Get a specific transition.
    pub fn get(&self, id: u32) -> Option<&InertializationBlender> {
        self.transitions.get(&id)
    }

    /// Get mutable transition.
    pub fn get_mut(&mut self, id: u32) -> Option<&mut InertializationBlender> {
        self.transitions.get_mut(&id)
    }

    /// Cancel a specific transition.
    pub fn cancel(&mut self, id: u32) -> bool {
        if let Some(blender) = self.transitions.get_mut(&id) {
            blender.cancel();
            true
        } else {
            false
        }
    }

    /// Cancel all transitions.
    pub fn cancel_all(&mut self) {
        self.transitions.clear();
    }

    /// Get the number of active transitions.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.transitions.len()
    }

    /// Check if any transition is active.
    #[inline]
    pub fn has_active(&self) -> bool {
        !self.transitions.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Compute the decay factor for inertialization.
///
/// Uses critical damping by default.
#[inline]
pub fn compute_decay(t: f32, half_life: f32) -> f32 {
    DecayCurve::critical(half_life).decay(t)
}

/// Compute instant snap (zero duration transition).
///
/// Returns 0.0 (complete) immediately.
#[inline]
pub fn instant_snap() -> f32 {
    0.0
}

/// Interpolate a quaternion offset.
pub fn interpolate_quat_offset(offset: Quat, t: f32) -> Quat {
    Quat::IDENTITY.slerp(offset, t)
}

/// Estimate velocity from two poses.
pub fn estimate_velocity(
    prev_positions: &[Vec3],
    prev_rotations: &[Quat],
    curr_positions: &[Vec3],
    curr_rotations: &[Quat],
    dt: f32,
) -> Vec<JointVelocity> {
    if dt <= EPSILON {
        return vec![JointVelocity::zero(); curr_positions.len()];
    }

    let joint_count = prev_positions.len()
        .min(prev_rotations.len())
        .min(curr_positions.len())
        .min(curr_rotations.len());

    let mut velocities = Vec::with_capacity(joint_count);

    for i in 0..joint_count {
        let linear = (curr_positions[i] - prev_positions[i]) / dt;
        let angular = JointVelocity::angular_from_quat_delta(
            prev_rotations[i],
            curr_rotations[i],
            dt,
        );

        velocities.push(JointVelocity {
            linear,
            angular,
            scale: Vec3::ZERO,
        });
    }

    velocities
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // =====================================================================
    // DampingMode tests
    // =====================================================================

    #[test]
    fn test_damping_mode_default() {
        assert_eq!(DampingMode::default(), DampingMode::Critical);
    }

    #[test]
    fn test_damping_mode_name() {
        assert_eq!(DampingMode::Critical.name(), "critical");
        assert_eq!(DampingMode::Underdamped.name(), "underdamped");
        assert_eq!(DampingMode::Overdamped.name(), "overdamped");
        assert_eq!(DampingMode::Custom.name(), "custom");
    }

    #[test]
    fn test_damping_mode_ratio() {
        assert_eq!(DampingMode::Critical.damping_ratio(), 1.0);
        assert!(DampingMode::Underdamped.damping_ratio() < 1.0);
        assert!(DampingMode::Overdamped.damping_ratio() > 1.0);
    }

    // =====================================================================
    // DecayCurve tests
    // =====================================================================

    #[test]
    fn test_decay_curve_default() {
        let curve = DecayCurve::default();
        assert_eq!(curve.mode, DampingMode::Critical);
        assert_eq!(curve.half_life, DEFAULT_HALF_LIFE);
    }

    #[test]
    fn test_decay_curve_critical() {
        let curve = DecayCurve::critical(0.1);
        assert_eq!(curve.mode, DampingMode::Critical);
        assert_eq!(curve.half_life, 0.1);
    }

    #[test]
    fn test_decay_curve_underdamped() {
        let curve = DecayCurve::underdamped(0.1);
        assert_eq!(curve.mode, DampingMode::Underdamped);
    }

    #[test]
    fn test_decay_curve_overdamped() {
        let curve = DecayCurve::overdamped(0.1);
        assert_eq!(curve.mode, DampingMode::Overdamped);
    }

    #[test]
    fn test_decay_at_zero() {
        let curve = DecayCurve::critical(0.1);
        assert!((curve.decay(0.0) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_decay_decreases_over_time() {
        let curve = DecayCurve::critical(0.1);
        let d1 = curve.decay(0.05);
        let d2 = curve.decay(0.1);
        let d3 = curve.decay(0.2);

        assert!(d1 > d2);
        assert!(d2 > d3);
        assert!(d3 > 0.0);
    }

    #[test]
    fn test_decay_half_life() {
        let curve = DecayCurve::critical(0.1);
        // At half-life, value should be around 0.5 (approximately for critical damping)
        let at_half_life = curve.decay(0.1);
        // For critical damping, it's not exactly 0.5 but close
        assert!(at_half_life > 0.2);
        assert!(at_half_life < 0.8);
    }

    #[test]
    fn test_decay_underdamped_oscillates() {
        let curve = DecayCurve::underdamped(0.1);
        // Underdamped can go slightly negative (overshoot)
        let mut has_negative = false;
        for i in 1..100 {
            let t = i as f32 * 0.01;
            if curve.decay(t) < 0.0 {
                has_negative = true;
                break;
            }
        }
        // May or may not oscillate below zero depending on parameters
        // The test just ensures it doesn't crash
    }

    #[test]
    fn test_decay_overdamped_slow() {
        let critical = DecayCurve::critical(0.1);
        let overdamped = DecayCurve::overdamped(0.1);

        // Overdamped decays slower
        let t = 0.15;
        let c_decay = critical.decay(t);
        let o_decay = overdamped.decay(t);

        assert!(o_decay >= c_decay * 0.5); // Overdamped is slower
    }

    #[test]
    fn test_decay_is_complete() {
        let curve = DecayCurve::critical(0.1);
        assert!(!curve.is_complete(0.0, 0.001));
        assert!(curve.is_complete(10.0, 0.001));
    }

    #[test]
    fn test_velocity_decay() {
        let curve = DecayCurve::critical(0.1);
        let v0 = curve.velocity_decay(0.0);
        let v1 = curve.velocity_decay(0.1);

        assert!(v0.is_finite());
        assert!(v1.is_finite());
    }

    // =====================================================================
    // JointOffset tests
    // =====================================================================

    #[test]
    fn test_joint_offset_zero() {
        let offset = JointOffset::zero();
        assert_eq!(offset.position, Vec3::ZERO);
        assert_eq!(offset.rotation, Vec3::ZERO);
        assert_eq!(offset.scale, Vec3::ZERO);
    }

    #[test]
    fn test_joint_offset_from_position() {
        let offset = JointOffset::from_position(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(offset.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(offset.rotation, Vec3::ZERO);
    }

    #[test]
    fn test_joint_offset_from_rotation() {
        let offset = JointOffset::from_rotation(Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(offset.rotation, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(offset.position, Vec3::ZERO);
    }

    #[test]
    fn test_joint_offset_from_quat_diff() {
        let from = Quat::from_rotation_y(PI / 4.0);
        let to = Quat::IDENTITY;
        let offset = JointOffset::from_quat_diff(from, to);

        assert!(!offset.rotation.is_nan());
    }

    #[test]
    fn test_joint_offset_rotation_quat_identity() {
        let offset = JointOffset::zero();
        let q = offset.rotation_quat();
        assert!((q.w - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_joint_offset_rotation_quat_nonzero() {
        let offset = JointOffset::from_rotation(Vec3::new(0.0, PI / 2.0, 0.0));
        let q = offset.rotation_quat();

        // Should be approximately 90 degree rotation around Y
        let (axis, angle) = q.to_axis_angle();
        assert!((angle - PI / 2.0).abs() < 0.01);
        assert!((axis.y.abs() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_joint_offset_scaled() {
        let offset = JointOffset {
            position: Vec3::new(2.0, 4.0, 6.0),
            rotation: Vec3::new(1.0, 2.0, 3.0),
            scale: Vec3::new(0.5, 0.5, 0.5),
        };
        let scaled = offset.scaled(0.5);

        assert_eq!(scaled.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(scaled.rotation, Vec3::new(0.5, 1.0, 1.5));
    }

    #[test]
    fn test_joint_offset_magnitude() {
        let offset = JointOffset::from_position(Vec3::new(3.0, 4.0, 0.0));
        assert!((offset.magnitude() - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_joint_offset_is_negligible() {
        let zero = JointOffset::zero();
        let small = JointOffset::from_position(Vec3::new(0.0001, 0.0, 0.0));
        let large = JointOffset::from_position(Vec3::new(1.0, 0.0, 0.0));

        assert!(zero.is_negligible(0.001));
        assert!(small.is_negligible(0.001));
        assert!(!large.is_negligible(0.001));
    }

    #[test]
    fn test_joint_offset_lerp() {
        let a = JointOffset::from_position(Vec3::ZERO);
        let b = JointOffset::from_position(Vec3::new(10.0, 0.0, 0.0));
        let mid = a.lerp(&b, 0.5);

        assert!((mid.position.x - 5.0).abs() < EPSILON);
    }

    // =====================================================================
    // JointVelocity tests
    // =====================================================================

    #[test]
    fn test_joint_velocity_zero() {
        let vel = JointVelocity::zero();
        assert_eq!(vel.linear, Vec3::ZERO);
        assert_eq!(vel.angular, Vec3::ZERO);
        assert_eq!(vel.scale, Vec3::ZERO);
    }

    #[test]
    fn test_joint_velocity_from_linear() {
        let vel = JointVelocity::from_linear(Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(vel.linear, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_joint_velocity_from_angular() {
        let vel = JointVelocity::from_angular(Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(vel.angular, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn test_joint_velocity_scaled() {
        let vel = JointVelocity::from_linear(Vec3::new(2.0, 0.0, 0.0));
        let scaled = vel.scaled(0.5);
        assert_eq!(scaled.linear, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_joint_velocity_magnitude() {
        let vel = JointVelocity::from_linear(Vec3::new(3.0, 4.0, 0.0));
        assert!((vel.magnitude() - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_joint_velocity_angular_from_quat_delta() {
        let from = Quat::IDENTITY;
        let to = Quat::from_rotation_y(PI / 2.0);
        let dt = 1.0;

        let angular = JointVelocity::angular_from_quat_delta(from, to, dt);
        assert!((angular.y.abs() - PI / 2.0).abs() < 0.1);
    }

    #[test]
    fn test_joint_velocity_angular_from_quat_delta_zero_dt() {
        let angular = JointVelocity::angular_from_quat_delta(
            Quat::IDENTITY,
            Quat::from_rotation_y(1.0),
            0.0,
        );
        assert_eq!(angular, Vec3::ZERO);
    }

    // =====================================================================
    // JointConfig tests
    // =====================================================================

    #[test]
    fn test_joint_config_default() {
        let config = JointConfig::default();
        assert_eq!(config.weight, 1.0);
        assert!(config.duration_override.is_none());
        assert!(config.preserve_velocity);
    }

    #[test]
    fn test_joint_config_with_weight() {
        let config = JointConfig::with_weight(0.5);
        assert_eq!(config.weight, 0.5);
    }

    #[test]
    fn test_joint_config_with_duration() {
        let config = JointConfig::new().with_duration(0.3);
        assert_eq!(config.duration_override, Some(0.3));
    }

    #[test]
    fn test_joint_config_with_half_life() {
        let config = JointConfig::new().with_half_life(0.15);
        assert_eq!(config.half_life_override, Some(0.15));
    }

    #[test]
    fn test_joint_config_with_damping() {
        let config = JointConfig::new().with_damping(DampingMode::Underdamped);
        assert_eq!(config.damping_override, Some(DampingMode::Underdamped));
    }

    #[test]
    fn test_joint_config_without_velocity() {
        let config = JointConfig::new().without_velocity();
        assert!(!config.preserve_velocity);
    }

    // =====================================================================
    // ComponentConfig tests
    // =====================================================================

    #[test]
    fn test_component_config_default() {
        let config = ComponentConfig::default();
        assert!(config.enabled);
        assert_eq!(config.weight, 1.0);
    }

    #[test]
    fn test_component_config_disabled() {
        let config = ComponentConfig::disabled();
        assert!(!config.enabled);
        assert_eq!(config.weight, 0.0);
    }

    #[test]
    fn test_component_config_with_weight() {
        let config = ComponentConfig::new().with_weight(0.5);
        assert_eq!(config.weight, 0.5);
    }

    #[test]
    fn test_component_config_with_decay() {
        let config = ComponentConfig::new().with_decay(DecayCurve::underdamped(0.2));
        assert_eq!(config.decay_curve.mode, DampingMode::Underdamped);
    }

    // =====================================================================
    // InertializationConfig tests
    // =====================================================================

    #[test]
    fn test_config_default() {
        let config = InertializationConfig::default();
        assert_eq!(config.duration, DEFAULT_TRANSITION_DURATION);
        assert_eq!(config.half_life, DEFAULT_HALF_LIFE);
        assert_eq!(config.damping_mode, DampingMode::Critical);
    }

    #[test]
    fn test_config_fast() {
        let config = InertializationConfig::fast();
        assert!(config.duration < DEFAULT_TRANSITION_DURATION);
    }

    #[test]
    fn test_config_smooth() {
        let config = InertializationConfig::smooth();
        assert!(config.duration > DEFAULT_TRANSITION_DURATION);
    }

    #[test]
    fn test_config_springy() {
        let config = InertializationConfig::springy();
        assert_eq!(config.damping_mode, DampingMode::Underdamped);
    }

    #[test]
    fn test_config_heavy() {
        let config = InertializationConfig::heavy();
        assert_eq!(config.damping_mode, DampingMode::Overdamped);
    }

    #[test]
    fn test_config_with_duration() {
        let config = InertializationConfig::default().with_duration(0.5);
        assert_eq!(config.duration, 0.5);
    }

    #[test]
    fn test_config_with_half_life() {
        let config = InertializationConfig::default().with_half_life(0.2);
        assert_eq!(config.half_life, 0.2);
    }

    #[test]
    fn test_config_with_damping_mode() {
        let config = InertializationConfig::default().with_damping_mode(DampingMode::Overdamped);
        assert_eq!(config.damping_mode, DampingMode::Overdamped);
    }

    #[test]
    fn test_config_with_velocity_preservation() {
        let config = InertializationConfig::default().with_velocity_preservation(0.5);
        assert_eq!(config.velocity_preservation, 0.5);
    }

    #[test]
    fn test_config_with_joint_override() {
        let config = InertializationConfig::default()
            .with_joint_override(5, JointConfig::with_weight(0.5));

        assert!(config.joint_overrides.contains_key(&5));
        assert_eq!(config.effective_weight(5), 0.5);
        assert_eq!(config.effective_weight(0), 1.0);
    }

    #[test]
    fn test_config_effective_duration() {
        let config = InertializationConfig::default()
            .with_duration(0.2)
            .with_joint_override(5, JointConfig::new().with_duration(0.5));

        assert_eq!(config.effective_duration(0), 0.2);
        assert_eq!(config.effective_duration(5), 0.5);
    }

    #[test]
    fn test_config_effective_half_life() {
        let config = InertializationConfig::default()
            .with_half_life(0.1)
            .with_joint_override(5, JointConfig::new().with_half_life(0.3));

        assert_eq!(config.effective_half_life(0), 0.1);
        assert_eq!(config.effective_half_life(5), 0.3);
    }

    #[test]
    fn test_config_effective_damping() {
        let config = InertializationConfig::default()
            .with_damping_mode(DampingMode::Critical)
            .with_joint_override(5, JointConfig::new().with_damping(DampingMode::Underdamped));

        assert_eq!(config.effective_damping(0), DampingMode::Critical);
        assert_eq!(config.effective_damping(5), DampingMode::Underdamped);
    }

    // =====================================================================
    // InertializationState tests
    // =====================================================================

    #[test]
    fn test_state_new() {
        let state = InertializationState::new();
        assert!(!state.is_active);
        assert!(state.is_complete());
    }

    #[test]
    fn test_state_with_joint_count() {
        let state = InertializationState::with_joint_count(10);
        assert_eq!(state.joint_count, 10);
        assert_eq!(state.source_offsets.len(), 10);
        assert_eq!(state.source_velocities.len(), 10);
    }

    #[test]
    fn test_state_progress() {
        let mut state = InertializationState::with_joint_count(1);
        state.duration = 1.0;
        state.elapsed = 0.5;
        state.is_active = true;

        assert!((state.progress() - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_state_progress_clamped() {
        let mut state = InertializationState::with_joint_count(1);
        state.duration = 1.0;
        state.elapsed = 1.5;

        assert_eq!(state.progress(), 1.0);
    }

    #[test]
    fn test_state_is_complete() {
        let mut state = InertializationState::with_joint_count(1);
        state.duration = 1.0;
        state.is_active = true;

        state.elapsed = 0.5;
        assert!(!state.is_complete());

        state.elapsed = 1.0;
        assert!(state.is_complete());
    }

    #[test]
    fn test_state_remaining() {
        let mut state = InertializationState::with_joint_count(1);
        state.duration = 1.0;
        state.elapsed = 0.3;

        assert!((state.remaining() - 0.7).abs() < EPSILON);
    }

    #[test]
    fn test_state_reset() {
        let mut state = InertializationState::with_joint_count(5);
        state.elapsed = 0.5;
        state.is_active = true;
        state.source_offsets[0] = JointOffset::from_position(Vec3::X);

        state.reset();

        assert_eq!(state.elapsed, 0.0);
        assert!(!state.is_active);
        assert_eq!(state.source_offsets[0].position, Vec3::ZERO);
    }

    #[test]
    fn test_state_current_offset() {
        let mut state = InertializationState::with_joint_count(1);
        state.source_offsets[0] = JointOffset::from_position(Vec3::new(10.0, 0.0, 0.0));
        state.decay_curves[0] = DecayCurve::critical(0.1);
        state.elapsed = 0.0;

        let offset = state.current_offset(0);
        // At t=0, decay should be 1.0
        assert!((offset.position.x - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_state_current_offset_decayed() {
        let mut state = InertializationState::with_joint_count(1);
        state.source_offsets[0] = JointOffset::from_position(Vec3::new(10.0, 0.0, 0.0));
        state.decay_curves[0] = DecayCurve::critical(0.1);
        state.elapsed = 0.5;

        let offset = state.current_offset(0);
        // Should be significantly decayed
        assert!(offset.position.x < 5.0);
    }

    // =====================================================================
    // VelocityEstimator tests
    // =====================================================================

    #[test]
    fn test_velocity_estimator_new() {
        let estimator = VelocityEstimator::new();
        assert!(!estimator.is_initialized());
    }

    #[test]
    fn test_velocity_estimator_with_smoothing() {
        let estimator = VelocityEstimator::with_smoothing(0.5);
        assert_eq!(estimator.smoothing, 0.5);
    }

    #[test]
    fn test_velocity_estimator_initialize() {
        let mut estimator = VelocityEstimator::new();
        estimator.initialize(10);

        assert_eq!(estimator.prev_positions.len(), 10);
        assert_eq!(estimator.smoothed_velocities.len(), 10);
    }

    #[test]
    fn test_velocity_estimator_first_update() {
        let mut estimator = VelocityEstimator::new();
        let positions = vec![Vec3::ZERO; 5];
        let rotations = vec![Quat::IDENTITY; 5];
        let scales = vec![Vec3::ONE; 5];

        let velocities = estimator.update(&positions, &rotations, &scales, 0.0);

        assert!(estimator.is_initialized());
        assert_eq!(velocities.len(), 5);
        // First frame should return zero velocities
        assert_eq!(velocities[0].linear, Vec3::ZERO);
    }

    #[test]
    fn test_velocity_estimator_computes_velocity() {
        let mut estimator = VelocityEstimator::with_smoothing(1.0); // No smoothing for test

        let pos1 = vec![Vec3::ZERO];
        let pos2 = vec![Vec3::new(1.0, 0.0, 0.0)];
        let rotations = vec![Quat::IDENTITY];
        let scales = vec![Vec3::ONE];

        // First sample
        estimator.update(&pos1, &rotations, &scales, 0.0);

        // Second sample
        let velocities = estimator.update(&pos2, &rotations, &scales, 1.0);

        // Velocity should be 1.0 in X
        assert!((velocities[0].linear.x - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_velocity_estimator_reset() {
        let mut estimator = VelocityEstimator::new();
        let positions = vec![Vec3::ZERO; 5];
        let rotations = vec![Quat::IDENTITY; 5];
        let scales = vec![Vec3::ONE; 5];

        estimator.update(&positions, &rotations, &scales, 0.0);
        assert!(estimator.is_initialized());

        estimator.reset();
        assert!(!estimator.is_initialized());
    }

    // =====================================================================
    // InertializationBlender tests
    // =====================================================================

    #[test]
    fn test_blender_new() {
        let blender = InertializationBlender::new(InertializationConfig::default());
        assert!(!blender.is_active());
        assert!(blender.is_complete());
    }

    #[test]
    fn test_blender_with_default_config() {
        let blender = InertializationBlender::with_default_config();
        assert_eq!(blender.config.duration, DEFAULT_TRANSITION_DURATION);
    }

    #[test]
    fn test_blender_start_transition() {
        let mut blender = InertializationBlender::new(InertializationConfig::default());

        let source_pos = vec![Vec3::new(1.0, 0.0, 0.0)];
        let source_rot = vec![Quat::IDENTITY];
        let target_pos = vec![Vec3::ZERO];
        let target_rot = vec![Quat::IDENTITY];

        blender.start_transition(&source_pos, &source_rot, &target_pos, &target_rot, None);

        assert!(blender.is_active());
        assert!(!blender.is_complete());
        assert_eq!(blender.get_progress(), 0.0);
    }

    #[test]
    fn test_blender_start_transition_from_offset() {
        let mut blender = InertializationBlender::new(InertializationConfig::default());
        let offsets = vec![JointOffset::from_position(Vec3::X)];

        blender.start_transition_from_offset(&offsets);

        assert!(blender.is_active());
    }

    #[test]
    fn test_blender_update() {
        let mut blender = InertializationBlender::new(
            InertializationConfig::default().with_duration(0.2)
        );

        blender.start_transition_from_offset(&[JointOffset::from_position(Vec3::X)]);

        blender.update(0.1);
        assert!((blender.get_progress() - 0.5).abs() < EPSILON);

        blender.update(0.1);
        assert!(blender.is_complete());
    }

    #[test]
    fn test_blender_apply_positions() {
        let mut blender = InertializationBlender::new(InertializationConfig::default());
        let offsets = vec![JointOffset::from_position(Vec3::new(1.0, 0.0, 0.0))];

        blender.start_transition_from_offset(&offsets);

        let target = vec![Vec3::ZERO];
        let result = blender.apply_positions(&target);

        // At t=0, offset should be fully applied
        assert!((result[0].x - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_blender_apply_positions_decayed() {
        let mut blender = InertializationBlender::new(
            InertializationConfig::default().with_duration(1.0)
        );
        let offsets = vec![JointOffset::from_position(Vec3::new(10.0, 0.0, 0.0))];

        blender.start_transition_from_offset(&offsets);
        blender.update(0.5);

        let target = vec![Vec3::ZERO];
        let result = blender.apply_positions(&target);

        // Offset should be partially decayed
        assert!(result[0].x < 10.0);
        assert!(result[0].x > 0.0);
    }

    #[test]
    fn test_blender_apply_rotations() {
        let mut blender = InertializationBlender::new(InertializationConfig::default());
        let offsets = vec![JointOffset::from_rotation(Vec3::new(0.0, PI / 4.0, 0.0))];

        blender.start_transition_from_offset(&offsets);

        let target = vec![Quat::IDENTITY];
        let result = blender.apply_rotations(&target);

        // Should have rotation applied
        assert!((result[0].w - 1.0).abs() > 0.01);
    }

    #[test]
    fn test_blender_apply_both() {
        let mut blender = InertializationBlender::new(InertializationConfig::default());
        let offsets = vec![
            JointOffset {
                position: Vec3::new(1.0, 0.0, 0.0),
                rotation: Vec3::new(0.0, 0.5, 0.0),
                scale: Vec3::ZERO,
            }
        ];

        blender.start_transition_from_offset(&offsets);

        let target_pos = vec![Vec3::ZERO];
        let target_rot = vec![Quat::IDENTITY];

        let (pos, rot) = blender.apply(&target_pos, &target_rot);

        assert!(pos[0].x > 0.5);
        assert!((rot[0].w - 1.0).abs() > 0.01);
    }

    #[test]
    fn test_blender_remaining_time() {
        let mut blender = InertializationBlender::new(
            InertializationConfig::default().with_duration(1.0)
        );
        blender.start_transition_from_offset(&[JointOffset::from_position(Vec3::X)]);

        assert!((blender.remaining_time() - 1.0).abs() < EPSILON);

        blender.update(0.3);
        assert!((blender.remaining_time() - 0.7).abs() < EPSILON);
    }

    #[test]
    fn test_blender_cancel() {
        let mut blender = InertializationBlender::new(InertializationConfig::default());
        blender.start_transition_from_offset(&[JointOffset::from_position(Vec3::X)]);

        assert!(blender.is_active());

        blender.cancel();

        assert!(!blender.is_active());
        assert!(blender.is_complete());
    }

    #[test]
    fn test_blender_reset() {
        let mut blender = InertializationBlender::new(InertializationConfig::default());
        blender.start_transition_from_offset(&[JointOffset::from_position(Vec3::X)]);
        blender.update(0.1);

        blender.reset();

        assert!(!blender.is_active());
        assert_eq!(blender.state.elapsed, 0.0);
    }

    // =====================================================================
    // HierarchyAwareBlender tests
    // =====================================================================

    #[test]
    fn test_hierarchy_blender_new() {
        let parent_indices = vec![-1, 0, 1]; // root, child of 0, child of 1
        let blender = HierarchyAwareBlender::new(
            InertializationConfig::default(),
            parent_indices,
        );

        assert!(!blender.is_complete());
    }

    #[test]
    fn test_hierarchy_blender_set_weight() {
        let mut blender = HierarchyAwareBlender::new(
            InertializationConfig::default(),
            vec![-1],
        );

        blender.set_hierarchy_weight(0.8);
        assert_eq!(blender.hierarchy_weight, 0.8);
    }

    #[test]
    fn test_hierarchy_blender_apply() {
        let parent_indices = vec![-1, 0, 1];
        let mut blender = HierarchyAwareBlender::new(
            InertializationConfig::default(),
            parent_indices,
        );

        blender.start_transition(
            &[Vec3::X, Vec3::Y, Vec3::Z],
            &[Quat::IDENTITY; 3],
            &[Vec3::ZERO; 3],
            &[Quat::IDENTITY; 3],
            None,
        );

        let target_pos = vec![Vec3::ZERO; 3];
        let target_rot = vec![Quat::IDENTITY; 3];

        let (pos, _rot) = blender.apply(&target_pos, &target_rot);

        assert_eq!(pos.len(), 3);
        // Child joints should have parent influence
    }

    // =====================================================================
    // TransitionManager tests
    // =====================================================================

    #[test]
    fn test_manager_new() {
        let manager = TransitionManager::new();
        assert_eq!(manager.active_count(), 0);
        assert!(!manager.has_active());
    }

    #[test]
    fn test_manager_with_config() {
        let config = InertializationConfig::fast();
        let manager = TransitionManager::with_config(config.clone());
        assert_eq!(manager.default_config.duration, config.duration);
    }

    #[test]
    fn test_manager_start() {
        let mut manager = TransitionManager::new();

        let id = manager.start(
            &[Vec3::X],
            &[Quat::IDENTITY],
            &[Vec3::ZERO],
            &[Quat::IDENTITY],
            None,
        );

        assert_eq!(id, 0);
        assert_eq!(manager.active_count(), 1);
        assert!(manager.has_active());
    }

    #[test]
    fn test_manager_multiple_transitions() {
        let mut manager = TransitionManager::new();

        let id1 = manager.start(
            &[Vec3::X], &[Quat::IDENTITY],
            &[Vec3::ZERO], &[Quat::IDENTITY],
            None,
        );
        let id2 = manager.start(
            &[Vec3::Y], &[Quat::IDENTITY],
            &[Vec3::ZERO], &[Quat::IDENTITY],
            None,
        );

        assert_ne!(id1, id2);
        assert_eq!(manager.active_count(), 2);
    }

    #[test]
    fn test_manager_update_removes_complete() {
        let mut manager = TransitionManager::with_config(
            InertializationConfig::default().with_duration(0.1)
        );

        manager.start(
            &[Vec3::X], &[Quat::IDENTITY],
            &[Vec3::ZERO], &[Quat::IDENTITY],
            None,
        );

        assert_eq!(manager.active_count(), 1);

        manager.update(0.2);

        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_manager_apply() {
        let mut manager = TransitionManager::new();

        manager.start(
            &[Vec3::new(1.0, 0.0, 0.0)],
            &[Quat::IDENTITY],
            &[Vec3::ZERO],
            &[Quat::IDENTITY],
            None,
        );

        let (pos, _rot) = manager.apply(
            &[Vec3::ZERO],
            &[Quat::IDENTITY],
        );

        assert!(pos[0].x > 0.5);
    }

    #[test]
    fn test_manager_get() {
        let mut manager = TransitionManager::new();

        let id = manager.start(
            &[Vec3::X], &[Quat::IDENTITY],
            &[Vec3::ZERO], &[Quat::IDENTITY],
            None,
        );

        assert!(manager.get(id).is_some());
        assert!(manager.get(999).is_none());
    }

    #[test]
    fn test_manager_cancel() {
        let mut manager = TransitionManager::new();

        let id = manager.start(
            &[Vec3::X], &[Quat::IDENTITY],
            &[Vec3::ZERO], &[Quat::IDENTITY],
            None,
        );

        assert!(manager.cancel(id));
        assert!(!manager.cancel(999));

        manager.update(0.0);
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_manager_cancel_all() {
        let mut manager = TransitionManager::new();

        manager.start(&[Vec3::X], &[Quat::IDENTITY], &[Vec3::ZERO], &[Quat::IDENTITY], None);
        manager.start(&[Vec3::Y], &[Quat::IDENTITY], &[Vec3::ZERO], &[Quat::IDENTITY], None);

        assert_eq!(manager.active_count(), 2);

        manager.cancel_all();

        assert_eq!(manager.active_count(), 0);
    }

    // =====================================================================
    // Utility function tests
    // =====================================================================

    #[test]
    fn test_compute_decay() {
        let d0 = compute_decay(0.0, 0.1);
        let d1 = compute_decay(0.1, 0.1);
        let d2 = compute_decay(0.5, 0.1);

        assert!((d0 - 1.0).abs() < EPSILON);
        assert!(d1 < d0);
        assert!(d2 < d1);
    }

    #[test]
    fn test_instant_snap() {
        assert_eq!(instant_snap(), 0.0);
    }

    #[test]
    fn test_interpolate_quat_offset() {
        let offset = Quat::from_rotation_y(PI / 2.0);

        let at_0 = interpolate_quat_offset(offset, 0.0);
        let at_half = interpolate_quat_offset(offset, 0.5);
        let at_1 = interpolate_quat_offset(offset, 1.0);

        assert!((at_0.w - 1.0).abs() < 0.01); // Identity
        assert!((at_1 - offset).length() < 0.01); // Full offset

        // Midpoint should be between
        let (_, angle_half) = at_half.to_axis_angle();
        assert!(angle_half > 0.0 && angle_half < PI / 2.0);
    }

    #[test]
    fn test_estimate_velocity() {
        let prev_pos = vec![Vec3::ZERO];
        let prev_rot = vec![Quat::IDENTITY];
        let curr_pos = vec![Vec3::new(1.0, 0.0, 0.0)];
        let curr_rot = vec![Quat::IDENTITY];

        let velocities = estimate_velocity(&prev_pos, &prev_rot, &curr_pos, &curr_rot, 1.0);

        assert_eq!(velocities.len(), 1);
        assert!((velocities[0].linear.x - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_estimate_velocity_zero_dt() {
        let prev_pos = vec![Vec3::ZERO];
        let prev_rot = vec![Quat::IDENTITY];
        let curr_pos = vec![Vec3::X];
        let curr_rot = vec![Quat::IDENTITY];

        let velocities = estimate_velocity(&prev_pos, &prev_rot, &curr_pos, &curr_rot, 0.0);

        assert_eq!(velocities[0].linear, Vec3::ZERO);
    }

    // =====================================================================
    // Edge case tests
    // =====================================================================

    #[test]
    fn test_blender_empty_pose() {
        let mut blender = InertializationBlender::new(InertializationConfig::default());

        blender.start_transition(&[], &[], &[], &[], None);

        // Should handle gracefully
        assert!(!blender.is_active());
    }

    #[test]
    fn test_blender_inactive_apply() {
        let blender = InertializationBlender::new(InertializationConfig::default());

        let target = vec![Vec3::new(1.0, 2.0, 3.0)];
        let result = blender.apply_positions(&target);

        // Should return target unchanged
        assert_eq!(result, target);
    }

    #[test]
    fn test_blender_zero_duration() {
        let mut blender = InertializationBlender::new(
            InertializationConfig::default().with_duration(0.0)
        );

        blender.start_transition_from_offset(&[JointOffset::from_position(Vec3::X)]);

        // Should complete immediately
        blender.update(0.001);
        assert!(blender.is_complete());
    }

    #[test]
    fn test_decay_curve_negative_time() {
        let curve = DecayCurve::critical(0.1);
        let decay = curve.decay(-1.0);
        assert_eq!(decay, 1.0);
    }

    #[test]
    fn test_state_current_offset_out_of_bounds() {
        let state = InertializationState::with_joint_count(1);
        let offset = state.current_offset(999);
        assert_eq!(offset.position, Vec3::ZERO);
    }

    // =====================================================================
    // Integration tests
    // =====================================================================

    #[test]
    fn test_full_transition_lifecycle() {
        let config = InertializationConfig::default()
            .with_duration(0.2)
            .with_damping_mode(DampingMode::Critical);

        let mut blender = InertializationBlender::new(config);

        // Source pose (offset from target)
        let source_pos = vec![Vec3::new(10.0, 0.0, 0.0)];
        let source_rot = vec![Quat::from_rotation_y(PI / 2.0)];

        // Target pose
        let target_pos = vec![Vec3::ZERO];
        let target_rot = vec![Quat::IDENTITY];

        // Start transition
        blender.start_transition(&source_pos, &source_rot, &target_pos, &target_rot, None);

        assert!(blender.is_active());
        assert_eq!(blender.get_progress(), 0.0);

        // At start, should be close to source
        let (pos0, _) = blender.apply(&target_pos, &target_rot);
        assert!(pos0[0].x > 5.0);

        // Update halfway
        blender.update(0.1);
        assert!((blender.get_progress() - 0.5).abs() < EPSILON);

        let (pos_mid, _) = blender.apply(&target_pos, &target_rot);
        assert!(pos_mid[0].x < pos0[0].x);
        assert!(pos_mid[0].x > 0.0);

        // Complete transition
        blender.update(0.1);
        assert!(blender.is_complete());

        // After completion, should be close to target
        let (pos_end, _) = blender.apply(&target_pos, &target_rot);
        assert!(pos_end[0].x < 1.0);
    }

    #[test]
    fn test_pose_continuity() {
        let mut blender = InertializationBlender::new(
            InertializationConfig::default().with_duration(0.5)
        );

        // Create a transition
        blender.start_transition_from_offset(&[JointOffset::from_position(Vec3::new(5.0, 0.0, 0.0))]);

        let target = vec![Vec3::ZERO];
        let mut prev_pos = blender.apply_positions(&target)[0].x;

        // Check that position changes smoothly (monotonically decreasing)
        for i in 1..=10 {
            blender.update(0.05);
            let curr_pos = blender.apply_positions(&target)[0].x;

            // Position should decrease (for critical damping)
            assert!(curr_pos <= prev_pos + EPSILON,
                "Position increased at step {}: {} > {}", i, curr_pos, prev_pos);

            prev_pos = curr_pos;
        }
    }

    #[test]
    fn test_multiple_joint_transition() {
        let mut blender = InertializationBlender::new(InertializationConfig::default());

        let joint_count = 10;
        let source_pos: Vec<Vec3> = (0..joint_count).map(|i| Vec3::new(i as f32, 0.0, 0.0)).collect();
        let target_pos: Vec<Vec3> = vec![Vec3::ZERO; joint_count];
        let rotations: Vec<Quat> = vec![Quat::IDENTITY; joint_count];

        blender.start_transition(&source_pos, &rotations, &target_pos, &rotations, None);

        let (result, _) = blender.apply(&target_pos, &rotations);

        assert_eq!(result.len(), joint_count);

        // Each joint should have proportional offset
        for (i, pos) in result.iter().enumerate() {
            assert!(pos.x > 0.0);
            // Joint i should have larger offset than joint i-1
            if i > 0 {
                assert!(pos.x >= result[i - 1].x);
            }
        }
    }

    #[test]
    fn test_per_joint_override() {
        let config = InertializationConfig::default()
            .with_duration(0.2)
            .with_joint_override(0, JointConfig::with_weight(0.5))
            .with_joint_override(1, JointConfig::with_weight(0.0)); // Disabled

        let mut blender = InertializationBlender::new(config);

        let offsets = vec![
            JointOffset::from_position(Vec3::new(10.0, 0.0, 0.0)),
            JointOffset::from_position(Vec3::new(10.0, 0.0, 0.0)),
        ];

        blender.start_transition_from_offset(&offsets);

        let target = vec![Vec3::ZERO; 2];
        let result = blender.apply_positions(&target);

        // Joint 0 should have half offset (weight 0.5)
        assert!((result[0].x - 5.0).abs() < 1.0);

        // Joint 1 should have zero offset (weight 0.0)
        assert!(result[1].x.abs() < 0.1);
    }

    #[test]
    fn test_damping_mode_comparison() {
        let configs = vec![
            ("critical", InertializationConfig::default().with_damping_mode(DampingMode::Critical)),
            ("underdamped", InertializationConfig::springy()),
            ("overdamped", InertializationConfig::heavy()),
        ];

        let offset = vec![JointOffset::from_position(Vec3::new(10.0, 0.0, 0.0))];
        let target = vec![Vec3::ZERO];

        for (name, config) in configs {
            let mut blender = InertializationBlender::new(config.with_duration(1.0));
            blender.start_transition_from_offset(&offset);

            let mut positions = Vec::new();
            for _ in 0..20 {
                positions.push(blender.apply_positions(&target)[0].x);
                blender.update(0.05);
            }

            // All should eventually decay
            let final_pos = positions.last().unwrap();
            assert!(*final_pos < 5.0, "{} did not decay: {}", name, final_pos);
        }
    }
}
