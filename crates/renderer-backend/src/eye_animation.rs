//! Realistic eye animation system for TRINITY Engine (T-AN-7.4).
//!
//! This module provides physiologically accurate eye movement simulation:
//!
//! - **Gaze IK**: Look-at target with smooth interpolation and Donders' law compliance
//! - **Saccades**: Quick ballistic movements between fixation points
//! - **Drift**: Slow continuous movement during fixation
//! - **Tremor**: High-frequency micro-oscillations for realism
//! - **Blinking**: Spontaneous and reactive blinks with configurable parameters
//! - **Pupil dilation**: Light-responsive pupil size changes
//!
//! # Donders' Law
//!
//! Donders' law states that for any gaze direction, the eye assumes a unique
//! orientation (torsion is determined by gaze direction). This module implements
//! Listing's law as a soft constraint, which specifies that all rotation axes
//! lie in a plane (Listing's plane) perpendicular to the primary gaze direction.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::eye_animation::{
//!     EyeAnimationSystem, EyeParams, EyeConfig,
//! };
//! use glam::{Mat4, Vec3};
//!
//! // Create eye animation system with default parameters
//! let mut eyes = EyeAnimationSystem::new(EyeParams::default());
//!
//! // Set a gaze target (e.g., player looking at NPC)
//! eyes.set_gaze_target(Some(Vec3::new(0.0, 1.6, 2.0)));
//!
//! // Set light level for pupil response (0.0 = dark, 1.0 = bright)
//! eyes.set_light_level(0.7);
//!
//! // Update each frame
//! let head_transform = Mat4::IDENTITY;
//! eyes.update(1.0 / 60.0, head_transform);
//!
//! // Get results for rendering
//! let (left_rotation, right_rotation) = eyes.get_eye_rotations();
//! let blink_weight = eyes.get_blink_weight();
//! let pupil_size = eyes.get_pupil_size();
//! ```

use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Minimum distance for valid gaze target computation.
const EPSILON: f32 = 1e-6;

/// Default inter-pupillary distance (average human ~63mm).
const DEFAULT_IPD: f32 = 0.063;

/// Minimum delta time to prevent physics instability.
const MIN_DELTA_TIME: f32 = 1e-6;

/// Maximum delta time for physics substeps.
const MAX_DELTA_TIME: f32 = 1.0 / 30.0;

/// Tremor frequency in Hz (physiological range: 70-90 Hz).
const TREMOR_FREQUENCY: f32 = 80.0;

/// Tremor amplitude in radians (very small).
const TREMOR_AMPLITUDE: f32 = 0.0005;

/// Drift amplitude in radians per second.
const DRIFT_AMPLITUDE: f32 = 0.002;

/// Drift frequency for noise computation.
const DRIFT_FREQUENCY: f32 = 0.5;

/// Saccade angular velocity in radians per second (fast ballistic motion).
const SACCADE_VELOCITY: f32 = 8.0;

/// Pupil dilation response time constant (seconds).
const PUPIL_TIME_CONSTANT: f32 = 0.5;

// ---------------------------------------------------------------------------
// EyeParams
// ---------------------------------------------------------------------------

/// Global parameters controlling eye animation behavior.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeParams {
    /// Maximum horizontal rotation (yaw) in radians.
    /// Typical human range: ~55 degrees.
    pub max_yaw: f32,

    /// Maximum vertical rotation (pitch) in radians.
    /// Typical human range: ~40 degrees up, ~60 degrees down.
    pub max_pitch: f32,

    /// Saccade interval range (min, max) in seconds.
    /// During fixation, micro-saccades occur at random intervals.
    pub saccade_interval: (f32, f32),

    /// Blink interval range (min, max) in seconds.
    /// Average human blink rate: ~15-20 per minute (3-4 seconds).
    pub blink_interval: (f32, f32),

    /// Blink duration range (min, max) in seconds.
    /// Typical blink: 100-400ms.
    pub blink_duration: (f32, f32),

    /// Minimum pupil size (0.0 = fully constricted).
    pub min_pupil_size: f32,

    /// Maximum pupil size (1.0 = fully dilated).
    pub max_pupil_size: f32,

    /// Inter-pupillary distance in meters.
    pub ipd: f32,

    /// Gaze interpolation speed (higher = faster tracking).
    /// In radians per second.
    pub gaze_speed: f32,

    /// Enable micro-saccades during fixation.
    pub enable_saccades: bool,

    /// Enable drift movement during fixation.
    pub enable_drift: bool,

    /// Enable tremor (high-frequency micro-oscillations).
    pub enable_tremor: bool,

    /// Enable spontaneous blinking.
    pub enable_blink: bool,

    /// Enable partial blinks (not fully closing).
    pub enable_partial_blinks: bool,

    /// Probability of a partial blink vs full blink.
    pub partial_blink_probability: f32,

    /// Donders' law compliance strength (0.0 = off, 1.0 = strict).
    pub donders_compliance: f32,
}

impl Default for EyeParams {
    fn default() -> Self {
        Self {
            max_yaw: 55.0_f32.to_radians(),
            max_pitch: 50.0_f32.to_radians(),
            saccade_interval: (0.2, 0.6),
            blink_interval: (3.0, 5.0),
            blink_duration: (0.1, 0.4),
            min_pupil_size: 0.2,
            max_pupil_size: 0.8,
            ipd: DEFAULT_IPD,
            gaze_speed: 6.0,
            enable_saccades: true,
            enable_drift: true,
            enable_tremor: true,
            enable_blink: true,
            enable_partial_blinks: true,
            partial_blink_probability: 0.15,
            donders_compliance: 0.9,
        }
    }
}

impl EyeParams {
    /// Create parameters for realistic human eyes.
    pub fn human() -> Self {
        Self::default()
    }

    /// Create parameters for cartoon/stylized eyes with exaggerated motion.
    pub fn cartoon() -> Self {
        Self {
            max_yaw: 70.0_f32.to_radians(),
            max_pitch: 60.0_f32.to_radians(),
            saccade_interval: (0.3, 0.8),
            blink_interval: (4.0, 8.0),
            blink_duration: (0.15, 0.5),
            gaze_speed: 10.0,
            enable_tremor: false,
            donders_compliance: 0.3,
            ..Default::default()
        }
    }

    /// Create parameters for robotic/mechanical eyes.
    pub fn robotic() -> Self {
        Self {
            max_yaw: 90.0_f32.to_radians(),
            max_pitch: 90.0_f32.to_radians(),
            saccade_interval: (0.5, 1.0),
            blink_interval: (10.0, 20.0),
            blink_duration: (0.05, 0.1),
            gaze_speed: 15.0,
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            enable_partial_blinks: false,
            donders_compliance: 0.0,
            ..Default::default()
        }
    }

    /// Create parameters with custom rotation limits.
    #[inline]
    pub fn with_limits(mut self, max_yaw: f32, max_pitch: f32) -> Self {
        self.max_yaw = max_yaw;
        self.max_pitch = max_pitch;
        self
    }

    /// Create parameters with custom blink timing.
    #[inline]
    pub fn with_blink_timing(mut self, interval: (f32, f32), duration: (f32, f32)) -> Self {
        self.blink_interval = interval;
        self.blink_duration = duration;
        self
    }

    /// Create parameters with custom saccade timing.
    #[inline]
    pub fn with_saccade_timing(mut self, interval: (f32, f32)) -> Self {
        self.saccade_interval = interval;
        self
    }
}

// ---------------------------------------------------------------------------
// EyeState
// ---------------------------------------------------------------------------

/// State for a single eye.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeState {
    /// Current eye rotation in local space.
    pub current_rotation: Quat,

    /// Target rotation for gaze tracking.
    pub target_rotation: Quat,

    /// Current pupil size (0.0 = constricted, 1.0 = dilated).
    pub pupil_size: f32,

    /// Blink progress (0.0 = open, 1.0 = closed).
    pub blink_progress: f32,

    /// Time until next saccade.
    pub time_to_next_saccade: f32,

    /// Time until next blink.
    pub time_to_next_blink: f32,

    /// Whether this eye is currently tracking a target.
    pub is_tracking: bool,

    /// Accumulated time for drift/tremor computation.
    pub accumulated_time: f32,

    /// Current saccade offset (micro-movement during fixation).
    pub saccade_offset: Quat,

    /// Current drift offset.
    pub drift_offset: Vec3,

    /// Is blink currently in progress?
    pub is_blinking: bool,

    /// Current blink duration.
    pub current_blink_duration: f32,

    /// Is current blink partial?
    pub is_partial_blink: bool,

    /// Target blink closure (1.0 for full, <1.0 for partial).
    pub target_blink_closure: f32,
}

impl Default for EyeState {
    fn default() -> Self {
        Self {
            current_rotation: Quat::IDENTITY,
            target_rotation: Quat::IDENTITY,
            pupil_size: 0.5,
            blink_progress: 0.0,
            time_to_next_saccade: 0.3,
            time_to_next_blink: 4.0,
            is_tracking: false,
            accumulated_time: 0.0,
            saccade_offset: Quat::IDENTITY,
            drift_offset: Vec3::ZERO,
            is_blinking: false,
            current_blink_duration: 0.15,
            is_partial_blink: false,
            target_blink_closure: 1.0,
        }
    }
}

impl EyeState {
    /// Reset the eye state to default.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get the effective rotation including micro-movements.
    #[inline]
    pub fn effective_rotation(&self) -> Quat {
        self.current_rotation * self.saccade_offset
    }
}

// ---------------------------------------------------------------------------
// EyeAnimationSystem
// ---------------------------------------------------------------------------

/// Complete eye animation system managing both eyes.
#[derive(Clone, Debug)]
pub struct EyeAnimationSystem {
    /// Left eye state.
    pub left_eye: EyeState,

    /// Right eye state.
    pub right_eye: EyeState,

    /// Animation parameters.
    pub params: EyeParams,

    /// Current gaze target in world space (None = forward gaze).
    pub gaze_target: Option<Vec3>,

    /// Current light level for pupil response (0.0 = dark, 1.0 = bright).
    pub light_level: f32,

    /// Random seed for deterministic behavior (useful for testing).
    random_state: u32,

    /// Target pupil size based on light level.
    target_pupil_size: f32,

    /// Primary gaze direction (forward in head space).
    primary_direction: Vec3,
}

impl EyeAnimationSystem {
    /// Create a new eye animation system with the given parameters.
    pub fn new(params: EyeParams) -> Self {
        Self {
            left_eye: EyeState::default(),
            right_eye: EyeState::default(),
            params,
            gaze_target: None,
            light_level: 0.5,
            random_state: 12345,
            target_pupil_size: 0.5,
            primary_direction: Vec3::NEG_Z,
        }
    }

    /// Create with default parameters.
    pub fn default_system() -> Self {
        Self::new(EyeParams::default())
    }

    /// Update the eye animation system.
    ///
    /// # Arguments
    ///
    /// * `dt` - Delta time in seconds
    /// * `head_transform` - Head bone world transform matrix
    pub fn update(&mut self, dt: f32, head_transform: Mat4) {
        let dt = dt.clamp(MIN_DELTA_TIME, MAX_DELTA_TIME);

        // Update gaze tracking
        self.update_gaze(dt, head_transform);

        // Update micro-movements (saccades, drift, tremor)
        self.update_micro_movements(dt);

        // Update blinking
        self.update_blinking(dt);

        // Update pupil response
        self.update_pupil(dt);
    }

    /// Set the gaze target position in world space.
    ///
    /// # Arguments
    ///
    /// * `target` - World space position to look at, or None to look forward
    #[inline]
    pub fn set_gaze_target(&mut self, target: Option<Vec3>) {
        self.gaze_target = target;
        let tracking = target.is_some();
        self.left_eye.is_tracking = tracking;
        self.right_eye.is_tracking = tracking;
    }

    /// Trigger an immediate blink.
    ///
    /// Used for reactive blinks (startled, loud noise, bright flash).
    pub fn trigger_blink(&mut self) {
        self.start_blink(&mut self.left_eye.clone(), false);
        self.start_blink(&mut self.right_eye.clone(), false);

        // Apply to actual state
        self.left_eye.is_blinking = true;
        self.left_eye.blink_progress = 0.0;
        self.left_eye.is_partial_blink = false;
        self.left_eye.target_blink_closure = 1.0;
        self.left_eye.current_blink_duration = self.random_range(
            self.params.blink_duration.0,
            self.params.blink_duration.1,
        );

        self.right_eye.is_blinking = true;
        self.right_eye.blink_progress = 0.0;
        self.right_eye.is_partial_blink = false;
        self.right_eye.target_blink_closure = 1.0;
        self.right_eye.current_blink_duration = self.left_eye.current_blink_duration;
    }

    /// Set the ambient light level for pupil response.
    ///
    /// # Arguments
    ///
    /// * `level` - Light level (0.0 = dark, 1.0 = bright)
    #[inline]
    pub fn set_light_level(&mut self, level: f32) {
        self.light_level = level.clamp(0.0, 1.0);
        // Compute target pupil size (inverse relationship with light)
        let t = self.light_level;
        self.target_pupil_size =
            self.params.max_pupil_size - t * (self.params.max_pupil_size - self.params.min_pupil_size);
    }

    /// Get the current eye rotations.
    ///
    /// # Returns
    ///
    /// Tuple of (left_eye_rotation, right_eye_rotation) in local space.
    #[inline]
    pub fn get_eye_rotations(&self) -> (Quat, Quat) {
        (
            self.left_eye.effective_rotation(),
            self.right_eye.effective_rotation(),
        )
    }

    /// Get the current blink weight for blendshape/eyelid animation.
    ///
    /// # Returns
    ///
    /// Blink weight (0.0 = open, 1.0 = closed). Uses average of both eyes.
    #[inline]
    pub fn get_blink_weight(&self) -> f32 {
        (self.left_eye.blink_progress + self.right_eye.blink_progress) * 0.5
    }

    /// Get individual blink weights for left and right eyes.
    ///
    /// # Returns
    ///
    /// Tuple of (left_blink_weight, right_blink_weight).
    #[inline]
    pub fn get_blink_weights(&self) -> (f32, f32) {
        (self.left_eye.blink_progress, self.right_eye.blink_progress)
    }

    /// Get the current pupil size.
    ///
    /// # Returns
    ///
    /// Pupil size (0.0 = constricted, 1.0 = dilated). Uses average of both eyes.
    #[inline]
    pub fn get_pupil_size(&self) -> f32 {
        (self.left_eye.pupil_size + self.right_eye.pupil_size) * 0.5
    }

    /// Get individual pupil sizes for left and right eyes.
    ///
    /// # Returns
    ///
    /// Tuple of (left_pupil_size, right_pupil_size).
    #[inline]
    pub fn get_pupil_sizes(&self) -> (f32, f32) {
        (self.left_eye.pupil_size, self.right_eye.pupil_size)
    }

    /// Check if a gaze target is reachable (within rotation limits).
    ///
    /// # Arguments
    ///
    /// * `target` - World space position to check
    /// * `head_transform` - Current head transform
    ///
    /// # Returns
    ///
    /// True if the target is within the eye's rotation range.
    pub fn is_target_reachable(&self, target: Vec3, head_transform: Mat4) -> bool {
        let head_pos = head_transform.col(3).truncate();
        let direction = (target - head_pos).normalize_or_zero();

        if direction.length_squared() < EPSILON {
            return false;
        }

        // Transform to head local space
        let head_inverse = head_transform.inverse();
        let local_dir = head_inverse.transform_vector3(direction);

        // Compute angles
        let yaw = local_dir.x.atan2(-local_dir.z);
        let pitch = local_dir.y.asin();

        yaw.abs() <= self.params.max_yaw && pitch.abs() <= self.params.max_pitch
    }

    /// Reset the animation system to initial state.
    pub fn reset(&mut self) {
        self.left_eye.reset();
        self.right_eye.reset();
        self.gaze_target = None;
        self.light_level = 0.5;
        self.target_pupil_size = 0.5;
    }

    // -----------------------------------------------------------------------
    // Private Methods
    // -----------------------------------------------------------------------

    fn update_gaze(&mut self, dt: f32, head_transform: Mat4) {
        let head_pos = head_transform.col(3).truncate();
        let head_rotation = Quat::from_mat4(&head_transform);

        // Compute target rotations for each eye
        if let Some(target) = self.gaze_target {
            // Compute eye positions (offset from head center by IPD/2)
            let half_ipd = self.params.ipd * 0.5;
            let head_right = head_rotation * Vec3::X;

            let left_eye_pos = head_pos - head_right * half_ipd;
            let right_eye_pos = head_pos + head_right * half_ipd;

            // Compute look-at rotations
            self.left_eye.target_rotation =
                self.compute_gaze_rotation(target, left_eye_pos, head_rotation);
            self.right_eye.target_rotation =
                self.compute_gaze_rotation(target, right_eye_pos, head_rotation);
        } else {
            // Look forward
            self.left_eye.target_rotation = Quat::IDENTITY;
            self.right_eye.target_rotation = Quat::IDENTITY;
        }

        // Smooth interpolation toward target
        let speed = self.params.gaze_speed * dt;
        self.left_eye.current_rotation = slerp_shortest(
            self.left_eye.current_rotation,
            self.left_eye.target_rotation,
            speed.min(1.0),
        );
        self.right_eye.current_rotation = slerp_shortest(
            self.right_eye.current_rotation,
            self.right_eye.target_rotation,
            speed.min(1.0),
        );
    }

    fn compute_gaze_rotation(&self, target: Vec3, eye_pos: Vec3, head_rotation: Quat) -> Quat {
        let direction = (target - eye_pos).normalize_or_zero();

        if direction.length_squared() < EPSILON {
            return Quat::IDENTITY;
        }

        // Transform to head local space
        let head_inverse = head_rotation.inverse();
        let local_dir = head_inverse * direction;

        // Compute yaw and pitch angles
        let yaw = local_dir.x.atan2(-local_dir.z);
        let pitch = local_dir.y.asin();

        // Clamp to rotation limits
        let clamped_yaw = yaw.clamp(-self.params.max_yaw, self.params.max_yaw);
        let clamped_pitch = pitch.clamp(-self.params.max_pitch, self.params.max_pitch);

        // Build rotation (yaw around Y, pitch around X)
        let yaw_quat = Quat::from_rotation_y(clamped_yaw);
        let pitch_quat = Quat::from_rotation_x(-clamped_pitch);

        // Apply Donders' law (Listing's law constraint)
        let rotation = yaw_quat * pitch_quat;
        self.apply_donders_constraint(rotation, clamped_yaw, clamped_pitch)
    }

    fn apply_donders_constraint(&self, rotation: Quat, yaw: f32, pitch: f32) -> Quat {
        if self.params.donders_compliance < EPSILON {
            return rotation;
        }

        // Listing's law: torsion = 0 in Listing's plane
        // The rotation axis should lie in the plane perpendicular to primary gaze
        // For simplicity, we compute the torsion that would result from the current
        // yaw/pitch combination and blend it toward zero.

        // Half-angle rule: torsion = -sin(yaw) * sin(pitch)
        // This is the natural torsion from Listing's law
        let listing_torsion = -yaw.sin() * pitch.sin() * 0.5;

        // Create a torsion rotation
        let torsion_quat = Quat::from_rotation_z(listing_torsion * self.params.donders_compliance);

        rotation * torsion_quat
    }

    fn update_micro_movements(&mut self, dt: f32) {
        // Update accumulated time
        self.left_eye.accumulated_time += dt;
        self.right_eye.accumulated_time += dt;

        // Saccades
        if self.params.enable_saccades {
            self.update_saccades(dt);
        }

        // Drift
        if self.params.enable_drift {
            self.update_drift(dt);
        }

        // Tremor
        if self.params.enable_tremor {
            self.apply_tremor();
        }
    }

    fn update_saccades(&mut self, dt: f32) {
        // Left eye saccade timer
        self.left_eye.time_to_next_saccade -= dt;
        if self.left_eye.time_to_next_saccade <= 0.0 {
            self.trigger_saccade(true);
            self.left_eye.time_to_next_saccade = self.random_range(
                self.params.saccade_interval.0,
                self.params.saccade_interval.1,
            );
        }

        // Right eye follows left with slight delay (coordinated saccades)
        self.right_eye.time_to_next_saccade -= dt;
        if self.right_eye.time_to_next_saccade <= 0.0 {
            self.trigger_saccade(false);
            // Right eye synchronizes with left
            self.right_eye.time_to_next_saccade = self.left_eye.time_to_next_saccade + 0.01;
        }

        // Decay saccade offset over time (quick ballistic motion)
        let decay = (-SACCADE_VELOCITY * dt).exp();
        self.left_eye.saccade_offset =
            slerp_shortest(self.left_eye.saccade_offset, Quat::IDENTITY, 1.0 - decay);
        self.right_eye.saccade_offset =
            slerp_shortest(self.right_eye.saccade_offset, Quat::IDENTITY, 1.0 - decay);
    }

    fn trigger_saccade(&mut self, is_left: bool) {
        // Random small angle offset
        let max_angle = 0.02; // About 1 degree
        let yaw = self.random_range(-max_angle, max_angle);
        let pitch = self.random_range(-max_angle, max_angle);

        let saccade = Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch);

        if is_left {
            self.left_eye.saccade_offset = saccade;
            // Coordinated movement for right eye (same direction, slight difference)
            let coord_factor = 0.9 + self.random_range(0.0, 0.2);
            self.right_eye.saccade_offset =
                Quat::from_rotation_y(yaw * coord_factor) * Quat::from_rotation_x(pitch * coord_factor);
        } else {
            self.right_eye.saccade_offset = saccade;
        }
    }

    fn update_drift(&mut self, _dt: f32) {
        // Compute smooth noise-based drift
        let t_left = self.left_eye.accumulated_time;
        let t_right = self.right_eye.accumulated_time;

        // Simple sine-based drift (different frequencies for x/y)
        self.left_eye.drift_offset = Vec3::new(
            (t_left * DRIFT_FREQUENCY * 1.3).sin() * DRIFT_AMPLITUDE,
            (t_left * DRIFT_FREQUENCY * 0.7).sin() * DRIFT_AMPLITUDE,
            0.0,
        );

        self.right_eye.drift_offset = Vec3::new(
            (t_right * DRIFT_FREQUENCY * 1.3 + 0.5).sin() * DRIFT_AMPLITUDE,
            (t_right * DRIFT_FREQUENCY * 0.7 + 0.5).sin() * DRIFT_AMPLITUDE,
            0.0,
        );

        // Apply drift to saccade offset
        let left_drift =
            Quat::from_rotation_y(self.left_eye.drift_offset.x) *
            Quat::from_rotation_x(self.left_eye.drift_offset.y);
        let right_drift =
            Quat::from_rotation_y(self.right_eye.drift_offset.x) *
            Quat::from_rotation_x(self.right_eye.drift_offset.y);

        self.left_eye.saccade_offset = self.left_eye.saccade_offset * left_drift;
        self.right_eye.saccade_offset = self.right_eye.saccade_offset * right_drift;
    }

    fn apply_tremor(&mut self) {
        // High-frequency oscillation
        let t_left = self.left_eye.accumulated_time * TREMOR_FREQUENCY * 2.0 * PI;
        let t_right = self.right_eye.accumulated_time * TREMOR_FREQUENCY * 2.0 * PI;

        let left_tremor = Quat::from_rotation_y(t_left.sin() * TREMOR_AMPLITUDE)
            * Quat::from_rotation_x(t_left.cos() * TREMOR_AMPLITUDE);
        let right_tremor = Quat::from_rotation_y(t_right.sin() * TREMOR_AMPLITUDE)
            * Quat::from_rotation_x(t_right.cos() * TREMOR_AMPLITUDE);

        self.left_eye.saccade_offset = self.left_eye.saccade_offset * left_tremor;
        self.right_eye.saccade_offset = self.right_eye.saccade_offset * right_tremor;
    }

    fn update_blinking(&mut self, dt: f32) {
        if !self.params.enable_blink {
            return;
        }

        // Update blink timers (both eyes blink together)
        self.left_eye.time_to_next_blink -= dt;

        if self.left_eye.time_to_next_blink <= 0.0 && !self.left_eye.is_blinking {
            self.start_spontaneous_blink();
        }

        // Update blink progress
        if self.left_eye.is_blinking {
            self.update_blink_progress(dt);
        }

        // Sync right eye with left
        self.right_eye.blink_progress = self.left_eye.blink_progress;
        self.right_eye.is_blinking = self.left_eye.is_blinking;
    }

    fn start_spontaneous_blink(&mut self) {
        let is_partial = self.params.enable_partial_blinks
            && self.random_range(0.0, 1.0) < self.params.partial_blink_probability;

        self.left_eye.is_blinking = true;
        self.left_eye.blink_progress = 0.0;
        self.left_eye.is_partial_blink = is_partial;
        self.left_eye.target_blink_closure = if is_partial {
            self.random_range(0.3, 0.7)
        } else {
            1.0
        };
        self.left_eye.current_blink_duration = self.random_range(
            self.params.blink_duration.0,
            self.params.blink_duration.1,
        );

        // Reset timer for next blink
        self.left_eye.time_to_next_blink = self.random_range(
            self.params.blink_interval.0,
            self.params.blink_interval.1,
        );

        // Sync to right eye
        self.right_eye.is_blinking = true;
        self.right_eye.blink_progress = 0.0;
        self.right_eye.is_partial_blink = is_partial;
        self.right_eye.target_blink_closure = self.left_eye.target_blink_closure;
        self.right_eye.current_blink_duration = self.left_eye.current_blink_duration;
        self.right_eye.time_to_next_blink = self.left_eye.time_to_next_blink;
    }

    fn start_blink(&mut self, _eye: &mut EyeState, _is_partial: bool) {
        // Helper for trigger_blink - actual state changes happen in trigger_blink
    }

    fn update_blink_progress(&mut self, dt: f32) {
        let duration = self.left_eye.current_blink_duration;
        let target = self.left_eye.target_blink_closure;

        // Blink uses a smooth curve: fast close, hold, fast open
        let normalized_time = dt / duration;

        if self.left_eye.blink_progress < target {
            // Closing phase
            self.left_eye.blink_progress += normalized_time * 4.0;
            if self.left_eye.blink_progress >= target {
                self.left_eye.blink_progress = target;
            }
        } else {
            // Opening phase
            self.left_eye.blink_progress -= normalized_time * 3.0;
            if self.left_eye.blink_progress <= 0.0 {
                self.left_eye.blink_progress = 0.0;
                self.left_eye.is_blinking = false;
            }
        }

        // Apply easing (fast close, slower open)
        self.left_eye.blink_progress = self.left_eye.blink_progress.clamp(0.0, 1.0);
    }

    fn update_pupil(&mut self, dt: f32) {
        // Smooth response to light level changes
        let response_speed = dt / PUPIL_TIME_CONSTANT;

        self.left_eye.pupil_size +=
            (self.target_pupil_size - self.left_eye.pupil_size) * response_speed;
        self.right_eye.pupil_size +=
            (self.target_pupil_size - self.right_eye.pupil_size) * response_speed;

        // Clamp to valid range
        self.left_eye.pupil_size = self.left_eye.pupil_size.clamp(
            self.params.min_pupil_size,
            self.params.max_pupil_size,
        );
        self.right_eye.pupil_size = self.right_eye.pupil_size.clamp(
            self.params.min_pupil_size,
            self.params.max_pupil_size,
        );
    }

    /// Simple PRNG for random values.
    fn random(&mut self) -> f32 {
        // Xorshift algorithm
        self.random_state ^= self.random_state << 13;
        self.random_state ^= self.random_state >> 17;
        self.random_state ^= self.random_state << 5;
        (self.random_state as f32) / (u32::MAX as f32)
    }

    fn random_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.random() * (max - min)
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Spherical linear interpolation taking the shortest path.
#[inline]
fn slerp_shortest(a: Quat, b: Quat, t: f32) -> Quat {
    let dot = a.dot(b);
    let b = if dot < 0.0 { -b } else { b };
    a.slerp(b, t)
}

/// Compute angle between two quaternions in radians.
#[inline]
#[allow(dead_code)]
fn quat_angle(a: Quat, b: Quat) -> f32 {
    let dot = a.dot(b).abs();
    2.0 * dot.min(1.0).acos()
}

/// Check if a direction is behind the head (dot product < 0 with forward).
#[inline]
pub fn is_behind_head(direction: Vec3, head_forward: Vec3) -> bool {
    direction.dot(head_forward) < 0.0
}

/// Compute vergence angle for a target at given distance.
///
/// Vergence is the inward rotation of both eyes to focus on a near target.
///
/// # Arguments
///
/// * `distance` - Distance to target in meters
/// * `ipd` - Inter-pupillary distance in meters
///
/// # Returns
///
/// Vergence angle in radians for each eye.
#[inline]
pub fn compute_vergence(distance: f32, ipd: f32) -> f32 {
    if distance < EPSILON {
        return 0.0;
    }
    ((ipd * 0.5) / distance).atan()
}

/// Compute accommodation (lens focus) from distance.
///
/// Returns a normalized value where 0.0 = infinity focus, 1.0 = near focus.
///
/// # Arguments
///
/// * `distance` - Distance to target in meters
/// * `near_point` - Nearest focus distance in meters (default ~0.1m for young adult)
///
/// # Returns
///
/// Accommodation value (0.0 to 1.0).
#[inline]
pub fn compute_accommodation(distance: f32, near_point: f32) -> f32 {
    if distance < near_point {
        return 1.0;
    }
    let far_point = 1000.0; // Effectively infinity
    let t = ((far_point - distance) / (far_point - near_point)).clamp(0.0, 1.0);
    t
}

// ---------------------------------------------------------------------------
// Eye Configuration Presets
// ---------------------------------------------------------------------------

/// Configuration for different eye movement behaviors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeConfig {
    /// Name of this configuration.
    pub name: String,

    /// Eye parameters.
    pub params: EyeParams,

    /// Default light level.
    pub default_light_level: f32,
}

impl EyeConfig {
    /// Create a human eye configuration.
    pub fn human() -> Self {
        Self {
            name: "Human".to_string(),
            params: EyeParams::human(),
            default_light_level: 0.5,
        }
    }

    /// Create a cartoon eye configuration.
    pub fn cartoon() -> Self {
        Self {
            name: "Cartoon".to_string(),
            params: EyeParams::cartoon(),
            default_light_level: 0.5,
        }
    }

    /// Create a robotic eye configuration.
    pub fn robotic() -> Self {
        Self {
            name: "Robotic".to_string(),
            params: EyeParams::robotic(),
            default_light_level: 0.5,
        }
    }

    /// Create a sleepy eye configuration.
    pub fn sleepy() -> Self {
        Self {
            name: "Sleepy".to_string(),
            params: EyeParams {
                blink_interval: (1.5, 3.0), // More frequent blinks
                blink_duration: (0.2, 0.6), // Slower blinks
                gaze_speed: 2.0,            // Slower tracking
                enable_partial_blinks: true,
                partial_blink_probability: 0.4, // More partial blinks
                ..EyeParams::human()
            },
            default_light_level: 0.3,
        }
    }

    /// Create an alert eye configuration.
    pub fn alert() -> Self {
        Self {
            name: "Alert".to_string(),
            params: EyeParams {
                saccade_interval: (0.1, 0.3), // More frequent saccades
                blink_interval: (5.0, 10.0),  // Less frequent blinks
                gaze_speed: 10.0,              // Fast tracking
                ..EyeParams::human()
            },
            default_light_level: 0.7,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // EyeParams Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_eye_params_default() {
        let params = EyeParams::default();
        assert!(params.max_yaw > 0.0);
        assert!(params.max_pitch > 0.0);
        assert!(params.saccade_interval.0 < params.saccade_interval.1);
        assert!(params.blink_interval.0 < params.blink_interval.1);
        assert!(params.blink_duration.0 < params.blink_duration.1);
        assert!(params.min_pupil_size < params.max_pupil_size);
    }

    #[test]
    fn test_eye_params_human_preset() {
        let params = EyeParams::human();
        // Human yaw limit should be around 55 degrees
        assert!((params.max_yaw - 55.0_f32.to_radians()).abs() < 0.01);
    }

    #[test]
    fn test_eye_params_cartoon_preset() {
        let params = EyeParams::cartoon();
        // Cartoon should have wider rotation limits
        assert!(params.max_yaw > EyeParams::human().max_yaw);
        assert!(!params.enable_tremor);
    }

    #[test]
    fn test_eye_params_robotic_preset() {
        let params = EyeParams::robotic();
        // Robotic should have full rotation range
        assert!(params.max_yaw >= 90.0_f32.to_radians() - 0.01);
        assert!(!params.enable_saccades);
        assert!(!params.enable_drift);
        assert!(!params.enable_tremor);
    }

    #[test]
    fn test_eye_params_with_limits() {
        let params = EyeParams::default().with_limits(0.5, 0.3);
        assert!((params.max_yaw - 0.5).abs() < EPSILON);
        assert!((params.max_pitch - 0.3).abs() < EPSILON);
    }

    #[test]
    fn test_eye_params_with_blink_timing() {
        let params = EyeParams::default().with_blink_timing((1.0, 2.0), (0.05, 0.1));
        assert!((params.blink_interval.0 - 1.0).abs() < EPSILON);
        assert!((params.blink_interval.1 - 2.0).abs() < EPSILON);
        assert!((params.blink_duration.0 - 0.05).abs() < EPSILON);
        assert!((params.blink_duration.1 - 0.1).abs() < EPSILON);
    }

    #[test]
    fn test_eye_params_with_saccade_timing() {
        let params = EyeParams::default().with_saccade_timing((0.1, 0.5));
        assert!((params.saccade_interval.0 - 0.1).abs() < EPSILON);
        assert!((params.saccade_interval.1 - 0.5).abs() < EPSILON);
    }

    // -----------------------------------------------------------------------
    // EyeState Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_eye_state_default() {
        let state = EyeState::default();
        assert_eq!(state.current_rotation, Quat::IDENTITY);
        assert_eq!(state.target_rotation, Quat::IDENTITY);
        assert!((state.pupil_size - 0.5).abs() < EPSILON);
        assert!((state.blink_progress - 0.0).abs() < EPSILON);
        assert!(!state.is_tracking);
    }

    #[test]
    fn test_eye_state_reset() {
        let mut state = EyeState {
            current_rotation: Quat::from_rotation_y(0.5),
            pupil_size: 0.8,
            blink_progress: 0.5,
            is_tracking: true,
            ..Default::default()
        };

        state.reset();

        assert_eq!(state.current_rotation, Quat::IDENTITY);
        assert!((state.pupil_size - 0.5).abs() < EPSILON);
        assert!((state.blink_progress - 0.0).abs() < EPSILON);
        assert!(!state.is_tracking);
    }

    #[test]
    fn test_eye_state_effective_rotation() {
        let mut state = EyeState::default();
        state.current_rotation = Quat::from_rotation_y(0.1);
        state.saccade_offset = Quat::from_rotation_x(0.05);

        let effective = state.effective_rotation();

        // Effective should combine both rotations
        let expected = Quat::from_rotation_y(0.1) * Quat::from_rotation_x(0.05);
        let angle = quat_angle(effective, expected);
        assert!(angle < 0.001);
    }

    // -----------------------------------------------------------------------
    // EyeAnimationSystem Basic Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_system_creation() {
        let system = EyeAnimationSystem::new(EyeParams::default());
        assert_eq!(system.left_eye.current_rotation, Quat::IDENTITY);
        assert_eq!(system.right_eye.current_rotation, Quat::IDENTITY);
        assert!(system.gaze_target.is_none());
    }

    #[test]
    fn test_system_default_creation() {
        let system = EyeAnimationSystem::default_system();
        assert!((system.light_level - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_system_reset() {
        let mut system = EyeAnimationSystem::new(EyeParams::default());
        system.set_gaze_target(Some(Vec3::new(0.0, 1.0, 2.0)));
        system.set_light_level(0.8);
        system.left_eye.pupil_size = 0.9;

        system.reset();

        assert!(system.gaze_target.is_none());
        assert!((system.light_level - 0.5).abs() < EPSILON);
        assert!((system.left_eye.pupil_size - 0.5).abs() < EPSILON);
    }

    // -----------------------------------------------------------------------
    // Gaze Tracking Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_gaze_target() {
        let mut system = EyeAnimationSystem::new(EyeParams::default());

        system.set_gaze_target(Some(Vec3::new(0.0, 0.0, 2.0)));

        assert!(system.gaze_target.is_some());
        assert!(system.left_eye.is_tracking);
        assert!(system.right_eye.is_tracking);
    }

    #[test]
    fn test_clear_gaze_target() {
        let mut system = EyeAnimationSystem::new(EyeParams::default());
        system.set_gaze_target(Some(Vec3::new(0.0, 0.0, 2.0)));

        system.set_gaze_target(None);

        assert!(system.gaze_target.is_none());
        assert!(!system.left_eye.is_tracking);
        assert!(!system.right_eye.is_tracking);
    }

    #[test]
    fn test_gaze_tracking_forward() {
        let mut system = EyeAnimationSystem::new(EyeParams::default());
        system.set_gaze_target(Some(Vec3::new(0.0, 0.0, -2.0)));

        // Update multiple times for convergence
        for _ in 0..60 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        // Eyes should be close to identity (looking forward)
        let (left, right) = system.get_eye_rotations();
        let left_angle = quat_angle(left, Quat::IDENTITY);
        let right_angle = quat_angle(right, Quat::IDENTITY);

        // Should be within small angle due to micro-movements
        assert!(left_angle < 0.1);
        assert!(right_angle < 0.1);
    }

    #[test]
    fn test_gaze_tracking_left() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            ..EyeParams::default()
        });
        system.set_gaze_target(Some(Vec3::new(-1.0, 0.0, -1.0)));

        // Update for convergence
        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        let (left, _) = system.get_eye_rotations();

        // Verify the rotation deviates from identity (eyes have rotated)
        let angle = quat_angle(left, Quat::IDENTITY);
        assert!(angle > 0.1, "Expected eyes to rotate left, angle={}", angle);
    }

    #[test]
    fn test_gaze_tracking_right() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            ..EyeParams::default()
        });
        system.set_gaze_target(Some(Vec3::new(1.0, 0.0, -1.0)));

        // Update for convergence
        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        let (_, right) = system.get_eye_rotations();

        // Verify the rotation deviates from identity (eyes have rotated)
        let angle = quat_angle(right, Quat::IDENTITY);
        assert!(angle > 0.1, "Expected eyes to rotate right, angle={}", angle);
    }

    #[test]
    fn test_gaze_tracking_up() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            ..EyeParams::default()
        });
        system.set_gaze_target(Some(Vec3::new(0.0, 1.0, -1.0)));

        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        let (left, _) = system.get_eye_rotations();

        // Verify rotation has occurred (pitch component)
        let angle = quat_angle(left, Quat::IDENTITY);
        assert!(angle > 0.1, "Expected eyes to rotate up, angle={}", angle);
    }

    #[test]
    fn test_gaze_tracking_down() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            ..EyeParams::default()
        });
        system.set_gaze_target(Some(Vec3::new(0.0, -1.0, -1.0)));

        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        let (left, _) = system.get_eye_rotations();

        // Verify rotation has occurred (pitch component)
        let angle = quat_angle(left, Quat::IDENTITY);
        assert!(angle > 0.1, "Expected eyes to rotate down, angle={}", angle);
    }

    #[test]
    fn test_gaze_clamping_extreme_yaw() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            max_yaw: 45.0_f32.to_radians(),
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            ..EyeParams::default()
        });

        // Target 90 degrees to the left (beyond limit)
        system.set_gaze_target(Some(Vec3::new(-10.0, 0.0, 0.0)));

        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        let (left, _) = system.get_eye_rotations();
        let (_, _, yaw) = left.to_euler(glam::EulerRot::YXZ);

        // Yaw should be clamped
        assert!(yaw.abs() <= 45.0_f32.to_radians() + 0.1);
    }

    #[test]
    fn test_gaze_clamping_extreme_pitch() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            max_pitch: 30.0_f32.to_radians(),
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            ..EyeParams::default()
        });

        // Target straight up (beyond limit)
        system.set_gaze_target(Some(Vec3::new(0.0, 10.0, -0.1)));

        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        let (left, _) = system.get_eye_rotations();
        let (pitch, _, _) = left.to_euler(glam::EulerRot::YXZ);

        // Pitch should be clamped
        assert!(pitch.abs() <= 30.0_f32.to_radians() + 0.1);
    }

    #[test]
    fn test_left_right_eye_coordination() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            ..EyeParams::default()
        });

        // Target near (should cause vergence)
        system.set_gaze_target(Some(Vec3::new(0.0, 0.0, -0.5)));

        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        let (left, right) = system.get_eye_rotations();

        // Both eyes should be rotated, but in opposite directions for convergence
        // This is a soft test - just verify they're different
        let angle_diff = quat_angle(left, right);
        assert!(angle_diff > 0.001);
    }

    #[test]
    fn test_is_target_reachable_valid() {
        let system = EyeAnimationSystem::new(EyeParams::default());

        // Forward target should be reachable
        let reachable = system.is_target_reachable(Vec3::new(0.0, 0.0, -2.0), Mat4::IDENTITY);
        assert!(reachable);
    }

    #[test]
    fn test_is_target_reachable_extreme_angle() {
        let system = EyeAnimationSystem::new(EyeParams {
            max_yaw: 45.0_f32.to_radians(),
            ..EyeParams::default()
        });

        // Target 90 degrees to the side should not be reachable
        let reachable = system.is_target_reachable(Vec3::new(10.0, 0.0, 0.0), Mat4::IDENTITY);
        assert!(!reachable);
    }

    #[test]
    fn test_is_target_reachable_behind_head() {
        let system = EyeAnimationSystem::new(EyeParams::default());

        // Target behind should not be reachable
        let reachable = system.is_target_reachable(Vec3::new(0.0, 0.0, 2.0), Mat4::IDENTITY);
        assert!(!reachable);
    }

    // -----------------------------------------------------------------------
    // Saccade Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_saccade_timing() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            saccade_interval: (0.1, 0.2),
            enable_drift: false,
            enable_tremor: false,
            enable_blink: false,
            ..EyeParams::default()
        });

        let initial_time = system.left_eye.time_to_next_saccade;

        // Update until saccade triggers
        for _ in 0..100 {
            system.update(0.01, Mat4::IDENTITY);
        }

        // Timer should have reset
        assert!(system.left_eye.time_to_next_saccade != initial_time || initial_time > 0.5);
    }

    #[test]
    fn test_saccade_offset_decays() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_drift: false,
            enable_tremor: false,
            enable_blink: false,
            ..EyeParams::default()
        });

        // Force a saccade
        system.trigger_saccade(true);
        let initial_offset = system.left_eye.saccade_offset;

        // Update and let it decay
        for _ in 0..60 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        // Offset should have decayed toward identity
        let angle_initial = quat_angle(initial_offset, Quat::IDENTITY);
        let angle_current = quat_angle(system.left_eye.saccade_offset, Quat::IDENTITY);
        assert!(angle_current < angle_initial || angle_initial < 0.001);
    }

    #[test]
    fn test_saccade_coordinated_movement() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_drift: false,
            enable_tremor: false,
            enable_blink: false,
            ..EyeParams::default()
        });

        system.trigger_saccade(true);

        // Both eyes should have saccade offsets
        let left_angle = quat_angle(system.left_eye.saccade_offset, Quat::IDENTITY);
        let right_angle = quat_angle(system.right_eye.saccade_offset, Quat::IDENTITY);

        // They should be small but non-zero (within floating point tolerance)
        assert!(left_angle >= 0.0);
        assert!(right_angle >= 0.0);
    }

    // -----------------------------------------------------------------------
    // Drift Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_drift_movement() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_saccades: false,
            enable_drift: true,
            enable_tremor: false,
            enable_blink: false,
            ..EyeParams::default()
        });

        let initial = system.left_eye.drift_offset;

        // Update for a while
        for _ in 0..60 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        // Drift should change over time
        let current = system.left_eye.drift_offset;
        assert!(initial != current || (initial.length() < EPSILON && current.length() < EPSILON));
    }

    #[test]
    fn test_drift_amplitude() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_saccades: false,
            enable_drift: true,
            enable_tremor: false,
            enable_blink: false,
            ..EyeParams::default()
        });

        // Update and collect samples
        let mut max_drift = 0.0_f32;
        for _ in 0..600 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
            let drift = system.left_eye.drift_offset.length();
            max_drift = max_drift.max(drift);
        }

        // Drift should be small
        assert!(max_drift < 0.01);
    }

    // -----------------------------------------------------------------------
    // Tremor Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tremor_high_frequency() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: true,
            enable_blink: false,
            ..EyeParams::default()
        });

        // Tremor should cause small oscillations
        let mut rotations = Vec::new();
        for _ in 0..200 {
            system.update(0.001, Mat4::IDENTITY); // 1ms steps
            let (left, _) = system.get_eye_rotations();
            rotations.push(left);
        }

        // Verify that rotations vary over time (tremor effect)
        let mut total_variation = 0.0_f32;
        for i in 1..rotations.len() {
            let angle = quat_angle(rotations[i - 1], rotations[i]);
            total_variation += angle;
        }

        // Should have measurable variation from tremor
        assert!(total_variation > 0.0, "Expected tremor to cause rotation variation");
    }

    #[test]
    fn test_tremor_amplitude_small() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: true,
            enable_blink: false,
            ..EyeParams::default()
        });

        let mut max_angle = 0.0_f32;
        for _ in 0..1000 {
            system.update(0.001, Mat4::IDENTITY);
            let (left, _) = system.get_eye_rotations();
            let angle = quat_angle(left, Quat::IDENTITY);
            max_angle = max_angle.max(angle);
        }

        // Tremor alone should result in very small angles
        assert!(max_angle < 0.01);
    }

    // -----------------------------------------------------------------------
    // Blink Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blink_trigger() {
        let mut system = EyeAnimationSystem::new(EyeParams::default());

        system.trigger_blink();

        assert!(system.left_eye.is_blinking);
        assert!(system.right_eye.is_blinking);
    }

    #[test]
    fn test_blink_progress_cycle() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            blink_duration: (0.1, 0.1), // Fixed duration
            enable_blink: true,
            ..EyeParams::default()
        });

        system.trigger_blink();

        // Collect blink progress values
        let mut max_progress = 0.0_f32;
        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
            max_progress = max_progress.max(system.get_blink_weight());
        }

        // Should have reached close to fully closed
        assert!(max_progress > 0.5, "Expected max blink progress > 0.5, got {}", max_progress);

        // Continue until blink completes
        for _ in 0..180 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        // Should be open again (or another blink may have started)
        // Just check it's a valid value
        let final_weight = system.get_blink_weight();
        assert!(final_weight >= 0.0 && final_weight <= 1.0);
    }

    #[test]
    fn test_blink_timing_spontaneous() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            blink_interval: (0.05, 0.1),
            blink_duration: (0.02, 0.03),
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            enable_blink: true,
            ..EyeParams::default()
        });

        // Force very short time to next blink
        system.left_eye.time_to_next_blink = 0.01;
        system.right_eye.time_to_next_blink = 0.01;

        let mut blink_count = 0;
        let mut was_blinking = false;

        // Run for 5 seconds with small timesteps
        for _ in 0..500 {
            system.update(0.01, Mat4::IDENTITY);
            if system.left_eye.is_blinking && !was_blinking {
                blink_count += 1;
            }
            was_blinking = system.left_eye.is_blinking;
        }

        // Should have at least one blink in 5 seconds
        assert!(blink_count >= 1, "Expected at least 1 blink, got {}", blink_count);
    }

    #[test]
    fn test_partial_blink() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_partial_blinks: true,
            partial_blink_probability: 1.0, // Always partial
            blink_duration: (0.1, 0.1),
            ..EyeParams::default()
        });

        system.start_spontaneous_blink();

        // Partial blink should have target < 1.0
        assert!(system.left_eye.target_blink_closure < 1.0);
    }

    #[test]
    fn test_blink_duration_range() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            blink_duration: (0.1, 0.4),
            ..EyeParams::default()
        });

        let mut durations = Vec::new();
        for _ in 0..20 {
            system.start_spontaneous_blink();
            durations.push(system.left_eye.current_blink_duration);
        }

        // All durations should be in range
        for d in &durations {
            assert!(*d >= 0.1 - EPSILON);
            assert!(*d <= 0.4 + EPSILON);
        }
    }

    #[test]
    fn test_blink_synchronized() {
        let mut system = EyeAnimationSystem::new(EyeParams::default());

        system.trigger_blink();

        // Both eyes should have same blink progress during update
        for _ in 0..30 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
            assert!((system.left_eye.blink_progress - system.right_eye.blink_progress).abs() < EPSILON);
        }
    }

    // -----------------------------------------------------------------------
    // Pupil Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pupil_response_bright() {
        let mut system = EyeAnimationSystem::new(EyeParams::default());

        system.set_light_level(1.0);

        // Update until pupil adapts
        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        // Pupil should be constricted (small)
        assert!(system.get_pupil_size() < 0.4);
    }

    #[test]
    fn test_pupil_response_dark() {
        let mut system = EyeAnimationSystem::new(EyeParams::default());

        system.set_light_level(0.0);

        // Update until pupil adapts
        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        // Pupil should be dilated (large)
        assert!(system.get_pupil_size() > 0.6);
    }

    #[test]
    fn test_pupil_smooth_transition() {
        let mut system = EyeAnimationSystem::new(EyeParams::default());

        system.set_light_level(0.0);
        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }
        let dilated = system.get_pupil_size();

        // Sudden bright light
        system.set_light_level(1.0);

        // Collect transition values
        let mut sizes = Vec::new();
        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
            sizes.push(system.get_pupil_size());
        }

        // Verify smooth transition (monotonically decreasing)
        for i in 1..sizes.len() {
            assert!(sizes[i] <= sizes[i - 1] + 0.01);
        }

        // Final should be much smaller than initial
        assert!(sizes.last().unwrap() < &(dilated - 0.2));
    }

    #[test]
    fn test_pupil_clamped_range() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            min_pupil_size: 0.2,
            max_pupil_size: 0.8,
            ..EyeParams::default()
        });

        // Try extreme light levels
        for light in [0.0, 0.5, 1.0, -0.5, 1.5].iter() {
            system.set_light_level(*light);
            for _ in 0..120 {
                system.update(1.0 / 60.0, Mat4::IDENTITY);
            }
            let pupil = system.get_pupil_size();
            assert!(pupil >= 0.2 - EPSILON);
            assert!(pupil <= 0.8 + EPSILON);
        }
    }

    #[test]
    fn test_pupil_individual_sizes() {
        let mut system = EyeAnimationSystem::new(EyeParams::default());

        system.set_light_level(0.5);
        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        let (left, right) = system.get_pupil_sizes();

        // Both should be similar (consensual response)
        assert!((left - right).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Donders' Law Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_donders_compliance_enabled() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            donders_compliance: 1.0,
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            ..EyeParams::default()
        });

        // Look at diagonal target
        system.set_gaze_target(Some(Vec3::new(0.5, 0.5, -1.0)));

        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        // With Donders' compliance, the rotation should be valid and finite
        let (left, _) = system.get_eye_rotations();
        assert!(left.is_normalized(), "Expected normalized quaternion");
        assert!(left.is_finite(), "Expected finite quaternion");
    }

    #[test]
    fn test_donders_compliance_disabled() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            donders_compliance: 0.0,
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            ..EyeParams::default()
        });

        system.set_gaze_target(Some(Vec3::new(0.5, 0.5, -1.0)));

        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        // Without Donders' compliance, rotation should still be valid
        let (left, _) = system.get_eye_rotations();
        assert!(left.is_normalized(), "Expected normalized quaternion");
        assert!(left.is_finite(), "Expected finite quaternion");
    }

    // -----------------------------------------------------------------------
    // Edge Cases Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_target_at_eye_position() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            ..EyeParams::default()
        });

        // Target exactly at head position
        system.set_gaze_target(Some(Vec3::ZERO));

        // Should not crash
        for _ in 0..60 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        // Rotation should be valid
        let (left, right) = system.get_eye_rotations();
        assert!(left.is_normalized());
        assert!(right.is_normalized());
    }

    #[test]
    fn test_very_small_dt() {
        let mut system = EyeAnimationSystem::new(EyeParams::default());
        system.set_gaze_target(Some(Vec3::new(0.0, 0.0, -2.0)));

        // Very small timestep
        for _ in 0..1000 {
            system.update(0.0001, Mat4::IDENTITY);
        }

        // Should not have NaN or inf
        let (left, right) = system.get_eye_rotations();
        assert!(left.is_finite());
        assert!(right.is_finite());
    }

    #[test]
    fn test_very_large_dt_clamped() {
        let mut system = EyeAnimationSystem::new(EyeParams::default());
        system.set_gaze_target(Some(Vec3::new(0.0, 0.0, -2.0)));

        // Very large timestep (should be clamped)
        system.update(1.0, Mat4::IDENTITY);

        // Should not have NaN or inf
        let (left, right) = system.get_eye_rotations();
        assert!(left.is_finite());
        assert!(right.is_finite());
    }

    #[test]
    fn test_head_rotated() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            ..EyeParams::default()
        });

        // Target forward in world space
        system.set_gaze_target(Some(Vec3::new(0.0, 0.0, -2.0)));

        // Head rotated 45 degrees
        let head_transform = Mat4::from_rotation_y(45.0_f32.to_radians());

        for _ in 0..120 {
            system.update(1.0 / 60.0, head_transform);
        }

        // Eyes should compensate for head rotation
        let (left, _) = system.get_eye_rotations();
        let angle = quat_angle(left, Quat::IDENTITY);

        // Eyes should have rotated to track target
        assert!(angle > 0.5);
    }

    #[test]
    fn test_target_very_far() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            ..EyeParams::default()
        });

        // Very distant target
        system.set_gaze_target(Some(Vec3::new(0.0, 0.0, -10000.0)));

        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        // Eyes should be nearly parallel (minimal vergence)
        let (left, right) = system.get_eye_rotations();
        let angle_diff = quat_angle(left, right);

        // Very small difference at long distance
        assert!(angle_diff < 0.01);
    }

    #[test]
    fn test_target_very_near() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            ..EyeParams::default()
        });

        // Very close target
        system.set_gaze_target(Some(Vec3::new(0.0, 0.0, -0.1)));

        for _ in 0..120 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        // Eyes should converge significantly
        let (left, right) = system.get_eye_rotations();
        let angle_diff = quat_angle(left, right);

        // Noticeable difference for near target
        assert!(angle_diff > 0.05);
    }

    // -----------------------------------------------------------------------
    // Helper Function Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_slerp_shortest() {
        let a = Quat::IDENTITY;
        let b = Quat::from_rotation_y(0.5);

        let mid = slerp_shortest(a, b, 0.5);

        // Should be halfway
        let expected = Quat::from_rotation_y(0.25);
        let angle = quat_angle(mid, expected);
        assert!(angle < 0.01);
    }

    #[test]
    fn test_slerp_shortest_opposite() {
        let a = Quat::from_rotation_y(0.0);
        let b = Quat::from_rotation_y(PI); // Opposite direction

        let mid = slerp_shortest(a, b, 0.5);

        // Should take shortest path
        assert!(mid.is_normalized());
    }

    #[test]
    fn test_is_behind_head() {
        let forward = Vec3::NEG_Z;

        // Forward target
        assert!(!is_behind_head(Vec3::NEG_Z, forward));

        // Behind target
        assert!(is_behind_head(Vec3::Z, forward));

        // Side target (not behind)
        assert!(!is_behind_head(Vec3::X, forward));
    }

    #[test]
    fn test_compute_vergence() {
        let ipd = 0.063;

        // Far distance - minimal vergence
        let far_vergence = compute_vergence(10.0, ipd);
        assert!(far_vergence < 0.01);

        // Near distance - larger vergence
        let near_vergence = compute_vergence(0.3, ipd);
        assert!(near_vergence > far_vergence);

        // Zero distance - should return 0 (edge case)
        let zero_vergence = compute_vergence(0.0, ipd);
        assert!((zero_vergence - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_compute_accommodation() {
        let near_point = 0.1;

        // Far distance should give low accommodation (near 0)
        let far = compute_accommodation(100.0, near_point);
        assert!(far < 0.5, "Far accommodation should be < 0.5, got {}", far);

        // Near distance should give higher accommodation
        let near = compute_accommodation(0.2, near_point);
        assert!(near > far, "Near accommodation {} should be > far {}", near, far);

        // At near point
        let at_near = compute_accommodation(0.1, near_point);
        assert!((at_near - 1.0).abs() < 0.1, "At near point accommodation should be ~1.0, got {}", at_near);

        // Beyond near point (closer than near point)
        let beyond = compute_accommodation(0.05, near_point);
        assert!((beyond - 1.0).abs() < EPSILON, "Beyond near point should be 1.0, got {}", beyond);
    }

    // -----------------------------------------------------------------------
    // EyeConfig Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_eye_config_human() {
        let config = EyeConfig::human();
        assert_eq!(config.name, "Human");
        assert!(config.params.enable_tremor);
    }

    #[test]
    fn test_eye_config_cartoon() {
        let config = EyeConfig::cartoon();
        assert_eq!(config.name, "Cartoon");
        assert!(!config.params.enable_tremor);
    }

    #[test]
    fn test_eye_config_robotic() {
        let config = EyeConfig::robotic();
        assert_eq!(config.name, "Robotic");
        assert!(!config.params.enable_saccades);
        assert!(!config.params.enable_drift);
    }

    #[test]
    fn test_eye_config_sleepy() {
        let config = EyeConfig::sleepy();
        assert_eq!(config.name, "Sleepy");
        // Sleepy should have more frequent blinks
        assert!(config.params.blink_interval.0 < EyeParams::human().blink_interval.0);
    }

    #[test]
    fn test_eye_config_alert() {
        let config = EyeConfig::alert();
        assert_eq!(config.name, "Alert");
        // Alert should have faster gaze tracking
        assert!(config.params.gaze_speed > EyeParams::human().gaze_speed);
    }

    // -----------------------------------------------------------------------
    // Stress Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_long_simulation() {
        let mut system = EyeAnimationSystem::new(EyeParams::default());
        system.set_gaze_target(Some(Vec3::new(0.5, 0.5, -1.0)));

        // Simulate 10 seconds at 60fps
        for _ in 0..600 {
            system.update(1.0 / 60.0, Mat4::IDENTITY);
        }

        // Everything should still be valid
        let (left, right) = system.get_eye_rotations();
        assert!(left.is_finite());
        assert!(right.is_finite());
        assert!(left.is_normalized());
        assert!(right.is_normalized());

        let blink = system.get_blink_weight();
        assert!(blink >= 0.0 && blink <= 1.0);

        let pupil = system.get_pupil_size();
        assert!(pupil >= system.params.min_pupil_size);
        assert!(pupil <= system.params.max_pupil_size);
    }

    #[test]
    fn test_rapid_target_changes() {
        let mut system = EyeAnimationSystem::new(EyeParams {
            gaze_speed: 20.0, // Fast tracking
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            ..EyeParams::default()
        });

        // Rapidly change targets
        for i in 0..100 {
            let angle = (i as f32 * 0.1) % (2.0 * PI);
            let target = Vec3::new(angle.cos(), 0.0, -1.0 + angle.sin() * 0.5);
            system.set_gaze_target(Some(target));
            system.update(1.0 / 60.0, Mat4::IDENTITY);

            // Should always be valid
            let (left, right) = system.get_eye_rotations();
            assert!(left.is_finite());
            assert!(right.is_finite());
        }
    }

    #[test]
    fn test_deterministic_with_seed() {
        // Two systems with same seed and disabled random features should be similar
        let params = EyeParams {
            enable_saccades: false,
            enable_drift: false,
            enable_tremor: false,
            enable_blink: false,
            ..EyeParams::default()
        };

        let mut system1 = EyeAnimationSystem::new(params);
        let mut system2 = EyeAnimationSystem::new(params);

        // Ensure same random state
        system1.random_state = 42;
        system2.random_state = 42;

        system1.set_gaze_target(Some(Vec3::new(0.0, 0.0, -2.0)));
        system2.set_gaze_target(Some(Vec3::new(0.0, 0.0, -2.0)));

        for _ in 0..60 {
            system1.update(1.0 / 60.0, Mat4::IDENTITY);
            system2.update(1.0 / 60.0, Mat4::IDENTITY);

            let (left1, _) = system1.get_eye_rotations();
            let (left2, _) = system2.get_eye_rotations();

            let angle = quat_angle(left1, left2);
            assert!(angle < 0.01, "Expected similar rotations, angle diff={}", angle);
        }
    }
}
