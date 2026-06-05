//! Spring/Jiggle bones for physics-based secondary motion (T-AN-7.5).
//!
//! This module provides spring-based physics simulation for secondary motion effects
//! such as hair dynamics, cloth/cape motion, accessories (earrings, pendants), and tails.
//!
//! # Physics Model
//!
//! Uses a damped mass-spring system with Verlet integration for stability:
//! - Each bone has mass, position, and velocity state
//! - Spring force pulls bone toward rest position
//! - Damping prevents perpetual oscillation
//! - External forces: gravity, wind
//!
//! # Collision Support
//!
//! - Sphere colliders for head/body
//! - Capsule colliders for limbs
//! - Simple push-out resolution
//!
//! # Chain Support
//!
//! Multiple bones can form a chain (hair strand, cape edge) with:
//! - Parent-child constraint propagation
//! - Length preservation between bones
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::spring_bone::{
//!     SpringBoneChain, SpringBoneParams, SpringBoneState,
//!     SphereCollider, simulate_spring_chain,
//! };
//! use glam::{Vec3, Mat4};
//!
//! // Create a hair strand chain
//! let mut chain = SpringBoneChain::new(vec![0, 1, 2, 3]);
//!
//! // Configure physics parameters
//! chain.params[0] = SpringBoneParams {
//!     stiffness: 50.0,
//!     damping: 0.5,
//!     gravity_scale: 1.0,
//!     mass: 0.5,
//!     wind_influence: 0.3,
//!     collision_radius: 0.05,
//! };
//!
//! // Head collider to prevent hair clipping
//! let colliders = vec![
//!     SphereCollider { center: Vec3::new(0.0, 1.7, 0.0), radius: 0.12 },
//! ];
//!
//! // Simulate one frame
//! let positions = simulate_spring_chain(
//!     &mut chain,
//!     Mat4::IDENTITY,
//!     1.0 / 60.0, // 60 FPS
//!     Vec3::new(0.0, -9.81, 0.0),
//!     Vec3::new(1.0, 0.0, 0.5), // light breeze
//!     &colliders,
//!     &[],
//! );
//! ```

use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of bones in a single spring chain.
pub const MAX_CHAIN_LENGTH: usize = 64;

/// Minimum delta time to prevent division by zero.
pub const MIN_DELTA_TIME: f32 = 1e-6;

/// Maximum delta time to prevent physics explosion.
pub const MAX_DELTA_TIME: f32 = 1.0 / 30.0;

/// Number of constraint iterations for length preservation.
pub const CONSTRAINT_ITERATIONS: usize = 3;

/// Small epsilon for collision detection.
pub const COLLISION_EPSILON: f32 = 1e-4;

// ---------------------------------------------------------------------------
// SpringBoneParams
// ---------------------------------------------------------------------------

/// Physics parameters for a spring bone.
///
/// These parameters control how the bone responds to forces and constraints.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpringBoneParams {
    /// Spring constant (stiffness). Higher values = stiffer spring.
    /// Typical range: 0.1 - 100.0
    /// - Hair: 10-30
    /// - Cloth: 20-50
    /// - Accessories: 30-80
    pub stiffness: f32,

    /// Damping coefficient. Controls how quickly oscillations decay.
    /// Range: 0.0 - 1.0
    /// - 0.0 = no damping (perpetual oscillation)
    /// - 0.5 = underdamped (some bounce)
    /// - 1.0 = critically damped (no overshoot)
    pub damping: f32,

    /// Gravity influence scale.
    /// - 0.0 = no gravity effect
    /// - 1.0 = full gravity
    /// - Negative = float upward
    pub gravity_scale: f32,

    /// Bone mass in arbitrary units. Affects inertia.
    /// Typical range: 0.1 - 10.0
    /// Higher mass = slower to accelerate, more momentum
    pub mass: f32,

    /// Wind influence factor.
    /// Range: 0.0 - 1.0
    /// - 0.0 = no wind effect
    /// - 1.0 = full wind force applied
    pub wind_influence: f32,

    /// Collision sphere radius around bone tip.
    /// Used for collision detection with body colliders.
    pub collision_radius: f32,
}

impl Default for SpringBoneParams {
    fn default() -> Self {
        Self {
            stiffness: 20.0,
            damping: 0.5,
            gravity_scale: 1.0,
            mass: 1.0,
            wind_influence: 0.3,
            collision_radius: 0.02,
        }
    }
}

impl SpringBoneParams {
    /// Create parameters for stiff hair.
    pub fn hair_stiff() -> Self {
        Self {
            stiffness: 40.0,
            damping: 0.6,
            gravity_scale: 0.8,
            mass: 0.3,
            wind_influence: 0.2,
            collision_radius: 0.015,
        }
    }

    /// Create parameters for soft hair.
    pub fn hair_soft() -> Self {
        Self {
            stiffness: 15.0,
            damping: 0.4,
            gravity_scale: 1.0,
            mass: 0.5,
            wind_influence: 0.5,
            collision_radius: 0.02,
        }
    }

    /// Create parameters for cloth/cape.
    pub fn cloth() -> Self {
        Self {
            stiffness: 25.0,
            damping: 0.55,
            gravity_scale: 1.0,
            mass: 0.8,
            wind_influence: 0.7,
            collision_radius: 0.03,
        }
    }

    /// Create parameters for accessories (earrings, pendants).
    pub fn accessory() -> Self {
        Self {
            stiffness: 60.0,
            damping: 0.4,
            gravity_scale: 1.0,
            mass: 0.2,
            wind_influence: 0.15,
            collision_radius: 0.01,
        }
    }

    /// Create parameters for a tail.
    pub fn tail() -> Self {
        Self {
            stiffness: 30.0,
            damping: 0.5,
            gravity_scale: 0.6,
            mass: 1.5,
            wind_influence: 0.2,
            collision_radius: 0.04,
        }
    }

    /// Validate parameters are within reasonable bounds.
    pub fn validate(&self) -> Result<(), SpringBoneError> {
        if self.stiffness < 0.0 {
            return Err(SpringBoneError::InvalidParameter(
                "stiffness must be non-negative".to_string(),
            ));
        }
        if self.damping < 0.0 || self.damping > 1.0 {
            return Err(SpringBoneError::InvalidParameter(
                "damping must be in [0, 1]".to_string(),
            ));
        }
        if self.mass <= 0.0 {
            return Err(SpringBoneError::InvalidParameter(
                "mass must be positive".to_string(),
            ));
        }
        if self.wind_influence < 0.0 || self.wind_influence > 1.0 {
            return Err(SpringBoneError::InvalidParameter(
                "wind_influence must be in [0, 1]".to_string(),
            ));
        }
        if self.collision_radius < 0.0 {
            return Err(SpringBoneError::InvalidParameter(
                "collision_radius must be non-negative".to_string(),
            ));
        }
        Ok(())
    }

    /// Calculate critical damping coefficient for this spring.
    /// Critical damping = 2 * sqrt(k * m)
    #[inline]
    pub fn critical_damping(&self) -> f32 {
        2.0 * (self.stiffness * self.mass).sqrt()
    }

    /// Check if this spring is overdamped (no oscillation).
    #[inline]
    pub fn is_overdamped(&self) -> bool {
        let critical = self.critical_damping();
        self.damping * critical > critical
    }

    /// Check if this spring is underdamped (will oscillate).
    #[inline]
    pub fn is_underdamped(&self) -> bool {
        let critical = self.critical_damping();
        self.damping * critical < critical
    }
}

// ---------------------------------------------------------------------------
// SpringBoneState
// ---------------------------------------------------------------------------

/// Runtime state for a spring bone.
///
/// Uses Verlet integration which stores current and previous position
/// to derive velocity implicitly.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpringBoneState {
    /// Current world-space position of bone tip.
    pub position: Vec3,

    /// Previous frame's position (for Verlet integration).
    pub prev_position: Vec3,

    /// Explicit velocity (used for initialization and external forces).
    pub velocity: Vec3,
}

impl Default for SpringBoneState {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            prev_position: Vec3::ZERO,
            velocity: Vec3::ZERO,
        }
    }
}

impl SpringBoneState {
    /// Create a new state at the given position.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            prev_position: position,
            velocity: Vec3::ZERO,
        }
    }

    /// Create state with initial velocity.
    pub fn with_velocity(position: Vec3, velocity: Vec3) -> Self {
        Self {
            position,
            prev_position: position - velocity * (1.0 / 60.0), // Assume 60fps for prev
            velocity,
        }
    }

    /// Reset state to a position (zero velocity).
    pub fn reset(&mut self, position: Vec3) {
        self.position = position;
        self.prev_position = position;
        self.velocity = Vec3::ZERO;
    }

    /// Compute implicit velocity from Verlet positions.
    #[inline]
    pub fn verlet_velocity(&self, delta_time: f32) -> Vec3 {
        if delta_time > MIN_DELTA_TIME {
            (self.position - self.prev_position) / delta_time
        } else {
            Vec3::ZERO
        }
    }

    /// Get the displacement from rest position.
    #[inline]
    pub fn displacement(&self, rest_position: Vec3) -> Vec3 {
        self.position - rest_position
    }

    /// Get the current kinetic energy (0.5 * m * v^2).
    #[inline]
    pub fn kinetic_energy(&self, mass: f32, delta_time: f32) -> f32 {
        let v = self.verlet_velocity(delta_time);
        0.5 * mass * v.length_squared()
    }
}

// ---------------------------------------------------------------------------
// Colliders
// ---------------------------------------------------------------------------

/// Sphere collider for body collision (head, torso).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SphereCollider {
    /// Center of sphere in world space.
    pub center: Vec3,

    /// Radius of sphere.
    pub radius: f32,
}

impl SphereCollider {
    /// Create a new sphere collider.
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Test if a point is inside or touching this collider.
    #[inline]
    pub fn contains(&self, point: Vec3, margin: f32) -> bool {
        let dist_sq = (point - self.center).length_squared();
        let total_radius = self.radius + margin;
        dist_sq <= total_radius * total_radius
    }

    /// Get the closest point on the surface to a given point.
    #[inline]
    pub fn closest_surface_point(&self, point: Vec3) -> Vec3 {
        let dir = point - self.center;
        let len = dir.length();
        if len > COLLISION_EPSILON {
            self.center + dir * (self.radius / len)
        } else {
            // Point is at center, push in arbitrary direction
            self.center + Vec3::Y * self.radius
        }
    }

    /// Push a point outside the collider if it's inside.
    /// Returns the new position and whether collision occurred.
    pub fn push_out(&self, point: Vec3, margin: f32) -> (Vec3, bool) {
        let to_point = point - self.center;
        let dist = to_point.length();
        let min_dist = self.radius + margin;

        if dist < min_dist && dist > COLLISION_EPSILON {
            // Push to surface
            let normal = to_point / dist;
            (self.center + normal * min_dist, true)
        } else if dist <= COLLISION_EPSILON {
            // At center, push up
            (self.center + Vec3::Y * min_dist, true)
        } else {
            (point, false)
        }
    }
}

/// Capsule collider for limb collision.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapsuleCollider {
    /// Start point of capsule axis.
    pub start: Vec3,

    /// End point of capsule axis.
    pub end: Vec3,

    /// Radius of capsule.
    pub radius: f32,
}

impl CapsuleCollider {
    /// Create a new capsule collider.
    pub fn new(start: Vec3, end: Vec3, radius: f32) -> Self {
        Self { start, end, radius }
    }

    /// Get the closest point on the capsule axis to a given point.
    #[inline]
    pub fn closest_axis_point(&self, point: Vec3) -> Vec3 {
        let axis = self.end - self.start;
        let len_sq = axis.length_squared();

        if len_sq < COLLISION_EPSILON {
            // Degenerate capsule (sphere)
            return self.start;
        }

        // Project point onto axis
        let t = ((point - self.start).dot(axis) / len_sq).clamp(0.0, 1.0);
        self.start + axis * t
    }

    /// Get the closest point on the capsule surface to a given point.
    #[inline]
    pub fn closest_surface_point(&self, point: Vec3) -> Vec3 {
        let axis_point = self.closest_axis_point(point);
        let dir = point - axis_point;
        let len = dir.length();

        if len > COLLISION_EPSILON {
            axis_point + dir * (self.radius / len)
        } else {
            // Point is on axis, push perpendicular
            let axis = self.end - self.start;
            let perp = if axis.x.abs() < 0.9 {
                axis.cross(Vec3::X).normalize()
            } else {
                axis.cross(Vec3::Y).normalize()
            };
            axis_point + perp * self.radius
        }
    }

    /// Test if a point is inside or touching this collider.
    #[inline]
    pub fn contains(&self, point: Vec3, margin: f32) -> bool {
        let axis_point = self.closest_axis_point(point);
        let dist_sq = (point - axis_point).length_squared();
        let total_radius = self.radius + margin;
        dist_sq <= total_radius * total_radius
    }

    /// Push a point outside the collider if it's inside.
    /// Returns the new position and whether collision occurred.
    pub fn push_out(&self, point: Vec3, margin: f32) -> (Vec3, bool) {
        let axis_point = self.closest_axis_point(point);
        let to_point = point - axis_point;
        let dist = to_point.length();
        let min_dist = self.radius + margin;

        if dist < min_dist && dist > COLLISION_EPSILON {
            let normal = to_point / dist;
            (axis_point + normal * min_dist, true)
        } else if dist <= COLLISION_EPSILON {
            // On axis, push perpendicular
            let axis = self.end - self.start;
            let perp = if axis.x.abs() < 0.9 {
                axis.cross(Vec3::X).normalize()
            } else {
                axis.cross(Vec3::Y).normalize()
            };
            (axis_point + perp * min_dist, true)
        } else {
            (point, false)
        }
    }
}

// ---------------------------------------------------------------------------
// SpringBoneChain
// ---------------------------------------------------------------------------

/// A chain of spring bones for secondary motion.
///
/// Chains represent connected bones (hair strand, cape edge) that should
/// maintain distance constraints between adjacent bones.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpringBoneChain {
    /// Bone indices in the skeleton (root to tip order).
    pub bones: Vec<usize>,

    /// Physics parameters per bone.
    pub params: Vec<SpringBoneParams>,

    /// Runtime state per bone.
    pub states: Vec<SpringBoneState>,

    /// Rest lengths between consecutive bones.
    /// Length is `bones.len() - 1`.
    pub rest_lengths: Vec<f32>,

    /// Rest positions in local space relative to parent.
    pub rest_positions: Vec<Vec3>,

    /// Whether the chain is enabled for simulation.
    pub enabled: bool,
}

impl SpringBoneChain {
    /// Create a new spring bone chain from bone indices.
    pub fn new(bones: Vec<usize>) -> Self {
        let count = bones.len();
        Self {
            bones,
            params: vec![SpringBoneParams::default(); count],
            states: vec![SpringBoneState::default(); count],
            rest_lengths: vec![0.1; count.saturating_sub(1)],
            rest_positions: vec![Vec3::ZERO; count],
            enabled: true,
        }
    }

    /// Create a chain with custom parameters.
    pub fn with_params(bones: Vec<usize>, params: Vec<SpringBoneParams>) -> Self {
        let count = bones.len();
        assert_eq!(params.len(), count, "params count must match bones count");
        Self {
            bones,
            params,
            states: vec![SpringBoneState::default(); count],
            rest_lengths: vec![0.1; count.saturating_sub(1)],
            rest_positions: vec![Vec3::ZERO; count],
            enabled: true,
        }
    }

    /// Get the number of bones in this chain.
    #[inline]
    pub fn len(&self) -> usize {
        self.bones.len()
    }

    /// Check if the chain is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bones.is_empty()
    }

    /// Initialize rest positions and lengths from world positions.
    pub fn initialize_rest(&mut self, world_positions: &[Vec3]) {
        if world_positions.len() != self.bones.len() {
            return;
        }

        self.rest_positions = world_positions.to_vec();
        self.rest_lengths.clear();

        for i in 0..world_positions.len().saturating_sub(1) {
            let length = (world_positions[i + 1] - world_positions[i]).length();
            self.rest_lengths.push(length.max(0.001)); // Prevent zero length
        }

        // Initialize states to rest positions
        for (state, &pos) in self.states.iter_mut().zip(world_positions.iter()) {
            state.reset(pos);
        }
    }

    /// Reset all states to rest positions.
    pub fn reset_to_rest(&mut self) {
        for (state, &pos) in self.states.iter_mut().zip(self.rest_positions.iter()) {
            state.reset(pos);
        }
    }

    /// Validate the chain configuration.
    pub fn validate(&self) -> Result<(), SpringBoneError> {
        if self.bones.is_empty() {
            return Err(SpringBoneError::EmptyChain);
        }

        if self.bones.len() > MAX_CHAIN_LENGTH {
            return Err(SpringBoneError::ChainTooLong(self.bones.len()));
        }

        if self.params.len() != self.bones.len() {
            return Err(SpringBoneError::MismatchedArrays(
                "params".to_string(),
                self.params.len(),
                self.bones.len(),
            ));
        }

        if self.states.len() != self.bones.len() {
            return Err(SpringBoneError::MismatchedArrays(
                "states".to_string(),
                self.states.len(),
                self.bones.len(),
            ));
        }

        if self.bones.len() > 1 && self.rest_lengths.len() != self.bones.len() - 1 {
            return Err(SpringBoneError::MismatchedArrays(
                "rest_lengths".to_string(),
                self.rest_lengths.len(),
                self.bones.len() - 1,
            ));
        }

        for (i, params) in self.params.iter().enumerate() {
            params.validate().map_err(|e| {
                SpringBoneError::InvalidParameter(format!("bone {}: {:?}", i, e))
            })?;
        }

        Ok(())
    }

    /// Get total kinetic energy of the chain.
    pub fn total_kinetic_energy(&self, delta_time: f32) -> f32 {
        self.states
            .iter()
            .zip(self.params.iter())
            .map(|(state, params)| state.kinetic_energy(params.mass, delta_time))
            .sum()
    }

    /// Check if the chain has settled (low energy).
    pub fn is_settled(&self, delta_time: f32, threshold: f32) -> bool {
        self.total_kinetic_energy(delta_time) < threshold
    }
}

// ---------------------------------------------------------------------------
// Error Type
// ---------------------------------------------------------------------------

/// Errors that can occur in spring bone simulation.
#[derive(Clone, Debug, PartialEq)]
pub enum SpringBoneError {
    /// A parameter value is invalid.
    InvalidParameter(String),
    /// The chain has no bones.
    EmptyChain,
    /// The chain exceeds maximum length.
    ChainTooLong(usize),
    /// Array sizes don't match.
    MismatchedArrays(String, usize, usize),
}

impl std::fmt::Display for SpringBoneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpringBoneError::InvalidParameter(msg) => write!(f, "invalid parameter: {}", msg),
            SpringBoneError::EmptyChain => write!(f, "chain has no bones"),
            SpringBoneError::ChainTooLong(len) => {
                write!(f, "chain too long: {} > {}", len, MAX_CHAIN_LENGTH)
            }
            SpringBoneError::MismatchedArrays(name, got, expected) => {
                write!(f, "{} array size {} != expected {}", name, got, expected)
            }
        }
    }
}

impl std::error::Error for SpringBoneError {}

// ---------------------------------------------------------------------------
// Simulation Functions
// ---------------------------------------------------------------------------

/// Simulate a spring bone chain for one frame.
///
/// Uses Verlet integration for stability with constraint iterations
/// for length preservation.
///
/// # Arguments
///
/// * `chain` - The spring bone chain to simulate
/// * `parent_transform` - World transform of the chain's parent bone
/// * `delta_time` - Time step in seconds
/// * `gravity` - Gravity vector in world space
/// * `wind` - Wind force vector in world space
/// * `sphere_colliders` - Sphere colliders for collision
/// * `capsule_colliders` - Capsule colliders for collision
///
/// # Returns
///
/// Updated world positions for each bone in the chain.
pub fn simulate_spring_chain(
    chain: &mut SpringBoneChain,
    parent_transform: Mat4,
    delta_time: f32,
    gravity: Vec3,
    wind: Vec3,
    sphere_colliders: &[SphereCollider],
    capsule_colliders: &[CapsuleCollider],
) -> Vec<Vec3> {
    if chain.bones.is_empty() || !chain.enabled {
        return chain.states.iter().map(|s| s.position).collect();
    }

    // Clamp delta time to reasonable bounds
    let dt = delta_time.clamp(MIN_DELTA_TIME, MAX_DELTA_TIME);
    let dt_sq = dt * dt;

    // Phase 1: Apply forces and integrate (Verlet)
    // Process root bone first (anchored to parent)
    if !chain.states.is_empty() {
        let rest_local = chain.rest_positions.first().copied().unwrap_or(Vec3::ZERO);
        let rest_world = parent_transform.transform_point3(rest_local);
        chain.states[0].position = rest_world;
        chain.states[0].prev_position = rest_world;
    }

    // Process remaining bones with index-based iteration to avoid borrow issues
    for i in 1..chain.states.len() {
        // Get previous bone position first
        let prev_pos = chain.states[i - 1].position;
        let params = &chain.params[i];

        // Calculate rest position
        let rest_pos = {
            let rest_offset = chain.rest_positions.get(i).copied().unwrap_or(Vec3::ZERO);
            prev_pos + rest_offset
        };

        // Get current state data
        let state = &chain.states[i];
        let current_pos = state.position;
        let prev_position = state.prev_position;

        // Spring force: F = -k * displacement
        let displacement = current_pos - rest_pos;
        let spring_force = -params.stiffness * displacement;

        // Damping force: F = -c * velocity
        let velocity = if dt > MIN_DELTA_TIME {
            (current_pos - prev_position) / dt
        } else {
            Vec3::ZERO
        };
        let damping_force = -params.damping * params.critical_damping() * velocity;

        // Gravity force
        let gravity_force = gravity * params.gravity_scale * params.mass;

        // Wind force
        let wind_force = wind * params.wind_influence;

        // Total acceleration: F = ma => a = F/m
        let total_force = spring_force + damping_force + gravity_force + wind_force;
        let acceleration = total_force / params.mass;

        // Verlet integration: x_new = 2*x - x_old + a*dt^2
        let new_pos = current_pos * 2.0 - prev_position + acceleration * dt_sq;

        // Update state
        let state = &mut chain.states[i];
        state.prev_position = current_pos;
        state.position = new_pos;
        state.velocity = (new_pos - current_pos) / dt;
    }

    // Phase 2: Length constraint enforcement
    for _ in 0..CONSTRAINT_ITERATIONS {
        for i in 1..chain.states.len() {
            let rest_length = chain.rest_lengths.get(i - 1).copied().unwrap_or(0.1);

            // Get positions (need to do this carefully to avoid borrow issues)
            let parent_pos = chain.states[i - 1].position;
            let child_state = &mut chain.states[i];

            let delta = child_state.position - parent_pos;
            let current_length = delta.length();

            if current_length > COLLISION_EPSILON {
                // Push child to maintain rest length
                let correction = delta * (rest_length / current_length);
                child_state.position = parent_pos + correction;
            }
        }
    }

    // Phase 3: Collision resolution
    for (state, params) in chain.states.iter_mut().zip(chain.params.iter()) {
        let margin = params.collision_radius;

        // Sphere colliders
        for collider in sphere_colliders {
            let (new_pos, _) = collider.push_out(state.position, margin);
            state.position = new_pos;
        }

        // Capsule colliders
        for collider in capsule_colliders {
            let (new_pos, _) = collider.push_out(state.position, margin);
            state.position = new_pos;
        }
    }

    // Return final positions
    chain.states.iter().map(|s| s.position).collect()
}

/// Simulate multiple spring bone chains in batch.
///
/// This is more efficient than calling `simulate_spring_chain` repeatedly
/// as it allows for better cache utilization.
pub fn simulate_spring_chains_batch(
    chains: &mut [SpringBoneChain],
    parent_transforms: &[Mat4],
    delta_time: f32,
    gravity: Vec3,
    wind: Vec3,
    sphere_colliders: &[SphereCollider],
    capsule_colliders: &[CapsuleCollider],
) -> Vec<Vec<Vec3>> {
    chains
        .iter_mut()
        .zip(parent_transforms.iter())
        .map(|(chain, &transform)| {
            simulate_spring_chain(
                chain,
                transform,
                delta_time,
                gravity,
                wind,
                sphere_colliders,
                capsule_colliders,
            )
        })
        .collect()
}

/// Apply wind noise to base wind vector for more natural motion.
///
/// Uses a simple turbulence approximation based on time.
pub fn calculate_wind_with_turbulence(
    base_wind: Vec3,
    time: f32,
    turbulence_scale: f32,
    turbulence_frequency: f32,
) -> Vec3 {
    // Simple sinusoidal turbulence
    let noise_x = (time * turbulence_frequency).sin() * turbulence_scale;
    let noise_y = (time * turbulence_frequency * 1.3 + 1.0).sin() * turbulence_scale * 0.5;
    let noise_z = (time * turbulence_frequency * 0.7 + 2.0).sin() * turbulence_scale * 0.8;

    base_wind + Vec3::new(noise_x, noise_y, noise_z)
}

/// Compute the natural frequency of a spring bone (rad/s).
///
/// omega = sqrt(k/m)
#[inline]
pub fn natural_frequency(params: &SpringBoneParams) -> f32 {
    (params.stiffness / params.mass).sqrt()
}

/// Compute the damped frequency of a spring bone (rad/s).
///
/// omega_d = omega * sqrt(1 - zeta^2) for underdamped systems
#[inline]
pub fn damped_frequency(params: &SpringBoneParams) -> f32 {
    let omega = natural_frequency(params);
    let zeta = params.damping;
    if zeta < 1.0 {
        omega * (1.0 - zeta * zeta).sqrt()
    } else {
        0.0 // Critically damped or overdamped - no oscillation
    }
}

/// Estimate settling time for a spring bone (95% settled).
///
/// For underdamped: t = 3 / (zeta * omega)
/// For critically damped: t = 5 / omega
/// For overdamped: longer, depends on damping ratio
#[inline]
pub fn settling_time(params: &SpringBoneParams) -> f32 {
    let omega = natural_frequency(params);
    if omega < COLLISION_EPSILON {
        return f32::INFINITY;
    }

    let zeta = params.damping;
    if zeta < 1.0 {
        // Underdamped
        3.0 / (zeta * omega)
    } else if (zeta - 1.0).abs() < 0.01 {
        // Critically damped
        5.0 / omega
    } else {
        // Overdamped
        5.0 / (omega * (zeta - (zeta * zeta - 1.0).sqrt()))
    }
}

// ---------------------------------------------------------------------------
// SpringBoneSystem
// ---------------------------------------------------------------------------

/// System for managing multiple spring bone chains.
///
/// Provides a higher-level interface for animation integration.
#[derive(Clone, Debug, Default)]
pub struct SpringBoneSystem {
    /// All spring bone chains.
    pub chains: Vec<SpringBoneChain>,

    /// Global gravity vector.
    pub gravity: Vec3,

    /// Global wind vector (before turbulence).
    pub wind: Vec3,

    /// Wind turbulence parameters.
    pub wind_turbulence_scale: f32,
    pub wind_turbulence_frequency: f32,

    /// Sphere colliders.
    pub sphere_colliders: Vec<SphereCollider>,

    /// Capsule colliders.
    pub capsule_colliders: Vec<CapsuleCollider>,

    /// Accumulated time for wind turbulence.
    pub time: f32,

    /// Whether simulation is paused.
    pub paused: bool,
}

impl SpringBoneSystem {
    /// Create a new spring bone system with default settings.
    pub fn new() -> Self {
        Self {
            chains: Vec::new(),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            wind: Vec3::ZERO,
            wind_turbulence_scale: 0.5,
            wind_turbulence_frequency: 2.0,
            sphere_colliders: Vec::new(),
            capsule_colliders: Vec::new(),
            time: 0.0,
            paused: false,
        }
    }

    /// Add a spring bone chain.
    pub fn add_chain(&mut self, chain: SpringBoneChain) -> usize {
        let index = self.chains.len();
        self.chains.push(chain);
        index
    }

    /// Remove a chain by index.
    pub fn remove_chain(&mut self, index: usize) -> Option<SpringBoneChain> {
        if index < self.chains.len() {
            Some(self.chains.remove(index))
        } else {
            None
        }
    }

    /// Add a sphere collider.
    pub fn add_sphere_collider(&mut self, collider: SphereCollider) {
        self.sphere_colliders.push(collider);
    }

    /// Add a capsule collider.
    pub fn add_capsule_collider(&mut self, collider: CapsuleCollider) {
        self.capsule_colliders.push(collider);
    }

    /// Update all spring bone chains.
    ///
    /// # Arguments
    ///
    /// * `delta_time` - Time step in seconds
    /// * `parent_transforms` - World transforms for each chain's parent bone
    ///   (must match chains length)
    pub fn update(&mut self, delta_time: f32, parent_transforms: &[Mat4]) -> Vec<Vec<Vec3>> {
        if self.paused {
            return self
                .chains
                .iter()
                .map(|c| c.states.iter().map(|s| s.position).collect())
                .collect();
        }

        self.time += delta_time;

        // Calculate wind with turbulence
        let wind = calculate_wind_with_turbulence(
            self.wind,
            self.time,
            self.wind_turbulence_scale,
            self.wind_turbulence_frequency,
        );

        simulate_spring_chains_batch(
            &mut self.chains,
            parent_transforms,
            delta_time,
            self.gravity,
            wind,
            &self.sphere_colliders,
            &self.capsule_colliders,
        )
    }

    /// Reset all chains to their rest positions.
    pub fn reset(&mut self) {
        for chain in &mut self.chains {
            chain.reset_to_rest();
        }
        self.time = 0.0;
    }

    /// Set global gravity.
    pub fn set_gravity(&mut self, gravity: Vec3) {
        self.gravity = gravity;
    }

    /// Set global wind.
    pub fn set_wind(&mut self, wind: Vec3) {
        self.wind = wind;
    }

    /// Pause simulation.
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Resume simulation.
    pub fn resume(&mut self) {
        self.paused = false;
    }

    /// Get total number of simulated bones.
    pub fn total_bones(&self) -> usize {
        self.chains.iter().map(|c| c.len()).sum()
    }

    /// Check if all chains have settled.
    pub fn all_settled(&self, delta_time: f32, threshold: f32) -> bool {
        self.chains.iter().all(|c| c.is_settled(delta_time, threshold))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    const EPSILON: f32 = 1e-4;

    // -----------------------------------------------------------------------
    // SpringBoneParams Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_params_default() {
        let params = SpringBoneParams::default();
        assert!(params.validate().is_ok());
        assert_eq!(params.stiffness, 20.0);
        assert_eq!(params.damping, 0.5);
        assert_eq!(params.gravity_scale, 1.0);
        assert_eq!(params.mass, 1.0);
    }

    #[test]
    fn test_params_presets() {
        assert!(SpringBoneParams::hair_stiff().validate().is_ok());
        assert!(SpringBoneParams::hair_soft().validate().is_ok());
        assert!(SpringBoneParams::cloth().validate().is_ok());
        assert!(SpringBoneParams::accessory().validate().is_ok());
        assert!(SpringBoneParams::tail().validate().is_ok());
    }

    #[test]
    fn test_params_validation() {
        // Negative stiffness
        let mut params = SpringBoneParams::default();
        params.stiffness = -1.0;
        assert!(params.validate().is_err());

        // Invalid damping
        params = SpringBoneParams::default();
        params.damping = 1.5;
        assert!(params.validate().is_err());

        params.damping = -0.1;
        assert!(params.validate().is_err());

        // Zero mass
        params = SpringBoneParams::default();
        params.mass = 0.0;
        assert!(params.validate().is_err());

        // Negative mass
        params.mass = -1.0;
        assert!(params.validate().is_err());

        // Invalid wind influence
        params = SpringBoneParams::default();
        params.wind_influence = 1.5;
        assert!(params.validate().is_err());

        // Negative collision radius
        params = SpringBoneParams::default();
        params.collision_radius = -0.1;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_critical_damping() {
        let params = SpringBoneParams {
            stiffness: 100.0,
            mass: 1.0,
            damping: 0.5,
            ..Default::default()
        };

        // critical = 2 * sqrt(k * m) = 2 * sqrt(100 * 1) = 20
        assert!((params.critical_damping() - 20.0).abs() < EPSILON);
    }

    #[test]
    fn test_damping_classification() {
        // Underdamped
        let underdamped = SpringBoneParams {
            stiffness: 100.0,
            mass: 1.0,
            damping: 0.3,
            ..Default::default()
        };
        assert!(underdamped.is_underdamped());
        assert!(!underdamped.is_overdamped());

        // Overdamped (high damping coefficient relative to critical)
        // When damping * critical > critical, it's overdamped
        // This happens when damping > 1.0
        // But our damping is clamped to [0,1], so we need special handling
        // In practice, with damping = 1.0, we get critically damped
    }

    // -----------------------------------------------------------------------
    // SpringBoneState Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_new() {
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let state = SpringBoneState::new(pos);

        assert_eq!(state.position, pos);
        assert_eq!(state.prev_position, pos);
        assert_eq!(state.velocity, Vec3::ZERO);
    }

    #[test]
    fn test_state_with_velocity() {
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let vel = Vec3::new(0.0, -1.0, 0.0);
        let state = SpringBoneState::with_velocity(pos, vel);

        assert_eq!(state.position, pos);
        assert!((state.velocity - vel).length() < EPSILON);
    }

    #[test]
    fn test_state_reset() {
        let mut state = SpringBoneState::new(Vec3::new(1.0, 2.0, 3.0));
        state.position = Vec3::new(5.0, 6.0, 7.0);
        state.velocity = Vec3::new(1.0, 1.0, 1.0);

        let new_pos = Vec3::new(10.0, 10.0, 10.0);
        state.reset(new_pos);

        assert_eq!(state.position, new_pos);
        assert_eq!(state.prev_position, new_pos);
        assert_eq!(state.velocity, Vec3::ZERO);
    }

    #[test]
    fn test_verlet_velocity() {
        let mut state = SpringBoneState::new(Vec3::ZERO);
        state.prev_position = Vec3::ZERO;
        state.position = Vec3::new(1.0, 0.0, 0.0);

        let vel = state.verlet_velocity(1.0 / 60.0);
        assert!((vel.x - 60.0).abs() < EPSILON);
    }

    #[test]
    fn test_verlet_velocity_zero_dt() {
        let state = SpringBoneState::new(Vec3::ZERO);
        let vel = state.verlet_velocity(0.0);
        assert_eq!(vel, Vec3::ZERO);
    }

    #[test]
    fn test_kinetic_energy() {
        let mut state = SpringBoneState::new(Vec3::ZERO);
        state.prev_position = Vec3::ZERO;
        state.position = Vec3::new(1.0, 0.0, 0.0);

        // v = 60 m/s, m = 2 kg
        // KE = 0.5 * 2 * 60^2 = 3600
        let ke = state.kinetic_energy(2.0, 1.0 / 60.0);
        assert!((ke - 3600.0).abs() < 1.0);
    }

    // -----------------------------------------------------------------------
    // SphereCollider Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sphere_contains() {
        let sphere = SphereCollider::new(Vec3::ZERO, 1.0);

        // Inside
        assert!(sphere.contains(Vec3::new(0.5, 0.0, 0.0), 0.0));

        // On surface
        assert!(sphere.contains(Vec3::new(1.0, 0.0, 0.0), 0.0));

        // Outside
        assert!(!sphere.contains(Vec3::new(1.5, 0.0, 0.0), 0.0));

        // Outside but within margin
        assert!(sphere.contains(Vec3::new(1.1, 0.0, 0.0), 0.2));
    }

    #[test]
    fn test_sphere_closest_surface() {
        let sphere = SphereCollider::new(Vec3::ZERO, 1.0);

        let closest = sphere.closest_surface_point(Vec3::new(2.0, 0.0, 0.0));
        assert!((closest - Vec3::new(1.0, 0.0, 0.0)).length() < EPSILON);

        let closest = sphere.closest_surface_point(Vec3::new(0.0, 3.0, 0.0));
        assert!((closest - Vec3::new(0.0, 1.0, 0.0)).length() < EPSILON);
    }

    #[test]
    fn test_sphere_push_out() {
        let sphere = SphereCollider::new(Vec3::ZERO, 1.0);

        // Point inside
        let (pos, collided) = sphere.push_out(Vec3::new(0.5, 0.0, 0.0), 0.0);
        assert!(collided);
        assert!((pos.length() - 1.0).abs() < EPSILON);

        // Point outside
        let (pos, collided) = sphere.push_out(Vec3::new(2.0, 0.0, 0.0), 0.0);
        assert!(!collided);
        assert!((pos - Vec3::new(2.0, 0.0, 0.0)).length() < EPSILON);

        // Point at center
        let (pos, collided) = sphere.push_out(Vec3::ZERO, 0.0);
        assert!(collided);
        assert!((pos.length() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_sphere_push_out_with_margin() {
        let sphere = SphereCollider::new(Vec3::ZERO, 1.0);

        // Point just outside sphere but within margin
        let (pos, collided) = sphere.push_out(Vec3::new(1.05, 0.0, 0.0), 0.1);
        assert!(collided);
        assert!((pos.length() - 1.1).abs() < EPSILON);
    }

    // -----------------------------------------------------------------------
    // CapsuleCollider Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_capsule_closest_axis() {
        let capsule = CapsuleCollider::new(Vec3::ZERO, Vec3::new(0.0, 2.0, 0.0), 0.5);

        // Point along axis
        let closest = capsule.closest_axis_point(Vec3::new(0.0, 1.0, 0.0));
        assert!((closest - Vec3::new(0.0, 1.0, 0.0)).length() < EPSILON);

        // Point off to side
        let closest = capsule.closest_axis_point(Vec3::new(1.0, 1.0, 0.0));
        assert!((closest - Vec3::new(0.0, 1.0, 0.0)).length() < EPSILON);

        // Point past end
        let closest = capsule.closest_axis_point(Vec3::new(0.0, 5.0, 0.0));
        assert!((closest - Vec3::new(0.0, 2.0, 0.0)).length() < EPSILON);

        // Point before start
        let closest = capsule.closest_axis_point(Vec3::new(0.0, -1.0, 0.0));
        assert!((closest - Vec3::ZERO).length() < EPSILON);
    }

    #[test]
    fn test_capsule_contains() {
        let capsule = CapsuleCollider::new(Vec3::ZERO, Vec3::new(0.0, 2.0, 0.0), 0.5);

        // Inside
        assert!(capsule.contains(Vec3::new(0.2, 1.0, 0.0), 0.0));

        // On surface
        assert!(capsule.contains(Vec3::new(0.5, 1.0, 0.0), 0.0));

        // Outside
        assert!(!capsule.contains(Vec3::new(1.0, 1.0, 0.0), 0.0));

        // In end cap region
        assert!(capsule.contains(Vec3::new(0.0, 2.3, 0.0), 0.0));
    }

    #[test]
    fn test_capsule_push_out() {
        let capsule = CapsuleCollider::new(Vec3::ZERO, Vec3::new(0.0, 2.0, 0.0), 0.5);

        // Point inside
        let (pos, collided) = capsule.push_out(Vec3::new(0.2, 1.0, 0.0), 0.0);
        assert!(collided);

        // Check it's on the surface
        let dist = (pos - Vec3::new(0.0, 1.0, 0.0)).length();
        assert!((dist - 0.5).abs() < EPSILON);

        // Point outside
        let (pos, collided) = capsule.push_out(Vec3::new(2.0, 1.0, 0.0), 0.0);
        assert!(!collided);
        assert!((pos - Vec3::new(2.0, 1.0, 0.0)).length() < EPSILON);
    }

    #[test]
    fn test_capsule_degenerate() {
        // Zero-length capsule (sphere)
        let capsule = CapsuleCollider::new(Vec3::ZERO, Vec3::ZERO, 0.5);

        let closest = capsule.closest_axis_point(Vec3::new(1.0, 0.0, 0.0));
        assert!((closest - Vec3::ZERO).length() < EPSILON);
    }

    // -----------------------------------------------------------------------
    // SpringBoneChain Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chain_new() {
        let chain = SpringBoneChain::new(vec![0, 1, 2, 3]);

        assert_eq!(chain.len(), 4);
        assert!(!chain.is_empty());
        assert_eq!(chain.params.len(), 4);
        assert_eq!(chain.states.len(), 4);
        assert_eq!(chain.rest_lengths.len(), 3);
    }

    #[test]
    fn test_chain_with_params() {
        let params = vec![
            SpringBoneParams::hair_stiff(),
            SpringBoneParams::hair_soft(),
            SpringBoneParams::hair_soft(),
        ];
        let chain = SpringBoneChain::with_params(vec![0, 1, 2], params.clone());

        assert_eq!(chain.params.len(), 3);
        assert_eq!(chain.params[0], params[0]);
    }

    #[test]
    fn test_chain_initialize_rest() {
        let mut chain = SpringBoneChain::new(vec![0, 1, 2]);
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, -0.1, 0.0),
            Vec3::new(0.0, -0.2, 0.0),
        ];

        chain.initialize_rest(&positions);

        assert_eq!(chain.rest_positions.len(), 3);
        assert_eq!(chain.rest_lengths.len(), 2);
        assert!((chain.rest_lengths[0] - 0.1).abs() < EPSILON);
        assert!((chain.rest_lengths[1] - 0.1).abs() < EPSILON);
    }

    #[test]
    fn test_chain_reset_to_rest() {
        let mut chain = SpringBoneChain::new(vec![0, 1, 2]);
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, -0.1, 0.0),
            Vec3::new(0.0, -0.2, 0.0),
        ];
        chain.initialize_rest(&positions);

        // Modify states
        chain.states[1].position = Vec3::new(1.0, 1.0, 1.0);

        chain.reset_to_rest();

        assert_eq!(chain.states[1].position, positions[1]);
    }

    #[test]
    fn test_chain_validation() {
        // Valid chain
        let chain = SpringBoneChain::new(vec![0, 1, 2]);
        assert!(chain.validate().is_ok());

        // Empty chain
        let empty = SpringBoneChain::new(vec![]);
        assert!(matches!(empty.validate(), Err(SpringBoneError::EmptyChain)));

        // Chain too long
        let long_bones: Vec<usize> = (0..MAX_CHAIN_LENGTH + 1).collect();
        let long = SpringBoneChain::new(long_bones);
        assert!(matches!(long.validate(), Err(SpringBoneError::ChainTooLong(_))));

        // Mismatched params
        let mut mismatch = SpringBoneChain::new(vec![0, 1, 2]);
        mismatch.params.pop();
        assert!(matches!(
            mismatch.validate(),
            Err(SpringBoneError::MismatchedArrays(_, _, _))
        ));
    }

    #[test]
    fn test_chain_kinetic_energy() {
        let mut chain = SpringBoneChain::new(vec![0, 1]);
        chain.states[0].position = Vec3::ZERO;
        chain.states[0].prev_position = Vec3::ZERO;
        chain.states[1].position = Vec3::new(1.0, 0.0, 0.0);
        chain.states[1].prev_position = Vec3::ZERO;

        let ke = chain.total_kinetic_energy(1.0 / 60.0);
        assert!(ke > 0.0);
    }

    #[test]
    fn test_chain_is_settled() {
        let chain = SpringBoneChain::new(vec![0, 1]);
        // Default state has zero velocity, so should be settled
        assert!(chain.is_settled(1.0 / 60.0, 0.001));
    }

    // -----------------------------------------------------------------------
    // Simulation Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simulate_empty_chain() {
        let mut chain = SpringBoneChain::new(vec![]);
        let positions = simulate_spring_chain(
            &mut chain,
            Mat4::IDENTITY,
            1.0 / 60.0,
            Vec3::new(0.0, -9.81, 0.0),
            Vec3::ZERO,
            &[],
            &[],
        );

        assert!(positions.is_empty());
    }

    #[test]
    fn test_simulate_disabled_chain() {
        let mut chain = SpringBoneChain::new(vec![0, 1, 2]);
        chain.enabled = false;

        let initial: Vec<Vec3> = chain.states.iter().map(|s| s.position).collect();
        let positions = simulate_spring_chain(
            &mut chain,
            Mat4::IDENTITY,
            1.0 / 60.0,
            Vec3::new(0.0, -9.81, 0.0),
            Vec3::ZERO,
            &[],
            &[],
        );

        assert_eq!(positions, initial);
    }

    #[test]
    fn test_simulate_gravity_response() {
        let mut chain = SpringBoneChain::new(vec![0, 1]);
        let init_pos = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.9, 0.0),
        ];
        chain.initialize_rest(&init_pos);

        // Simulate several frames with gravity
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        for _ in 0..10 {
            simulate_spring_chain(
                &mut chain,
                Mat4::IDENTITY,
                1.0 / 60.0,
                gravity,
                Vec3::ZERO,
                &[],
                &[],
            );
        }

        // Bone should have moved down due to gravity
        // (but spring pulls it back, so it won't go forever)
        // Just verify it's affected
        assert!(chain.states[1].position.y < init_pos[1].y);
    }

    #[test]
    fn test_simulate_wind_influence() {
        let mut chain = SpringBoneChain::new(vec![0, 1]);
        let init_pos = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.9, 0.0),
        ];
        chain.initialize_rest(&init_pos);
        chain.params[1].wind_influence = 1.0;

        // Simulate with wind
        let wind = Vec3::new(1.0, 0.0, 0.0);
        for _ in 0..30 {
            simulate_spring_chain(
                &mut chain,
                Mat4::IDENTITY,
                1.0 / 60.0,
                Vec3::ZERO, // No gravity
                wind,
                &[],
                &[],
            );
        }

        // Bone should have moved in wind direction
        assert!(chain.states[1].position.x > init_pos[1].x);
    }

    #[test]
    fn test_simulate_length_preservation() {
        let mut chain = SpringBoneChain::new(vec![0, 1, 2]);
        let init_pos = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.9, 0.0),
            Vec3::new(0.0, 0.8, 0.0),
        ];
        chain.initialize_rest(&init_pos);

        let expected_length_01 = 0.1;
        let expected_length_12 = 0.1;

        // Simulate with strong forces
        for _ in 0..100 {
            simulate_spring_chain(
                &mut chain,
                Mat4::IDENTITY,
                1.0 / 60.0,
                Vec3::new(0.0, -9.81, 0.0),
                Vec3::new(0.5, 0.0, 0.2),
                &[],
                &[],
            );
        }

        // Check length constraints are approximately preserved
        let len_01 = (chain.states[1].position - chain.states[0].position).length();
        let len_12 = (chain.states[2].position - chain.states[1].position).length();

        assert!((len_01 - expected_length_01).abs() < 0.02);
        assert!((len_12 - expected_length_12).abs() < 0.02);
    }

    #[test]
    fn test_simulate_collision_sphere() {
        let mut chain = SpringBoneChain::new(vec![0, 1]);
        let init_pos = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0), // Inside collider
        ];
        chain.initialize_rest(&init_pos);
        chain.params[1].collision_radius = 0.05;

        let collider = SphereCollider::new(Vec3::new(0.0, 0.5, 0.0), 0.3);

        simulate_spring_chain(
            &mut chain,
            Mat4::IDENTITY,
            1.0 / 60.0,
            Vec3::ZERO,
            Vec3::ZERO,
            &[collider],
            &[],
        );

        // Bone should be pushed outside collider
        let dist = (chain.states[1].position - collider.center).length();
        assert!(dist >= collider.radius + chain.params[1].collision_radius - EPSILON);
    }

    #[test]
    fn test_simulate_collision_capsule() {
        let mut chain = SpringBoneChain::new(vec![0, 1]);
        let init_pos = vec![
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0), // Near capsule
        ];
        chain.initialize_rest(&init_pos);
        chain.params[1].collision_radius = 0.05;

        let collider = CapsuleCollider::new(
            Vec3::new(-0.5, 1.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
            0.2,
        );

        // Move bone inside capsule
        chain.states[1].position = Vec3::new(0.0, 1.0, 0.0);
        chain.states[1].prev_position = chain.states[1].position;

        simulate_spring_chain(
            &mut chain,
            Mat4::IDENTITY,
            1.0 / 60.0,
            Vec3::ZERO,
            Vec3::ZERO,
            &[],
            &[collider],
        );

        // Bone should be pushed outside capsule
        let axis_point = collider.closest_axis_point(chain.states[1].position);
        let dist = (chain.states[1].position - axis_point).length();
        assert!(dist >= collider.radius + chain.params[1].collision_radius - EPSILON);
    }

    #[test]
    fn test_simulate_multiple_colliders() {
        let mut chain = SpringBoneChain::new(vec![0, 1, 2]);
        let init_pos = vec![
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 1.5, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        chain.initialize_rest(&init_pos);

        let sphere1 = SphereCollider::new(Vec3::new(0.0, 1.5, 0.0), 0.2);
        let sphere2 = SphereCollider::new(Vec3::new(0.0, 1.0, 0.0), 0.2);
        let capsule = CapsuleCollider::new(
            Vec3::new(-1.0, 1.25, 0.0),
            Vec3::new(1.0, 1.25, 0.0),
            0.1,
        );

        simulate_spring_chain(
            &mut chain,
            Mat4::IDENTITY,
            1.0 / 60.0,
            Vec3::ZERO,
            Vec3::ZERO,
            &[sphere1, sphere2],
            &[capsule],
        );

        // Just verify it doesn't crash with multiple colliders
        assert!(chain.states[1].position.is_finite());
        assert!(chain.states[2].position.is_finite());
    }

    #[test]
    fn test_simulate_zero_dt() {
        let mut chain = SpringBoneChain::new(vec![0, 1]);
        let init_pos = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.9, 0.0),
        ];
        chain.initialize_rest(&init_pos);

        let before = chain.states[1].position;

        simulate_spring_chain(
            &mut chain,
            Mat4::IDENTITY,
            0.0,
            Vec3::new(0.0, -9.81, 0.0),
            Vec3::ZERO,
            &[],
            &[],
        );

        // With zero dt (clamped to MIN_DELTA_TIME), state should change minimally
        let diff = (chain.states[1].position - before).length();
        assert!(diff < 0.001);
    }

    #[test]
    fn test_simulate_extreme_dt() {
        let mut chain = SpringBoneChain::new(vec![0, 1]);
        let init_pos = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.9, 0.0),
        ];
        chain.initialize_rest(&init_pos);

        // Very large dt should be clamped
        simulate_spring_chain(
            &mut chain,
            Mat4::IDENTITY,
            10.0, // Very large
            Vec3::new(0.0, -9.81, 0.0),
            Vec3::ZERO,
            &[],
            &[],
        );

        // Should not explode
        assert!(chain.states[1].position.is_finite());
    }

    #[test]
    fn test_simulate_extreme_params() {
        let mut chain = SpringBoneChain::new(vec![0, 1]);
        let init_pos = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.9, 0.0),
        ];
        chain.initialize_rest(&init_pos);

        // Extreme stiffness
        chain.params[1].stiffness = 1000.0;
        chain.params[1].damping = 0.99;

        for _ in 0..60 {
            simulate_spring_chain(
                &mut chain,
                Mat4::IDENTITY,
                1.0 / 60.0,
                Vec3::new(0.0, -9.81, 0.0),
                Vec3::ZERO,
                &[],
                &[],
            );
        }

        assert!(chain.states[1].position.is_finite());
    }

    // -----------------------------------------------------------------------
    // Spring Physics Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_spring_oscillation() {
        // Test that underdamped spring oscillates
        let mut chain = SpringBoneChain::new(vec![0, 1]);
        let init_pos = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.9, 0.0),
        ];
        chain.initialize_rest(&init_pos);
        chain.params[1].stiffness = 50.0;
        chain.params[1].damping = 0.1; // Underdamped

        // Perturb the system
        chain.states[1].position = Vec3::new(0.0, 0.7, 0.0);

        let mut positions_y: Vec<f32> = Vec::new();

        for _ in 0..120 {
            simulate_spring_chain(
                &mut chain,
                Mat4::IDENTITY,
                1.0 / 60.0,
                Vec3::ZERO, // No gravity for clean oscillation
                Vec3::ZERO,
                &[],
                &[],
            );
            positions_y.push(chain.states[1].position.y);
        }

        // Check for oscillation: there should be direction changes
        let mut direction_changes = 0;
        for i in 2..positions_y.len() {
            let prev_delta = positions_y[i - 1] - positions_y[i - 2];
            let curr_delta = positions_y[i] - positions_y[i - 1];
            if prev_delta * curr_delta < 0.0 {
                direction_changes += 1;
            }
        }

        assert!(direction_changes >= 2, "Underdamped spring should oscillate");
    }

    #[test]
    fn test_spring_settling() {
        // Test that spring settles to rest
        let mut chain = SpringBoneChain::new(vec![0, 1]);
        let init_pos = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.9, 0.0),
        ];
        chain.initialize_rest(&init_pos);
        chain.params[1].stiffness = 50.0;
        chain.params[1].damping = 0.8; // Higher damping

        // Perturb the system
        chain.states[1].position = Vec3::new(0.0, 0.5, 0.0);

        // Simulate until settled
        for _ in 0..300 {
            simulate_spring_chain(
                &mut chain,
                Mat4::IDENTITY,
                1.0 / 60.0,
                Vec3::ZERO,
                Vec3::ZERO,
                &[],
                &[],
            );
        }

        // Should be close to rest position
        let rest_y = init_pos[1].y;
        assert!(
            (chain.states[1].position.y - rest_y).abs() < 0.05,
            "Spring should settle near rest position"
        );
    }

    #[test]
    fn test_damping_effect() {
        // Compare settling between different damping values
        let test_damping = |damping: f32| -> f32 {
            let mut chain = SpringBoneChain::new(vec![0, 1]);
            let init_pos = vec![
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.9, 0.0),
            ];
            chain.initialize_rest(&init_pos);
            chain.params[1].stiffness = 50.0;
            chain.params[1].damping = damping;
            chain.states[1].position = Vec3::new(0.0, 0.5, 0.0);

            for _ in 0..100 {
                simulate_spring_chain(
                    &mut chain,
                    Mat4::IDENTITY,
                    1.0 / 60.0,
                    Vec3::ZERO,
                    Vec3::ZERO,
                    &[],
                    &[],
                );
            }

            chain.total_kinetic_energy(1.0 / 60.0)
        };

        let low_damping_energy = test_damping(0.1);
        let high_damping_energy = test_damping(0.9);

        // Higher damping should result in less energy (faster settling)
        assert!(
            high_damping_energy < low_damping_energy,
            "Higher damping should dissipate energy faster"
        );
    }

    // -----------------------------------------------------------------------
    // Batch Simulation Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_simulation() {
        let mut chains = vec![
            SpringBoneChain::new(vec![0, 1]),
            SpringBoneChain::new(vec![2, 3]),
        ];

        chains[0].initialize_rest(&[Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.9, 0.0)]);
        chains[1].initialize_rest(&[Vec3::new(1.0, 1.0, 0.0), Vec3::new(1.0, 0.9, 0.0)]);

        let transforms = vec![Mat4::IDENTITY, Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))];

        let results = simulate_spring_chains_batch(
            &mut chains,
            &transforms,
            1.0 / 60.0,
            Vec3::new(0.0, -9.81, 0.0),
            Vec3::ZERO,
            &[],
            &[],
        );

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 2);
    }

    // -----------------------------------------------------------------------
    // Wind Turbulence Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_wind_turbulence() {
        let base_wind = Vec3::new(1.0, 0.0, 0.0);

        let wind0 = calculate_wind_with_turbulence(base_wind, 0.0, 0.5, 2.0);
        let wind1 = calculate_wind_with_turbulence(base_wind, 0.5, 0.5, 2.0);
        let wind2 = calculate_wind_with_turbulence(base_wind, 1.0, 0.5, 2.0);

        // Winds should be different at different times
        assert!((wind0 - wind1).length() > 0.01);
        assert!((wind1 - wind2).length() > 0.01);

        // But all should be close to base wind
        assert!((wind0 - base_wind).length() < 2.0);
    }

    #[test]
    fn test_wind_zero_turbulence() {
        let base_wind = Vec3::new(1.0, 0.5, 0.25);
        let wind = calculate_wind_with_turbulence(base_wind, 0.0, 0.0, 2.0);

        // With zero turbulence scale, should equal base wind (at t=0)
        // Actually sin(0) = 0, so it will equal base wind
        assert!((wind - base_wind).length() < EPSILON);
    }

    // -----------------------------------------------------------------------
    // Frequency Analysis Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_natural_frequency() {
        let params = SpringBoneParams {
            stiffness: 100.0,
            mass: 1.0,
            ..Default::default()
        };

        // omega = sqrt(k/m) = sqrt(100/1) = 10 rad/s
        let omega = natural_frequency(&params);
        assert!((omega - 10.0).abs() < EPSILON);
    }

    #[test]
    fn test_damped_frequency() {
        let params = SpringBoneParams {
            stiffness: 100.0,
            mass: 1.0,
            damping: 0.5,
            ..Default::default()
        };

        // omega_d = omega * sqrt(1 - zeta^2) = 10 * sqrt(0.75)
        let omega_d = damped_frequency(&params);
        let expected = 10.0 * (0.75_f32).sqrt();
        assert!((omega_d - expected).abs() < EPSILON);
    }

    #[test]
    fn test_damped_frequency_critically_damped() {
        let params = SpringBoneParams {
            stiffness: 100.0,
            mass: 1.0,
            damping: 1.0, // Critically damped
            ..Default::default()
        };

        let omega_d = damped_frequency(&params);
        assert_eq!(omega_d, 0.0, "Critically damped should have zero damped frequency");
    }

    #[test]
    fn test_settling_time_underdamped() {
        let params = SpringBoneParams {
            stiffness: 100.0,
            mass: 1.0,
            damping: 0.5,
            ..Default::default()
        };

        let t = settling_time(&params);
        // t = 3 / (zeta * omega) = 3 / (0.5 * 10) = 0.6s
        let expected = 3.0 / (0.5 * 10.0);
        assert!((t - expected).abs() < EPSILON);
    }

    // -----------------------------------------------------------------------
    // SpringBoneSystem Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_system_new() {
        let system = SpringBoneSystem::new();

        assert!(system.chains.is_empty());
        assert_eq!(system.gravity, Vec3::new(0.0, -9.81, 0.0));
        assert_eq!(system.wind, Vec3::ZERO);
        assert!(!system.paused);
    }

    #[test]
    fn test_system_add_chain() {
        let mut system = SpringBoneSystem::new();
        let chain = SpringBoneChain::new(vec![0, 1, 2]);

        let idx = system.add_chain(chain);
        assert_eq!(idx, 0);
        assert_eq!(system.chains.len(), 1);

        let idx2 = system.add_chain(SpringBoneChain::new(vec![3, 4]));
        assert_eq!(idx2, 1);
        assert_eq!(system.chains.len(), 2);
    }

    #[test]
    fn test_system_remove_chain() {
        let mut system = SpringBoneSystem::new();
        system.add_chain(SpringBoneChain::new(vec![0, 1]));
        system.add_chain(SpringBoneChain::new(vec![2, 3]));

        let removed = system.remove_chain(0);
        assert!(removed.is_some());
        assert_eq!(system.chains.len(), 1);

        let removed = system.remove_chain(10);
        assert!(removed.is_none());
    }

    #[test]
    fn test_system_colliders() {
        let mut system = SpringBoneSystem::new();

        system.add_sphere_collider(SphereCollider::new(Vec3::ZERO, 0.5));
        system.add_capsule_collider(CapsuleCollider::new(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            0.2,
        ));

        assert_eq!(system.sphere_colliders.len(), 1);
        assert_eq!(system.capsule_colliders.len(), 1);
    }

    #[test]
    fn test_system_update() {
        let mut system = SpringBoneSystem::new();

        let mut chain = SpringBoneChain::new(vec![0, 1]);
        chain.initialize_rest(&[Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.9, 0.0)]);
        system.add_chain(chain);

        let transforms = vec![Mat4::IDENTITY];
        let results = system.update(1.0 / 60.0, &transforms);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 2);
    }

    #[test]
    fn test_system_pause_resume() {
        let mut system = SpringBoneSystem::new();

        let mut chain = SpringBoneChain::new(vec![0, 1]);
        chain.initialize_rest(&[Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.9, 0.0)]);
        system.add_chain(chain);

        system.pause();
        assert!(system.paused);

        let before = system.chains[0].states[1].position;
        system.update(1.0 / 60.0, &[Mat4::IDENTITY]);
        let after = system.chains[0].states[1].position;

        // Should not change when paused
        assert_eq!(before, after);

        system.resume();
        assert!(!system.paused);
    }

    #[test]
    fn test_system_reset() {
        let mut system = SpringBoneSystem::new();

        let mut chain = SpringBoneChain::new(vec![0, 1]);
        chain.initialize_rest(&[Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.9, 0.0)]);
        system.add_chain(chain);

        // Modify state
        system.chains[0].states[1].position = Vec3::new(5.0, 5.0, 5.0);
        system.time = 100.0;

        system.reset();

        assert_eq!(system.time, 0.0);
        assert_eq!(
            system.chains[0].states[1].position,
            Vec3::new(0.0, 0.9, 0.0)
        );
    }

    #[test]
    fn test_system_total_bones() {
        let mut system = SpringBoneSystem::new();
        system.add_chain(SpringBoneChain::new(vec![0, 1, 2]));
        system.add_chain(SpringBoneChain::new(vec![3, 4]));

        assert_eq!(system.total_bones(), 5);
    }

    #[test]
    fn test_system_all_settled() {
        let system = SpringBoneSystem::new();
        // Empty system is settled
        assert!(system.all_settled(1.0 / 60.0, 0.001));
    }

    // -----------------------------------------------------------------------
    // Error Type Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let e1 = SpringBoneError::InvalidParameter("test".to_string());
        assert!(e1.to_string().contains("invalid parameter"));

        let e2 = SpringBoneError::EmptyChain;
        assert!(e2.to_string().contains("no bones"));

        let e3 = SpringBoneError::ChainTooLong(100);
        assert!(e3.to_string().contains("too long"));

        let e4 = SpringBoneError::MismatchedArrays("params".to_string(), 5, 3);
        assert!(e4.to_string().contains("params"));
    }

    // -----------------------------------------------------------------------
    // Edge Case Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_bone_chain() {
        let mut chain = SpringBoneChain::new(vec![0]);
        chain.initialize_rest(&[Vec3::new(0.0, 1.0, 0.0)]);

        assert!(chain.validate().is_ok());
        assert!(chain.rest_lengths.is_empty());

        let positions = simulate_spring_chain(
            &mut chain,
            Mat4::IDENTITY,
            1.0 / 60.0,
            Vec3::new(0.0, -9.81, 0.0),
            Vec3::ZERO,
            &[],
            &[],
        );

        assert_eq!(positions.len(), 1);
    }

    #[test]
    fn test_parent_transform_effect() {
        let mut chain = SpringBoneChain::new(vec![0, 1]);
        chain.initialize_rest(&[Vec3::ZERO, Vec3::new(0.0, -0.1, 0.0)]);

        // Parent moved
        let parent_transform = Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0));

        let positions = simulate_spring_chain(
            &mut chain,
            parent_transform,
            1.0 / 60.0,
            Vec3::ZERO,
            Vec3::ZERO,
            &[],
            &[],
        );

        // Root bone should follow parent
        assert!((positions[0].x - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_degenerate_chain_zero_rest_lengths() {
        let mut chain = SpringBoneChain::new(vec![0, 1, 2]);
        // All at same position (degenerate)
        chain.initialize_rest(&[Vec3::ZERO, Vec3::ZERO, Vec3::ZERO]);

        // Rest lengths should be clamped to minimum
        assert!(chain.rest_lengths[0] >= 0.001);
        assert!(chain.rest_lengths[1] >= 0.001);

        // Should not crash
        simulate_spring_chain(
            &mut chain,
            Mat4::IDENTITY,
            1.0 / 60.0,
            Vec3::new(0.0, -9.81, 0.0),
            Vec3::ZERO,
            &[],
            &[],
        );
    }

    #[test]
    fn test_nan_protection() {
        let mut chain = SpringBoneChain::new(vec![0, 1]);
        chain.initialize_rest(&[Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.9, 0.0)]);

        // Extreme values that could cause numerical issues
        chain.params[1].stiffness = 10000.0;
        chain.params[1].mass = 0.001;

        // Simulate many frames
        for _ in 0..1000 {
            simulate_spring_chain(
                &mut chain,
                Mat4::IDENTITY,
                1.0 / 60.0,
                Vec3::new(0.0, -9.81, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                &[],
                &[],
            );
        }

        // Should remain finite
        assert!(chain.states[0].position.is_finite());
        assert!(chain.states[1].position.is_finite());
    }
}
