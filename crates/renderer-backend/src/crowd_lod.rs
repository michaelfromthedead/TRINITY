//! Crowd LOD System (T-AN-8.5)
//!
//! This module provides a comprehensive Level of Detail (LOD) system for crowd rendering.
//! It manages smooth transitions between different detail levels, budget-based quality
//! distribution, and priority-based LOD selection for important characters.
//!
//! # Architecture
//!
//! ```text
//! CrowdInstance                    LodSelector
//!       |                               |
//!       v                               v
//! +----------------+     +--------------------------------+
//! | Position, Rot  | --> | Distance calculation           |
//! | Scale, Flags   |     | Screen-size LOD option         |
//! +----------------+     | Priority-based selection       |
//!                        +--------------------------------+
//!                                       |
//!                       +---------------+---------------+
//!                       v                               v
//!                  LodLevel                    LodTransitionManager
//!                  (LOD0-LOD3)                 (crossfade/dither)
//!                       |                               |
//!                       v                               v
//!                  LodBudgetManager  <------->  LodMeshSet
//!                  (polygon budget)             (multi-LOD meshes)
//!                       |
//!                       v
//!                  CrowdLodSystem
//!                  (integration)
//! ```
//!
//! # LOD Levels
//!
//! - **LOD0**: Full mesh with all bones, full skinning (high detail)
//! - **LOD1**: Simplified mesh with reduced bones (medium detail)
//! - **LOD2**: Billboard impostor (low detail)
//! - **LOD3**: Point sprite (extreme distance, minimal cost)
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::crowd_lod::{
//!     CrowdLodSystem, LodConfig, LodLevel, LodSelector, LodBudgetManager,
//! };
//!
//! // Configure LOD system
//! let config = LodConfig::new()
//!     .with_distance_thresholds([20.0, 50.0, 100.0, 200.0])
//!     .with_hysteresis(2.0)
//!     .with_screen_size_lod(true);
//!
//! // Create LOD system
//! let mut system = CrowdLodSystem::new(config);
//!
//! // Add character type with LOD meshes
//! let mesh_set = LodMeshSet::new("soldier")
//!     .with_lod(LodLevel::Lod0, mesh_full, bone_count_full)
//!     .with_lod(LodLevel::Lod1, mesh_simplified, bone_count_reduced);
//! system.register_mesh_set(0, mesh_set);
//!
//! // Per-frame update
//! system.update(camera_pos, camera_forward, fov, instances.as_mut_slice(), delta_time);
//!
//! // Get statistics
//! let stats = system.stats();
//! println!("LOD0: {}, LOD1: {}, LOD2: {}, LOD3: {}",
//!     stats.lod_counts[0], stats.lod_counts[1], stats.lod_counts[2], stats.lod_counts[3]);
//! ```

use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of LOD levels supported.
pub const LOD_LEVEL_COUNT: usize = 4;

/// Default distance threshold for LOD0 -> LOD1 transition.
pub const DEFAULT_LOD0_DISTANCE: f32 = 20.0;

/// Default distance threshold for LOD1 -> LOD2 transition.
pub const DEFAULT_LOD1_DISTANCE: f32 = 50.0;

/// Default distance threshold for LOD2 -> LOD3 transition.
pub const DEFAULT_LOD2_DISTANCE: f32 = 100.0;

/// Default distance threshold for LOD3 -> culled transition.
pub const DEFAULT_LOD3_DISTANCE: f32 = 200.0;

/// Default hysteresis margin (prevents oscillation).
pub const DEFAULT_HYSTERESIS: f32 = 2.0;

/// Default transition duration in seconds.
pub const DEFAULT_TRANSITION_DURATION: f32 = 0.25;

/// Default polygon budget.
pub const DEFAULT_POLYGON_BUDGET: u32 = 500_000;

/// Priority flag: Hero character (never goes below LOD0).
pub const PRIORITY_HERO: u8 = 0x80;

/// Priority flag: Important character (never goes below LOD1).
pub const PRIORITY_IMPORTANT: u8 = 0x40;

/// Priority flag: Screen-center bonus.
pub const PRIORITY_SCREEN_CENTER: u8 = 0x20;

/// Transition flag: Active transition.
pub const TRANSITION_ACTIVE: u8 = 0x01;

/// Transition flag: Use dithering.
pub const TRANSITION_DITHER: u8 = 0x02;

/// Maximum instances per update batch.
pub const MAX_BATCH_SIZE: usize = 4096;

// ---------------------------------------------------------------------------
// Vec3 / Vec4 helper types
// ---------------------------------------------------------------------------

/// 3D vector type for LOD system.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    /// Create a new Vec3.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Zero vector.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// Calculate distance to another point.
    #[inline]
    pub fn distance(self, other: Self) -> f32 {
        self.distance_squared(other).sqrt()
    }

    /// Calculate squared distance to another point.
    #[inline]
    pub fn distance_squared(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Subtract two vectors.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    /// Normalize the vector.
    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 0.0001 {
            Self::new(self.x / len, self.y / len, self.z / len)
        } else {
            Self::ZERO
        }
    }

    /// Length of the vector.
    #[inline]
    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Check if all components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Convert to array.
    #[inline]
    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Create from array.
    #[inline]
    pub fn from_array(arr: [f32; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(arr: [f32; 3]) -> Self {
        Self::from_array(arr)
    }
}

impl From<Vec3> for [f32; 3] {
    fn from(v: Vec3) -> Self {
        v.to_array()
    }
}

// ---------------------------------------------------------------------------
// LodLevel
// ---------------------------------------------------------------------------

/// Level of detail enumeration.
///
/// Each level represents a different quality/performance tradeoff:
/// - **LOD0**: Full mesh, full bone skinning
/// - **LOD1**: Simplified mesh, reduced bones
/// - **LOD2**: Billboard impostor
/// - **LOD3**: Point sprite (extreme distance)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum LodLevel {
    /// Full mesh with all bones and full skinning.
    #[default]
    Lod0 = 0,

    /// Simplified mesh with reduced bone count.
    Lod1 = 1,

    /// Billboard impostor (single quad with pre-rendered texture).
    Lod2 = 2,

    /// Point sprite (extreme distance, minimal cost).
    Lod3 = 3,
}

impl LodLevel {
    /// Get name for this LOD level.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Lod0 => "LOD0 (Full)",
            Self::Lod1 => "LOD1 (Simplified)",
            Self::Lod2 => "LOD2 (Impostor)",
            Self::Lod3 => "LOD3 (Point)",
        }
    }

    /// Get short name for this LOD level.
    pub const fn short_name(&self) -> &'static str {
        match self {
            Self::Lod0 => "LOD0",
            Self::Lod1 => "LOD1",
            Self::Lod2 => "LOD2",
            Self::Lod3 => "LOD3",
        }
    }

    /// Parse from u8 value.
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Lod0),
            1 => Some(Self::Lod1),
            2 => Some(Self::Lod2),
            3 => Some(Self::Lod3),
            _ => None,
        }
    }

    /// Convert to u8 value.
    #[inline]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Get all LOD levels.
    pub const fn all() -> [Self; LOD_LEVEL_COUNT] {
        [Self::Lod0, Self::Lod1, Self::Lod2, Self::Lod3]
    }

    /// Check if this level uses skeletal animation.
    #[inline]
    pub const fn uses_skeleton(&self) -> bool {
        matches!(self, Self::Lod0 | Self::Lod1)
    }

    /// Check if this level uses billboard rendering.
    #[inline]
    pub const fn is_billboard(&self) -> bool {
        matches!(self, Self::Lod2)
    }

    /// Check if this level uses point sprite rendering.
    #[inline]
    pub const fn is_point_sprite(&self) -> bool {
        matches!(self, Self::Lod3)
    }

    /// Get the next lower detail level (if any).
    #[inline]
    pub const fn lower(&self) -> Option<Self> {
        match self {
            Self::Lod0 => Some(Self::Lod1),
            Self::Lod1 => Some(Self::Lod2),
            Self::Lod2 => Some(Self::Lod3),
            Self::Lod3 => None,
        }
    }

    /// Get the next higher detail level (if any).
    #[inline]
    pub const fn higher(&self) -> Option<Self> {
        match self {
            Self::Lod0 => None,
            Self::Lod1 => Some(Self::Lod0),
            Self::Lod2 => Some(Self::Lod1),
            Self::Lod3 => Some(Self::Lod2),
        }
    }
}

// ---------------------------------------------------------------------------
// LodError
// ---------------------------------------------------------------------------

/// Errors that can occur during LOD operations.
#[derive(Clone, Debug, PartialEq)]
pub enum LodError {
    /// Invalid configuration parameter.
    InvalidConfig { reason: &'static str },
    /// Distance thresholds not monotonically increasing.
    InvalidDistanceThresholds,
    /// Mesh set not found.
    MeshSetNotFound { id: u32 },
    /// LOD level not available in mesh set.
    LodLevelNotAvailable { level: LodLevel },
    /// Budget exceeded and cannot reduce quality further.
    BudgetExceeded { budget: u32, required: u32 },
    /// Invalid instance index.
    InvalidInstanceIndex { index: usize },
    /// Transition already in progress.
    TransitionInProgress { instance_id: u32 },
}

impl std::fmt::Display for LodError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig { reason } => write!(f, "invalid LOD config: {}", reason),
            Self::InvalidDistanceThresholds => {
                write!(f, "distance thresholds must be monotonically increasing")
            }
            Self::MeshSetNotFound { id } => write!(f, "mesh set {} not found", id),
            Self::LodLevelNotAvailable { level } => {
                write!(f, "LOD level {:?} not available in mesh set", level)
            }
            Self::BudgetExceeded { budget, required } => {
                write!(f, "polygon budget {} exceeded, need {}", budget, required)
            }
            Self::InvalidInstanceIndex { index } => {
                write!(f, "invalid instance index {}", index)
            }
            Self::TransitionInProgress { instance_id } => {
                write!(f, "transition already in progress for instance {}", instance_id)
            }
        }
    }
}

impl std::error::Error for LodError {}

// ---------------------------------------------------------------------------
// LodConfig
// ---------------------------------------------------------------------------

/// Configuration for the LOD system.
#[derive(Clone, Debug, PartialEq)]
pub struct LodConfig {
    /// Distance thresholds for each LOD level transition.
    /// Index i is the distance at which LODi transitions to LOD(i+1).
    pub distance_thresholds: [f32; LOD_LEVEL_COUNT],

    /// Hysteresis margin to prevent oscillation near threshold boundaries.
    pub hysteresis: f32,

    /// Enable screen-size based LOD selection.
    pub screen_size_lod: bool,

    /// Minimum screen-space size (pixels) for LOD0.
    pub screen_size_thresholds: [f32; LOD_LEVEL_COUNT],

    /// Default transition duration in seconds.
    pub transition_duration: f32,

    /// Use dithered transitions instead of alpha blend.
    pub use_dithering: bool,

    /// Total polygon budget for all crowd instances.
    pub polygon_budget: u32,

    /// Enable adaptive quality based on frame time.
    pub adaptive_quality: bool,

    /// Target frame time for adaptive quality (seconds).
    pub target_frame_time: f32,

    /// Enable priority-based LOD (hero characters stay high LOD).
    pub enable_priority: bool,

    /// Per-character type distance multipliers.
    pub type_distance_multipliers: HashMap<u32, f32>,
}

impl LodConfig {
    /// Create a new default configuration.
    pub fn new() -> Self {
        Self {
            distance_thresholds: [
                DEFAULT_LOD0_DISTANCE,
                DEFAULT_LOD1_DISTANCE,
                DEFAULT_LOD2_DISTANCE,
                DEFAULT_LOD3_DISTANCE,
            ],
            hysteresis: DEFAULT_HYSTERESIS,
            screen_size_lod: false,
            screen_size_thresholds: [100.0, 50.0, 25.0, 10.0],
            transition_duration: DEFAULT_TRANSITION_DURATION,
            use_dithering: true,
            polygon_budget: DEFAULT_POLYGON_BUDGET,
            adaptive_quality: false,
            target_frame_time: 1.0 / 60.0,
            enable_priority: true,
            type_distance_multipliers: HashMap::new(),
        }
    }

    /// Set distance thresholds for LOD transitions.
    pub fn with_distance_thresholds(mut self, thresholds: [f32; LOD_LEVEL_COUNT]) -> Self {
        self.distance_thresholds = thresholds;
        self
    }

    /// Set hysteresis margin.
    pub fn with_hysteresis(mut self, hysteresis: f32) -> Self {
        self.hysteresis = hysteresis.max(0.0);
        self
    }

    /// Enable or disable screen-size based LOD.
    pub fn with_screen_size_lod(mut self, enable: bool) -> Self {
        self.screen_size_lod = enable;
        self
    }

    /// Set screen-size thresholds.
    pub fn with_screen_size_thresholds(mut self, thresholds: [f32; LOD_LEVEL_COUNT]) -> Self {
        self.screen_size_thresholds = thresholds;
        self
    }

    /// Set transition duration.
    pub fn with_transition_duration(mut self, duration: f32) -> Self {
        self.transition_duration = duration.max(0.0);
        self
    }

    /// Enable or disable dithered transitions.
    pub fn with_dithering(mut self, enable: bool) -> Self {
        self.use_dithering = enable;
        self
    }

    /// Set polygon budget.
    pub fn with_polygon_budget(mut self, budget: u32) -> Self {
        self.polygon_budget = budget;
        self
    }

    /// Enable or disable adaptive quality.
    pub fn with_adaptive_quality(mut self, enable: bool) -> Self {
        self.adaptive_quality = enable;
        self
    }

    /// Set target frame time for adaptive quality.
    pub fn with_target_frame_time(mut self, time: f32) -> Self {
        self.target_frame_time = time.max(0.001);
        self
    }

    /// Enable or disable priority-based LOD.
    pub fn with_priority(mut self, enable: bool) -> Self {
        self.enable_priority = enable;
        self
    }

    /// Add a distance multiplier for a character type.
    pub fn with_type_distance_multiplier(mut self, type_id: u32, multiplier: f32) -> Self {
        self.type_distance_multipliers.insert(type_id, multiplier);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), LodError> {
        // Check that distance thresholds are monotonically increasing
        for i in 1..LOD_LEVEL_COUNT {
            if self.distance_thresholds[i] <= self.distance_thresholds[i - 1] {
                return Err(LodError::InvalidDistanceThresholds);
            }
        }

        // Check that thresholds are positive
        for threshold in &self.distance_thresholds {
            if *threshold <= 0.0 || !threshold.is_finite() {
                return Err(LodError::InvalidConfig {
                    reason: "distance thresholds must be positive and finite",
                });
            }
        }

        // Check hysteresis
        if !self.hysteresis.is_finite() {
            return Err(LodError::InvalidConfig {
                reason: "hysteresis must be finite",
            });
        }

        Ok(())
    }

    /// Get distance threshold for a specific LOD level with type multiplier.
    pub fn get_threshold(&self, level: LodLevel, type_id: Option<u32>) -> f32 {
        let base = self.distance_thresholds[level.to_u8() as usize];
        let multiplier = type_id
            .and_then(|id| self.type_distance_multipliers.get(&id).copied())
            .unwrap_or(1.0);
        base * multiplier
    }

    /// Configuration optimized for large crowds (10k+ instances).
    pub fn for_large_crowd() -> Self {
        Self::new()
            .with_distance_thresholds([15.0, 35.0, 70.0, 150.0])
            .with_polygon_budget(1_000_000)
            .with_adaptive_quality(true)
            .with_dithering(true)
    }

    /// Configuration optimized for small crowds (<1k instances).
    pub fn for_small_crowd() -> Self {
        Self::new()
            .with_distance_thresholds([30.0, 60.0, 120.0, 250.0])
            .with_polygon_budget(250_000)
            .with_adaptive_quality(false)
    }

    /// Configuration for cinematic quality (fewer but higher detail).
    pub fn for_cinematic() -> Self {
        Self::new()
            .with_distance_thresholds([50.0, 100.0, 200.0, 400.0])
            .with_polygon_budget(2_000_000)
            .with_transition_duration(0.5)
    }
}

impl Default for LodConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LodSelector
// ---------------------------------------------------------------------------

/// Selects the appropriate LOD level for each instance.
#[derive(Clone, Debug)]
pub struct LodSelector {
    /// Configuration.
    config: LodConfig,

    /// Camera position.
    camera_pos: Vec3,

    /// Camera forward direction (for screen-size LOD).
    camera_forward: Vec3,

    /// Camera field of view (radians).
    camera_fov: f32,

    /// Screen height in pixels.
    screen_height: f32,

    /// Previous frame's LOD levels (for hysteresis).
    previous_lods: Vec<LodLevel>,
}

impl LodSelector {
    /// Create a new LOD selector.
    pub fn new(config: LodConfig) -> Self {
        Self {
            config,
            camera_pos: Vec3::ZERO,
            camera_forward: Vec3::new(0.0, 0.0, -1.0),
            camera_fov: std::f32::consts::FRAC_PI_2,
            screen_height: 1080.0,
            previous_lods: Vec::new(),
        }
    }

    /// Update camera parameters.
    pub fn update_camera(
        &mut self,
        position: Vec3,
        forward: Vec3,
        fov: f32,
        screen_height: f32,
    ) {
        self.camera_pos = position;
        self.camera_forward = forward.normalize();
        self.camera_fov = fov;
        self.screen_height = screen_height;
    }

    /// Calculate LOD level for a single instance based on distance.
    #[inline]
    pub fn select_lod_distance(
        &self,
        position: Vec3,
        scale: f32,
        type_id: Option<u32>,
        priority: u8,
        previous_lod: Option<LodLevel>,
    ) -> LodLevel {
        let distance = self.camera_pos.distance(position);

        // Apply priority constraints
        let min_lod = if priority & PRIORITY_HERO != 0 {
            LodLevel::Lod0
        } else if priority & PRIORITY_IMPORTANT != 0 {
            LodLevel::Lod1
        } else {
            LodLevel::Lod3
        };

        // Calculate base LOD from distance
        let base_lod = self.distance_to_lod(distance, scale, type_id);

        // Apply hysteresis if we have previous LOD
        let lod_with_hysteresis = if let Some(prev) = previous_lod {
            self.apply_hysteresis(distance, scale, type_id, base_lod, prev)
        } else {
            base_lod
        };

        // Clamp to minimum LOD based on priority
        if lod_with_hysteresis > min_lod {
            min_lod
        } else {
            lod_with_hysteresis
        }
    }

    /// Calculate LOD level from distance.
    fn distance_to_lod(&self, distance: f32, scale: f32, type_id: Option<u32>) -> LodLevel {
        // Adjust distance by scale (larger characters stay higher LOD longer)
        let effective_distance = distance / scale.max(0.1);

        for (i, _) in self.config.distance_thresholds.iter().enumerate() {
            let adjusted_threshold = self.config.get_threshold(
                LodLevel::from_u8(i as u8).unwrap_or(LodLevel::Lod0),
                type_id,
            );
            if effective_distance < adjusted_threshold {
                return LodLevel::from_u8(i as u8).unwrap_or(LodLevel::Lod3);
            }
        }

        LodLevel::Lod3
    }

    /// Apply hysteresis to prevent oscillation.
    fn apply_hysteresis(
        &self,
        distance: f32,
        scale: f32,
        type_id: Option<u32>,
        new_lod: LodLevel,
        previous_lod: LodLevel,
    ) -> LodLevel {
        if new_lod == previous_lod {
            return new_lod;
        }

        let effective_distance = distance / scale.max(0.1);
        let hysteresis = self.config.hysteresis;

        // When transitioning to lower detail, require distance > threshold + hysteresis
        if new_lod > previous_lod {
            let threshold = self.config.get_threshold(previous_lod, type_id);
            if effective_distance < threshold + hysteresis {
                return previous_lod;
            }
        }

        // When transitioning to higher detail, require distance < threshold - hysteresis
        if new_lod < previous_lod {
            let threshold = self.config.get_threshold(new_lod, type_id);
            if effective_distance > threshold - hysteresis {
                return previous_lod;
            }
        }

        new_lod
    }

    /// Calculate LOD level based on screen-space size.
    pub fn select_lod_screen_size(
        &self,
        position: Vec3,
        bounding_radius: f32,
    ) -> LodLevel {
        let screen_size = self.calculate_screen_size(position, bounding_radius);

        for (i, &threshold) in self.config.screen_size_thresholds.iter().enumerate() {
            if screen_size >= threshold {
                return LodLevel::from_u8(i as u8).unwrap_or(LodLevel::Lod3);
            }
        }

        LodLevel::Lod3
    }

    /// Calculate screen-space size in pixels.
    fn calculate_screen_size(&self, position: Vec3, bounding_radius: f32) -> f32 {
        let distance = self.camera_pos.distance(position);
        if distance < 0.001 {
            return self.screen_height;
        }

        // Project sphere to screen space
        let projected_size = bounding_radius / (distance * (self.camera_fov * 0.5).tan());
        projected_size * self.screen_height * 0.5
    }

    /// Batch update LOD levels for multiple instances.
    pub fn batch_select(
        &mut self,
        positions: &[Vec3],
        scales: &[f32],
        priorities: &[u8],
        type_ids: &[Option<u32>],
    ) -> Vec<LodLevel> {
        let count = positions.len();

        // Ensure previous_lods is sized correctly
        if self.previous_lods.len() != count {
            self.previous_lods.resize(count, LodLevel::Lod0);
        }

        let mut results = Vec::with_capacity(count);

        for i in 0..count {
            let lod = if self.config.screen_size_lod {
                let distance_lod = self.select_lod_distance(
                    positions[i],
                    scales[i],
                    type_ids.get(i).copied().flatten(),
                    priorities[i],
                    Some(self.previous_lods[i]),
                );
                let screen_lod = self.select_lod_screen_size(positions[i], scales[i]);

                // Use higher detail of the two
                if distance_lod < screen_lod {
                    distance_lod
                } else {
                    screen_lod
                }
            } else {
                self.select_lod_distance(
                    positions[i],
                    scales[i],
                    type_ids.get(i).copied().flatten(),
                    priorities[i],
                    Some(self.previous_lods[i]),
                )
            };

            results.push(lod);
            self.previous_lods[i] = lod;
        }

        results
    }

    /// Clear previous LOD history.
    pub fn clear_history(&mut self) {
        self.previous_lods.clear();
    }

    /// Get current configuration.
    pub fn config(&self) -> &LodConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: LodConfig) {
        self.config = config;
    }
}

// ---------------------------------------------------------------------------
// LodTransition
// ---------------------------------------------------------------------------

/// State of an active LOD transition for a single instance.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LodTransition {
    /// Source LOD level.
    pub from_lod: LodLevel,

    /// Target LOD level.
    pub to_lod: LodLevel,

    /// Progress (0.0 = start, 1.0 = complete).
    pub progress: f32,

    /// Total duration of the transition.
    pub duration: f32,

    /// Transition flags.
    pub flags: u8,
}

impl LodTransition {
    /// Create a new transition.
    pub fn new(from: LodLevel, to: LodLevel, duration: f32, dithered: bool) -> Self {
        let mut flags = TRANSITION_ACTIVE;
        if dithered {
            flags |= TRANSITION_DITHER;
        }

        Self {
            from_lod: from,
            to_lod: to,
            progress: 0.0,
            duration,
            flags,
        }
    }

    /// Check if the transition is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        (self.flags & TRANSITION_ACTIVE) != 0
    }

    /// Check if the transition uses dithering.
    #[inline]
    pub fn is_dithered(&self) -> bool {
        (self.flags & TRANSITION_DITHER) != 0
    }

    /// Check if the transition is complete.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.progress >= 1.0
    }

    /// Advance the transition by delta time.
    pub fn advance(&mut self, delta_time: f32) {
        if !self.is_active() {
            return;
        }

        if self.duration > 0.0 {
            self.progress += delta_time / self.duration;
        } else {
            self.progress = 1.0;
        }

        if self.progress >= 1.0 {
            self.progress = 1.0;
            self.flags &= !TRANSITION_ACTIVE;
        }
    }

    /// Get the blend factor for crossfade (0 = from, 1 = to).
    #[inline]
    pub fn blend_factor(&self) -> f32 {
        self.progress.clamp(0.0, 1.0)
    }

    /// Get the dither threshold for the given screen position.
    #[inline]
    pub fn dither_threshold(&self, screen_x: f32, screen_y: f32) -> f32 {
        // 4x4 Bayer dithering matrix
        const BAYER_4X4: [[f32; 4]; 4] = [
            [0.0, 8.0, 2.0, 10.0],
            [12.0, 4.0, 14.0, 6.0],
            [3.0, 11.0, 1.0, 9.0],
            [15.0, 7.0, 13.0, 5.0],
        ];

        let x = (screen_x as usize) % 4;
        let y = (screen_y as usize) % 4;

        BAYER_4X4[y][x] / 16.0
    }

    /// Check if the instance should be rendered in the "from" LOD based on dithering.
    pub fn should_render_from(&self, screen_x: f32, screen_y: f32) -> bool {
        if !self.is_dithered() {
            return true;
        }
        self.dither_threshold(screen_x, screen_y) >= self.progress
    }

    /// Check if the instance should be rendered in the "to" LOD based on dithering.
    pub fn should_render_to(&self, screen_x: f32, screen_y: f32) -> bool {
        if !self.is_dithered() {
            return true;
        }
        self.dither_threshold(screen_x, screen_y) < self.progress
    }
}

// ---------------------------------------------------------------------------
// LodTransitionManager
// ---------------------------------------------------------------------------

/// Manages LOD transitions for all instances.
#[derive(Clone, Debug)]
pub struct LodTransitionManager {
    /// Active transitions indexed by instance ID.
    transitions: HashMap<u32, LodTransition>,

    /// Default transition duration.
    default_duration: f32,

    /// Use dithering by default.
    use_dithering: bool,

    /// Maximum concurrent transitions.
    max_transitions: usize,
}

impl LodTransitionManager {
    /// Create a new transition manager.
    pub fn new(default_duration: f32, use_dithering: bool) -> Self {
        Self {
            transitions: HashMap::new(),
            default_duration,
            use_dithering,
            max_transitions: 10000,
        }
    }

    /// Start a new transition for an instance.
    pub fn start_transition(
        &mut self,
        instance_id: u32,
        from: LodLevel,
        to: LodLevel,
    ) -> Result<(), LodError> {
        // Don't start transition if already transitioning
        if self.transitions.contains_key(&instance_id) {
            return Err(LodError::TransitionInProgress { instance_id });
        }

        // Don't start transition to same LOD
        if from == to {
            return Ok(());
        }

        // Respect max transitions limit
        if self.transitions.len() >= self.max_transitions {
            // Remove oldest completed transitions
            self.cleanup_completed();
        }

        let transition = LodTransition::new(from, to, self.default_duration, self.use_dithering);
        self.transitions.insert(instance_id, transition);

        Ok(())
    }

    /// Start or update transition (more lenient version).
    pub fn start_or_update_transition(
        &mut self,
        instance_id: u32,
        from: LodLevel,
        to: LodLevel,
    ) {
        if from == to {
            // Remove any existing transition
            self.transitions.remove(&instance_id);
            return;
        }

        if let Some(existing) = self.transitions.get_mut(&instance_id) {
            // Update target if different
            if existing.to_lod != to {
                existing.to_lod = to;
                // Optionally adjust progress
            }
        } else {
            let transition = LodTransition::new(from, to, self.default_duration, self.use_dithering);
            self.transitions.insert(instance_id, transition);
        }
    }

    /// Update all active transitions.
    pub fn update(&mut self, delta_time: f32) {
        for transition in self.transitions.values_mut() {
            transition.advance(delta_time);
        }

        // Remove completed transitions
        self.cleanup_completed();
    }

    /// Remove completed transitions.
    fn cleanup_completed(&mut self) {
        self.transitions.retain(|_, t| !t.is_complete());
    }

    /// Get transition for an instance.
    pub fn get_transition(&self, instance_id: u32) -> Option<&LodTransition> {
        self.transitions.get(&instance_id)
    }

    /// Check if an instance has an active transition.
    pub fn is_transitioning(&self, instance_id: u32) -> bool {
        self.transitions
            .get(&instance_id)
            .map(|t| t.is_active())
            .unwrap_or(false)
    }

    /// Get the current effective LOD for an instance.
    pub fn effective_lod(&self, instance_id: u32, base_lod: LodLevel) -> LodLevel {
        if let Some(transition) = self.transitions.get(&instance_id) {
            if transition.is_active() && transition.progress < 0.5 {
                transition.from_lod
            } else {
                transition.to_lod
            }
        } else {
            base_lod
        }
    }

    /// Get active transition count.
    pub fn active_count(&self) -> usize {
        self.transitions.values().filter(|t| t.is_active()).count()
    }

    /// Clear all transitions.
    pub fn clear(&mut self) {
        self.transitions.clear();
    }

    /// Set default duration.
    pub fn set_default_duration(&mut self, duration: f32) {
        self.default_duration = duration;
    }

    /// Set dithering mode.
    pub fn set_dithering(&mut self, use_dithering: bool) {
        self.use_dithering = use_dithering;
    }

    /// Remove transition for an instance.
    pub fn cancel_transition(&mut self, instance_id: u32) {
        self.transitions.remove(&instance_id);
    }
}

// ---------------------------------------------------------------------------
// LodBudgetManager
// ---------------------------------------------------------------------------

/// Manages polygon budget distribution across LOD levels.
#[derive(Clone, Debug)]
pub struct LodBudgetManager {
    /// Total polygon budget.
    budget: u32,

    /// Polygon cost per LOD level (estimated).
    lod_costs: [u32; LOD_LEVEL_COUNT],

    /// Current polygon usage.
    current_usage: u32,

    /// Quality bias (0 = budget priority, 1 = quality priority).
    quality_bias: f32,

    /// Frame time history for adaptive quality.
    frame_times: Vec<f32>,

    /// Target frame time.
    target_frame_time: f32,

    /// Adaptive quality enabled.
    adaptive: bool,
}

impl LodBudgetManager {
    /// Create a new budget manager.
    pub fn new(budget: u32) -> Self {
        Self {
            budget,
            // Default polygon costs per LOD level
            lod_costs: [
                5000,  // LOD0: Full mesh
                1000,  // LOD1: Simplified mesh
                10,    // LOD2: Billboard quad
                1,     // LOD3: Point sprite
            ],
            current_usage: 0,
            quality_bias: 0.5,
            frame_times: Vec::with_capacity(60),
            target_frame_time: 1.0 / 60.0,
            adaptive: false,
        }
    }

    /// Set polygon costs for each LOD level.
    pub fn with_lod_costs(mut self, costs: [u32; LOD_LEVEL_COUNT]) -> Self {
        self.lod_costs = costs;
        self
    }

    /// Set quality bias.
    pub fn with_quality_bias(mut self, bias: f32) -> Self {
        self.quality_bias = bias.clamp(0.0, 1.0);
        self
    }

    /// Enable adaptive quality.
    pub fn with_adaptive(mut self, enable: bool, target_frame_time: f32) -> Self {
        self.adaptive = enable;
        self.target_frame_time = target_frame_time;
        self
    }

    /// Get polygon cost for a LOD level.
    #[inline]
    pub fn lod_cost(&self, level: LodLevel) -> u32 {
        self.lod_costs[level.to_u8() as usize]
    }

    /// Calculate total cost for given LOD distribution.
    pub fn calculate_cost(&self, lod_counts: &[usize; LOD_LEVEL_COUNT]) -> u32 {
        lod_counts
            .iter()
            .enumerate()
            .map(|(i, &count)| count as u32 * self.lod_costs[i])
            .sum()
    }

    /// Check if a LOD assignment is within budget.
    pub fn is_within_budget(&self, lod_counts: &[usize; LOD_LEVEL_COUNT]) -> bool {
        self.calculate_cost(lod_counts) <= self.budget
    }

    /// Distribute LOD levels to fit within budget.
    ///
    /// Takes sorted instance indices (by importance) and assigns LODs.
    /// Higher importance instances get higher LOD levels.
    pub fn distribute_budget(
        &self,
        sorted_indices: &[usize],
        base_lods: &[LodLevel],
        importance_scores: &[f32],
    ) -> Vec<LodLevel> {
        let count = sorted_indices.len();
        let mut assigned_lods = vec![LodLevel::Lod3; count];
        let mut remaining_budget = self.budget;

        // Assign LODs in order of importance
        for &idx in sorted_indices {
            if idx >= base_lods.len() {
                continue;
            }

            let base_lod = base_lods[idx];
            let importance = importance_scores.get(idx).copied().unwrap_or(0.0);

            // Try to assign the base LOD or higher
            let mut best_lod = base_lod;

            // If high importance and quality bias is high, try higher LOD
            if importance > 0.5 && self.quality_bias > 0.5 {
                if let Some(higher) = base_lod.higher() {
                    let cost_diff = self.lod_cost(higher).saturating_sub(self.lod_cost(base_lod));
                    if cost_diff <= remaining_budget {
                        best_lod = higher;
                    }
                }
            }

            let cost = self.lod_cost(best_lod);

            // If we can't afford this LOD, try lower ones
            let mut final_lod = best_lod;
            let mut final_cost = cost;

            while final_cost > remaining_budget {
                if let Some(lower) = final_lod.lower() {
                    final_lod = lower;
                    final_cost = self.lod_cost(final_lod);
                } else {
                    break;
                }
            }

            if final_cost <= remaining_budget {
                assigned_lods[idx] = final_lod;
                remaining_budget = remaining_budget.saturating_sub(final_cost);
            }
        }

        assigned_lods
    }

    /// Update with frame time for adaptive quality.
    pub fn update_frame_time(&mut self, frame_time: f32) {
        if !self.adaptive {
            return;
        }

        self.frame_times.push(frame_time);

        // Keep last 60 frames
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }

        // Adjust quality bias based on frame time
        if self.frame_times.len() >= 10 {
            let avg_frame_time: f32 = self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;

            if avg_frame_time > self.target_frame_time * 1.1 {
                // Frame time too high, reduce quality
                self.quality_bias = (self.quality_bias - 0.01).max(0.0);
            } else if avg_frame_time < self.target_frame_time * 0.9 {
                // Frame time good, can increase quality
                self.quality_bias = (self.quality_bias + 0.005).min(1.0);
            }
        }
    }

    /// Get current usage.
    pub fn current_usage(&self) -> u32 {
        self.current_usage
    }

    /// Get budget.
    pub fn budget(&self) -> u32 {
        self.budget
    }

    /// Set budget.
    pub fn set_budget(&mut self, budget: u32) {
        self.budget = budget;
    }

    /// Get quality bias.
    pub fn quality_bias(&self) -> f32 {
        self.quality_bias
    }

    /// Calculate importance score based on distance and priority.
    pub fn calculate_importance(distance: f32, max_distance: f32, priority: u8) -> f32 {
        let distance_score = 1.0 - (distance / max_distance).clamp(0.0, 1.0);

        let priority_bonus = if priority & PRIORITY_HERO != 0 {
            0.5
        } else if priority & PRIORITY_IMPORTANT != 0 {
            0.3
        } else if priority & PRIORITY_SCREEN_CENTER != 0 {
            0.1
        } else {
            0.0
        };

        (distance_score + priority_bonus).min(1.0)
    }
}

// ---------------------------------------------------------------------------
// LodMeshData
// ---------------------------------------------------------------------------

/// Data for a single LOD level's mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct LodMeshData {
    /// Mesh handle/ID.
    pub mesh_id: u32,

    /// Vertex count.
    pub vertex_count: u32,

    /// Index count.
    pub index_count: u32,

    /// Bone count (for skeletal meshes).
    pub bone_count: u32,

    /// Bone index remapping from full skeleton.
    /// `remap[simplified_bone] = full_bone`
    pub bone_remap: Vec<u16>,
}

impl LodMeshData {
    /// Create new LOD mesh data.
    pub fn new(mesh_id: u32, vertex_count: u32, index_count: u32) -> Self {
        Self {
            mesh_id,
            vertex_count,
            index_count,
            bone_count: 0,
            bone_remap: Vec::new(),
        }
    }

    /// Set bone information.
    pub fn with_bones(mut self, bone_count: u32, bone_remap: Vec<u16>) -> Self {
        self.bone_count = bone_count;
        self.bone_remap = bone_remap;
        self
    }

    /// Get polygon count (triangles).
    #[inline]
    pub fn polygon_count(&self) -> u32 {
        self.index_count / 3
    }

    /// Remap a bone index from simplified to full skeleton.
    pub fn remap_bone(&self, simplified_index: u16) -> Option<u16> {
        self.bone_remap.get(simplified_index as usize).copied()
    }
}

// ---------------------------------------------------------------------------
// LodMeshSet
// ---------------------------------------------------------------------------

/// Collection of meshes for each LOD level.
#[derive(Clone, Debug)]
pub struct LodMeshSet {
    /// Name/identifier for this mesh set.
    pub name: String,

    /// Meshes for each LOD level.
    meshes: [Option<LodMeshData>; LOD_LEVEL_COUNT],

    /// Full bone count (LOD0).
    pub full_bone_count: u32,

    /// Bounding radius.
    pub bounding_radius: f32,
}

impl LodMeshSet {
    /// Create a new mesh set.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            meshes: Default::default(),
            full_bone_count: 0,
            bounding_radius: 1.0,
        }
    }

    /// Add a mesh for a LOD level.
    pub fn with_lod(mut self, level: LodLevel, mesh: LodMeshData) -> Self {
        if level == LodLevel::Lod0 {
            self.full_bone_count = mesh.bone_count;
        }
        self.meshes[level.to_u8() as usize] = Some(mesh);
        self
    }

    /// Set bounding radius.
    pub fn with_bounding_radius(mut self, radius: f32) -> Self {
        self.bounding_radius = radius;
        self
    }

    /// Get mesh for a LOD level.
    pub fn get_mesh(&self, level: LodLevel) -> Option<&LodMeshData> {
        self.meshes[level.to_u8() as usize].as_ref()
    }

    /// Get the best available mesh at or below the requested level.
    pub fn get_best_available(&self, requested: LodLevel) -> Option<(LodLevel, &LodMeshData)> {
        // Try requested level first
        if let Some(mesh) = self.get_mesh(requested) {
            return Some((requested, mesh));
        }

        // Try lower detail levels
        let mut level = requested;
        while let Some(lower) = level.lower() {
            if let Some(mesh) = self.get_mesh(lower) {
                return Some((lower, mesh));
            }
            level = lower;
        }

        // Try higher detail levels
        level = requested;
        while let Some(higher) = level.higher() {
            if let Some(mesh) = self.get_mesh(higher) {
                return Some((higher, mesh));
            }
            level = higher;
        }

        None
    }

    /// Check if a LOD level is available.
    pub fn has_lod(&self, level: LodLevel) -> bool {
        self.meshes[level.to_u8() as usize].is_some()
    }

    /// Get all available LOD levels.
    pub fn available_levels(&self) -> Vec<LodLevel> {
        LodLevel::all()
            .into_iter()
            .filter(|&level| self.has_lod(level))
            .collect()
    }

    /// Get bone remap from simplified LOD to full skeleton.
    pub fn get_bone_remap(&self, level: LodLevel) -> Option<&[u16]> {
        self.get_mesh(level).map(|m| m.bone_remap.as_slice())
    }

    /// Validate seamless LOD boundaries (check bone counts are compatible).
    pub fn validate_bone_compatibility(&self) -> Result<(), LodError> {
        let mut prev_bones = self.full_bone_count;

        for level in LodLevel::all() {
            if let Some(mesh) = self.get_mesh(level) {
                // Bone count should not increase in lower LODs
                if mesh.bone_count > prev_bones && level != LodLevel::Lod0 {
                    return Err(LodError::InvalidConfig {
                        reason: "lower LOD has more bones than higher LOD",
                    });
                }

                // Check bone remap validity
                for &remapped in &mesh.bone_remap {
                    if remapped as u32 >= self.full_bone_count {
                        return Err(LodError::InvalidConfig {
                            reason: "bone remap index out of range",
                        });
                    }
                }

                prev_bones = mesh.bone_count;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CrowdLodInstance
// ---------------------------------------------------------------------------

/// Extended instance data for LOD system.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CrowdLodInstance {
    /// World-space position.
    pub position: Vec3,
    /// Uniform scale factor.
    pub scale: f32,
    /// Current LOD level.
    pub current_lod: u8,
    /// Priority flags.
    pub priority: u8,
    /// Character type ID.
    pub type_id: u8,
    /// Transition flags.
    pub transition_flags: u8,
    /// Transition blend factor (0-255 mapped to 0.0-1.0).
    pub transition_blend: u8,
    /// Previous LOD level (for transitions).
    pub previous_lod: u8,
    /// Padding.
    pub _padding: [u8; 2],
    /// Instance ID.
    pub instance_id: u32,
}

impl CrowdLodInstance {
    /// Create a new instance.
    pub fn new(position: impl Into<Vec3>, instance_id: u32) -> Self {
        Self {
            position: position.into(),
            scale: 1.0,
            current_lod: LodLevel::Lod0.to_u8(),
            priority: 0,
            type_id: 0,
            transition_flags: 0,
            transition_blend: 0,
            previous_lod: LodLevel::Lod0.to_u8(),
            _padding: [0; 2],
            instance_id,
        }
    }

    /// Set scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Set type ID.
    pub fn with_type(mut self, type_id: u8) -> Self {
        self.type_id = type_id;
        self
    }

    /// Get current LOD level.
    #[inline]
    pub fn lod_level(&self) -> LodLevel {
        LodLevel::from_u8(self.current_lod).unwrap_or(LodLevel::Lod0)
    }

    /// Set LOD level.
    pub fn set_lod(&mut self, level: LodLevel) {
        self.previous_lod = self.current_lod;
        self.current_lod = level.to_u8();
    }

    /// Check if instance has hero priority.
    #[inline]
    pub fn is_hero(&self) -> bool {
        (self.priority & PRIORITY_HERO) != 0
    }

    /// Check if instance has important priority.
    #[inline]
    pub fn is_important(&self) -> bool {
        (self.priority & PRIORITY_IMPORTANT) != 0
    }

    /// Get transition blend as float (0.0-1.0).
    #[inline]
    pub fn blend_factor(&self) -> f32 {
        self.transition_blend as f32 / 255.0
    }

    /// Set transition blend from float.
    pub fn set_blend_factor(&mut self, factor: f32) {
        self.transition_blend = (factor.clamp(0.0, 1.0) * 255.0) as u8;
    }

    /// Check if transitioning.
    #[inline]
    pub fn is_transitioning(&self) -> bool {
        (self.transition_flags & TRANSITION_ACTIVE) != 0
    }
}

// ---------------------------------------------------------------------------
// CrowdLodStats
// ---------------------------------------------------------------------------

/// Statistics for the LOD system.
#[derive(Clone, Debug, Default)]
pub struct CrowdLodStats {
    /// Instance count per LOD level.
    pub lod_counts: [usize; LOD_LEVEL_COUNT],

    /// Total instance count.
    pub total_instances: usize,

    /// Active transition count.
    pub active_transitions: usize,

    /// Estimated polygon count.
    pub polygon_count: u32,

    /// Budget usage (0.0-1.0).
    pub budget_usage: f32,

    /// Average LOD level.
    pub average_lod: f32,

    /// Quality bias (from adaptive system).
    pub quality_bias: f32,

    /// Frame number.
    pub frame: u64,
}

impl CrowdLodStats {
    /// Calculate average LOD level.
    pub fn calculate_average(&mut self) {
        if self.total_instances > 0 {
            let weighted_sum: f32 = self.lod_counts
                .iter()
                .enumerate()
                .map(|(i, &count)| i as f32 * count as f32)
                .sum();
            self.average_lod = weighted_sum / self.total_instances as f32;
        }
    }
}

// ---------------------------------------------------------------------------
// CrowdLodSystem
// ---------------------------------------------------------------------------

/// Complete LOD system integrating all components.
#[derive(Debug)]
pub struct CrowdLodSystem {
    /// Configuration.
    pub config: LodConfig,

    /// LOD selector.
    selector: LodSelector,

    /// Transition manager.
    transitions: LodTransitionManager,

    /// Budget manager.
    budget: LodBudgetManager,

    /// Registered mesh sets.
    mesh_sets: HashMap<u32, LodMeshSet>,

    /// Current frame number.
    frame: u64,

    /// Statistics.
    stats: CrowdLodStats,

    /// Debug mode.
    debug_mode: bool,
}

impl CrowdLodSystem {
    /// Create a new LOD system with the given configuration.
    pub fn new(config: LodConfig) -> Self {
        let selector = LodSelector::new(config.clone());
        let transitions = LodTransitionManager::new(
            config.transition_duration,
            config.use_dithering,
        );
        let budget = LodBudgetManager::new(config.polygon_budget);

        Self {
            config,
            selector,
            transitions,
            budget,
            mesh_sets: HashMap::new(),
            frame: 0,
            stats: CrowdLodStats::default(),
            debug_mode: false,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(LodConfig::default())
    }

    /// Register a mesh set for a character type.
    pub fn register_mesh_set(&mut self, type_id: u32, mesh_set: LodMeshSet) {
        self.mesh_sets.insert(type_id, mesh_set);
    }

    /// Unregister a mesh set.
    pub fn unregister_mesh_set(&mut self, type_id: u32) -> Option<LodMeshSet> {
        self.mesh_sets.remove(&type_id)
    }

    /// Get a mesh set.
    pub fn get_mesh_set(&self, type_id: u32) -> Option<&LodMeshSet> {
        self.mesh_sets.get(&type_id)
    }

    /// Update camera parameters.
    pub fn update_camera(
        &mut self,
        camera_pos: Vec3,
        camera_forward: Vec3,
        fov: f32,
        screen_height: f32,
    ) {
        self.selector.update_camera(camera_pos, camera_forward, fov, screen_height);
    }

    /// Update LODs for all instances.
    pub fn update(
        &mut self,
        camera_pos: Vec3,
        camera_forward: Vec3,
        fov: f32,
        instances: &mut [CrowdLodInstance],
        delta_time: f32,
    ) {
        // Update camera
        self.selector.update_camera(camera_pos, camera_forward, fov, 1080.0);

        // Update transitions
        self.transitions.update(delta_time);

        // Calculate LODs
        self.update_instance_lods(instances);

        // Update statistics
        self.update_stats(instances);

        // Update budget with frame time
        self.budget.update_frame_time(delta_time);

        self.frame += 1;
    }

    /// Update LOD levels for all instances.
    fn update_instance_lods(&mut self, instances: &mut [CrowdLodInstance]) {
        for instance in instances.iter_mut() {
            let type_id = if instance.type_id > 0 {
                Some(instance.type_id as u32)
            } else {
                None
            };

            let previous_lod = LodLevel::from_u8(instance.previous_lod);
            let new_lod = self.selector.select_lod_distance(
                instance.position,
                instance.scale,
                type_id,
                instance.priority,
                previous_lod,
            );

            let current_lod = instance.lod_level();

            if new_lod != current_lod {
                // Start transition
                self.transitions.start_or_update_transition(
                    instance.instance_id,
                    current_lod,
                    new_lod,
                );
                instance.set_lod(new_lod);
            }

            // Update transition state
            if let Some(transition) = self.transitions.get_transition(instance.instance_id) {
                instance.transition_flags = transition.flags;
                instance.set_blend_factor(transition.blend_factor());
            } else {
                instance.transition_flags = 0;
                instance.transition_blend = 255;
            }
        }
    }

    /// Update statistics.
    fn update_stats(&mut self, instances: &[CrowdLodInstance]) {
        self.stats = CrowdLodStats {
            lod_counts: [0; LOD_LEVEL_COUNT],
            total_instances: instances.len(),
            active_transitions: self.transitions.active_count(),
            polygon_count: 0,
            budget_usage: 0.0,
            average_lod: 0.0,
            quality_bias: self.budget.quality_bias(),
            frame: self.frame,
        };

        for instance in instances {
            let lod = instance.lod_level();
            self.stats.lod_counts[lod.to_u8() as usize] += 1;
            self.stats.polygon_count += self.budget.lod_cost(lod);
        }

        self.stats.budget_usage = self.stats.polygon_count as f32 / self.budget.budget() as f32;
        self.stats.calculate_average();
    }

    /// Get current statistics.
    pub fn stats(&self) -> &CrowdLodStats {
        &self.stats
    }

    /// Get mesh for instance's current LOD.
    pub fn get_instance_mesh(&self, instance: &CrowdLodInstance) -> Option<&LodMeshData> {
        let type_id = instance.type_id as u32;
        let level = instance.lod_level();

        self.mesh_sets
            .get(&type_id)
            .and_then(|set| set.get_mesh(level))
    }

    /// Get best available mesh for instance.
    pub fn get_instance_best_mesh(&self, instance: &CrowdLodInstance) -> Option<(LodLevel, &LodMeshData)> {
        let type_id = instance.type_id as u32;
        let level = instance.lod_level();

        self.mesh_sets
            .get(&type_id)
            .and_then(|set| set.get_best_available(level))
    }

    /// Handle camera teleport (clear hysteresis history).
    pub fn handle_camera_teleport(&mut self) {
        self.selector.clear_history();
        self.transitions.clear();
    }

    /// Handle spawn burst (many new instances).
    pub fn handle_spawn_burst(&mut self, instance_ids: &[u32]) {
        // Cancel any pending transitions for new instances
        for &id in instance_ids {
            self.transitions.cancel_transition(id);
        }
    }

    /// Set debug mode.
    pub fn set_debug_mode(&mut self, enabled: bool) {
        self.debug_mode = enabled;
    }

    /// Hot-reload configuration.
    pub fn reload_config(&mut self, config: LodConfig) -> Result<(), LodError> {
        config.validate()?;

        self.config = config.clone();
        self.selector.set_config(config.clone());
        self.transitions.set_default_duration(config.transition_duration);
        self.transitions.set_dithering(config.use_dithering);
        self.budget.set_budget(config.polygon_budget);

        Ok(())
    }

    /// Get selector reference.
    pub fn selector(&self) -> &LodSelector {
        &self.selector
    }

    /// Get transition manager reference.
    pub fn transitions(&self) -> &LodTransitionManager {
        &self.transitions
    }

    /// Get budget manager reference.
    pub fn budget(&self) -> &LodBudgetManager {
        &self.budget
    }

    /// Get current frame number.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// Calculate LOD distribution for a set of distances.
    pub fn calculate_distribution(&self, distances: &[f32]) -> [usize; LOD_LEVEL_COUNT] {
        let mut counts = [0usize; LOD_LEVEL_COUNT];

        for &distance in distances {
            for (i, &threshold) in self.config.distance_thresholds.iter().enumerate() {
                if distance < threshold {
                    counts[i] += 1;
                    break;
                }
                if i == LOD_LEVEL_COUNT - 1 {
                    counts[LOD_LEVEL_COUNT - 1] += 1;
                }
            }
        }

        counts
    }
}

impl Default for CrowdLodSystem {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // LodLevel Tests
    // ========================================================================

    #[test]
    fn test_lod_level_from_u8() {
        assert_eq!(LodLevel::from_u8(0), Some(LodLevel::Lod0));
        assert_eq!(LodLevel::from_u8(1), Some(LodLevel::Lod1));
        assert_eq!(LodLevel::from_u8(2), Some(LodLevel::Lod2));
        assert_eq!(LodLevel::from_u8(3), Some(LodLevel::Lod3));
        assert_eq!(LodLevel::from_u8(4), None);
    }

    #[test]
    fn test_lod_level_to_u8() {
        assert_eq!(LodLevel::Lod0.to_u8(), 0);
        assert_eq!(LodLevel::Lod1.to_u8(), 1);
        assert_eq!(LodLevel::Lod2.to_u8(), 2);
        assert_eq!(LodLevel::Lod3.to_u8(), 3);
    }

    #[test]
    fn test_lod_level_all() {
        let all = LodLevel::all();
        assert_eq!(all.len(), LOD_LEVEL_COUNT);
        assert_eq!(all[0], LodLevel::Lod0);
        assert_eq!(all[3], LodLevel::Lod3);
    }

    #[test]
    fn test_lod_level_uses_skeleton() {
        assert!(LodLevel::Lod0.uses_skeleton());
        assert!(LodLevel::Lod1.uses_skeleton());
        assert!(!LodLevel::Lod2.uses_skeleton());
        assert!(!LodLevel::Lod3.uses_skeleton());
    }

    #[test]
    fn test_lod_level_is_billboard() {
        assert!(!LodLevel::Lod0.is_billboard());
        assert!(!LodLevel::Lod1.is_billboard());
        assert!(LodLevel::Lod2.is_billboard());
        assert!(!LodLevel::Lod3.is_billboard());
    }

    #[test]
    fn test_lod_level_is_point_sprite() {
        assert!(!LodLevel::Lod0.is_point_sprite());
        assert!(!LodLevel::Lod1.is_point_sprite());
        assert!(!LodLevel::Lod2.is_point_sprite());
        assert!(LodLevel::Lod3.is_point_sprite());
    }

    #[test]
    fn test_lod_level_lower() {
        assert_eq!(LodLevel::Lod0.lower(), Some(LodLevel::Lod1));
        assert_eq!(LodLevel::Lod1.lower(), Some(LodLevel::Lod2));
        assert_eq!(LodLevel::Lod2.lower(), Some(LodLevel::Lod3));
        assert_eq!(LodLevel::Lod3.lower(), None);
    }

    #[test]
    fn test_lod_level_higher() {
        assert_eq!(LodLevel::Lod0.higher(), None);
        assert_eq!(LodLevel::Lod1.higher(), Some(LodLevel::Lod0));
        assert_eq!(LodLevel::Lod2.higher(), Some(LodLevel::Lod1));
        assert_eq!(LodLevel::Lod3.higher(), Some(LodLevel::Lod2));
    }

    #[test]
    fn test_lod_level_ordering() {
        assert!(LodLevel::Lod0 < LodLevel::Lod1);
        assert!(LodLevel::Lod1 < LodLevel::Lod2);
        assert!(LodLevel::Lod2 < LodLevel::Lod3);
    }

    #[test]
    fn test_lod_level_name() {
        assert_eq!(LodLevel::Lod0.short_name(), "LOD0");
        assert_eq!(LodLevel::Lod3.short_name(), "LOD3");
    }

    // ========================================================================
    // LodConfig Tests
    // ========================================================================

    #[test]
    fn test_config_default() {
        let config = LodConfig::default();
        assert_eq!(config.distance_thresholds[0], DEFAULT_LOD0_DISTANCE);
        assert_eq!(config.hysteresis, DEFAULT_HYSTERESIS);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = LodConfig::new()
            .with_distance_thresholds([10.0, 30.0, 60.0, 120.0])
            .with_hysteresis(3.0)
            .with_screen_size_lod(true)
            .with_polygon_budget(1_000_000);

        assert_eq!(config.distance_thresholds[0], 10.0);
        assert_eq!(config.hysteresis, 3.0);
        assert!(config.screen_size_lod);
        assert_eq!(config.polygon_budget, 1_000_000);
    }

    #[test]
    fn test_config_validate_valid() {
        let config = LodConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_invalid_thresholds() {
        let config = LodConfig::new()
            .with_distance_thresholds([50.0, 30.0, 100.0, 200.0]); // Not increasing
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_zero_threshold() {
        let config = LodConfig::new()
            .with_distance_thresholds([0.0, 30.0, 60.0, 120.0]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_type_distance_multiplier() {
        let config = LodConfig::new()
            .with_type_distance_multiplier(1, 2.0);

        assert_eq!(config.get_threshold(LodLevel::Lod0, Some(1)), DEFAULT_LOD0_DISTANCE * 2.0);
        assert_eq!(config.get_threshold(LodLevel::Lod0, None), DEFAULT_LOD0_DISTANCE);
    }

    #[test]
    fn test_config_for_large_crowd() {
        let config = LodConfig::for_large_crowd();
        assert!(config.adaptive_quality);
        assert!(config.polygon_budget > DEFAULT_POLYGON_BUDGET);
    }

    #[test]
    fn test_config_for_cinematic() {
        let config = LodConfig::for_cinematic();
        assert!(config.transition_duration > DEFAULT_TRANSITION_DURATION);
    }

    // ========================================================================
    // LodSelector Tests
    // ========================================================================

    #[test]
    fn test_selector_new() {
        let config = LodConfig::default();
        let selector = LodSelector::new(config);
        assert_eq!(selector.camera_pos, Vec3::ZERO);
    }

    #[test]
    fn test_selector_select_lod_close() {
        let config = LodConfig::default();
        let selector = LodSelector::new(config);

        // Close to camera
        let lod = selector.select_lod_distance(
            Vec3::new(5.0, 0.0, 0.0),
            1.0,
            None,
            0,
            None,
        );
        assert_eq!(lod, LodLevel::Lod0);
    }

    #[test]
    fn test_selector_select_lod_medium() {
        let config = LodConfig::default();
        let selector = LodSelector::new(config);

        // Medium distance
        let lod = selector.select_lod_distance(
            Vec3::new(35.0, 0.0, 0.0),
            1.0,
            None,
            0,
            None,
        );
        assert_eq!(lod, LodLevel::Lod1);
    }

    #[test]
    fn test_selector_select_lod_far() {
        let config = LodConfig::default();
        let selector = LodSelector::new(config);

        // Far distance
        let lod = selector.select_lod_distance(
            Vec3::new(80.0, 0.0, 0.0),
            1.0,
            None,
            0,
            None,
        );
        assert_eq!(lod, LodLevel::Lod2);
    }

    #[test]
    fn test_selector_select_lod_very_far() {
        let config = LodConfig::default();
        let selector = LodSelector::new(config);

        // Very far distance
        let lod = selector.select_lod_distance(
            Vec3::new(150.0, 0.0, 0.0),
            1.0,
            None,
            0,
            None,
        );
        assert_eq!(lod, LodLevel::Lod3);
    }

    #[test]
    fn test_selector_hero_priority() {
        let config = LodConfig::default();
        let selector = LodSelector::new(config);

        // Hero stays at LOD0 regardless of distance
        let lod = selector.select_lod_distance(
            Vec3::new(150.0, 0.0, 0.0),
            1.0,
            None,
            PRIORITY_HERO,
            None,
        );
        assert_eq!(lod, LodLevel::Lod0);
    }

    #[test]
    fn test_selector_important_priority() {
        let config = LodConfig::default();
        let selector = LodSelector::new(config);

        // Important character stays at LOD1 or better
        let lod = selector.select_lod_distance(
            Vec3::new(150.0, 0.0, 0.0),
            1.0,
            None,
            PRIORITY_IMPORTANT,
            None,
        );
        assert_eq!(lod, LodLevel::Lod1);
    }

    #[test]
    fn test_selector_scale_affects_lod() {
        let config = LodConfig::default();
        let selector = LodSelector::new(config);

        // Larger characters stay higher LOD longer
        let lod_small = selector.select_lod_distance(
            Vec3::new(40.0, 0.0, 0.0),
            1.0,
            None,
            0,
            None,
        );
        let lod_large = selector.select_lod_distance(
            Vec3::new(40.0, 0.0, 0.0),
            3.0,
            None,
            0,
            None,
        );

        // Large character should be higher or equal LOD
        assert!(lod_large <= lod_small);
    }

    #[test]
    fn test_selector_hysteresis() {
        let config = LodConfig::new().with_hysteresis(5.0);
        let selector = LodSelector::new(config);

        // At threshold boundary, should stay at previous LOD
        let lod = selector.select_lod_distance(
            Vec3::new(22.0, 0.0, 0.0), // Just past LOD0 threshold
            1.0,
            None,
            0,
            Some(LodLevel::Lod0),
        );
        // With hysteresis, should stay at LOD0
        assert_eq!(lod, LodLevel::Lod0);
    }

    #[test]
    fn test_selector_batch_select() {
        let config = LodConfig::default();
        let mut selector = LodSelector::new(config);

        let positions = vec![
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(35.0, 0.0, 0.0),
            Vec3::new(80.0, 0.0, 0.0),
        ];
        let scales = vec![1.0, 1.0, 1.0];
        let priorities = vec![0, 0, 0];
        let type_ids: Vec<Option<u32>> = vec![None, None, None];

        let lods = selector.batch_select(&positions, &scales, &priorities, &type_ids);

        assert_eq!(lods.len(), 3);
        assert_eq!(lods[0], LodLevel::Lod0);
        assert_eq!(lods[1], LodLevel::Lod1);
        assert_eq!(lods[2], LodLevel::Lod2);
    }

    // ========================================================================
    // LodTransition Tests
    // ========================================================================

    #[test]
    fn test_transition_new() {
        let transition = LodTransition::new(LodLevel::Lod0, LodLevel::Lod1, 0.5, false);
        assert!(transition.is_active());
        assert!(!transition.is_dithered());
        assert_eq!(transition.progress, 0.0);
    }

    #[test]
    fn test_transition_advance() {
        let mut transition = LodTransition::new(LodLevel::Lod0, LodLevel::Lod1, 1.0, false);

        transition.advance(0.25);
        assert!((transition.progress - 0.25).abs() < 0.001);
        assert!(transition.is_active());

        transition.advance(0.75);
        assert!((transition.progress - 1.0).abs() < 0.001);
        assert!(!transition.is_active());
    }

    #[test]
    fn test_transition_blend_factor() {
        let mut transition = LodTransition::new(LodLevel::Lod0, LodLevel::Lod1, 1.0, false);

        transition.advance(0.5);
        assert!((transition.blend_factor() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_transition_dither_threshold() {
        let transition = LodTransition::new(LodLevel::Lod0, LodLevel::Lod1, 1.0, true);

        let threshold = transition.dither_threshold(0.0, 0.0);
        assert!(threshold >= 0.0 && threshold <= 1.0);
    }

    #[test]
    fn test_transition_complete() {
        let mut transition = LodTransition::new(LodLevel::Lod0, LodLevel::Lod1, 0.5, false);

        assert!(!transition.is_complete());
        transition.advance(1.0);
        assert!(transition.is_complete());
    }

    // ========================================================================
    // LodTransitionManager Tests
    // ========================================================================

    #[test]
    fn test_transition_manager_new() {
        let manager = LodTransitionManager::new(0.25, true);
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_transition_manager_start() {
        let mut manager = LodTransitionManager::new(0.25, true);

        let result = manager.start_transition(1, LodLevel::Lod0, LodLevel::Lod1);
        assert!(result.is_ok());
        assert!(manager.is_transitioning(1));
    }

    #[test]
    fn test_transition_manager_start_same_lod() {
        let mut manager = LodTransitionManager::new(0.25, true);

        let result = manager.start_transition(1, LodLevel::Lod0, LodLevel::Lod0);
        assert!(result.is_ok());
        assert!(!manager.is_transitioning(1));
    }

    #[test]
    fn test_transition_manager_duplicate() {
        let mut manager = LodTransitionManager::new(0.25, true);

        manager.start_transition(1, LodLevel::Lod0, LodLevel::Lod1).unwrap();
        let result = manager.start_transition(1, LodLevel::Lod0, LodLevel::Lod2);
        assert!(result.is_err());
    }

    #[test]
    fn test_transition_manager_update() {
        let mut manager = LodTransitionManager::new(0.5, false);

        manager.start_transition(1, LodLevel::Lod0, LodLevel::Lod1).unwrap();
        manager.update(0.25);

        let transition = manager.get_transition(1).unwrap();
        assert!((transition.progress - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_transition_manager_cleanup() {
        let mut manager = LodTransitionManager::new(0.25, false);

        manager.start_transition(1, LodLevel::Lod0, LodLevel::Lod1).unwrap();
        manager.update(1.0); // Complete transition

        // Should be cleaned up
        assert!(!manager.is_transitioning(1));
    }

    #[test]
    fn test_transition_manager_effective_lod() {
        let mut manager = LodTransitionManager::new(1.0, false);

        // No transition
        assert_eq!(manager.effective_lod(1, LodLevel::Lod1), LodLevel::Lod1);

        // During transition
        manager.start_transition(1, LodLevel::Lod0, LodLevel::Lod1).unwrap();
        assert_eq!(manager.effective_lod(1, LodLevel::Lod1), LodLevel::Lod0);

        manager.update(0.6); // Past 50%
        assert_eq!(manager.effective_lod(1, LodLevel::Lod1), LodLevel::Lod1);
    }

    // ========================================================================
    // LodBudgetManager Tests
    // ========================================================================

    #[test]
    fn test_budget_manager_new() {
        let manager = LodBudgetManager::new(500_000);
        assert_eq!(manager.budget(), 500_000);
    }

    #[test]
    fn test_budget_manager_lod_cost() {
        let manager = LodBudgetManager::new(500_000);

        assert!(manager.lod_cost(LodLevel::Lod0) > manager.lod_cost(LodLevel::Lod1));
        assert!(manager.lod_cost(LodLevel::Lod1) > manager.lod_cost(LodLevel::Lod2));
        assert!(manager.lod_cost(LodLevel::Lod2) > manager.lod_cost(LodLevel::Lod3));
    }

    #[test]
    fn test_budget_manager_calculate_cost() {
        let manager = LodBudgetManager::new(500_000);

        let counts = [10, 20, 30, 40];
        let cost = manager.calculate_cost(&counts);

        let expected = 10 * manager.lod_cost(LodLevel::Lod0)
            + 20 * manager.lod_cost(LodLevel::Lod1)
            + 30 * manager.lod_cost(LodLevel::Lod2)
            + 40 * manager.lod_cost(LodLevel::Lod3);

        assert_eq!(cost, expected);
    }

    #[test]
    fn test_budget_manager_is_within_budget() {
        let manager = LodBudgetManager::new(500_000);

        let small_counts = [10, 20, 30, 40];
        assert!(manager.is_within_budget(&small_counts));

        let large_counts = [1000, 2000, 3000, 4000];
        assert!(!manager.is_within_budget(&large_counts));
    }

    #[test]
    fn test_budget_manager_calculate_importance() {
        let importance_close = LodBudgetManager::calculate_importance(10.0, 100.0, 0);
        let importance_far = LodBudgetManager::calculate_importance(90.0, 100.0, 0);

        assert!(importance_close > importance_far);

        let importance_hero = LodBudgetManager::calculate_importance(50.0, 100.0, PRIORITY_HERO);
        let importance_normal = LodBudgetManager::calculate_importance(50.0, 100.0, 0);

        assert!(importance_hero > importance_normal);
    }

    #[test]
    fn test_budget_manager_distribute_budget() {
        let manager = LodBudgetManager::new(100_000);

        let sorted_indices: Vec<usize> = (0..10).collect();
        let base_lods = vec![LodLevel::Lod0; 10];
        let importance_scores: Vec<f32> = (0..10).map(|i| 1.0 - i as f32 * 0.1).collect();

        let assigned = manager.distribute_budget(&sorted_indices, &base_lods, &importance_scores);

        assert_eq!(assigned.len(), 10);
    }

    // ========================================================================
    // LodMeshData Tests
    // ========================================================================

    #[test]
    fn test_mesh_data_new() {
        let mesh = LodMeshData::new(1, 1000, 3000);
        assert_eq!(mesh.mesh_id, 1);
        assert_eq!(mesh.vertex_count, 1000);
        assert_eq!(mesh.index_count, 3000);
        assert_eq!(mesh.polygon_count(), 1000);
    }

    #[test]
    fn test_mesh_data_with_bones() {
        let remap = vec![0, 1, 5, 10];
        let mesh = LodMeshData::new(1, 1000, 3000)
            .with_bones(4, remap.clone());

        assert_eq!(mesh.bone_count, 4);
        assert_eq!(mesh.remap_bone(2), Some(5));
    }

    // ========================================================================
    // LodMeshSet Tests
    // ========================================================================

    #[test]
    fn test_mesh_set_new() {
        let set = LodMeshSet::new("soldier");
        assert_eq!(set.name, "soldier");
        assert!(set.available_levels().is_empty());
    }

    #[test]
    fn test_mesh_set_with_lod() {
        let set = LodMeshSet::new("soldier")
            .with_lod(LodLevel::Lod0, LodMeshData::new(0, 5000, 15000).with_bones(64, vec![]))
            .with_lod(LodLevel::Lod1, LodMeshData::new(1, 1000, 3000).with_bones(16, vec![0, 5, 10, 15]));

        assert!(set.has_lod(LodLevel::Lod0));
        assert!(set.has_lod(LodLevel::Lod1));
        assert!(!set.has_lod(LodLevel::Lod2));
        assert_eq!(set.full_bone_count, 64);
    }

    #[test]
    fn test_mesh_set_get_best_available() {
        let set = LodMeshSet::new("soldier")
            .with_lod(LodLevel::Lod0, LodMeshData::new(0, 5000, 15000))
            .with_lod(LodLevel::Lod2, LodMeshData::new(2, 10, 6));

        // Request LOD1, should get LOD2 (next available lower)
        let (level, _mesh) = set.get_best_available(LodLevel::Lod1).unwrap();
        assert_eq!(level, LodLevel::Lod2);

        // Request LOD0, should get LOD0
        let (level, _mesh) = set.get_best_available(LodLevel::Lod0).unwrap();
        assert_eq!(level, LodLevel::Lod0);
    }

    #[test]
    fn test_mesh_set_available_levels() {
        let set = LodMeshSet::new("soldier")
            .with_lod(LodLevel::Lod0, LodMeshData::new(0, 5000, 15000))
            .with_lod(LodLevel::Lod2, LodMeshData::new(2, 10, 6));

        let levels = set.available_levels();
        assert_eq!(levels.len(), 2);
        assert!(levels.contains(&LodLevel::Lod0));
        assert!(levels.contains(&LodLevel::Lod2));
    }

    // ========================================================================
    // CrowdLodInstance Tests
    // ========================================================================

    #[test]
    fn test_crowd_lod_instance_new() {
        let instance = CrowdLodInstance::new([10.0, 0.0, 5.0], 42);
        assert_eq!(instance.position.x, 10.0);
        assert_eq!(instance.instance_id, 42);
        assert_eq!(instance.lod_level(), LodLevel::Lod0);
    }

    #[test]
    fn test_crowd_lod_instance_set_lod() {
        let mut instance = CrowdLodInstance::new([0.0, 0.0, 0.0], 1);
        instance.set_lod(LodLevel::Lod2);

        assert_eq!(instance.lod_level(), LodLevel::Lod2);
        assert_eq!(instance.previous_lod, LodLevel::Lod0.to_u8());
    }

    #[test]
    fn test_crowd_lod_instance_priority() {
        let hero = CrowdLodInstance::new([0.0, 0.0, 0.0], 1).with_priority(PRIORITY_HERO);
        assert!(hero.is_hero());
        assert!(!hero.is_important());

        let important = CrowdLodInstance::new([0.0, 0.0, 0.0], 2).with_priority(PRIORITY_IMPORTANT);
        assert!(!important.is_hero());
        assert!(important.is_important());
    }

    #[test]
    fn test_crowd_lod_instance_blend_factor() {
        let mut instance = CrowdLodInstance::new([0.0, 0.0, 0.0], 1);

        instance.set_blend_factor(0.5);
        assert!((instance.blend_factor() - 0.5).abs() < 0.01);

        instance.set_blend_factor(1.0);
        assert!((instance.blend_factor() - 1.0).abs() < 0.01);
    }

    // ========================================================================
    // CrowdLodSystem Tests
    // ========================================================================

    #[test]
    fn test_system_new() {
        let system = CrowdLodSystem::new(LodConfig::default());
        assert_eq!(system.frame(), 0);
    }

    #[test]
    fn test_system_register_mesh_set() {
        let mut system = CrowdLodSystem::new(LodConfig::default());

        let mesh_set = LodMeshSet::new("soldier")
            .with_lod(LodLevel::Lod0, LodMeshData::new(0, 5000, 15000));

        system.register_mesh_set(0, mesh_set);

        assert!(system.get_mesh_set(0).is_some());
        assert!(system.get_mesh_set(1).is_none());
    }

    #[test]
    fn test_system_update() {
        let mut system = CrowdLodSystem::new(LodConfig::default());

        let mut instances = vec![
            CrowdLodInstance::new([10.0, 0.0, 0.0], 0),
            CrowdLodInstance::new([35.0, 0.0, 0.0], 1),
            CrowdLodInstance::new([80.0, 0.0, 0.0], 2),
        ];

        system.update(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
            &mut instances,
            0.016,
        );

        assert_eq!(instances[0].lod_level(), LodLevel::Lod0);
        assert_eq!(instances[1].lod_level(), LodLevel::Lod1);
        assert_eq!(instances[2].lod_level(), LodLevel::Lod2);
    }

    #[test]
    fn test_system_stats() {
        let mut system = CrowdLodSystem::new(LodConfig::default());

        let mut instances = vec![
            CrowdLodInstance::new([10.0, 0.0, 0.0], 0),
            CrowdLodInstance::new([10.0, 0.0, 0.0], 1),
            CrowdLodInstance::new([80.0, 0.0, 0.0], 2),
        ];

        system.update(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 1.0, &mut instances, 0.016);

        let stats = system.stats();
        assert_eq!(stats.total_instances, 3);
        assert_eq!(stats.lod_counts[0], 2); // Two at LOD0
        assert_eq!(stats.lod_counts[2], 1); // One at LOD2
    }

    #[test]
    fn test_system_handle_camera_teleport() {
        let mut system = CrowdLodSystem::new(LodConfig::default());

        let mut instances = vec![CrowdLodInstance::new([10.0, 0.0, 0.0], 0)];

        system.update(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 1.0, &mut instances, 0.016);
        system.handle_camera_teleport();

        // Should clear history without error
        assert_eq!(system.transitions.active_count(), 0);
    }

    #[test]
    fn test_system_reload_config() {
        let mut system = CrowdLodSystem::new(LodConfig::default());

        let new_config = LodConfig::new()
            .with_distance_thresholds([30.0, 60.0, 120.0, 240.0])
            .with_hysteresis(5.0);

        let result = system.reload_config(new_config);
        assert!(result.is_ok());
        assert_eq!(system.config.distance_thresholds[0], 30.0);
    }

    #[test]
    fn test_system_calculate_distribution() {
        let system = CrowdLodSystem::new(LodConfig::default());

        let distances = vec![5.0, 10.0, 35.0, 80.0, 150.0];
        let distribution = system.calculate_distribution(&distances);

        assert_eq!(distribution[0], 2); // 5.0 and 10.0 at LOD0
        assert_eq!(distribution[1], 1); // 35.0 at LOD1
        assert_eq!(distribution[2], 1); // 80.0 at LOD2
        assert_eq!(distribution[3], 1); // 150.0 at LOD3
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_zero_distance() {
        let config = LodConfig::default();
        let selector = LodSelector::new(config);

        let lod = selector.select_lod_distance(Vec3::ZERO, 1.0, None, 0, None);
        assert_eq!(lod, LodLevel::Lod0);
    }

    #[test]
    fn test_zero_scale() {
        let config = LodConfig::default();
        let selector = LodSelector::new(config);

        // Zero scale should be handled gracefully
        let lod = selector.select_lod_distance(Vec3::new(35.0, 0.0, 0.0), 0.0, None, 0, None);
        // Should use minimum scale internally
        assert!(lod >= LodLevel::Lod0);
    }

    #[test]
    fn test_extremely_far_distance() {
        let config = LodConfig::default();
        let selector = LodSelector::new(config);

        let lod = selector.select_lod_distance(Vec3::new(10000.0, 0.0, 0.0), 1.0, None, 0, None);
        assert_eq!(lod, LodLevel::Lod3);
    }

    #[test]
    fn test_transition_zero_duration() {
        let mut transition = LodTransition::new(LodLevel::Lod0, LodLevel::Lod1, 0.0, false);

        transition.advance(0.001);
        assert!(transition.is_complete());
    }

    #[test]
    fn test_empty_instance_update() {
        let mut system = CrowdLodSystem::new(LodConfig::default());
        let mut instances: Vec<CrowdLodInstance> = vec![];

        // Should handle empty slice gracefully
        system.update(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 1.0, &mut instances, 0.016);
        assert_eq!(system.stats().total_instances, 0);
    }

    // ========================================================================
    // Error Tests
    // ========================================================================

    #[test]
    fn test_error_display() {
        let err = LodError::InvalidConfig { reason: "test reason" };
        assert!(err.to_string().contains("test reason"));

        let err = LodError::InvalidDistanceThresholds;
        assert!(err.to_string().contains("monotonically"));

        let err = LodError::MeshSetNotFound { id: 42 };
        assert!(err.to_string().contains("42"));

        let err = LodError::BudgetExceeded { budget: 100, required: 200 };
        assert!(err.to_string().contains("100"));
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_full_pipeline() {
        // Create LOD system with custom config
        let config = LodConfig::new()
            .with_distance_thresholds([15.0, 40.0, 80.0, 150.0])
            .with_hysteresis(2.0)
            .with_polygon_budget(200_000);

        let mut system = CrowdLodSystem::new(config);

        // Register mesh sets
        let soldier_mesh = LodMeshSet::new("soldier")
            .with_lod(LodLevel::Lod0, LodMeshData::new(0, 5000, 15000).with_bones(64, vec![]))
            .with_lod(LodLevel::Lod1, LodMeshData::new(1, 1000, 3000).with_bones(16, vec![]))
            .with_lod(LodLevel::Lod2, LodMeshData::new(2, 4, 6))
            .with_lod(LodLevel::Lod3, LodMeshData::new(3, 1, 0));
        system.register_mesh_set(0, soldier_mesh);

        // Create instances at various distances
        let mut instances: Vec<CrowdLodInstance> = (0..100)
            .map(|i| {
                let distance = (i as f32 / 100.0) * 200.0;
                let priority = if i == 0 { PRIORITY_HERO } else if i < 5 { PRIORITY_IMPORTANT } else { 0 };
                CrowdLodInstance::new([distance, 0.0, 0.0], i as u32)
                    .with_type(0)
                    .with_priority(priority)
            })
            .collect();

        // Update system
        system.update(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 1.0, &mut instances, 0.016);

        // Verify hero stays at LOD0
        assert_eq!(instances[0].lod_level(), LodLevel::Lod0);

        // Verify stats
        let stats = system.stats();
        assert_eq!(stats.total_instances, 100);
        assert!(stats.lod_counts[0] > 0);
        assert!(stats.polygon_count > 0);

        // Simulate multiple frames with transitions
        for _ in 0..60 {
            system.update(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 1.0, &mut instances, 0.016);
        }

        // All transitions should be complete
        assert_eq!(system.transitions().active_count(), 0);
    }

    #[test]
    fn test_camera_movement() {
        let mut system = CrowdLodSystem::new(LodConfig::default());

        let mut instances = vec![
            CrowdLodInstance::new([50.0, 0.0, 0.0], 0),
        ];

        // Camera starts at origin
        system.update(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 1.0, &mut instances, 0.016);
        let initial_lod = instances[0].lod_level();

        // Camera moves closer
        system.update(Vec3::new(45.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0, &mut instances, 0.016);

        // LOD should improve (be lower or equal)
        assert!(instances[0].lod_level() <= initial_lod);
    }

    #[test]
    fn test_spawn_burst_handling() {
        let mut system = CrowdLodSystem::new(LodConfig::default());

        let instance_ids: Vec<u32> = (0..100).collect();

        // Simulate spawn burst
        system.handle_spawn_burst(&instance_ids);

        // Should not have any active transitions for these instances
        for id in instance_ids {
            assert!(!system.transitions().is_transitioning(id));
        }
    }

    #[test]
    fn test_bytemuck_compliance() {
        // Verify Pod and Zeroable for instance struct
        let instance = CrowdLodInstance::default();
        let bytes: &[u8] = bytemuck::bytes_of(&instance);
        assert_eq!(bytes.len(), std::mem::size_of::<CrowdLodInstance>());

        // Should be able to cast slice
        let instances = vec![CrowdLodInstance::default(); 10];
        let _batch_bytes: &[u8] = bytemuck::cast_slice(&instances);
    }
}
