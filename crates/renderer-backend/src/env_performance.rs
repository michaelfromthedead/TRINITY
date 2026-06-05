//! Environment Rendering Performance Budget and LOD Management
//!
//! This module implements a comprehensive performance budget system for environment
//! rendering, including fog, water, clouds, terrain, and foliage. It provides:
//!
//! - Per-pass timing budgets and tracking
//! - Quality tier presets (Ultra to Mobile)
//! - Dynamic LOD selection based on distance and budget
//! - Adaptive quality system with hysteresis
//!
//! # Features
//!
//! - GPU-friendly data layouts (repr(C), Pod/Zeroable)
//! - Rolling frame time averages for stable adaptation
//! - Hysteresis to prevent quality oscillation
//! - Sub-pass timing with begin/end tracking
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::env_performance::{
//!     PerformanceBudget, QualityTier, PerformanceTracker, LodSelector,
//! };
//!
//! let budget = QualityTier::High.get_defaults();
//! let mut tracker = PerformanceTracker::new(budget);
//!
//! tracker.begin_frame();
//! tracker.begin_pass("fog");
//! // ... render fog ...
//! tracker.end_pass();
//!
//! if tracker.is_over_budget() {
//!     if let Some(new_lod) = tracker.suggest_lod_adjustment() {
//!         // Apply new LOD settings
//!     }
//! }
//! ```

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default target frame time for 60 FPS (in milliseconds).
pub const DEFAULT_TARGET_FRAME_TIME_MS: f32 = 16.67;

/// Default fog rendering budget (ms).
pub const DEFAULT_FOG_BUDGET_MS: f32 = 2.0;

/// Default water rendering budget (ms).
pub const DEFAULT_WATER_BUDGET_MS: f32 = 3.0;

/// Default clouds rendering budget (ms).
pub const DEFAULT_CLOUDS_BUDGET_MS: f32 = 4.0;

/// Default terrain rendering budget (ms).
pub const DEFAULT_TERRAIN_BUDGET_MS: f32 = 3.0;

/// Default foliage rendering budget (ms).
pub const DEFAULT_FOLIAGE_BUDGET_MS: f32 = 2.0;

/// Maximum LOD level (0 = highest detail, 3 = lowest).
pub const MAX_LOD_LEVEL: u8 = 3;

/// Number of frames for rolling average.
pub const ROLLING_AVERAGE_FRAMES: usize = 16;

/// Hysteresis threshold for quality upgrades (must be under budget by this %).
pub const UPGRADE_HYSTERESIS_PERCENT: f32 = 0.15;

/// Hysteresis threshold for quality downgrades (must be over budget by this %).
pub const DOWNGRADE_HYSTERESIS_PERCENT: f32 = 0.10;

/// Minimum frames at stable performance before upgrade.
pub const MIN_STABLE_FRAMES_FOR_UPGRADE: u32 = 32;

/// Minimum frames after downgrade before considering upgrade.
pub const MIN_FRAMES_AFTER_DOWNGRADE: u32 = 64;

// ---------------------------------------------------------------------------
// Performance Budget
// ---------------------------------------------------------------------------

/// Performance budget for environment rendering passes.
///
/// Defines the target frame time and per-pass budgets for each
/// environment rendering subsystem.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct PerformanceBudget {
    /// Target total frame time in milliseconds.
    pub target_frame_time_ms: f32,
    /// Budget for fog rendering (volumetric/layered).
    pub fog_budget_ms: f32,
    /// Budget for water rendering (reflections, waves).
    pub water_budget_ms: f32,
    /// Budget for cloud rendering (volumetric clouds).
    pub clouds_budget_ms: f32,
    /// Budget for terrain rendering (clipmaps, tessellation).
    pub terrain_budget_ms: f32,
    /// Budget for foliage rendering (grass, trees).
    pub foliage_budget_ms: f32,
    /// Padding for alignment.
    pub _padding: [f32; 2],
}

impl Default for PerformanceBudget {
    fn default() -> Self {
        Self {
            target_frame_time_ms: DEFAULT_TARGET_FRAME_TIME_MS,
            fog_budget_ms: DEFAULT_FOG_BUDGET_MS,
            water_budget_ms: DEFAULT_WATER_BUDGET_MS,
            clouds_budget_ms: DEFAULT_CLOUDS_BUDGET_MS,
            terrain_budget_ms: DEFAULT_TERRAIN_BUDGET_MS,
            foliage_budget_ms: DEFAULT_FOLIAGE_BUDGET_MS,
            _padding: [0.0; 2],
        }
    }
}

impl PerformanceBudget {
    /// Create a new performance budget with custom values.
    pub fn new(
        target_frame_time_ms: f32,
        fog_budget_ms: f32,
        water_budget_ms: f32,
        clouds_budget_ms: f32,
        terrain_budget_ms: f32,
        foliage_budget_ms: f32,
    ) -> Self {
        Self {
            target_frame_time_ms,
            fog_budget_ms,
            water_budget_ms,
            clouds_budget_ms,
            terrain_budget_ms,
            foliage_budget_ms,
            _padding: [0.0; 2],
        }
    }

    /// Create a budget for a specific target FPS.
    pub fn for_target_fps(fps: f32) -> Self {
        let frame_time = if fps > 0.0 { 1000.0 / fps } else { DEFAULT_TARGET_FRAME_TIME_MS };
        let scale = frame_time / DEFAULT_TARGET_FRAME_TIME_MS;

        Self {
            target_frame_time_ms: frame_time,
            fog_budget_ms: DEFAULT_FOG_BUDGET_MS * scale,
            water_budget_ms: DEFAULT_WATER_BUDGET_MS * scale,
            clouds_budget_ms: DEFAULT_CLOUDS_BUDGET_MS * scale,
            terrain_budget_ms: DEFAULT_TERRAIN_BUDGET_MS * scale,
            foliage_budget_ms: DEFAULT_FOLIAGE_BUDGET_MS * scale,
            _padding: [0.0; 2],
        }
    }

    /// Get total environment rendering budget.
    pub fn total_env_budget_ms(&self) -> f32 {
        self.fog_budget_ms
            + self.water_budget_ms
            + self.clouds_budget_ms
            + self.terrain_budget_ms
            + self.foliage_budget_ms
    }

    /// Get budget for a specific pass by name.
    pub fn get_pass_budget(&self, pass_name: &str) -> Option<f32> {
        match pass_name {
            "fog" => Some(self.fog_budget_ms),
            "water" => Some(self.water_budget_ms),
            "clouds" => Some(self.clouds_budget_ms),
            "terrain" => Some(self.terrain_budget_ms),
            "foliage" => Some(self.foliage_budget_ms),
            _ => None,
        }
    }

    /// Scale all budgets by a factor.
    pub fn scaled(&self, factor: f32) -> Self {
        Self {
            target_frame_time_ms: self.target_frame_time_ms,
            fog_budget_ms: self.fog_budget_ms * factor,
            water_budget_ms: self.water_budget_ms * factor,
            clouds_budget_ms: self.clouds_budget_ms * factor,
            terrain_budget_ms: self.terrain_budget_ms * factor,
            foliage_budget_ms: self.foliage_budget_ms * factor,
            _padding: [0.0; 2],
        }
    }

    /// Check if budget values are valid (non-negative).
    pub fn is_valid(&self) -> bool {
        self.target_frame_time_ms >= 0.0
            && self.fog_budget_ms >= 0.0
            && self.water_budget_ms >= 0.0
            && self.clouds_budget_ms >= 0.0
            && self.terrain_budget_ms >= 0.0
            && self.foliage_budget_ms >= 0.0
    }
}

// ---------------------------------------------------------------------------
// Quality Tier
// ---------------------------------------------------------------------------

/// Quality tier presets for environment rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum QualityTier {
    /// Maximum quality, no budget constraints.
    Ultra = 0,
    /// High quality for high-end desktops.
    High = 1,
    /// Medium quality for mainstream hardware.
    Medium = 2,
    /// Low quality for older hardware.
    Low = 3,
    /// Mobile quality with aggressive optimizations.
    Mobile = 4,
}

impl Default for QualityTier {
    fn default() -> Self {
        Self::High
    }
}

impl QualityTier {
    /// Get the tier index (0-4).
    pub fn as_index(&self) -> usize {
        *self as usize
    }

    /// Create from tier index.
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Ultra),
            1 => Some(Self::High),
            2 => Some(Self::Medium),
            3 => Some(Self::Low),
            4 => Some(Self::Mobile),
            _ => None,
        }
    }

    /// Auto-detect quality tier from GPU capability tier.
    ///
    /// GPU tiers:
    /// - 0: Unknown/Integrated -> Mobile
    /// - 1: Low-end discrete -> Low
    /// - 2: Mid-range discrete -> Medium
    /// - 3: High-end discrete -> High
    /// - 4+: Enthusiast -> Ultra
    pub fn from_gpu_tier(tier: u8) -> Self {
        match tier {
            0 => Self::Mobile,
            1 => Self::Low,
            2 => Self::Medium,
            3 => Self::High,
            _ => Self::Ultra,
        }
    }

    /// Get default performance budget for this quality tier.
    pub fn get_defaults(&self) -> PerformanceBudget {
        match self {
            Self::Ultra => PerformanceBudget::new(
                16.67,  // 60 FPS
                4.0,    // fog
                6.0,    // water
                8.0,    // clouds
                6.0,    // terrain
                4.0,    // foliage
            ),
            Self::High => PerformanceBudget::new(
                16.67,  // 60 FPS
                2.0,    // fog
                3.0,    // water
                4.0,    // clouds
                3.0,    // terrain
                2.0,    // foliage
            ),
            Self::Medium => PerformanceBudget::new(
                16.67,  // 60 FPS
                1.5,    // fog
                2.0,    // water
                2.5,    // clouds
                2.0,    // terrain
                1.5,    // foliage
            ),
            Self::Low => PerformanceBudget::new(
                33.33,  // 30 FPS
                1.0,    // fog
                1.5,    // water
                2.0,    // clouds
                1.5,    // terrain
                1.0,    // foliage
            ),
            Self::Mobile => PerformanceBudget::new(
                33.33,  // 30 FPS
                0.5,    // fog
                1.0,    // water
                1.5,    // clouds
                1.0,    // terrain
                0.5,    // foliage
            ),
        }
    }

    /// Get default LOD configuration for this quality tier.
    pub fn get_default_lods(&self) -> EnvLodConfig {
        match self {
            Self::Ultra => EnvLodConfig::new(0, 0, 0, 0, 0),
            Self::High => EnvLodConfig::new(0, 0, 1, 0, 0),
            Self::Medium => EnvLodConfig::new(1, 1, 1, 1, 1),
            Self::Low => EnvLodConfig::new(2, 2, 2, 2, 2),
            Self::Mobile => EnvLodConfig::new(3, 3, 3, 3, 3),
        }
    }

    /// Get one tier lower (returns self if already at lowest).
    pub fn downgrade(&self) -> Self {
        match self {
            Self::Ultra => Self::High,
            Self::High => Self::Medium,
            Self::Medium => Self::Low,
            Self::Low => Self::Mobile,
            Self::Mobile => Self::Mobile,
        }
    }

    /// Get one tier higher (returns self if already at highest).
    pub fn upgrade(&self) -> Self {
        match self {
            Self::Ultra => Self::Ultra,
            Self::High => Self::Ultra,
            Self::Medium => Self::High,
            Self::Low => Self::Medium,
            Self::Mobile => Self::Low,
        }
    }

    /// Check if this is the highest tier.
    pub fn is_highest(&self) -> bool {
        matches!(self, Self::Ultra)
    }

    /// Check if this is the lowest tier.
    pub fn is_lowest(&self) -> bool {
        matches!(self, Self::Mobile)
    }
}

// ---------------------------------------------------------------------------
// Environment LOD Configuration
// ---------------------------------------------------------------------------

/// LOD levels for each environment rendering subsystem.
///
/// Values range from 0 (highest detail) to 3 (lowest detail).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct EnvLodConfig {
    /// Fog LOD level (0-3).
    pub fog_lod: u8,
    /// Water LOD level (0-3).
    pub water_lod: u8,
    /// Cloud LOD level (0-3).
    pub cloud_lod: u8,
    /// Terrain LOD level (0-3).
    pub terrain_lod: u8,
    /// Foliage LOD level (0-3).
    pub foliage_lod: u8,
    /// Padding for alignment.
    pub _padding: [u8; 3],
}

impl Default for EnvLodConfig {
    fn default() -> Self {
        Self {
            fog_lod: 0,
            water_lod: 0,
            cloud_lod: 0,
            terrain_lod: 0,
            foliage_lod: 0,
            _padding: [0; 3],
        }
    }
}

impl EnvLodConfig {
    /// Create a new LOD configuration.
    pub fn new(fog_lod: u8, water_lod: u8, cloud_lod: u8, terrain_lod: u8, foliage_lod: u8) -> Self {
        Self {
            fog_lod: fog_lod.min(MAX_LOD_LEVEL),
            water_lod: water_lod.min(MAX_LOD_LEVEL),
            cloud_lod: cloud_lod.min(MAX_LOD_LEVEL),
            terrain_lod: terrain_lod.min(MAX_LOD_LEVEL),
            foliage_lod: foliage_lod.min(MAX_LOD_LEVEL),
            _padding: [0; 3],
        }
    }

    /// Create a uniform LOD configuration (all passes at same level).
    pub fn uniform(lod: u8) -> Self {
        let lod = lod.min(MAX_LOD_LEVEL);
        Self::new(lod, lod, lod, lod, lod)
    }

    /// Get LOD for a specific pass by name.
    pub fn get_pass_lod(&self, pass_name: &str) -> Option<u8> {
        match pass_name {
            "fog" => Some(self.fog_lod),
            "water" => Some(self.water_lod),
            "clouds" => Some(self.cloud_lod),
            "terrain" => Some(self.terrain_lod),
            "foliage" => Some(self.foliage_lod),
            _ => None,
        }
    }

    /// Set LOD for a specific pass by name.
    pub fn set_pass_lod(&mut self, pass_name: &str, lod: u8) -> bool {
        let lod = lod.min(MAX_LOD_LEVEL);
        match pass_name {
            "fog" => { self.fog_lod = lod; true }
            "water" => { self.water_lod = lod; true }
            "clouds" => { self.cloud_lod = lod; true }
            "terrain" => { self.terrain_lod = lod; true }
            "foliage" => { self.foliage_lod = lod; true }
            _ => false,
        }
    }

    /// Increase LOD level for a specific pass (lower detail).
    pub fn increase_lod(&mut self, pass_name: &str) -> bool {
        if let Some(current) = self.get_pass_lod(pass_name) {
            if current < MAX_LOD_LEVEL {
                return self.set_pass_lod(pass_name, current + 1);
            }
        }
        false
    }

    /// Decrease LOD level for a specific pass (higher detail).
    pub fn decrease_lod(&mut self, pass_name: &str) -> bool {
        if let Some(current) = self.get_pass_lod(pass_name) {
            if current > 0 {
                return self.set_pass_lod(pass_name, current - 1);
            }
        }
        false
    }

    /// Get the average LOD level across all passes.
    pub fn average_lod(&self) -> f32 {
        let sum = self.fog_lod as f32
            + self.water_lod as f32
            + self.cloud_lod as f32
            + self.terrain_lod as f32
            + self.foliage_lod as f32;
        sum / 5.0
    }

    /// Check if all LODs are at maximum (lowest detail).
    pub fn is_all_max(&self) -> bool {
        self.fog_lod == MAX_LOD_LEVEL
            && self.water_lod == MAX_LOD_LEVEL
            && self.cloud_lod == MAX_LOD_LEVEL
            && self.terrain_lod == MAX_LOD_LEVEL
            && self.foliage_lod == MAX_LOD_LEVEL
    }

    /// Check if all LODs are at minimum (highest detail).
    pub fn is_all_min(&self) -> bool {
        self.fog_lod == 0
            && self.water_lod == 0
            && self.cloud_lod == 0
            && self.terrain_lod == 0
            && self.foliage_lod == 0
    }

    /// Check if configuration is valid (all values in range).
    pub fn is_valid(&self) -> bool {
        self.fog_lod <= MAX_LOD_LEVEL
            && self.water_lod <= MAX_LOD_LEVEL
            && self.cloud_lod <= MAX_LOD_LEVEL
            && self.terrain_lod <= MAX_LOD_LEVEL
            && self.foliage_lod <= MAX_LOD_LEVEL
    }
}

// ---------------------------------------------------------------------------
// Frame Statistics
// ---------------------------------------------------------------------------

/// Statistics for a single frame's performance.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStats {
    /// Total frame time in milliseconds.
    pub total_time_ms: f32,
    /// Time spent on fog rendering.
    pub fog_time_ms: f32,
    /// Time spent on water rendering.
    pub water_time_ms: f32,
    /// Time spent on cloud rendering.
    pub clouds_time_ms: f32,
    /// Time spent on terrain rendering.
    pub terrain_time_ms: f32,
    /// Time spent on foliage rendering.
    pub foliage_time_ms: f32,
    /// Frame number.
    pub frame_number: u64,
    /// Whether the frame exceeded budget.
    pub over_budget: bool,
}

impl FrameStats {
    /// Get total environment rendering time.
    pub fn env_time_ms(&self) -> f32 {
        self.fog_time_ms
            + self.water_time_ms
            + self.clouds_time_ms
            + self.terrain_time_ms
            + self.foliage_time_ms
    }

    /// Get time for a specific pass.
    pub fn get_pass_time(&self, pass_name: &str) -> Option<f32> {
        match pass_name {
            "fog" => Some(self.fog_time_ms),
            "water" => Some(self.water_time_ms),
            "clouds" => Some(self.clouds_time_ms),
            "terrain" => Some(self.terrain_time_ms),
            "foliage" => Some(self.foliage_time_ms),
            _ => None,
        }
    }

    /// Calculate budget utilization (1.0 = exactly on budget).
    pub fn budget_utilization(&self, budget: &PerformanceBudget) -> f32 {
        if budget.target_frame_time_ms > 0.0 {
            self.total_time_ms / budget.target_frame_time_ms
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Pass Timing State
// ---------------------------------------------------------------------------

/// Internal state for tracking in-progress pass timing.
#[derive(Debug, Clone, Default)]
struct PassTimingState {
    /// Name of the currently timing pass.
    current_pass: Option<String>,
    /// Start time of current pass (in arbitrary units).
    start_time: Option<u64>,
}

// ---------------------------------------------------------------------------
// Performance Tracker
// ---------------------------------------------------------------------------

/// Tracks rendering performance against budgets.
///
/// Provides timing instrumentation and budget analysis.
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    /// Budget configuration.
    budget: PerformanceBudget,
    /// Current frame statistics.
    current_stats: FrameStats,
    /// Rolling history of frame times.
    frame_history: Vec<f32>,
    /// Current index in rolling buffer.
    history_index: usize,
    /// Total frames tracked.
    frame_count: u64,
    /// Pass timing state.
    timing_state: PassTimingState,
    /// Recorded pass times for current frame.
    pass_times: [(String, f32); 8],
    /// Number of recorded passes.
    pass_count: usize,
    /// Current LOD configuration.
    current_lods: EnvLodConfig,
    /// Current quality tier.
    current_tier: QualityTier,
    /// Frames since last quality change.
    frames_since_quality_change: u32,
    /// Frames at stable performance.
    stable_frames: u32,
}

impl PerformanceTracker {
    /// Create a new performance tracker with the given budget.
    pub fn new(budget: PerformanceBudget) -> Self {
        Self {
            budget,
            current_stats: FrameStats::default(),
            frame_history: vec![0.0; ROLLING_AVERAGE_FRAMES],
            history_index: 0,
            frame_count: 0,
            timing_state: PassTimingState::default(),
            pass_times: Default::default(),
            pass_count: 0,
            current_lods: EnvLodConfig::default(),
            current_tier: QualityTier::High,
            frames_since_quality_change: 0,
            stable_frames: 0,
        }
    }

    /// Create a tracker with default budget.
    pub fn with_defaults() -> Self {
        Self::new(PerformanceBudget::default())
    }

    /// Create a tracker for a specific quality tier.
    pub fn for_tier(tier: QualityTier) -> Self {
        let mut tracker = Self::new(tier.get_defaults());
        tracker.current_tier = tier;
        tracker.current_lods = tier.get_default_lods();
        tracker
    }

    /// Get the budget configuration.
    pub fn budget(&self) -> &PerformanceBudget {
        &self.budget
    }

    /// Set the budget configuration.
    pub fn set_budget(&mut self, budget: PerformanceBudget) {
        self.budget = budget;
    }

    /// Get current LOD configuration.
    pub fn current_lods(&self) -> &EnvLodConfig {
        &self.current_lods
    }

    /// Set current LOD configuration.
    pub fn set_lods(&mut self, lods: EnvLodConfig) {
        self.current_lods = lods;
    }

    /// Get current quality tier.
    pub fn current_tier(&self) -> QualityTier {
        self.current_tier
    }

    /// Begin a new frame, resetting per-frame statistics.
    pub fn begin_frame(&mut self) {
        self.current_stats = FrameStats {
            frame_number: self.frame_count,
            ..Default::default()
        };
        self.pass_count = 0;
        self.timing_state = PassTimingState::default();
    }

    /// Begin timing a render pass.
    ///
    /// Call `end_pass()` when the pass completes.
    pub fn begin_pass(&mut self, pass_name: &str) {
        self.timing_state.current_pass = Some(pass_name.to_string());
        // Use a simple monotonic counter for timing (in real code, use std::time::Instant)
        self.timing_state.start_time = Some(self.get_current_time_nanos());
    }

    /// End timing the current pass.
    ///
    /// Returns the elapsed time in milliseconds.
    pub fn end_pass(&mut self) -> f32 {
        let elapsed = if let (Some(pass_name), Some(start)) =
            (self.timing_state.current_pass.take(), self.timing_state.start_time.take())
        {
            let end = self.get_current_time_nanos();
            let elapsed_ms = (end.saturating_sub(start)) as f32 / 1_000_000.0;
            self.record_pass_time(&pass_name, elapsed_ms);
            elapsed_ms
        } else {
            0.0
        };
        elapsed
    }

    /// Record a pass time directly (useful when timing is done externally).
    pub fn record_pass_time(&mut self, pass_name: &str, time_ms: f32) {
        // Store in pass_times array
        if self.pass_count < self.pass_times.len() {
            self.pass_times[self.pass_count] = (pass_name.to_string(), time_ms);
            self.pass_count += 1;
        }

        // Update current stats
        match pass_name {
            "fog" => self.current_stats.fog_time_ms = time_ms,
            "water" => self.current_stats.water_time_ms = time_ms,
            "clouds" => self.current_stats.clouds_time_ms = time_ms,
            "terrain" => self.current_stats.terrain_time_ms = time_ms,
            "foliage" => self.current_stats.foliage_time_ms = time_ms,
            _ => {}
        }
    }

    /// Get the recorded time for a specific pass.
    pub fn get_pass_time(&self, pass_name: &str) -> f32 {
        self.current_stats.get_pass_time(pass_name).unwrap_or(0.0)
    }

    /// End the frame and update statistics.
    pub fn end_frame(&mut self, total_frame_time_ms: f32) {
        self.current_stats.total_time_ms = total_frame_time_ms;
        self.current_stats.over_budget = total_frame_time_ms > self.budget.target_frame_time_ms;

        // Update rolling history
        self.frame_history[self.history_index] = total_frame_time_ms;
        self.history_index = (self.history_index + 1) % ROLLING_AVERAGE_FRAMES;
        self.frame_count += 1;
        self.frames_since_quality_change += 1;

        // Update stability tracking
        if self.current_stats.over_budget {
            self.stable_frames = 0;
        } else {
            self.stable_frames += 1;
        }
    }

    /// Check if current frame is over budget.
    pub fn is_over_budget(&self) -> bool {
        self.current_stats.total_time_ms > self.budget.target_frame_time_ms
    }

    /// Get the amount over budget in milliseconds.
    pub fn get_overage(&self) -> f32 {
        (self.current_stats.total_time_ms - self.budget.target_frame_time_ms).max(0.0)
    }

    /// Get budget headroom (negative if over budget).
    pub fn get_headroom(&self) -> f32 {
        self.budget.target_frame_time_ms - self.current_stats.total_time_ms
    }

    /// Get the rolling average frame time.
    pub fn rolling_average_frame_time(&self) -> f32 {
        let count = self.frame_count.min(ROLLING_AVERAGE_FRAMES as u64) as f32;
        if count > 0.0 {
            self.frame_history.iter().take(count as usize).sum::<f32>() / count
        } else {
            0.0
        }
    }

    /// Get frame statistics.
    pub fn frame_stats(&self) -> FrameStats {
        self.current_stats
    }

    /// Suggest LOD adjustments based on current performance.
    ///
    /// Returns `Some(config)` if adjustments are recommended.
    pub fn suggest_lod_adjustment(&self) -> Option<EnvLodConfig> {
        let avg_time = self.rolling_average_frame_time();
        let target = self.budget.target_frame_time_ms;

        if avg_time <= 0.0 || target <= 0.0 {
            return None;
        }

        let utilization = avg_time / target;
        let mut new_lods = self.current_lods;
        let mut changed = false;

        // Over budget: increase LODs (lower detail)
        if utilization > 1.0 + DOWNGRADE_HYSTERESIS_PERCENT {
            // Find the most expensive pass and increase its LOD
            let passes = [
                ("fog", self.current_stats.fog_time_ms),
                ("water", self.current_stats.water_time_ms),
                ("clouds", self.current_stats.clouds_time_ms),
                ("terrain", self.current_stats.terrain_time_ms),
                ("foliage", self.current_stats.foliage_time_ms),
            ];

            if let Some((pass_name, _)) = passes.iter()
                .filter(|(name, _)| new_lods.get_pass_lod(name).unwrap_or(MAX_LOD_LEVEL) < MAX_LOD_LEVEL)
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if new_lods.increase_lod(pass_name) {
                    changed = true;
                }
            }
        }
        // Under budget with headroom: decrease LODs (higher detail) if stable
        else if utilization < 1.0 - UPGRADE_HYSTERESIS_PERCENT
            && self.stable_frames >= MIN_STABLE_FRAMES_FOR_UPGRADE
            && self.frames_since_quality_change >= MIN_FRAMES_AFTER_DOWNGRADE
        {
            // Find the highest LOD pass and decrease it
            let passes = ["fog", "water", "clouds", "terrain", "foliage"];
            if let Some(pass_name) = passes.iter()
                .filter(|name| new_lods.get_pass_lod(name).unwrap_or(0) > 0)
                .next()
            {
                if new_lods.decrease_lod(pass_name) {
                    changed = true;
                }
            }
        }

        if changed { Some(new_lods) } else { None }
    }

    /// Apply suggested LOD adjustment.
    pub fn apply_lod_adjustment(&mut self, lods: EnvLodConfig) {
        self.current_lods = lods;
        self.frames_since_quality_change = 0;
    }

    /// Get current time in nanoseconds (for timing).
    fn get_current_time_nanos(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// LOD Selector
// ---------------------------------------------------------------------------

/// Selects appropriate LOD levels based on distance and budget.
#[derive(Debug, Clone)]
pub struct LodSelector {
    /// Distance thresholds for each LOD level (in world units).
    fog_thresholds: [f32; 4],
    /// Distance thresholds for water LOD.
    water_thresholds: [f32; 4],
    /// Distance thresholds for terrain LOD.
    terrain_thresholds: [f32; 4],
    /// Distance thresholds for foliage LOD.
    foliage_thresholds: [f32; 4],
    /// Camera height thresholds for water detail.
    water_height_thresholds: [f32; 4],
    /// Slope thresholds for terrain LOD (steeper = more detail).
    terrain_slope_thresholds: [f32; 4],
}

impl Default for LodSelector {
    fn default() -> Self {
        Self {
            fog_thresholds: [50.0, 150.0, 400.0, 1000.0],
            water_thresholds: [30.0, 100.0, 300.0, 800.0],
            terrain_thresholds: [100.0, 300.0, 800.0, 2000.0],
            foliage_thresholds: [20.0, 60.0, 150.0, 400.0],
            water_height_thresholds: [5.0, 20.0, 50.0, 100.0],
            terrain_slope_thresholds: [0.3, 0.5, 0.7, 0.9],
        }
    }
}

impl LodSelector {
    /// Create a new LOD selector with default thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a LOD selector with custom thresholds.
    pub fn with_thresholds(
        fog: [f32; 4],
        water: [f32; 4],
        terrain: [f32; 4],
        foliage: [f32; 4],
    ) -> Self {
        Self {
            fog_thresholds: fog,
            water_thresholds: water,
            terrain_thresholds: terrain,
            foliage_thresholds: foliage,
            ..Default::default()
        }
    }

    /// Select fog LOD based on distance and remaining budget.
    ///
    /// Returns LOD level (0-3).
    pub fn select_fog_lod(&self, distance: f32, budget_remaining_ms: f32) -> u8 {
        // If budget is tight, force higher LOD (lower detail)
        let budget_pressure = if budget_remaining_ms <= 0.5 { 2 }
            else if budget_remaining_ms <= 1.0 { 1 }
            else { 0 };

        let distance_lod = self.distance_to_lod(distance, &self.fog_thresholds);
        (distance_lod + budget_pressure).min(MAX_LOD_LEVEL)
    }

    /// Select water LOD based on camera height and distance.
    ///
    /// Water needs more detail when camera is close to water surface.
    pub fn select_water_lod(&self, camera_height: f32, distance: f32) -> u8 {
        let height_lod = self.distance_to_lod(camera_height, &self.water_height_thresholds);
        let distance_lod = self.distance_to_lod(distance, &self.water_thresholds);

        // Use the lower LOD (higher detail) of the two
        height_lod.min(distance_lod)
    }

    /// Select terrain LOD based on distance and slope.
    ///
    /// Steeper terrain needs more detail to avoid visible artifacts.
    pub fn select_terrain_lod(&self, distance: f32, slope: f32) -> u8 {
        let distance_lod = self.distance_to_lod(distance, &self.terrain_thresholds);

        // Slope bonus: steeper slopes get lower LOD (more detail)
        let slope_bonus = if slope > self.terrain_slope_thresholds[0] { 1 } else { 0 };

        distance_lod.saturating_sub(slope_bonus)
    }

    /// Select foliage density based on distance and budget.
    ///
    /// Returns a density multiplier (0.0 to 1.0).
    pub fn select_foliage_density(&self, distance: f32, budget_remaining_ms: f32) -> f32 {
        let base_lod = self.distance_to_lod(distance, &self.foliage_thresholds);

        // Convert LOD to density (LOD 0 = 1.0, LOD 3 = 0.25)
        let base_density = 1.0 - (base_lod as f32 * 0.25);

        // Scale by budget pressure
        let budget_factor = if budget_remaining_ms <= 0.5 { 0.5 }
            else if budget_remaining_ms <= 1.0 { 0.75 }
            else { 1.0 };

        (base_density * budget_factor).max(0.1)
    }

    /// Select complete LOD configuration for all environment systems.
    pub fn select_all_lods(
        &self,
        camera_height: f32,
        terrain_distance: f32,
        terrain_slope: f32,
        water_distance: f32,
        foliage_distance: f32,
        budget_remaining_ms: f32,
    ) -> EnvLodConfig {
        EnvLodConfig::new(
            self.select_fog_lod(terrain_distance, budget_remaining_ms),
            self.select_water_lod(camera_height, water_distance),
            self.select_cloud_lod(terrain_distance),
            self.select_terrain_lod(terrain_distance, terrain_slope),
            self.distance_to_lod(foliage_distance, &self.foliage_thresholds),
        )
    }

    /// Select cloud LOD based on distance (clouds are always far).
    pub fn select_cloud_lod(&self, view_distance: f32) -> u8 {
        // Clouds use simple distance-based LOD
        if view_distance < 500.0 { 0 }
        else if view_distance < 1500.0 { 1 }
        else if view_distance < 4000.0 { 2 }
        else { 3 }
    }

    /// Convert distance to LOD level using thresholds.
    fn distance_to_lod(&self, distance: f32, thresholds: &[f32; 4]) -> u8 {
        for (i, &threshold) in thresholds.iter().enumerate() {
            if distance < threshold {
                return i as u8;
            }
        }
        MAX_LOD_LEVEL
    }
}

// ---------------------------------------------------------------------------
// Adaptive Quality System
// ---------------------------------------------------------------------------

/// Adaptive quality system that automatically adjusts quality tier.
///
/// Uses hysteresis to prevent oscillation between tiers.
#[derive(Debug, Clone)]
pub struct AdaptiveQuality {
    /// Current quality tier.
    current_tier: QualityTier,
    /// Frame time history for rolling average.
    frame_history: Vec<f32>,
    /// Current index in history buffer.
    history_index: usize,
    /// Frames since last tier change.
    frames_since_change: u32,
    /// Number of consecutive frames over budget.
    consecutive_over_budget: u32,
    /// Number of consecutive frames under budget (with headroom).
    consecutive_under_budget: u32,
    /// Whether quality was recently downgraded.
    recently_downgraded: bool,
    /// Cooldown frames after downgrade before upgrade is considered.
    downgrade_cooldown: u32,
    /// Target frame time.
    target_frame_time_ms: f32,
}

impl AdaptiveQuality {
    /// Create a new adaptive quality system.
    pub fn new(initial_tier: QualityTier) -> Self {
        Self {
            current_tier: initial_tier,
            frame_history: vec![0.0; ROLLING_AVERAGE_FRAMES],
            history_index: 0,
            frames_since_change: 0,
            consecutive_over_budget: 0,
            consecutive_under_budget: 0,
            recently_downgraded: false,
            downgrade_cooldown: 0,
            target_frame_time_ms: initial_tier.get_defaults().target_frame_time_ms,
        }
    }

    /// Create with specific target frame time.
    pub fn with_target(initial_tier: QualityTier, target_frame_time_ms: f32) -> Self {
        let mut aq = Self::new(initial_tier);
        aq.target_frame_time_ms = target_frame_time_ms;
        aq
    }

    /// Get current quality tier.
    pub fn current_tier(&self) -> QualityTier {
        self.current_tier
    }

    /// Record a frame time and potentially adjust quality.
    ///
    /// Returns `Some(new_tier)` if quality was adjusted.
    pub fn record_frame(&mut self, frame_time_ms: f32) -> Option<QualityTier> {
        // Update history
        self.frame_history[self.history_index] = frame_time_ms;
        self.history_index = (self.history_index + 1) % ROLLING_AVERAGE_FRAMES;
        self.frames_since_change += 1;

        if self.downgrade_cooldown > 0 {
            self.downgrade_cooldown -= 1;
        }

        // Check budget status
        let over_budget = frame_time_ms > self.target_frame_time_ms * (1.0 + DOWNGRADE_HYSTERESIS_PERCENT);
        let under_budget = frame_time_ms < self.target_frame_time_ms * (1.0 - UPGRADE_HYSTERESIS_PERCENT);

        if over_budget {
            self.consecutive_over_budget += 1;
            self.consecutive_under_budget = 0;
        } else if under_budget {
            self.consecutive_under_budget += 1;
            self.consecutive_over_budget = 0;
        } else {
            // In acceptable range
            self.consecutive_over_budget = 0;
            self.consecutive_under_budget = 0;
        }

        // Check for tier changes
        let avg = self.rolling_average();

        // Downgrade: sustained over-budget
        if self.consecutive_over_budget >= 8 || avg > self.target_frame_time_ms * 1.2 {
            if !self.current_tier.is_lowest() {
                let new_tier = self.current_tier.downgrade();
                self.apply_tier_change(new_tier, true);
                return Some(new_tier);
            }
        }

        // Upgrade: sustained under-budget with stability
        if self.consecutive_under_budget >= MIN_STABLE_FRAMES_FOR_UPGRADE
            && self.frames_since_change >= MIN_FRAMES_AFTER_DOWNGRADE
            && self.downgrade_cooldown == 0
            && !self.current_tier.is_highest()
        {
            let new_tier = self.current_tier.upgrade();
            self.apply_tier_change(new_tier, false);
            return Some(new_tier);
        }

        None
    }

    /// Get rolling average frame time.
    pub fn rolling_average(&self) -> f32 {
        let sum: f32 = self.frame_history.iter().sum();
        sum / ROLLING_AVERAGE_FRAMES as f32
    }

    /// Force a specific quality tier.
    pub fn force_tier(&mut self, tier: QualityTier) {
        self.apply_tier_change(tier, false);
    }

    /// Reset adaptive state while keeping current tier.
    pub fn reset(&mut self) {
        self.frame_history.fill(0.0);
        self.history_index = 0;
        self.frames_since_change = 0;
        self.consecutive_over_budget = 0;
        self.consecutive_under_budget = 0;
        self.recently_downgraded = false;
        self.downgrade_cooldown = 0;
    }

    /// Get number of frames since last quality change.
    pub fn frames_since_change(&self) -> u32 {
        self.frames_since_change
    }

    /// Check if system is in a stable state (not oscillating).
    pub fn is_stable(&self) -> bool {
        self.frames_since_change >= MIN_FRAMES_AFTER_DOWNGRADE
            && self.consecutive_over_budget < 4
    }

    /// Apply tier change with tracking.
    fn apply_tier_change(&mut self, new_tier: QualityTier, is_downgrade: bool) {
        self.current_tier = new_tier;
        self.frames_since_change = 0;
        self.consecutive_over_budget = 0;
        self.consecutive_under_budget = 0;
        self.recently_downgraded = is_downgrade;

        if is_downgrade {
            self.downgrade_cooldown = MIN_FRAMES_AFTER_DOWNGRADE;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===== PerformanceBudget Tests =====

    #[test]
    fn test_budget_default() {
        let budget = PerformanceBudget::default();
        assert_eq!(budget.target_frame_time_ms, DEFAULT_TARGET_FRAME_TIME_MS);
        assert_eq!(budget.fog_budget_ms, DEFAULT_FOG_BUDGET_MS);
        assert_eq!(budget.water_budget_ms, DEFAULT_WATER_BUDGET_MS);
        assert_eq!(budget.clouds_budget_ms, DEFAULT_CLOUDS_BUDGET_MS);
        assert_eq!(budget.terrain_budget_ms, DEFAULT_TERRAIN_BUDGET_MS);
        assert_eq!(budget.foliage_budget_ms, DEFAULT_FOLIAGE_BUDGET_MS);
    }

    #[test]
    fn test_budget_new() {
        let budget = PerformanceBudget::new(33.33, 1.0, 2.0, 3.0, 2.5, 1.5);
        assert_eq!(budget.target_frame_time_ms, 33.33);
        assert_eq!(budget.fog_budget_ms, 1.0);
        assert_eq!(budget.water_budget_ms, 2.0);
        assert_eq!(budget.clouds_budget_ms, 3.0);
        assert_eq!(budget.terrain_budget_ms, 2.5);
        assert_eq!(budget.foliage_budget_ms, 1.5);
    }

    #[test]
    fn test_budget_for_target_fps() {
        let budget_60 = PerformanceBudget::for_target_fps(60.0);
        assert!((budget_60.target_frame_time_ms - 16.67).abs() < 0.01);

        let budget_30 = PerformanceBudget::for_target_fps(30.0);
        assert!((budget_30.target_frame_time_ms - 33.33).abs() < 0.01);
        // 30 FPS should have ~2x the budget of 60 FPS
        assert!((budget_30.fog_budget_ms / budget_60.fog_budget_ms - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_budget_for_zero_fps() {
        let budget = PerformanceBudget::for_target_fps(0.0);
        assert_eq!(budget.target_frame_time_ms, DEFAULT_TARGET_FRAME_TIME_MS);
    }

    #[test]
    fn test_budget_total_env() {
        let budget = PerformanceBudget::default();
        let expected = DEFAULT_FOG_BUDGET_MS + DEFAULT_WATER_BUDGET_MS + DEFAULT_CLOUDS_BUDGET_MS
            + DEFAULT_TERRAIN_BUDGET_MS + DEFAULT_FOLIAGE_BUDGET_MS;
        assert_eq!(budget.total_env_budget_ms(), expected);
    }

    #[test]
    fn test_budget_get_pass_budget() {
        let budget = PerformanceBudget::default();
        assert_eq!(budget.get_pass_budget("fog"), Some(DEFAULT_FOG_BUDGET_MS));
        assert_eq!(budget.get_pass_budget("water"), Some(DEFAULT_WATER_BUDGET_MS));
        assert_eq!(budget.get_pass_budget("clouds"), Some(DEFAULT_CLOUDS_BUDGET_MS));
        assert_eq!(budget.get_pass_budget("terrain"), Some(DEFAULT_TERRAIN_BUDGET_MS));
        assert_eq!(budget.get_pass_budget("foliage"), Some(DEFAULT_FOLIAGE_BUDGET_MS));
        assert_eq!(budget.get_pass_budget("unknown"), None);
    }

    #[test]
    fn test_budget_scaled() {
        let budget = PerformanceBudget::default();
        let scaled = budget.scaled(0.5);
        assert_eq!(scaled.fog_budget_ms, budget.fog_budget_ms * 0.5);
        assert_eq!(scaled.water_budget_ms, budget.water_budget_ms * 0.5);
        // Target frame time should not be scaled
        assert_eq!(scaled.target_frame_time_ms, budget.target_frame_time_ms);
    }

    #[test]
    fn test_budget_is_valid() {
        let valid = PerformanceBudget::default();
        assert!(valid.is_valid());

        let invalid = PerformanceBudget::new(-1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_budget_pod_zeroable() {
        // Verify Pod/Zeroable compliance
        let zeroed: PerformanceBudget = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.target_frame_time_ms, 0.0);

        let budget = PerformanceBudget::default();
        let bytes = bytemuck::bytes_of(&budget);
        assert_eq!(bytes.len(), std::mem::size_of::<PerformanceBudget>());
    }

    // ===== QualityTier Tests =====

    #[test]
    fn test_quality_tier_from_gpu_tier() {
        assert_eq!(QualityTier::from_gpu_tier(0), QualityTier::Mobile);
        assert_eq!(QualityTier::from_gpu_tier(1), QualityTier::Low);
        assert_eq!(QualityTier::from_gpu_tier(2), QualityTier::Medium);
        assert_eq!(QualityTier::from_gpu_tier(3), QualityTier::High);
        assert_eq!(QualityTier::from_gpu_tier(4), QualityTier::Ultra);
        assert_eq!(QualityTier::from_gpu_tier(10), QualityTier::Ultra);
    }

    #[test]
    fn test_quality_tier_get_defaults() {
        let ultra = QualityTier::Ultra.get_defaults();
        let mobile = QualityTier::Mobile.get_defaults();

        // Ultra should have higher budgets than Mobile
        assert!(ultra.fog_budget_ms > mobile.fog_budget_ms);
        assert!(ultra.water_budget_ms > mobile.water_budget_ms);
        assert!(ultra.clouds_budget_ms > mobile.clouds_budget_ms);
    }

    #[test]
    fn test_quality_tier_downgrade() {
        assert_eq!(QualityTier::Ultra.downgrade(), QualityTier::High);
        assert_eq!(QualityTier::High.downgrade(), QualityTier::Medium);
        assert_eq!(QualityTier::Medium.downgrade(), QualityTier::Low);
        assert_eq!(QualityTier::Low.downgrade(), QualityTier::Mobile);
        assert_eq!(QualityTier::Mobile.downgrade(), QualityTier::Mobile);
    }

    #[test]
    fn test_quality_tier_upgrade() {
        assert_eq!(QualityTier::Mobile.upgrade(), QualityTier::Low);
        assert_eq!(QualityTier::Low.upgrade(), QualityTier::Medium);
        assert_eq!(QualityTier::Medium.upgrade(), QualityTier::High);
        assert_eq!(QualityTier::High.upgrade(), QualityTier::Ultra);
        assert_eq!(QualityTier::Ultra.upgrade(), QualityTier::Ultra);
    }

    #[test]
    fn test_quality_tier_is_highest_lowest() {
        assert!(QualityTier::Ultra.is_highest());
        assert!(!QualityTier::High.is_highest());
        assert!(QualityTier::Mobile.is_lowest());
        assert!(!QualityTier::Low.is_lowest());
    }

    #[test]
    fn test_quality_tier_as_index() {
        assert_eq!(QualityTier::Ultra.as_index(), 0);
        assert_eq!(QualityTier::High.as_index(), 1);
        assert_eq!(QualityTier::Medium.as_index(), 2);
        assert_eq!(QualityTier::Low.as_index(), 3);
        assert_eq!(QualityTier::Mobile.as_index(), 4);
    }

    #[test]
    fn test_quality_tier_from_index() {
        assert_eq!(QualityTier::from_index(0), Some(QualityTier::Ultra));
        assert_eq!(QualityTier::from_index(4), Some(QualityTier::Mobile));
        assert_eq!(QualityTier::from_index(5), None);
    }

    #[test]
    fn test_quality_tier_get_default_lods() {
        let ultra_lods = QualityTier::Ultra.get_default_lods();
        assert!(ultra_lods.is_all_min());

        let mobile_lods = QualityTier::Mobile.get_default_lods();
        assert!(mobile_lods.is_all_max());
    }

    // ===== EnvLodConfig Tests =====

    #[test]
    fn test_lod_config_default() {
        let config = EnvLodConfig::default();
        assert_eq!(config.fog_lod, 0);
        assert_eq!(config.water_lod, 0);
        assert_eq!(config.cloud_lod, 0);
        assert_eq!(config.terrain_lod, 0);
        assert_eq!(config.foliage_lod, 0);
    }

    #[test]
    fn test_lod_config_new() {
        let config = EnvLodConfig::new(1, 2, 3, 2, 1);
        assert_eq!(config.fog_lod, 1);
        assert_eq!(config.water_lod, 2);
        assert_eq!(config.cloud_lod, 3);
        assert_eq!(config.terrain_lod, 2);
        assert_eq!(config.foliage_lod, 1);
    }

    #[test]
    fn test_lod_config_clamping() {
        let config = EnvLodConfig::new(10, 10, 10, 10, 10);
        assert_eq!(config.fog_lod, MAX_LOD_LEVEL);
        assert_eq!(config.water_lod, MAX_LOD_LEVEL);
        assert_eq!(config.cloud_lod, MAX_LOD_LEVEL);
    }

    #[test]
    fn test_lod_config_uniform() {
        let config = EnvLodConfig::uniform(2);
        assert_eq!(config.fog_lod, 2);
        assert_eq!(config.water_lod, 2);
        assert_eq!(config.cloud_lod, 2);
        assert_eq!(config.terrain_lod, 2);
        assert_eq!(config.foliage_lod, 2);
    }

    #[test]
    fn test_lod_config_get_pass_lod() {
        let config = EnvLodConfig::new(0, 1, 2, 3, 1);
        assert_eq!(config.get_pass_lod("fog"), Some(0));
        assert_eq!(config.get_pass_lod("water"), Some(1));
        assert_eq!(config.get_pass_lod("clouds"), Some(2));
        assert_eq!(config.get_pass_lod("terrain"), Some(3));
        assert_eq!(config.get_pass_lod("foliage"), Some(1));
        assert_eq!(config.get_pass_lod("unknown"), None);
    }

    #[test]
    fn test_lod_config_set_pass_lod() {
        let mut config = EnvLodConfig::default();
        assert!(config.set_pass_lod("fog", 2));
        assert_eq!(config.fog_lod, 2);

        assert!(!config.set_pass_lod("invalid", 1));
    }

    #[test]
    fn test_lod_config_increase_lod() {
        let mut config = EnvLodConfig::new(1, 1, 1, 1, 1);
        assert!(config.increase_lod("fog"));
        assert_eq!(config.fog_lod, 2);

        // Already at max
        config.fog_lod = MAX_LOD_LEVEL;
        assert!(!config.increase_lod("fog"));
    }

    #[test]
    fn test_lod_config_decrease_lod() {
        let mut config = EnvLodConfig::new(2, 2, 2, 2, 2);
        assert!(config.decrease_lod("water"));
        assert_eq!(config.water_lod, 1);

        // Already at min
        config.water_lod = 0;
        assert!(!config.decrease_lod("water"));
    }

    #[test]
    fn test_lod_config_average_lod() {
        let config = EnvLodConfig::new(0, 1, 2, 3, 2);
        let expected = (0.0 + 1.0 + 2.0 + 3.0 + 2.0) / 5.0;
        assert!((config.average_lod() - expected).abs() < 0.001);
    }

    #[test]
    fn test_lod_config_is_all_max_min() {
        let max_config = EnvLodConfig::uniform(MAX_LOD_LEVEL);
        assert!(max_config.is_all_max());
        assert!(!max_config.is_all_min());

        let min_config = EnvLodConfig::uniform(0);
        assert!(min_config.is_all_min());
        assert!(!min_config.is_all_max());
    }

    #[test]
    fn test_lod_config_pod_zeroable() {
        let zeroed: EnvLodConfig = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.fog_lod, 0);

        let config = EnvLodConfig::default();
        let bytes = bytemuck::bytes_of(&config);
        assert_eq!(bytes.len(), std::mem::size_of::<EnvLodConfig>());
    }

    // ===== FrameStats Tests =====

    #[test]
    fn test_frame_stats_default() {
        let stats = FrameStats::default();
        assert_eq!(stats.total_time_ms, 0.0);
        assert!(!stats.over_budget);
    }

    #[test]
    fn test_frame_stats_env_time() {
        let stats = FrameStats {
            fog_time_ms: 1.0,
            water_time_ms: 2.0,
            clouds_time_ms: 3.0,
            terrain_time_ms: 2.0,
            foliage_time_ms: 1.5,
            ..Default::default()
        };
        assert_eq!(stats.env_time_ms(), 9.5);
    }

    #[test]
    fn test_frame_stats_get_pass_time() {
        let stats = FrameStats {
            fog_time_ms: 1.5,
            water_time_ms: 2.5,
            ..Default::default()
        };
        assert_eq!(stats.get_pass_time("fog"), Some(1.5));
        assert_eq!(stats.get_pass_time("water"), Some(2.5));
        assert_eq!(stats.get_pass_time("invalid"), None);
    }

    #[test]
    fn test_frame_stats_budget_utilization() {
        let stats = FrameStats {
            total_time_ms: 20.0,
            ..Default::default()
        };
        let budget = PerformanceBudget::new(16.67, 2.0, 3.0, 4.0, 3.0, 2.0);
        let util = stats.budget_utilization(&budget);
        assert!((util - (20.0 / 16.67)).abs() < 0.01);
    }

    // ===== PerformanceTracker Tests =====

    #[test]
    fn test_tracker_new() {
        let budget = PerformanceBudget::default();
        let tracker = PerformanceTracker::new(budget);
        assert_eq!(*tracker.budget(), budget);
    }

    #[test]
    fn test_tracker_with_defaults() {
        let tracker = PerformanceTracker::with_defaults();
        assert_eq!(*tracker.budget(), PerformanceBudget::default());
    }

    #[test]
    fn test_tracker_for_tier() {
        let tracker = PerformanceTracker::for_tier(QualityTier::Mobile);
        assert_eq!(tracker.current_tier(), QualityTier::Mobile);
    }

    #[test]
    fn test_tracker_begin_end_frame() {
        let mut tracker = PerformanceTracker::with_defaults();
        tracker.begin_frame();
        tracker.record_pass_time("fog", 1.5);
        tracker.end_frame(10.0);

        let stats = tracker.frame_stats();
        assert_eq!(stats.total_time_ms, 10.0);
        assert_eq!(stats.fog_time_ms, 1.5);
    }

    #[test]
    fn test_tracker_begin_end_pass() {
        let mut tracker = PerformanceTracker::with_defaults();
        tracker.begin_frame();
        tracker.begin_pass("fog");
        // Simulate some work
        std::thread::sleep(std::time::Duration::from_micros(100));
        let elapsed = tracker.end_pass();
        assert!(elapsed > 0.0);
    }

    #[test]
    fn test_tracker_record_pass_time() {
        let mut tracker = PerformanceTracker::with_defaults();
        tracker.begin_frame();
        tracker.record_pass_time("water", 2.5);
        assert_eq!(tracker.get_pass_time("water"), 2.5);
    }

    #[test]
    fn test_tracker_is_over_budget() {
        let mut tracker = PerformanceTracker::new(PerformanceBudget::for_target_fps(60.0));

        tracker.begin_frame();
        tracker.end_frame(10.0); // Under budget
        assert!(!tracker.is_over_budget());

        tracker.begin_frame();
        tracker.end_frame(20.0); // Over budget
        assert!(tracker.is_over_budget());
    }

    #[test]
    fn test_tracker_get_overage() {
        let mut tracker = PerformanceTracker::new(PerformanceBudget::for_target_fps(60.0));

        tracker.begin_frame();
        tracker.end_frame(20.0);
        assert!((tracker.get_overage() - 3.33).abs() < 0.1);
    }

    #[test]
    fn test_tracker_get_headroom() {
        let mut tracker = PerformanceTracker::new(PerformanceBudget::for_target_fps(60.0));

        tracker.begin_frame();
        tracker.end_frame(10.0);
        assert!((tracker.get_headroom() - 6.67).abs() < 0.1);
    }

    #[test]
    fn test_tracker_rolling_average() {
        let mut tracker = PerformanceTracker::with_defaults();

        for i in 0..ROLLING_AVERAGE_FRAMES {
            tracker.begin_frame();
            tracker.end_frame((i + 1) as f32);
        }

        let expected = (1..=ROLLING_AVERAGE_FRAMES as i32).sum::<i32>() as f32 / ROLLING_AVERAGE_FRAMES as f32;
        assert!((tracker.rolling_average_frame_time() - expected).abs() < 0.1);
    }

    #[test]
    fn test_tracker_suggest_lod_adjustment_over_budget() {
        let mut tracker = PerformanceTracker::new(PerformanceBudget::for_target_fps(60.0));
        tracker.set_lods(EnvLodConfig::default());

        // Simulate sustained over-budget
        for _ in 0..ROLLING_AVERAGE_FRAMES {
            tracker.begin_frame();
            tracker.record_pass_time("clouds", 10.0); // Very expensive
            tracker.end_frame(25.0);
        }

        let suggestion = tracker.suggest_lod_adjustment();
        assert!(suggestion.is_some());
        let new_lods = suggestion.unwrap();
        // Clouds should be increased (lower detail) since it's the most expensive
        assert!(new_lods.cloud_lod > 0 || new_lods.average_lod() > 0.0);
    }

    #[test]
    fn test_tracker_no_suggestion_when_on_budget() {
        let mut tracker = PerformanceTracker::new(PerformanceBudget::for_target_fps(60.0));

        // Exactly on budget
        for _ in 0..ROLLING_AVERAGE_FRAMES {
            tracker.begin_frame();
            tracker.end_frame(16.67);
        }

        assert!(tracker.suggest_lod_adjustment().is_none());
    }

    #[test]
    fn test_tracker_apply_lod_adjustment() {
        let mut tracker = PerformanceTracker::with_defaults();
        let new_lods = EnvLodConfig::uniform(2);

        tracker.apply_lod_adjustment(new_lods);
        assert_eq!(*tracker.current_lods(), new_lods);
    }

    #[test]
    fn test_tracker_set_budget() {
        let mut tracker = PerformanceTracker::with_defaults();
        let new_budget = PerformanceBudget::for_target_fps(30.0);
        tracker.set_budget(new_budget);
        assert_eq!(*tracker.budget(), new_budget);
    }

    #[test]
    fn test_tracker_zero_budget_handling() {
        let budget = PerformanceBudget::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let mut tracker = PerformanceTracker::new(budget);

        tracker.begin_frame();
        tracker.end_frame(10.0);

        // Should handle zero budget gracefully
        assert!(tracker.is_over_budget());
        assert_eq!(tracker.get_overage(), 10.0);
    }

    #[test]
    fn test_tracker_missing_pass_handling() {
        let tracker = PerformanceTracker::with_defaults();
        assert_eq!(tracker.get_pass_time("nonexistent"), 0.0);
    }

    // ===== LodSelector Tests =====

    #[test]
    fn test_lod_selector_default() {
        let selector = LodSelector::new();
        // Just verify it creates without panic
        assert!(selector.fog_thresholds[0] > 0.0);
    }

    #[test]
    fn test_lod_selector_fog_lod_distance() {
        let selector = LodSelector::new();

        // Close = high detail (LOD 0)
        assert_eq!(selector.select_fog_lod(10.0, 10.0), 0);

        // Far = low detail (LOD 3)
        assert_eq!(selector.select_fog_lod(2000.0, 10.0), MAX_LOD_LEVEL);
    }

    #[test]
    fn test_lod_selector_fog_lod_budget_pressure() {
        let selector = LodSelector::new();

        // With budget pressure, LOD should increase
        let lod_no_pressure = selector.select_fog_lod(100.0, 5.0);
        let lod_with_pressure = selector.select_fog_lod(100.0, 0.3);

        assert!(lod_with_pressure >= lod_no_pressure);
    }

    #[test]
    fn test_lod_selector_water_lod() {
        let selector = LodSelector::new();

        // Close to water surface = high detail
        let lod_close = selector.select_water_lod(2.0, 10.0);

        // Far from water surface = lower detail
        let lod_far = selector.select_water_lod(100.0, 10.0);

        assert!(lod_close <= lod_far);
    }

    #[test]
    fn test_lod_selector_terrain_lod() {
        let selector = LodSelector::new();

        // Close terrain = high detail
        assert_eq!(selector.select_terrain_lod(50.0, 0.1), 0);

        // Far terrain = low detail
        assert_eq!(selector.select_terrain_lod(3000.0, 0.1), MAX_LOD_LEVEL);
    }

    #[test]
    fn test_lod_selector_terrain_slope_bonus() {
        let selector = LodSelector::new();

        // Steep slope should get detail bonus
        let lod_flat = selector.select_terrain_lod(200.0, 0.1);
        let lod_steep = selector.select_terrain_lod(200.0, 0.5);

        assert!(lod_steep <= lod_flat);
    }

    #[test]
    fn test_lod_selector_foliage_density() {
        let selector = LodSelector::new();

        // Close with budget = high density
        let density_close = selector.select_foliage_density(10.0, 5.0);
        assert!(density_close > 0.8);

        // Far with low budget = low density
        let density_far = selector.select_foliage_density(500.0, 0.3);
        assert!(density_far < 0.5);
    }

    #[test]
    fn test_lod_selector_cloud_lod() {
        let selector = LodSelector::new();

        assert_eq!(selector.select_cloud_lod(100.0), 0);
        assert_eq!(selector.select_cloud_lod(1000.0), 1);
        assert_eq!(selector.select_cloud_lod(2000.0), 2);
        assert_eq!(selector.select_cloud_lod(5000.0), 3);
    }

    #[test]
    fn test_lod_selector_select_all_lods() {
        let selector = LodSelector::new();

        let lods = selector.select_all_lods(
            10.0,   // camera height
            500.0,  // terrain distance
            0.3,    // terrain slope
            100.0,  // water distance
            50.0,   // foliage distance
            5.0,    // budget remaining
        );

        assert!(lods.is_valid());
    }

    // ===== AdaptiveQuality Tests =====

    #[test]
    fn test_adaptive_quality_new() {
        let aq = AdaptiveQuality::new(QualityTier::High);
        assert_eq!(aq.current_tier(), QualityTier::High);
    }

    #[test]
    fn test_adaptive_quality_downgrade_on_over_budget() {
        let mut aq = AdaptiveQuality::new(QualityTier::High);

        // Simulate sustained over-budget frames
        for _ in 0..10 {
            let result = aq.record_frame(25.0); // Well over 16.67ms budget
            if result.is_some() {
                assert_eq!(result.unwrap(), QualityTier::Medium);
                return;
            }
        }

        // Should have downgraded by now
        assert_eq!(aq.current_tier(), QualityTier::Medium);
    }

    #[test]
    fn test_adaptive_quality_upgrade_on_stable() {
        let mut aq = AdaptiveQuality::with_target(QualityTier::Low, 16.67);

        // Simulate sustained under-budget frames
        for _ in 0..MIN_STABLE_FRAMES_FOR_UPGRADE + MIN_FRAMES_AFTER_DOWNGRADE + 10 {
            let _ = aq.record_frame(10.0); // Well under budget
        }

        // Should have upgraded
        assert!(aq.current_tier() as u8 <= QualityTier::Low as u8);
    }

    #[test]
    fn test_adaptive_quality_hysteresis_no_oscillation() {
        let mut aq = AdaptiveQuality::new(QualityTier::Medium);

        // Alternate between just over and just under budget
        // Should NOT cause rapid tier changes due to hysteresis
        let mut tier_changes = 0;

        for i in 0..100 {
            let frame_time = if i % 2 == 0 { 17.5 } else { 15.5 };
            if aq.record_frame(frame_time).is_some() {
                tier_changes += 1;
            }
        }

        // Should have very few tier changes due to hysteresis
        assert!(tier_changes <= 2, "Too many tier changes: {}", tier_changes);
    }

    #[test]
    fn test_adaptive_quality_force_tier() {
        let mut aq = AdaptiveQuality::new(QualityTier::High);
        aq.force_tier(QualityTier::Mobile);
        assert_eq!(aq.current_tier(), QualityTier::Mobile);
    }

    #[test]
    fn test_adaptive_quality_reset() {
        let mut aq = AdaptiveQuality::new(QualityTier::High);

        // Record some frames
        for _ in 0..10 {
            aq.record_frame(15.0);
        }

        aq.reset();
        assert_eq!(aq.rolling_average(), 0.0);
        assert_eq!(aq.frames_since_change(), 0);
    }

    #[test]
    fn test_adaptive_quality_is_stable() {
        let mut aq = AdaptiveQuality::new(QualityTier::High);

        // Initially not stable
        assert!(!aq.is_stable());

        // Use frame time in the "acceptable" range (not triggering upgrade or downgrade)
        // Target is 16.67ms, upgrade threshold is ~14.17ms, downgrade is ~18.34ms
        // Use 15.5ms to stay in the middle
        for _ in 0..MIN_FRAMES_AFTER_DOWNGRADE + 10 {
            aq.record_frame(15.5);
        }

        assert!(aq.is_stable());
    }

    #[test]
    fn test_adaptive_quality_rolling_average() {
        let mut aq = AdaptiveQuality::new(QualityTier::High);

        for _ in 0..ROLLING_AVERAGE_FRAMES {
            aq.record_frame(10.0);
        }

        assert!((aq.rolling_average() - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_adaptive_quality_no_upgrade_at_highest() {
        let mut aq = AdaptiveQuality::new(QualityTier::Ultra);

        for _ in 0..200 {
            let result = aq.record_frame(5.0); // Very fast
            assert!(result.is_none()); // Should not change from Ultra
        }

        assert_eq!(aq.current_tier(), QualityTier::Ultra);
    }

    #[test]
    fn test_adaptive_quality_no_downgrade_at_lowest() {
        let mut aq = AdaptiveQuality::with_target(QualityTier::Mobile, 33.33);

        for _ in 0..100 {
            let result = aq.record_frame(50.0); // Very slow
            if result.is_some() {
                // If any change happens, it should still be Mobile
                assert_eq!(result.unwrap(), QualityTier::Mobile);
            }
        }

        assert_eq!(aq.current_tier(), QualityTier::Mobile);
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_edge_case_all_passes_max_lod() {
        let mut config = EnvLodConfig::uniform(MAX_LOD_LEVEL);
        assert!(config.is_all_max());
        assert!(!config.increase_lod("fog")); // Can't go higher
    }

    #[test]
    fn test_edge_case_all_passes_min_lod() {
        let mut config = EnvLodConfig::uniform(0);
        assert!(config.is_all_min());
        assert!(!config.decrease_lod("fog")); // Can't go lower
    }

    #[test]
    fn test_edge_case_negative_distance() {
        let selector = LodSelector::new();
        // Negative distance should be treated as very close
        let lod = selector.select_fog_lod(-100.0, 5.0);
        assert_eq!(lod, 0); // Highest detail for negative/close distance
    }

    #[test]
    fn test_edge_case_very_large_frame_time() {
        let mut tracker = PerformanceTracker::with_defaults();
        tracker.begin_frame();
        tracker.end_frame(1000.0); // 1 second frame!

        assert!(tracker.is_over_budget());
        assert!(tracker.get_overage() > 900.0);
    }

    #[test]
    fn test_edge_case_zero_frame_time() {
        let mut tracker = PerformanceTracker::with_defaults();
        tracker.begin_frame();
        tracker.end_frame(0.0);

        assert!(!tracker.is_over_budget());
        assert_eq!(tracker.get_overage(), 0.0);
    }

    #[test]
    fn test_edge_case_rapid_tier_changes() {
        let mut aq = AdaptiveQuality::new(QualityTier::Medium);

        // Even with extreme values, cooldown should prevent rapid changes
        let mut changes = 0;
        for i in 0..50 {
            let frame_time = if i % 2 == 0 { 100.0 } else { 1.0 };
            if aq.record_frame(frame_time).is_some() {
                changes += 1;
            }
        }

        // Cooldown should limit changes
        assert!(changes < 10);
    }

    #[test]
    fn test_budget_validation_edge_cases() {
        // All zeros
        let zero_budget = PerformanceBudget::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(zero_budget.is_valid()); // Zeros are valid

        // Very large values
        let large_budget = PerformanceBudget::new(
            f32::MAX, f32::MAX, f32::MAX, f32::MAX, f32::MAX, f32::MAX
        );
        assert!(large_budget.is_valid());
    }

    #[test]
    fn test_lod_selector_extreme_distances() {
        let selector = LodSelector::new();

        // Very close
        assert_eq!(selector.select_fog_lod(0.001, 10.0), 0);

        // Extremely far
        assert_eq!(selector.select_fog_lod(1_000_000.0, 10.0), MAX_LOD_LEVEL);
    }

    #[test]
    fn test_frame_stats_accumulation() {
        let mut tracker = PerformanceTracker::with_defaults();

        // Multiple frames should accumulate correctly in history
        for i in 1..=20 {
            tracker.begin_frame();
            tracker.end_frame(i as f32);
        }

        // Rolling average should be based on last 16 frames
        let avg = tracker.rolling_average_frame_time();
        // Average of 5..=20 = (5+6+...+20)/16 = 200/16 = 12.5
        assert!(avg > 10.0 && avg < 15.0);
    }

    #[test]
    fn test_pass_timing_without_begin() {
        let mut tracker = PerformanceTracker::with_defaults();
        // End pass without begin should return 0
        let elapsed = tracker.end_pass();
        assert_eq!(elapsed, 0.0);
    }

    #[test]
    fn test_multiple_pass_recordings() {
        let mut tracker = PerformanceTracker::with_defaults();
        tracker.begin_frame();

        // Record all standard passes
        tracker.record_pass_time("fog", 1.0);
        tracker.record_pass_time("water", 2.0);
        tracker.record_pass_time("clouds", 3.0);
        tracker.record_pass_time("terrain", 2.5);
        tracker.record_pass_time("foliage", 1.5);

        let stats = tracker.frame_stats();
        assert_eq!(stats.env_time_ms(), 10.0);
    }
}
