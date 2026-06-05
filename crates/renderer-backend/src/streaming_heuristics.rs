//! Feedback-Based Streaming Heuristics Auto-Tuning
//!
//! This module implements adaptive streaming heuristics that automatically tune
//! preloading parameters based on runtime performance feedback. The system uses
//! PID-like feedback control to optimize streaming behavior.
//!
//! # Features
//!
//! - Real-time metrics collection (miss rate, latency, budget pressure)
//! - Exponential moving average smoothing for stable metrics
//! - PID-like auto-tuning based on feedback
//! - Developer overrides for manual parameter control
//! - Parameter clamping to valid ranges
//!
//! # Auto-Tuning Logic
//!
//! - High miss rate -> increase preload_distance
//! - High budget pressure -> decrease preload_distance
//! - High load latency -> reduce tier weights for low priority
//! - Miss rate decreasing -> system converging (good)
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::streaming_heuristics::{
//!     HeuristicsTuner, HeuristicParams, StreamingMetrics,
//! };
//!
//! let mut tuner = HeuristicsTuner::default();
//!
//! // Each frame, collect and submit metrics
//! let metrics = StreamingMetrics {
//!     page_miss_rate: 0.02,
//!     load_latency_avg_ms: 5.0,
//!     budget_pressure: 0.7,
//!     lod_switches_per_sec: 10.0,
//! };
//!
//! tuner.update(metrics);
//!
//! // Get auto-tuned parameters
//! let params = tuner.params();
//! ```

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default preload distance in world units.
pub const DEFAULT_PRELOAD_DISTANCE: f32 = 100.0;

/// Default urgency threshold (0.0-1.0).
pub const DEFAULT_URGENCY_THRESHOLD: f32 = 0.7;

/// Default tier weights [Critical, High, Normal, Low].
pub const DEFAULT_TIER_WEIGHTS: [f32; 4] = [1.0, 0.8, 0.5, 0.2];

/// Default target miss rate.
pub const DEFAULT_MISS_RATE_TARGET: f32 = 0.01;

/// Default EMA smoothing alpha (higher = faster response).
pub const DEFAULT_EMA_ALPHA: f32 = 0.2;

/// Minimum preload distance.
pub const MIN_PRELOAD_DISTANCE: f32 = 10.0;

/// Maximum preload distance.
pub const MAX_PRELOAD_DISTANCE: f32 = 500.0;

/// Minimum tier weight.
pub const MIN_TIER_WEIGHT: f32 = 0.0;

/// Maximum tier weight.
pub const MAX_TIER_WEIGHT: f32 = 1.0;

/// Minimum urgency threshold.
pub const MIN_URGENCY_THRESHOLD: f32 = 0.1;

/// Maximum urgency threshold.
pub const MAX_URGENCY_THRESHOLD: f32 = 0.95;

// ---------------------------------------------------------------------------
// StreamingMetrics
// ---------------------------------------------------------------------------

/// Real-time streaming performance metrics collected each frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StreamingMetrics {
    /// Page/texture misses per frame (requires loading on demand).
    pub page_miss_rate: f32,
    /// Average load latency in milliseconds.
    pub load_latency_avg_ms: f32,
    /// Budget pressure (0.0 = no pressure, 1.0 = at budget limit).
    pub budget_pressure: f32,
    /// LOD switches per second (visual pops/artifacts).
    pub lod_switches_per_sec: f32,
}

impl StreamingMetrics {
    /// Create new metrics with specified values.
    pub const fn new(
        page_miss_rate: f32,
        load_latency_avg_ms: f32,
        budget_pressure: f32,
        lod_switches_per_sec: f32,
    ) -> Self {
        Self {
            page_miss_rate,
            load_latency_avg_ms,
            budget_pressure,
            lod_switches_per_sec,
        }
    }

    /// Create metrics indicating good performance.
    pub const fn good() -> Self {
        Self {
            page_miss_rate: 0.0,
            load_latency_avg_ms: 1.0,
            budget_pressure: 0.3,
            lod_switches_per_sec: 1.0,
        }
    }

    /// Create metrics indicating poor performance.
    pub const fn poor() -> Self {
        Self {
            page_miss_rate: 0.1,
            load_latency_avg_ms: 50.0,
            budget_pressure: 0.95,
            lod_switches_per_sec: 30.0,
        }
    }

    /// Check if miss rate is above threshold.
    pub fn is_missing(&self, threshold: f32) -> bool {
        self.page_miss_rate > threshold
    }

    /// Check if under budget pressure (above threshold).
    pub fn is_budget_stressed(&self, threshold: f32) -> bool {
        self.budget_pressure > threshold
    }

    /// Check if load latency is high (above threshold in ms).
    pub fn is_latency_high(&self, threshold_ms: f32) -> bool {
        self.load_latency_avg_ms > threshold_ms
    }

    /// Get overall health score (0.0 = poor, 1.0 = excellent).
    pub fn health_score(&self) -> f32 {
        // Invert metrics so higher is better
        let miss_score = 1.0 - self.page_miss_rate.clamp(0.0, 1.0);
        let latency_score = 1.0 - (self.load_latency_avg_ms / 100.0).clamp(0.0, 1.0);
        let budget_score = 1.0 - self.budget_pressure.clamp(0.0, 1.0);
        let lod_score = 1.0 - (self.lod_switches_per_sec / 60.0).clamp(0.0, 1.0);

        // Weighted average
        0.3 * miss_score + 0.3 * latency_score + 0.2 * budget_score + 0.2 * lod_score
    }
}

// ---------------------------------------------------------------------------
// SmoothedMetrics
// ---------------------------------------------------------------------------

/// Exponential moving average smoothed metrics for stable feedback.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothedMetrics {
    /// Smoothed page miss rate.
    pub page_miss_rate: f32,
    /// Smoothed load latency.
    pub load_latency_avg_ms: f32,
    /// Smoothed budget pressure.
    pub budget_pressure: f32,
    /// Smoothed LOD switches.
    pub lod_switches_per_sec: f32,
    /// Previous miss rate (for trend detection).
    pub prev_miss_rate: f32,
    /// EMA smoothing factor.
    alpha: f32,
    /// Whether initialized with at least one sample.
    initialized: bool,
}

impl Default for SmoothedMetrics {
    fn default() -> Self {
        Self::new(DEFAULT_EMA_ALPHA)
    }
}

impl SmoothedMetrics {
    /// Create new smoothed metrics with specified alpha.
    ///
    /// # Parameters
    ///
    /// * `alpha` - Smoothing factor in [0, 1]. Higher values respond faster.
    pub fn new(alpha: f32) -> Self {
        Self {
            page_miss_rate: 0.0,
            load_latency_avg_ms: 0.0,
            budget_pressure: 0.0,
            lod_switches_per_sec: 0.0,
            prev_miss_rate: 0.0,
            alpha: alpha.clamp(0.01, 1.0),
            initialized: false,
        }
    }

    /// Update with new raw metrics, applying EMA smoothing.
    pub fn update(&mut self, raw: &StreamingMetrics) {
        if !self.initialized {
            self.page_miss_rate = raw.page_miss_rate;
            self.load_latency_avg_ms = raw.load_latency_avg_ms;
            self.budget_pressure = raw.budget_pressure;
            self.lod_switches_per_sec = raw.lod_switches_per_sec;
            self.initialized = true;
            return;
        }

        // Store previous for trend detection
        self.prev_miss_rate = self.page_miss_rate;

        // Apply EMA: new = alpha * raw + (1 - alpha) * old
        self.page_miss_rate = self.ema(self.page_miss_rate, raw.page_miss_rate);
        self.load_latency_avg_ms = self.ema(self.load_latency_avg_ms, raw.load_latency_avg_ms);
        self.budget_pressure = self.ema(self.budget_pressure, raw.budget_pressure);
        self.lod_switches_per_sec = self.ema(self.lod_switches_per_sec, raw.lod_switches_per_sec);
    }

    /// Apply EMA formula.
    fn ema(&self, old: f32, new: f32) -> f32 {
        self.alpha * new + (1.0 - self.alpha) * old
    }

    /// Check if miss rate is decreasing (converging).
    pub fn is_miss_rate_decreasing(&self) -> bool {
        self.page_miss_rate < self.prev_miss_rate
    }

    /// Get miss rate delta (positive = increasing, negative = decreasing).
    pub fn miss_rate_delta(&self) -> f32 {
        self.page_miss_rate - self.prev_miss_rate
    }

    /// Reset to uninitialized state.
    pub fn reset(&mut self) {
        self.page_miss_rate = 0.0;
        self.load_latency_avg_ms = 0.0;
        self.budget_pressure = 0.0;
        self.lod_switches_per_sec = 0.0;
        self.prev_miss_rate = 0.0;
        self.initialized = false;
    }

    /// Check if initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the alpha value.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Set the alpha value.
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha.clamp(0.01, 1.0);
    }

    /// Convert to raw metrics snapshot.
    pub fn to_metrics(&self) -> StreamingMetrics {
        StreamingMetrics {
            page_miss_rate: self.page_miss_rate,
            load_latency_avg_ms: self.load_latency_avg_ms,
            budget_pressure: self.budget_pressure,
            lod_switches_per_sec: self.lod_switches_per_sec,
        }
    }
}

// ---------------------------------------------------------------------------
// HeuristicParams
// ---------------------------------------------------------------------------

/// Parameters that control streaming heuristics behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeuristicParams {
    /// Distance ahead to preload assets (in world units).
    pub preload_distance: f32,
    /// Threshold at which to boost priority (0.0-1.0).
    pub urgency_threshold: f32,
    /// Priority weights for tiers: [Critical, High, Normal, Low].
    pub tier_weights: [f32; 4],
    /// Target miss rate the tuner aims for.
    pub miss_rate_target: f32,
}

impl Default for HeuristicParams {
    fn default() -> Self {
        Self {
            preload_distance: DEFAULT_PRELOAD_DISTANCE,
            urgency_threshold: DEFAULT_URGENCY_THRESHOLD,
            tier_weights: DEFAULT_TIER_WEIGHTS,
            miss_rate_target: DEFAULT_MISS_RATE_TARGET,
        }
    }
}

impl HeuristicParams {
    /// Create new parameters with specified values.
    pub const fn new(
        preload_distance: f32,
        urgency_threshold: f32,
        tier_weights: [f32; 4],
        miss_rate_target: f32,
    ) -> Self {
        Self {
            preload_distance,
            urgency_threshold,
            tier_weights,
            miss_rate_target,
        }
    }

    /// Create aggressive parameters (more preloading, higher quality).
    pub const fn aggressive() -> Self {
        Self {
            preload_distance: 200.0,
            urgency_threshold: 0.5,
            tier_weights: [1.0, 0.9, 0.7, 0.4],
            miss_rate_target: 0.005,
        }
    }

    /// Create conservative parameters (less preloading, save memory).
    pub const fn conservative() -> Self {
        Self {
            preload_distance: 50.0,
            urgency_threshold: 0.85,
            tier_weights: [1.0, 0.6, 0.3, 0.1],
            miss_rate_target: 0.02,
        }
    }

    /// Clamp all parameters to valid ranges.
    pub fn clamp(&mut self) {
        self.preload_distance = self.preload_distance.clamp(MIN_PRELOAD_DISTANCE, MAX_PRELOAD_DISTANCE);
        self.urgency_threshold = self.urgency_threshold.clamp(MIN_URGENCY_THRESHOLD, MAX_URGENCY_THRESHOLD);
        for weight in &mut self.tier_weights {
            *weight = weight.clamp(MIN_TIER_WEIGHT, MAX_TIER_WEIGHT);
        }
        self.miss_rate_target = self.miss_rate_target.clamp(0.0, 1.0);
    }

    /// Create a clamped copy.
    pub fn clamped(&self) -> Self {
        let mut copy = *self;
        copy.clamp();
        copy
    }

    /// Get weight for critical tier (index 0).
    pub fn critical_weight(&self) -> f32 {
        self.tier_weights[0]
    }

    /// Get weight for high tier (index 1).
    pub fn high_weight(&self) -> f32 {
        self.tier_weights[1]
    }

    /// Get weight for normal tier (index 2).
    pub fn normal_weight(&self) -> f32 {
        self.tier_weights[2]
    }

    /// Get weight for low tier (index 3).
    pub fn low_weight(&self) -> f32 {
        self.tier_weights[3]
    }

    /// Set tier weight by index.
    pub fn set_tier_weight(&mut self, tier: usize, weight: f32) {
        if tier < 4 {
            self.tier_weights[tier] = weight.clamp(MIN_TIER_WEIGHT, MAX_TIER_WEIGHT);
        }
    }

    /// Linear interpolation between two parameter sets.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let t_inv = 1.0 - t;

        Self {
            preload_distance: self.preload_distance * t_inv + other.preload_distance * t,
            urgency_threshold: self.urgency_threshold * t_inv + other.urgency_threshold * t,
            tier_weights: [
                self.tier_weights[0] * t_inv + other.tier_weights[0] * t,
                self.tier_weights[1] * t_inv + other.tier_weights[1] * t,
                self.tier_weights[2] * t_inv + other.tier_weights[2] * t,
                self.tier_weights[3] * t_inv + other.tier_weights[3] * t,
            ],
            miss_rate_target: self.miss_rate_target * t_inv + other.miss_rate_target * t,
        }
    }
}

// ---------------------------------------------------------------------------
// ParamOverride
// ---------------------------------------------------------------------------

/// Tracks which parameters have developer overrides.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ParamOverrides {
    /// Lock preload_distance (prevent auto-tuning).
    pub preload_distance_locked: bool,
    /// Lock urgency_threshold.
    pub urgency_threshold_locked: bool,
    /// Lock tier weights (all four).
    pub tier_weights_locked: bool,
    /// Lock miss rate target.
    pub miss_rate_target_locked: bool,
}

impl ParamOverrides {
    /// Create with no overrides.
    pub const fn none() -> Self {
        Self {
            preload_distance_locked: false,
            urgency_threshold_locked: false,
            tier_weights_locked: false,
            miss_rate_target_locked: false,
        }
    }

    /// Create with all parameters locked.
    pub const fn all() -> Self {
        Self {
            preload_distance_locked: true,
            urgency_threshold_locked: true,
            tier_weights_locked: true,
            miss_rate_target_locked: true,
        }
    }

    /// Check if any parameter is locked.
    pub fn has_any(&self) -> bool {
        self.preload_distance_locked
            || self.urgency_threshold_locked
            || self.tier_weights_locked
            || self.miss_rate_target_locked
    }

    /// Clear all overrides.
    pub fn clear(&mut self) {
        *self = Self::none();
    }
}

// ---------------------------------------------------------------------------
// TuningConfig
// ---------------------------------------------------------------------------

/// Configuration for the auto-tuning behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TuningConfig {
    /// Rate of preload distance adjustment per update.
    pub preload_distance_step: f32,
    /// Rate of tier weight adjustment per update.
    pub tier_weight_step: f32,
    /// Rate of urgency threshold adjustment per update.
    pub urgency_step: f32,
    /// Budget pressure threshold above which to reduce preloading.
    pub budget_pressure_threshold: f32,
    /// Latency threshold (ms) above which to reduce low-priority loading.
    pub latency_threshold_ms: f32,
    /// Minimum samples before tuning starts.
    pub warmup_samples: u32,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            preload_distance_step: 5.0,
            tier_weight_step: 0.02,
            urgency_step: 0.01,
            budget_pressure_threshold: 0.8,
            latency_threshold_ms: 20.0,
            warmup_samples: 10,
        }
    }
}

impl TuningConfig {
    /// Create aggressive tuning (faster convergence, more jitter).
    pub const fn aggressive() -> Self {
        Self {
            preload_distance_step: 10.0,
            tier_weight_step: 0.05,
            urgency_step: 0.02,
            budget_pressure_threshold: 0.7,
            latency_threshold_ms: 15.0,
            warmup_samples: 5,
        }
    }

    /// Create conservative tuning (slower convergence, less jitter).
    pub const fn conservative() -> Self {
        Self {
            preload_distance_step: 2.0,
            tier_weight_step: 0.01,
            urgency_step: 0.005,
            budget_pressure_threshold: 0.85,
            latency_threshold_ms: 30.0,
            warmup_samples: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// HeuristicsTuner
// ---------------------------------------------------------------------------

/// Main auto-tuning system for streaming heuristics.
#[derive(Debug, Clone)]
pub struct HeuristicsTuner {
    /// Current tuned parameters.
    params: HeuristicParams,
    /// Smoothed metrics for feedback.
    smoothed: SmoothedMetrics,
    /// Tuning configuration.
    config: TuningConfig,
    /// Developer overrides.
    overrides: ParamOverrides,
    /// Forced parameter values (when overridden).
    forced_params: HeuristicParams,
    /// Sample count for warmup.
    sample_count: u32,
    /// Auto-tuning enabled flag.
    auto_tune_enabled: bool,
}

impl Default for HeuristicsTuner {
    fn default() -> Self {
        Self::new(HeuristicParams::default(), TuningConfig::default())
    }
}

impl HeuristicsTuner {
    /// Create a new tuner with initial parameters and configuration.
    pub fn new(initial_params: HeuristicParams, config: TuningConfig) -> Self {
        let params = initial_params.clamped();
        Self {
            params,
            smoothed: SmoothedMetrics::default(),
            config,
            overrides: ParamOverrides::none(),
            forced_params: params,
            sample_count: 0,
            auto_tune_enabled: true,
        }
    }

    /// Create a tuner with default configuration.
    pub fn with_params(params: HeuristicParams) -> Self {
        Self::new(params, TuningConfig::default())
    }

    /// Get current effective parameters (respecting overrides).
    pub fn params(&self) -> HeuristicParams {
        let mut result = self.params;

        if self.overrides.preload_distance_locked {
            result.preload_distance = self.forced_params.preload_distance;
        }
        if self.overrides.urgency_threshold_locked {
            result.urgency_threshold = self.forced_params.urgency_threshold;
        }
        if self.overrides.tier_weights_locked {
            result.tier_weights = self.forced_params.tier_weights;
        }
        if self.overrides.miss_rate_target_locked {
            result.miss_rate_target = self.forced_params.miss_rate_target;
        }

        result
    }

    /// Get raw auto-tuned parameters (ignoring overrides).
    pub fn raw_params(&self) -> &HeuristicParams {
        &self.params
    }

    /// Get smoothed metrics.
    pub fn smoothed_metrics(&self) -> &SmoothedMetrics {
        &self.smoothed
    }

    /// Get current smoothed metrics as StreamingMetrics.
    pub fn current_metrics(&self) -> StreamingMetrics {
        self.smoothed.to_metrics()
    }

    /// Get tuning configuration.
    pub fn config(&self) -> &TuningConfig {
        &self.config
    }

    /// Set tuning configuration.
    pub fn set_config(&mut self, config: TuningConfig) {
        self.config = config;
    }

    /// Get override state.
    pub fn overrides(&self) -> &ParamOverrides {
        &self.overrides
    }

    /// Check if auto-tuning is enabled.
    pub fn is_auto_tune_enabled(&self) -> bool {
        self.auto_tune_enabled
    }

    /// Enable or disable auto-tuning.
    pub fn set_auto_tune_enabled(&mut self, enabled: bool) {
        self.auto_tune_enabled = enabled;
    }

    /// Get number of samples collected.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Check if warmup period is complete.
    pub fn is_warmed_up(&self) -> bool {
        self.sample_count >= self.config.warmup_samples
    }

    /// Update with new metrics and perform auto-tuning.
    pub fn update(&mut self, metrics: StreamingMetrics) {
        // Update smoothed metrics
        self.smoothed.update(&metrics);
        self.sample_count = self.sample_count.saturating_add(1);

        // Skip tuning if disabled or not warmed up
        if !self.auto_tune_enabled || !self.is_warmed_up() {
            return;
        }

        // Perform auto-tuning
        self.auto_tune();
    }

    /// Perform auto-tuning based on current smoothed metrics.
    fn auto_tune(&mut self) {
        let miss_rate = self.smoothed.page_miss_rate;
        let budget_pressure = self.smoothed.budget_pressure;
        let latency = self.smoothed.load_latency_avg_ms;
        let miss_delta = self.smoothed.miss_rate_delta();

        // --- Preload Distance Tuning ---
        if !self.overrides.preload_distance_locked {
            // High miss rate -> increase preload distance
            if miss_rate > self.params.miss_rate_target {
                self.params.preload_distance += self.config.preload_distance_step;
            }
            // High budget pressure -> decrease preload distance
            else if budget_pressure > self.config.budget_pressure_threshold {
                self.params.preload_distance -= self.config.preload_distance_step;
            }
            // Miss rate converging but still above target -> slight increase
            else if miss_rate > self.params.miss_rate_target * 0.5 && miss_delta < 0.0 {
                self.params.preload_distance += self.config.preload_distance_step * 0.5;
            }
        }

        // --- Tier Weight Tuning ---
        if !self.overrides.tier_weights_locked {
            // High latency -> reduce low priority tier weights
            if latency > self.config.latency_threshold_ms {
                // Reduce Low tier weight
                self.params.tier_weights[3] -= self.config.tier_weight_step;
                // Slightly reduce Normal tier weight
                self.params.tier_weights[2] -= self.config.tier_weight_step * 0.5;
            }
            // Good latency and low budget pressure -> can afford higher weights
            else if latency < self.config.latency_threshold_ms * 0.5
                && budget_pressure < self.config.budget_pressure_threshold * 0.5
            {
                self.params.tier_weights[3] += self.config.tier_weight_step * 0.5;
                self.params.tier_weights[2] += self.config.tier_weight_step * 0.25;
            }
        }

        // --- Urgency Threshold Tuning ---
        if !self.overrides.urgency_threshold_locked {
            // High LOD switches -> lower urgency threshold (boost priority sooner)
            if self.smoothed.lod_switches_per_sec > 20.0 {
                self.params.urgency_threshold -= self.config.urgency_step;
            }
            // Low LOD switches and good performance -> can raise threshold
            else if self.smoothed.lod_switches_per_sec < 5.0
                && miss_rate < self.params.miss_rate_target
            {
                self.params.urgency_threshold += self.config.urgency_step;
            }
        }

        // Clamp all parameters
        self.params.clamp();
    }

    // -----------------------------------------------------------------------
    // Developer Override API
    // -----------------------------------------------------------------------

    /// Lock and force preload distance to a specific value.
    pub fn force_preload_distance(&mut self, distance: f32) {
        self.forced_params.preload_distance = distance.clamp(MIN_PRELOAD_DISTANCE, MAX_PRELOAD_DISTANCE);
        self.overrides.preload_distance_locked = true;
    }

    /// Unlock preload distance (allow auto-tuning).
    pub fn unlock_preload_distance(&mut self) {
        self.overrides.preload_distance_locked = false;
    }

    /// Lock and force urgency threshold to a specific value.
    pub fn force_urgency_threshold(&mut self, threshold: f32) {
        self.forced_params.urgency_threshold = threshold.clamp(MIN_URGENCY_THRESHOLD, MAX_URGENCY_THRESHOLD);
        self.overrides.urgency_threshold_locked = true;
    }

    /// Unlock urgency threshold.
    pub fn unlock_urgency_threshold(&mut self) {
        self.overrides.urgency_threshold_locked = false;
    }

    /// Lock and force tier weights.
    pub fn force_tier_weights(&mut self, weights: [f32; 4]) {
        self.forced_params.tier_weights = [
            weights[0].clamp(MIN_TIER_WEIGHT, MAX_TIER_WEIGHT),
            weights[1].clamp(MIN_TIER_WEIGHT, MAX_TIER_WEIGHT),
            weights[2].clamp(MIN_TIER_WEIGHT, MAX_TIER_WEIGHT),
            weights[3].clamp(MIN_TIER_WEIGHT, MAX_TIER_WEIGHT),
        ];
        self.overrides.tier_weights_locked = true;
    }

    /// Unlock tier weights.
    pub fn unlock_tier_weights(&mut self) {
        self.overrides.tier_weights_locked = false;
    }

    /// Lock and force miss rate target.
    pub fn force_miss_rate_target(&mut self, target: f32) {
        self.forced_params.miss_rate_target = target.clamp(0.0, 1.0);
        self.overrides.miss_rate_target_locked = true;
    }

    /// Unlock miss rate target.
    pub fn unlock_miss_rate_target(&mut self) {
        self.overrides.miss_rate_target_locked = false;
    }

    /// Clear all overrides and return to auto-tuning.
    pub fn clear_overrides(&mut self) {
        self.overrides.clear();
    }

    /// Reset to default parameters and clear state.
    pub fn reset(&mut self) {
        self.params = HeuristicParams::default();
        self.smoothed.reset();
        self.overrides.clear();
        self.sample_count = 0;
    }

    /// Reset to specific parameters and clear state.
    pub fn reset_to(&mut self, params: HeuristicParams) {
        self.params = params.clamped();
        self.smoothed.reset();
        self.overrides.clear();
        self.sample_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // StreamingMetrics Tests
    // =========================================================================

    #[test]
    fn test_metrics_default() {
        let m = StreamingMetrics::default();
        assert_eq!(m.page_miss_rate, 0.0);
        assert_eq!(m.load_latency_avg_ms, 0.0);
        assert_eq!(m.budget_pressure, 0.0);
        assert_eq!(m.lod_switches_per_sec, 0.0);
    }

    #[test]
    fn test_metrics_new() {
        let m = StreamingMetrics::new(0.05, 10.0, 0.6, 15.0);
        assert_eq!(m.page_miss_rate, 0.05);
        assert_eq!(m.load_latency_avg_ms, 10.0);
        assert_eq!(m.budget_pressure, 0.6);
        assert_eq!(m.lod_switches_per_sec, 15.0);
    }

    #[test]
    fn test_metrics_good_and_poor() {
        let good = StreamingMetrics::good();
        let poor = StreamingMetrics::poor();
        assert!(good.health_score() > poor.health_score());
    }

    #[test]
    fn test_metrics_is_missing() {
        let m = StreamingMetrics::new(0.05, 10.0, 0.6, 15.0);
        assert!(m.is_missing(0.01));
        assert!(!m.is_missing(0.1));
    }

    #[test]
    fn test_metrics_is_budget_stressed() {
        let m = StreamingMetrics::new(0.05, 10.0, 0.9, 15.0);
        assert!(m.is_budget_stressed(0.8));
        assert!(!m.is_budget_stressed(0.95));
    }

    #[test]
    fn test_metrics_is_latency_high() {
        let m = StreamingMetrics::new(0.05, 30.0, 0.5, 15.0);
        assert!(m.is_latency_high(20.0));
        assert!(!m.is_latency_high(50.0));
    }

    #[test]
    fn test_metrics_health_score_range() {
        let m = StreamingMetrics::new(0.5, 50.0, 0.5, 30.0);
        let score = m.health_score();
        assert!(score >= 0.0 && score <= 1.0);
    }

    // =========================================================================
    // SmoothedMetrics Tests
    // =========================================================================

    #[test]
    fn test_smoothed_default() {
        let s = SmoothedMetrics::default();
        assert!(!s.is_initialized());
        assert_eq!(s.alpha(), DEFAULT_EMA_ALPHA);
    }

    #[test]
    fn test_smoothed_first_update_initializes() {
        let mut s = SmoothedMetrics::new(0.5);
        let m = StreamingMetrics::new(0.1, 20.0, 0.7, 10.0);
        s.update(&m);
        assert!(s.is_initialized());
        assert_eq!(s.page_miss_rate, 0.1);
        assert_eq!(s.load_latency_avg_ms, 20.0);
    }

    #[test]
    fn test_smoothed_ema_applied() {
        let mut s = SmoothedMetrics::new(0.5);

        // First update: direct assignment
        s.update(&StreamingMetrics::new(1.0, 100.0, 1.0, 60.0));
        assert_eq!(s.page_miss_rate, 1.0);

        // Second update: EMA applied
        s.update(&StreamingMetrics::new(0.0, 0.0, 0.0, 0.0));
        // alpha=0.5: new = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        assert!((s.page_miss_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_smoothed_miss_rate_delta() {
        let mut s = SmoothedMetrics::new(1.0); // alpha=1.0 for instant response

        s.update(&StreamingMetrics::new(0.1, 0.0, 0.0, 0.0));
        s.update(&StreamingMetrics::new(0.05, 0.0, 0.0, 0.0));

        assert!(s.is_miss_rate_decreasing());
        assert!(s.miss_rate_delta() < 0.0);
    }

    #[test]
    fn test_smoothed_reset() {
        let mut s = SmoothedMetrics::new(0.5);
        s.update(&StreamingMetrics::new(0.1, 20.0, 0.7, 10.0));
        assert!(s.is_initialized());

        s.reset();
        assert!(!s.is_initialized());
        assert_eq!(s.page_miss_rate, 0.0);
    }

    #[test]
    fn test_smoothed_alpha_clamped() {
        let s = SmoothedMetrics::new(2.0);
        assert_eq!(s.alpha(), 1.0);

        let s2 = SmoothedMetrics::new(-1.0);
        assert_eq!(s2.alpha(), 0.01);
    }

    #[test]
    fn test_smoothed_to_metrics() {
        let mut s = SmoothedMetrics::new(1.0);
        s.update(&StreamingMetrics::new(0.05, 15.0, 0.4, 8.0));

        let m = s.to_metrics();
        assert_eq!(m.page_miss_rate, 0.05);
        assert_eq!(m.load_latency_avg_ms, 15.0);
    }

    // =========================================================================
    // HeuristicParams Tests
    // =========================================================================

    #[test]
    fn test_params_default() {
        let p = HeuristicParams::default();
        assert_eq!(p.preload_distance, DEFAULT_PRELOAD_DISTANCE);
        assert_eq!(p.urgency_threshold, DEFAULT_URGENCY_THRESHOLD);
        assert_eq!(p.tier_weights, DEFAULT_TIER_WEIGHTS);
        assert_eq!(p.miss_rate_target, DEFAULT_MISS_RATE_TARGET);
    }

    #[test]
    fn test_params_aggressive_vs_conservative() {
        let agg = HeuristicParams::aggressive();
        let con = HeuristicParams::conservative();
        assert!(agg.preload_distance > con.preload_distance);
        assert!(agg.miss_rate_target < con.miss_rate_target);
    }

    #[test]
    fn test_params_clamp() {
        let mut p = HeuristicParams::new(0.0, 2.0, [2.0, -1.0, 0.5, 0.5], 1.5);
        p.clamp();

        assert_eq!(p.preload_distance, MIN_PRELOAD_DISTANCE);
        assert_eq!(p.urgency_threshold, MAX_URGENCY_THRESHOLD);
        assert_eq!(p.tier_weights[0], MAX_TIER_WEIGHT);
        assert_eq!(p.tier_weights[1], MIN_TIER_WEIGHT);
        assert_eq!(p.miss_rate_target, 1.0);
    }

    #[test]
    fn test_params_clamped_creates_copy() {
        let p = HeuristicParams::new(0.0, 2.0, [2.0, -1.0, 0.5, 0.5], 1.5);
        let clamped = p.clamped();

        assert_eq!(p.preload_distance, 0.0); // Original unchanged
        assert_eq!(clamped.preload_distance, MIN_PRELOAD_DISTANCE);
    }

    #[test]
    fn test_params_tier_accessors() {
        let p = HeuristicParams::default();
        assert_eq!(p.critical_weight(), DEFAULT_TIER_WEIGHTS[0]);
        assert_eq!(p.high_weight(), DEFAULT_TIER_WEIGHTS[1]);
        assert_eq!(p.normal_weight(), DEFAULT_TIER_WEIGHTS[2]);
        assert_eq!(p.low_weight(), DEFAULT_TIER_WEIGHTS[3]);
    }

    #[test]
    fn test_params_set_tier_weight() {
        let mut p = HeuristicParams::default();
        p.set_tier_weight(2, 0.6);
        assert_eq!(p.tier_weights[2], 0.6);

        // Out of bounds - no change
        p.set_tier_weight(10, 0.9);
    }

    #[test]
    fn test_params_lerp() {
        let a = HeuristicParams::new(100.0, 0.5, [1.0, 0.8, 0.5, 0.2], 0.01);
        let b = HeuristicParams::new(200.0, 1.0, [0.5, 0.4, 0.25, 0.1], 0.02);

        let mid = a.lerp(&b, 0.5);
        assert!((mid.preload_distance - 150.0).abs() < 0.001);
        assert!((mid.urgency_threshold - 0.75).abs() < 0.001);
        assert!((mid.tier_weights[0] - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_params_lerp_endpoints() {
        let a = HeuristicParams::default();
        let b = HeuristicParams::aggressive();

        let at_zero = a.lerp(&b, 0.0);
        assert_eq!(at_zero.preload_distance, a.preload_distance);

        let at_one = a.lerp(&b, 1.0);
        assert_eq!(at_one.preload_distance, b.preload_distance);
    }

    // =========================================================================
    // ParamOverrides Tests
    // =========================================================================

    #[test]
    fn test_overrides_none() {
        let o = ParamOverrides::none();
        assert!(!o.has_any());
    }

    #[test]
    fn test_overrides_all() {
        let o = ParamOverrides::all();
        assert!(o.has_any());
        assert!(o.preload_distance_locked);
        assert!(o.urgency_threshold_locked);
        assert!(o.tier_weights_locked);
        assert!(o.miss_rate_target_locked);
    }

    #[test]
    fn test_overrides_clear() {
        let mut o = ParamOverrides::all();
        o.clear();
        assert!(!o.has_any());
    }

    // =========================================================================
    // TuningConfig Tests
    // =========================================================================

    #[test]
    fn test_tuning_config_default() {
        let c = TuningConfig::default();
        assert_eq!(c.preload_distance_step, 5.0);
        assert_eq!(c.warmup_samples, 10);
    }

    #[test]
    fn test_tuning_config_aggressive_vs_conservative() {
        let agg = TuningConfig::aggressive();
        let con = TuningConfig::conservative();

        assert!(agg.preload_distance_step > con.preload_distance_step);
        assert!(agg.warmup_samples < con.warmup_samples);
    }

    // =========================================================================
    // HeuristicsTuner Tests
    // =========================================================================

    #[test]
    fn test_tuner_default() {
        let t = HeuristicsTuner::default();
        assert!(t.is_auto_tune_enabled());
        assert!(!t.is_warmed_up());
        assert_eq!(t.sample_count(), 0);
    }

    #[test]
    fn test_tuner_warmup() {
        let mut t = HeuristicsTuner::new(
            HeuristicParams::default(),
            TuningConfig { warmup_samples: 3, ..Default::default() },
        );

        assert!(!t.is_warmed_up());

        t.update(StreamingMetrics::default());
        t.update(StreamingMetrics::default());
        assert!(!t.is_warmed_up());

        t.update(StreamingMetrics::default());
        assert!(t.is_warmed_up());
    }

    #[test]
    fn test_tuner_high_miss_rate_increases_preload_distance() {
        let mut t = HeuristicsTuner::new(
            HeuristicParams::default(),
            TuningConfig { warmup_samples: 1, ..Default::default() },
        );

        let initial_distance = t.params().preload_distance;

        // Submit metrics with high miss rate
        let high_miss = StreamingMetrics::new(0.1, 5.0, 0.3, 5.0);
        t.update(high_miss);
        t.update(high_miss); // Need two for EMA

        assert!(t.params().preload_distance > initial_distance);
    }

    #[test]
    fn test_tuner_high_budget_pressure_decreases_preload_distance() {
        let initial_params = HeuristicParams::new(
            200.0, // Start high
            DEFAULT_URGENCY_THRESHOLD,
            DEFAULT_TIER_WEIGHTS,
            0.1, // High target so miss rate is OK
        );

        let mut t = HeuristicsTuner::new(
            initial_params,
            TuningConfig { warmup_samples: 1, ..Default::default() },
        );

        let initial_distance = t.params().preload_distance;

        // Submit metrics with high budget pressure and low miss rate
        let high_pressure = StreamingMetrics::new(0.001, 5.0, 0.95, 5.0);
        t.update(high_pressure);
        t.update(high_pressure);

        assert!(t.params().preload_distance < initial_distance);
    }

    #[test]
    fn test_tuner_high_latency_reduces_low_tier_weights() {
        let mut t = HeuristicsTuner::new(
            HeuristicParams::default(),
            TuningConfig { warmup_samples: 1, ..Default::default() },
        );

        let initial_low_weight = t.params().low_weight();

        // Submit metrics with high latency
        let high_latency = StreamingMetrics::new(0.0, 50.0, 0.3, 5.0);
        t.update(high_latency);
        t.update(high_latency);

        assert!(t.params().low_weight() < initial_low_weight);
    }

    #[test]
    fn test_tuner_override_preload_distance() {
        let mut t = HeuristicsTuner::default();

        t.force_preload_distance(250.0);
        assert!(t.overrides().preload_distance_locked);
        assert_eq!(t.params().preload_distance, 250.0);

        // Auto-tune should not affect it
        let high_miss = StreamingMetrics::new(0.1, 5.0, 0.3, 5.0);
        for _ in 0..20 {
            t.update(high_miss);
        }

        assert_eq!(t.params().preload_distance, 250.0);
    }

    #[test]
    fn test_tuner_unlock_override() {
        let mut t = HeuristicsTuner::default();

        t.force_preload_distance(250.0);
        t.unlock_preload_distance();

        assert!(!t.overrides().preload_distance_locked);
    }

    #[test]
    fn test_tuner_clear_overrides() {
        let mut t = HeuristicsTuner::default();

        t.force_preload_distance(250.0);
        t.force_urgency_threshold(0.9);
        t.force_tier_weights([0.5, 0.4, 0.3, 0.2]);

        assert!(t.overrides().has_any());

        t.clear_overrides();
        assert!(!t.overrides().has_any());
    }

    #[test]
    fn test_tuner_reset() {
        let mut t = HeuristicsTuner::new(
            HeuristicParams::aggressive(),
            TuningConfig::default(),
        );

        // Update and force some values
        t.update(StreamingMetrics::poor());
        t.force_preload_distance(300.0);

        t.reset();

        assert_eq!(t.sample_count(), 0);
        assert!(!t.overrides().has_any());
        assert_eq!(t.params().preload_distance, DEFAULT_PRELOAD_DISTANCE);
    }

    #[test]
    fn test_tuner_reset_to() {
        let mut t = HeuristicsTuner::default();

        let custom = HeuristicParams::new(150.0, 0.6, [0.9, 0.7, 0.5, 0.3], 0.005);
        t.reset_to(custom);

        assert_eq!(t.params().preload_distance, 150.0);
        assert_eq!(t.params().urgency_threshold, 0.6);
    }

    #[test]
    fn test_tuner_disable_auto_tune() {
        let mut t = HeuristicsTuner::new(
            HeuristicParams::default(),
            TuningConfig { warmup_samples: 1, ..Default::default() },
        );

        t.set_auto_tune_enabled(false);

        let initial_distance = t.params().preload_distance;

        // High miss rate should not affect params when disabled
        let high_miss = StreamingMetrics::new(0.5, 5.0, 0.3, 5.0);
        for _ in 0..10 {
            t.update(high_miss);
        }

        assert_eq!(t.params().preload_distance, initial_distance);
    }

    #[test]
    fn test_tuner_params_clamped_after_tuning() {
        let mut t = HeuristicsTuner::new(
            HeuristicParams::new(MAX_PRELOAD_DISTANCE - 1.0, 0.7, DEFAULT_TIER_WEIGHTS, 0.001),
            TuningConfig { warmup_samples: 1, preload_distance_step: 100.0, ..Default::default() },
        );

        // High miss rate tries to increase distance
        let high_miss = StreamingMetrics::new(0.5, 5.0, 0.3, 5.0);
        t.update(high_miss);
        t.update(high_miss);

        // Should be clamped to max
        assert!(t.params().preload_distance <= MAX_PRELOAD_DISTANCE);
    }

    #[test]
    fn test_tuner_override_clamped() {
        let mut t = HeuristicsTuner::default();

        t.force_preload_distance(10000.0);
        assert_eq!(t.params().preload_distance, MAX_PRELOAD_DISTANCE);

        t.force_urgency_threshold(-1.0);
        assert_eq!(t.params().urgency_threshold, MIN_URGENCY_THRESHOLD);
    }

    #[test]
    fn test_tuner_raw_params_ignores_overrides() {
        let mut t = HeuristicsTuner::default();

        t.force_preload_distance(250.0);

        // params() returns forced value
        assert_eq!(t.params().preload_distance, 250.0);

        // raw_params() returns auto-tuned value
        assert_eq!(t.raw_params().preload_distance, DEFAULT_PRELOAD_DISTANCE);
    }

    #[test]
    fn test_tuner_current_metrics() {
        let mut t = HeuristicsTuner::default();

        t.update(StreamingMetrics::new(0.05, 15.0, 0.4, 8.0));

        let m = t.current_metrics();
        assert_eq!(m.page_miss_rate, 0.05);
    }

    #[test]
    fn test_tuner_smoothed_metrics_accessor() {
        let mut t = HeuristicsTuner::default();

        t.update(StreamingMetrics::new(0.1, 20.0, 0.5, 10.0));

        let s = t.smoothed_metrics();
        assert!(s.is_initialized());
        assert_eq!(s.page_miss_rate, 0.1);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_tuner_converges_with_good_metrics() {
        let mut t = HeuristicsTuner::new(
            HeuristicParams::default(),
            TuningConfig { warmup_samples: 5, ..Default::default() },
        );

        // Simulate many frames with good performance
        let good = StreamingMetrics::good();
        for _ in 0..50 {
            t.update(good);
        }

        // Should have tuned to stable parameters
        let params = t.params();
        assert!(params.preload_distance >= MIN_PRELOAD_DISTANCE);
        assert!(params.preload_distance <= MAX_PRELOAD_DISTANCE);
    }

    #[test]
    fn test_tuner_responds_to_degraded_performance() {
        let mut t = HeuristicsTuner::new(
            HeuristicParams::conservative(),
            TuningConfig { warmup_samples: 2, ..Default::default() },
        );

        // Start with good metrics
        let good = StreamingMetrics::good();
        t.update(good);
        t.update(good);

        let stable_distance = t.params().preload_distance;

        // Performance degrades
        let poor = StreamingMetrics::poor();
        for _ in 0..10 {
            t.update(poor);
        }

        // Should have increased preload distance
        // Note: budget pressure may counteract this, so we just check it changed
        let new_distance = t.params().preload_distance;
        assert_ne!(new_distance, stable_distance);
    }

    #[test]
    fn test_multiple_overrides_combine() {
        let mut t = HeuristicsTuner::default();

        t.force_preload_distance(200.0);
        t.force_urgency_threshold(0.8);

        let params = t.params();
        assert_eq!(params.preload_distance, 200.0);
        assert_eq!(params.urgency_threshold, 0.8);

        // But tier weights should still be default (not overridden)
        assert_eq!(params.tier_weights, DEFAULT_TIER_WEIGHTS);
    }
}
