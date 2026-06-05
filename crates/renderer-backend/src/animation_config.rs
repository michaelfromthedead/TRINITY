//! AnimationConfig Resource - Global animation system configuration.
//!
//! This module provides the global configuration resource for the animation system,
//! including settings for IK chains, motion matching budgets, LOD distances, and
//! skeleton constraints.
//!
//! # Example
//!
//! ```rust
//! use renderer_backend::animation_config::{AnimationConfig, QualityPreset};
//!
//! // Create with builder pattern
//! let config = AnimationConfig::builder()
//!     .global_speed(1.5)
//!     .max_active_ik_chains(8)
//!     .motion_matching_budget_ms(2.0)
//!     .build()
//!     .expect("valid config");
//!
//! // Or use a quality preset
//! let high_quality = AnimationConfig::with_performance_mode(QualityPreset::High);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

/// Error type for animation configuration validation.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    /// Global speed must be positive and finite.
    InvalidGlobalSpeed(f32),
    /// Motion matching budget must be positive and finite.
    InvalidMotionMatchingBudget(f32),
    /// Default blend time must be positive and finite.
    InvalidDefaultBlendTime(f32),
    /// Tick rate must be positive and finite.
    InvalidTickRate(f32),
    /// LOD distances must be in ascending order.
    InvalidLodDistanceOrder {
        full_quality: f32,
        reduced_quality: f32,
        minimal_quality: f32,
        culled: f32,
    },
    /// LOD distance must be non-negative.
    NegativeLodDistance(f32),
    /// Max bones must be at least 1.
    InvalidMaxBones(u32),
    /// Max IK chains must be at least 1 (or 0 to disable).
    InvalidMaxIkChains(u32),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGlobalSpeed(v) => {
                write!(f, "global_speed must be positive and finite, got {v}")
            }
            Self::InvalidMotionMatchingBudget(v) => {
                write!(
                    f,
                    "motion_matching_budget_ms must be positive and finite, got {v}"
                )
            }
            Self::InvalidDefaultBlendTime(v) => {
                write!(f, "default_blend_time must be positive and finite, got {v}")
            }
            Self::InvalidTickRate(v) => {
                write!(f, "tick_rate must be positive and finite, got {v}")
            }
            Self::InvalidLodDistanceOrder {
                full_quality,
                reduced_quality,
                minimal_quality,
                culled,
            } => {
                write!(
                    f,
                    "LOD distances must be in ascending order: full_quality({full_quality}) < \
                     reduced_quality({reduced_quality}) < minimal_quality({minimal_quality}) < \
                     culled({culled})"
                )
            }
            Self::NegativeLodDistance(v) => {
                write!(f, "LOD distance must be non-negative, got {v}")
            }
            Self::InvalidMaxBones(v) => {
                write!(f, "max_bones_per_skeleton must be at least 1, got {v}")
            }
            Self::InvalidMaxIkChains(v) => {
                write!(
                    f,
                    "max_active_ik_chains validation error (internal): {v}"
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// LOD (Level of Detail) distance thresholds for animation quality.
///
/// These distances determine when to switch between animation quality levels
/// based on the distance from the camera.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LodDistances {
    /// Distance below which full quality animation is used.
    /// Full bone updates, full blend precision, full IK solving.
    pub full_quality: f32,

    /// Distance below which reduced quality animation is used.
    /// Reduced keyframe interpolation, simplified IK.
    pub reduced_quality: f32,

    /// Distance below which minimal quality animation is used.
    /// Minimal bone updates, snapped keyframes.
    pub minimal_quality: f32,

    /// Distance at or beyond which animation is culled entirely.
    /// No animation updates, static pose or T-pose.
    pub culled: f32,
}

impl Default for LodDistances {
    fn default() -> Self {
        Self {
            full_quality: 10.0,
            reduced_quality: 25.0,
            minimal_quality: 50.0,
            culled: 100.0,
        }
    }
}

impl LodDistances {
    /// Creates LOD distances with explicit values.
    pub fn new(
        full_quality: f32,
        reduced_quality: f32,
        minimal_quality: f32,
        culled: f32,
    ) -> Self {
        Self {
            full_quality,
            reduced_quality,
            minimal_quality,
            culled,
        }
    }

    /// Validates that LOD distances are in ascending order and non-negative.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Check for negative values
        if self.full_quality < 0.0 {
            return Err(ConfigError::NegativeLodDistance(self.full_quality));
        }
        if self.reduced_quality < 0.0 {
            return Err(ConfigError::NegativeLodDistance(self.reduced_quality));
        }
        if self.minimal_quality < 0.0 {
            return Err(ConfigError::NegativeLodDistance(self.minimal_quality));
        }
        if self.culled < 0.0 {
            return Err(ConfigError::NegativeLodDistance(self.culled));
        }

        // Check ascending order
        if !(self.full_quality < self.reduced_quality
            && self.reduced_quality < self.minimal_quality
            && self.minimal_quality < self.culled)
        {
            return Err(ConfigError::InvalidLodDistanceOrder {
                full_quality: self.full_quality,
                reduced_quality: self.reduced_quality,
                minimal_quality: self.minimal_quality,
                culled: self.culled,
            });
        }

        Ok(())
    }

    /// Returns the quality level for a given distance.
    pub fn quality_for_distance(&self, distance: f32) -> AnimationQualityLevel {
        if distance < self.full_quality {
            AnimationQualityLevel::Full
        } else if distance < self.reduced_quality {
            AnimationQualityLevel::Reduced
        } else if distance < self.minimal_quality {
            AnimationQualityLevel::Minimal
        } else {
            AnimationQualityLevel::Culled
        }
    }
}

/// Animation quality level based on LOD distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnimationQualityLevel {
    /// Full quality: all bones, full precision, full IK.
    Full,
    /// Reduced quality: fewer keyframes, simplified IK.
    Reduced,
    /// Minimal quality: minimal updates, snapped keyframes.
    Minimal,
    /// Culled: no animation updates.
    Culled,
}

/// Quality preset for quick configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QualityPreset {
    /// Low quality: conservative settings for low-end hardware.
    Low,
    /// Medium quality: balanced settings for mid-range hardware.
    Medium,
    /// High quality: high-fidelity settings for high-end hardware.
    High,
    /// Ultra quality: maximum quality, no compromises.
    Ultra,
}

/// Global animation system configuration resource.
///
/// This struct holds all configurable parameters for the animation system,
/// including performance budgets, LOD settings, and skeleton constraints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimationConfig {
    /// Global animation speed multiplier (1.0 = normal speed).
    /// Values less than 1.0 slow down animations, greater than 1.0 speed them up.
    pub global_speed: f32,

    /// Maximum number of active IK (Inverse Kinematics) chains per frame.
    /// Limits computational cost of IK solving. Set to 0 to disable IK.
    pub max_active_ik_chains: u32,

    /// Budget in milliseconds for motion matching per frame.
    /// Motion matching searches animation databases for best-matching poses.
    pub motion_matching_budget_ms: f32,

    /// LOD distance thresholds for animation quality.
    pub lod_distances: LodDistances,

    /// Default blend time in seconds for animation transitions.
    pub default_blend_time: f32,

    /// Maximum bones per skeleton.
    /// Skeletons exceeding this limit will be rejected during import.
    pub max_bones_per_skeleton: u32,

    /// Animation tick rate (updates per second).
    /// Higher values provide smoother animation but increase CPU cost.
    pub tick_rate: f32,
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            global_speed: 1.0,
            max_active_ik_chains: 16,
            motion_matching_budget_ms: 2.0,
            lod_distances: LodDistances::default(),
            default_blend_time: 0.2,
            max_bones_per_skeleton: 256,
            tick_rate: 60.0,
        }
    }
}

impl AnimationConfig {
    /// Creates a new AnimationConfig builder.
    pub fn builder() -> AnimationConfigBuilder {
        AnimationConfigBuilder::new()
    }

    /// Validates the configuration for consistency and valid ranges.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate global_speed
        if !self.global_speed.is_finite() || self.global_speed <= 0.0 {
            return Err(ConfigError::InvalidGlobalSpeed(self.global_speed));
        }

        // Validate motion_matching_budget_ms
        if !self.motion_matching_budget_ms.is_finite() || self.motion_matching_budget_ms <= 0.0 {
            return Err(ConfigError::InvalidMotionMatchingBudget(
                self.motion_matching_budget_ms,
            ));
        }

        // Validate default_blend_time
        if !self.default_blend_time.is_finite() || self.default_blend_time <= 0.0 {
            return Err(ConfigError::InvalidDefaultBlendTime(self.default_blend_time));
        }

        // Validate tick_rate
        if !self.tick_rate.is_finite() || self.tick_rate <= 0.0 {
            return Err(ConfigError::InvalidTickRate(self.tick_rate));
        }

        // Validate max_bones_per_skeleton
        if self.max_bones_per_skeleton == 0 {
            return Err(ConfigError::InvalidMaxBones(self.max_bones_per_skeleton));
        }

        // Validate LOD distances
        self.lod_distances.validate()?;

        Ok(())
    }

    /// Creates a configuration optimized for a specific quality preset.
    pub fn with_performance_mode(quality: QualityPreset) -> Self {
        match quality {
            QualityPreset::Low => Self {
                global_speed: 1.0,
                max_active_ik_chains: 4,
                motion_matching_budget_ms: 0.5,
                lod_distances: LodDistances {
                    full_quality: 5.0,
                    reduced_quality: 15.0,
                    minimal_quality: 30.0,
                    culled: 50.0,
                },
                default_blend_time: 0.15,
                max_bones_per_skeleton: 128,
                tick_rate: 30.0,
            },
            QualityPreset::Medium => Self {
                global_speed: 1.0,
                max_active_ik_chains: 8,
                motion_matching_budget_ms: 1.0,
                lod_distances: LodDistances {
                    full_quality: 8.0,
                    reduced_quality: 20.0,
                    minimal_quality: 40.0,
                    culled: 75.0,
                },
                default_blend_time: 0.2,
                max_bones_per_skeleton: 192,
                tick_rate: 45.0,
            },
            QualityPreset::High => Self {
                global_speed: 1.0,
                max_active_ik_chains: 16,
                motion_matching_budget_ms: 2.0,
                lod_distances: LodDistances {
                    full_quality: 10.0,
                    reduced_quality: 25.0,
                    minimal_quality: 50.0,
                    culled: 100.0,
                },
                default_blend_time: 0.2,
                max_bones_per_skeleton: 256,
                tick_rate: 60.0,
            },
            QualityPreset::Ultra => Self {
                global_speed: 1.0,
                max_active_ik_chains: 32,
                motion_matching_budget_ms: 4.0,
                lod_distances: LodDistances {
                    full_quality: 15.0,
                    reduced_quality: 35.0,
                    minimal_quality: 70.0,
                    culled: 150.0,
                },
                default_blend_time: 0.25,
                max_bones_per_skeleton: 512,
                tick_rate: 120.0,
            },
        }
    }

    /// Loads configuration from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serializes configuration to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serializes configuration to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Loads configuration from a JSON file and validates it.
    pub fn load_and_validate(json: &str) -> Result<Self, ConfigLoadError> {
        let config = Self::from_json(json).map_err(ConfigLoadError::ParseError)?;
        config.validate().map_err(ConfigLoadError::ValidationError)?;
        Ok(config)
    }
}

/// Error type for loading configuration from JSON.
#[derive(Debug)]
pub enum ConfigLoadError {
    /// JSON parsing error.
    ParseError(serde_json::Error),
    /// Validation error after parsing.
    ValidationError(ConfigError),
}

impl fmt::Display for ConfigLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError(e) => write!(f, "JSON parse error: {e}"),
            Self::ValidationError(e) => write!(f, "validation error: {e}"),
        }
    }
}

impl std::error::Error for ConfigLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ParseError(e) => Some(e),
            Self::ValidationError(e) => Some(e),
        }
    }
}

/// Builder for AnimationConfig with fluent API.
#[derive(Debug, Clone)]
pub struct AnimationConfigBuilder {
    config: AnimationConfig,
}

impl AnimationConfigBuilder {
    /// Creates a new builder with default values.
    pub fn new() -> Self {
        Self {
            config: AnimationConfig::default(),
        }
    }

    /// Sets the global animation speed multiplier.
    pub fn global_speed(mut self, speed: f32) -> Self {
        self.config.global_speed = speed;
        self
    }

    /// Sets the maximum number of active IK chains per frame.
    pub fn max_active_ik_chains(mut self, count: u32) -> Self {
        self.config.max_active_ik_chains = count;
        self
    }

    /// Sets the motion matching budget in milliseconds.
    pub fn motion_matching_budget_ms(mut self, budget: f32) -> Self {
        self.config.motion_matching_budget_ms = budget;
        self
    }

    /// Sets the LOD distances.
    pub fn lod_distances(mut self, distances: LodDistances) -> Self {
        self.config.lod_distances = distances;
        self
    }

    /// Sets individual LOD distance thresholds.
    pub fn lod_distance_thresholds(
        mut self,
        full_quality: f32,
        reduced_quality: f32,
        minimal_quality: f32,
        culled: f32,
    ) -> Self {
        self.config.lod_distances = LodDistances::new(
            full_quality,
            reduced_quality,
            minimal_quality,
            culled,
        );
        self
    }

    /// Sets the default blend time in seconds.
    pub fn default_blend_time(mut self, time: f32) -> Self {
        self.config.default_blend_time = time;
        self
    }

    /// Sets the maximum bones per skeleton.
    pub fn max_bones_per_skeleton(mut self, count: u32) -> Self {
        self.config.max_bones_per_skeleton = count;
        self
    }

    /// Sets the animation tick rate (updates per second).
    pub fn tick_rate(mut self, rate: f32) -> Self {
        self.config.tick_rate = rate;
        self
    }

    /// Applies a quality preset to the builder.
    pub fn with_preset(mut self, preset: QualityPreset) -> Self {
        self.config = AnimationConfig::with_performance_mode(preset);
        self
    }

    /// Builds the AnimationConfig, validating all parameters.
    pub fn build(self) -> Result<AnimationConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Builds the AnimationConfig without validation.
    /// Use with caution - invalid configurations may cause runtime errors.
    pub fn build_unchecked(self) -> AnimationConfig {
        self.config
    }
}

impl Default for AnimationConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Default Value Tests ====================

    #[test]
    fn test_default_values() {
        let config = AnimationConfig::default();
        assert_eq!(config.global_speed, 1.0);
        assert_eq!(config.max_active_ik_chains, 16);
        assert_eq!(config.motion_matching_budget_ms, 2.0);
        assert_eq!(config.default_blend_time, 0.2);
        assert_eq!(config.max_bones_per_skeleton, 256);
        assert_eq!(config.tick_rate, 60.0);
    }

    #[test]
    fn test_default_lod_distances() {
        let lod = LodDistances::default();
        assert_eq!(lod.full_quality, 10.0);
        assert_eq!(lod.reduced_quality, 25.0);
        assert_eq!(lod.minimal_quality, 50.0);
        assert_eq!(lod.culled, 100.0);
    }

    #[test]
    fn test_default_config_validates() {
        let config = AnimationConfig::default();
        assert!(config.validate().is_ok());
    }

    // ==================== Builder Pattern Tests ====================

    #[test]
    fn test_builder_default() {
        let config = AnimationConfig::builder().build().unwrap();
        assert_eq!(config, AnimationConfig::default());
    }

    #[test]
    fn test_builder_global_speed() {
        let config = AnimationConfig::builder()
            .global_speed(2.0)
            .build()
            .unwrap();
        assert_eq!(config.global_speed, 2.0);
    }

    #[test]
    fn test_builder_max_active_ik_chains() {
        let config = AnimationConfig::builder()
            .max_active_ik_chains(32)
            .build()
            .unwrap();
        assert_eq!(config.max_active_ik_chains, 32);
    }

    #[test]
    fn test_builder_motion_matching_budget() {
        let config = AnimationConfig::builder()
            .motion_matching_budget_ms(4.0)
            .build()
            .unwrap();
        assert_eq!(config.motion_matching_budget_ms, 4.0);
    }

    #[test]
    fn test_builder_lod_distances() {
        let lod = LodDistances::new(5.0, 15.0, 30.0, 60.0);
        let config = AnimationConfig::builder()
            .lod_distances(lod)
            .build()
            .unwrap();
        assert_eq!(config.lod_distances.full_quality, 5.0);
        assert_eq!(config.lod_distances.reduced_quality, 15.0);
        assert_eq!(config.lod_distances.minimal_quality, 30.0);
        assert_eq!(config.lod_distances.culled, 60.0);
    }

    #[test]
    fn test_builder_lod_distance_thresholds() {
        let config = AnimationConfig::builder()
            .lod_distance_thresholds(3.0, 10.0, 20.0, 40.0)
            .build()
            .unwrap();
        assert_eq!(config.lod_distances.full_quality, 3.0);
        assert_eq!(config.lod_distances.reduced_quality, 10.0);
        assert_eq!(config.lod_distances.minimal_quality, 20.0);
        assert_eq!(config.lod_distances.culled, 40.0);
    }

    #[test]
    fn test_builder_default_blend_time() {
        let config = AnimationConfig::builder()
            .default_blend_time(0.5)
            .build()
            .unwrap();
        assert_eq!(config.default_blend_time, 0.5);
    }

    #[test]
    fn test_builder_max_bones() {
        let config = AnimationConfig::builder()
            .max_bones_per_skeleton(512)
            .build()
            .unwrap();
        assert_eq!(config.max_bones_per_skeleton, 512);
    }

    #[test]
    fn test_builder_tick_rate() {
        let config = AnimationConfig::builder()
            .tick_rate(120.0)
            .build()
            .unwrap();
        assert_eq!(config.tick_rate, 120.0);
    }

    #[test]
    fn test_builder_chained() {
        let config = AnimationConfig::builder()
            .global_speed(1.5)
            .max_active_ik_chains(8)
            .motion_matching_budget_ms(3.0)
            .default_blend_time(0.3)
            .max_bones_per_skeleton(384)
            .tick_rate(90.0)
            .build()
            .unwrap();

        assert_eq!(config.global_speed, 1.5);
        assert_eq!(config.max_active_ik_chains, 8);
        assert_eq!(config.motion_matching_budget_ms, 3.0);
        assert_eq!(config.default_blend_time, 0.3);
        assert_eq!(config.max_bones_per_skeleton, 384);
        assert_eq!(config.tick_rate, 90.0);
    }

    #[test]
    fn test_builder_with_preset() {
        let config = AnimationConfig::builder()
            .with_preset(QualityPreset::High)
            .build()
            .unwrap();
        assert_eq!(config, AnimationConfig::with_performance_mode(QualityPreset::High));
    }

    #[test]
    fn test_builder_unchecked() {
        // Build without validation - can create invalid configs
        let config = AnimationConfig::builder()
            .global_speed(-1.0) // Invalid!
            .build_unchecked();
        assert_eq!(config.global_speed, -1.0);
        // But validation should fail
        assert!(config.validate().is_err());
    }

    // ==================== Validation Tests ====================

    #[test]
    fn test_validate_invalid_global_speed_zero() {
        let result = AnimationConfig::builder()
            .global_speed(0.0)
            .build();
        assert!(matches!(result, Err(ConfigError::InvalidGlobalSpeed(_))));
    }

    #[test]
    fn test_validate_invalid_global_speed_negative() {
        let result = AnimationConfig::builder()
            .global_speed(-1.0)
            .build();
        assert!(matches!(result, Err(ConfigError::InvalidGlobalSpeed(_))));
    }

    #[test]
    fn test_validate_invalid_global_speed_nan() {
        let result = AnimationConfig::builder()
            .global_speed(f32::NAN)
            .build();
        assert!(matches!(result, Err(ConfigError::InvalidGlobalSpeed(_))));
    }

    #[test]
    fn test_validate_invalid_global_speed_infinity() {
        let result = AnimationConfig::builder()
            .global_speed(f32::INFINITY)
            .build();
        assert!(matches!(result, Err(ConfigError::InvalidGlobalSpeed(_))));
    }

    #[test]
    fn test_validate_invalid_motion_matching_budget_zero() {
        let result = AnimationConfig::builder()
            .motion_matching_budget_ms(0.0)
            .build();
        assert!(matches!(
            result,
            Err(ConfigError::InvalidMotionMatchingBudget(_))
        ));
    }

    #[test]
    fn test_validate_invalid_motion_matching_budget_negative() {
        let result = AnimationConfig::builder()
            .motion_matching_budget_ms(-1.0)
            .build();
        assert!(matches!(
            result,
            Err(ConfigError::InvalidMotionMatchingBudget(_))
        ));
    }

    #[test]
    fn test_validate_invalid_default_blend_time_zero() {
        let result = AnimationConfig::builder()
            .default_blend_time(0.0)
            .build();
        assert!(matches!(result, Err(ConfigError::InvalidDefaultBlendTime(_))));
    }

    #[test]
    fn test_validate_invalid_tick_rate_zero() {
        let result = AnimationConfig::builder()
            .tick_rate(0.0)
            .build();
        assert!(matches!(result, Err(ConfigError::InvalidTickRate(_))));
    }

    #[test]
    fn test_validate_invalid_max_bones_zero() {
        let result = AnimationConfig::builder()
            .max_bones_per_skeleton(0)
            .build();
        assert!(matches!(result, Err(ConfigError::InvalidMaxBones(_))));
    }

    #[test]
    fn test_validate_lod_distances_wrong_order() {
        let result = AnimationConfig::builder()
            .lod_distance_thresholds(20.0, 15.0, 10.0, 5.0) // Descending instead of ascending
            .build();
        assert!(matches!(result, Err(ConfigError::InvalidLodDistanceOrder { .. })));
    }

    #[test]
    fn test_validate_lod_distances_equal_values() {
        let result = AnimationConfig::builder()
            .lod_distance_thresholds(10.0, 10.0, 20.0, 30.0) // Equal values
            .build();
        assert!(matches!(result, Err(ConfigError::InvalidLodDistanceOrder { .. })));
    }

    #[test]
    fn test_validate_lod_distances_negative() {
        let result = AnimationConfig::builder()
            .lod_distance_thresholds(-5.0, 10.0, 20.0, 30.0)
            .build();
        assert!(matches!(result, Err(ConfigError::NegativeLodDistance(_))));
    }

    // ==================== Quality Preset Tests ====================

    #[test]
    fn test_quality_preset_low() {
        let config = AnimationConfig::with_performance_mode(QualityPreset::Low);
        assert!(config.validate().is_ok());
        assert_eq!(config.max_active_ik_chains, 4);
        assert_eq!(config.motion_matching_budget_ms, 0.5);
        assert_eq!(config.tick_rate, 30.0);
        assert_eq!(config.max_bones_per_skeleton, 128);
    }

    #[test]
    fn test_quality_preset_medium() {
        let config = AnimationConfig::with_performance_mode(QualityPreset::Medium);
        assert!(config.validate().is_ok());
        assert_eq!(config.max_active_ik_chains, 8);
        assert_eq!(config.motion_matching_budget_ms, 1.0);
        assert_eq!(config.tick_rate, 45.0);
    }

    #[test]
    fn test_quality_preset_high() {
        let config = AnimationConfig::with_performance_mode(QualityPreset::High);
        assert!(config.validate().is_ok());
        assert_eq!(config.max_active_ik_chains, 16);
        assert_eq!(config.motion_matching_budget_ms, 2.0);
        assert_eq!(config.tick_rate, 60.0);
    }

    #[test]
    fn test_quality_preset_ultra() {
        let config = AnimationConfig::with_performance_mode(QualityPreset::Ultra);
        assert!(config.validate().is_ok());
        assert_eq!(config.max_active_ik_chains, 32);
        assert_eq!(config.motion_matching_budget_ms, 4.0);
        assert_eq!(config.tick_rate, 120.0);
        assert_eq!(config.max_bones_per_skeleton, 512);
    }

    #[test]
    fn test_all_presets_validate() {
        for preset in [
            QualityPreset::Low,
            QualityPreset::Medium,
            QualityPreset::High,
            QualityPreset::Ultra,
        ] {
            let config = AnimationConfig::with_performance_mode(preset);
            assert!(config.validate().is_ok(), "Preset {:?} failed validation", preset);
        }
    }

    // ==================== Serialization Tests ====================

    #[test]
    fn test_serialization_roundtrip() {
        let original = AnimationConfig::default();
        let json = original.to_json().unwrap();
        let restored = AnimationConfig::from_json(&json).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn test_serialization_roundtrip_custom_config() {
        let original = AnimationConfig::builder()
            .global_speed(1.5)
            .max_active_ik_chains(24)
            .motion_matching_budget_ms(3.5)
            .lod_distance_thresholds(8.0, 20.0, 45.0, 90.0)
            .default_blend_time(0.35)
            .max_bones_per_skeleton(384)
            .tick_rate(75.0)
            .build()
            .unwrap();

        let json = original.to_json().unwrap();
        let restored = AnimationConfig::from_json(&json).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn test_serialization_roundtrip_all_presets() {
        for preset in [
            QualityPreset::Low,
            QualityPreset::Medium,
            QualityPreset::High,
            QualityPreset::Ultra,
        ] {
            let original = AnimationConfig::with_performance_mode(preset);
            let json = original.to_json().unwrap();
            let restored = AnimationConfig::from_json(&json).unwrap();
            assert_eq!(original, restored, "Roundtrip failed for {:?}", preset);
        }
    }

    #[test]
    fn test_to_json_pretty() {
        let config = AnimationConfig::default();
        let json = config.to_json_pretty().unwrap();
        assert!(json.contains('\n'));
        assert!(json.contains("global_speed"));
    }

    #[test]
    fn test_from_json_invalid() {
        let result = AnimationConfig::from_json("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_and_validate_success() {
        let json = r#"{
            "global_speed": 1.0,
            "max_active_ik_chains": 16,
            "motion_matching_budget_ms": 2.0,
            "lod_distances": {
                "full_quality": 10.0,
                "reduced_quality": 25.0,
                "minimal_quality": 50.0,
                "culled": 100.0
            },
            "default_blend_time": 0.2,
            "max_bones_per_skeleton": 256,
            "tick_rate": 60.0
        }"#;
        let result = AnimationConfig::load_and_validate(json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_and_validate_invalid_json() {
        let result = AnimationConfig::load_and_validate("invalid json");
        assert!(matches!(result, Err(ConfigLoadError::ParseError(_))));
    }

    #[test]
    fn test_load_and_validate_invalid_config() {
        let json = r#"{
            "global_speed": 0.0,
            "max_active_ik_chains": 16,
            "motion_matching_budget_ms": 2.0,
            "lod_distances": {
                "full_quality": 10.0,
                "reduced_quality": 25.0,
                "minimal_quality": 50.0,
                "culled": 100.0
            },
            "default_blend_time": 0.2,
            "max_bones_per_skeleton": 256,
            "tick_rate": 60.0
        }"#;
        let result = AnimationConfig::load_and_validate(json);
        assert!(matches!(result, Err(ConfigLoadError::ValidationError(_))));
    }

    // ==================== LOD Distance Tests ====================

    #[test]
    fn test_lod_quality_for_distance_full() {
        let lod = LodDistances::default();
        assert_eq!(lod.quality_for_distance(0.0), AnimationQualityLevel::Full);
        assert_eq!(lod.quality_for_distance(5.0), AnimationQualityLevel::Full);
        assert_eq!(lod.quality_for_distance(9.99), AnimationQualityLevel::Full);
    }

    #[test]
    fn test_lod_quality_for_distance_reduced() {
        let lod = LodDistances::default();
        assert_eq!(lod.quality_for_distance(10.0), AnimationQualityLevel::Reduced);
        assert_eq!(lod.quality_for_distance(20.0), AnimationQualityLevel::Reduced);
        assert_eq!(lod.quality_for_distance(24.99), AnimationQualityLevel::Reduced);
    }

    #[test]
    fn test_lod_quality_for_distance_minimal() {
        let lod = LodDistances::default();
        assert_eq!(lod.quality_for_distance(25.0), AnimationQualityLevel::Minimal);
        assert_eq!(lod.quality_for_distance(40.0), AnimationQualityLevel::Minimal);
        assert_eq!(lod.quality_for_distance(49.99), AnimationQualityLevel::Minimal);
    }

    #[test]
    fn test_lod_quality_for_distance_culled() {
        let lod = LodDistances::default();
        assert_eq!(lod.quality_for_distance(50.0), AnimationQualityLevel::Culled);
        assert_eq!(lod.quality_for_distance(100.0), AnimationQualityLevel::Culled);
        assert_eq!(lod.quality_for_distance(1000.0), AnimationQualityLevel::Culled);
    }

    // ==================== Error Display Tests ====================

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::InvalidGlobalSpeed(-1.0);
        let msg = format!("{}", err);
        assert!(msg.contains("global_speed"));
        assert!(msg.contains("-1"));
    }

    #[test]
    fn test_config_load_error_display() {
        let err = ConfigLoadError::ValidationError(ConfigError::InvalidGlobalSpeed(-1.0));
        let msg = format!("{}", err);
        assert!(msg.contains("validation error"));
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_very_small_valid_values() {
        let config = AnimationConfig::builder()
            .global_speed(0.001)
            .motion_matching_budget_ms(0.001)
            .default_blend_time(0.001)
            .tick_rate(0.001)
            .build();
        assert!(config.is_ok());
    }

    #[test]
    fn test_very_large_valid_values() {
        let config = AnimationConfig::builder()
            .global_speed(100.0)
            .max_active_ik_chains(1000)
            .motion_matching_budget_ms(100.0)
            .max_bones_per_skeleton(10000)
            .tick_rate(1000.0)
            .lod_distance_thresholds(100.0, 500.0, 1000.0, 5000.0)
            .build();
        assert!(config.is_ok());
    }

    #[test]
    fn test_ik_chains_zero_allowed() {
        // Zero IK chains is valid (disables IK)
        let config = AnimationConfig::builder()
            .max_active_ik_chains(0)
            .build();
        assert!(config.is_ok());
    }

    #[test]
    fn test_lod_distances_validate_independently() {
        let valid = LodDistances::new(1.0, 2.0, 3.0, 4.0);
        assert!(valid.validate().is_ok());

        let invalid = LodDistances::new(4.0, 3.0, 2.0, 1.0);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_lod_distances_new() {
        let lod = LodDistances::new(5.0, 15.0, 30.0, 60.0);
        assert_eq!(lod.full_quality, 5.0);
        assert_eq!(lod.reduced_quality, 15.0);
        assert_eq!(lod.minimal_quality, 30.0);
        assert_eq!(lod.culled, 60.0);
    }

    #[test]
    fn test_config_clone() {
        let original = AnimationConfig::default();
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_config_debug() {
        let config = AnimationConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("AnimationConfig"));
        assert!(debug.contains("global_speed"));
    }
}
