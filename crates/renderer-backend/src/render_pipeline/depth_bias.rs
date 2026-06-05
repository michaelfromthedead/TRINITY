//! Depth bias (polygon offset) configuration for render pipelines.
//!
//! This module provides depth bias abstractions for wgpu 25.x render pipelines
//! with validation, builder patterns, and common presets.
//!
//! # Depth Bias Overview
//!
//! Depth bias adjusts the depth value of rasterized fragments to prevent
//! z-fighting artifacts. It applies a depth offset calculated as:
//!
//! ```text
//! depth_bias = constant * factor + slope * max_slope
//! final_depth = clamp(fragment_depth + depth_bias, -clamp, clamp)
//! ```
//!
//! Where:
//! - `constant`: Fixed depth offset in depth buffer units
//! - `slope_scale`: Scales the maximum depth slope of the polygon
//! - `clamp`: Maximum absolute depth bias value (0 = no clamping)
//!
//! # Common Use Cases
//!
//! | Use Case | Constant | Slope Scale | Clamp | Notes |
//! |----------|----------|-------------|-------|-------|
//! | Shadow mapping | 2 | 2.0 | 0.0 | Prevents shadow acne |
//! | Decals | 1 | 1.0 | 0.0 | Overlay rendering |
//! | Outline rendering | -1 | -1.0 | 0.0 | Back-face offset |
//! | Polygon offset fill | 1 | 1.0 | 0.0 | Standard offset |
//!
//! # wgpu API Reference
//!
//! ```ignore
//! pub struct DepthBiasState {
//!     pub constant: i32,
//!     pub slope_scale: f32,
//!     pub clamp: f32,
//! }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::render_pipeline::depth_bias::{DepthBias, DepthBiasBuilder};
//!
//! // Shadow map preset
//! let shadow_bias = DepthBias::shadow_map();
//!
//! // Custom bias via builder
//! let custom_bias = DepthBiasBuilder::new()
//!     .constant(4)
//!     .slope_scale(1.5)
//!     .clamp(0.01)
//!     .build()?;
//!
//! // Convert to wgpu type
//! let wgpu_bias: wgpu::DepthBiasState = custom_bias.into();
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// DepthBias
// ---------------------------------------------------------------------------

/// Describes depth bias (polygon offset) settings for depth testing.
///
/// # Fields
///
/// | Field | Type | Description |
/// |-------|------|-------------|
/// | `constant` | `i32` | Fixed depth offset in depth buffer units |
/// | `slope_scale` | `f32` | Multiplier for maximum depth slope |
/// | `clamp` | `f32` | Maximum absolute bias value (0 = unclamped) |
///
/// # Depth Bias Calculation
///
/// The GPU calculates depth bias as:
/// ```text
/// bias = constant * r + slope_scale * max_slope
/// ```
///
/// Where `r` is the minimum resolvable depth difference and `max_slope`
/// is the maximum depth slope of the polygon being rasterized.
///
/// # Defaults
///
/// Default is no bias (all values zero), which is appropriate for most
/// standard rendering without z-fighting issues.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthBias {
    /// Constant depth bias added to each fragment.
    ///
    /// Measured in units of the minimum resolvable depth difference.
    /// Positive values push fragments away from the camera.
    pub constant: i32,

    /// Slope-scaled depth bias factor.
    ///
    /// Multiplied by the maximum depth slope of the polygon to produce
    /// additional bias proportional to the polygon's angle relative to
    /// the view direction.
    pub slope_scale: f32,

    /// Maximum absolute depth bias clamp value.
    ///
    /// When non-zero, the calculated bias is clamped to [-clamp, clamp].
    /// A value of 0.0 means no clamping (unlimited bias).
    pub clamp: f32,
}

impl Default for DepthBias {
    fn default() -> Self {
        Self {
            constant: 0,
            slope_scale: 0.0,
            clamp: 0.0,
        }
    }
}

impl DepthBias {
    /// Create a new depth bias with default values (no bias).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let bias = DepthBias::new();
    /// assert_eq!(bias.constant, 0);
    /// assert_eq!(bias.slope_scale, 0.0);
    /// assert_eq!(bias.clamp, 0.0);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a depth bias with no offset (all zeros).
    ///
    /// Equivalent to `default()` but more explicitly named.
    pub fn none() -> Self {
        Self::default()
    }

    /// Create a preset for shadow map rendering.
    ///
    /// Uses typical values to prevent shadow acne:
    /// - `constant`: 2
    /// - `slope_scale`: 2.0
    /// - `clamp`: 0.0 (no clamping)
    ///
    /// # Shadow Acne
    ///
    /// Shadow acne occurs when depth precision issues cause surfaces to
    /// incorrectly self-shadow. The bias pushes shadow map samples slightly
    /// toward the light, preventing these artifacts.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let shadow_bias = DepthBias::shadow_map();
    /// // Use in shadow pass pipeline
    /// ```
    pub fn shadow_map() -> Self {
        Self {
            constant: 2,
            slope_scale: 2.0,
            clamp: 0.0,
        }
    }

    /// Create a preset for cascaded shadow maps (CSM).
    ///
    /// Slightly higher bias values for larger shadow map coverage:
    /// - `constant`: 4
    /// - `slope_scale`: 3.0
    /// - `clamp`: 0.0
    ///
    /// CSM typically needs more aggressive bias due to lower depth
    /// precision in distant cascades.
    pub fn cascaded_shadow_map() -> Self {
        Self {
            constant: 4,
            slope_scale: 3.0,
            clamp: 0.0,
        }
    }

    /// Create a preset for polygon offset fill rendering.
    ///
    /// Standard polygon offset values:
    /// - `constant`: 1
    /// - `slope_scale`: 1.0
    /// - `clamp`: 0.0
    ///
    /// Useful for rendering filled polygons that overlap exactly
    /// with line or point primitives (e.g., wireframe over solid).
    pub fn polygon_offset() -> Self {
        Self {
            constant: 1,
            slope_scale: 1.0,
            clamp: 0.0,
        }
    }

    /// Create a preset for decal rendering.
    ///
    /// Conservative bias for decals projected onto surfaces:
    /// - `constant`: 1
    /// - `slope_scale`: 1.0
    /// - `clamp`: 0.0
    ///
    /// Decals need bias to render above the underlying surface without
    /// z-fighting, but not so much that they float visibly above it.
    pub fn decal() -> Self {
        Self {
            constant: 1,
            slope_scale: 1.0,
            clamp: 0.0,
        }
    }

    /// Create a preset for outline/silhouette rendering.
    ///
    /// Negative bias for back-face offset:
    /// - `constant`: -1
    /// - `slope_scale`: -1.0
    /// - `clamp`: 0.0
    ///
    /// Used when rendering back faces slightly pushed away for
    /// cartoon-style outlines.
    pub fn outline() -> Self {
        Self {
            constant: -1,
            slope_scale: -1.0,
            clamp: 0.0,
        }
    }

    /// Create a preset for contact shadows.
    ///
    /// Aggressive bias for screen-space contact shadows:
    /// - `constant`: 8
    /// - `slope_scale`: 4.0
    /// - `clamp`: 0.0
    pub fn contact_shadow() -> Self {
        Self {
            constant: 8,
            slope_scale: 4.0,
            clamp: 0.0,
        }
    }

    /// Set the constant depth bias.
    pub fn constant(mut self, constant: i32) -> Self {
        self.constant = constant;
        self
    }

    /// Set the slope-scaled depth bias.
    pub fn slope_scale(mut self, slope_scale: f32) -> Self {
        self.slope_scale = slope_scale;
        self
    }

    /// Set the depth bias clamp value.
    pub fn clamp(mut self, clamp: f32) -> Self {
        self.clamp = clamp;
        self
    }

    /// Validate the depth bias configuration.
    ///
    /// # Validation Rules
    ///
    /// - `clamp` must be >= 0.0 (negative clamp values are invalid)
    /// - `slope_scale` and `constant` can be any value (including negative)
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, or a `DepthBiasError` describing the issue.
    pub fn validate(&self) -> Result<(), DepthBiasError> {
        if self.clamp < 0.0 {
            return Err(DepthBiasError::NegativeClamp(self.clamp));
        }
        // NaN checks
        if self.slope_scale.is_nan() {
            return Err(DepthBiasError::InvalidSlopeScale(self.slope_scale));
        }
        if self.clamp.is_nan() {
            return Err(DepthBiasError::InvalidClamp(self.clamp));
        }
        Ok(())
    }

    /// Check if the depth bias configuration is valid.
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }

    /// Check if this depth bias represents "no bias" (all zeros).
    pub fn is_none(&self) -> bool {
        self.constant == 0 && self.slope_scale == 0.0 && self.clamp == 0.0
    }

    /// Check if the bias is non-zero (will affect depth values).
    pub fn is_active(&self) -> bool {
        !self.is_none()
    }

    /// Scale all bias values by a factor.
    ///
    /// Useful for adjusting bias intensity globally.
    ///
    /// # Arguments
    ///
    /// * `factor` - Multiplier for constant and slope_scale (clamp unchanged)
    pub fn scaled(mut self, factor: f32) -> Self {
        self.constant = (self.constant as f32 * factor) as i32;
        self.slope_scale *= factor;
        self
    }

    /// Create an inverted bias (negative of current values).
    ///
    /// Useful for effects like outline rendering where you want
    /// the opposite direction of normal bias.
    pub fn inverted(self) -> Self {
        Self {
            constant: -self.constant,
            slope_scale: -self.slope_scale,
            clamp: self.clamp, // Clamp stays positive
        }
    }
}

// Thread-safety: DepthBias contains only Copy types
unsafe impl Send for DepthBias {}
unsafe impl Sync for DepthBias {}

impl From<DepthBias> for wgpu::DepthBiasState {
    fn from(bias: DepthBias) -> Self {
        wgpu::DepthBiasState {
            constant: bias.constant,
            slope_scale: bias.slope_scale,
            clamp: bias.clamp,
        }
    }
}

impl From<wgpu::DepthBiasState> for DepthBias {
    fn from(state: wgpu::DepthBiasState) -> Self {
        Self {
            constant: state.constant,
            slope_scale: state.slope_scale,
            clamp: state.clamp,
        }
    }
}

impl From<(i32, f32, f32)> for DepthBias {
    fn from((constant, slope_scale, clamp): (i32, f32, f32)) -> Self {
        Self {
            constant,
            slope_scale,
            clamp,
        }
    }
}

// ---------------------------------------------------------------------------
// DepthBiasBuilder
// ---------------------------------------------------------------------------

/// Builder for creating depth bias configurations with fluent API and validation.
///
/// # Example
///
/// ```ignore
/// let bias = DepthBiasBuilder::new()
///     .constant(4)
///     .slope_scale(2.0)
///     .clamp(0.01)
///     .build()?;
/// ```
#[derive(Debug, Clone)]
pub struct DepthBiasBuilder {
    bias: DepthBias,
}

impl Default for DepthBiasBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DepthBiasBuilder {
    /// Create a new depth bias builder with default values (no bias).
    pub fn new() -> Self {
        Self {
            bias: DepthBias::default(),
        }
    }

    /// Start building from a preset.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Start from shadow map preset, then adjust
    /// let bias = DepthBiasBuilder::from_preset(DepthBias::shadow_map())
    ///     .constant(4)  // Override constant
    ///     .build()?;
    /// ```
    pub fn from_preset(preset: DepthBias) -> Self {
        Self { bias: preset }
    }

    /// Start building from a shadow map preset.
    pub fn shadow_map() -> Self {
        Self::from_preset(DepthBias::shadow_map())
    }

    /// Start building from a polygon offset preset.
    pub fn polygon_offset() -> Self {
        Self::from_preset(DepthBias::polygon_offset())
    }

    /// Set the constant depth bias.
    ///
    /// # Arguments
    ///
    /// * `constant` - Fixed offset in depth buffer units
    pub fn constant(mut self, constant: i32) -> Self {
        self.bias.constant = constant;
        self
    }

    /// Set the slope-scaled depth bias.
    ///
    /// # Arguments
    ///
    /// * `slope_scale` - Multiplier for maximum depth slope
    pub fn slope_scale(mut self, slope_scale: f32) -> Self {
        self.bias.slope_scale = slope_scale;
        self
    }

    /// Set the depth bias clamp value.
    ///
    /// # Arguments
    ///
    /// * `clamp` - Maximum absolute bias (0 = no clamping)
    pub fn clamp(mut self, clamp: f32) -> Self {
        self.bias.clamp = clamp;
        self
    }

    /// Scale all bias values by a factor.
    pub fn scale(mut self, factor: f32) -> Self {
        self.bias = self.bias.scaled(factor);
        self
    }

    /// Invert all bias values.
    pub fn invert(mut self) -> Self {
        self.bias = self.bias.inverted();
        self
    }

    /// Build the depth bias with validation.
    ///
    /// # Returns
    ///
    /// `Ok(DepthBias)` if valid, or an error describing the issue.
    pub fn build(self) -> Result<DepthBias, DepthBiasError> {
        self.bias.validate()?;
        Ok(self.bias)
    }

    /// Build the depth bias without validation.
    ///
    /// Use when you've already validated or need intentionally unusual values.
    pub fn build_unchecked(self) -> DepthBias {
        self.bias
    }
}

// ---------------------------------------------------------------------------
// DepthBiasInfo
// ---------------------------------------------------------------------------

/// Metadata about a depth bias preset configuration.
///
/// Provides descriptive information for tooling, debugging, and documentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthBiasInfo {
    /// Human-readable name for the preset.
    pub name: &'static str,
    /// Description of the depth bias configuration.
    pub description: &'static str,
    /// Typical use cases for this preset.
    pub use_cases: &'static [&'static str],
    /// The preset values.
    pub preset: DepthBias,
}

/// Common depth bias configurations with documentation.
pub const DEPTH_BIAS_PRESETS: [DepthBiasInfo; 7] = [
    DepthBiasInfo {
        name: "None",
        description: "No depth bias applied (all values zero)",
        use_cases: &["standard opaque rendering", "no z-fighting expected"],
        preset: DepthBias {
            constant: 0,
            slope_scale: 0.0,
            clamp: 0.0,
        },
    },
    DepthBiasInfo {
        name: "Shadow Map",
        description: "Standard depth bias for shadow mapping to prevent shadow acne",
        use_cases: &["directional shadows", "point light shadows", "spot light shadows"],
        preset: DepthBias {
            constant: 2,
            slope_scale: 2.0,
            clamp: 0.0,
        },
    },
    DepthBiasInfo {
        name: "Cascaded Shadow Map",
        description: "Higher bias for cascaded shadow maps with larger coverage",
        use_cases: &["CSM", "large outdoor scenes", "distant shadows"],
        preset: DepthBias {
            constant: 4,
            slope_scale: 3.0,
            clamp: 0.0,
        },
    },
    DepthBiasInfo {
        name: "Polygon Offset",
        description: "Standard polygon offset for overlapping geometry",
        use_cases: &["wireframe over solid", "decals", "coplanar surfaces"],
        preset: DepthBias {
            constant: 1,
            slope_scale: 1.0,
            clamp: 0.0,
        },
    },
    DepthBiasInfo {
        name: "Decal",
        description: "Conservative bias for projected decals",
        use_cases: &["bullet holes", "blood splatters", "graffiti", "surface detail"],
        preset: DepthBias {
            constant: 1,
            slope_scale: 1.0,
            clamp: 0.0,
        },
    },
    DepthBiasInfo {
        name: "Outline",
        description: "Negative bias for back-face based outlines",
        use_cases: &["cartoon rendering", "silhouette edges", "selection highlight"],
        preset: DepthBias {
            constant: -1,
            slope_scale: -1.0,
            clamp: 0.0,
        },
    },
    DepthBiasInfo {
        name: "Contact Shadow",
        description: "Aggressive bias for screen-space contact shadows",
        use_cases: &["SSAO enhancement", "contact hardening", "small-scale AO"],
        preset: DepthBias {
            constant: 8,
            slope_scale: 4.0,
            clamp: 0.0,
        },
    },
];

/// Get depth bias info by preset name.
///
/// # Example
///
/// ```ignore
/// if let Some(info) = get_depth_bias_info("Shadow Map") {
///     println!("Use cases: {:?}", info.use_cases);
///     let bias = info.preset;
/// }
/// ```
pub fn get_depth_bias_info(name: &str) -> Option<&'static DepthBiasInfo> {
    DEPTH_BIAS_PRESETS.iter().find(|info| info.name == name)
}

/// Get the depth bias preset by name.
pub fn get_preset(name: &str) -> Option<DepthBias> {
    get_depth_bias_info(name).map(|info| info.preset)
}

/// List all available preset names.
pub fn preset_names() -> impl Iterator<Item = &'static str> {
    DEPTH_BIAS_PRESETS.iter().map(|info| info.name)
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during depth bias validation.
#[derive(Debug, Clone, PartialEq)]
pub enum DepthBiasError {
    /// Clamp value must be non-negative.
    NegativeClamp(f32),
    /// Slope scale contains invalid value (NaN).
    InvalidSlopeScale(f32),
    /// Clamp contains invalid value (NaN).
    InvalidClamp(f32),
}

impl fmt::Display for DepthBiasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DepthBiasError::NegativeClamp(v) => {
                write!(f, "Invalid depth bias clamp: {} (must be >= 0.0)", v)
            }
            DepthBiasError::InvalidSlopeScale(v) => {
                write!(f, "Invalid slope_scale value: {} (NaN not allowed)", v)
            }
            DepthBiasError::InvalidClamp(v) => {
                write!(f, "Invalid clamp value: {} (NaN not allowed)", v)
            }
        }
    }
}

impl std::error::Error for DepthBiasError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // DepthBias Basic Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_depth_bias_default() {
        let bias = DepthBias::default();
        assert_eq!(bias.constant, 0);
        assert_eq!(bias.slope_scale, 0.0);
        assert_eq!(bias.clamp, 0.0);
    }

    #[test]
    fn test_depth_bias_new() {
        let bias = DepthBias::new();
        assert_eq!(bias, DepthBias::default());
    }

    #[test]
    fn test_depth_bias_none() {
        let bias = DepthBias::none();
        assert_eq!(bias, DepthBias::default());
        assert!(bias.is_none());
    }

    // -------------------------------------------------------------------------
    // Preset Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shadow_map_preset() {
        let bias = DepthBias::shadow_map();
        assert_eq!(bias.constant, 2);
        assert!((bias.slope_scale - 2.0).abs() < f32::EPSILON);
        assert!((bias.clamp - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cascaded_shadow_map_preset() {
        let bias = DepthBias::cascaded_shadow_map();
        assert_eq!(bias.constant, 4);
        assert!((bias.slope_scale - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_polygon_offset_preset() {
        let bias = DepthBias::polygon_offset();
        assert_eq!(bias.constant, 1);
        assert!((bias.slope_scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_decal_preset() {
        let bias = DepthBias::decal();
        assert_eq!(bias.constant, 1);
        assert!((bias.slope_scale - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_outline_preset() {
        let bias = DepthBias::outline();
        assert_eq!(bias.constant, -1);
        assert!((bias.slope_scale - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_contact_shadow_preset() {
        let bias = DepthBias::contact_shadow();
        assert_eq!(bias.constant, 8);
        assert!((bias.slope_scale - 4.0).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Builder Method Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fluent_builder_methods() {
        let bias = DepthBias::new()
            .constant(10)
            .slope_scale(2.5)
            .clamp(0.05);

        assert_eq!(bias.constant, 10);
        assert!((bias.slope_scale - 2.5).abs() < f32::EPSILON);
        assert!((bias.clamp - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn test_negative_values() {
        let bias = DepthBias::new()
            .constant(-5)
            .slope_scale(-2.0);

        assert_eq!(bias.constant, -5);
        assert!((bias.slope_scale - (-2.0)).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_valid() {
        let bias = DepthBias::shadow_map();
        assert!(bias.validate().is_ok());
        assert!(bias.is_valid());
    }

    #[test]
    fn test_validate_negative_clamp() {
        let bias = DepthBias::new().clamp(-0.1);
        assert!(matches!(
            bias.validate(),
            Err(DepthBiasError::NegativeClamp(_))
        ));
        assert!(!bias.is_valid());
    }

    #[test]
    fn test_validate_nan_slope_scale() {
        let bias = DepthBias::new().slope_scale(f32::NAN);
        assert!(matches!(
            bias.validate(),
            Err(DepthBiasError::InvalidSlopeScale(_))
        ));
    }

    #[test]
    fn test_validate_nan_clamp() {
        let bias = DepthBias::new().clamp(f32::NAN);
        // NaN comparison is tricky - first check passes because NaN < 0.0 is false
        // but NaN check should catch it
        let result = bias.validate();
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // State Query Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_none() {
        assert!(DepthBias::none().is_none());
        assert!(DepthBias::default().is_none());
        assert!(!DepthBias::shadow_map().is_none());
    }

    #[test]
    fn test_is_active() {
        assert!(!DepthBias::none().is_active());
        assert!(DepthBias::shadow_map().is_active());
        assert!(DepthBias::new().constant(1).is_active());
    }

    // -------------------------------------------------------------------------
    // Transform Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_scaled() {
        let bias = DepthBias::new()
            .constant(2)
            .slope_scale(2.0)
            .scaled(2.0);

        assert_eq!(bias.constant, 4);
        assert!((bias.slope_scale - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_inverted() {
        let bias = DepthBias::shadow_map().inverted();
        assert_eq!(bias.constant, -2);
        assert!((bias.slope_scale - (-2.0)).abs() < f32::EPSILON);
        // Clamp stays positive
        assert!((bias.clamp - 0.0).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Conversion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_into_wgpu() {
        let bias = DepthBias::new()
            .constant(5)
            .slope_scale(3.0)
            .clamp(0.02);

        let wgpu_bias: wgpu::DepthBiasState = bias.into();

        assert_eq!(wgpu_bias.constant, 5);
        assert!((wgpu_bias.slope_scale - 3.0).abs() < f32::EPSILON);
        assert!((wgpu_bias.clamp - 0.02).abs() < f32::EPSILON);
    }

    #[test]
    fn test_from_wgpu() {
        let wgpu_bias = wgpu::DepthBiasState {
            constant: 7,
            slope_scale: 2.5,
            clamp: 0.03,
        };

        let bias: DepthBias = wgpu_bias.into();

        assert_eq!(bias.constant, 7);
        assert!((bias.slope_scale - 2.5).abs() < f32::EPSILON);
        assert!((bias.clamp - 0.03).abs() < f32::EPSILON);
    }

    #[test]
    fn test_from_tuple() {
        let bias: DepthBias = (3, 1.5, 0.01).into();
        assert_eq!(bias.constant, 3);
        assert!((bias.slope_scale - 1.5).abs() < f32::EPSILON);
        assert!((bias.clamp - 0.01).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_new() {
        let builder = DepthBiasBuilder::new();
        let bias = builder.build_unchecked();
        assert!(bias.is_none());
    }

    #[test]
    fn test_builder_from_preset() {
        let bias = DepthBiasBuilder::from_preset(DepthBias::shadow_map())
            .constant(4)
            .build()
            .unwrap();

        assert_eq!(bias.constant, 4);
        assert!((bias.slope_scale - 2.0).abs() < f32::EPSILON); // From preset
    }

    #[test]
    fn test_builder_shadow_map() {
        let bias = DepthBiasBuilder::shadow_map()
            .build()
            .unwrap();

        assert_eq!(bias.constant, 2);
    }

    #[test]
    fn test_builder_polygon_offset() {
        let bias = DepthBiasBuilder::polygon_offset()
            .build()
            .unwrap();

        assert_eq!(bias.constant, 1);
    }

    #[test]
    fn test_builder_chain() {
        let bias = DepthBiasBuilder::new()
            .constant(10)
            .slope_scale(5.0)
            .clamp(0.1)
            .build()
            .unwrap();

        assert_eq!(bias.constant, 10);
        assert!((bias.slope_scale - 5.0).abs() < f32::EPSILON);
        assert!((bias.clamp - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_builder_scale() {
        let bias = DepthBiasBuilder::shadow_map()
            .scale(2.0)
            .build()
            .unwrap();

        assert_eq!(bias.constant, 4);
        assert!((bias.slope_scale - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_builder_invert() {
        let bias = DepthBiasBuilder::shadow_map()
            .invert()
            .build()
            .unwrap();

        assert_eq!(bias.constant, -2);
        assert!((bias.slope_scale - (-2.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_builder_validation_error() {
        let result = DepthBiasBuilder::new()
            .clamp(-0.5)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_unchecked() {
        let bias = DepthBiasBuilder::new()
            .clamp(-0.5) // Invalid but allowed with unchecked
            .build_unchecked();

        assert!((bias.clamp - (-0.5)).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Preset Info Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_depth_bias_info() {
        let info = get_depth_bias_info("Shadow Map").unwrap();
        assert_eq!(info.name, "Shadow Map");
        assert_eq!(info.preset.constant, 2);
    }

    #[test]
    fn test_get_depth_bias_info_not_found() {
        assert!(get_depth_bias_info("NonExistent").is_none());
    }

    #[test]
    fn test_get_preset() {
        let preset = get_preset("Shadow Map").unwrap();
        assert_eq!(preset.constant, 2);
    }

    #[test]
    fn test_preset_names() {
        let names: Vec<_> = preset_names().collect();
        assert!(names.contains(&"Shadow Map"));
        assert!(names.contains(&"None"));
        assert!(names.contains(&"Polygon Offset"));
        assert_eq!(names.len(), 7);
    }

    #[test]
    fn test_all_presets_valid() {
        for info in &DEPTH_BIAS_PRESETS {
            assert!(info.preset.is_valid(), "Preset '{}' is invalid", info.name);
        }
    }

    // -------------------------------------------------------------------------
    // Equality Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_equality() {
        let bias1 = DepthBias::shadow_map();
        let bias2 = DepthBias::shadow_map();
        let bias3 = DepthBias::polygon_offset();

        assert_eq!(bias1, bias2);
        assert_ne!(bias1, bias3);
    }

    #[test]
    fn test_clone() {
        let bias = DepthBias::shadow_map();
        let cloned = bias.clone();
        assert_eq!(bias, cloned);
    }

    #[test]
    fn test_copy() {
        let bias = DepthBias::shadow_map();
        let copied = bias;
        assert_eq!(bias, copied);
    }

    // -------------------------------------------------------------------------
    // Error Display Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_display_negative_clamp() {
        let err = DepthBiasError::NegativeClamp(-0.5);
        let msg = format!("{}", err);
        assert!(msg.contains("-0.5"));
        assert!(msg.contains("clamp"));
    }

    #[test]
    fn test_error_display_invalid_slope() {
        let err = DepthBiasError::InvalidSlopeScale(f32::NAN);
        let msg = format!("{}", err);
        assert!(msg.contains("slope_scale"));
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<DepthBias>();
        assert_sync::<DepthBias>();
        assert_send::<DepthBiasBuilder>();
        assert_sync::<DepthBiasBuilder>();
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_extreme_values() {
        let bias = DepthBias::new()
            .constant(i32::MAX)
            .slope_scale(f32::MAX)
            .clamp(f32::MAX);

        assert_eq!(bias.constant, i32::MAX);
        assert!(bias.is_valid());
    }

    #[test]
    fn test_min_values() {
        let bias = DepthBias::new()
            .constant(i32::MIN)
            .slope_scale(f32::MIN_POSITIVE);

        assert_eq!(bias.constant, i32::MIN);
        assert!(bias.is_valid());
    }

    #[test]
    fn test_infinity_slope_scale() {
        let bias = DepthBias::new().slope_scale(f32::INFINITY);
        // Infinity is technically valid (not NaN)
        assert!(bias.is_valid());
    }

    #[test]
    fn test_zero_clamp() {
        let bias = DepthBias::new().clamp(0.0);
        assert!(bias.is_valid());
        assert!((bias.clamp - 0.0).abs() < f32::EPSILON);
    }

    // =========================================================================
    // ADDITIONAL WHITEBOX TESTS - T-WGPU-P3.4.2
    // =========================================================================

    // -------------------------------------------------------------------------
    // DepthBias Construction Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_construction_all_zeros() {
        let bias = DepthBias {
            constant: 0,
            slope_scale: 0.0,
            clamp: 0.0,
        };
        assert!(bias.is_none());
        assert!(!bias.is_active());
        assert!(bias.is_valid());
    }

    #[test]
    fn test_construction_negative_constant_only() {
        let bias = DepthBias {
            constant: -100,
            slope_scale: 0.0,
            clamp: 0.0,
        };
        assert!(!bias.is_none());
        assert!(bias.is_active());
        assert!(bias.is_valid());
    }

    #[test]
    fn test_construction_slope_scale_only() {
        let bias = DepthBias {
            constant: 0,
            slope_scale: 0.001,
            clamp: 0.0,
        };
        assert!(!bias.is_none());
        assert!(bias.is_active());
    }

    #[test]
    fn test_construction_clamp_only() {
        let bias = DepthBias {
            constant: 0,
            slope_scale: 0.0,
            clamp: 0.1,
        };
        // Clamp alone doesn't make bias "active" by is_none definition
        assert!(!bias.is_none());
        assert!(bias.is_active());
    }

    // -------------------------------------------------------------------------
    // Constant Bias Variations
    // -------------------------------------------------------------------------

    #[test]
    fn test_constant_i32_min() {
        let bias = DepthBias::new().constant(i32::MIN);
        assert_eq!(bias.constant, i32::MIN);
        assert!(bias.is_valid());

        // Verify conversion to wgpu preserves value
        let wgpu: wgpu::DepthBiasState = bias.into();
        assert_eq!(wgpu.constant, i32::MIN);
    }

    #[test]
    fn test_constant_i32_max() {
        let bias = DepthBias::new().constant(i32::MAX);
        assert_eq!(bias.constant, i32::MAX);
        assert!(bias.is_valid());

        let wgpu: wgpu::DepthBiasState = bias.into();
        assert_eq!(wgpu.constant, i32::MAX);
    }

    #[test]
    fn test_constant_typical_shadow_values() {
        // Test common shadow map constant values
        for val in [1, 2, 4, 8, 16, 32, 64] {
            let bias = DepthBias::new().constant(val);
            assert_eq!(bias.constant, val);
            assert!(bias.is_valid());
        }
    }

    #[test]
    fn test_constant_negative_for_backface() {
        // Negative constants used for backface culling tricks
        for val in [-1, -2, -4, -8] {
            let bias = DepthBias::new().constant(val);
            assert_eq!(bias.constant, val);
            assert!(bias.is_valid());
        }
    }

    // -------------------------------------------------------------------------
    // Slope Scale Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_slope_scale_zero() {
        let bias = DepthBias::new().slope_scale(0.0);
        assert!((bias.slope_scale - 0.0).abs() < f32::EPSILON);
        assert!(bias.is_valid());
    }

    #[test]
    fn test_slope_scale_negative_zero() {
        let bias = DepthBias::new().slope_scale(-0.0);
        // -0.0 == 0.0 in IEEE 754
        assert!((bias.slope_scale - 0.0).abs() < f32::EPSILON);
        assert!(bias.is_valid());
    }

    #[test]
    fn test_slope_scale_very_small() {
        let bias = DepthBias::new().slope_scale(f32::MIN_POSITIVE);
        assert!(bias.slope_scale > 0.0);
        assert!(bias.is_valid());
    }

    #[test]
    fn test_slope_scale_very_small_negative() {
        let bias = DepthBias::new().slope_scale(-f32::MIN_POSITIVE);
        assert!(bias.slope_scale < 0.0);
        assert!(bias.is_valid());
    }

    #[test]
    fn test_slope_scale_large_positive() {
        let bias = DepthBias::new().slope_scale(1_000_000.0);
        assert!((bias.slope_scale - 1_000_000.0).abs() < 1.0);
        assert!(bias.is_valid());
    }

    #[test]
    fn test_slope_scale_large_negative() {
        let bias = DepthBias::new().slope_scale(-1_000_000.0);
        assert!((bias.slope_scale - (-1_000_000.0)).abs() < 1.0);
        assert!(bias.is_valid());
    }

    #[test]
    fn test_slope_scale_positive_infinity() {
        let bias = DepthBias::new().slope_scale(f32::INFINITY);
        assert!(bias.slope_scale.is_infinite());
        assert!(bias.slope_scale.is_sign_positive());
        assert!(bias.is_valid()); // Infinity is valid per implementation
    }

    #[test]
    fn test_slope_scale_negative_infinity() {
        let bias = DepthBias::new().slope_scale(f32::NEG_INFINITY);
        assert!(bias.slope_scale.is_infinite());
        assert!(bias.slope_scale.is_sign_negative());
        assert!(bias.is_valid());
    }

    #[test]
    fn test_slope_scale_subnormal() {
        // Test subnormal (denormalized) floating point
        let subnormal = f32::MIN_POSITIVE / 2.0;
        let bias = DepthBias::new().slope_scale(subnormal);
        assert!(bias.is_valid());
    }

    // -------------------------------------------------------------------------
    // Clamp Validation
    // -------------------------------------------------------------------------

    #[test]
    fn test_clamp_zero_valid() {
        let bias = DepthBias::new().clamp(0.0);
        assert!(bias.is_valid());
        assert!(bias.validate().is_ok());
    }

    #[test]
    fn test_clamp_positive_small() {
        let bias = DepthBias::new().clamp(0.001);
        assert!(bias.is_valid());
    }

    #[test]
    fn test_clamp_positive_large() {
        let bias = DepthBias::new().clamp(1000.0);
        assert!(bias.is_valid());
    }

    #[test]
    fn test_clamp_negative_small() {
        let bias = DepthBias::new().clamp(-0.001);
        assert!(!bias.is_valid());
        assert!(matches!(
            bias.validate(),
            Err(DepthBiasError::NegativeClamp(v)) if (v - (-0.001)).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn test_clamp_negative_large() {
        let bias = DepthBias::new().clamp(-1000.0);
        assert!(!bias.is_valid());
    }

    #[test]
    fn test_clamp_positive_infinity() {
        let bias = DepthBias::new().clamp(f32::INFINITY);
        assert!(bias.is_valid()); // Positive infinity is >= 0
    }

    #[test]
    fn test_clamp_negative_infinity() {
        let bias = DepthBias::new().clamp(f32::NEG_INFINITY);
        assert!(!bias.is_valid());
        assert!(matches!(
            bias.validate(),
            Err(DepthBiasError::NegativeClamp(_))
        ));
    }

    // -------------------------------------------------------------------------
    // Builder Pattern Chains
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_overwrite_constant_multiple() {
        let bias = DepthBiasBuilder::new()
            .constant(1)
            .constant(2)
            .constant(3)
            .build()
            .unwrap();
        assert_eq!(bias.constant, 3); // Last value wins
    }

    #[test]
    fn test_builder_overwrite_slope_multiple() {
        let bias = DepthBiasBuilder::new()
            .slope_scale(1.0)
            .slope_scale(2.0)
            .slope_scale(3.0)
            .build()
            .unwrap();
        assert!((bias.slope_scale - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_builder_overwrite_clamp_multiple() {
        let bias = DepthBiasBuilder::new()
            .clamp(0.1)
            .clamp(0.2)
            .clamp(0.3)
            .build()
            .unwrap();
        assert!((bias.clamp - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_builder_all_combinations() {
        // Test all orderings of builder methods
        let b1 = DepthBiasBuilder::new()
            .constant(1).slope_scale(2.0).clamp(0.3)
            .build().unwrap();
        let b2 = DepthBiasBuilder::new()
            .slope_scale(2.0).constant(1).clamp(0.3)
            .build().unwrap();
        let b3 = DepthBiasBuilder::new()
            .clamp(0.3).constant(1).slope_scale(2.0)
            .build().unwrap();

        assert_eq!(b1, b2);
        assert_eq!(b2, b3);
    }

    #[test]
    fn test_builder_scale_then_modify() {
        let bias = DepthBiasBuilder::shadow_map()
            .scale(2.0)
            .constant(100) // Override scaled constant
            .build()
            .unwrap();
        assert_eq!(bias.constant, 100);
        assert!((bias.slope_scale - 4.0).abs() < f32::EPSILON); // Still scaled
    }

    #[test]
    fn test_builder_invert_then_modify() {
        let bias = DepthBiasBuilder::shadow_map()
            .invert()
            .slope_scale(10.0) // Override inverted slope
            .build()
            .unwrap();
        assert_eq!(bias.constant, -2); // Still inverted
        assert!((bias.slope_scale - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_builder_scale_and_invert() {
        let bias = DepthBiasBuilder::shadow_map()
            .scale(2.0)
            .invert()
            .build()
            .unwrap();
        assert_eq!(bias.constant, -4);
        assert!((bias.slope_scale - (-4.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_builder_invert_and_scale() {
        let bias = DepthBiasBuilder::shadow_map()
            .invert()
            .scale(2.0)
            .build()
            .unwrap();
        assert_eq!(bias.constant, -4);
        assert!((bias.slope_scale - (-4.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_builder_double_invert() {
        let original = DepthBias::shadow_map();
        let double_inverted = DepthBiasBuilder::from_preset(original)
            .invert()
            .invert()
            .build()
            .unwrap();
        assert_eq!(original, double_inverted);
    }

    #[test]
    fn test_builder_default_trait() {
        let builder = DepthBiasBuilder::default();
        let bias = builder.build().unwrap();
        assert!(bias.is_none());
    }

    #[test]
    fn test_builder_clone() {
        let builder1 = DepthBiasBuilder::new().constant(5);
        let builder2 = builder1.clone();

        let bias1 = builder1.slope_scale(1.0).build().unwrap();
        let bias2 = builder2.slope_scale(2.0).build().unwrap();

        assert_eq!(bias1.constant, 5);
        assert_eq!(bias2.constant, 5);
        assert!((bias1.slope_scale - 1.0).abs() < f32::EPSILON);
        assert!((bias2.slope_scale - 2.0).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // DepthBiasInfo Metadata Coverage
    // -------------------------------------------------------------------------

    #[test]
    fn test_depth_bias_info_none_preset() {
        let info = get_depth_bias_info("None").unwrap();
        assert_eq!(info.name, "None");
        assert!(info.description.contains("No depth bias"));
        assert!(info.use_cases.len() > 0);
        assert!(info.preset.is_none());
    }

    #[test]
    fn test_depth_bias_info_shadow_map_use_cases() {
        let info = get_depth_bias_info("Shadow Map").unwrap();
        assert!(info.use_cases.iter().any(|u| u.contains("shadow")));
    }

    #[test]
    fn test_depth_bias_info_csm_use_cases() {
        let info = get_depth_bias_info("Cascaded Shadow Map").unwrap();
        assert!(info.use_cases.iter().any(|u| u.contains("CSM") || u.contains("outdoor")));
    }

    #[test]
    fn test_depth_bias_info_polygon_offset_use_cases() {
        let info = get_depth_bias_info("Polygon Offset").unwrap();
        assert!(info.use_cases.iter().any(|u| u.contains("wireframe") || u.contains("coplanar")));
    }

    #[test]
    fn test_depth_bias_info_decal_use_cases() {
        let info = get_depth_bias_info("Decal").unwrap();
        assert!(info.use_cases.iter().any(|u| u.contains("bullet") || u.contains("surface")));
    }

    #[test]
    fn test_depth_bias_info_outline_use_cases() {
        let info = get_depth_bias_info("Outline").unwrap();
        assert!(info.use_cases.iter().any(|u| u.contains("cartoon") || u.contains("silhouette")));
    }

    #[test]
    fn test_depth_bias_info_contact_shadow_use_cases() {
        let info = get_depth_bias_info("Contact Shadow").unwrap();
        assert!(info.use_cases.iter().any(|u| u.contains("SSAO") || u.contains("contact")));
    }

    #[test]
    fn test_preset_info_matches_method() {
        // Verify preset info matches the method implementations
        let shadow_info = get_depth_bias_info("Shadow Map").unwrap();
        assert_eq!(shadow_info.preset, DepthBias::shadow_map());

        let csm_info = get_depth_bias_info("Cascaded Shadow Map").unwrap();
        assert_eq!(csm_info.preset, DepthBias::cascaded_shadow_map());

        let poly_info = get_depth_bias_info("Polygon Offset").unwrap();
        assert_eq!(poly_info.preset, DepthBias::polygon_offset());

        let decal_info = get_depth_bias_info("Decal").unwrap();
        assert_eq!(decal_info.preset, DepthBias::decal());

        let outline_info = get_depth_bias_info("Outline").unwrap();
        assert_eq!(outline_info.preset, DepthBias::outline());

        let contact_info = get_depth_bias_info("Contact Shadow").unwrap();
        assert_eq!(contact_info.preset, DepthBias::contact_shadow());
    }

    #[test]
    fn test_all_preset_names_retrievable() {
        for name in preset_names() {
            let info = get_depth_bias_info(name);
            assert!(info.is_some(), "Preset '{}' not found", name);

            let preset = get_preset(name);
            assert!(preset.is_some(), "Preset '{}' value not found", name);
        }
    }

    #[test]
    fn test_depth_bias_info_equality() {
        let info1 = get_depth_bias_info("Shadow Map").unwrap();
        let info2 = get_depth_bias_info("Shadow Map").unwrap();
        assert_eq!(info1, info2);

        let info3 = get_depth_bias_info("Outline").unwrap();
        assert_ne!(info1, info3);
    }

    // -------------------------------------------------------------------------
    // Conversion Roundtrip Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_wgpu_roundtrip() {
        let original = DepthBias::new()
            .constant(42)
            .slope_scale(3.14159)
            .clamp(0.123);

        let wgpu: wgpu::DepthBiasState = original.into();
        let roundtrip: DepthBias = wgpu.into();

        assert_eq!(original, roundtrip);
    }

    #[test]
    fn test_wgpu_roundtrip_presets() {
        for info in &DEPTH_BIAS_PRESETS {
            let wgpu: wgpu::DepthBiasState = info.preset.into();
            let roundtrip: DepthBias = wgpu.into();
            assert_eq!(info.preset, roundtrip, "Roundtrip failed for {}", info.name);
        }
    }

    #[test]
    fn test_tuple_conversion_roundtrip() {
        let tuple = (7, 2.5_f32, 0.05_f32);
        let bias: DepthBias = tuple.into();

        assert_eq!(bias.constant, tuple.0);
        assert!((bias.slope_scale - tuple.1).abs() < f32::EPSILON);
        assert!((bias.clamp - tuple.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tuple_negative_values() {
        let bias: DepthBias = (-10, -5.0_f32, 0.0_f32).into();
        assert_eq!(bias.constant, -10);
        assert!((bias.slope_scale - (-5.0)).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Display/Debug Trait Implementations
    // -------------------------------------------------------------------------

    #[test]
    fn test_depth_bias_debug() {
        let bias = DepthBias::shadow_map();
        let debug_str = format!("{:?}", bias);

        assert!(debug_str.contains("DepthBias"));
        assert!(debug_str.contains("constant"));
        assert!(debug_str.contains("slope_scale"));
        assert!(debug_str.contains("clamp"));
    }

    #[test]
    fn test_depth_bias_builder_debug() {
        let builder = DepthBiasBuilder::new().constant(5);
        let debug_str = format!("{:?}", builder);

        assert!(debug_str.contains("DepthBiasBuilder"));
    }

    #[test]
    fn test_depth_bias_info_debug() {
        let info = get_depth_bias_info("Shadow Map").unwrap();
        let debug_str = format!("{:?}", info);

        assert!(debug_str.contains("DepthBiasInfo"));
        assert!(debug_str.contains("Shadow Map"));
    }

    #[test]
    fn test_error_debug() {
        let err = DepthBiasError::NegativeClamp(-0.5);
        let debug_str = format!("{:?}", err);

        assert!(debug_str.contains("NegativeClamp"));
    }

    #[test]
    fn test_error_display_invalid_clamp() {
        let err = DepthBiasError::InvalidClamp(f32::NAN);
        let msg = format!("{}", err);
        assert!(msg.contains("clamp"));
        assert!(msg.contains("NaN"));
    }

    #[test]
    fn test_error_std_error_trait() {
        let err: &dyn std::error::Error = &DepthBiasError::NegativeClamp(-1.0);
        // Verify Error trait is implemented
        let _ = err.to_string();
    }

    #[test]
    fn test_error_clone() {
        let err = DepthBiasError::NegativeClamp(-0.5);
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    // -------------------------------------------------------------------------
    // Thread Safety Verification
    // -------------------------------------------------------------------------

    #[test]
    fn test_send_across_thread() {
        use std::thread;

        let bias = DepthBias::shadow_map();
        let handle = thread::spawn(move || {
            assert_eq!(bias.constant, 2);
            bias
        });
        let result = handle.join().unwrap();
        assert_eq!(result.constant, 2);
    }

    #[test]
    fn test_sync_shared_reference() {
        use std::sync::Arc;
        use std::thread;

        let bias = Arc::new(DepthBias::shadow_map());
        let bias_clone = Arc::clone(&bias);

        let handle = thread::spawn(move || {
            assert_eq!(bias_clone.constant, 2);
        });

        assert_eq!(bias.constant, 2);
        handle.join().unwrap();
    }

    #[test]
    fn test_builder_send() {
        use std::thread;

        let builder = DepthBiasBuilder::new().constant(5);
        let handle = thread::spawn(move || {
            builder.slope_scale(2.0).build().unwrap()
        });
        let result = handle.join().unwrap();
        assert_eq!(result.constant, 5);
    }

    // -------------------------------------------------------------------------
    // Scaled Transform Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_scaled_by_zero() {
        let bias = DepthBias::shadow_map().scaled(0.0);
        assert_eq!(bias.constant, 0);
        assert!((bias.slope_scale - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_scaled_by_negative() {
        let bias = DepthBias::shadow_map().scaled(-1.0);
        assert_eq!(bias.constant, -2);
        assert!((bias.slope_scale - (-2.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_scaled_by_fraction() {
        let bias = DepthBias::new().constant(10).slope_scale(10.0).scaled(0.5);
        assert_eq!(bias.constant, 5);
        assert!((bias.slope_scale - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_scaled_preserves_clamp() {
        let bias = DepthBias::new()
            .constant(2)
            .slope_scale(2.0)
            .clamp(0.5)
            .scaled(10.0);

        // Clamp is NOT scaled per implementation
        assert!((bias.clamp - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_scaled_large_factor() {
        let bias = DepthBias::new().constant(1).slope_scale(1.0).scaled(1000.0);
        assert_eq!(bias.constant, 1000);
        assert!((bias.slope_scale - 1000.0).abs() < 1.0);
    }

    // -------------------------------------------------------------------------
    // Inverted Transform Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_inverted_zero_bias() {
        let bias = DepthBias::none().inverted();
        assert!(bias.is_none()); // Inverting zeros gives zeros
    }

    #[test]
    fn test_inverted_preserves_positive_clamp() {
        let bias = DepthBias::new()
            .constant(5)
            .slope_scale(3.0)
            .clamp(0.1)
            .inverted();

        assert!((bias.clamp - 0.1).abs() < f32::EPSILON); // Clamp stays positive
    }

    #[test]
    fn test_inverted_symmetry() {
        let original = DepthBias::new().constant(10).slope_scale(5.0);
        let inverted = original.inverted();
        let double_inverted = inverted.inverted();

        assert_eq!(original, double_inverted);
    }

    // -------------------------------------------------------------------------
    // Validation Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_multiple_issues() {
        // NaN clamp is checked after negative clamp
        let bias = DepthBias {
            constant: 0,
            slope_scale: f32::NAN,
            clamp: -1.0,
        };

        // First error should be negative clamp
        let result = bias.validate();
        assert!(matches!(result, Err(DepthBiasError::NegativeClamp(_))));
    }

    #[test]
    fn test_validate_nan_slope_with_valid_clamp() {
        let bias = DepthBias {
            constant: 0,
            slope_scale: f32::NAN,
            clamp: 0.0,
        };

        assert!(matches!(
            bias.validate(),
            Err(DepthBiasError::InvalidSlopeScale(_))
        ));
    }

    #[test]
    fn test_validate_inf_slope_valid() {
        let bias = DepthBias::new().slope_scale(f32::INFINITY);
        assert!(bias.validate().is_ok());
    }

    #[test]
    fn test_validate_neg_inf_slope_valid() {
        let bias = DepthBias::new().slope_scale(f32::NEG_INFINITY);
        assert!(bias.validate().is_ok());
    }

    // -------------------------------------------------------------------------
    // is_none / is_active Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_none_with_negative_zero() {
        let bias = DepthBias {
            constant: 0,
            slope_scale: -0.0,
            clamp: 0.0,
        };
        // -0.0 == 0.0 in IEEE 754
        assert!(bias.is_none());
    }

    #[test]
    fn test_is_active_with_tiny_slope() {
        let bias = DepthBias::new().slope_scale(f32::MIN_POSITIVE);
        assert!(bias.is_active());
    }

    #[test]
    fn test_is_active_constant_only() {
        let bias = DepthBias {
            constant: 1,
            slope_scale: 0.0,
            clamp: 0.0,
        };
        assert!(bias.is_active());
        assert!(!bias.is_none());
    }

    // -------------------------------------------------------------------------
    // Preset Correctness
    // -------------------------------------------------------------------------

    #[test]
    fn test_shadow_map_values_correct() {
        let bias = DepthBias::shadow_map();
        assert_eq!(bias.constant, 2);
        assert_eq!(bias.slope_scale, 2.0);
        assert_eq!(bias.clamp, 0.0);
    }

    #[test]
    fn test_polygon_offset_values_correct() {
        let bias = DepthBias::polygon_offset();
        assert_eq!(bias.constant, 1);
        assert_eq!(bias.slope_scale, 1.0);
        assert_eq!(bias.clamp, 0.0);
    }

    #[test]
    fn test_decal_matches_polygon_offset() {
        // Per docs, decal uses same values as polygon_offset
        let decal = DepthBias::decal();
        let poly = DepthBias::polygon_offset();
        assert_eq!(decal, poly);
    }

    #[test]
    fn test_outline_negative_values() {
        let bias = DepthBias::outline();
        assert!(bias.constant < 0);
        assert!(bias.slope_scale < 0.0);
    }

    #[test]
    fn test_contact_shadow_aggressive() {
        let contact = DepthBias::contact_shadow();
        let shadow = DepthBias::shadow_map();

        // Contact shadow should have higher bias values
        assert!(contact.constant > shadow.constant);
        assert!(contact.slope_scale > shadow.slope_scale);
    }

    #[test]
    fn test_csm_higher_than_shadow_map() {
        let csm = DepthBias::cascaded_shadow_map();
        let shadow = DepthBias::shadow_map();

        assert!(csm.constant > shadow.constant);
        assert!(csm.slope_scale > shadow.slope_scale);
    }

    // -------------------------------------------------------------------------
    // Builder From Preset Modifications
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_from_all_presets() {
        let presets = [
            DepthBias::none(),
            DepthBias::shadow_map(),
            DepthBias::cascaded_shadow_map(),
            DepthBias::polygon_offset(),
            DepthBias::decal(),
            DepthBias::outline(),
            DepthBias::contact_shadow(),
        ];

        for preset in presets {
            let bias = DepthBiasBuilder::from_preset(preset)
                .constant(999)
                .build()
                .unwrap();

            assert_eq!(bias.constant, 999);
            // Other values preserved from preset
            assert!((bias.slope_scale - preset.slope_scale).abs() < f32::EPSILON);
            assert!((bias.clamp - preset.clamp).abs() < f32::EPSILON);
        }
    }

    // -------------------------------------------------------------------------
    // Copy/Clone Semantics
    // -------------------------------------------------------------------------

    #[test]
    fn test_bias_is_copy_not_move() {
        let bias1 = DepthBias::shadow_map();
        let bias2 = bias1; // Copy, not move
        let bias3 = bias1; // Can copy again

        assert_eq!(bias1, bias2);
        assert_eq!(bias2, bias3);
    }

    #[test]
    fn test_info_is_copy() {
        let info1 = DEPTH_BIAS_PRESETS[0];
        let info2 = info1; // Copy
        assert_eq!(info1, info2);
    }

    // -------------------------------------------------------------------------
    // DepthBiasInfo Clone
    // -------------------------------------------------------------------------

    #[test]
    fn test_depth_bias_info_clone() {
        let info = get_depth_bias_info("Shadow Map").unwrap();
        let cloned = *info; // Copy since DepthBiasInfo is Copy
        assert_eq!(*info, cloned);
    }

    // -------------------------------------------------------------------------
    // Error Variants Coverage
    // -------------------------------------------------------------------------

    #[test]
    fn test_all_error_variants_display() {
        let errors = [
            DepthBiasError::NegativeClamp(-0.5),
            DepthBiasError::InvalidSlopeScale(f32::NAN),
            DepthBiasError::InvalidClamp(f32::NAN),
        ];

        for err in &errors {
            let msg = format!("{}", err);
            assert!(!msg.is_empty());
        }
    }

    #[test]
    fn test_error_partialeq() {
        let err1 = DepthBiasError::NegativeClamp(-0.5);
        let err2 = DepthBiasError::NegativeClamp(-0.5);
        let err3 = DepthBiasError::NegativeClamp(-0.6);

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }
}
