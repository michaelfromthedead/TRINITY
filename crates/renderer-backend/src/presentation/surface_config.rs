//! Surface configuration builder, presets, and validation for the TRINITY renderer.
//!
//! This module provides higher-level configuration tools for surface setup:
//!
//! - [`SurfaceConfigBuilder`] - Fluent builder for constructing surface configurations
//! - [`ConfigPreset`] - Pre-defined configuration profiles for common scenarios
//! - [`ConfigValidationError`] - Detailed validation error types
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::presentation::{
//!     SurfaceConfigBuilder, ConfigPreset, ConfigValidationError,
//! };
//!
//! // Build configuration with preset
//! let config = SurfaceConfigBuilder::new()
//!     .with_size(1920, 1080)
//!     .with_preset(ConfigPreset::LowLatency)
//!     .build();
//!
//! // Validate against capabilities
//! let validation = config.validate_strict(&caps);
//! match validation {
//!     Ok(()) => println!("Configuration valid"),
//!     Err(ConfigValidationError::UnsupportedFormat(fmt)) => {
//!         println!("Format {:?} not supported", fmt);
//!     }
//!     Err(e) => println!("Validation failed: {:?}", e),
//! }
//! ```

use std::fmt;

use super::surface::SurfaceCapabilities;

// ============================================================================
// Configuration Validation Errors
// ============================================================================

/// Detailed validation errors for surface configuration.
///
/// Unlike [`SurfaceError`], this enum provides specific error variants
/// for each validation failure, allowing for more targeted error handling
/// and recovery strategies.
///
/// # Example
///
/// ```ignore
/// match config.validate_strict(&caps) {
///     Err(ConfigValidationError::ZeroDimensions) => {
///         // Handle minimized window
///         return;
///     }
///     Err(ConfigValidationError::UnsupportedFormat(fmt)) => {
///         // Fall back to a supported format
///         config = config.with_format(caps.preferred_format().unwrap());
///     }
///     Ok(()) => {}
///     Err(e) => return Err(e.into()),
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigValidationError {
    /// Surface dimensions are zero (e.g., minimized window).
    ZeroDimensions,

    /// The requested texture format is not supported by the surface.
    UnsupportedFormat(wgpu::TextureFormat),

    /// The requested present mode is not supported by the surface.
    UnsupportedPresentMode(wgpu::PresentMode),

    /// The requested alpha compositing mode is not supported.
    UnsupportedAlphaMode(wgpu::CompositeAlphaMode),

    /// A view format in `view_formats` is invalid or not compatible
    /// with the base format.
    InvalidViewFormat(wgpu::TextureFormat),
}

impl ConfigValidationError {
    /// Check if this error is recoverable without recreating the surface.
    ///
    /// `ZeroDimensions` is recoverable (wait for window to be restored).
    /// Format/mode errors require configuration changes.
    pub fn is_recoverable(&self) -> bool {
        matches!(self, ConfigValidationError::ZeroDimensions)
    }

    /// Get a suggested fix for this validation error.
    pub fn suggested_fix(&self) -> &'static str {
        match self {
            ConfigValidationError::ZeroDimensions => {
                "Wait for window to be restored from minimized state"
            }
            ConfigValidationError::UnsupportedFormat(_) => {
                "Use SurfaceCapabilities::preferred_format() to select a supported format"
            }
            ConfigValidationError::UnsupportedPresentMode(_) => {
                "Use SurfaceCapabilities::preferred_present_mode() or fall back to Fifo"
            }
            ConfigValidationError::UnsupportedAlphaMode(_) => {
                "Use SurfaceCapabilities::preferred_alpha_mode() or Auto"
            }
            ConfigValidationError::InvalidViewFormat(_) => {
                "Remove invalid format from view_formats or use get_srgb_companion_format()"
            }
        }
    }
}

impl fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigValidationError::ZeroDimensions => {
                write!(f, "surface dimensions must be non-zero")
            }
            ConfigValidationError::UnsupportedFormat(fmt) => {
                write!(f, "texture format {:?} is not supported by this surface", fmt)
            }
            ConfigValidationError::UnsupportedPresentMode(mode) => {
                write!(f, "present mode {:?} is not supported by this surface", mode)
            }
            ConfigValidationError::UnsupportedAlphaMode(mode) => {
                write!(f, "alpha mode {:?} is not supported by this surface", mode)
            }
            ConfigValidationError::InvalidViewFormat(fmt) => {
                write!(f, "view format {:?} is not compatible with the base format", fmt)
            }
        }
    }
}

impl std::error::Error for ConfigValidationError {}

// ============================================================================
// Configuration Presets
// ============================================================================

/// Pre-defined configuration profiles for common rendering scenarios.
///
/// Presets provide sensible defaults for specific use cases, balancing
/// performance, quality, and power consumption appropriately.
///
/// # Preset Overview
///
/// | Preset       | Present Mode | Frame Latency | Use Case                    |
/// |--------------|--------------|---------------|-----------------------------|
/// | `Default`    | Fifo         | 2             | General purpose rendering   |
/// | `LowLatency` | Mailbox      | 1             | Competitive gaming, VR      |
/// | `PowerSaving`| Fifo         | 3             | Mobile, battery-powered     |
/// | `HighQuality`| Fifo         | 2             | Cinematic, high fidelity    |
/// | `HDR`        | Fifo         | 2             | HDR displays, wide gamut    |
///
/// # Example
///
/// ```ignore
/// let config = SurfaceConfigBuilder::new()
///     .with_size(1920, 1080)
///     .with_preset(ConfigPreset::LowLatency)
///     .build();
///
/// // Or apply preset to existing configuration
/// let mut config = SurfaceConfiguration::new(1920, 1080);
/// ConfigPreset::HighQuality.apply(&mut config);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ConfigPreset {
    /// General purpose rendering with vsync.
    ///
    /// - Present mode: `Fifo` (vsync)
    /// - Frame latency: 2
    /// - Format: `Bgra8UnormSrgb`
    #[default]
    Default,

    /// Low-latency rendering for competitive gaming and VR.
    ///
    /// - Present mode: `Mailbox` (triple buffering, lowest latency)
    /// - Frame latency: 1
    /// - Format: `Bgra8UnormSrgb`
    LowLatency,

    /// Power-efficient rendering for mobile and battery-powered devices.
    ///
    /// - Present mode: `Fifo` (vsync, minimizes GPU wake-ups)
    /// - Frame latency: 3 (more buffering, smoother on variable loads)
    /// - Format: `Bgra8UnormSrgb`
    PowerSaving,

    /// High-quality rendering for cinematic experiences.
    ///
    /// - Present mode: `Fifo` (vsync)
    /// - Frame latency: 2
    /// - Format: `Bgra8UnormSrgb` with sRGB view format enabled
    HighQuality,

    /// HDR rendering for displays supporting wide color gamut.
    ///
    /// - Present mode: `Fifo` (vsync)
    /// - Frame latency: 2
    /// - Format: `Rgba16Float` for HDR content
    HDR,
}

impl ConfigPreset {
    /// Apply this preset to a surface configuration.
    ///
    /// This modifies the configuration in-place with preset values.
    /// The dimensions are preserved.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut config = SurfaceConfiguration::new(1920, 1080);
    /// ConfigPreset::LowLatency.apply(&mut config);
    /// assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
    /// ```
    pub fn apply(&self, config: &mut super::SurfaceConfiguration) {
        match self {
            ConfigPreset::Default => {
                config.format = wgpu::TextureFormat::Bgra8UnormSrgb;
                config.present_mode = wgpu::PresentMode::Fifo;
                config.alpha_mode = wgpu::CompositeAlphaMode::Auto;
                config.desired_maximum_frame_latency = 2;
            }
            ConfigPreset::LowLatency => {
                config.format = wgpu::TextureFormat::Bgra8UnormSrgb;
                config.present_mode = wgpu::PresentMode::Mailbox;
                config.alpha_mode = wgpu::CompositeAlphaMode::Auto;
                config.desired_maximum_frame_latency = 1;
            }
            ConfigPreset::PowerSaving => {
                config.format = wgpu::TextureFormat::Bgra8UnormSrgb;
                config.present_mode = wgpu::PresentMode::Fifo;
                config.alpha_mode = wgpu::CompositeAlphaMode::Auto;
                config.desired_maximum_frame_latency = 3;
            }
            ConfigPreset::HighQuality => {
                config.format = wgpu::TextureFormat::Bgra8UnormSrgb;
                config.present_mode = wgpu::PresentMode::Fifo;
                config.alpha_mode = wgpu::CompositeAlphaMode::Auto;
                config.desired_maximum_frame_latency = 2;
                // Enable sRGB companion format for high-quality rendering
                if !config.view_formats.contains(&wgpu::TextureFormat::Bgra8Unorm) {
                    config.view_formats.push(wgpu::TextureFormat::Bgra8Unorm);
                }
            }
            ConfigPreset::HDR => {
                config.format = wgpu::TextureFormat::Rgba16Float;
                config.present_mode = wgpu::PresentMode::Fifo;
                config.alpha_mode = wgpu::CompositeAlphaMode::Auto;
                config.desired_maximum_frame_latency = 2;
            }
        }
    }

    /// Get a human-readable description of this preset.
    ///
    /// # Example
    ///
    /// ```ignore
    /// println!("Using preset: {}", ConfigPreset::LowLatency.description());
    /// // Output: "Using preset: Low-latency rendering for competitive gaming and VR"
    /// ```
    pub fn description(&self) -> &'static str {
        match self {
            ConfigPreset::Default => "General purpose rendering with vsync",
            ConfigPreset::LowLatency => "Low-latency rendering for competitive gaming and VR",
            ConfigPreset::PowerSaving => "Power-efficient rendering for mobile devices",
            ConfigPreset::HighQuality => "High-quality rendering for cinematic experiences",
            ConfigPreset::HDR => "HDR rendering for wide color gamut displays",
        }
    }

    /// Get the present mode this preset uses.
    pub fn present_mode(&self) -> wgpu::PresentMode {
        match self {
            ConfigPreset::LowLatency => wgpu::PresentMode::Mailbox,
            _ => wgpu::PresentMode::Fifo,
        }
    }

    /// Get the frame latency this preset uses.
    pub fn frame_latency(&self) -> u32 {
        match self {
            ConfigPreset::LowLatency => 1,
            ConfigPreset::PowerSaving => 3,
            _ => 2,
        }
    }

    /// Get the texture format this preset prefers.
    pub fn preferred_format(&self) -> wgpu::TextureFormat {
        match self {
            ConfigPreset::HDR => wgpu::TextureFormat::Rgba16Float,
            _ => wgpu::TextureFormat::Bgra8UnormSrgb,
        }
    }

    /// Check if this preset requires HDR surface support.
    pub fn requires_hdr(&self) -> bool {
        matches!(self, ConfigPreset::HDR)
    }

    /// Get all available presets.
    pub fn all() -> &'static [ConfigPreset] {
        &[
            ConfigPreset::Default,
            ConfigPreset::LowLatency,
            ConfigPreset::PowerSaving,
            ConfigPreset::HighQuality,
            ConfigPreset::HDR,
        ]
    }
}

impl fmt::Display for ConfigPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigPreset::Default => write!(f, "Default"),
            ConfigPreset::LowLatency => write!(f, "Low Latency"),
            ConfigPreset::PowerSaving => write!(f, "Power Saving"),
            ConfigPreset::HighQuality => write!(f, "High Quality"),
            ConfigPreset::HDR => write!(f, "HDR"),
        }
    }
}

// ============================================================================
// Surface Configuration Builder
// ============================================================================

/// Fluent builder for constructing surface configurations.
///
/// The builder provides a convenient way to construct [`SurfaceConfiguration`]
/// instances with validation and preset support.
///
/// # Example
///
/// ```ignore
/// use renderer_backend::presentation::{SurfaceConfigBuilder, ConfigPreset};
///
/// // Basic usage
/// let config = SurfaceConfigBuilder::new()
///     .with_size(1920, 1080)
///     .with_format(wgpu::TextureFormat::Bgra8UnormSrgb)
///     .with_present_mode(wgpu::PresentMode::Mailbox)
///     .build();
///
/// // Using presets
/// let gaming_config = SurfaceConfigBuilder::new()
///     .with_size(2560, 1440)
///     .with_preset(ConfigPreset::LowLatency)
///     .build();
///
/// // With validation
/// let validated = SurfaceConfigBuilder::new()
///     .with_size(1920, 1080)
///     .with_preset(ConfigPreset::HDR)
///     .build_validated(&caps)?;
/// ```
#[derive(Debug, Clone)]
pub struct SurfaceConfigBuilder {
    width: u32,
    height: u32,
    format: Option<wgpu::TextureFormat>,
    present_mode: Option<wgpu::PresentMode>,
    alpha_mode: Option<wgpu::CompositeAlphaMode>,
    desired_maximum_frame_latency: Option<u32>,
    view_formats: Vec<wgpu::TextureFormat>,
    preset: Option<ConfigPreset>,
}

impl SurfaceConfigBuilder {
    /// Create a new builder with default values.
    ///
    /// Default dimensions are 1x1 (must be set before use).
    pub fn new() -> Self {
        Self {
            width: 1,
            height: 1,
            format: None,
            present_mode: None,
            alpha_mode: None,
            desired_maximum_frame_latency: None,
            view_formats: Vec::new(),
            preset: None,
        }
    }

    /// Set the surface dimensions.
    ///
    /// Dimensions are clamped to a minimum of 1.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width.max(1);
        self.height = height.max(1);
        self
    }

    /// Set the texture format.
    ///
    /// This overrides any format set by a preset.
    pub fn with_format(mut self, format: wgpu::TextureFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Set the present mode.
    ///
    /// This overrides any present mode set by a preset.
    pub fn with_present_mode(mut self, mode: wgpu::PresentMode) -> Self {
        self.present_mode = Some(mode);
        self
    }

    /// Set the alpha compositing mode.
    ///
    /// This overrides any alpha mode set by a preset.
    pub fn with_alpha_mode(mut self, mode: wgpu::CompositeAlphaMode) -> Self {
        self.alpha_mode = Some(mode);
        self
    }

    /// Set the maximum frame latency (number of frames that can be queued).
    pub fn with_frame_latency(mut self, latency: u32) -> Self {
        self.desired_maximum_frame_latency = Some(latency.max(1));
        self
    }

    /// Add view formats for texture views.
    pub fn with_view_formats(mut self, formats: &[wgpu::TextureFormat]) -> Self {
        self.view_formats = formats.to_vec();
        self
    }

    /// Add the sRGB companion format to view formats.
    ///
    /// This enables the sRGB toggle pattern.
    pub fn with_srgb_view_format(mut self) -> Self {
        let base_format = self.format.unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);
        if let Some(companion) = super::get_srgb_companion_format(base_format) {
            if !self.view_formats.contains(&companion) {
                self.view_formats.push(companion);
            }
        }
        self
    }

    /// Apply a configuration preset.
    ///
    /// The preset sets default values for format, present mode, alpha mode,
    /// and frame latency. Any values explicitly set after this call will
    /// override the preset.
    pub fn with_preset(mut self, preset: ConfigPreset) -> Self {
        self.preset = Some(preset);
        self
    }

    /// Build the surface configuration.
    ///
    /// Values not explicitly set use sensible defaults:
    /// - Format: `Bgra8UnormSrgb`
    /// - Present mode: `Fifo`
    /// - Alpha mode: `Auto`
    /// - Frame latency: 2
    pub fn build(self) -> super::SurfaceConfiguration {
        let mut config = super::SurfaceConfiguration::new(self.width, self.height);

        // Apply preset first (if any)
        if let Some(preset) = self.preset {
            preset.apply(&mut config);
        }

        // Override with explicit values
        if let Some(format) = self.format {
            config.format = format;
        }
        if let Some(mode) = self.present_mode {
            config.present_mode = mode;
        }
        if let Some(mode) = self.alpha_mode {
            config.alpha_mode = mode;
        }
        if let Some(latency) = self.desired_maximum_frame_latency {
            config.desired_maximum_frame_latency = latency;
        }
        if !self.view_formats.is_empty() {
            config.view_formats = self.view_formats;
        }

        config
    }

    /// Build and validate the configuration against surface capabilities.
    ///
    /// Returns `Err` if the configuration is invalid for the given surface.
    pub fn build_validated(
        self,
        caps: &SurfaceCapabilities,
    ) -> Result<super::SurfaceConfiguration, ConfigValidationError> {
        let config = self.build();
        validate_config(&config, caps)?;
        Ok(config)
    }

    /// Build from surface capabilities, selecting optimal values.
    ///
    /// This is a convenience method that:
    /// 1. Builds with preset values (if any)
    /// 2. Falls back to capability-preferred values for unset options
    pub fn build_from_capabilities(self, caps: &SurfaceCapabilities) -> super::SurfaceConfiguration {
        // Capture these before build() consumes self
        let format_was_none = self.format.is_none();
        let preset_was_none = self.preset.is_none();

        let mut config = self.build();

        // If format wasn't explicitly set and preset didn't set it,
        // use capability-preferred format
        if format_was_none && preset_was_none {
            if let Some(fmt) = caps.preferred_format() {
                config.format = fmt;
            }
        }

        // Validate format against capabilities, fall back if needed
        if !caps.supports_format(config.format) {
            if let Some(fmt) = caps.preferred_format() {
                config.format = fmt;
            }
        }

        // Validate present mode against capabilities, fall back if needed
        if !caps.supports_present_mode(config.present_mode) {
            config.present_mode = caps.preferred_present_mode();
        }

        // Validate alpha mode against capabilities, fall back if needed
        if !caps.alpha_modes.contains(&config.alpha_mode) {
            config.alpha_mode = caps.preferred_alpha_mode();
        }

        config
    }
}

impl Default for SurfaceConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Validation Function
// ============================================================================

/// Validate a surface configuration against capabilities with detailed errors.
///
/// This function performs strict validation and returns specific error types
/// for each validation failure.
///
/// # Example
///
/// ```ignore
/// let config = SurfaceConfiguration::new(1920, 1080)
///     .with_format(wgpu::TextureFormat::Rgba16Float);
///
/// match validate_config(&config, &caps) {
///     Ok(()) => println!("Valid configuration"),
///     Err(ConfigValidationError::UnsupportedFormat(fmt)) => {
///         println!("Format {:?} not supported, falling back", fmt);
///     }
///     Err(e) => return Err(e.into()),
/// }
/// ```
pub fn validate_config(
    config: &super::SurfaceConfiguration,
    caps: &SurfaceCapabilities,
) -> Result<(), ConfigValidationError> {
    // Check dimensions
    if config.width == 0 || config.height == 0 {
        return Err(ConfigValidationError::ZeroDimensions);
    }

    // Check format
    if !caps.supports_format(config.format) {
        return Err(ConfigValidationError::UnsupportedFormat(config.format));
    }

    // Check present mode
    if !caps.supports_present_mode(config.present_mode) {
        return Err(ConfigValidationError::UnsupportedPresentMode(config.present_mode));
    }

    // Check alpha mode
    if !caps.alpha_modes.contains(&config.alpha_mode) {
        return Err(ConfigValidationError::UnsupportedAlphaMode(config.alpha_mode));
    }

    // Validate view formats
    for &view_format in &config.view_formats {
        // View format must be compatible with base format
        // For now, we check if it's in the supported formats or is
        // a valid sRGB companion
        let is_srgb_companion = super::are_srgb_companions(config.format, view_format);
        let is_same_format = config.format == view_format;

        if !is_srgb_companion && !is_same_format && !caps.supports_format(view_format) {
            return Err(ConfigValidationError::InvalidViewFormat(view_format));
        }
    }

    Ok(())
}

// ============================================================================
// Extension Methods for SurfaceConfiguration
// ============================================================================

/// Extension trait adding strict validation to SurfaceConfiguration.
pub trait SurfaceConfigValidation {
    /// Validate with detailed error types.
    fn validate_strict(&self, caps: &SurfaceCapabilities) -> Result<(), ConfigValidationError>;
}

impl SurfaceConfigValidation for super::SurfaceConfiguration {
    fn validate_strict(&self, caps: &SurfaceCapabilities) -> Result<(), ConfigValidationError> {
        validate_config(self, caps)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_capabilities() -> SurfaceCapabilities {
        SurfaceCapabilities {
            formats: vec![
                wgpu::TextureFormat::Bgra8UnormSrgb,
                wgpu::TextureFormat::Bgra8Unorm,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                wgpu::TextureFormat::Rgba8Unorm,
            ],
            present_modes: vec![
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::Mailbox,
            ],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::Auto,
                wgpu::CompositeAlphaMode::Opaque,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    fn mock_capabilities_hdr() -> SurfaceCapabilities {
        SurfaceCapabilities {
            formats: vec![
                wgpu::TextureFormat::Bgra8UnormSrgb,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureFormat::Rgb10a2Unorm,
            ],
            present_modes: vec![
                wgpu::PresentMode::Fifo,
                wgpu::PresentMode::Mailbox,
            ],
            alpha_modes: vec![
                wgpu::CompositeAlphaMode::Auto,
            ],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        }
    }

    // ========================================================================
    // SurfaceConfiguration method tests
    // ========================================================================

    #[test]
    fn test_surface_config_aspect_ratio_standard() {
        let config = super::super::SurfaceConfiguration::new(1920, 1080);
        let ratio = config.aspect_ratio();
        assert!((ratio - 1.777).abs() < 0.01);
    }

    #[test]
    fn test_surface_config_aspect_ratio_square() {
        let config = super::super::SurfaceConfiguration::new(1000, 1000);
        let ratio = config.aspect_ratio();
        assert!((ratio - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_surface_config_aspect_ratio_portrait() {
        let config = super::super::SurfaceConfiguration::new(1080, 1920);
        let ratio = config.aspect_ratio();
        assert!((ratio - 0.5625).abs() < 0.001);
    }

    #[test]
    fn test_surface_config_aspect_ratio_zero_height() {
        let mut config = super::super::SurfaceConfiguration::new(1920, 1);
        config.height = 0; // Bypass constructor clamping
        assert_eq!(config.aspect_ratio(), 1.0);
    }

    #[test]
    fn test_surface_config_is_hdr_false_srgb() {
        let config = super::super::SurfaceConfiguration::new(1920, 1080);
        assert!(!config.is_hdr());
    }

    #[test]
    fn test_surface_config_is_hdr_false_linear() {
        let config = super::super::SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm);
        assert!(!config.is_hdr());
    }

    #[test]
    fn test_surface_config_is_hdr_true_rgba16float() {
        let config = super::super::SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba16Float);
        assert!(config.is_hdr());
    }

    #[test]
    fn test_surface_config_is_hdr_true_rgb10a2() {
        let config = super::super::SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgb10a2Unorm);
        assert!(config.is_hdr());
    }

    #[test]
    fn test_surface_config_is_hdr_true_rg11b10float() {
        let config = super::super::SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rg11b10Float);
        assert!(config.is_hdr());
    }

    #[test]
    fn test_surface_config_to_wgpu_config_alias() {
        let config = super::super::SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8Unorm);
        let wgpu1 = config.to_wgpu();
        let wgpu2 = config.to_wgpu_config();
        assert_eq!(wgpu1.width, wgpu2.width);
        assert_eq!(wgpu1.height, wgpu2.height);
        assert_eq!(wgpu1.format, wgpu2.format);
    }

    // ========================================================================
    // ConfigValidationError tests
    // ========================================================================

    #[test]
    fn test_config_validation_error_zero_dimensions() {
        let err = ConfigValidationError::ZeroDimensions;
        assert!(err.is_recoverable());
        assert!(err.to_string().contains("non-zero"));
        assert!(!err.suggested_fix().is_empty());
    }

    #[test]
    fn test_config_validation_error_unsupported_format() {
        let err = ConfigValidationError::UnsupportedFormat(wgpu::TextureFormat::Rgba16Float);
        assert!(!err.is_recoverable());
        assert!(err.to_string().contains("Rgba16Float"));
        assert!(err.suggested_fix().contains("preferred_format"));
    }

    #[test]
    fn test_config_validation_error_unsupported_present_mode() {
        let err = ConfigValidationError::UnsupportedPresentMode(wgpu::PresentMode::Mailbox);
        assert!(!err.is_recoverable());
        assert!(err.to_string().contains("Mailbox"));
    }

    #[test]
    fn test_config_validation_error_unsupported_alpha_mode() {
        let err = ConfigValidationError::UnsupportedAlphaMode(wgpu::CompositeAlphaMode::PreMultiplied);
        assert!(!err.is_recoverable());
        assert!(err.to_string().contains("PreMultiplied"));
    }

    #[test]
    fn test_config_validation_error_invalid_view_format() {
        let err = ConfigValidationError::InvalidViewFormat(wgpu::TextureFormat::R8Unorm);
        assert!(!err.is_recoverable());
        assert!(err.to_string().contains("R8Unorm"));
    }

    #[test]
    fn test_config_validation_error_display() {
        let err = ConfigValidationError::ZeroDimensions;
        let display = format!("{}", err);
        assert!(!display.is_empty());
    }

    #[test]
    fn test_config_validation_error_debug() {
        let err = ConfigValidationError::ZeroDimensions;
        let debug = format!("{:?}", err);
        assert!(debug.contains("ZeroDimensions"));
    }

    // ========================================================================
    // ConfigPreset tests
    // ========================================================================

    #[test]
    fn test_config_preset_default() {
        let preset = ConfigPreset::Default;
        assert_eq!(preset.present_mode(), wgpu::PresentMode::Fifo);
        assert_eq!(preset.frame_latency(), 2);
        assert_eq!(preset.preferred_format(), wgpu::TextureFormat::Bgra8UnormSrgb);
        assert!(!preset.requires_hdr());
    }

    #[test]
    fn test_config_preset_low_latency() {
        let preset = ConfigPreset::LowLatency;
        assert_eq!(preset.present_mode(), wgpu::PresentMode::Mailbox);
        assert_eq!(preset.frame_latency(), 1);
        assert!(!preset.requires_hdr());
    }

    #[test]
    fn test_config_preset_power_saving() {
        let preset = ConfigPreset::PowerSaving;
        assert_eq!(preset.present_mode(), wgpu::PresentMode::Fifo);
        assert_eq!(preset.frame_latency(), 3);
    }

    #[test]
    fn test_config_preset_high_quality() {
        let preset = ConfigPreset::HighQuality;
        assert_eq!(preset.present_mode(), wgpu::PresentMode::Fifo);
        assert_eq!(preset.frame_latency(), 2);
    }

    #[test]
    fn test_config_preset_hdr() {
        let preset = ConfigPreset::HDR;
        assert_eq!(preset.preferred_format(), wgpu::TextureFormat::Rgba16Float);
        assert!(preset.requires_hdr());
    }

    #[test]
    fn test_config_preset_apply_default() {
        let mut config = super::super::SurfaceConfiguration::new(1920, 1080);
        ConfigPreset::Default.apply(&mut config);
        assert_eq!(config.present_mode, wgpu::PresentMode::Fifo);
        assert_eq!(config.desired_maximum_frame_latency, 2);
    }

    #[test]
    fn test_config_preset_apply_low_latency() {
        let mut config = super::super::SurfaceConfiguration::new(1920, 1080);
        ConfigPreset::LowLatency.apply(&mut config);
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
        assert_eq!(config.desired_maximum_frame_latency, 1);
    }

    #[test]
    fn test_config_preset_apply_hdr() {
        let mut config = super::super::SurfaceConfiguration::new(1920, 1080);
        ConfigPreset::HDR.apply(&mut config);
        assert_eq!(config.format, wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn test_config_preset_apply_high_quality_adds_view_format() {
        let mut config = super::super::SurfaceConfiguration::new(1920, 1080);
        ConfigPreset::HighQuality.apply(&mut config);
        assert!(config.view_formats.contains(&wgpu::TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn test_config_preset_description_not_empty() {
        for preset in ConfigPreset::all() {
            assert!(!preset.description().is_empty());
        }
    }

    #[test]
    fn test_config_preset_display() {
        assert_eq!(format!("{}", ConfigPreset::Default), "Default");
        assert_eq!(format!("{}", ConfigPreset::LowLatency), "Low Latency");
        assert_eq!(format!("{}", ConfigPreset::PowerSaving), "Power Saving");
        assert_eq!(format!("{}", ConfigPreset::HighQuality), "High Quality");
        assert_eq!(format!("{}", ConfigPreset::HDR), "HDR");
    }

    #[test]
    fn test_config_preset_all() {
        let all = ConfigPreset::all();
        assert_eq!(all.len(), 5);
        assert!(all.contains(&ConfigPreset::Default));
        assert!(all.contains(&ConfigPreset::HDR));
    }

    #[test]
    fn test_config_preset_default_trait() {
        let preset: ConfigPreset = Default::default();
        assert_eq!(preset, ConfigPreset::Default);
    }

    // ========================================================================
    // SurfaceConfigBuilder tests
    // ========================================================================

    #[test]
    fn test_builder_new() {
        let builder = SurfaceConfigBuilder::new();
        let config = builder.build();
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn test_builder_with_size() {
        let config = SurfaceConfigBuilder::new()
            .with_size(1920, 1080)
            .build();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn test_builder_with_size_clamps_zero() {
        let config = SurfaceConfigBuilder::new()
            .with_size(0, 0)
            .build();
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    #[test]
    fn test_builder_with_format() {
        let config = SurfaceConfigBuilder::new()
            .with_size(800, 600)
            .with_format(wgpu::TextureFormat::Rgba8UnormSrgb)
            .build();
        assert_eq!(config.format, wgpu::TextureFormat::Rgba8UnormSrgb);
    }

    #[test]
    fn test_builder_with_present_mode() {
        let config = SurfaceConfigBuilder::new()
            .with_size(800, 600)
            .with_present_mode(wgpu::PresentMode::Mailbox)
            .build();
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
    }

    #[test]
    fn test_builder_with_alpha_mode() {
        let config = SurfaceConfigBuilder::new()
            .with_size(800, 600)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Opaque)
            .build();
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
    }

    #[test]
    fn test_builder_with_frame_latency() {
        let config = SurfaceConfigBuilder::new()
            .with_size(800, 600)
            .with_frame_latency(3)
            .build();
        assert_eq!(config.desired_maximum_frame_latency, 3);
    }

    #[test]
    fn test_builder_with_frame_latency_clamps_zero() {
        let config = SurfaceConfigBuilder::new()
            .with_size(800, 600)
            .with_frame_latency(0)
            .build();
        assert_eq!(config.desired_maximum_frame_latency, 1);
    }

    #[test]
    fn test_builder_with_view_formats() {
        let config = SurfaceConfigBuilder::new()
            .with_size(800, 600)
            .with_view_formats(&[wgpu::TextureFormat::Bgra8Unorm])
            .build();
        assert!(config.view_formats.contains(&wgpu::TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn test_builder_with_preset() {
        let config = SurfaceConfigBuilder::new()
            .with_size(1920, 1080)
            .with_preset(ConfigPreset::LowLatency)
            .build();
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
        assert_eq!(config.desired_maximum_frame_latency, 1);
    }

    #[test]
    fn test_builder_preset_then_override() {
        let config = SurfaceConfigBuilder::new()
            .with_size(1920, 1080)
            .with_preset(ConfigPreset::LowLatency)
            .with_frame_latency(2) // Override preset value
            .build();
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
        assert_eq!(config.desired_maximum_frame_latency, 2);
    }

    #[test]
    fn test_builder_complete_chain() {
        let config = SurfaceConfigBuilder::new()
            .with_size(2560, 1440)
            .with_format(wgpu::TextureFormat::Bgra8UnormSrgb)
            .with_present_mode(wgpu::PresentMode::Mailbox)
            .with_alpha_mode(wgpu::CompositeAlphaMode::Opaque)
            .with_frame_latency(2)
            .with_view_formats(&[wgpu::TextureFormat::Bgra8Unorm])
            .build();

        assert_eq!(config.width, 2560);
        assert_eq!(config.height, 1440);
        assert_eq!(config.format, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(config.present_mode, wgpu::PresentMode::Mailbox);
        assert_eq!(config.alpha_mode, wgpu::CompositeAlphaMode::Opaque);
        assert_eq!(config.desired_maximum_frame_latency, 2);
        assert!(config.view_formats.contains(&wgpu::TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn test_builder_build_validated_success() {
        let caps = mock_capabilities();
        let result = SurfaceConfigBuilder::new()
            .with_size(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8UnormSrgb)
            .with_present_mode(wgpu::PresentMode::Fifo)
            .build_validated(&caps);

        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_build_validated_error_format() {
        let caps = mock_capabilities();
        let result = SurfaceConfigBuilder::new()
            .with_size(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba16Float) // Not in mock caps
            .build_validated(&caps);

        assert!(matches!(
            result,
            Err(ConfigValidationError::UnsupportedFormat(_))
        ));
    }

    #[test]
    fn test_builder_build_validated_error_present_mode() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            present_modes: vec![wgpu::PresentMode::Fifo], // Only Fifo
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let result = SurfaceConfigBuilder::new()
            .with_size(1920, 1080)
            .with_present_mode(wgpu::PresentMode::Immediate)
            .build_validated(&caps);

        assert!(matches!(
            result,
            Err(ConfigValidationError::UnsupportedPresentMode(_))
        ));
    }

    #[test]
    fn test_builder_build_from_capabilities() {
        let caps = mock_capabilities();
        let config = SurfaceConfigBuilder::new()
            .with_size(1920, 1080)
            .build_from_capabilities(&caps);

        // Should use capability-preferred values
        assert!(caps.supports_format(config.format));
        assert!(caps.supports_present_mode(config.present_mode));
    }

    #[test]
    fn test_builder_build_from_capabilities_fallback() {
        let caps = mock_capabilities();
        let config = SurfaceConfigBuilder::new()
            .with_size(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba16Float) // Not supported
            .build_from_capabilities(&caps);

        // Should fall back to supported format
        assert!(caps.supports_format(config.format));
        assert_ne!(config.format, wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn test_builder_default() {
        let builder: SurfaceConfigBuilder = Default::default();
        let config = builder.build();
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
    }

    // ========================================================================
    // validate_config function tests
    // ========================================================================

    #[test]
    fn test_validate_config_success() {
        let caps = mock_capabilities();
        let config = super::super::SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8UnormSrgb);
        assert!(validate_config(&config, &caps).is_ok());
    }

    #[test]
    fn test_validate_config_zero_width() {
        let caps = mock_capabilities();
        let mut config = super::super::SurfaceConfiguration::new(1, 1080);
        config.width = 0;
        assert!(matches!(
            validate_config(&config, &caps),
            Err(ConfigValidationError::ZeroDimensions)
        ));
    }

    #[test]
    fn test_validate_config_zero_height() {
        let caps = mock_capabilities();
        let mut config = super::super::SurfaceConfiguration::new(1920, 1);
        config.height = 0;
        assert!(matches!(
            validate_config(&config, &caps),
            Err(ConfigValidationError::ZeroDimensions)
        ));
    }

    #[test]
    fn test_validate_config_unsupported_format() {
        let caps = mock_capabilities();
        let config = super::super::SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Rgba16Float);
        assert!(matches!(
            validate_config(&config, &caps),
            Err(ConfigValidationError::UnsupportedFormat(wgpu::TextureFormat::Rgba16Float))
        ));
    }

    #[test]
    fn test_validate_config_unsupported_present_mode() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let config = super::super::SurfaceConfiguration::new(1920, 1080)
            .with_present_mode(wgpu::PresentMode::Immediate);
        assert!(matches!(
            validate_config(&config, &caps),
            Err(ConfigValidationError::UnsupportedPresentMode(_))
        ));
    }

    #[test]
    fn test_validate_config_unsupported_alpha_mode() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let config = super::super::SurfaceConfiguration::new(1920, 1080)
            .with_alpha_mode(wgpu::CompositeAlphaMode::PreMultiplied);
        assert!(matches!(
            validate_config(&config, &caps),
            Err(ConfigValidationError::UnsupportedAlphaMode(_))
        ));
    }

    #[test]
    fn test_validate_config_valid_srgb_view_format() {
        let caps = mock_capabilities();
        let config = super::super::SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8UnormSrgb)
            .with_view_formats(&[wgpu::TextureFormat::Bgra8Unorm]); // sRGB companion
        assert!(validate_config(&config, &caps).is_ok());
    }

    #[test]
    fn test_validate_config_invalid_view_format() {
        let caps = SurfaceCapabilities {
            formats: vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            present_modes: vec![wgpu::PresentMode::Fifo],
            alpha_modes: vec![wgpu::CompositeAlphaMode::Auto],
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let config = super::super::SurfaceConfiguration::new(1920, 1080)
            .with_format(wgpu::TextureFormat::Bgra8UnormSrgb)
            .with_view_formats(&[wgpu::TextureFormat::R8Unorm]); // Incompatible
        assert!(matches!(
            validate_config(&config, &caps),
            Err(ConfigValidationError::InvalidViewFormat(_))
        ));
    }

    // ========================================================================
    // Extension trait tests
    // ========================================================================

    #[test]
    fn test_surface_config_validation_trait() {
        let caps = mock_capabilities();
        let config = super::super::SurfaceConfiguration::new(1920, 1080);
        assert!(config.validate_strict(&caps).is_ok());
    }

    #[test]
    fn test_surface_config_validation_trait_error() {
        let caps = mock_capabilities();
        let mut config = super::super::SurfaceConfiguration::new(1920, 1080);
        config.width = 0;
        assert!(config.validate_strict(&caps).is_err());
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn test_config_minimum_dimensions() {
        let config = SurfaceConfigBuilder::new()
            .with_size(1, 1)
            .build();
        assert_eq!(config.width, 1);
        assert_eq!(config.height, 1);
        assert!((config.aspect_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_config_large_dimensions() {
        let config = SurfaceConfigBuilder::new()
            .with_size(7680, 4320) // 8K
            .build();
        assert_eq!(config.width, 7680);
        assert_eq!(config.height, 4320);
    }

    #[test]
    fn test_builder_hdr_preset_with_hdr_caps() {
        let caps = mock_capabilities_hdr();
        let config = SurfaceConfigBuilder::new()
            .with_size(3840, 2160)
            .with_preset(ConfigPreset::HDR)
            .build_validated(&caps);

        assert!(config.is_ok());
        let config = config.unwrap();
        assert!(config.is_hdr());
        assert_eq!(config.format, wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn test_builder_hdr_preset_without_hdr_caps() {
        let caps = mock_capabilities(); // No HDR formats
        let result = SurfaceConfigBuilder::new()
            .with_size(1920, 1080)
            .with_preset(ConfigPreset::HDR)
            .build_validated(&caps);

        assert!(matches!(
            result,
            Err(ConfigValidationError::UnsupportedFormat(_))
        ));
    }
}
