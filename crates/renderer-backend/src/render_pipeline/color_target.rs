//! Color target state configuration for render pipelines.
//!
//! This module provides color target abstractions for wgpu 25.x render pipelines
//! with format selection, blend state configuration, write masks, and MRT support.
//!
//! # Overview
//!
//! A color target describes how the fragment shader output maps to a render
//! target (texture attachment) in a render pass. Each target specifies:
//!
//! - **Format**: The texture format of the render target
//! - **Blend**: Optional blending operations for combining with existing values
//! - **Write Mask**: Which color channels to write (R, G, B, A)
//!
//! # Multiple Render Targets (MRT)
//!
//! Modern GPUs support writing to multiple render targets simultaneously.
//! This is commonly used for:
//!
//! - **Deferred Rendering**: G-buffer passes write albedo, normal, material properties
//! - **HDR Rendering**: Separate brightness for bloom extraction
//! - **Velocity Buffers**: Motion vectors for temporal effects
//!
//! Typical MRT limits:
//! - Most GPUs: 8 simultaneous color attachments
//! - Mobile: 4-8 depending on hardware
//!
//! # wgpu API Reference
//!
//! ```ignore
//! pub struct ColorTargetState {
//!     pub format: TextureFormat,
//!     pub blend: Option<BlendState>,
//!     pub write_mask: ColorWrites,
//! }
//!
//! pub struct ColorWrites: u32 {
//!     const RED = 1;
//!     const GREEN = 2;
//!     const BLUE = 4;
//!     const ALPHA = 8;
//!     const COLOR = RED | GREEN | BLUE;
//!     const ALL = RED | GREEN | BLUE | ALPHA;
//! }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::render_pipeline::color_target::{ColorTarget, ColorTargetBuilder};
//!
//! // Simple swapchain target
//! let swapchain = ColorTarget::bgra8_unorm_srgb();
//!
//! // HDR with alpha blending
//! let hdr = ColorTargetBuilder::new()
//!     .format(wgpu::TextureFormat::Rgba16Float)
//!     .alpha_blend()
//!     .build();
//!
//! // G-buffer setup for deferred rendering
//! let gbuffer = ColorTargetBuilder::gbuffer()
//!     .albedo(wgpu::TextureFormat::Rgba8Unorm)
//!     .normal(wgpu::TextureFormat::Rgba16Float)
//!     .material(wgpu::TextureFormat::Rgba8Unorm)
//!     .build_targets();
//!
//! // Convert to wgpu type
//! let wgpu_state: wgpu::ColorTargetState = swapchain.into();
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// ColorTarget
// ---------------------------------------------------------------------------

/// Describes a single color target (render target attachment) for a render pipeline.
///
/// # Fields
///
/// | Field | Type | Description |
/// |-------|------|-------------|
/// | `format` | `TextureFormat` | Texture format of the render target |
/// | `blend` | `Option<BlendState>` | Optional blending configuration |
/// | `write_mask` | `ColorWrites` | Which color channels to write |
///
/// # Blending
///
/// When `blend` is `None`, the fragment shader output replaces the existing
/// value. When `Some(BlendState)`, the output is combined with the existing
/// value according to the blend equation:
///
/// ```text
/// result = src_factor * src_value <operation> dst_factor * dst_value
/// ```
///
/// # Write Mask
///
/// Controls which channels are written to the render target:
/// - `ColorWrites::RED` - Write red channel
/// - `ColorWrites::GREEN` - Write green channel
/// - `ColorWrites::BLUE` - Write blue channel
/// - `ColorWrites::ALPHA` - Write alpha channel
/// - `ColorWrites::COLOR` - Write RGB (not alpha)
/// - `ColorWrites::ALL` - Write all channels
///
/// # Defaults
///
/// Default is `Rgba8UnormSrgb` format with no blending and all channels written.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorTarget {
    /// Texture format of the render target.
    pub format: wgpu::TextureFormat,
    /// Optional blend state for combining with existing values.
    pub blend: Option<wgpu::BlendState>,
    /// Which color channels to write to the target.
    pub write_mask: wgpu::ColorWrites,
}

impl Default for ColorTarget {
    fn default() -> Self {
        Self {
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }
    }
}

impl ColorTarget {
    /// Create a new color target with the specified format.
    ///
    /// Uses default write mask (ALL) and no blending.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let target = ColorTarget::new(wgpu::TextureFormat::Rgba8Unorm);
    /// ```
    pub fn new(format: wgpu::TextureFormat) -> Self {
        Self {
            format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }
    }

    // -------------------------------------------------------------------------
    // Format Presets
    // -------------------------------------------------------------------------

    /// Create an RGBA8 unorm target (standard 8-bit per channel).
    ///
    /// Linear color space - use for non-sRGB rendering.
    pub fn rgba8_unorm() -> Self {
        Self::new(wgpu::TextureFormat::Rgba8Unorm)
    }

    /// Create an RGBA8 unorm sRGB target (standard sRGB).
    ///
    /// sRGB color space - use for final output or sRGB textures.
    pub fn rgba8_unorm_srgb() -> Self {
        Self::new(wgpu::TextureFormat::Rgba8UnormSrgb)
    }

    /// Create a BGRA8 unorm target (swapchain compatible).
    ///
    /// Linear color space - common swapchain format on Windows/macOS.
    pub fn bgra8_unorm() -> Self {
        Self::new(wgpu::TextureFormat::Bgra8Unorm)
    }

    /// Create a BGRA8 unorm sRGB target (swapchain compatible).
    ///
    /// sRGB color space - common swapchain format with gamma correction.
    pub fn bgra8_unorm_srgb() -> Self {
        Self::new(wgpu::TextureFormat::Bgra8UnormSrgb)
    }

    /// Create an RGBA16 float target (HDR rendering).
    ///
    /// 16-bit floating point per channel for HDR content.
    /// Supports values outside [0, 1] range and negative values.
    pub fn rgba16_float() -> Self {
        Self::new(wgpu::TextureFormat::Rgba16Float)
    }

    /// Create an RGBA32 float target (high precision HDR).
    ///
    /// 32-bit floating point per channel for maximum precision.
    /// Use sparingly due to memory bandwidth cost.
    pub fn rgba32_float() -> Self {
        Self::new(wgpu::TextureFormat::Rgba32Float)
    }

    /// Create an RGB10A2 unorm target (10-bit color + 2-bit alpha).
    ///
    /// Higher precision for color with minimal alpha.
    /// Good for HDR tone mapping output.
    pub fn rgb10a2_unorm() -> Self {
        Self::new(wgpu::TextureFormat::Rgb10a2Unorm)
    }

    /// Create an RG11B10 float target (shared exponent HDR).
    ///
    /// Compact HDR format with no alpha channel.
    /// Good for light accumulation buffers.
    pub fn rg11b10_float() -> Self {
        Self::new(wgpu::TextureFormat::Rg11b10Float)
    }

    /// Create an R8 unorm target (single channel).
    ///
    /// 8-bit single channel - good for masks, stencil copies, luminance.
    pub fn r8_unorm() -> Self {
        Self::new(wgpu::TextureFormat::R8Unorm)
    }

    /// Create an RG8 unorm target (two channels).
    ///
    /// 8-bit two channels - good for normal maps (XY), motion vectors.
    pub fn rg8_unorm() -> Self {
        Self::new(wgpu::TextureFormat::Rg8Unorm)
    }

    /// Create an R16 float target (single channel HDR).
    ///
    /// 16-bit float single channel - good for depth, luminance, masks.
    pub fn r16_float() -> Self {
        Self::new(wgpu::TextureFormat::R16Float)
    }

    /// Create an RG16 float target (two channel HDR).
    ///
    /// 16-bit float two channels - good for velocity buffers, normal maps.
    pub fn rg16_float() -> Self {
        Self::new(wgpu::TextureFormat::Rg16Float)
    }

    /// Create an R32 float target (single channel high precision).
    ///
    /// 32-bit float single channel - good for depth linearization, distance fields.
    pub fn r32_float() -> Self {
        Self::new(wgpu::TextureFormat::R32Float)
    }

    /// Create an RG32 float target (two channel high precision).
    ///
    /// 32-bit float two channels - good for high precision motion vectors.
    pub fn rg32_float() -> Self {
        Self::new(wgpu::TextureFormat::Rg32Float)
    }

    // -------------------------------------------------------------------------
    // Blend State Methods
    // -------------------------------------------------------------------------

    /// Set a custom blend state.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let target = ColorTarget::rgba8_unorm()
    ///     .blend(wgpu::BlendState::ALPHA_BLENDING);
    /// ```
    pub fn blend(mut self, blend: wgpu::BlendState) -> Self {
        self.blend = Some(blend);
        self
    }

    /// Set optional blend state (None = no blending).
    pub fn blend_opt(mut self, blend: Option<wgpu::BlendState>) -> Self {
        self.blend = blend;
        self
    }

    /// Enable standard alpha blending.
    ///
    /// Blends source with destination using source alpha:
    /// - Color: `src_alpha * src + (1 - src_alpha) * dst`
    /// - Alpha: `one * src + (1 - src_alpha) * dst`
    pub fn alpha_blend(mut self) -> Self {
        self.blend = Some(wgpu::BlendState::ALPHA_BLENDING);
        self
    }

    /// Enable premultiplied alpha blending.
    ///
    /// For textures where RGB is pre-multiplied by alpha:
    /// - Color: `one * src + (1 - src_alpha) * dst`
    /// - Alpha: `one * src + (1 - src_alpha) * dst`
    pub fn premultiplied_alpha(mut self) -> Self {
        self.blend = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        self
    }

    /// Enable additive blending.
    ///
    /// Adds source to destination (good for particles, glow):
    /// - Color: `one * src + one * dst`
    /// - Alpha: `one * src + one * dst`
    pub fn additive(mut self) -> Self {
        self.blend = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        });
        self
    }

    /// Enable multiply blending.
    ///
    /// Multiplies source with destination (good for shadows, tinting):
    /// - Color: `dst * src + zero * dst` = `dst * src`
    pub fn multiply(mut self) -> Self {
        self.blend = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::DstAlpha,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
        });
        self
    }

    /// Enable replace (no blending).
    ///
    /// Source completely replaces destination:
    /// - Color: `one * src + zero * dst` = `src`
    pub fn replace(mut self) -> Self {
        self.blend = Some(wgpu::BlendState::REPLACE);
        self
    }

    /// Disable blending (same as not setting blend).
    pub fn no_blend(mut self) -> Self {
        self.blend = None;
        self
    }

    // -------------------------------------------------------------------------
    // Write Mask Methods
    // -------------------------------------------------------------------------

    /// Set the color write mask.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Only write RGB, preserve alpha
    /// let target = ColorTarget::rgba8_unorm()
    ///     .write_mask(wgpu::ColorWrites::COLOR);
    /// ```
    pub fn write_mask(mut self, mask: wgpu::ColorWrites) -> Self {
        self.write_mask = mask;
        self
    }

    /// Write all color channels (R, G, B, A).
    pub fn write_all(mut self) -> Self {
        self.write_mask = wgpu::ColorWrites::ALL;
        self
    }

    /// Write only RGB channels (no alpha).
    pub fn write_color(mut self) -> Self {
        self.write_mask = wgpu::ColorWrites::COLOR;
        self
    }

    /// Write only red channel.
    pub fn write_red(mut self) -> Self {
        self.write_mask = wgpu::ColorWrites::RED;
        self
    }

    /// Write only green channel.
    pub fn write_green(mut self) -> Self {
        self.write_mask = wgpu::ColorWrites::GREEN;
        self
    }

    /// Write only blue channel.
    pub fn write_blue(mut self) -> Self {
        self.write_mask = wgpu::ColorWrites::BLUE;
        self
    }

    /// Write only alpha channel.
    pub fn write_alpha(mut self) -> Self {
        self.write_mask = wgpu::ColorWrites::ALPHA;
        self
    }

    /// Write no channels (useful for depth-only with color attachment).
    pub fn write_none(mut self) -> Self {
        self.write_mask = wgpu::ColorWrites::empty();
        self
    }

    // -------------------------------------------------------------------------
    // Query Methods
    // -------------------------------------------------------------------------

    /// Check if blending is enabled.
    pub fn has_blend(&self) -> bool {
        self.blend.is_some()
    }

    /// Check if all channels are written.
    pub fn writes_all(&self) -> bool {
        self.write_mask == wgpu::ColorWrites::ALL
    }

    /// Check if any channel is written.
    pub fn writes_any(&self) -> bool {
        !self.write_mask.is_empty()
    }

    /// Check if the format supports blending.
    ///
    /// Not all formats support blending (e.g., integer formats).
    pub fn format_supports_blend(&self) -> bool {
        // Integer formats don't support blending
        !matches!(
            self.format,
            wgpu::TextureFormat::R8Uint
                | wgpu::TextureFormat::R8Sint
                | wgpu::TextureFormat::R16Uint
                | wgpu::TextureFormat::R16Sint
                | wgpu::TextureFormat::R32Uint
                | wgpu::TextureFormat::R32Sint
                | wgpu::TextureFormat::Rg8Uint
                | wgpu::TextureFormat::Rg8Sint
                | wgpu::TextureFormat::Rg16Uint
                | wgpu::TextureFormat::Rg16Sint
                | wgpu::TextureFormat::Rg32Uint
                | wgpu::TextureFormat::Rg32Sint
                | wgpu::TextureFormat::Rgba8Uint
                | wgpu::TextureFormat::Rgba8Sint
                | wgpu::TextureFormat::Rgba16Uint
                | wgpu::TextureFormat::Rgba16Sint
                | wgpu::TextureFormat::Rgba32Uint
                | wgpu::TextureFormat::Rgba32Sint
        )
    }

    /// Get the number of bytes per pixel for this format.
    ///
    /// Returns `None` for compressed or block formats.
    pub fn bytes_per_pixel(&self) -> Option<u32> {
        match self.format {
            // 1 byte per pixel
            wgpu::TextureFormat::R8Unorm
            | wgpu::TextureFormat::R8Snorm
            | wgpu::TextureFormat::R8Uint
            | wgpu::TextureFormat::R8Sint => Some(1),

            // 2 bytes per pixel
            wgpu::TextureFormat::R16Uint
            | wgpu::TextureFormat::R16Sint
            | wgpu::TextureFormat::R16Unorm
            | wgpu::TextureFormat::R16Snorm
            | wgpu::TextureFormat::R16Float
            | wgpu::TextureFormat::Rg8Unorm
            | wgpu::TextureFormat::Rg8Snorm
            | wgpu::TextureFormat::Rg8Uint
            | wgpu::TextureFormat::Rg8Sint => Some(2),

            // 4 bytes per pixel
            wgpu::TextureFormat::R32Uint
            | wgpu::TextureFormat::R32Sint
            | wgpu::TextureFormat::R32Float
            | wgpu::TextureFormat::Rg16Uint
            | wgpu::TextureFormat::Rg16Sint
            | wgpu::TextureFormat::Rg16Unorm
            | wgpu::TextureFormat::Rg16Snorm
            | wgpu::TextureFormat::Rg16Float
            | wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Rgba8Snorm
            | wgpu::TextureFormat::Rgba8Uint
            | wgpu::TextureFormat::Rgba8Sint
            | wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Bgra8UnormSrgb
            | wgpu::TextureFormat::Rgb10a2Unorm
            | wgpu::TextureFormat::Rg11b10Float
            | wgpu::TextureFormat::Rgb10a2Uint => Some(4),

            // 8 bytes per pixel
            wgpu::TextureFormat::Rg32Uint
            | wgpu::TextureFormat::Rg32Sint
            | wgpu::TextureFormat::Rg32Float
            | wgpu::TextureFormat::Rgba16Uint
            | wgpu::TextureFormat::Rgba16Sint
            | wgpu::TextureFormat::Rgba16Unorm
            | wgpu::TextureFormat::Rgba16Snorm
            | wgpu::TextureFormat::Rgba16Float => Some(8),

            // 16 bytes per pixel
            wgpu::TextureFormat::Rgba32Uint
            | wgpu::TextureFormat::Rgba32Sint
            | wgpu::TextureFormat::Rgba32Float => Some(16),

            // Compressed/depth/stencil formats - not applicable
            _ => None,
        }
    }

    /// Check if this is an sRGB format.
    pub fn is_srgb(&self) -> bool {
        matches!(
            self.format,
            wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb
        )
    }

    /// Check if this is an HDR format (floating point).
    pub fn is_hdr(&self) -> bool {
        matches!(
            self.format,
            wgpu::TextureFormat::R16Float
                | wgpu::TextureFormat::Rg16Float
                | wgpu::TextureFormat::Rgba16Float
                | wgpu::TextureFormat::R32Float
                | wgpu::TextureFormat::Rg32Float
                | wgpu::TextureFormat::Rgba32Float
                | wgpu::TextureFormat::Rg11b10Float
        )
    }

    /// Validate the color target configuration.
    ///
    /// # Validation Rules
    ///
    /// - If blend is enabled, format must support blending
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, or a `ColorTargetError` describing the issue.
    pub fn validate(&self) -> Result<(), ColorTargetError> {
        if self.blend.is_some() && !self.format_supports_blend() {
            return Err(ColorTargetError::BlendNotSupported(self.format));
        }
        Ok(())
    }

    /// Check if the color target configuration is valid.
    pub fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }
}

// Thread-safety: ColorTarget contains only Copy/Clone types
unsafe impl Send for ColorTarget {}
unsafe impl Sync for ColorTarget {}

impl From<ColorTarget> for wgpu::ColorTargetState {
    fn from(target: ColorTarget) -> Self {
        wgpu::ColorTargetState {
            format: target.format,
            blend: target.blend,
            write_mask: target.write_mask,
        }
    }
}

impl From<&ColorTarget> for wgpu::ColorTargetState {
    fn from(target: &ColorTarget) -> Self {
        wgpu::ColorTargetState {
            format: target.format,
            blend: target.blend,
            write_mask: target.write_mask,
        }
    }
}

impl From<wgpu::ColorTargetState> for ColorTarget {
    fn from(state: wgpu::ColorTargetState) -> Self {
        Self {
            format: state.format,
            blend: state.blend,
            write_mask: state.write_mask,
        }
    }
}

impl From<wgpu::TextureFormat> for ColorTarget {
    fn from(format: wgpu::TextureFormat) -> Self {
        Self::new(format)
    }
}

// ---------------------------------------------------------------------------
// ColorTargetBuilder
// ---------------------------------------------------------------------------

/// Builder for creating color target configurations with fluent API.
///
/// # Example
///
/// ```ignore
/// let target = ColorTargetBuilder::new()
///     .format(wgpu::TextureFormat::Rgba16Float)
///     .alpha_blend()
///     .write_color()
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct ColorTargetBuilder {
    target: ColorTarget,
}

impl Default for ColorTargetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ColorTargetBuilder {
    /// Create a new builder with default values.
    pub fn new() -> Self {
        Self {
            target: ColorTarget::default(),
        }
    }

    /// Start building from an existing color target.
    pub fn from_target(target: ColorTarget) -> Self {
        Self { target }
    }

    /// Start building from a preset.
    pub fn from_preset(preset: ColorTarget) -> Self {
        Self { target: preset }
    }

    /// Set the texture format.
    pub fn format(mut self, format: wgpu::TextureFormat) -> Self {
        self.target.format = format;
        self
    }

    /// Set a custom blend state.
    pub fn blend(mut self, blend: wgpu::BlendState) -> Self {
        self.target.blend = Some(blend);
        self
    }

    /// Enable standard alpha blending.
    pub fn alpha_blend(mut self) -> Self {
        self.target = self.target.alpha_blend();
        self
    }

    /// Enable premultiplied alpha blending.
    pub fn premultiplied_alpha(mut self) -> Self {
        self.target = self.target.premultiplied_alpha();
        self
    }

    /// Enable additive blending.
    pub fn additive(mut self) -> Self {
        self.target = self.target.additive();
        self
    }

    /// Enable multiply blending.
    pub fn multiply(mut self) -> Self {
        self.target = self.target.multiply();
        self
    }

    /// Enable replace blending (overwrite).
    pub fn replace(mut self) -> Self {
        self.target = self.target.replace();
        self
    }

    /// Disable blending.
    pub fn no_blend(mut self) -> Self {
        self.target.blend = None;
        self
    }

    /// Set the color write mask.
    pub fn write_mask(mut self, mask: wgpu::ColorWrites) -> Self {
        self.target.write_mask = mask;
        self
    }

    /// Write all color channels.
    pub fn write_all(mut self) -> Self {
        self.target.write_mask = wgpu::ColorWrites::ALL;
        self
    }

    /// Write only RGB channels.
    pub fn write_color(mut self) -> Self {
        self.target.write_mask = wgpu::ColorWrites::COLOR;
        self
    }

    /// Write only red channel.
    pub fn write_red(mut self) -> Self {
        self.target.write_mask = wgpu::ColorWrites::RED;
        self
    }

    /// Write only alpha channel.
    pub fn write_alpha(mut self) -> Self {
        self.target.write_mask = wgpu::ColorWrites::ALPHA;
        self
    }

    /// Write no channels.
    pub fn write_none(mut self) -> Self {
        self.target.write_mask = wgpu::ColorWrites::empty();
        self
    }

    /// Build the color target with validation.
    ///
    /// # Returns
    ///
    /// `Ok(ColorTarget)` if valid, or an error describing the issue.
    pub fn build(self) -> Result<ColorTarget, ColorTargetError> {
        self.target.validate()?;
        Ok(self.target)
    }

    /// Build the color target without validation.
    pub fn build_unchecked(self) -> ColorTarget {
        self.target
    }
}

// ---------------------------------------------------------------------------
// ColorTargetArray
// ---------------------------------------------------------------------------

/// Builder for creating multiple render target (MRT) configurations.
///
/// Provides convenient methods for setting up common MRT patterns
/// like G-buffer rendering and HDR pipelines.
///
/// # Example
///
/// ```ignore
/// // G-buffer for deferred shading
/// let targets = ColorTargetArray::gbuffer()
///     .albedo(wgpu::TextureFormat::Rgba8Unorm)
///     .normal(wgpu::TextureFormat::Rgba16Float)
///     .material(wgpu::TextureFormat::Rgba8Unorm)
///     .build();
///
/// // HDR with bloom
/// let targets = ColorTargetArray::new()
///     .target(ColorTarget::rgba16_float())
///     .target(ColorTarget::rg11b10_float()) // Bloom brightness
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct ColorTargetArray {
    targets: Vec<Option<ColorTarget>>,
}

impl ColorTargetArray {
    /// Create a new empty color target array.
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
        }
    }

    /// Create an array configured for G-buffer rendering.
    ///
    /// Standard G-buffer layout:
    /// - Target 0: Albedo (RGBA8)
    /// - Target 1: Normal (RGBA16F)
    /// - Target 2: Material properties (RGBA8)
    pub fn gbuffer() -> Self {
        Self {
            targets: Vec::with_capacity(4),
        }
    }

    /// Add a color target.
    pub fn target(mut self, target: ColorTarget) -> Self {
        self.targets.push(Some(target));
        self
    }

    /// Add a color target from a format.
    pub fn target_format(mut self, format: wgpu::TextureFormat) -> Self {
        self.targets.push(Some(ColorTarget::new(format)));
        self
    }

    /// Add a null target (skip this attachment slot).
    pub fn null_target(mut self) -> Self {
        self.targets.push(None);
        self
    }

    /// Add an albedo target (for G-buffer).
    pub fn albedo(mut self, format: wgpu::TextureFormat) -> Self {
        self.targets.push(Some(ColorTarget::new(format)));
        self
    }

    /// Add a normal target (for G-buffer).
    pub fn normal(mut self, format: wgpu::TextureFormat) -> Self {
        self.targets.push(Some(ColorTarget::new(format)));
        self
    }

    /// Add a material properties target (for G-buffer).
    pub fn material(mut self, format: wgpu::TextureFormat) -> Self {
        self.targets.push(Some(ColorTarget::new(format)));
        self
    }

    /// Add a position/depth target (for G-buffer).
    pub fn position(mut self, format: wgpu::TextureFormat) -> Self {
        self.targets.push(Some(ColorTarget::new(format)));
        self
    }

    /// Add a velocity target (for motion blur/TAA).
    pub fn velocity(mut self, format: wgpu::TextureFormat) -> Self {
        self.targets.push(Some(ColorTarget::new(format)));
        self
    }

    /// Add a bloom brightness target.
    pub fn bloom(mut self, format: wgpu::TextureFormat) -> Self {
        self.targets.push(Some(ColorTarget::new(format)));
        self
    }

    /// Get the number of targets.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Check if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Build the color target array.
    pub fn build(self) -> Vec<Option<ColorTarget>> {
        self.targets
    }

    /// Build as wgpu ColorTargetState array.
    pub fn build_wgpu(self) -> Vec<Option<wgpu::ColorTargetState>> {
        self.targets
            .into_iter()
            .map(|opt| opt.map(|t| t.into()))
            .collect()
    }

    /// Validate all targets in the array.
    pub fn validate(&self) -> Result<(), ColorTargetError> {
        for (i, target) in self.targets.iter().enumerate() {
            if let Some(t) = target {
                t.validate().map_err(|e| ColorTargetError::ArrayError {
                    index: i,
                    error: Box::new(e),
                })?;
            }
        }
        if self.targets.len() > MAX_COLOR_ATTACHMENTS {
            return Err(ColorTargetError::TooManyTargets {
                count: self.targets.len(),
                max: MAX_COLOR_ATTACHMENTS,
            });
        }
        Ok(())
    }
}

/// Maximum number of color attachments supported by most GPUs.
pub const MAX_COLOR_ATTACHMENTS: usize = 8;

// ---------------------------------------------------------------------------
// ColorTargetInfo
// ---------------------------------------------------------------------------

/// Metadata about a color target preset configuration.
///
/// Provides descriptive information for tooling, debugging, and documentation.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorTargetInfo {
    /// Human-readable name for the preset.
    pub name: &'static str,
    /// Description of the color target configuration.
    pub description: &'static str,
    /// Typical use cases for this preset.
    pub use_cases: &'static [&'static str],
    /// The preset texture format.
    pub format: wgpu::TextureFormat,
    /// Bytes per pixel for this format.
    pub bytes_per_pixel: u32,
    /// Whether this is an sRGB format.
    pub is_srgb: bool,
    /// Whether this is an HDR format.
    pub is_hdr: bool,
}

/// Common color target configurations with documentation.
pub const COLOR_TARGET_PRESETS: [ColorTargetInfo; 12] = [
    ColorTargetInfo {
        name: "RGBA8 Unorm",
        description: "Standard 8-bit RGBA in linear color space",
        use_cases: &["intermediate buffers", "non-sRGB rendering", "G-buffer albedo"],
        format: wgpu::TextureFormat::Rgba8Unorm,
        bytes_per_pixel: 4,
        is_srgb: false,
        is_hdr: false,
    },
    ColorTargetInfo {
        name: "RGBA8 Unorm sRGB",
        description: "Standard 8-bit RGBA with sRGB gamma",
        use_cases: &["final output", "sRGB textures", "UI rendering"],
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        bytes_per_pixel: 4,
        is_srgb: true,
        is_hdr: false,
    },
    ColorTargetInfo {
        name: "BGRA8 Unorm",
        description: "8-bit BGRA in linear color space (swapchain compatible)",
        use_cases: &["swapchain target", "Windows/macOS output"],
        format: wgpu::TextureFormat::Bgra8Unorm,
        bytes_per_pixel: 4,
        is_srgb: false,
        is_hdr: false,
    },
    ColorTargetInfo {
        name: "BGRA8 Unorm sRGB",
        description: "8-bit BGRA with sRGB gamma (swapchain compatible)",
        use_cases: &["final swapchain output", "display rendering"],
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        bytes_per_pixel: 4,
        is_srgb: true,
        is_hdr: false,
    },
    ColorTargetInfo {
        name: "RGBA16 Float",
        description: "16-bit float RGBA for HDR rendering",
        use_cases: &["HDR rendering", "post-processing", "lighting accumulation"],
        format: wgpu::TextureFormat::Rgba16Float,
        bytes_per_pixel: 8,
        is_srgb: false,
        is_hdr: true,
    },
    ColorTargetInfo {
        name: "RGBA32 Float",
        description: "32-bit float RGBA for maximum precision",
        use_cases: &["scientific visualization", "precision-critical rendering"],
        format: wgpu::TextureFormat::Rgba32Float,
        bytes_per_pixel: 16,
        is_srgb: false,
        is_hdr: true,
    },
    ColorTargetInfo {
        name: "RGB10A2 Unorm",
        description: "10-bit RGB with 2-bit alpha",
        use_cases: &["HDR tone mapping output", "high color precision"],
        format: wgpu::TextureFormat::Rgb10a2Unorm,
        bytes_per_pixel: 4,
        is_srgb: false,
        is_hdr: false,
    },
    ColorTargetInfo {
        name: "RG11B10 Float",
        description: "Packed HDR RGB (no alpha)",
        use_cases: &["light accumulation", "HDR intermediate buffers", "bloom"],
        format: wgpu::TextureFormat::Rg11b10Float,
        bytes_per_pixel: 4,
        is_srgb: false,
        is_hdr: true,
    },
    ColorTargetInfo {
        name: "R8 Unorm",
        description: "Single channel 8-bit",
        use_cases: &["masks", "stencil copies", "luminance"],
        format: wgpu::TextureFormat::R8Unorm,
        bytes_per_pixel: 1,
        is_srgb: false,
        is_hdr: false,
    },
    ColorTargetInfo {
        name: "RG16 Float",
        description: "Two channel 16-bit float",
        use_cases: &["velocity buffers", "normal maps", "motion vectors"],
        format: wgpu::TextureFormat::Rg16Float,
        bytes_per_pixel: 4,
        is_srgb: false,
        is_hdr: true,
    },
    ColorTargetInfo {
        name: "R16 Float",
        description: "Single channel 16-bit float",
        use_cases: &["depth linearization", "single-channel HDR", "luminance"],
        format: wgpu::TextureFormat::R16Float,
        bytes_per_pixel: 2,
        is_srgb: false,
        is_hdr: true,
    },
    ColorTargetInfo {
        name: "R32 Float",
        description: "Single channel 32-bit float",
        use_cases: &["high precision depth", "distance fields", "raymarching"],
        format: wgpu::TextureFormat::R32Float,
        bytes_per_pixel: 4,
        is_srgb: false,
        is_hdr: true,
    },
];

/// Get color target info by preset name.
///
/// # Example
///
/// ```ignore
/// if let Some(info) = get_color_target_info("RGBA16 Float") {
///     println!("Use cases: {:?}", info.use_cases);
///     println!("Bytes per pixel: {}", info.bytes_per_pixel);
/// }
/// ```
pub fn get_color_target_info(name: &str) -> Option<&'static ColorTargetInfo> {
    COLOR_TARGET_PRESETS.iter().find(|info| info.name == name)
}

/// Get the color target preset by name.
pub fn get_preset(name: &str) -> Option<ColorTarget> {
    get_color_target_info(name).map(|info| ColorTarget::new(info.format))
}

/// List all available preset names.
pub fn preset_names() -> impl Iterator<Item = &'static str> {
    COLOR_TARGET_PRESETS.iter().map(|info| info.name)
}

/// Get all HDR format presets.
pub fn hdr_presets() -> impl Iterator<Item = &'static ColorTargetInfo> {
    COLOR_TARGET_PRESETS.iter().filter(|info| info.is_hdr)
}

/// Get all sRGB format presets.
pub fn srgb_presets() -> impl Iterator<Item = &'static ColorTargetInfo> {
    COLOR_TARGET_PRESETS.iter().filter(|info| info.is_srgb)
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during color target validation.
#[derive(Debug, Clone)]
pub enum ColorTargetError {
    /// Blend is enabled but format doesn't support it.
    BlendNotSupported(wgpu::TextureFormat),
    /// Too many color targets for the GPU.
    TooManyTargets { count: usize, max: usize },
    /// Error in a specific target of an array.
    ArrayError {
        index: usize,
        error: Box<ColorTargetError>,
    },
}

impl fmt::Display for ColorTargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColorTargetError::BlendNotSupported(format) => {
                write!(
                    f,
                    "Blend state not supported for format {:?} (integer formats cannot blend)",
                    format
                )
            }
            ColorTargetError::TooManyTargets { count, max } => {
                write!(
                    f,
                    "Too many color targets: {} (maximum is {})",
                    count, max
                )
            }
            ColorTargetError::ArrayError { index, error } => {
                write!(f, "Error in color target {}: {}", index, error)
            }
        }
    }
}

impl std::error::Error for ColorTargetError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ColorTarget Basic Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_color_target_default() {
        let target = ColorTarget::default();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba8UnormSrgb);
        assert!(target.blend.is_none());
        assert_eq!(target.write_mask, wgpu::ColorWrites::ALL);
    }

    #[test]
    fn test_color_target_new() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgba16Float);
        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
        assert!(target.blend.is_none());
        assert_eq!(target.write_mask, wgpu::ColorWrites::ALL);
    }

    // -------------------------------------------------------------------------
    // Format Preset Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rgba8_unorm() {
        let target = ColorTarget::rgba8_unorm();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba8Unorm);
        assert!(!target.is_srgb());
        assert!(!target.is_hdr());
    }

    #[test]
    fn test_rgba8_unorm_srgb() {
        let target = ColorTarget::rgba8_unorm_srgb();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba8UnormSrgb);
        assert!(target.is_srgb());
        assert!(!target.is_hdr());
    }

    #[test]
    fn test_bgra8_unorm() {
        let target = ColorTarget::bgra8_unorm();
        assert_eq!(target.format, wgpu::TextureFormat::Bgra8Unorm);
    }

    #[test]
    fn test_bgra8_unorm_srgb() {
        let target = ColorTarget::bgra8_unorm_srgb();
        assert_eq!(target.format, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert!(target.is_srgb());
    }

    #[test]
    fn test_rgba16_float() {
        let target = ColorTarget::rgba16_float();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
        assert!(target.is_hdr());
        assert!(!target.is_srgb());
    }

    #[test]
    fn test_rgba32_float() {
        let target = ColorTarget::rgba32_float();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba32Float);
        assert!(target.is_hdr());
    }

    #[test]
    fn test_rgb10a2_unorm() {
        let target = ColorTarget::rgb10a2_unorm();
        assert_eq!(target.format, wgpu::TextureFormat::Rgb10a2Unorm);
    }

    #[test]
    fn test_rg11b10_float() {
        let target = ColorTarget::rg11b10_float();
        assert_eq!(target.format, wgpu::TextureFormat::Rg11b10Float);
        assert!(target.is_hdr());
    }

    #[test]
    fn test_r8_unorm() {
        let target = ColorTarget::r8_unorm();
        assert_eq!(target.format, wgpu::TextureFormat::R8Unorm);
    }

    #[test]
    fn test_rg8_unorm() {
        let target = ColorTarget::rg8_unorm();
        assert_eq!(target.format, wgpu::TextureFormat::Rg8Unorm);
    }

    #[test]
    fn test_r16_float() {
        let target = ColorTarget::r16_float();
        assert_eq!(target.format, wgpu::TextureFormat::R16Float);
        assert!(target.is_hdr());
    }

    #[test]
    fn test_rg16_float() {
        let target = ColorTarget::rg16_float();
        assert_eq!(target.format, wgpu::TextureFormat::Rg16Float);
        assert!(target.is_hdr());
    }

    #[test]
    fn test_r32_float() {
        let target = ColorTarget::r32_float();
        assert_eq!(target.format, wgpu::TextureFormat::R32Float);
        assert!(target.is_hdr());
    }

    #[test]
    fn test_rg32_float() {
        let target = ColorTarget::rg32_float();
        assert_eq!(target.format, wgpu::TextureFormat::Rg32Float);
        assert!(target.is_hdr());
    }

    // -------------------------------------------------------------------------
    // Blend State Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_alpha_blend() {
        let target = ColorTarget::rgba8_unorm().alpha_blend();
        assert!(target.blend.is_some());
        let blend = target.blend.unwrap();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    }

    #[test]
    fn test_premultiplied_alpha() {
        let target = ColorTarget::rgba8_unorm().premultiplied_alpha();
        assert!(target.blend.is_some());
        let blend = target.blend.unwrap();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    }

    #[test]
    fn test_additive_blend() {
        let target = ColorTarget::rgba8_unorm().additive();
        assert!(target.blend.is_some());
        let blend = target.blend.unwrap();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_multiply_blend() {
        let target = ColorTarget::rgba8_unorm().multiply();
        assert!(target.blend.is_some());
        let blend = target.blend.unwrap();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::Dst);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::Zero);
    }

    #[test]
    fn test_replace_blend() {
        let target = ColorTarget::rgba8_unorm().replace();
        assert!(target.blend.is_some());
    }

    #[test]
    fn test_no_blend() {
        let target = ColorTarget::rgba8_unorm().alpha_blend().no_blend();
        assert!(target.blend.is_none());
    }

    #[test]
    fn test_custom_blend() {
        let custom = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Constant,
                dst_factor: wgpu::BlendFactor::OneMinusConstant,
                operation: wgpu::BlendOperation::Max,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let target = ColorTarget::rgba8_unorm().blend(custom);
        assert!(target.blend.is_some());
        let blend = target.blend.unwrap();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::Constant);
    }

    // -------------------------------------------------------------------------
    // Write Mask Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_write_mask_all() {
        let target = ColorTarget::rgba8_unorm().write_all();
        assert_eq!(target.write_mask, wgpu::ColorWrites::ALL);
        assert!(target.writes_all());
    }

    #[test]
    fn test_write_mask_color() {
        let target = ColorTarget::rgba8_unorm().write_color();
        assert_eq!(target.write_mask, wgpu::ColorWrites::COLOR);
        assert!(target.write_mask.contains(wgpu::ColorWrites::RED));
        assert!(target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(target.write_mask.contains(wgpu::ColorWrites::BLUE));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::ALPHA));
    }

    #[test]
    fn test_write_mask_red() {
        let target = ColorTarget::rgba8_unorm().write_red();
        assert_eq!(target.write_mask, wgpu::ColorWrites::RED);
    }

    #[test]
    fn test_write_mask_green() {
        let target = ColorTarget::rgba8_unorm().write_green();
        assert_eq!(target.write_mask, wgpu::ColorWrites::GREEN);
    }

    #[test]
    fn test_write_mask_blue() {
        let target = ColorTarget::rgba8_unorm().write_blue();
        assert_eq!(target.write_mask, wgpu::ColorWrites::BLUE);
    }

    #[test]
    fn test_write_mask_alpha() {
        let target = ColorTarget::rgba8_unorm().write_alpha();
        assert_eq!(target.write_mask, wgpu::ColorWrites::ALPHA);
    }

    #[test]
    fn test_write_mask_none() {
        let target = ColorTarget::rgba8_unorm().write_none();
        assert_eq!(target.write_mask, wgpu::ColorWrites::empty());
        assert!(!target.writes_any());
    }

    #[test]
    fn test_write_mask_custom() {
        let target = ColorTarget::rgba8_unorm()
            .write_mask(wgpu::ColorWrites::RED | wgpu::ColorWrites::BLUE);
        assert!(target.write_mask.contains(wgpu::ColorWrites::RED));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(target.write_mask.contains(wgpu::ColorWrites::BLUE));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::ALPHA));
    }

    // -------------------------------------------------------------------------
    // Query Method Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_has_blend() {
        let no_blend = ColorTarget::rgba8_unorm();
        let with_blend = ColorTarget::rgba8_unorm().alpha_blend();
        assert!(!no_blend.has_blend());
        assert!(with_blend.has_blend());
    }

    #[test]
    fn test_format_supports_blend() {
        let rgba = ColorTarget::rgba8_unorm();
        assert!(rgba.format_supports_blend());

        let uint = ColorTarget::new(wgpu::TextureFormat::Rgba8Uint);
        assert!(!uint.format_supports_blend());

        let sint = ColorTarget::new(wgpu::TextureFormat::Rgba16Sint);
        assert!(!sint.format_supports_blend());
    }

    #[test]
    fn test_bytes_per_pixel() {
        assert_eq!(ColorTarget::r8_unorm().bytes_per_pixel(), Some(1));
        assert_eq!(ColorTarget::rg8_unorm().bytes_per_pixel(), Some(2));
        assert_eq!(ColorTarget::rgba8_unorm().bytes_per_pixel(), Some(4));
        assert_eq!(ColorTarget::rgba16_float().bytes_per_pixel(), Some(8));
        assert_eq!(ColorTarget::rgba32_float().bytes_per_pixel(), Some(16));
    }

    // -------------------------------------------------------------------------
    // Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_valid() {
        let target = ColorTarget::rgba8_unorm().alpha_blend();
        assert!(target.validate().is_ok());
        assert!(target.is_valid());
    }

    #[test]
    fn test_validate_blend_not_supported() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgba8Uint).alpha_blend();
        assert!(matches!(
            target.validate(),
            Err(ColorTargetError::BlendNotSupported(_))
        ));
        assert!(!target.is_valid());
    }

    // -------------------------------------------------------------------------
    // Conversion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_into_wgpu() {
        let target = ColorTarget::rgba16_float().alpha_blend().write_color();
        let wgpu_state: wgpu::ColorTargetState = target.into();
        assert_eq!(wgpu_state.format, wgpu::TextureFormat::Rgba16Float);
        assert!(wgpu_state.blend.is_some());
        assert_eq!(wgpu_state.write_mask, wgpu::ColorWrites::COLOR);
    }

    #[test]
    fn test_from_wgpu() {
        let wgpu_state = wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        };
        let target: ColorTarget = wgpu_state.into();
        assert_eq!(target.format, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert!(target.blend.is_some());
    }

    #[test]
    fn test_from_format() {
        let target: ColorTarget = wgpu::TextureFormat::Rgba16Float.into();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_default() {
        let target = ColorTargetBuilder::new().build_unchecked();
        assert_eq!(target, ColorTarget::default());
    }

    #[test]
    fn test_builder_format() {
        let target = ColorTargetBuilder::new()
            .format(wgpu::TextureFormat::Rgba16Float)
            .build_unchecked();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn test_builder_blend() {
        let target = ColorTargetBuilder::new()
            .alpha_blend()
            .build_unchecked();
        assert!(target.blend.is_some());
    }

    #[test]
    fn test_builder_write_mask() {
        let target = ColorTargetBuilder::new()
            .write_color()
            .build_unchecked();
        assert_eq!(target.write_mask, wgpu::ColorWrites::COLOR);
    }

    #[test]
    fn test_builder_chained() {
        let target = ColorTargetBuilder::new()
            .format(wgpu::TextureFormat::Rgba16Float)
            .premultiplied_alpha()
            .write_color()
            .build()
            .unwrap();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
        assert!(target.blend.is_some());
        assert_eq!(target.write_mask, wgpu::ColorWrites::COLOR);
    }

    #[test]
    fn test_builder_from_preset() {
        let target = ColorTargetBuilder::from_preset(ColorTarget::rgba16_float())
            .alpha_blend()
            .build_unchecked();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
        assert!(target.blend.is_some());
    }

    #[test]
    fn test_builder_validation() {
        let result = ColorTargetBuilder::new()
            .format(wgpu::TextureFormat::Rgba8Uint)
            .alpha_blend()
            .build();
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // ColorTargetArray Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_array_empty() {
        let array = ColorTargetArray::new();
        assert!(array.is_empty());
        assert_eq!(array.len(), 0);
    }

    #[test]
    fn test_array_single_target() {
        let targets = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm())
            .build();
        assert_eq!(targets.len(), 1);
        assert!(targets[0].is_some());
    }

    #[test]
    fn test_array_multiple_targets() {
        let targets = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm())
            .target(ColorTarget::rgba16_float())
            .target(ColorTarget::rg16_float())
            .build();
        assert_eq!(targets.len(), 3);
    }

    #[test]
    fn test_array_with_null() {
        let targets = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm())
            .null_target()
            .target(ColorTarget::rgba16_float())
            .build();
        assert_eq!(targets.len(), 3);
        assert!(targets[0].is_some());
        assert!(targets[1].is_none());
        assert!(targets[2].is_some());
    }

    #[test]
    fn test_array_gbuffer() {
        let targets = ColorTargetArray::gbuffer()
            .albedo(wgpu::TextureFormat::Rgba8Unorm)
            .normal(wgpu::TextureFormat::Rgba16Float)
            .material(wgpu::TextureFormat::Rgba8Unorm)
            .position(wgpu::TextureFormat::Rgba32Float)
            .build();
        assert_eq!(targets.len(), 4);
        assert_eq!(targets[0].as_ref().unwrap().format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(targets[1].as_ref().unwrap().format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(targets[2].as_ref().unwrap().format, wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(targets[3].as_ref().unwrap().format, wgpu::TextureFormat::Rgba32Float);
    }

    #[test]
    fn test_array_build_wgpu() {
        let targets = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm().alpha_blend())
            .target(ColorTarget::rgba16_float())
            .build_wgpu();
        assert_eq!(targets.len(), 2);
        assert!(targets[0].as_ref().unwrap().blend.is_some());
        assert!(targets[1].as_ref().unwrap().blend.is_none());
    }

    #[test]
    fn test_array_validation() {
        let array = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm());
        assert!(array.validate().is_ok());
    }

    #[test]
    fn test_array_too_many_targets() {
        let mut array = ColorTargetArray::new();
        for _ in 0..10 {
            array = array.target(ColorTarget::rgba8_unorm());
        }
        assert!(matches!(
            array.validate(),
            Err(ColorTargetError::TooManyTargets { .. })
        ));
    }

    // -------------------------------------------------------------------------
    // ColorTargetInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_color_target_info() {
        let info = get_color_target_info("RGBA16 Float");
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.format, wgpu::TextureFormat::Rgba16Float);
        assert!(info.is_hdr);
        assert_eq!(info.bytes_per_pixel, 8);
    }

    #[test]
    fn test_get_color_target_info_not_found() {
        let info = get_color_target_info("NonExistent");
        assert!(info.is_none());
    }

    #[test]
    fn test_get_preset() {
        let target = get_preset("BGRA8 Unorm sRGB");
        assert!(target.is_some());
        assert_eq!(target.unwrap().format, wgpu::TextureFormat::Bgra8UnormSrgb);
    }

    #[test]
    fn test_preset_names() {
        let names: Vec<_> = preset_names().collect();
        assert!(names.contains(&"RGBA8 Unorm"));
        assert!(names.contains(&"RGBA16 Float"));
        assert!(names.contains(&"BGRA8 Unorm sRGB"));
    }

    #[test]
    fn test_hdr_presets() {
        let hdr: Vec<_> = hdr_presets().collect();
        assert!(!hdr.is_empty());
        for info in &hdr {
            assert!(info.is_hdr);
        }
    }

    #[test]
    fn test_srgb_presets() {
        let srgb: Vec<_> = srgb_presets().collect();
        assert!(!srgb.is_empty());
        for info in &srgb {
            assert!(info.is_srgb);
        }
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<ColorTarget>();
        assert_sync::<ColorTarget>();
    }

    // -------------------------------------------------------------------------
    // Clone and Equality Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_clone() {
        let original = ColorTarget::rgba16_float().alpha_blend().write_color();
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_equality() {
        let a = ColorTarget::rgba8_unorm();
        let b = ColorTarget::rgba8_unorm();
        let c = ColorTarget::rgba16_float();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_debug() {
        let target = ColorTarget::rgba8_unorm().alpha_blend();
        let debug_str = format!("{:?}", target);
        assert!(debug_str.contains("Rgba8Unorm"));
    }

    // -------------------------------------------------------------------------
    // Error Display Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_display_blend_not_supported() {
        let err = ColorTargetError::BlendNotSupported(wgpu::TextureFormat::Rgba8Uint);
        let msg = format!("{}", err);
        assert!(msg.contains("Blend state not supported"));
        assert!(msg.contains("Rgba8Uint"));
    }

    #[test]
    fn test_error_display_too_many_targets() {
        let err = ColorTargetError::TooManyTargets { count: 10, max: 8 };
        let msg = format!("{}", err);
        assert!(msg.contains("Too many color targets"));
        assert!(msg.contains("10"));
        assert!(msg.contains("8"));
    }

    // =========================================================================
    // WHITEBOX TESTS - Additional Edge Cases and Internal Coverage
    // =========================================================================

    // -------------------------------------------------------------------------
    // Additional TextureFormat Variant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_r8_snorm() {
        let target = ColorTarget::new(wgpu::TextureFormat::R8Snorm);
        assert_eq!(target.bytes_per_pixel(), Some(1));
        assert!(target.format_supports_blend());
        assert!(!target.is_hdr());
        assert!(!target.is_srgb());
    }

    #[test]
    fn test_format_r8_uint() {
        let target = ColorTarget::new(wgpu::TextureFormat::R8Uint);
        assert_eq!(target.bytes_per_pixel(), Some(1));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_r8_sint() {
        let target = ColorTarget::new(wgpu::TextureFormat::R8Sint);
        assert_eq!(target.bytes_per_pixel(), Some(1));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_r16_uint() {
        let target = ColorTarget::new(wgpu::TextureFormat::R16Uint);
        assert_eq!(target.bytes_per_pixel(), Some(2));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_r16_sint() {
        let target = ColorTarget::new(wgpu::TextureFormat::R16Sint);
        assert_eq!(target.bytes_per_pixel(), Some(2));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_r16_unorm() {
        let target = ColorTarget::new(wgpu::TextureFormat::R16Unorm);
        assert_eq!(target.bytes_per_pixel(), Some(2));
        assert!(target.format_supports_blend());
    }

    #[test]
    fn test_format_r16_snorm() {
        let target = ColorTarget::new(wgpu::TextureFormat::R16Snorm);
        assert_eq!(target.bytes_per_pixel(), Some(2));
        assert!(target.format_supports_blend());
    }

    #[test]
    fn test_format_rg8_snorm() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rg8Snorm);
        assert_eq!(target.bytes_per_pixel(), Some(2));
        assert!(target.format_supports_blend());
    }

    #[test]
    fn test_format_rg8_uint() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rg8Uint);
        assert_eq!(target.bytes_per_pixel(), Some(2));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_rg8_sint() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rg8Sint);
        assert_eq!(target.bytes_per_pixel(), Some(2));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_r32_uint() {
        let target = ColorTarget::new(wgpu::TextureFormat::R32Uint);
        assert_eq!(target.bytes_per_pixel(), Some(4));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_r32_sint() {
        let target = ColorTarget::new(wgpu::TextureFormat::R32Sint);
        assert_eq!(target.bytes_per_pixel(), Some(4));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_rg16_uint() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rg16Uint);
        assert_eq!(target.bytes_per_pixel(), Some(4));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_rg16_sint() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rg16Sint);
        assert_eq!(target.bytes_per_pixel(), Some(4));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_rg16_unorm() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rg16Unorm);
        assert_eq!(target.bytes_per_pixel(), Some(4));
        assert!(target.format_supports_blend());
    }

    #[test]
    fn test_format_rg16_snorm() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rg16Snorm);
        assert_eq!(target.bytes_per_pixel(), Some(4));
        assert!(target.format_supports_blend());
    }

    #[test]
    fn test_format_rgba8_snorm() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgba8Snorm);
        assert_eq!(target.bytes_per_pixel(), Some(4));
        assert!(target.format_supports_blend());
    }

    #[test]
    fn test_format_rgba8_uint() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgba8Uint);
        assert_eq!(target.bytes_per_pixel(), Some(4));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_rgba8_sint() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgba8Sint);
        assert_eq!(target.bytes_per_pixel(), Some(4));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_rgb10a2_uint() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgb10a2Uint);
        assert_eq!(target.bytes_per_pixel(), Some(4));
        // Rgb10a2Uint is not in the non-blendable list, check if it supports blend
        assert!(target.format_supports_blend());
    }

    #[test]
    fn test_format_rg32_uint() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rg32Uint);
        assert_eq!(target.bytes_per_pixel(), Some(8));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_rg32_sint() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rg32Sint);
        assert_eq!(target.bytes_per_pixel(), Some(8));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_rgba16_uint() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgba16Uint);
        assert_eq!(target.bytes_per_pixel(), Some(8));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_rgba16_sint() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgba16Sint);
        assert_eq!(target.bytes_per_pixel(), Some(8));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_rgba16_unorm() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgba16Unorm);
        assert_eq!(target.bytes_per_pixel(), Some(8));
        assert!(target.format_supports_blend());
    }

    #[test]
    fn test_format_rgba16_snorm() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgba16Snorm);
        assert_eq!(target.bytes_per_pixel(), Some(8));
        assert!(target.format_supports_blend());
    }

    #[test]
    fn test_format_rgba32_uint() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgba32Uint);
        assert_eq!(target.bytes_per_pixel(), Some(16));
        assert!(!target.format_supports_blend());
    }

    #[test]
    fn test_format_rgba32_sint() {
        let target = ColorTarget::new(wgpu::TextureFormat::Rgba32Sint);
        assert_eq!(target.bytes_per_pixel(), Some(16));
        assert!(!target.format_supports_blend());
    }

    // -------------------------------------------------------------------------
    // Additional Blend Mode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blend_operation_add() {
        let target = ColorTarget::rgba8_unorm().additive();
        let blend = target.blend.unwrap();
        assert_eq!(blend.color.operation, wgpu::BlendOperation::Add);
        assert_eq!(blend.alpha.operation, wgpu::BlendOperation::Add);
    }

    #[test]
    fn test_blend_multiply_alpha_component() {
        let target = ColorTarget::rgba8_unorm().multiply();
        let blend = target.blend.unwrap();
        assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::DstAlpha);
        assert_eq!(blend.alpha.dst_factor, wgpu::BlendFactor::Zero);
    }

    #[test]
    fn test_blend_replace_is_replace() {
        let target = ColorTarget::rgba8_unorm().replace();
        let blend = target.blend.unwrap();
        assert_eq!(blend, wgpu::BlendState::REPLACE);
    }

    #[test]
    fn test_blend_opt_some() {
        let target = ColorTarget::rgba8_unorm()
            .blend_opt(Some(wgpu::BlendState::ALPHA_BLENDING));
        assert!(target.blend.is_some());
    }

    #[test]
    fn test_blend_opt_none() {
        let target = ColorTarget::rgba8_unorm()
            .alpha_blend()
            .blend_opt(None);
        assert!(target.blend.is_none());
    }

    #[test]
    fn test_alpha_blend_alpha_component() {
        let target = ColorTarget::rgba8_unorm().alpha_blend();
        let blend = target.blend.unwrap();
        // Standard alpha blending alpha component
        assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(blend.alpha.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
    }

    #[test]
    fn test_premultiplied_alpha_full_check() {
        let target = ColorTarget::rgba8_unorm().premultiplied_alpha();
        let blend = target.blend.unwrap();
        // Premultiplied alpha blending
        assert_eq!(blend, wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
    }

    // -------------------------------------------------------------------------
    // Write Mask Combination Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_write_mask_rg() {
        let target = ColorTarget::rgba8_unorm()
            .write_mask(wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN);
        assert!(target.write_mask.contains(wgpu::ColorWrites::RED));
        assert!(target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::BLUE));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::ALPHA));
        assert!(target.writes_any());
        assert!(!target.writes_all());
    }

    #[test]
    fn test_write_mask_rb() {
        let target = ColorTarget::rgba8_unorm()
            .write_mask(wgpu::ColorWrites::RED | wgpu::ColorWrites::BLUE);
        assert!(target.write_mask.contains(wgpu::ColorWrites::RED));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(target.write_mask.contains(wgpu::ColorWrites::BLUE));
    }

    #[test]
    fn test_write_mask_ra() {
        let target = ColorTarget::rgba8_unorm()
            .write_mask(wgpu::ColorWrites::RED | wgpu::ColorWrites::ALPHA);
        assert!(target.write_mask.contains(wgpu::ColorWrites::RED));
        assert!(target.write_mask.contains(wgpu::ColorWrites::ALPHA));
    }

    #[test]
    fn test_write_mask_gb() {
        let target = ColorTarget::rgba8_unorm()
            .write_mask(wgpu::ColorWrites::GREEN | wgpu::ColorWrites::BLUE);
        assert!(!target.write_mask.contains(wgpu::ColorWrites::RED));
        assert!(target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(target.write_mask.contains(wgpu::ColorWrites::BLUE));
    }

    #[test]
    fn test_write_mask_ga() {
        let target = ColorTarget::rgba8_unorm()
            .write_mask(wgpu::ColorWrites::GREEN | wgpu::ColorWrites::ALPHA);
        assert!(target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(target.write_mask.contains(wgpu::ColorWrites::ALPHA));
    }

    #[test]
    fn test_write_mask_ba() {
        let target = ColorTarget::rgba8_unorm()
            .write_mask(wgpu::ColorWrites::BLUE | wgpu::ColorWrites::ALPHA);
        assert!(target.write_mask.contains(wgpu::ColorWrites::BLUE));
        assert!(target.write_mask.contains(wgpu::ColorWrites::ALPHA));
    }

    #[test]
    fn test_write_mask_rgb() {
        let target = ColorTarget::rgba8_unorm()
            .write_mask(wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN | wgpu::ColorWrites::BLUE);
        assert_eq!(target.write_mask, wgpu::ColorWrites::COLOR);
    }

    #[test]
    fn test_write_mask_rga() {
        let target = ColorTarget::rgba8_unorm()
            .write_mask(wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN | wgpu::ColorWrites::ALPHA);
        assert!(target.write_mask.contains(wgpu::ColorWrites::RED));
        assert!(target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::BLUE));
        assert!(target.write_mask.contains(wgpu::ColorWrites::ALPHA));
    }

    #[test]
    fn test_write_mask_rba() {
        let target = ColorTarget::rgba8_unorm()
            .write_mask(wgpu::ColorWrites::RED | wgpu::ColorWrites::BLUE | wgpu::ColorWrites::ALPHA);
        assert!(target.write_mask.contains(wgpu::ColorWrites::RED));
        assert!(!target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(target.write_mask.contains(wgpu::ColorWrites::BLUE));
        assert!(target.write_mask.contains(wgpu::ColorWrites::ALPHA));
    }

    #[test]
    fn test_write_mask_gba() {
        let target = ColorTarget::rgba8_unorm()
            .write_mask(wgpu::ColorWrites::GREEN | wgpu::ColorWrites::BLUE | wgpu::ColorWrites::ALPHA);
        assert!(!target.write_mask.contains(wgpu::ColorWrites::RED));
        assert!(target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(target.write_mask.contains(wgpu::ColorWrites::BLUE));
        assert!(target.write_mask.contains(wgpu::ColorWrites::ALPHA));
    }

    // -------------------------------------------------------------------------
    // Builder Pattern Chain Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_all_blend_modes() {
        // Test all blend methods can be chained
        let target = ColorTargetBuilder::new()
            .format(wgpu::TextureFormat::Rgba8Unorm)
            .alpha_blend()
            .no_blend()
            .premultiplied_alpha()
            .no_blend()
            .additive()
            .no_blend()
            .multiply()
            .no_blend()
            .replace()
            .build_unchecked();
        // Final state should be replace
        assert!(target.blend.is_some());
    }

    #[test]
    fn test_builder_all_write_masks() {
        // Test all write mask methods can be chained
        let target = ColorTargetBuilder::new()
            .write_all()
            .write_color()
            .write_red()
            .write_alpha()
            .write_none()
            .write_all()
            .build_unchecked();
        assert_eq!(target.write_mask, wgpu::ColorWrites::ALL);
    }

    #[test]
    fn test_builder_from_target() {
        let original = ColorTarget::rgba16_float().alpha_blend().write_color();
        let target = ColorTargetBuilder::from_target(original)
            .no_blend()
            .build_unchecked();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
        assert!(target.blend.is_none());
        assert_eq!(target.write_mask, wgpu::ColorWrites::COLOR);
    }

    #[test]
    fn test_builder_default_impl() {
        let builder = ColorTargetBuilder::default();
        let target = builder.build_unchecked();
        assert_eq!(target, ColorTarget::default());
    }

    #[test]
    fn test_builder_custom_blend() {
        let custom_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Src,
                dst_factor: wgpu::BlendFactor::Dst,
                operation: wgpu::BlendOperation::Subtract,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::ReverseSubtract,
            },
        };
        let target = ColorTargetBuilder::new()
            .blend(custom_blend)
            .build_unchecked();
        let blend = target.blend.unwrap();
        assert_eq!(blend.color.operation, wgpu::BlendOperation::Subtract);
        assert_eq!(blend.alpha.operation, wgpu::BlendOperation::ReverseSubtract);
    }

    #[test]
    fn test_builder_write_mask_custom() {
        let target = ColorTargetBuilder::new()
            .write_mask(wgpu::ColorWrites::GREEN | wgpu::ColorWrites::ALPHA)
            .build_unchecked();
        assert!(target.write_mask.contains(wgpu::ColorWrites::GREEN));
        assert!(target.write_mask.contains(wgpu::ColorWrites::ALPHA));
    }

    // -------------------------------------------------------------------------
    // ColorTargetInfo Metadata Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_preset_info_rgba8_unorm_use_cases() {
        let info = get_color_target_info("RGBA8 Unorm").unwrap();
        assert!(info.use_cases.len() > 0);
        assert!(info.use_cases.contains(&"G-buffer albedo"));
    }

    #[test]
    fn test_preset_info_rgba8_unorm_srgb_use_cases() {
        let info = get_color_target_info("RGBA8 Unorm sRGB").unwrap();
        assert!(info.use_cases.contains(&"final output"));
        assert!(info.use_cases.contains(&"UI rendering"));
    }

    #[test]
    fn test_preset_info_rgba16_float_use_cases() {
        let info = get_color_target_info("RGBA16 Float").unwrap();
        assert!(info.use_cases.contains(&"HDR rendering"));
        assert!(info.use_cases.contains(&"post-processing"));
    }

    #[test]
    fn test_preset_info_rg16_float_use_cases() {
        let info = get_color_target_info("RG16 Float").unwrap();
        assert!(info.use_cases.contains(&"velocity buffers"));
        assert!(info.use_cases.contains(&"motion vectors"));
    }

    #[test]
    fn test_preset_info_all_have_descriptions() {
        for info in COLOR_TARGET_PRESETS.iter() {
            assert!(!info.description.is_empty(), "Preset {} missing description", info.name);
            assert!(!info.name.is_empty(), "Preset has empty name");
        }
    }

    #[test]
    fn test_preset_info_bytes_match() {
        for info in COLOR_TARGET_PRESETS.iter() {
            let target = ColorTarget::new(info.format);
            if let Some(bpp) = target.bytes_per_pixel() {
                assert_eq!(
                    bpp, info.bytes_per_pixel,
                    "Bytes per pixel mismatch for {}",
                    info.name
                );
            }
        }
    }

    #[test]
    fn test_preset_info_srgb_match() {
        for info in COLOR_TARGET_PRESETS.iter() {
            let target = ColorTarget::new(info.format);
            assert_eq!(
                target.is_srgb(), info.is_srgb,
                "sRGB flag mismatch for {}",
                info.name
            );
        }
    }

    #[test]
    fn test_preset_info_hdr_match() {
        for info in COLOR_TARGET_PRESETS.iter() {
            let target = ColorTarget::new(info.format);
            assert_eq!(
                target.is_hdr(), info.is_hdr,
                "HDR flag mismatch for {}",
                info.name
            );
        }
    }

    #[test]
    fn test_preset_names_count() {
        let names: Vec<_> = preset_names().collect();
        assert_eq!(names.len(), COLOR_TARGET_PRESETS.len());
    }

    #[test]
    fn test_all_presets_unique_names() {
        let names: Vec<_> = preset_names().collect();
        let mut unique_names = names.clone();
        unique_names.sort();
        unique_names.dedup();
        assert_eq!(names.len(), unique_names.len(), "Duplicate preset names found");
    }

    // -------------------------------------------------------------------------
    // HDR Format Preset Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hdr_presets_count() {
        let hdr: Vec<_> = hdr_presets().collect();
        // Should have: R16Float, Rg16Float, Rgba16Float, R32Float, Rg32Float, Rgba32Float, Rg11b10Float
        assert!(hdr.len() >= 6);
    }

    #[test]
    fn test_hdr_presets_all_float() {
        for info in hdr_presets() {
            let format_name = format!("{:?}", info.format);
            assert!(
                format_name.contains("Float"),
                "HDR preset {} is not a float format",
                info.name
            );
        }
    }

    #[test]
    fn test_srgb_presets_count() {
        let srgb: Vec<_> = srgb_presets().collect();
        // Should have: Rgba8UnormSrgb, Bgra8UnormSrgb
        assert_eq!(srgb.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Per-Target Array Configuration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_array_target_format() {
        let targets = ColorTargetArray::new()
            .target_format(wgpu::TextureFormat::Rgba16Float)
            .target_format(wgpu::TextureFormat::R8Unorm)
            .build();
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0].as_ref().unwrap().format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(targets[1].as_ref().unwrap().format, wgpu::TextureFormat::R8Unorm);
    }

    #[test]
    fn test_array_velocity_target() {
        let targets = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm())
            .velocity(wgpu::TextureFormat::Rg16Float)
            .build();
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[1].as_ref().unwrap().format, wgpu::TextureFormat::Rg16Float);
    }

    #[test]
    fn test_array_bloom_target() {
        let targets = ColorTargetArray::new()
            .target(ColorTarget::rgba16_float())
            .bloom(wgpu::TextureFormat::Rg11b10Float)
            .build();
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[1].as_ref().unwrap().format, wgpu::TextureFormat::Rg11b10Float);
    }

    #[test]
    fn test_array_max_attachments() {
        let mut array = ColorTargetArray::new();
        for _ in 0..MAX_COLOR_ATTACHMENTS {
            array = array.target(ColorTarget::rgba8_unorm());
        }
        assert_eq!(array.len(), MAX_COLOR_ATTACHMENTS);
        assert!(array.validate().is_ok());
    }

    #[test]
    fn test_array_exactly_max_targets() {
        let mut array = ColorTargetArray::new();
        for _ in 0..8 {
            array = array.target(ColorTarget::rgba8_unorm());
        }
        assert_eq!(array.len(), 8);
        assert!(array.validate().is_ok());
    }

    #[test]
    fn test_array_one_over_max() {
        let mut array = ColorTargetArray::new();
        for _ in 0..9 {
            array = array.target(ColorTarget::rgba8_unorm());
        }
        assert!(matches!(
            array.validate(),
            Err(ColorTargetError::TooManyTargets { count: 9, max: 8 })
        ));
    }

    #[test]
    fn test_array_validation_with_invalid_target() {
        let array = ColorTargetArray::new()
            .target(ColorTarget::new(wgpu::TextureFormat::Rgba8Uint).alpha_blend());
        let result = array.validate();
        assert!(matches!(
            result,
            Err(ColorTargetError::ArrayError { index: 0, .. })
        ));
    }

    #[test]
    fn test_array_validation_second_invalid() {
        let array = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm())
            .target(ColorTarget::new(wgpu::TextureFormat::Rgba16Sint).alpha_blend());
        let result = array.validate();
        assert!(matches!(
            result,
            Err(ColorTargetError::ArrayError { index: 1, .. })
        ));
    }

    #[test]
    fn test_array_mixed_null_and_valid() {
        let targets = ColorTargetArray::new()
            .null_target()
            .target(ColorTarget::rgba8_unorm())
            .null_target()
            .null_target()
            .target(ColorTarget::rgba16_float())
            .build();
        assert_eq!(targets.len(), 5);
        assert!(targets[0].is_none());
        assert!(targets[1].is_some());
        assert!(targets[2].is_none());
        assert!(targets[3].is_none());
        assert!(targets[4].is_some());
    }

    // -------------------------------------------------------------------------
    // Display/Debug Trait Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_color_target_debug_format() {
        let target = ColorTarget::rgba16_float().alpha_blend().write_color();
        let debug_str = format!("{:?}", target);
        assert!(debug_str.contains("Rgba16Float"));
        assert!(debug_str.contains("blend"));
        assert!(debug_str.contains("write_mask"));
    }

    #[test]
    fn test_color_target_debug_no_blend() {
        let target = ColorTarget::rgba8_unorm();
        let debug_str = format!("{:?}", target);
        assert!(debug_str.contains("None"));
    }

    #[test]
    fn test_builder_debug() {
        let builder = ColorTargetBuilder::new()
            .format(wgpu::TextureFormat::Rgba16Float)
            .alpha_blend();
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("ColorTargetBuilder"));
    }

    #[test]
    fn test_builder_clone() {
        let builder1 = ColorTargetBuilder::new()
            .format(wgpu::TextureFormat::Rgba16Float)
            .alpha_blend();
        let builder2 = builder1.clone();
        let target1 = builder1.build_unchecked();
        let target2 = builder2.build_unchecked();
        assert_eq!(target1, target2);
    }

    #[test]
    fn test_array_debug() {
        let array = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm())
            .target(ColorTarget::rgba16_float());
        let debug_str = format!("{:?}", array);
        assert!(debug_str.contains("ColorTargetArray"));
    }

    #[test]
    fn test_array_clone() {
        let array1 = ColorTargetArray::new()
            .target(ColorTarget::rgba8_unorm())
            .target(ColorTarget::rgba16_float());
        let array2 = array1.clone();
        let targets1 = array1.build();
        let targets2 = array2.build();
        assert_eq!(targets1.len(), targets2.len());
    }

    #[test]
    fn test_error_array_error_display() {
        let inner = ColorTargetError::BlendNotSupported(wgpu::TextureFormat::Rgba8Uint);
        let err = ColorTargetError::ArrayError {
            index: 2,
            error: Box::new(inner),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Error in color target 2"));
        assert!(msg.contains("Blend state not supported"));
    }

    #[test]
    fn test_error_is_std_error() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<ColorTargetError>();
    }

    // -------------------------------------------------------------------------
    // Thread Safety Verification Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_color_target_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ColorTarget>();
    }

    #[test]
    fn test_color_target_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ColorTarget>();
    }

    #[test]
    fn test_color_target_builder_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ColorTargetBuilder>();
    }

    #[test]
    fn test_color_target_builder_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ColorTargetBuilder>();
    }

    #[test]
    fn test_color_target_array_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ColorTargetArray>();
    }

    #[test]
    fn test_color_target_array_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ColorTargetArray>();
    }

    #[test]
    fn test_color_target_error_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ColorTargetError>();
    }

    #[test]
    fn test_color_target_error_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ColorTargetError>();
    }

    // -------------------------------------------------------------------------
    // Conversion Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_from_ref_conversion() {
        let target = ColorTarget::rgba16_float().alpha_blend();
        let wgpu_state: wgpu::ColorTargetState = (&target).into();
        assert_eq!(wgpu_state.format, wgpu::TextureFormat::Rgba16Float);
        assert!(wgpu_state.blend.is_some());
        // Original target still usable
        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn test_round_trip_conversion() {
        let original = ColorTarget::rgba16_float().alpha_blend().write_color();
        let wgpu_state: wgpu::ColorTargetState = original.clone().into();
        let back: ColorTarget = wgpu_state.into();
        assert_eq!(original, back);
    }

    #[test]
    fn test_format_conversion_preserves_defaults() {
        let target: ColorTarget = wgpu::TextureFormat::Rgba32Float.into();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba32Float);
        assert!(target.blend.is_none());
        assert_eq!(target.write_mask, wgpu::ColorWrites::ALL);
    }

    // -------------------------------------------------------------------------
    // Validation Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_all_integer_formats_fail_with_blend() {
        let integer_formats = [
            wgpu::TextureFormat::R8Uint,
            wgpu::TextureFormat::R8Sint,
            wgpu::TextureFormat::R16Uint,
            wgpu::TextureFormat::R16Sint,
            wgpu::TextureFormat::R32Uint,
            wgpu::TextureFormat::R32Sint,
            wgpu::TextureFormat::Rg8Uint,
            wgpu::TextureFormat::Rg8Sint,
            wgpu::TextureFormat::Rg16Uint,
            wgpu::TextureFormat::Rg16Sint,
            wgpu::TextureFormat::Rg32Uint,
            wgpu::TextureFormat::Rg32Sint,
            wgpu::TextureFormat::Rgba8Uint,
            wgpu::TextureFormat::Rgba8Sint,
            wgpu::TextureFormat::Rgba16Uint,
            wgpu::TextureFormat::Rgba16Sint,
            wgpu::TextureFormat::Rgba32Uint,
            wgpu::TextureFormat::Rgba32Sint,
        ];

        for format in integer_formats {
            let target = ColorTarget::new(format).alpha_blend();
            assert!(
                !target.is_valid(),
                "Format {:?} should not support blending",
                format
            );
        }
    }

    #[test]
    fn test_validate_all_float_formats_succeed_with_blend() {
        let float_formats = [
            wgpu::TextureFormat::R16Float,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureFormat::Rg16Float,
            wgpu::TextureFormat::Rg32Float,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureFormat::Rg11b10Float,
        ];

        for format in float_formats {
            let target = ColorTarget::new(format).alpha_blend();
            assert!(
                target.is_valid(),
                "Format {:?} should support blending",
                format
            );
        }
    }

    #[test]
    fn test_validate_all_unorm_formats_succeed_with_blend() {
        let unorm_formats = [
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureFormat::R16Unorm,
            wgpu::TextureFormat::Rg8Unorm,
            wgpu::TextureFormat::Rg16Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Rgba16Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgb10a2Unorm,
        ];

        for format in unorm_formats {
            let target = ColorTarget::new(format).alpha_blend();
            assert!(
                target.is_valid(),
                "Format {:?} should support blending",
                format
            );
        }
    }

    // -------------------------------------------------------------------------
    // Bytes Per Pixel Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_bytes_per_pixel_compressed_format() {
        // Compressed formats should return None
        let target = ColorTarget::new(wgpu::TextureFormat::Bc1RgbaUnorm);
        assert!(target.bytes_per_pixel().is_none());
    }

    #[test]
    fn test_bytes_per_pixel_depth_format() {
        // Depth formats should return None
        let target = ColorTarget::new(wgpu::TextureFormat::Depth32Float);
        assert!(target.bytes_per_pixel().is_none());
    }

    // -------------------------------------------------------------------------
    // ColorTargetInfo PartialEq Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_color_target_info_equality() {
        let info1 = &COLOR_TARGET_PRESETS[0];
        let info2 = &COLOR_TARGET_PRESETS[0];
        assert_eq!(info1, info2);

        let info3 = &COLOR_TARGET_PRESETS[1];
        assert_ne!(info1, info3);
    }

    // -------------------------------------------------------------------------
    // MAX_COLOR_ATTACHMENTS Constant Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_max_color_attachments_value() {
        assert_eq!(MAX_COLOR_ATTACHMENTS, 8);
    }

    // -------------------------------------------------------------------------
    // Builder Validation Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_build_validates() {
        // Valid configuration should succeed
        let result = ColorTargetBuilder::new()
            .format(wgpu::TextureFormat::Rgba8Unorm)
            .alpha_blend()
            .build();
        assert!(result.is_ok());

        // Invalid configuration should fail
        let result = ColorTargetBuilder::new()
            .format(wgpu::TextureFormat::Rgba8Uint)
            .alpha_blend()
            .build();
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // Query Method Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_writes_any_all_combinations() {
        // Empty writes nothing
        let empty = ColorTarget::rgba8_unorm().write_none();
        assert!(!empty.writes_any());

        // Single channel writes something
        let red = ColorTarget::rgba8_unorm().write_red();
        assert!(red.writes_any());

        // All channels writes something
        let all = ColorTarget::rgba8_unorm().write_all();
        assert!(all.writes_any());
    }

    #[test]
    fn test_writes_all_only_true_for_all() {
        // All four channels
        let all = ColorTarget::rgba8_unorm().write_all();
        assert!(all.writes_all());

        // Three channels (COLOR)
        let color = ColorTarget::rgba8_unorm().write_color();
        assert!(!color.writes_all());

        // Single channel
        let red = ColorTarget::rgba8_unorm().write_red();
        assert!(!red.writes_all());
    }
}
