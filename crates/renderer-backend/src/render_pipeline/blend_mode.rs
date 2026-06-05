//! Blend mode presets and configuration for wgpu 25.x render pipelines.
//!
//! This module provides blend state abstractions with common presets and a fluent
//! builder API for custom blend configurations.
//!
//! # Overview
//!
//! Blending determines how fragment shader output combines with existing render
//! target values. The blend equation is:
//!
//! ```text
//! result = src_factor * src_value <operation> dst_factor * dst_value
//! ```
//!
//! Where:
//! - `src_value` = fragment shader output
//! - `dst_value` = existing value in render target
//! - `src_factor` / `dst_factor` = multipliers from `BlendFactor`
//! - `operation` = combining operation from `BlendOperation`
//!
//! # Blend Factors (13 values)
//!
//! | Factor | Value | Description |
//! |--------|-------|-------------|
//! | `Zero` | 0 | Multiply by zero |
//! | `One` | 1 | Multiply by one (identity) |
//! | `Src` | src | Multiply by source color |
//! | `OneMinusSrc` | 1 - src | Multiply by inverse source color |
//! | `SrcAlpha` | src.a | Multiply by source alpha |
//! | `OneMinusSrcAlpha` | 1 - src.a | Multiply by inverse source alpha |
//! | `Dst` | dst | Multiply by destination color |
//! | `OneMinusDst` | 1 - dst | Multiply by inverse destination color |
//! | `DstAlpha` | dst.a | Multiply by destination alpha |
//! | `OneMinusDstAlpha` | 1 - dst.a | Multiply by inverse destination alpha |
//! | `SrcAlphaSaturated` | min(src.a, 1-dst.a) | Saturated source alpha |
//! | `Constant` | const | Multiply by blend constant |
//! | `OneMinusConstant` | 1 - const | Multiply by inverse blend constant |
//!
//! # Blend Operations (5 values)
//!
//! | Operation | Result | Description |
//! |-----------|--------|-------------|
//! | `Add` | src + dst | Add source and destination |
//! | `Subtract` | src - dst | Subtract destination from source |
//! | `ReverseSubtract` | dst - src | Subtract source from destination |
//! | `Min` | min(src, dst) | Component-wise minimum |
//! | `Max` | max(src, dst) | Component-wise maximum |
//!
//! # Common Blend Mode Presets
//!
//! | Mode | Color Equation | Use Case |
//! |------|----------------|----------|
//! | Alpha | src*srcA + dst*(1-srcA) | Standard transparency |
//! | Premultiplied | src + dst*(1-srcA) | Pre-multiplied alpha textures |
//! | Additive | src + dst | Glow, particles, light |
//! | Multiply | src * dst | Shadows, darkening |
//! | Screen | 1 - (1-src)*(1-dst) | Lightening, highlights |
//! | Overlay | Multiply + Screen | Contrast enhancement |
//! | Replace | src | No blending (overwrite) |
//!
//! # wgpu API Reference
//!
//! ```ignore
//! pub enum BlendFactor {
//!     Zero, One, Src, OneMinusSrc, SrcAlpha, OneMinusSrcAlpha,
//!     Dst, OneMinusDst, DstAlpha, OneMinusDstAlpha,
//!     SrcAlphaSaturated, Constant, OneMinusConstant
//! }
//!
//! pub enum BlendOperation {
//!     Add, Subtract, ReverseSubtract, Min, Max
//! }
//!
//! pub struct BlendComponent {
//!     pub src_factor: BlendFactor,
//!     pub dst_factor: BlendFactor,
//!     pub operation: BlendOperation,
//! }
//!
//! pub struct BlendState {
//!     pub color: BlendComponent,
//!     pub alpha: BlendComponent,
//! }
//! ```
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::render_pipeline::blend_mode::{BlendMode, BlendModeBuilder};
//!
//! // Use a preset
//! let alpha_blend = BlendMode::alpha();
//! let additive = BlendMode::additive();
//!
//! // Custom blend mode
//! let custom = BlendModeBuilder::new()
//!     .color_src_factor(wgpu::BlendFactor::SrcAlpha)
//!     .color_dst_factor(wgpu::BlendFactor::One)
//!     .color_operation(wgpu::BlendOperation::Add)
//!     .alpha_src_factor(wgpu::BlendFactor::One)
//!     .alpha_dst_factor(wgpu::BlendFactor::One)
//!     .alpha_operation(wgpu::BlendOperation::Max)
//!     .build();
//!
//! // Convert to wgpu type
//! let wgpu_state: wgpu::BlendState = alpha_blend.into();
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// BlendMode
// ---------------------------------------------------------------------------

/// Describes a blend mode configuration for combining fragment shader output
/// with existing render target values.
///
/// # Fields
///
/// | Field | Type | Description |
/// |-------|------|-------------|
/// | `color` | `BlendComponent` | Blend configuration for RGB channels |
/// | `alpha` | `BlendComponent` | Blend configuration for alpha channel |
///
/// # Blend Equation
///
/// The blend equation is applied separately for color and alpha:
///
/// ```text
/// result.rgb = src_factor * src.rgb <operation> dst_factor * dst.rgb
/// result.a   = src_factor * src.a   <operation> dst_factor * dst.a
/// ```
///
/// # Separate Color/Alpha Configuration
///
/// Color and alpha blending can be configured independently. This is useful for:
///
/// - Preserving alpha while blending color (e.g., UI rendering)
/// - Accumulating alpha values differently than color (e.g., order-independent transparency)
/// - Special effects that treat alpha as a mask
///
/// # Defaults
///
/// Default is standard alpha blending:
/// - Color: `src_alpha * src + (1 - src_alpha) * dst`
/// - Alpha: `one * src + (1 - src_alpha) * dst`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendMode {
    /// Blend configuration for RGB color channels.
    pub color: wgpu::BlendComponent,
    /// Blend configuration for alpha channel.
    pub alpha: wgpu::BlendComponent,
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::alpha()
    }
}

impl BlendMode {
    /// Create a new blend mode with explicit color and alpha configurations.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let blend = BlendMode::new(
    ///     wgpu::BlendComponent {
    ///         src_factor: wgpu::BlendFactor::SrcAlpha,
    ///         dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
    ///         operation: wgpu::BlendOperation::Add,
    ///     },
    ///     wgpu::BlendComponent {
    ///         src_factor: wgpu::BlendFactor::One,
    ///         dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
    ///         operation: wgpu::BlendOperation::Add,
    ///     },
    /// );
    /// ```
    pub const fn new(color: wgpu::BlendComponent, alpha: wgpu::BlendComponent) -> Self {
        Self { color, alpha }
    }

    /// Create a blend mode with the same configuration for color and alpha.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let additive = BlendMode::uniform(wgpu::BlendComponent {
    ///     src_factor: wgpu::BlendFactor::One,
    ///     dst_factor: wgpu::BlendFactor::One,
    ///     operation: wgpu::BlendOperation::Add,
    /// });
    /// ```
    pub const fn uniform(component: wgpu::BlendComponent) -> Self {
        Self {
            color: component,
            alpha: component,
        }
    }

    // -------------------------------------------------------------------------
    // Preset Blend Modes
    // -------------------------------------------------------------------------

    /// Standard alpha blending.
    ///
    /// Blends source with destination using source alpha:
    /// - Color: `src_alpha * src + (1 - src_alpha) * dst`
    /// - Alpha: `one * src + (1 - src_alpha) * dst`
    ///
    /// # Use Cases
    ///
    /// - Transparent UI elements
    /// - Standard sprite rendering
    /// - General transparency effects
    ///
    /// # Formula
    ///
    /// ```text
    /// result.rgb = src.a * src.rgb + (1 - src.a) * dst.rgb
    /// result.a   = 1 * src.a + (1 - src.a) * dst.a
    /// ```
    pub const fn alpha() -> Self {
        Self {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }
    }

    /// Premultiplied alpha blending.
    ///
    /// For textures where RGB is pre-multiplied by alpha:
    /// - Color: `one * src + (1 - src_alpha) * dst`
    /// - Alpha: `one * src + (1 - src_alpha) * dst`
    ///
    /// # Use Cases
    ///
    /// - Pre-multiplied alpha textures
    /// - Photoshop-style layer blending
    /// - WebGL textures with premultiplied alpha
    /// - Anti-aliased text rendering
    ///
    /// # Benefits
    ///
    /// - More accurate blending at edges
    /// - Correct filtering for semi-transparent textures
    /// - Better mipmap generation
    ///
    /// # Formula
    ///
    /// ```text
    /// result.rgb = src.rgb + (1 - src.a) * dst.rgb
    /// result.a   = src.a + (1 - src.a) * dst.a
    /// ```
    pub const fn premultiplied_alpha() -> Self {
        Self {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }
    }

    /// Additive blending.
    ///
    /// Adds source to destination:
    /// - Color: `one * src + one * dst`
    /// - Alpha: `one * src + one * dst`
    ///
    /// # Use Cases
    ///
    /// - Particle effects (fire, sparks, magic)
    /// - Glow and bloom effects
    /// - Light accumulation
    /// - Laser/energy effects
    /// - Lens flares
    ///
    /// # Characteristics
    ///
    /// - Always brightens the image
    /// - Multiple overlapping sources accumulate
    /// - Can exceed [0,1] range (HDR saturation)
    ///
    /// # Formula
    ///
    /// ```text
    /// result.rgb = src.rgb + dst.rgb
    /// result.a   = src.a + dst.a
    /// ```
    pub const fn additive() -> Self {
        Self {
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
        }
    }

    /// Additive blending with alpha modulation.
    ///
    /// Like additive, but source alpha controls contribution:
    /// - Color: `src_alpha * src + one * dst`
    /// - Alpha: `one * src + one * dst`
    ///
    /// # Use Cases
    ///
    /// - Fading particles
    /// - Controllable glow intensity
    /// - Soft additive effects
    ///
    /// # Formula
    ///
    /// ```text
    /// result.rgb = src.a * src.rgb + dst.rgb
    /// result.a   = src.a + dst.a
    /// ```
    pub const fn additive_alpha() -> Self {
        Self {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }
    }

    /// Multiply blending.
    ///
    /// Multiplies source with destination:
    /// - Color: `dst * src + zero * dst` = `src * dst`
    /// - Alpha: `dst_alpha * src + zero * dst` = `src * dst_alpha`
    ///
    /// # Use Cases
    ///
    /// - Shadows and darkening
    /// - Color tinting
    /// - Stained glass effects
    /// - Photoshop-style multiply layers
    ///
    /// # Characteristics
    ///
    /// - Always darkens (unless source is white)
    /// - Black source = black result
    /// - White source = unchanged destination
    /// - Commutative: A*B = B*A
    ///
    /// # Formula
    ///
    /// ```text
    /// result.rgb = src.rgb * dst.rgb
    /// result.a   = src.a * dst.a
    /// ```
    pub const fn multiply() -> Self {
        Self {
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
        }
    }

    /// Screen blending.
    ///
    /// Inverse multiply - brightens the image:
    /// - Color: `one * src + one_minus_src * dst`
    /// - Alpha: `one * src + one_minus_src_alpha * dst`
    ///
    /// Equivalent to: `1 - (1 - src) * (1 - dst)`
    ///
    /// # Use Cases
    ///
    /// - Lightening effects
    /// - Highlights and glare
    /// - Light leak effects
    /// - Photoshop-style screen layers
    ///
    /// # Characteristics
    ///
    /// - Always lightens (unless source is black)
    /// - White source = white result
    /// - Black source = unchanged destination
    /// - Commutative: A screen B = B screen A
    ///
    /// # Formula
    ///
    /// ```text
    /// result.rgb = src.rgb + (1 - src.rgb) * dst.rgb
    ///            = 1 - (1 - src.rgb) * (1 - dst.rgb)
    /// ```
    pub const fn screen() -> Self {
        Self {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }
    }

    /// Soft light blending (approximation).
    ///
    /// A gentler version of overlay - slightly darkens or lightens
    /// depending on the blend color.
    ///
    /// This is an approximation using standard blend modes:
    /// - Color: `dst * src + dst * one_minus_src`
    ///
    /// # Use Cases
    ///
    /// - Subtle lighting adjustments
    /// - Soft shadows
    /// - Gentle color grading
    ///
    /// # Note
    ///
    /// True soft light requires per-pixel math not expressible in
    /// standard blend equations. This approximation works well for
    /// many use cases but may differ from Photoshop's soft light.
    pub const fn soft_light() -> Self {
        Self {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::DstAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        }
    }

    /// Replace blending (no blending).
    ///
    /// Source completely replaces destination:
    /// - Color: `one * src + zero * dst` = `src`
    /// - Alpha: `one * src + zero * dst` = `src`
    ///
    /// # Use Cases
    ///
    /// - Opaque geometry
    /// - Clearing render targets
    /// - First pass of multi-pass rendering
    /// - When blending is explicitly not wanted
    ///
    /// # Formula
    ///
    /// ```text
    /// result = src
    /// ```
    pub const fn replace() -> Self {
        Self {
            color: wgpu::BlendComponent::REPLACE,
            alpha: wgpu::BlendComponent::REPLACE,
        }
    }

    /// Subtractive blending.
    ///
    /// Subtracts source from destination:
    /// - Color: `one * src - one * dst` (src - dst clamped to 0)
    /// - Alpha: `one * src - one * dst`
    ///
    /// Note: Uses Subtract operation which computes (src - dst).
    ///
    /// # Use Cases
    ///
    /// - Color correction
    /// - Removing light/color
    /// - Special effects
    ///
    /// # Characteristics
    ///
    /// - Can result in negative values (clamped to 0)
    /// - Not commutative
    pub const fn subtract() -> Self {
        Self {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::ReverseSubtract,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::ReverseSubtract,
            },
        }
    }

    /// Min blending.
    ///
    /// Takes component-wise minimum:
    /// - Color: `min(src, dst)`
    /// - Alpha: `min(src, dst)`
    ///
    /// # Use Cases
    ///
    /// - Shadow volume stencil operations
    /// - Masking effects
    /// - Finding darkest overlapping value
    ///
    /// # Note
    ///
    /// Blend factors are ignored in Min/Max operations.
    pub const fn min() -> Self {
        Self {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Min,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Min,
            },
        }
    }

    /// Max blending.
    ///
    /// Takes component-wise maximum:
    /// - Color: `max(src, dst)`
    /// - Alpha: `max(src, dst)`
    ///
    /// # Use Cases
    ///
    /// - Light accumulation with clamping
    /// - Finding brightest overlapping value
    /// - HDR bloom threshold extraction
    ///
    /// # Note
    ///
    /// Blend factors are ignored in Min/Max operations.
    pub const fn max() -> Self {
        Self {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Max,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Max,
            },
        }
    }

    // -------------------------------------------------------------------------
    // Modifier Methods
    // -------------------------------------------------------------------------

    /// Set the color blend component.
    pub fn with_color(mut self, color: wgpu::BlendComponent) -> Self {
        self.color = color;
        self
    }

    /// Set the alpha blend component.
    pub fn with_alpha(mut self, alpha: wgpu::BlendComponent) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the color source factor.
    pub fn with_color_src_factor(mut self, factor: wgpu::BlendFactor) -> Self {
        self.color.src_factor = factor;
        self
    }

    /// Set the color destination factor.
    pub fn with_color_dst_factor(mut self, factor: wgpu::BlendFactor) -> Self {
        self.color.dst_factor = factor;
        self
    }

    /// Set the color blend operation.
    pub fn with_color_operation(mut self, operation: wgpu::BlendOperation) -> Self {
        self.color.operation = operation;
        self
    }

    /// Set the alpha source factor.
    pub fn with_alpha_src_factor(mut self, factor: wgpu::BlendFactor) -> Self {
        self.alpha.src_factor = factor;
        self
    }

    /// Set the alpha destination factor.
    pub fn with_alpha_dst_factor(mut self, factor: wgpu::BlendFactor) -> Self {
        self.alpha.dst_factor = factor;
        self
    }

    /// Set the alpha blend operation.
    pub fn with_alpha_operation(mut self, operation: wgpu::BlendOperation) -> Self {
        self.alpha.operation = operation;
        self
    }

    // -------------------------------------------------------------------------
    // Query Methods
    // -------------------------------------------------------------------------

    /// Check if this blend mode is effectively a replace (no blending).
    ///
    /// Returns true if the blend will simply overwrite the destination.
    pub fn is_replace(&self) -> bool {
        self.color.src_factor == wgpu::BlendFactor::One
            && self.color.dst_factor == wgpu::BlendFactor::Zero
            && self.color.operation == wgpu::BlendOperation::Add
            && self.alpha.src_factor == wgpu::BlendFactor::One
            && self.alpha.dst_factor == wgpu::BlendFactor::Zero
            && self.alpha.operation == wgpu::BlendOperation::Add
    }

    /// Check if this blend mode is additive.
    ///
    /// Returns true if both factors are One with Add operation.
    pub fn is_additive(&self) -> bool {
        self.color.src_factor == wgpu::BlendFactor::One
            && self.color.dst_factor == wgpu::BlendFactor::One
            && self.color.operation == wgpu::BlendOperation::Add
    }

    /// Check if this blend mode uses alpha blending.
    ///
    /// Returns true if either src or dst factor depends on alpha.
    pub fn uses_alpha(&self) -> bool {
        self.factor_uses_alpha(self.color.src_factor)
            || self.factor_uses_alpha(self.color.dst_factor)
    }

    /// Check if a blend factor uses alpha values.
    fn factor_uses_alpha(&self, factor: wgpu::BlendFactor) -> bool {
        matches!(
            factor,
            wgpu::BlendFactor::SrcAlpha
                | wgpu::BlendFactor::OneMinusSrcAlpha
                | wgpu::BlendFactor::DstAlpha
                | wgpu::BlendFactor::OneMinusDstAlpha
                | wgpu::BlendFactor::SrcAlphaSaturated
        )
    }

    /// Check if this blend mode uses blend constants.
    ///
    /// If true, `set_blend_constant` must be called on the render pass.
    pub fn uses_constant(&self) -> bool {
        self.factor_uses_constant(self.color.src_factor)
            || self.factor_uses_constant(self.color.dst_factor)
            || self.factor_uses_constant(self.alpha.src_factor)
            || self.factor_uses_constant(self.alpha.dst_factor)
    }

    /// Check if a blend factor uses blend constants.
    fn factor_uses_constant(&self, factor: wgpu::BlendFactor) -> bool {
        matches!(
            factor,
            wgpu::BlendFactor::Constant | wgpu::BlendFactor::OneMinusConstant
        )
    }

    /// Check if color and alpha use the same blend configuration.
    pub fn is_uniform(&self) -> bool {
        self.color == self.alpha
    }

    /// Get the blend operation for color.
    pub fn color_operation(&self) -> wgpu::BlendOperation {
        self.color.operation
    }

    /// Get the blend operation for alpha.
    pub fn alpha_operation(&self) -> wgpu::BlendOperation {
        self.alpha.operation
    }

    /// Check if this blend mode can produce values > 1.0 (HDR).
    ///
    /// Returns true for additive modes that can exceed the [0,1] range.
    pub fn can_exceed_one(&self) -> bool {
        // Additive blending with Add operation can exceed 1.0
        self.color.operation == wgpu::BlendOperation::Add
            && self.color.src_factor == wgpu::BlendFactor::One
            && self.color.dst_factor == wgpu::BlendFactor::One
    }

    /// Check if this blend mode only darkens (like multiply).
    pub fn only_darkens(&self) -> bool {
        // Multiply mode always darkens
        self.color.src_factor == wgpu::BlendFactor::Dst
            && self.color.dst_factor == wgpu::BlendFactor::Zero
    }

    /// Check if this blend mode only lightens (like screen).
    pub fn only_lightens(&self) -> bool {
        // Screen mode always lightens
        self.color.src_factor == wgpu::BlendFactor::One
            && self.color.dst_factor == wgpu::BlendFactor::OneMinusSrc
            && self.color.operation == wgpu::BlendOperation::Add
    }
}

// Thread-safety: BlendMode contains only Copy types
unsafe impl Send for BlendMode {}
unsafe impl Sync for BlendMode {}

impl From<BlendMode> for wgpu::BlendState {
    fn from(mode: BlendMode) -> Self {
        wgpu::BlendState {
            color: mode.color,
            alpha: mode.alpha,
        }
    }
}

impl From<&BlendMode> for wgpu::BlendState {
    fn from(mode: &BlendMode) -> Self {
        wgpu::BlendState {
            color: mode.color,
            alpha: mode.alpha,
        }
    }
}

impl From<wgpu::BlendState> for BlendMode {
    fn from(state: wgpu::BlendState) -> Self {
        Self {
            color: state.color,
            alpha: state.alpha,
        }
    }
}

// ---------------------------------------------------------------------------
// BlendModeBuilder
// ---------------------------------------------------------------------------

/// Builder for creating custom blend mode configurations with a fluent API.
///
/// # Example
///
/// ```ignore
/// let custom = BlendModeBuilder::new()
///     .color_src_factor(wgpu::BlendFactor::SrcAlpha)
///     .color_dst_factor(wgpu::BlendFactor::OneMinusSrcAlpha)
///     .color_operation(wgpu::BlendOperation::Add)
///     .alpha_src_factor(wgpu::BlendFactor::One)
///     .alpha_dst_factor(wgpu::BlendFactor::Zero)
///     .alpha_operation(wgpu::BlendOperation::Add)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct BlendModeBuilder {
    mode: BlendMode,
}

impl Default for BlendModeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BlendModeBuilder {
    /// Create a new builder with default alpha blending.
    pub fn new() -> Self {
        Self {
            mode: BlendMode::alpha(),
        }
    }

    /// Start building from a preset.
    pub fn from_preset(preset: BlendMode) -> Self {
        Self { mode: preset }
    }

    /// Start building from replace (no blending).
    pub fn from_replace() -> Self {
        Self {
            mode: BlendMode::replace(),
        }
    }

    // -------------------------------------------------------------------------
    // Color Component Configuration
    // -------------------------------------------------------------------------

    /// Set the entire color blend component.
    pub fn color(mut self, component: wgpu::BlendComponent) -> Self {
        self.mode.color = component;
        self
    }

    /// Set the color source factor.
    pub fn color_src_factor(mut self, factor: wgpu::BlendFactor) -> Self {
        self.mode.color.src_factor = factor;
        self
    }

    /// Set the color destination factor.
    pub fn color_dst_factor(mut self, factor: wgpu::BlendFactor) -> Self {
        self.mode.color.dst_factor = factor;
        self
    }

    /// Set the color blend operation.
    pub fn color_operation(mut self, operation: wgpu::BlendOperation) -> Self {
        self.mode.color.operation = operation;
        self
    }

    // -------------------------------------------------------------------------
    // Alpha Component Configuration
    // -------------------------------------------------------------------------

    /// Set the entire alpha blend component.
    pub fn alpha(mut self, component: wgpu::BlendComponent) -> Self {
        self.mode.alpha = component;
        self
    }

    /// Set the alpha source factor.
    pub fn alpha_src_factor(mut self, factor: wgpu::BlendFactor) -> Self {
        self.mode.alpha.src_factor = factor;
        self
    }

    /// Set the alpha destination factor.
    pub fn alpha_dst_factor(mut self, factor: wgpu::BlendFactor) -> Self {
        self.mode.alpha.dst_factor = factor;
        self
    }

    /// Set the alpha blend operation.
    pub fn alpha_operation(mut self, operation: wgpu::BlendOperation) -> Self {
        self.mode.alpha.operation = operation;
        self
    }

    // -------------------------------------------------------------------------
    // Uniform Configuration (same for color and alpha)
    // -------------------------------------------------------------------------

    /// Set both color and alpha to the same component.
    pub fn uniform(mut self, component: wgpu::BlendComponent) -> Self {
        self.mode.color = component;
        self.mode.alpha = component;
        self
    }

    /// Set both color and alpha source factor.
    pub fn src_factor(mut self, factor: wgpu::BlendFactor) -> Self {
        self.mode.color.src_factor = factor;
        self.mode.alpha.src_factor = factor;
        self
    }

    /// Set both color and alpha destination factor.
    pub fn dst_factor(mut self, factor: wgpu::BlendFactor) -> Self {
        self.mode.color.dst_factor = factor;
        self.mode.alpha.dst_factor = factor;
        self
    }

    /// Set both color and alpha operation.
    pub fn operation(mut self, operation: wgpu::BlendOperation) -> Self {
        self.mode.color.operation = operation;
        self.mode.alpha.operation = operation;
        self
    }

    // -------------------------------------------------------------------------
    // Build
    // -------------------------------------------------------------------------

    /// Build the blend mode.
    pub fn build(self) -> BlendMode {
        self.mode
    }

    /// Build and convert to wgpu BlendState.
    pub fn build_wgpu(self) -> wgpu::BlendState {
        self.mode.into()
    }
}

// ---------------------------------------------------------------------------
// BlendModeInfo
// ---------------------------------------------------------------------------

/// Metadata about a blend mode preset.
///
/// Provides descriptive information for tooling, debugging, and documentation.
#[derive(Debug, Clone, PartialEq)]
pub struct BlendModeInfo {
    /// Human-readable name for the preset.
    pub name: &'static str,
    /// Brief description of the blend mode.
    pub description: &'static str,
    /// The color blend equation (human-readable).
    pub color_equation: &'static str,
    /// Typical use cases for this blend mode.
    pub use_cases: &'static [&'static str],
    /// Whether this mode can produce values > 1.0 (requires HDR target).
    pub can_exceed_one: bool,
    /// Whether this mode uses alpha values.
    pub uses_alpha: bool,
    /// Whether this mode uses blend constants.
    pub uses_constant: bool,
}

/// Information about all 13 blend factors.
pub const BLEND_FACTORS: [BlendFactorInfo; 13] = [
    BlendFactorInfo {
        factor: wgpu::BlendFactor::Zero,
        name: "Zero",
        description: "Multiply by zero",
        value: "0",
    },
    BlendFactorInfo {
        factor: wgpu::BlendFactor::One,
        name: "One",
        description: "Multiply by one (identity)",
        value: "1",
    },
    BlendFactorInfo {
        factor: wgpu::BlendFactor::Src,
        name: "Src",
        description: "Multiply by source color",
        value: "src",
    },
    BlendFactorInfo {
        factor: wgpu::BlendFactor::OneMinusSrc,
        name: "OneMinusSrc",
        description: "Multiply by inverse source color",
        value: "1 - src",
    },
    BlendFactorInfo {
        factor: wgpu::BlendFactor::SrcAlpha,
        name: "SrcAlpha",
        description: "Multiply by source alpha",
        value: "src.a",
    },
    BlendFactorInfo {
        factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        name: "OneMinusSrcAlpha",
        description: "Multiply by inverse source alpha",
        value: "1 - src.a",
    },
    BlendFactorInfo {
        factor: wgpu::BlendFactor::Dst,
        name: "Dst",
        description: "Multiply by destination color",
        value: "dst",
    },
    BlendFactorInfo {
        factor: wgpu::BlendFactor::OneMinusDst,
        name: "OneMinusDst",
        description: "Multiply by inverse destination color",
        value: "1 - dst",
    },
    BlendFactorInfo {
        factor: wgpu::BlendFactor::DstAlpha,
        name: "DstAlpha",
        description: "Multiply by destination alpha",
        value: "dst.a",
    },
    BlendFactorInfo {
        factor: wgpu::BlendFactor::OneMinusDstAlpha,
        name: "OneMinusDstAlpha",
        description: "Multiply by inverse destination alpha",
        value: "1 - dst.a",
    },
    BlendFactorInfo {
        factor: wgpu::BlendFactor::SrcAlphaSaturated,
        name: "SrcAlphaSaturated",
        description: "Saturated source alpha",
        value: "min(src.a, 1 - dst.a)",
    },
    BlendFactorInfo {
        factor: wgpu::BlendFactor::Constant,
        name: "Constant",
        description: "Multiply by blend constant",
        value: "const",
    },
    BlendFactorInfo {
        factor: wgpu::BlendFactor::OneMinusConstant,
        name: "OneMinusConstant",
        description: "Multiply by inverse blend constant",
        value: "1 - const",
    },
];

/// Information about a blend factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendFactorInfo {
    /// The wgpu blend factor.
    pub factor: wgpu::BlendFactor,
    /// Human-readable name.
    pub name: &'static str,
    /// Description of what this factor does.
    pub description: &'static str,
    /// Mathematical representation.
    pub value: &'static str,
}

/// Information about all 5 blend operations.
pub const BLEND_OPERATIONS: [BlendOperationInfo; 5] = [
    BlendOperationInfo {
        operation: wgpu::BlendOperation::Add,
        name: "Add",
        description: "Add source and destination",
        formula: "src + dst",
    },
    BlendOperationInfo {
        operation: wgpu::BlendOperation::Subtract,
        name: "Subtract",
        description: "Subtract destination from source",
        formula: "src - dst",
    },
    BlendOperationInfo {
        operation: wgpu::BlendOperation::ReverseSubtract,
        name: "ReverseSubtract",
        description: "Subtract source from destination",
        formula: "dst - src",
    },
    BlendOperationInfo {
        operation: wgpu::BlendOperation::Min,
        name: "Min",
        description: "Component-wise minimum",
        formula: "min(src, dst)",
    },
    BlendOperationInfo {
        operation: wgpu::BlendOperation::Max,
        name: "Max",
        description: "Component-wise maximum",
        formula: "max(src, dst)",
    },
];

/// Information about a blend operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendOperationInfo {
    /// The wgpu blend operation.
    pub operation: wgpu::BlendOperation,
    /// Human-readable name.
    pub name: &'static str,
    /// Description of what this operation does.
    pub description: &'static str,
    /// Mathematical formula.
    pub formula: &'static str,
}

/// Common blend mode presets with documentation.
pub const BLEND_MODE_PRESETS: [BlendModeInfo; 12] = [
    BlendModeInfo {
        name: "Alpha",
        description: "Standard alpha blending for transparency",
        color_equation: "src.a * src + (1 - src.a) * dst",
        use_cases: &["transparent UI", "sprites", "general transparency"],
        can_exceed_one: false,
        uses_alpha: true,
        uses_constant: false,
    },
    BlendModeInfo {
        name: "Premultiplied Alpha",
        description: "For textures with pre-multiplied alpha",
        color_equation: "src + (1 - src.a) * dst",
        use_cases: &["pre-multiplied textures", "anti-aliased text", "WebGL"],
        can_exceed_one: false,
        uses_alpha: true,
        uses_constant: false,
    },
    BlendModeInfo {
        name: "Additive",
        description: "Add source to destination",
        color_equation: "src + dst",
        use_cases: &["particles", "glow", "light accumulation", "lasers"],
        can_exceed_one: true,
        uses_alpha: false,
        uses_constant: false,
    },
    BlendModeInfo {
        name: "Additive Alpha",
        description: "Additive with alpha modulation",
        color_equation: "src.a * src + dst",
        use_cases: &["fading particles", "controllable glow"],
        can_exceed_one: true,
        uses_alpha: true,
        uses_constant: false,
    },
    BlendModeInfo {
        name: "Multiply",
        description: "Multiply source with destination",
        color_equation: "src * dst",
        use_cases: &["shadows", "darkening", "color tinting"],
        can_exceed_one: false,
        uses_alpha: false,
        uses_constant: false,
    },
    BlendModeInfo {
        name: "Screen",
        description: "Inverse multiply for lightening",
        color_equation: "1 - (1 - src) * (1 - dst)",
        use_cases: &["highlights", "light leak", "lightening"],
        can_exceed_one: false,
        uses_alpha: false,
        uses_constant: false,
    },
    BlendModeInfo {
        name: "Soft Light",
        description: "Gentle darkening or lightening",
        color_equation: "approximation of soft light",
        use_cases: &["subtle adjustments", "soft shadows", "color grading"],
        can_exceed_one: false,
        uses_alpha: false,
        uses_constant: false,
    },
    BlendModeInfo {
        name: "Replace",
        description: "No blending, source replaces destination",
        color_equation: "src",
        use_cases: &["opaque geometry", "clearing", "first pass"],
        can_exceed_one: false,
        uses_alpha: false,
        uses_constant: false,
    },
    BlendModeInfo {
        name: "Subtract",
        description: "Subtract source from destination",
        color_equation: "dst - src",
        use_cases: &["color correction", "special effects"],
        can_exceed_one: false,
        uses_alpha: false,
        uses_constant: false,
    },
    BlendModeInfo {
        name: "Min",
        description: "Component-wise minimum",
        color_equation: "min(src, dst)",
        use_cases: &["shadow volumes", "masking"],
        can_exceed_one: false,
        uses_alpha: false,
        uses_constant: false,
    },
    BlendModeInfo {
        name: "Max",
        description: "Component-wise maximum",
        color_equation: "max(src, dst)",
        use_cases: &["light accumulation", "bloom extraction"],
        can_exceed_one: false,
        uses_alpha: false,
        uses_constant: false,
    },
    BlendModeInfo {
        name: "Constant",
        description: "Uses blend constant for factor",
        color_equation: "const * src + (1 - const) * dst",
        use_cases: &["fade effects", "cross-dissolve", "dynamic blending"],
        can_exceed_one: false,
        uses_alpha: false,
        uses_constant: true,
    },
];

/// Get blend mode info by preset name.
///
/// # Example
///
/// ```ignore
/// if let Some(info) = get_blend_mode_info("Alpha") {
///     println!("Use cases: {:?}", info.use_cases);
///     println!("Equation: {}", info.color_equation);
/// }
/// ```
pub fn get_blend_mode_info(name: &str) -> Option<&'static BlendModeInfo> {
    BLEND_MODE_PRESETS.iter().find(|info| info.name == name)
}

/// Get blend factor info by factor.
pub fn get_blend_factor_info(factor: wgpu::BlendFactor) -> Option<&'static BlendFactorInfo> {
    BLEND_FACTORS.iter().find(|info| info.factor == factor)
}

/// Get blend operation info by operation.
pub fn get_blend_operation_info(
    operation: wgpu::BlendOperation,
) -> Option<&'static BlendOperationInfo> {
    BLEND_OPERATIONS
        .iter()
        .find(|info| info.operation == operation)
}

/// Get a blend mode preset by name.
pub fn get_preset(name: &str) -> Option<BlendMode> {
    match name {
        "Alpha" => Some(BlendMode::alpha()),
        "Premultiplied Alpha" => Some(BlendMode::premultiplied_alpha()),
        "Additive" => Some(BlendMode::additive()),
        "Additive Alpha" => Some(BlendMode::additive_alpha()),
        "Multiply" => Some(BlendMode::multiply()),
        "Screen" => Some(BlendMode::screen()),
        "Soft Light" => Some(BlendMode::soft_light()),
        "Replace" => Some(BlendMode::replace()),
        "Subtract" => Some(BlendMode::subtract()),
        "Min" => Some(BlendMode::min()),
        "Max" => Some(BlendMode::max()),
        _ => None,
    }
}

/// List all available preset names.
pub fn preset_names() -> impl Iterator<Item = &'static str> {
    BLEND_MODE_PRESETS.iter().map(|info| info.name)
}

/// Get all blend factor names.
pub fn blend_factor_names() -> impl Iterator<Item = &'static str> {
    BLEND_FACTORS.iter().map(|info| info.name)
}

/// Get all blend operation names.
pub fn blend_operation_names() -> impl Iterator<Item = &'static str> {
    BLEND_OPERATIONS.iter().map(|info| info.name)
}

/// Get presets that can exceed [0,1] range (for HDR).
pub fn hdr_presets() -> impl Iterator<Item = &'static BlendModeInfo> {
    BLEND_MODE_PRESETS.iter().filter(|info| info.can_exceed_one)
}

/// Get presets that use alpha blending.
pub fn alpha_presets() -> impl Iterator<Item = &'static BlendModeInfo> {
    BLEND_MODE_PRESETS.iter().filter(|info| info.uses_alpha)
}

/// Get presets that use blend constants.
pub fn constant_presets() -> impl Iterator<Item = &'static BlendModeInfo> {
    BLEND_MODE_PRESETS.iter().filter(|info| info.uses_constant)
}

// ---------------------------------------------------------------------------
// Display implementations
// ---------------------------------------------------------------------------

impl fmt::Display for BlendMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BlendMode {{ color: {:?} * src {:?} {:?} * dst, alpha: {:?} * src {:?} {:?} * dst }}",
            self.color.src_factor,
            self.color.operation,
            self.color.dst_factor,
            self.alpha.src_factor,
            self.alpha.operation,
            self.alpha.dst_factor,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // BlendMode Basic Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blend_mode_default() {
        let mode = BlendMode::default();
        // Default is alpha blending
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Add);
    }

    #[test]
    fn test_blend_mode_new() {
        let color = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        };
        let alpha = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        };
        let mode = BlendMode::new(color, alpha);
        assert_eq!(mode.color, color);
        assert_eq!(mode.alpha, alpha);
    }

    #[test]
    fn test_blend_mode_uniform() {
        let component = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let mode = BlendMode::uniform(component);
        assert_eq!(mode.color, component);
        assert_eq!(mode.alpha, component);
        assert!(mode.is_uniform());
    }

    // -------------------------------------------------------------------------
    // Preset Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_alpha_preset() {
        let mode = BlendMode::alpha();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Add);
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert!(mode.uses_alpha());
    }

    #[test]
    fn test_premultiplied_alpha_preset() {
        let mode = BlendMode::premultiplied_alpha();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert!(mode.uses_alpha());
    }

    #[test]
    fn test_additive_preset() {
        let mode = BlendMode::additive();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Add);
        assert!(mode.is_additive());
        assert!(mode.can_exceed_one());
        assert!(!mode.uses_alpha());
    }

    #[test]
    fn test_additive_alpha_preset() {
        let mode = BlendMode::additive_alpha();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::One);
        assert!(mode.uses_alpha());
    }

    #[test]
    fn test_multiply_preset() {
        let mode = BlendMode::multiply();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::Dst);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::Zero);
        assert!(mode.only_darkens());
        assert!(!mode.uses_alpha());
    }

    #[test]
    fn test_screen_preset() {
        let mode = BlendMode::screen();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::OneMinusSrc);
        assert!(mode.only_lightens());
    }

    #[test]
    fn test_soft_light_preset() {
        let mode = BlendMode::soft_light();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::Dst);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::OneMinusSrc);
    }

    #[test]
    fn test_replace_preset() {
        let mode = BlendMode::replace();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::Zero);
        assert!(mode.is_replace());
    }

    #[test]
    fn test_subtract_preset() {
        let mode = BlendMode::subtract();
        assert_eq!(mode.color.operation, wgpu::BlendOperation::ReverseSubtract);
    }

    #[test]
    fn test_min_preset() {
        let mode = BlendMode::min();
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Min);
        assert_eq!(mode.alpha.operation, wgpu::BlendOperation::Min);
    }

    #[test]
    fn test_max_preset() {
        let mode = BlendMode::max();
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Max);
        assert_eq!(mode.alpha.operation, wgpu::BlendOperation::Max);
    }

    // -------------------------------------------------------------------------
    // Modifier Method Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_with_color() {
        let component = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Constant,
            dst_factor: wgpu::BlendFactor::OneMinusConstant,
            operation: wgpu::BlendOperation::Subtract,
        };
        let mode = BlendMode::alpha().with_color(component);
        assert_eq!(mode.color, component);
    }

    #[test]
    fn test_with_alpha() {
        let component = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::DstAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusDstAlpha,
            operation: wgpu::BlendOperation::Max,
        };
        let mode = BlendMode::alpha().with_alpha(component);
        assert_eq!(mode.alpha, component);
    }

    #[test]
    fn test_with_color_factors() {
        let mode = BlendMode::alpha()
            .with_color_src_factor(wgpu::BlendFactor::One)
            .with_color_dst_factor(wgpu::BlendFactor::Zero)
            .with_color_operation(wgpu::BlendOperation::Max);
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::Zero);
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Max);
    }

    #[test]
    fn test_with_alpha_factors() {
        let mode = BlendMode::alpha()
            .with_alpha_src_factor(wgpu::BlendFactor::DstAlpha)
            .with_alpha_dst_factor(wgpu::BlendFactor::SrcAlpha)
            .with_alpha_operation(wgpu::BlendOperation::Min);
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::DstAlpha);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(mode.alpha.operation, wgpu::BlendOperation::Min);
    }

    // -------------------------------------------------------------------------
    // Query Method Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_replace() {
        assert!(BlendMode::replace().is_replace());
        assert!(!BlendMode::alpha().is_replace());
        assert!(!BlendMode::additive().is_replace());
    }

    #[test]
    fn test_is_additive() {
        assert!(BlendMode::additive().is_additive());
        assert!(!BlendMode::alpha().is_additive());
        assert!(!BlendMode::multiply().is_additive());
    }

    #[test]
    fn test_uses_alpha() {
        assert!(BlendMode::alpha().uses_alpha());
        assert!(BlendMode::premultiplied_alpha().uses_alpha());
        assert!(BlendMode::additive_alpha().uses_alpha());
        assert!(!BlendMode::additive().uses_alpha());
        assert!(!BlendMode::multiply().uses_alpha());
    }

    #[test]
    fn test_uses_constant() {
        let with_constant = BlendModeBuilder::new()
            .color_src_factor(wgpu::BlendFactor::Constant)
            .build();
        assert!(with_constant.uses_constant());
        assert!(!BlendMode::alpha().uses_constant());
    }

    #[test]
    fn test_is_uniform() {
        assert!(BlendMode::additive().is_uniform());
        assert!(BlendMode::replace().is_uniform());
        assert!(!BlendMode::alpha().is_uniform()); // Color and alpha differ
    }

    #[test]
    fn test_color_operation() {
        assert_eq!(BlendMode::alpha().color_operation(), wgpu::BlendOperation::Add);
        assert_eq!(BlendMode::min().color_operation(), wgpu::BlendOperation::Min);
        assert_eq!(BlendMode::max().color_operation(), wgpu::BlendOperation::Max);
    }

    #[test]
    fn test_alpha_operation() {
        assert_eq!(BlendMode::alpha().alpha_operation(), wgpu::BlendOperation::Add);
        assert_eq!(BlendMode::min().alpha_operation(), wgpu::BlendOperation::Min);
    }

    #[test]
    fn test_can_exceed_one() {
        assert!(BlendMode::additive().can_exceed_one());
        assert!(!BlendMode::alpha().can_exceed_one());
        assert!(!BlendMode::multiply().can_exceed_one());
    }

    #[test]
    fn test_only_darkens() {
        assert!(BlendMode::multiply().only_darkens());
        assert!(!BlendMode::alpha().only_darkens());
        assert!(!BlendMode::screen().only_darkens());
    }

    #[test]
    fn test_only_lightens() {
        assert!(BlendMode::screen().only_lightens());
        assert!(!BlendMode::alpha().only_lightens());
        assert!(!BlendMode::multiply().only_lightens());
    }

    // -------------------------------------------------------------------------
    // Conversion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_into_wgpu_blend_state() {
        let mode = BlendMode::alpha();
        let state: wgpu::BlendState = mode.into();
        assert_eq!(state.color, mode.color);
        assert_eq!(state.alpha, mode.alpha);
    }

    #[test]
    fn test_from_wgpu_blend_state() {
        let state = wgpu::BlendState::ALPHA_BLENDING;
        let mode: BlendMode = state.into();
        assert_eq!(mode.color, state.color);
        assert_eq!(mode.alpha, state.alpha);
    }

    #[test]
    fn test_ref_into_wgpu() {
        let mode = BlendMode::additive();
        let state: wgpu::BlendState = (&mode).into();
        assert_eq!(state.color, mode.color);
        assert_eq!(state.alpha, mode.alpha);
    }

    // -------------------------------------------------------------------------
    // Builder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_default() {
        let mode = BlendModeBuilder::new().build();
        assert_eq!(mode, BlendMode::alpha());
    }

    #[test]
    fn test_builder_from_preset() {
        let mode = BlendModeBuilder::from_preset(BlendMode::additive())
            .color_dst_factor(wgpu::BlendFactor::SrcAlpha)
            .build();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::SrcAlpha);
    }

    #[test]
    fn test_builder_from_replace() {
        let mode = BlendModeBuilder::from_replace().build();
        assert!(mode.is_replace());
    }

    #[test]
    fn test_builder_color_configuration() {
        let mode = BlendModeBuilder::new()
            .color(wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Dst,
                dst_factor: wgpu::BlendFactor::Src,
                operation: wgpu::BlendOperation::Subtract,
            })
            .build();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::Dst);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::Src);
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Subtract);
    }

    #[test]
    fn test_builder_alpha_configuration() {
        let mode = BlendModeBuilder::new()
            .alpha(wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::DstAlpha,
                dst_factor: wgpu::BlendFactor::SrcAlpha,
                operation: wgpu::BlendOperation::Max,
            })
            .build();
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::DstAlpha);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(mode.alpha.operation, wgpu::BlendOperation::Max);
    }

    #[test]
    fn test_builder_uniform() {
        let component = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let mode = BlendModeBuilder::new().uniform(component).build();
        assert_eq!(mode.color, component);
        assert_eq!(mode.alpha, component);
    }

    #[test]
    fn test_builder_src_dst_operation() {
        let mode = BlendModeBuilder::new()
            .src_factor(wgpu::BlendFactor::One)
            .dst_factor(wgpu::BlendFactor::Zero)
            .operation(wgpu::BlendOperation::Add)
            .build();
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::Zero);
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::Zero);
    }

    #[test]
    fn test_builder_build_wgpu() {
        let state = BlendModeBuilder::new()
            .src_factor(wgpu::BlendFactor::One)
            .dst_factor(wgpu::BlendFactor::One)
            .build_wgpu();
        assert_eq!(state.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(state.color.dst_factor, wgpu::BlendFactor::One);
    }

    // -------------------------------------------------------------------------
    // Blend Factor Info Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blend_factors_count() {
        assert_eq!(BLEND_FACTORS.len(), 13);
    }

    #[test]
    fn test_get_blend_factor_info() {
        let info = get_blend_factor_info(wgpu::BlendFactor::SrcAlpha);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "SrcAlpha");
        assert_eq!(info.value, "src.a");
    }

    #[test]
    fn test_all_blend_factors_have_info() {
        let factors = [
            wgpu::BlendFactor::Zero,
            wgpu::BlendFactor::One,
            wgpu::BlendFactor::Src,
            wgpu::BlendFactor::OneMinusSrc,
            wgpu::BlendFactor::SrcAlpha,
            wgpu::BlendFactor::OneMinusSrcAlpha,
            wgpu::BlendFactor::Dst,
            wgpu::BlendFactor::OneMinusDst,
            wgpu::BlendFactor::DstAlpha,
            wgpu::BlendFactor::OneMinusDstAlpha,
            wgpu::BlendFactor::SrcAlphaSaturated,
            wgpu::BlendFactor::Constant,
            wgpu::BlendFactor::OneMinusConstant,
        ];
        for factor in factors {
            assert!(
                get_blend_factor_info(factor).is_some(),
                "Missing info for {:?}",
                factor
            );
        }
    }

    // -------------------------------------------------------------------------
    // Blend Operation Info Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blend_operations_count() {
        assert_eq!(BLEND_OPERATIONS.len(), 5);
    }

    #[test]
    fn test_get_blend_operation_info() {
        let info = get_blend_operation_info(wgpu::BlendOperation::Add);
        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "Add");
        assert_eq!(info.formula, "src + dst");
    }

    #[test]
    fn test_all_blend_operations_have_info() {
        let operations = [
            wgpu::BlendOperation::Add,
            wgpu::BlendOperation::Subtract,
            wgpu::BlendOperation::ReverseSubtract,
            wgpu::BlendOperation::Min,
            wgpu::BlendOperation::Max,
        ];
        for op in operations {
            assert!(
                get_blend_operation_info(op).is_some(),
                "Missing info for {:?}",
                op
            );
        }
    }

    // -------------------------------------------------------------------------
    // Preset Info Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_blend_mode_presets_count() {
        assert_eq!(BLEND_MODE_PRESETS.len(), 12);
    }

    #[test]
    fn test_get_blend_mode_info() {
        let info = get_blend_mode_info("Alpha");
        assert!(info.is_some());
        let info = info.unwrap();
        assert!(info.uses_alpha);
        assert!(!info.can_exceed_one);
    }

    #[test]
    fn test_get_preset() {
        let mode = get_preset("Alpha");
        assert!(mode.is_some());
        assert_eq!(mode.unwrap(), BlendMode::alpha());

        let mode = get_preset("Additive");
        assert!(mode.is_some());
        assert_eq!(mode.unwrap(), BlendMode::additive());

        let mode = get_preset("NonExistent");
        assert!(mode.is_none());
    }

    #[test]
    fn test_preset_names() {
        let names: Vec<_> = preset_names().collect();
        assert!(names.contains(&"Alpha"));
        assert!(names.contains(&"Additive"));
        assert!(names.contains(&"Multiply"));
        assert!(names.contains(&"Screen"));
    }

    #[test]
    fn test_blend_factor_names() {
        let names: Vec<_> = blend_factor_names().collect();
        assert_eq!(names.len(), 13);
        assert!(names.contains(&"Zero"));
        assert!(names.contains(&"One"));
        assert!(names.contains(&"SrcAlpha"));
    }

    #[test]
    fn test_blend_operation_names() {
        let names: Vec<_> = blend_operation_names().collect();
        assert_eq!(names.len(), 5);
        assert!(names.contains(&"Add"));
        assert!(names.contains(&"Subtract"));
        assert!(names.contains(&"Min"));
        assert!(names.contains(&"Max"));
    }

    #[test]
    fn test_hdr_presets() {
        let hdr: Vec<_> = hdr_presets().collect();
        assert!(!hdr.is_empty());
        for info in &hdr {
            assert!(info.can_exceed_one);
        }
    }

    #[test]
    fn test_alpha_presets() {
        let alpha: Vec<_> = alpha_presets().collect();
        assert!(!alpha.is_empty());
        for info in &alpha {
            assert!(info.uses_alpha);
        }
    }

    #[test]
    fn test_constant_presets() {
        let constant: Vec<_> = constant_presets().collect();
        assert!(!constant.is_empty());
        for info in &constant {
            assert!(info.uses_constant);
        }
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<BlendMode>();
        assert_sync::<BlendMode>();
    }

    // -------------------------------------------------------------------------
    // Clone and Equality Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_clone() {
        let original = BlendMode::alpha();
        let cloned = original;
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_equality() {
        let a = BlendMode::alpha();
        let b = BlendMode::alpha();
        let c = BlendMode::additive();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_debug() {
        let mode = BlendMode::alpha();
        let debug_str = format!("{:?}", mode);
        assert!(debug_str.contains("BlendMode"));
    }

    #[test]
    fn test_display() {
        let mode = BlendMode::alpha();
        let display_str = format!("{}", mode);
        assert!(display_str.contains("BlendMode"));
        assert!(display_str.contains("src"));
        assert!(display_str.contains("dst"));
    }

    // -------------------------------------------------------------------------
    // Separate Color/Alpha Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_separate_color_alpha() {
        let mode = BlendModeBuilder::new()
            .color_src_factor(wgpu::BlendFactor::One)
            .color_dst_factor(wgpu::BlendFactor::One)
            .alpha_src_factor(wgpu::BlendFactor::Zero)
            .alpha_dst_factor(wgpu::BlendFactor::One)
            .build();

        // Color is additive
        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::One);

        // Alpha preserves destination
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::Zero);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::One);

        assert!(!mode.is_uniform());
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_all_blend_factor_variants() {
        // Ensure all factor variants can be used in a mode
        let factors = [
            wgpu::BlendFactor::Zero,
            wgpu::BlendFactor::One,
            wgpu::BlendFactor::Src,
            wgpu::BlendFactor::OneMinusSrc,
            wgpu::BlendFactor::SrcAlpha,
            wgpu::BlendFactor::OneMinusSrcAlpha,
            wgpu::BlendFactor::Dst,
            wgpu::BlendFactor::OneMinusDst,
            wgpu::BlendFactor::DstAlpha,
            wgpu::BlendFactor::OneMinusDstAlpha,
            wgpu::BlendFactor::SrcAlphaSaturated,
            wgpu::BlendFactor::Constant,
            wgpu::BlendFactor::OneMinusConstant,
        ];

        for factor in factors {
            let mode = BlendMode::alpha().with_color_src_factor(factor);
            assert_eq!(mode.color.src_factor, factor);
        }
    }

    #[test]
    fn test_all_blend_operation_variants() {
        // Ensure all operation variants can be used in a mode
        let operations = [
            wgpu::BlendOperation::Add,
            wgpu::BlendOperation::Subtract,
            wgpu::BlendOperation::ReverseSubtract,
            wgpu::BlendOperation::Min,
            wgpu::BlendOperation::Max,
        ];

        for op in operations {
            let mode = BlendMode::alpha().with_color_operation(op);
            assert_eq!(mode.color.operation, op);
        }
    }

    #[test]
    fn test_wgpu_premultiplied_alpha_matches() {
        let mode = BlendMode::premultiplied_alpha();
        let wgpu_state = wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;
        assert_eq!(mode.color, wgpu_state.color);
        assert_eq!(mode.alpha, wgpu_state.alpha);
    }

    #[test]
    fn test_wgpu_alpha_blending_matches() {
        let mode = BlendMode::alpha();
        let wgpu_state = wgpu::BlendState::ALPHA_BLENDING;
        assert_eq!(mode.color, wgpu_state.color);
        assert_eq!(mode.alpha, wgpu_state.alpha);
    }

    #[test]
    fn test_wgpu_replace_matches() {
        let mode = BlendMode::replace();
        let wgpu_state = wgpu::BlendState::REPLACE;
        assert_eq!(mode.color, wgpu_state.color);
        assert_eq!(mode.alpha, wgpu_state.alpha);
    }

    // -------------------------------------------------------------------------
    // Additional Whitebox Tests - T-WGPU-P3.5.2
    // -------------------------------------------------------------------------

    // ---- All Preset Combinations ----

    #[test]
    fn test_all_presets_exist() {
        // Verify all 11 documented presets are accessible
        let presets = [
            BlendMode::alpha(),
            BlendMode::premultiplied_alpha(),
            BlendMode::additive(),
            BlendMode::additive_alpha(),
            BlendMode::multiply(),
            BlendMode::screen(),
            BlendMode::soft_light(),
            BlendMode::replace(),
            BlendMode::subtract(),
            BlendMode::min(),
            BlendMode::max(),
        ];
        assert_eq!(presets.len(), 11);
    }

    #[test]
    fn test_preset_distinctness() {
        // All presets should be distinct from each other
        let presets = [
            BlendMode::alpha(),
            BlendMode::premultiplied_alpha(),
            BlendMode::additive(),
            BlendMode::additive_alpha(),
            BlendMode::multiply(),
            BlendMode::screen(),
            BlendMode::soft_light(),
            BlendMode::replace(),
            BlendMode::subtract(),
            BlendMode::min(),
            BlendMode::max(),
        ];

        for i in 0..presets.len() {
            for j in (i + 1)..presets.len() {
                assert_ne!(
                    presets[i], presets[j],
                    "Presets {} and {} should be distinct",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_preset_consistency_after_conversion() {
        // Presets should survive wgpu roundtrip
        let presets = [
            BlendMode::alpha(),
            BlendMode::premultiplied_alpha(),
            BlendMode::additive(),
            BlendMode::multiply(),
            BlendMode::screen(),
            BlendMode::replace(),
        ];

        for preset in presets {
            let wgpu_state: wgpu::BlendState = preset.into();
            let back: BlendMode = wgpu_state.into();
            assert_eq!(preset, back);
        }
    }

    // ---- BlendFactor Complete Coverage ----

    #[test]
    fn test_blend_factor_zero_behavior() {
        let info = get_blend_factor_info(wgpu::BlendFactor::Zero).unwrap();
        assert_eq!(info.value, "0");
        assert_eq!(info.name, "Zero");
    }

    #[test]
    fn test_blend_factor_one_behavior() {
        let info = get_blend_factor_info(wgpu::BlendFactor::One).unwrap();
        assert_eq!(info.value, "1");
        assert_eq!(info.name, "One");
    }

    #[test]
    fn test_blend_factor_src_variants() {
        let src = get_blend_factor_info(wgpu::BlendFactor::Src).unwrap();
        assert_eq!(src.value, "src");

        let one_minus_src = get_blend_factor_info(wgpu::BlendFactor::OneMinusSrc).unwrap();
        assert_eq!(one_minus_src.value, "1 - src");

        let src_alpha = get_blend_factor_info(wgpu::BlendFactor::SrcAlpha).unwrap();
        assert_eq!(src_alpha.value, "src.a");

        let one_minus_src_alpha = get_blend_factor_info(wgpu::BlendFactor::OneMinusSrcAlpha).unwrap();
        assert_eq!(one_minus_src_alpha.value, "1 - src.a");

        let src_alpha_sat = get_blend_factor_info(wgpu::BlendFactor::SrcAlphaSaturated).unwrap();
        assert_eq!(src_alpha_sat.value, "min(src.a, 1 - dst.a)");
    }

    #[test]
    fn test_blend_factor_dst_variants() {
        let dst = get_blend_factor_info(wgpu::BlendFactor::Dst).unwrap();
        assert_eq!(dst.value, "dst");

        let one_minus_dst = get_blend_factor_info(wgpu::BlendFactor::OneMinusDst).unwrap();
        assert_eq!(one_minus_dst.value, "1 - dst");

        let dst_alpha = get_blend_factor_info(wgpu::BlendFactor::DstAlpha).unwrap();
        assert_eq!(dst_alpha.value, "dst.a");

        let one_minus_dst_alpha = get_blend_factor_info(wgpu::BlendFactor::OneMinusDstAlpha).unwrap();
        assert_eq!(one_minus_dst_alpha.value, "1 - dst.a");
    }

    #[test]
    fn test_blend_factor_constant_variants() {
        let constant = get_blend_factor_info(wgpu::BlendFactor::Constant).unwrap();
        assert_eq!(constant.value, "const");

        let one_minus_constant = get_blend_factor_info(wgpu::BlendFactor::OneMinusConstant).unwrap();
        assert_eq!(one_minus_constant.value, "1 - const");
    }

    #[test]
    fn test_blend_factor_descriptions_not_empty() {
        for info in &BLEND_FACTORS {
            assert!(!info.name.is_empty(), "Factor name should not be empty");
            assert!(
                !info.description.is_empty(),
                "Factor description should not be empty for {}",
                info.name
            );
            assert!(!info.value.is_empty(), "Factor value should not be empty for {}", info.name);
        }
    }

    // ---- BlendOperation Complete Coverage ----

    #[test]
    fn test_blend_operation_add() {
        let info = get_blend_operation_info(wgpu::BlendOperation::Add).unwrap();
        assert_eq!(info.formula, "src + dst");
    }

    #[test]
    fn test_blend_operation_subtract() {
        let info = get_blend_operation_info(wgpu::BlendOperation::Subtract).unwrap();
        assert_eq!(info.formula, "src - dst");
    }

    #[test]
    fn test_blend_operation_reverse_subtract() {
        let info = get_blend_operation_info(wgpu::BlendOperation::ReverseSubtract).unwrap();
        assert_eq!(info.formula, "dst - src");
    }

    #[test]
    fn test_blend_operation_min() {
        let info = get_blend_operation_info(wgpu::BlendOperation::Min).unwrap();
        assert_eq!(info.formula, "min(src, dst)");
    }

    #[test]
    fn test_blend_operation_max() {
        let info = get_blend_operation_info(wgpu::BlendOperation::Max).unwrap();
        assert_eq!(info.formula, "max(src, dst)");
    }

    #[test]
    fn test_blend_operation_descriptions_not_empty() {
        for info in &BLEND_OPERATIONS {
            assert!(!info.name.is_empty());
            assert!(!info.description.is_empty());
            assert!(!info.formula.is_empty());
        }
    }

    // ---- Color/Alpha Separate Configuration ----

    #[test]
    fn test_color_only_additive_alpha_preserve() {
        // Common pattern: additive color, preserve destination alpha
        let mode = BlendModeBuilder::new()
            .color_src_factor(wgpu::BlendFactor::One)
            .color_dst_factor(wgpu::BlendFactor::One)
            .color_operation(wgpu::BlendOperation::Add)
            .alpha_src_factor(wgpu::BlendFactor::Zero)
            .alpha_dst_factor(wgpu::BlendFactor::One)
            .alpha_operation(wgpu::BlendOperation::Add)
            .build();

        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::Zero);
        assert!(!mode.is_uniform());
    }

    #[test]
    fn test_alpha_max_color_add() {
        // Color blends additively, alpha takes max
        let mode = BlendModeBuilder::new()
            .color_operation(wgpu::BlendOperation::Add)
            .alpha_operation(wgpu::BlendOperation::Max)
            .build();

        assert_eq!(mode.color_operation(), wgpu::BlendOperation::Add);
        assert_eq!(mode.alpha_operation(), wgpu::BlendOperation::Max);
    }

    #[test]
    fn test_separate_factors_separate_operations() {
        let mode = BlendModeBuilder::new()
            .color_src_factor(wgpu::BlendFactor::SrcAlpha)
            .color_dst_factor(wgpu::BlendFactor::OneMinusSrcAlpha)
            .color_operation(wgpu::BlendOperation::Add)
            .alpha_src_factor(wgpu::BlendFactor::One)
            .alpha_dst_factor(wgpu::BlendFactor::One)
            .alpha_operation(wgpu::BlendOperation::Max)
            .build();

        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::One);
        assert_ne!(mode.color.operation, mode.alpha.operation);
    }

    // ---- Builder Pattern Chain Combinations ----

    #[test]
    fn test_builder_full_chain() {
        let mode = BlendModeBuilder::new()
            .color_src_factor(wgpu::BlendFactor::SrcAlpha)
            .color_dst_factor(wgpu::BlendFactor::OneMinusSrcAlpha)
            .color_operation(wgpu::BlendOperation::Add)
            .alpha_src_factor(wgpu::BlendFactor::One)
            .alpha_dst_factor(wgpu::BlendFactor::Zero)
            .alpha_operation(wgpu::BlendOperation::Add)
            .build();

        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(mode.color.operation, wgpu::BlendOperation::Add);
        assert_eq!(mode.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::Zero);
        assert_eq!(mode.alpha.operation, wgpu::BlendOperation::Add);
    }

    #[test]
    fn test_builder_override_chain() {
        // Build from preset then override specific values
        let mode = BlendModeBuilder::from_preset(BlendMode::additive())
            .color_src_factor(wgpu::BlendFactor::SrcAlpha)
            .alpha_operation(wgpu::BlendOperation::Max)
            .build();

        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::One); // From additive
        assert_eq!(mode.alpha.operation, wgpu::BlendOperation::Max);
    }

    #[test]
    fn test_builder_uniform_then_override() {
        let component = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };

        let mode = BlendModeBuilder::new()
            .uniform(component)
            .alpha_dst_factor(wgpu::BlendFactor::Zero)
            .build();

        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::One);
        assert_eq!(mode.alpha.dst_factor, wgpu::BlendFactor::Zero);
    }

    #[test]
    fn test_builder_default_impl() {
        let builder1 = BlendModeBuilder::default();
        let builder2 = BlendModeBuilder::new();
        assert_eq!(builder1.build(), builder2.build());
    }

    #[test]
    fn test_builder_clone() {
        let builder = BlendModeBuilder::new()
            .color_src_factor(wgpu::BlendFactor::Constant);
        let cloned = builder.clone();
        assert_eq!(builder.build(), cloned.build());
    }

    // ---- BlendModeInfo Metadata Coverage ----

    #[test]
    fn test_blend_mode_info_all_have_use_cases() {
        for info in &BLEND_MODE_PRESETS {
            assert!(
                !info.use_cases.is_empty(),
                "Preset '{}' should have use cases",
                info.name
            );
        }
    }

    #[test]
    fn test_blend_mode_info_all_have_equations() {
        for info in &BLEND_MODE_PRESETS {
            assert!(
                !info.color_equation.is_empty(),
                "Preset '{}' should have equation",
                info.name
            );
        }
    }

    #[test]
    fn test_blend_mode_info_consistency() {
        // Check that info matches actual preset behavior
        let alpha_info = get_blend_mode_info("Alpha").unwrap();
        let alpha_mode = get_preset("Alpha").unwrap();
        assert_eq!(alpha_info.uses_alpha, alpha_mode.uses_alpha());

        let additive_info = get_blend_mode_info("Additive").unwrap();
        let additive_mode = get_preset("Additive").unwrap();
        assert_eq!(additive_info.can_exceed_one, additive_mode.can_exceed_one());
    }

    #[test]
    fn test_blend_mode_info_constant_flag() {
        let constant_info = get_blend_mode_info("Constant").unwrap();
        assert!(constant_info.uses_constant);

        // Non-constant modes should not have the flag
        let alpha_info = get_blend_mode_info("Alpha").unwrap();
        assert!(!alpha_info.uses_constant);
    }

    // ---- wgpu BlendState Conversion Roundtrip ----

    #[test]
    fn test_wgpu_conversion_all_presets() {
        let test_cases = [
            ("Alpha", BlendMode::alpha()),
            ("Premultiplied", BlendMode::premultiplied_alpha()),
            ("Additive", BlendMode::additive()),
            ("Multiply", BlendMode::multiply()),
            ("Screen", BlendMode::screen()),
            ("Replace", BlendMode::replace()),
            ("Min", BlendMode::min()),
            ("Max", BlendMode::max()),
        ];

        for (name, original) in test_cases {
            let wgpu_state: wgpu::BlendState = original.into();
            let roundtrip: BlendMode = wgpu_state.into();
            assert_eq!(
                original, roundtrip,
                "Roundtrip failed for preset: {}",
                name
            );
        }
    }

    #[test]
    fn test_wgpu_conversion_custom_mode() {
        let custom = BlendModeBuilder::new()
            .color_src_factor(wgpu::BlendFactor::Constant)
            .color_dst_factor(wgpu::BlendFactor::OneMinusConstant)
            .color_operation(wgpu::BlendOperation::Subtract)
            .alpha_src_factor(wgpu::BlendFactor::SrcAlphaSaturated)
            .alpha_dst_factor(wgpu::BlendFactor::DstAlpha)
            .alpha_operation(wgpu::BlendOperation::ReverseSubtract)
            .build();

        let wgpu_state: wgpu::BlendState = custom.into();
        let roundtrip: BlendMode = wgpu_state.into();
        assert_eq!(custom, roundtrip);
    }

    #[test]
    fn test_wgpu_conversion_preserves_components() {
        let mode = BlendMode::alpha();
        let state: wgpu::BlendState = mode.into();

        // Verify component-level equality
        assert_eq!(state.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(state.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(state.color.operation, wgpu::BlendOperation::Add);
        assert_eq!(state.alpha.src_factor, wgpu::BlendFactor::One);
        assert_eq!(state.alpha.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(state.alpha.operation, wgpu::BlendOperation::Add);
    }

    // ---- Edge Cases: Identity and Inverse Operations ----

    #[test]
    fn test_identity_blend_src_one_dst_zero() {
        // src * 1 + dst * 0 = src (identity/replace)
        let mode = BlendModeBuilder::new()
            .src_factor(wgpu::BlendFactor::One)
            .dst_factor(wgpu::BlendFactor::Zero)
            .operation(wgpu::BlendOperation::Add)
            .build();

        assert!(mode.is_replace());
    }

    #[test]
    fn test_identity_blend_src_zero_dst_one() {
        // src * 0 + dst * 1 = dst (no change)
        let mode = BlendModeBuilder::new()
            .src_factor(wgpu::BlendFactor::Zero)
            .dst_factor(wgpu::BlendFactor::One)
            .operation(wgpu::BlendOperation::Add)
            .build();

        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::Zero);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::One);
    }

    #[test]
    fn test_inverse_factors_src_one_minus() {
        // OneMinusSrc is inverse of Src
        let mode = BlendModeBuilder::new()
            .color_src_factor(wgpu::BlendFactor::Src)
            .color_dst_factor(wgpu::BlendFactor::OneMinusSrc)
            .build();

        assert_eq!(mode.color.src_factor, wgpu::BlendFactor::Src);
        assert_eq!(mode.color.dst_factor, wgpu::BlendFactor::OneMinusSrc);
    }

    #[test]
    fn test_inverse_factors_alpha() {
        let mode = BlendModeBuilder::new()
            .alpha_src_factor(wgpu::BlendFactor::SrcAlpha)
            .alpha_dst_factor(wgpu::BlendFactor::OneMinusSrcAlpha)
            .build();

        // These are the standard alpha blending factors
        assert!(mode.uses_alpha());
    }

    #[test]
    fn test_subtract_vs_reverse_subtract() {
        let subtract = BlendModeBuilder::new()
            .operation(wgpu::BlendOperation::Subtract)
            .build();

        let reverse = BlendModeBuilder::new()
            .operation(wgpu::BlendOperation::ReverseSubtract)
            .build();

        assert_ne!(subtract, reverse);
        assert_eq!(subtract.color.operation, wgpu::BlendOperation::Subtract);
        assert_eq!(reverse.color.operation, wgpu::BlendOperation::ReverseSubtract);
    }

    // ---- Display/Debug Trait Implementations ----

    #[test]
    fn test_display_contains_factors() {
        let mode = BlendMode::alpha();
        let display = format!("{}", mode);
        assert!(display.contains("SrcAlpha"), "Display should contain src factor");
    }

    #[test]
    fn test_display_all_presets() {
        let presets = [
            BlendMode::alpha(),
            BlendMode::premultiplied_alpha(),
            BlendMode::additive(),
            BlendMode::multiply(),
            BlendMode::screen(),
            BlendMode::replace(),
        ];

        for preset in presets {
            let display = format!("{}", preset);
            // Should not panic and should contain key info
            assert!(display.contains("BlendMode"));
            assert!(display.contains("color:"));
            assert!(display.contains("alpha:"));
        }
    }

    #[test]
    fn test_debug_format() {
        let mode = BlendMode::additive();
        let debug = format!("{:?}", mode);
        assert!(debug.contains("color"));
        assert!(debug.contains("alpha"));
        assert!(debug.contains("src_factor"));
        assert!(debug.contains("dst_factor"));
    }

    #[test]
    fn test_debug_builder() {
        let builder = BlendModeBuilder::new();
        let debug = format!("{:?}", builder);
        assert!(debug.contains("BlendModeBuilder"));
    }

    #[test]
    fn test_debug_blend_factor_info() {
        let info = &BLEND_FACTORS[0];
        let debug = format!("{:?}", info);
        assert!(debug.contains("BlendFactorInfo"));
    }

    #[test]
    fn test_debug_blend_operation_info() {
        let info = &BLEND_OPERATIONS[0];
        let debug = format!("{:?}", info);
        assert!(debug.contains("BlendOperationInfo"));
    }

    #[test]
    fn test_debug_blend_mode_info() {
        let info = &BLEND_MODE_PRESETS[0];
        let debug = format!("{:?}", info);
        assert!(debug.contains("BlendModeInfo"));
    }

    // ---- Thread Safety Verification ----

    #[test]
    fn test_blend_mode_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<BlendMode>();
    }

    #[test]
    fn test_blend_mode_builder_is_clone() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<BlendModeBuilder>();
    }

    #[test]
    fn test_blend_factor_info_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<BlendFactorInfo>();
    }

    #[test]
    fn test_blend_operation_info_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<BlendOperationInfo>();
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let mode = Arc::new(BlendMode::alpha());
        let mut handles = vec![];

        for _ in 0..4 {
            let mode_clone = Arc::clone(&mode);
            handles.push(thread::spawn(move || {
                // Read operations should be safe
                let _ = mode_clone.is_replace();
                let _ = mode_clone.uses_alpha();
                let _ = mode_clone.can_exceed_one();
                let _state: wgpu::BlendState = (*mode_clone).into();
            }));
        }

        for handle in handles {
            handle.join().expect("Thread should complete");
        }
    }

    // ---- Query Method Edge Cases ----

    #[test]
    fn test_uses_constant_all_positions() {
        // Constant in color src
        let mode1 = BlendModeBuilder::new()
            .color_src_factor(wgpu::BlendFactor::Constant)
            .build();
        assert!(mode1.uses_constant());

        // Constant in color dst
        let mode2 = BlendModeBuilder::new()
            .color_dst_factor(wgpu::BlendFactor::Constant)
            .build();
        assert!(mode2.uses_constant());

        // Constant in alpha src
        let mode3 = BlendModeBuilder::new()
            .alpha_src_factor(wgpu::BlendFactor::Constant)
            .build();
        assert!(mode3.uses_constant());

        // Constant in alpha dst
        let mode4 = BlendModeBuilder::new()
            .alpha_dst_factor(wgpu::BlendFactor::Constant)
            .build();
        assert!(mode4.uses_constant());

        // OneMinusConstant variants
        let mode5 = BlendModeBuilder::new()
            .color_src_factor(wgpu::BlendFactor::OneMinusConstant)
            .build();
        assert!(mode5.uses_constant());
    }

    #[test]
    fn test_uses_alpha_all_factors() {
        let alpha_factors = [
            wgpu::BlendFactor::SrcAlpha,
            wgpu::BlendFactor::OneMinusSrcAlpha,
            wgpu::BlendFactor::DstAlpha,
            wgpu::BlendFactor::OneMinusDstAlpha,
            wgpu::BlendFactor::SrcAlphaSaturated,
        ];

        for factor in alpha_factors {
            let mode = BlendModeBuilder::new()
                .color_src_factor(factor)
                .build();
            assert!(
                mode.uses_alpha(),
                "Mode with {:?} should use alpha",
                factor
            );
        }
    }

    #[test]
    fn test_non_alpha_factors() {
        let non_alpha_factors = [
            wgpu::BlendFactor::Zero,
            wgpu::BlendFactor::One,
            wgpu::BlendFactor::Src,
            wgpu::BlendFactor::OneMinusSrc,
            wgpu::BlendFactor::Dst,
            wgpu::BlendFactor::OneMinusDst,
            wgpu::BlendFactor::Constant,
            wgpu::BlendFactor::OneMinusConstant,
        ];

        for factor in non_alpha_factors {
            let mode = BlendModeBuilder::new()
                .src_factor(factor)
                .dst_factor(factor)
                .build();
            assert!(
                !mode.uses_alpha(),
                "Mode with {:?} should not use alpha",
                factor
            );
        }
    }

    // ---- Preset Lookup Edge Cases ----

    #[test]
    fn test_get_preset_case_sensitive() {
        assert!(get_preset("Alpha").is_some());
        assert!(get_preset("alpha").is_none());
        assert!(get_preset("ALPHA").is_none());
    }

    #[test]
    fn test_get_preset_all_valid_names() {
        let valid_names = [
            "Alpha",
            "Premultiplied Alpha",
            "Additive",
            "Additive Alpha",
            "Multiply",
            "Screen",
            "Soft Light",
            "Replace",
            "Subtract",
            "Min",
            "Max",
        ];

        for name in valid_names {
            assert!(
                get_preset(name).is_some(),
                "Preset '{}' should exist",
                name
            );
        }
    }

    #[test]
    fn test_get_blend_mode_info_not_found() {
        assert!(get_blend_mode_info("NonExistent").is_none());
        assert!(get_blend_mode_info("").is_none());
    }

    #[test]
    fn test_preset_names_iterator_count() {
        let count = preset_names().count();
        assert_eq!(count, BLEND_MODE_PRESETS.len());
    }

    // ---- Modifier Method Chaining ----

    #[test]
    fn test_modifier_chain_preserves_other_fields() {
        let original = BlendMode::additive();

        let modified = original
            .with_color_src_factor(wgpu::BlendFactor::SrcAlpha);

        // Only color src should change
        assert_eq!(modified.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(modified.color.dst_factor, original.color.dst_factor);
        assert_eq!(modified.color.operation, original.color.operation);
        assert_eq!(modified.alpha, original.alpha);
    }

    #[test]
    fn test_modifier_multiple_changes() {
        let mode = BlendMode::replace()
            .with_color_src_factor(wgpu::BlendFactor::SrcAlpha)
            .with_color_dst_factor(wgpu::BlendFactor::OneMinusSrcAlpha)
            .with_alpha_src_factor(wgpu::BlendFactor::One)
            .with_alpha_dst_factor(wgpu::BlendFactor::OneMinusSrcAlpha);

        // Should match alpha preset
        assert_eq!(mode, BlendMode::alpha());
    }

    // ---- Iterator Functions ----

    #[test]
    fn test_hdr_presets_contains_additive() {
        let hdr: Vec<_> = hdr_presets().collect();
        assert!(hdr.iter().any(|info| info.name == "Additive"));
        assert!(hdr.iter().any(|info| info.name == "Additive Alpha"));
    }

    #[test]
    fn test_alpha_presets_contains_expected() {
        let alpha: Vec<_> = alpha_presets().collect();
        assert!(alpha.iter().any(|info| info.name == "Alpha"));
        assert!(alpha.iter().any(|info| info.name == "Premultiplied Alpha"));
        assert!(alpha.iter().any(|info| info.name == "Additive Alpha"));
    }

    #[test]
    fn test_constant_presets_contains_constant() {
        let constant: Vec<_> = constant_presets().collect();
        assert!(constant.iter().any(|info| info.name == "Constant"));
    }

    // ---- PartialEq for Info Types ----

    #[test]
    fn test_blend_factor_info_equality() {
        let info1 = &BLEND_FACTORS[0];
        let info2 = &BLEND_FACTORS[0];
        let info3 = &BLEND_FACTORS[1];

        assert_eq!(info1, info2);
        assert_ne!(info1, info3);
    }

    #[test]
    fn test_blend_operation_info_equality() {
        let info1 = &BLEND_OPERATIONS[0];
        let info2 = &BLEND_OPERATIONS[0];
        let info3 = &BLEND_OPERATIONS[1];

        assert_eq!(info1, info2);
        assert_ne!(info1, info3);
    }

    #[test]
    fn test_blend_mode_info_equality() {
        let info1 = &BLEND_MODE_PRESETS[0];
        let info2 = &BLEND_MODE_PRESETS[0];
        let info3 = &BLEND_MODE_PRESETS[1];

        assert_eq!(info1, info2);
        assert_ne!(info1, info3);
    }

    // ---- Const Function Tests ----

    #[test]
    fn test_const_new() {
        const MODE: BlendMode = BlendMode::new(
            wgpu::BlendComponent::REPLACE,
            wgpu::BlendComponent::REPLACE,
        );
        assert!(MODE.is_replace());
    }

    #[test]
    fn test_const_uniform() {
        const MODE: BlendMode = BlendMode::uniform(wgpu::BlendComponent::REPLACE);
        assert!(MODE.is_uniform());
    }

    #[test]
    fn test_const_presets() {
        // All preset constructors should be const
        const ALPHA: BlendMode = BlendMode::alpha();
        const PREMULT: BlendMode = BlendMode::premultiplied_alpha();
        const ADD: BlendMode = BlendMode::additive();
        const ADD_ALPHA: BlendMode = BlendMode::additive_alpha();
        const MULT: BlendMode = BlendMode::multiply();
        const SCREEN: BlendMode = BlendMode::screen();
        const SOFT: BlendMode = BlendMode::soft_light();
        const REPLACE: BlendMode = BlendMode::replace();
        const SUB: BlendMode = BlendMode::subtract();
        const MIN: BlendMode = BlendMode::min();
        const MAX: BlendMode = BlendMode::max();

        // Verify they're valid
        assert!(ALPHA.uses_alpha());
        assert!(PREMULT.uses_alpha());
        assert!(ADD.is_additive());
        assert!(ADD_ALPHA.uses_alpha());
        assert!(MULT.only_darkens());
        assert!(SCREEN.only_lightens());
        assert!(!SOFT.is_uniform());
        assert!(REPLACE.is_replace());
        assert_eq!(SUB.color.operation, wgpu::BlendOperation::ReverseSubtract);
        assert_eq!(MIN.color.operation, wgpu::BlendOperation::Min);
        assert_eq!(MAX.color.operation, wgpu::BlendOperation::Max);
    }
}
