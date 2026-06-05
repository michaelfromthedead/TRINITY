//! Sampler creation and management for TRINITY.
//!
//! This module provides the sampler creation API for the TRINITY wgpu abstraction layer.
//! It wraps wgpu's sampler creation with validation, builder patterns, and common presets.
//!
//! # Overview
//!
//! Sampler creation in wgpu requires specifying address modes, filter modes, and other
//! parameters. This module provides:
//!
//! - [`TrinitySampler`] - Wrapper around wgpu::Sampler with metadata
//! - [`TrinitySamplerDescriptor`] - Sampler creation parameters with builder pattern
//! - [`create_sampler`] - Validated sampler creation
//! - Preset methods for common sampler configurations
//!
//! # Address Modes
//!
//! | Mode | Behavior |
//! |------|----------|
//! | ClampToEdge | Coordinates clamped to [0, 1] |
//! | Repeat | Coordinates wrap around |
//! | MirrorRepeat | Coordinates mirror at boundaries |
//! | ClampToBorder | Out-of-bounds returns border color |
//!
//! # Filter Modes
//!
//! | Mode | Behavior |
//! |------|----------|
//! | Nearest | Nearest texel (pixelated) |
//! | Linear | Bilinear interpolation (smooth) |
//!
//! # Anisotropic Filtering
//!
//! Anisotropy improves texture quality at oblique viewing angles.
//! Values typically range from 1 (disabled) to 16 (maximum quality).
//! The actual max is device-dependent (`limits.max_sampler_anisotropy`).
//!
//! # Comparison Samplers
//!
//! Used for shadow mapping, comparison samplers return 0.0 or 1.0 based on
//! comparing the sampled value against a reference. Typically used with
//! depth textures and `CompareFunction::Less` or `LessEqual`.
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::sampler::{TrinitySamplerDescriptor, create_sampler};
//! use wgpu::{AddressMode, FilterMode};
//!
//! # fn example(device: &wgpu::Device) {
//! // Using builder pattern
//! let desc = TrinitySamplerDescriptor::new()
//!     .label("diffuse_sampler")
//!     .filter(FilterMode::Linear)
//!     .address_mode(AddressMode::Repeat)
//!     .anisotropy(8);
//!
//! let sampler = create_sampler(device, &desc);
//!
//! // Using presets
//! let linear_sampler = create_sampler(device, &TrinitySamplerDescriptor::linear_clamp());
//! let shadow_sampler = create_sampler(device, &TrinitySamplerDescriptor::shadow());
//! # }
//! ```

use log::debug;
use std::fmt;
use wgpu::{Device, Sampler, SamplerDescriptor};

// Re-export wgpu types for convenience
pub use wgpu::{AddressMode, CompareFunction, FilterMode, SamplerBorderColor};

// ============================================================================
// TrinitySamplerDescriptor
// ============================================================================

/// Sampler creation descriptor with builder pattern.
///
/// This struct describes the parameters for creating a new sampler.
/// All fields have sensible defaults (linear filtering, clamp to edge).
///
/// # Builder Pattern
///
/// The descriptor supports method chaining for convenient configuration:
///
/// ```no_run
/// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
/// use wgpu::{AddressMode, FilterMode};
///
/// let desc = TrinitySamplerDescriptor::new()
///     .label("my_sampler")
///     .filter(FilterMode::Linear)
///     .address_mode(AddressMode::Repeat)
///     .anisotropy(4);
/// ```
///
/// # Presets
///
/// Common configurations are available as static methods:
///
/// - [`linear_clamp()`](Self::linear_clamp) - Linear filtering, clamp to edge
/// - [`linear_repeat()`](Self::linear_repeat) - Linear filtering, repeat wrapping
/// - [`nearest_clamp()`](Self::nearest_clamp) - Nearest filtering, clamp to edge
/// - [`nearest_repeat()`](Self::nearest_repeat) - Nearest filtering, repeat wrapping
/// - [`trilinear()`](Self::trilinear) - Linear with mipmap linear
/// - [`shadow()`](Self::shadow) - Comparison sampler for shadow maps
#[derive(Debug, Clone)]
pub struct TrinitySamplerDescriptor {
    /// Debug label for the sampler.
    pub label: Option<String>,

    /// Address mode for the U (horizontal) texture coordinate.
    pub address_mode_u: AddressMode,

    /// Address mode for the V (vertical) texture coordinate.
    pub address_mode_v: AddressMode,

    /// Address mode for the W (depth) texture coordinate.
    pub address_mode_w: AddressMode,

    /// Magnification filter (used when texture is larger than sampled area).
    pub mag_filter: FilterMode,

    /// Minification filter (used when texture is smaller than sampled area).
    pub min_filter: FilterMode,

    /// Mipmap filter (used when selecting between mip levels).
    pub mipmap_filter: FilterMode,

    /// Minimum LOD (level of detail) clamp.
    ///
    /// Restricts sampling to mip levels >= this value.
    /// Default: 0.0
    pub lod_min_clamp: f32,

    /// Maximum LOD (level of detail) clamp.
    ///
    /// Restricts sampling to mip levels <= this value.
    /// Default: 32.0 (effectively no clamping)
    pub lod_max_clamp: f32,

    /// Comparison function for comparison samplers.
    ///
    /// When set, the sampler compares the sampled depth value against
    /// a reference value, returning 0.0 or 1.0.
    /// Used for shadow mapping.
    pub compare: Option<CompareFunction>,

    /// Maximum anisotropy level.
    ///
    /// Higher values improve quality at oblique viewing angles.
    /// - 1 = disabled
    /// - 2-16 = typical values
    /// - Value is clamped to device limit
    pub anisotropy_clamp: u16,

    /// Border color for ClampToBorder address mode.
    ///
    /// Only used when address mode is ClampToBorder.
    /// Options: TransparentBlack, OpaqueBlack, OpaqueWhite
    pub border_color: Option<SamplerBorderColor>,
}

impl Default for TrinitySamplerDescriptor {
    fn default() -> Self {
        Self {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        }
    }
}

impl TrinitySamplerDescriptor {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Creates a new descriptor with default settings.
    ///
    /// Default configuration:
    /// - Linear filtering (mag, min, mipmap)
    /// - Clamp to edge address mode
    /// - No anisotropy
    /// - No comparison function
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    ///
    /// let desc = TrinitySamplerDescriptor::new();
    /// assert_eq!(desc.mag_filter, wgpu::FilterMode::Linear);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    // ========================================================================
    // Builder Methods
    // ========================================================================

    /// Sets the debug label.
    ///
    /// The label appears in GPU debugging tools and error messages.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    ///
    /// let desc = TrinitySamplerDescriptor::new().label("diffuse_sampler");
    /// assert_eq!(desc.label.as_deref(), Some("diffuse_sampler"));
    /// ```
    #[inline]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Sets all three address modes (U, V, W) to the same value.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use wgpu::AddressMode;
    ///
    /// let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::Repeat);
    /// assert_eq!(desc.address_mode_u, AddressMode::Repeat);
    /// assert_eq!(desc.address_mode_v, AddressMode::Repeat);
    /// assert_eq!(desc.address_mode_w, AddressMode::Repeat);
    /// ```
    #[inline]
    pub fn address_mode(mut self, mode: AddressMode) -> Self {
        self.address_mode_u = mode;
        self.address_mode_v = mode;
        self.address_mode_w = mode;
        self
    }

    /// Sets address modes for U, V, W coordinates separately.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use wgpu::AddressMode;
    ///
    /// let desc = TrinitySamplerDescriptor::new()
    ///     .address_mode_uvw(AddressMode::Repeat, AddressMode::ClampToEdge, AddressMode::MirrorRepeat);
    /// assert_eq!(desc.address_mode_u, AddressMode::Repeat);
    /// assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
    /// assert_eq!(desc.address_mode_w, AddressMode::MirrorRepeat);
    /// ```
    #[inline]
    pub fn address_mode_uvw(mut self, u: AddressMode, v: AddressMode, w: AddressMode) -> Self {
        self.address_mode_u = u;
        self.address_mode_v = v;
        self.address_mode_w = w;
        self
    }

    /// Sets all three filter modes (mag, min, mipmap) to the same value.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use wgpu::FilterMode;
    ///
    /// let desc = TrinitySamplerDescriptor::new().filter(FilterMode::Nearest);
    /// assert_eq!(desc.mag_filter, FilterMode::Nearest);
    /// assert_eq!(desc.min_filter, FilterMode::Nearest);
    /// assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    /// ```
    #[inline]
    pub fn filter(mut self, filter: FilterMode) -> Self {
        self.mag_filter = filter;
        self.min_filter = filter;
        self.mipmap_filter = filter;
        self
    }

    /// Sets filter modes for mag, min, and mipmap separately.
    ///
    /// # Arguments
    ///
    /// * `mag` - Magnification filter
    /// * `min` - Minification filter
    /// * `mip` - Mipmap filter
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use wgpu::FilterMode;
    ///
    /// let desc = TrinitySamplerDescriptor::new()
    ///     .filter_separate(FilterMode::Linear, FilterMode::Linear, FilterMode::Nearest);
    /// assert_eq!(desc.mag_filter, FilterMode::Linear);
    /// assert_eq!(desc.min_filter, FilterMode::Linear);
    /// assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    /// ```
    #[inline]
    pub fn filter_separate(mut self, mag: FilterMode, min: FilterMode, mip: FilterMode) -> Self {
        self.mag_filter = mag;
        self.min_filter = min;
        self.mipmap_filter = mip;
        self
    }

    /// Sets the LOD (level of detail) clamp range.
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum LOD (default: 0.0)
    /// * `max` - Maximum LOD (default: 32.0)
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    ///
    /// // Only use mip levels 0-4
    /// let desc = TrinitySamplerDescriptor::new().lod_clamp(0.0, 4.0);
    /// assert_eq!(desc.lod_min_clamp, 0.0);
    /// assert_eq!(desc.lod_max_clamp, 4.0);
    /// ```
    #[inline]
    pub fn lod_clamp(mut self, min: f32, max: f32) -> Self {
        self.lod_min_clamp = min;
        self.lod_max_clamp = max;
        self
    }

    /// Sets the maximum anisotropy level.
    ///
    /// Higher values improve texture quality at oblique angles but cost performance.
    /// The value is clamped to the device's `max_sampler_anisotropy` limit.
    ///
    /// # Arguments
    ///
    /// * `max_anisotropy` - Maximum anisotropy (1 = disabled, 2-16 typical)
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    ///
    /// let desc = TrinitySamplerDescriptor::new().anisotropy(8);
    /// assert_eq!(desc.anisotropy_clamp, 8);
    /// ```
    #[inline]
    pub fn anisotropy(mut self, max_anisotropy: u16) -> Self {
        self.anisotropy_clamp = max_anisotropy;
        self
    }

    /// Sets the comparison function for comparison samplers.
    ///
    /// When set, the sampler performs depth comparison instead of normal sampling.
    /// Used for shadow mapping with depth textures.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use wgpu::CompareFunction;
    ///
    /// let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Less);
    /// assert_eq!(desc.compare, Some(CompareFunction::Less));
    /// ```
    #[inline]
    pub fn compare(mut self, func: CompareFunction) -> Self {
        self.compare = Some(func);
        self
    }

    /// Sets the border color for ClampToBorder address mode.
    ///
    /// Only meaningful when using `AddressMode::ClampToBorder`.
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use wgpu::{AddressMode, SamplerBorderColor};
    ///
    /// let desc = TrinitySamplerDescriptor::new()
    ///     .address_mode(AddressMode::ClampToBorder)
    ///     .border_color(SamplerBorderColor::OpaqueBlack);
    /// assert_eq!(desc.border_color, Some(SamplerBorderColor::OpaqueBlack));
    /// ```
    #[inline]
    pub fn border_color(mut self, color: SamplerBorderColor) -> Self {
        self.border_color = Some(color);
        self
    }

    // ========================================================================
    // Presets
    // ========================================================================

    /// Linear filtering with clamp to edge.
    ///
    /// Good default for most textures (smooth, no wrapping).
    ///
    /// Configuration:
    /// - Filter: Linear (all)
    /// - Address: ClampToEdge (all)
    /// - Mipmap: Linear
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use wgpu::{AddressMode, FilterMode};
    ///
    /// let desc = TrinitySamplerDescriptor::linear_clamp();
    /// assert_eq!(desc.mag_filter, FilterMode::Linear);
    /// assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
    /// ```
    #[inline]
    pub fn linear_clamp() -> Self {
        Self::new()
            .label("linear_clamp")
            .filter(FilterMode::Linear)
            .address_mode(AddressMode::ClampToEdge)
    }

    /// Linear filtering with repeat wrapping.
    ///
    /// Good for tiled/repeating textures.
    ///
    /// Configuration:
    /// - Filter: Linear (all)
    /// - Address: Repeat (all)
    /// - Mipmap: Linear
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use wgpu::{AddressMode, FilterMode};
    ///
    /// let desc = TrinitySamplerDescriptor::linear_repeat();
    /// assert_eq!(desc.mag_filter, FilterMode::Linear);
    /// assert_eq!(desc.address_mode_u, AddressMode::Repeat);
    /// ```
    #[inline]
    pub fn linear_repeat() -> Self {
        Self::new()
            .label("linear_repeat")
            .filter(FilterMode::Linear)
            .address_mode(AddressMode::Repeat)
    }

    /// Nearest (point) filtering with clamp to edge.
    ///
    /// Good for pixel art or when sharp texel boundaries are desired.
    ///
    /// Configuration:
    /// - Filter: Nearest (all)
    /// - Address: ClampToEdge (all)
    /// - Mipmap: Nearest
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use wgpu::{AddressMode, FilterMode};
    ///
    /// let desc = TrinitySamplerDescriptor::nearest_clamp();
    /// assert_eq!(desc.mag_filter, FilterMode::Nearest);
    /// assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
    /// ```
    #[inline]
    pub fn nearest_clamp() -> Self {
        Self::new()
            .label("nearest_clamp")
            .filter(FilterMode::Nearest)
            .address_mode(AddressMode::ClampToEdge)
    }

    /// Nearest (point) filtering with repeat wrapping.
    ///
    /// Good for pixel art tiles.
    ///
    /// Configuration:
    /// - Filter: Nearest (all)
    /// - Address: Repeat (all)
    /// - Mipmap: Nearest
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use wgpu::{AddressMode, FilterMode};
    ///
    /// let desc = TrinitySamplerDescriptor::nearest_repeat();
    /// assert_eq!(desc.mag_filter, FilterMode::Nearest);
    /// assert_eq!(desc.address_mode_u, AddressMode::Repeat);
    /// ```
    #[inline]
    pub fn nearest_repeat() -> Self {
        Self::new()
            .label("nearest_repeat")
            .filter(FilterMode::Nearest)
            .address_mode(AddressMode::Repeat)
    }

    /// Comparison sampler for shadow mapping.
    ///
    /// Configured for typical shadow map sampling with depth comparison.
    ///
    /// Configuration:
    /// - Filter: Linear (enables PCF when supported)
    /// - Address: ClampToEdge (avoid sampling outside shadow map)
    /// - Compare: Less (shadow test: is fragment closer than shadow map?)
    /// - Mipmap: Nearest (shadow maps typically don't use mipmaps)
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use wgpu::{AddressMode, CompareFunction, FilterMode};
    ///
    /// let desc = TrinitySamplerDescriptor::shadow();
    /// assert_eq!(desc.compare, Some(CompareFunction::Less));
    /// assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
    /// ```
    #[inline]
    pub fn shadow() -> Self {
        Self::new()
            .label("shadow")
            .filter_separate(FilterMode::Linear, FilterMode::Linear, FilterMode::Nearest)
            .address_mode(AddressMode::ClampToEdge)
            .compare(CompareFunction::Less)
    }

    /// Trilinear filtering sampler.
    ///
    /// Linear filtering with linear mipmap blending for smooth transitions
    /// between mip levels.
    ///
    /// Configuration:
    /// - Mag/Min Filter: Linear
    /// - Mipmap Filter: Linear (enables trilinear)
    /// - Address: ClampToEdge
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::sampler::TrinitySamplerDescriptor;
    /// use wgpu::FilterMode;
    ///
    /// let desc = TrinitySamplerDescriptor::trilinear();
    /// assert_eq!(desc.mag_filter, FilterMode::Linear);
    /// assert_eq!(desc.min_filter, FilterMode::Linear);
    /// assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    /// ```
    #[inline]
    pub fn trilinear() -> Self {
        Self::new()
            .label("trilinear")
            .filter(FilterMode::Linear)
            .address_mode(AddressMode::ClampToEdge)
    }
}

// ============================================================================
// Validation
// ============================================================================

/// Errors that can occur during sampler descriptor validation.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplerValidationError {
    /// Requested anisotropy exceeds device limit.
    InvalidAnisotropy {
        /// The requested anisotropy value.
        requested: u16,
        /// The maximum supported by the device.
        max_supported: u16,
    },

    /// LOD range is invalid (min > max).
    InvalidLodRange {
        /// The minimum LOD value.
        min: f32,
        /// The maximum LOD value.
        max: f32,
    },

    /// Border color specified without ClampToBorder address mode.
    BorderColorRequiresClampToBorder,

    /// LOD values must be non-negative.
    NegativeLod {
        /// The invalid LOD value.
        value: f32,
    },
}

impl fmt::Display for SamplerValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAnisotropy {
                requested,
                max_supported,
            } => {
                write!(
                    f,
                    "anisotropy {} exceeds device limit {}",
                    requested, max_supported
                )
            }
            Self::InvalidLodRange { min, max } => {
                write!(f, "invalid LOD range: min {} > max {}", min, max)
            }
            Self::BorderColorRequiresClampToBorder => {
                write!(
                    f,
                    "border_color requires at least one address mode to be ClampToBorder"
                )
            }
            Self::NegativeLod { value } => {
                write!(f, "LOD value {} must be non-negative", value)
            }
        }
    }
}

impl std::error::Error for SamplerValidationError {}

/// Validates a sampler descriptor.
///
/// This function checks:
/// - Anisotropy is in valid range (1-16)
/// - LOD range is valid (min <= max, both >= 0)
/// - Border color is only used with ClampToBorder address mode
///
/// Note: wgpu internally clamps anisotropy to device limits, so we only
/// validate that the value is in the standard range (1-16).
///
/// # Arguments
///
/// * `desc` - The sampler descriptor to validate
///
/// # Returns
///
/// `Ok(())` if valid, or `Err(SamplerValidationError)` describing the issue.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::sampler::{TrinitySamplerDescriptor, validate_descriptor};
///
/// let desc = TrinitySamplerDescriptor::new().anisotropy(8);
///
/// match validate_descriptor(&desc) {
///     Ok(()) => println!("Descriptor is valid"),
///     Err(e) => println!("Validation failed: {}", e),
/// }
/// ```
pub fn validate_descriptor(desc: &TrinitySamplerDescriptor) -> Result<(), SamplerValidationError> {
    // Check anisotropy range (wgpu clamps internally, but we validate sane input)
    // Standard range is 1-16, though some hardware supports up to 32
    if desc.anisotropy_clamp > 16 {
        return Err(SamplerValidationError::InvalidAnisotropy {
            requested: desc.anisotropy_clamp,
            max_supported: 16,
        });
    }

    // Check LOD range
    if desc.lod_min_clamp < 0.0 {
        return Err(SamplerValidationError::NegativeLod {
            value: desc.lod_min_clamp,
        });
    }
    if desc.lod_max_clamp < 0.0 {
        return Err(SamplerValidationError::NegativeLod {
            value: desc.lod_max_clamp,
        });
    }
    if desc.lod_min_clamp > desc.lod_max_clamp {
        return Err(SamplerValidationError::InvalidLodRange {
            min: desc.lod_min_clamp,
            max: desc.lod_max_clamp,
        });
    }

    // Check border color usage
    if desc.border_color.is_some() {
        let uses_clamp_to_border = desc.address_mode_u == AddressMode::ClampToBorder
            || desc.address_mode_v == AddressMode::ClampToBorder
            || desc.address_mode_w == AddressMode::ClampToBorder;

        if !uses_clamp_to_border {
            return Err(SamplerValidationError::BorderColorRequiresClampToBorder);
        }
    }

    Ok(())
}

// ============================================================================
// TrinitySampler
// ============================================================================

/// TRINITY sampler wrapper with metadata.
///
/// This struct wraps a wgpu [`Sampler`] with its creation descriptor,
/// allowing inspection of sampler configuration after creation.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::sampler::{TrinitySamplerDescriptor, create_sampler};
/// use wgpu::FilterMode;
///
/// # fn example(device: &wgpu::Device) {
/// let sampler = create_sampler(device, &TrinitySamplerDescriptor::linear_clamp());
///
/// // Access the underlying wgpu sampler
/// let wgpu_sampler = sampler.inner();
///
/// // Check the configuration
/// assert_eq!(sampler.descriptor().mag_filter, FilterMode::Linear);
/// # }
/// ```
pub struct TrinitySampler {
    /// The underlying wgpu sampler.
    sampler: Sampler,
    /// The descriptor used to create this sampler.
    descriptor: TrinitySamplerDescriptor,
}

impl TrinitySampler {
    /// Returns a reference to the underlying wgpu sampler.
    ///
    /// Use this to bind the sampler in render passes or bind groups.
    #[inline]
    pub fn inner(&self) -> &Sampler {
        &self.sampler
    }

    /// Returns a reference to the descriptor used to create this sampler.
    ///
    /// Useful for inspecting sampler configuration after creation.
    #[inline]
    pub fn descriptor(&self) -> &TrinitySamplerDescriptor {
        &self.descriptor
    }

    /// Returns the debug label, if any.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.descriptor.label.as_deref()
    }

    /// Returns the magnification filter mode.
    #[inline]
    pub fn mag_filter(&self) -> FilterMode {
        self.descriptor.mag_filter
    }

    /// Returns the minification filter mode.
    #[inline]
    pub fn min_filter(&self) -> FilterMode {
        self.descriptor.min_filter
    }

    /// Returns the mipmap filter mode.
    #[inline]
    pub fn mipmap_filter(&self) -> FilterMode {
        self.descriptor.mipmap_filter
    }

    /// Returns the U address mode.
    #[inline]
    pub fn address_mode_u(&self) -> AddressMode {
        self.descriptor.address_mode_u
    }

    /// Returns the V address mode.
    #[inline]
    pub fn address_mode_v(&self) -> AddressMode {
        self.descriptor.address_mode_v
    }

    /// Returns the W address mode.
    #[inline]
    pub fn address_mode_w(&self) -> AddressMode {
        self.descriptor.address_mode_w
    }

    /// Returns the anisotropy clamp value.
    #[inline]
    pub fn anisotropy_clamp(&self) -> u16 {
        self.descriptor.anisotropy_clamp
    }

    /// Returns the comparison function, if any.
    #[inline]
    pub fn compare(&self) -> Option<CompareFunction> {
        self.descriptor.compare
    }

    /// Returns true if this is a comparison sampler (for shadow mapping).
    #[inline]
    pub fn is_comparison_sampler(&self) -> bool {
        self.descriptor.compare.is_some()
    }

    /// Returns true if anisotropic filtering is enabled.
    #[inline]
    pub fn is_anisotropic(&self) -> bool {
        self.descriptor.anisotropy_clamp > 1
    }

    /// Consumes the wrapper and returns the inner wgpu sampler.
    #[inline]
    pub fn into_inner(self) -> Sampler {
        self.sampler
    }
}

impl fmt::Debug for TrinitySampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrinitySampler")
            .field("label", &self.descriptor.label)
            .field("mag_filter", &self.descriptor.mag_filter)
            .field("min_filter", &self.descriptor.min_filter)
            .field("mipmap_filter", &self.descriptor.mipmap_filter)
            .field("address_mode_u", &self.descriptor.address_mode_u)
            .field("address_mode_v", &self.descriptor.address_mode_v)
            .field("address_mode_w", &self.descriptor.address_mode_w)
            .field("anisotropy_clamp", &self.descriptor.anisotropy_clamp)
            .field("compare", &self.descriptor.compare)
            .finish()
    }
}

// ============================================================================
// Sampler Creation
// ============================================================================

/// Creates a sampler with validation and logging.
///
/// This function creates a wgpu sampler with the specified parameters.
/// Anisotropy values are clamped to the standard range (1-16).
/// Note: wgpu internally clamps to device-specific limits.
///
/// # Arguments
///
/// * `device` - The wgpu device to create the sampler on
/// * `desc` - The sampler descriptor specifying parameters
///
/// # Returns
///
/// A [`TrinitySampler`] wrapping the created wgpu sampler.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::sampler::{TrinitySamplerDescriptor, create_sampler};
/// use wgpu::{AddressMode, FilterMode};
///
/// # fn example(device: &wgpu::Device) {
/// let sampler = create_sampler(device, &TrinitySamplerDescriptor::new()
///     .label("my_sampler")
///     .filter(FilterMode::Linear)
///     .address_mode(AddressMode::Repeat)
///     .anisotropy(8));
///
/// // Use sampler.inner() to bind in a bind group
/// # }
/// ```
pub fn create_sampler(device: &Device, desc: &TrinitySamplerDescriptor) -> TrinitySampler {
    let label = desc.label.as_deref();

    // Clamp anisotropy to standard range (1-16)
    // wgpu will further clamp to device-specific limits internally
    let anisotropy = desc.anisotropy_clamp.clamp(1, 16);

    if desc.anisotropy_clamp > 16 {
        debug!(
            "Clamping anisotropy from {} to 16 (standard max) for sampler {:?}",
            desc.anisotropy_clamp, label
        );
    }

    let wgpu_desc = SamplerDescriptor {
        label,
        address_mode_u: desc.address_mode_u,
        address_mode_v: desc.address_mode_v,
        address_mode_w: desc.address_mode_w,
        mag_filter: desc.mag_filter,
        min_filter: desc.min_filter,
        mipmap_filter: desc.mipmap_filter,
        lod_min_clamp: desc.lod_min_clamp,
        lod_max_clamp: desc.lod_max_clamp,
        compare: desc.compare,
        anisotropy_clamp: anisotropy,
        border_color: desc.border_color,
    };

    let sampler = device.create_sampler(&wgpu_desc);

    debug!(
        "Created sampler {:?}: mag={:?}, min={:?}, mip={:?}, addr_u={:?}, aniso={}{}",
        label,
        desc.mag_filter,
        desc.min_filter,
        desc.mipmap_filter,
        desc.address_mode_u,
        anisotropy,
        if desc.compare.is_some() {
            ", comparison"
        } else {
            ""
        }
    );

    // Store the clamped anisotropy in the descriptor copy
    let mut stored_desc = desc.clone();
    stored_desc.anisotropy_clamp = anisotropy;

    TrinitySampler {
        sampler,
        descriptor: stored_desc,
    }
}

/// Creates a sampler with explicit validation.
///
/// Unlike [`create_sampler`], this function returns an error if validation fails
/// instead of automatically clamping values.
///
/// # Arguments
///
/// * `device` - The wgpu device to create the sampler on
/// * `desc` - The sampler descriptor specifying parameters
///
/// # Returns
///
/// `Ok(TrinitySampler)` on success, or `Err(SamplerValidationError)` if validation fails.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::sampler::{TrinitySamplerDescriptor, try_create_sampler};
///
/// # fn example(device: &wgpu::Device) {
/// let desc = TrinitySamplerDescriptor::new().anisotropy(16);
///
/// match try_create_sampler(device, &desc) {
///     Ok(sampler) => println!("Created sampler"),
///     Err(e) => println!("Failed to create sampler: {}", e),
/// }
/// # }
/// ```
pub fn try_create_sampler(
    device: &Device,
    desc: &TrinitySamplerDescriptor,
) -> Result<TrinitySampler, SamplerValidationError> {
    validate_descriptor(desc)?;

    // Validation passed, create the sampler
    Ok(create_sampler(device, desc))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Default Value Tests
    // ========================================================================

    #[test]
    fn test_default_descriptor() {
        let desc = TrinitySamplerDescriptor::default();

        assert!(desc.label.is_none());
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToEdge);
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
        assert_eq!(desc.lod_min_clamp, 0.0);
        assert_eq!(desc.lod_max_clamp, 32.0);
        assert!(desc.compare.is_none());
        assert_eq!(desc.anisotropy_clamp, 1);
        assert!(desc.border_color.is_none());
    }

    #[test]
    fn test_new_equals_default() {
        let new = TrinitySamplerDescriptor::new();
        let default = TrinitySamplerDescriptor::default();

        assert_eq!(new.address_mode_u, default.address_mode_u);
        assert_eq!(new.mag_filter, default.mag_filter);
        assert_eq!(new.anisotropy_clamp, default.anisotropy_clamp);
    }

    // ========================================================================
    // Builder Pattern Tests
    // ========================================================================

    #[test]
    fn test_builder_label() {
        let desc = TrinitySamplerDescriptor::new().label("test_sampler");
        assert_eq!(desc.label.as_deref(), Some("test_sampler"));
    }

    #[test]
    fn test_builder_label_string() {
        let name = String::from("dynamic_name");
        let desc = TrinitySamplerDescriptor::new().label(name);
        assert_eq!(desc.label.as_deref(), Some("dynamic_name"));
    }

    #[test]
    fn test_builder_address_mode_all() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::Repeat);

        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
        assert_eq!(desc.address_mode_v, AddressMode::Repeat);
        assert_eq!(desc.address_mode_w, AddressMode::Repeat);
    }

    #[test]
    fn test_builder_address_mode_uvw() {
        let desc = TrinitySamplerDescriptor::new().address_mode_uvw(
            AddressMode::Repeat,
            AddressMode::MirrorRepeat,
            AddressMode::ClampToEdge,
        );

        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
        assert_eq!(desc.address_mode_v, AddressMode::MirrorRepeat);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToEdge);
    }

    #[test]
    fn test_builder_filter_all() {
        let desc = TrinitySamplerDescriptor::new().filter(FilterMode::Nearest);

        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_builder_filter_separate() {
        let desc = TrinitySamplerDescriptor::new().filter_separate(
            FilterMode::Linear,
            FilterMode::Nearest,
            FilterMode::Linear,
        );

        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    #[test]
    fn test_builder_lod_clamp() {
        let desc = TrinitySamplerDescriptor::new().lod_clamp(1.0, 8.0);

        assert_eq!(desc.lod_min_clamp, 1.0);
        assert_eq!(desc.lod_max_clamp, 8.0);
    }

    #[test]
    fn test_builder_anisotropy() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(8);
        assert_eq!(desc.anisotropy_clamp, 8);
    }

    #[test]
    fn test_builder_compare() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::LessEqual);
        assert_eq!(desc.compare, Some(CompareFunction::LessEqual));
    }

    #[test]
    fn test_builder_border_color() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::OpaqueWhite);

        assert_eq!(desc.border_color, Some(SamplerBorderColor::OpaqueWhite));
    }

    #[test]
    fn test_builder_chaining() {
        let desc = TrinitySamplerDescriptor::new()
            .label("chained")
            .filter(FilterMode::Linear)
            .address_mode(AddressMode::Repeat)
            .anisotropy(4)
            .lod_clamp(0.0, 10.0);

        assert_eq!(desc.label.as_deref(), Some("chained"));
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
        assert_eq!(desc.anisotropy_clamp, 4);
        assert_eq!(desc.lod_max_clamp, 10.0);
    }

    // ========================================================================
    // Preset Tests
    // ========================================================================

    #[test]
    fn test_preset_linear_clamp() {
        let desc = TrinitySamplerDescriptor::linear_clamp();

        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_v, AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_w, AddressMode::ClampToEdge);
    }

    #[test]
    fn test_preset_linear_repeat() {
        let desc = TrinitySamplerDescriptor::linear_repeat();

        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
        assert_eq!(desc.address_mode_v, AddressMode::Repeat);
        assert_eq!(desc.address_mode_w, AddressMode::Repeat);
    }

    #[test]
    fn test_preset_nearest_clamp() {
        let desc = TrinitySamplerDescriptor::nearest_clamp();

        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
    }

    #[test]
    fn test_preset_nearest_repeat() {
        let desc = TrinitySamplerDescriptor::nearest_repeat();

        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
    }

    #[test]
    fn test_preset_shadow() {
        let desc = TrinitySamplerDescriptor::shadow();

        assert_eq!(desc.compare, Some(CompareFunction::Less));
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_preset_trilinear() {
        let desc = TrinitySamplerDescriptor::trilinear();

        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
    }

    // ========================================================================
    // Validation Tests
    // ========================================================================

    #[test]
    fn test_validation_valid_descriptor() {
        let desc = TrinitySamplerDescriptor::new();
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validation_valid_anisotropy_16() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(16);
        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validation_invalid_anisotropy() {
        let desc = TrinitySamplerDescriptor::new().anisotropy(32);

        let result = validate_descriptor(&desc);
        assert!(matches!(
            result,
            Err(SamplerValidationError::InvalidAnisotropy {
                requested: 32,
                max_supported: 16
            })
        ));
    }

    #[test]
    fn test_validation_invalid_lod_range() {
        let mut desc = TrinitySamplerDescriptor::new();
        desc.lod_min_clamp = 10.0;
        desc.lod_max_clamp = 5.0;

        let result = validate_descriptor(&desc);
        assert!(matches!(
            result,
            Err(SamplerValidationError::InvalidLodRange { min: _, max: _ })
        ));
    }

    #[test]
    fn test_validation_negative_lod_min() {
        let mut desc = TrinitySamplerDescriptor::new();
        desc.lod_min_clamp = -1.0;

        let result = validate_descriptor(&desc);
        assert!(matches!(
            result,
            Err(SamplerValidationError::NegativeLod { value: _ })
        ));
    }

    #[test]
    fn test_validation_negative_lod_max() {
        let mut desc = TrinitySamplerDescriptor::new();
        desc.lod_max_clamp = -1.0;

        let result = validate_descriptor(&desc);
        assert!(matches!(
            result,
            Err(SamplerValidationError::NegativeLod { value: _ })
        ));
    }

    #[test]
    fn test_validation_border_color_without_clamp_to_border() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::Repeat)
            .border_color(SamplerBorderColor::OpaqueBlack);

        let result = validate_descriptor(&desc);
        assert!(matches!(
            result,
            Err(SamplerValidationError::BorderColorRequiresClampToBorder)
        ));
    }

    #[test]
    fn test_validation_border_color_with_clamp_to_border() {
        let desc = TrinitySamplerDescriptor::new()
            .address_mode(AddressMode::ClampToBorder)
            .border_color(SamplerBorderColor::OpaqueBlack);

        assert!(validate_descriptor(&desc).is_ok());
    }

    #[test]
    fn test_validation_border_color_partial_clamp_to_border() {
        // Only U uses ClampToBorder, which should be valid
        let desc = TrinitySamplerDescriptor::new()
            .address_mode_uvw(
                AddressMode::ClampToBorder,
                AddressMode::Repeat,
                AddressMode::Repeat,
            )
            .border_color(SamplerBorderColor::TransparentBlack);

        assert!(validate_descriptor(&desc).is_ok());
    }

    // ========================================================================
    // Address Mode Coverage Tests
    // ========================================================================

    #[test]
    fn test_address_mode_clamp_to_edge() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::ClampToEdge);
        assert_eq!(desc.address_mode_u, AddressMode::ClampToEdge);
    }

    #[test]
    fn test_address_mode_repeat() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::Repeat);
        assert_eq!(desc.address_mode_u, AddressMode::Repeat);
    }

    #[test]
    fn test_address_mode_mirror_repeat() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::MirrorRepeat);
        assert_eq!(desc.address_mode_u, AddressMode::MirrorRepeat);
    }

    #[test]
    fn test_address_mode_clamp_to_border() {
        let desc = TrinitySamplerDescriptor::new().address_mode(AddressMode::ClampToBorder);
        assert_eq!(desc.address_mode_u, AddressMode::ClampToBorder);
    }

    // ========================================================================
    // Filter Mode Coverage Tests
    // ========================================================================

    #[test]
    fn test_filter_mode_nearest() {
        let desc = TrinitySamplerDescriptor::new().filter(FilterMode::Nearest);
        assert_eq!(desc.mag_filter, FilterMode::Nearest);
        assert_eq!(desc.min_filter, FilterMode::Nearest);
        assert_eq!(desc.mipmap_filter, FilterMode::Nearest);
    }

    #[test]
    fn test_filter_mode_linear() {
        let desc = TrinitySamplerDescriptor::new().filter(FilterMode::Linear);
        assert_eq!(desc.mag_filter, FilterMode::Linear);
        assert_eq!(desc.min_filter, FilterMode::Linear);
        assert_eq!(desc.mipmap_filter, FilterMode::Linear);
    }

    // ========================================================================
    // Compare Function Tests
    // ========================================================================

    #[test]
    fn test_compare_function_less() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Less);
        assert_eq!(desc.compare, Some(CompareFunction::Less));
    }

    #[test]
    fn test_compare_function_less_equal() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::LessEqual);
        assert_eq!(desc.compare, Some(CompareFunction::LessEqual));
    }

    #[test]
    fn test_compare_function_greater() {
        let desc = TrinitySamplerDescriptor::new().compare(CompareFunction::Greater);
        assert_eq!(desc.compare, Some(CompareFunction::Greater));
    }

    // ========================================================================
    // Error Display Tests
    // ========================================================================

    #[test]
    fn test_error_display_invalid_anisotropy() {
        let err = SamplerValidationError::InvalidAnisotropy {
            requested: 32,
            max_supported: 16,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("32"));
        assert!(msg.contains("16"));
    }

    #[test]
    fn test_error_display_invalid_lod_range() {
        let err = SamplerValidationError::InvalidLodRange {
            min: 10.0,
            max: 5.0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_error_display_border_color_requires_clamp() {
        let err = SamplerValidationError::BorderColorRequiresClampToBorder;
        let msg = format!("{}", err);
        assert!(msg.contains("border_color"));
        assert!(msg.contains("ClampToBorder"));
    }

    #[test]
    fn test_error_display_negative_lod() {
        let err = SamplerValidationError::NegativeLod { value: -1.5 };
        let msg = format!("{}", err);
        assert!(msg.contains("-1.5"));
        assert!(msg.contains("non-negative"));
    }

    // ========================================================================
    // Debug Format Tests
    // ========================================================================

    #[test]
    fn test_descriptor_debug() {
        let desc = TrinitySamplerDescriptor::new().label("debug_test");
        let debug_str = format!("{:?}", desc);
        assert!(debug_str.contains("TrinitySamplerDescriptor"));
        assert!(debug_str.contains("debug_test"));
    }

    // ========================================================================
    // Clone Tests
    // ========================================================================

    #[test]
    fn test_descriptor_clone() {
        let desc = TrinitySamplerDescriptor::new()
            .label("original")
            .anisotropy(8)
            .compare(CompareFunction::Less);

        let cloned = desc.clone();

        assert_eq!(cloned.label, desc.label);
        assert_eq!(cloned.anisotropy_clamp, desc.anisotropy_clamp);
        assert_eq!(cloned.compare, desc.compare);
    }

    // ========================================================================
    // Validation Error Equality Tests
    // ========================================================================

    #[test]
    fn test_validation_error_equality() {
        let err1 = SamplerValidationError::InvalidAnisotropy {
            requested: 32,
            max_supported: 16,
        };
        let err2 = SamplerValidationError::InvalidAnisotropy {
            requested: 32,
            max_supported: 16,
        };

        assert_eq!(err1, err2);
    }

    #[test]
    fn test_validation_error_inequality() {
        let err1 = SamplerValidationError::InvalidAnisotropy {
            requested: 32,
            max_supported: 16,
        };
        let err2 = SamplerValidationError::InvalidLodRange {
            min: 10.0,
            max: 5.0,
        };

        assert_ne!(err1, err2);
    }
}
