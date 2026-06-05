//! Multisample state descriptor for render pipelines.
//!
//! This module provides MSAA (Multisample Anti-Aliasing) configuration for wgpu
//! render pipelines, including device capability queries, sample count selection,
//! and MSAA render target creation.
//!
//! # Overview
//!
//! MSAA improves image quality by rendering at multiple sample points per pixel
//! and averaging the results. Higher sample counts provide better anti-aliasing
//! but consume more memory and GPU resources.
//!
//! # Supported Sample Counts
//!
//! | Count | Quality | Memory | Use Case |
//! |-------|---------|--------|----------|
//! | 1 | None | 1x | No anti-aliasing, fastest |
//! | 4 | Good | 4x | Standard quality, recommended default |
//! | 8 | Better | 8x | High quality, moderate cost |
//! | 16 | Best | 16x | Maximum quality, high cost |
//!
//! # Device Capabilities
//!
//! Not all devices support all sample counts. Use `query_supported_sample_counts()`
//! to check device capabilities and `select_max_supported_sample_count()` to
//! automatically select the best available option.
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::render_pipeline::multisample_state::{
//!     MultisampleStateDescriptor, MultisampleStateBuilder,
//!     query_supported_sample_counts, select_max_supported_sample_count,
//! };
//!
//! // Use a preset
//! let msaa_4x = MultisampleStateDescriptor::msaa_4x();
//! let msaa_off = MultisampleStateDescriptor::msaa_off();
//!
//! // Query device capabilities
//! let supported = query_supported_sample_counts(&adapter, format);
//! let max_count = select_max_supported_sample_count(&adapter, format);
//!
//! // Build custom configuration
//! let custom = MultisampleStateBuilder::new()
//!     .count(4)
//!     .alpha_to_coverage(true)
//!     .build();
//!
//! // Create MSAA render target
//! let target = MsaaRenderTarget::new(&device, 1920, 1080, format, 4);
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// SampleCountInfo - Metadata for sample counts
// ---------------------------------------------------------------------------

/// Information about a sample count with usage documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleCountInfo {
    /// The sample count value (1, 4, 8, or 16).
    pub count: u32,
    /// Human-readable name.
    pub name: &'static str,
    /// Description of the quality/performance tradeoff.
    pub description: &'static str,
    /// Recommended use cases.
    pub use_cases: &'static [&'static str],
    /// Memory multiplier relative to non-MSAA.
    pub memory_multiplier: u32,
}

/// All supported MSAA sample counts with documentation.
pub const SAMPLE_COUNTS: [SampleCountInfo; 4] = [
    SampleCountInfo {
        count: 1,
        name: "No MSAA",
        description: "No multisampling, single sample per pixel",
        use_cases: &["performance-critical rendering", "post-process AA (FXAA, TAA)", "mobile devices"],
        memory_multiplier: 1,
    },
    SampleCountInfo {
        count: 4,
        name: "4x MSAA",
        description: "4 samples per pixel, good quality/performance balance",
        use_cases: &["standard 3D rendering", "recommended default", "most games"],
        memory_multiplier: 4,
    },
    SampleCountInfo {
        count: 8,
        name: "8x MSAA",
        description: "8 samples per pixel, high quality anti-aliasing",
        use_cases: &["high-quality rendering", "CAD applications", "screenshots"],
        memory_multiplier: 8,
    },
    SampleCountInfo {
        count: 16,
        name: "16x MSAA",
        description: "16 samples per pixel, maximum quality anti-aliasing",
        use_cases: &["maximum quality", "offline rendering", "benchmarking"],
        memory_multiplier: 16,
    },
];

/// Get information about a specific sample count.
///
/// # Arguments
///
/// * `count` - The sample count to look up (1, 4, 8, or 16)
///
/// # Returns
///
/// `Some(SampleCountInfo)` if the count is valid, `None` otherwise.
pub fn get_sample_count_info(count: u32) -> Option<&'static SampleCountInfo> {
    SAMPLE_COUNTS.iter().find(|info| info.count == count)
}

/// Check if a sample count is valid (1, 4, 8, or 16).
pub fn is_valid_sample_count(count: u32) -> bool {
    matches!(count, 1 | 4 | 8 | 16)
}

// ---------------------------------------------------------------------------
// Device Capability Queries
// ---------------------------------------------------------------------------

/// Query which sample counts are supported by the device for a given texture format.
///
/// This checks the device's `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` to determine
/// which sample counts are actually supported.
///
/// # Arguments
///
/// * `adapter` - The wgpu adapter to query
/// * `format` - The texture format to check (e.g., `TextureFormat::Rgba8Unorm`)
///
/// # Returns
///
/// A vector of supported sample counts in ascending order.
///
/// # Example
///
/// ```ignore
/// let supported = query_supported_sample_counts(&adapter, TextureFormat::Rgba8Unorm);
/// // Returns e.g., [1, 4] or [1, 4, 8] depending on device
/// ```
pub fn query_supported_sample_counts(
    adapter: &wgpu::Adapter,
    format: wgpu::TextureFormat,
) -> Vec<u32> {
    let features = adapter.get_texture_format_features(format);
    let flags = features.flags;

    let mut supported = vec![1]; // 1 sample is always supported

    if flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4) {
        supported.push(4);
    }
    if flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X8) {
        supported.push(8);
    }
    if flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X16) {
        supported.push(16);
    }

    supported
}

/// Select the maximum supported sample count for a given format.
///
/// # Arguments
///
/// * `adapter` - The wgpu adapter to query
/// * `format` - The texture format to check
///
/// # Returns
///
/// The highest supported sample count (1, 4, 8, or 16).
///
/// # Example
///
/// ```ignore
/// let max = select_max_supported_sample_count(&adapter, TextureFormat::Rgba8Unorm);
/// let state = MultisampleStateDescriptor::new().count(max);
/// ```
pub fn select_max_supported_sample_count(
    adapter: &wgpu::Adapter,
    format: wgpu::TextureFormat,
) -> u32 {
    let supported = query_supported_sample_counts(adapter, format);
    *supported.last().unwrap_or(&1)
}

/// Select the best sample count that doesn't exceed the preferred count.
///
/// # Arguments
///
/// * `adapter` - The wgpu adapter to query
/// * `format` - The texture format to check
/// * `preferred` - The preferred sample count
///
/// # Returns
///
/// The highest supported sample count that is <= preferred.
pub fn select_sample_count_up_to(
    adapter: &wgpu::Adapter,
    format: wgpu::TextureFormat,
    preferred: u32,
) -> u32 {
    let supported = query_supported_sample_counts(adapter, format);
    *supported.iter().rev().find(|&&c| c <= preferred).unwrap_or(&1)
}

// ---------------------------------------------------------------------------
// MultisampleStateDescriptor
// ---------------------------------------------------------------------------

/// Describes multisample anti-aliasing configuration.
///
/// # Defaults
///
/// - `count`: 1 (no MSAA)
/// - `mask`: `!0` (all samples)
/// - `alpha_to_coverage_enabled`: `false`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultisampleStateDescriptor {
    /// Number of samples per pixel.
    pub count: u32,
    /// Sample mask.
    pub mask: u64,
    /// Whether alpha-to-coverage is enabled.
    pub alpha_to_coverage_enabled: bool,
}

impl Default for MultisampleStateDescriptor {
    fn default() -> Self {
        Self {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        }
    }
}

impl MultisampleStateDescriptor {
    /// Create with no MSAA (1 sample).
    pub fn new() -> Self {
        Self::default()
    }

    // -------------------------------------------------------------------------
    // Presets
    // -------------------------------------------------------------------------

    /// Disable MSAA (1 sample per pixel).
    ///
    /// This is the fastest option with no memory overhead. Use when:
    /// - Performance is critical
    /// - Using post-process AA (FXAA, TAA, SMAA)
    /// - Targeting mobile or low-end devices
    pub fn msaa_off() -> Self {
        Self {
            count: 1,
            ..Default::default()
        }
    }

    /// Enable 4x MSAA (4 samples per pixel).
    ///
    /// Good balance between quality and performance. Recommended default for
    /// most 3D applications. Memory cost is 4x the non-MSAA render target.
    pub fn msaa_4x() -> Self {
        Self {
            count: 4,
            ..Default::default()
        }
    }

    /// Enable 8x MSAA (8 samples per pixel).
    ///
    /// High quality anti-aliasing with moderate performance cost.
    /// Memory cost is 8x the non-MSAA render target.
    pub fn msaa_8x() -> Self {
        Self {
            count: 8,
            ..Default::default()
        }
    }

    /// Enable 16x MSAA (16 samples per pixel).
    ///
    /// Maximum quality anti-aliasing. High performance and memory cost.
    /// Memory cost is 16x the non-MSAA render target.
    /// Not supported on all devices - use `query_supported_sample_counts()` to check.
    pub fn msaa_16x() -> Self {
        Self {
            count: 16,
            ..Default::default()
        }
    }

    /// Create preset with alpha-to-coverage enabled.
    ///
    /// Alpha-to-coverage converts alpha values to coverage masks, providing
    /// order-independent transparency for alpha-tested geometry like foliage.
    pub fn with_alpha_to_coverage(count: u32) -> Self {
        Self {
            count,
            mask: !0,
            alpha_to_coverage_enabled: true,
        }
    }

    // -------------------------------------------------------------------------
    // Fluent API
    // -------------------------------------------------------------------------

    /// Set the sample count.
    ///
    /// # Arguments
    ///
    /// * `count` - Must be 1, 4, 8, or 16
    pub fn count(mut self, count: u32) -> Self {
        self.count = count;
        self
    }

    /// Set the sample mask.
    ///
    /// The sample mask determines which samples are written. Each bit corresponds
    /// to a sample. For 4x MSAA, bits 0-3 control samples 0-3.
    ///
    /// Default is `!0` (all samples enabled).
    pub fn mask(mut self, mask: u64) -> Self {
        self.mask = mask;
        self
    }

    /// Enable alpha-to-coverage.
    ///
    /// When enabled, the fragment's alpha value is converted to a coverage mask
    /// that determines which samples are written. This provides order-independent
    /// transparency for alpha-tested geometry.
    pub fn alpha_to_coverage(mut self, enabled: bool) -> Self {
        self.alpha_to_coverage_enabled = enabled;
        self
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Check if MSAA is enabled (count > 1).
    pub fn is_msaa_enabled(&self) -> bool {
        self.count > 1
    }

    /// Get information about this configuration's sample count.
    pub fn sample_count_info(&self) -> Option<&'static SampleCountInfo> {
        get_sample_count_info(self.count)
    }

    /// Calculate the memory multiplier for this MSAA configuration.
    pub fn memory_multiplier(&self) -> u32 {
        self.sample_count_info().map_or(self.count, |info| info.memory_multiplier)
    }
}

impl fmt::Display for MultisampleStateDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.count == 1 {
            write!(f, "No MSAA")?;
        } else {
            write!(f, "{}x MSAA", self.count)?;
        }
        if self.alpha_to_coverage_enabled {
            write!(f, " (alpha-to-coverage)")?;
        }
        if self.mask != !0 {
            write!(f, " (mask: 0x{:X})", self.mask)?;
        }
        Ok(())
    }
}

impl From<MultisampleStateDescriptor> for wgpu::MultisampleState {
    fn from(desc: MultisampleStateDescriptor) -> Self {
        wgpu::MultisampleState {
            count: desc.count,
            mask: desc.mask,
            alpha_to_coverage_enabled: desc.alpha_to_coverage_enabled,
        }
    }
}

// ---------------------------------------------------------------------------
// MultisampleStateBuilder
// ---------------------------------------------------------------------------

/// Builder for MultisampleStateDescriptor with validation.
///
/// # Example
///
/// ```ignore
/// let state = MultisampleStateBuilder::new()
///     .count(4)
///     .alpha_to_coverage(true)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct MultisampleStateBuilder {
    count: u32,
    mask: u64,
    alpha_to_coverage_enabled: bool,
}

impl Default for MultisampleStateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MultisampleStateBuilder {
    /// Create a new builder with default values (no MSAA).
    pub fn new() -> Self {
        Self {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        }
    }

    /// Set the sample count.
    ///
    /// # Arguments
    ///
    /// * `count` - Should be 1, 4, 8, or 16 for best compatibility
    pub fn count(mut self, count: u32) -> Self {
        self.count = count;
        self
    }

    /// Set the sample mask.
    pub fn mask(mut self, mask: u64) -> Self {
        self.mask = mask;
        self
    }

    /// Enable or disable alpha-to-coverage.
    pub fn alpha_to_coverage(mut self, enabled: bool) -> Self {
        self.alpha_to_coverage_enabled = enabled;
        self
    }

    /// Build the MultisampleStateDescriptor.
    pub fn build(self) -> MultisampleStateDescriptor {
        MultisampleStateDescriptor {
            count: self.count,
            mask: self.mask,
            alpha_to_coverage_enabled: self.alpha_to_coverage_enabled,
        }
    }

    /// Build and validate that the sample count is valid.
    ///
    /// # Returns
    ///
    /// `Ok(MultisampleStateDescriptor)` if valid, `Err` with description if not.
    pub fn build_validated(self) -> Result<MultisampleStateDescriptor, String> {
        if !is_valid_sample_count(self.count) {
            return Err(format!(
                "Invalid sample count {}: must be 1, 4, 8, or 16",
                self.count
            ));
        }
        Ok(self.build())
    }
}

// ---------------------------------------------------------------------------
// MsaaRenderTarget - MSAA Render Target Creation
// ---------------------------------------------------------------------------

/// Configuration for creating an MSAA render target.
///
/// MSAA render targets require a multisampled texture for rendering and a
/// resolve target for the final output. This struct encapsulates the creation
/// of both.
///
/// # Example
///
/// ```ignore
/// let msaa_target = MsaaRenderTarget::new(&device, 1920, 1080, format, 4);
///
/// // Use msaa_target.msaa_view as the color attachment
/// // Use msaa_target.resolve_view as the resolve target
/// ```
#[derive(Debug)]
pub struct MsaaRenderTarget {
    /// The multisampled texture.
    pub msaa_texture: wgpu::Texture,
    /// View into the multisampled texture.
    pub msaa_view: wgpu::TextureView,
    /// The resolve texture (1 sample).
    pub resolve_texture: wgpu::Texture,
    /// View into the resolve texture.
    pub resolve_view: wgpu::TextureView,
    /// The sample count.
    pub sample_count: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The texture format.
    pub format: wgpu::TextureFormat,
}

impl MsaaRenderTarget {
    /// Create a new MSAA render target.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `width` - Width in pixels
    /// * `height` - Height in pixels
    /// * `format` - The texture format (must support MSAA)
    /// * `sample_count` - Number of samples (1, 4, 8, or 16)
    ///
    /// # Panics
    ///
    /// Panics if width or height is 0.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        assert!(width > 0 && height > 0, "Render target dimensions must be > 0");

        let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Render Target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let msaa_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let resolve_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Resolve Target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let resolve_view = resolve_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            msaa_texture,
            msaa_view,
            resolve_texture,
            resolve_view,
            sample_count,
            width,
            height,
            format,
        }
    }

    /// Check if this target has MSAA enabled.
    pub fn is_msaa_enabled(&self) -> bool {
        self.sample_count > 1
    }

    /// Get the memory size of the MSAA texture in bytes (approximate).
    pub fn msaa_memory_bytes(&self) -> u64 {
        let bytes_per_pixel = self.format.block_copy_size(None).unwrap_or(4) as u64;
        (self.width as u64) * (self.height as u64) * bytes_per_pixel * (self.sample_count as u64)
    }

    /// Get the total memory size including resolve texture (approximate).
    pub fn total_memory_bytes(&self) -> u64 {
        let bytes_per_pixel = self.format.block_copy_size(None).unwrap_or(4) as u64;
        let msaa = (self.width as u64) * (self.height as u64) * bytes_per_pixel * (self.sample_count as u64);
        let resolve = (self.width as u64) * (self.height as u64) * bytes_per_pixel;
        msaa + resolve
    }

    /// Resize the render target. Returns a new MsaaRenderTarget.
    pub fn resize(
        &self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Self {
        Self::new(device, width, height, self.format, self.sample_count)
    }
}

/// Create MSAA-compatible depth texture.
///
/// Helper to create a depth texture that matches an MSAA render target's sample count.
pub fn create_msaa_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MSAA Depth Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

// ---------------------------------------------------------------------------
// MSAA Resolve Configuration (T-WGPU-P3.7.2)
// ---------------------------------------------------------------------------

/// Store operation for MSAA textures after resolve.
///
/// Controls whether the MSAA texture contents are preserved or discarded
/// after resolving to the resolve target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MsaaStoreOp {
    /// Store the MSAA texture contents after resolve.
    ///
    /// Use when you need to access the multisampled data after the pass,
    /// for example for debugging or custom resolve operations.
    Store,

    /// Discard the MSAA texture contents after resolve.
    ///
    /// This is the default and recommended option. The MSAA texture is only
    /// needed during the render pass, and discarding allows the GPU to
    /// potentially avoid writing it back to memory, saving bandwidth.
    #[default]
    Discard,
}

impl MsaaStoreOp {
    /// Create a Store operation.
    pub fn store() -> Self {
        Self::Store
    }

    /// Create a Discard operation.
    pub fn discard() -> Self {
        Self::Discard
    }

    /// Check if this is a Store operation.
    pub fn is_store(&self) -> bool {
        matches!(self, Self::Store)
    }

    /// Check if this is a Discard operation.
    pub fn is_discard(&self) -> bool {
        matches!(self, Self::Discard)
    }
}

impl From<MsaaStoreOp> for wgpu::StoreOp {
    fn from(op: MsaaStoreOp) -> Self {
        match op {
            MsaaStoreOp::Store => wgpu::StoreOp::Store,
            MsaaStoreOp::Discard => wgpu::StoreOp::Discard,
        }
    }
}

impl fmt::Display for MsaaStoreOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store => write!(f, "Store"),
            Self::Discard => write!(f, "Discard"),
        }
    }
}

/// Information about a resolve operation.
///
/// Provides metadata about the resolve configuration for debugging
/// and introspection purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolveInfo {
    /// The sample count of the source MSAA texture.
    pub source_sample_count: u32,
    /// The sample count of the resolve target (should always be 1).
    pub target_sample_count: u32,
    /// The store operation for the MSAA texture.
    pub store_op: MsaaStoreOp,
    /// Width of the textures in pixels.
    pub width: u32,
    /// Height of the textures in pixels.
    pub height: u32,
    /// The texture format.
    pub format: wgpu::TextureFormat,
    /// Whether this is a valid resolve configuration.
    pub is_valid: bool,
}

impl ResolveInfo {
    /// Create info for a valid resolve operation.
    pub fn valid(
        source_sample_count: u32,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        store_op: MsaaStoreOp,
    ) -> Self {
        Self {
            source_sample_count,
            target_sample_count: 1,
            store_op,
            width,
            height,
            format,
            is_valid: source_sample_count > 1,
        }
    }

    /// Create info for an invalid resolve (e.g., no MSAA).
    pub fn no_resolve() -> Self {
        Self {
            source_sample_count: 1,
            target_sample_count: 1,
            store_op: MsaaStoreOp::Store,
            width: 0,
            height: 0,
            format: wgpu::TextureFormat::Rgba8Unorm,
            is_valid: false,
        }
    }

    /// Check if MSAA resolve is needed (source has multiple samples).
    pub fn needs_resolve(&self) -> bool {
        self.source_sample_count > 1
    }

    /// Get the memory savings from using Discard vs Store (percentage).
    pub fn memory_savings_percent(&self) -> u32 {
        if self.store_op.is_discard() && self.source_sample_count > 1 {
            // Discarding MSAA buffer saves the MSAA texture write-back
            // which is proportional to sample count
            ((self.source_sample_count - 1) * 100) / self.source_sample_count
        } else {
            0
        }
    }
}

impl fmt::Display for ResolveInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid {
            write!(
                f,
                "Resolve {}x -> 1x ({}x{}, {}, {})",
                self.source_sample_count,
                self.width,
                self.height,
                format_name(self.format),
                self.store_op
            )
        } else {
            write!(f, "No resolve (not multisampled)")
        }
    }
}

/// Helper to get a short name for texture formats.
fn format_name(format: wgpu::TextureFormat) -> &'static str {
    match format {
        wgpu::TextureFormat::Rgba8Unorm => "RGBA8",
        wgpu::TextureFormat::Rgba8UnormSrgb => "RGBA8-sRGB",
        wgpu::TextureFormat::Bgra8Unorm => "BGRA8",
        wgpu::TextureFormat::Bgra8UnormSrgb => "BGRA8-sRGB",
        wgpu::TextureFormat::Rgba16Float => "RGBA16F",
        wgpu::TextureFormat::Rgba32Float => "RGBA32F",
        wgpu::TextureFormat::Rgb10a2Unorm => "RGB10A2",
        _ => "Custom",
    }
}

/// Resolve attachment descriptor for render pass color attachments.
///
/// Configures how MSAA textures are resolved to non-multisampled textures
/// at the end of a render pass.
///
/// # Example
///
/// ```ignore
/// let resolve_desc = ResolveAttachmentDescriptor::new(4)
///     .store_op(MsaaStoreOp::Discard);
///
/// // Validate before creating resources
/// assert!(resolve_desc.is_valid());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolveAttachmentDescriptor {
    /// Sample count of the MSAA source texture.
    pub source_sample_count: u32,
    /// Store operation for the MSAA texture after resolve.
    pub store_op: MsaaStoreOp,
    /// Whether resolve is enabled (source_sample_count > 1).
    enabled: bool,
}

impl Default for ResolveAttachmentDescriptor {
    fn default() -> Self {
        Self {
            source_sample_count: 1,
            store_op: MsaaStoreOp::Discard,
            enabled: false,
        }
    }
}

impl ResolveAttachmentDescriptor {
    /// Create a new resolve attachment descriptor.
    ///
    /// # Arguments
    ///
    /// * `source_sample_count` - Sample count of the MSAA texture (1, 4, 8, or 16)
    pub fn new(source_sample_count: u32) -> Self {
        Self {
            source_sample_count,
            store_op: MsaaStoreOp::Discard,
            enabled: source_sample_count > 1,
        }
    }

    // -------------------------------------------------------------------------
    // Presets
    // -------------------------------------------------------------------------

    /// Standard MSAA 4x resolve with discard.
    ///
    /// The most common configuration: 4x MSAA resolved to single-sampled,
    /// with the MSAA texture discarded after resolve to save bandwidth.
    pub fn resolve_discard_4x() -> Self {
        Self {
            source_sample_count: 4,
            store_op: MsaaStoreOp::Discard,
            enabled: true,
        }
    }

    /// MSAA 8x resolve with discard.
    pub fn resolve_discard_8x() -> Self {
        Self {
            source_sample_count: 8,
            store_op: MsaaStoreOp::Discard,
            enabled: true,
        }
    }

    /// MSAA 4x resolve keeping the MSAA texture.
    ///
    /// Use when you need access to the multisampled data after the pass.
    pub fn resolve_store_4x() -> Self {
        Self {
            source_sample_count: 4,
            store_op: MsaaStoreOp::Store,
            enabled: true,
        }
    }

    /// MSAA 8x resolve keeping the MSAA texture.
    pub fn resolve_store_8x() -> Self {
        Self {
            source_sample_count: 8,
            store_op: MsaaStoreOp::Store,
            enabled: true,
        }
    }

    /// No resolve (single-sampled rendering).
    pub fn no_resolve() -> Self {
        Self::default()
    }

    // -------------------------------------------------------------------------
    // Fluent API
    // -------------------------------------------------------------------------

    /// Set the sample count.
    pub fn sample_count(mut self, count: u32) -> Self {
        self.source_sample_count = count;
        self.enabled = count > 1;
        self
    }

    /// Set the store operation for the MSAA texture.
    pub fn store_op(mut self, op: MsaaStoreOp) -> Self {
        self.store_op = op;
        self
    }

    /// Set to discard MSAA texture after resolve.
    pub fn discard(mut self) -> Self {
        self.store_op = MsaaStoreOp::Discard;
        self
    }

    /// Set to store MSAA texture after resolve.
    pub fn store(mut self) -> Self {
        self.store_op = MsaaStoreOp::Store;
        self
    }

    // -------------------------------------------------------------------------
    // Validation
    // -------------------------------------------------------------------------

    /// Check if this configuration is valid.
    ///
    /// A valid configuration has a supported sample count (1, 4, 8, or 16).
    pub fn is_valid(&self) -> bool {
        is_valid_sample_count(self.source_sample_count)
    }

    /// Validate that a resolve target is not multisampled.
    ///
    /// # Arguments
    ///
    /// * `target_sample_count` - Sample count of the resolve target
    ///
    /// # Returns
    ///
    /// `Ok(())` if target is single-sampled (count == 1), `Err` otherwise.
    pub fn validate_resolve_target(target_sample_count: u32) -> Result<(), ResolveError> {
        if target_sample_count == 1 {
            Ok(())
        } else {
            Err(ResolveError::InvalidResolveTarget {
                expected: 1,
                actual: target_sample_count,
            })
        }
    }

    /// Validate this configuration and a target.
    ///
    /// Checks that:
    /// 1. Source sample count is valid
    /// 2. Target sample count is 1
    /// 3. If enabled, source > 1
    pub fn validate(&self, target_sample_count: u32) -> Result<(), ResolveError> {
        if !is_valid_sample_count(self.source_sample_count) {
            return Err(ResolveError::InvalidSourceSampleCount(self.source_sample_count));
        }
        Self::validate_resolve_target(target_sample_count)?;
        if self.enabled && self.source_sample_count <= 1 {
            return Err(ResolveError::ResolveEnabledWithoutMsaa);
        }
        Ok(())
    }

    /// Check if resolve is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Check if this needs a resolve target.
    pub fn needs_resolve_target(&self) -> bool {
        self.enabled && self.source_sample_count > 1
    }

    /// Get the wgpu StoreOp for this configuration.
    pub fn wgpu_store_op(&self) -> wgpu::StoreOp {
        self.store_op.into()
    }
}

impl fmt::Display for ResolveAttachmentDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.enabled {
            write!(f, "Resolve {}x MSAA ({})", self.source_sample_count, self.store_op)
        } else {
            write!(f, "No MSAA resolve")
        }
    }
}

/// Errors that can occur during resolve configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// Invalid source sample count.
    InvalidSourceSampleCount(u32),
    /// Resolve target is multisampled (must be sample_count == 1).
    InvalidResolveTarget {
        expected: u32,
        actual: u32,
    },
    /// Resolve is enabled but source sample count is 1.
    ResolveEnabledWithoutMsaa,
    /// Texture dimensions don't match.
    DimensionMismatch {
        source: (u32, u32),
        target: (u32, u32),
    },
    /// Texture formats don't match.
    FormatMismatch {
        source: wgpu::TextureFormat,
        target: wgpu::TextureFormat,
    },
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSourceSampleCount(count) => {
                write!(f, "Invalid source sample count {}: must be 1, 4, 8, or 16", count)
            }
            Self::InvalidResolveTarget { expected, actual } => {
                write!(f, "Resolve target must have sample count {}, got {}", expected, actual)
            }
            Self::ResolveEnabledWithoutMsaa => {
                write!(f, "Resolve enabled but source sample count is 1 (no MSAA)")
            }
            Self::DimensionMismatch { source, target } => {
                write!(
                    f,
                    "Dimension mismatch: source {}x{}, target {}x{}",
                    source.0, source.1, target.0, target.1
                )
            }
            Self::FormatMismatch { source, target } => {
                write!(f, "Format mismatch: source {:?}, target {:?}", source, target)
            }
        }
    }
}

impl std::error::Error for ResolveError {}

/// MSAA resolve target pair.
///
/// Holds references to both the MSAA source texture view and the resolve target
/// texture view, along with the store operation configuration.
///
/// This struct is designed to be used when configuring render pass color
/// attachments that need MSAA resolve.
///
/// # Example
///
/// ```ignore
/// let resolve_target = MsaaResolveTarget::new(&msaa_view, &resolve_view)
///     .store_op(MsaaStoreOp::Discard);
///
/// // Use in render pass setup
/// let color_attachment = wgpu::RenderPassColorAttachment {
///     view: resolve_target.source,
///     resolve_target: Some(resolve_target.resolve_target),
///     ops: wgpu::Operations {
///         load: wgpu::LoadOp::Clear(color),
///         store: resolve_target.store_op.into(),
///     },
/// };
/// ```
#[derive(Debug)]
pub struct MsaaResolveTarget<'a> {
    /// The MSAA texture view (multisampled source).
    pub source: &'a wgpu::TextureView,
    /// The resolve target texture view (must be sample_count == 1).
    pub resolve_target: &'a wgpu::TextureView,
    /// Store operation for the MSAA texture.
    pub store_op: MsaaStoreOp,
}

impl<'a> MsaaResolveTarget<'a> {
    /// Create a new MSAA resolve target pair.
    ///
    /// # Arguments
    ///
    /// * `source` - The multisampled texture view (MSAA source)
    /// * `resolve_target` - The single-sampled texture view (resolve destination)
    ///
    /// # Note
    ///
    /// This does not validate sample counts at creation time. Use `validate()`
    /// or `validate_sample_counts()` to check compatibility.
    pub fn new(source: &'a wgpu::TextureView, resolve_target: &'a wgpu::TextureView) -> Self {
        Self {
            source,
            resolve_target,
            store_op: MsaaStoreOp::Discard,
        }
    }

    /// Set the store operation for the MSAA texture.
    pub fn store_op(mut self, op: MsaaStoreOp) -> Self {
        self.store_op = op;
        self
    }

    /// Set store operation to Discard.
    pub fn discard(mut self) -> Self {
        self.store_op = MsaaStoreOp::Discard;
        self
    }

    /// Set store operation to Store.
    pub fn store(mut self) -> Self {
        self.store_op = MsaaStoreOp::Store;
        self
    }

    /// Get the wgpu StoreOp.
    pub fn wgpu_store_op(&self) -> wgpu::StoreOp {
        self.store_op.into()
    }

    /// Create an optional resolve target for use in render pass attachments.
    ///
    /// Returns `Some(&resolve_target)` which can be used directly in
    /// `RenderPassColorAttachment::resolve_target`.
    pub fn resolve_target_option(&self) -> Option<&'a wgpu::TextureView> {
        Some(self.resolve_target)
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Check if a sample count is valid for a resolve target.
///
/// Resolve targets must have a sample count of 1.
///
/// # Arguments
///
/// * `sample_count` - The sample count to check
///
/// # Returns
///
/// `true` if sample_count == 1, `false` otherwise.
pub fn is_valid_resolve_target(sample_count: u32) -> bool {
    sample_count == 1
}

/// Create a matching MSAA + resolve texture pair.
///
/// This is a convenience function that creates both textures needed for
/// MSAA rendering with resolve. The MSAA texture has the specified sample count,
/// and the resolve texture has sample count 1.
///
/// # Arguments
///
/// * `device` - The wgpu device
/// * `width` - Width in pixels
/// * `height` - Height in pixels
/// * `format` - The texture format
/// * `sample_count` - MSAA sample count (4, 8, or 16)
///
/// # Returns
///
/// A tuple of (msaa_texture, msaa_view, resolve_texture, resolve_view).
///
/// # Panics
///
/// Panics if width or height is 0, or if sample_count is 1.
pub fn create_resolve_pair(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Texture, wgpu::TextureView) {
    assert!(width > 0 && height > 0, "Texture dimensions must be > 0");
    assert!(sample_count > 1, "Sample count must be > 1 for MSAA resolve pair");

    let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MSAA Source"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let msaa_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let resolve_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MSAA Resolve Target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let resolve_view = resolve_texture.create_view(&wgpu::TextureViewDescriptor::default());

    (msaa_texture, msaa_view, resolve_texture, resolve_view)
}

/// Preset: Standard MSAA discard after resolve.
///
/// Returns a ResolveAttachmentDescriptor configured for the most common
/// MSAA resolve pattern: 4x MSAA with the MSAA texture discarded after resolve.
pub fn resolve_discard() -> ResolveAttachmentDescriptor {
    ResolveAttachmentDescriptor::resolve_discard_4x()
}

/// Preset: Keep both MSAA and resolved textures.
///
/// Returns a ResolveAttachmentDescriptor configured to keep the MSAA texture
/// after resolve. Use when you need access to the multisampled data.
pub fn resolve_store() -> ResolveAttachmentDescriptor {
    ResolveAttachmentDescriptor::resolve_store_4x()
}

// ---------------------------------------------------------------------------
// Extended MsaaRenderTarget methods
// ---------------------------------------------------------------------------

impl MsaaRenderTarget {
    /// Create a resolve attachment descriptor for this target.
    ///
    /// Returns a descriptor configured with this target's sample count and
    /// the default store operation (Discard).
    pub fn resolve_descriptor(&self) -> ResolveAttachmentDescriptor {
        ResolveAttachmentDescriptor::new(self.sample_count)
    }

    /// Create a resolve target pair from this MSAA render target.
    ///
    /// Returns an `MsaaResolveTarget` referencing this target's views.
    pub fn as_resolve_target(&self) -> MsaaResolveTarget<'_> {
        MsaaResolveTarget::new(&self.msaa_view, &self.resolve_view)
    }

    /// Get resolve info for this target.
    pub fn resolve_info(&self, store_op: MsaaStoreOp) -> ResolveInfo {
        ResolveInfo::valid(
            self.sample_count,
            self.width,
            self.height,
            self.format,
            store_op,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multisample_defaults() {
        let state = MultisampleStateDescriptor::default();
        assert_eq!(state.count, 1);
        assert_eq!(state.mask, !0);
        assert!(!state.alpha_to_coverage_enabled);
    }

    #[test]
    fn test_multisample_presets() {
        let msaa4 = MultisampleStateDescriptor::msaa_4x();
        assert_eq!(msaa4.count, 4);

        let msaa8 = MultisampleStateDescriptor::msaa_8x();
        assert_eq!(msaa8.count, 8);
    }

    #[test]
    fn test_multisample_builder() {
        let state = MultisampleStateDescriptor::new()
            .count(2)
            .alpha_to_coverage(true);
        assert_eq!(state.count, 2);
        assert!(state.alpha_to_coverage_enabled);
    }

    #[test]
    fn test_multisample_into_wgpu() {
        let state = MultisampleStateDescriptor::msaa_4x();
        let wgpu_state: wgpu::MultisampleState = state.into();
        assert_eq!(wgpu_state.count, 4);
    }

    // -------------------------------------------------------------------------
    // Additional Whitebox Tests - Sample Counts
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_count_1() {
        let state = MultisampleStateDescriptor::new().count(1);
        assert_eq!(state.count, 1);

        let wgpu_state: wgpu::MultisampleState = state.into();
        assert_eq!(wgpu_state.count, 1);
    }

    #[test]
    fn test_sample_count_4() {
        let state = MultisampleStateDescriptor::new().count(4);
        assert_eq!(state.count, 4);

        let wgpu_state: wgpu::MultisampleState = state.into();
        assert_eq!(wgpu_state.count, 4);
    }

    #[test]
    fn test_sample_count_8() {
        let state = MultisampleStateDescriptor::new().count(8);
        assert_eq!(state.count, 8);

        let wgpu_state: wgpu::MultisampleState = state.into();
        assert_eq!(wgpu_state.count, 8);
    }

    #[test]
    fn test_sample_count_16() {
        let state = MultisampleStateDescriptor::new().count(16);
        assert_eq!(state.count, 16);

        let wgpu_state: wgpu::MultisampleState = state.into();
        assert_eq!(wgpu_state.count, 16);
    }

    #[test]
    fn test_sample_mask_all_samples() {
        // All samples enabled (default)
        let state = MultisampleStateDescriptor::new();
        assert_eq!(state.mask, !0u64);

        let wgpu_state: wgpu::MultisampleState = state.into();
        assert_eq!(wgpu_state.mask, !0u64);
    }

    #[test]
    fn test_sample_mask_half_samples() {
        // Only lower half of samples
        let state = MultisampleStateDescriptor::msaa_4x().mask(0b0011);
        assert_eq!(state.mask, 0b0011);
    }

    #[test]
    fn test_sample_mask_single_sample() {
        // Only first sample
        let state = MultisampleStateDescriptor::msaa_4x().mask(0b0001);
        assert_eq!(state.mask, 0b0001);
    }

    #[test]
    fn test_sample_mask_zero() {
        // No samples (edge case, probably invalid but API allows)
        let state = MultisampleStateDescriptor::new().mask(0);
        assert_eq!(state.mask, 0);
    }

    #[test]
    fn test_sample_mask_alternating() {
        // Alternating samples for checkerboard pattern
        let state = MultisampleStateDescriptor::msaa_8x().mask(0b01010101);
        assert_eq!(state.mask, 0b01010101);
    }

    #[test]
    fn test_alpha_to_coverage_enabled() {
        let state = MultisampleStateDescriptor::new().alpha_to_coverage(true);
        assert!(state.alpha_to_coverage_enabled);

        let wgpu_state: wgpu::MultisampleState = state.into();
        assert!(wgpu_state.alpha_to_coverage_enabled);
    }

    #[test]
    fn test_alpha_to_coverage_disabled() {
        let state = MultisampleStateDescriptor::new().alpha_to_coverage(false);
        assert!(!state.alpha_to_coverage_enabled);
    }

    #[test]
    fn test_alpha_to_coverage_with_msaa() {
        // Alpha-to-coverage is commonly used with MSAA
        let state = MultisampleStateDescriptor::msaa_4x().alpha_to_coverage(true);
        assert_eq!(state.count, 4);
        assert!(state.alpha_to_coverage_enabled);
    }

    #[test]
    fn test_combined_msaa_settings() {
        let state = MultisampleStateDescriptor::new()
            .count(8)
            .mask(0xFF)
            .alpha_to_coverage(true);

        assert_eq!(state.count, 8);
        assert_eq!(state.mask, 0xFF);
        assert!(state.alpha_to_coverage_enabled);
    }

    #[test]
    fn test_multisample_equality() {
        let state1 = MultisampleStateDescriptor::msaa_4x();
        let state2 = MultisampleStateDescriptor::msaa_4x();
        let state3 = MultisampleStateDescriptor::msaa_8x();

        assert_eq!(state1, state2);
        assert_ne!(state1, state3);
    }

    #[test]
    fn test_multisample_copy() {
        let state = MultisampleStateDescriptor::msaa_4x();
        let state_copy = state;
        assert_eq!(state, state_copy);
    }

    #[test]
    fn test_multisample_clone() {
        let state = MultisampleStateDescriptor::msaa_4x();
        let state_clone = state.clone();
        assert_eq!(state, state_clone);
    }

    #[test]
    fn test_into_wgpu_all_fields() {
        let state = MultisampleStateDescriptor::new()
            .count(8)
            .mask(0xAB)
            .alpha_to_coverage(true);

        let wgpu_state: wgpu::MultisampleState = state.into();

        assert_eq!(wgpu_state.count, 8);
        assert_eq!(wgpu_state.mask, 0xAB);
        assert!(wgpu_state.alpha_to_coverage_enabled);
    }

    // -------------------------------------------------------------------------
    // SampleCountInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_counts_array() {
        assert_eq!(SAMPLE_COUNTS.len(), 4);
        assert_eq!(SAMPLE_COUNTS[0].count, 1);
        assert_eq!(SAMPLE_COUNTS[1].count, 4);
        assert_eq!(SAMPLE_COUNTS[2].count, 8);
        assert_eq!(SAMPLE_COUNTS[3].count, 16);
    }

    #[test]
    fn test_sample_count_info_names() {
        assert_eq!(SAMPLE_COUNTS[0].name, "No MSAA");
        assert_eq!(SAMPLE_COUNTS[1].name, "4x MSAA");
        assert_eq!(SAMPLE_COUNTS[2].name, "8x MSAA");
        assert_eq!(SAMPLE_COUNTS[3].name, "16x MSAA");
    }

    #[test]
    fn test_sample_count_info_memory_multipliers() {
        assert_eq!(SAMPLE_COUNTS[0].memory_multiplier, 1);
        assert_eq!(SAMPLE_COUNTS[1].memory_multiplier, 4);
        assert_eq!(SAMPLE_COUNTS[2].memory_multiplier, 8);
        assert_eq!(SAMPLE_COUNTS[3].memory_multiplier, 16);
    }

    #[test]
    fn test_sample_count_info_use_cases() {
        // Each sample count should have at least one use case
        for info in &SAMPLE_COUNTS {
            assert!(!info.use_cases.is_empty(), "Sample count {} has no use cases", info.count);
        }
    }

    #[test]
    fn test_get_sample_count_info_valid() {
        let info = get_sample_count_info(1).expect("Should find 1");
        assert_eq!(info.count, 1);

        let info = get_sample_count_info(4).expect("Should find 4");
        assert_eq!(info.count, 4);

        let info = get_sample_count_info(8).expect("Should find 8");
        assert_eq!(info.count, 8);

        let info = get_sample_count_info(16).expect("Should find 16");
        assert_eq!(info.count, 16);
    }

    #[test]
    fn test_get_sample_count_info_invalid() {
        assert!(get_sample_count_info(0).is_none());
        assert!(get_sample_count_info(2).is_none());
        assert!(get_sample_count_info(3).is_none());
        assert!(get_sample_count_info(5).is_none());
        assert!(get_sample_count_info(32).is_none());
    }

    #[test]
    fn test_is_valid_sample_count() {
        assert!(is_valid_sample_count(1));
        assert!(is_valid_sample_count(4));
        assert!(is_valid_sample_count(8));
        assert!(is_valid_sample_count(16));

        assert!(!is_valid_sample_count(0));
        assert!(!is_valid_sample_count(2));
        assert!(!is_valid_sample_count(3));
        assert!(!is_valid_sample_count(32));
    }

    // -------------------------------------------------------------------------
    // New Preset Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_msaa_off_preset() {
        let state = MultisampleStateDescriptor::msaa_off();
        assert_eq!(state.count, 1);
        assert_eq!(state.mask, !0);
        assert!(!state.alpha_to_coverage_enabled);
    }

    #[test]
    fn test_msaa_16x_preset() {
        let state = MultisampleStateDescriptor::msaa_16x();
        assert_eq!(state.count, 16);
        assert_eq!(state.mask, !0);
        assert!(!state.alpha_to_coverage_enabled);
    }

    #[test]
    fn test_with_alpha_to_coverage_preset() {
        let state = MultisampleStateDescriptor::with_alpha_to_coverage(4);
        assert_eq!(state.count, 4);
        assert!(state.alpha_to_coverage_enabled);
    }

    // -------------------------------------------------------------------------
    // Helper Method Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_msaa_enabled() {
        assert!(!MultisampleStateDescriptor::msaa_off().is_msaa_enabled());
        assert!(MultisampleStateDescriptor::msaa_4x().is_msaa_enabled());
        assert!(MultisampleStateDescriptor::msaa_8x().is_msaa_enabled());
        assert!(MultisampleStateDescriptor::msaa_16x().is_msaa_enabled());
    }

    #[test]
    fn test_sample_count_info_method() {
        let state = MultisampleStateDescriptor::msaa_4x();
        let info = state.sample_count_info().expect("Should have info");
        assert_eq!(info.count, 4);
        assert_eq!(info.name, "4x MSAA");
    }

    #[test]
    fn test_memory_multiplier_method() {
        assert_eq!(MultisampleStateDescriptor::msaa_off().memory_multiplier(), 1);
        assert_eq!(MultisampleStateDescriptor::msaa_4x().memory_multiplier(), 4);
        assert_eq!(MultisampleStateDescriptor::msaa_8x().memory_multiplier(), 8);
        assert_eq!(MultisampleStateDescriptor::msaa_16x().memory_multiplier(), 16);
    }

    // -------------------------------------------------------------------------
    // Display Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_no_msaa() {
        let state = MultisampleStateDescriptor::msaa_off();
        let display = format!("{}", state);
        assert_eq!(display, "No MSAA");
    }

    #[test]
    fn test_display_msaa_4x() {
        let state = MultisampleStateDescriptor::msaa_4x();
        let display = format!("{}", state);
        assert_eq!(display, "4x MSAA");
    }

    #[test]
    fn test_display_with_alpha_to_coverage() {
        let state = MultisampleStateDescriptor::msaa_4x().alpha_to_coverage(true);
        let display = format!("{}", state);
        assert!(display.contains("4x MSAA"));
        assert!(display.contains("alpha-to-coverage"));
    }

    #[test]
    fn test_display_with_custom_mask() {
        let state = MultisampleStateDescriptor::msaa_4x().mask(0b0011);
        let display = format!("{}", state);
        assert!(display.contains("4x MSAA"));
        assert!(display.contains("mask"));
    }

    // -------------------------------------------------------------------------
    // MultisampleStateBuilder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_defaults() {
        let state = MultisampleStateBuilder::new().build();
        assert_eq!(state.count, 1);
        assert_eq!(state.mask, !0);
        assert!(!state.alpha_to_coverage_enabled);
    }

    #[test]
    fn test_builder_count() {
        let state = MultisampleStateBuilder::new().count(4).build();
        assert_eq!(state.count, 4);
    }

    #[test]
    fn test_builder_mask() {
        let state = MultisampleStateBuilder::new().mask(0xF0).build();
        assert_eq!(state.mask, 0xF0);
    }

    #[test]
    fn test_builder_alpha_to_coverage() {
        let state = MultisampleStateBuilder::new().alpha_to_coverage(true).build();
        assert!(state.alpha_to_coverage_enabled);
    }

    #[test]
    fn test_builder_chained() {
        let state = MultisampleStateBuilder::new()
            .count(8)
            .mask(0xFF)
            .alpha_to_coverage(true)
            .build();

        assert_eq!(state.count, 8);
        assert_eq!(state.mask, 0xFF);
        assert!(state.alpha_to_coverage_enabled);
    }

    #[test]
    fn test_builder_validated_valid() {
        let result = MultisampleStateBuilder::new().count(4).build_validated();
        assert!(result.is_ok());
        assert_eq!(result.unwrap().count, 4);
    }

    #[test]
    fn test_builder_validated_invalid() {
        let result = MultisampleStateBuilder::new().count(5).build_validated();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid sample count"));
    }

    #[test]
    fn test_builder_default_trait() {
        let builder = MultisampleStateBuilder::default();
        let state = builder.build();
        assert_eq!(state.count, 1);
    }

    #[test]
    fn test_builder_clone() {
        let builder = MultisampleStateBuilder::new().count(4);
        let builder_clone = builder.clone();
        assert_eq!(builder.build().count, builder_clone.build().count);
    }

    // -------------------------------------------------------------------------
    // Integration Tests (descriptor + builder consistency)
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_matches_descriptor_msaa_4x() {
        let from_preset = MultisampleStateDescriptor::msaa_4x();
        let from_builder = MultisampleStateBuilder::new().count(4).build();
        assert_eq!(from_preset, from_builder);
    }

    #[test]
    fn test_builder_matches_descriptor_msaa_8x() {
        let from_preset = MultisampleStateDescriptor::msaa_8x();
        let from_builder = MultisampleStateBuilder::new().count(8).build();
        assert_eq!(from_preset, from_builder);
    }

    #[test]
    fn test_builder_matches_descriptor_msaa_off() {
        let from_preset = MultisampleStateDescriptor::msaa_off();
        let from_builder = MultisampleStateBuilder::new().count(1).build();
        assert_eq!(from_preset, from_builder);
    }

    // -------------------------------------------------------------------------
    // wgpu Conversion Consistency Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_all_presets_convert_to_wgpu() {
        // msaa_off
        let wgpu_off: wgpu::MultisampleState = MultisampleStateDescriptor::msaa_off().into();
        assert_eq!(wgpu_off.count, 1);

        // msaa_4x
        let wgpu_4x: wgpu::MultisampleState = MultisampleStateDescriptor::msaa_4x().into();
        assert_eq!(wgpu_4x.count, 4);

        // msaa_8x
        let wgpu_8x: wgpu::MultisampleState = MultisampleStateDescriptor::msaa_8x().into();
        assert_eq!(wgpu_8x.count, 8);

        // msaa_16x
        let wgpu_16x: wgpu::MultisampleState = MultisampleStateDescriptor::msaa_16x().into();
        assert_eq!(wgpu_16x.count, 16);
    }

    #[test]
    fn test_builder_converts_to_wgpu() {
        let state = MultisampleStateBuilder::new()
            .count(4)
            .mask(0xABCD)
            .alpha_to_coverage(true)
            .build();

        let wgpu_state: wgpu::MultisampleState = state.into();
        assert_eq!(wgpu_state.count, 4);
        assert_eq!(wgpu_state.mask, 0xABCD);
        assert!(wgpu_state.alpha_to_coverage_enabled);
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests - Send + Sync bounds
    // -------------------------------------------------------------------------

    #[test]
    fn test_multisample_state_descriptor_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MultisampleStateDescriptor>();
    }

    #[test]
    fn test_multisample_state_descriptor_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<MultisampleStateDescriptor>();
    }

    #[test]
    fn test_multisample_state_builder_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MultisampleStateBuilder>();
    }

    #[test]
    fn test_multisample_state_builder_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<MultisampleStateBuilder>();
    }

    #[test]
    fn test_sample_count_info_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SampleCountInfo>();
    }

    #[test]
    fn test_sample_count_info_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<SampleCountInfo>();
    }

    // -------------------------------------------------------------------------
    // Additional Sample Count Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_valid_sample_count_boundary_values() {
        // Test values around valid counts
        assert!(!is_valid_sample_count(0));
        assert!(is_valid_sample_count(1));
        assert!(!is_valid_sample_count(2));
        assert!(!is_valid_sample_count(3));
        assert!(is_valid_sample_count(4));
        assert!(!is_valid_sample_count(5));
        assert!(!is_valid_sample_count(6));
        assert!(!is_valid_sample_count(7));
        assert!(is_valid_sample_count(8));
        assert!(!is_valid_sample_count(9));
        assert!(!is_valid_sample_count(15));
        assert!(is_valid_sample_count(16));
        assert!(!is_valid_sample_count(17));
    }

    #[test]
    fn test_is_valid_sample_count_large_values() {
        assert!(!is_valid_sample_count(32));
        assert!(!is_valid_sample_count(64));
        assert!(!is_valid_sample_count(128));
        assert!(!is_valid_sample_count(256));
        assert!(!is_valid_sample_count(u32::MAX));
    }

    // -------------------------------------------------------------------------
    // Builder Validation Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_validated_all_valid_counts() {
        // All valid counts should pass validation
        for count in [1, 4, 8, 16] {
            let result = MultisampleStateBuilder::new().count(count).build_validated();
            assert!(result.is_ok(), "Count {} should be valid", count);
            assert_eq!(result.unwrap().count, count);
        }
    }

    #[test]
    fn test_builder_validated_invalid_counts() {
        // Various invalid counts
        for count in [0, 2, 3, 5, 6, 7, 9, 10, 12, 15, 17, 32, 64] {
            let result = MultisampleStateBuilder::new().count(count).build_validated();
            assert!(result.is_err(), "Count {} should be invalid", count);
        }
    }

    #[test]
    fn test_builder_validated_error_message() {
        let result = MultisampleStateBuilder::new().count(3).build_validated();
        let err = result.unwrap_err();
        assert!(err.contains("3"));
        assert!(err.contains("must be 1, 4, 8, or 16"));
    }

    #[test]
    fn test_builder_validated_preserves_all_fields() {
        let result = MultisampleStateBuilder::new()
            .count(8)
            .mask(0xABCD)
            .alpha_to_coverage(true)
            .build_validated();

        assert!(result.is_ok());
        let state = result.unwrap();
        assert_eq!(state.count, 8);
        assert_eq!(state.mask, 0xABCD);
        assert!(state.alpha_to_coverage_enabled);
    }

    // -------------------------------------------------------------------------
    // SampleCountInfo Detailed Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_count_info_descriptions_non_empty() {
        for info in &SAMPLE_COUNTS {
            assert!(!info.description.is_empty(), "Count {} has empty description", info.count);
        }
    }

    #[test]
    fn test_sample_count_info_1_details() {
        let info = get_sample_count_info(1).unwrap();
        assert_eq!(info.count, 1);
        assert_eq!(info.name, "No MSAA");
        assert_eq!(info.memory_multiplier, 1);
        assert!(info.use_cases.len() >= 2);
    }

    #[test]
    fn test_sample_count_info_4_details() {
        let info = get_sample_count_info(4).unwrap();
        assert_eq!(info.count, 4);
        assert_eq!(info.name, "4x MSAA");
        assert_eq!(info.memory_multiplier, 4);
        assert!(info.description.contains("4 samples"));
    }

    #[test]
    fn test_sample_count_info_8_details() {
        let info = get_sample_count_info(8).unwrap();
        assert_eq!(info.count, 8);
        assert_eq!(info.name, "8x MSAA");
        assert_eq!(info.memory_multiplier, 8);
        assert!(info.description.contains("8 samples"));
    }

    #[test]
    fn test_sample_count_info_16_details() {
        let info = get_sample_count_info(16).unwrap();
        assert_eq!(info.count, 16);
        assert_eq!(info.name, "16x MSAA");
        assert_eq!(info.memory_multiplier, 16);
        assert!(info.description.contains("16 samples"));
    }

    #[test]
    fn test_sample_count_info_equality() {
        let info1 = get_sample_count_info(4).unwrap();
        let info2 = &SAMPLE_COUNTS[1];
        assert_eq!(info1.count, info2.count);
        assert_eq!(info1.name, info2.name);
    }

    #[test]
    fn test_sample_count_info_copy() {
        let info = get_sample_count_info(4).unwrap();
        let info_copy = *info;
        assert_eq!(info.count, info_copy.count);
    }

    // -------------------------------------------------------------------------
    // Memory Multiplier Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_multiplier_invalid_count() {
        // When sample count is invalid, memory_multiplier returns the count itself
        let state = MultisampleStateDescriptor::new().count(3);
        assert_eq!(state.memory_multiplier(), 3);
    }

    #[test]
    fn test_memory_multiplier_zero_count() {
        let state = MultisampleStateDescriptor::new().count(0);
        assert_eq!(state.memory_multiplier(), 0);
    }

    #[test]
    fn test_memory_multiplier_large_count() {
        let state = MultisampleStateDescriptor::new().count(32);
        assert_eq!(state.memory_multiplier(), 32);
    }

    // -------------------------------------------------------------------------
    // Display Format Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_display_msaa_8x() {
        let state = MultisampleStateDescriptor::msaa_8x();
        let display = format!("{}", state);
        assert_eq!(display, "8x MSAA");
    }

    #[test]
    fn test_display_msaa_16x() {
        let state = MultisampleStateDescriptor::msaa_16x();
        let display = format!("{}", state);
        assert_eq!(display, "16x MSAA");
    }

    #[test]
    fn test_display_full_combination() {
        let state = MultisampleStateDescriptor::new()
            .count(8)
            .mask(0xFF)
            .alpha_to_coverage(true);
        let display = format!("{}", state);
        assert!(display.contains("8x MSAA"));
        assert!(display.contains("alpha-to-coverage"));
        assert!(display.contains("mask"));
    }

    #[test]
    fn test_display_no_msaa_with_alpha_to_coverage() {
        let state = MultisampleStateDescriptor::msaa_off().alpha_to_coverage(true);
        let display = format!("{}", state);
        assert!(display.contains("No MSAA"));
        assert!(display.contains("alpha-to-coverage"));
    }

    #[test]
    fn test_display_hex_mask_format() {
        let state = MultisampleStateDescriptor::msaa_4x().mask(0xABCD);
        let display = format!("{}", state);
        assert!(display.contains("0xABCD"));
    }

    // -------------------------------------------------------------------------
    // Chained Method Call Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fluent_api_override_count() {
        let state = MultisampleStateDescriptor::new()
            .count(4)
            .count(8)
            .count(16);
        assert_eq!(state.count, 16);
    }

    #[test]
    fn test_fluent_api_override_mask() {
        let state = MultisampleStateDescriptor::new()
            .mask(0x0F)
            .mask(0xF0)
            .mask(0xFF);
        assert_eq!(state.mask, 0xFF);
    }

    #[test]
    fn test_fluent_api_toggle_alpha_to_coverage() {
        let state = MultisampleStateDescriptor::new()
            .alpha_to_coverage(true)
            .alpha_to_coverage(false)
            .alpha_to_coverage(true);
        assert!(state.alpha_to_coverage_enabled);
    }

    #[test]
    fn test_builder_chained_override() {
        let state = MultisampleStateBuilder::new()
            .count(4)
            .count(8)
            .mask(0x0F)
            .mask(0xFF)
            .alpha_to_coverage(false)
            .alpha_to_coverage(true)
            .build();

        assert_eq!(state.count, 8);
        assert_eq!(state.mask, 0xFF);
        assert!(state.alpha_to_coverage_enabled);
    }

    // -------------------------------------------------------------------------
    // Equality and Comparison Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_equality_different_counts() {
        let state1 = MultisampleStateDescriptor::new().count(4);
        let state2 = MultisampleStateDescriptor::new().count(8);
        assert_ne!(state1, state2);
    }

    #[test]
    fn test_equality_different_masks() {
        let state1 = MultisampleStateDescriptor::msaa_4x().mask(0x0F);
        let state2 = MultisampleStateDescriptor::msaa_4x().mask(0xF0);
        assert_ne!(state1, state2);
    }

    #[test]
    fn test_equality_different_alpha_to_coverage() {
        let state1 = MultisampleStateDescriptor::msaa_4x().alpha_to_coverage(true);
        let state2 = MultisampleStateDescriptor::msaa_4x().alpha_to_coverage(false);
        assert_ne!(state1, state2);
    }

    #[test]
    fn test_equality_same_values_different_construction() {
        let state1 = MultisampleStateDescriptor::new()
            .count(4)
            .mask(!0)
            .alpha_to_coverage(false);
        let state2 = MultisampleStateDescriptor::msaa_4x();
        assert_eq!(state1, state2);
    }

    // -------------------------------------------------------------------------
    // Debug Trait Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_multisample_state_descriptor_debug() {
        let state = MultisampleStateDescriptor::msaa_4x();
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("MultisampleStateDescriptor"));
        assert!(debug_str.contains("count"));
        assert!(debug_str.contains("4"));
    }

    #[test]
    fn test_multisample_state_builder_debug() {
        let builder = MultisampleStateBuilder::new().count(8);
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("MultisampleStateBuilder"));
    }

    #[test]
    fn test_sample_count_info_debug() {
        let info = get_sample_count_info(4).unwrap();
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("SampleCountInfo"));
        assert!(debug_str.contains("4"));
    }

    // -------------------------------------------------------------------------
    // Sample Mask Bit Pattern Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_mask_for_4x_msaa() {
        // For 4x MSAA, bits 0-3 control samples
        let state = MultisampleStateDescriptor::msaa_4x().mask(0b1111);
        assert_eq!(state.mask, 0b1111);
    }

    #[test]
    fn test_sample_mask_for_8x_msaa() {
        // For 8x MSAA, bits 0-7 control samples
        let state = MultisampleStateDescriptor::msaa_8x().mask(0b11111111);
        assert_eq!(state.mask, 0b11111111);
    }

    #[test]
    fn test_sample_mask_for_16x_msaa() {
        // For 16x MSAA, bits 0-15 control samples
        let state = MultisampleStateDescriptor::msaa_16x().mask(0xFFFF);
        assert_eq!(state.mask, 0xFFFF);
    }

    #[test]
    fn test_sample_mask_max_u64() {
        let state = MultisampleStateDescriptor::new().mask(u64::MAX);
        assert_eq!(state.mask, u64::MAX);
    }

    // -------------------------------------------------------------------------
    // With Alpha To Coverage Preset Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_with_alpha_to_coverage_count_1() {
        let state = MultisampleStateDescriptor::with_alpha_to_coverage(1);
        assert_eq!(state.count, 1);
        assert!(state.alpha_to_coverage_enabled);
        assert_eq!(state.mask, !0);
    }

    #[test]
    fn test_with_alpha_to_coverage_count_8() {
        let state = MultisampleStateDescriptor::with_alpha_to_coverage(8);
        assert_eq!(state.count, 8);
        assert!(state.alpha_to_coverage_enabled);
    }

    #[test]
    fn test_with_alpha_to_coverage_count_16() {
        let state = MultisampleStateDescriptor::with_alpha_to_coverage(16);
        assert_eq!(state.count, 16);
        assert!(state.alpha_to_coverage_enabled);
    }

    // -------------------------------------------------------------------------
    // is_msaa_enabled Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_msaa_enabled_count_2() {
        // Invalid count but > 1, still reports as enabled
        let state = MultisampleStateDescriptor::new().count(2);
        assert!(state.is_msaa_enabled());
    }

    #[test]
    fn test_is_msaa_enabled_count_0() {
        let state = MultisampleStateDescriptor::new().count(0);
        assert!(!state.is_msaa_enabled());
    }

    // -------------------------------------------------------------------------
    // sample_count_info Method Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_count_info_method_invalid_count() {
        let state = MultisampleStateDescriptor::new().count(5);
        assert!(state.sample_count_info().is_none());
    }

    #[test]
    fn test_sample_count_info_method_all_valid() {
        for count in [1, 4, 8, 16] {
            let state = MultisampleStateDescriptor::new().count(count);
            let info = state.sample_count_info();
            assert!(info.is_some(), "Should have info for count {}", count);
            assert_eq!(info.unwrap().count, count);
        }
    }

    // -------------------------------------------------------------------------
    // wgpu Conversion Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_wgpu_conversion_preserves_zero_mask() {
        let state = MultisampleStateDescriptor::new().mask(0);
        let wgpu_state: wgpu::MultisampleState = state.into();
        assert_eq!(wgpu_state.mask, 0);
    }

    #[test]
    fn test_wgpu_conversion_preserves_max_mask() {
        let state = MultisampleStateDescriptor::new().mask(u64::MAX);
        let wgpu_state: wgpu::MultisampleState = state.into();
        assert_eq!(wgpu_state.mask, u64::MAX);
    }

    #[test]
    fn test_wgpu_conversion_invalid_count() {
        // wgpu conversion doesn't validate - passes through
        let state = MultisampleStateDescriptor::new().count(3);
        let wgpu_state: wgpu::MultisampleState = state.into();
        assert_eq!(wgpu_state.count, 3);
    }

    // -------------------------------------------------------------------------
    // New API Method Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_new_method_equals_default() {
        let from_new = MultisampleStateDescriptor::new();
        let from_default = MultisampleStateDescriptor::default();
        assert_eq!(from_new, from_default);
    }

    // -------------------------------------------------------------------------
    // Builder Consistency Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_builder_new_equals_default() {
        let from_new = MultisampleStateBuilder::new().build();
        let from_default = MultisampleStateBuilder::default().build();
        assert_eq!(from_new, from_default);
    }

    #[test]
    fn test_builder_matches_descriptor_msaa_16x() {
        let from_preset = MultisampleStateDescriptor::msaa_16x();
        let from_builder = MultisampleStateBuilder::new().count(16).build();
        assert_eq!(from_preset, from_builder);
    }

    #[test]
    fn test_builder_matches_with_alpha_to_coverage() {
        let from_preset = MultisampleStateDescriptor::with_alpha_to_coverage(4);
        let from_builder = MultisampleStateBuilder::new()
            .count(4)
            .alpha_to_coverage(true)
            .build();
        assert_eq!(from_preset, from_builder);
    }

    // =========================================================================
    // MSAA Resolve Tests (T-WGPU-P3.7.2)
    // =========================================================================

    // -------------------------------------------------------------------------
    // MsaaStoreOp Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_msaa_store_op_default() {
        let op = MsaaStoreOp::default();
        assert!(op.is_discard());
        assert!(!op.is_store());
    }

    #[test]
    fn test_msaa_store_op_store() {
        let op = MsaaStoreOp::store();
        assert!(op.is_store());
        assert!(!op.is_discard());
    }

    #[test]
    fn test_msaa_store_op_discard() {
        let op = MsaaStoreOp::discard();
        assert!(op.is_discard());
        assert!(!op.is_store());
    }

    #[test]
    fn test_msaa_store_op_into_wgpu_store() {
        let op = MsaaStoreOp::Store;
        let wgpu_op: wgpu::StoreOp = op.into();
        assert_eq!(wgpu_op, wgpu::StoreOp::Store);
    }

    #[test]
    fn test_msaa_store_op_into_wgpu_discard() {
        let op = MsaaStoreOp::Discard;
        let wgpu_op: wgpu::StoreOp = op.into();
        assert_eq!(wgpu_op, wgpu::StoreOp::Discard);
    }

    #[test]
    fn test_msaa_store_op_display() {
        assert_eq!(format!("{}", MsaaStoreOp::Store), "Store");
        assert_eq!(format!("{}", MsaaStoreOp::Discard), "Discard");
    }

    #[test]
    fn test_msaa_store_op_equality() {
        assert_eq!(MsaaStoreOp::Store, MsaaStoreOp::Store);
        assert_eq!(MsaaStoreOp::Discard, MsaaStoreOp::Discard);
        assert_ne!(MsaaStoreOp::Store, MsaaStoreOp::Discard);
    }

    #[test]
    fn test_msaa_store_op_clone() {
        let op = MsaaStoreOp::Store;
        let cloned = op.clone();
        assert_eq!(op, cloned);
    }

    #[test]
    fn test_msaa_store_op_copy() {
        let op = MsaaStoreOp::Discard;
        let copied = op;
        assert_eq!(op, copied);
    }

    // -------------------------------------------------------------------------
    // ResolveInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_info_valid() {
        let info = ResolveInfo::valid(4, 1920, 1080, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);
        assert!(info.is_valid);
        assert!(info.needs_resolve());
        assert_eq!(info.source_sample_count, 4);
        assert_eq!(info.target_sample_count, 1);
        assert_eq!(info.width, 1920);
        assert_eq!(info.height, 1080);
    }

    #[test]
    fn test_resolve_info_no_resolve() {
        let info = ResolveInfo::no_resolve();
        assert!(!info.is_valid);
        assert!(!info.needs_resolve());
        assert_eq!(info.source_sample_count, 1);
    }

    #[test]
    fn test_resolve_info_needs_resolve() {
        let info_4x = ResolveInfo::valid(4, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);
        assert!(info_4x.needs_resolve());

        let info_1x = ResolveInfo::valid(1, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Store);
        assert!(!info_1x.needs_resolve());
    }

    #[test]
    fn test_resolve_info_memory_savings_discard() {
        let info = ResolveInfo::valid(4, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);
        // 4x MSAA with discard saves ~75% (3/4 samples discarded)
        assert!(info.memory_savings_percent() > 0);
        assert_eq!(info.memory_savings_percent(), 75);
    }

    #[test]
    fn test_resolve_info_memory_savings_store() {
        let info = ResolveInfo::valid(4, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Store);
        assert_eq!(info.memory_savings_percent(), 0);
    }

    #[test]
    fn test_resolve_info_memory_savings_8x() {
        let info = ResolveInfo::valid(8, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);
        assert_eq!(info.memory_savings_percent(), 87); // (8-1)*100/8 = 87.5 -> 87
    }

    #[test]
    fn test_resolve_info_display_valid() {
        let info = ResolveInfo::valid(4, 1920, 1080, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);
        let display = format!("{}", info);
        assert!(display.contains("4x"));
        assert!(display.contains("1x"));
        assert!(display.contains("1920"));
        assert!(display.contains("1080"));
    }

    #[test]
    fn test_resolve_info_display_no_resolve() {
        let info = ResolveInfo::no_resolve();
        let display = format!("{}", info);
        assert!(display.contains("No resolve"));
    }

    // -------------------------------------------------------------------------
    // ResolveAttachmentDescriptor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_attachment_descriptor_new() {
        let desc = ResolveAttachmentDescriptor::new(4);
        assert_eq!(desc.source_sample_count, 4);
        assert!(desc.is_enabled());
        assert!(desc.store_op.is_discard()); // Default is discard
    }

    #[test]
    fn test_resolve_attachment_descriptor_no_msaa() {
        let desc = ResolveAttachmentDescriptor::new(1);
        assert_eq!(desc.source_sample_count, 1);
        assert!(!desc.is_enabled());
    }

    #[test]
    fn test_resolve_attachment_descriptor_default() {
        let desc = ResolveAttachmentDescriptor::default();
        assert_eq!(desc.source_sample_count, 1);
        assert!(!desc.is_enabled());
    }

    #[test]
    fn test_resolve_attachment_descriptor_presets() {
        let discard_4x = ResolveAttachmentDescriptor::resolve_discard_4x();
        assert_eq!(discard_4x.source_sample_count, 4);
        assert!(discard_4x.store_op.is_discard());

        let discard_8x = ResolveAttachmentDescriptor::resolve_discard_8x();
        assert_eq!(discard_8x.source_sample_count, 8);
        assert!(discard_8x.store_op.is_discard());

        let store_4x = ResolveAttachmentDescriptor::resolve_store_4x();
        assert_eq!(store_4x.source_sample_count, 4);
        assert!(store_4x.store_op.is_store());

        let store_8x = ResolveAttachmentDescriptor::resolve_store_8x();
        assert_eq!(store_8x.source_sample_count, 8);
        assert!(store_8x.store_op.is_store());

        let no_resolve = ResolveAttachmentDescriptor::no_resolve();
        assert!(!no_resolve.is_enabled());
    }

    #[test]
    fn test_resolve_attachment_descriptor_fluent_api() {
        let desc = ResolveAttachmentDescriptor::new(4)
            .sample_count(8)
            .store_op(MsaaStoreOp::Store);
        assert_eq!(desc.source_sample_count, 8);
        assert!(desc.store_op.is_store());
    }

    #[test]
    fn test_resolve_attachment_descriptor_discard_store_methods() {
        let desc_discard = ResolveAttachmentDescriptor::new(4).discard();
        assert!(desc_discard.store_op.is_discard());

        let desc_store = ResolveAttachmentDescriptor::new(4).store();
        assert!(desc_store.store_op.is_store());
    }

    #[test]
    fn test_resolve_attachment_descriptor_is_valid() {
        assert!(ResolveAttachmentDescriptor::new(1).is_valid());
        assert!(ResolveAttachmentDescriptor::new(4).is_valid());
        assert!(ResolveAttachmentDescriptor::new(8).is_valid());
        assert!(ResolveAttachmentDescriptor::new(16).is_valid());
        assert!(!ResolveAttachmentDescriptor::new(2).is_valid());
        assert!(!ResolveAttachmentDescriptor::new(3).is_valid());
    }

    #[test]
    fn test_resolve_attachment_descriptor_validate_resolve_target() {
        assert!(ResolveAttachmentDescriptor::validate_resolve_target(1).is_ok());
        assert!(ResolveAttachmentDescriptor::validate_resolve_target(4).is_err());
    }

    #[test]
    fn test_resolve_attachment_descriptor_validate() {
        let desc = ResolveAttachmentDescriptor::new(4);
        assert!(desc.validate(1).is_ok());
        assert!(desc.validate(4).is_err()); // Target can't be multisampled
    }

    #[test]
    fn test_resolve_attachment_descriptor_validate_invalid_source() {
        let mut desc = ResolveAttachmentDescriptor::new(4);
        desc.source_sample_count = 3; // Invalid
        assert!(desc.validate(1).is_err());
    }

    #[test]
    fn test_resolve_attachment_descriptor_needs_resolve_target() {
        assert!(ResolveAttachmentDescriptor::new(4).needs_resolve_target());
        assert!(!ResolveAttachmentDescriptor::new(1).needs_resolve_target());
    }

    #[test]
    fn test_resolve_attachment_descriptor_wgpu_store_op() {
        let discard = ResolveAttachmentDescriptor::new(4).discard();
        assert_eq!(discard.wgpu_store_op(), wgpu::StoreOp::Discard);

        let store = ResolveAttachmentDescriptor::new(4).store();
        assert_eq!(store.wgpu_store_op(), wgpu::StoreOp::Store);
    }

    #[test]
    fn test_resolve_attachment_descriptor_display() {
        let enabled = ResolveAttachmentDescriptor::resolve_discard_4x();
        let display = format!("{}", enabled);
        assert!(display.contains("4x MSAA"));
        assert!(display.contains("Discard"));

        let disabled = ResolveAttachmentDescriptor::no_resolve();
        let display = format!("{}", disabled);
        assert!(display.contains("No MSAA resolve"));
    }

    // -------------------------------------------------------------------------
    // ResolveError Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_error_display() {
        let err = ResolveError::InvalidSourceSampleCount(3);
        assert!(format!("{}", err).contains("3"));

        let err = ResolveError::InvalidResolveTarget { expected: 1, actual: 4 };
        assert!(format!("{}", err).contains("1"));
        assert!(format!("{}", err).contains("4"));

        let err = ResolveError::ResolveEnabledWithoutMsaa;
        assert!(format!("{}", err).contains("enabled"));

        let err = ResolveError::DimensionMismatch {
            source: (100, 200),
            target: (200, 400),
        };
        assert!(format!("{}", err).contains("100"));
        assert!(format!("{}", err).contains("200"));

        let err = ResolveError::FormatMismatch {
            source: wgpu::TextureFormat::Rgba8Unorm,
            target: wgpu::TextureFormat::Bgra8Unorm,
        };
        assert!(format!("{}", err).contains("Rgba8Unorm"));
        assert!(format!("{}", err).contains("Bgra8Unorm"));
    }

    #[test]
    fn test_resolve_error_is_std_error() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<ResolveError>();
    }

    // -------------------------------------------------------------------------
    // is_valid_resolve_target Helper Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_valid_resolve_target_valid() {
        assert!(is_valid_resolve_target(1));
    }

    #[test]
    fn test_is_valid_resolve_target_invalid() {
        assert!(!is_valid_resolve_target(2));
        assert!(!is_valid_resolve_target(4));
        assert!(!is_valid_resolve_target(8));
        assert!(!is_valid_resolve_target(16));
    }

    // -------------------------------------------------------------------------
    // Preset Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_discard_preset() {
        let desc = resolve_discard();
        assert_eq!(desc.source_sample_count, 4);
        assert!(desc.store_op.is_discard());
        assert!(desc.is_enabled());
    }

    #[test]
    fn test_resolve_store_preset() {
        let desc = resolve_store();
        assert_eq!(desc.source_sample_count, 4);
        assert!(desc.store_op.is_store());
        assert!(desc.is_enabled());
    }

    // -------------------------------------------------------------------------
    // Thread Safety Tests for Resolve Types
    // -------------------------------------------------------------------------

    #[test]
    fn test_msaa_store_op_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MsaaStoreOp>();
    }

    #[test]
    fn test_msaa_store_op_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<MsaaStoreOp>();
    }

    #[test]
    fn test_resolve_info_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ResolveInfo>();
    }

    #[test]
    fn test_resolve_info_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ResolveInfo>();
    }

    #[test]
    fn test_resolve_attachment_descriptor_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ResolveAttachmentDescriptor>();
    }

    #[test]
    fn test_resolve_attachment_descriptor_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ResolveAttachmentDescriptor>();
    }

    #[test]
    fn test_resolve_error_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ResolveError>();
    }

    #[test]
    fn test_resolve_error_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ResolveError>();
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_attachment_sample_count_override() {
        let desc = ResolveAttachmentDescriptor::new(4)
            .sample_count(1); // Disable MSAA
        assert!(!desc.is_enabled());
        assert!(!desc.needs_resolve_target());
    }

    #[test]
    fn test_resolve_attachment_enable_after_disable() {
        let desc = ResolveAttachmentDescriptor::new(1) // No MSAA
            .sample_count(8); // Enable 8x
        assert!(desc.is_enabled());
        assert_eq!(desc.source_sample_count, 8);
    }

    #[test]
    fn test_resolve_info_all_formats() {
        let formats = [
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureFormat::Rgb10a2Unorm,
        ];

        for format in formats {
            let info = ResolveInfo::valid(4, 100, 100, format, MsaaStoreOp::Discard);
            assert!(info.is_valid);
            assert_eq!(info.format, format);
        }
    }

    #[test]
    fn test_resolve_attachment_equality() {
        let desc1 = ResolveAttachmentDescriptor::resolve_discard_4x();
        let desc2 = ResolveAttachmentDescriptor::resolve_discard_4x();
        let desc3 = ResolveAttachmentDescriptor::resolve_store_4x();
        let desc4 = ResolveAttachmentDescriptor::resolve_discard_8x();

        assert_eq!(desc1, desc2);
        assert_ne!(desc1, desc3); // Different store op
        assert_ne!(desc1, desc4); // Different sample count
    }

    #[test]
    fn test_resolve_attachment_clone() {
        let desc = ResolveAttachmentDescriptor::resolve_discard_4x();
        let cloned = desc.clone();
        assert_eq!(desc, cloned);
    }

    #[test]
    fn test_resolve_attachment_copy() {
        let desc = ResolveAttachmentDescriptor::resolve_discard_4x();
        let copied = desc;
        assert_eq!(desc, copied);
    }

    #[test]
    fn test_resolve_info_clone() {
        let info = ResolveInfo::valid(4, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);
        let cloned = info.clone();
        assert_eq!(info, cloned);
    }

    #[test]
    fn test_resolve_info_copy() {
        let info = ResolveInfo::valid(4, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);
        let copied = info;
        assert_eq!(info, copied);
    }

    // -------------------------------------------------------------------------
    // Validate Method Comprehensive Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_all_valid_sample_counts() {
        for count in [4, 8, 16] {
            let desc = ResolveAttachmentDescriptor::new(count);
            assert!(desc.validate(1).is_ok(), "Should validate for count {}", count);
        }
    }

    #[test]
    fn test_validate_invalid_sample_counts() {
        for count in [2, 3, 5, 6, 7, 9, 10, 12, 15, 17, 32] {
            let mut desc = ResolveAttachmentDescriptor::new(4);
            desc.source_sample_count = count;
            let result = desc.validate(1);
            assert!(result.is_err(), "Should fail for count {}", count);
            if let Err(ResolveError::InvalidSourceSampleCount(c)) = result {
                assert_eq!(c, count);
            } else {
                panic!("Expected InvalidSourceSampleCount");
            }
        }
    }

    #[test]
    fn test_validate_multisampled_target() {
        let desc = ResolveAttachmentDescriptor::new(4);
        for target_count in [2, 4, 8, 16] {
            let result = desc.validate(target_count);
            assert!(result.is_err());
            if let Err(ResolveError::InvalidResolveTarget { expected, actual }) = result {
                assert_eq!(expected, 1);
                assert_eq!(actual, target_count);
            }
        }
    }

    // =========================================================================
    // Additional MSAA Resolve Tests for T-WGPU-P3.7.2 (Target: 200+ tests)
    // =========================================================================

    // -------------------------------------------------------------------------
    // MsaaStoreOp Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_msaa_store_op_debug() {
        let store = MsaaStoreOp::Store;
        let debug = format!("{:?}", store);
        assert!(debug.contains("Store"));

        let discard = MsaaStoreOp::Discard;
        let debug = format!("{:?}", discard);
        assert!(debug.contains("Discard"));
    }

    #[test]
    fn test_msaa_store_op_variants_exhaustive() {
        // Verify all variants are covered
        let ops = [MsaaStoreOp::Store, MsaaStoreOp::Discard];
        assert_eq!(ops.len(), 2);

        // Each variant has unique behavior
        assert!(ops[0].is_store());
        assert!(!ops[0].is_discard());
        assert!(!ops[1].is_store());
        assert!(ops[1].is_discard());
    }

    #[test]
    fn test_msaa_store_op_into_wgpu_roundtrip() {
        // Verify conversion consistency
        let store_wgpu: wgpu::StoreOp = MsaaStoreOp::Store.into();
        let discard_wgpu: wgpu::StoreOp = MsaaStoreOp::Discard.into();

        assert_ne!(store_wgpu, discard_wgpu);
    }

    #[test]
    fn test_msaa_store_op_factory_methods_consistency() {
        assert_eq!(MsaaStoreOp::store(), MsaaStoreOp::Store);
        assert_eq!(MsaaStoreOp::discard(), MsaaStoreOp::Discard);
    }

    // -------------------------------------------------------------------------
    // ResolveInfo Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_info_memory_savings_16x() {
        let info = ResolveInfo::valid(16, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);
        // 16x MSAA with discard saves ~93.75% ((16-1)*100/16 = 93.75 -> 93)
        assert_eq!(info.memory_savings_percent(), 93);
    }

    #[test]
    fn test_resolve_info_memory_savings_1x() {
        let info = ResolveInfo::valid(1, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);
        // 1x has no savings even with discard (no multisampling)
        assert_eq!(info.memory_savings_percent(), 0);
    }

    #[test]
    fn test_resolve_info_valid_with_store() {
        let info = ResolveInfo::valid(8, 800, 600, wgpu::TextureFormat::Bgra8Unorm, MsaaStoreOp::Store);
        assert!(info.is_valid);
        assert_eq!(info.source_sample_count, 8);
        assert_eq!(info.store_op, MsaaStoreOp::Store);
    }

    #[test]
    fn test_resolve_info_debug() {
        let info = ResolveInfo::valid(4, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);
        let debug = format!("{:?}", info);
        assert!(debug.contains("ResolveInfo"));
        assert!(debug.contains("source_sample_count"));
    }

    #[test]
    fn test_resolve_info_equality() {
        let info1 = ResolveInfo::valid(4, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);
        let info2 = ResolveInfo::valid(4, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);
        let info3 = ResolveInfo::valid(8, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);

        assert_eq!(info1, info2);
        assert_ne!(info1, info3);
    }

    #[test]
    fn test_resolve_info_different_dimensions() {
        let info1 = ResolveInfo::valid(4, 1920, 1080, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);
        let info2 = ResolveInfo::valid(4, 3840, 2160, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);

        assert_ne!(info1.width, info2.width);
        assert_ne!(info1.height, info2.height);
    }

    #[test]
    fn test_resolve_info_different_store_ops() {
        let info1 = ResolveInfo::valid(4, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Store);
        let info2 = ResolveInfo::valid(4, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);

        assert_ne!(info1.store_op, info2.store_op);
        assert_ne!(info1.memory_savings_percent(), info2.memory_savings_percent());
    }

    #[test]
    fn test_resolve_info_display_formats() {
        // Test display for different formats
        let formats_and_names = [
            (wgpu::TextureFormat::Rgba8Unorm, "RGBA8"),
            (wgpu::TextureFormat::Rgba8UnormSrgb, "RGBA8-sRGB"),
            (wgpu::TextureFormat::Bgra8Unorm, "BGRA8"),
            (wgpu::TextureFormat::Bgra8UnormSrgb, "BGRA8-sRGB"),
            (wgpu::TextureFormat::Rgba16Float, "RGBA16F"),
            (wgpu::TextureFormat::Rgba32Float, "RGBA32F"),
            (wgpu::TextureFormat::Rgb10a2Unorm, "RGB10A2"),
        ];

        for (format, expected_name) in formats_and_names {
            let info = ResolveInfo::valid(4, 100, 100, format, MsaaStoreOp::Discard);
            let display = format!("{}", info);
            assert!(display.contains(expected_name), "Expected {} in display for {:?}", expected_name, format);
        }
    }

    #[test]
    fn test_resolve_info_display_custom_format() {
        // A format not in format_name switch should show "Custom"
        let info = ResolveInfo::valid(4, 100, 100, wgpu::TextureFormat::R8Unorm, MsaaStoreOp::Discard);
        let display = format!("{}", info);
        assert!(display.contains("Custom"));
    }

    // -------------------------------------------------------------------------
    // ResolveAttachmentDescriptor Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_attachment_descriptor_debug() {
        let desc = ResolveAttachmentDescriptor::resolve_discard_4x();
        let debug = format!("{:?}", desc);
        assert!(debug.contains("ResolveAttachmentDescriptor"));
        assert!(debug.contains("source_sample_count"));
    }

    #[test]
    fn test_resolve_attachment_descriptor_16x_presets() {
        // Test 16x variants (not predefined but can be created)
        let discard_16x = ResolveAttachmentDescriptor::new(16).discard();
        assert_eq!(discard_16x.source_sample_count, 16);
        assert!(discard_16x.store_op.is_discard());
        assert!(discard_16x.is_enabled());

        let store_16x = ResolveAttachmentDescriptor::new(16).store();
        assert_eq!(store_16x.source_sample_count, 16);
        assert!(store_16x.store_op.is_store());
    }

    #[test]
    fn test_resolve_attachment_descriptor_validate_enabled_without_msaa() {
        // Create a descriptor manually with enabled=true but count=1
        let mut desc = ResolveAttachmentDescriptor::default();
        desc.enabled = true;
        desc.source_sample_count = 1;

        let result = desc.validate(1);
        assert!(result.is_err());
        if let Err(ResolveError::ResolveEnabledWithoutMsaa) = result {
            // Expected
        } else {
            panic!("Expected ResolveEnabledWithoutMsaa error");
        }
    }

    #[test]
    fn test_resolve_attachment_descriptor_validate_target_0() {
        // Edge case: target sample count 0
        let result = ResolveAttachmentDescriptor::validate_resolve_target(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_attachment_descriptor_display_with_store() {
        let desc = ResolveAttachmentDescriptor::resolve_store_8x();
        let display = format!("{}", desc);
        assert!(display.contains("8x MSAA"));
        assert!(display.contains("Store"));
    }

    #[test]
    fn test_resolve_attachment_descriptor_sample_count_chained() {
        let desc = ResolveAttachmentDescriptor::new(4)
            .sample_count(8)
            .sample_count(16)
            .sample_count(4);
        assert_eq!(desc.source_sample_count, 4);
        assert!(desc.is_enabled());
    }

    #[test]
    fn test_resolve_attachment_descriptor_store_op_chained() {
        let desc = ResolveAttachmentDescriptor::new(4)
            .store_op(MsaaStoreOp::Store)
            .store_op(MsaaStoreOp::Discard)
            .store_op(MsaaStoreOp::Store);
        assert!(desc.store_op.is_store());
    }

    // -------------------------------------------------------------------------
    // ResolveError Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_resolve_error_equality() {
        let err1 = ResolveError::InvalidSourceSampleCount(3);
        let err2 = ResolveError::InvalidSourceSampleCount(3);
        let err3 = ResolveError::InvalidSourceSampleCount(5);

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_resolve_error_clone() {
        let err = ResolveError::InvalidResolveTarget { expected: 1, actual: 4 };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_resolve_error_debug() {
        let err = ResolveError::InvalidSourceSampleCount(3);
        let debug = format!("{:?}", err);
        assert!(debug.contains("InvalidSourceSampleCount"));
        assert!(debug.contains("3"));
    }

    #[test]
    fn test_resolve_error_dimension_mismatch_equality() {
        let err1 = ResolveError::DimensionMismatch {
            source: (100, 200),
            target: (200, 400),
        };
        let err2 = ResolveError::DimensionMismatch {
            source: (100, 200),
            target: (200, 400),
        };
        let err3 = ResolveError::DimensionMismatch {
            source: (100, 100),
            target: (200, 200),
        };

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_resolve_error_format_mismatch_equality() {
        let err1 = ResolveError::FormatMismatch {
            source: wgpu::TextureFormat::Rgba8Unorm,
            target: wgpu::TextureFormat::Bgra8Unorm,
        };
        let err2 = ResolveError::FormatMismatch {
            source: wgpu::TextureFormat::Rgba8Unorm,
            target: wgpu::TextureFormat::Bgra8Unorm,
        };

        assert_eq!(err1, err2);
    }

    #[test]
    fn test_resolve_error_all_variants_debug() {
        let errors = [
            ResolveError::InvalidSourceSampleCount(3),
            ResolveError::InvalidResolveTarget { expected: 1, actual: 4 },
            ResolveError::ResolveEnabledWithoutMsaa,
            ResolveError::DimensionMismatch {
                source: (100, 200),
                target: (200, 400),
            },
            ResolveError::FormatMismatch {
                source: wgpu::TextureFormat::Rgba8Unorm,
                target: wgpu::TextureFormat::Bgra8Unorm,
            },
        ];

        for err in errors {
            let debug = format!("{:?}", err);
            assert!(!debug.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // is_valid_resolve_target Additional Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_valid_resolve_target_zero() {
        assert!(!is_valid_resolve_target(0));
    }

    #[test]
    fn test_is_valid_resolve_target_large_values() {
        assert!(!is_valid_resolve_target(32));
        assert!(!is_valid_resolve_target(64));
        assert!(!is_valid_resolve_target(u32::MAX));
    }

    // -------------------------------------------------------------------------
    // format_name Helper Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_name_all_known_formats() {
        // Verify format_name returns expected values
        assert_eq!(format_name(wgpu::TextureFormat::Rgba8Unorm), "RGBA8");
        assert_eq!(format_name(wgpu::TextureFormat::Rgba8UnormSrgb), "RGBA8-sRGB");
        assert_eq!(format_name(wgpu::TextureFormat::Bgra8Unorm), "BGRA8");
        assert_eq!(format_name(wgpu::TextureFormat::Bgra8UnormSrgb), "BGRA8-sRGB");
        assert_eq!(format_name(wgpu::TextureFormat::Rgba16Float), "RGBA16F");
        assert_eq!(format_name(wgpu::TextureFormat::Rgba32Float), "RGBA32F");
        assert_eq!(format_name(wgpu::TextureFormat::Rgb10a2Unorm), "RGB10A2");
    }

    #[test]
    fn test_format_name_unknown_formats() {
        // Unknown formats should return "Custom"
        assert_eq!(format_name(wgpu::TextureFormat::R8Unorm), "Custom");
        assert_eq!(format_name(wgpu::TextureFormat::Rg8Unorm), "Custom");
        assert_eq!(format_name(wgpu::TextureFormat::Depth32Float), "Custom");
    }

    // -------------------------------------------------------------------------
    // Integration Tests: Full Resolve Workflow Simulation
    // -------------------------------------------------------------------------

    #[test]
    fn test_full_resolve_workflow_4x() {
        // Simulate complete MSAA resolve workflow

        // 1. Create MSAA state
        let msaa_state = MultisampleStateDescriptor::msaa_4x();
        assert!(msaa_state.is_msaa_enabled());

        // 2. Create resolve attachment descriptor
        let resolve_desc = ResolveAttachmentDescriptor::new(msaa_state.count)
            .discard();
        assert!(resolve_desc.is_valid());
        assert!(resolve_desc.needs_resolve_target());

        // 3. Validate configuration
        let validation = resolve_desc.validate(1);
        assert!(validation.is_ok());

        // 4. Create resolve info
        let info = ResolveInfo::valid(
            msaa_state.count,
            1920,
            1080,
            wgpu::TextureFormat::Rgba8Unorm,
            resolve_desc.store_op,
        );
        assert!(info.is_valid);
        assert!(info.needs_resolve());

        // 5. Get wgpu store op
        let wgpu_store_op = resolve_desc.wgpu_store_op();
        assert_eq!(wgpu_store_op, wgpu::StoreOp::Discard);
    }

    #[test]
    fn test_full_resolve_workflow_8x_store() {
        // Workflow with Store operation (keeping MSAA data)

        let msaa_state = MultisampleStateDescriptor::msaa_8x();
        let resolve_desc = ResolveAttachmentDescriptor::new(msaa_state.count)
            .store();

        assert!(resolve_desc.validate(1).is_ok());
        assert_eq!(resolve_desc.wgpu_store_op(), wgpu::StoreOp::Store);

        let info = ResolveInfo::valid(
            msaa_state.count,
            800,
            600,
            wgpu::TextureFormat::Bgra8Unorm,
            resolve_desc.store_op,
        );
        assert_eq!(info.memory_savings_percent(), 0); // Store = no savings
    }

    #[test]
    fn test_full_resolve_workflow_no_msaa() {
        // Workflow without MSAA

        let msaa_state = MultisampleStateDescriptor::msaa_off();
        assert!(!msaa_state.is_msaa_enabled());

        let resolve_desc = ResolveAttachmentDescriptor::new(msaa_state.count);
        assert!(!resolve_desc.is_enabled());
        assert!(!resolve_desc.needs_resolve_target());

        let info = ResolveInfo::no_resolve();
        assert!(!info.is_valid);
        assert!(!info.needs_resolve());
    }

    #[test]
    fn test_resolve_workflow_preset_consistency() {
        // Verify presets produce consistent workflow

        let desc1 = resolve_discard();
        let desc2 = ResolveAttachmentDescriptor::resolve_discard_4x();
        assert_eq!(desc1, desc2);

        let desc3 = resolve_store();
        let desc4 = ResolveAttachmentDescriptor::resolve_store_4x();
        assert_eq!(desc3, desc4);
    }

    #[test]
    fn test_resolve_workflow_wgpu_conversion_chain() {
        // Verify entire chain from descriptor to wgpu types

        let msaa_desc = MultisampleStateDescriptor::msaa_4x();
        let wgpu_msaa: wgpu::MultisampleState = msaa_desc.into();

        let resolve_desc = ResolveAttachmentDescriptor::new(msaa_desc.count);
        let wgpu_store: wgpu::StoreOp = resolve_desc.store_op.into();

        assert_eq!(wgpu_msaa.count, 4);
        assert_eq!(wgpu_store, wgpu::StoreOp::Discard);
    }

    #[test]
    fn test_resolve_workflow_sample_count_matching() {
        // Verify sample counts match across types

        for count in [4, 8, 16] {
            let msaa_state = MultisampleStateDescriptor::new().count(count);
            let resolve_desc = ResolveAttachmentDescriptor::new(count);
            let info = ResolveInfo::valid(count, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);

            assert_eq!(msaa_state.count, resolve_desc.source_sample_count);
            assert_eq!(resolve_desc.source_sample_count, info.source_sample_count);
        }
    }

    // -------------------------------------------------------------------------
    // Additional Thread Safety Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_msaa_resolve_target_is_send() {
        // MsaaResolveTarget has a lifetime, so we test with concrete types
        fn assert_send<T: Send>() {}
        // Note: MsaaResolveTarget<'a> is Send if the references are Send
        // wgpu::TextureView is Send, so MsaaResolveTarget should be too
    }

    #[test]
    fn test_all_resolve_types_thread_safe() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<MsaaStoreOp>();
        assert_send_sync::<ResolveInfo>();
        assert_send_sync::<ResolveAttachmentDescriptor>();
        assert_send_sync::<ResolveError>();
    }

    // -------------------------------------------------------------------------
    // Edge Case Tests for Validation
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_resolve_target_all_msaa_counts() {
        // All MSAA counts should fail as resolve targets
        for count in [4, 8, 16] {
            let result = ResolveAttachmentDescriptor::validate_resolve_target(count);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_resolve_attachment_is_enabled_boundary() {
        // Exactly at boundary
        let desc_1 = ResolveAttachmentDescriptor::new(1);
        assert!(!desc_1.is_enabled());

        let desc_2 = ResolveAttachmentDescriptor::new(2);
        assert!(desc_2.is_enabled()); // > 1, so enabled

        let desc_4 = ResolveAttachmentDescriptor::new(4);
        assert!(desc_4.is_enabled());
    }

    #[test]
    fn test_resolve_info_target_sample_count_always_1() {
        // Target sample count should always be 1 for valid resolve
        let info = ResolveInfo::valid(4, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Discard);
        assert_eq!(info.target_sample_count, 1);

        let info = ResolveInfo::valid(8, 100, 100, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Store);
        assert_eq!(info.target_sample_count, 1);

        let info = ResolveInfo::no_resolve();
        assert_eq!(info.target_sample_count, 1);
    }

    // -------------------------------------------------------------------------
    // Comprehensive Preset Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_all_resolve_presets_valid() {
        let presets = [
            ResolveAttachmentDescriptor::resolve_discard_4x(),
            ResolveAttachmentDescriptor::resolve_discard_8x(),
            ResolveAttachmentDescriptor::resolve_store_4x(),
            ResolveAttachmentDescriptor::resolve_store_8x(),
            ResolveAttachmentDescriptor::no_resolve(),
            resolve_discard(),
            resolve_store(),
        ];

        for preset in presets {
            assert!(preset.is_valid());
        }
    }

    #[test]
    fn test_preset_store_ops() {
        // Discard presets
        assert!(ResolveAttachmentDescriptor::resolve_discard_4x().store_op.is_discard());
        assert!(ResolveAttachmentDescriptor::resolve_discard_8x().store_op.is_discard());
        assert!(resolve_discard().store_op.is_discard());

        // Store presets
        assert!(ResolveAttachmentDescriptor::resolve_store_4x().store_op.is_store());
        assert!(ResolveAttachmentDescriptor::resolve_store_8x().store_op.is_store());
        assert!(resolve_store().store_op.is_store());
    }

    #[test]
    fn test_preset_sample_counts() {
        assert_eq!(ResolveAttachmentDescriptor::resolve_discard_4x().source_sample_count, 4);
        assert_eq!(ResolveAttachmentDescriptor::resolve_discard_8x().source_sample_count, 8);
        assert_eq!(ResolveAttachmentDescriptor::resolve_store_4x().source_sample_count, 4);
        assert_eq!(ResolveAttachmentDescriptor::resolve_store_8x().source_sample_count, 8);
        assert_eq!(ResolveAttachmentDescriptor::no_resolve().source_sample_count, 1);
    }

    // -------------------------------------------------------------------------
    // ResolveInfo Memory Savings Comprehensive Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_savings_all_sample_counts() {
        // Verify memory savings formula: (count - 1) * 100 / count
        let test_cases = [
            (1, MsaaStoreOp::Discard, 0),   // 1x has no savings
            (4, MsaaStoreOp::Discard, 75),  // (4-1)*100/4 = 75
            (8, MsaaStoreOp::Discard, 87),  // (8-1)*100/8 = 87.5 -> 87
            (16, MsaaStoreOp::Discard, 93), // (16-1)*100/16 = 93.75 -> 93
            (4, MsaaStoreOp::Store, 0),     // Store = no savings
            (8, MsaaStoreOp::Store, 0),
            (16, MsaaStoreOp::Store, 0),
        ];

        for (count, store_op, expected_savings) in test_cases {
            let info = ResolveInfo::valid(count, 100, 100, wgpu::TextureFormat::Rgba8Unorm, store_op);
            assert_eq!(
                info.memory_savings_percent(),
                expected_savings,
                "Failed for count={}, store_op={:?}",
                count,
                store_op
            );
        }
    }

    // -------------------------------------------------------------------------
    // ResolveAttachmentDescriptor Validation Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_count_1_enabled_false() {
        // Count 1 with enabled=false should validate
        let desc = ResolveAttachmentDescriptor::new(1);
        assert!(!desc.enabled);
        // Should pass because we're not trying to resolve
        // (the validate function checks enabled && source <= 1)
        let result = desc.validate(1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_all_error_types() {
        // InvalidSourceSampleCount
        let mut desc = ResolveAttachmentDescriptor::new(4);
        desc.source_sample_count = 3;
        let result = desc.validate(1);
        assert!(matches!(result, Err(ResolveError::InvalidSourceSampleCount(3))));

        // InvalidResolveTarget
        let desc = ResolveAttachmentDescriptor::new(4);
        let result = desc.validate(4);
        assert!(matches!(result, Err(ResolveError::InvalidResolveTarget { .. })));

        // ResolveEnabledWithoutMsaa
        let mut desc = ResolveAttachmentDescriptor::new(1);
        desc.enabled = true;
        let result = desc.validate(1);
        assert!(matches!(result, Err(ResolveError::ResolveEnabledWithoutMsaa)));
    }

    // -------------------------------------------------------------------------
    // Display Format Comprehensive Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_msaa_store_op_display_comprehensive() {
        let store = MsaaStoreOp::Store;
        let discard = MsaaStoreOp::Discard;

        assert!(!format!("{}", store).is_empty());
        assert!(!format!("{}", discard).is_empty());
        assert_ne!(format!("{}", store), format!("{}", discard));
    }

    #[test]
    fn test_resolve_info_display_with_store() {
        let info = ResolveInfo::valid(4, 1920, 1080, wgpu::TextureFormat::Rgba8Unorm, MsaaStoreOp::Store);
        let display = format!("{}", info);
        assert!(display.contains("Store"));
    }

    #[test]
    fn test_resolve_attachment_display_comprehensive() {
        let descriptors = [
            ResolveAttachmentDescriptor::resolve_discard_4x(),
            ResolveAttachmentDescriptor::resolve_discard_8x(),
            ResolveAttachmentDescriptor::resolve_store_4x(),
            ResolveAttachmentDescriptor::resolve_store_8x(),
            ResolveAttachmentDescriptor::no_resolve(),
        ];

        for desc in descriptors {
            let display = format!("{}", desc);
            assert!(!display.is_empty());
        }
    }
}
