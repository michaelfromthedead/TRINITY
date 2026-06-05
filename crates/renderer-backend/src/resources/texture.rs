//! Texture creation and management for TRINITY.
//!
//! This module provides the texture creation API for the TRINITY wgpu abstraction layer.
//! It wraps wgpu's texture creation with validation, memory estimation, and metadata tracking.
//!
//! # Overview
//!
//! Texture creation in wgpu requires specifying dimensions, format, usage, and other parameters.
//! This module provides:
//!
//! - [`TrinityTexture`] - Wrapper around wgpu::Texture with metadata and default view
//! - [`TrinityTextureDescriptor`] - Texture creation parameters with view_formats support
//! - [`create_texture`] - Validated texture creation with logging and memory estimation
//! - [`texture_usages`] - Common usage flag presets
//! - Helper functions for mip count calculation and memory estimation
//!
//! # Texture Format Groups
//!
//! Formats are grouped by their characteristics:
//!
//! | Group | Formats | Bytes/Pixel |
//! |-------|---------|-------------|
//! | 8-bit | R8Unorm, R8Snorm, R8Uint, R8Sint | 1 |
//! | 16-bit | R16Uint, R16Sint, R16Float, Rg8Unorm, etc. | 2 |
//! | 32-bit | R32Uint, R32Sint, R32Float, Rgba8Unorm, Bgra8Unorm | 4 |
//! | 64-bit | Rg32Uint, Rg32Sint, Rg32Float, Rgba16Float | 8 |
//! | 128-bit | Rgba32Uint, Rgba32Sint, Rgba32Float | 16 |
//! | Depth | Depth16Unorm (2), Depth24Plus (3), Depth32Float (4) | varies |
//! | Compressed | BC1-BC7, ETC2, ASTC | varies by block |
//!
//! # View Format Reinterpretation
//!
//! The `view_formats` field allows creating texture views with compatible formats.
//! Compatible formats must have the same block size. Common patterns:
//!
//! - Rgba8Unorm <-> Rgba8UnormSrgb (sRGB reinterpretation)
//! - Bgra8Unorm <-> Bgra8UnormSrgb (sRGB reinterpretation)
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::texture::{TrinityTextureDescriptor, create_texture, texture_usages};
//! use wgpu::{TextureDimension, TextureFormat, Extent3d};
//!
//! # fn example(device: &wgpu::Device) {
//! let desc = TrinityTextureDescriptor {
//!     label: Some("diffuse_texture"),
//!     size: Extent3d { width: 1024, height: 1024, depth_or_array_layers: 1 },
//!     mip_level_count: 0, // Auto-calculate
//!     sample_count: 1,
//!     dimension: TextureDimension::D2,
//!     format: TextureFormat::Rgba8UnormSrgb,
//!     usage: texture_usages::SAMPLED,
//!     view_formats: &[TextureFormat::Rgba8Unorm], // Allow non-sRGB view
//! };
//!
//! let texture = create_texture(device, &desc);
//! println!("Created texture: {}x{}, {} mips",
//!     texture.size().width, texture.size().height, texture.mip_level_count());
//! # }
//! ```

use log::debug;
use wgpu::{
    Device, Extent3d, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureView, TextureViewDescriptor,
};

// ============================================================================
// Texture Usage Presets
// ============================================================================

/// Common texture usage presets for typical use cases.
///
/// These presets combine the necessary flags for common texture patterns,
/// reducing boilerplate and preventing invalid combinations.
///
/// # Usage Presets Reference
///
/// | Preset | Flags | Use Case |
/// |--------|-------|----------|
/// | `SAMPLED` | `TEXTURE_BINDING \| COPY_DST` | Shader sampling with CPU upload |
/// | `STORAGE` | `STORAGE_BINDING \| COPY_DST` | Compute shader read/write |
/// | `RENDER_TARGET` | `RENDER_ATTACHMENT \| TEXTURE_BINDING` | Render to texture |
/// | `RENDER_TARGET_COPY` | `RENDER_ATTACHMENT \| COPY_SRC` | Render and readback |
/// | `DEPTH_TARGET` | `RENDER_ATTACHMENT \| TEXTURE_BINDING` | Depth buffer (same as RENDER_TARGET) |
/// | `FULL` | All common flags | Maximum flexibility |
pub mod texture_usages {
    use wgpu::TextureUsages;

    /// Texture for shader sampling with CPU upload capability.
    ///
    /// Combines `TEXTURE_BINDING` for use in shaders with `COPY_DST`
    /// for uploading texture data via staging buffer or queue write.
    pub const SAMPLED: TextureUsages =
        TextureUsages::TEXTURE_BINDING.union(TextureUsages::COPY_DST);

    /// Storage texture for compute shader read/write.
    ///
    /// Combines `STORAGE_BINDING` for compute shader access with `COPY_DST`
    /// for initial data upload.
    pub const STORAGE: TextureUsages =
        TextureUsages::STORAGE_BINDING.union(TextureUsages::COPY_DST);

    /// Render target that can also be sampled.
    ///
    /// Combines `RENDER_ATTACHMENT` for use as a render target with
    /// `TEXTURE_BINDING` for sampling in subsequent passes.
    pub const RENDER_TARGET: TextureUsages =
        TextureUsages::RENDER_ATTACHMENT.union(TextureUsages::TEXTURE_BINDING);

    /// Render target with copy source capability.
    ///
    /// Combines `RENDER_ATTACHMENT` for rendering with `COPY_SRC`
    /// for reading back results or copying to other textures.
    pub const RENDER_TARGET_COPY: TextureUsages =
        TextureUsages::RENDER_ATTACHMENT.union(TextureUsages::COPY_SRC);

    /// Depth buffer that can be sampled.
    ///
    /// Same as `RENDER_TARGET` but semantically for depth textures.
    pub const DEPTH_TARGET: TextureUsages =
        TextureUsages::RENDER_ATTACHMENT.union(TextureUsages::TEXTURE_BINDING);

    /// Full capability texture.
    ///
    /// All common flags combined for maximum flexibility. Use when you
    /// need the texture for multiple purposes or are prototyping.
    pub const FULL: TextureUsages = TextureUsages::TEXTURE_BINDING
        .union(TextureUsages::STORAGE_BINDING)
        .union(TextureUsages::RENDER_ATTACHMENT)
        .union(TextureUsages::COPY_SRC)
        .union(TextureUsages::COPY_DST);
}

// ============================================================================
// Format Utilities
// ============================================================================

/// Returns the number of bytes per texel (pixel) for a given texture format.
///
/// For compressed formats, this returns the bytes per block divided by the
/// block area, which gives an approximate bytes per texel for estimation.
///
/// # Arguments
///
/// * `format` - The texture format to query
///
/// # Returns
///
/// Bytes per texel (or approximate for compressed formats).
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture::bytes_per_pixel;
/// use wgpu::TextureFormat;
///
/// assert_eq!(bytes_per_pixel(TextureFormat::R8Unorm), 1);
/// assert_eq!(bytes_per_pixel(TextureFormat::Rgba8Unorm), 4);
/// assert_eq!(bytes_per_pixel(TextureFormat::Rgba16Float), 8);
/// assert_eq!(bytes_per_pixel(TextureFormat::Rgba32Float), 16);
/// ```
pub const fn bytes_per_pixel(format: TextureFormat) -> u32 {
    match format {
        // 8-bit formats (1 byte)
        TextureFormat::R8Unorm
        | TextureFormat::R8Snorm
        | TextureFormat::R8Uint
        | TextureFormat::R8Sint
        | TextureFormat::Stencil8 => 1,

        // 16-bit formats (2 bytes)
        TextureFormat::R16Uint
        | TextureFormat::R16Sint
        | TextureFormat::R16Float
        | TextureFormat::Rg8Unorm
        | TextureFormat::Rg8Snorm
        | TextureFormat::Rg8Uint
        | TextureFormat::Rg8Sint
        | TextureFormat::Depth16Unorm => 2,

        // 24-bit formats (3 bytes, but often padded to 4 in practice)
        TextureFormat::Depth24Plus => 4, // Implementation-dependent, assume 4

        // 32-bit formats (4 bytes)
        TextureFormat::R32Uint
        | TextureFormat::R32Sint
        | TextureFormat::R32Float
        | TextureFormat::Rg16Uint
        | TextureFormat::Rg16Sint
        | TextureFormat::Rg16Float
        | TextureFormat::Rgba8Unorm
        | TextureFormat::Rgba8UnormSrgb
        | TextureFormat::Rgba8Snorm
        | TextureFormat::Rgba8Uint
        | TextureFormat::Rgba8Sint
        | TextureFormat::Bgra8Unorm
        | TextureFormat::Bgra8UnormSrgb
        | TextureFormat::Rgb9e5Ufloat
        | TextureFormat::Rgb10a2Uint
        | TextureFormat::Rgb10a2Unorm
        | TextureFormat::Rg11b10Float
        | TextureFormat::Depth32Float
        | TextureFormat::Depth24PlusStencil8 => 4,

        // 64-bit formats (8 bytes)
        TextureFormat::Rg32Uint
        | TextureFormat::Rg32Sint
        | TextureFormat::Rg32Float
        | TextureFormat::Rgba16Uint
        | TextureFormat::Rgba16Sint
        | TextureFormat::Rgba16Float
        | TextureFormat::Depth32FloatStencil8 => 8,

        // 128-bit formats (16 bytes)
        TextureFormat::Rgba32Uint | TextureFormat::Rgba32Sint | TextureFormat::Rgba32Float => 16,

        // BC compressed formats (4x4 blocks)
        // BC1: 8 bytes per 16 pixels = 0.5 bytes/pixel
        TextureFormat::Bc1RgbaUnorm | TextureFormat::Bc1RgbaUnormSrgb => 1,
        // BC2, BC3: 16 bytes per 16 pixels = 1 byte/pixel
        TextureFormat::Bc2RgbaUnorm
        | TextureFormat::Bc2RgbaUnormSrgb
        | TextureFormat::Bc3RgbaUnorm
        | TextureFormat::Bc3RgbaUnormSrgb => 1,
        // BC4: 8 bytes per 16 pixels = 0.5 bytes/pixel
        TextureFormat::Bc4RUnorm | TextureFormat::Bc4RSnorm => 1,
        // BC5: 16 bytes per 16 pixels = 1 byte/pixel
        TextureFormat::Bc5RgUnorm | TextureFormat::Bc5RgSnorm => 1,
        // BC6H, BC7: 16 bytes per 16 pixels = 1 byte/pixel
        TextureFormat::Bc6hRgbUfloat
        | TextureFormat::Bc6hRgbFloat
        | TextureFormat::Bc7RgbaUnorm
        | TextureFormat::Bc7RgbaUnormSrgb => 1,

        // ETC2 compressed formats (4x4 blocks)
        TextureFormat::Etc2Rgb8Unorm
        | TextureFormat::Etc2Rgb8UnormSrgb
        | TextureFormat::Etc2Rgb8A1Unorm
        | TextureFormat::Etc2Rgb8A1UnormSrgb => 1,
        TextureFormat::Etc2Rgba8Unorm | TextureFormat::Etc2Rgba8UnormSrgb => 1,
        TextureFormat::EacR11Unorm | TextureFormat::EacR11Snorm => 1,
        TextureFormat::EacRg11Unorm | TextureFormat::EacRg11Snorm => 1,

        // ASTC compressed formats (various block sizes, use 1 as approximation)
        TextureFormat::Astc { .. } => 1,

        // Fallback for unknown/future formats
        _ => 4,
    }
}

/// Calculates the number of bytes per texel for compressed formats more accurately.
///
/// This function returns the actual bytes for a 4x4 block (or other block size),
/// useful for precise memory calculations.
///
/// # Arguments
///
/// * `format` - The texture format to query
///
/// # Returns
///
/// A tuple of (bytes_per_block, block_width, block_height).
pub const fn block_info(format: TextureFormat) -> (u32, u32, u32) {
    match format {
        // Uncompressed formats: 1x1 blocks
        TextureFormat::R8Unorm
        | TextureFormat::R8Snorm
        | TextureFormat::R8Uint
        | TextureFormat::R8Sint => (1, 1, 1),

        TextureFormat::R16Uint
        | TextureFormat::R16Sint
        | TextureFormat::R16Float
        | TextureFormat::Rg8Unorm
        | TextureFormat::Rg8Snorm
        | TextureFormat::Rg8Uint
        | TextureFormat::Rg8Sint => (2, 1, 1),

        TextureFormat::R32Uint
        | TextureFormat::R32Sint
        | TextureFormat::R32Float
        | TextureFormat::Rg16Uint
        | TextureFormat::Rg16Sint
        | TextureFormat::Rg16Float
        | TextureFormat::Rgba8Unorm
        | TextureFormat::Rgba8UnormSrgb
        | TextureFormat::Rgba8Snorm
        | TextureFormat::Rgba8Uint
        | TextureFormat::Rgba8Sint
        | TextureFormat::Bgra8Unorm
        | TextureFormat::Bgra8UnormSrgb
        | TextureFormat::Rgb10a2Unorm
        | TextureFormat::Rgb10a2Uint
        | TextureFormat::Rg11b10Float
        | TextureFormat::Rgb9e5Ufloat => (4, 1, 1),

        TextureFormat::Rg32Uint
        | TextureFormat::Rg32Sint
        | TextureFormat::Rg32Float
        | TextureFormat::Rgba16Uint
        | TextureFormat::Rgba16Sint
        | TextureFormat::Rgba16Float => (8, 1, 1),

        TextureFormat::Rgba32Uint | TextureFormat::Rgba32Sint | TextureFormat::Rgba32Float => {
            (16, 1, 1)
        }

        // Depth/stencil formats
        TextureFormat::Stencil8 => (1, 1, 1),
        TextureFormat::Depth16Unorm => (2, 1, 1),
        TextureFormat::Depth24Plus | TextureFormat::Depth32Float => (4, 1, 1),
        TextureFormat::Depth24PlusStencil8 => (4, 1, 1),
        TextureFormat::Depth32FloatStencil8 => (8, 1, 1),

        // BC compressed formats (4x4 blocks)
        TextureFormat::Bc1RgbaUnorm | TextureFormat::Bc1RgbaUnormSrgb => (8, 4, 4),
        TextureFormat::Bc2RgbaUnorm
        | TextureFormat::Bc2RgbaUnormSrgb
        | TextureFormat::Bc3RgbaUnorm
        | TextureFormat::Bc3RgbaUnormSrgb => (16, 4, 4),
        TextureFormat::Bc4RUnorm | TextureFormat::Bc4RSnorm => (8, 4, 4),
        TextureFormat::Bc5RgUnorm | TextureFormat::Bc5RgSnorm => (16, 4, 4),
        TextureFormat::Bc6hRgbUfloat | TextureFormat::Bc6hRgbFloat => (16, 4, 4),
        TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc7RgbaUnormSrgb => (16, 4, 4),

        // ETC2 compressed formats (4x4 blocks)
        TextureFormat::Etc2Rgb8Unorm
        | TextureFormat::Etc2Rgb8UnormSrgb
        | TextureFormat::Etc2Rgb8A1Unorm
        | TextureFormat::Etc2Rgb8A1UnormSrgb => (8, 4, 4),
        TextureFormat::Etc2Rgba8Unorm | TextureFormat::Etc2Rgba8UnormSrgb => (16, 4, 4),
        TextureFormat::EacR11Unorm | TextureFormat::EacR11Snorm => (8, 4, 4),
        TextureFormat::EacRg11Unorm | TextureFormat::EacRg11Snorm => (16, 4, 4),

        // ASTC compressed formats (various block sizes)
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B4x4,
            ..
        } => (16, 4, 4),
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B5x4,
            ..
        } => (16, 5, 4),
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B5x5,
            ..
        } => (16, 5, 5),
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B6x5,
            ..
        } => (16, 6, 5),
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B6x6,
            ..
        } => (16, 6, 6),
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x5,
            ..
        } => (16, 8, 5),
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x6,
            ..
        } => (16, 8, 6),
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x8,
            ..
        } => (16, 8, 8),
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B10x5,
            ..
        } => (16, 10, 5),
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B10x6,
            ..
        } => (16, 10, 6),
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B10x8,
            ..
        } => (16, 10, 8),
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B10x10,
            ..
        } => (16, 10, 10),
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B12x10,
            ..
        } => (16, 12, 10),
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B12x12,
            ..
        } => (16, 12, 12),

        // Fallback for unknown formats
        _ => (4, 1, 1),
    }
}

/// Calculates the recommended number of mip levels for a texture.
///
/// The mip chain goes down to 1x1 (or the smallest dimension reaching 1).
///
/// # Arguments
///
/// * `width` - Texture width in pixels
/// * `height` - Texture height in pixels
///
/// # Returns
///
/// The number of mip levels including the base level.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture::calculate_mip_count;
///
/// assert_eq!(calculate_mip_count(1024, 1024), 11); // 1024 -> 1
/// assert_eq!(calculate_mip_count(512, 256), 10);   // 512 -> 1
/// assert_eq!(calculate_mip_count(1, 1), 1);        // Already at minimum
/// assert_eq!(calculate_mip_count(4, 4), 3);        // 4 -> 2 -> 1
/// ```
pub const fn calculate_mip_count(width: u32, height: u32) -> u32 {
    let max_dim = if width > height { width } else { height };
    if max_dim == 0 {
        return 1;
    }
    // log2(max_dim) + 1 = number of mip levels
    32 - max_dim.leading_zeros()
}

/// Calculates the recommended number of mip levels for a 3D texture.
///
/// # Arguments
///
/// * `width` - Texture width in pixels
/// * `height` - Texture height in pixels
/// * `depth` - Texture depth in pixels
///
/// # Returns
///
/// The number of mip levels including the base level.
pub const fn calculate_mip_count_3d(width: u32, height: u32, depth: u32) -> u32 {
    let max_dim = if width > height {
        if width > depth {
            width
        } else {
            depth
        }
    } else if height > depth {
        height
    } else {
        depth
    };

    if max_dim == 0 {
        return 1;
    }
    32 - max_dim.leading_zeros()
}

/// Estimates the total memory size of a texture including all mip levels.
///
/// This function calculates the memory for 2D textures and 2D arrays.
/// For `depth_or_array_layers > 1`, each layer gets the full mip chain.
///
/// # Arguments
///
/// * `extent` - The texture dimensions (width, height, depth_or_array_layers)
/// * `format` - The texture format
/// * `mip_levels` - Number of mip levels (1 for no mipmaps)
/// * `_array_layers` - Deprecated, use extent.depth_or_array_layers instead
///
/// # Returns
///
/// Estimated memory size in bytes.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture::estimate_texture_size;
/// use wgpu::{Extent3d, TextureFormat};
///
/// // 1024x1024 RGBA8 with full mip chain
/// let size = estimate_texture_size(
///     Extent3d { width: 1024, height: 1024, depth_or_array_layers: 1 },
///     TextureFormat::Rgba8Unorm,
///     11,
///     1,
/// );
/// // Base: 1024*1024*4 = 4MB, with mips ~5.33MB
/// assert!(size > 4_000_000 && size < 6_000_000);
/// ```
pub fn estimate_texture_size(
    extent: Extent3d,
    format: TextureFormat,
    mip_levels: u32,
    _array_layers: u32, // Kept for API compatibility, use extent.depth_or_array_layers
) -> u64 {
    let (bytes_per_block, block_w, block_h) = block_info(format);
    let layers = extent.depth_or_array_layers.max(1);

    let mut total_size: u64 = 0;

    for mip in 0..mip_levels {
        // Calculate dimensions at this mip level
        let mip_width = (extent.width >> mip).max(1);
        let mip_height = (extent.height >> mip).max(1);

        // Calculate block count (round up for compressed formats)
        let blocks_x = (mip_width + block_w - 1) / block_w;
        let blocks_y = (mip_height + block_h - 1) / block_h;

        let mip_size = (blocks_x as u64) * (blocks_y as u64) * (bytes_per_block as u64);
        total_size += mip_size;
    }

    total_size * (layers as u64)
}

/// Formats a byte size as a human-readable string with appropriate unit.
fn format_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2}MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{}B", bytes)
    }
}

// ============================================================================
// TrinityTextureDescriptor
// ============================================================================

/// Texture creation descriptor.
///
/// This struct describes the parameters for creating a new texture.
/// The `view_formats` field allows specifying compatible formats for
/// texture view reinterpretation.
///
/// # Automatic Mip Level Calculation
///
/// If `mip_level_count` is 0, the full mip chain will be calculated
/// automatically based on the texture dimensions.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::texture::{TrinityTextureDescriptor, texture_usages};
/// use wgpu::{TextureDimension, TextureFormat, Extent3d};
///
/// let desc = TrinityTextureDescriptor {
///     label: Some("albedo_texture"),
///     size: Extent3d { width: 2048, height: 2048, depth_or_array_layers: 1 },
///     mip_level_count: 0, // Auto-calculate full mip chain
///     sample_count: 1,
///     dimension: TextureDimension::D2,
///     format: TextureFormat::Rgba8UnormSrgb,
///     usage: texture_usages::SAMPLED,
///     view_formats: &[TextureFormat::Rgba8Unorm], // Allow linear view
/// };
/// ```
#[derive(Debug, Clone)]
pub struct TrinityTextureDescriptor<'a> {
    /// Debug label for the texture.
    ///
    /// This label appears in GPU debugging tools and error messages.
    pub label: Option<&'a str>,

    /// Texture dimensions.
    ///
    /// For 2D textures: width, height, and array_layers.
    /// For 3D textures: width, height, and depth.
    /// For cube maps: width (=height), and 6 * array_layers.
    pub size: Extent3d,

    /// Number of mip levels.
    ///
    /// Set to 0 for automatic calculation of the full mip chain.
    /// Set to 1 for no mipmapping.
    pub mip_level_count: u32,

    /// Number of MSAA samples.
    ///
    /// Must be 1 (no multisampling) or a power of 2 (2, 4, 8).
    /// Multisampled textures cannot have mipmaps.
    pub sample_count: u32,

    /// Texture dimension (1D, 2D, or 3D).
    pub dimension: TextureDimension,

    /// Texture format.
    pub format: TextureFormat,

    /// Usage flags for the texture.
    pub usage: TextureUsages,

    /// Additional formats that views can use.
    ///
    /// Allows creating texture views with formats different from the
    /// texture's creation format. Formats must be compatible (same block size).
    ///
    /// Common use case: `Rgba8UnormSrgb` texture with `Rgba8Unorm` view format
    /// for shaders that want linear color space access.
    pub view_formats: &'a [TextureFormat],
}

impl Default for TrinityTextureDescriptor<'_> {
    fn default() -> Self {
        Self {
            label: None,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: texture_usages::SAMPLED,
            view_formats: &[],
        }
    }
}

// ============================================================================
// TrinityTexture
// ============================================================================

/// TRINITY texture wrapper with metadata and default view.
///
/// This struct wraps a wgpu [`Texture`] with its default view and metadata.
/// The default view covers all mip levels and array layers.
///
/// # Creating Additional Views
///
/// Use [`create_view`](Self::create_view) to create views with different
/// parameters (e.g., single mip level, different format, subset of array layers).
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::texture::{TrinityTextureDescriptor, create_texture};
/// use wgpu::{TextureDimension, TextureFormat, Extent3d, TextureViewDescriptor};
///
/// # fn example(device: &wgpu::Device) {
/// let texture = create_texture(device, &TrinityTextureDescriptor {
///     label: Some("my_texture"),
///     size: Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
///     format: TextureFormat::Rgba8Unorm,
///     ..Default::default()
/// });
///
/// // Use the default view
/// let default_view = texture.view();
///
/// // Create a custom view for a single mip level
/// let mip_view = texture.create_view(&TextureViewDescriptor {
///     label: Some("mip0_view"),
///     base_mip_level: 0,
///     mip_level_count: Some(1),
///     ..Default::default()
/// });
/// # }
/// ```
pub struct TrinityTexture {
    /// The underlying wgpu texture.
    texture: Texture,
    /// Default texture view (all mips, all layers).
    view: TextureView,
    /// Texture dimensions.
    size: Extent3d,
    /// Texture format.
    format: TextureFormat,
    /// Texture usage flags.
    usage: TextureUsages,
    /// Number of mip levels.
    mip_level_count: u32,
    /// Number of samples.
    sample_count: u32,
    /// Texture dimension.
    dimension: TextureDimension,
    /// Optional debug label.
    label: Option<String>,
    /// Estimated memory size in bytes.
    estimated_size: u64,
}

impl TrinityTexture {
    /// Creates a TrinityTexture from raw wgpu components.
    ///
    /// Use this when wrapping textures created through other means.
    ///
    /// # Arguments
    ///
    /// * `texture` - The wgpu texture
    /// * `view` - The default texture view
    /// * `size` - Texture dimensions
    /// * `format` - Texture format
    /// * `usage` - Usage flags
    /// * `mip_level_count` - Number of mip levels
    /// * `sample_count` - MSAA sample count
    /// * `dimension` - Texture dimension
    /// * `label` - Optional debug label
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw(
        texture: Texture,
        view: TextureView,
        size: Extent3d,
        format: TextureFormat,
        usage: TextureUsages,
        mip_level_count: u32,
        sample_count: u32,
        dimension: TextureDimension,
        label: Option<String>,
    ) -> Self {
        let estimated_size = estimate_texture_size(
            size,
            format,
            mip_level_count,
            size.depth_or_array_layers,
        );

        Self {
            texture,
            view,
            size,
            format,
            usage,
            mip_level_count,
            sample_count,
            dimension,
            label,
            estimated_size,
        }
    }

    /// Returns a reference to the underlying wgpu texture.
    #[inline]
    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    /// Returns a reference to the default texture view.
    ///
    /// The default view covers all mip levels and array layers.
    #[inline]
    pub fn view(&self) -> &TextureView {
        &self.view
    }

    /// Returns the texture dimensions.
    #[inline]
    pub fn size(&self) -> Extent3d {
        self.size
    }

    /// Returns the texture width in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.size.width
    }

    /// Returns the texture height in pixels.
    #[inline]
    pub fn height(&self) -> u32 {
        self.size.height
    }

    /// Returns the texture depth or array layer count.
    #[inline]
    pub fn depth_or_array_layers(&self) -> u32 {
        self.size.depth_or_array_layers
    }

    /// Returns the texture format.
    #[inline]
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// Returns the texture usage flags.
    #[inline]
    pub fn usage(&self) -> TextureUsages {
        self.usage
    }

    /// Returns the number of mip levels.
    #[inline]
    pub fn mip_level_count(&self) -> u32 {
        self.mip_level_count
    }

    /// Returns the MSAA sample count.
    #[inline]
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Returns the texture dimension (1D, 2D, or 3D).
    #[inline]
    pub fn dimension(&self) -> TextureDimension {
        self.dimension
    }

    /// Returns the debug label, if any.
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Returns the estimated memory size in bytes.
    #[inline]
    pub fn estimated_size(&self) -> u64 {
        self.estimated_size
    }

    /// Creates a new texture view with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `desc` - The view descriptor specifying the view parameters
    ///
    /// # Returns
    ///
    /// A new texture view.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::resources::texture::{TrinityTextureDescriptor, create_texture};
    /// # use wgpu::{TextureViewDescriptor, TextureViewDimension, TextureAspect, Extent3d, TextureFormat};
    /// # fn example(device: &wgpu::Device) {
    /// # let texture = create_texture(device, &TrinityTextureDescriptor::default());
    /// // Create a view for a single mip level
    /// let mip_view = texture.create_view(&TextureViewDescriptor {
    ///     label: Some("mip_1_view"),
    ///     format: None, // Use texture format
    ///     dimension: None, // Use texture dimension
    ///     aspect: TextureAspect::All,
    ///     base_mip_level: 1,
    ///     mip_level_count: Some(1),
    ///     base_array_layer: 0,
    ///     array_layer_count: None,
    /// });
    /// # }
    /// ```
    pub fn create_view(&self, desc: &TextureViewDescriptor) -> TextureView {
        self.texture.create_view(desc)
    }

    /// Consumes the wrapper and returns the inner wgpu texture.
    ///
    /// The view is dropped; create a new view from the returned texture if needed.
    #[inline]
    pub fn into_inner(self) -> Texture {
        self.texture
    }
}

impl std::fmt::Debug for TrinityTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrinityTexture")
            .field("label", &self.label)
            .field("size", &self.size)
            .field("format", &self.format)
            .field("usage", &self.usage)
            .field("mip_level_count", &self.mip_level_count)
            .field("sample_count", &self.sample_count)
            .field("dimension", &self.dimension)
            .field("estimated_size", &format_size(self.estimated_size))
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Texture Creation
// ============================================================================

/// Creates a texture with validation and logging.
///
/// This function creates a wgpu texture with the specified parameters,
/// performing validation and automatic mip level calculation if requested.
///
/// # Arguments
///
/// * `device` - The wgpu device to create the texture on
/// * `desc` - The texture descriptor specifying parameters
///
/// # Returns
///
/// A [`TrinityTexture`] wrapping the created wgpu texture and its default view.
///
/// # Panics
///
/// Panics if:
/// - `size.width` or `size.height` is 0
/// - `usage` is empty
/// - `sample_count` is multisampled (>1) and `mip_level_count` > 1
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::texture::{TrinityTextureDescriptor, create_texture, texture_usages};
/// use wgpu::{TextureDimension, TextureFormat, Extent3d};
///
/// # fn example(device: &wgpu::Device) {
/// let texture = create_texture(device, &TrinityTextureDescriptor {
///     label: Some("diffuse_map"),
///     size: Extent3d { width: 1024, height: 1024, depth_or_array_layers: 1 },
///     mip_level_count: 0, // Auto mips
///     sample_count: 1,
///     dimension: TextureDimension::D2,
///     format: TextureFormat::Rgba8UnormSrgb,
///     usage: texture_usages::SAMPLED,
///     view_formats: &[],
/// });
///
/// assert_eq!(texture.width(), 1024);
/// assert_eq!(texture.mip_level_count(), 11); // Full chain
/// # }
/// ```
pub fn create_texture(device: &Device, desc: &TrinityTextureDescriptor) -> TrinityTexture {
    // Validate dimensions
    assert!(
        desc.size.width > 0 && desc.size.height > 0,
        "Texture dimensions must be greater than 0"
    );

    // Validate usage
    assert!(
        !desc.usage.is_empty(),
        "Texture usage flags must not be empty"
    );

    // Calculate mip levels if auto (0)
    let mip_level_count = if desc.mip_level_count == 0 {
        match desc.dimension {
            TextureDimension::D3 => calculate_mip_count_3d(
                desc.size.width,
                desc.size.height,
                desc.size.depth_or_array_layers,
            ),
            _ => calculate_mip_count(desc.size.width, desc.size.height),
        }
    } else {
        desc.mip_level_count
    };

    // Validate multisampling + mipmapping
    if desc.sample_count > 1 && mip_level_count > 1 {
        panic!("Multisampled textures cannot have mipmaps");
    }

    // Create the wgpu texture
    let texture = device.create_texture(&TextureDescriptor {
        label: desc.label,
        size: desc.size,
        mip_level_count,
        sample_count: desc.sample_count,
        dimension: desc.dimension,
        format: desc.format,
        usage: desc.usage,
        view_formats: desc.view_formats,
    });

    // Create default view
    let view = texture.create_view(&TextureViewDescriptor::default());

    // Estimate memory size
    let estimated_size = estimate_texture_size(
        desc.size,
        desc.format,
        mip_level_count,
        desc.size.depth_or_array_layers,
    );

    // Log allocation
    debug!(
        "Created texture '{}' {}x{}{} {:?} (~{})",
        desc.label.unwrap_or("<unnamed>"),
        desc.size.width,
        desc.size.height,
        if desc.size.depth_or_array_layers > 1 {
            format!("x{}", desc.size.depth_or_array_layers)
        } else {
            String::new()
        },
        desc.format,
        format_size(estimated_size),
    );

    TrinityTexture {
        texture,
        view,
        size: desc.size,
        format: desc.format,
        usage: desc.usage,
        mip_level_count,
        sample_count: desc.sample_count,
        dimension: desc.dimension,
        label: desc.label.map(String::from),
        estimated_size,
    }
}

/// Creates a texture with validation, returning an error on failure.
///
/// This is the fallible version of [`create_texture`].
///
/// # Arguments
///
/// * `device` - The wgpu device to create the texture on
/// * `desc` - The texture descriptor specifying parameters
///
/// # Returns
///
/// `Ok(TrinityTexture)` on success, or `Err(TextureCreationError)` on failure.
pub fn try_create_texture(
    device: &Device,
    desc: &TrinityTextureDescriptor,
) -> Result<TrinityTexture, TextureCreationError> {
    // Validate dimensions
    if desc.size.width == 0 || desc.size.height == 0 {
        return Err(TextureCreationError::ZeroDimension);
    }

    // Validate usage
    if desc.usage.is_empty() {
        return Err(TextureCreationError::EmptyUsage);
    }

    // Calculate mip levels if auto (0)
    let mip_level_count = if desc.mip_level_count == 0 {
        match desc.dimension {
            TextureDimension::D3 => calculate_mip_count_3d(
                desc.size.width,
                desc.size.height,
                desc.size.depth_or_array_layers,
            ),
            _ => calculate_mip_count(desc.size.width, desc.size.height),
        }
    } else {
        desc.mip_level_count
    };

    // Validate multisampling + mipmapping
    if desc.sample_count > 1 && mip_level_count > 1 {
        return Err(TextureCreationError::MultisampleWithMipmaps);
    }

    // Create the wgpu texture
    let texture = device.create_texture(&TextureDescriptor {
        label: desc.label,
        size: desc.size,
        mip_level_count,
        sample_count: desc.sample_count,
        dimension: desc.dimension,
        format: desc.format,
        usage: desc.usage,
        view_formats: desc.view_formats,
    });

    // Create default view
    let view = texture.create_view(&TextureViewDescriptor::default());

    // Estimate memory size
    let estimated_size = estimate_texture_size(
        desc.size,
        desc.format,
        mip_level_count,
        desc.size.depth_or_array_layers,
    );

    // Log allocation
    debug!(
        "Created texture '{}' {}x{}{} {:?} (~{})",
        desc.label.unwrap_or("<unnamed>"),
        desc.size.width,
        desc.size.height,
        if desc.size.depth_or_array_layers > 1 {
            format!("x{}", desc.size.depth_or_array_layers)
        } else {
            String::new()
        },
        desc.format,
        format_size(estimated_size),
    );

    Ok(TrinityTexture {
        texture,
        view,
        size: desc.size,
        format: desc.format,
        usage: desc.usage,
        mip_level_count,
        sample_count: desc.sample_count,
        dimension: desc.dimension,
        label: desc.label.map(String::from),
        estimated_size,
    })
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during texture creation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextureCreationError {
    /// Texture width or height was 0.
    ZeroDimension,
    /// Texture usage flags were empty.
    EmptyUsage,
    /// Multisampled texture cannot have mipmaps.
    MultisampleWithMipmaps,
    /// Invalid view format (not compatible with base format).
    InvalidViewFormat(TextureFormat),
}

impl std::fmt::Display for TextureCreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextureCreationError::ZeroDimension => {
                write!(f, "texture width and height must be greater than 0")
            }
            TextureCreationError::EmptyUsage => {
                write!(f, "texture usage flags must not be empty")
            }
            TextureCreationError::MultisampleWithMipmaps => {
                write!(f, "multisampled textures cannot have mipmaps")
            }
            TextureCreationError::InvalidViewFormat(fmt) => {
                write!(f, "view format {:?} is not compatible with base format", fmt)
            }
        }
    }
}

impl std::error::Error for TextureCreationError {}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // bytes_per_pixel tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bytes_per_pixel_8bit() {
        assert_eq!(bytes_per_pixel(TextureFormat::R8Unorm), 1);
        assert_eq!(bytes_per_pixel(TextureFormat::R8Snorm), 1);
        assert_eq!(bytes_per_pixel(TextureFormat::R8Uint), 1);
        assert_eq!(bytes_per_pixel(TextureFormat::R8Sint), 1);
    }

    #[test]
    fn test_bytes_per_pixel_16bit() {
        assert_eq!(bytes_per_pixel(TextureFormat::R16Float), 2);
        assert_eq!(bytes_per_pixel(TextureFormat::Rg8Unorm), 2);
        assert_eq!(bytes_per_pixel(TextureFormat::Depth16Unorm), 2);
    }

    #[test]
    fn test_bytes_per_pixel_32bit() {
        assert_eq!(bytes_per_pixel(TextureFormat::R32Float), 4);
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba8Unorm), 4);
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba8UnormSrgb), 4);
        assert_eq!(bytes_per_pixel(TextureFormat::Bgra8Unorm), 4);
        assert_eq!(bytes_per_pixel(TextureFormat::Depth32Float), 4);
    }

    #[test]
    fn test_bytes_per_pixel_64bit() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rg32Float), 8);
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba16Float), 8);
    }

    #[test]
    fn test_bytes_per_pixel_128bit() {
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba32Float), 16);
        assert_eq!(bytes_per_pixel(TextureFormat::Rgba32Uint), 16);
    }

    // -------------------------------------------------------------------------
    // block_info tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_block_info_uncompressed() {
        assert_eq!(block_info(TextureFormat::R8Unorm), (1, 1, 1));
        assert_eq!(block_info(TextureFormat::Rgba8Unorm), (4, 1, 1));
        assert_eq!(block_info(TextureFormat::Rgba16Float), (8, 1, 1));
        assert_eq!(block_info(TextureFormat::Rgba32Float), (16, 1, 1));
    }

    #[test]
    fn test_block_info_bc_compressed() {
        // BC1: 8 bytes per 4x4 block
        assert_eq!(block_info(TextureFormat::Bc1RgbaUnorm), (8, 4, 4));
        // BC3: 16 bytes per 4x4 block
        assert_eq!(block_info(TextureFormat::Bc3RgbaUnorm), (16, 4, 4));
        // BC7: 16 bytes per 4x4 block
        assert_eq!(block_info(TextureFormat::Bc7RgbaUnorm), (16, 4, 4));
    }

    // -------------------------------------------------------------------------
    // calculate_mip_count tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mip_count_powers_of_two() {
        assert_eq!(calculate_mip_count(1, 1), 1);
        assert_eq!(calculate_mip_count(2, 2), 2);
        assert_eq!(calculate_mip_count(4, 4), 3);
        assert_eq!(calculate_mip_count(8, 8), 4);
        assert_eq!(calculate_mip_count(16, 16), 5);
        assert_eq!(calculate_mip_count(256, 256), 9);
        assert_eq!(calculate_mip_count(1024, 1024), 11);
        assert_eq!(calculate_mip_count(2048, 2048), 12);
    }

    #[test]
    fn test_mip_count_non_square() {
        assert_eq!(calculate_mip_count(512, 256), 10); // max is 512
        assert_eq!(calculate_mip_count(1024, 512), 11); // max is 1024
        assert_eq!(calculate_mip_count(2048, 1), 12); // max is 2048
    }

    #[test]
    fn test_mip_count_non_power_of_two() {
        assert_eq!(calculate_mip_count(100, 100), 7); // 100 > 64, 100 < 128
        assert_eq!(calculate_mip_count(1920, 1080), 11); // HD resolution
    }

    #[test]
    fn test_mip_count_3d() {
        assert_eq!(calculate_mip_count_3d(64, 64, 64), 7);
        assert_eq!(calculate_mip_count_3d(256, 128, 64), 9); // max is 256
    }

    // -------------------------------------------------------------------------
    // estimate_texture_size tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_size_estimation_simple() {
        // 256x256 RGBA8, no mips
        let size = estimate_texture_size(
            Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            TextureFormat::Rgba8Unorm,
            1,
            1,
        );
        assert_eq!(size, 256 * 256 * 4);
    }

    #[test]
    fn test_size_estimation_with_mips() {
        // 256x256 RGBA8, full mip chain (9 levels)
        // Base: 256*256*4 = 262144
        // Mip1: 128*128*4 = 65536
        // Mip2: 64*64*4 = 16384
        // ... sum converges to ~1.33x base
        let size = estimate_texture_size(
            Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            TextureFormat::Rgba8Unorm,
            9,
            1,
        );
        // Should be approximately 349524 bytes (sum of mip sizes)
        assert!(size > 256 * 256 * 4);
        assert!(size < 256 * 256 * 4 * 2);
    }

    #[test]
    fn test_size_estimation_array() {
        // 256x256 RGBA8, 4 array layers, no mips
        // Array layers are specified in extent.depth_or_array_layers
        let size = estimate_texture_size(
            Extent3d { width: 256, height: 256, depth_or_array_layers: 4 },
            TextureFormat::Rgba8Unorm,
            1,
            1, // Ignored, uses extent.depth_or_array_layers
        );
        assert_eq!(size, 256 * 256 * 4 * 4);
    }

    #[test]
    fn test_size_estimation_compressed() {
        // 256x256 BC1 (8 bytes per 4x4 block)
        let size = estimate_texture_size(
            Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            TextureFormat::Bc1RgbaUnorm,
            1,
            1,
        );
        // 64x64 blocks * 8 bytes = 32768
        assert_eq!(size, 64 * 64 * 8);
    }

    // -------------------------------------------------------------------------
    // TrinityTextureDescriptor tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_default() {
        let desc = TrinityTextureDescriptor::default();
        assert_eq!(desc.size.width, 1);
        assert_eq!(desc.size.height, 1);
        assert_eq!(desc.mip_level_count, 1);
        assert_eq!(desc.sample_count, 1);
        assert_eq!(desc.format, TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn test_descriptor_with_view_formats() {
        let view_formats = &[TextureFormat::Rgba8Unorm];
        let desc = TrinityTextureDescriptor {
            label: Some("test"),
            size: Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
            format: TextureFormat::Rgba8UnormSrgb,
            view_formats,
            ..Default::default()
        };
        assert_eq!(desc.view_formats.len(), 1);
        assert_eq!(desc.view_formats[0], TextureFormat::Rgba8Unorm);
    }

    // -------------------------------------------------------------------------
    // TextureCreationError tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_display() {
        assert!(TextureCreationError::ZeroDimension.to_string().contains("0"));
        assert!(TextureCreationError::EmptyUsage.to_string().contains("usage"));
        assert!(TextureCreationError::MultisampleWithMipmaps.to_string().contains("multisample"));
    }

    // -------------------------------------------------------------------------
    // texture_usages tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_usage_presets() {
        assert!(texture_usages::SAMPLED.contains(TextureUsages::TEXTURE_BINDING));
        assert!(texture_usages::SAMPLED.contains(TextureUsages::COPY_DST));

        assert!(texture_usages::RENDER_TARGET.contains(TextureUsages::RENDER_ATTACHMENT));
        assert!(texture_usages::RENDER_TARGET.contains(TextureUsages::TEXTURE_BINDING));

        assert!(texture_usages::STORAGE.contains(TextureUsages::STORAGE_BINDING));

        assert!(texture_usages::FULL.contains(TextureUsages::TEXTURE_BINDING));
        assert!(texture_usages::FULL.contains(TextureUsages::STORAGE_BINDING));
        assert!(texture_usages::FULL.contains(TextureUsages::RENDER_ATTACHMENT));
        assert!(texture_usages::FULL.contains(TextureUsages::COPY_SRC));
        assert!(texture_usages::FULL.contains(TextureUsages::COPY_DST));
    }

    // -------------------------------------------------------------------------
    // format_size tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_size_display() {
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(1024), "1.0KB");
        assert_eq!(format_size(1536), "1.5KB");
        assert_eq!(format_size(1048576), "1.00MB");
        assert_eq!(format_size(2621440), "2.50MB");
    }
}
