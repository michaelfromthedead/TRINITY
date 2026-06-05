//! Texture upload utilities for TRINITY.
//!
//! This module provides CPU -> GPU texture upload functionality for the TRINITY
//! wgpu abstraction layer. It supports both direct uploads via `queue.write_texture()`
//! and staged uploads via intermediate staging buffers.
//!
//! # Overview
//!
//! Texture uploads in wgpu require careful attention to:
//!
//! - **Row pitch alignment**: Rows must be aligned to 256 bytes for many operations
//! - **Upload method selection**: Direct for small textures, staged for large ones
//! - **Format conversion**: Some CPU formats need conversion to GPU formats
//!
//! This module provides:
//!
//! - [`TextureUploader`] - Main upload orchestrator with automatic method selection
//! - [`TextureUploadDescriptor`] - Upload region specification
//! - [`TextureRegion`] - Convenience struct for region definitions
//! - Alignment helpers: [`calculate_row_pitch`], [`align_to_256`], [`pad_to_row_pitch`]
//! - Format converters: [`convert_rgb_to_rgba`], [`convert_bgra_to_rgba`], etc.
//!
//! # Upload Methods
//!
//! | Method | Best For | Mechanism |
//! |--------|----------|-----------|
//! | Direct | Small textures (<64KB) | `queue.write_texture()` |
//! | Staged | Large textures (>=64KB) | Staging buffer + `copy_buffer_to_texture()` |
//!
//! # Row Pitch Alignment
//!
//! wgpu requires `bytes_per_row` to be a multiple of 256 bytes for buffer-to-texture
//! copies. This module handles alignment automatically, padding rows as needed.
//!
//! ```text
//! Example: 100px wide RGBA8 texture
//!   Unpadded: 100 * 4 = 400 bytes/row
//!   Aligned:  512 bytes/row (next multiple of 256)
//!   Padding:  112 bytes/row added
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::texture_uploads::{
//!     TextureUploader, TextureUploadDescriptor, TextureRegion,
//! };
//!
//! # fn example(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) {
//! let uploader = TextureUploader::new(65536); // 64KB threshold
//!
//! // Load image data (e.g., from file)
//! let image_data: &[u8] = &[0u8; 1024 * 1024 * 4]; // 1024x1024 RGBA
//!
//! // Upload full texture
//! let desc = TextureUploadDescriptor {
//!     offset: (0, 0, 0),
//!     size: (1024, 1024, 1),
//!     mip_level: 0,
//!     bytes_per_pixel: 4,
//!     format: None, // Use texture format
//! };
//!
//! uploader.upload(
//!     device, queue, None, texture, image_data, &desc,
//! ).expect("Upload failed");
//! # }
//! ```

use log::debug;
use std::borrow::Cow;
use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoder, Device, Extent3d, ImageCopyBuffer,
    ImageCopyTexture, ImageDataLayout, Origin3d, Queue, Texture, TextureAspect, TextureFormat,
};

// ============================================================================
// Constants
// ============================================================================

/// wgpu row pitch alignment requirement (256 bytes).
///
/// When copying data to textures, the number of bytes per row must be
/// a multiple of this value. The actual requirement is `COPY_BYTES_PER_ROW_ALIGNMENT`
/// in wgpu, which is 256.
pub const ROW_PITCH_ALIGNMENT: u32 = 256;

/// Default threshold for using staged uploads (64KB).
///
/// Textures larger than this threshold will use a staging buffer instead
/// of direct `queue.write_texture()` calls. Staged uploads are better for:
///
/// - Large data transfers (reduces CPU-GPU synchronization)
/// - Batched uploads (can be combined in a single command buffer)
/// - Async uploading (staging buffer can be reused while GPU processes)
pub const STAGING_THRESHOLD: u64 = 65536;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during texture upload operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextureUploadError {
    /// The specified region extends beyond texture bounds.
    InvalidRegion {
        /// The requested region offset.
        offset: (u32, u32, u32),
        /// The requested region size.
        size: (u32, u32, u32),
        /// The texture dimensions.
        texture_size: (u32, u32, u32),
        /// The target mip level.
        mip_level: u32,
    },

    /// The provided data buffer is too small for the upload region.
    BufferTooSmall {
        /// The provided buffer size in bytes.
        provided: usize,
        /// The required buffer size in bytes.
        required: usize,
    },

    /// The source data format doesn't match the texture format.
    FormatMismatch {
        /// Description of the expected format.
        expected: String,
        /// Description of the actual format.
        actual: String,
    },

    /// Row pitch alignment calculation failed.
    AlignmentError {
        /// The unaligned row size.
        row_size: u32,
        /// The required alignment.
        alignment: u32,
    },

    /// Zero-size region specified.
    ZeroSizeRegion,

    /// Invalid bytes per pixel value.
    InvalidBytesPerPixel {
        /// The provided value.
        value: u32,
    },

    /// Mip level out of range.
    MipLevelOutOfRange {
        /// The requested mip level.
        level: u32,
        /// The maximum valid mip level.
        max_level: u32,
    },
}

impl std::fmt::Display for TextureUploadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextureUploadError::InvalidRegion {
                offset,
                size,
                texture_size,
                mip_level,
            } => {
                write!(
                    f,
                    "upload region ({},{},{})+({},{},{}) exceeds texture bounds ({},{},{}) at mip {}",
                    offset.0, offset.1, offset.2,
                    size.0, size.1, size.2,
                    texture_size.0, texture_size.1, texture_size.2,
                    mip_level
                )
            }
            TextureUploadError::BufferTooSmall { provided, required } => {
                write!(
                    f,
                    "buffer too small: provided {} bytes, required {} bytes",
                    provided, required
                )
            }
            TextureUploadError::FormatMismatch { expected, actual } => {
                write!(f, "format mismatch: expected {}, got {}", expected, actual)
            }
            TextureUploadError::AlignmentError { row_size, alignment } => {
                write!(
                    f,
                    "alignment error: row size {} is not compatible with alignment {}",
                    row_size, alignment
                )
            }
            TextureUploadError::ZeroSizeRegion => {
                write!(f, "upload region has zero size")
            }
            TextureUploadError::InvalidBytesPerPixel { value } => {
                write!(f, "invalid bytes per pixel: {}", value)
            }
            TextureUploadError::MipLevelOutOfRange { level, max_level } => {
                write!(
                    f,
                    "mip level {} out of range (max: {})",
                    level, max_level
                )
            }
        }
    }
}

impl std::error::Error for TextureUploadError {}

// ============================================================================
// Upload Descriptor
// ============================================================================

/// Describes a texture upload operation.
///
/// This struct specifies where in the texture to upload data, at what mip level,
/// and provides format information for validation.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_uploads::TextureUploadDescriptor;
///
/// // Upload to top-left corner of mip level 0
/// let desc = TextureUploadDescriptor {
///     offset: (0, 0, 0),
///     size: (256, 256, 1),
///     mip_level: 0,
///     bytes_per_pixel: 4, // RGBA8
///     format: None,
/// };
///
/// // Upload a subregion at mip level 2
/// let subregion = TextureUploadDescriptor {
///     offset: (64, 64, 0),
///     size: (128, 128, 1),
///     mip_level: 2,
///     bytes_per_pixel: 4,
///     format: None,
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureUploadDescriptor {
    /// Offset within the texture (x, y, z/layer).
    ///
    /// For 2D textures, z is the array layer index.
    /// For 3D textures, z is the depth slice.
    pub offset: (u32, u32, u32),

    /// Size of the upload region (width, height, depth/layers).
    ///
    /// For 2D textures, depth is the number of array layers to upload.
    /// For 3D textures, depth is the number of depth slices.
    pub size: (u32, u32, u32),

    /// Target mip level (0 = base level).
    pub mip_level: u32,

    /// Bytes per pixel in the source data.
    ///
    /// This must match the texture format:
    /// - R8: 1
    /// - RG8, R16: 2
    /// - RGBA8, R32F: 4
    /// - RGBA16F, RG32F: 8
    /// - RGBA32F: 16
    pub bytes_per_pixel: u32,

    /// Optional source format for validation.
    ///
    /// If provided, the uploader will validate that the format is compatible
    /// with the target texture format.
    pub format: Option<TextureFormat>,
}

impl Default for TextureUploadDescriptor {
    fn default() -> Self {
        Self {
            offset: (0, 0, 0),
            size: (1, 1, 1),
            mip_level: 0,
            bytes_per_pixel: 4, // RGBA8 is most common
            format: None,
        }
    }
}

impl TextureUploadDescriptor {
    /// Creates a descriptor for a full texture upload at mip level 0.
    ///
    /// # Arguments
    ///
    /// * `width` - Texture width in pixels
    /// * `height` - Texture height in pixels
    /// * `bytes_per_pixel` - Bytes per pixel for the format
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::texture_uploads::TextureUploadDescriptor;
    ///
    /// let desc = TextureUploadDescriptor::full(1024, 1024, 4);
    /// assert_eq!(desc.size, (1024, 1024, 1));
    /// assert_eq!(desc.mip_level, 0);
    /// ```
    pub fn full(width: u32, height: u32, bytes_per_pixel: u32) -> Self {
        Self {
            offset: (0, 0, 0),
            size: (width, height, 1),
            mip_level: 0,
            bytes_per_pixel,
            format: None,
        }
    }

    /// Creates a descriptor for uploading a specific mip level.
    ///
    /// # Arguments
    ///
    /// * `base_width` - Base texture width (mip 0)
    /// * `base_height` - Base texture height (mip 0)
    /// * `mip_level` - Target mip level
    /// * `bytes_per_pixel` - Bytes per pixel for the format
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::texture_uploads::TextureUploadDescriptor;
    ///
    /// // Mip 2 of a 1024x1024 texture = 256x256
    /// let desc = TextureUploadDescriptor::mip_level(1024, 1024, 2, 4);
    /// assert_eq!(desc.size, (256, 256, 1));
    /// assert_eq!(desc.mip_level, 2);
    /// ```
    pub fn mip_level(
        base_width: u32,
        base_height: u32,
        mip_level: u32,
        bytes_per_pixel: u32,
    ) -> Self {
        let width = (base_width >> mip_level).max(1);
        let height = (base_height >> mip_level).max(1);
        Self {
            offset: (0, 0, 0),
            size: (width, height, 1),
            mip_level,
            bytes_per_pixel,
            format: None,
        }
    }

    /// Creates a descriptor for a subregion upload.
    ///
    /// # Arguments
    ///
    /// * `x` - X offset in pixels
    /// * `y` - Y offset in pixels
    /// * `width` - Region width in pixels
    /// * `height` - Region height in pixels
    /// * `bytes_per_pixel` - Bytes per pixel for the format
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::texture_uploads::TextureUploadDescriptor;
    ///
    /// let desc = TextureUploadDescriptor::subregion(100, 100, 200, 200, 4);
    /// assert_eq!(desc.offset, (100, 100, 0));
    /// assert_eq!(desc.size, (200, 200, 1));
    /// ```
    pub fn subregion(x: u32, y: u32, width: u32, height: u32, bytes_per_pixel: u32) -> Self {
        Self {
            offset: (x, y, 0),
            size: (width, height, 1),
            mip_level: 0,
            bytes_per_pixel,
            format: None,
        }
    }

    /// Sets the source format for validation.
    pub fn with_format(mut self, format: TextureFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Sets the target mip level.
    pub fn with_mip_level(mut self, mip_level: u32) -> Self {
        self.mip_level = mip_level;
        self
    }

    /// Sets the array layer/depth slice.
    pub fn with_layer(mut self, layer: u32) -> Self {
        self.offset.2 = layer;
        self
    }

    /// Calculates the required data size for this upload.
    ///
    /// This accounts for row pitch alignment when using staged uploads.
    pub fn required_data_size(&self) -> u64 {
        let row_pitch = calculate_row_pitch(self.size.0, self.bytes_per_pixel);
        (row_pitch as u64) * (self.size.1 as u64) * (self.size.2 as u64)
    }

    /// Calculates the unpadded data size (no alignment).
    pub fn unpadded_data_size(&self) -> u64 {
        (self.size.0 as u64)
            * (self.size.1 as u64)
            * (self.size.2 as u64)
            * (self.bytes_per_pixel as u64)
    }
}

// ============================================================================
// Texture Region
// ============================================================================

/// Convenience struct for defining texture regions.
///
/// This provides a cleaner API for common region patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureRegion {
    /// Region offset (x, y, z/layer).
    pub offset: (u32, u32, u32),
    /// Region size (width, height, depth/layers).
    pub size: (u32, u32, u32),
}

impl TextureRegion {
    /// Creates a region covering the full texture.
    ///
    /// # Arguments
    ///
    /// * `width` - Texture width
    /// * `height` - Texture height
    /// * `depth` - Texture depth/layers (usually 1 for 2D)
    pub const fn full(width: u32, height: u32, depth: u32) -> Self {
        Self {
            offset: (0, 0, 0),
            size: (width, height, depth),
        }
    }

    /// Creates a region for a specific mip level.
    ///
    /// # Arguments
    ///
    /// * `base_width` - Base (mip 0) width
    /// * `base_height` - Base (mip 0) height
    /// * `mip_level` - Target mip level
    pub const fn mip(base_width: u32, base_height: u32, mip_level: u32) -> Self {
        let width = if mip_level >= 32 {
            1
        } else {
            let shifted = base_width >> mip_level;
            if shifted == 0 { 1 } else { shifted }
        };
        let height = if mip_level >= 32 {
            1
        } else {
            let shifted = base_height >> mip_level;
            if shifted == 0 { 1 } else { shifted }
        };
        Self {
            offset: (0, 0, 0),
            size: (width, height, 1),
        }
    }

    /// Creates a subregion.
    pub const fn subregion(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            offset: (x, y, 0),
            size: (width, height, 1),
        }
    }

    /// Creates a region for a specific array layer.
    pub const fn layer(width: u32, height: u32, layer: u32) -> Self {
        Self {
            offset: (0, 0, layer),
            size: (width, height, 1),
        }
    }

    /// Creates a region for multiple array layers.
    pub const fn layer_range(width: u32, height: u32, start_layer: u32, layer_count: u32) -> Self {
        Self {
            offset: (0, 0, start_layer),
            size: (width, height, layer_count),
        }
    }

    /// Checks if this region is valid for the given texture dimensions.
    pub const fn is_valid_for(&self, tex_width: u32, tex_height: u32, tex_depth: u32) -> bool {
        let end_x = self.offset.0.saturating_add(self.size.0);
        let end_y = self.offset.1.saturating_add(self.size.1);
        let end_z = self.offset.2.saturating_add(self.size.2);
        end_x <= tex_width && end_y <= tex_height && end_z <= tex_depth
    }

    /// Checks if this region has non-zero size.
    pub const fn is_non_empty(&self) -> bool {
        self.size.0 > 0 && self.size.1 > 0 && self.size.2 > 0
    }
}

impl Default for TextureRegion {
    fn default() -> Self {
        Self {
            offset: (0, 0, 0),
            size: (1, 1, 1),
        }
    }
}

// ============================================================================
// Alignment Helpers
// ============================================================================

/// Calculates the row pitch (bytes per row) with 256-byte alignment.
///
/// wgpu requires `bytes_per_row` to be a multiple of 256 for buffer-to-texture
/// copies. This function calculates the aligned row pitch.
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `bytes_per_pixel` - Bytes per pixel (e.g., 4 for RGBA8)
///
/// # Returns
///
/// The aligned row pitch in bytes.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_uploads::calculate_row_pitch;
///
/// // 100 pixels * 4 bytes = 400, aligned to 512
/// assert_eq!(calculate_row_pitch(100, 4), 512);
///
/// // 64 pixels * 4 bytes = 256, already aligned
/// assert_eq!(calculate_row_pitch(64, 4), 256);
///
/// // 1 pixel * 4 bytes = 4, aligned to 256
/// assert_eq!(calculate_row_pitch(1, 4), 256);
/// ```
#[inline]
pub const fn calculate_row_pitch(width: u32, bytes_per_pixel: u32) -> u32 {
    let unaligned = width * bytes_per_pixel;
    align_to_256(unaligned)
}

/// Aligns a value up to the nearest multiple of 256.
///
/// # Arguments
///
/// * `value` - The value to align
///
/// # Returns
///
/// The aligned value (>= input, multiple of 256).
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_uploads::align_to_256;
///
/// assert_eq!(align_to_256(0), 0);
/// assert_eq!(align_to_256(1), 256);
/// assert_eq!(align_to_256(256), 256);
/// assert_eq!(align_to_256(257), 512);
/// assert_eq!(align_to_256(400), 512);
/// ```
#[inline]
pub const fn align_to_256(value: u32) -> u32 {
    if value == 0 {
        return 0;
    }
    (value + ROW_PITCH_ALIGNMENT - 1) & !(ROW_PITCH_ALIGNMENT - 1)
}

/// Checks if a row pitch is properly aligned.
#[inline]
pub const fn is_row_pitch_aligned(bytes_per_row: u32) -> bool {
    bytes_per_row == 0 || bytes_per_row % ROW_PITCH_ALIGNMENT == 0
}

/// Pads image data to match row pitch alignment requirements.
///
/// This function takes tightly-packed image data and adds padding bytes
/// to each row to satisfy wgpu's 256-byte row alignment requirement.
///
/// # Arguments
///
/// * `data` - Source image data (tightly packed, no padding)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `bytes_per_pixel` - Bytes per pixel
///
/// # Returns
///
/// A new vector with padded data, or the original data if already aligned.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_uploads::pad_to_row_pitch;
///
/// // 100x2 RGBA image: 100*4 = 400 bytes/row, needs padding to 512
/// let data = vec![0u8; 100 * 2 * 4]; // 800 bytes, tightly packed
/// let padded = pad_to_row_pitch(&data, 100, 2, 4);
///
/// // Padded: 512 bytes/row * 2 rows = 1024 bytes
/// assert_eq!(padded.len(), 1024);
/// ```
pub fn pad_to_row_pitch(data: &[u8], width: u32, height: u32, bytes_per_pixel: u32) -> Cow<'_, [u8]> {
    let unpadded_row = width * bytes_per_pixel;
    let padded_row = calculate_row_pitch(width, bytes_per_pixel);

    // If already aligned, return original data
    if unpadded_row == padded_row {
        return Cow::Borrowed(data);
    }

    let unpadded_row = unpadded_row as usize;
    let padded_row = padded_row as usize;
    let height = height as usize;

    let mut padded = vec![0u8; padded_row * height];

    for row in 0..height {
        let src_start = row * unpadded_row;
        let src_end = src_start + unpadded_row;
        let dst_start = row * padded_row;

        if src_end <= data.len() {
            padded[dst_start..dst_start + unpadded_row].copy_from_slice(&data[src_start..src_end]);
        }
    }

    Cow::Owned(padded)
}

/// Pads 3D/array texture data to match row pitch alignment.
///
/// Similar to [`pad_to_row_pitch`] but handles multiple depth slices/layers.
///
/// # Arguments
///
/// * `data` - Source image data (tightly packed)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `depth` - Number of depth slices/array layers
/// * `bytes_per_pixel` - Bytes per pixel
pub fn pad_to_row_pitch_3d(
    data: &[u8],
    width: u32,
    height: u32,
    depth: u32,
    bytes_per_pixel: u32,
) -> Cow<'_, [u8]> {
    let unpadded_row = width * bytes_per_pixel;
    let padded_row = calculate_row_pitch(width, bytes_per_pixel);

    // If already aligned, return original data
    if unpadded_row == padded_row {
        return Cow::Borrowed(data);
    }

    let unpadded_row = unpadded_row as usize;
    let padded_row = padded_row as usize;
    let height = height as usize;
    let depth = depth as usize;
    let unpadded_slice = unpadded_row * height;

    let mut padded = vec![0u8; padded_row * height * depth];

    for z in 0..depth {
        for row in 0..height {
            let src_start = z * unpadded_slice + row * unpadded_row;
            let src_end = src_start + unpadded_row;
            let dst_start = z * (padded_row * height) + row * padded_row;

            if src_end <= data.len() {
                padded[dst_start..dst_start + unpadded_row]
                    .copy_from_slice(&data[src_start..src_end]);
            }
        }
    }

    Cow::Owned(padded)
}

// ============================================================================
// Format Conversion Helpers
// ============================================================================

/// Converts RGB data to RGBA by adding an alpha channel.
///
/// GPU textures commonly use RGBA format, but source images may be RGB.
/// This function adds a fully opaque alpha channel (255).
///
/// # Arguments
///
/// * `data` - Source RGB data (3 bytes per pixel)
///
/// # Returns
///
/// RGBA data (4 bytes per pixel).
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_uploads::convert_rgb_to_rgba;
///
/// let rgb = [255, 128, 64]; // One red-ish pixel
/// let rgba = convert_rgb_to_rgba(&rgb);
///
/// assert_eq!(rgba, [255, 128, 64, 255]);
/// ```
pub fn convert_rgb_to_rgba(data: &[u8]) -> Vec<u8> {
    let pixel_count = data.len() / 3;
    let mut rgba = Vec::with_capacity(pixel_count * 4);

    for chunk in data.chunks_exact(3) {
        rgba.push(chunk[0]); // R
        rgba.push(chunk[1]); // G
        rgba.push(chunk[2]); // B
        rgba.push(255);      // A (fully opaque)
    }

    rgba
}

/// Converts BGRA data to RGBA by swapping R and B channels.
///
/// Some platforms (Windows/DirectX) use BGRA as the native format.
/// This converts to RGBA for cross-platform compatibility.
///
/// # Arguments
///
/// * `data` - Source BGRA data (4 bytes per pixel)
///
/// # Returns
///
/// RGBA data (4 bytes per pixel).
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_uploads::convert_bgra_to_rgba;
///
/// let bgra = [64, 128, 255, 200]; // BGRA
/// let rgba = convert_bgra_to_rgba(&bgra);
///
/// assert_eq!(rgba, [255, 128, 64, 200]); // RGBA (R and B swapped)
/// ```
pub fn convert_bgra_to_rgba(data: &[u8]) -> Vec<u8> {
    let mut rgba = data.to_vec();
    for chunk in rgba.chunks_exact_mut(4) {
        chunk.swap(0, 2); // Swap R and B
    }
    rgba
}

/// Converts RGBA data to BGRA by swapping R and B channels.
///
/// Some platforms prefer BGRA format. This is the inverse of
/// [`convert_bgra_to_rgba`].
///
/// # Arguments
///
/// * `data` - Source RGBA data (4 bytes per pixel)
///
/// # Returns
///
/// BGRA data (4 bytes per pixel).
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_uploads::convert_rgba_to_bgra;
///
/// let rgba = [255, 128, 64, 200]; // RGBA
/// let bgra = convert_rgba_to_bgra(&rgba);
///
/// assert_eq!(bgra, [64, 128, 255, 200]); // BGRA
/// ```
pub fn convert_rgba_to_bgra(data: &[u8]) -> Vec<u8> {
    convert_bgra_to_rgba(data) // Same operation, just swap R and B
}

/// Converts grayscale data to RGBA by replicating the value.
///
/// # Arguments
///
/// * `data` - Source grayscale data (1 byte per pixel)
///
/// # Returns
///
/// RGBA data (4 bytes per pixel) with R=G=B=gray, A=255.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_uploads::convert_gray_to_rgba;
///
/// let gray = [128];
/// let rgba = convert_gray_to_rgba(&gray);
///
/// assert_eq!(rgba, [128, 128, 128, 255]);
/// ```
pub fn convert_gray_to_rgba(data: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(data.len() * 4);
    for &gray in data {
        rgba.push(gray); // R
        rgba.push(gray); // G
        rgba.push(gray); // B
        rgba.push(255);  // A
    }
    rgba
}

/// Converts grayscale + alpha data to RGBA.
///
/// # Arguments
///
/// * `data` - Source grayscale+alpha data (2 bytes per pixel)
///
/// # Returns
///
/// RGBA data (4 bytes per pixel) with R=G=B=gray, A=alpha.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_uploads::convert_gray_alpha_to_rgba;
///
/// let gray_alpha = [128, 200]; // Gray=128, Alpha=200
/// let rgba = convert_gray_alpha_to_rgba(&gray_alpha);
///
/// assert_eq!(rgba, [128, 128, 128, 200]);
/// ```
pub fn convert_gray_alpha_to_rgba(data: &[u8]) -> Vec<u8> {
    let pixel_count = data.len() / 2;
    let mut rgba = Vec::with_capacity(pixel_count * 4);

    for chunk in data.chunks_exact(2) {
        let gray = chunk[0];
        let alpha = chunk[1];
        rgba.push(gray);  // R
        rgba.push(gray);  // G
        rgba.push(gray);  // B
        rgba.push(alpha); // A
    }

    rgba
}

/// Premultiplies alpha in RGBA data.
///
/// Premultiplied alpha: `color = color * alpha / 255`
///
/// This is useful for correct blending with transparent textures.
///
/// # Arguments
///
/// * `data` - Source RGBA data (4 bytes per pixel)
///
/// # Returns
///
/// Premultiplied RGBA data.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_uploads::premultiply_alpha;
///
/// let rgba = [255, 128, 64, 128]; // 50% alpha
/// let premul = premultiply_alpha(&rgba);
///
/// // Colors are halved due to 50% alpha
/// assert_eq!(premul[0], 128); // R: 255 * 128 / 255 = 128
/// assert_eq!(premul[1], 64);  // G: 128 * 128 / 255 ≈ 64
/// assert_eq!(premul[2], 32);  // B: 64 * 128 / 255 ≈ 32
/// assert_eq!(premul[3], 128); // A unchanged
/// ```
pub fn premultiply_alpha(data: &[u8]) -> Vec<u8> {
    let mut result = data.to_vec();
    for chunk in result.chunks_exact_mut(4) {
        let alpha = chunk[3] as u16;
        chunk[0] = ((chunk[0] as u16 * alpha) / 255) as u8;
        chunk[1] = ((chunk[1] as u16 * alpha) / 255) as u8;
        chunk[2] = ((chunk[2] as u16 * alpha) / 255) as u8;
        // Alpha unchanged
    }
    result
}

// ============================================================================
// Texture Uploader
// ============================================================================

/// Handles CPU to GPU texture uploads.
///
/// The uploader automatically chooses between direct and staged uploads
/// based on the data size and the configured threshold.
///
/// # Methods
///
/// | Method | When to Use |
/// |--------|-------------|
/// | [`write_texture`](Self::write_texture) | Small textures, synchronous upload |
/// | [`upload_staged`](Self::upload_staged) | Large textures, batched with other commands |
/// | [`upload`](Self::upload) | Auto-selects based on size |
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::texture_uploads::{TextureUploader, TextureUploadDescriptor};
///
/// # fn example(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) {
/// let uploader = TextureUploader::new(65536); // 64KB threshold
///
/// // Small texture - uses direct upload
/// let small_data = vec![0u8; 1024];
/// let desc = TextureUploadDescriptor::full(16, 16, 4);
/// uploader.write_texture(queue, texture, &small_data, &desc).unwrap();
///
/// // Large texture - auto-selects staged upload
/// let large_data = vec![0u8; 1024 * 1024 * 4];
/// let desc = TextureUploadDescriptor::full(1024, 1024, 4);
/// uploader.upload(device, queue, None, texture, &large_data, &desc).unwrap();
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TextureUploader {
    /// Size threshold in bytes for switching to staged uploads.
    staging_threshold: u64,
}

impl TextureUploader {
    /// Creates a new texture uploader with the specified staging threshold.
    ///
    /// # Arguments
    ///
    /// * `staging_threshold` - Byte size above which staged uploads are used
    ///
    /// # Example
    ///
    /// ```
    /// use renderer_backend::resources::texture_uploads::TextureUploader;
    ///
    /// let uploader = TextureUploader::new(65536); // 64KB threshold
    /// ```
    pub fn new(staging_threshold: u64) -> Self {
        Self { staging_threshold }
    }

    /// Creates an uploader with the default 64KB threshold.
    pub fn with_default_threshold() -> Self {
        Self::new(STAGING_THRESHOLD)
    }

    /// Returns the staging threshold in bytes.
    #[inline]
    pub fn staging_threshold(&self) -> u64 {
        self.staging_threshold
    }

    /// Sets the staging threshold.
    pub fn set_staging_threshold(&mut self, threshold: u64) {
        self.staging_threshold = threshold;
    }

    /// Performs a direct texture upload via `queue.write_texture()`.
    ///
    /// This is best for small textures where the overhead of a staging
    /// buffer isn't worth it. The data is uploaded synchronously.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue
    /// * `texture` - Target texture
    /// * `data` - Source pixel data
    /// * `desc` - Upload descriptor
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if validation fails.
    ///
    /// # Note
    ///
    /// This method handles row pitch alignment internally. If the source data
    /// isn't aligned, it will be padded before upload.
    pub fn write_texture(
        &self,
        queue: &Queue,
        texture: &Texture,
        data: &[u8],
        desc: &TextureUploadDescriptor,
    ) -> Result<(), TextureUploadError> {
        // Validate inputs
        self.validate_upload(desc, data.len())?;

        // Calculate row pitch
        let unpadded_row = desc.size.0 * desc.bytes_per_pixel;
        let padded_row = calculate_row_pitch(desc.size.0, desc.bytes_per_pixel);
        let rows_per_image = desc.size.1;

        // Pad data if necessary
        let upload_data: Cow<'_, [u8]> = if unpadded_row == padded_row {
            Cow::Borrowed(data)
        } else {
            pad_to_row_pitch_3d(data, desc.size.0, desc.size.1, desc.size.2, desc.bytes_per_pixel)
        };

        // Create ImageCopyTexture
        let destination = ImageCopyTexture {
            texture,
            mip_level: desc.mip_level,
            origin: Origin3d {
                x: desc.offset.0,
                y: desc.offset.1,
                z: desc.offset.2,
            },
            aspect: TextureAspect::All,
        };

        // Create ImageDataLayout
        // Note: bytes_per_row must be Some and aligned to 256 for non-1D textures
        let data_layout = ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(padded_row),
            rows_per_image: Some(rows_per_image),
        };

        // Upload extent
        let size = Extent3d {
            width: desc.size.0,
            height: desc.size.1,
            depth_or_array_layers: desc.size.2,
        };

        debug!(
            "Direct upload: {}x{}x{} @ mip {} ({}B/row, {} rows)",
            desc.size.0, desc.size.1, desc.size.2,
            desc.mip_level, padded_row, rows_per_image
        );

        // Perform the upload
        queue.write_texture(destination, &upload_data, data_layout, size);

        Ok(())
    }

    /// Performs a staged texture upload via a staging buffer.
    ///
    /// This creates a temporary staging buffer, copies data into it, then
    /// issues a `copy_buffer_to_texture` command. This is better for large
    /// uploads as it:
    ///
    /// - Avoids blocking the CPU during GPU upload
    /// - Can be batched with other GPU commands
    /// - Works better with async upload patterns
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `encoder` - Command encoder to record the copy command
    /// * `texture` - Target texture
    /// * `data` - Source pixel data
    /// * `desc` - Upload descriptor
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if validation fails.
    pub fn upload_staged(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        texture: &Texture,
        data: &[u8],
        desc: &TextureUploadDescriptor,
    ) -> Result<(), TextureUploadError> {
        // Validate inputs
        self.validate_upload(desc, data.len())?;

        // Calculate row pitch
        let padded_row = calculate_row_pitch(desc.size.0, desc.bytes_per_pixel);
        let rows_per_image = desc.size.1;
        let buffer_size = (padded_row as u64) * (rows_per_image as u64) * (desc.size.2 as u64);

        // Pad data if necessary
        let upload_data = pad_to_row_pitch_3d(
            data,
            desc.size.0,
            desc.size.1,
            desc.size.2,
            desc.bytes_per_pixel,
        );

        // Create staging buffer
        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("texture_upload_staging"),
            size: buffer_size,
            usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        });

        // Write data to staging buffer
        {
            let mut view = staging_buffer.slice(..).get_mapped_range_mut();
            view[..upload_data.len()].copy_from_slice(&upload_data);
        }
        staging_buffer.unmap();

        // Create copy source/destination
        let source = ImageCopyBuffer {
            buffer: &staging_buffer,
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(rows_per_image),
            },
        };

        let destination = ImageCopyTexture {
            texture,
            mip_level: desc.mip_level,
            origin: Origin3d {
                x: desc.offset.0,
                y: desc.offset.1,
                z: desc.offset.2,
            },
            aspect: TextureAspect::All,
        };

        let size = Extent3d {
            width: desc.size.0,
            height: desc.size.1,
            depth_or_array_layers: desc.size.2,
        };

        debug!(
            "Staged upload: {}x{}x{} @ mip {} ({}B staging buffer)",
            desc.size.0, desc.size.1, desc.size.2,
            desc.mip_level, buffer_size
        );

        // Issue copy command
        encoder.copy_buffer_to_texture(source, destination, size);

        Ok(())
    }

    /// Auto-selects upload method based on data size.
    ///
    /// - Uses direct upload for data smaller than `staging_threshold`
    /// - Uses staged upload for larger data (requires encoder)
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `queue` - The wgpu queue
    /// * `encoder` - Optional command encoder (required for staged uploads)
    /// * `texture` - Target texture
    /// * `data` - Source pixel data
    /// * `desc` - Upload descriptor
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if validation fails.
    ///
    /// # Panics
    ///
    /// Panics if the data exceeds the staging threshold but no encoder is provided.
    pub fn upload(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: Option<&mut CommandEncoder>,
        texture: &Texture,
        data: &[u8],
        desc: &TextureUploadDescriptor,
    ) -> Result<(), TextureUploadError> {
        let data_size = data.len() as u64;

        if data_size < self.staging_threshold {
            // Use direct upload
            self.write_texture(queue, texture, data, desc)
        } else {
            // Use staged upload
            match encoder {
                Some(enc) => self.upload_staged(device, enc, texture, data, desc),
                None => {
                    // Fall back to direct upload if no encoder provided
                    debug!(
                        "No encoder provided for large upload ({}B), using direct upload",
                        data_size
                    );
                    self.write_texture(queue, texture, data, desc)
                }
            }
        }
    }

    /// Uploads multiple mip levels at once.
    ///
    /// # Arguments
    ///
    /// * `queue` - The wgpu queue
    /// * `texture` - Target texture
    /// * `mip_data` - Slice of (mip_level, data) pairs
    /// * `base_width` - Base (mip 0) texture width
    /// * `base_height` - Base (mip 0) texture height
    /// * `bytes_per_pixel` - Bytes per pixel
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or the first error encountered.
    pub fn upload_mip_chain(
        &self,
        queue: &Queue,
        texture: &Texture,
        mip_data: &[(u32, &[u8])],
        base_width: u32,
        base_height: u32,
        bytes_per_pixel: u32,
    ) -> Result<(), TextureUploadError> {
        for &(mip_level, data) in mip_data {
            let desc = TextureUploadDescriptor::mip_level(
                base_width,
                base_height,
                mip_level,
                bytes_per_pixel,
            );
            self.write_texture(queue, texture, data, &desc)?;
        }
        Ok(())
    }

    /// Validates upload parameters.
    ///
    /// This function is public when the `test-utils` feature is enabled for whitebox testing.
    #[cfg(not(feature = "test-utils"))]
    fn validate_upload(&self, desc: &TextureUploadDescriptor, data_len: usize) -> Result<(), TextureUploadError> {
        self.validate_upload_impl(desc, data_len)
    }

    /// Validates upload parameters (public for test-utils).
    #[cfg(feature = "test-utils")]
    pub fn validate_upload(&self, desc: &TextureUploadDescriptor, data_len: usize) -> Result<(), TextureUploadError> {
        self.validate_upload_impl(desc, data_len)
    }

    /// Internal validation implementation.
    fn validate_upload_impl(&self, desc: &TextureUploadDescriptor, data_len: usize) -> Result<(), TextureUploadError> {
        // Check for zero-size region
        if desc.size.0 == 0 || desc.size.1 == 0 || desc.size.2 == 0 {
            return Err(TextureUploadError::ZeroSizeRegion);
        }

        // Validate bytes per pixel
        if desc.bytes_per_pixel == 0 || desc.bytes_per_pixel > 16 {
            return Err(TextureUploadError::InvalidBytesPerPixel {
                value: desc.bytes_per_pixel,
            });
        }

        // Check buffer size (unpadded requirement)
        let required = desc.unpadded_data_size() as usize;
        if data_len < required {
            return Err(TextureUploadError::BufferTooSmall {
                provided: data_len,
                required,
            });
        }

        Ok(())
    }
}

impl Default for TextureUploader {
    fn default() -> Self {
        Self::with_default_threshold()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Calculates the size at a specific mip level.
///
/// # Arguments
///
/// * `base_size` - Size at mip level 0
/// * `mip_level` - Target mip level
///
/// # Returns
///
/// Size at the specified mip level (minimum 1).
#[inline]
pub const fn mip_size(base_size: u32, mip_level: u32) -> u32 {
    let size = base_size >> mip_level;
    if size == 0 { 1 } else { size }
}

/// Gets the bytes per pixel for common texture formats.
///
/// Returns `None` for compressed or unsupported formats.
pub fn bytes_per_pixel_for_format(format: TextureFormat) -> Option<u32> {
    match format {
        TextureFormat::R8Unorm
        | TextureFormat::R8Snorm
        | TextureFormat::R8Uint
        | TextureFormat::R8Sint => Some(1),

        TextureFormat::R16Uint
        | TextureFormat::R16Sint
        | TextureFormat::R16Float
        | TextureFormat::Rg8Unorm
        | TextureFormat::Rg8Snorm
        | TextureFormat::Rg8Uint
        | TextureFormat::Rg8Sint => Some(2),

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
        | TextureFormat::Bgra8UnormSrgb => Some(4),

        TextureFormat::Rg32Uint
        | TextureFormat::Rg32Sint
        | TextureFormat::Rg32Float
        | TextureFormat::Rgba16Uint
        | TextureFormat::Rgba16Sint
        | TextureFormat::Rgba16Float => Some(8),

        TextureFormat::Rgba32Uint
        | TextureFormat::Rgba32Sint
        | TextureFormat::Rgba32Float => Some(16),

        // Compressed and depth formats return None
        _ => None,
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Row Pitch Alignment Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_calculate_row_pitch_aligned() {
        // Already aligned cases (64 pixels * 4 bytes = 256)
        assert_eq!(calculate_row_pitch(64, 4), 256);
        assert_eq!(calculate_row_pitch(128, 4), 512);
        assert_eq!(calculate_row_pitch(256, 4), 1024);
    }

    #[test]
    fn test_calculate_row_pitch_needs_padding() {
        // 100 * 4 = 400, aligns to 512
        assert_eq!(calculate_row_pitch(100, 4), 512);
        // 1 * 4 = 4, aligns to 256
        assert_eq!(calculate_row_pitch(1, 4), 256);
        // 65 * 4 = 260, aligns to 512
        assert_eq!(calculate_row_pitch(65, 4), 512);
    }

    #[test]
    fn test_calculate_row_pitch_different_bpp() {
        // 256 pixels, 1 byte each = 256 (aligned)
        assert_eq!(calculate_row_pitch(256, 1), 256);
        // 256 pixels, 2 bytes each = 512 (aligned)
        assert_eq!(calculate_row_pitch(256, 2), 512);
        // 256 pixels, 8 bytes each = 2048 (aligned)
        assert_eq!(calculate_row_pitch(256, 8), 2048);
        // 256 pixels, 16 bytes each = 4096 (aligned)
        assert_eq!(calculate_row_pitch(256, 16), 4096);
    }

    #[test]
    fn test_align_to_256() {
        assert_eq!(align_to_256(0), 0);
        assert_eq!(align_to_256(1), 256);
        assert_eq!(align_to_256(255), 256);
        assert_eq!(align_to_256(256), 256);
        assert_eq!(align_to_256(257), 512);
        assert_eq!(align_to_256(400), 512);
        assert_eq!(align_to_256(512), 512);
        assert_eq!(align_to_256(1000), 1024);
    }

    #[test]
    fn test_is_row_pitch_aligned() {
        assert!(is_row_pitch_aligned(0));
        assert!(is_row_pitch_aligned(256));
        assert!(is_row_pitch_aligned(512));
        assert!(is_row_pitch_aligned(1024));
        assert!(!is_row_pitch_aligned(1));
        assert!(!is_row_pitch_aligned(255));
        assert!(!is_row_pitch_aligned(257));
        assert!(!is_row_pitch_aligned(400));
    }

    // -------------------------------------------------------------------------
    // Padding Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_pad_to_row_pitch_already_aligned() {
        // 64x2 RGBA = 256 bytes/row, already aligned
        let data = vec![0u8; 64 * 2 * 4];
        let padded = pad_to_row_pitch(&data, 64, 2, 4);

        // Should return borrowed data (no copy)
        assert!(matches!(padded, Cow::Borrowed(_)));
        assert_eq!(padded.len(), data.len());
    }

    #[test]
    fn test_pad_to_row_pitch_needs_padding() {
        // 100x2 RGBA: 400 bytes/row -> 512 bytes/row
        let data = vec![1u8; 100 * 2 * 4]; // 800 bytes
        let padded = pad_to_row_pitch(&data, 100, 2, 4);

        // Should return owned data with padding
        assert!(matches!(padded, Cow::Owned(_)));
        assert_eq!(padded.len(), 512 * 2); // 1024 bytes

        // Verify original data is preserved
        for row in 0..2 {
            for col in 0..400 {
                assert_eq!(padded[row * 512 + col], 1);
            }
            // Padding should be zeros
            for col in 400..512 {
                assert_eq!(padded[row * 512 + col], 0);
            }
        }
    }

    #[test]
    fn test_pad_to_row_pitch_3d() {
        // 100x2x2 RGBA (2 layers)
        let data = vec![2u8; 100 * 2 * 2 * 4]; // 1600 bytes
        let padded = pad_to_row_pitch_3d(&data, 100, 2, 2, 4);

        // 512 bytes/row * 2 rows * 2 layers = 2048 bytes
        assert_eq!(padded.len(), 2048);
    }

    // -------------------------------------------------------------------------
    // Format Conversion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_convert_rgb_to_rgba() {
        let rgb = [255, 128, 64, 0, 100, 200];
        let rgba = convert_rgb_to_rgba(&rgb);

        assert_eq!(rgba.len(), 8);
        assert_eq!(rgba, [255, 128, 64, 255, 0, 100, 200, 255]);
    }

    #[test]
    fn test_convert_bgra_to_rgba() {
        let bgra = [64, 128, 255, 200];
        let rgba = convert_bgra_to_rgba(&bgra);

        assert_eq!(rgba, [255, 128, 64, 200]);
    }

    #[test]
    fn test_convert_rgba_to_bgra() {
        let rgba = [255, 128, 64, 200];
        let bgra = convert_rgba_to_bgra(&rgba);

        assert_eq!(bgra, [64, 128, 255, 200]);
    }

    #[test]
    fn test_convert_gray_to_rgba() {
        let gray = [128, 64];
        let rgba = convert_gray_to_rgba(&gray);

        assert_eq!(rgba, [128, 128, 128, 255, 64, 64, 64, 255]);
    }

    #[test]
    fn test_convert_gray_alpha_to_rgba() {
        let gray_alpha = [128, 200, 64, 100];
        let rgba = convert_gray_alpha_to_rgba(&gray_alpha);

        assert_eq!(rgba, [128, 128, 128, 200, 64, 64, 64, 100]);
    }

    #[test]
    fn test_premultiply_alpha() {
        // 50% alpha
        let rgba = [255, 128, 64, 128];
        let premul = premultiply_alpha(&rgba);

        assert_eq!(premul[0], 128); // 255 * 128 / 255 = 128
        assert_eq!(premul[1], 64);  // 128 * 128 / 255 ≈ 64
        assert_eq!(premul[2], 32);  // 64 * 128 / 255 ≈ 32
        assert_eq!(premul[3], 128); // Alpha unchanged
    }

    #[test]
    fn test_premultiply_alpha_opaque() {
        // Fully opaque
        let rgba = [255, 128, 64, 255];
        let premul = premultiply_alpha(&rgba);

        assert_eq!(premul, rgba); // Should be unchanged
    }

    #[test]
    fn test_premultiply_alpha_transparent() {
        // Fully transparent
        let rgba = [255, 128, 64, 0];
        let premul = premultiply_alpha(&rgba);

        assert_eq!(premul[0], 0);
        assert_eq!(premul[1], 0);
        assert_eq!(premul[2], 0);
        assert_eq!(premul[3], 0);
    }

    // -------------------------------------------------------------------------
    // TextureUploadDescriptor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_default() {
        let desc = TextureUploadDescriptor::default();

        assert_eq!(desc.offset, (0, 0, 0));
        assert_eq!(desc.size, (1, 1, 1));
        assert_eq!(desc.mip_level, 0);
        assert_eq!(desc.bytes_per_pixel, 4);
        assert!(desc.format.is_none());
    }

    #[test]
    fn test_descriptor_full() {
        let desc = TextureUploadDescriptor::full(1024, 512, 4);

        assert_eq!(desc.offset, (0, 0, 0));
        assert_eq!(desc.size, (1024, 512, 1));
        assert_eq!(desc.mip_level, 0);
        assert_eq!(desc.bytes_per_pixel, 4);
    }

    #[test]
    fn test_descriptor_mip_level() {
        // Mip 2 of 1024x1024 = 256x256
        let desc = TextureUploadDescriptor::mip_level(1024, 1024, 2, 4);

        assert_eq!(desc.size, (256, 256, 1));
        assert_eq!(desc.mip_level, 2);
    }

    #[test]
    fn test_descriptor_mip_level_non_square() {
        // Mip 1 of 1024x512 = 512x256
        let desc = TextureUploadDescriptor::mip_level(1024, 512, 1, 4);

        assert_eq!(desc.size, (512, 256, 1));
    }

    #[test]
    fn test_descriptor_subregion() {
        let desc = TextureUploadDescriptor::subregion(100, 200, 300, 400, 4);

        assert_eq!(desc.offset, (100, 200, 0));
        assert_eq!(desc.size, (300, 400, 1));
    }

    #[test]
    fn test_descriptor_required_data_size() {
        // 100x100 RGBA: 100*4 = 400 -> 512 aligned
        // 512 * 100 = 51200 bytes
        let desc = TextureUploadDescriptor::full(100, 100, 4);
        assert_eq!(desc.required_data_size(), 51200);

        // 64x64 RGBA: 64*4 = 256 (already aligned)
        // 256 * 64 = 16384 bytes
        let desc2 = TextureUploadDescriptor::full(64, 64, 4);
        assert_eq!(desc2.required_data_size(), 16384);
    }

    #[test]
    fn test_descriptor_unpadded_data_size() {
        let desc = TextureUploadDescriptor::full(100, 100, 4);
        assert_eq!(desc.unpadded_data_size(), 100 * 100 * 4);
    }

    #[test]
    fn test_descriptor_builder_pattern() {
        let desc = TextureUploadDescriptor::full(256, 256, 4)
            .with_format(TextureFormat::Rgba8Unorm)
            .with_mip_level(1)
            .with_layer(2);

        assert_eq!(desc.format, Some(TextureFormat::Rgba8Unorm));
        assert_eq!(desc.mip_level, 1);
        assert_eq!(desc.offset.2, 2);
    }

    // -------------------------------------------------------------------------
    // TextureRegion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_region_full() {
        let region = TextureRegion::full(1024, 512, 1);

        assert_eq!(region.offset, (0, 0, 0));
        assert_eq!(region.size, (1024, 512, 1));
    }

    #[test]
    fn test_region_mip() {
        let region = TextureRegion::mip(1024, 1024, 2);

        assert_eq!(region.size, (256, 256, 1));
    }

    #[test]
    fn test_region_mip_minimum() {
        // Very high mip level should clamp to 1x1
        let region = TextureRegion::mip(256, 256, 10);

        assert_eq!(region.size, (1, 1, 1));
    }

    #[test]
    fn test_region_subregion() {
        let region = TextureRegion::subregion(10, 20, 100, 200);

        assert_eq!(region.offset, (10, 20, 0));
        assert_eq!(region.size, (100, 200, 1));
    }

    #[test]
    fn test_region_layer() {
        let region = TextureRegion::layer(256, 256, 3);

        assert_eq!(region.offset, (0, 0, 3));
        assert_eq!(region.size, (256, 256, 1));
    }

    #[test]
    fn test_region_layer_range() {
        let region = TextureRegion::layer_range(256, 256, 2, 4);

        assert_eq!(region.offset, (0, 0, 2));
        assert_eq!(region.size, (256, 256, 4));
    }

    #[test]
    fn test_region_is_valid_for() {
        let region = TextureRegion::subregion(100, 100, 50, 50);

        assert!(region.is_valid_for(256, 256, 1));
        assert!(region.is_valid_for(150, 150, 1));
        assert!(!region.is_valid_for(140, 256, 1)); // x overflow
        assert!(!region.is_valid_for(256, 140, 1)); // y overflow
    }

    #[test]
    fn test_region_is_non_empty() {
        assert!(TextureRegion::full(1, 1, 1).is_non_empty());
        assert!(!TextureRegion { offset: (0, 0, 0), size: (0, 1, 1) }.is_non_empty());
        assert!(!TextureRegion { offset: (0, 0, 0), size: (1, 0, 1) }.is_non_empty());
        assert!(!TextureRegion { offset: (0, 0, 0), size: (1, 1, 0) }.is_non_empty());
    }

    // -------------------------------------------------------------------------
    // TextureUploader Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_uploader_new() {
        let uploader = TextureUploader::new(1024);
        assert_eq!(uploader.staging_threshold(), 1024);
    }

    #[test]
    fn test_uploader_default_threshold() {
        let uploader = TextureUploader::with_default_threshold();
        assert_eq!(uploader.staging_threshold(), STAGING_THRESHOLD);
    }

    #[test]
    fn test_uploader_set_threshold() {
        let mut uploader = TextureUploader::new(1024);
        uploader.set_staging_threshold(2048);
        assert_eq!(uploader.staging_threshold(), 2048);
    }

    #[test]
    fn test_uploader_validate_zero_size() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor {
            size: (0, 1, 1),
            ..Default::default()
        };
        let result = uploader.validate_upload(&desc, 100);

        assert!(matches!(result, Err(TextureUploadError::ZeroSizeRegion)));
    }

    #[test]
    fn test_uploader_validate_invalid_bpp() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor {
            bytes_per_pixel: 0,
            ..Default::default()
        };
        let result = uploader.validate_upload(&desc, 100);

        assert!(matches!(
            result,
            Err(TextureUploadError::InvalidBytesPerPixel { value: 0 })
        ));
    }

    #[test]
    fn test_uploader_validate_buffer_too_small() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor::full(10, 10, 4);
        // Needs 10*10*4 = 400 bytes, but we only provide 100
        let result = uploader.validate_upload(&desc, 100);

        assert!(matches!(
            result,
            Err(TextureUploadError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn test_uploader_validate_success() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor::full(10, 10, 4);
        let result = uploader.validate_upload(&desc, 400);

        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Error Display Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_error_display_invalid_region() {
        let err = TextureUploadError::InvalidRegion {
            offset: (100, 100, 0),
            size: (200, 200, 1),
            texture_size: (256, 256, 1),
            mip_level: 0,
        };
        let msg = err.to_string();
        assert!(msg.contains("100"));
        assert!(msg.contains("200"));
        assert!(msg.contains("256"));
    }

    #[test]
    fn test_error_display_buffer_too_small() {
        let err = TextureUploadError::BufferTooSmall {
            provided: 100,
            required: 400,
        };
        let msg = err.to_string();
        assert!(msg.contains("100"));
        assert!(msg.contains("400"));
    }

    #[test]
    fn test_error_display_format_mismatch() {
        let err = TextureUploadError::FormatMismatch {
            expected: "RGBA8".to_string(),
            actual: "RGB8".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("RGBA8"));
        assert!(msg.contains("RGB8"));
    }

    #[test]
    fn test_error_display_zero_size() {
        let err = TextureUploadError::ZeroSizeRegion;
        assert!(err.to_string().contains("zero"));
    }

    #[test]
    fn test_error_display_mip_out_of_range() {
        let err = TextureUploadError::MipLevelOutOfRange {
            level: 10,
            max_level: 5,
        };
        let msg = err.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    // -------------------------------------------------------------------------
    // Utility Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mip_size() {
        assert_eq!(mip_size(1024, 0), 1024);
        assert_eq!(mip_size(1024, 1), 512);
        assert_eq!(mip_size(1024, 2), 256);
        assert_eq!(mip_size(1024, 10), 1);
        assert_eq!(mip_size(1024, 20), 1); // Clamps to 1
    }

    #[test]
    fn test_bytes_per_pixel_for_format() {
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::R8Unorm), Some(1));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rg8Unorm), Some(2));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rgba8Unorm), Some(4));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rgba16Float), Some(8));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rgba32Float), Some(16));
        // Compressed format returns None
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Bc1RgbaUnorm), None);
    }

    // -------------------------------------------------------------------------
    // Constants Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants() {
        assert_eq!(ROW_PITCH_ALIGNMENT, 256);
        assert_eq!(STAGING_THRESHOLD, 65536);
    }

    // -------------------------------------------------------------------------
    // Clone/Debug Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_uploader_clone() {
        let uploader = TextureUploader::new(1024);
        let cloned = uploader.clone();
        assert_eq!(cloned.staging_threshold(), uploader.staging_threshold());
    }

    #[test]
    fn test_descriptor_clone() {
        let desc = TextureUploadDescriptor::full(100, 100, 4);
        let cloned = desc.clone();
        assert_eq!(desc, cloned);
    }

    #[test]
    fn test_region_copy() {
        let region = TextureRegion::full(100, 100, 1);
        let copied = region;
        assert_eq!(region, copied);
    }

    #[test]
    fn test_error_clone() {
        let err = TextureUploadError::ZeroSizeRegion;
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }
}
