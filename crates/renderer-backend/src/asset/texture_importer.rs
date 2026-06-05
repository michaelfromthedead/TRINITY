//! Base Texture Importer (stb_image-compatible)
//!
//! Implements the core texture import pipeline using the `image` crate
//! (Rust equivalent of stb_image) for decoding PNG, JPEG, TGA, and BMP formats.
//!
//! # Features
//!
//! - Decode PNG (8-bit and 16-bit), JPEG, TGA, BMP
//! - Map source pixel formats to GPU formats (R8G8B8A8_UNORM, R8G8B8A8_SRGB, R16G16B16A16_FLOAT)
//! - sRGB detection and gamma-corrected pipeline
//! - Memory budget tracking per-texture
//! - Texture state tracking: PENDING -> UPLOADING -> READY -> EVICTED
//! - Extension-based format detection for @asset decorator integration
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::{TextureImporter, MemoryBudgetTracker, GpuTextureFormat};
//!
//! let mut budget = MemoryBudgetTracker::new(1024 * 1024 * 512); // 512 MB
//! let importer = TextureImporter::new();
//!
//! let png_data = std::fs::read("texture.png")?;
//! let asset = importer.import_from_bytes(&png_data, Some("png"), &mut budget)?;
//!
//! assert_eq!(asset.metadata.format, GpuTextureFormat::R8G8B8A8Srgb);
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use image::{DynamicImage, GenericImageView, ImageFormat, ImageReader};
use parking_lot::RwLock;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

/// Errors that can occur during texture import.
#[derive(Debug, Clone)]
pub enum TextureImportError {
    /// File format not supported
    UnsupportedFormat(String),
    /// File data is invalid or corrupted
    InvalidData(String),
    /// Image decoding failed
    DecodeFailed(String),
    /// Texture exceeds memory budget
    BudgetExceeded { required: usize, available: usize },
    /// Texture dimensions are invalid (zero or too large)
    InvalidDimensions { width: u32, height: u32 },
    /// I/O error
    IoError(String),
}

impl fmt::Display for TextureImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextureImportError::UnsupportedFormat(fmt) => {
                write!(f, "unsupported texture format: {}", fmt)
            }
            TextureImportError::InvalidData(msg) => write!(f, "invalid data: {}", msg),
            TextureImportError::DecodeFailed(msg) => write!(f, "decode failed: {}", msg),
            TextureImportError::BudgetExceeded { required, available } => {
                write!(
                    f,
                    "texture exceeds memory budget: requires {} bytes, {} available",
                    required, available
                )
            }
            TextureImportError::InvalidDimensions { width, height } => {
                write!(f, "invalid dimensions: {}x{}", width, height)
            }
            TextureImportError::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for TextureImportError {}

impl From<image::ImageError> for TextureImportError {
    fn from(err: image::ImageError) -> Self {
        match err {
            image::ImageError::Decoding(e) => {
                TextureImportError::DecodeFailed(e.to_string())
            }
            image::ImageError::Unsupported(e) => {
                TextureImportError::UnsupportedFormat(e.to_string())
            }
            image::ImageError::IoError(e) => TextureImportError::IoError(e.to_string()),
            _ => TextureImportError::InvalidData(err.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// GPU Texture Format
// ---------------------------------------------------------------------------

/// Target GPU texture format for imported textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuTextureFormat {
    /// 8-bit RGBA, linear color space (UNORM)
    R8G8B8A8Unorm,
    /// 8-bit RGBA, sRGB color space
    R8G8B8A8Srgb,
    /// 16-bit RGBA, floating point (for HDR)
    R16G16B16A16Float,
    /// 8-bit single channel, linear
    R8Unorm,
    /// 8-bit dual channel, linear (RG)
    R8G8Unorm,
    /// 16-bit single channel, linear
    R16Unorm,
    /// 16-bit dual channel, linear
    R16G16Unorm,
    /// 16-bit RGBA, linear
    R16G16B16A16Unorm,
}

impl GpuTextureFormat {
    /// Returns bytes per pixel for this format.
    #[inline]
    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            GpuTextureFormat::R8Unorm => 1,
            GpuTextureFormat::R8G8Unorm => 2,
            GpuTextureFormat::R16Unorm => 2,
            GpuTextureFormat::R8G8B8A8Unorm | GpuTextureFormat::R8G8B8A8Srgb => 4,
            GpuTextureFormat::R16G16Unorm => 4,
            GpuTextureFormat::R16G16B16A16Unorm | GpuTextureFormat::R16G16B16A16Float => 8,
        }
    }

    /// Returns true if this format uses sRGB color space.
    #[inline]
    pub const fn is_srgb(&self) -> bool {
        matches!(self, GpuTextureFormat::R8G8B8A8Srgb)
    }

    /// Returns true if this format uses floating point.
    #[inline]
    pub const fn is_float(&self) -> bool {
        matches!(self, GpuTextureFormat::R16G16B16A16Float)
    }

    /// Returns the number of channels.
    #[inline]
    pub const fn channel_count(&self) -> u32 {
        match self {
            GpuTextureFormat::R8Unorm | GpuTextureFormat::R16Unorm => 1,
            GpuTextureFormat::R8G8Unorm | GpuTextureFormat::R16G16Unorm => 2,
            GpuTextureFormat::R8G8B8A8Unorm
            | GpuTextureFormat::R8G8B8A8Srgb
            | GpuTextureFormat::R16G16B16A16Unorm
            | GpuTextureFormat::R16G16B16A16Float => 4,
        }
    }

    /// Map to wgpu TextureFormat string representation.
    pub fn to_wgpu_format_str(&self) -> &'static str {
        match self {
            GpuTextureFormat::R8Unorm => "R8Unorm",
            GpuTextureFormat::R8G8Unorm => "Rg8Unorm",
            GpuTextureFormat::R16Unorm => "R16Unorm",
            GpuTextureFormat::R16G16Unorm => "Rg16Unorm",
            GpuTextureFormat::R8G8B8A8Unorm => "Rgba8Unorm",
            GpuTextureFormat::R8G8B8A8Srgb => "Rgba8UnormSrgb",
            GpuTextureFormat::R16G16B16A16Unorm => "Rgba16Unorm",
            GpuTextureFormat::R16G16B16A16Float => "Rgba16Float",
        }
    }
}

impl fmt::Display for GpuTextureFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuTextureFormat::R8Unorm => write!(f, "R8_UNORM"),
            GpuTextureFormat::R8G8Unorm => write!(f, "R8G8_UNORM"),
            GpuTextureFormat::R16Unorm => write!(f, "R16_UNORM"),
            GpuTextureFormat::R16G16Unorm => write!(f, "R16G16_UNORM"),
            GpuTextureFormat::R8G8B8A8Unorm => write!(f, "R8G8B8A8_UNORM"),
            GpuTextureFormat::R8G8B8A8Srgb => write!(f, "R8G8B8A8_SRGB"),
            GpuTextureFormat::R16G16B16A16Unorm => write!(f, "R16G16B16A16_UNORM"),
            GpuTextureFormat::R16G16B16A16Float => write!(f, "R16G16B16A16_FLOAT"),
        }
    }
}

// ---------------------------------------------------------------------------
// Texture State
// ---------------------------------------------------------------------------

/// State of a texture in the asset pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureState {
    /// Texture import requested but not yet processed
    Pending,
    /// Texture is being uploaded to GPU
    Uploading,
    /// Texture is ready for use
    Ready,
    /// Texture has been evicted from GPU memory
    Evicted,
}

impl fmt::Display for TextureState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextureState::Pending => write!(f, "PENDING"),
            TextureState::Uploading => write!(f, "UPLOADING"),
            TextureState::Ready => write!(f, "READY"),
            TextureState::Evicted => write!(f, "EVICTED"),
        }
    }
}

// ---------------------------------------------------------------------------
// Source Format Detection
// ---------------------------------------------------------------------------

/// Detected source image format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    Png,
    Jpeg,
    Tga,
    Bmp,
}

impl SourceFormat {
    /// Detect format from magic bytes.
    pub fn from_magic(data: &[u8]) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }

        // PNG: 89 50 4E 47 (0x89 'P' 'N' 'G')
        if data.len() >= 8
            && data[0] == 0x89
            && data[1] == b'P'
            && data[2] == b'N'
            && data[3] == b'G'
        {
            return Some(SourceFormat::Png);
        }

        // JPEG: FF D8 FF
        if data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
            return Some(SourceFormat::Jpeg);
        }

        // BMP: 'B' 'M'
        if data[0] == b'B' && data[1] == b'M' {
            return Some(SourceFormat::Bmp);
        }

        // TGA has no magic bytes, so we rely on extension hint
        None
    }

    /// Get format from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "png" => Some(SourceFormat::Png),
            "jpg" | "jpeg" | "jpe" => Some(SourceFormat::Jpeg),
            "tga" | "targa" => Some(SourceFormat::Tga),
            "bmp" | "dib" => Some(SourceFormat::Bmp),
            _ => None,
        }
    }

    /// Convert to image crate format.
    pub fn to_image_format(self) -> ImageFormat {
        match self {
            SourceFormat::Png => ImageFormat::Png,
            SourceFormat::Jpeg => ImageFormat::Jpeg,
            SourceFormat::Tga => ImageFormat::Tga,
            SourceFormat::Bmp => ImageFormat::Bmp,
        }
    }
}

// ---------------------------------------------------------------------------
// Texture Metadata
// ---------------------------------------------------------------------------

/// Metadata about an imported texture.
#[derive(Debug, Clone)]
pub struct TextureMetadata {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// GPU texture format
    pub format: GpuTextureFormat,
    /// Memory size in bytes
    pub memory_size: usize,
    /// Whether the texture uses sRGB color space
    pub is_srgb: bool,
    /// Source format (PNG, JPEG, etc.)
    pub source_format: SourceFormat,
    /// Bit depth per channel in source (8 or 16)
    pub source_bit_depth: u8,
    /// Number of channels in source
    pub source_channels: u8,
}

impl TextureMetadata {
    /// Calculate memory size from dimensions and format.
    pub fn calculate_memory_size(width: u32, height: u32, format: GpuTextureFormat) -> usize {
        width as usize * height as usize * format.bytes_per_pixel()
    }
}

// ---------------------------------------------------------------------------
// Texture Asset
// ---------------------------------------------------------------------------

/// A fully imported texture asset ready for GPU upload.
#[derive(Debug, Clone)]
pub struct TextureAsset {
    /// Unique asset ID
    pub id: u64,
    /// Texture metadata
    pub metadata: TextureMetadata,
    /// Raw pixel data (GPU-ready format)
    pub data: Vec<u8>,
    /// Current state
    pub state: TextureState,
}

impl TextureAsset {
    /// Transition to a new state.
    pub fn set_state(&mut self, new_state: TextureState) {
        self.state = new_state;
    }

    /// Check if texture is ready for rendering.
    pub fn is_ready(&self) -> bool {
        self.state == TextureState::Ready
    }

    /// Check if texture data is valid.
    pub fn is_valid(&self) -> bool {
        let expected = self.metadata.memory_size;
        self.data.len() == expected
    }
}

// ---------------------------------------------------------------------------
// Memory Budget Tracker
// ---------------------------------------------------------------------------

/// Tracks memory usage against a budget for textures.
#[derive(Debug)]
pub struct MemoryBudgetTracker {
    /// Total budget in bytes
    budget: usize,
    /// Current usage in bytes (atomic for thread safety)
    usage: AtomicUsize,
    /// Per-texture allocations (texture_id -> bytes)
    allocations: RwLock<HashMap<u64, usize>>,
}

impl MemoryBudgetTracker {
    /// Create a new budget tracker with the given budget in bytes.
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            usage: AtomicUsize::new(0),
            allocations: RwLock::new(HashMap::new()),
        }
    }

    /// Get the total budget.
    #[inline]
    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Get current usage.
    #[inline]
    pub fn usage(&self) -> usize {
        self.usage.load(Ordering::Relaxed)
    }

    /// Get available budget.
    #[inline]
    pub fn available(&self) -> usize {
        self.budget.saturating_sub(self.usage())
    }

    /// Check if allocation would exceed budget.
    #[inline]
    pub fn can_allocate(&self, size: usize) -> bool {
        self.usage() + size <= self.budget
    }

    /// Try to allocate memory for a texture.
    ///
    /// Returns Ok(()) if successful, Err with budget info if exceeded.
    pub fn allocate(&self, texture_id: u64, size: usize) -> Result<(), TextureImportError> {
        let available = self.available();
        if size > available {
            return Err(TextureImportError::BudgetExceeded {
                required: size,
                available,
            });
        }

        self.usage.fetch_add(size, Ordering::Relaxed);
        self.allocations.write().insert(texture_id, size);
        Ok(())
    }

    /// Deallocate memory for a texture.
    pub fn deallocate(&self, texture_id: u64) {
        if let Some(size) = self.allocations.write().remove(&texture_id) {
            self.usage.fetch_sub(size, Ordering::Relaxed);
        }
    }

    /// Get allocation for a specific texture.
    pub fn get_allocation(&self, texture_id: u64) -> Option<usize> {
        self.allocations.read().get(&texture_id).copied()
    }

    /// Get number of tracked textures.
    pub fn texture_count(&self) -> usize {
        self.allocations.read().len()
    }

    /// Get utilization as a percentage (0.0 - 1.0).
    #[inline]
    pub fn utilization(&self) -> f32 {
        if self.budget == 0 {
            return 1.0;
        }
        self.usage() as f32 / self.budget as f32
    }

    /// Set a new budget (does not affect current allocations).
    pub fn set_budget(&mut self, new_budget: usize) {
        self.budget = new_budget;
    }

    /// Reset all allocations (for testing/cleanup).
    pub fn reset(&self) {
        self.allocations.write().clear();
        self.usage.store(0, Ordering::Relaxed);
    }
}

impl Clone for MemoryBudgetTracker {
    fn clone(&self) -> Self {
        Self {
            budget: self.budget,
            usage: AtomicUsize::new(self.usage.load(Ordering::Relaxed)),
            allocations: RwLock::new(self.allocations.read().clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// sRGB Detection
// ---------------------------------------------------------------------------

/// Hints for sRGB detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrgbHint {
    /// Auto-detect based on file metadata and heuristics
    Auto,
    /// Force sRGB interpretation
    ForceSrgb,
    /// Force linear interpretation
    ForceLinear,
}

impl Default for SrgbHint {
    fn default() -> Self {
        SrgbHint::Auto
    }
}

/// Detect if an image should be treated as sRGB.
///
/// Heuristics:
/// - JPEG files are typically sRGB (photos)
/// - PNG with 'sRGB' chunk should be sRGB
/// - Grayscale images are typically linear
/// - Normal maps (detected by name patterns) are linear
fn detect_srgb(
    source_format: SourceFormat,
    data: &[u8],
    channels: u8,
    hint: SrgbHint,
) -> bool {
    match hint {
        SrgbHint::ForceSrgb => return true,
        SrgbHint::ForceLinear => return false,
        SrgbHint::Auto => {}
    }

    // Grayscale images are typically linear
    if channels == 1 || channels == 2 {
        return false;
    }

    match source_format {
        // JPEG is almost always sRGB (camera photos, etc.)
        SourceFormat::Jpeg => true,
        // PNG: check for sRGB chunk
        SourceFormat::Png => detect_png_srgb_chunk(data),
        // TGA: typically sRGB for color images
        SourceFormat::Tga => channels >= 3,
        // BMP: typically sRGB
        SourceFormat::Bmp => channels >= 3,
    }
}

/// Check if PNG has an sRGB chunk.
fn detect_png_srgb_chunk(data: &[u8]) -> bool {
    if data.len() < 12 {
        return true; // Default to sRGB for short data
    }

    // PNG chunk structure: 4 bytes length + 4 bytes type + data + 4 bytes CRC
    // Skip signature (8 bytes)
    let mut pos = 8;

    while pos + 8 <= data.len() {
        let length = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
            as usize;
        let chunk_type = &data[pos + 4..pos + 8];

        // Check for sRGB chunk
        if chunk_type == b"sRGB" {
            return true;
        }

        // Check for gAMA chunk (gamma) - presence suggests color management
        if chunk_type == b"gAMA" {
            // If gamma is ~2.2 (45455 in PNG units), it's likely sRGB
            if pos + 12 <= data.len() {
                let gamma = u32::from_be_bytes([
                    data[pos + 8],
                    data[pos + 9],
                    data[pos + 10],
                    data[pos + 11],
                ]);
                // PNG gamma is stored as 1/gamma * 100000
                // sRGB gamma ~2.2 -> 45454-45455
                if gamma >= 45000 && gamma <= 46000 {
                    return true;
                }
            }
        }

        // Stop at IDAT (image data) - metadata chunks come before
        if chunk_type == b"IDAT" {
            break;
        }

        // Move to next chunk (length + type + data + CRC)
        pos += 12 + length;
    }

    // Default to sRGB for color images
    true
}

// ---------------------------------------------------------------------------
// Texture Importer
// ---------------------------------------------------------------------------

/// Texture importer with stb_image-compatible format support.
pub struct TextureImporter {
    /// Counter for generating unique texture IDs
    next_id: AtomicU64,
    /// Default sRGB hint
    default_srgb_hint: SrgbHint,
    /// Maximum texture dimension (0 = unlimited)
    max_dimension: u32,
}

impl TextureImporter {
    /// Maximum supported texture dimension (16K x 16K).
    pub const MAX_SUPPORTED_DIMENSION: u32 = 16384;

    /// Create a new texture importer with default settings.
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            default_srgb_hint: SrgbHint::Auto,
            max_dimension: Self::MAX_SUPPORTED_DIMENSION,
        }
    }

    /// Create with custom settings.
    pub fn with_settings(srgb_hint: SrgbHint, max_dimension: u32) -> Self {
        Self {
            next_id: AtomicU64::new(1),
            default_srgb_hint: srgb_hint,
            max_dimension: if max_dimension == 0 {
                Self::MAX_SUPPORTED_DIMENSION
            } else {
                max_dimension
            },
        }
    }

    /// Set the default sRGB hint.
    pub fn set_srgb_hint(&mut self, hint: SrgbHint) {
        self.default_srgb_hint = hint;
    }

    /// Generate a new unique texture ID.
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Import a texture from raw bytes.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw file bytes
    /// * `extension_hint` - Optional file extension (e.g., "png", "jpg")
    /// * `budget` - Memory budget tracker
    ///
    /// # Returns
    ///
    /// A fully decoded texture asset, or an error.
    pub fn import_from_bytes(
        &self,
        data: &[u8],
        extension_hint: Option<&str>,
        budget: &MemoryBudgetTracker,
    ) -> Result<TextureAsset, TextureImportError> {
        self.import_with_hint(data, extension_hint, self.default_srgb_hint, budget)
    }

    /// Import with explicit sRGB hint.
    pub fn import_with_hint(
        &self,
        data: &[u8],
        extension_hint: Option<&str>,
        srgb_hint: SrgbHint,
        budget: &MemoryBudgetTracker,
    ) -> Result<TextureAsset, TextureImportError> {
        // Detect source format
        let source_format = self.detect_format(data, extension_hint)?;

        // Decode the image
        let image = self.decode_image(data, source_format)?;

        // Get dimensions
        let (width, height) = image.dimensions();

        // Validate dimensions
        self.validate_dimensions(width, height)?;

        // Determine source properties
        let (source_bit_depth, source_channels) = self.analyze_image(&image);

        // Detect sRGB
        let is_srgb = detect_srgb(source_format, data, source_channels, srgb_hint);

        // Choose GPU format and convert
        let (format, pixel_data) = self.convert_to_gpu_format(&image, is_srgb, source_bit_depth)?;

        // Calculate memory size
        let memory_size = TextureMetadata::calculate_memory_size(width, height, format);

        // Check budget
        let texture_id = self.next_id();
        budget.allocate(texture_id, memory_size)?;

        // Build metadata
        let metadata = TextureMetadata {
            width,
            height,
            format,
            memory_size,
            is_srgb,
            source_format,
            source_bit_depth,
            source_channels,
        };

        Ok(TextureAsset {
            id: texture_id,
            metadata,
            data: pixel_data,
            state: TextureState::Pending,
        })
    }

    /// Detect the source format from magic bytes and/or extension hint.
    fn detect_format(
        &self,
        data: &[u8],
        extension_hint: Option<&str>,
    ) -> Result<SourceFormat, TextureImportError> {
        // Try magic bytes first
        if let Some(format) = SourceFormat::from_magic(data) {
            return Ok(format);
        }

        // Fall back to extension
        if let Some(ext) = extension_hint {
            if let Some(format) = SourceFormat::from_extension(ext) {
                return Ok(format);
            }
        }

        Err(TextureImportError::UnsupportedFormat(
            extension_hint.unwrap_or("unknown").to_string(),
        ))
    }

    /// Decode the image using the image crate.
    fn decode_image(
        &self,
        data: &[u8],
        format: SourceFormat,
    ) -> Result<DynamicImage, TextureImportError> {
        let cursor = std::io::Cursor::new(data);
        let reader = ImageReader::with_format(cursor, format.to_image_format());

        reader.decode().map_err(TextureImportError::from)
    }

    /// Analyze the image to determine bit depth and channel count.
    fn analyze_image(&self, image: &DynamicImage) -> (u8, u8) {
        match image {
            DynamicImage::ImageLuma8(_) => (8, 1),
            DynamicImage::ImageLumaA8(_) => (8, 2),
            DynamicImage::ImageRgb8(_) => (8, 3),
            DynamicImage::ImageRgba8(_) => (8, 4),
            DynamicImage::ImageLuma16(_) => (16, 1),
            DynamicImage::ImageLumaA16(_) => (16, 2),
            DynamicImage::ImageRgb16(_) => (16, 3),
            DynamicImage::ImageRgba16(_) => (16, 4),
            DynamicImage::ImageRgb32F(_) => (32, 3),
            DynamicImage::ImageRgba32F(_) => (32, 4),
            _ => (8, 4), // Default fallback
        }
    }

    /// Validate texture dimensions.
    fn validate_dimensions(&self, width: u32, height: u32) -> Result<(), TextureImportError> {
        if width == 0 || height == 0 {
            return Err(TextureImportError::InvalidDimensions { width, height });
        }

        if width > self.max_dimension || height > self.max_dimension {
            return Err(TextureImportError::InvalidDimensions { width, height });
        }

        Ok(())
    }

    /// Convert image to GPU-ready format.
    fn convert_to_gpu_format(
        &self,
        image: &DynamicImage,
        is_srgb: bool,
        _source_bit_depth: u8,
    ) -> Result<(GpuTextureFormat, Vec<u8>), TextureImportError> {
        match image {
            // 8-bit grayscale
            DynamicImage::ImageLuma8(img) => {
                let data = img.as_raw().clone();
                Ok((GpuTextureFormat::R8Unorm, data))
            }

            // 8-bit grayscale + alpha
            DynamicImage::ImageLumaA8(img) => {
                let data = img.as_raw().clone();
                Ok((GpuTextureFormat::R8G8Unorm, data))
            }

            // 8-bit RGB -> expand to RGBA
            DynamicImage::ImageRgb8(_) => {
                let rgba = image.to_rgba8();
                let format = if is_srgb {
                    GpuTextureFormat::R8G8B8A8Srgb
                } else {
                    GpuTextureFormat::R8G8B8A8Unorm
                };
                Ok((format, rgba.into_raw()))
            }

            // 8-bit RGBA
            DynamicImage::ImageRgba8(img) => {
                let format = if is_srgb {
                    GpuTextureFormat::R8G8B8A8Srgb
                } else {
                    GpuTextureFormat::R8G8B8A8Unorm
                };
                Ok((format, img.as_raw().clone()))
            }

            // 16-bit grayscale
            DynamicImage::ImageLuma16(img) => {
                let raw = img.as_raw();
                let data: Vec<u8> = raw
                    .iter()
                    .flat_map(|&v| v.to_le_bytes())
                    .collect();
                Ok((GpuTextureFormat::R16Unorm, data))
            }

            // 16-bit grayscale + alpha
            DynamicImage::ImageLumaA16(img) => {
                let raw = img.as_raw();
                let data: Vec<u8> = raw
                    .iter()
                    .flat_map(|&v| v.to_le_bytes())
                    .collect();
                Ok((GpuTextureFormat::R16G16Unorm, data))
            }

            // 16-bit RGB -> expand to RGBA16
            DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgba16(_) => {
                let rgba16 = image.to_rgba16();
                let raw = rgba16.as_raw();
                let data: Vec<u8> = raw
                    .iter()
                    .flat_map(|&v| v.to_le_bytes())
                    .collect();
                Ok((GpuTextureFormat::R16G16B16A16Unorm, data))
            }

            // 32-bit float -> convert to 16-bit float
            DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_) => {
                let rgba32 = image.to_rgba32f();
                let raw = rgba32.as_raw();
                // Convert f32 to f16 (half precision)
                let data: Vec<u8> = raw
                    .iter()
                    .flat_map(|&v| {
                        let h = half::f16::from_f32(v);
                        h.to_le_bytes()
                    })
                    .collect();
                Ok((GpuTextureFormat::R16G16B16A16Float, data))
            }

            // Fallback: convert to RGBA8
            _ => {
                let rgba = image.to_rgba8();
                let format = if is_srgb {
                    GpuTextureFormat::R8G8B8A8Srgb
                } else {
                    GpuTextureFormat::R8G8B8A8Unorm
                };
                Ok((format, rgba.into_raw()))
            }
        }
    }

    /// Supported file extensions for @asset decorator integration.
    pub fn supported_extensions() -> &'static [&'static str] {
        &["png", "jpg", "jpeg", "jpe", "tga", "targa", "bmp", "dib"]
    }

    /// Check if an extension is supported.
    pub fn is_extension_supported(ext: &str) -> bool {
        SourceFormat::from_extension(ext).is_some()
    }
}

impl Default for TextureImporter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // PNG test data generators
    fn create_minimal_png(width: u32, height: u32, bit_depth: u8, color_type: u8) -> Vec<u8> {
        // PNG signature
        let mut data = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

        // IHDR chunk
        let ihdr_data = [
            (width >> 24) as u8,
            (width >> 16) as u8,
            (width >> 8) as u8,
            width as u8,
            (height >> 24) as u8,
            (height >> 16) as u8,
            (height >> 8) as u8,
            height as u8,
            bit_depth,
            color_type,
            0, // compression
            0, // filter
            0, // interlace
        ];

        // IHDR chunk: length (4) + "IHDR" (4) + data (13) + CRC (4)
        data.extend_from_slice(&[0, 0, 0, 13]); // length
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&ihdr_data);
        // Simplified CRC (not valid but sufficient for header parsing)
        data.extend_from_slice(&[0, 0, 0, 0]);

        data
    }

    fn create_valid_png_8bit_rgba(width: u32, height: u32) -> Vec<u8> {
        // Create a minimal valid PNG with actual image data
        use std::io::Write;

        let mut png_data = Vec::new();

        // We'll use a simple approach: create raw pixel data and encode it
        let pixel_data: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let r = ((x * 255) / width.max(1)) as u8;
                    let g = ((y * 255) / height.max(1)) as u8;
                    let b = 128u8;
                    let a = 255u8;
                    vec![r, g, b, a]
                })
            })
            .collect();

        // Use the image crate to create a valid PNG
        let img = image::RgbaImage::from_raw(width, height, pixel_data)
            .expect("Failed to create image");

        let mut cursor = std::io::Cursor::new(&mut png_data);
        img.write_to(&mut cursor, ImageFormat::Png)
            .expect("Failed to write PNG");

        png_data
    }

    fn create_valid_png_16bit_rgba(width: u32, height: u32) -> Vec<u8> {
        use image::{ImageBuffer, Rgba};

        let pixel_data: Vec<u16> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let r = ((x as u32 * 65535) / width.max(1)) as u16;
                    let g = ((y as u32 * 65535) / height.max(1)) as u16;
                    let b = 32768u16;
                    let a = 65535u16;
                    vec![r, g, b, a]
                })
            })
            .collect();

        let img: ImageBuffer<Rgba<u16>, Vec<u16>> =
            ImageBuffer::from_raw(width, height, pixel_data)
                .expect("Failed to create image");

        let mut png_data: Vec<u8> = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut png_data);
        img.write_to(&mut cursor, ImageFormat::Png)
            .expect("Failed to write PNG");

        png_data
    }

    fn create_valid_png_grayscale(width: u32, height: u32) -> Vec<u8> {
        let pixel_data: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| ((x + y) * 255 / (width + height).max(1)) as u8)
            })
            .collect();

        let img = image::GrayImage::from_raw(width, height, pixel_data)
            .expect("Failed to create image");

        let mut png_data = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut png_data);
        img.write_to(&mut cursor, ImageFormat::Png)
            .expect("Failed to write PNG");

        png_data
    }

    fn create_valid_jpeg(width: u32, height: u32) -> Vec<u8> {
        let pixel_data: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let r = ((x * 255) / width.max(1)) as u8;
                    let g = ((y * 255) / height.max(1)) as u8;
                    let b = 100u8;
                    vec![r, g, b]
                })
            })
            .collect();

        let img = image::RgbImage::from_raw(width, height, pixel_data)
            .expect("Failed to create image");

        let mut jpeg_data = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut jpeg_data);
        img.write_to(&mut cursor, ImageFormat::Jpeg)
            .expect("Failed to write JPEG");

        jpeg_data
    }

    fn create_valid_bmp(width: u32, height: u32) -> Vec<u8> {
        let pixel_data: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let r = ((x * 255) / width.max(1)) as u8;
                    let g = ((y * 255) / height.max(1)) as u8;
                    let b = 50u8;
                    let a = 255u8;
                    vec![r, g, b, a]
                })
            })
            .collect();

        let img = image::RgbaImage::from_raw(width, height, pixel_data)
            .expect("Failed to create image");

        let mut bmp_data = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut bmp_data);
        img.write_to(&mut cursor, ImageFormat::Bmp)
            .expect("Failed to write BMP");

        bmp_data
    }

    fn create_valid_tga(width: u32, height: u32) -> Vec<u8> {
        let pixel_data: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let r = ((x * 255) / width.max(1)) as u8;
                    let g = ((y * 255) / height.max(1)) as u8;
                    let b = 75u8;
                    let a = 255u8;
                    vec![r, g, b, a]
                })
            })
            .collect();

        let img = image::RgbaImage::from_raw(width, height, pixel_data)
            .expect("Failed to create image");

        let mut tga_data = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut tga_data);
        img.write_to(&mut cursor, ImageFormat::Tga)
            .expect("Failed to write TGA");

        tga_data
    }

    // ---------------------------------------------------------------------------
    // PNG Decode Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_decode_png_8bit_rgba() {
        let png_data = create_valid_png_8bit_rgba(64, 64);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let result = importer.import_from_bytes(&png_data, Some("png"), &budget);
        assert!(result.is_ok(), "Failed to decode PNG: {:?}", result.err());

        let asset = result.unwrap();
        assert_eq!(asset.metadata.width, 64);
        assert_eq!(asset.metadata.height, 64);
        assert_eq!(asset.metadata.source_format, SourceFormat::Png);
        assert_eq!(asset.metadata.source_bit_depth, 8);
        assert_eq!(asset.metadata.source_channels, 4);
        assert!(asset.metadata.is_srgb);
        assert_eq!(asset.metadata.format, GpuTextureFormat::R8G8B8A8Srgb);
        assert_eq!(asset.state, TextureState::Pending);
    }

    #[test]
    fn test_decode_png_16bit_rgba() {
        let png_data = create_valid_png_16bit_rgba(32, 32);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let result = importer.import_from_bytes(&png_data, Some("png"), &budget);
        assert!(result.is_ok(), "Failed to decode 16-bit PNG: {:?}", result.err());

        let asset = result.unwrap();
        assert_eq!(asset.metadata.width, 32);
        assert_eq!(asset.metadata.height, 32);
        assert_eq!(asset.metadata.source_bit_depth, 16);
        assert_eq!(asset.metadata.format, GpuTextureFormat::R16G16B16A16Unorm);
        // 16-bit textures: 32 * 32 * 8 bytes = 8192 bytes
        assert_eq!(asset.metadata.memory_size, 32 * 32 * 8);
    }

    #[test]
    fn test_decode_png_grayscale() {
        let png_data = create_valid_png_grayscale(16, 16);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let result = importer.import_from_bytes(&png_data, Some("png"), &budget);
        assert!(result.is_ok(), "Failed to decode grayscale PNG: {:?}", result.err());

        let asset = result.unwrap();
        assert_eq!(asset.metadata.width, 16);
        assert_eq!(asset.metadata.height, 16);
        assert_eq!(asset.metadata.source_channels, 1);
        // Grayscale is NOT sRGB
        assert!(!asset.metadata.is_srgb);
        assert_eq!(asset.metadata.format, GpuTextureFormat::R8Unorm);
    }

    // ---------------------------------------------------------------------------
    // JPEG Decode Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_decode_jpeg() {
        let jpeg_data = create_valid_jpeg(128, 128);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let result = importer.import_from_bytes(&jpeg_data, Some("jpg"), &budget);
        assert!(result.is_ok(), "Failed to decode JPEG: {:?}", result.err());

        let asset = result.unwrap();
        assert_eq!(asset.metadata.width, 128);
        assert_eq!(asset.metadata.height, 128);
        assert_eq!(asset.metadata.source_format, SourceFormat::Jpeg);
        // JPEG is always sRGB
        assert!(asset.metadata.is_srgb);
        assert_eq!(asset.metadata.format, GpuTextureFormat::R8G8B8A8Srgb);
    }

    #[test]
    fn test_decode_jpeg_magic_detection() {
        let jpeg_data = create_valid_jpeg(64, 64);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        // No extension hint - should detect from magic bytes
        let result = importer.import_from_bytes(&jpeg_data, None, &budget);
        assert!(result.is_ok(), "Failed to detect JPEG from magic: {:?}", result.err());

        let asset = result.unwrap();
        assert_eq!(asset.metadata.source_format, SourceFormat::Jpeg);
    }

    // ---------------------------------------------------------------------------
    // TGA Decode Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_decode_tga() {
        let tga_data = create_valid_tga(64, 64);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        // TGA requires extension hint (no magic bytes)
        let result = importer.import_from_bytes(&tga_data, Some("tga"), &budget);
        assert!(result.is_ok(), "Failed to decode TGA: {:?}", result.err());

        let asset = result.unwrap();
        assert_eq!(asset.metadata.width, 64);
        assert_eq!(asset.metadata.height, 64);
        assert_eq!(asset.metadata.source_format, SourceFormat::Tga);
    }

    #[test]
    fn test_decode_tga_with_targa_extension() {
        let tga_data = create_valid_tga(32, 32);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let result = importer.import_from_bytes(&tga_data, Some("targa"), &budget);
        assert!(result.is_ok(), "Failed to decode TGA with .targa: {:?}", result.err());
    }

    // ---------------------------------------------------------------------------
    // BMP Decode Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_decode_bmp() {
        let bmp_data = create_valid_bmp(64, 64);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let result = importer.import_from_bytes(&bmp_data, Some("bmp"), &budget);
        assert!(result.is_ok(), "Failed to decode BMP: {:?}", result.err());

        let asset = result.unwrap();
        assert_eq!(asset.metadata.width, 64);
        assert_eq!(asset.metadata.height, 64);
        assert_eq!(asset.metadata.source_format, SourceFormat::Bmp);
    }

    #[test]
    fn test_decode_bmp_magic_detection() {
        let bmp_data = create_valid_bmp(32, 32);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        // BMP has magic bytes "BM"
        let result = importer.import_from_bytes(&bmp_data, None, &budget);
        assert!(result.is_ok(), "Failed to detect BMP from magic: {:?}", result.err());

        let asset = result.unwrap();
        assert_eq!(asset.metadata.source_format, SourceFormat::Bmp);
    }

    // ---------------------------------------------------------------------------
    // sRGB Detection Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_srgb_detection_jpeg() {
        let jpeg_data = create_valid_jpeg(16, 16);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let asset = importer.import_from_bytes(&jpeg_data, Some("jpg"), &budget).unwrap();
        assert!(asset.metadata.is_srgb, "JPEG should be detected as sRGB");
    }

    #[test]
    fn test_srgb_force_linear() {
        let png_data = create_valid_png_8bit_rgba(16, 16);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let asset = importer
            .import_with_hint(&png_data, Some("png"), SrgbHint::ForceLinear, &budget)
            .unwrap();
        assert!(!asset.metadata.is_srgb, "Should be forced to linear");
        assert_eq!(asset.metadata.format, GpuTextureFormat::R8G8B8A8Unorm);
    }

    #[test]
    fn test_srgb_force_srgb() {
        let png_data = create_valid_png_grayscale(16, 16);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        // Grayscale would normally be linear, but we force sRGB
        // Note: grayscale stays R8 format, but is_srgb flag changes
        let asset = importer
            .import_with_hint(&png_data, Some("png"), SrgbHint::ForceSrgb, &budget)
            .unwrap();
        assert!(asset.metadata.is_srgb, "Should be forced to sRGB");
    }

    // ---------------------------------------------------------------------------
    // Format Mapping Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_format_mapping_rgba8_srgb() {
        let png_data = create_valid_png_8bit_rgba(8, 8);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let asset = importer.import_from_bytes(&png_data, Some("png"), &budget).unwrap();
        assert_eq!(asset.metadata.format, GpuTextureFormat::R8G8B8A8Srgb);
        assert_eq!(asset.metadata.format.bytes_per_pixel(), 4);
        assert!(asset.metadata.format.is_srgb());
    }

    #[test]
    fn test_format_mapping_rgba8_unorm() {
        let png_data = create_valid_png_8bit_rgba(8, 8);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let asset = importer
            .import_with_hint(&png_data, Some("png"), SrgbHint::ForceLinear, &budget)
            .unwrap();
        assert_eq!(asset.metadata.format, GpuTextureFormat::R8G8B8A8Unorm);
        assert!(!asset.metadata.format.is_srgb());
    }

    #[test]
    fn test_format_mapping_r16g16b16a16_float() {
        // Test the R16G16B16A16Float format properties directly
        // Note: PNG doesn't support float encoding, so we just test the format properties
        let format = GpuTextureFormat::R16G16B16A16Float;
        assert_eq!(format.bytes_per_pixel(), 8);
        assert!(format.is_float());
        assert!(!format.is_srgb());
        assert_eq!(format.channel_count(), 4);
    }

    #[test]
    fn test_format_wgpu_mapping() {
        assert_eq!(GpuTextureFormat::R8G8B8A8Unorm.to_wgpu_format_str(), "Rgba8Unorm");
        assert_eq!(GpuTextureFormat::R8G8B8A8Srgb.to_wgpu_format_str(), "Rgba8UnormSrgb");
        assert_eq!(GpuTextureFormat::R16G16B16A16Float.to_wgpu_format_str(), "Rgba16Float");
        assert_eq!(GpuTextureFormat::R8Unorm.to_wgpu_format_str(), "R8Unorm");
    }

    // ---------------------------------------------------------------------------
    // Memory Budget Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_memory_budget_tracking() {
        let budget = MemoryBudgetTracker::new(1024 * 1024); // 1 MB
        assert_eq!(budget.budget(), 1024 * 1024);
        assert_eq!(budget.usage(), 0);
        assert_eq!(budget.available(), 1024 * 1024);
    }

    #[test]
    fn test_memory_budget_allocation() {
        let budget = MemoryBudgetTracker::new(1024 * 1024);

        // Allocate 512 KB
        assert!(budget.allocate(1, 512 * 1024).is_ok());
        assert_eq!(budget.usage(), 512 * 1024);
        assert_eq!(budget.available(), 512 * 1024);
        assert_eq!(budget.texture_count(), 1);
    }

    #[test]
    fn test_memory_budget_exceeded() {
        let budget = MemoryBudgetTracker::new(1024); // 1 KB

        let result = budget.allocate(1, 2048);
        assert!(matches!(result, Err(TextureImportError::BudgetExceeded { .. })));
    }

    #[test]
    fn test_memory_budget_deallocation() {
        let budget = MemoryBudgetTracker::new(1024 * 1024);

        budget.allocate(1, 256 * 1024).unwrap();
        budget.allocate(2, 256 * 1024).unwrap();
        assert_eq!(budget.usage(), 512 * 1024);
        assert_eq!(budget.texture_count(), 2);

        budget.deallocate(1);
        assert_eq!(budget.usage(), 256 * 1024);
        assert_eq!(budget.texture_count(), 1);

        budget.deallocate(2);
        assert_eq!(budget.usage(), 0);
        assert_eq!(budget.texture_count(), 0);
    }

    #[test]
    fn test_memory_budget_utilization() {
        let budget = MemoryBudgetTracker::new(1000);

        budget.allocate(1, 500).unwrap();
        assert!((budget.utilization() - 0.5).abs() < 0.001);

        budget.allocate(2, 250).unwrap();
        assert!((budget.utilization() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_memory_budget_can_allocate() {
        let budget = MemoryBudgetTracker::new(1024);

        assert!(budget.can_allocate(512));
        assert!(budget.can_allocate(1024));
        assert!(!budget.can_allocate(1025));

        budget.allocate(1, 512).unwrap();
        assert!(budget.can_allocate(512));
        assert!(!budget.can_allocate(513));
    }

    #[test]
    fn test_memory_budget_reset() {
        let budget = MemoryBudgetTracker::new(1024 * 1024);

        budget.allocate(1, 100).unwrap();
        budget.allocate(2, 200).unwrap();
        assert_eq!(budget.usage(), 300);

        budget.reset();
        assert_eq!(budget.usage(), 0);
        assert_eq!(budget.texture_count(), 0);
    }

    #[test]
    fn test_import_updates_budget() {
        let png_data = create_valid_png_8bit_rgba(64, 64);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let initial_usage = budget.usage();
        let asset = importer.import_from_bytes(&png_data, Some("png"), &budget).unwrap();

        let expected_size = 64 * 64 * 4; // RGBA8
        assert_eq!(budget.usage(), initial_usage + expected_size);
        assert_eq!(budget.get_allocation(asset.id), Some(expected_size));
    }

    #[test]
    fn test_import_fails_on_budget_exceeded() {
        let png_data = create_valid_png_8bit_rgba(64, 64);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(100); // Very small budget

        let result = importer.import_from_bytes(&png_data, Some("png"), &budget);
        assert!(matches!(result, Err(TextureImportError::BudgetExceeded { .. })));
    }

    // ---------------------------------------------------------------------------
    // State Transition Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_state_initial_pending() {
        let png_data = create_valid_png_8bit_rgba(8, 8);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024);

        let asset = importer.import_from_bytes(&png_data, Some("png"), &budget).unwrap();
        assert_eq!(asset.state, TextureState::Pending);
    }

    #[test]
    fn test_state_transitions() {
        let png_data = create_valid_png_8bit_rgba(8, 8);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024);

        let mut asset = importer.import_from_bytes(&png_data, Some("png"), &budget).unwrap();

        // PENDING -> UPLOADING
        asset.set_state(TextureState::Uploading);
        assert_eq!(asset.state, TextureState::Uploading);
        assert!(!asset.is_ready());

        // UPLOADING -> READY
        asset.set_state(TextureState::Ready);
        assert_eq!(asset.state, TextureState::Ready);
        assert!(asset.is_ready());

        // READY -> EVICTED
        asset.set_state(TextureState::Evicted);
        assert_eq!(asset.state, TextureState::Evicted);
        assert!(!asset.is_ready());
    }

    // ---------------------------------------------------------------------------
    // Invalid File Handling Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_invalid_empty_data() {
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024);

        let result = importer.import_from_bytes(&[], Some("png"), &budget);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_truncated_png() {
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024);

        // Just the PNG signature, no actual data
        let truncated = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let result = importer.import_from_bytes(&truncated, Some("png"), &budget);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_corrupt_jpeg() {
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024);

        // JPEG header followed by garbage
        let corrupt = vec![0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0, 0, 0];
        let result = importer.import_from_bytes(&corrupt, Some("jpg"), &budget);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_unknown_format() {
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024);

        let random_data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let result = importer.import_from_bytes(&random_data, Some("xyz"), &budget);
        assert!(matches!(result, Err(TextureImportError::UnsupportedFormat(_))));
    }

    #[test]
    fn test_invalid_no_format_hint() {
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024);

        // Random data with no magic bytes and no hint
        let random_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let result = importer.import_from_bytes(&random_data, None, &budget);
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------------------
    // Large Texture Handling Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_large_texture_within_limits() {
        // 256x256 is reasonable
        let png_data = create_valid_png_8bit_rgba(256, 256);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let result = importer.import_from_bytes(&png_data, Some("png"), &budget);
        assert!(result.is_ok());

        let asset = result.unwrap();
        assert_eq!(asset.metadata.width, 256);
        assert_eq!(asset.metadata.height, 256);
        assert_eq!(asset.metadata.memory_size, 256 * 256 * 4);
    }

    #[test]
    fn test_max_dimension_limit() {
        let importer = TextureImporter::with_settings(SrgbHint::Auto, 64);
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        // 128x128 exceeds our custom 64 max dimension
        let png_data = create_valid_png_8bit_rgba(128, 128);
        let result = importer.import_from_bytes(&png_data, Some("png"), &budget);
        assert!(matches!(result, Err(TextureImportError::InvalidDimensions { .. })));
    }

    // ---------------------------------------------------------------------------
    // Asset Validation Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_asset_is_valid() {
        let png_data = create_valid_png_8bit_rgba(16, 16);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024);

        let asset = importer.import_from_bytes(&png_data, Some("png"), &budget).unwrap();
        assert!(asset.is_valid());
        assert_eq!(asset.data.len(), asset.metadata.memory_size);
    }

    #[test]
    fn test_asset_unique_ids() {
        let png_data = create_valid_png_8bit_rgba(8, 8);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let asset1 = importer.import_from_bytes(&png_data, Some("png"), &budget).unwrap();
        let asset2 = importer.import_from_bytes(&png_data, Some("png"), &budget).unwrap();
        let asset3 = importer.import_from_bytes(&png_data, Some("png"), &budget).unwrap();

        assert_ne!(asset1.id, asset2.id);
        assert_ne!(asset2.id, asset3.id);
        assert_ne!(asset1.id, asset3.id);
    }

    // ---------------------------------------------------------------------------
    // Extension Support Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_supported_extensions() {
        let extensions = TextureImporter::supported_extensions();
        assert!(extensions.contains(&"png"));
        assert!(extensions.contains(&"jpg"));
        assert!(extensions.contains(&"jpeg"));
        assert!(extensions.contains(&"tga"));
        assert!(extensions.contains(&"bmp"));
    }

    #[test]
    fn test_is_extension_supported() {
        assert!(TextureImporter::is_extension_supported("png"));
        assert!(TextureImporter::is_extension_supported("PNG"));
        assert!(TextureImporter::is_extension_supported("jpg"));
        assert!(TextureImporter::is_extension_supported("jpeg"));
        assert!(TextureImporter::is_extension_supported("tga"));
        assert!(TextureImporter::is_extension_supported("bmp"));
        assert!(!TextureImporter::is_extension_supported("gif"));
        assert!(!TextureImporter::is_extension_supported("webp"));
    }

    // ---------------------------------------------------------------------------
    // Format Display Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_gpu_format_display() {
        assert_eq!(format!("{}", GpuTextureFormat::R8G8B8A8Unorm), "R8G8B8A8_UNORM");
        assert_eq!(format!("{}", GpuTextureFormat::R8G8B8A8Srgb), "R8G8B8A8_SRGB");
        assert_eq!(format!("{}", GpuTextureFormat::R16G16B16A16Float), "R16G16B16A16_FLOAT");
    }

    #[test]
    fn test_state_display() {
        assert_eq!(format!("{}", TextureState::Pending), "PENDING");
        assert_eq!(format!("{}", TextureState::Uploading), "UPLOADING");
        assert_eq!(format!("{}", TextureState::Ready), "READY");
        assert_eq!(format!("{}", TextureState::Evicted), "EVICTED");
    }

    #[test]
    fn test_error_display() {
        let err = TextureImportError::UnsupportedFormat("xyz".to_string());
        assert!(err.to_string().contains("unsupported"));

        let err = TextureImportError::BudgetExceeded {
            required: 1000,
            available: 500,
        };
        assert!(err.to_string().contains("1000"));
        assert!(err.to_string().contains("500"));
    }

    // ---------------------------------------------------------------------------
    // Integration Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_full_import_pipeline_png() {
        let png_data = create_valid_png_8bit_rgba(32, 32);
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024);

        // Import
        let mut asset = importer.import_from_bytes(&png_data, Some("png"), &budget).unwrap();

        // Verify initial state
        assert_eq!(asset.state, TextureState::Pending);
        assert!(asset.is_valid());

        // Simulate upload
        asset.set_state(TextureState::Uploading);
        assert!(!asset.is_ready());

        // Simulate completion
        asset.set_state(TextureState::Ready);
        assert!(asset.is_ready());

        // Verify budget tracking
        assert!(budget.get_allocation(asset.id).is_some());

        // Simulate eviction
        asset.set_state(TextureState::Evicted);
        budget.deallocate(asset.id);
        assert_eq!(budget.usage(), 0);
    }

    #[test]
    fn test_multiple_format_imports() {
        let importer = TextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let png = create_valid_png_8bit_rgba(16, 16);
        let jpg = create_valid_jpeg(16, 16);
        let bmp = create_valid_bmp(16, 16);
        let tga = create_valid_tga(16, 16);

        let png_asset = importer.import_from_bytes(&png, Some("png"), &budget).unwrap();
        let jpg_asset = importer.import_from_bytes(&jpg, Some("jpg"), &budget).unwrap();
        let bmp_asset = importer.import_from_bytes(&bmp, Some("bmp"), &budget).unwrap();
        let tga_asset = importer.import_from_bytes(&tga, Some("tga"), &budget).unwrap();

        assert_eq!(png_asset.metadata.source_format, SourceFormat::Png);
        assert_eq!(jpg_asset.metadata.source_format, SourceFormat::Jpeg);
        assert_eq!(bmp_asset.metadata.source_format, SourceFormat::Bmp);
        assert_eq!(tga_asset.metadata.source_format, SourceFormat::Tga);

        assert_eq!(budget.texture_count(), 4);
    }
}
