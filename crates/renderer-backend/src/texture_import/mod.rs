//! Texture importer plugin system.
//!
//! Provides a flexible plugin architecture for importing various texture formats.
//! Importers implement the [`FormatImporter`] trait and are registered with an
//! [`ImporterRegistry`] for format resolution and import dispatch.
//!
//! # Example
//!
//! ```
//! use renderer_backend::texture_import::{ImporterRegistry, TextureFormat};
//!
//! let registry = ImporterRegistry::with_defaults();
//!
//! // Import by magic bytes (auto-detect format)
//! let png_data = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
//! if let Some(importer) = registry.resolve_by_magic(png_data) {
//!     // importer.import(full_png_data) would decode the texture
//! }
//!
//! // Import by extension
//! if let Some(importer) = registry.resolve_by_extension("png") {
//!     // Use the PNG importer
//! }
//! ```

mod cook;
#[cfg(test)]
mod cook_tests;
mod fbx_importer;
mod importers;
mod ktx2_importer;
mod usd_importer;

use std::fmt;
use std::io;

pub use cook::{
    box_filter_2x2, calculate_mip_levels, CookError, CookedTexture, GpuTextureFormat,
    TextureCooker, TextureUsage,
};
pub use fbx_importer::{FbxImporter, FbxTextureRef, FbxHeader, FbxEmbeddedTexture};
pub use importers::{BmpImporter, JpegImporter, PngImporter, TgaImporter};
pub use ktx2_importer::{Ktx2Importer, Ktx2Header, Ktx2LevelIndex, Ktx2TextureType, VkFormat};
pub use usd_importer::{UsdImporter, UsdTextureRef};

// ---------------------------------------------------------------------------
// TextureFormat
// ---------------------------------------------------------------------------

/// Pixel format for imported texture data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// Single 8-bit channel (grayscale).
    R8,
    /// Two 8-bit channels (red, green).
    Rg8,
    /// Four 8-bit channels (red, green, blue, alpha), linear.
    Rgba8,
    /// Four 8-bit channels (red, green, blue, alpha), sRGB.
    Rgba8Srgb,
    /// Single 16-bit channel.
    R16,
    /// Two 16-bit channels.
    Rg16,
    /// Four 16-bit channels.
    Rgba16,
    /// Single 32-bit floating point channel.
    R32F,
    /// Two 32-bit floating point channels.
    Rg32F,
    /// Four 32-bit floating point channels.
    Rgba32F,
}

impl TextureFormat {
    /// Returns the number of bytes per pixel for this format.
    #[inline]
    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            TextureFormat::R8 => 1,
            TextureFormat::Rg8 => 2,
            TextureFormat::Rgba8 | TextureFormat::Rgba8Srgb => 4,
            TextureFormat::R16 => 2,
            TextureFormat::Rg16 => 4,
            TextureFormat::Rgba16 => 8,
            TextureFormat::R32F => 4,
            TextureFormat::Rg32F => 8,
            TextureFormat::Rgba32F => 16,
        }
    }

    /// Returns the number of channels in this format.
    #[inline]
    pub const fn channels(&self) -> u8 {
        match self {
            TextureFormat::R8 | TextureFormat::R16 | TextureFormat::R32F => 1,
            TextureFormat::Rg8 | TextureFormat::Rg16 | TextureFormat::Rg32F => 2,
            TextureFormat::Rgba8
            | TextureFormat::Rgba8Srgb
            | TextureFormat::Rgba16
            | TextureFormat::Rgba32F => 4,
        }
    }
}

impl fmt::Display for TextureFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextureFormat::R8 => write!(f, "R8"),
            TextureFormat::Rg8 => write!(f, "RG8"),
            TextureFormat::Rgba8 => write!(f, "RGBA8"),
            TextureFormat::Rgba8Srgb => write!(f, "RGBA8_sRGB"),
            TextureFormat::R16 => write!(f, "R16"),
            TextureFormat::Rg16 => write!(f, "RG16"),
            TextureFormat::Rgba16 => write!(f, "RGBA16"),
            TextureFormat::R32F => write!(f, "R32F"),
            TextureFormat::Rg32F => write!(f, "RG32F"),
            TextureFormat::Rgba32F => write!(f, "RGBA32F"),
        }
    }
}

// ---------------------------------------------------------------------------
// TextureData
// ---------------------------------------------------------------------------

/// Raw texture data produced by an importer.
///
/// Contains pixel data in row-major order (top-to-bottom), along with
/// dimensions and format information.
#[derive(Debug, Clone, PartialEq)]
pub struct TextureData {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format of the data.
    pub format: TextureFormat,
    /// Raw pixel data (row-major, top-to-bottom).
    pub data: Vec<u8>,
    /// Number of mip levels (1 = base level only, no additional mips).
    pub mip_levels: u32,
}

impl TextureData {
    /// Create new texture data with the given parameters.
    pub fn new(
        width: u32,
        height: u32,
        format: TextureFormat,
        data: Vec<u8>,
        mip_levels: u32,
    ) -> Self {
        Self {
            width,
            height,
            format,
            data,
            mip_levels,
        }
    }

    /// Returns the expected byte size for the base mip level.
    #[inline]
    pub fn expected_size(&self) -> usize {
        self.width as usize * self.height as usize * self.format.bytes_per_pixel()
    }

    /// Validates that the data length matches the expected size.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.data.len() >= self.expected_size()
    }
}

// ---------------------------------------------------------------------------
// ImportError
// ---------------------------------------------------------------------------

/// Errors that can occur during texture import.
#[derive(Debug)]
pub enum ImportError {
    /// The format is not supported by any registered importer.
    UnsupportedFormat(String),
    /// The input data is invalid or corrupted.
    InvalidData(String),
    /// An I/O error occurred during import.
    IoError(io::Error),
    /// The decoder failed to process the data.
    DecodeFailed(String),
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImportError::UnsupportedFormat(fmt_name) => {
                write!(f, "unsupported format: {}", fmt_name)
            }
            ImportError::InvalidData(msg) => {
                write!(f, "invalid data: {}", msg)
            }
            ImportError::IoError(err) => {
                write!(f, "I/O error: {}", err)
            }
            ImportError::DecodeFailed(msg) => {
                write!(f, "decode failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for ImportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ImportError::IoError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for ImportError {
    fn from(err: io::Error) -> Self {
        ImportError::IoError(err)
    }
}

impl Clone for ImportError {
    fn clone(&self) -> Self {
        match self {
            ImportError::UnsupportedFormat(s) => ImportError::UnsupportedFormat(s.clone()),
            ImportError::InvalidData(s) => ImportError::InvalidData(s.clone()),
            ImportError::IoError(e) => {
                ImportError::IoError(io::Error::new(e.kind(), e.to_string()))
            }
            ImportError::DecodeFailed(s) => ImportError::DecodeFailed(s.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// FormatImporter trait
// ---------------------------------------------------------------------------

/// Trait for texture format importers.
///
/// Implement this trait to add support for a new texture format. Importers
/// are registered with an [`ImporterRegistry`] and selected based on file
/// extension, MIME type, or magic bytes.
pub trait FormatImporter: Send + Sync {
    /// File extensions this importer handles (e.g., `["png", "PNG"]`).
    ///
    /// Extensions should be lowercase for consistent matching.
    fn extensions(&self) -> &[&str];

    /// MIME types this importer handles (e.g., `["image/png"]`).
    fn mime_types(&self) -> &[&str];

    /// Priority when multiple importers match (higher = preferred).
    ///
    /// Default is 100. Built-in importers use priorities in the range 50-150.
    fn priority(&self) -> u32 {
        100
    }

    /// Import texture data from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the data cannot be decoded.
    fn import(&self, data: &[u8]) -> Result<TextureData, ImportError>;

    /// Check if this importer can handle the given data.
    ///
    /// This should be a fast check based on magic bytes at the start of the
    /// data. Return `false` if the magic bytes don't match.
    fn can_import(&self, data: &[u8]) -> bool;

    /// Returns the format name for debugging/logging.
    fn format_name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// ImporterRegistry
// ---------------------------------------------------------------------------

/// Registry of texture format importers.
///
/// Maintains a collection of importers and provides methods to resolve the
/// appropriate importer for a given format or data.
pub struct ImporterRegistry {
    importers: Vec<Box<dyn FormatImporter>>,
}

impl ImporterRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            importers: Vec::new(),
        }
    }

    /// Create a registry with default importers registered.
    ///
    /// Includes: PNG, JPEG, BMP, TGA, KTX2, USD, FBX
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        // Basic image formats
        registry.register(Box::new(PngImporter));
        registry.register(Box::new(JpegImporter));
        registry.register(Box::new(BmpImporter));
        registry.register(Box::new(TgaImporter));
        // Advanced formats
        registry.register(Box::new(Ktx2Importer));
        registry.register(Box::new(UsdImporter));
        registry.register(Box::new(FbxImporter));
        registry
    }

    /// Register a new importer.
    ///
    /// Importers are stored and searched in registration order, but priority
    /// is used to resolve conflicts when multiple importers match.
    pub fn register(&mut self, importer: Box<dyn FormatImporter>) {
        self.importers.push(importer);
    }

    /// Find an importer by file extension.
    ///
    /// Returns the highest-priority importer that handles the given extension.
    pub fn resolve_by_extension(&self, ext: &str) -> Option<&dyn FormatImporter> {
        let ext_lower = ext.to_lowercase();
        self.importers
            .iter()
            .filter(|imp| {
                imp.extensions()
                    .iter()
                    .any(|e| e.to_lowercase() == ext_lower)
            })
            .max_by_key(|imp| imp.priority())
            .map(|imp| imp.as_ref())
    }

    /// Find an importer by MIME type.
    ///
    /// Returns the highest-priority importer that handles the given MIME type.
    pub fn resolve_by_mime(&self, mime: &str) -> Option<&dyn FormatImporter> {
        let mime_lower = mime.to_lowercase();
        self.importers
            .iter()
            .filter(|imp| {
                imp.mime_types()
                    .iter()
                    .any(|m| m.to_lowercase() == mime_lower)
            })
            .max_by_key(|imp| imp.priority())
            .map(|imp| imp.as_ref())
    }

    /// Find an importer by examining magic bytes in the data.
    ///
    /// Returns the highest-priority importer whose `can_import` returns true.
    pub fn resolve_by_magic(&self, data: &[u8]) -> Option<&dyn FormatImporter> {
        self.importers
            .iter()
            .filter(|imp| imp.can_import(data))
            .max_by_key(|imp| imp.priority())
            .map(|imp| imp.as_ref())
    }

    /// Import texture data, using a hint to help resolve the format.
    ///
    /// The hint can be a file extension (e.g., "png") or MIME type
    /// (e.g., "image/png"). If no hint is provided or the hint doesn't
    /// match, falls back to magic byte detection.
    ///
    /// # Errors
    ///
    /// Returns an error if no importer can be found or if decoding fails.
    pub fn import(&self, data: &[u8], hint: Option<&str>) -> Result<TextureData, ImportError> {
        // Try hint-based resolution first
        let importer = if let Some(hint) = hint {
            self.resolve_by_extension(hint)
                .or_else(|| self.resolve_by_mime(hint))
        } else {
            None
        };

        // Fall back to magic byte detection
        let importer = importer.or_else(|| self.resolve_by_magic(data));

        match importer {
            Some(imp) => imp.import(data),
            None => Err(ImportError::UnsupportedFormat(
                hint.unwrap_or("unknown").to_string(),
            )),
        }
    }

    /// Returns the number of registered importers.
    pub fn len(&self) -> usize {
        self.importers.len()
    }

    /// Returns true if no importers are registered.
    pub fn is_empty(&self) -> bool {
        self.importers.is_empty()
    }

    /// Returns an iterator over all registered importers.
    pub fn iter(&self) -> impl Iterator<Item = &dyn FormatImporter> {
        self.importers.iter().map(|imp| imp.as_ref())
    }
}

impl Default for ImporterRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl fmt::Debug for ImporterRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImporterRegistry")
            .field("importers", &self.importers.len())
            .finish()
    }
}

#[cfg(test)]
mod tests;
