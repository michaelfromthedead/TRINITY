//! KTX (v1) and DDS Container Format Parsers (T-AS-2.7)
//!
//! Implements parsing for pre-compressed texture containers:
//! - KTX v1: Khronos Texture format with GL format codes
//! - DDS: DirectDraw Surface with DX10 header extension
//!
//! # Features
//!
//! - Parse KTX v1 headers, mip levels, cubemaps, and texture arrays
//! - Parse DDS headers including DX10 extension for modern formats
//! - Detect BCn (BC1-BC7) and ASTC compressed formats from container metadata
//! - Pass compressed data directly to GPU (no decode/re-encode)
//! - Integration with @asset decorator via extension mapping (.ktx, .dds)
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::compressed_formats::{KtxParser, DdsParser, CompressedFormat};
//!
//! // Parse KTX v1 file
//! let ktx_data = std::fs::read("texture.ktx")?;
//! let ktx = KtxParser::parse(&ktx_data)?;
//! assert!(ktx.format.is_compressed());
//!
//! // Parse DDS file with DX10 extension
//! let dds_data = std::fs::read("texture.dds")?;
//! let dds = DdsParser::parse(&dds_data)?;
//! println!("Format: {:?}, {} mip levels", dds.format, dds.mip_count);
//! ```

use std::fmt;

use super::texture_importer::{TextureImportError, TextureState};

// ---------------------------------------------------------------------------
// Compressed GPU Formats
// ---------------------------------------------------------------------------

/// Block-compressed texture formats for GPU upload.
///
/// These formats are passed directly to the GPU without decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressedFormat {
    // BCn formats (Desktop / Vulkan / Metal)
    /// BC1 RGB (DXT1) - 4 bits/pixel, no alpha
    Bc1Rgb,
    /// BC1 RGBA (DXT1) - 4 bits/pixel, 1-bit alpha
    Bc1Rgba,
    /// BC1 sRGB
    Bc1RgbSrgb,
    /// BC1 sRGB with alpha
    Bc1RgbaSrgb,
    /// BC2 (DXT3) - 8 bits/pixel, explicit alpha
    Bc2,
    /// BC2 sRGB
    Bc2Srgb,
    /// BC3 (DXT5) - 8 bits/pixel, interpolated alpha
    Bc3,
    /// BC3 sRGB
    Bc3Srgb,
    /// BC4 - single channel (grayscale)
    Bc4Unorm,
    /// BC4 signed
    Bc4Snorm,
    /// BC5 - two channels (normal maps)
    Bc5Unorm,
    /// BC5 signed
    Bc5Snorm,
    /// BC6H unsigned float (HDR)
    Bc6hUfloat,
    /// BC6H signed float (HDR)
    Bc6hSfloat,
    /// BC7 - high quality RGBA
    Bc7Unorm,
    /// BC7 sRGB
    Bc7Srgb,

    // ASTC formats (Mobile / Vulkan)
    /// ASTC 4x4 block
    Astc4x4,
    /// ASTC 4x4 sRGB
    Astc4x4Srgb,
    /// ASTC 5x4 block
    Astc5x4,
    /// ASTC 5x4 sRGB
    Astc5x4Srgb,
    /// ASTC 5x5 block
    Astc5x5,
    /// ASTC 5x5 sRGB
    Astc5x5Srgb,
    /// ASTC 6x5 block
    Astc6x5,
    /// ASTC 6x5 sRGB
    Astc6x5Srgb,
    /// ASTC 6x6 block
    Astc6x6,
    /// ASTC 6x6 sRGB
    Astc6x6Srgb,
    /// ASTC 8x5 block
    Astc8x5,
    /// ASTC 8x5 sRGB
    Astc8x5Srgb,
    /// ASTC 8x6 block
    Astc8x6,
    /// ASTC 8x6 sRGB
    Astc8x6Srgb,
    /// ASTC 8x8 block
    Astc8x8,
    /// ASTC 8x8 sRGB
    Astc8x8Srgb,
    /// ASTC 10x5 block
    Astc10x5,
    /// ASTC 10x5 sRGB
    Astc10x5Srgb,
    /// ASTC 10x6 block
    Astc10x6,
    /// ASTC 10x6 sRGB
    Astc10x6Srgb,
    /// ASTC 10x8 block
    Astc10x8,
    /// ASTC 10x8 sRGB
    Astc10x8Srgb,
    /// ASTC 10x10 block
    Astc10x10,
    /// ASTC 10x10 sRGB
    Astc10x10Srgb,
    /// ASTC 12x10 block
    Astc12x10,
    /// ASTC 12x10 sRGB
    Astc12x10Srgb,
    /// ASTC 12x12 block
    Astc12x12,
    /// ASTC 12x12 sRGB
    Astc12x12Srgb,

    // ETC formats (OpenGL ES)
    /// ETC1 RGB
    Etc1Rgb,
    /// ETC2 RGB
    Etc2Rgb,
    /// ETC2 RGB sRGB
    Etc2RgbSrgb,
    /// ETC2 RGBA (punchthrough alpha)
    Etc2RgbA1,
    /// ETC2 RGBA sRGB (punchthrough alpha)
    Etc2RgbA1Srgb,
    /// ETC2 RGBA
    Etc2Rgba,
    /// ETC2 RGBA sRGB
    Etc2RgbaSrgb,
    /// EAC R11
    EacR11Unorm,
    /// EAC R11 signed
    EacR11Snorm,
    /// EAC RG11
    EacRg11Unorm,
    /// EAC RG11 signed
    EacRg11Snorm,
}

impl CompressedFormat {
    /// Returns the block dimensions (width, height) for this format.
    #[inline]
    pub const fn block_dimensions(&self) -> (u32, u32) {
        match self {
            // BCn formats all use 4x4 blocks
            CompressedFormat::Bc1Rgb
            | CompressedFormat::Bc1Rgba
            | CompressedFormat::Bc1RgbSrgb
            | CompressedFormat::Bc1RgbaSrgb
            | CompressedFormat::Bc2
            | CompressedFormat::Bc2Srgb
            | CompressedFormat::Bc3
            | CompressedFormat::Bc3Srgb
            | CompressedFormat::Bc4Unorm
            | CompressedFormat::Bc4Snorm
            | CompressedFormat::Bc5Unorm
            | CompressedFormat::Bc5Snorm
            | CompressedFormat::Bc6hUfloat
            | CompressedFormat::Bc6hSfloat
            | CompressedFormat::Bc7Unorm
            | CompressedFormat::Bc7Srgb => (4, 4),

            // ASTC variable block sizes
            CompressedFormat::Astc4x4 | CompressedFormat::Astc4x4Srgb => (4, 4),
            CompressedFormat::Astc5x4 | CompressedFormat::Astc5x4Srgb => (5, 4),
            CompressedFormat::Astc5x5 | CompressedFormat::Astc5x5Srgb => (5, 5),
            CompressedFormat::Astc6x5 | CompressedFormat::Astc6x5Srgb => (6, 5),
            CompressedFormat::Astc6x6 | CompressedFormat::Astc6x6Srgb => (6, 6),
            CompressedFormat::Astc8x5 | CompressedFormat::Astc8x5Srgb => (8, 5),
            CompressedFormat::Astc8x6 | CompressedFormat::Astc8x6Srgb => (8, 6),
            CompressedFormat::Astc8x8 | CompressedFormat::Astc8x8Srgb => (8, 8),
            CompressedFormat::Astc10x5 | CompressedFormat::Astc10x5Srgb => (10, 5),
            CompressedFormat::Astc10x6 | CompressedFormat::Astc10x6Srgb => (10, 6),
            CompressedFormat::Astc10x8 | CompressedFormat::Astc10x8Srgb => (10, 8),
            CompressedFormat::Astc10x10 | CompressedFormat::Astc10x10Srgb => (10, 10),
            CompressedFormat::Astc12x10 | CompressedFormat::Astc12x10Srgb => (12, 10),
            CompressedFormat::Astc12x12 | CompressedFormat::Astc12x12Srgb => (12, 12),

            // ETC formats use 4x4 blocks
            CompressedFormat::Etc1Rgb
            | CompressedFormat::Etc2Rgb
            | CompressedFormat::Etc2RgbSrgb
            | CompressedFormat::Etc2RgbA1
            | CompressedFormat::Etc2RgbA1Srgb
            | CompressedFormat::Etc2Rgba
            | CompressedFormat::Etc2RgbaSrgb
            | CompressedFormat::EacR11Unorm
            | CompressedFormat::EacR11Snorm
            | CompressedFormat::EacRg11Unorm
            | CompressedFormat::EacRg11Snorm => (4, 4),
        }
    }

    /// Returns bytes per block for this format.
    #[inline]
    pub const fn bytes_per_block(&self) -> usize {
        match self {
            // BC1 and BC4: 8 bytes per 4x4 block
            CompressedFormat::Bc1Rgb
            | CompressedFormat::Bc1Rgba
            | CompressedFormat::Bc1RgbSrgb
            | CompressedFormat::Bc1RgbaSrgb
            | CompressedFormat::Bc4Unorm
            | CompressedFormat::Bc4Snorm => 8,

            // BC2, BC3, BC5, BC6H, BC7: 16 bytes per 4x4 block
            CompressedFormat::Bc2
            | CompressedFormat::Bc2Srgb
            | CompressedFormat::Bc3
            | CompressedFormat::Bc3Srgb
            | CompressedFormat::Bc5Unorm
            | CompressedFormat::Bc5Snorm
            | CompressedFormat::Bc6hUfloat
            | CompressedFormat::Bc6hSfloat
            | CompressedFormat::Bc7Unorm
            | CompressedFormat::Bc7Srgb => 16,

            // ASTC: always 16 bytes per block (variable block size)
            CompressedFormat::Astc4x4
            | CompressedFormat::Astc4x4Srgb
            | CompressedFormat::Astc5x4
            | CompressedFormat::Astc5x4Srgb
            | CompressedFormat::Astc5x5
            | CompressedFormat::Astc5x5Srgb
            | CompressedFormat::Astc6x5
            | CompressedFormat::Astc6x5Srgb
            | CompressedFormat::Astc6x6
            | CompressedFormat::Astc6x6Srgb
            | CompressedFormat::Astc8x5
            | CompressedFormat::Astc8x5Srgb
            | CompressedFormat::Astc8x6
            | CompressedFormat::Astc8x6Srgb
            | CompressedFormat::Astc8x8
            | CompressedFormat::Astc8x8Srgb
            | CompressedFormat::Astc10x5
            | CompressedFormat::Astc10x5Srgb
            | CompressedFormat::Astc10x6
            | CompressedFormat::Astc10x6Srgb
            | CompressedFormat::Astc10x8
            | CompressedFormat::Astc10x8Srgb
            | CompressedFormat::Astc10x10
            | CompressedFormat::Astc10x10Srgb
            | CompressedFormat::Astc12x10
            | CompressedFormat::Astc12x10Srgb
            | CompressedFormat::Astc12x12
            | CompressedFormat::Astc12x12Srgb => 16,

            // ETC1/ETC2 RGB: 8 bytes per 4x4 block
            CompressedFormat::Etc1Rgb
            | CompressedFormat::Etc2Rgb
            | CompressedFormat::Etc2RgbSrgb
            | CompressedFormat::Etc2RgbA1
            | CompressedFormat::Etc2RgbA1Srgb
            | CompressedFormat::EacR11Unorm
            | CompressedFormat::EacR11Snorm => 8,

            // ETC2 RGBA, EAC RG11: 16 bytes per 4x4 block
            CompressedFormat::Etc2Rgba
            | CompressedFormat::Etc2RgbaSrgb
            | CompressedFormat::EacRg11Unorm
            | CompressedFormat::EacRg11Snorm => 16,
        }
    }

    /// Calculate total data size for a texture with given dimensions.
    #[inline]
    pub fn calculate_size(&self, width: u32, height: u32) -> usize {
        let (block_w, block_h) = self.block_dimensions();
        let blocks_x = (width + block_w - 1) / block_w;
        let blocks_y = (height + block_h - 1) / block_h;
        (blocks_x as usize) * (blocks_y as usize) * self.bytes_per_block()
    }

    /// Returns true if this is an sRGB format.
    #[inline]
    pub const fn is_srgb(&self) -> bool {
        matches!(
            self,
            CompressedFormat::Bc1RgbSrgb
                | CompressedFormat::Bc1RgbaSrgb
                | CompressedFormat::Bc2Srgb
                | CompressedFormat::Bc3Srgb
                | CompressedFormat::Bc7Srgb
                | CompressedFormat::Astc4x4Srgb
                | CompressedFormat::Astc5x4Srgb
                | CompressedFormat::Astc5x5Srgb
                | CompressedFormat::Astc6x5Srgb
                | CompressedFormat::Astc6x6Srgb
                | CompressedFormat::Astc8x5Srgb
                | CompressedFormat::Astc8x6Srgb
                | CompressedFormat::Astc8x8Srgb
                | CompressedFormat::Astc10x5Srgb
                | CompressedFormat::Astc10x6Srgb
                | CompressedFormat::Astc10x8Srgb
                | CompressedFormat::Astc10x10Srgb
                | CompressedFormat::Astc12x10Srgb
                | CompressedFormat::Astc12x12Srgb
                | CompressedFormat::Etc2RgbSrgb
                | CompressedFormat::Etc2RgbA1Srgb
                | CompressedFormat::Etc2RgbaSrgb
        )
    }

    /// Returns true if this is an HDR format.
    #[inline]
    pub const fn is_hdr(&self) -> bool {
        matches!(
            self,
            CompressedFormat::Bc6hUfloat | CompressedFormat::Bc6hSfloat
        )
    }

    /// Returns true if this is a BCn format (desktop GPU).
    #[inline]
    pub const fn is_bcn(&self) -> bool {
        matches!(
            self,
            CompressedFormat::Bc1Rgb
                | CompressedFormat::Bc1Rgba
                | CompressedFormat::Bc1RgbSrgb
                | CompressedFormat::Bc1RgbaSrgb
                | CompressedFormat::Bc2
                | CompressedFormat::Bc2Srgb
                | CompressedFormat::Bc3
                | CompressedFormat::Bc3Srgb
                | CompressedFormat::Bc4Unorm
                | CompressedFormat::Bc4Snorm
                | CompressedFormat::Bc5Unorm
                | CompressedFormat::Bc5Snorm
                | CompressedFormat::Bc6hUfloat
                | CompressedFormat::Bc6hSfloat
                | CompressedFormat::Bc7Unorm
                | CompressedFormat::Bc7Srgb
        )
    }

    /// Returns true if this is an ASTC format (mobile GPU).
    #[inline]
    pub const fn is_astc(&self) -> bool {
        matches!(
            self,
            CompressedFormat::Astc4x4
                | CompressedFormat::Astc4x4Srgb
                | CompressedFormat::Astc5x4
                | CompressedFormat::Astc5x4Srgb
                | CompressedFormat::Astc5x5
                | CompressedFormat::Astc5x5Srgb
                | CompressedFormat::Astc6x5
                | CompressedFormat::Astc6x5Srgb
                | CompressedFormat::Astc6x6
                | CompressedFormat::Astc6x6Srgb
                | CompressedFormat::Astc8x5
                | CompressedFormat::Astc8x5Srgb
                | CompressedFormat::Astc8x6
                | CompressedFormat::Astc8x6Srgb
                | CompressedFormat::Astc8x8
                | CompressedFormat::Astc8x8Srgb
                | CompressedFormat::Astc10x5
                | CompressedFormat::Astc10x5Srgb
                | CompressedFormat::Astc10x6
                | CompressedFormat::Astc10x6Srgb
                | CompressedFormat::Astc10x8
                | CompressedFormat::Astc10x8Srgb
                | CompressedFormat::Astc10x10
                | CompressedFormat::Astc10x10Srgb
                | CompressedFormat::Astc12x10
                | CompressedFormat::Astc12x10Srgb
                | CompressedFormat::Astc12x12
                | CompressedFormat::Astc12x12Srgb
        )
    }

    /// Map to wgpu TextureFormat string representation.
    pub fn to_wgpu_format_str(&self) -> &'static str {
        match self {
            CompressedFormat::Bc1Rgb | CompressedFormat::Bc1Rgba => "Bc1RgbaUnorm",
            CompressedFormat::Bc1RgbSrgb | CompressedFormat::Bc1RgbaSrgb => "Bc1RgbaUnormSrgb",
            CompressedFormat::Bc2 => "Bc2RgbaUnorm",
            CompressedFormat::Bc2Srgb => "Bc2RgbaUnormSrgb",
            CompressedFormat::Bc3 => "Bc3RgbaUnorm",
            CompressedFormat::Bc3Srgb => "Bc3RgbaUnormSrgb",
            CompressedFormat::Bc4Unorm => "Bc4RUnorm",
            CompressedFormat::Bc4Snorm => "Bc4RSnorm",
            CompressedFormat::Bc5Unorm => "Bc5RgUnorm",
            CompressedFormat::Bc5Snorm => "Bc5RgSnorm",
            CompressedFormat::Bc6hUfloat => "Bc6hRgbUfloat",
            CompressedFormat::Bc6hSfloat => "Bc6hRgbFloat",
            CompressedFormat::Bc7Unorm => "Bc7RgbaUnorm",
            CompressedFormat::Bc7Srgb => "Bc7RgbaUnormSrgb",
            CompressedFormat::Astc4x4 => "Astc4x4Unorm",
            CompressedFormat::Astc4x4Srgb => "Astc4x4UnormSrgb",
            CompressedFormat::Astc5x4 => "Astc5x4Unorm",
            CompressedFormat::Astc5x4Srgb => "Astc5x4UnormSrgb",
            CompressedFormat::Astc5x5 => "Astc5x5Unorm",
            CompressedFormat::Astc5x5Srgb => "Astc5x5UnormSrgb",
            CompressedFormat::Astc6x5 => "Astc6x5Unorm",
            CompressedFormat::Astc6x5Srgb => "Astc6x5UnormSrgb",
            CompressedFormat::Astc6x6 => "Astc6x6Unorm",
            CompressedFormat::Astc6x6Srgb => "Astc6x6UnormSrgb",
            CompressedFormat::Astc8x5 => "Astc8x5Unorm",
            CompressedFormat::Astc8x5Srgb => "Astc8x5UnormSrgb",
            CompressedFormat::Astc8x6 => "Astc8x6Unorm",
            CompressedFormat::Astc8x6Srgb => "Astc8x6UnormSrgb",
            CompressedFormat::Astc8x8 => "Astc8x8Unorm",
            CompressedFormat::Astc8x8Srgb => "Astc8x8UnormSrgb",
            CompressedFormat::Astc10x5 => "Astc10x5Unorm",
            CompressedFormat::Astc10x5Srgb => "Astc10x5UnormSrgb",
            CompressedFormat::Astc10x6 => "Astc10x6Unorm",
            CompressedFormat::Astc10x6Srgb => "Astc10x6UnormSrgb",
            CompressedFormat::Astc10x8 => "Astc10x8Unorm",
            CompressedFormat::Astc10x8Srgb => "Astc10x8UnormSrgb",
            CompressedFormat::Astc10x10 => "Astc10x10Unorm",
            CompressedFormat::Astc10x10Srgb => "Astc10x10UnormSrgb",
            CompressedFormat::Astc12x10 => "Astc12x10Unorm",
            CompressedFormat::Astc12x10Srgb => "Astc12x10UnormSrgb",
            CompressedFormat::Astc12x12 => "Astc12x12Unorm",
            CompressedFormat::Astc12x12Srgb => "Astc12x12UnormSrgb",
            CompressedFormat::Etc1Rgb => "Etc2Rgb8Unorm",
            CompressedFormat::Etc2Rgb => "Etc2Rgb8Unorm",
            CompressedFormat::Etc2RgbSrgb => "Etc2Rgb8UnormSrgb",
            CompressedFormat::Etc2RgbA1 => "Etc2Rgb8A1Unorm",
            CompressedFormat::Etc2RgbA1Srgb => "Etc2Rgb8A1UnormSrgb",
            CompressedFormat::Etc2Rgba => "Etc2Rgba8Unorm",
            CompressedFormat::Etc2RgbaSrgb => "Etc2Rgba8UnormSrgb",
            CompressedFormat::EacR11Unorm => "EacR11Unorm",
            CompressedFormat::EacR11Snorm => "EacR11Snorm",
            CompressedFormat::EacRg11Unorm => "EacRg11Unorm",
            CompressedFormat::EacRg11Snorm => "EacRg11Snorm",
        }
    }
}

impl fmt::Display for CompressedFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_wgpu_format_str())
    }
}

// ---------------------------------------------------------------------------
// Mip Level Data
// ---------------------------------------------------------------------------

/// Data for a single mip level.
#[derive(Debug, Clone)]
pub struct MipLevelData {
    /// Mip level index (0 = base)
    pub level: u32,
    /// Width at this mip level
    pub width: u32,
    /// Height at this mip level
    pub height: u32,
    /// Byte offset in the source data
    pub offset: usize,
    /// Byte length of this mip level
    pub size: usize,
}

impl MipLevelData {
    /// Create a new mip level descriptor.
    pub fn new(level: u32, width: u32, height: u32, offset: usize, size: usize) -> Self {
        Self {
            level,
            width,
            height,
            offset,
            size,
        }
    }
}

// ---------------------------------------------------------------------------
// Face Data (for cubemaps)
// ---------------------------------------------------------------------------

/// Data for a single cubemap face or array layer.
#[derive(Debug, Clone)]
pub struct FaceData {
    /// Face index (0-5 for cubemap, 0-N for array)
    pub face_index: u32,
    /// Mip levels for this face
    pub mip_levels: Vec<MipLevelData>,
}

impl FaceData {
    /// Create new face data.
    pub fn new(face_index: u32) -> Self {
        Self {
            face_index,
            mip_levels: Vec::new(),
        }
    }

    /// Add a mip level to this face.
    pub fn add_mip(&mut self, level: MipLevelData) {
        self.mip_levels.push(level);
    }
}

// ---------------------------------------------------------------------------
// Texture Type
// ---------------------------------------------------------------------------

/// Type of texture in the container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureType {
    /// Standard 2D texture
    Texture2D,
    /// Cubemap (6 faces)
    Cubemap,
    /// 2D texture array
    Array2D,
    /// Cubemap array
    CubemapArray,
    /// 3D volume texture
    Volume,
}

impl fmt::Display for TextureType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextureType::Texture2D => write!(f, "2D"),
            TextureType::Cubemap => write!(f, "Cubemap"),
            TextureType::Array2D => write!(f, "2D Array"),
            TextureType::CubemapArray => write!(f, "Cubemap Array"),
            TextureType::Volume => write!(f, "Volume"),
        }
    }
}

// ===========================================================================
// KTX v1 Parser
// ===========================================================================

/// KTX v1 file identifier (12 bytes).
const KTX1_IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, // 'KTX '
    0x20, 0x31, 0x31, 0xBB, // ' 11'
    0x0D, 0x0A, 0x1A, 0x0A, // '\r\n\x1a\n'
];

/// KTX v1 header (64 bytes total).
#[derive(Debug, Clone)]
pub struct KtxHeader {
    /// Endianness marker (0x04030201 = little-endian)
    pub endianness: u32,
    /// OpenGL type (e.g., GL_UNSIGNED_BYTE)
    pub gl_type: u32,
    /// OpenGL type size (e.g., 1 for GL_UNSIGNED_BYTE)
    pub gl_type_size: u32,
    /// OpenGL format (e.g., GL_RGBA)
    pub gl_format: u32,
    /// OpenGL internal format (e.g., GL_RGBA8)
    pub gl_internal_format: u32,
    /// OpenGL base internal format
    pub gl_base_internal_format: u32,
    /// Texture width at base mip level
    pub pixel_width: u32,
    /// Texture height at base mip level
    pub pixel_height: u32,
    /// Texture depth (for 3D textures)
    pub pixel_depth: u32,
    /// Number of array elements (0 = not an array)
    pub number_of_array_elements: u32,
    /// Number of faces (1 = 2D, 6 = cubemap)
    pub number_of_faces: u32,
    /// Number of mipmap levels
    pub number_of_mipmap_levels: u32,
    /// Byte length of key/value data
    pub bytes_of_key_value_data: u32,
}

/// Parsed KTX v1 texture.
#[derive(Debug, Clone)]
pub struct KtxTexture {
    /// Parsed header
    pub header: KtxHeader,
    /// Detected compressed format (None if uncompressed)
    pub format: Option<CompressedFormat>,
    /// Texture type
    pub texture_type: TextureType,
    /// Number of mip levels
    pub mip_count: u32,
    /// Number of array layers (1 for non-arrays)
    pub array_count: u32,
    /// Number of faces (1 for 2D, 6 for cubemap)
    pub face_count: u32,
    /// Width at base level
    pub width: u32,
    /// Height at base level
    pub height: u32,
    /// Depth at base level (for volume textures)
    pub depth: u32,
    /// Face/layer data with mip levels
    pub faces: Vec<FaceData>,
    /// Offset to first image data (after header + KV data)
    pub data_offset: usize,
    /// Total size of all image data
    pub data_size: usize,
    /// Is the file little-endian?
    pub is_little_endian: bool,
}

/// KTX v1 parser implementation.
pub struct KtxParser;

impl KtxParser {
    /// Check if data appears to be a KTX v1 file.
    #[inline]
    pub fn is_ktx(data: &[u8]) -> bool {
        data.len() >= 12 && data[0..12] == KTX1_IDENTIFIER
    }

    /// Parse a KTX v1 file from raw bytes.
    pub fn parse(data: &[u8]) -> Result<KtxTexture, TextureImportError> {
        if data.len() < 64 {
            return Err(TextureImportError::InvalidData(
                "KTX file too short for header".to_string(),
            ));
        }

        // Check identifier
        if data[0..12] != KTX1_IDENTIFIER {
            return Err(TextureImportError::InvalidData(
                "Invalid KTX v1 identifier".to_string(),
            ));
        }

        // Parse header
        let header = Self::parse_header(data)?;

        // Determine endianness
        let is_little_endian = header.endianness == 0x04030201;

        // Validate header
        Self::validate_header(&header)?;

        // Detect format
        let format = Self::detect_format(&header);

        // Determine texture type
        let texture_type = Self::determine_texture_type(&header);

        // Calculate data offset (header + key/value data)
        let data_offset = 64 + header.bytes_of_key_value_data as usize;

        // Parse mip level data
        let (faces, data_size) = Self::parse_mip_levels(data, &header, data_offset, format)?;

        let mip_count = if header.number_of_mipmap_levels == 0 {
            1
        } else {
            header.number_of_mipmap_levels
        };

        let array_count = if header.number_of_array_elements == 0 {
            1
        } else {
            header.number_of_array_elements
        };

        Ok(KtxTexture {
            header,
            format,
            texture_type,
            mip_count,
            array_count,
            face_count: if texture_type == TextureType::Cubemap
                || texture_type == TextureType::CubemapArray
            {
                6
            } else {
                1
            },
            width: 0, // Set below
            height: 0,
            depth: 0,
            faces,
            data_offset,
            data_size,
            is_little_endian,
        })
        .map(|mut tex| {
            tex.width = tex.header.pixel_width;
            tex.height = tex.header.pixel_height;
            tex.depth = if tex.header.pixel_depth == 0 {
                1
            } else {
                tex.header.pixel_depth
            };
            tex
        })
    }

    /// Parse the KTX v1 header.
    fn parse_header(data: &[u8]) -> Result<KtxHeader, TextureImportError> {
        let read_u32 = |offset: usize| -> u32 {
            u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        };

        let endianness = read_u32(12);
        let is_little_endian = endianness == 0x04030201;

        // Re-read with correct endianness
        let read_u32_endian = |offset: usize| -> u32 {
            let bytes = [
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ];
            if is_little_endian {
                u32::from_le_bytes(bytes)
            } else {
                u32::from_be_bytes(bytes)
            }
        };

        Ok(KtxHeader {
            endianness,
            gl_type: read_u32_endian(16),
            gl_type_size: read_u32_endian(20),
            gl_format: read_u32_endian(24),
            gl_internal_format: read_u32_endian(28),
            gl_base_internal_format: read_u32_endian(32),
            pixel_width: read_u32_endian(36),
            pixel_height: read_u32_endian(40),
            pixel_depth: read_u32_endian(44),
            number_of_array_elements: read_u32_endian(48),
            number_of_faces: read_u32_endian(52),
            number_of_mipmap_levels: read_u32_endian(56),
            bytes_of_key_value_data: read_u32_endian(60),
        })
    }

    /// Validate the KTX header.
    fn validate_header(header: &KtxHeader) -> Result<(), TextureImportError> {
        // Width must be non-zero
        if header.pixel_width == 0 {
            return Err(TextureImportError::InvalidDimensions {
                width: 0,
                height: header.pixel_height,
            });
        }

        // Faces must be 1 or 6
        if header.number_of_faces != 1 && header.number_of_faces != 6 {
            return Err(TextureImportError::InvalidData(format!(
                "Invalid face count: {} (must be 1 or 6)",
                header.number_of_faces
            )));
        }

        // Cubemap must be square
        if header.number_of_faces == 6 && header.pixel_width != header.pixel_height {
            return Err(TextureImportError::InvalidData(
                "Cubemap faces must be square".to_string(),
            ));
        }

        // Validate endianness marker
        if header.endianness != 0x04030201 && header.endianness != 0x01020304 {
            return Err(TextureImportError::InvalidData(format!(
                "Invalid endianness marker: 0x{:08X}",
                header.endianness
            )));
        }

        Ok(())
    }

    /// Detect compressed format from GL internal format.
    fn detect_format(header: &KtxHeader) -> Option<CompressedFormat> {
        // GL compressed format constants
        const GL_COMPRESSED_RGB_S3TC_DXT1_EXT: u32 = 0x83F0;
        const GL_COMPRESSED_RGBA_S3TC_DXT1_EXT: u32 = 0x83F1;
        const GL_COMPRESSED_RGBA_S3TC_DXT3_EXT: u32 = 0x83F2;
        const GL_COMPRESSED_RGBA_S3TC_DXT5_EXT: u32 = 0x83F3;
        const GL_COMPRESSED_SRGB_S3TC_DXT1_EXT: u32 = 0x8C4C;
        const GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT: u32 = 0x8C4D;
        const GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT: u32 = 0x8C4E;
        const GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT: u32 = 0x8C4F;

        const GL_COMPRESSED_RED_RGTC1: u32 = 0x8DBB;
        const GL_COMPRESSED_SIGNED_RED_RGTC1: u32 = 0x8DBC;
        const GL_COMPRESSED_RG_RGTC2: u32 = 0x8DBD;
        const GL_COMPRESSED_SIGNED_RG_RGTC2: u32 = 0x8DBE;

        const GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT: u32 = 0x8E8F;
        const GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT: u32 = 0x8E8E;
        const GL_COMPRESSED_RGBA_BPTC_UNORM: u32 = 0x8E8C;
        const GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM: u32 = 0x8E8D;

        // ASTC format bases (linear = 0x93B0, sRGB = 0x93D0)
        const ASTC_LINEAR_BASE: u32 = 0x93B0;
        const ASTC_SRGB_BASE: u32 = 0x93D0;

        // ETC2/EAC formats
        const GL_COMPRESSED_RGB8_ETC2: u32 = 0x9274;
        const GL_COMPRESSED_SRGB8_ETC2: u32 = 0x9275;
        const GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2: u32 = 0x9276;
        const GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2: u32 = 0x9277;
        const GL_COMPRESSED_RGBA8_ETC2_EAC: u32 = 0x9278;
        const GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC: u32 = 0x9279;
        const GL_COMPRESSED_R11_EAC: u32 = 0x9270;
        const GL_COMPRESSED_SIGNED_R11_EAC: u32 = 0x9271;
        const GL_COMPRESSED_RG11_EAC: u32 = 0x9272;
        const GL_COMPRESSED_SIGNED_RG11_EAC: u32 = 0x9273;

        match header.gl_internal_format {
            // BC1 (DXT1)
            GL_COMPRESSED_RGB_S3TC_DXT1_EXT => Some(CompressedFormat::Bc1Rgb),
            GL_COMPRESSED_RGBA_S3TC_DXT1_EXT => Some(CompressedFormat::Bc1Rgba),
            GL_COMPRESSED_SRGB_S3TC_DXT1_EXT => Some(CompressedFormat::Bc1RgbSrgb),
            GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT => Some(CompressedFormat::Bc1RgbaSrgb),

            // BC2 (DXT3)
            GL_COMPRESSED_RGBA_S3TC_DXT3_EXT => Some(CompressedFormat::Bc2),
            GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT => Some(CompressedFormat::Bc2Srgb),

            // BC3 (DXT5)
            GL_COMPRESSED_RGBA_S3TC_DXT5_EXT => Some(CompressedFormat::Bc3),
            GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT => Some(CompressedFormat::Bc3Srgb),

            // BC4 (RGTC1)
            GL_COMPRESSED_RED_RGTC1 => Some(CompressedFormat::Bc4Unorm),
            GL_COMPRESSED_SIGNED_RED_RGTC1 => Some(CompressedFormat::Bc4Snorm),

            // BC5 (RGTC2)
            GL_COMPRESSED_RG_RGTC2 => Some(CompressedFormat::Bc5Unorm),
            GL_COMPRESSED_SIGNED_RG_RGTC2 => Some(CompressedFormat::Bc5Snorm),

            // BC6H (BPTC float)
            GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT => Some(CompressedFormat::Bc6hUfloat),
            GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT => Some(CompressedFormat::Bc6hSfloat),

            // BC7 (BPTC)
            GL_COMPRESSED_RGBA_BPTC_UNORM => Some(CompressedFormat::Bc7Unorm),
            GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM => Some(CompressedFormat::Bc7Srgb),

            // ASTC (checking range for all block sizes)
            fmt if fmt >= ASTC_LINEAR_BASE && fmt <= 0x93BD => {
                Self::detect_astc_format(fmt, false)
            }
            fmt if fmt >= ASTC_SRGB_BASE && fmt <= 0x93DD => {
                Self::detect_astc_format(fmt, true)
            }

            // ETC2/EAC
            GL_COMPRESSED_RGB8_ETC2 => Some(CompressedFormat::Etc2Rgb),
            GL_COMPRESSED_SRGB8_ETC2 => Some(CompressedFormat::Etc2RgbSrgb),
            GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2 => Some(CompressedFormat::Etc2RgbA1),
            GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2 => Some(CompressedFormat::Etc2RgbA1Srgb),
            GL_COMPRESSED_RGBA8_ETC2_EAC => Some(CompressedFormat::Etc2Rgba),
            GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC => Some(CompressedFormat::Etc2RgbaSrgb),
            GL_COMPRESSED_R11_EAC => Some(CompressedFormat::EacR11Unorm),
            GL_COMPRESSED_SIGNED_R11_EAC => Some(CompressedFormat::EacR11Snorm),
            GL_COMPRESSED_RG11_EAC => Some(CompressedFormat::EacRg11Unorm),
            GL_COMPRESSED_SIGNED_RG11_EAC => Some(CompressedFormat::EacRg11Snorm),

            _ => None,
        }
    }

    /// Detect ASTC format from GL constant.
    fn detect_astc_format(gl_format: u32, is_srgb: bool) -> Option<CompressedFormat> {
        // ASTC formats are sequential: 4x4, 5x4, 5x5, 6x5, 6x6, 8x5, 8x6, 8x8, 10x5, 10x6, 10x8, 10x10, 12x10, 12x12
        let base = if is_srgb { 0x93D0 } else { 0x93B0 };
        let index = gl_format - base;

        match index {
            0 => Some(if is_srgb {
                CompressedFormat::Astc4x4Srgb
            } else {
                CompressedFormat::Astc4x4
            }),
            1 => Some(if is_srgb {
                CompressedFormat::Astc5x4Srgb
            } else {
                CompressedFormat::Astc5x4
            }),
            2 => Some(if is_srgb {
                CompressedFormat::Astc5x5Srgb
            } else {
                CompressedFormat::Astc5x5
            }),
            3 => Some(if is_srgb {
                CompressedFormat::Astc6x5Srgb
            } else {
                CompressedFormat::Astc6x5
            }),
            4 => Some(if is_srgb {
                CompressedFormat::Astc6x6Srgb
            } else {
                CompressedFormat::Astc6x6
            }),
            5 => Some(if is_srgb {
                CompressedFormat::Astc8x5Srgb
            } else {
                CompressedFormat::Astc8x5
            }),
            6 => Some(if is_srgb {
                CompressedFormat::Astc8x6Srgb
            } else {
                CompressedFormat::Astc8x6
            }),
            7 => Some(if is_srgb {
                CompressedFormat::Astc8x8Srgb
            } else {
                CompressedFormat::Astc8x8
            }),
            8 => Some(if is_srgb {
                CompressedFormat::Astc10x5Srgb
            } else {
                CompressedFormat::Astc10x5
            }),
            9 => Some(if is_srgb {
                CompressedFormat::Astc10x6Srgb
            } else {
                CompressedFormat::Astc10x6
            }),
            10 => Some(if is_srgb {
                CompressedFormat::Astc10x8Srgb
            } else {
                CompressedFormat::Astc10x8
            }),
            11 => Some(if is_srgb {
                CompressedFormat::Astc10x10Srgb
            } else {
                CompressedFormat::Astc10x10
            }),
            12 => Some(if is_srgb {
                CompressedFormat::Astc12x10Srgb
            } else {
                CompressedFormat::Astc12x10
            }),
            13 => Some(if is_srgb {
                CompressedFormat::Astc12x12Srgb
            } else {
                CompressedFormat::Astc12x12
            }),
            _ => None,
        }
    }

    /// Determine texture type from header.
    fn determine_texture_type(header: &KtxHeader) -> TextureType {
        let is_array = header.number_of_array_elements > 0;
        let is_cubemap = header.number_of_faces == 6;
        let is_3d = header.pixel_depth > 0;

        if is_3d {
            TextureType::Volume
        } else if is_cubemap && is_array {
            TextureType::CubemapArray
        } else if is_cubemap {
            TextureType::Cubemap
        } else if is_array {
            TextureType::Array2D
        } else {
            TextureType::Texture2D
        }
    }

    /// Parse mip level data from the file.
    fn parse_mip_levels(
        data: &[u8],
        header: &KtxHeader,
        data_offset: usize,
        format: Option<CompressedFormat>,
    ) -> Result<(Vec<FaceData>, usize), TextureImportError> {
        let is_little_endian = header.endianness == 0x04030201;
        let mip_count = if header.number_of_mipmap_levels == 0 {
            1
        } else {
            header.number_of_mipmap_levels
        };
        let face_count = header.number_of_faces;
        let array_count = if header.number_of_array_elements == 0 {
            1
        } else {
            header.number_of_array_elements
        };

        let total_faces = face_count * array_count;
        let mut faces: Vec<FaceData> = (0..total_faces).map(FaceData::new).collect();

        let mut offset = data_offset;
        let mut total_data_size = 0usize;

        for mip_level in 0..mip_count {
            // Read image size (4 bytes)
            if offset + 4 > data.len() {
                return Err(TextureImportError::InvalidData(
                    "KTX file truncated at mip level size".to_string(),
                ));
            }

            let image_size = if is_little_endian {
                u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
            } else {
                u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
            };
            offset += 4;

            // Calculate dimensions at this mip level
            let width = (header.pixel_width >> mip_level).max(1);
            let height = (header.pixel_height >> mip_level).max(1);

            // Calculate expected size per face
            let face_size = if let Some(fmt) = format {
                fmt.calculate_size(width, height)
            } else {
                // Uncompressed: use gl_type_size and pixel count
                let channels = match header.gl_base_internal_format {
                    0x1903 => 1, // GL_RED
                    0x8227 => 2, // GL_RG
                    0x1907 => 3, // GL_RGB
                    0x1908 => 4, // GL_RGBA
                    _ => 4,
                };
                (width as usize) * (height as usize) * channels * (header.gl_type_size as usize)
            };

            // For each face/layer
            for face_idx in 0..total_faces {
                let mip_data = MipLevelData::new(mip_level, width, height, offset, face_size);
                faces[face_idx as usize].add_mip(mip_data);
                offset += face_size;

                // Mip padding to 4-byte boundary
                let padding = (4 - (face_size % 4)) % 4;
                offset += padding;
            }

            total_data_size += image_size as usize;

            // Level padding (cubePadding)
            let level_padding = (4 - (offset % 4)) % 4;
            offset += level_padding;
        }

        Ok((faces, total_data_size))
    }

    /// Get compressed data for a specific mip level.
    ///
    /// Returns a slice to the compressed data that can be passed directly to GPU.
    pub fn get_mip_data<'a>(
        &self,
        data: &'a [u8],
        texture: &KtxTexture,
        face: u32,
        mip_level: u32,
    ) -> Option<&'a [u8]> {
        let face_data = texture.faces.get(face as usize)?;
        let mip_data = face_data.mip_levels.get(mip_level as usize)?;

        if mip_data.offset + mip_data.size <= data.len() {
            Some(&data[mip_data.offset..mip_data.offset + mip_data.size])
        } else {
            None
        }
    }
}

// ===========================================================================
// DDS Parser
// ===========================================================================

/// DDS magic number "DDS " (0x20534444).
const DDS_MAGIC: u32 = 0x20534444;

/// DDS pixel format flags.
#[derive(Debug, Clone, Copy)]
pub struct DdsPixelFormatFlags;

impl DdsPixelFormatFlags {
    pub const ALPHAPIXELS: u32 = 0x00000001;
    pub const ALPHA: u32 = 0x00000002;
    pub const FOURCC: u32 = 0x00000004;
    pub const RGB: u32 = 0x00000040;
    pub const YUV: u32 = 0x00000200;
    pub const LUMINANCE: u32 = 0x00020000;
}

/// DDS header flags.
#[derive(Debug, Clone, Copy)]
pub struct DdsHeaderFlags;

impl DdsHeaderFlags {
    pub const CAPS: u32 = 0x00000001;
    pub const HEIGHT: u32 = 0x00000002;
    pub const WIDTH: u32 = 0x00000004;
    pub const PITCH: u32 = 0x00000008;
    pub const PIXELFORMAT: u32 = 0x00001000;
    pub const MIPMAPCOUNT: u32 = 0x00020000;
    pub const LINEARSIZE: u32 = 0x00080000;
    pub const DEPTH: u32 = 0x00800000;
}

/// DDS caps flags.
#[derive(Debug, Clone, Copy)]
pub struct DdsCapsFlags;

impl DdsCapsFlags {
    pub const COMPLEX: u32 = 0x00000008;
    pub const MIPMAP: u32 = 0x00400000;
    pub const TEXTURE: u32 = 0x00001000;
}

/// DDS caps2 flags.
#[derive(Debug, Clone, Copy)]
pub struct DdsCaps2Flags;

impl DdsCaps2Flags {
    pub const CUBEMAP: u32 = 0x00000200;
    pub const CUBEMAP_POSITIVEX: u32 = 0x00000400;
    pub const CUBEMAP_NEGATIVEX: u32 = 0x00000800;
    pub const CUBEMAP_POSITIVEY: u32 = 0x00001000;
    pub const CUBEMAP_NEGATIVEY: u32 = 0x00002000;
    pub const CUBEMAP_POSITIVEZ: u32 = 0x00004000;
    pub const CUBEMAP_NEGATIVEZ: u32 = 0x00008000;
    pub const CUBEMAP_ALL_FACES: u32 = 0x0000FC00;
    pub const VOLUME: u32 = 0x00200000;
}

/// DDS pixel format structure (32 bytes).
#[derive(Debug, Clone)]
pub struct DdsPixelFormat {
    /// Size of structure (always 32)
    pub size: u32,
    /// Flags indicating valid fields
    pub flags: u32,
    /// FourCC code for compressed formats
    pub four_cc: [u8; 4],
    /// Number of bits per pixel
    pub rgb_bit_count: u32,
    /// Red channel mask
    pub r_bit_mask: u32,
    /// Green channel mask
    pub g_bit_mask: u32,
    /// Blue channel mask
    pub b_bit_mask: u32,
    /// Alpha channel mask
    pub a_bit_mask: u32,
}

impl DdsPixelFormat {
    /// Get FourCC as string.
    pub fn four_cc_str(&self) -> String {
        String::from_utf8_lossy(&self.four_cc).to_string()
    }

    /// Check if this is a compressed format.
    pub fn is_compressed(&self) -> bool {
        (self.flags & DdsPixelFormatFlags::FOURCC) != 0
    }
}

/// DDS header (124 bytes after magic).
#[derive(Debug, Clone)]
pub struct DdsHeader {
    /// Size of header (always 124)
    pub size: u32,
    /// Flags indicating valid fields
    pub flags: u32,
    /// Texture height
    pub height: u32,
    /// Texture width
    pub width: u32,
    /// Pitch or linear size
    pub pitch_or_linear_size: u32,
    /// Depth (for volume textures)
    pub depth: u32,
    /// Number of mipmap levels
    pub mip_map_count: u32,
    /// Reserved (11 DWORDs)
    pub reserved1: [u32; 11],
    /// Pixel format
    pub pixel_format: DdsPixelFormat,
    /// Capability flags
    pub caps: u32,
    /// Additional capability flags
    pub caps2: u32,
    /// Caps3 (unused)
    pub caps3: u32,
    /// Caps4 (unused)
    pub caps4: u32,
    /// Reserved2
    pub reserved2: u32,
}

/// DDS DX10 header extension (20 bytes).
#[derive(Debug, Clone)]
pub struct DdsDx10Header {
    /// DXGI format
    pub dxgi_format: u32,
    /// Resource dimension (1D, 2D, 3D)
    pub resource_dimension: u32,
    /// Misc flags (cubemap)
    pub misc_flag: u32,
    /// Array size
    pub array_size: u32,
    /// Misc flags 2 (alpha mode)
    pub misc_flags2: u32,
}

/// Parsed DDS texture.
#[derive(Debug, Clone)]
pub struct DdsTexture {
    /// Main header
    pub header: DdsHeader,
    /// DX10 extension header (if present)
    pub dx10_header: Option<DdsDx10Header>,
    /// Detected compressed format
    pub format: Option<CompressedFormat>,
    /// Texture type
    pub texture_type: TextureType,
    /// Number of mip levels
    pub mip_count: u32,
    /// Number of array layers
    pub array_count: u32,
    /// Number of faces (1 for 2D, 6 for cubemap)
    pub face_count: u32,
    /// Width at base level
    pub width: u32,
    /// Height at base level
    pub height: u32,
    /// Depth at base level
    pub depth: u32,
    /// Face/layer data with mip levels
    pub faces: Vec<FaceData>,
    /// Offset to first image data
    pub data_offset: usize,
    /// Total data size
    pub data_size: usize,
}

/// DDS parser implementation.
pub struct DdsParser;

impl DdsParser {
    /// Check if data appears to be a DDS file.
    #[inline]
    pub fn is_dds(data: &[u8]) -> bool {
        if data.len() < 4 {
            return false;
        }
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        magic == DDS_MAGIC
    }

    /// Parse a DDS file from raw bytes.
    pub fn parse(data: &[u8]) -> Result<DdsTexture, TextureImportError> {
        if data.len() < 128 {
            return Err(TextureImportError::InvalidData(
                "DDS file too short for header".to_string(),
            ));
        }

        // Check magic
        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if magic != DDS_MAGIC {
            return Err(TextureImportError::InvalidData(
                "Invalid DDS magic number".to_string(),
            ));
        }

        // Parse main header
        let header = Self::parse_header(&data[4..128])?;

        // Check for DX10 extension
        let (dx10_header, data_offset) =
            if header.pixel_format.four_cc == *b"DX10" {
                if data.len() < 148 {
                    return Err(TextureImportError::InvalidData(
                        "DDS file too short for DX10 header".to_string(),
                    ));
                }
                let dx10 = Self::parse_dx10_header(&data[128..148])?;
                (Some(dx10), 148)
            } else {
                (None, 128)
            };

        // Detect format
        let format = Self::detect_format(&header, dx10_header.as_ref());

        // Determine texture type
        let texture_type = Self::determine_texture_type(&header, dx10_header.as_ref());

        // Calculate face/array counts
        let (face_count, array_count) = Self::calculate_counts(&header, dx10_header.as_ref());

        // Calculate mip count
        let mip_count = if (header.flags & DdsHeaderFlags::MIPMAPCOUNT) != 0 {
            header.mip_map_count.max(1)
        } else {
            1
        };

        // Parse mip level data
        let depth = if (header.flags & DdsHeaderFlags::DEPTH) != 0 {
            header.depth.max(1)
        } else {
            1
        };

        let (faces, data_size) = Self::parse_mip_levels(
            data,
            &header,
            format,
            data_offset,
            mip_count,
            face_count,
            array_count,
            depth,
        )?;

        Ok(DdsTexture {
            header,
            dx10_header,
            format,
            texture_type,
            mip_count,
            array_count,
            face_count,
            width: 0,
            height: 0,
            depth,
            faces,
            data_offset,
            data_size,
        })
        .map(|mut tex| {
            tex.width = tex.header.width;
            tex.height = tex.header.height;
            tex
        })
    }

    /// Parse the main DDS header.
    fn parse_header(data: &[u8]) -> Result<DdsHeader, TextureImportError> {
        let read_u32 = |offset: usize| -> u32 {
            u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        };

        let size = read_u32(0);
        if size != 124 {
            return Err(TextureImportError::InvalidData(format!(
                "Invalid DDS header size: {} (expected 124)",
                size
            )));
        }

        // Parse pixel format (at offset 72)
        let pf_offset = 72;
        let pf_size = read_u32(pf_offset);
        if pf_size != 32 {
            return Err(TextureImportError::InvalidData(format!(
                "Invalid DDS pixel format size: {} (expected 32)",
                pf_size
            )));
        }

        let pixel_format = DdsPixelFormat {
            size: pf_size,
            flags: read_u32(pf_offset + 4),
            four_cc: [
                data[pf_offset + 8],
                data[pf_offset + 9],
                data[pf_offset + 10],
                data[pf_offset + 11],
            ],
            rgb_bit_count: read_u32(pf_offset + 12),
            r_bit_mask: read_u32(pf_offset + 16),
            g_bit_mask: read_u32(pf_offset + 20),
            b_bit_mask: read_u32(pf_offset + 24),
            a_bit_mask: read_u32(pf_offset + 28),
        };

        // Parse reserved1 array
        let mut reserved1 = [0u32; 11];
        for i in 0..11 {
            reserved1[i] = read_u32(28 + i * 4);
        }

        Ok(DdsHeader {
            size,
            flags: read_u32(4),
            height: read_u32(8),
            width: read_u32(12),
            pitch_or_linear_size: read_u32(16),
            depth: read_u32(20),
            mip_map_count: read_u32(24),
            reserved1,
            pixel_format,
            caps: read_u32(104),
            caps2: read_u32(108),
            caps3: read_u32(112),
            caps4: read_u32(116),
            reserved2: read_u32(120),
        })
    }

    /// Parse the DX10 extension header.
    fn parse_dx10_header(data: &[u8]) -> Result<DdsDx10Header, TextureImportError> {
        let read_u32 = |offset: usize| -> u32 {
            u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        };

        Ok(DdsDx10Header {
            dxgi_format: read_u32(0),
            resource_dimension: read_u32(4),
            misc_flag: read_u32(8),
            array_size: read_u32(12),
            misc_flags2: read_u32(16),
        })
    }

    /// Detect format from header.
    fn detect_format(
        header: &DdsHeader,
        dx10: Option<&DdsDx10Header>,
    ) -> Option<CompressedFormat> {
        // DX10 extension takes priority
        if let Some(dx10) = dx10 {
            return Self::detect_dxgi_format(dx10.dxgi_format);
        }

        // Legacy FourCC detection
        if header.pixel_format.is_compressed() {
            return Self::detect_fourcc_format(&header.pixel_format.four_cc);
        }

        None
    }

    /// Detect format from DXGI format code.
    fn detect_dxgi_format(dxgi_format: u32) -> Option<CompressedFormat> {
        // DXGI format constants
        const DXGI_FORMAT_BC1_UNORM: u32 = 71;
        const DXGI_FORMAT_BC1_UNORM_SRGB: u32 = 72;
        const DXGI_FORMAT_BC2_UNORM: u32 = 74;
        const DXGI_FORMAT_BC2_UNORM_SRGB: u32 = 75;
        const DXGI_FORMAT_BC3_UNORM: u32 = 77;
        const DXGI_FORMAT_BC3_UNORM_SRGB: u32 = 78;
        const DXGI_FORMAT_BC4_UNORM: u32 = 80;
        const DXGI_FORMAT_BC4_SNORM: u32 = 81;
        const DXGI_FORMAT_BC5_UNORM: u32 = 83;
        const DXGI_FORMAT_BC5_SNORM: u32 = 84;
        const DXGI_FORMAT_BC6H_UF16: u32 = 95;
        const DXGI_FORMAT_BC6H_SF16: u32 = 96;
        const DXGI_FORMAT_BC7_UNORM: u32 = 98;
        const DXGI_FORMAT_BC7_UNORM_SRGB: u32 = 99;

        // ASTC formats (extended DXGI)
        const DXGI_FORMAT_ASTC_4X4_UNORM: u32 = 134;
        const DXGI_FORMAT_ASTC_4X4_UNORM_SRGB: u32 = 135;

        match dxgi_format {
            DXGI_FORMAT_BC1_UNORM => Some(CompressedFormat::Bc1Rgba),
            DXGI_FORMAT_BC1_UNORM_SRGB => Some(CompressedFormat::Bc1RgbaSrgb),
            DXGI_FORMAT_BC2_UNORM => Some(CompressedFormat::Bc2),
            DXGI_FORMAT_BC2_UNORM_SRGB => Some(CompressedFormat::Bc2Srgb),
            DXGI_FORMAT_BC3_UNORM => Some(CompressedFormat::Bc3),
            DXGI_FORMAT_BC3_UNORM_SRGB => Some(CompressedFormat::Bc3Srgb),
            DXGI_FORMAT_BC4_UNORM => Some(CompressedFormat::Bc4Unorm),
            DXGI_FORMAT_BC4_SNORM => Some(CompressedFormat::Bc4Snorm),
            DXGI_FORMAT_BC5_UNORM => Some(CompressedFormat::Bc5Unorm),
            DXGI_FORMAT_BC5_SNORM => Some(CompressedFormat::Bc5Snorm),
            DXGI_FORMAT_BC6H_UF16 => Some(CompressedFormat::Bc6hUfloat),
            DXGI_FORMAT_BC6H_SF16 => Some(CompressedFormat::Bc6hSfloat),
            DXGI_FORMAT_BC7_UNORM => Some(CompressedFormat::Bc7Unorm),
            DXGI_FORMAT_BC7_UNORM_SRGB => Some(CompressedFormat::Bc7Srgb),
            DXGI_FORMAT_ASTC_4X4_UNORM => Some(CompressedFormat::Astc4x4),
            DXGI_FORMAT_ASTC_4X4_UNORM_SRGB => Some(CompressedFormat::Astc4x4Srgb),
            _ => None,
        }
    }

    /// Detect format from FourCC code.
    fn detect_fourcc_format(four_cc: &[u8; 4]) -> Option<CompressedFormat> {
        match four_cc {
            b"DXT1" => Some(CompressedFormat::Bc1Rgba),
            b"DXT2" | b"DXT3" => Some(CompressedFormat::Bc2),
            b"DXT4" | b"DXT5" => Some(CompressedFormat::Bc3),
            b"BC4U" | b"ATI1" => Some(CompressedFormat::Bc4Unorm),
            b"BC4S" => Some(CompressedFormat::Bc4Snorm),
            b"BC5U" | b"ATI2" => Some(CompressedFormat::Bc5Unorm),
            b"BC5S" => Some(CompressedFormat::Bc5Snorm),
            _ => None,
        }
    }

    /// Determine texture type.
    fn determine_texture_type(
        header: &DdsHeader,
        dx10: Option<&DdsDx10Header>,
    ) -> TextureType {
        let is_cubemap = (header.caps2 & DdsCaps2Flags::CUBEMAP) != 0;
        let is_volume = (header.caps2 & DdsCaps2Flags::VOLUME) != 0;

        let is_array = if let Some(dx10) = dx10 {
            dx10.array_size > 1
        } else {
            false
        };

        if is_volume {
            TextureType::Volume
        } else if is_cubemap && is_array {
            TextureType::CubemapArray
        } else if is_cubemap {
            TextureType::Cubemap
        } else if is_array {
            TextureType::Array2D
        } else {
            TextureType::Texture2D
        }
    }

    /// Calculate face and array counts.
    fn calculate_counts(
        header: &DdsHeader,
        dx10: Option<&DdsDx10Header>,
    ) -> (u32, u32) {
        let face_count = if (header.caps2 & DdsCaps2Flags::CUBEMAP) != 0 {
            // Count set face bits
            let face_bits = header.caps2 & DdsCaps2Flags::CUBEMAP_ALL_FACES;
            (face_bits >> 10).count_ones()
        } else {
            1
        };

        let array_count = if let Some(dx10) = dx10 {
            dx10.array_size.max(1)
        } else {
            1
        };

        (face_count, array_count)
    }

    /// Parse mip level data.
    fn parse_mip_levels(
        data: &[u8],
        header: &DdsHeader,
        format: Option<CompressedFormat>,
        data_offset: usize,
        mip_count: u32,
        face_count: u32,
        array_count: u32,
        depth: u32,
    ) -> Result<(Vec<FaceData>, usize), TextureImportError> {
        let total_faces = face_count * array_count;
        let mut faces: Vec<FaceData> = (0..total_faces).map(FaceData::new).collect();

        let mut offset = data_offset;
        let start_offset = offset;

        // DDS stores data as: for each face/layer { for each mip level { data } }
        for face_idx in 0..total_faces {
            for mip_level in 0..mip_count {
                let width = (header.width >> mip_level).max(1);
                let height = (header.height >> mip_level).max(1);
                let mip_depth = (depth >> mip_level).max(1);

                let size = if let Some(fmt) = format {
                    fmt.calculate_size(width, height) * mip_depth as usize
                } else {
                    // Uncompressed
                    let bits_per_pixel = header.pixel_format.rgb_bit_count;
                    ((width * height * mip_depth * bits_per_pixel + 7) / 8) as usize
                };

                if offset + size > data.len() {
                    return Err(TextureImportError::InvalidData(format!(
                        "DDS file truncated at face {}, mip {}",
                        face_idx, mip_level
                    )));
                }

                let mip_data = MipLevelData::new(mip_level, width, height, offset, size);
                faces[face_idx as usize].add_mip(mip_data);
                offset += size;
            }
        }

        Ok((faces, offset - start_offset))
    }

    /// Get compressed data for a specific mip level.
    ///
    /// Returns a slice to the compressed data that can be passed directly to GPU.
    pub fn get_mip_data<'a>(
        &self,
        data: &'a [u8],
        texture: &DdsTexture,
        face: u32,
        mip_level: u32,
    ) -> Option<&'a [u8]> {
        let face_data = texture.faces.get(face as usize)?;
        let mip_data = face_data.mip_levels.get(mip_level as usize)?;

        if mip_data.offset + mip_data.size <= data.len() {
            Some(&data[mip_data.offset..mip_data.offset + mip_data.size])
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Compressed Texture Asset
// ---------------------------------------------------------------------------

/// A compressed texture asset ready for GPU upload.
///
/// Unlike decoded textures, compressed textures pass data directly to GPU
/// without decode/re-encode.
#[derive(Debug, Clone)]
pub struct CompressedTextureAsset {
    /// Unique asset ID
    pub id: u64,
    /// Texture width at base level
    pub width: u32,
    /// Texture height at base level
    pub height: u32,
    /// Texture depth (1 for 2D textures)
    pub depth: u32,
    /// Compressed format
    pub format: CompressedFormat,
    /// Texture type
    pub texture_type: TextureType,
    /// Number of mip levels
    pub mip_count: u32,
    /// Number of array layers
    pub array_count: u32,
    /// Number of faces (1 for 2D, 6 for cubemap)
    pub face_count: u32,
    /// Compressed data (all mips, faces, layers)
    pub data: Vec<u8>,
    /// Face/layer structure
    pub faces: Vec<FaceData>,
    /// Total memory size
    pub memory_size: usize,
    /// Current state
    pub state: TextureState,
}

impl CompressedTextureAsset {
    /// Check if texture is ready.
    pub fn is_ready(&self) -> bool {
        self.state == TextureState::Ready
    }

    /// Get data for a specific face and mip level.
    pub fn get_mip_data(&self, face: u32, mip_level: u32) -> Option<&[u8]> {
        let face_data = self.faces.get(face as usize)?;
        let mip_data = face_data.mip_levels.get(mip_level as usize)?;

        // Adjust offset relative to our data buffer (original offset was relative to file)
        // We need to recalculate based on our stored structure
        let mut offset = 0;
        for f in 0..face as usize {
            if let Some(fd) = self.faces.get(f) {
                for ml in &fd.mip_levels {
                    offset += ml.size;
                }
            }
        }
        for m in 0..mip_level as usize {
            if let Some(ml) = face_data.mip_levels.get(m) {
                offset += ml.size;
            }
        }

        if offset + mip_data.size <= self.data.len() {
            Some(&self.data[offset..offset + mip_data.size])
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Extension Detection for @asset decorator
// ---------------------------------------------------------------------------

/// Check if a file extension indicates KTX v1 format.
#[inline]
pub fn is_ktx_extension(ext: &str) -> bool {
    let ext_lower = ext.to_lowercase();
    ext_lower == "ktx"
}

/// Check if a file extension indicates DDS format.
#[inline]
pub fn is_dds_extension(ext: &str) -> bool {
    let ext_lower = ext.to_lowercase();
    ext_lower == "dds"
}

/// Detect container format from magic bytes.
pub fn detect_container_format(data: &[u8]) -> Option<&'static str> {
    if KtxParser::is_ktx(data) {
        Some("ktx")
    } else if DdsParser::is_dds(data) {
        Some("dds")
    } else {
        None
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // CompressedFormat tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compressed_format_block_dimensions() {
        assert_eq!(CompressedFormat::Bc1Rgb.block_dimensions(), (4, 4));
        assert_eq!(CompressedFormat::Bc7Unorm.block_dimensions(), (4, 4));
        assert_eq!(CompressedFormat::Astc4x4.block_dimensions(), (4, 4));
        assert_eq!(CompressedFormat::Astc5x5.block_dimensions(), (5, 5));
        assert_eq!(CompressedFormat::Astc8x8.block_dimensions(), (8, 8));
        assert_eq!(CompressedFormat::Astc12x12.block_dimensions(), (12, 12));
    }

    #[test]
    fn test_compressed_format_bytes_per_block() {
        assert_eq!(CompressedFormat::Bc1Rgb.bytes_per_block(), 8);
        assert_eq!(CompressedFormat::Bc4Unorm.bytes_per_block(), 8);
        assert_eq!(CompressedFormat::Bc2.bytes_per_block(), 16);
        assert_eq!(CompressedFormat::Bc3.bytes_per_block(), 16);
        assert_eq!(CompressedFormat::Bc7Unorm.bytes_per_block(), 16);
        assert_eq!(CompressedFormat::Astc4x4.bytes_per_block(), 16);
    }

    #[test]
    fn test_compressed_format_calculate_size() {
        // BC1: 4x4 blocks, 8 bytes each
        // 256x256 = 64x64 blocks = 4096 blocks * 8 = 32768 bytes
        assert_eq!(CompressedFormat::Bc1Rgb.calculate_size(256, 256), 32768);

        // BC3: 4x4 blocks, 16 bytes each
        // 256x256 = 64x64 blocks = 4096 blocks * 16 = 65536 bytes
        assert_eq!(CompressedFormat::Bc3.calculate_size(256, 256), 65536);

        // Non-power-of-2: 100x100
        // (100+3)/4 = 25 blocks wide, (100+3)/4 = 25 blocks tall
        // 25 * 25 = 625 blocks * 8 = 5000 bytes for BC1
        assert_eq!(CompressedFormat::Bc1Rgb.calculate_size(100, 100), 5000);
    }

    #[test]
    fn test_compressed_format_is_srgb() {
        assert!(!CompressedFormat::Bc1Rgb.is_srgb());
        assert!(CompressedFormat::Bc1RgbSrgb.is_srgb());
        assert!(!CompressedFormat::Bc7Unorm.is_srgb());
        assert!(CompressedFormat::Bc7Srgb.is_srgb());
        assert!(!CompressedFormat::Astc4x4.is_srgb());
        assert!(CompressedFormat::Astc4x4Srgb.is_srgb());
    }

    #[test]
    fn test_compressed_format_is_hdr() {
        assert!(!CompressedFormat::Bc1Rgb.is_hdr());
        assert!(!CompressedFormat::Bc7Unorm.is_hdr());
        assert!(CompressedFormat::Bc6hUfloat.is_hdr());
        assert!(CompressedFormat::Bc6hSfloat.is_hdr());
    }

    #[test]
    fn test_compressed_format_is_bcn() {
        assert!(CompressedFormat::Bc1Rgb.is_bcn());
        assert!(CompressedFormat::Bc7Srgb.is_bcn());
        assert!(!CompressedFormat::Astc4x4.is_bcn());
        assert!(!CompressedFormat::Etc2Rgb.is_bcn());
    }

    #[test]
    fn test_compressed_format_is_astc() {
        assert!(!CompressedFormat::Bc1Rgb.is_astc());
        assert!(CompressedFormat::Astc4x4.is_astc());
        assert!(CompressedFormat::Astc12x12Srgb.is_astc());
    }

    #[test]
    fn test_compressed_format_wgpu_str() {
        assert_eq!(CompressedFormat::Bc1Rgb.to_wgpu_format_str(), "Bc1RgbaUnorm");
        assert_eq!(
            CompressedFormat::Bc7Srgb.to_wgpu_format_str(),
            "Bc7RgbaUnormSrgb"
        );
        assert_eq!(
            CompressedFormat::Astc8x8.to_wgpu_format_str(),
            "Astc8x8Unorm"
        );
    }

    // -----------------------------------------------------------------------
    // KTX v1 Parser tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ktx_is_ktx() {
        assert!(KtxParser::is_ktx(&KTX1_IDENTIFIER));
        assert!(!KtxParser::is_ktx(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]));
        assert!(!KtxParser::is_ktx(&[0, 1, 2]));
    }

    #[test]
    fn test_ktx_parse_too_short() {
        let result = KtxParser::parse(&[0u8; 32]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ktx_parse_invalid_identifier() {
        let data = [0u8; 128];
        let result = KtxParser::parse(&data);
        assert!(result.is_err());
    }

    fn create_minimal_ktx() -> Vec<u8> {
        let mut data = Vec::new();

        // Identifier (12 bytes)
        data.extend_from_slice(&KTX1_IDENTIFIER);

        // Endianness (4 bytes)
        data.extend_from_slice(&0x04030201u32.to_le_bytes());

        // glType (4 bytes) - 0 for compressed
        data.extend_from_slice(&0u32.to_le_bytes());

        // glTypeSize (4 bytes)
        data.extend_from_slice(&1u32.to_le_bytes());

        // glFormat (4 bytes) - 0 for compressed
        data.extend_from_slice(&0u32.to_le_bytes());

        // glInternalFormat (4 bytes) - BC1 (DXT1)
        data.extend_from_slice(&0x83F0u32.to_le_bytes()); // GL_COMPRESSED_RGB_S3TC_DXT1_EXT

        // glBaseInternalFormat (4 bytes)
        data.extend_from_slice(&0x1907u32.to_le_bytes()); // GL_RGB

        // pixelWidth (4 bytes)
        data.extend_from_slice(&256u32.to_le_bytes());

        // pixelHeight (4 bytes)
        data.extend_from_slice(&256u32.to_le_bytes());

        // pixelDepth (4 bytes)
        data.extend_from_slice(&0u32.to_le_bytes());

        // numberOfArrayElements (4 bytes)
        data.extend_from_slice(&0u32.to_le_bytes());

        // numberOfFaces (4 bytes)
        data.extend_from_slice(&1u32.to_le_bytes());

        // numberOfMipmapLevels (4 bytes)
        data.extend_from_slice(&1u32.to_le_bytes());

        // bytesOfKeyValueData (4 bytes)
        data.extend_from_slice(&0u32.to_le_bytes());

        // Image size for mip 0 (BC1: 64x64 blocks * 8 bytes = 32768)
        data.extend_from_slice(&32768u32.to_le_bytes());

        // Image data (32768 bytes)
        data.extend(std::iter::repeat(0u8).take(32768));

        data
    }

    #[test]
    fn test_ktx_parse_valid_bc1() {
        let data = create_minimal_ktx();
        let result = KtxParser::parse(&data);
        assert!(result.is_ok());

        let tex = result.unwrap();
        assert_eq!(tex.width, 256);
        assert_eq!(tex.height, 256);
        assert_eq!(tex.mip_count, 1);
        assert_eq!(tex.face_count, 1);
        assert_eq!(tex.texture_type, TextureType::Texture2D);
        assert_eq!(tex.format, Some(CompressedFormat::Bc1Rgb));
    }

    #[test]
    fn test_ktx_parse_cubemap() {
        let mut data = Vec::new();

        // Identifier
        data.extend_from_slice(&KTX1_IDENTIFIER);

        // Endianness
        data.extend_from_slice(&0x04030201u32.to_le_bytes());

        // glType
        data.extend_from_slice(&0u32.to_le_bytes());

        // glTypeSize
        data.extend_from_slice(&1u32.to_le_bytes());

        // glFormat
        data.extend_from_slice(&0u32.to_le_bytes());

        // glInternalFormat (BC1)
        data.extend_from_slice(&0x83F0u32.to_le_bytes());

        // glBaseInternalFormat
        data.extend_from_slice(&0x1907u32.to_le_bytes());

        // pixelWidth
        data.extend_from_slice(&64u32.to_le_bytes());

        // pixelHeight
        data.extend_from_slice(&64u32.to_le_bytes());

        // pixelDepth
        data.extend_from_slice(&0u32.to_le_bytes());

        // numberOfArrayElements
        data.extend_from_slice(&0u32.to_le_bytes());

        // numberOfFaces (6 for cubemap)
        data.extend_from_slice(&6u32.to_le_bytes());

        // numberOfMipmapLevels
        data.extend_from_slice(&1u32.to_le_bytes());

        // bytesOfKeyValueData
        data.extend_from_slice(&0u32.to_le_bytes());

        // Image size for mip 0 (BC1: 16x16 blocks * 8 bytes * 6 faces = 12288)
        data.extend_from_slice(&12288u32.to_le_bytes());

        // Image data for all 6 faces
        data.extend(std::iter::repeat(0u8).take(2048 * 6 + 24)); // 2048 per face + padding

        let result = KtxParser::parse(&data);
        assert!(result.is_ok());

        let tex = result.unwrap();
        assert_eq!(tex.texture_type, TextureType::Cubemap);
        assert_eq!(tex.face_count, 6);
    }

    #[test]
    fn test_ktx_parse_texture_array() {
        let mut data = Vec::new();

        // Identifier
        data.extend_from_slice(&KTX1_IDENTIFIER);

        // Endianness
        data.extend_from_slice(&0x04030201u32.to_le_bytes());

        // glType
        data.extend_from_slice(&0u32.to_le_bytes());

        // glTypeSize
        data.extend_from_slice(&1u32.to_le_bytes());

        // glFormat
        data.extend_from_slice(&0u32.to_le_bytes());

        // glInternalFormat (BC3)
        data.extend_from_slice(&0x83F3u32.to_le_bytes());

        // glBaseInternalFormat
        data.extend_from_slice(&0x1908u32.to_le_bytes()); // GL_RGBA

        // pixelWidth
        data.extend_from_slice(&64u32.to_le_bytes());

        // pixelHeight
        data.extend_from_slice(&64u32.to_le_bytes());

        // pixelDepth
        data.extend_from_slice(&0u32.to_le_bytes());

        // numberOfArrayElements (4 layers)
        data.extend_from_slice(&4u32.to_le_bytes());

        // numberOfFaces
        data.extend_from_slice(&1u32.to_le_bytes());

        // numberOfMipmapLevels
        data.extend_from_slice(&1u32.to_le_bytes());

        // bytesOfKeyValueData
        data.extend_from_slice(&0u32.to_le_bytes());

        // Image size for mip 0 (BC3: 16x16 blocks * 16 bytes * 4 layers = 16384)
        data.extend_from_slice(&16384u32.to_le_bytes());

        // Image data for all 4 layers
        data.extend(std::iter::repeat(0u8).take(4096 * 4));

        let result = KtxParser::parse(&data);
        assert!(result.is_ok());

        let tex = result.unwrap();
        assert_eq!(tex.texture_type, TextureType::Array2D);
        assert_eq!(tex.array_count, 4);
        assert_eq!(tex.format, Some(CompressedFormat::Bc3));
    }

    #[test]
    fn test_ktx_mip_levels() {
        let mut data = Vec::new();

        // Identifier
        data.extend_from_slice(&KTX1_IDENTIFIER);

        // Endianness
        data.extend_from_slice(&0x04030201u32.to_le_bytes());

        // glType
        data.extend_from_slice(&0u32.to_le_bytes());

        // glTypeSize
        data.extend_from_slice(&1u32.to_le_bytes());

        // glFormat
        data.extend_from_slice(&0u32.to_le_bytes());

        // glInternalFormat (BC1)
        data.extend_from_slice(&0x83F0u32.to_le_bytes());

        // glBaseInternalFormat
        data.extend_from_slice(&0x1907u32.to_le_bytes());

        // pixelWidth
        data.extend_from_slice(&64u32.to_le_bytes());

        // pixelHeight
        data.extend_from_slice(&64u32.to_le_bytes());

        // pixelDepth
        data.extend_from_slice(&0u32.to_le_bytes());

        // numberOfArrayElements
        data.extend_from_slice(&0u32.to_le_bytes());

        // numberOfFaces
        data.extend_from_slice(&1u32.to_le_bytes());

        // numberOfMipmapLevels (7 levels: 64, 32, 16, 8, 4, 2, 1)
        data.extend_from_slice(&7u32.to_le_bytes());

        // bytesOfKeyValueData
        data.extend_from_slice(&0u32.to_le_bytes());

        // Mip 0: 64x64 -> 16x16 blocks -> 2048 bytes
        data.extend_from_slice(&2048u32.to_le_bytes());
        data.extend(std::iter::repeat(0u8).take(2048));

        // Mip 1: 32x32 -> 8x8 blocks -> 512 bytes
        data.extend_from_slice(&512u32.to_le_bytes());
        data.extend(std::iter::repeat(0u8).take(512));

        // Mip 2: 16x16 -> 4x4 blocks -> 128 bytes
        data.extend_from_slice(&128u32.to_le_bytes());
        data.extend(std::iter::repeat(0u8).take(128));

        // Mip 3: 8x8 -> 2x2 blocks -> 32 bytes
        data.extend_from_slice(&32u32.to_le_bytes());
        data.extend(std::iter::repeat(0u8).take(32));

        // Mip 4: 4x4 -> 1x1 blocks -> 8 bytes
        data.extend_from_slice(&8u32.to_le_bytes());
        data.extend(std::iter::repeat(0u8).take(8));

        // Mip 5: 2x2 -> 1x1 blocks -> 8 bytes
        data.extend_from_slice(&8u32.to_le_bytes());
        data.extend(std::iter::repeat(0u8).take(8));

        // Mip 6: 1x1 -> 1x1 blocks -> 8 bytes
        data.extend_from_slice(&8u32.to_le_bytes());
        data.extend(std::iter::repeat(0u8).take(8));

        let result = KtxParser::parse(&data);
        assert!(result.is_ok());

        let tex = result.unwrap();
        assert_eq!(tex.mip_count, 7);
        assert_eq!(tex.faces.len(), 1);
        assert_eq!(tex.faces[0].mip_levels.len(), 7);

        // Check mip dimensions
        assert_eq!(tex.faces[0].mip_levels[0].width, 64);
        assert_eq!(tex.faces[0].mip_levels[1].width, 32);
        assert_eq!(tex.faces[0].mip_levels[2].width, 16);
        assert_eq!(tex.faces[0].mip_levels[3].width, 8);
        assert_eq!(tex.faces[0].mip_levels[4].width, 4);
        assert_eq!(tex.faces[0].mip_levels[5].width, 2);
        assert_eq!(tex.faces[0].mip_levels[6].width, 1);
    }

    // -----------------------------------------------------------------------
    // DDS Parser tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dds_is_dds() {
        assert!(DdsParser::is_dds(b"DDS "));
        assert!(!DdsParser::is_dds(b"PNG "));
        assert!(!DdsParser::is_dds(&[0, 1, 2]));
    }

    #[test]
    fn test_dds_parse_too_short() {
        let result = DdsParser::parse(&[0u8; 64]);
        assert!(result.is_err());
    }

    fn create_minimal_dds_dxt1() -> Vec<u8> {
        let mut data = Vec::new();

        // Magic
        data.extend_from_slice(b"DDS ");

        // Header size
        data.extend_from_slice(&124u32.to_le_bytes());

        // Flags
        let flags = DdsHeaderFlags::CAPS
            | DdsHeaderFlags::HEIGHT
            | DdsHeaderFlags::WIDTH
            | DdsHeaderFlags::PIXELFORMAT
            | DdsHeaderFlags::MIPMAPCOUNT
            | DdsHeaderFlags::LINEARSIZE;
        data.extend_from_slice(&flags.to_le_bytes());

        // Height
        data.extend_from_slice(&256u32.to_le_bytes());

        // Width
        data.extend_from_slice(&256u32.to_le_bytes());

        // Pitch or linear size (BC1: 32768 bytes for 256x256)
        data.extend_from_slice(&32768u32.to_le_bytes());

        // Depth
        data.extend_from_slice(&0u32.to_le_bytes());

        // Mip map count
        data.extend_from_slice(&1u32.to_le_bytes());

        // Reserved1 (11 DWORDs = 44 bytes)
        data.extend(std::iter::repeat(0u8).take(44));

        // Pixel format
        // Size
        data.extend_from_slice(&32u32.to_le_bytes());
        // Flags (FOURCC)
        data.extend_from_slice(&DdsPixelFormatFlags::FOURCC.to_le_bytes());
        // FourCC
        data.extend_from_slice(b"DXT1");
        // RGB bit count
        data.extend_from_slice(&0u32.to_le_bytes());
        // R mask
        data.extend_from_slice(&0u32.to_le_bytes());
        // G mask
        data.extend_from_slice(&0u32.to_le_bytes());
        // B mask
        data.extend_from_slice(&0u32.to_le_bytes());
        // A mask
        data.extend_from_slice(&0u32.to_le_bytes());

        // Caps
        data.extend_from_slice(
            &(DdsCapsFlags::TEXTURE | DdsCapsFlags::COMPLEX | DdsCapsFlags::MIPMAP).to_le_bytes(),
        );

        // Caps2
        data.extend_from_slice(&0u32.to_le_bytes());

        // Caps3
        data.extend_from_slice(&0u32.to_le_bytes());

        // Caps4
        data.extend_from_slice(&0u32.to_le_bytes());

        // Reserved2
        data.extend_from_slice(&0u32.to_le_bytes());

        // Image data (32768 bytes)
        data.extend(std::iter::repeat(0u8).take(32768));

        data
    }

    #[test]
    fn test_dds_parse_valid_dxt1() {
        let data = create_minimal_dds_dxt1();
        let result = DdsParser::parse(&data);
        assert!(result.is_ok());

        let tex = result.unwrap();
        assert_eq!(tex.width, 256);
        assert_eq!(tex.height, 256);
        assert_eq!(tex.mip_count, 1);
        assert_eq!(tex.face_count, 1);
        assert_eq!(tex.texture_type, TextureType::Texture2D);
        assert_eq!(tex.format, Some(CompressedFormat::Bc1Rgba));
    }

    #[test]
    fn test_dds_parse_bc7_srgb() {
        // Test DXGI format detection
        let format = DdsParser::detect_dxgi_format(99); // BC7_UNORM_SRGB
        assert_eq!(format, Some(CompressedFormat::Bc7Srgb));
    }

    #[test]
    fn test_dds_fourcc_detection() {
        assert_eq!(
            DdsParser::detect_fourcc_format(b"DXT1"),
            Some(CompressedFormat::Bc1Rgba)
        );
        assert_eq!(
            DdsParser::detect_fourcc_format(b"DXT3"),
            Some(CompressedFormat::Bc2)
        );
        assert_eq!(
            DdsParser::detect_fourcc_format(b"DXT5"),
            Some(CompressedFormat::Bc3)
        );
        assert_eq!(
            DdsParser::detect_fourcc_format(b"ATI1"),
            Some(CompressedFormat::Bc4Unorm)
        );
        assert_eq!(
            DdsParser::detect_fourcc_format(b"ATI2"),
            Some(CompressedFormat::Bc5Unorm)
        );
    }

    fn create_dds_cubemap() -> Vec<u8> {
        let mut data = Vec::new();

        // Magic
        data.extend_from_slice(b"DDS ");

        // Header size
        data.extend_from_slice(&124u32.to_le_bytes());

        // Flags
        let flags = DdsHeaderFlags::CAPS
            | DdsHeaderFlags::HEIGHT
            | DdsHeaderFlags::WIDTH
            | DdsHeaderFlags::PIXELFORMAT;
        data.extend_from_slice(&flags.to_le_bytes());

        // Height
        data.extend_from_slice(&64u32.to_le_bytes());

        // Width
        data.extend_from_slice(&64u32.to_le_bytes());

        // Pitch or linear size
        data.extend_from_slice(&2048u32.to_le_bytes());

        // Depth
        data.extend_from_slice(&0u32.to_le_bytes());

        // Mip map count
        data.extend_from_slice(&1u32.to_le_bytes());

        // Reserved1
        data.extend(std::iter::repeat(0u8).take(44));

        // Pixel format
        data.extend_from_slice(&32u32.to_le_bytes());
        data.extend_from_slice(&DdsPixelFormatFlags::FOURCC.to_le_bytes());
        data.extend_from_slice(b"DXT1");
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        // Caps
        data.extend_from_slice(&(DdsCapsFlags::TEXTURE | DdsCapsFlags::COMPLEX).to_le_bytes());

        // Caps2 (cubemap with all faces)
        data.extend_from_slice(
            &(DdsCaps2Flags::CUBEMAP | DdsCaps2Flags::CUBEMAP_ALL_FACES).to_le_bytes(),
        );

        // Caps3, Caps4, Reserved2
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        // Image data for 6 faces (2048 bytes each)
        data.extend(std::iter::repeat(0u8).take(2048 * 6));

        data
    }

    #[test]
    fn test_dds_parse_cubemap() {
        let data = create_dds_cubemap();
        let result = DdsParser::parse(&data);
        assert!(result.is_ok());

        let tex = result.unwrap();
        assert_eq!(tex.texture_type, TextureType::Cubemap);
        assert_eq!(tex.face_count, 6);
    }

    fn create_dds_dx10_bc7() -> Vec<u8> {
        let mut data = Vec::new();

        // Magic
        data.extend_from_slice(b"DDS ");

        // Header size
        data.extend_from_slice(&124u32.to_le_bytes());

        // Flags
        let flags = DdsHeaderFlags::CAPS
            | DdsHeaderFlags::HEIGHT
            | DdsHeaderFlags::WIDTH
            | DdsHeaderFlags::PIXELFORMAT;
        data.extend_from_slice(&flags.to_le_bytes());

        // Height
        data.extend_from_slice(&128u32.to_le_bytes());

        // Width
        data.extend_from_slice(&128u32.to_le_bytes());

        // Pitch or linear size
        data.extend_from_slice(&16384u32.to_le_bytes());

        // Depth
        data.extend_from_slice(&0u32.to_le_bytes());

        // Mip map count
        data.extend_from_slice(&1u32.to_le_bytes());

        // Reserved1
        data.extend(std::iter::repeat(0u8).take(44));

        // Pixel format - DX10 extension marker
        data.extend_from_slice(&32u32.to_le_bytes());
        data.extend_from_slice(&DdsPixelFormatFlags::FOURCC.to_le_bytes());
        data.extend_from_slice(b"DX10");
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        // Caps
        data.extend_from_slice(&DdsCapsFlags::TEXTURE.to_le_bytes());

        // Caps2, Caps3, Caps4, Reserved2
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        // DX10 header
        data.extend_from_slice(&98u32.to_le_bytes()); // BC7_UNORM
        data.extend_from_slice(&3u32.to_le_bytes()); // 2D
        data.extend_from_slice(&0u32.to_le_bytes()); // Misc
        data.extend_from_slice(&1u32.to_le_bytes()); // Array size
        data.extend_from_slice(&0u32.to_le_bytes()); // Misc2

        // Image data
        data.extend(std::iter::repeat(0u8).take(16384));

        data
    }

    #[test]
    fn test_dds_parse_dx10_bc7() {
        let data = create_dds_dx10_bc7();
        let result = DdsParser::parse(&data);
        assert!(result.is_ok());

        let tex = result.unwrap();
        assert!(tex.dx10_header.is_some());
        assert_eq!(tex.format, Some(CompressedFormat::Bc7Unorm));
        assert_eq!(tex.width, 128);
        assert_eq!(tex.height, 128);
    }

    // -----------------------------------------------------------------------
    // Extension detection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_ktx_extension() {
        assert!(is_ktx_extension("ktx"));
        assert!(is_ktx_extension("KTX"));
        assert!(is_ktx_extension("Ktx"));
        assert!(!is_ktx_extension("ktx2"));
        assert!(!is_ktx_extension("dds"));
    }

    #[test]
    fn test_is_dds_extension() {
        assert!(is_dds_extension("dds"));
        assert!(is_dds_extension("DDS"));
        assert!(is_dds_extension("Dds"));
        assert!(!is_dds_extension("ktx"));
    }

    #[test]
    fn test_detect_container_format() {
        let ktx_data = create_minimal_ktx();
        assert_eq!(detect_container_format(&ktx_data), Some("ktx"));

        let dds_data = create_minimal_dds_dxt1();
        assert_eq!(detect_container_format(&dds_data), Some("dds"));

        let unknown = [0u8; 64];
        assert_eq!(detect_container_format(&unknown), None);
    }

    // -----------------------------------------------------------------------
    // Compressed data passthrough tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compressed_data_passthrough_ktx() {
        let mut data = create_minimal_ktx();
        // Fill image data with recognizable pattern
        for i in 0..32768 {
            data[68 + i] = (i % 256) as u8;
        }

        let result = KtxParser::parse(&data);
        assert!(result.is_ok());

        let tex = result.unwrap();
        let parser = KtxParser;
        let mip_data = parser.get_mip_data(&data, &tex, 0, 0);
        assert!(mip_data.is_some());

        let slice = mip_data.unwrap();
        assert_eq!(slice.len(), 32768);
        // Verify data passthrough
        for i in 0..100 {
            assert_eq!(slice[i], (i % 256) as u8);
        }
    }

    #[test]
    fn test_compressed_data_passthrough_dds() {
        let mut data = create_minimal_dds_dxt1();
        // Fill image data with recognizable pattern
        for i in 0..32768 {
            data[128 + i] = (i % 256) as u8;
        }

        let result = DdsParser::parse(&data);
        assert!(result.is_ok());

        let tex = result.unwrap();
        let parser = DdsParser;
        let mip_data = parser.get_mip_data(&data, &tex, 0, 0);
        assert!(mip_data.is_some());

        let slice = mip_data.unwrap();
        assert_eq!(slice.len(), 32768);
        // Verify data passthrough
        for i in 0..100 {
            assert_eq!(slice[i], (i % 256) as u8);
        }
    }

    // -----------------------------------------------------------------------
    // Invalid file handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ktx_invalid_face_count() {
        let mut data = create_minimal_ktx();
        // Set invalid face count (3 instead of 1 or 6)
        data[52] = 3;

        let result = KtxParser::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_ktx_invalid_endianness() {
        let mut data = create_minimal_ktx();
        // Set invalid endianness
        data[12] = 0xFF;
        data[13] = 0xFF;
        data[14] = 0xFF;
        data[15] = 0xFF;

        let result = KtxParser::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_ktx_zero_width() {
        let mut data = create_minimal_ktx();
        // Set width to 0
        data[36] = 0;
        data[37] = 0;
        data[38] = 0;
        data[39] = 0;

        let result = KtxParser::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dds_invalid_header_size() {
        let mut data = create_minimal_dds_dxt1();
        // Set invalid header size
        data[4] = 100;

        let result = DdsParser::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dds_invalid_pf_size() {
        let mut data = create_minimal_dds_dxt1();
        // Set invalid pixel format size (at offset 4 + 72 = 76)
        data[76] = 16;

        let result = DdsParser::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dds_truncated_data() {
        let data = create_minimal_dds_dxt1();
        // Truncate before image data completes
        let truncated = &data[..150];

        let result = DdsParser::parse(truncated);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // ASTC format detection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ktx_astc_detection() {
        // GL_COMPRESSED_RGBA_ASTC_4x4_KHR = 0x93B0
        let format = KtxParser::detect_astc_format(0x93B0, false);
        assert_eq!(format, Some(CompressedFormat::Astc4x4));

        let format = KtxParser::detect_astc_format(0x93D0, true);
        assert_eq!(format, Some(CompressedFormat::Astc4x4Srgb));

        // 8x8 is index 7
        let format = KtxParser::detect_astc_format(0x93B0 + 7, false);
        assert_eq!(format, Some(CompressedFormat::Astc8x8));
    }
}
