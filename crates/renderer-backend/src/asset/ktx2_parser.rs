//! KTX2 Container Parser with Basis Universal Supercompression (T-AS-2.3)
//!
//! Implements parsing for KTX 2.0 texture container format with:
//! - Full KTX2 header and metadata parsing
//! - Basis Universal supercompressed texture support (UASTC and ETC1S modes)
//! - Transcoding to BCn, ASTC, ETC2, and RGBA8 formats
//! - Cubemap and texture array support
//! - Mip level extraction and GPU upload path
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::ktx2_parser::{Ktx2Parser, TranscodeTarget};
//!
//! // Parse KTX2 file
//! let ktx2_data = std::fs::read("texture.ktx2")?;
//! let ktx2 = Ktx2Parser::parse(&ktx2_data)?;
//!
//! // Check Basis Universal mode
//! if let Some(basis_mode) = ktx2.basis_mode() {
//!     println!("Basis mode: {:?}", basis_mode);
//! }
//!
//! // Transcode to BC7 for desktop GPU
//! let transcoded = ktx2.transcode_to(&ktx2_data, TranscodeTarget::Bc7)?;
//! ```

use std::fmt;

use super::compressed_formats::{CompressedFormat, TextureType};
use super::texture_importer::{TextureImportError, TextureState};

// ===========================================================================
// KTX2 Constants and Types
// ===========================================================================

/// KTX2 file identifier (12 bytes).
pub const KTX2_IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, // 'KTX '
    0x20, 0x32, 0x30, 0xBB, // ' 20'
    0x0D, 0x0A, 0x1A, 0x0A, // '\r\n\x1a\n'
];

/// Vulkan format codes used in KTX2.
/// Based on VK_FORMAT_* constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum VkFormat {
    Undefined = 0,
    // Uncompressed formats
    R8Unorm = 9,
    R8G8Unorm = 16,
    R8G8B8Unorm = 23,
    R8G8B8A8Unorm = 37,
    R8G8B8A8Srgb = 43,
    R16Sfloat = 76,
    R16G16Sfloat = 83,
    R16G16B16A16Sfloat = 97,
    R32Sfloat = 100,
    R32G32B32A32Sfloat = 109,
    // BCn compressed
    Bc1RgbUnorm = 131,
    Bc1RgbSrgb = 132,
    Bc1RgbaUnorm = 133,
    Bc1RgbaSrgb = 134,
    Bc2Unorm = 135,
    Bc2Srgb = 136,
    Bc3Unorm = 137,
    Bc3Srgb = 138,
    Bc4Unorm = 139,
    Bc4Snorm = 140,
    Bc5Unorm = 141,
    Bc5Snorm = 142,
    Bc6hUfloat = 143,
    Bc6hSfloat = 144,
    Bc7Unorm = 145,
    Bc7Srgb = 146,
    // ETC2 compressed
    Etc2R8G8B8Unorm = 147,
    Etc2R8G8B8Srgb = 148,
    Etc2R8G8B8A1Unorm = 149,
    Etc2R8G8B8A1Srgb = 150,
    Etc2R8G8B8A8Unorm = 151,
    Etc2R8G8B8A8Srgb = 152,
    EacR11Unorm = 153,
    EacR11Snorm = 154,
    EacR11G11Unorm = 155,
    EacR11G11Snorm = 156,
    // ASTC compressed (4x4 to 12x12)
    Astc4x4Unorm = 157,
    Astc4x4Srgb = 158,
    Astc5x4Unorm = 159,
    Astc5x4Srgb = 160,
    Astc5x5Unorm = 161,
    Astc5x5Srgb = 162,
    Astc6x5Unorm = 163,
    Astc6x5Srgb = 164,
    Astc6x6Unorm = 165,
    Astc6x6Srgb = 166,
    Astc8x5Unorm = 167,
    Astc8x5Srgb = 168,
    Astc8x6Unorm = 169,
    Astc8x6Srgb = 170,
    Astc8x8Unorm = 171,
    Astc8x8Srgb = 172,
    Astc10x5Unorm = 173,
    Astc10x5Srgb = 174,
    Astc10x6Unorm = 175,
    Astc10x6Srgb = 176,
    Astc10x8Unorm = 177,
    Astc10x8Srgb = 178,
    Astc10x10Unorm = 179,
    Astc10x10Srgb = 180,
    Astc12x10Unorm = 181,
    Astc12x10Srgb = 182,
    Astc12x12Unorm = 183,
    Astc12x12Srgb = 184,
}

impl VkFormat {
    /// Create from raw u32 value.
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => VkFormat::Undefined,
            9 => VkFormat::R8Unorm,
            16 => VkFormat::R8G8Unorm,
            23 => VkFormat::R8G8B8Unorm,
            37 => VkFormat::R8G8B8A8Unorm,
            43 => VkFormat::R8G8B8A8Srgb,
            76 => VkFormat::R16Sfloat,
            83 => VkFormat::R16G16Sfloat,
            97 => VkFormat::R16G16B16A16Sfloat,
            100 => VkFormat::R32Sfloat,
            109 => VkFormat::R32G32B32A32Sfloat,
            131 => VkFormat::Bc1RgbUnorm,
            132 => VkFormat::Bc1RgbSrgb,
            133 => VkFormat::Bc1RgbaUnorm,
            134 => VkFormat::Bc1RgbaSrgb,
            135 => VkFormat::Bc2Unorm,
            136 => VkFormat::Bc2Srgb,
            137 => VkFormat::Bc3Unorm,
            138 => VkFormat::Bc3Srgb,
            139 => VkFormat::Bc4Unorm,
            140 => VkFormat::Bc4Snorm,
            141 => VkFormat::Bc5Unorm,
            142 => VkFormat::Bc5Snorm,
            143 => VkFormat::Bc6hUfloat,
            144 => VkFormat::Bc6hSfloat,
            145 => VkFormat::Bc7Unorm,
            146 => VkFormat::Bc7Srgb,
            147 => VkFormat::Etc2R8G8B8Unorm,
            148 => VkFormat::Etc2R8G8B8Srgb,
            149 => VkFormat::Etc2R8G8B8A1Unorm,
            150 => VkFormat::Etc2R8G8B8A1Srgb,
            151 => VkFormat::Etc2R8G8B8A8Unorm,
            152 => VkFormat::Etc2R8G8B8A8Srgb,
            153 => VkFormat::EacR11Unorm,
            154 => VkFormat::EacR11Snorm,
            155 => VkFormat::EacR11G11Unorm,
            156 => VkFormat::EacR11G11Snorm,
            157 => VkFormat::Astc4x4Unorm,
            158 => VkFormat::Astc4x4Srgb,
            159 => VkFormat::Astc5x4Unorm,
            160 => VkFormat::Astc5x4Srgb,
            161 => VkFormat::Astc5x5Unorm,
            162 => VkFormat::Astc5x5Srgb,
            163 => VkFormat::Astc6x5Unorm,
            164 => VkFormat::Astc6x5Srgb,
            165 => VkFormat::Astc6x6Unorm,
            166 => VkFormat::Astc6x6Srgb,
            167 => VkFormat::Astc8x5Unorm,
            168 => VkFormat::Astc8x5Srgb,
            169 => VkFormat::Astc8x6Unorm,
            170 => VkFormat::Astc8x6Srgb,
            171 => VkFormat::Astc8x8Unorm,
            172 => VkFormat::Astc8x8Srgb,
            173 => VkFormat::Astc10x5Unorm,
            174 => VkFormat::Astc10x5Srgb,
            175 => VkFormat::Astc10x6Unorm,
            176 => VkFormat::Astc10x6Srgb,
            177 => VkFormat::Astc10x8Unorm,
            178 => VkFormat::Astc10x8Srgb,
            179 => VkFormat::Astc10x10Unorm,
            180 => VkFormat::Astc10x10Srgb,
            181 => VkFormat::Astc12x10Unorm,
            182 => VkFormat::Astc12x10Srgb,
            183 => VkFormat::Astc12x12Unorm,
            184 => VkFormat::Astc12x12Srgb,
            _ => VkFormat::Undefined,
        }
    }

    /// Check if this is a compressed format.
    pub fn is_compressed(&self) -> bool {
        matches!(
            self,
            VkFormat::Bc1RgbUnorm
                | VkFormat::Bc1RgbSrgb
                | VkFormat::Bc1RgbaUnorm
                | VkFormat::Bc1RgbaSrgb
                | VkFormat::Bc2Unorm
                | VkFormat::Bc2Srgb
                | VkFormat::Bc3Unorm
                | VkFormat::Bc3Srgb
                | VkFormat::Bc4Unorm
                | VkFormat::Bc4Snorm
                | VkFormat::Bc5Unorm
                | VkFormat::Bc5Snorm
                | VkFormat::Bc6hUfloat
                | VkFormat::Bc6hSfloat
                | VkFormat::Bc7Unorm
                | VkFormat::Bc7Srgb
                | VkFormat::Etc2R8G8B8Unorm
                | VkFormat::Etc2R8G8B8Srgb
                | VkFormat::Etc2R8G8B8A1Unorm
                | VkFormat::Etc2R8G8B8A1Srgb
                | VkFormat::Etc2R8G8B8A8Unorm
                | VkFormat::Etc2R8G8B8A8Srgb
                | VkFormat::EacR11Unorm
                | VkFormat::EacR11Snorm
                | VkFormat::EacR11G11Unorm
                | VkFormat::EacR11G11Snorm
                | VkFormat::Astc4x4Unorm
                | VkFormat::Astc4x4Srgb
                | VkFormat::Astc5x4Unorm
                | VkFormat::Astc5x4Srgb
                | VkFormat::Astc5x5Unorm
                | VkFormat::Astc5x5Srgb
                | VkFormat::Astc6x5Unorm
                | VkFormat::Astc6x5Srgb
                | VkFormat::Astc6x6Unorm
                | VkFormat::Astc6x6Srgb
                | VkFormat::Astc8x5Unorm
                | VkFormat::Astc8x5Srgb
                | VkFormat::Astc8x6Unorm
                | VkFormat::Astc8x6Srgb
                | VkFormat::Astc8x8Unorm
                | VkFormat::Astc8x8Srgb
                | VkFormat::Astc10x5Unorm
                | VkFormat::Astc10x5Srgb
                | VkFormat::Astc10x6Unorm
                | VkFormat::Astc10x6Srgb
                | VkFormat::Astc10x8Unorm
                | VkFormat::Astc10x8Srgb
                | VkFormat::Astc10x10Unorm
                | VkFormat::Astc10x10Srgb
                | VkFormat::Astc12x10Unorm
                | VkFormat::Astc12x10Srgb
                | VkFormat::Astc12x12Unorm
                | VkFormat::Astc12x12Srgb
        )
    }

    /// Check if this is an sRGB format.
    pub fn is_srgb(&self) -> bool {
        matches!(
            self,
            VkFormat::R8G8B8A8Srgb
                | VkFormat::Bc1RgbSrgb
                | VkFormat::Bc1RgbaSrgb
                | VkFormat::Bc2Srgb
                | VkFormat::Bc3Srgb
                | VkFormat::Bc7Srgb
                | VkFormat::Etc2R8G8B8Srgb
                | VkFormat::Etc2R8G8B8A1Srgb
                | VkFormat::Etc2R8G8B8A8Srgb
                | VkFormat::Astc4x4Srgb
                | VkFormat::Astc5x4Srgb
                | VkFormat::Astc5x5Srgb
                | VkFormat::Astc6x5Srgb
                | VkFormat::Astc6x6Srgb
                | VkFormat::Astc8x5Srgb
                | VkFormat::Astc8x6Srgb
                | VkFormat::Astc8x8Srgb
                | VkFormat::Astc10x5Srgb
                | VkFormat::Astc10x6Srgb
                | VkFormat::Astc10x8Srgb
                | VkFormat::Astc10x10Srgb
                | VkFormat::Astc12x10Srgb
                | VkFormat::Astc12x12Srgb
        )
    }

    /// Convert to CompressedFormat if applicable.
    pub fn to_compressed_format(&self) -> Option<CompressedFormat> {
        match self {
            VkFormat::Bc1RgbUnorm => Some(CompressedFormat::Bc1Rgb),
            VkFormat::Bc1RgbSrgb => Some(CompressedFormat::Bc1RgbSrgb),
            VkFormat::Bc1RgbaUnorm => Some(CompressedFormat::Bc1Rgba),
            VkFormat::Bc1RgbaSrgb => Some(CompressedFormat::Bc1RgbaSrgb),
            VkFormat::Bc2Unorm => Some(CompressedFormat::Bc2),
            VkFormat::Bc2Srgb => Some(CompressedFormat::Bc2Srgb),
            VkFormat::Bc3Unorm => Some(CompressedFormat::Bc3),
            VkFormat::Bc3Srgb => Some(CompressedFormat::Bc3Srgb),
            VkFormat::Bc4Unorm => Some(CompressedFormat::Bc4Unorm),
            VkFormat::Bc4Snorm => Some(CompressedFormat::Bc4Snorm),
            VkFormat::Bc5Unorm => Some(CompressedFormat::Bc5Unorm),
            VkFormat::Bc5Snorm => Some(CompressedFormat::Bc5Snorm),
            VkFormat::Bc6hUfloat => Some(CompressedFormat::Bc6hUfloat),
            VkFormat::Bc6hSfloat => Some(CompressedFormat::Bc6hSfloat),
            VkFormat::Bc7Unorm => Some(CompressedFormat::Bc7Unorm),
            VkFormat::Bc7Srgb => Some(CompressedFormat::Bc7Srgb),
            VkFormat::Etc2R8G8B8Unorm => Some(CompressedFormat::Etc2Rgb),
            VkFormat::Etc2R8G8B8Srgb => Some(CompressedFormat::Etc2RgbSrgb),
            VkFormat::Etc2R8G8B8A1Unorm => Some(CompressedFormat::Etc2RgbA1),
            VkFormat::Etc2R8G8B8A1Srgb => Some(CompressedFormat::Etc2RgbA1Srgb),
            VkFormat::Etc2R8G8B8A8Unorm => Some(CompressedFormat::Etc2Rgba),
            VkFormat::Etc2R8G8B8A8Srgb => Some(CompressedFormat::Etc2RgbaSrgb),
            VkFormat::EacR11Unorm => Some(CompressedFormat::EacR11Unorm),
            VkFormat::EacR11Snorm => Some(CompressedFormat::EacR11Snorm),
            VkFormat::EacR11G11Unorm => Some(CompressedFormat::EacRg11Unorm),
            VkFormat::EacR11G11Snorm => Some(CompressedFormat::EacRg11Snorm),
            VkFormat::Astc4x4Unorm => Some(CompressedFormat::Astc4x4),
            VkFormat::Astc4x4Srgb => Some(CompressedFormat::Astc4x4Srgb),
            VkFormat::Astc5x4Unorm => Some(CompressedFormat::Astc5x4),
            VkFormat::Astc5x4Srgb => Some(CompressedFormat::Astc5x4Srgb),
            VkFormat::Astc5x5Unorm => Some(CompressedFormat::Astc5x5),
            VkFormat::Astc5x5Srgb => Some(CompressedFormat::Astc5x5Srgb),
            VkFormat::Astc6x5Unorm => Some(CompressedFormat::Astc6x5),
            VkFormat::Astc6x5Srgb => Some(CompressedFormat::Astc6x5Srgb),
            VkFormat::Astc6x6Unorm => Some(CompressedFormat::Astc6x6),
            VkFormat::Astc6x6Srgb => Some(CompressedFormat::Astc6x6Srgb),
            VkFormat::Astc8x5Unorm => Some(CompressedFormat::Astc8x5),
            VkFormat::Astc8x5Srgb => Some(CompressedFormat::Astc8x5Srgb),
            VkFormat::Astc8x6Unorm => Some(CompressedFormat::Astc8x6),
            VkFormat::Astc8x6Srgb => Some(CompressedFormat::Astc8x6Srgb),
            VkFormat::Astc8x8Unorm => Some(CompressedFormat::Astc8x8),
            VkFormat::Astc8x8Srgb => Some(CompressedFormat::Astc8x8Srgb),
            VkFormat::Astc10x5Unorm => Some(CompressedFormat::Astc10x5),
            VkFormat::Astc10x5Srgb => Some(CompressedFormat::Astc10x5Srgb),
            VkFormat::Astc10x6Unorm => Some(CompressedFormat::Astc10x6),
            VkFormat::Astc10x6Srgb => Some(CompressedFormat::Astc10x6Srgb),
            VkFormat::Astc10x8Unorm => Some(CompressedFormat::Astc10x8),
            VkFormat::Astc10x8Srgb => Some(CompressedFormat::Astc10x8Srgb),
            VkFormat::Astc10x10Unorm => Some(CompressedFormat::Astc10x10),
            VkFormat::Astc10x10Srgb => Some(CompressedFormat::Astc10x10Srgb),
            VkFormat::Astc12x10Unorm => Some(CompressedFormat::Astc12x10),
            VkFormat::Astc12x10Srgb => Some(CompressedFormat::Astc12x10Srgb),
            VkFormat::Astc12x12Unorm => Some(CompressedFormat::Astc12x12),
            VkFormat::Astc12x12Srgb => Some(CompressedFormat::Astc12x12Srgb),
            _ => None,
        }
    }
}

/// Supercompression scheme used in KTX2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum SupercompressionScheme {
    /// No supercompression
    None = 0,
    /// Basis LZ (legacy, ETC1S-only)
    BasisLz = 1,
    /// Zstandard
    Zstd = 2,
    /// Zlib
    Zlib = 3,
}

impl SupercompressionScheme {
    /// Create from raw u32 value.
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => SupercompressionScheme::None,
            1 => SupercompressionScheme::BasisLz,
            2 => SupercompressionScheme::Zstd,
            3 => SupercompressionScheme::Zlib,
            _ => SupercompressionScheme::None,
        }
    }
}

/// Basis Universal encoding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasisMode {
    /// ETC1S: Low quality, high compression, fast transcoding
    /// Uses a global codebook for all blocks
    Etc1s,
    /// UASTC: High quality, moderate compression
    /// Uses 8 bits per texel, higher quality than ETC1S
    Uastc,
}

impl fmt::Display for BasisMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BasisMode::Etc1s => write!(f, "ETC1S"),
            BasisMode::Uastc => write!(f, "UASTC"),
        }
    }
}

/// Target format for Basis Universal transcoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TranscodeTarget {
    /// BC1 (DXT1) - 4 bits/pixel, no alpha
    Bc1,
    /// BC3 (DXT5) - 8 bits/pixel, alpha
    Bc3,
    /// BC4 - single channel
    Bc4,
    /// BC5 - dual channel
    Bc5,
    /// BC7 - high quality
    Bc7,
    /// ASTC 4x4
    Astc4x4,
    /// ETC1 (legacy)
    Etc1,
    /// ETC2 RGB
    Etc2Rgb,
    /// ETC2 RGBA
    Etc2Rgba,
    /// Uncompressed RGBA8
    Rgba8,
    /// Uncompressed RGB565
    Rgb565,
    /// Uncompressed RGBA4444
    Rgba4444,
}

impl TranscodeTarget {
    /// Get the CompressedFormat for this target, if applicable.
    pub fn to_compressed_format(&self, is_srgb: bool) -> Option<CompressedFormat> {
        match self {
            TranscodeTarget::Bc1 => Some(if is_srgb {
                CompressedFormat::Bc1RgbaSrgb
            } else {
                CompressedFormat::Bc1Rgba
            }),
            TranscodeTarget::Bc3 => Some(if is_srgb {
                CompressedFormat::Bc3Srgb
            } else {
                CompressedFormat::Bc3
            }),
            TranscodeTarget::Bc4 => Some(CompressedFormat::Bc4Unorm),
            TranscodeTarget::Bc5 => Some(CompressedFormat::Bc5Unorm),
            TranscodeTarget::Bc7 => Some(if is_srgb {
                CompressedFormat::Bc7Srgb
            } else {
                CompressedFormat::Bc7Unorm
            }),
            TranscodeTarget::Astc4x4 => Some(if is_srgb {
                CompressedFormat::Astc4x4Srgb
            } else {
                CompressedFormat::Astc4x4
            }),
            TranscodeTarget::Etc1 => Some(CompressedFormat::Etc1Rgb),
            TranscodeTarget::Etc2Rgb => Some(if is_srgb {
                CompressedFormat::Etc2RgbSrgb
            } else {
                CompressedFormat::Etc2Rgb
            }),
            TranscodeTarget::Etc2Rgba => Some(if is_srgb {
                CompressedFormat::Etc2RgbaSrgb
            } else {
                CompressedFormat::Etc2Rgba
            }),
            _ => None,
        }
    }

    /// Get bytes per pixel/block for this target.
    pub fn bytes_per_unit(&self) -> usize {
        match self {
            TranscodeTarget::Bc1 | TranscodeTarget::Bc4 | TranscodeTarget::Etc1 => 8,
            TranscodeTarget::Bc3 | TranscodeTarget::Bc5 | TranscodeTarget::Bc7 => 16,
            TranscodeTarget::Astc4x4 => 16,
            TranscodeTarget::Etc2Rgb => 8,
            TranscodeTarget::Etc2Rgba => 16,
            TranscodeTarget::Rgba8 => 4,
            TranscodeTarget::Rgb565 | TranscodeTarget::Rgba4444 => 2,
        }
    }
}

// ===========================================================================
// KTX2 Header Structure
// ===========================================================================

/// KTX2 header (80 bytes total).
#[derive(Debug, Clone)]
pub struct Ktx2Header {
    /// Vulkan format (VK_FORMAT_*)
    pub vk_format: VkFormat,
    /// Type size (1 for compressed, >1 for uncompressed)
    pub type_size: u32,
    /// Texture width at base level
    pub pixel_width: u32,
    /// Texture height at base level (0 for 1D textures)
    pub pixel_height: u32,
    /// Texture depth at base level (0 for non-3D textures)
    pub pixel_depth: u32,
    /// Number of array layers (0 = not an array)
    pub layer_count: u32,
    /// Number of cubemap faces (1 = not a cubemap, 6 = cubemap)
    pub face_count: u32,
    /// Number of mipmap levels
    pub level_count: u32,
    /// Supercompression scheme
    pub supercompression_scheme: SupercompressionScheme,
}

/// KTX2 level index entry.
#[derive(Debug, Clone)]
pub struct Ktx2LevelIndex {
    /// Byte offset to level data from start of file
    pub byte_offset: u64,
    /// Byte length of level data (compressed if using supercompression)
    pub byte_length: u64,
    /// Uncompressed byte length
    pub uncompressed_byte_length: u64,
}

/// KTX2 Data Format Descriptor (DFD) sample.
#[derive(Debug, Clone)]
pub struct DfdSample {
    /// Bit offset within texel
    pub bit_offset: u16,
    /// Bit length of sample
    pub bit_length: u8,
    /// Channel type
    pub channel_type: u8,
    /// Sample position (4 bits each for x, y, z, w)
    pub sample_position: u32,
    /// Sample lower bound
    pub sample_lower: u32,
    /// Sample upper bound
    pub sample_upper: u32,
}

/// KTX2 Data Format Descriptor.
#[derive(Debug, Clone)]
pub struct Ktx2Dfd {
    /// Vendor ID (0 = Khronos)
    pub vendor_id: u32,
    /// Descriptor type
    pub descriptor_type: u32,
    /// Version number
    pub version_number: u32,
    /// Descriptor block size
    pub descriptor_block_size: u32,
    /// Color model
    pub color_model: u8,
    /// Color primaries
    pub color_primaries: u8,
    /// Transfer function
    pub transfer_function: u8,
    /// Flags
    pub flags: u8,
    /// Texel block dimensions
    pub texel_block_dimension: [u8; 4],
    /// Bytes per plane
    pub bytes_plane: [u8; 8],
    /// Samples
    pub samples: Vec<DfdSample>,
}

impl Ktx2Dfd {
    /// Check if this is sRGB transfer function.
    pub fn is_srgb(&self) -> bool {
        // KHR_DF_TRANSFER_SRGB = 2
        self.transfer_function == 2
    }
}

/// KTX2 key-value pair.
#[derive(Debug, Clone)]
pub struct Ktx2KeyValue {
    /// Key string
    pub key: String,
    /// Value bytes
    pub value: Vec<u8>,
}

/// Basis Universal global data (for BasisLZ supercompression).
#[derive(Debug, Clone)]
pub struct BasisGlobalData {
    /// Number of endpoints
    pub endpoint_count: u32,
    /// Number of selectors
    pub selector_count: u32,
    /// Endpoints byte length
    pub endpoints_byte_length: u32,
    /// Selectors byte length
    pub selectors_byte_length: u32,
    /// Tables byte length
    pub tables_byte_length: u32,
    /// Extended byte length
    pub extended_byte_length: u32,
    /// Basis mode
    pub mode: BasisMode,
    /// Has alpha slices
    pub has_alpha: bool,
    /// Raw endpoints data
    pub endpoints: Vec<u8>,
    /// Raw selectors data
    pub selectors: Vec<u8>,
    /// Raw tables data
    pub tables: Vec<u8>,
}

// ===========================================================================
// Parsed KTX2 Texture
// ===========================================================================

/// Fully parsed KTX2 texture.
#[derive(Debug, Clone)]
pub struct Ktx2Texture {
    /// Parsed header
    pub header: Ktx2Header,
    /// Level indices
    pub level_indices: Vec<Ktx2LevelIndex>,
    /// Data format descriptor
    pub dfd: Option<Ktx2Dfd>,
    /// Key-value pairs
    pub key_values: Vec<Ktx2KeyValue>,
    /// Basis global data (if BasisLZ supercompressed)
    pub basis_global_data: Option<BasisGlobalData>,
    /// Detected texture type
    pub texture_type: TextureType,
    /// Width at base level
    pub width: u32,
    /// Height at base level
    pub height: u32,
    /// Depth at base level
    pub depth: u32,
    /// Number of mip levels
    pub mip_count: u32,
    /// Number of array layers
    pub array_count: u32,
    /// Number of faces
    pub face_count: u32,
    /// Is sRGB color space
    pub is_srgb: bool,
    /// Compressed format (if not Basis)
    pub compressed_format: Option<CompressedFormat>,
    /// Byte offset to data section
    pub data_offset: usize,
}

impl Ktx2Texture {
    /// Get the Basis Universal mode, if this is a Basis texture.
    pub fn basis_mode(&self) -> Option<BasisMode> {
        self.basis_global_data.as_ref().map(|bg| bg.mode)
    }

    /// Check if this texture uses Basis Universal supercompression.
    pub fn is_basis(&self) -> bool {
        self.basis_global_data.is_some()
            || self.header.supercompression_scheme == SupercompressionScheme::BasisLz
    }

    /// Check if this is a UASTC Basis texture.
    pub fn is_uastc(&self) -> bool {
        self.basis_mode() == Some(BasisMode::Uastc)
    }

    /// Check if this is an ETC1S Basis texture.
    pub fn is_etc1s(&self) -> bool {
        self.basis_mode() == Some(BasisMode::Etc1s)
    }

    /// Get a key-value pair by key name.
    pub fn get_key_value(&self, key: &str) -> Option<&[u8]> {
        self.key_values
            .iter()
            .find(|kv| kv.key == key)
            .map(|kv| kv.value.as_slice())
    }

    /// Get metadata for a specific mip level.
    pub fn get_level_metadata(&self, level: u32) -> Option<&Ktx2LevelIndex> {
        self.level_indices.get(level as usize)
    }
}

// ===========================================================================
// KTX2 Parser Implementation
// ===========================================================================

/// KTX2 parser implementation.
pub struct Ktx2Parser;

impl Ktx2Parser {
    /// Check if data appears to be a KTX2 file.
    #[inline]
    pub fn is_ktx2(data: &[u8]) -> bool {
        data.len() >= 12 && data[0..12] == KTX2_IDENTIFIER
    }

    /// Parse a KTX2 file from raw bytes.
    pub fn parse(data: &[u8]) -> Result<Ktx2Texture, TextureImportError> {
        // Minimum size: identifier (12) + header (68) = 80 bytes
        if data.len() < 80 {
            return Err(TextureImportError::InvalidData(
                "KTX2 file too short for header".to_string(),
            ));
        }

        // Check identifier
        if data[0..12] != KTX2_IDENTIFIER {
            return Err(TextureImportError::InvalidData(
                "Invalid KTX2 identifier".to_string(),
            ));
        }

        // Parse header (36 bytes starting at offset 12)
        let header = Self::parse_header(&data[12..48])?;

        // Validate header
        Self::validate_header(&header)?;

        // Parse index section
        // KTX2 header layout:
        // - Identifier: 12 bytes (0-11)
        // - Header fields (vkFormat through supercompressionScheme): 36 bytes (12-47)
        // - Index section (dfdByteOffset through sgdByteLength): 32 bytes (48-79)
        // - Level indices: starts at offset 80
        if data.len() < 80 {
            return Err(TextureImportError::InvalidData(
                "KTX2 file too short for index section".to_string(),
            ));
        }

        let read_u32 = |offset: usize| -> u32 {
            u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        };

        let read_u64 = |offset: usize| -> u64 {
            u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ])
        };

        // Index section is at offsets 48-79
        let dfd_byte_offset = read_u32(48) as usize;
        let dfd_byte_length = read_u32(52) as usize;
        let kvd_byte_offset = read_u32(56) as usize;
        let kvd_byte_length = read_u32(60) as usize;
        let sgd_byte_offset = read_u64(64) as usize;
        let sgd_byte_length = read_u64(72) as usize;

        // Parse level indices (starts at offset 80)
        // Each level index is 24 bytes: byte_offset (8), byte_length (8), uncompressed_byte_length (8)
        let level_count = header.level_count.max(1);
        let level_index_start = 80;
        let level_index_end = level_index_start + (level_count as usize * 24);

        if data.len() < level_index_end {
            return Err(TextureImportError::InvalidData(
                "KTX2 file too short for level indices".to_string(),
            ));
        }

        let mut level_indices = Vec::with_capacity(level_count as usize);
        for i in 0..level_count as usize {
            let offset = level_index_start + i * 24;
            level_indices.push(Ktx2LevelIndex {
                byte_offset: read_u64(offset),
                byte_length: read_u64(offset + 8),
                uncompressed_byte_length: read_u64(offset + 16),
            });
        }

        // Parse DFD if present
        let dfd = if dfd_byte_length > 0 && dfd_byte_offset > 0 {
            Self::parse_dfd(data, dfd_byte_offset, dfd_byte_length)?
        } else {
            None
        };

        // Parse key-value data if present
        let key_values = if kvd_byte_length > 0 && kvd_byte_offset > 0 {
            Self::parse_key_values(data, kvd_byte_offset, kvd_byte_length)?
        } else {
            Vec::new()
        };

        // Parse supercompression global data if present
        let basis_global_data =
            if header.supercompression_scheme == SupercompressionScheme::BasisLz
                && sgd_byte_length > 0
                && sgd_byte_offset > 0
            {
                Self::parse_basis_global_data(data, sgd_byte_offset, sgd_byte_length)?
            } else {
                // Check for UASTC in DFD even without BasisLZ supercompression
                Self::detect_uastc_from_dfd(&dfd)
            };

        // Determine texture type
        let texture_type = Self::determine_texture_type(&header);

        // Determine if sRGB
        let is_srgb = header.vk_format.is_srgb()
            || dfd.as_ref().map(|d| d.is_srgb()).unwrap_or(false);

        // Get compressed format
        let compressed_format = header.vk_format.to_compressed_format();

        // Calculate data offset (first level's offset)
        let data_offset = level_indices
            .first()
            .map(|l| l.byte_offset as usize)
            .unwrap_or(level_index_end);

        Ok(Ktx2Texture {
            header: header.clone(),
            level_indices,
            dfd,
            key_values,
            basis_global_data,
            texture_type,
            width: header.pixel_width,
            height: header.pixel_height.max(1),
            depth: header.pixel_depth.max(1),
            mip_count: level_count,
            array_count: header.layer_count.max(1),
            face_count: header.face_count,
            is_srgb,
            compressed_format,
            data_offset,
        })
    }

    /// Parse the KTX2 header (36 bytes starting at offset 12).
    fn parse_header(data: &[u8]) -> Result<Ktx2Header, TextureImportError> {
        let read_u32 = |offset: usize| -> u32 {
            u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        };

        let vk_format_raw = read_u32(0);
        let type_size = read_u32(4);
        let pixel_width = read_u32(8);
        let pixel_height = read_u32(12);
        let pixel_depth = read_u32(16);
        let layer_count = read_u32(20);
        let face_count = read_u32(24);
        let level_count = read_u32(28);
        let supercompression_scheme_raw = read_u32(32);

        Ok(Ktx2Header {
            vk_format: VkFormat::from_u32(vk_format_raw),
            type_size,
            pixel_width,
            pixel_height,
            pixel_depth,
            layer_count,
            face_count,
            level_count,
            supercompression_scheme: SupercompressionScheme::from_u32(supercompression_scheme_raw),
        })
    }

    /// Validate the KTX2 header.
    fn validate_header(header: &Ktx2Header) -> Result<(), TextureImportError> {
        // Width must be non-zero
        if header.pixel_width == 0 {
            return Err(TextureImportError::InvalidDimensions {
                width: 0,
                height: header.pixel_height,
            });
        }

        // Face count must be 1 or 6
        if header.face_count != 1 && header.face_count != 6 {
            return Err(TextureImportError::InvalidData(format!(
                "Invalid face count: {} (must be 1 or 6)",
                header.face_count
            )));
        }

        // Cubemap must be square
        if header.face_count == 6 && header.pixel_width != header.pixel_height {
            return Err(TextureImportError::InvalidData(
                "Cubemap faces must be square".to_string(),
            ));
        }

        Ok(())
    }

    /// Parse the Data Format Descriptor.
    fn parse_dfd(
        data: &[u8],
        offset: usize,
        length: usize,
    ) -> Result<Option<Ktx2Dfd>, TextureImportError> {
        if offset + length > data.len() || length < 24 {
            return Ok(None);
        }

        let read_u32 = |off: usize| -> u32 {
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        };

        let read_u16 = |off: usize| -> u16 { u16::from_le_bytes([data[off], data[off + 1]]) };

        // DFD total size
        let _dfd_total_size = read_u32(offset);

        // Basic descriptor block starts at offset + 4
        let block_offset = offset + 4;

        let vendor_id = read_u16(block_offset) as u32;
        let descriptor_type = read_u16(block_offset + 2) as u32;
        let version_number = data[block_offset + 4] as u32;
        let descriptor_block_size = read_u16(block_offset + 5) as u32;

        let color_model = data[block_offset + 8];
        let color_primaries = data[block_offset + 9];
        let transfer_function = data[block_offset + 10];
        let flags = data[block_offset + 11];

        let texel_block_dimension = [
            data[block_offset + 12],
            data[block_offset + 13],
            data[block_offset + 14],
            data[block_offset + 15],
        ];

        let bytes_plane = [
            data[block_offset + 16],
            data[block_offset + 17],
            data[block_offset + 18],
            data[block_offset + 19],
            data[block_offset + 20],
            data[block_offset + 21],
            data[block_offset + 22],
            data[block_offset + 23],
        ];

        // Parse samples (each sample is 16 bytes)
        let samples_offset = block_offset + 24;
        let remaining = (descriptor_block_size as usize).saturating_sub(24);
        let sample_count = remaining / 16;
        let mut samples = Vec::with_capacity(sample_count);

        for i in 0..sample_count {
            let s_off = samples_offset + i * 16;
            if s_off + 16 > data.len() {
                break;
            }
            samples.push(DfdSample {
                bit_offset: read_u16(s_off),
                bit_length: data[s_off + 2],
                channel_type: data[s_off + 3],
                sample_position: read_u32(s_off + 4),
                sample_lower: read_u32(s_off + 8),
                sample_upper: read_u32(s_off + 12),
            });
        }

        Ok(Some(Ktx2Dfd {
            vendor_id,
            descriptor_type,
            version_number,
            descriptor_block_size,
            color_model,
            color_primaries,
            transfer_function,
            flags,
            texel_block_dimension,
            bytes_plane,
            samples,
        }))
    }

    /// Parse key-value data section.
    fn parse_key_values(
        data: &[u8],
        offset: usize,
        length: usize,
    ) -> Result<Vec<Ktx2KeyValue>, TextureImportError> {
        if offset + length > data.len() {
            return Ok(Vec::new());
        }

        let mut key_values = Vec::new();
        let mut pos = offset;
        let end = offset + length;

        while pos + 4 <= end {
            // Key-value entry: keyAndValueByteLength (4 bytes), key (NUL-terminated), value
            let entry_length = u32::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
            ]) as usize;

            pos += 4;

            if entry_length == 0 || pos + entry_length > end {
                break;
            }

            // Find NUL terminator for key
            let key_end = data[pos..pos + entry_length]
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(entry_length);

            let key = String::from_utf8_lossy(&data[pos..pos + key_end]).to_string();
            let value_start = pos + key_end + 1;
            let value_end = pos + entry_length;

            let value = if value_start < value_end {
                data[value_start..value_end].to_vec()
            } else {
                Vec::new()
            };

            key_values.push(Ktx2KeyValue { key, value });

            // Move to next entry (4-byte aligned)
            pos += entry_length;
            pos = (pos + 3) & !3;
        }

        Ok(key_values)
    }

    /// Parse Basis Universal global data.
    fn parse_basis_global_data(
        data: &[u8],
        offset: usize,
        length: usize,
    ) -> Result<Option<BasisGlobalData>, TextureImportError> {
        if offset + length > data.len() || length < 20 {
            return Ok(None);
        }

        let read_u16 = |off: usize| -> u16 { u16::from_le_bytes([data[off], data[off + 1]]) };
        let read_u32 = |off: usize| -> u32 {
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        };

        // SGD header
        let endpoint_count = read_u16(offset) as u32;
        let selector_count = read_u16(offset + 2) as u32;
        let endpoints_byte_length = read_u32(offset + 4);
        let selectors_byte_length = read_u32(offset + 8);
        let tables_byte_length = read_u32(offset + 12);
        let extended_byte_length = read_u32(offset + 16);

        // Determine mode from DFD or format
        // UASTC uses different DFD color model (166 = UASTC)
        // ETC1S uses BasisLZ supercompression
        let mode = BasisMode::Etc1s; // Default for BasisLZ

        // Check flags for alpha
        let has_alpha = false; // Would need to check DFD

        // Extract data sections
        let mut pos = offset + 20;
        let endpoints = if endpoints_byte_length > 0 && pos + endpoints_byte_length as usize <= data.len() {
            let e = data[pos..pos + endpoints_byte_length as usize].to_vec();
            pos += endpoints_byte_length as usize;
            e
        } else {
            Vec::new()
        };

        let selectors = if selectors_byte_length > 0 && pos + selectors_byte_length as usize <= data.len() {
            let s = data[pos..pos + selectors_byte_length as usize].to_vec();
            pos += selectors_byte_length as usize;
            s
        } else {
            Vec::new()
        };

        let tables = if tables_byte_length > 0 && pos + tables_byte_length as usize <= data.len() {
            data[pos..pos + tables_byte_length as usize].to_vec()
        } else {
            Vec::new()
        };

        Ok(Some(BasisGlobalData {
            endpoint_count,
            selector_count,
            endpoints_byte_length,
            selectors_byte_length,
            tables_byte_length,
            extended_byte_length,
            mode,
            has_alpha,
            endpoints,
            selectors,
            tables,
        }))
    }

    /// Detect UASTC from DFD (color model 166).
    fn detect_uastc_from_dfd(dfd: &Option<Ktx2Dfd>) -> Option<BasisGlobalData> {
        // KHR_DF_MODEL_UASTC = 166
        const KHR_DF_MODEL_UASTC: u8 = 166;

        if let Some(dfd) = dfd {
            if dfd.color_model == KHR_DF_MODEL_UASTC {
                return Some(BasisGlobalData {
                    endpoint_count: 0,
                    selector_count: 0,
                    endpoints_byte_length: 0,
                    selectors_byte_length: 0,
                    tables_byte_length: 0,
                    extended_byte_length: 0,
                    mode: BasisMode::Uastc,
                    has_alpha: false,
                    endpoints: Vec::new(),
                    selectors: Vec::new(),
                    tables: Vec::new(),
                });
            }
        }
        None
    }

    /// Determine texture type from header.
    fn determine_texture_type(header: &Ktx2Header) -> TextureType {
        let is_array = header.layer_count > 0;
        let is_cubemap = header.face_count == 6;
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

    /// Get raw mip level data from the file.
    pub fn get_level_data<'a>(
        data: &'a [u8],
        texture: &Ktx2Texture,
        level: u32,
    ) -> Option<&'a [u8]> {
        let index = texture.level_indices.get(level as usize)?;
        let start = index.byte_offset as usize;
        let end = start + index.byte_length as usize;

        if end <= data.len() {
            Some(&data[start..end])
        } else {
            None
        }
    }

    /// Get uncompressed size for a mip level.
    pub fn get_level_uncompressed_size(texture: &Ktx2Texture, level: u32) -> Option<usize> {
        texture
            .level_indices
            .get(level as usize)
            .map(|idx| idx.uncompressed_byte_length as usize)
    }
}

// ===========================================================================
// Basis Universal Transcoder
// ===========================================================================

/// Basis Universal transcoder for KTX2 textures.
///
/// This provides the transcoding interface for Basis Universal supercompressed
/// textures (UASTC and ETC1S modes).
pub struct BasisTranscoder;

impl BasisTranscoder {
    /// Check if transcoding is supported for this texture and target.
    pub fn can_transcode(texture: &Ktx2Texture, target: TranscodeTarget) -> bool {
        if !texture.is_basis() {
            return false;
        }

        match (texture.basis_mode(), target) {
            // UASTC can transcode to all targets
            (Some(BasisMode::Uastc), _) => true,
            // ETC1S has some limitations
            (Some(BasisMode::Etc1s), TranscodeTarget::Bc7) => true,
            (Some(BasisMode::Etc1s), TranscodeTarget::Bc1) => true,
            (Some(BasisMode::Etc1s), TranscodeTarget::Bc3) => true,
            (Some(BasisMode::Etc1s), TranscodeTarget::Etc1) => true,
            (Some(BasisMode::Etc1s), TranscodeTarget::Etc2Rgb) => true,
            (Some(BasisMode::Etc1s), TranscodeTarget::Etc2Rgba) => true,
            (Some(BasisMode::Etc1s), TranscodeTarget::Rgba8) => true,
            (Some(BasisMode::Etc1s), TranscodeTarget::Astc4x4) => true,
            _ => false,
        }
    }

    /// Transcode a Basis Universal texture to the target format.
    ///
    /// This is a stub implementation. Full transcoding requires the basis-universal
    /// library or a Rust implementation of the transcoder.
    pub fn transcode(
        data: &[u8],
        texture: &Ktx2Texture,
        target: TranscodeTarget,
        level: u32,
    ) -> Result<Vec<u8>, TextureImportError> {
        if !Self::can_transcode(texture, target) {
            return Err(TextureImportError::UnsupportedFormat(format!(
                "Cannot transcode {:?} to {:?}",
                texture.basis_mode(),
                target
            )));
        }

        let level_data = Ktx2Parser::get_level_data(data, texture, level).ok_or_else(|| {
            TextureImportError::InvalidData(format!("Level {} not found", level))
        })?;

        // Calculate output size
        let width = (texture.width >> level).max(1);
        let height = (texture.height >> level).max(1);

        let output_size = match target {
            TranscodeTarget::Rgba8 => (width * height * 4) as usize,
            TranscodeTarget::Rgb565 | TranscodeTarget::Rgba4444 => (width * height * 2) as usize,
            _ => {
                // Block-compressed formats
                let block_w = (width + 3) / 4;
                let block_h = (height + 3) / 4;
                (block_w * block_h) as usize * target.bytes_per_unit()
            }
        };

        // Stub: In a real implementation, this would call the basis_universal transcoder
        // For now, we return placeholder data for testing
        match texture.basis_mode() {
            Some(BasisMode::Uastc) => {
                // UASTC transcoding stub
                Self::transcode_uastc(level_data, width, height, target, output_size)
            }
            Some(BasisMode::Etc1s) => {
                // ETC1S transcoding stub
                Self::transcode_etc1s(
                    level_data,
                    texture.basis_global_data.as_ref(),
                    width,
                    height,
                    target,
                    output_size,
                )
            }
            None => Err(TextureImportError::InvalidData(
                "Not a Basis Universal texture".to_string(),
            )),
        }
    }

    /// Transcode UASTC data (stub implementation).
    fn transcode_uastc(
        _data: &[u8],
        _width: u32,
        _height: u32,
        target: TranscodeTarget,
        output_size: usize,
    ) -> Result<Vec<u8>, TextureImportError> {
        // UASTC is a fixed-rate 128-bit per 4x4 block format
        // Real implementation would decode UASTC blocks and re-encode to target

        // For testing, return zeroed output of correct size
        match target {
            TranscodeTarget::Rgba8 => {
                // Return magenta pixels for visual debugging
                let mut output = Vec::with_capacity(output_size);
                for _ in 0..output_size / 4 {
                    output.extend_from_slice(&[255, 0, 255, 255]); // Magenta
                }
                Ok(output)
            }
            _ => {
                // Return zeroed compressed data
                Ok(vec![0u8; output_size])
            }
        }
    }

    /// Transcode ETC1S data (stub implementation).
    fn transcode_etc1s(
        _data: &[u8],
        _global_data: Option<&BasisGlobalData>,
        _width: u32,
        _height: u32,
        target: TranscodeTarget,
        output_size: usize,
    ) -> Result<Vec<u8>, TextureImportError> {
        // ETC1S uses a global codebook (endpoints + selectors)
        // Real implementation would:
        // 1. Decode endpoint/selector indices from slice data
        // 2. Look up endpoints/selectors from global data
        // 3. Reconstruct ETC1 blocks
        // 4. Transcode to target format

        match target {
            TranscodeTarget::Rgba8 => {
                // Return cyan pixels for visual debugging (different from UASTC)
                let mut output = Vec::with_capacity(output_size);
                for _ in 0..output_size / 4 {
                    output.extend_from_slice(&[0, 255, 255, 255]); // Cyan
                }
                Ok(output)
            }
            _ => {
                // Return zeroed compressed data
                Ok(vec![0u8; output_size])
            }
        }
    }
}

// ===========================================================================
// Transcoded Texture Asset
// ===========================================================================

/// A transcoded KTX2 texture asset ready for GPU upload.
#[derive(Debug, Clone)]
pub struct Ktx2TextureAsset {
    /// Unique asset ID
    pub id: u64,
    /// Width at base level
    pub width: u32,
    /// Height at base level
    pub height: u32,
    /// Depth at base level
    pub depth: u32,
    /// Original Basis mode (if applicable)
    pub basis_mode: Option<BasisMode>,
    /// Transcoded format
    pub transcoded_format: Option<TranscodeTarget>,
    /// Compressed format (from original or transcoding)
    pub compressed_format: Option<CompressedFormat>,
    /// Texture type
    pub texture_type: TextureType,
    /// Is sRGB
    pub is_srgb: bool,
    /// Number of mip levels
    pub mip_count: u32,
    /// Number of array layers
    pub array_count: u32,
    /// Number of faces
    pub face_count: u32,
    /// Transcoded/decompressed data per level
    pub level_data: Vec<Vec<u8>>,
    /// Total memory size
    pub memory_size: usize,
    /// Current state
    pub state: TextureState,
}

impl Ktx2TextureAsset {
    /// Check if texture is ready.
    pub fn is_ready(&self) -> bool {
        self.state == TextureState::Ready
    }

    /// Get data for a specific mip level.
    pub fn get_level_data(&self, level: u32) -> Option<&[u8]> {
        self.level_data.get(level as usize).map(|v| v.as_slice())
    }
}

// ===========================================================================
// Extension Detection
// ===========================================================================

/// Check if a file extension indicates KTX2 format.
#[inline]
pub fn is_ktx2_extension(ext: &str) -> bool {
    let ext_lower = ext.to_lowercase();
    ext_lower == "ktx2"
}

/// Detect KTX2 from magic bytes.
#[inline]
pub fn detect_ktx2_format(data: &[u8]) -> bool {
    Ktx2Parser::is_ktx2(data)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test helpers
    // -------------------------------------------------------------------------

    fn create_minimal_ktx2() -> Vec<u8> {
        let mut data = Vec::new();

        // KTX2 layout:
        // - Identifier: 12 bytes (0-11)
        // - Header fields: 36 bytes (12-47)
        // - Index section: 32 bytes (48-79)
        // - Level indices: 24 bytes per level (80+)

        // Identifier (12 bytes)
        data.extend_from_slice(&KTX2_IDENTIFIER);

        // Header fields (36 bytes, offset 12-47)
        data.extend_from_slice(&37u32.to_le_bytes()); // vkFormat = R8G8B8A8_UNORM
        data.extend_from_slice(&1u32.to_le_bytes()); // typeSize
        data.extend_from_slice(&64u32.to_le_bytes()); // pixelWidth
        data.extend_from_slice(&64u32.to_le_bytes()); // pixelHeight
        data.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth
        data.extend_from_slice(&0u32.to_le_bytes()); // layerCount
        data.extend_from_slice(&1u32.to_le_bytes()); // faceCount
        data.extend_from_slice(&1u32.to_le_bytes()); // levelCount
        data.extend_from_slice(&0u32.to_le_bytes()); // supercompressionScheme = None

        // Index section (32 bytes, offset 48-79)
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteLength
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteLength
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteOffset
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteLength

        // Level index (24 bytes, offset 80-103)
        let data_offset = 80 + 24; // After level index
        let data_size = 64 * 64 * 4; // RGBA8
        data.extend_from_slice(&(data_offset as u64).to_le_bytes());
        data.extend_from_slice(&(data_size as u64).to_le_bytes());
        data.extend_from_slice(&(data_size as u64).to_le_bytes());

        // Image data (at offset 104)
        data.extend(std::iter::repeat(0u8).take(data_size as usize));

        data
    }

    fn create_ktx2_bc7() -> Vec<u8> {
        let mut data = Vec::new();

        // KTX2 layout:
        // - Identifier: 12 bytes (0-11)
        // - Header fields: 36 bytes (12-47)
        // - Index section: 32 bytes (48-79)
        // - Level indices: 24 bytes per level (80+)

        // Identifier (12 bytes)
        data.extend_from_slice(&KTX2_IDENTIFIER);

        // Header fields (36 bytes, offset 12-47)
        data.extend_from_slice(&145u32.to_le_bytes()); // vkFormat (VK_FORMAT_BC7_UNORM_BLOCK)
        data.extend_from_slice(&1u32.to_le_bytes()); // typeSize
        data.extend_from_slice(&128u32.to_le_bytes()); // pixelWidth
        data.extend_from_slice(&128u32.to_le_bytes()); // pixelHeight
        data.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth
        data.extend_from_slice(&0u32.to_le_bytes()); // layerCount
        data.extend_from_slice(&1u32.to_le_bytes()); // faceCount
        data.extend_from_slice(&1u32.to_le_bytes()); // levelCount
        data.extend_from_slice(&0u32.to_le_bytes()); // supercompressionScheme

        // Index section (32 bytes, offset 48-79)
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteLength
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteLength
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteOffset
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteLength

        // Level index (24 bytes, offset 80-103)
        let data_offset = 80 + 24; // After level index
        let data_size = (128 / 4) * (128 / 4) * 16; // BC7: 32x32 blocks * 16 bytes
        data.extend_from_slice(&(data_offset as u64).to_le_bytes());
        data.extend_from_slice(&(data_size as u64).to_le_bytes());
        data.extend_from_slice(&(data_size as u64).to_le_bytes());

        // Image data (at offset 104)
        data.extend(std::iter::repeat(0u8).take(data_size as usize));

        data
    }

    fn create_ktx2_cubemap() -> Vec<u8> {
        let mut data = Vec::new();

        // KTX2 layout:
        // - Identifier: 12 bytes (0-11)
        // - Header fields: 36 bytes (12-47)
        // - Index section: 32 bytes (48-79)
        // - Level indices: 24 bytes per level (80+)

        // Identifier (12 bytes)
        data.extend_from_slice(&KTX2_IDENTIFIER);

        // Header fields (36 bytes, offset 12-47)
        data.extend_from_slice(&37u32.to_le_bytes()); // vkFormat (RGBA8)
        data.extend_from_slice(&1u32.to_le_bytes()); // typeSize
        data.extend_from_slice(&32u32.to_le_bytes()); // pixelWidth
        data.extend_from_slice(&32u32.to_le_bytes()); // pixelHeight
        data.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth
        data.extend_from_slice(&0u32.to_le_bytes()); // layerCount
        data.extend_from_slice(&6u32.to_le_bytes()); // faceCount = 6 for cubemap
        data.extend_from_slice(&1u32.to_le_bytes()); // levelCount
        data.extend_from_slice(&0u32.to_le_bytes()); // supercompressionScheme

        // Index section (32 bytes, offset 48-79)
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteLength
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteLength
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteOffset
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteLength

        // Level index (24 bytes, offset 80-103)
        let data_offset = 80 + 24; // After level index
        let data_size = 32 * 32 * 4 * 6; // RGBA8 * 6 faces
        data.extend_from_slice(&(data_offset as u64).to_le_bytes());
        data.extend_from_slice(&(data_size as u64).to_le_bytes());
        data.extend_from_slice(&(data_size as u64).to_le_bytes());

        // Image data (at offset 104)
        data.extend(std::iter::repeat(0u8).take(data_size as usize));

        data
    }

    fn create_ktx2_array() -> Vec<u8> {
        let mut data = Vec::new();

        // KTX2 layout:
        // - Identifier: 12 bytes (0-11)
        // - Header fields: 36 bytes (12-47)
        // - Index section: 32 bytes (48-79)
        // - Level indices: 24 bytes per level (80+)

        // Identifier (12 bytes)
        data.extend_from_slice(&KTX2_IDENTIFIER);

        // Header fields (36 bytes, offset 12-47)
        data.extend_from_slice(&37u32.to_le_bytes()); // vkFormat (RGBA8)
        data.extend_from_slice(&1u32.to_le_bytes()); // typeSize
        data.extend_from_slice(&64u32.to_le_bytes()); // pixelWidth
        data.extend_from_slice(&64u32.to_le_bytes()); // pixelHeight
        data.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth
        data.extend_from_slice(&4u32.to_le_bytes()); // layerCount = 4 for array
        data.extend_from_slice(&1u32.to_le_bytes()); // faceCount
        data.extend_from_slice(&1u32.to_le_bytes()); // levelCount
        data.extend_from_slice(&0u32.to_le_bytes()); // supercompressionScheme

        // Index section (32 bytes, offset 48-79)
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteLength
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteLength
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteOffset
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteLength

        // Level index (24 bytes, offset 80-103)
        let data_offset = 80 + 24; // After level index
        let data_size = 64 * 64 * 4 * 4; // RGBA8 * 4 layers
        data.extend_from_slice(&(data_offset as u64).to_le_bytes());
        data.extend_from_slice(&(data_size as u64).to_le_bytes());
        data.extend_from_slice(&(data_size as u64).to_le_bytes());

        // Image data (at offset 104)
        data.extend(std::iter::repeat(0u8).take(data_size as usize));

        data
    }

    fn create_ktx2_mipmaps() -> Vec<u8> {
        let mut data = Vec::new();

        // KTX2 layout:
        // - Identifier: 12 bytes (0-11)
        // - Header fields: 36 bytes (12-47)
        // - Index section: 32 bytes (48-79)
        // - Level indices: 24 bytes per level (80+)

        // Identifier (12 bytes)
        data.extend_from_slice(&KTX2_IDENTIFIER);

        // Header fields with 7 mip levels (64, 32, 16, 8, 4, 2, 1)
        data.extend_from_slice(&37u32.to_le_bytes()); // vkFormat (RGBA8)
        data.extend_from_slice(&1u32.to_le_bytes()); // typeSize
        data.extend_from_slice(&64u32.to_le_bytes()); // pixelWidth
        data.extend_from_slice(&64u32.to_le_bytes()); // pixelHeight
        data.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth
        data.extend_from_slice(&0u32.to_le_bytes()); // layerCount
        data.extend_from_slice(&1u32.to_le_bytes()); // faceCount
        data.extend_from_slice(&7u32.to_le_bytes()); // levelCount = 7
        data.extend_from_slice(&0u32.to_le_bytes()); // supercompressionScheme

        // Index section (32 bytes, offset 48-79)
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteLength
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteLength
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteOffset
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteLength

        // Calculate level sizes and offsets
        let level_sizes: Vec<usize> = (0..7)
            .map(|i| {
                let w = (64 >> i).max(1);
                let h = (64 >> i).max(1);
                w * h * 4
            })
            .collect();

        // Level indices (7 levels * 24 bytes each, starting at offset 80)
        let mut offset = 80 + (7 * 24); // After all level indices
        for size in &level_sizes {
            data.extend_from_slice(&(offset as u64).to_le_bytes());
            data.extend_from_slice(&(*size as u64).to_le_bytes());
            data.extend_from_slice(&(*size as u64).to_le_bytes());
            offset += size;
        }

        // Image data for all levels
        for size in &level_sizes {
            data.extend(std::iter::repeat(0u8).take(*size));
        }

        data
    }

    fn create_ktx2_uastc() -> Vec<u8> {
        let mut data = Vec::new();

        // KTX2 file layout:
        // Identifier: 12 bytes (0-11)
        // Header fields: 36 bytes (12-47)
        // Index section: 32 bytes (48-79)
        // Level indices: 24 bytes per level (80+)
        // Then DFD, KVD, SGD, and image data

        // Identifier (12 bytes)
        data.extend_from_slice(&KTX2_IDENTIFIER);

        // Header fields (36 bytes, offset 12-47)
        data.extend_from_slice(&0u32.to_le_bytes()); // vkFormat = UNDEFINED
        data.extend_from_slice(&1u32.to_le_bytes()); // typeSize
        data.extend_from_slice(&64u32.to_le_bytes()); // pixelWidth
        data.extend_from_slice(&64u32.to_le_bytes()); // pixelHeight
        data.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth
        data.extend_from_slice(&0u32.to_le_bytes()); // layerCount
        data.extend_from_slice(&1u32.to_le_bytes()); // faceCount
        data.extend_from_slice(&1u32.to_le_bytes()); // levelCount
        data.extend_from_slice(&0u32.to_le_bytes()); // supercompressionScheme = None

        // Index section (32 bytes, offset 48-79)
        // DFD comes after level indices
        let level_index_start = 80;
        let level_index_size = 24; // 1 level * 24 bytes
        let dfd_offset = level_index_start + level_index_size; // 104
        let dfd_size = 48; // DFD total size including 4-byte size prefix

        data.extend_from_slice(&(dfd_offset as u32).to_le_bytes()); // dfdByteOffset
        data.extend_from_slice(&(dfd_size as u32).to_le_bytes()); // dfdByteLength
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteLength
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteOffset
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteLength

        // Level index (24 bytes, offset 80-103)
        let level_data_offset = dfd_offset + dfd_size; // 152
        let level_data_size = (64 / 4) * (64 / 4) * 16; // UASTC: 16 bytes per 4x4 block = 4096
        data.extend_from_slice(&(level_data_offset as u64).to_le_bytes());
        data.extend_from_slice(&(level_data_size as u64).to_le_bytes());
        data.extend_from_slice(&(level_data_size as u64).to_le_bytes());

        // DFD with UASTC color model (166) - 48 bytes at offset 104
        // DFD total size (4 bytes)
        data.extend_from_slice(&(dfd_size as u32).to_le_bytes());

        // Basic descriptor block (44 bytes)
        data.extend_from_slice(&0u16.to_le_bytes()); // vendorId (2)
        data.extend_from_slice(&0u16.to_le_bytes()); // descriptorType (2)
        data.push(0); // versionNumber (1)
        data.extend_from_slice(&44u16.to_le_bytes()); // descriptorBlockSize (2)
        data.push(0); // reserved (1)
        data.push(166u8); // colorModel = UASTC (1)
        data.push(1u8); // colorPrimaries = BT709 (1)
        data.push(2u8); // transferFunction = SRGB (1)
        data.push(0u8); // flags (1)
        // texelBlockDimension (4)
        data.extend_from_slice(&[3, 3, 0, 0]); // 4x4 block (0-indexed)
        // bytesPlane (8)
        data.extend_from_slice(&[16, 0, 0, 0, 0, 0, 0, 0]);
        // Sample (16 bytes)
        data.extend_from_slice(&0u16.to_le_bytes()); // bitOffset (2)
        data.push(127); // bitLength (1)
        data.push(0); // channelType (1)
        data.extend_from_slice(&0u32.to_le_bytes()); // samplePosition (4)
        data.extend_from_slice(&0u32.to_le_bytes()); // sampleLower (4)
        data.extend_from_slice(&0xFFFFFFFFu32.to_le_bytes()); // sampleUpper (4)
        // Padding to match dfd_size = 48 (4 + 44 = 48)
        data.extend(std::iter::repeat(0u8).take(4));
        // Total: 4 + 2+2+1+2+1+1+1+1+1+4+8+2+1+1+4+4+4+4 = 4 + 44 = 48 bytes

        // Image data (4096 bytes at offset 152)
        data.extend(std::iter::repeat(0u8).take(level_data_size as usize));

        data
    }

    fn create_ktx2_etc1s() -> Vec<u8> {
        let mut data = Vec::new();

        // KTX2 layout:
        // - Identifier: 12 bytes (0-11)
        // - Header fields: 36 bytes (12-47)
        // - Index section: 32 bytes (48-79)
        // - Level indices: 24 bytes per level (80+)
        // - Then SGD for BasisLZ

        // Identifier (12 bytes)
        data.extend_from_slice(&KTX2_IDENTIFIER);

        // Header fields (36 bytes, offset 12-47)
        data.extend_from_slice(&0u32.to_le_bytes()); // vkFormat = UNDEFINED
        data.extend_from_slice(&1u32.to_le_bytes()); // typeSize
        data.extend_from_slice(&64u32.to_le_bytes()); // pixelWidth
        data.extend_from_slice(&64u32.to_le_bytes()); // pixelHeight
        data.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth
        data.extend_from_slice(&0u32.to_le_bytes()); // layerCount
        data.extend_from_slice(&1u32.to_le_bytes()); // faceCount
        data.extend_from_slice(&1u32.to_le_bytes()); // levelCount
        data.extend_from_slice(&1u32.to_le_bytes()); // supercompressionScheme = BasisLZ

        // Calculate offsets
        let level_index_start = 80;
        let level_index_size = 24; // 1 level * 24 bytes
        let sgd_offset = level_index_start + level_index_size; // 104
        let sgd_size = 60; // SGD header (20) + endpoints (16) + selectors (16) + tables (8) = 60

        // Index section (32 bytes, offset 48-79)
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteLength
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteLength
        data.extend_from_slice(&(sgd_offset as u64).to_le_bytes()); // sgdByteOffset
        data.extend_from_slice(&(sgd_size as u64).to_le_bytes()); // sgdByteLength

        // Level index (24 bytes, offset 80-103)
        let level_data_offset = sgd_offset + sgd_size; // 164
        let level_data_size = 1024; // Compressed ETC1S data
        data.extend_from_slice(&(level_data_offset as u64).to_le_bytes());
        data.extend_from_slice(&(level_data_size as u64).to_le_bytes());
        data.extend_from_slice(&(level_data_size as u64).to_le_bytes());

        // SGD (Supercompression Global Data) - 60 bytes at offset 104
        data.extend_from_slice(&256u16.to_le_bytes()); // endpointCount (2)
        data.extend_from_slice(&256u16.to_le_bytes()); // selectorCount (2)
        data.extend_from_slice(&16u32.to_le_bytes()); // endpointsByteLength (4)
        data.extend_from_slice(&16u32.to_le_bytes()); // selectorsByteLength (4)
        data.extend_from_slice(&8u32.to_le_bytes()); // tablesByteLength (4)
        data.extend_from_slice(&0u32.to_le_bytes()); // extendedByteLength (4)
        // Total header: 2+2+4+4+4+4 = 20 bytes
        // Endpoints data (16 bytes)
        data.extend(std::iter::repeat(0u8).take(16));
        // Selectors data (16 bytes)
        data.extend(std::iter::repeat(0u8).take(16));
        // Tables data (8 bytes)
        data.extend(std::iter::repeat(0u8).take(8));
        // Total SGD: 20 + 16 + 16 + 8 = 60 bytes

        // Image data (1024 bytes at offset 164)
        data.extend(std::iter::repeat(0u8).take(level_data_size as usize));

        data
    }

    // -------------------------------------------------------------------------
    // KTX2 Header Parsing Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ktx2_is_ktx2() {
        assert!(Ktx2Parser::is_ktx2(&KTX2_IDENTIFIER));
        let ktx1_id = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];
        assert!(!Ktx2Parser::is_ktx2(&ktx1_id)); // KTX1 should not match
        assert!(!Ktx2Parser::is_ktx2(&[0, 1, 2, 3]));
    }

    #[test]
    fn test_ktx2_parse_header() {
        let data = create_minimal_ktx2();
        let result = Ktx2Parser::parse(&data);
        assert!(result.is_ok(), "Parse failed: {:?}", result.err());

        let tex = result.unwrap();
        assert_eq!(tex.width, 64);
        assert_eq!(tex.height, 64);
        assert_eq!(tex.mip_count, 1);
        assert_eq!(tex.face_count, 1);
        assert_eq!(tex.texture_type, TextureType::Texture2D);
    }

    #[test]
    fn test_ktx2_parse_too_short() {
        let result = Ktx2Parser::parse(&[0u8; 50]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ktx2_parse_invalid_identifier() {
        let mut data = create_minimal_ktx2();
        data[0] = 0; // Corrupt identifier
        let result = Ktx2Parser::parse(&data);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // KTX2 Metadata Extraction Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ktx2_metadata_extraction() {
        let data = create_minimal_ktx2();
        let tex = Ktx2Parser::parse(&data).unwrap();

        assert_eq!(tex.header.vk_format, VkFormat::R8G8B8A8Unorm);
        assert_eq!(tex.header.type_size, 1);
        assert_eq!(tex.header.pixel_width, 64);
        assert_eq!(tex.header.pixel_height, 64);
        assert_eq!(tex.header.supercompression_scheme, SupercompressionScheme::None);
    }

    // -------------------------------------------------------------------------
    // KTX2 Mip Level Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ktx2_mip_levels() {
        let data = create_ktx2_mipmaps();
        let tex = Ktx2Parser::parse(&data).unwrap();

        assert_eq!(tex.mip_count, 7);
        assert_eq!(tex.level_indices.len(), 7);

        // Verify level dimensions decrease
        for i in 0..7 {
            let expected_size = (64 >> i).max(1) * (64 >> i).max(1) * 4;
            assert_eq!(
                tex.level_indices[i].uncompressed_byte_length,
                expected_size as u64
            );
        }
    }

    #[test]
    fn test_ktx2_get_level_data() {
        let data = create_minimal_ktx2();
        let tex = Ktx2Parser::parse(&data).unwrap();

        let level_data = Ktx2Parser::get_level_data(&data, &tex, 0);
        assert!(level_data.is_some());
        assert_eq!(level_data.unwrap().len(), 64 * 64 * 4);

        // Non-existent level
        let invalid = Ktx2Parser::get_level_data(&data, &tex, 5);
        assert!(invalid.is_none());
    }

    // -------------------------------------------------------------------------
    // KTX2 Cubemap Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ktx2_cubemap() {
        let data = create_ktx2_cubemap();
        let tex = Ktx2Parser::parse(&data).unwrap();

        assert_eq!(tex.texture_type, TextureType::Cubemap);
        assert_eq!(tex.face_count, 6);
        assert_eq!(tex.width, 32);
        assert_eq!(tex.height, 32);
    }

    #[test]
    fn test_ktx2_cubemap_must_be_square() {
        let mut data = create_ktx2_cubemap();
        // Make non-square (offset 20 is pixelHeight in header)
        data[12 + 12] = 48; // Set height to 48
        let result = Ktx2Parser::parse(&data);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // KTX2 Texture Array Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ktx2_texture_array() {
        let data = create_ktx2_array();
        let tex = Ktx2Parser::parse(&data).unwrap();

        assert_eq!(tex.texture_type, TextureType::Array2D);
        assert_eq!(tex.array_count, 4);
    }

    // -------------------------------------------------------------------------
    // UASTC Detection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ktx2_uastc_detection() {
        let data = create_ktx2_uastc();
        let tex = Ktx2Parser::parse(&data).unwrap();

        assert!(tex.is_basis());
        assert!(tex.is_uastc());
        assert!(!tex.is_etc1s());
        assert_eq!(tex.basis_mode(), Some(BasisMode::Uastc));
    }

    // -------------------------------------------------------------------------
    // ETC1S Detection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ktx2_etc1s_detection() {
        let data = create_ktx2_etc1s();
        let tex = Ktx2Parser::parse(&data).unwrap();

        assert!(tex.is_basis());
        assert!(tex.is_etc1s());
        assert!(!tex.is_uastc());
        assert_eq!(tex.basis_mode(), Some(BasisMode::Etc1s));
    }

    // -------------------------------------------------------------------------
    // Transcode Target Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transcode_to_bc7() {
        let data = create_ktx2_uastc();
        let tex = Ktx2Parser::parse(&data).unwrap();

        // Verify the test data is correctly sized
        assert!(tex.level_indices.len() > 0, "No level indices");
        let level = &tex.level_indices[0];
        let expected_end = level.byte_offset as usize + level.byte_length as usize;
        assert!(
            data.len() >= expected_end,
            "Data too short: {} < {} (offset={}, length={})",
            data.len(),
            expected_end,
            level.byte_offset,
            level.byte_length
        );

        assert!(BasisTranscoder::can_transcode(&tex, TranscodeTarget::Bc7));

        let result = BasisTranscoder::transcode(&data, &tex, TranscodeTarget::Bc7, 0);
        assert!(result.is_ok(), "Transcode failed: {:?}", result.err());

        let transcoded = result.unwrap();
        // BC7: (64/4) * (64/4) * 16 = 16 * 16 * 16 = 4096 bytes
        assert_eq!(transcoded.len(), 4096);
    }

    #[test]
    fn test_transcode_to_astc() {
        let data = create_ktx2_uastc();
        let tex = Ktx2Parser::parse(&data).unwrap();

        assert!(BasisTranscoder::can_transcode(&tex, TranscodeTarget::Astc4x4));

        let result = BasisTranscoder::transcode(&data, &tex, TranscodeTarget::Astc4x4, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transcode_to_etc2() {
        let data = create_ktx2_etc1s();
        let tex = Ktx2Parser::parse(&data).unwrap();

        assert!(BasisTranscoder::can_transcode(&tex, TranscodeTarget::Etc2Rgb));

        let result = BasisTranscoder::transcode(&data, &tex, TranscodeTarget::Etc2Rgb, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transcode_to_rgba8() {
        let data = create_ktx2_uastc();
        let tex = Ktx2Parser::parse(&data).unwrap();

        assert!(BasisTranscoder::can_transcode(&tex, TranscodeTarget::Rgba8));

        let result = BasisTranscoder::transcode(&data, &tex, TranscodeTarget::Rgba8, 0);
        assert!(result.is_ok());

        let transcoded = result.unwrap();
        assert_eq!(transcoded.len(), 64 * 64 * 4);
    }

    // -------------------------------------------------------------------------
    // Invalid File Handling Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ktx2_invalid_face_count() {
        let mut data = create_minimal_ktx2();
        // Set invalid face count (offset 12 + 24 = 36)
        data[36] = 3; // Invalid: must be 1 or 6
        let result = Ktx2Parser::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_ktx2_zero_width() {
        let mut data = create_minimal_ktx2();
        // Set width to 0 (offset 12 + 8 = 20)
        data[20] = 0;
        data[21] = 0;
        data[22] = 0;
        data[23] = 0;
        let result = Ktx2Parser::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_ktx2_truncated_level_data() {
        let mut data = create_minimal_ktx2();
        // Truncate before all level data
        data.truncate(150);
        let tex = Ktx2Parser::parse(&data);
        // Should still parse header successfully
        assert!(tex.is_ok());

        // But getting level data should fail
        let tex = tex.unwrap();
        let level_data = Ktx2Parser::get_level_data(&data, &tex, 0);
        assert!(level_data.is_none());
    }

    // -------------------------------------------------------------------------
    // VkFormat Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_vk_format_is_compressed() {
        assert!(VkFormat::Bc1RgbaUnorm.is_compressed());
        assert!(VkFormat::Bc7Srgb.is_compressed());
        assert!(VkFormat::Astc4x4Unorm.is_compressed());
        assert!(VkFormat::Etc2R8G8B8Unorm.is_compressed());
        assert!(!VkFormat::R8G8B8A8Unorm.is_compressed());
        assert!(!VkFormat::R16G16B16A16Sfloat.is_compressed());
    }

    #[test]
    fn test_vk_format_is_srgb() {
        assert!(VkFormat::R8G8B8A8Srgb.is_srgb());
        assert!(VkFormat::Bc7Srgb.is_srgb());
        assert!(VkFormat::Astc4x4Srgb.is_srgb());
        assert!(!VkFormat::R8G8B8A8Unorm.is_srgb());
        assert!(!VkFormat::Bc7Unorm.is_srgb());
    }

    #[test]
    fn test_vk_format_to_compressed() {
        assert_eq!(
            VkFormat::Bc7Unorm.to_compressed_format(),
            Some(CompressedFormat::Bc7Unorm)
        );
        assert_eq!(
            VkFormat::Astc4x4Srgb.to_compressed_format(),
            Some(CompressedFormat::Astc4x4Srgb)
        );
        assert_eq!(VkFormat::R8G8B8A8Unorm.to_compressed_format(), None);
    }

    // -------------------------------------------------------------------------
    // BC7 Format Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ktx2_bc7_format() {
        let data = create_ktx2_bc7();
        let tex = Ktx2Parser::parse(&data).unwrap();

        assert_eq!(tex.header.vk_format, VkFormat::Bc7Unorm);
        assert_eq!(tex.compressed_format, Some(CompressedFormat::Bc7Unorm));
        assert_eq!(tex.width, 128);
        assert_eq!(tex.height, 128);
    }

    // -------------------------------------------------------------------------
    // Extension Detection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_ktx2_extension() {
        assert!(is_ktx2_extension("ktx2"));
        assert!(is_ktx2_extension("KTX2"));
        assert!(is_ktx2_extension("Ktx2"));
        assert!(!is_ktx2_extension("ktx"));
        assert!(!is_ktx2_extension("dds"));
    }

    #[test]
    fn test_detect_ktx2_format() {
        let ktx2_data = create_minimal_ktx2();
        assert!(detect_ktx2_format(&ktx2_data));

        let not_ktx2 = [0u8; 64];
        assert!(!detect_ktx2_format(&not_ktx2));
    }

    // -------------------------------------------------------------------------
    // TranscodeTarget Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transcode_target_compressed_format() {
        assert_eq!(
            TranscodeTarget::Bc7.to_compressed_format(false),
            Some(CompressedFormat::Bc7Unorm)
        );
        assert_eq!(
            TranscodeTarget::Bc7.to_compressed_format(true),
            Some(CompressedFormat::Bc7Srgb)
        );
        assert_eq!(
            TranscodeTarget::Astc4x4.to_compressed_format(true),
            Some(CompressedFormat::Astc4x4Srgb)
        );
        assert_eq!(TranscodeTarget::Rgba8.to_compressed_format(false), None);
    }

    #[test]
    fn test_transcode_target_bytes_per_unit() {
        assert_eq!(TranscodeTarget::Bc1.bytes_per_unit(), 8);
        assert_eq!(TranscodeTarget::Bc3.bytes_per_unit(), 16);
        assert_eq!(TranscodeTarget::Bc7.bytes_per_unit(), 16);
        assert_eq!(TranscodeTarget::Astc4x4.bytes_per_unit(), 16);
        assert_eq!(TranscodeTarget::Rgba8.bytes_per_unit(), 4);
        assert_eq!(TranscodeTarget::Rgb565.bytes_per_unit(), 2);
    }

    // -------------------------------------------------------------------------
    // Supercompression Scheme Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_supercompression_scheme() {
        assert_eq!(
            SupercompressionScheme::from_u32(0),
            SupercompressionScheme::None
        );
        assert_eq!(
            SupercompressionScheme::from_u32(1),
            SupercompressionScheme::BasisLz
        );
        assert_eq!(
            SupercompressionScheme::from_u32(2),
            SupercompressionScheme::Zstd
        );
        assert_eq!(
            SupercompressionScheme::from_u32(3),
            SupercompressionScheme::Zlib
        );
        assert_eq!(
            SupercompressionScheme::from_u32(99),
            SupercompressionScheme::None
        );
    }

    // -------------------------------------------------------------------------
    // BasisMode Display Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_basis_mode_display() {
        assert_eq!(format!("{}", BasisMode::Etc1s), "ETC1S");
        assert_eq!(format!("{}", BasisMode::Uastc), "UASTC");
    }

    // -------------------------------------------------------------------------
    // Non-Basis Texture Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_non_basis_texture() {
        let data = create_minimal_ktx2();
        let tex = Ktx2Parser::parse(&data).unwrap();

        assert!(!tex.is_basis());
        assert!(tex.basis_mode().is_none());
        assert!(!BasisTranscoder::can_transcode(&tex, TranscodeTarget::Bc7));
    }

    // -------------------------------------------------------------------------
    // Key-Value Pair Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ktx2_key_value_pairs() {
        // Create KTX2 with key-value data
        let mut data = Vec::new();

        // KTX2 layout:
        // - Identifier: 12 bytes (0-11)
        // - Header fields: 36 bytes (12-47)
        // - Index section: 32 bytes (48-79)
        // - Level indices: 24 bytes per level (80+)

        // Identifier (12 bytes)
        data.extend_from_slice(&KTX2_IDENTIFIER);

        // Header fields (36 bytes, offset 12-47)
        data.extend_from_slice(&37u32.to_le_bytes()); // vkFormat (RGBA8)
        data.extend_from_slice(&1u32.to_le_bytes()); // typeSize
        data.extend_from_slice(&32u32.to_le_bytes()); // pixelWidth
        data.extend_from_slice(&32u32.to_le_bytes()); // pixelHeight
        data.extend_from_slice(&0u32.to_le_bytes()); // pixelDepth
        data.extend_from_slice(&0u32.to_le_bytes()); // layerCount
        data.extend_from_slice(&1u32.to_le_bytes()); // faceCount
        data.extend_from_slice(&1u32.to_le_bytes()); // levelCount
        data.extend_from_slice(&0u32.to_le_bytes()); // supercompressionScheme

        // Index section (32 bytes, offset 48-79)
        let kvd_offset = 80 + 24; // After level index (104)
        let kvd_size = 24; // "KTXwriter\0test\0" + length prefix + padding
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteLength
        data.extend_from_slice(&(kvd_offset as u32).to_le_bytes()); // kvdByteOffset
        data.extend_from_slice(&(kvd_size as u32).to_le_bytes()); // kvdByteLength
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteOffset
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteLength

        // Level index (24 bytes, offset 80-103)
        let level_data_offset = kvd_offset + kvd_size; // 128
        let level_data_size = 32 * 32 * 4;
        data.extend_from_slice(&(level_data_offset as u64).to_le_bytes());
        data.extend_from_slice(&(level_data_size as u64).to_le_bytes());
        data.extend_from_slice(&(level_data_size as u64).to_le_bytes());

        // Key-value data (at offset 104)
        // Entry length
        data.extend_from_slice(&15u32.to_le_bytes()); // "KTXwriter\0test\0" = 15 bytes
        // Key
        data.extend_from_slice(b"KTXwriter\0");
        // Value
        data.extend_from_slice(b"test\0");
        // Padding to align to 4 bytes (15 + 4 = 19, need 1 byte to reach 20, then 4 more to reach 24)
        data.extend(std::iter::repeat(0u8).take(5));

        // Image data (at offset 128)
        data.extend(std::iter::repeat(0u8).take(level_data_size as usize));

        let tex = Ktx2Parser::parse(&data).unwrap();
        assert_eq!(tex.key_values.len(), 1);
        assert_eq!(tex.key_values[0].key, "KTXwriter");

        let value = tex.get_key_value("KTXwriter");
        assert!(value.is_some());
    }
}
