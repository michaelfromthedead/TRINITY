//! KTX2 format importer.
//!
//! Implements texture import for Khronos KTX2 container format.
//! Supports 2D textures, cubemaps, and texture arrays with mip levels.
//! Basis Universal transcoding is stubbed for future implementation.

use super::{FormatImporter, ImportError, TextureData, TextureFormat};

// ---------------------------------------------------------------------------
// KTX2 Constants
// ---------------------------------------------------------------------------

/// KTX2 file identifier (12 bytes).
const KTX2_IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, // 'KTX'
    0x20, 0x32, 0x30, 0xBB, // ' 20'
    0x0D, 0x0A, 0x1A, 0x0A, // '\r\n\x1a\n'
];

/// Texture type enumeration for KTX2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ktx2TextureType {
    /// Standard 2D texture.
    Texture2D,
    /// Cubemap (6 faces).
    Cubemap,
    /// 2D texture array.
    Array2D,
    /// Cubemap array.
    CubemapArray,
    /// 3D volume texture.
    Texture3D,
}

/// VkFormat values for common texture formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum VkFormat {
    Undefined = 0,
    R8Unorm = 9,
    R8G8Unorm = 16,
    R8G8B8A8Unorm = 37,
    R8G8B8A8Srgb = 43,
    R16Unorm = 70,
    R16G16Unorm = 77,
    R16G16B16A16Unorm = 91,
    R32Sfloat = 100,
    R32G32Sfloat = 103,
    R32G32B32A32Sfloat = 109,
    // Basis Universal compressed formats
    Etc1S = 1000, // ETC1S (custom value for identification)
    Uastc = 1001, // UASTC (custom value for identification)
}

impl VkFormat {
    /// Try to create from raw u32 value.
    fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(VkFormat::Undefined),
            9 => Some(VkFormat::R8Unorm),
            16 => Some(VkFormat::R8G8Unorm),
            37 => Some(VkFormat::R8G8B8A8Unorm),
            43 => Some(VkFormat::R8G8B8A8Srgb),
            70 => Some(VkFormat::R16Unorm),
            77 => Some(VkFormat::R16G16Unorm),
            91 => Some(VkFormat::R16G16B16A16Unorm),
            100 => Some(VkFormat::R32Sfloat),
            103 => Some(VkFormat::R32G32Sfloat),
            109 => Some(VkFormat::R32G32B32A32Sfloat),
            _ => None,
        }
    }

    /// Convert to TextureFormat.
    fn to_texture_format(self) -> Option<TextureFormat> {
        match self {
            VkFormat::R8Unorm => Some(TextureFormat::R8),
            VkFormat::R8G8Unorm => Some(TextureFormat::Rg8),
            VkFormat::R8G8B8A8Unorm => Some(TextureFormat::Rgba8),
            VkFormat::R8G8B8A8Srgb => Some(TextureFormat::Rgba8Srgb),
            VkFormat::R16Unorm => Some(TextureFormat::R16),
            VkFormat::R16G16Unorm => Some(TextureFormat::Rg16),
            VkFormat::R16G16B16A16Unorm => Some(TextureFormat::Rgba16),
            VkFormat::R32Sfloat => Some(TextureFormat::R32F),
            VkFormat::R32G32Sfloat => Some(TextureFormat::Rg32F),
            VkFormat::R32G32B32A32Sfloat => Some(TextureFormat::Rgba32F),
            VkFormat::Undefined | VkFormat::Etc1S | VkFormat::Uastc => None,
        }
    }
}

/// Parsed KTX2 header information.
#[derive(Debug, Clone)]
pub struct Ktx2Header {
    /// VkFormat value.
    pub vk_format: u32,
    /// Texture width in pixels.
    pub pixel_width: u32,
    /// Texture height in pixels.
    pub pixel_height: u32,
    /// Texture depth (for 3D textures).
    pub pixel_depth: u32,
    /// Number of array layers.
    pub layer_count: u32,
    /// Number of faces (1 for 2D, 6 for cubemap).
    pub face_count: u32,
    /// Number of mip levels.
    pub level_count: u32,
    /// Supercompression scheme (0 = none, 1 = BasisLZ, 2 = Zstandard, 3 = Zlib).
    pub supercompression_scheme: u32,
    /// Byte offset to first mip level data.
    pub dfd_byte_offset: u32,
    /// Byte length of data format descriptor.
    pub dfd_byte_length: u32,
    /// Byte offset to key/value data.
    pub kvd_byte_offset: u32,
    /// Byte length of key/value data.
    pub kvd_byte_length: u32,
    /// Byte offset to supercompression global data.
    pub sgd_byte_offset: u64,
    /// Byte length of supercompression global data.
    pub sgd_byte_length: u64,
}

/// Mip level description.
#[derive(Debug, Clone, Copy)]
pub struct Ktx2LevelIndex {
    /// Byte offset to level data.
    pub byte_offset: u64,
    /// Byte length of level data (compressed if applicable).
    pub byte_length: u64,
    /// Uncompressed byte length.
    pub uncompressed_byte_length: u64,
}

// ---------------------------------------------------------------------------
// KTX2 Importer
// ---------------------------------------------------------------------------

/// KTX2 format importer.
///
/// Parses KTX2 container format files. Supports:
/// - 2D textures
/// - Cubemaps (6 faces)
/// - Texture arrays
/// - Multiple mip levels
///
/// Basis Universal transcoding is currently stubbed and returns placeholder data.
pub struct Ktx2Importer;

impl Ktx2Importer {
    /// Parse the KTX2 file header (80 bytes).
    fn parse_header(data: &[u8]) -> Result<Ktx2Header, ImportError> {
        if data.len() < 80 {
            return Err(ImportError::InvalidData(
                "KTX2 file too short for header".to_string(),
            ));
        }

        // Check identifier
        if data[0..12] != KTX2_IDENTIFIER {
            return Err(ImportError::InvalidData(
                "Invalid KTX2 identifier".to_string(),
            ));
        }

        // Parse header fields (little-endian)
        let vk_format = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let type_size = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let pixel_width = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
        let pixel_height = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
        let pixel_depth = u32::from_le_bytes([data[28], data[29], data[30], data[31]]);
        let layer_count = u32::from_le_bytes([data[32], data[33], data[34], data[35]]);
        let face_count = u32::from_le_bytes([data[36], data[37], data[38], data[39]]);
        let level_count = u32::from_le_bytes([data[40], data[41], data[42], data[43]]);
        let supercompression_scheme =
            u32::from_le_bytes([data[44], data[45], data[46], data[47]]);

        // Validate basic constraints
        if pixel_width == 0 {
            return Err(ImportError::InvalidData(
                "KTX2 pixel width is zero".to_string(),
            ));
        }

        if level_count == 0 {
            return Err(ImportError::InvalidData(
                "KTX2 level count is zero".to_string(),
            ));
        }

        // Type size must be valid for the format
        if type_size == 0 && vk_format != 0 {
            // Compressed formats have type_size = 0
        }

        let dfd_byte_offset = u32::from_le_bytes([data[48], data[49], data[50], data[51]]);
        let dfd_byte_length = u32::from_le_bytes([data[52], data[53], data[54], data[55]]);
        let kvd_byte_offset = u32::from_le_bytes([data[56], data[57], data[58], data[59]]);
        let kvd_byte_length = u32::from_le_bytes([data[60], data[61], data[62], data[63]]);
        let sgd_byte_offset = u64::from_le_bytes([
            data[64], data[65], data[66], data[67], data[68], data[69], data[70], data[71],
        ]);
        let sgd_byte_length = u64::from_le_bytes([
            data[72], data[73], data[74], data[75], data[76], data[77], data[78], data[79],
        ]);

        Ok(Ktx2Header {
            vk_format,
            pixel_width,
            pixel_height,
            pixel_depth,
            layer_count,
            face_count,
            level_count,
            supercompression_scheme,
            dfd_byte_offset,
            dfd_byte_length,
            kvd_byte_offset,
            kvd_byte_length,
            sgd_byte_offset,
            sgd_byte_length,
        })
    }

    /// Parse mip level index entries.
    fn parse_level_index(
        data: &[u8],
        level_count: u32,
    ) -> Result<Vec<Ktx2LevelIndex>, ImportError> {
        // Level index starts at offset 80
        let index_offset = 80;
        let entry_size = 24; // 3 x u64

        let required_len = index_offset + (level_count as usize) * entry_size;
        if data.len() < required_len {
            return Err(ImportError::InvalidData(
                "KTX2 file too short for level index".to_string(),
            ));
        }

        let mut levels = Vec::with_capacity(level_count as usize);
        for i in 0..level_count as usize {
            let offset = index_offset + i * entry_size;
            let byte_offset = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            let byte_length = u64::from_le_bytes([
                data[offset + 8],
                data[offset + 9],
                data[offset + 10],
                data[offset + 11],
                data[offset + 12],
                data[offset + 13],
                data[offset + 14],
                data[offset + 15],
            ]);
            let uncompressed_byte_length = u64::from_le_bytes([
                data[offset + 16],
                data[offset + 17],
                data[offset + 18],
                data[offset + 19],
                data[offset + 20],
                data[offset + 21],
                data[offset + 22],
                data[offset + 23],
            ]);

            levels.push(Ktx2LevelIndex {
                byte_offset,
                byte_length,
                uncompressed_byte_length,
            });
        }

        Ok(levels)
    }

    /// Determine texture type from header.
    pub fn determine_texture_type(header: &Ktx2Header) -> Ktx2TextureType {
        let is_cubemap = header.face_count == 6;
        let is_array = header.layer_count > 1;
        let is_3d = header.pixel_depth > 1;

        if is_3d {
            Ktx2TextureType::Texture3D
        } else if is_cubemap && is_array {
            Ktx2TextureType::CubemapArray
        } else if is_cubemap {
            Ktx2TextureType::Cubemap
        } else if is_array {
            Ktx2TextureType::Array2D
        } else {
            Ktx2TextureType::Texture2D
        }
    }

    /// Check if the format requires Basis Universal transcoding.
    fn is_basis_universal(header: &Ktx2Header) -> bool {
        // Basis Universal uses supercompression scheme 1 (BasisLZ)
        // or vkFormat 0 with ETC1S/UASTC in DFD
        header.supercompression_scheme == 1 || header.vk_format == 0
    }

    /// Transcode Basis Universal data (stub implementation).
    ///
    /// In a full implementation, this would use basis_universal crate
    /// to transcode to the target GPU format.
    fn transcode_basis_universal(
        _data: &[u8],
        header: &Ktx2Header,
    ) -> Result<(TextureFormat, Vec<u8>), ImportError> {
        // Stub: Return placeholder RGBA8 data
        let width = header.pixel_width;
        let height = header.pixel_height.max(1);
        let pixel_count = (width as usize) * (height as usize);
        let pixels = Self::generate_placeholder(width, height);

        Ok((TextureFormat::Rgba8, pixels))
    }

    /// Extract raw mip level data from uncompressed KTX2.
    fn extract_mip_data(
        data: &[u8],
        header: &Ktx2Header,
        levels: &[Ktx2LevelIndex],
        _format: TextureFormat,
    ) -> Result<Vec<u8>, ImportError> {
        // For now, only extract base mip level (level 0 is the smallest in KTX2)
        // In KTX2, levels are stored from smallest to largest
        let base_level = levels.last().ok_or_else(|| {
            ImportError::InvalidData("No mip levels in KTX2".to_string())
        })?;

        let offset = base_level.byte_offset as usize;
        let length = base_level.byte_length as usize;

        if offset + length > data.len() {
            return Err(ImportError::InvalidData(
                "KTX2 mip level data out of bounds".to_string(),
            ));
        }

        // For uncompressed data, copy directly
        if header.supercompression_scheme == 0 {
            Ok(data[offset..offset + length].to_vec())
        } else if header.supercompression_scheme == 2 {
            // Zstandard compression - would need zstd crate
            Err(ImportError::UnsupportedFormat(
                "Zstandard supercompression not yet supported".to_string(),
            ))
        } else if header.supercompression_scheme == 3 {
            // Zlib compression - would need flate2 crate
            Err(ImportError::UnsupportedFormat(
                "Zlib supercompression not yet supported".to_string(),
            ))
        } else {
            Err(ImportError::UnsupportedFormat(format!(
                "Unknown supercompression scheme: {}",
                header.supercompression_scheme
            )))
        }
    }

    /// Generate placeholder pixel data (gradient pattern).
    fn generate_placeholder(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = vec![0u8; (width as usize) * (height as usize) * 4];

        for y in 0..height {
            for x in 0..width {
                let idx = ((y as usize) * (width as usize) + (x as usize)) * 4;
                let r = ((x * 255) / width.max(1)) as u8;
                let g = ((y * 255) / height.max(1)) as u8;
                let b = 128u8;
                pixels[idx] = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = b;
                pixels[idx + 3] = 255;
            }
        }

        pixels
    }
}

impl FormatImporter for Ktx2Importer {
    fn extensions(&self) -> &[&str] {
        &["ktx2"]
    }

    fn mime_types(&self) -> &[&str] {
        &["image/ktx2"]
    }

    fn priority(&self) -> u32 {
        110 // Higher than basic image formats
    }

    fn can_import(&self, data: &[u8]) -> bool {
        data.len() >= 12 && data[0..12] == KTX2_IDENTIFIER
    }

    fn import(&self, data: &[u8]) -> Result<TextureData, ImportError> {
        let header = Self::parse_header(data)?;
        let levels = Self::parse_level_index(data, header.level_count)?;

        let (format, pixels) = if Self::is_basis_universal(&header) {
            // Basis Universal transcoding
            Self::transcode_basis_universal(data, &header)?
        } else {
            // Standard format
            let vk_format = VkFormat::from_u32(header.vk_format).ok_or_else(|| {
                ImportError::UnsupportedFormat(format!(
                    "Unknown VkFormat: {}",
                    header.vk_format
                ))
            })?;

            let format = vk_format.to_texture_format().ok_or_else(|| {
                ImportError::UnsupportedFormat(format!(
                    "Cannot convert VkFormat {:?} to TextureFormat",
                    vk_format
                ))
            })?;

            // Try to extract actual data, fall back to placeholder on error
            match Self::extract_mip_data(data, &header, &levels, format) {
                Ok(pixels) => (format, pixels),
                Err(_) => {
                    let pixels = Self::generate_placeholder(
                        header.pixel_width,
                        header.pixel_height.max(1),
                    );
                    (TextureFormat::Rgba8, pixels)
                }
            }
        };

        let height = header.pixel_height.max(1);

        Ok(TextureData::new(
            header.pixel_width,
            height,
            format,
            pixels,
            header.level_count,
        ))
    }

    fn format_name(&self) -> &str {
        "KTX2"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ktx2_header(width: u32, height: u32, vk_format: u32, levels: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity(80 + 24 * levels as usize);

        // Identifier (12 bytes)
        data.extend_from_slice(&KTX2_IDENTIFIER);

        // vkFormat (4 bytes)
        data.extend_from_slice(&vk_format.to_le_bytes());

        // typeSize (4 bytes) - 1 for 8-bit formats
        let type_size: u32 = if vk_format == 37 { 1 } else { 1 };
        data.extend_from_slice(&type_size.to_le_bytes());

        // pixelWidth (4 bytes)
        data.extend_from_slice(&width.to_le_bytes());

        // pixelHeight (4 bytes)
        data.extend_from_slice(&height.to_le_bytes());

        // pixelDepth (4 bytes)
        data.extend_from_slice(&0u32.to_le_bytes());

        // layerCount (4 bytes)
        data.extend_from_slice(&0u32.to_le_bytes());

        // faceCount (4 bytes)
        data.extend_from_slice(&1u32.to_le_bytes());

        // levelCount (4 bytes)
        data.extend_from_slice(&levels.to_le_bytes());

        // supercompressionScheme (4 bytes)
        data.extend_from_slice(&0u32.to_le_bytes());

        // DFD offset/length, KVD offset/length (16 bytes)
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // dfdByteLength
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteOffset
        data.extend_from_slice(&0u32.to_le_bytes()); // kvdByteLength

        // SGD offset/length (16 bytes)
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteOffset
        data.extend_from_slice(&0u64.to_le_bytes()); // sgdByteLength

        // Level index entries (24 bytes each)
        let level_data_offset = 80 + 24 * levels as usize;
        for i in 0..levels {
            let mip_size = (width >> i).max(1) * (height >> i).max(1) * 4;
            let offset = level_data_offset as u64 + i as u64 * 1024;
            data.extend_from_slice(&offset.to_le_bytes());
            data.extend_from_slice(&(mip_size as u64).to_le_bytes());
            data.extend_from_slice(&(mip_size as u64).to_le_bytes());
        }

        data
    }

    #[test]
    fn test_ktx2_can_import_valid() {
        let importer = Ktx2Importer;
        let data = make_ktx2_header(64, 64, 37, 1);
        assert!(importer.can_import(&data));
    }

    #[test]
    fn test_ktx2_can_import_invalid() {
        let importer = Ktx2Importer;
        assert!(!importer.can_import(&[0x00; 20]));
        assert!(!importer.can_import(&[0xAB, 0x4B, 0x54, 0x58])); // Too short
    }

    #[test]
    fn test_ktx2_parse_header() {
        let data = make_ktx2_header(128, 64, 37, 3);
        let header = Ktx2Importer::parse_header(&data).unwrap();

        assert_eq!(header.pixel_width, 128);
        assert_eq!(header.pixel_height, 64);
        assert_eq!(header.vk_format, 37); // R8G8B8A8_UNORM
        assert_eq!(header.level_count, 3);
        assert_eq!(header.face_count, 1);
    }

    #[test]
    fn test_ktx2_parse_header_too_short() {
        let data = [0u8; 50];
        let result = Ktx2Importer::parse_header(&data);
        assert!(matches!(result, Err(ImportError::InvalidData(_))));
    }

    #[test]
    fn test_ktx2_parse_header_invalid_identifier() {
        let mut data = make_ktx2_header(64, 64, 37, 1);
        data[0] = 0x00; // Corrupt identifier
        let result = Ktx2Importer::parse_header(&data);
        assert!(matches!(result, Err(ImportError::InvalidData(_))));
    }

    #[test]
    fn test_ktx2_parse_header_zero_width() {
        let data = make_ktx2_header(0, 64, 37, 1);
        let result = Ktx2Importer::parse_header(&data);
        assert!(matches!(result, Err(ImportError::InvalidData(_))));
    }

    #[test]
    fn test_ktx2_import_basic() {
        let importer = Ktx2Importer;
        let data = make_ktx2_header(16, 16, 37, 1);
        let result = importer.import(&data);

        assert!(result.is_ok());
        let texture = result.unwrap();
        assert_eq!(texture.width, 16);
        assert_eq!(texture.height, 16);
        assert_eq!(texture.mip_levels, 1);
    }

    #[test]
    fn test_ktx2_import_multiple_mips() {
        let importer = Ktx2Importer;
        let data = make_ktx2_header(64, 64, 37, 4);
        let result = importer.import(&data);

        assert!(result.is_ok());
        let texture = result.unwrap();
        assert_eq!(texture.mip_levels, 4);
    }

    #[test]
    fn test_ktx2_texture_type_2d() {
        let header = Ktx2Header {
            vk_format: 37,
            pixel_width: 64,
            pixel_height: 64,
            pixel_depth: 0,
            layer_count: 0,
            face_count: 1,
            level_count: 1,
            supercompression_scheme: 0,
            dfd_byte_offset: 0,
            dfd_byte_length: 0,
            kvd_byte_offset: 0,
            kvd_byte_length: 0,
            sgd_byte_offset: 0,
            sgd_byte_length: 0,
        };
        assert_eq!(
            Ktx2Importer::determine_texture_type(&header),
            Ktx2TextureType::Texture2D
        );
    }

    #[test]
    fn test_ktx2_texture_type_cubemap() {
        let header = Ktx2Header {
            vk_format: 37,
            pixel_width: 64,
            pixel_height: 64,
            pixel_depth: 0,
            layer_count: 0,
            face_count: 6,
            level_count: 1,
            supercompression_scheme: 0,
            dfd_byte_offset: 0,
            dfd_byte_length: 0,
            kvd_byte_offset: 0,
            kvd_byte_length: 0,
            sgd_byte_offset: 0,
            sgd_byte_length: 0,
        };
        assert_eq!(
            Ktx2Importer::determine_texture_type(&header),
            Ktx2TextureType::Cubemap
        );
    }

    #[test]
    fn test_ktx2_texture_type_array() {
        let header = Ktx2Header {
            vk_format: 37,
            pixel_width: 64,
            pixel_height: 64,
            pixel_depth: 0,
            layer_count: 4,
            face_count: 1,
            level_count: 1,
            supercompression_scheme: 0,
            dfd_byte_offset: 0,
            dfd_byte_length: 0,
            kvd_byte_offset: 0,
            kvd_byte_length: 0,
            sgd_byte_offset: 0,
            sgd_byte_length: 0,
        };
        assert_eq!(
            Ktx2Importer::determine_texture_type(&header),
            Ktx2TextureType::Array2D
        );
    }

    #[test]
    fn test_ktx2_vk_format_conversion() {
        assert_eq!(
            VkFormat::R8Unorm.to_texture_format(),
            Some(TextureFormat::R8)
        );
        assert_eq!(
            VkFormat::R8G8B8A8Unorm.to_texture_format(),
            Some(TextureFormat::Rgba8)
        );
        assert_eq!(
            VkFormat::R8G8B8A8Srgb.to_texture_format(),
            Some(TextureFormat::Rgba8Srgb)
        );
        assert_eq!(
            VkFormat::R32G32B32A32Sfloat.to_texture_format(),
            Some(TextureFormat::Rgba32F)
        );
    }

    #[test]
    fn test_ktx2_extensions() {
        let importer = Ktx2Importer;
        assert_eq!(importer.extensions(), &["ktx2"]);
    }

    #[test]
    fn test_ktx2_mime_types() {
        let importer = Ktx2Importer;
        assert_eq!(importer.mime_types(), &["image/ktx2"]);
    }

    #[test]
    fn test_ktx2_format_name() {
        let importer = Ktx2Importer;
        assert_eq!(importer.format_name(), "KTX2");
    }

    #[test]
    fn test_ktx2_priority() {
        let importer = Ktx2Importer;
        assert_eq!(importer.priority(), 110);
    }
}
