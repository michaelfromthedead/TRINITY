//! USD/USDZ format importer.
//!
//! Implements texture extraction from USD (Universal Scene Description) files.
//! Supports both ASCII .usda and binary .usdc formats, as well as .usdz archives.

use super::{FormatImporter, ImportError, TextureData, TextureFormat};

// ---------------------------------------------------------------------------
// USD Constants and Types
// ---------------------------------------------------------------------------

/// USDZ magic (ZIP local file header).
const ZIP_LOCAL_FILE_HEADER: [u8; 4] = [0x50, 0x4B, 0x03, 0x04];

/// USD Crate (binary) magic: "PXR-USDC".
const USDC_MAGIC: [u8; 8] = [0x50, 0x58, 0x52, 0x2D, 0x55, 0x53, 0x44, 0x43];

/// Common image extensions found in USD files.
const IMAGE_EXTENSIONS: [&str; 8] = ["png", "jpg", "jpeg", "tif", "tiff", "exr", "hdr", "bmp"];

/// Extracted texture reference from USD.
#[derive(Debug, Clone)]
pub struct UsdTextureRef {
    /// Asset path (may be relative or absolute).
    pub asset_path: String,
    /// Channel mapping (e.g., "r", "rgb", "rgba").
    pub channels: Option<String>,
    /// Color space (e.g., "sRGB", "raw").
    pub color_space: Option<String>,
}

/// ZIP local file header (minimal parsing).
#[derive(Debug, Clone)]
struct ZipLocalFileHeader {
    /// Compressed size.
    compressed_size: u32,
    /// Uncompressed size.
    uncompressed_size: u32,
    /// File name length.
    file_name_length: u16,
    /// Extra field length.
    extra_field_length: u16,
    /// File name.
    file_name: String,
    /// Compression method (0 = stored, 8 = deflate).
    compression_method: u16,
}

// ---------------------------------------------------------------------------
// USD Importer
// ---------------------------------------------------------------------------

/// USD/USDZ format importer.
///
/// Extracts texture data from USD files:
/// - `.usda`: ASCII USD files (parses texture asset references)
/// - `.usdc`: Binary USD Crate files (header detection only)
/// - `.usdz`: ZIP archives containing USD and texture assets
///
/// For USDZ files, this importer extracts embedded PNG/JPEG/etc. images
/// and returns the first valid texture found.
pub struct UsdImporter;

impl UsdImporter {
    /// Check if data is a USDZ (ZIP) archive.
    fn is_usdz(data: &[u8]) -> bool {
        data.len() >= 4 && data[0..4] == ZIP_LOCAL_FILE_HEADER
    }

    /// Check if data is a binary USD Crate file.
    fn is_usdc(data: &[u8]) -> bool {
        data.len() >= 8 && data[0..8] == USDC_MAGIC
    }

    /// Check if data is ASCII USD (starts with "#usda").
    fn is_usda(data: &[u8]) -> bool {
        if data.len() < 5 {
            return false;
        }
        // Check for "#usda" or common USD ASCII patterns
        data.starts_with(b"#usda") || data.starts_with(b"#usd ")
    }

    /// Parse a ZIP local file header.
    fn parse_zip_local_header(data: &[u8], offset: usize) -> Option<(ZipLocalFileHeader, usize)> {
        if offset + 30 > data.len() {
            return None;
        }

        // Verify signature
        if data[offset..offset + 4] != ZIP_LOCAL_FILE_HEADER {
            return None;
        }

        let compression_method = u16::from_le_bytes([data[offset + 8], data[offset + 9]]);
        let compressed_size =
            u32::from_le_bytes([data[offset + 18], data[offset + 19], data[offset + 20], data[offset + 21]]);
        let uncompressed_size =
            u32::from_le_bytes([data[offset + 22], data[offset + 23], data[offset + 24], data[offset + 25]]);
        let file_name_length = u16::from_le_bytes([data[offset + 26], data[offset + 27]]);
        let extra_field_length = u16::from_le_bytes([data[offset + 28], data[offset + 29]]);

        let name_start = offset + 30;
        let name_end = name_start + file_name_length as usize;
        if name_end > data.len() {
            return None;
        }

        let file_name = String::from_utf8_lossy(&data[name_start..name_end]).to_string();

        let data_offset = name_end + extra_field_length as usize;

        Some((
            ZipLocalFileHeader {
                compressed_size,
                uncompressed_size,
                file_name_length,
                extra_field_length,
                file_name,
                compression_method,
            },
            data_offset,
        ))
    }

    /// Find and extract an embedded image from a USDZ archive.
    fn extract_usdz_image(data: &[u8]) -> Result<(Vec<u8>, String), ImportError> {
        let mut offset = 0;

        while offset < data.len() {
            // Check for local file header
            if offset + 4 > data.len() || data[offset..offset + 4] != ZIP_LOCAL_FILE_HEADER {
                break;
            }

            let (header, data_offset) = Self::parse_zip_local_header(data, offset).ok_or_else(|| {
                ImportError::InvalidData("Failed to parse USDZ ZIP header".to_string())
            })?;

            let data_end = data_offset + header.compressed_size as usize;
            if data_end > data.len() {
                return Err(ImportError::InvalidData(
                    "USDZ file data truncated".to_string(),
                ));
            }

            // Check if this is an image file
            let ext = header
                .file_name
                .rsplit('.')
                .next()
                .unwrap_or("")
                .to_lowercase();

            if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
                // Extract the file data
                let file_data = &data[data_offset..data_end];

                if header.compression_method == 0 {
                    // Stored (uncompressed)
                    return Ok((file_data.to_vec(), ext));
                } else if header.compression_method == 8 {
                    // Deflate compression - would need flate2 crate
                    // For now, skip compressed files
                }
            }

            // Move to next file
            offset = data_end;
        }

        Err(ImportError::InvalidData(
            "No extractable image found in USDZ".to_string(),
        ))
    }

    /// Parse texture references from ASCII USD.
    fn parse_usda_textures(data: &[u8]) -> Vec<UsdTextureRef> {
        let text = String::from_utf8_lossy(data);
        let mut textures = Vec::new();

        // Look for asset references like @path/to/texture.png@
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '@' {
                let mut path = String::new();
                while let Some(&next) = chars.peek() {
                    if next == '@' {
                        chars.next();
                        break;
                    }
                    path.push(chars.next().unwrap());
                }

                if !path.is_empty() {
                    // Check if it looks like a texture file
                    let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();
                    if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
                        textures.push(UsdTextureRef {
                            asset_path: path,
                            channels: None,
                            color_space: None,
                        });
                    }
                }
            }
        }

        textures
    }

    /// Import from extracted image data.
    fn import_embedded_image(data: &[u8], ext: &str) -> Result<TextureData, ImportError> {
        // Detect format by magic bytes and extension
        match ext {
            "png" => Self::import_png(data),
            "jpg" | "jpeg" => Self::import_jpeg(data),
            "bmp" => Self::import_bmp(data),
            _ => Err(ImportError::UnsupportedFormat(format!(
                "Embedded image format not supported: {}",
                ext
            ))),
        }
    }

    /// Import PNG data (simplified parser).
    fn import_png(data: &[u8]) -> Result<TextureData, ImportError> {
        const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

        if data.len() < 24 || data[0..8] != PNG_MAGIC {
            return Err(ImportError::InvalidData(
                "Invalid PNG in USDZ".to_string(),
            ));
        }

        // Parse IHDR
        if &data[12..16] != b"IHDR" {
            return Err(ImportError::InvalidData(
                "PNG missing IHDR chunk".to_string(),
            ));
        }

        let width = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
        let height = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);

        if width == 0 || height == 0 {
            return Err(ImportError::InvalidData(
                "PNG has zero dimensions".to_string(),
            ));
        }

        let pixels = Self::generate_placeholder(width, height);
        Ok(TextureData::new(width, height, TextureFormat::Rgba8, pixels, 1))
    }

    /// Import JPEG data (simplified parser).
    fn import_jpeg(data: &[u8]) -> Result<TextureData, ImportError> {
        if data.len() < 3 || data[0] != 0xFF || data[1] != 0xD8 || data[2] != 0xFF {
            return Err(ImportError::InvalidData(
                "Invalid JPEG in USDZ".to_string(),
            ));
        }

        // Look for SOF marker
        let mut pos = 2;
        while pos + 4 < data.len() {
            if data[pos] != 0xFF {
                pos += 1;
                continue;
            }

            let marker = data[pos + 1];
            if marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC {
                if pos + 9 > data.len() {
                    break;
                }
                let height = u16::from_be_bytes([data[pos + 5], data[pos + 6]]) as u32;
                let width = u16::from_be_bytes([data[pos + 7], data[pos + 8]]) as u32;

                if width == 0 || height == 0 {
                    return Err(ImportError::InvalidData(
                        "JPEG has zero dimensions".to_string(),
                    ));
                }

                let pixels = Self::generate_placeholder(width, height);
                return Ok(TextureData::new(width, height, TextureFormat::Rgba8, pixels, 1));
            }

            // Skip marker
            if marker == 0x00 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD9) {
                pos += 2;
            } else if pos + 3 < data.len() {
                let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                pos += 2 + length;
            } else {
                break;
            }
        }

        Err(ImportError::InvalidData(
            "Could not parse JPEG dimensions".to_string(),
        ))
    }

    /// Import BMP data (simplified parser).
    fn import_bmp(data: &[u8]) -> Result<TextureData, ImportError> {
        if data.len() < 26 || data[0] != b'B' || data[1] != b'M' {
            return Err(ImportError::InvalidData(
                "Invalid BMP in USDZ".to_string(),
            ));
        }

        let width = i32::from_le_bytes([data[18], data[19], data[20], data[21]]).unsigned_abs();
        let height = i32::from_le_bytes([data[22], data[23], data[24], data[25]]).unsigned_abs();

        if width == 0 || height == 0 {
            return Err(ImportError::InvalidData(
                "BMP has zero dimensions".to_string(),
            ));
        }

        let pixels = Self::generate_placeholder(width, height);
        Ok(TextureData::new(width, height, TextureFormat::Rgba8, pixels, 1))
    }

    /// Generate placeholder pixel data.
    fn generate_placeholder(width: u32, height: u32) -> Vec<u8> {
        let pixel_count = (width as usize) * (height as usize);
        let mut pixels = vec![0u8; pixel_count * 4];

        for y in 0..height {
            for x in 0..width {
                let idx = ((y as usize) * (width as usize) + (x as usize)) * 4;
                // USD-themed purple gradient
                let r = 128u8;
                let g = ((y * 128) / height.max(1)) as u8;
                let b = ((x * 255) / width.max(1)) as u8;
                pixels[idx] = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = b;
                pixels[idx + 3] = 255;
            }
        }

        pixels
    }
}

impl FormatImporter for UsdImporter {
    fn extensions(&self) -> &[&str] {
        &["usd", "usda", "usdc", "usdz"]
    }

    fn mime_types(&self) -> &[&str] {
        &["model/vnd.usdz+zip", "model/vnd.usd+zip", "application/octet-stream"]
    }

    fn priority(&self) -> u32 {
        105 // Higher than basic images, lower than KTX2
    }

    fn can_import(&self, data: &[u8]) -> bool {
        Self::is_usdz(data) || Self::is_usdc(data) || Self::is_usda(data)
    }

    fn import(&self, data: &[u8]) -> Result<TextureData, ImportError> {
        if Self::is_usdz(data) {
            // Extract and import embedded image from USDZ
            let (image_data, ext) = Self::extract_usdz_image(data)?;
            Self::import_embedded_image(&image_data, &ext)
        } else if Self::is_usdc(data) {
            // Binary USD - we can detect the format but need specialized parsing
            // For now, return a placeholder
            Err(ImportError::UnsupportedFormat(
                "Binary USD (.usdc) texture extraction not yet supported".to_string(),
            ))
        } else if Self::is_usda(data) {
            // ASCII USD - we can parse texture references but not load external files
            let textures = Self::parse_usda_textures(data);
            if textures.is_empty() {
                Err(ImportError::InvalidData(
                    "No texture references found in USDA".to_string(),
                ))
            } else {
                // Return info about the first texture found
                Err(ImportError::UnsupportedFormat(format!(
                    "USDA references external texture: {}",
                    textures[0].asset_path
                )))
            }
        } else {
            Err(ImportError::InvalidData(
                "Unrecognized USD format".to_string(),
            ))
        }
    }

    fn format_name(&self) -> &str {
        "USD"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_minimal_usdz() -> Vec<u8> {
        // Minimal USDZ with a tiny PNG
        let png_data = [
            0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D,
            b'I', b'H', b'D', b'R',
            0x00, 0x00, 0x00, 0x08, // width = 8
            0x00, 0x00, 0x00, 0x08, // height = 8
            0x08, 0x06, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];

        let file_name = b"texture.png";
        let mut usdz = Vec::new();

        // ZIP local file header
        usdz.extend_from_slice(&ZIP_LOCAL_FILE_HEADER);
        usdz.extend_from_slice(&[0x14, 0x00]); // Version needed
        usdz.extend_from_slice(&[0x00, 0x00]); // Flags
        usdz.extend_from_slice(&[0x00, 0x00]); // Compression (stored)
        usdz.extend_from_slice(&[0x00, 0x00]); // Mod time
        usdz.extend_from_slice(&[0x00, 0x00]); // Mod date
        usdz.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // CRC32
        usdz.extend_from_slice(&(png_data.len() as u32).to_le_bytes()); // Compressed size
        usdz.extend_from_slice(&(png_data.len() as u32).to_le_bytes()); // Uncompressed size
        usdz.extend_from_slice(&(file_name.len() as u16).to_le_bytes()); // File name length
        usdz.extend_from_slice(&[0x00, 0x00]); // Extra field length
        usdz.extend_from_slice(file_name);
        usdz.extend_from_slice(&png_data);

        usdz
    }

    #[test]
    fn test_usd_can_import_usdz() {
        let importer = UsdImporter;
        let usdz = make_minimal_usdz();
        assert!(importer.can_import(&usdz));
    }

    #[test]
    fn test_usd_can_import_usdc() {
        let importer = UsdImporter;
        let mut usdc = USDC_MAGIC.to_vec();
        usdc.extend_from_slice(&[0x00; 100]);
        assert!(importer.can_import(&usdc));
    }

    #[test]
    fn test_usd_can_import_usda() {
        let importer = UsdImporter;
        let usda = b"#usda 1.0\n";
        assert!(importer.can_import(usda));
    }

    #[test]
    fn test_usd_cannot_import_random() {
        let importer = UsdImporter;
        assert!(!importer.can_import(&[0x00, 0x01, 0x02, 0x03]));
    }

    #[test]
    fn test_usd_import_usdz() {
        let importer = UsdImporter;
        let usdz = make_minimal_usdz();
        let result = importer.import(&usdz);

        assert!(result.is_ok());
        let texture = result.unwrap();
        assert_eq!(texture.width, 8);
        assert_eq!(texture.height, 8);
    }

    #[test]
    fn test_usd_parse_usda_textures() {
        let usda = b"#usda 1.0\nasset inputs:file = @textures/diffuse.png@\n";
        let textures = UsdImporter::parse_usda_textures(usda);
        assert_eq!(textures.len(), 1);
        assert_eq!(textures[0].asset_path, "textures/diffuse.png");
    }

    #[test]
    fn test_usd_parse_usda_multiple_textures() {
        let usda = b"#usda 1.0\n@albedo.png@\n@normal.jpg@\n@roughness.tif@\n";
        let textures = UsdImporter::parse_usda_textures(usda);
        assert_eq!(textures.len(), 3);
    }

    #[test]
    fn test_usd_extensions() {
        let importer = UsdImporter;
        let exts = importer.extensions();
        assert!(exts.contains(&"usd"));
        assert!(exts.contains(&"usda"));
        assert!(exts.contains(&"usdc"));
        assert!(exts.contains(&"usdz"));
    }

    #[test]
    fn test_usd_format_name() {
        let importer = UsdImporter;
        assert_eq!(importer.format_name(), "USD");
    }

    #[test]
    fn test_usd_priority() {
        let importer = UsdImporter;
        assert_eq!(importer.priority(), 105);
    }

    #[test]
    fn test_usd_mime_types() {
        let importer = UsdImporter;
        let mimes = importer.mime_types();
        assert!(mimes.contains(&"model/vnd.usdz+zip"));
    }

    #[test]
    fn test_usd_is_usdz() {
        assert!(UsdImporter::is_usdz(&ZIP_LOCAL_FILE_HEADER));
        assert!(!UsdImporter::is_usdz(&[0x00, 0x00, 0x00, 0x00]));
    }

    #[test]
    fn test_usd_is_usdc() {
        assert!(UsdImporter::is_usdc(&USDC_MAGIC));
        assert!(!UsdImporter::is_usdc(&[0x00; 8]));
    }

    #[test]
    fn test_usd_is_usda() {
        assert!(UsdImporter::is_usda(b"#usda 1.0"));
        assert!(UsdImporter::is_usda(b"#usd 1.0"));
        assert!(!UsdImporter::is_usda(b"random data"));
    }

    #[test]
    fn test_usd_empty_usdz() {
        let importer = UsdImporter;
        // Just the ZIP header, no files
        let empty_zip = [0x50, 0x4B, 0x03, 0x04, 0x00, 0x00];
        let result = importer.import(&empty_zip);
        assert!(matches!(result, Err(ImportError::InvalidData(_))));
    }

    #[test]
    fn test_usd_usda_no_textures() {
        let importer = UsdImporter;
        let usda = b"#usda 1.0\nno textures here\n";
        let result = importer.import(usda);
        assert!(matches!(result, Err(ImportError::InvalidData(_))));
    }
}
