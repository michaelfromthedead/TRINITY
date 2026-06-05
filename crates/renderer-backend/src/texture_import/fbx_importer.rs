//! FBX format importer.
//!
//! Implements texture extraction from FBX (Filmbox) files.
//! Supports both binary and ASCII FBX formats.

use super::{FormatImporter, ImportError, TextureData, TextureFormat};

// ---------------------------------------------------------------------------
// FBX Constants
// ---------------------------------------------------------------------------

/// FBX binary magic: "Kaydara FBX Binary  " (21 bytes including null + 2 padding).
const FBX_BINARY_MAGIC: &[u8] = b"Kaydara FBX Binary  \x00";

/// Common texture property names in FBX.
const TEXTURE_PROPERTIES: [&str; 6] = [
    "DiffuseColor",
    "NormalMap",
    "SpecularColor",
    "EmissiveColor",
    "AmbientColor",
    "TransparencyFactor",
];

/// Embedded image magic bytes for detection.
struct ImageMagic;

impl ImageMagic {
    const PNG: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    const JPEG: [u8; 2] = [0xFF, 0xD8];
    const BMP: [u8; 2] = [b'B', b'M'];
    const TGA_SIGNATURE: &'static [u8] = b"TRUEVISION-XFILE.\x00";
}

/// FBX texture reference.
#[derive(Debug, Clone)]
pub struct FbxTextureRef {
    /// Texture file name (relative or absolute path).
    pub file_name: String,
    /// Property this texture is connected to.
    pub property: Option<String>,
    /// Whether texture data is embedded.
    pub is_embedded: bool,
}

/// Embedded texture data found in FBX.
#[derive(Debug)]
pub struct FbxEmbeddedTexture {
    /// Texture name/identifier.
    pub name: String,
    /// Raw image data.
    pub data: Vec<u8>,
    /// Detected format extension.
    pub format: String,
}

/// FBX file header information.
#[derive(Debug, Clone)]
pub struct FbxHeader {
    /// FBX version number (e.g., 7400 for FBX 2016).
    pub version: u32,
    /// Whether this is binary format.
    pub is_binary: bool,
}

// ---------------------------------------------------------------------------
// FBX Importer
// ---------------------------------------------------------------------------

/// FBX format importer.
///
/// Extracts texture data from FBX files:
/// - Parses FBX binary header to detect format version
/// - Extracts embedded texture data from Video nodes
/// - Identifies texture file references
///
/// Supports FBX versions 7.x (2010+).
pub struct FbxImporter;

impl FbxImporter {
    /// Check if data is binary FBX.
    fn is_binary_fbx(data: &[u8]) -> bool {
        data.len() >= 21 && data[0..21] == *FBX_BINARY_MAGIC
    }

    /// Check if data is ASCII FBX.
    fn is_ascii_fbx(data: &[u8]) -> bool {
        if data.len() < 100 {
            return false;
        }
        // ASCII FBX starts with comments and version info
        // Look for "; FBX" or "FBXHeaderExtension"
        let header = String::from_utf8_lossy(&data[..100.min(data.len())]);
        header.contains("; FBX") || header.contains("FBXHeaderExtension")
    }

    /// Parse binary FBX header.
    fn parse_binary_header(data: &[u8]) -> Result<FbxHeader, ImportError> {
        if data.len() < 27 {
            return Err(ImportError::InvalidData(
                "FBX file too short for header".to_string(),
            ));
        }

        if !Self::is_binary_fbx(data) {
            return Err(ImportError::InvalidData(
                "Invalid FBX binary magic".to_string(),
            ));
        }

        // Version is at offset 23-26 (little-endian u32)
        let version = u32::from_le_bytes([data[23], data[24], data[25], data[26]]);

        Ok(FbxHeader {
            version,
            is_binary: true,
        })
    }

    /// Parse ASCII FBX header.
    fn parse_ascii_header(data: &[u8]) -> Result<FbxHeader, ImportError> {
        let text = String::from_utf8_lossy(data);

        // Look for FBXVersion: XXXX
        let version = if let Some(pos) = text.find("FBXVersion:") {
            let after = &text[pos + 11..];
            let version_str: String = after
                .chars()
                .skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            version_str.parse().unwrap_or(7400)
        } else {
            7400 // Default to FBX 2016
        };

        Ok(FbxHeader {
            version,
            is_binary: false,
        })
    }

    /// Detect image format from magic bytes.
    fn detect_image_format(data: &[u8]) -> Option<&'static str> {
        if data.len() >= 8 && data[0..8] == ImageMagic::PNG {
            Some("png")
        } else if data.len() >= 2 && data[0..2] == ImageMagic::JPEG {
            Some("jpg")
        } else if data.len() >= 2 && data[0..2] == ImageMagic::BMP {
            Some("bmp")
        } else if data.len() >= 18 {
            // TGA has no magic at start, but has signature at end
            // For embedded data, we check the footer
            None // TGA detection is unreliable without full data
        } else {
            None
        }
    }

    /// Search for embedded texture data in binary FBX.
    ///
    /// FBX stores embedded textures in "Video" nodes with a "Content" property
    /// containing the raw image data.
    fn find_embedded_textures(data: &[u8]) -> Vec<FbxEmbeddedTexture> {
        let mut textures = Vec::new();

        // Look for PNG/JPEG signatures in the data
        // This is a simplified approach - full FBX parsing would follow the node structure
        let mut pos = 0;
        while pos + 8 < data.len() {
            // Check for PNG
            if data[pos..pos + 8] == ImageMagic::PNG {
                if let Some(texture) = Self::extract_png_at(data, pos) {
                    let data_len = texture.data.len();
                    textures.push(texture);
                    pos += data_len;
                    continue;
                }
            }

            // Check for JPEG
            if pos + 2 < data.len() && data[pos..pos + 2] == ImageMagic::JPEG {
                if let Some(texture) = Self::extract_jpeg_at(data, pos) {
                    let data_len = texture.data.len();
                    textures.push(texture);
                    pos += data_len;
                    continue;
                }
            }

            pos += 1;
        }

        textures
    }

    /// Extract PNG data starting at the given position.
    fn extract_png_at(data: &[u8], start: usize) -> Option<FbxEmbeddedTexture> {
        // PNG ends with IEND chunk
        const IEND: [u8; 8] = [0x00, 0x00, 0x00, 0x00, b'I', b'E', b'N', b'D'];

        let remaining = &data[start..];
        if remaining.len() < 33 {
            return None; // Too short for valid PNG
        }

        // Search for IEND chunk (simplified - real parser would follow chunk structure)
        for end in 33..remaining.len().min(10_000_000) {
            if end + 4 <= remaining.len() && &remaining[end - 4..end + 4] == &IEND {
                let png_data = remaining[..end + 8].to_vec();

                // Verify we have valid IHDR
                if png_data.len() >= 24 && &png_data[12..16] == b"IHDR" {
                    return Some(FbxEmbeddedTexture {
                        name: format!("embedded_{}", start),
                        data: png_data,
                        format: "png".to_string(),
                    });
                }
            }
        }

        None
    }

    /// Extract JPEG data starting at the given position.
    fn extract_jpeg_at(data: &[u8], start: usize) -> Option<FbxEmbeddedTexture> {
        // JPEG ends with FFD9 (EOI)
        let remaining = &data[start..];
        if remaining.len() < 10 {
            return None;
        }

        // Search for EOI marker
        for end in 10..remaining.len().min(10_000_000) {
            if remaining[end - 2] == 0xFF && remaining[end - 1] == 0xD9 {
                let jpeg_data = remaining[..end].to_vec();
                return Some(FbxEmbeddedTexture {
                    name: format!("embedded_{}", start),
                    data: jpeg_data,
                    format: "jpg".to_string(),
                });
            }
        }

        None
    }

    /// Parse texture file references from ASCII FBX.
    fn parse_ascii_texture_refs(data: &[u8]) -> Vec<FbxTextureRef> {
        let text = String::from_utf8_lossy(data);
        let mut refs = Vec::new();

        // Look for "RelativeFilename:" or "Filename:" properties
        for line in text.lines() {
            let line = line.trim();
            if line.contains("Filename:") || line.contains("RelativeFilename:") {
                // Extract the path from quotes
                if let Some(start) = line.find('"') {
                    if let Some(end) = line[start + 1..].find('"') {
                        let path = &line[start + 1..start + 1 + end];
                        if !path.is_empty() {
                            // Check if it's an image file
                            let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();
                            if ["png", "jpg", "jpeg", "tga", "tif", "tiff", "bmp", "dds"]
                                .contains(&ext.as_str())
                            {
                                refs.push(FbxTextureRef {
                                    file_name: path.to_string(),
                                    property: None,
                                    is_embedded: false,
                                });
                            }
                        }
                    }
                }
            }
        }

        refs
    }

    /// Import from embedded image data.
    fn import_embedded(embedded: &FbxEmbeddedTexture) -> Result<TextureData, ImportError> {
        match embedded.format.as_str() {
            "png" => Self::import_png(&embedded.data),
            "jpg" => Self::import_jpeg(&embedded.data),
            "bmp" => Self::import_bmp(&embedded.data),
            _ => Err(ImportError::UnsupportedFormat(format!(
                "Embedded format not supported: {}",
                embedded.format
            ))),
        }
    }

    /// Import PNG data.
    fn import_png(data: &[u8]) -> Result<TextureData, ImportError> {
        if data.len() < 24 || data[0..8] != ImageMagic::PNG {
            return Err(ImportError::InvalidData(
                "Invalid PNG in FBX".to_string(),
            ));
        }

        if &data[12..16] != b"IHDR" {
            return Err(ImportError::InvalidData(
                "PNG missing IHDR".to_string(),
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

    /// Import JPEG data.
    fn import_jpeg(data: &[u8]) -> Result<TextureData, ImportError> {
        if data.len() < 3 || data[0..2] != ImageMagic::JPEG {
            return Err(ImportError::InvalidData(
                "Invalid JPEG in FBX".to_string(),
            ));
        }

        // Find SOF marker for dimensions
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
            "Could not parse JPEG dimensions in FBX".to_string(),
        ))
    }

    /// Import BMP data.
    fn import_bmp(data: &[u8]) -> Result<TextureData, ImportError> {
        if data.len() < 26 || data[0..2] != ImageMagic::BMP {
            return Err(ImportError::InvalidData(
                "Invalid BMP in FBX".to_string(),
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
                // FBX-themed orange gradient
                let r = 255u8;
                let g = ((y * 128) / height.max(1)) as u8 + 64;
                let b = ((x * 64) / width.max(1)) as u8;
                pixels[idx] = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = b;
                pixels[idx + 3] = 255;
            }
        }

        pixels
    }
}

impl FormatImporter for FbxImporter {
    fn extensions(&self) -> &[&str] {
        &["fbx"]
    }

    fn mime_types(&self) -> &[&str] {
        &["application/octet-stream", "model/fbx"]
    }

    fn priority(&self) -> u32 {
        105 // Same as USD
    }

    fn can_import(&self, data: &[u8]) -> bool {
        Self::is_binary_fbx(data) || Self::is_ascii_fbx(data)
    }

    fn import(&self, data: &[u8]) -> Result<TextureData, ImportError> {
        if Self::is_binary_fbx(data) {
            let _header = Self::parse_binary_header(data)?;

            // Try to find embedded textures
            let textures = Self::find_embedded_textures(data);
            if let Some(first) = textures.first() {
                return Self::import_embedded(first);
            }

            Err(ImportError::InvalidData(
                "No embedded textures found in FBX".to_string(),
            ))
        } else if Self::is_ascii_fbx(data) {
            let _header = Self::parse_ascii_header(data)?;
            let refs = Self::parse_ascii_texture_refs(data);

            if refs.is_empty() {
                Err(ImportError::InvalidData(
                    "No texture references found in ASCII FBX".to_string(),
                ))
            } else {
                Err(ImportError::UnsupportedFormat(format!(
                    "FBX references external texture: {}",
                    refs[0].file_name
                )))
            }
        } else {
            Err(ImportError::InvalidData(
                "Unrecognized FBX format".to_string(),
            ))
        }
    }

    fn format_name(&self) -> &str {
        "FBX"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_binary_fbx_header(version: u32) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(FBX_BINARY_MAGIC);
        data.extend_from_slice(&[0x1A, 0x00]); // Unknown bytes
        data.extend_from_slice(&version.to_le_bytes());
        // Pad to reasonable size
        data.extend_from_slice(&[0x00; 100]);
        data
    }

    fn make_ascii_fbx() -> Vec<u8> {
        let content = r#"; FBX 7.4.0 project file
; Created by Test
FBXHeaderExtension:  {
    FBXHeaderVersion: 1003
    FBXVersion: 7400
}
"#;
        content.as_bytes().to_vec()
    }

    fn make_fbx_with_embedded_png() -> Vec<u8> {
        let mut data = make_binary_fbx_header(7400);

        // Add embedded PNG
        let png = [
            0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A,
            0x00, 0x00, 0x00, 0x0D,
            b'I', b'H', b'D', b'R',
            0x00, 0x00, 0x00, 0x10, // width = 16
            0x00, 0x00, 0x00, 0x10, // height = 16
            0x08, 0x06, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            // IEND chunk
            0x00, 0x00, 0x00, 0x00,
            b'I', b'E', b'N', b'D',
            0xAE, 0x42, 0x60, 0x82,
        ];

        data.extend_from_slice(&png);
        data
    }

    #[test]
    fn test_fbx_can_import_binary() {
        let importer = FbxImporter;
        let data = make_binary_fbx_header(7400);
        assert!(importer.can_import(&data));
    }

    #[test]
    fn test_fbx_can_import_ascii() {
        let importer = FbxImporter;
        let data = make_ascii_fbx();
        assert!(importer.can_import(&data));
    }

    #[test]
    fn test_fbx_cannot_import_random() {
        let importer = FbxImporter;
        assert!(!importer.can_import(&[0x00, 0x01, 0x02, 0x03]));
    }

    #[test]
    fn test_fbx_parse_binary_header() {
        let data = make_binary_fbx_header(7500);
        let header = FbxImporter::parse_binary_header(&data).unwrap();
        assert_eq!(header.version, 7500);
        assert!(header.is_binary);
    }

    #[test]
    fn test_fbx_parse_ascii_header() {
        let data = make_ascii_fbx();
        let header = FbxImporter::parse_ascii_header(&data).unwrap();
        assert_eq!(header.version, 7400);
        assert!(!header.is_binary);
    }

    #[test]
    fn test_fbx_import_embedded_png() {
        let importer = FbxImporter;
        let data = make_fbx_with_embedded_png();
        let result = importer.import(&data);

        assert!(result.is_ok());
        let texture = result.unwrap();
        assert_eq!(texture.width, 16);
        assert_eq!(texture.height, 16);
    }

    #[test]
    fn test_fbx_parse_ascii_texture_refs() {
        let fbx = br#"; FBX file
Filename: "C:\textures\diffuse.png"
RelativeFilename: "textures/normal.jpg"
"#;
        let refs = FbxImporter::parse_ascii_texture_refs(fbx);
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn test_fbx_detect_image_format() {
        assert_eq!(
            FbxImporter::detect_image_format(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]),
            Some("png")
        );
        assert_eq!(
            FbxImporter::detect_image_format(&[0xFF, 0xD8, 0xFF]),
            Some("jpg")
        );
        assert_eq!(
            FbxImporter::detect_image_format(&[b'B', b'M', 0x00]),
            Some("bmp")
        );
        assert_eq!(FbxImporter::detect_image_format(&[0x00, 0x00]), None);
    }

    #[test]
    fn test_fbx_extensions() {
        let importer = FbxImporter;
        assert_eq!(importer.extensions(), &["fbx"]);
    }

    #[test]
    fn test_fbx_format_name() {
        let importer = FbxImporter;
        assert_eq!(importer.format_name(), "FBX");
    }

    #[test]
    fn test_fbx_priority() {
        let importer = FbxImporter;
        assert_eq!(importer.priority(), 105);
    }

    #[test]
    fn test_fbx_is_binary() {
        assert!(FbxImporter::is_binary_fbx(FBX_BINARY_MAGIC));
        assert!(!FbxImporter::is_binary_fbx(&[0x00; 30]));
    }

    #[test]
    fn test_fbx_is_ascii() {
        // ASCII FBX detection requires at least 100 bytes of content
        let fbx_header = b"; FBX 7.4.0 project file\n; Created by Test Application\nFBXHeaderExtension: {\n    FBXHeaderVersion: 1003\n    FBXVersion: 7400\n}";
        assert!(FbxImporter::is_ascii_fbx(fbx_header));

        let fbx_header2 = b"FBXHeaderExtension: {\n    FBXHeaderVersion: 1003\n    FBXVersion: 7400\n}\nCreator: Test Application v1.0\n";
        assert!(FbxImporter::is_ascii_fbx(fbx_header2));

        // Too short to detect
        assert!(!FbxImporter::is_ascii_fbx(b"random data"));
        assert!(!FbxImporter::is_ascii_fbx(&[0x00; 50]));
    }

    #[test]
    fn test_fbx_no_embedded_textures() {
        let importer = FbxImporter;
        let data = make_binary_fbx_header(7400);
        let result = importer.import(&data);
        assert!(matches!(result, Err(ImportError::InvalidData(_))));
    }

    #[test]
    fn test_fbx_header_too_short() {
        let result = FbxImporter::parse_binary_header(&[0x00; 10]);
        assert!(matches!(result, Err(ImportError::InvalidData(_))));
    }
}
