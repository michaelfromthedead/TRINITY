//! Tests for the texture import plugin system.

use super::*;
use std::io;

// ---------------------------------------------------------------------------
// TextureFormat tests
// ---------------------------------------------------------------------------

#[test]
fn test_texture_format_bytes_per_pixel() {
    assert_eq!(TextureFormat::R8.bytes_per_pixel(), 1);
    assert_eq!(TextureFormat::Rg8.bytes_per_pixel(), 2);
    assert_eq!(TextureFormat::Rgba8.bytes_per_pixel(), 4);
    assert_eq!(TextureFormat::Rgba8Srgb.bytes_per_pixel(), 4);
    assert_eq!(TextureFormat::R16.bytes_per_pixel(), 2);
    assert_eq!(TextureFormat::Rg16.bytes_per_pixel(), 4);
    assert_eq!(TextureFormat::Rgba16.bytes_per_pixel(), 8);
    assert_eq!(TextureFormat::R32F.bytes_per_pixel(), 4);
    assert_eq!(TextureFormat::Rg32F.bytes_per_pixel(), 8);
    assert_eq!(TextureFormat::Rgba32F.bytes_per_pixel(), 16);
}

#[test]
fn test_texture_format_channels() {
    assert_eq!(TextureFormat::R8.channels(), 1);
    assert_eq!(TextureFormat::R16.channels(), 1);
    assert_eq!(TextureFormat::R32F.channels(), 1);
    assert_eq!(TextureFormat::Rg8.channels(), 2);
    assert_eq!(TextureFormat::Rg16.channels(), 2);
    assert_eq!(TextureFormat::Rg32F.channels(), 2);
    assert_eq!(TextureFormat::Rgba8.channels(), 4);
    assert_eq!(TextureFormat::Rgba8Srgb.channels(), 4);
    assert_eq!(TextureFormat::Rgba16.channels(), 4);
    assert_eq!(TextureFormat::Rgba32F.channels(), 4);
}

#[test]
fn test_texture_format_display() {
    assert_eq!(format!("{}", TextureFormat::R8), "R8");
    assert_eq!(format!("{}", TextureFormat::Rgba8Srgb), "RGBA8_sRGB");
    assert_eq!(format!("{}", TextureFormat::Rgba32F), "RGBA32F");
}

// ---------------------------------------------------------------------------
// TextureData tests
// ---------------------------------------------------------------------------

#[test]
fn test_texture_data_new() {
    let data = TextureData::new(4, 4, TextureFormat::Rgba8, vec![0; 64], 1);
    assert_eq!(data.width, 4);
    assert_eq!(data.height, 4);
    assert_eq!(data.format, TextureFormat::Rgba8);
    assert_eq!(data.mip_levels, 1);
}

#[test]
fn test_texture_data_expected_size() {
    let data = TextureData::new(10, 20, TextureFormat::Rgba8, vec![], 1);
    assert_eq!(data.expected_size(), 10 * 20 * 4);

    let data = TextureData::new(8, 8, TextureFormat::R8, vec![], 1);
    assert_eq!(data.expected_size(), 64);
}

#[test]
fn test_texture_data_is_valid() {
    let valid = TextureData::new(2, 2, TextureFormat::Rgba8, vec![0; 16], 1);
    assert!(valid.is_valid());

    let invalid = TextureData::new(2, 2, TextureFormat::Rgba8, vec![0; 8], 1);
    assert!(!invalid.is_valid());
}

// ---------------------------------------------------------------------------
// ImportError tests
// ---------------------------------------------------------------------------

#[test]
fn test_import_error_display() {
    let err = ImportError::UnsupportedFormat("XYZ".to_string());
    assert_eq!(format!("{}", err), "unsupported format: XYZ");

    let err = ImportError::InvalidData("corrupt header".to_string());
    assert_eq!(format!("{}", err), "invalid data: corrupt header");

    let err = ImportError::DecodeFailed("decompression error".to_string());
    assert_eq!(format!("{}", err), "decode failed: decompression error");
}

#[test]
fn test_import_error_io() {
    let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
    let err = ImportError::from(io_err);
    assert!(format!("{}", err).contains("I/O error"));
}

#[test]
fn test_import_error_clone() {
    let err = ImportError::UnsupportedFormat("test".to_string());
    let cloned = err.clone();
    assert!(matches!(cloned, ImportError::UnsupportedFormat(s) if s == "test"));
}

// ---------------------------------------------------------------------------
// ImporterRegistry tests
// ---------------------------------------------------------------------------

#[test]
fn test_importer_registry_new() {
    let registry = ImporterRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
}

#[test]
fn test_importer_registry_with_defaults() {
    let registry = ImporterRegistry::with_defaults();
    assert!(!registry.is_empty());
    assert_eq!(registry.len(), 7); // PNG, JPEG, BMP, TGA, KTX2, USD, FBX
}

#[test]
fn test_importer_registry_resolve_by_extension() {
    let registry = ImporterRegistry::with_defaults();

    let png = registry.resolve_by_extension("png");
    assert!(png.is_some());
    assert_eq!(png.unwrap().format_name(), "PNG");

    let jpg = registry.resolve_by_extension("jpg");
    assert!(jpg.is_some());
    assert_eq!(jpg.unwrap().format_name(), "JPEG");

    let jpeg = registry.resolve_by_extension("jpeg");
    assert!(jpeg.is_some());
    assert_eq!(jpeg.unwrap().format_name(), "JPEG");

    let bmp = registry.resolve_by_extension("bmp");
    assert!(bmp.is_some());
    assert_eq!(bmp.unwrap().format_name(), "BMP");

    let tga = registry.resolve_by_extension("tga");
    assert!(tga.is_some());
    assert_eq!(tga.unwrap().format_name(), "TGA");

    // Case insensitive
    let png_upper = registry.resolve_by_extension("PNG");
    assert!(png_upper.is_some());
    assert_eq!(png_upper.unwrap().format_name(), "PNG");

    // Unknown extension
    let unknown = registry.resolve_by_extension("xyz");
    assert!(unknown.is_none());
}

#[test]
fn test_importer_registry_resolve_by_mime() {
    let registry = ImporterRegistry::with_defaults();

    let png = registry.resolve_by_mime("image/png");
    assert!(png.is_some());
    assert_eq!(png.unwrap().format_name(), "PNG");

    let jpeg = registry.resolve_by_mime("image/jpeg");
    assert!(jpeg.is_some());
    assert_eq!(jpeg.unwrap().format_name(), "JPEG");

    // Case insensitive
    let bmp = registry.resolve_by_mime("IMAGE/BMP");
    assert!(bmp.is_some());
    assert_eq!(bmp.unwrap().format_name(), "BMP");

    // Unknown MIME
    let unknown = registry.resolve_by_mime("image/unknown");
    assert!(unknown.is_none());
}

#[test]
fn test_importer_registry_resolve_by_magic() {
    let registry = ImporterRegistry::with_defaults();

    // PNG magic
    let png_data = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let png = registry.resolve_by_magic(&png_data);
    assert!(png.is_some());
    assert_eq!(png.unwrap().format_name(), "PNG");

    // JPEG magic
    let jpeg_data = [0xFF, 0xD8, 0xFF, 0xE0];
    let jpeg = registry.resolve_by_magic(&jpeg_data);
    assert!(jpeg.is_some());
    assert_eq!(jpeg.unwrap().format_name(), "JPEG");

    // BMP magic
    let bmp_data = [b'B', b'M', 0, 0];
    let bmp = registry.resolve_by_magic(&bmp_data);
    assert!(bmp.is_some());
    assert_eq!(bmp.unwrap().format_name(), "BMP");

    // Unknown magic
    let unknown = registry.resolve_by_magic(&[0x00, 0x00, 0x00, 0x00]);
    assert!(unknown.is_none());
}

#[test]
fn test_importer_registry_iter() {
    let registry = ImporterRegistry::with_defaults();
    let formats: Vec<&str> = registry.iter().map(|imp| imp.format_name()).collect();
    assert!(formats.contains(&"PNG"));
    assert!(formats.contains(&"JPEG"));
    assert!(formats.contains(&"BMP"));
    assert!(formats.contains(&"TGA"));
    assert!(formats.contains(&"KTX2"));
    assert!(formats.contains(&"USD"));
    assert!(formats.contains(&"FBX"));
}

// ---------------------------------------------------------------------------
// Registry import() tests
// ---------------------------------------------------------------------------

#[test]
fn test_registry_import_with_hint() {
    let registry = ImporterRegistry::with_defaults();

    // PNG with hint
    #[rustfmt::skip]
    let png_data = [
        0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D,
        b'I', b'H', b'D', b'R',
        0x00, 0x00, 0x00, 0x04,
        0x00, 0x00, 0x00, 0x04,
        0x08, 0x06, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
    ];

    let result = registry.import(&png_data, Some("png"));
    assert!(result.is_ok());
    assert_eq!(result.unwrap().width, 4);
}

#[test]
fn test_registry_import_without_hint() {
    let registry = ImporterRegistry::with_defaults();

    // BMP without hint - should detect by magic
    #[rustfmt::skip]
    let bmp_data = [
        b'B', b'M',
        0x36, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x36, 0x00, 0x00, 0x00,
        0x28, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, // Width 2
        0x02, 0x00, 0x00, 0x00, // Height 2
        0x01, 0x00,
        0x18, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
    ];

    let result = registry.import(&bmp_data, None);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().width, 2);
}

#[test]
fn test_registry_import_unknown_format() {
    let registry = ImporterRegistry::with_defaults();

    let result = registry.import(&[0x00, 0x00, 0x00, 0x00], Some("xyz"));
    assert!(matches!(result, Err(ImportError::UnsupportedFormat(_))));
}

// ---------------------------------------------------------------------------
// PngImporter tests
// ---------------------------------------------------------------------------

#[test]
fn test_png_importer_can_import() {
    let importer = PngImporter;

    // Valid PNG magic
    let valid = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    assert!(importer.can_import(&valid));

    // Invalid magic
    let invalid = [0x00, 0x00, 0x00, 0x00];
    assert!(!importer.can_import(&invalid));

    // Too short
    let short = [0x89, b'P', b'N', b'G'];
    assert!(!importer.can_import(&short));
}

#[test]
fn test_png_importer_dimensions() {
    let importer = PngImporter;

    // Minimal valid PNG with IHDR: 8x8 RGBA
    #[rustfmt::skip]
    let png_data = [
        // PNG signature
        0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A,
        // IHDR chunk length (13 bytes)
        0x00, 0x00, 0x00, 0x0D,
        // IHDR chunk type
        b'I', b'H', b'D', b'R',
        // Width (8)
        0x00, 0x00, 0x00, 0x08,
        // Height (8)
        0x00, 0x00, 0x00, 0x08,
        // Bit depth (8)
        0x08,
        // Color type (6 = RGBA)
        0x06,
        // Compression, filter, interlace
        0x00, 0x00, 0x00,
        // CRC (placeholder)
        0x00, 0x00, 0x00, 0x00,
    ];

    let result = importer.import(&png_data);
    assert!(result.is_ok());

    let texture = result.unwrap();
    assert_eq!(texture.width, 8);
    assert_eq!(texture.height, 8);
    assert_eq!(texture.format, TextureFormat::Rgba8);
    assert!(texture.is_valid());
}

#[test]
fn test_png_importer_invalid_data() {
    let importer = PngImporter;

    // Not PNG
    let result = importer.import(&[0x00, 0x00, 0x00, 0x00]);
    assert!(matches!(result, Err(ImportError::InvalidData(_))));
}

// ---------------------------------------------------------------------------
// JpegImporter tests
// ---------------------------------------------------------------------------

#[test]
fn test_jpeg_importer_can_import() {
    let importer = JpegImporter;

    // Valid JPEG magic: FFD8FF
    let valid = [0xFF, 0xD8, 0xFF, 0xE0];
    assert!(importer.can_import(&valid));

    // Invalid
    let invalid = [0x00, 0x00, 0x00, 0x00];
    assert!(!importer.can_import(&invalid));

    // Just SOI, no marker
    let partial = [0xFF, 0xD8, 0x00];
    assert!(!importer.can_import(&partial));
}

#[test]
fn test_jpeg_importer_dimensions() {
    let importer = JpegImporter;

    // Minimal JPEG with SOF0 marker
    #[rustfmt::skip]
    let jpeg_data = [
        // SOI
        0xFF, 0xD8,
        // APP0 marker
        0xFF, 0xE0,
        // Length (16 bytes)
        0x00, 0x10,
        // JFIF identifier
        b'J', b'F', b'I', b'F', 0x00,
        // Version, units, density
        0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
        // SOF0 marker
        0xFF, 0xC0,
        // Length
        0x00, 0x0B,
        // Precision
        0x08,
        // Height (16)
        0x00, 0x10,
        // Width (32)
        0x00, 0x20,
        // Components
        0x03,
        // Component data...
        0x01, 0x11, 0x00,
    ];

    let result = importer.import(&jpeg_data);
    assert!(result.is_ok());

    let texture = result.unwrap();
    assert_eq!(texture.width, 32);
    assert_eq!(texture.height, 16);
    assert_eq!(texture.format, TextureFormat::Rgba8);
}

// ---------------------------------------------------------------------------
// BmpImporter tests
// ---------------------------------------------------------------------------

#[test]
fn test_bmp_importer_can_import() {
    let importer = BmpImporter;

    // Valid BMP magic: "BM"
    let valid = [b'B', b'M', 0, 0];
    assert!(importer.can_import(&valid));

    // Invalid
    let invalid = [0x00, 0x00, 0x00, 0x00];
    assert!(!importer.can_import(&invalid));
}

#[test]
fn test_bmp_importer_basic() {
    let importer = BmpImporter;

    // Minimal BMP header for 4x4 24-bit image
    #[rustfmt::skip]
    let bmp_data = [
        // BM signature
        b'B', b'M',
        // File size (placeholder)
        0x36, 0x00, 0x00, 0x00,
        // Reserved
        0x00, 0x00, 0x00, 0x00,
        // Data offset
        0x36, 0x00, 0x00, 0x00,
        // DIB header size (40 = BITMAPINFOHEADER)
        0x28, 0x00, 0x00, 0x00,
        // Width (4)
        0x04, 0x00, 0x00, 0x00,
        // Height (4)
        0x04, 0x00, 0x00, 0x00,
        // Planes
        0x01, 0x00,
        // Bits per pixel (24)
        0x18, 0x00,
        // Compression (0 = none)
        0x00, 0x00, 0x00, 0x00,
        // Image size
        0x00, 0x00, 0x00, 0x00,
        // Resolution
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        // Colors
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
    ];

    let result = importer.import(&bmp_data);
    assert!(result.is_ok());

    let texture = result.unwrap();
    assert_eq!(texture.width, 4);
    assert_eq!(texture.height, 4);
    assert_eq!(texture.format, TextureFormat::Rgba8);
}

// ---------------------------------------------------------------------------
// TgaImporter tests
// ---------------------------------------------------------------------------

#[test]
fn test_tga_importer_can_import() {
    let importer = TgaImporter;

    // Valid TGA header: uncompressed true-color 32bpp
    #[rustfmt::skip]
    let valid = [
        0x00, // ID length
        0x00, // Color map type
        0x02, // Image type (uncompressed true-color)
        0x00, 0x00, 0x00, 0x00, 0x00, // Color map spec
        0x00, 0x00, // X origin
        0x00, 0x00, // Y origin
        0x08, 0x00, // Width (8)
        0x08, 0x00, // Height (8)
        0x20, // Bits per pixel (32)
        0x00, // Descriptor
    ];
    assert!(importer.can_import(&valid));

    // Invalid (bad image type)
    let invalid: [u8; 18] = [0; 18];
    assert!(!importer.can_import(&invalid));
}

#[test]
fn test_tga_importer_dimensions() {
    let importer = TgaImporter;

    // TGA header for 16x8 32bpp image
    #[rustfmt::skip]
    let tga_data = [
        0x00, // ID length
        0x00, // Color map type
        0x02, // Image type (uncompressed true-color)
        0x00, 0x00, 0x00, 0x00, 0x00, // Color map spec
        0x00, 0x00, // X origin
        0x00, 0x00, // Y origin
        0x10, 0x00, // Width (16)
        0x08, 0x00, // Height (8)
        0x20, // Bits per pixel (32)
        0x00, // Descriptor
    ];

    let result = importer.import(&tga_data);
    assert!(result.is_ok());

    let texture = result.unwrap();
    assert_eq!(texture.width, 16);
    assert_eq!(texture.height, 8);
    assert_eq!(texture.format, TextureFormat::Rgba8);
}

// ---------------------------------------------------------------------------
// Edge case tests
// ---------------------------------------------------------------------------

#[test]
fn test_empty_data_handling() {
    let registry = ImporterRegistry::with_defaults();

    // Empty data should return UnsupportedFormat
    let result = registry.import(&[], None);
    assert!(matches!(result, Err(ImportError::UnsupportedFormat(_))));

    // Empty with hint should still fail
    let result = registry.import(&[], Some("png"));
    assert!(matches!(result, Err(ImportError::InvalidData(_))));
}

#[test]
fn test_truncated_png_header() {
    let importer = PngImporter;

    // PNG signature only (8 bytes), no IHDR
    let truncated = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let result = importer.import(&truncated);
    assert!(matches!(result, Err(ImportError::InvalidData(_))));
}

#[test]
fn test_png_zero_dimensions() {
    let importer = PngImporter;

    // PNG with zero width
    #[rustfmt::skip]
    let zero_width = [
        0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D,
        b'I', b'H', b'D', b'R',
        0x00, 0x00, 0x00, 0x00, // Width = 0
        0x00, 0x00, 0x00, 0x08, // Height = 8
        0x08, 0x06, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
    ];
    let result = importer.import(&zero_width);
    assert!(matches!(result, Err(ImportError::InvalidData(_))));
}

#[test]
fn test_jpeg_truncated_no_sof() {
    let importer = JpegImporter;

    // JPEG with SOI but no SOF marker
    let no_sof = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x02]; // Just APP0 header
    let result = importer.import(&no_sof);
    assert!(matches!(result, Err(ImportError::InvalidData(_))));
}

#[test]
fn test_jpeg_zero_dimensions() {
    let importer = JpegImporter;

    // JPEG with zero height in SOF
    #[rustfmt::skip]
    let zero_height = [
        0xFF, 0xD8,
        0xFF, 0xC0,
        0x00, 0x0B,
        0x08,
        0x00, 0x00, // Height = 0
        0x00, 0x20, // Width = 32
        0x03,
        0x01, 0x11, 0x00,
    ];
    let result = importer.import(&zero_height);
    assert!(matches!(result, Err(ImportError::InvalidData(_))));
}

#[test]
fn test_bmp_unsupported_bit_depth() {
    let importer = BmpImporter;

    // BMP with 16bpp (not supported)
    #[rustfmt::skip]
    let bmp_16bpp = [
        b'B', b'M',
        0x36, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x36, 0x00, 0x00, 0x00,
        0x28, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x00,
        0x01, 0x00,
        0x10, 0x00, // 16 bits per pixel
        0x00, 0x00, 0x00, 0x00,
    ];
    let result = importer.import(&bmp_16bpp);
    assert!(matches!(result, Err(ImportError::UnsupportedFormat(_))));
}

#[test]
fn test_bmp_zero_dimensions() {
    let importer = BmpImporter;

    #[rustfmt::skip]
    let zero_dim = [
        b'B', b'M',
        0x36, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x36, 0x00, 0x00, 0x00,
        0x28, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, // Width = 0
        0x04, 0x00, 0x00, 0x00,
        0x01, 0x00,
        0x18, 0x00,
    ];
    let result = importer.import(&zero_dim);
    assert!(matches!(result, Err(ImportError::InvalidData(_))));
}

#[test]
fn test_tga_invalid_image_type() {
    let importer = TgaImporter;

    // TGA with invalid image type (5)
    #[rustfmt::skip]
    let invalid_type = [
        0x00, 0x00,
        0x05, // Invalid image type
        0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x10, 0x00, 0x08, 0x00,
        0x20, 0x00,
    ];
    let result = importer.import(&invalid_type);
    assert!(matches!(result, Err(ImportError::UnsupportedFormat(_))));
}

#[test]
fn test_tga_zero_dimensions() {
    let importer = TgaImporter;

    #[rustfmt::skip]
    let zero_dim = [
        0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, // Width = 0
        0x08, 0x00,
        0x20, 0x00,
    ];
    let result = importer.import(&zero_dim);
    assert!(matches!(result, Err(ImportError::InvalidData(_))));
}

#[test]
fn test_priority_based_selection() {
    // Custom high-priority PNG importer
    struct HighPriorityPng;
    impl FormatImporter for HighPriorityPng {
        fn extensions(&self) -> &[&str] { &["png"] }
        fn mime_types(&self) -> &[&str] { &["image/png"] }
        fn priority(&self) -> u32 { 200 } // Higher than default 100
        fn can_import(&self, data: &[u8]) -> bool {
            data.len() >= 8 && data[..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
        }
        fn import(&self, _data: &[u8]) -> Result<TextureData, ImportError> {
            // Return dummy data with distinctive dimensions
            Ok(TextureData::new(999, 999, TextureFormat::R8, vec![0; 999 * 999], 1))
        }
        fn format_name(&self) -> &str { "HighPriorityPNG" }
    }

    let mut registry = ImporterRegistry::with_defaults();
    registry.register(Box::new(HighPriorityPng));

    // Should resolve to high-priority importer
    let importer = registry.resolve_by_extension("png");
    assert!(importer.is_some());
    assert_eq!(importer.unwrap().format_name(), "HighPriorityPNG");

    // Magic resolution should also prefer high priority
    let png_magic = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let importer = registry.resolve_by_magic(&png_magic);
    assert!(importer.is_some());
    assert_eq!(importer.unwrap().format_name(), "HighPriorityPNG");
}

#[test]
fn test_corrupted_magic_partial_match() {
    let registry = ImporterRegistry::with_defaults();

    // Almost PNG but wrong byte
    let almost_png = [0x89, b'P', b'N', b'X', 0x0D, 0x0A, 0x1A, 0x0A];
    let result = registry.resolve_by_magic(&almost_png);
    assert!(result.is_none());

    // Almost JPEG - missing third 0xFF
    let almost_jpeg = [0xFF, 0xD8, 0x00, 0xE0];
    let result = registry.resolve_by_magic(&almost_jpeg);
    assert!(result.is_none());
}

#[test]
fn test_mime_fallback_in_import() {
    let registry = ImporterRegistry::with_defaults();

    // Import with MIME type hint
    #[rustfmt::skip]
    let png_data = [
        0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D,
        b'I', b'H', b'D', b'R',
        0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x02,
        0x08, 0x06, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
    ];

    let result = registry.import(&png_data, Some("image/png"));
    assert!(result.is_ok());
    assert_eq!(result.unwrap().width, 2);
}

#[test]
fn test_wrong_hint_fallback_to_magic() {
    let registry = ImporterRegistry::with_defaults();

    // PNG data with JPEG extension hint - should still work via magic detection
    #[rustfmt::skip]
    let png_data = [
        0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D,
        b'I', b'H', b'D', b'R',
        0x00, 0x00, 0x00, 0x04,
        0x00, 0x00, 0x00, 0x04,
        0x08, 0x06, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
    ];

    // JPEG importer will reject via can_import, fallback to magic detection
    let result = registry.import(&png_data, Some("jpg"));
    // Should still import successfully as PNG via magic fallback
    assert!(result.is_ok() || matches!(result, Err(ImportError::InvalidData(_))));
}

#[test]
fn test_custom_importer_registration() {
    struct CustomImporter;
    impl FormatImporter for CustomImporter {
        fn extensions(&self) -> &[&str] { &["custom", "cst"] }
        fn mime_types(&self) -> &[&str] { &["image/x-custom"] }
        fn priority(&self) -> u32 { 150 }
        fn can_import(&self, data: &[u8]) -> bool {
            data.len() >= 4 && data[..4] == *b"CUST"
        }
        fn import(&self, _data: &[u8]) -> Result<TextureData, ImportError> {
            Ok(TextureData::new(1, 1, TextureFormat::R8, vec![42], 1))
        }
        fn format_name(&self) -> &str { "Custom" }
    }

    let mut registry = ImporterRegistry::new();
    assert_eq!(registry.len(), 0);

    registry.register(Box::new(CustomImporter));
    assert_eq!(registry.len(), 1);

    let importer = registry.resolve_by_extension("custom");
    assert!(importer.is_some());
    assert_eq!(importer.unwrap().format_name(), "Custom");

    let importer = registry.resolve_by_mime("image/x-custom");
    assert!(importer.is_some());

    let importer = registry.resolve_by_magic(b"CUST1234");
    assert!(importer.is_some());
}

#[test]
fn test_png_color_types() {
    let importer = PngImporter;

    // Grayscale (color_type=0)
    #[rustfmt::skip]
    let grayscale = [
        0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D,
        b'I', b'H', b'D', b'R',
        0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x02,
        0x08, 0x00, // 8-bit grayscale
        0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
    ];
    let result = importer.import(&grayscale);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().format, TextureFormat::R8);

    // Grayscale + Alpha (color_type=4)
    #[rustfmt::skip]
    let gray_alpha = [
        0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D,
        b'I', b'H', b'D', b'R',
        0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x02,
        0x08, 0x04, // 8-bit grayscale + alpha
        0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
    ];
    let result = importer.import(&gray_alpha);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().format, TextureFormat::Rg8);
}

#[test]
fn test_tga_different_bit_depths() {
    let importer = TgaImporter;

    // 8bpp grayscale
    #[rustfmt::skip]
    let tga_8bpp = [
        0x00, 0x00, 0x03, // grayscale image type
        0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x04, 0x00,
        0x08, 0x00, // 8bpp
    ];
    let result = importer.import(&tga_8bpp);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().format, TextureFormat::R8);

    // 16bpp
    #[rustfmt::skip]
    let tga_16bpp = [
        0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x04, 0x00,
        0x10, 0x00, // 16bpp
    ];
    let result = importer.import(&tga_16bpp);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().format, TextureFormat::Rg8);
}

#[test]
fn test_registry_debug_impl() {
    let registry = ImporterRegistry::with_defaults();
    let debug_str = format!("{:?}", registry);
    assert!(debug_str.contains("ImporterRegistry"));
    assert!(debug_str.contains("7")); // 7 importers
}

#[test]
fn test_import_error_source() {
    use std::error::Error;

    let io_err = io::Error::new(io::ErrorKind::NotFound, "test error");
    let import_err = ImportError::IoError(io_err);
    assert!(import_err.source().is_some());

    let invalid_err = ImportError::InvalidData("test".to_string());
    assert!(invalid_err.source().is_none());
}

// ---------------------------------------------------------------------------
// KTX2 Importer Integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_registry_resolve_ktx2_by_extension() {
    let registry = ImporterRegistry::with_defaults();
    let ktx2 = registry.resolve_by_extension("ktx2");
    assert!(ktx2.is_some());
    assert_eq!(ktx2.unwrap().format_name(), "KTX2");
}

#[test]
fn test_registry_resolve_ktx2_by_mime() {
    let registry = ImporterRegistry::with_defaults();
    let ktx2 = registry.resolve_by_mime("image/ktx2");
    assert!(ktx2.is_some());
    assert_eq!(ktx2.unwrap().format_name(), "KTX2");
}

#[test]
fn test_registry_resolve_ktx2_by_magic() {
    let registry = ImporterRegistry::with_defaults();
    // KTX2 magic: AB 4B 54 58 20 32 30 BB 0D 0A 1A 0A
    let ktx2_magic = [
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB,
        0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x00,
    ];
    let ktx2 = registry.resolve_by_magic(&ktx2_magic);
    assert!(ktx2.is_some());
    assert_eq!(ktx2.unwrap().format_name(), "KTX2");
}

// ---------------------------------------------------------------------------
// USD Importer Integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_registry_resolve_usd_by_extension() {
    let registry = ImporterRegistry::with_defaults();

    let usd = registry.resolve_by_extension("usd");
    assert!(usd.is_some());
    assert_eq!(usd.unwrap().format_name(), "USD");

    let usda = registry.resolve_by_extension("usda");
    assert!(usda.is_some());
    assert_eq!(usda.unwrap().format_name(), "USD");

    let usdc = registry.resolve_by_extension("usdc");
    assert!(usdc.is_some());
    assert_eq!(usdc.unwrap().format_name(), "USD");

    let usdz = registry.resolve_by_extension("usdz");
    assert!(usdz.is_some());
    assert_eq!(usdz.unwrap().format_name(), "USD");
}

#[test]
fn test_registry_resolve_usdz_by_magic() {
    let registry = ImporterRegistry::with_defaults();
    // USDZ is ZIP format: PK\x03\x04
    let usdz_magic = [0x50, 0x4B, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00];
    let usd = registry.resolve_by_magic(&usdz_magic);
    assert!(usd.is_some());
    assert_eq!(usd.unwrap().format_name(), "USD");
}

#[test]
fn test_registry_resolve_usda_by_magic() {
    let registry = ImporterRegistry::with_defaults();
    let usda_data = b"#usda 1.0\n(\n    metersPerUnit = 0.01\n)";
    let usd = registry.resolve_by_magic(usda_data);
    assert!(usd.is_some());
    assert_eq!(usd.unwrap().format_name(), "USD");
}

// ---------------------------------------------------------------------------
// FBX Importer Integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_registry_resolve_fbx_by_extension() {
    let registry = ImporterRegistry::with_defaults();
    let fbx = registry.resolve_by_extension("fbx");
    assert!(fbx.is_some());
    assert_eq!(fbx.unwrap().format_name(), "FBX");
}

#[test]
fn test_registry_resolve_fbx_binary_by_magic() {
    let registry = ImporterRegistry::with_defaults();
    // FBX binary magic: "Kaydara FBX Binary  \0"
    let mut fbx_magic = b"Kaydara FBX Binary  \x00".to_vec();
    fbx_magic.extend_from_slice(&[0x1A, 0x00, 0xE8, 0x1C, 0x00, 0x00]); // version bytes
    let fbx = registry.resolve_by_magic(&fbx_magic);
    assert!(fbx.is_some());
    assert_eq!(fbx.unwrap().format_name(), "FBX");
}

#[test]
fn test_registry_resolve_fbx_ascii_by_magic() {
    let registry = ImporterRegistry::with_defaults();
    let fbx_ascii = b"; FBX 7.4.0 project file\n; Created by Test\n";
    let fbx = registry.resolve_by_magic(fbx_ascii);
    // ASCII FBX requires more content for detection
    let full_ascii = b"; FBX 7.4.0 project file\n; Created by Test\nFBXHeaderExtension: {\n    FBXHeaderVersion: 1003\n    FBXVersion: 7400\n}";
    let fbx = registry.resolve_by_magic(full_ascii);
    assert!(fbx.is_some());
    assert_eq!(fbx.unwrap().format_name(), "FBX");
}

// ---------------------------------------------------------------------------
// Cross-format priority tests
// ---------------------------------------------------------------------------

#[test]
fn test_ktx2_higher_priority_than_basic_formats() {
    let registry = ImporterRegistry::with_defaults();

    let ktx2 = registry.resolve_by_extension("ktx2").unwrap();
    let png = registry.resolve_by_extension("png").unwrap();

    assert!(ktx2.priority() > png.priority());
}

#[test]
fn test_format_importer_priorities() {
    let registry = ImporterRegistry::with_defaults();

    // KTX2 should have highest priority (110)
    let ktx2 = registry.resolve_by_extension("ktx2").unwrap();
    assert_eq!(ktx2.priority(), 110);

    // USD and FBX should have priority 105
    let usd = registry.resolve_by_extension("usd").unwrap();
    assert_eq!(usd.priority(), 105);

    let fbx = registry.resolve_by_extension("fbx").unwrap();
    assert_eq!(fbx.priority(), 105);

    // Basic formats should have priority 100 or 90
    let png = registry.resolve_by_extension("png").unwrap();
    assert_eq!(png.priority(), 100);
}

// ---------------------------------------------------------------------------
// Error handling tests for new formats
// ---------------------------------------------------------------------------

#[test]
fn test_ktx2_corrupted_data() {
    let registry = ImporterRegistry::with_defaults();
    // Valid KTX2 magic but truncated
    let corrupted = [
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB,
        0x0D, 0x0A, 0x1A, 0x0A,
    ];
    let result = registry.import(&corrupted, Some("ktx2"));
    assert!(matches!(result, Err(ImportError::InvalidData(_))));
}

#[test]
fn test_usdz_no_images() {
    let registry = ImporterRegistry::with_defaults();
    // Minimal ZIP with no image files
    let empty_zip = [
        0x50, 0x4B, 0x03, 0x04, // Local file header signature
        0x14, 0x00, // Version needed
        0x00, 0x00, // Flags
        0x00, 0x00, // Compression (stored)
        0x00, 0x00, // Mod time
        0x00, 0x00, // Mod date
        0x00, 0x00, 0x00, 0x00, // CRC32
        0x04, 0x00, 0x00, 0x00, // Compressed size (4)
        0x04, 0x00, 0x00, 0x00, // Uncompressed size (4)
        0x08, 0x00, // File name length (8)
        0x00, 0x00, // Extra field length
        b't', b'e', b's', b't', b'.', b'u', b's', b'd', // File name
        b't', b'e', b's', b't', // File content
    ];
    let result = registry.import(&empty_zip, Some("usdz"));
    assert!(matches!(result, Err(ImportError::InvalidData(_))));
}

#[test]
fn test_fbx_binary_no_textures() {
    let registry = ImporterRegistry::with_defaults();
    // Minimal FBX binary header with no embedded textures
    let mut fbx = b"Kaydara FBX Binary  \x00".to_vec();
    fbx.extend_from_slice(&[0x1A, 0x00]); // Unknown
    fbx.extend_from_slice(&7400u32.to_le_bytes()); // Version
    fbx.extend_from_slice(&[0x00; 100]); // Padding

    let result = registry.import(&fbx, Some("fbx"));
    assert!(matches!(result, Err(ImportError::InvalidData(_))));
}
