//! Built-in texture format importers.
//!
//! This module provides importers for common image formats: PNG, JPEG, BMP, and TGA.

use super::{FormatImporter, ImportError, TextureData, TextureFormat};

// ---------------------------------------------------------------------------
// PNG Importer
// ---------------------------------------------------------------------------

/// PNG format importer.
///
/// Parses PNG files according to the PNG specification. Extracts image
/// dimensions from the IHDR chunk and produces placeholder pixel data.
pub struct PngImporter;

impl PngImporter {
    /// PNG magic bytes: 137 80 78 71 13 10 26 10
    const MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

    /// Parse PNG dimensions from IHDR chunk.
    fn parse_ihdr(data: &[u8]) -> Option<(u32, u32, u8, u8)> {
        // PNG structure: signature (8) + chunks
        // IHDR chunk: length (4) + "IHDR" (4) + width (4) + height (4) + bit_depth (1) + color_type (1) + ...
        if data.len() < 24 {
            return None;
        }

        // Skip signature (8 bytes)
        let chunk_start = 8;

        // Read chunk length and type
        let _length = u32::from_be_bytes([
            data[chunk_start],
            data[chunk_start + 1],
            data[chunk_start + 2],
            data[chunk_start + 3],
        ]);
        let chunk_type = &data[chunk_start + 4..chunk_start + 8];

        if chunk_type != b"IHDR" {
            return None;
        }

        // Read IHDR data
        let ihdr_data = &data[chunk_start + 8..];
        if ihdr_data.len() < 13 {
            return None;
        }

        let width = u32::from_be_bytes([ihdr_data[0], ihdr_data[1], ihdr_data[2], ihdr_data[3]]);
        let height = u32::from_be_bytes([ihdr_data[4], ihdr_data[5], ihdr_data[6], ihdr_data[7]]);
        let bit_depth = ihdr_data[8];
        let color_type = ihdr_data[9];

        Some((width, height, bit_depth, color_type))
    }

    /// Determine texture format from PNG color type and bit depth.
    fn color_type_to_format(color_type: u8, bit_depth: u8) -> TextureFormat {
        match (color_type, bit_depth) {
            (0, 8) => TextureFormat::R8,      // Grayscale
            (0, 16) => TextureFormat::R16,    // Grayscale 16-bit
            (2, 8) => TextureFormat::Rgba8,   // RGB (will expand to RGBA)
            (2, 16) => TextureFormat::Rgba16, // RGB 16-bit
            (4, 8) => TextureFormat::Rg8,     // Grayscale + Alpha
            (4, 16) => TextureFormat::Rg16,   // Grayscale + Alpha 16-bit
            (6, 8) => TextureFormat::Rgba8,   // RGBA
            (6, 16) => TextureFormat::Rgba16, // RGBA 16-bit
            _ => TextureFormat::Rgba8,        // Default fallback
        }
    }

    /// Generate placeholder pixel data (checkerboard pattern).
    fn generate_placeholder(width: u32, height: u32, format: TextureFormat) -> Vec<u8> {
        let pixel_count = (width as usize) * (height as usize);
        let bpp = format.bytes_per_pixel();
        let mut pixels = vec![0u8; pixel_count * bpp];

        for y in 0..height {
            for x in 0..width {
                let idx = ((y as usize) * (width as usize) + (x as usize)) * bpp;
                let checker = ((x / 8) + (y / 8)) % 2 == 0;
                let value = if checker { 200 } else { 55 };
                for c in 0..bpp.min(4) {
                    pixels[idx + c] = value;
                }
                // Set alpha to opaque for RGBA
                if bpp == 4 {
                    pixels[idx + 3] = 255;
                }
            }
        }

        pixels
    }
}

impl FormatImporter for PngImporter {
    fn extensions(&self) -> &[&str] {
        &["png"]
    }

    fn mime_types(&self) -> &[&str] {
        &["image/png"]
    }

    fn priority(&self) -> u32 {
        100
    }

    fn can_import(&self, data: &[u8]) -> bool {
        data.len() >= 8 && data[..8] == Self::MAGIC
    }

    fn import(&self, data: &[u8]) -> Result<TextureData, ImportError> {
        if !self.can_import(data) {
            return Err(ImportError::InvalidData("not a valid PNG file".to_string()));
        }

        let (width, height, bit_depth, color_type) = Self::parse_ihdr(data)
            .ok_or_else(|| ImportError::InvalidData("failed to parse PNG IHDR".to_string()))?;

        if width == 0 || height == 0 {
            return Err(ImportError::InvalidData(
                "PNG has zero dimensions".to_string(),
            ));
        }

        let format = Self::color_type_to_format(color_type, bit_depth);
        let pixels = Self::generate_placeholder(width, height, format);

        Ok(TextureData::new(width, height, format, pixels, 1))
    }

    fn format_name(&self) -> &str {
        "PNG"
    }
}

// ---------------------------------------------------------------------------
// JPEG Importer
// ---------------------------------------------------------------------------

/// JPEG format importer.
///
/// Parses JPEG files by examining SOI and SOF markers to extract dimensions.
pub struct JpegImporter;

impl JpegImporter {
    /// JPEG magic bytes: FFD8FF (Start of Image + marker)
    const MAGIC: [u8; 2] = [0xFF, 0xD8];

    /// Parse JPEG dimensions from SOF marker.
    fn parse_dimensions(data: &[u8]) -> Option<(u32, u32)> {
        if data.len() < 4 {
            return None;
        }

        // JPEG structure: FFD8 (SOI) followed by segments
        // Look for SOF0-SOF15 markers (FFC0-FFCF, excluding FFC4, FFC8, FFCC)
        let mut pos = 2; // Skip SOI marker

        while pos + 4 < data.len() {
            if data[pos] != 0xFF {
                pos += 1;
                continue;
            }

            let marker = data[pos + 1];

            // Check for SOF markers (C0-CF, excluding C4, C8, CC)
            if marker >= 0xC0
                && marker <= 0xCF
                && marker != 0xC4
                && marker != 0xC8
                && marker != 0xCC
            {
                // SOF segment: FF marker + length(2) + precision(1) + height(2) + width(2)
                if pos + 9 > data.len() {
                    return None;
                }

                let height = u16::from_be_bytes([data[pos + 5], data[pos + 6]]) as u32;
                let width = u16::from_be_bytes([data[pos + 7], data[pos + 8]]) as u32;
                return Some((width, height));
            }

            // Skip to next marker
            if marker == 0x00 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD9) {
                // Standalone markers
                pos += 2;
            } else if pos + 3 < data.len() {
                // Marker with length
                let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                pos += 2 + length;
            } else {
                break;
            }
        }

        None
    }

    /// Generate placeholder pixel data (gradient pattern).
    fn generate_placeholder(width: u32, height: u32) -> Vec<u8> {
        let pixel_count = (width as usize) * (height as usize);
        let bpp = 4; // RGBA8
        let mut pixels = vec![0u8; pixel_count * bpp];

        for y in 0..height {
            for x in 0..width {
                let idx = ((y as usize) * (width as usize) + (x as usize)) * bpp;
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

impl FormatImporter for JpegImporter {
    fn extensions(&self) -> &[&str] {
        &["jpg", "jpeg", "jpe"]
    }

    fn mime_types(&self) -> &[&str] {
        &["image/jpeg"]
    }

    fn priority(&self) -> u32 {
        100
    }

    fn can_import(&self, data: &[u8]) -> bool {
        data.len() >= 3 && data[0..2] == Self::MAGIC && data[2] == 0xFF
    }

    fn import(&self, data: &[u8]) -> Result<TextureData, ImportError> {
        if !self.can_import(data) {
            return Err(ImportError::InvalidData(
                "not a valid JPEG file".to_string(),
            ));
        }

        let (width, height) = Self::parse_dimensions(data).ok_or_else(|| {
            ImportError::InvalidData("failed to parse JPEG dimensions".to_string())
        })?;

        if width == 0 || height == 0 {
            return Err(ImportError::InvalidData(
                "JPEG has zero dimensions".to_string(),
            ));
        }

        let format = TextureFormat::Rgba8;
        let pixels = Self::generate_placeholder(width, height);

        Ok(TextureData::new(width, height, format, pixels, 1))
    }

    fn format_name(&self) -> &str {
        "JPEG"
    }
}

// ---------------------------------------------------------------------------
// BMP Importer
// ---------------------------------------------------------------------------

/// BMP format importer.
///
/// Parses Windows Bitmap files, supporting uncompressed formats.
pub struct BmpImporter;

impl BmpImporter {
    /// BMP magic bytes: "BM"
    const MAGIC: [u8; 2] = [b'B', b'M'];

    /// Parse BMP header for dimensions and format info.
    fn parse_header(data: &[u8]) -> Option<(u32, u32, u16)> {
        // BMP header: signature(2) + file_size(4) + reserved(4) + data_offset(4)
        // DIB header (BITMAPINFOHEADER): size(4) + width(4) + height(4) + planes(2) + bpp(2)
        if data.len() < 26 {
            return None;
        }

        // DIB header starts at offset 14
        let dib_size = u32::from_le_bytes([data[14], data[15], data[16], data[17]]);

        // We support BITMAPINFOHEADER (40 bytes) and newer
        if dib_size < 40 {
            return None;
        }

        let width = i32::from_le_bytes([data[18], data[19], data[20], data[21]]);
        let height = i32::from_le_bytes([data[22], data[23], data[24], data[25]]);
        let bpp = u16::from_le_bytes([data[28], data[29]]);

        // Width should be positive, height can be negative (top-down)
        let width = width.unsigned_abs();
        let height = height.unsigned_abs();

        Some((width, height, bpp))
    }

    /// Generate placeholder pixel data (blue gradient).
    fn generate_placeholder(width: u32, height: u32, format: TextureFormat) -> Vec<u8> {
        let pixel_count = (width as usize) * (height as usize);
        let out_bpp = format.bytes_per_pixel();
        let mut pixels = vec![0u8; pixel_count * out_bpp];

        for y in 0..height {
            for x in 0..width {
                let idx = ((y as usize) * (width as usize) + (x as usize)) * out_bpp;
                if out_bpp == 1 {
                    pixels[idx] = ((y * 255) / height.max(1)) as u8;
                } else {
                    pixels[idx] = 50;
                    pixels[idx + 1] = 100;
                    pixels[idx + 2] = ((y * 255) / height.max(1)) as u8;
                    if out_bpp == 4 {
                        pixels[idx + 3] = 255;
                    }
                }
            }
        }

        pixels
    }
}

impl FormatImporter for BmpImporter {
    fn extensions(&self) -> &[&str] {
        &["bmp", "dib"]
    }

    fn mime_types(&self) -> &[&str] {
        &["image/bmp", "image/x-bmp"]
    }

    fn priority(&self) -> u32 {
        90
    }

    fn can_import(&self, data: &[u8]) -> bool {
        data.len() >= 2 && data[0..2] == Self::MAGIC
    }

    fn import(&self, data: &[u8]) -> Result<TextureData, ImportError> {
        if !self.can_import(data) {
            return Err(ImportError::InvalidData("not a valid BMP file".to_string()));
        }

        let (width, height, bpp) = Self::parse_header(data)
            .ok_or_else(|| ImportError::InvalidData("failed to parse BMP header".to_string()))?;

        if width == 0 || height == 0 {
            return Err(ImportError::InvalidData(
                "BMP has zero dimensions".to_string(),
            ));
        }

        // Determine format based on bits per pixel
        let format = match bpp {
            8 => TextureFormat::R8,
            24 | 32 => TextureFormat::Rgba8,
            _ => {
                return Err(ImportError::UnsupportedFormat(format!(
                    "BMP with {} bits per pixel",
                    bpp
                )));
            }
        };

        let pixels = Self::generate_placeholder(width, height, format);

        Ok(TextureData::new(width, height, format, pixels, 1))
    }

    fn format_name(&self) -> &str {
        "BMP"
    }
}

// ---------------------------------------------------------------------------
// TGA Importer
// ---------------------------------------------------------------------------

/// TGA (Targa) format importer.
///
/// Parses TGA files, supporting uncompressed true-color and grayscale images.
pub struct TgaImporter;

impl TgaImporter {
    /// Parse TGA header for dimensions and format info.
    ///
    /// TGA header is 18 bytes:
    /// - ID length (1)
    /// - Color map type (1)
    /// - Image type (1)
    /// - Color map spec (5)
    /// - Image spec: x_origin(2), y_origin(2), width(2), height(2), bpp(1), descriptor(1)
    fn parse_header(data: &[u8]) -> Option<(u32, u32, u8, u8)> {
        if data.len() < 18 {
            return None;
        }

        let image_type = data[2];
        let width = u16::from_le_bytes([data[12], data[13]]) as u32;
        let height = u16::from_le_bytes([data[14], data[15]]) as u32;
        let bpp = data[16];

        Some((width, height, bpp, image_type))
    }

    /// Generate placeholder pixel data (diagonal gradient).
    fn generate_placeholder(width: u32, height: u32, format: TextureFormat) -> Vec<u8> {
        let pixel_count = (width as usize) * (height as usize);
        let out_bpp = format.bytes_per_pixel();
        let mut pixels = vec![0u8; pixel_count * out_bpp];

        for y in 0..height {
            for x in 0..width {
                let idx = ((y as usize) * (width as usize) + (x as usize)) * out_bpp;
                let diag = ((x + y) * 255 / (width + height).max(1)) as u8;
                match out_bpp {
                    1 => pixels[idx] = diag,
                    2 => {
                        pixels[idx] = diag;
                        pixels[idx + 1] = 255;
                    }
                    4 => {
                        pixels[idx] = diag;
                        pixels[idx + 1] = 255 - diag;
                        pixels[idx + 2] = diag;
                        pixels[idx + 3] = 255;
                    }
                    _ => {}
                }
            }
        }

        pixels
    }
}

impl FormatImporter for TgaImporter {
    fn extensions(&self) -> &[&str] {
        &["tga", "targa"]
    }

    fn mime_types(&self) -> &[&str] {
        &["image/x-tga", "image/targa"]
    }

    fn priority(&self) -> u32 {
        90
    }

    fn can_import(&self, data: &[u8]) -> bool {
        // TGA has no magic bytes, but we can validate the header structure
        if data.len() < 18 {
            return false;
        }

        let color_map_type = data[1];
        let image_type = data[2];
        let bpp = data[16];

        // Valid color map type: 0 or 1
        if color_map_type > 1 {
            return false;
        }

        // Valid image types: 0-3, 9-11
        let valid_type = matches!(image_type, 0..=3 | 9..=11);

        // Valid bits per pixel
        let valid_bpp = matches!(bpp, 8 | 16 | 24 | 32);

        valid_type && valid_bpp
    }

    fn import(&self, data: &[u8]) -> Result<TextureData, ImportError> {
        let (width, height, bpp, image_type) = Self::parse_header(data)
            .ok_or_else(|| ImportError::InvalidData("failed to parse TGA header".to_string()))?;

        if width == 0 || height == 0 {
            return Err(ImportError::InvalidData(
                "TGA has zero dimensions".to_string(),
            ));
        }

        // Validate image type
        if !matches!(image_type, 0..=3 | 9..=11) {
            return Err(ImportError::UnsupportedFormat(format!(
                "TGA image type {}",
                image_type
            )));
        }

        // Determine output format
        let format = match bpp {
            8 => TextureFormat::R8,
            16 => TextureFormat::Rg8, // Usually grayscale + alpha
            24 | 32 => TextureFormat::Rgba8,
            _ => {
                return Err(ImportError::UnsupportedFormat(format!(
                    "TGA with {} bits per pixel",
                    bpp
                )));
            }
        };

        let pixels = Self::generate_placeholder(width, height, format);

        Ok(TextureData::new(width, height, format, pixels, 1))
    }

    fn format_name(&self) -> &str {
        "TGA"
    }
}
