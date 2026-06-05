//! HDR and Advanced Format Support
//!
//! Extends the texture importer with high dynamic range format decoding:
//! - OpenEXR (.exr) - 16-bit and 32-bit float, lossless/lossy compression
//! - Radiance RGBE (.hdr) - HDR environment maps
//! - TIFF (.tiff, .tif) - Uncompressed multi-page support
//! - PSD (.psd) - Photoshop flattened composite
//!
//! # Tone Mapping
//!
//! Provides preview tone mapping operators for HDR content:
//! - Reinhard global operator
//! - ACES filmic (Academy Color Encoding System)
//! - Exposure adjustment
//!
//! # GPU Format Selection
//!
//! Maps HDR source data to appropriate GPU formats:
//! - R16G16B16A16_FLOAT for full HDR
//! - RGB10_A2_UNORM for 10-bit HDR displays
//! - R11G11B10_FLOAT for compact HDR

use std::fmt;
use std::io::Cursor;

use exr::prelude::ReadChannels;
use image::{DynamicImage, GenericImageView, ImageFormat, ImageReader};

use super::texture_importer::{
    GpuTextureFormat, MemoryBudgetTracker, TextureAsset, TextureImportError, TextureMetadata,
    TextureState,
};

// ---------------------------------------------------------------------------
// Extended GPU Formats for HDR
// ---------------------------------------------------------------------------

/// Extended GPU texture formats including HDR-specific formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HdrGpuFormat {
    /// Standard formats from base importer
    Standard(GpuTextureFormat),
    /// RGB10_A2 unsigned normalized (10-bit per channel + 2-bit alpha)
    Rgb10A2Unorm,
    /// R11G11B10 unsigned float (compact HDR)
    R11G11B10UFloat,
    /// R32G32B32A32 float (full precision HDR)
    R32G32B32A32Float,
}

impl HdrGpuFormat {
    /// Returns bytes per pixel for this format.
    #[inline]
    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            HdrGpuFormat::Standard(fmt) => fmt.bytes_per_pixel(),
            HdrGpuFormat::Rgb10A2Unorm => 4,
            HdrGpuFormat::R11G11B10UFloat => 4,
            HdrGpuFormat::R32G32B32A32Float => 16,
        }
    }

    /// Returns true if this format uses floating point.
    #[inline]
    pub const fn is_float(&self) -> bool {
        match self {
            HdrGpuFormat::Standard(fmt) => fmt.is_float(),
            HdrGpuFormat::Rgb10A2Unorm => false,
            HdrGpuFormat::R11G11B10UFloat => true,
            HdrGpuFormat::R32G32B32A32Float => true,
        }
    }

    /// Returns true if this format is HDR-capable.
    #[inline]
    pub const fn is_hdr(&self) -> bool {
        matches!(
            self,
            HdrGpuFormat::Standard(GpuTextureFormat::R16G16B16A16Float)
                | HdrGpuFormat::R11G11B10UFloat
                | HdrGpuFormat::R32G32B32A32Float
        )
    }

    /// Map to wgpu TextureFormat string representation.
    pub fn to_wgpu_format_str(&self) -> &'static str {
        match self {
            HdrGpuFormat::Standard(fmt) => fmt.to_wgpu_format_str(),
            HdrGpuFormat::Rgb10A2Unorm => "Rgb10a2Unorm",
            HdrGpuFormat::R11G11B10UFloat => "Rg11b10Ufloat",
            HdrGpuFormat::R32G32B32A32Float => "Rgba32Float",
        }
    }

    /// Convert to base GpuTextureFormat if applicable.
    pub fn to_base_format(&self) -> Option<GpuTextureFormat> {
        match self {
            HdrGpuFormat::Standard(fmt) => Some(*fmt),
            _ => None,
        }
    }
}

impl fmt::Display for HdrGpuFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HdrGpuFormat::Standard(fmt) => write!(f, "{}", fmt),
            HdrGpuFormat::Rgb10A2Unorm => write!(f, "RGB10_A2_UNORM"),
            HdrGpuFormat::R11G11B10UFloat => write!(f, "R11G11B10_UFLOAT"),
            HdrGpuFormat::R32G32B32A32Float => write!(f, "R32G32B32A32_FLOAT"),
        }
    }
}

impl From<GpuTextureFormat> for HdrGpuFormat {
    fn from(fmt: GpuTextureFormat) -> Self {
        HdrGpuFormat::Standard(fmt)
    }
}

// ---------------------------------------------------------------------------
// HDR Source Format Detection
// ---------------------------------------------------------------------------

/// HDR source image format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrSourceFormat {
    /// OpenEXR (.exr)
    Exr,
    /// Radiance RGBE (.hdr)
    RadianceHdr,
    /// TIFF (.tiff, .tif)
    Tiff,
    /// Photoshop PSD (.psd)
    Psd,
}

impl HdrSourceFormat {
    /// Detect format from magic bytes.
    pub fn from_magic(data: &[u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }

        // OpenEXR: magic number 0x76 0x2F 0x31 0x01
        if data.len() >= 4 && data[0] == 0x76 && data[1] == 0x2F && data[2] == 0x31 && data[3] == 0x01
        {
            return Some(HdrSourceFormat::Exr);
        }

        // Radiance HDR: starts with "#?" (typically "#?RADIANCE")
        if data.len() >= 2 && data[0] == b'#' && data[1] == b'?' {
            return Some(HdrSourceFormat::RadianceHdr);
        }

        // TIFF: II (little-endian) or MM (big-endian) followed by 42
        if data.len() >= 4 {
            if (data[0] == b'I' && data[1] == b'I' && data[2] == 42 && data[3] == 0)
                || (data[0] == b'M' && data[1] == b'M' && data[2] == 0 && data[3] == 42)
            {
                return Some(HdrSourceFormat::Tiff);
            }
        }

        // PSD: "8BPS" magic
        if data.len() >= 4 && &data[0..4] == b"8BPS" {
            return Some(HdrSourceFormat::Psd);
        }

        None
    }

    /// Get format from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "exr" => Some(HdrSourceFormat::Exr),
            "hdr" | "rgbe" | "pic" => Some(HdrSourceFormat::RadianceHdr),
            "tiff" | "tif" => Some(HdrSourceFormat::Tiff),
            "psd" | "psb" => Some(HdrSourceFormat::Psd),
            _ => None,
        }
    }

    /// Returns true if this format is HDR.
    pub const fn is_hdr(&self) -> bool {
        matches!(self, HdrSourceFormat::Exr | HdrSourceFormat::RadianceHdr)
    }

    /// Convert to image crate format if supported.
    pub fn to_image_format(&self) -> Option<ImageFormat> {
        match self {
            HdrSourceFormat::RadianceHdr => Some(ImageFormat::Hdr),
            HdrSourceFormat::Tiff => Some(ImageFormat::Tiff),
            // EXR and PSD need special handling
            HdrSourceFormat::Exr | HdrSourceFormat::Psd => None,
        }
    }
}

impl fmt::Display for HdrSourceFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HdrSourceFormat::Exr => write!(f, "OpenEXR"),
            HdrSourceFormat::RadianceHdr => write!(f, "Radiance HDR"),
            HdrSourceFormat::Tiff => write!(f, "TIFF"),
            HdrSourceFormat::Psd => write!(f, "Photoshop PSD"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tone Mapping
// ---------------------------------------------------------------------------

/// Tone mapping operator for HDR preview.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToneMapOperator {
    /// No tone mapping (raw HDR values)
    None,
    /// Reinhard global operator
    Reinhard,
    /// ACES filmic curve (Academy Color Encoding System)
    AcesFilmic,
    /// Simple exposure adjustment (multiplier)
    Exposure(f32),
    /// Reinhard with white point
    ReinhardExtended { white_point: f32 },
    /// Uncharted 2 filmic (John Hable)
    Uncharted2,
}

impl Default for ToneMapOperator {
    fn default() -> Self {
        ToneMapOperator::AcesFilmic
    }
}

impl ToneMapOperator {
    /// Apply tone mapping to a single RGB value.
    #[inline]
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        match self {
            ToneMapOperator::None => (r, g, b),
            ToneMapOperator::Reinhard => {
                let r = r / (1.0 + r);
                let g = g / (1.0 + g);
                let b = b / (1.0 + b);
                (r, g, b)
            }
            ToneMapOperator::AcesFilmic => {
                // ACES filmic curve approximation
                let aces_a = 2.51;
                let aces_b = 0.03;
                let aces_c = 2.43;
                let aces_d = 0.59;
                let aces_e = 0.14;

                let r = ((r * (aces_a * r + aces_b)) / (r * (aces_c * r + aces_d) + aces_e))
                    .clamp(0.0, 1.0);
                let g = ((g * (aces_a * g + aces_b)) / (g * (aces_c * g + aces_d) + aces_e))
                    .clamp(0.0, 1.0);
                let b = ((b * (aces_a * b + aces_b)) / (b * (aces_c * b + aces_d) + aces_e))
                    .clamp(0.0, 1.0);
                (r, g, b)
            }
            ToneMapOperator::Exposure(exp) => {
                let r = (r * exp).clamp(0.0, 1.0);
                let g = (g * exp).clamp(0.0, 1.0);
                let b = (b * exp).clamp(0.0, 1.0);
                (r, g, b)
            }
            ToneMapOperator::ReinhardExtended { white_point } => {
                let wp2 = white_point * white_point;
                let r = (r * (1.0 + r / wp2)) / (1.0 + r);
                let g = (g * (1.0 + g / wp2)) / (1.0 + g);
                let b = (b * (1.0 + b / wp2)) / (1.0 + b);
                (r, g, b)
            }
            ToneMapOperator::Uncharted2 => {
                // Uncharted 2 filmic curve
                fn uc2_tonemap(x: f32) -> f32 {
                    const A: f32 = 0.15;
                    const B: f32 = 0.50;
                    const C: f32 = 0.10;
                    const D: f32 = 0.20;
                    const E: f32 = 0.02;
                    const F: f32 = 0.30;
                    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
                }

                const W: f32 = 11.2;
                let white_scale = 1.0 / uc2_tonemap(W);

                let r = uc2_tonemap(r) * white_scale;
                let g = uc2_tonemap(g) * white_scale;
                let b = uc2_tonemap(b) * white_scale;
                (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
            }
        }
    }

    /// Apply gamma correction after tone mapping.
    #[inline]
    pub fn apply_gamma(value: f32, gamma: f32) -> f32 {
        value.powf(1.0 / gamma)
    }
}

// ---------------------------------------------------------------------------
// HDR Metadata
// ---------------------------------------------------------------------------

/// Extended metadata for HDR textures.
#[derive(Debug, Clone)]
pub struct HdrTextureMetadata {
    /// Base metadata
    pub base: TextureMetadata,
    /// HDR-specific source format
    pub hdr_source_format: HdrSourceFormat,
    /// Whether source has alpha channel
    pub has_alpha: bool,
    /// Source bit depth per channel (16 or 32 for float)
    pub float_bit_depth: Option<u8>,
    /// Dynamic range info (min/max luminance)
    pub dynamic_range: Option<DynamicRange>,
    /// Color primaries (if available)
    pub color_primaries: Option<ColorPrimaries>,
    /// EXR compression mode (if EXR)
    pub exr_compression: Option<ExrCompression>,
}

/// Dynamic range information.
#[derive(Debug, Clone, Copy)]
pub struct DynamicRange {
    /// Minimum luminance value
    pub min_luminance: f32,
    /// Maximum luminance value
    pub max_luminance: f32,
    /// Average luminance value
    pub avg_luminance: f32,
}

impl DynamicRange {
    /// Calculate the contrast ratio.
    #[inline]
    pub fn contrast_ratio(&self) -> f32 {
        if self.min_luminance > 0.0 {
            self.max_luminance / self.min_luminance
        } else {
            f32::INFINITY
        }
    }

    /// Calculate the number of stops of dynamic range.
    #[inline]
    pub fn stops(&self) -> f32 {
        if self.max_luminance > 0.0 && self.min_luminance > 0.0 {
            (self.max_luminance / self.min_luminance).log2()
        } else {
            0.0
        }
    }

    /// Calculate from HDR pixel data.
    pub fn from_f32_pixels(pixels: &[f32], channels: usize) -> Self {
        let mut min_lum = f32::MAX;
        let mut max_lum = f32::MIN;
        let mut sum_lum = 0.0;
        let mut count = 0usize;

        for chunk in pixels.chunks(channels) {
            if chunk.len() >= 3 {
                // Luminance from Rec. 709 coefficients
                let lum = 0.2126 * chunk[0] + 0.7152 * chunk[1] + 0.0722 * chunk[2];
                if lum.is_finite() && lum >= 0.0 {
                    min_lum = min_lum.min(lum);
                    max_lum = max_lum.max(lum);
                    sum_lum += lum;
                    count += 1;
                }
            }
        }

        DynamicRange {
            min_luminance: if min_lum == f32::MAX { 0.0 } else { min_lum },
            max_luminance: if max_lum == f32::MIN { 1.0 } else { max_lum },
            avg_luminance: if count > 0 { sum_lum / count as f32 } else { 0.5 },
        }
    }
}

/// Color primaries for HDR content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorPrimaries {
    /// sRGB/Rec. 709
    Srgb,
    /// DCI-P3
    DciP3,
    /// Rec. 2020
    Rec2020,
    /// ACES AP0
    AcesAp0,
    /// ACES AP1
    AcesAp1,
    /// Unknown/custom
    Unknown,
}

/// EXR compression mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExrCompression {
    /// No compression
    None,
    /// RLE compression
    Rle,
    /// ZIP compression (lossless)
    Zip,
    /// ZIP per scanline
    ZipScanline,
    /// PIZ compression (lossless, best for noisy images)
    Piz,
    /// PXR24 (lossy, 24-bit float)
    Pxr24,
    /// B44 (lossy, fixed rate)
    B44,
    /// B44A (lossy, fixed rate with alpha)
    B44A,
    /// DWAA (lossy wavelet)
    Dwaa,
    /// DWAB (lossy wavelet)
    Dwab,
}

// ---------------------------------------------------------------------------
// HDR Texture Asset
// ---------------------------------------------------------------------------

/// An HDR texture asset with extended metadata.
#[derive(Debug, Clone)]
pub struct HdrTextureAsset {
    /// Base texture asset
    pub base: TextureAsset,
    /// HDR-specific metadata
    pub hdr_metadata: HdrTextureMetadata,
    /// Original HDR data (f32 per channel) for preview/processing
    pub hdr_data: Option<Vec<f32>>,
    /// Tone-mapped preview data (8-bit RGBA)
    pub preview_data: Option<Vec<u8>>,
    /// Tone map operator used for preview
    pub preview_tone_map: Option<ToneMapOperator>,
}

impl HdrTextureAsset {
    /// Generate a tone-mapped preview from HDR data.
    pub fn generate_preview(&mut self, operator: ToneMapOperator, gamma: f32) {
        if let Some(ref hdr_data) = self.hdr_data {
            let width = self.base.metadata.width as usize;
            let height = self.base.metadata.height as usize;
            let channels = self.hdr_metadata.hdr_source_format.channel_count();

            let mut preview = Vec::with_capacity(width * height * 4);

            for chunk in hdr_data.chunks(channels) {
                let (r, g, b) = if chunk.len() >= 3 {
                    (chunk[0], chunk[1], chunk[2])
                } else if chunk.len() >= 1 {
                    (chunk[0], chunk[0], chunk[0])
                } else {
                    (0.0, 0.0, 0.0)
                };

                let a = if chunk.len() >= 4 { chunk[3] } else { 1.0 };

                // Apply tone mapping
                let (tr, tg, tb) = operator.apply(r, g, b);

                // Apply gamma correction
                let gr = ToneMapOperator::apply_gamma(tr, gamma);
                let gg = ToneMapOperator::apply_gamma(tg, gamma);
                let gb = ToneMapOperator::apply_gamma(tb, gamma);

                // Convert to 8-bit
                preview.push((gr * 255.0).clamp(0.0, 255.0) as u8);
                preview.push((gg * 255.0).clamp(0.0, 255.0) as u8);
                preview.push((gb * 255.0).clamp(0.0, 255.0) as u8);
                preview.push((a * 255.0).clamp(0.0, 255.0) as u8);
            }

            self.preview_data = Some(preview);
            self.preview_tone_map = Some(operator);
        }
    }

    /// Get the effective GPU format for HDR content.
    pub fn effective_gpu_format(&self) -> HdrGpuFormat {
        if self.hdr_metadata.hdr_source_format.is_hdr() {
            match self.hdr_metadata.float_bit_depth {
                Some(32) => HdrGpuFormat::R32G32B32A32Float,
                Some(16) | None => {
                    HdrGpuFormat::Standard(GpuTextureFormat::R16G16B16A16Float)
                }
                Some(_) => {
                    // Fallback for other bit depths
                    HdrGpuFormat::Standard(GpuTextureFormat::R16G16B16A16Float)
                }
            }
        } else {
            HdrGpuFormat::Standard(self.base.metadata.format)
        }
    }

    /// Discard the original HDR data to save memory.
    pub fn discard_hdr_data(&mut self) {
        self.hdr_data = None;
    }

    /// Check if texture has HDR data available.
    pub fn has_hdr_data(&self) -> bool {
        self.hdr_data.is_some()
    }
}

impl HdrSourceFormat {
    /// Returns the number of channels for this format.
    pub const fn channel_count(&self) -> usize {
        match self {
            HdrSourceFormat::Exr => 4,       // Typically RGBA
            HdrSourceFormat::RadianceHdr => 3, // RGB only
            HdrSourceFormat::Tiff => 4,      // Usually RGBA
            HdrSourceFormat::Psd => 4,       // RGBA
        }
    }
}

// ---------------------------------------------------------------------------
// EXR Decoding Helper
// ---------------------------------------------------------------------------

/// Decode EXR file to RGBA f32 pixel data.
/// This is a standalone function to handle the exr crate's lifetime requirements.
fn decode_exr_to_rgba(data: &[u8]) -> Result<(u32, u32, Vec<f32>), TextureImportError> {
    use exr::prelude::*;

    // Read the EXR file with rgba_channels and all_layers
    let image = read()
        .no_deep_data()
        .largest_resolution_level()
        .rgba_channels(
            |resolution, _| vec![0.0f32; resolution.width() * resolution.height() * 4],
            |buffer, position, (r, g, b, a): (f32, f32, f32, f32)| {
                let idx = (position.y() * position.width() + position.x()) * 4;
                buffer[idx] = r;
                buffer[idx + 1] = g;
                buffer[idx + 2] = b;
                buffer[idx + 3] = a;
            },
        )
        .all_layers()
        .all_attributes()
        .from_buffered(Cursor::new(data))
        .map_err(|e| TextureImportError::DecodeFailed(format!("EXR decode error: {}", e)))?;

    // Get the first layer
    let layer = image.layer_data.first()
        .ok_or_else(|| TextureImportError::DecodeFailed("EXR has no layers".to_string()))?;

    let width = layer.size.width() as u32;
    let height = layer.size.height() as u32;
    let pixels = layer.channel_data.pixels.clone();

    Ok((width, height, pixels))
}

// ---------------------------------------------------------------------------
// HDR Importer
// ---------------------------------------------------------------------------

/// Configuration for HDR texture import.
#[derive(Debug, Clone)]
pub struct HdrImportConfig {
    /// Preferred output format for HDR textures
    pub preferred_format: HdrGpuFormat,
    /// Generate tone-mapped preview
    pub generate_preview: bool,
    /// Tone mapping operator for preview
    pub preview_tone_map: ToneMapOperator,
    /// Preview gamma value
    pub preview_gamma: f32,
    /// Keep original HDR data in memory
    pub keep_hdr_data: bool,
    /// Calculate dynamic range statistics
    pub calculate_dynamic_range: bool,
    /// Maximum texture dimension (0 = unlimited)
    pub max_dimension: u32,
}

impl Default for HdrImportConfig {
    fn default() -> Self {
        Self {
            preferred_format: HdrGpuFormat::Standard(GpuTextureFormat::R16G16B16A16Float),
            generate_preview: true,
            preview_tone_map: ToneMapOperator::AcesFilmic,
            preview_gamma: 2.2,
            keep_hdr_data: false,
            calculate_dynamic_range: true,
            max_dimension: 16384,
        }
    }
}

/// HDR texture importer supporting EXR, HDR, TIFF, and PSD formats.
pub struct HdrTextureImporter {
    /// Configuration
    config: HdrImportConfig,
    /// Next asset ID
    next_id: std::sync::atomic::AtomicU64,
}

impl HdrTextureImporter {
    /// Create a new HDR texture importer with default configuration.
    pub fn new() -> Self {
        Self {
            config: HdrImportConfig::default(),
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: HdrImportConfig) -> Self {
        Self {
            config,
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &HdrImportConfig {
        &self.config
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: HdrImportConfig) {
        self.config = config;
    }

    /// Generate a new unique texture ID.
    fn next_id(&self) -> u64 {
        self.next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Import an HDR texture from raw bytes.
    pub fn import_from_bytes(
        &self,
        data: &[u8],
        extension_hint: Option<&str>,
        budget: &MemoryBudgetTracker,
    ) -> Result<HdrTextureAsset, TextureImportError> {
        // Detect format
        let format = self.detect_format(data, extension_hint)?;

        // Dispatch to format-specific decoder
        match format {
            HdrSourceFormat::Exr => self.decode_exr(data, budget),
            HdrSourceFormat::RadianceHdr => self.decode_radiance_hdr(data, budget),
            HdrSourceFormat::Tiff => self.decode_tiff(data, budget),
            HdrSourceFormat::Psd => self.decode_psd(data, budget),
        }
    }

    /// Detect the HDR source format.
    fn detect_format(
        &self,
        data: &[u8],
        extension_hint: Option<&str>,
    ) -> Result<HdrSourceFormat, TextureImportError> {
        // Try magic bytes first
        if let Some(format) = HdrSourceFormat::from_magic(data) {
            return Ok(format);
        }

        // Fall back to extension
        if let Some(ext) = extension_hint {
            if let Some(format) = HdrSourceFormat::from_extension(ext) {
                return Ok(format);
            }
        }

        Err(TextureImportError::UnsupportedFormat(
            extension_hint.unwrap_or("unknown").to_string(),
        ))
    }

    /// Decode OpenEXR file.
    fn decode_exr(
        &self,
        data: &[u8],
        budget: &MemoryBudgetTracker,
    ) -> Result<HdrTextureAsset, TextureImportError> {
        // Decode EXR to RGBA using standalone helper function
        let (width, height, hdr_pixels) = decode_exr_to_rgba(data)?;

        // Validate dimensions
        self.validate_dimensions(width, height)?;

        // Determine bit depth from attributes (default to 16-bit half)
        let float_bit_depth = Some(16u8); // EXR typically uses half-float

        // Calculate memory size for GPU format
        let memory_size = (width as usize) * (height as usize) * 8; // R16G16B16A16Float

        // Check budget
        let texture_id = self.next_id();
        budget.allocate(texture_id, memory_size)?;

        // Calculate dynamic range
        let dynamic_range = if self.config.calculate_dynamic_range {
            Some(DynamicRange::from_f32_pixels(&hdr_pixels, 4))
        } else {
            None
        };

        // Convert f32 to f16 for GPU upload
        let gpu_data: Vec<u8> = hdr_pixels
            .iter()
            .flat_map(|&v| {
                let h = half::f16::from_f32(v);
                h.to_le_bytes()
            })
            .collect();

        // Build base metadata (use a placeholder source format)
        let base_metadata = TextureMetadata {
            width,
            height,
            format: GpuTextureFormat::R16G16B16A16Float,
            memory_size,
            is_srgb: false, // HDR is linear
            source_format: super::texture_importer::SourceFormat::Png, // placeholder
            source_bit_depth: 32,
            source_channels: 4,
        };

        let hdr_metadata = HdrTextureMetadata {
            base: base_metadata.clone(),
            hdr_source_format: HdrSourceFormat::Exr,
            has_alpha: true,
            float_bit_depth,
            dynamic_range,
            color_primaries: Some(ColorPrimaries::Unknown),
            exr_compression: Some(ExrCompression::None), // Would parse from attributes
        };

        let base_asset = TextureAsset {
            id: texture_id,
            metadata: base_metadata,
            data: gpu_data,
            state: TextureState::Pending,
        };

        let mut hdr_asset = HdrTextureAsset {
            base: base_asset,
            hdr_metadata,
            hdr_data: if self.config.keep_hdr_data {
                Some(hdr_pixels)
            } else {
                None
            },
            preview_data: None,
            preview_tone_map: None,
        };

        // Generate preview if requested
        if self.config.generate_preview && hdr_asset.hdr_data.is_some() {
            hdr_asset.generate_preview(self.config.preview_tone_map, self.config.preview_gamma);
        }

        Ok(hdr_asset)
    }

    /// Decode Radiance HDR (.hdr) file.
    fn decode_radiance_hdr(
        &self,
        data: &[u8],
        budget: &MemoryBudgetTracker,
    ) -> Result<HdrTextureAsset, TextureImportError> {
        // Use image crate's HDR decoder
        let cursor = Cursor::new(data);
        let reader = ImageReader::with_format(cursor, ImageFormat::Hdr);
        let image = reader
            .decode()
            .map_err(|e| TextureImportError::DecodeFailed(format!("HDR decode error: {}", e)))?;

        let (width, height) = image.dimensions();
        self.validate_dimensions(width, height)?;

        // Convert to RGBA f32
        let rgba32f = image.to_rgba32f();
        let hdr_pixels: Vec<f32> = rgba32f.as_raw().to_vec();

        // Calculate memory size for GPU format
        let memory_size = (width as usize) * (height as usize) * 8; // R16G16B16A16Float

        // Check budget
        let texture_id = self.next_id();
        budget.allocate(texture_id, memory_size)?;

        // Calculate dynamic range
        let dynamic_range = if self.config.calculate_dynamic_range {
            Some(DynamicRange::from_f32_pixels(&hdr_pixels, 4))
        } else {
            None
        };

        // Convert f32 to f16 for GPU upload
        let gpu_data: Vec<u8> = hdr_pixels
            .iter()
            .flat_map(|&v| {
                let h = half::f16::from_f32(v);
                h.to_le_bytes()
            })
            .collect();

        let base_metadata = TextureMetadata {
            width,
            height,
            format: GpuTextureFormat::R16G16B16A16Float,
            memory_size,
            is_srgb: false,
            source_format: super::texture_importer::SourceFormat::Png, // placeholder
            source_bit_depth: 32,
            source_channels: 3, // HDR is RGB
        };

        let hdr_metadata = HdrTextureMetadata {
            base: base_metadata.clone(),
            hdr_source_format: HdrSourceFormat::RadianceHdr,
            has_alpha: false, // Radiance HDR doesn't have alpha
            float_bit_depth: Some(32),
            dynamic_range,
            color_primaries: Some(ColorPrimaries::Srgb),
            exr_compression: None,
        };

        let base_asset = TextureAsset {
            id: texture_id,
            metadata: base_metadata,
            data: gpu_data,
            state: TextureState::Pending,
        };

        // Determine if we need to keep HDR data
        let keep_hdr = self.config.keep_hdr_data || self.config.generate_preview;

        let mut hdr_asset = HdrTextureAsset {
            base: base_asset,
            hdr_metadata,
            hdr_data: if keep_hdr { Some(hdr_pixels) } else { None },
            preview_data: None,
            preview_tone_map: None,
        };

        // Generate preview if requested
        if self.config.generate_preview && hdr_asset.hdr_data.is_some() {
            hdr_asset.generate_preview(self.config.preview_tone_map, self.config.preview_gamma);
            // Discard HDR data after preview if not needed
            if !self.config.keep_hdr_data {
                hdr_asset.hdr_data = None;
            }
        }

        Ok(hdr_asset)
    }

    /// Decode TIFF file.
    fn decode_tiff(
        &self,
        data: &[u8],
        budget: &MemoryBudgetTracker,
    ) -> Result<HdrTextureAsset, TextureImportError> {
        // Use image crate's TIFF decoder
        let cursor = Cursor::new(data);
        let reader = ImageReader::with_format(cursor, ImageFormat::Tiff);
        let image = reader
            .decode()
            .map_err(|e| TextureImportError::DecodeFailed(format!("TIFF decode error: {}", e)))?;

        let (width, height) = image.dimensions();
        self.validate_dimensions(width, height)?;

        // Analyze source image type
        let (source_bit_depth, source_channels, is_float) = match &image {
            DynamicImage::ImageLuma8(_) => (8, 1, false),
            DynamicImage::ImageLumaA8(_) => (8, 2, false),
            DynamicImage::ImageRgb8(_) => (8, 3, false),
            DynamicImage::ImageRgba8(_) => (8, 4, false),
            DynamicImage::ImageLuma16(_) => (16, 1, false),
            DynamicImage::ImageLumaA16(_) => (16, 2, false),
            DynamicImage::ImageRgb16(_) => (16, 3, false),
            DynamicImage::ImageRgba16(_) => (16, 4, false),
            DynamicImage::ImageRgb32F(_) => (32, 3, true),
            DynamicImage::ImageRgba32F(_) => (32, 4, true),
            _ => (8, 4, false),
        };

        // Choose appropriate format and convert
        let (format, gpu_data, hdr_pixels) = if is_float || source_bit_depth == 16 {
            // High bit depth: use 16-bit float
            let rgba32f = image.to_rgba32f();
            let pixels: Vec<f32> = rgba32f.as_raw().to_vec();

            let data: Vec<u8> = pixels
                .iter()
                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                .collect();

            (GpuTextureFormat::R16G16B16A16Float, data, Some(pixels))
        } else {
            // 8-bit: use standard RGBA8
            let rgba8 = image.to_rgba8();
            (GpuTextureFormat::R8G8B8A8Unorm, rgba8.into_raw(), None)
        };

        let memory_size = (width as usize) * (height as usize) * format.bytes_per_pixel();

        // Check budget
        let texture_id = self.next_id();
        budget.allocate(texture_id, memory_size)?;

        // Calculate dynamic range for HDR TIFF
        let dynamic_range = if self.config.calculate_dynamic_range && hdr_pixels.is_some() {
            Some(DynamicRange::from_f32_pixels(hdr_pixels.as_ref().unwrap(), 4))
        } else {
            None
        };

        let base_metadata = TextureMetadata {
            width,
            height,
            format,
            memory_size,
            is_srgb: !is_float && source_bit_depth == 8,
            source_format: super::texture_importer::SourceFormat::Png, // placeholder
            source_bit_depth,
            source_channels,
        };

        let hdr_metadata = HdrTextureMetadata {
            base: base_metadata.clone(),
            hdr_source_format: HdrSourceFormat::Tiff,
            has_alpha: source_channels == 4 || source_channels == 2,
            float_bit_depth: if is_float { Some(32) } else { None },
            dynamic_range,
            color_primaries: Some(ColorPrimaries::Srgb),
            exr_compression: None,
        };

        let base_asset = TextureAsset {
            id: texture_id,
            metadata: base_metadata,
            data: gpu_data,
            state: TextureState::Pending,
        };

        let mut hdr_asset = HdrTextureAsset {
            base: base_asset,
            hdr_metadata,
            hdr_data: if self.config.keep_hdr_data {
                hdr_pixels
            } else {
                None
            },
            preview_data: None,
            preview_tone_map: None,
        };

        // Generate preview for HDR TIFF
        if self.config.generate_preview && is_float && hdr_asset.hdr_data.is_some() {
            hdr_asset.generate_preview(self.config.preview_tone_map, self.config.preview_gamma);
        }

        Ok(hdr_asset)
    }

    /// Decode PSD file (flattened composite).
    fn decode_psd(
        &self,
        data: &[u8],
        budget: &MemoryBudgetTracker,
    ) -> Result<HdrTextureAsset, TextureImportError> {
        // PSD format parsing
        // PSD header structure:
        // - 4 bytes: signature "8BPS"
        // - 2 bytes: version (1 = PSD, 2 = PSB)
        // - 6 bytes: reserved
        // - 2 bytes: channels (1-56)
        // - 4 bytes: height
        // - 4 bytes: width
        // - 2 bytes: depth (1, 8, 16, 32)
        // - 2 bytes: color mode

        if data.len() < 26 {
            return Err(TextureImportError::InvalidData(
                "PSD file too small".to_string(),
            ));
        }

        // Verify signature
        if &data[0..4] != b"8BPS" {
            return Err(TextureImportError::InvalidData(
                "Invalid PSD signature".to_string(),
            ));
        }

        // Parse header
        let version = u16::from_be_bytes([data[4], data[5]]);
        if version != 1 && version != 2 {
            return Err(TextureImportError::UnsupportedFormat(format!(
                "Unsupported PSD version: {}",
                version
            )));
        }

        let channels = u16::from_be_bytes([data[12], data[13]]);
        let height = u32::from_be_bytes([data[14], data[15], data[16], data[17]]);
        let width = u32::from_be_bytes([data[18], data[19], data[20], data[21]]);
        let depth = u16::from_be_bytes([data[22], data[23]]);
        let color_mode = u16::from_be_bytes([data[24], data[25]]);

        self.validate_dimensions(width, height)?;

        // For now, we only support RGB(A) mode (3) with 8 or 16-bit depth
        if color_mode != 3 {
            return Err(TextureImportError::UnsupportedFormat(format!(
                "Unsupported PSD color mode: {} (only RGB supported)",
                color_mode
            )));
        }

        if depth != 8 && depth != 16 {
            return Err(TextureImportError::UnsupportedFormat(format!(
                "Unsupported PSD bit depth: {} (only 8/16 supported)",
                depth
            )));
        }

        // Parse color mode data section
        let mut offset = 26;
        if offset + 4 > data.len() {
            return Err(TextureImportError::InvalidData(
                "PSD truncated at color mode section".to_string(),
            ));
        }
        let color_mode_len =
            u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
                as usize;
        offset += 4 + color_mode_len;

        // Parse image resources section
        if offset + 4 > data.len() {
            return Err(TextureImportError::InvalidData(
                "PSD truncated at resources section".to_string(),
            ));
        }
        let resources_len =
            u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
                as usize;
        offset += 4 + resources_len;

        // Parse layer and mask information section
        if offset + 4 > data.len() {
            return Err(TextureImportError::InvalidData(
                "PSD truncated at layer section".to_string(),
            ));
        }
        let layer_len =
            u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
                as usize;
        offset += 4 + layer_len;

        // Parse image data section
        if offset + 2 > data.len() {
            return Err(TextureImportError::InvalidData(
                "PSD truncated at image data section".to_string(),
            ));
        }
        let compression = u16::from_be_bytes([data[offset], data[offset + 1]]);
        offset += 2;

        // For now, only support uncompressed (0) or RLE (1)
        if compression != 0 && compression != 1 {
            return Err(TextureImportError::UnsupportedFormat(format!(
                "Unsupported PSD compression: {}",
                compression
            )));
        }

        let pixel_count = (width as usize) * (height as usize);
        let bytes_per_channel = if depth == 16 { 2 } else { 1 };
        let channel_count = channels.min(4) as usize; // Limit to RGBA

        let (format, gpu_data) = if compression == 0 {
            // Uncompressed: planar format (all R, then all G, then all B, then all A)
            self.decode_psd_uncompressed(
                &data[offset..],
                width,
                height,
                channel_count,
                depth,
            )?
        } else {
            // RLE compressed
            self.decode_psd_rle(&data[offset..], width, height, channel_count, depth)?
        };

        let memory_size = (width as usize) * (height as usize) * format.bytes_per_pixel();

        // Check budget
        let texture_id = self.next_id();
        budget.allocate(texture_id, memory_size)?;

        let base_metadata = TextureMetadata {
            width,
            height,
            format,
            memory_size,
            is_srgb: depth == 8,
            source_format: super::texture_importer::SourceFormat::Png, // placeholder
            source_bit_depth: depth as u8,
            source_channels: channel_count as u8,
        };

        let hdr_metadata = HdrTextureMetadata {
            base: base_metadata.clone(),
            hdr_source_format: HdrSourceFormat::Psd,
            has_alpha: channel_count >= 4,
            float_bit_depth: None,
            dynamic_range: None,
            color_primaries: Some(ColorPrimaries::Srgb),
            exr_compression: None,
        };

        let base_asset = TextureAsset {
            id: texture_id,
            metadata: base_metadata,
            data: gpu_data,
            state: TextureState::Pending,
        };

        let hdr_asset = HdrTextureAsset {
            base: base_asset,
            hdr_metadata,
            hdr_data: None,
            preview_data: None,
            preview_tone_map: None,
        };

        Ok(hdr_asset)
    }

    /// Decode uncompressed PSD image data (planar format).
    fn decode_psd_uncompressed(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        depth: u16,
    ) -> Result<(GpuTextureFormat, Vec<u8>), TextureImportError> {
        let pixel_count = (width as usize) * (height as usize);
        let bytes_per_channel = if depth == 16 { 2 } else { 1 };
        let expected_size = pixel_count * channels * bytes_per_channel;

        if data.len() < expected_size {
            return Err(TextureImportError::InvalidData(format!(
                "PSD image data too small: expected {}, got {}",
                expected_size,
                data.len()
            )));
        }

        if depth == 8 {
            // 8-bit: interleave to RGBA8
            let mut output = vec![0u8; pixel_count * 4];

            for c in 0..channels.min(4) {
                let channel_offset = c * pixel_count;
                for i in 0..pixel_count {
                    output[i * 4 + c] = data[channel_offset + i];
                }
            }

            // Fill alpha if missing
            if channels < 4 {
                for i in 0..pixel_count {
                    output[i * 4 + 3] = 255;
                }
            }

            Ok((GpuTextureFormat::R8G8B8A8Unorm, output))
        } else {
            // 16-bit: interleave to RGBA16
            let mut output = vec![0u8; pixel_count * 8];

            for c in 0..channels.min(4) {
                let channel_offset = c * pixel_count * 2;
                for i in 0..pixel_count {
                    // PSD stores big-endian, convert to little-endian for GPU
                    let hi = data[channel_offset + i * 2];
                    let lo = data[channel_offset + i * 2 + 1];
                    let out_offset = i * 8 + c * 2;
                    output[out_offset] = lo;
                    output[out_offset + 1] = hi;
                }
            }

            // Fill alpha if missing
            if channels < 4 {
                for i in 0..pixel_count {
                    let out_offset = i * 8 + 6;
                    output[out_offset] = 0xFF;
                    output[out_offset + 1] = 0xFF;
                }
            }

            Ok((GpuTextureFormat::R16G16B16A16Unorm, output))
        }
    }

    /// Decode RLE-compressed PSD image data.
    fn decode_psd_rle(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        depth: u16,
    ) -> Result<(GpuTextureFormat, Vec<u8>), TextureImportError> {
        let pixel_count = (width as usize) * (height as usize);
        let scanlines = (height as usize) * channels;

        // First, read the byte counts for each scanline
        if data.len() < scanlines * 2 {
            return Err(TextureImportError::InvalidData(
                "PSD RLE byte counts truncated".to_string(),
            ));
        }

        let mut byte_counts = Vec::with_capacity(scanlines);
        for i in 0..scanlines {
            let count = u16::from_be_bytes([data[i * 2], data[i * 2 + 1]]) as usize;
            byte_counts.push(count);
        }

        let mut offset = scanlines * 2;

        // Decode each channel
        let bytes_per_channel = if depth == 16 { 2 } else { 1 };
        let scanline_bytes = (width as usize) * bytes_per_channel;
        let mut channel_data = vec![vec![0u8; pixel_count * bytes_per_channel]; channels];

        for c in 0..channels {
            let mut channel_offset = 0;
            for row in 0..(height as usize) {
                let scanline_idx = c * (height as usize) + row;
                let compressed_len = byte_counts[scanline_idx];

                if offset + compressed_len > data.len() {
                    return Err(TextureImportError::InvalidData(
                        "PSD RLE data truncated".to_string(),
                    ));
                }

                // Decode PackBits RLE
                let mut src = offset;
                let mut dst = channel_offset;
                let end = offset + compressed_len;

                while src < end && dst < channel_offset + scanline_bytes {
                    let header = data[src] as i8;
                    src += 1;

                    if header >= 0 {
                        // Literal run: copy n+1 bytes
                        let count = (header as usize) + 1;
                        let copy_count = count.min(channel_offset + scanline_bytes - dst);
                        if src + copy_count > data.len() {
                            break;
                        }
                        channel_data[c][dst..dst + copy_count]
                            .copy_from_slice(&data[src..src + copy_count]);
                        src += count;
                        dst += copy_count;
                    } else if header != -128 {
                        // Repeat run: repeat next byte -n+1 times
                        let count = (-(header as i16) + 1) as usize;
                        if src >= data.len() {
                            break;
                        }
                        let value = data[src];
                        src += 1;
                        let fill_count = count.min(channel_offset + scanline_bytes - dst);
                        for i in 0..fill_count {
                            channel_data[c][dst + i] = value;
                        }
                        dst += fill_count;
                    }
                    // header == -128: no-op
                }

                offset += compressed_len;
                channel_offset += scanline_bytes;
            }
        }

        // Interleave channels
        if depth == 8 {
            let mut output = vec![0u8; pixel_count * 4];

            for c in 0..channels.min(4) {
                for i in 0..pixel_count {
                    output[i * 4 + c] = channel_data[c][i];
                }
            }

            // Fill alpha if missing
            if channels < 4 {
                for i in 0..pixel_count {
                    output[i * 4 + 3] = 255;
                }
            }

            Ok((GpuTextureFormat::R8G8B8A8Unorm, output))
        } else {
            let mut output = vec![0u8; pixel_count * 8];

            for c in 0..channels.min(4) {
                for i in 0..pixel_count {
                    // Convert big-endian to little-endian
                    let hi = channel_data[c][i * 2];
                    let lo = channel_data[c][i * 2 + 1];
                    let out_offset = i * 8 + c * 2;
                    output[out_offset] = lo;
                    output[out_offset + 1] = hi;
                }
            }

            // Fill alpha if missing
            if channels < 4 {
                for i in 0..pixel_count {
                    let out_offset = i * 8 + 6;
                    output[out_offset] = 0xFF;
                    output[out_offset + 1] = 0xFF;
                }
            }

            Ok((GpuTextureFormat::R16G16B16A16Unorm, output))
        }
    }

    /// Validate texture dimensions.
    fn validate_dimensions(&self, width: u32, height: u32) -> Result<(), TextureImportError> {
        if width == 0 || height == 0 {
            return Err(TextureImportError::InvalidDimensions { width, height });
        }

        if self.config.max_dimension > 0
            && (width > self.config.max_dimension || height > self.config.max_dimension)
        {
            return Err(TextureImportError::InvalidDimensions { width, height });
        }

        Ok(())
    }

    /// Supported file extensions.
    pub fn supported_extensions() -> &'static [&'static str] {
        &["exr", "hdr", "rgbe", "pic", "tiff", "tif", "psd", "psb"]
    }

    /// Check if an extension is supported.
    pub fn is_extension_supported(ext: &str) -> bool {
        HdrSourceFormat::from_extension(ext).is_some()
    }
}

impl Default for HdrTextureImporter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GPU Format Selection
// ---------------------------------------------------------------------------

/// Select the optimal GPU format for HDR content.
pub fn select_hdr_gpu_format(
    source_format: HdrSourceFormat,
    bit_depth: u8,
    prefer_compact: bool,
) -> HdrGpuFormat {
    match source_format {
        HdrSourceFormat::Exr | HdrSourceFormat::RadianceHdr => {
            if prefer_compact {
                HdrGpuFormat::R11G11B10UFloat
            } else if bit_depth >= 32 {
                HdrGpuFormat::R32G32B32A32Float
            } else {
                HdrGpuFormat::Standard(GpuTextureFormat::R16G16B16A16Float)
            }
        }
        HdrSourceFormat::Tiff => {
            if bit_depth >= 16 {
                HdrGpuFormat::Standard(GpuTextureFormat::R16G16B16A16Float)
            } else {
                HdrGpuFormat::Standard(GpuTextureFormat::R8G8B8A8Unorm)
            }
        }
        HdrSourceFormat::Psd => {
            if bit_depth >= 16 {
                HdrGpuFormat::Standard(GpuTextureFormat::R16G16B16A16Unorm)
            } else {
                HdrGpuFormat::Standard(GpuTextureFormat::R8G8B8A8Unorm)
            }
        }
    }
}

/// Convert HDR data to RGB10_A2 format (10-bit per channel).
pub fn convert_to_rgb10a2(hdr_pixels: &[f32], tone_map: ToneMapOperator) -> Vec<u8> {
    let pixel_count = hdr_pixels.len() / 4;
    let mut output = Vec::with_capacity(pixel_count * 4);

    for chunk in hdr_pixels.chunks(4) {
        if chunk.len() < 4 {
            break;
        }

        // Apply tone mapping
        let (r, g, b) = tone_map.apply(chunk[0], chunk[1], chunk[2]);
        let a = chunk[3].clamp(0.0, 1.0);

        // Convert to 10-bit (0-1023) and 2-bit (0-3)
        let r10 = (r * 1023.0).clamp(0.0, 1023.0) as u32;
        let g10 = (g * 1023.0).clamp(0.0, 1023.0) as u32;
        let b10 = (b * 1023.0).clamp(0.0, 1023.0) as u32;
        let a2 = (a * 3.0).clamp(0.0, 3.0) as u32;

        // Pack into 32-bit: RRRRRRRRRR GGGGGGGGGG BBBBBBBBBB AA
        let packed = (r10) | (g10 << 10) | (b10 << 20) | (a2 << 30);
        output.extend_from_slice(&packed.to_le_bytes());
    }

    output
}

/// Convert HDR data to R11G11B10 format (compact HDR).
pub fn convert_to_r11g11b10(hdr_pixels: &[f32]) -> Vec<u8> {
    let pixel_count = hdr_pixels.len() / 4;
    let mut output = Vec::with_capacity(pixel_count * 4);

    for chunk in hdr_pixels.chunks(4) {
        if chunk.len() < 3 {
            break;
        }

        // Convert to unsigned float representation
        // R11G11B10 uses 5-bit exponent and 6/5 bits mantissa
        let r = float_to_r11(chunk[0]);
        let g = float_to_g11(chunk[1]);
        let b = float_to_b10(chunk[2]);

        let packed = r | (g << 11) | (b << 22);
        output.extend_from_slice(&packed.to_le_bytes());
    }

    output
}

/// Convert f32 to 11-bit unsigned float (R11).
fn float_to_r11(value: f32) -> u32 {
    if value <= 0.0 {
        return 0;
    }
    if value.is_nan() {
        return 0x7C0; // NaN
    }
    if value.is_infinite() {
        return 0x7C0; // Inf
    }

    let bits = value.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    // R11: 5-bit exponent (bias 15), 6-bit mantissa
    if exp <= 112 {
        // Denormalized
        let shift = 113 - exp;
        if shift >= 7 {
            0
        } else {
            ((mantissa | 0x800000) >> (17 + shift)) as u32
        }
    } else if exp >= 143 {
        // Overflow to infinity
        0x7C0
    } else {
        let e = ((exp - 112) as u32) << 6;
        let m = (mantissa >> 17) as u32;
        e | m
    }
}

/// Convert f32 to 11-bit unsigned float (G11).
fn float_to_g11(value: f32) -> u32 {
    float_to_r11(value) // Same format as R11
}

/// Convert f32 to 10-bit unsigned float (B10).
fn float_to_b10(value: f32) -> u32 {
    if value <= 0.0 {
        return 0;
    }
    if value.is_nan() {
        return 0x3E0; // NaN
    }
    if value.is_infinite() {
        return 0x3E0; // Inf
    }

    let bits = value.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    // B10: 5-bit exponent (bias 15), 5-bit mantissa
    if exp <= 112 {
        // Denormalized
        let shift = 113 - exp;
        if shift >= 6 {
            0
        } else {
            ((mantissa | 0x800000) >> (18 + shift)) as u32
        }
    } else if exp >= 143 {
        // Overflow to infinity
        0x3E0
    } else {
        let e = ((exp - 112) as u32) << 5;
        let m = (mantissa >> 18) as u32;
        e | m
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // HDR Format Detection Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_hdr_format_detection_exr_magic() {
        let exr_magic = [0x76, 0x2F, 0x31, 0x01, 0, 0, 0, 0];
        assert_eq!(
            HdrSourceFormat::from_magic(&exr_magic),
            Some(HdrSourceFormat::Exr)
        );
    }

    #[test]
    fn test_hdr_format_detection_radiance_magic() {
        let hdr_magic = b"#?RADIANCE\n";
        assert_eq!(
            HdrSourceFormat::from_magic(hdr_magic),
            Some(HdrSourceFormat::RadianceHdr)
        );
    }

    #[test]
    fn test_hdr_format_detection_tiff_le_magic() {
        let tiff_le = [b'I', b'I', 42, 0, 0, 0, 0, 0];
        assert_eq!(
            HdrSourceFormat::from_magic(&tiff_le),
            Some(HdrSourceFormat::Tiff)
        );
    }

    #[test]
    fn test_hdr_format_detection_tiff_be_magic() {
        let tiff_be = [b'M', b'M', 0, 42, 0, 0, 0, 0];
        assert_eq!(
            HdrSourceFormat::from_magic(&tiff_be),
            Some(HdrSourceFormat::Tiff)
        );
    }

    #[test]
    fn test_hdr_format_detection_psd_magic() {
        let psd_magic = b"8BPS\x00\x01";
        assert_eq!(
            HdrSourceFormat::from_magic(psd_magic),
            Some(HdrSourceFormat::Psd)
        );
    }

    #[test]
    fn test_hdr_format_detection_from_extension() {
        assert_eq!(
            HdrSourceFormat::from_extension("exr"),
            Some(HdrSourceFormat::Exr)
        );
        assert_eq!(
            HdrSourceFormat::from_extension("EXR"),
            Some(HdrSourceFormat::Exr)
        );
        assert_eq!(
            HdrSourceFormat::from_extension("hdr"),
            Some(HdrSourceFormat::RadianceHdr)
        );
        assert_eq!(
            HdrSourceFormat::from_extension("rgbe"),
            Some(HdrSourceFormat::RadianceHdr)
        );
        assert_eq!(
            HdrSourceFormat::from_extension("tiff"),
            Some(HdrSourceFormat::Tiff)
        );
        assert_eq!(
            HdrSourceFormat::from_extension("tif"),
            Some(HdrSourceFormat::Tiff)
        );
        assert_eq!(
            HdrSourceFormat::from_extension("psd"),
            Some(HdrSourceFormat::Psd)
        );
        assert_eq!(
            HdrSourceFormat::from_extension("psb"),
            Some(HdrSourceFormat::Psd)
        );
        assert_eq!(HdrSourceFormat::from_extension("png"), None);
    }

    // ---------------------------------------------------------------------------
    // Tone Mapping Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_tone_map_none() {
        let op = ToneMapOperator::None;
        let (r, g, b) = op.apply(2.0, 1.0, 0.5);
        assert!((r - 2.0).abs() < 0.001);
        assert!((g - 1.0).abs() < 0.001);
        assert!((b - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_tone_map_reinhard() {
        let op = ToneMapOperator::Reinhard;

        // Input 1.0 should map to 0.5
        let (r, _g, _b) = op.apply(1.0, 1.0, 1.0);
        assert!((r - 0.5).abs() < 0.001);

        // Input 0.0 should stay 0.0
        let (r, _, _) = op.apply(0.0, 0.0, 0.0);
        assert!(r.abs() < 0.001);

        // High values should be compressed
        let (r, _, _) = op.apply(100.0, 1.0, 1.0);
        assert!(r > 0.99 && r < 1.0);
    }

    #[test]
    fn test_tone_map_aces_filmic() {
        let op = ToneMapOperator::AcesFilmic;

        // Should produce values in 0-1 range
        let (r, g, b) = op.apply(2.0, 1.0, 0.5);
        assert!(r >= 0.0 && r <= 1.0);
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);

        // Higher input should produce higher output (monotonic)
        let (r1, _, _) = op.apply(1.0, 0.0, 0.0);
        let (r2, _, _) = op.apply(2.0, 0.0, 0.0);
        assert!(r2 > r1);
    }

    #[test]
    fn test_tone_map_exposure() {
        let op = ToneMapOperator::Exposure(2.0);

        let (r, g, b) = op.apply(0.25, 0.5, 0.75);
        assert!((r - 0.5).abs() < 0.001);
        assert!((g - 1.0).abs() < 0.001); // Clamped to 1.0
        assert!((b - 1.0).abs() < 0.001); // Clamped to 1.0
    }

    #[test]
    fn test_tone_map_reinhard_extended() {
        let op = ToneMapOperator::ReinhardExtended { white_point: 4.0 };

        // At white point, output should be close to 1.0
        let (r, _, _) = op.apply(4.0, 0.0, 0.0);
        assert!(r > 0.9);

        // Below white point
        let (r, _, _) = op.apply(1.0, 0.0, 0.0);
        assert!(r > 0.4 && r < 0.7);
    }

    #[test]
    fn test_tone_map_uncharted2() {
        let op = ToneMapOperator::Uncharted2;

        // Should produce values in 0-1 range
        let (r, g, b) = op.apply(5.0, 2.0, 1.0);
        assert!(r >= 0.0 && r <= 1.0);
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn test_gamma_correction() {
        // Gamma 2.2
        let result = ToneMapOperator::apply_gamma(0.5, 2.2);
        assert!(result > 0.7 && result < 0.8);

        // Gamma 1.0 (linear) should not change
        let result = ToneMapOperator::apply_gamma(0.5, 1.0);
        assert!((result - 0.5).abs() < 0.001);
    }

    // ---------------------------------------------------------------------------
    // Dynamic Range Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_dynamic_range_calculation() {
        let pixels = vec![
            0.01, 0.01, 0.01, 1.0, // Dark pixel
            1.0, 1.0, 1.0, 1.0,   // Mid pixel
            10.0, 10.0, 10.0, 1.0, // Bright pixel
        ];

        let dr = DynamicRange::from_f32_pixels(&pixels, 4);

        assert!(dr.min_luminance > 0.0 && dr.min_luminance < 0.02);
        assert!(dr.max_luminance > 9.0 && dr.max_luminance < 11.0);
        assert!(dr.stops() > 9.0); // log2(10/0.01) ~ 10 stops
    }

    #[test]
    fn test_dynamic_range_contrast_ratio() {
        let dr = DynamicRange {
            min_luminance: 0.001,
            max_luminance: 1000.0,
            avg_luminance: 1.0,
        };

        assert!((dr.contrast_ratio() - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_dynamic_range_stops() {
        let dr = DynamicRange {
            min_luminance: 1.0,
            max_luminance: 1024.0,
            avg_luminance: 10.0,
        };

        assert!((dr.stops() - 10.0).abs() < 0.01); // log2(1024) = 10
    }

    // ---------------------------------------------------------------------------
    // HDR GPU Format Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_hdr_gpu_format_bytes_per_pixel() {
        assert_eq!(HdrGpuFormat::Rgb10A2Unorm.bytes_per_pixel(), 4);
        assert_eq!(HdrGpuFormat::R11G11B10UFloat.bytes_per_pixel(), 4);
        assert_eq!(HdrGpuFormat::R32G32B32A32Float.bytes_per_pixel(), 16);
        assert_eq!(
            HdrGpuFormat::Standard(GpuTextureFormat::R16G16B16A16Float).bytes_per_pixel(),
            8
        );
    }

    #[test]
    fn test_hdr_gpu_format_is_float() {
        assert!(HdrGpuFormat::R11G11B10UFloat.is_float());
        assert!(HdrGpuFormat::R32G32B32A32Float.is_float());
        assert!(!HdrGpuFormat::Rgb10A2Unorm.is_float());
    }

    #[test]
    fn test_hdr_gpu_format_is_hdr() {
        assert!(HdrGpuFormat::R11G11B10UFloat.is_hdr());
        assert!(HdrGpuFormat::R32G32B32A32Float.is_hdr());
        assert!(HdrGpuFormat::Standard(GpuTextureFormat::R16G16B16A16Float).is_hdr());
        assert!(!HdrGpuFormat::Standard(GpuTextureFormat::R8G8B8A8Unorm).is_hdr());
    }

    #[test]
    fn test_hdr_gpu_format_wgpu_str() {
        assert_eq!(HdrGpuFormat::Rgb10A2Unorm.to_wgpu_format_str(), "Rgb10a2Unorm");
        assert_eq!(
            HdrGpuFormat::R11G11B10UFloat.to_wgpu_format_str(),
            "Rg11b10Ufloat"
        );
        assert_eq!(
            HdrGpuFormat::R32G32B32A32Float.to_wgpu_format_str(),
            "Rgba32Float"
        );
    }

    // ---------------------------------------------------------------------------
    // GPU Format Selection Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_select_hdr_gpu_format_exr_full() {
        let format = select_hdr_gpu_format(HdrSourceFormat::Exr, 32, false);
        assert_eq!(format, HdrGpuFormat::R32G32B32A32Float);
    }

    #[test]
    fn test_select_hdr_gpu_format_exr_compact() {
        let format = select_hdr_gpu_format(HdrSourceFormat::Exr, 16, true);
        assert_eq!(format, HdrGpuFormat::R11G11B10UFloat);
    }

    #[test]
    fn test_select_hdr_gpu_format_hdr() {
        let format = select_hdr_gpu_format(HdrSourceFormat::RadianceHdr, 32, false);
        assert_eq!(
            format,
            HdrGpuFormat::Standard(GpuTextureFormat::R16G16B16A16Float)
        );
    }

    #[test]
    fn test_select_hdr_gpu_format_tiff_8bit() {
        let format = select_hdr_gpu_format(HdrSourceFormat::Tiff, 8, false);
        assert_eq!(
            format,
            HdrGpuFormat::Standard(GpuTextureFormat::R8G8B8A8Unorm)
        );
    }

    #[test]
    fn test_select_hdr_gpu_format_tiff_16bit() {
        let format = select_hdr_gpu_format(HdrSourceFormat::Tiff, 16, false);
        assert_eq!(
            format,
            HdrGpuFormat::Standard(GpuTextureFormat::R16G16B16A16Float)
        );
    }

    #[test]
    fn test_select_hdr_gpu_format_psd() {
        let format = select_hdr_gpu_format(HdrSourceFormat::Psd, 16, false);
        assert_eq!(
            format,
            HdrGpuFormat::Standard(GpuTextureFormat::R16G16B16A16Unorm)
        );
    }

    // ---------------------------------------------------------------------------
    // RGB10A2 Conversion Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_convert_to_rgb10a2_black() {
        let pixels = vec![0.0, 0.0, 0.0, 1.0];
        let result = convert_to_rgb10a2(&pixels, ToneMapOperator::None);
        assert_eq!(result.len(), 4);

        let packed = u32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        assert_eq!(packed & 0x3FF, 0); // R = 0
        assert_eq!((packed >> 10) & 0x3FF, 0); // G = 0
        assert_eq!((packed >> 20) & 0x3FF, 0); // B = 0
        assert_eq!((packed >> 30) & 0x3, 3); // A = 3 (full)
    }

    #[test]
    fn test_convert_to_rgb10a2_white() {
        let pixels = vec![1.0, 1.0, 1.0, 1.0];
        let result = convert_to_rgb10a2(&pixels, ToneMapOperator::None);

        let packed = u32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        assert_eq!(packed & 0x3FF, 1023); // R = 1023
        assert_eq!((packed >> 10) & 0x3FF, 1023); // G = 1023
        assert_eq!((packed >> 20) & 0x3FF, 1023); // B = 1023
    }

    // ---------------------------------------------------------------------------
    // R11G11B10 Conversion Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_convert_to_r11g11b10_zero() {
        let pixels = vec![0.0, 0.0, 0.0, 1.0];
        let result = convert_to_r11g11b10(&pixels);
        assert_eq!(result.len(), 4);

        let packed = u32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        assert_eq!(packed, 0);
    }

    #[test]
    fn test_convert_to_r11g11b10_positive() {
        let pixels = vec![1.0, 1.0, 1.0, 1.0];
        let result = convert_to_r11g11b10(&pixels);

        let packed = u32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        // R11: exponent 15 (value 1.0), mantissa depends on precision
        assert!(packed & 0x7FF > 0); // R non-zero
        assert!((packed >> 11) & 0x7FF > 0); // G non-zero
        assert!((packed >> 22) & 0x3FF > 0); // B non-zero
    }

    // ---------------------------------------------------------------------------
    // HDR Importer Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_hdr_importer_creation() {
        let importer = HdrTextureImporter::new();
        assert_eq!(
            importer.config().preferred_format,
            HdrGpuFormat::Standard(GpuTextureFormat::R16G16B16A16Float)
        );
        assert!(importer.config().generate_preview);
    }

    #[test]
    fn test_hdr_importer_with_config() {
        let config = HdrImportConfig {
            preferred_format: HdrGpuFormat::R32G32B32A32Float,
            generate_preview: false,
            ..Default::default()
        };
        let importer = HdrTextureImporter::with_config(config);
        assert_eq!(
            importer.config().preferred_format,
            HdrGpuFormat::R32G32B32A32Float
        );
        assert!(!importer.config().generate_preview);
    }

    #[test]
    fn test_hdr_importer_supported_extensions() {
        let extensions = HdrTextureImporter::supported_extensions();
        assert!(extensions.contains(&"exr"));
        assert!(extensions.contains(&"hdr"));
        assert!(extensions.contains(&"tiff"));
        assert!(extensions.contains(&"psd"));
    }

    #[test]
    fn test_hdr_importer_is_extension_supported() {
        assert!(HdrTextureImporter::is_extension_supported("exr"));
        assert!(HdrTextureImporter::is_extension_supported("HDR"));
        assert!(HdrTextureImporter::is_extension_supported("tiff"));
        assert!(!HdrTextureImporter::is_extension_supported("png"));
    }

    // ---------------------------------------------------------------------------
    // TIFF Decode Tests (using image crate)
    // ---------------------------------------------------------------------------

    fn create_valid_tiff_8bit(width: u32, height: u32) -> Vec<u8> {
        use image::{ImageBuffer, Rgba, ImageFormat};

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

        let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
            ImageBuffer::from_raw(width, height, pixel_data).expect("Failed to create image");

        let mut tiff_data = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut tiff_data);
        img.write_to(&mut cursor, ImageFormat::Tiff)
            .expect("Failed to write TIFF");

        tiff_data
    }

    #[test]
    fn test_decode_tiff_8bit() {
        let tiff_data = create_valid_tiff_8bit(32, 32);
        let importer = HdrTextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let result = importer.import_from_bytes(&tiff_data, Some("tiff"), &budget);
        assert!(result.is_ok(), "Failed to decode TIFF: {:?}", result.err());

        let asset = result.unwrap();
        assert_eq!(asset.base.metadata.width, 32);
        assert_eq!(asset.base.metadata.height, 32);
        assert_eq!(
            asset.hdr_metadata.hdr_source_format,
            HdrSourceFormat::Tiff
        );
    }

    // ---------------------------------------------------------------------------
    // Radiance HDR Decode Tests
    // ---------------------------------------------------------------------------

    fn create_minimal_radiance_hdr(width: u32, height: u32) -> Vec<u8> {
        use image::{ImageBuffer, Rgb, ImageFormat};

        // Create float RGB image
        let pixel_data: Vec<f32> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    let r = (x as f32) / (width as f32).max(1.0);
                    let g = (y as f32) / (height as f32).max(1.0);
                    let b = 0.5f32;
                    vec![r, g, b]
                })
            })
            .collect();

        let img: ImageBuffer<Rgb<f32>, Vec<f32>> =
            ImageBuffer::from_raw(width, height, pixel_data).expect("Failed to create image");

        let mut hdr_data = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut hdr_data);
        img.write_to(&mut cursor, ImageFormat::Hdr)
            .expect("Failed to write HDR");

        hdr_data
    }

    #[test]
    fn test_decode_radiance_hdr() {
        let hdr_data = create_minimal_radiance_hdr(64, 64);
        let importer = HdrTextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let result = importer.import_from_bytes(&hdr_data, Some("hdr"), &budget);
        assert!(result.is_ok(), "Failed to decode HDR: {:?}", result.err());

        let asset = result.unwrap();
        assert_eq!(asset.base.metadata.width, 64);
        assert_eq!(asset.base.metadata.height, 64);
        assert_eq!(
            asset.hdr_metadata.hdr_source_format,
            HdrSourceFormat::RadianceHdr
        );
        assert_eq!(
            asset.base.metadata.format,
            GpuTextureFormat::R16G16B16A16Float
        );
    }

    #[test]
    fn test_decode_radiance_hdr_dynamic_range() {
        let hdr_data = create_minimal_radiance_hdr(16, 16);
        let mut config = HdrImportConfig::default();
        config.calculate_dynamic_range = true;
        config.keep_hdr_data = true;

        let importer = HdrTextureImporter::with_config(config);
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let asset = importer
            .import_from_bytes(&hdr_data, Some("hdr"), &budget)
            .unwrap();

        assert!(asset.hdr_metadata.dynamic_range.is_some());
        let dr = asset.hdr_metadata.dynamic_range.unwrap();
        assert!(dr.min_luminance >= 0.0);
        assert!(dr.max_luminance >= dr.min_luminance);
    }

    // ---------------------------------------------------------------------------
    // Preview Generation Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_hdr_preview_generation() {
        let hdr_data = create_minimal_radiance_hdr(16, 16);
        let mut config = HdrImportConfig::default();
        config.keep_hdr_data = true;
        config.generate_preview = true;

        let importer = HdrTextureImporter::with_config(config);
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let asset = importer
            .import_from_bytes(&hdr_data, Some("hdr"), &budget)
            .unwrap();

        assert!(asset.preview_data.is_some());
        let preview = asset.preview_data.as_ref().unwrap();
        assert_eq!(preview.len(), 16 * 16 * 4); // RGBA8
    }

    #[test]
    fn test_hdr_asset_generate_preview_different_operators() {
        let hdr_data = create_minimal_radiance_hdr(8, 8);
        let mut config = HdrImportConfig::default();
        config.keep_hdr_data = true;
        config.generate_preview = false;

        let importer = HdrTextureImporter::with_config(config);
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let mut asset = importer
            .import_from_bytes(&hdr_data, Some("hdr"), &budget)
            .unwrap();

        // Generate with Reinhard
        asset.generate_preview(ToneMapOperator::Reinhard, 2.2);
        assert!(asset.preview_data.is_some());
        assert_eq!(asset.preview_tone_map, Some(ToneMapOperator::Reinhard));

        // Generate with ACES
        asset.generate_preview(ToneMapOperator::AcesFilmic, 2.2);
        assert_eq!(asset.preview_tone_map, Some(ToneMapOperator::AcesFilmic));
    }

    // ---------------------------------------------------------------------------
    // HDR Asset Methods Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_hdr_asset_effective_gpu_format() {
        let hdr_data = create_minimal_radiance_hdr(8, 8);
        let mut config = HdrImportConfig::default();
        config.keep_hdr_data = true;

        let importer = HdrTextureImporter::with_config(config);
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let asset = importer
            .import_from_bytes(&hdr_data, Some("hdr"), &budget)
            .unwrap();

        let format = asset.effective_gpu_format();
        assert!(format.is_hdr());
    }

    #[test]
    fn test_hdr_asset_discard_hdr_data() {
        let hdr_data = create_minimal_radiance_hdr(8, 8);
        let mut config = HdrImportConfig::default();
        config.keep_hdr_data = true;

        let importer = HdrTextureImporter::with_config(config);
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let mut asset = importer
            .import_from_bytes(&hdr_data, Some("hdr"), &budget)
            .unwrap();

        assert!(asset.has_hdr_data());
        asset.discard_hdr_data();
        assert!(!asset.has_hdr_data());
    }

    // ---------------------------------------------------------------------------
    // PSD Decode Tests
    // ---------------------------------------------------------------------------

    fn create_minimal_psd_8bit(width: u32, height: u32) -> Vec<u8> {
        let mut data = Vec::new();

        // Signature
        data.extend_from_slice(b"8BPS");
        // Version
        data.extend_from_slice(&1u16.to_be_bytes());
        // Reserved (6 bytes)
        data.extend_from_slice(&[0u8; 6]);
        // Channels (3 = RGB)
        data.extend_from_slice(&3u16.to_be_bytes());
        // Height
        data.extend_from_slice(&height.to_be_bytes());
        // Width
        data.extend_from_slice(&width.to_be_bytes());
        // Depth (8 bit)
        data.extend_from_slice(&8u16.to_be_bytes());
        // Color mode (3 = RGB)
        data.extend_from_slice(&3u16.to_be_bytes());

        // Color mode data section (empty)
        data.extend_from_slice(&0u32.to_be_bytes());

        // Image resources section (empty)
        data.extend_from_slice(&0u32.to_be_bytes());

        // Layer and mask section (empty)
        data.extend_from_slice(&0u32.to_be_bytes());

        // Image data section
        // Compression: 0 = raw
        data.extend_from_slice(&0u16.to_be_bytes());

        // Planar data: R channel, then G, then B
        let pixel_count = (width * height) as usize;
        for c in 0..3 {
            for _ in 0..pixel_count {
                data.push((85 * (c + 1)) as u8); // Simple gradient
            }
        }

        data
    }

    #[test]
    fn test_decode_psd_8bit() {
        let psd_data = create_minimal_psd_8bit(16, 16);
        let importer = HdrTextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let result = importer.import_from_bytes(&psd_data, Some("psd"), &budget);
        assert!(result.is_ok(), "Failed to decode PSD: {:?}", result.err());

        let asset = result.unwrap();
        assert_eq!(asset.base.metadata.width, 16);
        assert_eq!(asset.base.metadata.height, 16);
        assert_eq!(
            asset.hdr_metadata.hdr_source_format,
            HdrSourceFormat::Psd
        );
        assert_eq!(
            asset.base.metadata.format,
            GpuTextureFormat::R8G8B8A8Unorm
        );
    }

    #[test]
    fn test_psd_magic_detection() {
        let psd_data = create_minimal_psd_8bit(8, 8);
        let importer = HdrTextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        // Should detect from magic bytes without extension
        let result = importer.import_from_bytes(&psd_data, None, &budget);
        assert!(result.is_ok(), "Failed to detect PSD from magic: {:?}", result.err());
    }

    #[test]
    fn test_psd_invalid_signature() {
        let invalid_data = b"XXXX\x00\x01";
        let importer = HdrTextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let result = importer.import_from_bytes(invalid_data, Some("psd"), &budget);
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------------------
    // Large Texture Handling Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_hdr_large_texture_within_limits() {
        let hdr_data = create_minimal_radiance_hdr(256, 256);
        let importer = HdrTextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let result = importer.import_from_bytes(&hdr_data, Some("hdr"), &budget);
        assert!(result.is_ok());

        let asset = result.unwrap();
        assert_eq!(asset.base.metadata.width, 256);
        assert_eq!(asset.base.metadata.height, 256);
    }

    #[test]
    fn test_hdr_max_dimension_limit() {
        let mut config = HdrImportConfig::default();
        config.max_dimension = 32;

        let importer = HdrTextureImporter::with_config(config);
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        // 64x64 exceeds our 32 max dimension
        let hdr_data = create_minimal_radiance_hdr(64, 64);
        let result = importer.import_from_bytes(&hdr_data, Some("hdr"), &budget);
        assert!(matches!(
            result,
            Err(TextureImportError::InvalidDimensions { .. })
        ));
    }

    // ---------------------------------------------------------------------------
    // Memory Budget Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_hdr_import_updates_budget() {
        let hdr_data = create_minimal_radiance_hdr(32, 32);
        let importer = HdrTextureImporter::new();
        let budget = MemoryBudgetTracker::new(1024 * 1024 * 100);

        let initial_usage = budget.usage();
        let asset = importer
            .import_from_bytes(&hdr_data, Some("hdr"), &budget)
            .unwrap();

        let expected_size = 32 * 32 * 8; // R16G16B16A16Float
        assert_eq!(budget.usage(), initial_usage + expected_size);
        assert_eq!(budget.get_allocation(asset.base.id), Some(expected_size));
    }

    #[test]
    fn test_hdr_import_fails_on_budget_exceeded() {
        let hdr_data = create_minimal_radiance_hdr(64, 64);
        let importer = HdrTextureImporter::new();
        let budget = MemoryBudgetTracker::new(100); // Very small budget

        let result = importer.import_from_bytes(&hdr_data, Some("hdr"), &budget);
        assert!(matches!(
            result,
            Err(TextureImportError::BudgetExceeded { .. })
        ));
    }

    // ---------------------------------------------------------------------------
    // Display Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_hdr_source_format_display() {
        assert_eq!(format!("{}", HdrSourceFormat::Exr), "OpenEXR");
        assert_eq!(format!("{}", HdrSourceFormat::RadianceHdr), "Radiance HDR");
        assert_eq!(format!("{}", HdrSourceFormat::Tiff), "TIFF");
        assert_eq!(format!("{}", HdrSourceFormat::Psd), "Photoshop PSD");
    }

    #[test]
    fn test_hdr_gpu_format_display() {
        assert_eq!(format!("{}", HdrGpuFormat::Rgb10A2Unorm), "RGB10_A2_UNORM");
        assert_eq!(
            format!("{}", HdrGpuFormat::R11G11B10UFloat),
            "R11G11B10_UFLOAT"
        );
        assert_eq!(
            format!("{}", HdrGpuFormat::R32G32B32A32Float),
            "R32G32B32A32_FLOAT"
        );
    }
}
