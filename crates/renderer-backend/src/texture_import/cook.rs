//! Texture cooking pipeline.
//!
//! Converts raw texture data to GPU-optimized formats with automatic format
//! selection based on texture usage, mip generation, and compression.
//!
//! # Pipeline
//!
//! 1. **Format Selection**: Choose optimal GPU format based on texture usage
//!    - BaseColor: SRGB, BC7
//!    - NormalMap: UNORM, BC5
//!    - Roughness/Metallic/Occlusion: Linear, BC4
//!    - Emissive: HDR, BC6H
//!    - Data: R32F
//!
//! 2. **Mip Generation**: Generate mip levels using box filtering
//!
//! 3. **Output**: Cooked texture ready for GPU upload
//!
//! # Example
//!
//! ```
//! use renderer_backend::texture_import::{TextureData, TextureFormat, TextureUsage, TextureCooker};
//!
//! let data = TextureData::new(64, 64, TextureFormat::Rgba8, vec![128; 64 * 64 * 4], 1);
//! let cooker = TextureCooker::new().with_mips(true);
//! let cooked = cooker.cook(&data, TextureUsage::BaseColor).unwrap();
//!
//! assert_eq!(cooked.width, 64);
//! assert_eq!(cooked.mip_data.len(), 7); // 64, 32, 16, 8, 4, 2, 1
//! ```

use std::fmt;

use super::{TextureData, TextureFormat};

// ---------------------------------------------------------------------------
// TextureUsage
// ---------------------------------------------------------------------------

/// Describes how a texture will be used, enabling optimal format selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureUsage {
    /// Base color/albedo texture (sRGB color space, BC7 compression).
    BaseColor,
    /// Normal map (linear, BC5 two-channel compression).
    NormalMap,
    /// Roughness map (linear, single-channel BC4).
    Roughness,
    /// Metallic map (linear, single-channel BC4).
    Metallic,
    /// Ambient occlusion map (linear, single-channel BC4).
    Occlusion,
    /// Emissive/HDR texture (BC6H HDR compression).
    Emissive,
    /// Raw data texture (R32 float).
    Data,
    /// Unknown usage (fallback to RGBA8).
    Unknown,
}

impl fmt::Display for TextureUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextureUsage::BaseColor => write!(f, "BaseColor"),
            TextureUsage::NormalMap => write!(f, "NormalMap"),
            TextureUsage::Roughness => write!(f, "Roughness"),
            TextureUsage::Metallic => write!(f, "Metallic"),
            TextureUsage::Occlusion => write!(f, "Occlusion"),
            TextureUsage::Emissive => write!(f, "Emissive"),
            TextureUsage::Data => write!(f, "Data"),
            TextureUsage::Unknown => write!(f, "Unknown"),
        }
    }
}

impl Default for TextureUsage {
    fn default() -> Self {
        TextureUsage::Unknown
    }
}

// ---------------------------------------------------------------------------
// GpuTextureFormat
// ---------------------------------------------------------------------------

/// Target GPU texture format for cooked textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuTextureFormat {
    /// RGBA 8-bit unorm (linear).
    Rgba8Unorm,
    /// RGBA 8-bit sRGB.
    Rgba8Srgb,
    /// BC4 single-channel compression (unorm).
    Bc4Unorm,
    /// BC5 two-channel compression (unorm, for normal maps).
    Bc5Unorm,
    /// BC6H HDR compression (unsigned float).
    Bc6hFloat,
    /// BC7 high-quality RGBA compression (unorm).
    Bc7Unorm,
    /// BC7 high-quality RGBA compression (sRGB).
    Bc7Srgb,
    /// Single-channel 32-bit float.
    R32Float,
    /// Single-channel 8-bit unorm.
    R8Unorm,
    /// Two-channel 8-bit unorm.
    Rg8Unorm,
}

impl GpuTextureFormat {
    /// Returns the number of bytes per pixel (or per block for compressed formats).
    ///
    /// For block-compressed formats, this returns the bytes per 4x4 block.
    #[inline]
    pub const fn bytes_per_pixel_or_block(&self) -> usize {
        match self {
            GpuTextureFormat::Rgba8Unorm | GpuTextureFormat::Rgba8Srgb => 4,
            GpuTextureFormat::Bc4Unorm => 8,  // 8 bytes per 4x4 block
            GpuTextureFormat::Bc5Unorm => 16, // 16 bytes per 4x4 block
            GpuTextureFormat::Bc6hFloat => 16,
            GpuTextureFormat::Bc7Unorm | GpuTextureFormat::Bc7Srgb => 16,
            GpuTextureFormat::R32Float => 4,
            GpuTextureFormat::R8Unorm => 1,
            GpuTextureFormat::Rg8Unorm => 2,
        }
    }

    /// Returns true if this is a block-compressed format.
    #[inline]
    pub const fn is_block_compressed(&self) -> bool {
        matches!(
            self,
            GpuTextureFormat::Bc4Unorm
                | GpuTextureFormat::Bc5Unorm
                | GpuTextureFormat::Bc6hFloat
                | GpuTextureFormat::Bc7Unorm
                | GpuTextureFormat::Bc7Srgb
        )
    }

    /// Returns the block size (4 for BC formats, 1 for uncompressed).
    #[inline]
    pub const fn block_size(&self) -> u32 {
        if self.is_block_compressed() {
            4
        } else {
            1
        }
    }

    /// Returns true if this format uses sRGB color space.
    #[inline]
    pub const fn is_srgb(&self) -> bool {
        matches!(self, GpuTextureFormat::Rgba8Srgb | GpuTextureFormat::Bc7Srgb)
    }
}

impl fmt::Display for GpuTextureFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuTextureFormat::Rgba8Unorm => write!(f, "RGBA8_UNORM"),
            GpuTextureFormat::Rgba8Srgb => write!(f, "RGBA8_SRGB"),
            GpuTextureFormat::Bc4Unorm => write!(f, "BC4_UNORM"),
            GpuTextureFormat::Bc5Unorm => write!(f, "BC5_UNORM"),
            GpuTextureFormat::Bc6hFloat => write!(f, "BC6H_FLOAT"),
            GpuTextureFormat::Bc7Unorm => write!(f, "BC7_UNORM"),
            GpuTextureFormat::Bc7Srgb => write!(f, "BC7_SRGB"),
            GpuTextureFormat::R32Float => write!(f, "R32_FLOAT"),
            GpuTextureFormat::R8Unorm => write!(f, "R8_UNORM"),
            GpuTextureFormat::Rg8Unorm => write!(f, "RG8_UNORM"),
        }
    }
}

// ---------------------------------------------------------------------------
// CookedTexture
// ---------------------------------------------------------------------------

/// A texture that has been processed and is ready for GPU upload.
#[derive(Debug, Clone)]
pub struct CookedTexture {
    /// Target GPU format.
    pub format: GpuTextureFormat,
    /// Width at mip level 0.
    pub width: u32,
    /// Height at mip level 0.
    pub height: u32,
    /// Mip level data (index 0 = full resolution).
    pub mip_data: Vec<Vec<u8>>,
    /// Usage hint that was used for format selection.
    pub usage: TextureUsage,
}

impl CookedTexture {
    /// Returns the number of mip levels.
    #[inline]
    pub fn mip_count(&self) -> u32 {
        self.mip_data.len() as u32
    }

    /// Returns the dimensions of a specific mip level.
    ///
    /// Returns `None` if the mip level doesn't exist.
    pub fn mip_dimensions(&self, level: u32) -> Option<(u32, u32)> {
        if level >= self.mip_count() {
            return None;
        }
        let divisor = 1 << level;
        let w = (self.width / divisor).max(1);
        let h = (self.height / divisor).max(1);
        Some((w, h))
    }

    /// Returns the data for a specific mip level.
    pub fn mip_level(&self, level: u32) -> Option<&[u8]> {
        self.mip_data.get(level as usize).map(|v| v.as_slice())
    }

    /// Returns total size of all mip data in bytes.
    pub fn total_size(&self) -> usize {
        self.mip_data.iter().map(|m| m.len()).sum()
    }

    /// Validates that all mip levels have correct sizes.
    pub fn is_valid(&self) -> bool {
        if self.mip_data.is_empty() {
            return false;
        }

        for (level, data) in self.mip_data.iter().enumerate() {
            let (w, h) = match self.mip_dimensions(level as u32) {
                Some(dims) => dims,
                None => return false,
            };

            let expected_size = if self.format.is_block_compressed() {
                let blocks_x = (w + 3) / 4;
                let blocks_y = (h + 3) / 4;
                (blocks_x * blocks_y) as usize * self.format.bytes_per_pixel_or_block()
            } else {
                (w * h) as usize * self.format.bytes_per_pixel_or_block()
            };

            if data.len() != expected_size {
                return false;
            }
        }

        true
    }
}

// ---------------------------------------------------------------------------
// CookError
// ---------------------------------------------------------------------------

/// Errors that can occur during texture cooking.
#[derive(Debug, Clone)]
pub enum CookError {
    /// The source format is not supported for cooking.
    UnsupportedFormat(String),
    /// Invalid texture dimensions.
    InvalidDimensions { width: u32, height: u32 },
    /// Mip generation failed.
    MipGenerationFailed(String),
    /// The input data is invalid.
    InvalidInput(String),
}

impl fmt::Display for CookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CookError::UnsupportedFormat(fmt) => {
                write!(f, "unsupported format for cooking: {}", fmt)
            }
            CookError::InvalidDimensions { width, height } => {
                write!(f, "invalid dimensions: {}x{}", width, height)
            }
            CookError::MipGenerationFailed(msg) => {
                write!(f, "mip generation failed: {}", msg)
            }
            CookError::InvalidInput(msg) => {
                write!(f, "invalid input: {}", msg)
            }
        }
    }
}

impl std::error::Error for CookError {}

// ---------------------------------------------------------------------------
// TextureCooker
// ---------------------------------------------------------------------------

/// Texture cooking pipeline configuration and execution.
///
/// Converts raw texture data to GPU-optimized formats with optional mip generation.
#[derive(Debug, Clone)]
pub struct TextureCooker {
    /// Whether to generate mip levels.
    pub(crate) generate_mips: bool,
    /// Maximum number of mip levels to generate (0 = full chain).
    pub(crate) max_mip_levels: u32,
    /// Use block compression when possible.
    pub(crate) use_compression: bool,
}

impl TextureCooker {
    /// Create a new texture cooker with default settings.
    ///
    /// Default: mips enabled, full chain, compression disabled (uncompressed output).
    pub fn new() -> Self {
        Self {
            generate_mips: true,
            max_mip_levels: 0,
            use_compression: false,
        }
    }

    /// Enable or disable mip generation.
    pub fn with_mips(mut self, generate: bool) -> Self {
        self.generate_mips = generate;
        self
    }

    /// Set maximum number of mip levels (0 = full chain down to 1x1).
    pub fn with_max_mip_levels(mut self, max_levels: u32) -> Self {
        self.max_mip_levels = max_levels;
        self
    }

    /// Enable or disable block compression output.
    ///
    /// When disabled, outputs uncompressed formats that can be read back.
    /// When enabled, outputs BC formats (requires compression at upload time).
    pub fn with_compression(mut self, compress: bool) -> Self {
        self.use_compression = compress;
        self
    }

    /// Select optimal GPU format based on texture usage.
    ///
    /// # Arguments
    ///
    /// * `usage` - How the texture will be used
    /// * `has_alpha` - Whether the texture has meaningful alpha channel
    /// * `use_compression` - Whether to use block compression
    pub fn select_format(
        usage: TextureUsage,
        has_alpha: bool,
        use_compression: bool,
    ) -> GpuTextureFormat {
        if use_compression {
            match usage {
                TextureUsage::BaseColor => GpuTextureFormat::Bc7Srgb,
                TextureUsage::NormalMap => GpuTextureFormat::Bc5Unorm,
                TextureUsage::Roughness | TextureUsage::Metallic | TextureUsage::Occlusion => {
                    GpuTextureFormat::Bc4Unorm
                }
                TextureUsage::Emissive => GpuTextureFormat::Bc6hFloat,
                TextureUsage::Data => GpuTextureFormat::R32Float,
                TextureUsage::Unknown => {
                    if has_alpha {
                        GpuTextureFormat::Bc7Unorm
                    } else {
                        GpuTextureFormat::Bc7Unorm
                    }
                }
            }
        } else {
            // Uncompressed formats
            match usage {
                TextureUsage::BaseColor => GpuTextureFormat::Rgba8Srgb,
                TextureUsage::NormalMap => GpuTextureFormat::Rg8Unorm,
                TextureUsage::Roughness | TextureUsage::Metallic | TextureUsage::Occlusion => {
                    GpuTextureFormat::R8Unorm
                }
                TextureUsage::Emissive => GpuTextureFormat::Rgba8Unorm, // HDR needs float, fallback
                TextureUsage::Data => GpuTextureFormat::R32Float,
                TextureUsage::Unknown => {
                    if has_alpha {
                        GpuTextureFormat::Rgba8Unorm
                    } else {
                        GpuTextureFormat::Rgba8Unorm
                    }
                }
            }
        }
    }

    /// Cook a texture with automatic format selection.
    ///
    /// # Arguments
    ///
    /// * `input` - Raw texture data to cook
    /// * `usage` - How the texture will be used
    ///
    /// # Errors
    ///
    /// Returns an error if the input is invalid or cooking fails.
    pub fn cook(&self, input: &TextureData, usage: TextureUsage) -> Result<CookedTexture, CookError> {
        // Validate input
        if input.width == 0 || input.height == 0 {
            return Err(CookError::InvalidDimensions {
                width: input.width,
                height: input.height,
            });
        }

        if !input.is_valid() {
            return Err(CookError::InvalidInput(
                "texture data size doesn't match dimensions".to_string(),
            ));
        }

        // Detect alpha
        let has_alpha = Self::detect_alpha(input);

        // Select output format
        let format = Self::select_format(usage, has_alpha, self.use_compression);

        // Convert to working format (RGBA8 for processing)
        let rgba_data = self.convert_to_rgba8(input)?;

        // Generate mip chain
        let mip_data = if self.generate_mips {
            self.generate_mip_chain(&rgba_data, input.width, input.height, format, usage)?
        } else {
            vec![self.convert_to_output_format(&rgba_data, input.width, input.height, format, usage)?]
        };

        Ok(CookedTexture {
            format,
            width: input.width,
            height: input.height,
            mip_data,
            usage,
        })
    }

    /// Detect if texture has meaningful alpha values.
    pub(crate) fn detect_alpha(input: &TextureData) -> bool {
        match input.format {
            TextureFormat::Rgba8 | TextureFormat::Rgba8Srgb => {
                // Check if any alpha value differs from 255
                input
                    .data
                    .chunks(4)
                    .any(|pixel| pixel.len() == 4 && pixel[3] != 255)
            }
            TextureFormat::Rgba16 => {
                // 16-bit RGBA: check if alpha differs from max
                input.data.chunks(8).any(|pixel| {
                    if pixel.len() == 8 {
                        let alpha = u16::from_le_bytes([pixel[6], pixel[7]]);
                        alpha != 65535
                    } else {
                        false
                    }
                })
            }
            TextureFormat::Rgba32F => {
                // 32-bit float RGBA
                input.data.chunks(16).any(|pixel| {
                    if pixel.len() == 16 {
                        let alpha_bytes: [u8; 4] = [pixel[12], pixel[13], pixel[14], pixel[15]];
                        let alpha = f32::from_le_bytes(alpha_bytes);
                        (alpha - 1.0).abs() > 0.001
                    } else {
                        false
                    }
                })
            }
            TextureFormat::Rg8 | TextureFormat::Rg16 | TextureFormat::Rg32F => {
                // Second channel might be alpha for grayscale+alpha
                true
            }
            _ => false,
        }
    }

    /// Convert input texture to RGBA8 working format.
    fn convert_to_rgba8(&self, input: &TextureData) -> Result<Vec<u8>, CookError> {
        let pixel_count = (input.width as usize) * (input.height as usize);
        let mut rgba = vec![0u8; pixel_count * 4];

        match input.format {
            TextureFormat::R8 => {
                for (i, &v) in input.data.iter().take(pixel_count).enumerate() {
                    rgba[i * 4] = v;
                    rgba[i * 4 + 1] = v;
                    rgba[i * 4 + 2] = v;
                    rgba[i * 4 + 3] = 255;
                }
            }
            TextureFormat::Rg8 => {
                for (i, chunk) in input.data.chunks(2).take(pixel_count).enumerate() {
                    rgba[i * 4] = chunk[0];
                    rgba[i * 4 + 1] = chunk.get(1).copied().unwrap_or(0);
                    rgba[i * 4 + 2] = 0;
                    rgba[i * 4 + 3] = 255;
                }
            }
            TextureFormat::Rgba8 | TextureFormat::Rgba8Srgb => {
                rgba.copy_from_slice(&input.data[..pixel_count * 4]);
            }
            TextureFormat::R16 => {
                for (i, chunk) in input.data.chunks(2).take(pixel_count).enumerate() {
                    let v = u16::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0)]);
                    let v8 = (v >> 8) as u8;
                    rgba[i * 4] = v8;
                    rgba[i * 4 + 1] = v8;
                    rgba[i * 4 + 2] = v8;
                    rgba[i * 4 + 3] = 255;
                }
            }
            TextureFormat::Rg16 => {
                for (i, chunk) in input.data.chunks(4).take(pixel_count).enumerate() {
                    let r = u16::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0)]);
                    let g = u16::from_le_bytes([
                        chunk.get(2).copied().unwrap_or(0),
                        chunk.get(3).copied().unwrap_or(0),
                    ]);
                    rgba[i * 4] = (r >> 8) as u8;
                    rgba[i * 4 + 1] = (g >> 8) as u8;
                    rgba[i * 4 + 2] = 0;
                    rgba[i * 4 + 3] = 255;
                }
            }
            TextureFormat::Rgba16 => {
                for (i, chunk) in input.data.chunks(8).take(pixel_count).enumerate() {
                    for c in 0..4 {
                        let idx = c * 2;
                        let v = u16::from_le_bytes([
                            chunk.get(idx).copied().unwrap_or(0),
                            chunk.get(idx + 1).copied().unwrap_or(0),
                        ]);
                        rgba[i * 4 + c] = (v >> 8) as u8;
                    }
                }
            }
            TextureFormat::R32F => {
                for (i, chunk) in input.data.chunks(4).take(pixel_count).enumerate() {
                    let bytes: [u8; 4] = [
                        chunk[0],
                        chunk.get(1).copied().unwrap_or(0),
                        chunk.get(2).copied().unwrap_or(0),
                        chunk.get(3).copied().unwrap_or(0),
                    ];
                    let v = f32::from_le_bytes(bytes);
                    let v8 = (v.clamp(0.0, 1.0) * 255.0) as u8;
                    rgba[i * 4] = v8;
                    rgba[i * 4 + 1] = v8;
                    rgba[i * 4 + 2] = v8;
                    rgba[i * 4 + 3] = 255;
                }
            }
            TextureFormat::Rg32F => {
                for (i, chunk) in input.data.chunks(8).take(pixel_count).enumerate() {
                    let r_bytes: [u8; 4] = [
                        chunk[0],
                        chunk.get(1).copied().unwrap_or(0),
                        chunk.get(2).copied().unwrap_or(0),
                        chunk.get(3).copied().unwrap_or(0),
                    ];
                    let g_bytes: [u8; 4] = [
                        chunk.get(4).copied().unwrap_or(0),
                        chunk.get(5).copied().unwrap_or(0),
                        chunk.get(6).copied().unwrap_or(0),
                        chunk.get(7).copied().unwrap_or(0),
                    ];
                    let r = f32::from_le_bytes(r_bytes);
                    let g = f32::from_le_bytes(g_bytes);
                    rgba[i * 4] = (r.clamp(0.0, 1.0) * 255.0) as u8;
                    rgba[i * 4 + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
                    rgba[i * 4 + 2] = 0;
                    rgba[i * 4 + 3] = 255;
                }
            }
            TextureFormat::Rgba32F => {
                for (i, chunk) in input.data.chunks(16).take(pixel_count).enumerate() {
                    for c in 0..4 {
                        let idx = c * 4;
                        let bytes: [u8; 4] = [
                            chunk.get(idx).copied().unwrap_or(0),
                            chunk.get(idx + 1).copied().unwrap_or(0),
                            chunk.get(idx + 2).copied().unwrap_or(0),
                            chunk.get(idx + 3).copied().unwrap_or(0),
                        ];
                        let v = f32::from_le_bytes(bytes);
                        rgba[i * 4 + c] = (v.clamp(0.0, 1.0) * 255.0) as u8;
                    }
                }
            }
        }

        Ok(rgba)
    }

    /// Generate full mip chain for a texture.
    fn generate_mip_chain(
        &self,
        rgba_data: &[u8],
        width: u32,
        height: u32,
        format: GpuTextureFormat,
        usage: TextureUsage,
    ) -> Result<Vec<Vec<u8>>, CookError> {
        // Calculate number of mip levels
        let max_dim = width.max(height);
        let full_chain_levels = (max_dim as f32).log2().floor() as u32 + 1;
        let num_levels = if self.max_mip_levels > 0 {
            full_chain_levels.min(self.max_mip_levels)
        } else {
            full_chain_levels
        };

        let mut mip_data = Vec::with_capacity(num_levels as usize);

        // Level 0: convert to output format
        mip_data.push(self.convert_to_output_format(rgba_data, width, height, format, usage)?);

        // Generate subsequent levels
        let mut current_rgba = rgba_data.to_vec();
        let mut current_w = width;
        let mut current_h = height;

        for _ in 1..num_levels {
            // Downsample
            let new_w = (current_w / 2).max(1);
            let new_h = (current_h / 2).max(1);

            let downsampled = box_filter_2x2(&current_rgba, current_w, current_h, 4);

            // Convert to output format
            let output = self.convert_to_output_format(&downsampled, new_w, new_h, format, usage)?;
            mip_data.push(output);

            current_rgba = downsampled;
            current_w = new_w;
            current_h = new_h;
        }

        Ok(mip_data)
    }

    /// Convert RGBA8 data to the target output format.
    fn convert_to_output_format(
        &self,
        rgba_data: &[u8],
        width: u32,
        height: u32,
        format: GpuTextureFormat,
        _usage: TextureUsage,
    ) -> Result<Vec<u8>, CookError> {
        let pixel_count = (width as usize) * (height as usize);

        match format {
            GpuTextureFormat::Rgba8Unorm | GpuTextureFormat::Rgba8Srgb => {
                Ok(rgba_data[..pixel_count * 4].to_vec())
            }
            GpuTextureFormat::R8Unorm => {
                // Extract red channel (or compute luminance)
                let mut output = vec![0u8; pixel_count];
                for (i, chunk) in rgba_data.chunks(4).take(pixel_count).enumerate() {
                    // Simple luminance: 0.299R + 0.587G + 0.114B
                    let r = chunk[0] as f32;
                    let g = chunk.get(1).copied().unwrap_or(0) as f32;
                    let b = chunk.get(2).copied().unwrap_or(0) as f32;
                    let lum = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                    output[i] = lum;
                }
                Ok(output)
            }
            GpuTextureFormat::Rg8Unorm => {
                // Extract RG channels (for normal maps)
                let mut output = vec![0u8; pixel_count * 2];
                for (i, chunk) in rgba_data.chunks(4).take(pixel_count).enumerate() {
                    output[i * 2] = chunk[0];
                    output[i * 2 + 1] = chunk.get(1).copied().unwrap_or(0);
                }
                Ok(output)
            }
            GpuTextureFormat::R32Float => {
                // Convert to float
                let mut output = vec![0u8; pixel_count * 4];
                for (i, chunk) in rgba_data.chunks(4).take(pixel_count).enumerate() {
                    let r = chunk[0] as f32 / 255.0;
                    let bytes = r.to_le_bytes();
                    output[i * 4..i * 4 + 4].copy_from_slice(&bytes);
                }
                Ok(output)
            }
            // Block compressed formats - output placeholder data
            // In a real implementation, you'd use a BC encoder library
            GpuTextureFormat::Bc4Unorm => {
                let blocks_x = (width + 3) / 4;
                let blocks_y = (height + 3) / 4;
                let block_count = (blocks_x * blocks_y) as usize;
                // BC4: 8 bytes per block
                let mut output = vec![0u8; block_count * 8];
                // Simple encoding: store average value per block
                for by in 0..blocks_y {
                    for bx in 0..blocks_x {
                        let block_idx = (by * blocks_x + bx) as usize;
                        let mut sum = 0u32;
                        let mut count = 0u32;
                        for py in 0..4 {
                            for px in 0..4 {
                                let x = (bx * 4 + px) as usize;
                                let y = (by * 4 + py) as usize;
                                if x < width as usize && y < height as usize {
                                    let idx = (y * width as usize + x) * 4;
                                    sum += rgba_data[idx] as u32;
                                    count += 1;
                                }
                            }
                        }
                        let avg = if count > 0 { (sum / count) as u8 } else { 0 };
                        // Simplified BC4: endpoints = avg, indices = 0
                        output[block_idx * 8] = avg;
                        output[block_idx * 8 + 1] = avg;
                    }
                }
                Ok(output)
            }
            GpuTextureFormat::Bc5Unorm => {
                let blocks_x = (width + 3) / 4;
                let blocks_y = (height + 3) / 4;
                let block_count = (blocks_x * blocks_y) as usize;
                // BC5: 16 bytes per block (two BC4 blocks)
                let mut output = vec![0u8; block_count * 16];
                for by in 0..blocks_y {
                    for bx in 0..blocks_x {
                        let block_idx = (by * blocks_x + bx) as usize;
                        let mut sum_r = 0u32;
                        let mut sum_g = 0u32;
                        let mut count = 0u32;
                        for py in 0..4 {
                            for px in 0..4 {
                                let x = (bx * 4 + px) as usize;
                                let y = (by * 4 + py) as usize;
                                if x < width as usize && y < height as usize {
                                    let idx = (y * width as usize + x) * 4;
                                    sum_r += rgba_data[idx] as u32;
                                    sum_g += rgba_data.get(idx + 1).copied().unwrap_or(0) as u32;
                                    count += 1;
                                }
                            }
                        }
                        let avg_r = if count > 0 { (sum_r / count) as u8 } else { 0 };
                        let avg_g = if count > 0 { (sum_g / count) as u8 } else { 0 };
                        // First BC4 block (red)
                        output[block_idx * 16] = avg_r;
                        output[block_idx * 16 + 1] = avg_r;
                        // Second BC4 block (green)
                        output[block_idx * 16 + 8] = avg_g;
                        output[block_idx * 16 + 9] = avg_g;
                    }
                }
                Ok(output)
            }
            GpuTextureFormat::Bc6hFloat | GpuTextureFormat::Bc7Unorm | GpuTextureFormat::Bc7Srgb => {
                let blocks_x = (width + 3) / 4;
                let blocks_y = (height + 3) / 4;
                let block_count = (blocks_x * blocks_y) as usize;
                // BC6H/BC7: 16 bytes per block
                // For now, output placeholder data
                Ok(vec![0u8; block_count * 16])
            }
        }
    }
}

impl Default for TextureCooker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Mip generation helpers
// ---------------------------------------------------------------------------

/// Downsample an image by 2x using box filtering.
///
/// Takes RGBA8 data and produces half-resolution RGBA8 data.
pub fn box_filter_2x2(data: &[u8], width: u32, height: u32, channels: u32) -> Vec<u8> {
    let new_w = (width / 2).max(1);
    let new_h = (height / 2).max(1);
    let stride = (width * channels) as usize;
    let new_stride = (new_w * channels) as usize;
    let mut output = vec![0u8; (new_w * new_h * channels) as usize];

    for y in 0..new_h {
        for x in 0..new_w {
            let src_x = (x * 2) as usize;
            let src_y = (y * 2) as usize;

            for c in 0..(channels as usize) {
                let mut sum = 0u32;
                let mut count = 0u32;

                // Sample 2x2 block
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = src_x + dx;
                        let sy = src_y + dy;
                        if sx < width as usize && sy < height as usize {
                            let idx = sy * stride + sx * (channels as usize) + c;
                            sum += data.get(idx).copied().unwrap_or(0) as u32;
                            count += 1;
                        }
                    }
                }

                let avg = if count > 0 { (sum / count) as u8 } else { 0 };
                let out_idx = (y as usize) * new_stride + (x as usize) * (channels as usize) + c;
                output[out_idx] = avg;
            }
        }
    }

    output
}

/// Calculate the maximum number of mip levels for given dimensions.
#[inline]
pub fn calculate_mip_levels(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height);
    if max_dim == 0 {
        0
    } else {
        (max_dim as f32).log2().floor() as u32 + 1
    }
}

// ---------------------------------------------------------------------------
// TextureData extension
// ---------------------------------------------------------------------------

impl TextureData {
    /// Cook this texture for GPU upload.
    ///
    /// # Arguments
    ///
    /// * `usage` - How the texture will be used
    ///
    /// # Errors
    ///
    /// Returns an error if cooking fails.
    pub fn cook(&self, usage: TextureUsage) -> Result<CookedTexture, CookError> {
        TextureCooker::new().cook(self, usage)
    }

    /// Cook this texture with custom cooker settings.
    pub fn cook_with(&self, cooker: &TextureCooker, usage: TextureUsage) -> Result<CookedTexture, CookError> {
        cooker.cook(self, usage)
    }
}

