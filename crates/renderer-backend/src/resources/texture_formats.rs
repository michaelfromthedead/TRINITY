//! Texture format selection with platform awareness for TRINITY.
//!
//! This module provides intelligent texture format selection based on platform
//! capabilities, backend type, and use case requirements.
//!
//! # Overview
//!
//! Different platforms have different optimal texture formats:
//!
//! - **Windows/Linux (Vulkan/DX12)**: Prefer BC compression, sRGB for color
//! - **macOS/iOS (Metal)**: Prefer BGRA for swapchain, ASTC for compression
//! - **Mobile (Vulkan/GLES)**: ASTC or ETC2 compression, BGRA varies
//! - **WebGPU**: Limited format support, fallbacks required
//!
//! # Architecture
//!
//! ```text
//! TextureFormatSelector
//!     - Captures adapter backend and features at creation
//!     - Provides format selection methods for each use case
//!     - Implements fallback chains for unsupported formats
//!
//! Format Tables
//!     - COLOR_FORMATS: Common color texture formats
//!     - DEPTH_FORMATS: Depth/stencil formats
//!     - COMPRESSED_BC: BC compression formats (desktop)
//!     - COMPRESSED_ASTC: ASTC compression formats (mobile/Apple)
//!     - COMPRESSED_ETC2: ETC2 compression formats (mobile fallback)
//!     - NORMAL_MAP_FORMATS: Formats suitable for normal maps
//! ```
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::texture_formats::TextureFormatSelector;
//!
//! # fn example(adapter: &wgpu::Adapter) {
//! let selector = TextureFormatSelector::new(adapter);
//!
//! // Get optimal formats for each use case
//! let color_fmt = selector.color_attachment(false);
//! let hdr_fmt = selector.color_attachment(true);
//! let depth_fmt = selector.depth(false);
//! let depth_stencil_fmt = selector.depth(true);
//! let normal_fmt = selector.normal_map(false);
//! let compressed_fmt = selector.compressed_color(false);
//!
//! println!("Backend: {:?}", selector.backend());
//! println!("Color attachment: {:?}", color_fmt);
//! println!("Depth format: {:?}", depth_fmt);
//! # }
//! ```

use wgpu::{Adapter, Backend, Features, TextureFormat};

// ============================================================================
// Format Tables
// ============================================================================

/// Common texture format tables organized by use case.
///
/// These tables provide reference lists of formats for each category,
/// ordered by preference (most preferred first).
pub mod format_tables {
    use wgpu::TextureFormat;

    /// Color texture formats for render targets and general use.
    ///
    /// Ordered by typical preference for color attachments.
    pub const COLOR_FORMATS: &[TextureFormat] = &[
        // sRGB formats (preferred for color-correct rendering)
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8UnormSrgb,
        // Linear formats (for post-processing or compute)
        TextureFormat::Rgba8Unorm,
        TextureFormat::Bgra8Unorm,
        // High precision
        TextureFormat::Rgba16Float,
        TextureFormat::Rgb10a2Unorm,
        // Packed formats
        TextureFormat::Rg11b10Float,
        TextureFormat::Rgb9e5Ufloat,
    ];

    /// HDR-capable color formats.
    ///
    /// These formats support values outside [0, 1] for high dynamic range.
    pub const HDR_COLOR_FORMATS: &[TextureFormat] = &[
        TextureFormat::Rgba16Float,
        TextureFormat::Rg11b10Float, // Unsigned only, but high precision
        TextureFormat::Rgba32Float,  // Maximum precision, rarely needed
    ];

    /// Depth texture formats ordered by precision.
    ///
    /// Higher precision formats are listed first.
    pub const DEPTH_FORMATS: &[TextureFormat] = &[
        TextureFormat::Depth32Float,
        TextureFormat::Depth24Plus,
        TextureFormat::Depth16Unorm,
    ];

    /// Depth-stencil combined formats.
    ///
    /// Use these when both depth testing and stencil operations are needed.
    pub const DEPTH_STENCIL_FORMATS: &[TextureFormat] = &[
        TextureFormat::Depth32FloatStencil8,
        TextureFormat::Depth24PlusStencil8,
    ];

    /// BC compressed texture formats (Windows/Linux desktop).
    ///
    /// These formats require the `TEXTURE_COMPRESSION_BC` feature.
    pub const COMPRESSED_BC: &[TextureFormat] = &[
        // BC7: High quality RGBA (best quality, larger)
        TextureFormat::Bc7RgbaUnorm,
        TextureFormat::Bc7RgbaUnormSrgb,
        // BC3: RGBA with separate alpha (DXT5)
        TextureFormat::Bc3RgbaUnorm,
        TextureFormat::Bc3RgbaUnormSrgb,
        // BC1: RGB with optional 1-bit alpha (DXT1)
        TextureFormat::Bc1RgbaUnorm,
        TextureFormat::Bc1RgbaUnormSrgb,
        // BC4: Single channel (useful for heightmaps)
        TextureFormat::Bc4RUnorm,
        TextureFormat::Bc4RSnorm,
        // BC5: Two channel (normal maps)
        TextureFormat::Bc5RgUnorm,
        TextureFormat::Bc5RgSnorm,
        // BC6H: HDR RGB
        TextureFormat::Bc6hRgbUfloat,
        TextureFormat::Bc6hRgbFloat,
    ];

    /// BC formats for sRGB color textures.
    pub const COMPRESSED_BC_SRGB: &[TextureFormat] = &[
        TextureFormat::Bc7RgbaUnormSrgb,
        TextureFormat::Bc3RgbaUnormSrgb,
        TextureFormat::Bc1RgbaUnormSrgb,
    ];

    /// BC formats for linear color textures.
    pub const COMPRESSED_BC_LINEAR: &[TextureFormat] = &[
        TextureFormat::Bc7RgbaUnorm,
        TextureFormat::Bc3RgbaUnorm,
        TextureFormat::Bc1RgbaUnorm,
    ];

    /// ASTC compressed texture formats (Apple/Mobile).
    ///
    /// These formats require the `TEXTURE_COMPRESSION_ASTC` feature.
    /// Listed from smallest block (highest quality) to largest (smallest size).
    pub const COMPRESSED_ASTC: &[TextureFormat] = &[
        // 4x4 block - highest quality
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B4x4,
            channel: wgpu::AstcChannel::Unorm,
        },
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B4x4,
            channel: wgpu::AstcChannel::UnormSrgb,
        },
        // 5x5 block
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B5x5,
            channel: wgpu::AstcChannel::Unorm,
        },
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B5x5,
            channel: wgpu::AstcChannel::UnormSrgb,
        },
        // 6x6 block
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B6x6,
            channel: wgpu::AstcChannel::Unorm,
        },
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B6x6,
            channel: wgpu::AstcChannel::UnormSrgb,
        },
        // 8x8 block - smaller size
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x8,
            channel: wgpu::AstcChannel::Unorm,
        },
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x8,
            channel: wgpu::AstcChannel::UnormSrgb,
        },
    ];

    /// ASTC formats for sRGB color textures.
    pub const COMPRESSED_ASTC_SRGB: &[TextureFormat] = &[
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B4x4,
            channel: wgpu::AstcChannel::UnormSrgb,
        },
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B5x5,
            channel: wgpu::AstcChannel::UnormSrgb,
        },
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B6x6,
            channel: wgpu::AstcChannel::UnormSrgb,
        },
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x8,
            channel: wgpu::AstcChannel::UnormSrgb,
        },
    ];

    /// ASTC formats for linear color textures.
    pub const COMPRESSED_ASTC_LINEAR: &[TextureFormat] = &[
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B4x4,
            channel: wgpu::AstcChannel::Unorm,
        },
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B5x5,
            channel: wgpu::AstcChannel::Unorm,
        },
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B6x6,
            channel: wgpu::AstcChannel::Unorm,
        },
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B8x8,
            channel: wgpu::AstcChannel::Unorm,
        },
    ];

    /// ETC2 compressed texture formats (mobile fallback).
    ///
    /// These formats require the `TEXTURE_COMPRESSION_ETC2` feature.
    /// Used as fallback on older mobile devices without ASTC support.
    pub const COMPRESSED_ETC2: &[TextureFormat] = &[
        // RGBA
        TextureFormat::Etc2Rgba8Unorm,
        TextureFormat::Etc2Rgba8UnormSrgb,
        // RGB
        TextureFormat::Etc2Rgb8Unorm,
        TextureFormat::Etc2Rgb8UnormSrgb,
        // RGB with punch-through alpha
        TextureFormat::Etc2Rgb8A1Unorm,
        TextureFormat::Etc2Rgb8A1UnormSrgb,
        // Single channel
        TextureFormat::EacR11Unorm,
        TextureFormat::EacR11Snorm,
        // Two channel (normal maps)
        TextureFormat::EacRg11Unorm,
        TextureFormat::EacRg11Snorm,
    ];

    /// ETC2 formats for sRGB color textures.
    pub const COMPRESSED_ETC2_SRGB: &[TextureFormat] = &[
        TextureFormat::Etc2Rgba8UnormSrgb,
        TextureFormat::Etc2Rgb8UnormSrgb,
        TextureFormat::Etc2Rgb8A1UnormSrgb,
    ];

    /// ETC2 formats for linear color textures.
    pub const COMPRESSED_ETC2_LINEAR: &[TextureFormat] = &[
        TextureFormat::Etc2Rgba8Unorm,
        TextureFormat::Etc2Rgb8Unorm,
        TextureFormat::Etc2Rgb8A1Unorm,
    ];

    /// Normal map texture formats.
    ///
    /// Ordered by quality/precision. Signed formats preserve negative values
    /// for normal vectors centered around zero.
    pub const NORMAL_MAP_FORMATS: &[TextureFormat] = &[
        // High precision signed (best for tangent-space normals)
        TextureFormat::Rg16Snorm,
        // Compressed signed (good quality, small size)
        TextureFormat::Bc5RgSnorm,
        // Standard signed RGBA
        TextureFormat::Rgba8Snorm,
        // Unsigned fallbacks (require [0,1] to [-1,1] remapping in shader)
        TextureFormat::Rg16Unorm,
        TextureFormat::Rgba8Unorm,
    ];

    /// Compressed normal map formats.
    pub const NORMAL_MAP_COMPRESSED: &[TextureFormat] = &[
        // BC5 for desktop
        TextureFormat::Bc5RgSnorm,
        TextureFormat::Bc5RgUnorm,
        // ETC2 RG for mobile
        TextureFormat::EacRg11Snorm,
        TextureFormat::EacRg11Unorm,
    ];

    /// Single-channel texture formats (heightmaps, masks, etc).
    pub const SINGLE_CHANNEL_FORMATS: &[TextureFormat] = &[
        TextureFormat::R16Float,
        TextureFormat::R32Float,
        TextureFormat::R16Unorm,
        TextureFormat::R8Unorm,
        TextureFormat::Bc4RUnorm,
        TextureFormat::EacR11Unorm,
    ];

    /// Two-channel texture formats (normal XY, flow maps, etc).
    pub const TWO_CHANNEL_FORMATS: &[TextureFormat] = &[
        TextureFormat::Rg16Snorm,
        TextureFormat::Rg16Float,
        TextureFormat::Rg16Unorm,
        TextureFormat::Rg8Snorm,
        TextureFormat::Rg8Unorm,
    ];
}

// ============================================================================
// Platform Detection
// ============================================================================

/// Identifies the platform category for format selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// Windows or Linux with Vulkan or DirectX backend.
    Desktop,
    /// macOS or iOS with Metal backend.
    Apple,
    /// Android or other mobile with Vulkan/GLES.
    Mobile,
    /// WebGPU in browser.
    Web,
    /// Unknown or unidentified platform.
    Unknown,
}

impl Platform {
    /// Determines the platform from the wgpu backend.
    pub fn from_backend(backend: Backend) -> Self {
        match backend {
            Backend::Vulkan => {
                // Vulkan can be desktop or mobile; assume desktop for now
                // A more sophisticated check would query the device type
                Platform::Desktop
            }
            Backend::Dx12 => Platform::Desktop,
            Backend::Metal => Platform::Apple,
            Backend::Gl => Platform::Mobile, // OpenGL ES is typically mobile
            Backend::BrowserWebGpu => Platform::Web,
            _ => Platform::Unknown,
        }
    }

    /// Returns true if this is an Apple platform (Metal backend).
    pub fn is_apple(&self) -> bool {
        matches!(self, Platform::Apple)
    }

    /// Returns true if this is a desktop platform (Windows/Linux).
    pub fn is_desktop(&self) -> bool {
        matches!(self, Platform::Desktop)
    }

    /// Returns true if this is a mobile platform.
    pub fn is_mobile(&self) -> bool {
        matches!(self, Platform::Mobile)
    }

    /// Returns true if this is a web platform (WebGPU).
    pub fn is_web(&self) -> bool {
        matches!(self, Platform::Web)
    }
}

// ============================================================================
// TextureFormatSelector
// ============================================================================

/// Intelligent texture format selector with platform awareness.
///
/// This struct captures the adapter's backend and feature set at creation,
/// then provides methods to select optimal texture formats for each use case.
///
/// # Platform-Specific Behavior
///
/// - **Windows/Linux (Vulkan/DX12)**: Prefers sRGB formats, BC compression
/// - **macOS/iOS (Metal)**: Prefers BGRA for swapchain compatibility, ASTC compression
/// - **Mobile**: Prefers ASTC or ETC2 compression
/// - **WebGPU**: Conservative format selection with fallbacks
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::texture_formats::TextureFormatSelector;
///
/// # fn example(adapter: &wgpu::Adapter) {
/// let selector = TextureFormatSelector::new(adapter);
///
/// // For color render targets
/// let swapchain_format = selector.color_attachment(false);
///
/// // For HDR rendering
/// let hdr_format = selector.color_attachment(true);
///
/// // For shadow maps
/// let shadow_format = selector.depth(false);
///
/// // For normal maps
/// let normal_format = selector.normal_map(false);
///
/// // For compressed textures
/// let diffuse_format = selector.compressed_color(true); // sRGB
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TextureFormatSelector {
    /// The wgpu backend in use.
    backend: Backend,
    /// Available features on this device.
    features: Features,
    /// Derived platform category.
    platform: Platform,
}

impl TextureFormatSelector {
    /// Creates a new format selector from an adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The wgpu adapter to query for backend and features
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(adapter: &wgpu::Adapter) {
    /// use renderer_backend::resources::texture_formats::TextureFormatSelector;
    ///
    /// let selector = TextureFormatSelector::new(adapter);
    /// println!("Using backend: {:?}", selector.backend());
    /// # }
    /// ```
    pub fn new(adapter: &Adapter) -> Self {
        let info = adapter.get_info();
        let backend = info.backend;
        let features = adapter.features();
        let platform = Platform::from_backend(backend);

        Self {
            backend,
            features,
            platform,
        }
    }

    /// Creates a format selector with explicit backend and features.
    ///
    /// Useful for testing or when adapter is not available.
    ///
    /// # Arguments
    ///
    /// * `backend` - The wgpu backend
    /// * `features` - The device features
    pub fn with_backend_features(backend: Backend, features: Features) -> Self {
        let platform = Platform::from_backend(backend);
        Self {
            backend,
            features,
            platform,
        }
    }

    /// Returns the wgpu backend in use.
    #[inline]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Returns the device features.
    #[inline]
    pub fn features(&self) -> Features {
        self.features
    }

    /// Returns the detected platform.
    #[inline]
    pub fn platform(&self) -> Platform {
        self.platform
    }

    /// Checks if a specific texture format is supported.
    ///
    /// This checks whether the required feature for the format is available.
    /// Note: This does not query the device for runtime support, just feature flags.
    ///
    /// # Arguments
    ///
    /// * `format` - The texture format to check
    ///
    /// # Returns
    ///
    /// `true` if the format should be supported based on features.
    pub fn supports(&self, format: TextureFormat) -> bool {
        match format {
            // BC formats require TEXTURE_COMPRESSION_BC
            TextureFormat::Bc1RgbaUnorm
            | TextureFormat::Bc1RgbaUnormSrgb
            | TextureFormat::Bc2RgbaUnorm
            | TextureFormat::Bc2RgbaUnormSrgb
            | TextureFormat::Bc3RgbaUnorm
            | TextureFormat::Bc3RgbaUnormSrgb
            | TextureFormat::Bc4RUnorm
            | TextureFormat::Bc4RSnorm
            | TextureFormat::Bc5RgUnorm
            | TextureFormat::Bc5RgSnorm
            | TextureFormat::Bc6hRgbUfloat
            | TextureFormat::Bc6hRgbFloat
            | TextureFormat::Bc7RgbaUnorm
            | TextureFormat::Bc7RgbaUnormSrgb => {
                self.features.contains(Features::TEXTURE_COMPRESSION_BC)
            }

            // ETC2 formats require TEXTURE_COMPRESSION_ETC2
            TextureFormat::Etc2Rgb8Unorm
            | TextureFormat::Etc2Rgb8UnormSrgb
            | TextureFormat::Etc2Rgb8A1Unorm
            | TextureFormat::Etc2Rgb8A1UnormSrgb
            | TextureFormat::Etc2Rgba8Unorm
            | TextureFormat::Etc2Rgba8UnormSrgb
            | TextureFormat::EacR11Unorm
            | TextureFormat::EacR11Snorm
            | TextureFormat::EacRg11Unorm
            | TextureFormat::EacRg11Snorm => {
                self.features.contains(Features::TEXTURE_COMPRESSION_ETC2)
            }

            // ASTC formats require TEXTURE_COMPRESSION_ASTC
            TextureFormat::Astc { .. } => {
                self.features.contains(Features::TEXTURE_COMPRESSION_ASTC)
            }

            // 16-bit normalized formats may require specific features
            TextureFormat::Rg16Snorm | TextureFormat::Rg16Unorm => {
                // These are widely supported, but check for normalization support
                // wgpu 22 should support these on most backends
                true
            }

            // Depth32FloatStencil8 may require a feature on some backends
            TextureFormat::Depth32FloatStencil8 => {
                // Generally supported, but some older hardware may not
                true
            }

            // All other standard formats are assumed supported
            _ => true,
        }
    }

    /// Returns true if BC texture compression is available.
    #[inline]
    pub fn has_bc_compression(&self) -> bool {
        self.features.contains(Features::TEXTURE_COMPRESSION_BC)
    }

    /// Returns true if ASTC texture compression is available.
    #[inline]
    pub fn has_astc_compression(&self) -> bool {
        self.features.contains(Features::TEXTURE_COMPRESSION_ASTC)
    }

    /// Returns true if ETC2 texture compression is available.
    #[inline]
    pub fn has_etc2_compression(&self) -> bool {
        self.features.contains(Features::TEXTURE_COMPRESSION_ETC2)
    }

    // =========================================================================
    // Format Selection Methods
    // =========================================================================

    /// Selects the optimal color attachment format.
    ///
    /// # Arguments
    ///
    /// * `hdr` - If true, returns an HDR-capable format (Rgba16Float)
    ///
    /// # Returns
    ///
    /// - **HDR**: `Rgba16Float` for wide gamut rendering
    /// - **Windows/Linux**: `Rgba8UnormSrgb` for sRGB color-correct rendering
    /// - **macOS/iOS (Metal)**: `Bgra8Unorm` for Metal swapchain compatibility
    /// - **Mobile/Web**: `Bgra8Unorm` (widely compatible)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(selector: &renderer_backend::resources::texture_formats::TextureFormatSelector) {
    /// // Standard color attachment
    /// let color_fmt = selector.color_attachment(false);
    ///
    /// // HDR render target
    /// let hdr_fmt = selector.color_attachment(true);
    /// # }
    /// ```
    pub fn color_attachment(&self, hdr: bool) -> TextureFormat {
        if hdr {
            // HDR always uses Rgba16Float for wide gamut
            TextureFormat::Rgba16Float
        } else {
            match self.platform {
                Platform::Desktop => {
                    // Windows/Linux prefer sRGB for color-correct rendering
                    TextureFormat::Rgba8UnormSrgb
                }
                Platform::Apple => {
                    // Metal prefers BGRA for native swapchain format
                    // Using linear here; sRGB conversion happens in shader or via view
                    TextureFormat::Bgra8Unorm
                }
                Platform::Mobile | Platform::Web | Platform::Unknown => {
                    // BGRA is widely compatible across platforms
                    TextureFormat::Bgra8Unorm
                }
            }
        }
    }

    /// Selects the optimal depth texture format.
    ///
    /// # Arguments
    ///
    /// * `with_stencil` - If true, returns a depth-stencil combined format
    ///
    /// # Returns
    ///
    /// - **With stencil**: `Depth24PlusStencil8` (or `Depth32FloatStencil8` if available)
    /// - **Without stencil**: `Depth32Float` for maximum precision
    /// - **Fallback**: `Depth24Plus` if Depth32Float is unavailable
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(selector: &renderer_backend::resources::texture_formats::TextureFormatSelector) {
    /// // Depth-only for shadow maps
    /// let shadow_depth = selector.depth(false);
    ///
    /// // Depth-stencil for deferred rendering
    /// let gbuffer_depth = selector.depth(true);
    /// # }
    /// ```
    pub fn depth(&self, with_stencil: bool) -> TextureFormat {
        if with_stencil {
            // Prefer Depth24PlusStencil8 for compatibility
            // Depth32FloatStencil8 has better precision but may not be available
            TextureFormat::Depth24PlusStencil8
        } else {
            // Depth32Float for maximum precision (shadow maps, etc.)
            TextureFormat::Depth32Float
        }
    }

    /// Returns the fallback depth format if the primary is unavailable.
    ///
    /// # Arguments
    ///
    /// * `with_stencil` - If true, returns a depth-stencil fallback
    ///
    /// # Returns
    ///
    /// A fallback depth format with potentially lower precision.
    pub fn depth_fallback(&self, with_stencil: bool) -> TextureFormat {
        if with_stencil {
            // Only fallback for depth-stencil is Depth24PlusStencil8
            TextureFormat::Depth24PlusStencil8
        } else {
            // Fallback from Depth32Float to Depth24Plus
            TextureFormat::Depth24Plus
        }
    }

    /// Selects the optimal normal map texture format.
    ///
    /// # Arguments
    ///
    /// * `compressed` - If true, prefers compressed format (BC5/ETC2)
    ///
    /// # Returns
    ///
    /// - **Compressed (desktop)**: `Bc5RgSnorm` for signed normal XY
    /// - **Compressed (mobile)**: `EacRg11Snorm` for signed normal XY
    /// - **Uncompressed (high precision)**: `Rg16Snorm` for signed 16-bit normals
    /// - **Fallback**: `Rgba8Snorm` for signed 8-bit normals
    ///
    /// Normal maps typically only need RG channels (XY), with Z reconstructed
    /// in the shader: `Z = sqrt(1 - X*X - Y*Y)`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(selector: &renderer_backend::resources::texture_formats::TextureFormatSelector) {
    /// // High quality uncompressed
    /// let hq_normal = selector.normal_map(false);
    ///
    /// // Compressed for production
    /// let compressed_normal = selector.normal_map(true);
    /// # }
    /// ```
    pub fn normal_map(&self, compressed: bool) -> TextureFormat {
        if compressed {
            // Try BC5 first (desktop), then ETC2 RG (mobile)
            if self.has_bc_compression() {
                TextureFormat::Bc5RgSnorm
            } else if self.has_etc2_compression() {
                TextureFormat::EacRg11Snorm
            } else {
                // No compression available, use uncompressed
                TextureFormat::Rgba8Snorm
            }
        } else {
            // High precision uncompressed
            // Rg16Snorm provides 16-bit precision for tangent-space normals
            if self.supports(TextureFormat::Rg16Snorm) {
                TextureFormat::Rg16Snorm
            } else {
                // Fallback to 8-bit signed
                TextureFormat::Rgba8Snorm
            }
        }
    }

    /// Selects the optimal compressed color texture format.
    ///
    /// # Arguments
    ///
    /// * `srgb` - If true, returns an sRGB format for color textures
    ///
    /// # Returns
    ///
    /// - **Windows/Linux (BC)**: `Bc7RgbaUnormSrgb` or `Bc7RgbaUnorm`
    /// - **macOS/iOS (ASTC)**: `Astc4x4UnormSrgb` or `Astc4x4Unorm`
    /// - **Mobile (ASTC/ETC2)**: ASTC if available, else ETC2
    /// - **No compression**: `Rgba8UnormSrgb` or `Rgba8Unorm`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn example(selector: &renderer_backend::resources::texture_formats::TextureFormatSelector) {
    /// // Compressed diffuse texture (sRGB)
    /// let diffuse_fmt = selector.compressed_color(true);
    ///
    /// // Compressed data texture (linear)
    /// let data_fmt = selector.compressed_color(false);
    /// # }
    /// ```
    pub fn compressed_color(&self, srgb: bool) -> TextureFormat {
        // Priority: BC (desktop) > ASTC (mobile/Apple) > ETC2 (fallback) > uncompressed
        if self.has_bc_compression() {
            if srgb {
                TextureFormat::Bc7RgbaUnormSrgb
            } else {
                TextureFormat::Bc7RgbaUnorm
            }
        } else if self.has_astc_compression() {
            if srgb {
                TextureFormat::Astc {
                    block: wgpu::AstcBlock::B4x4,
                    channel: wgpu::AstcChannel::UnormSrgb,
                }
            } else {
                TextureFormat::Astc {
                    block: wgpu::AstcBlock::B4x4,
                    channel: wgpu::AstcChannel::Unorm,
                }
            }
        } else if self.has_etc2_compression() {
            if srgb {
                TextureFormat::Etc2Rgba8UnormSrgb
            } else {
                TextureFormat::Etc2Rgba8Unorm
            }
        } else {
            // No compression available
            if srgb {
                TextureFormat::Rgba8UnormSrgb
            } else {
                TextureFormat::Rgba8Unorm
            }
        }
    }

    /// Returns the best available compression scheme name.
    ///
    /// Useful for logging and debugging.
    pub fn compression_scheme(&self) -> &'static str {
        if self.has_bc_compression() {
            "BC"
        } else if self.has_astc_compression() {
            "ASTC"
        } else if self.has_etc2_compression() {
            "ETC2"
        } else {
            "None"
        }
    }

    /// Returns the optimal swapchain surface format for this platform.
    ///
    /// This returns the format that should match the surface's preferred format
    /// for optimal performance without format conversion.
    ///
    /// # Returns
    ///
    /// - **Metal**: `Bgra8UnormSrgb` (Metal's native swapchain format)
    /// - **Vulkan/DX12**: `Bgra8UnormSrgb` (widely supported)
    /// - **WebGPU**: `Bgra8UnormSrgb` (standard for web)
    pub fn swapchain_format(&self) -> TextureFormat {
        // BGRA8 sRGB is the most universally supported swapchain format
        TextureFormat::Bgra8UnormSrgb
    }

    /// Returns all supported color formats for this platform.
    ///
    /// Useful for format enumeration or fallback iteration.
    pub fn supported_color_formats(&self) -> Vec<TextureFormat> {
        format_tables::COLOR_FORMATS
            .iter()
            .copied()
            .filter(|f| self.supports(*f))
            .collect()
    }

    /// Returns all supported compressed formats for this platform.
    pub fn supported_compressed_formats(&self) -> Vec<TextureFormat> {
        let mut formats = Vec::new();

        if self.has_bc_compression() {
            formats.extend(format_tables::COMPRESSED_BC.iter().copied());
        }
        if self.has_astc_compression() {
            formats.extend(format_tables::COMPRESSED_ASTC.iter().copied());
        }
        if self.has_etc2_compression() {
            formats.extend(format_tables::COMPRESSED_ETC2.iter().copied());
        }

        formats
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a selector with specific features
    fn selector_with_features(backend: Backend, features: Features) -> TextureFormatSelector {
        TextureFormatSelector::with_backend_features(backend, features)
    }

    // -------------------------------------------------------------------------
    // Platform detection tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_platform_from_backend() {
        assert_eq!(Platform::from_backend(Backend::Vulkan), Platform::Desktop);
        assert_eq!(Platform::from_backend(Backend::Dx12), Platform::Desktop);
        assert_eq!(Platform::from_backend(Backend::Metal), Platform::Apple);
        assert_eq!(Platform::from_backend(Backend::Gl), Platform::Mobile);
        assert_eq!(
            Platform::from_backend(Backend::BrowserWebGpu),
            Platform::Web
        );
    }

    #[test]
    fn test_platform_is_methods() {
        assert!(Platform::Desktop.is_desktop());
        assert!(!Platform::Desktop.is_apple());

        assert!(Platform::Apple.is_apple());
        assert!(!Platform::Apple.is_desktop());

        assert!(Platform::Mobile.is_mobile());
        assert!(Platform::Web.is_web());
    }

    // -------------------------------------------------------------------------
    // Color attachment format tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_color_attachment_desktop_sdr() {
        let selector = selector_with_features(Backend::Vulkan, Features::empty());
        assert_eq!(
            selector.color_attachment(false),
            TextureFormat::Rgba8UnormSrgb
        );
    }

    #[test]
    fn test_color_attachment_metal_sdr() {
        let selector = selector_with_features(Backend::Metal, Features::empty());
        assert_eq!(
            selector.color_attachment(false),
            TextureFormat::Bgra8Unorm
        );
    }

    #[test]
    fn test_color_attachment_hdr() {
        let selector = selector_with_features(Backend::Vulkan, Features::empty());
        assert_eq!(selector.color_attachment(true), TextureFormat::Rgba16Float);

        // HDR should be the same on Metal
        let metal_selector = selector_with_features(Backend::Metal, Features::empty());
        assert_eq!(
            metal_selector.color_attachment(true),
            TextureFormat::Rgba16Float
        );
    }

    #[test]
    fn test_color_attachment_web() {
        let selector = selector_with_features(Backend::BrowserWebGpu, Features::empty());
        assert_eq!(
            selector.color_attachment(false),
            TextureFormat::Bgra8Unorm
        );
    }

    // -------------------------------------------------------------------------
    // Depth format tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_depth_without_stencil() {
        let selector = selector_with_features(Backend::Vulkan, Features::empty());
        assert_eq!(selector.depth(false), TextureFormat::Depth32Float);
    }

    #[test]
    fn test_depth_with_stencil() {
        let selector = selector_with_features(Backend::Vulkan, Features::empty());
        assert_eq!(selector.depth(true), TextureFormat::Depth24PlusStencil8);
    }

    #[test]
    fn test_depth_fallback() {
        let selector = selector_with_features(Backend::Vulkan, Features::empty());
        assert_eq!(selector.depth_fallback(false), TextureFormat::Depth24Plus);
        assert_eq!(
            selector.depth_fallback(true),
            TextureFormat::Depth24PlusStencil8
        );
    }

    // -------------------------------------------------------------------------
    // Normal map format tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_normal_map_uncompressed() {
        let selector = selector_with_features(Backend::Vulkan, Features::empty());
        assert_eq!(selector.normal_map(false), TextureFormat::Rg16Snorm);
    }

    #[test]
    fn test_normal_map_compressed_bc() {
        let selector =
            selector_with_features(Backend::Vulkan, Features::TEXTURE_COMPRESSION_BC);
        assert_eq!(selector.normal_map(true), TextureFormat::Bc5RgSnorm);
    }

    #[test]
    fn test_normal_map_compressed_etc2() {
        let selector =
            selector_with_features(Backend::Gl, Features::TEXTURE_COMPRESSION_ETC2);
        assert_eq!(selector.normal_map(true), TextureFormat::EacRg11Snorm);
    }

    #[test]
    fn test_normal_map_no_compression() {
        let selector = selector_with_features(Backend::Vulkan, Features::empty());
        // Without compression features, should fall back to uncompressed
        assert_eq!(selector.normal_map(true), TextureFormat::Rgba8Snorm);
    }

    // -------------------------------------------------------------------------
    // Compressed color format tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compressed_color_bc_srgb() {
        let selector =
            selector_with_features(Backend::Vulkan, Features::TEXTURE_COMPRESSION_BC);
        assert_eq!(
            selector.compressed_color(true),
            TextureFormat::Bc7RgbaUnormSrgb
        );
    }

    #[test]
    fn test_compressed_color_bc_linear() {
        let selector =
            selector_with_features(Backend::Vulkan, Features::TEXTURE_COMPRESSION_BC);
        assert_eq!(
            selector.compressed_color(false),
            TextureFormat::Bc7RgbaUnorm
        );
    }

    #[test]
    fn test_compressed_color_astc_srgb() {
        let selector =
            selector_with_features(Backend::Metal, Features::TEXTURE_COMPRESSION_ASTC);
        assert_eq!(
            selector.compressed_color(true),
            TextureFormat::Astc {
                block: wgpu::AstcBlock::B4x4,
                channel: wgpu::AstcChannel::UnormSrgb,
            }
        );
    }

    #[test]
    fn test_compressed_color_astc_linear() {
        let selector =
            selector_with_features(Backend::Metal, Features::TEXTURE_COMPRESSION_ASTC);
        assert_eq!(
            selector.compressed_color(false),
            TextureFormat::Astc {
                block: wgpu::AstcBlock::B4x4,
                channel: wgpu::AstcChannel::Unorm,
            }
        );
    }

    #[test]
    fn test_compressed_color_etc2() {
        let selector =
            selector_with_features(Backend::Gl, Features::TEXTURE_COMPRESSION_ETC2);
        assert_eq!(
            selector.compressed_color(true),
            TextureFormat::Etc2Rgba8UnormSrgb
        );
        assert_eq!(
            selector.compressed_color(false),
            TextureFormat::Etc2Rgba8Unorm
        );
    }

    #[test]
    fn test_compressed_color_no_compression() {
        let selector = selector_with_features(Backend::BrowserWebGpu, Features::empty());
        assert_eq!(
            selector.compressed_color(true),
            TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(
            selector.compressed_color(false),
            TextureFormat::Rgba8Unorm
        );
    }

    // -------------------------------------------------------------------------
    // Feature detection tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_has_compression_features() {
        let bc_selector =
            selector_with_features(Backend::Vulkan, Features::TEXTURE_COMPRESSION_BC);
        assert!(bc_selector.has_bc_compression());
        assert!(!bc_selector.has_astc_compression());
        assert!(!bc_selector.has_etc2_compression());

        let astc_selector =
            selector_with_features(Backend::Metal, Features::TEXTURE_COMPRESSION_ASTC);
        assert!(!astc_selector.has_bc_compression());
        assert!(astc_selector.has_astc_compression());
        assert!(!astc_selector.has_etc2_compression());

        let etc2_selector =
            selector_with_features(Backend::Gl, Features::TEXTURE_COMPRESSION_ETC2);
        assert!(!etc2_selector.has_bc_compression());
        assert!(!etc2_selector.has_astc_compression());
        assert!(etc2_selector.has_etc2_compression());
    }

    #[test]
    fn test_supports_format() {
        let bc_selector =
            selector_with_features(Backend::Vulkan, Features::TEXTURE_COMPRESSION_BC);
        assert!(bc_selector.supports(TextureFormat::Bc7RgbaUnorm));
        assert!(bc_selector.supports(TextureFormat::Bc5RgSnorm));
        assert!(!bc_selector.supports(TextureFormat::Etc2Rgba8Unorm));

        let no_compression = selector_with_features(Backend::Vulkan, Features::empty());
        assert!(!no_compression.supports(TextureFormat::Bc7RgbaUnorm));
        assert!(no_compression.supports(TextureFormat::Rgba8Unorm));
    }

    // -------------------------------------------------------------------------
    // Compression scheme tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compression_scheme() {
        assert_eq!(
            selector_with_features(Backend::Vulkan, Features::TEXTURE_COMPRESSION_BC)
                .compression_scheme(),
            "BC"
        );
        assert_eq!(
            selector_with_features(Backend::Metal, Features::TEXTURE_COMPRESSION_ASTC)
                .compression_scheme(),
            "ASTC"
        );
        assert_eq!(
            selector_with_features(Backend::Gl, Features::TEXTURE_COMPRESSION_ETC2)
                .compression_scheme(),
            "ETC2"
        );
        assert_eq!(
            selector_with_features(Backend::BrowserWebGpu, Features::empty())
                .compression_scheme(),
            "None"
        );
    }

    // -------------------------------------------------------------------------
    // Swapchain format tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_swapchain_format() {
        let vulkan = selector_with_features(Backend::Vulkan, Features::empty());
        assert_eq!(vulkan.swapchain_format(), TextureFormat::Bgra8UnormSrgb);

        let metal = selector_with_features(Backend::Metal, Features::empty());
        assert_eq!(metal.swapchain_format(), TextureFormat::Bgra8UnormSrgb);

        let webgpu = selector_with_features(Backend::BrowserWebGpu, Features::empty());
        assert_eq!(webgpu.swapchain_format(), TextureFormat::Bgra8UnormSrgb);
    }

    // -------------------------------------------------------------------------
    // Format table tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_format_tables_not_empty() {
        assert!(!format_tables::COLOR_FORMATS.is_empty());
        assert!(!format_tables::HDR_COLOR_FORMATS.is_empty());
        assert!(!format_tables::DEPTH_FORMATS.is_empty());
        assert!(!format_tables::DEPTH_STENCIL_FORMATS.is_empty());
        assert!(!format_tables::COMPRESSED_BC.is_empty());
        assert!(!format_tables::COMPRESSED_ASTC.is_empty());
        assert!(!format_tables::COMPRESSED_ETC2.is_empty());
        assert!(!format_tables::NORMAL_MAP_FORMATS.is_empty());
        assert!(!format_tables::SINGLE_CHANNEL_FORMATS.is_empty());
        assert!(!format_tables::TWO_CHANNEL_FORMATS.is_empty());
    }

    #[test]
    fn test_format_tables_color() {
        // Verify color formats include key formats
        assert!(format_tables::COLOR_FORMATS.contains(&TextureFormat::Rgba8UnormSrgb));
        assert!(format_tables::COLOR_FORMATS.contains(&TextureFormat::Bgra8Unorm));
        assert!(format_tables::COLOR_FORMATS.contains(&TextureFormat::Rgba16Float));
    }

    #[test]
    fn test_format_tables_depth() {
        // Verify depth formats include key formats
        assert!(format_tables::DEPTH_FORMATS.contains(&TextureFormat::Depth32Float));
        assert!(format_tables::DEPTH_FORMATS.contains(&TextureFormat::Depth24Plus));

        assert!(format_tables::DEPTH_STENCIL_FORMATS
            .contains(&TextureFormat::Depth24PlusStencil8));
    }

    #[test]
    fn test_format_tables_normal_map() {
        // Verify normal map formats include signed formats
        assert!(format_tables::NORMAL_MAP_FORMATS.contains(&TextureFormat::Rg16Snorm));
        assert!(format_tables::NORMAL_MAP_FORMATS.contains(&TextureFormat::Rgba8Snorm));
    }

    // -------------------------------------------------------------------------
    // Supported formats enumeration tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_supported_color_formats() {
        let selector = selector_with_features(Backend::Vulkan, Features::empty());
        let formats = selector.supported_color_formats();
        assert!(!formats.is_empty());
        assert!(formats.contains(&TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn test_supported_compressed_formats_bc() {
        let selector =
            selector_with_features(Backend::Vulkan, Features::TEXTURE_COMPRESSION_BC);
        let formats = selector.supported_compressed_formats();
        assert!(!formats.is_empty());
        assert!(formats.contains(&TextureFormat::Bc7RgbaUnorm));
    }

    #[test]
    fn test_supported_compressed_formats_none() {
        let selector = selector_with_features(Backend::BrowserWebGpu, Features::empty());
        let formats = selector.supported_compressed_formats();
        assert!(formats.is_empty());
    }

    // -------------------------------------------------------------------------
    // Backend and platform accessor tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_backend_accessor() {
        let selector = selector_with_features(Backend::Metal, Features::empty());
        assert_eq!(selector.backend(), Backend::Metal);
    }

    #[test]
    fn test_features_accessor() {
        let features = Features::TEXTURE_COMPRESSION_BC | Features::TEXTURE_COMPRESSION_ASTC;
        let selector = selector_with_features(Backend::Vulkan, features);
        assert!(selector.features().contains(Features::TEXTURE_COMPRESSION_BC));
        assert!(selector.features().contains(Features::TEXTURE_COMPRESSION_ASTC));
    }

    #[test]
    fn test_platform_accessor() {
        let vulkan_selector = selector_with_features(Backend::Vulkan, Features::empty());
        assert_eq!(vulkan_selector.platform(), Platform::Desktop);

        let metal_selector = selector_with_features(Backend::Metal, Features::empty());
        assert_eq!(metal_selector.platform(), Platform::Apple);
    }
}
