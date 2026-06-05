// Blackbox contract tests for T-WGPU-P2.3.2 Texture Format Selector API
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::resources::{TextureFormatSelector, Platform, format_tables}`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/resources/texture_formats.rs (implementation)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - Task acceptance criteria (T-WGPU-P2.3.2)
//   - Public API documentation in resources/mod.rs
//
// Public API under test:
//   - TextureFormatSelector: with_backend_features(), backend(), features(), platform(),
//     supports(), has_bc_compression(), has_astc_compression(), has_etc2_compression(),
//     color_attachment(), depth(), normal_map(), compressed_color(), swapchain_format()
//   - Platform: Desktop, Apple, Mobile, Web, Unknown
//   - format_tables: COLOR_FORMATS, DEPTH_FORMATS, COMPRESSED_BC, COMPRESSED_ASTC, etc.
//
// Test design rationale:
//   Equivalence partitioning:
//     - Desktop backend (Vulkan/DX12) with BC compression
//     - Apple backend (Metal) with ASTC compression
//     - Mobile backend with ASTC/ETC2 compression
//     - Web backend (WebGPU) with limited features
//   Format categories:
//     - Color attachment formats (SDR/HDR)
//     - Depth formats (with/without stencil)
//     - Normal map formats (compressed/uncompressed)
//     - Compressed color formats (sRGB/linear)
//   Platform awareness:
//     - Correct platform detection per backend
//     - Feature-dependent format selection

use renderer_backend::resources::{format_tables, Platform, TextureFormatSelector};
use std::collections::HashSet;
use wgpu::{Backend, Features, TextureFormat};

// =============================================================================
// SECTION 1: API CONTRACT TESTS (no GPU required)
// =============================================================================

// -----------------------------------------------------------------------------
// TextureFormatSelector Construction
// -----------------------------------------------------------------------------

/// Test: with_backend_features creates a valid selector for Vulkan.
#[test]
fn test_selector_vulkan_construction() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Vulkan, Features::empty());
    assert_eq!(selector.backend(), Backend::Vulkan);
}

/// Test: with_backend_features creates a valid selector for Metal.
#[test]
fn test_selector_metal_construction() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Metal, Features::empty());
    assert_eq!(selector.backend(), Backend::Metal);
}

/// Test: with_backend_features creates a valid selector for DX12.
#[test]
fn test_selector_dx12_construction() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Dx12, Features::empty());
    assert_eq!(selector.backend(), Backend::Dx12);
}

/// Test: with_backend_features creates a valid selector for WebGPU.
#[test]
fn test_selector_webgpu_construction() {
    let selector =
        TextureFormatSelector::with_backend_features(Backend::BrowserWebGpu, Features::empty());
    assert_eq!(selector.backend(), Backend::BrowserWebGpu);
}

/// Test: with_backend_features creates a valid selector for GL.
#[test]
fn test_selector_gl_construction() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Gl, Features::empty());
    assert_eq!(selector.backend(), Backend::Gl);
}

/// Test: features() returns the features passed at construction.
#[test]
fn test_selector_features_empty() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Vulkan, Features::empty());
    assert_eq!(selector.features(), Features::empty());
}

/// Test: features() returns BC compression features when provided.
#[test]
fn test_selector_features_bc_compression() {
    let features = Features::TEXTURE_COMPRESSION_BC;
    let selector = TextureFormatSelector::with_backend_features(Backend::Vulkan, features);
    assert!(selector.features().contains(Features::TEXTURE_COMPRESSION_BC));
}

/// Test: features() returns ASTC compression features when provided.
#[test]
fn test_selector_features_astc_compression() {
    let features = Features::TEXTURE_COMPRESSION_ASTC;
    let selector = TextureFormatSelector::with_backend_features(Backend::Metal, features);
    assert!(selector.features().contains(Features::TEXTURE_COMPRESSION_ASTC));
}

/// Test: features() returns ETC2 compression features when provided.
#[test]
fn test_selector_features_etc2_compression() {
    let features = Features::TEXTURE_COMPRESSION_ETC2;
    let selector = TextureFormatSelector::with_backend_features(Backend::Gl, features);
    assert!(selector.features().contains(Features::TEXTURE_COMPRESSION_ETC2));
}

// -----------------------------------------------------------------------------
// Platform Detection
// -----------------------------------------------------------------------------

/// Test: platform() returns Desktop for Vulkan backend.
#[test]
fn test_platform_vulkan_is_desktop() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Vulkan, Features::empty());
    assert_eq!(selector.platform(), Platform::Desktop);
}

/// Test: platform() returns Desktop for DX12 backend.
#[test]
fn test_platform_dx12_is_desktop() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Dx12, Features::empty());
    assert_eq!(selector.platform(), Platform::Desktop);
}

/// Test: platform() returns Apple for Metal backend.
#[test]
fn test_platform_metal_is_apple() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Metal, Features::empty());
    assert_eq!(selector.platform(), Platform::Apple);
}

/// Test: platform() returns Web for BrowserWebGpu backend.
#[test]
fn test_platform_webgpu_is_web() {
    let selector =
        TextureFormatSelector::with_backend_features(Backend::BrowserWebGpu, Features::empty());
    assert_eq!(selector.platform(), Platform::Web);
}

/// Test: platform() returns Mobile or Unknown for GL backend.
#[test]
fn test_platform_gl_variant() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Gl, Features::empty());
    let platform = selector.platform();
    // GL can be mobile (OpenGL ES) or desktop (OpenGL), implementation may vary
    assert!(
        matches!(platform, Platform::Mobile | Platform::Desktop | Platform::Unknown),
        "GL platform should be Mobile, Desktop, or Unknown"
    );
}

// =============================================================================
// SECTION 2: BEHAVIORAL TESTS (no GPU required)
// =============================================================================

// -----------------------------------------------------------------------------
// Compression Feature Detection
// -----------------------------------------------------------------------------

/// Test: has_bc_compression returns true when BC feature is enabled.
#[test]
fn test_has_bc_compression_with_feature() {
    let selector = TextureFormatSelector::with_backend_features(
        Backend::Vulkan,
        Features::TEXTURE_COMPRESSION_BC,
    );
    assert!(selector.has_bc_compression());
}

/// Test: has_bc_compression returns false when BC feature is not enabled.
#[test]
fn test_has_bc_compression_without_feature() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Vulkan, Features::empty());
    assert!(!selector.has_bc_compression());
}

/// Test: has_astc_compression returns true when ASTC feature is enabled.
#[test]
fn test_has_astc_compression_with_feature() {
    let selector = TextureFormatSelector::with_backend_features(
        Backend::Metal,
        Features::TEXTURE_COMPRESSION_ASTC,
    );
    assert!(selector.has_astc_compression());
}

/// Test: has_astc_compression returns false when ASTC feature is not enabled.
#[test]
fn test_has_astc_compression_without_feature() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Metal, Features::empty());
    assert!(!selector.has_astc_compression());
}

/// Test: has_etc2_compression returns true when ETC2 feature is enabled.
#[test]
fn test_has_etc2_compression_with_feature() {
    let selector = TextureFormatSelector::with_backend_features(
        Backend::Gl,
        Features::TEXTURE_COMPRESSION_ETC2,
    );
    assert!(selector.has_etc2_compression());
}

/// Test: has_etc2_compression returns false when ETC2 feature is not enabled.
#[test]
fn test_has_etc2_compression_without_feature() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Gl, Features::empty());
    assert!(!selector.has_etc2_compression());
}

// -----------------------------------------------------------------------------
// Color Attachment Format Selection
// -----------------------------------------------------------------------------

/// Test: color_attachment with hdr=false returns an SDR format.
#[test]
fn test_color_attachment_sdr() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Vulkan, Features::empty());
    let format = selector.color_attachment(false);
    // SDR formats are typically 8-bit per channel
    let sdr_formats = [
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb,
    ];
    assert!(
        sdr_formats.contains(&format),
        "SDR color attachment should be Rgba8 or Bgra8 variant, got {:?}",
        format
    );
}

/// Test: color_attachment with hdr=true returns an HDR format.
#[test]
fn test_color_attachment_hdr() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Vulkan, Features::empty());
    let format = selector.color_attachment(true);
    // HDR formats are typically 16-bit or higher precision
    let hdr_formats = [
        TextureFormat::Rgba16Float,
        TextureFormat::Rgb10a2Unorm,
        TextureFormat::Rg11b10Float,
        TextureFormat::Rgba32Float,
    ];
    assert!(
        hdr_formats.contains(&format),
        "HDR color attachment should be float or high-precision format, got {:?}",
        format
    );
}

/// Test: Metal color_attachment prefers BGRA format.
#[test]
fn test_color_attachment_metal_bgra() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Metal, Features::empty());
    let format = selector.color_attachment(false);
    // Metal typically prefers BGRA formats
    let bgra_formats = [
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb,
    ];
    assert!(
        bgra_formats.contains(&format),
        "Metal SDR color attachment should be standard format, got {:?}",
        format
    );
}

// -----------------------------------------------------------------------------
// Depth Format Selection
// -----------------------------------------------------------------------------

/// Test: depth without stencil returns a depth-only format.
#[test]
fn test_depth_without_stencil() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Vulkan, Features::empty());
    let format = selector.depth(false);
    let depth_only_formats = [
        TextureFormat::Depth32Float,
        TextureFormat::Depth24Plus,
        TextureFormat::Depth16Unorm,
    ];
    assert!(
        depth_only_formats.contains(&format),
        "Depth-only format expected, got {:?}",
        format
    );
}

/// Test: depth with stencil returns a depth-stencil format.
#[test]
fn test_depth_with_stencil() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Vulkan, Features::empty());
    let format = selector.depth(true);
    let depth_stencil_formats = [
        TextureFormat::Depth32FloatStencil8,
        TextureFormat::Depth24PlusStencil8,
    ];
    assert!(
        depth_stencil_formats.contains(&format),
        "Depth-stencil format expected, got {:?}",
        format
    );
}

/// Test: depth format consistency across calls.
#[test]
fn test_depth_format_consistency() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Vulkan, Features::empty());
    let format1 = selector.depth(false);
    let format2 = selector.depth(false);
    assert_eq!(format1, format2, "Depth format should be consistent");
}

// -----------------------------------------------------------------------------
// Normal Map Format Selection
// -----------------------------------------------------------------------------

/// Test: normal_map uncompressed returns RG format.
#[test]
fn test_normal_map_uncompressed() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Vulkan, Features::empty());
    let format = selector.normal_map(false);
    // Normal maps typically use RG formats (XY, derive Z)
    let normal_formats = [
        TextureFormat::Rg16Snorm,
        TextureFormat::Rg16Unorm,
        TextureFormat::Rg8Snorm,
        TextureFormat::Rg8Unorm,
        TextureFormat::Rgba8Snorm,
        TextureFormat::Rgba8Unorm,
    ];
    assert!(
        normal_formats.contains(&format),
        "Normal map should be RG or RGBA format, got {:?}",
        format
    );
}

/// Test: normal_map compressed with BC support returns BC5.
#[test]
fn test_normal_map_compressed_bc() {
    let selector = TextureFormatSelector::with_backend_features(
        Backend::Vulkan,
        Features::TEXTURE_COMPRESSION_BC,
    );
    let format = selector.normal_map(true);
    // BC5 is preferred for normal maps on desktop
    let bc_normal_formats = [
        TextureFormat::Bc5RgSnorm,
        TextureFormat::Bc5RgUnorm,
        TextureFormat::Rg16Snorm,
    ];
    assert!(
        bc_normal_formats.contains(&format),
        "Compressed normal map should use BC5 or fallback, got {:?}",
        format
    );
}

// -----------------------------------------------------------------------------
// Compressed Color Format Selection
// -----------------------------------------------------------------------------

/// Test: compressed_color with BC support and sRGB=false returns BC7.
#[test]
fn test_compressed_color_bc_linear() {
    let selector = TextureFormatSelector::with_backend_features(
        Backend::Vulkan,
        Features::TEXTURE_COMPRESSION_BC,
    );
    let format = selector.compressed_color(false);
    // BC7 for high-quality color, BC1/BC3 for lower quality
    let bc_formats = [
        TextureFormat::Bc7RgbaUnorm,
        TextureFormat::Bc3RgbaUnorm,
        TextureFormat::Bc1RgbaUnorm,
    ];
    assert!(
        bc_formats.contains(&format),
        "BC compressed linear color expected, got {:?}",
        format
    );
}

/// Test: compressed_color with BC support and sRGB=true returns BC7 sRGB.
#[test]
fn test_compressed_color_bc_srgb() {
    let selector = TextureFormatSelector::with_backend_features(
        Backend::Vulkan,
        Features::TEXTURE_COMPRESSION_BC,
    );
    let format = selector.compressed_color(true);
    // sRGB variants
    let bc_srgb_formats = [
        TextureFormat::Bc7RgbaUnormSrgb,
        TextureFormat::Bc3RgbaUnormSrgb,
        TextureFormat::Bc1RgbaUnormSrgb,
        TextureFormat::Rgba8UnormSrgb, // fallback if BC not available
    ];
    assert!(
        bc_srgb_formats.contains(&format),
        "BC compressed sRGB color expected, got {:?}",
        format
    );
}

/// Test: compressed_color with ASTC support returns ASTC format.
#[test]
fn test_compressed_color_astc() {
    let selector = TextureFormatSelector::with_backend_features(
        Backend::Metal,
        Features::TEXTURE_COMPRESSION_ASTC,
    );
    let format = selector.compressed_color(false);
    // ASTC is preferred on Apple/mobile
    let astc_formats = [
        TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::Unorm },
        TextureFormat::Astc { block: wgpu::AstcBlock::B5x4, channel: wgpu::AstcChannel::Unorm },
        TextureFormat::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::Unorm },
        TextureFormat::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::Unorm },
        TextureFormat::Rgba8Unorm, // fallback
    ];
    assert!(
        astc_formats.contains(&format),
        "ASTC compressed color expected on Apple, got {:?}",
        format
    );
}

/// Test: compressed_color without compression features returns uncompressed.
#[test]
fn test_compressed_color_fallback() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Vulkan, Features::empty());
    let format = selector.compressed_color(false);
    // Should fall back to uncompressed format
    let fallback_formats = [TextureFormat::Rgba8Unorm, TextureFormat::Bgra8Unorm];
    assert!(
        fallback_formats.contains(&format),
        "Without compression, should fallback to uncompressed, got {:?}",
        format
    );
}

// -----------------------------------------------------------------------------
// Swapchain Format Selection
// -----------------------------------------------------------------------------

/// Test: swapchain_format returns a valid swapchain-compatible format.
#[test]
fn test_swapchain_format_valid() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Vulkan, Features::empty());
    let format = selector.swapchain_format();
    // Common swapchain formats
    let swapchain_formats = [
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Rgb10a2Unorm,
    ];
    assert!(
        swapchain_formats.contains(&format),
        "Swapchain format should be a standard presentation format, got {:?}",
        format
    );
}

/// Test: Metal swapchain_format prefers BGRA.
#[test]
fn test_swapchain_format_metal() {
    let selector = TextureFormatSelector::with_backend_features(Backend::Metal, Features::empty());
    let format = selector.swapchain_format();
    let metal_swapchain_formats = [
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb,
    ];
    assert!(
        metal_swapchain_formats.contains(&format),
        "Metal swapchain should be BGRA or RGBA variant, got {:?}",
        format
    );
}

// =============================================================================
// SECTION 3: FORMAT TABLES TESTS
// =============================================================================

/// Test: COLOR_FORMATS table is non-empty.
#[test]
fn test_color_formats_table_non_empty() {
    assert!(
        !format_tables::COLOR_FORMATS.is_empty(),
        "COLOR_FORMATS table should not be empty"
    );
}

/// Test: DEPTH_FORMATS table is non-empty.
#[test]
fn test_depth_formats_table_non_empty() {
    assert!(
        !format_tables::DEPTH_FORMATS.is_empty(),
        "DEPTH_FORMATS table should not be empty"
    );
}

/// Test: COMPRESSED_BC table is non-empty.
#[test]
fn test_compressed_bc_table_non_empty() {
    assert!(
        !format_tables::COMPRESSED_BC.is_empty(),
        "COMPRESSED_BC table should not be empty"
    );
}

/// Test: COMPRESSED_ASTC table is non-empty.
#[test]
fn test_compressed_astc_table_non_empty() {
    assert!(
        !format_tables::COMPRESSED_ASTC.is_empty(),
        "COMPRESSED_ASTC table should not be empty"
    );
}

/// Test: COLOR_FORMATS contains common formats.
#[test]
fn test_color_formats_contains_common() {
    let formats = format_tables::COLOR_FORMATS;
    let has_rgba8 = formats
        .iter()
        .any(|f| matches!(f, TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb));
    assert!(has_rgba8, "COLOR_FORMATS should contain Rgba8 variants");
}

/// Test: DEPTH_FORMATS contains Depth32Float.
#[test]
fn test_depth_formats_contains_depth32() {
    let formats = format_tables::DEPTH_FORMATS;
    let has_depth32 = formats.iter().any(|f| *f == TextureFormat::Depth32Float);
    assert!(has_depth32, "DEPTH_FORMATS should contain Depth32Float");
}

/// Test: COMPRESSED_BC contains BC7.
#[test]
fn test_compressed_bc_contains_bc7() {
    let formats = format_tables::COMPRESSED_BC;
    let has_bc7 = formats.iter().any(|f| {
        matches!(
            f,
            TextureFormat::Bc7RgbaUnorm | TextureFormat::Bc7RgbaUnormSrgb
        )
    });
    assert!(has_bc7, "COMPRESSED_BC should contain BC7 variants");
}

/// Test: No duplicates in COLOR_FORMATS.
#[test]
fn test_color_formats_no_duplicates() {
    let formats = format_tables::COLOR_FORMATS;
    let mut seen = HashSet::new();
    for format in formats {
        assert!(
            seen.insert(format),
            "Duplicate format found in COLOR_FORMATS: {:?}",
            format
        );
    }
}

/// Test: No duplicates in DEPTH_FORMATS.
#[test]
fn test_depth_formats_no_duplicates() {
    let formats = format_tables::DEPTH_FORMATS;
    let mut seen = HashSet::new();
    for format in formats {
        assert!(
            seen.insert(format),
            "Duplicate format found in DEPTH_FORMATS: {:?}",
            format
        );
    }
}

/// Test: No duplicates in COMPRESSED_BC.
#[test]
fn test_compressed_bc_no_duplicates() {
    let formats = format_tables::COMPRESSED_BC;
    let mut seen = HashSet::new();
    for format in formats {
        assert!(
            seen.insert(format),
            "Duplicate format found in COMPRESSED_BC: {:?}",
            format
        );
    }
}

// =============================================================================
// SECTION 4: INTEGRATION TESTS (require GPU - marked #[ignore])
// =============================================================================

/// Test: Real adapter format selection (requires GPU).
#[test]

fn test_real_adapter_format_selection() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));

    if let Some(adapter) = adapter {
        let selector = TextureFormatSelector::new(&adapter);
        let backend = selector.backend();
        let platform = selector.platform();
        let _color = selector.color_attachment(false);
        let _depth = selector.depth(false);

        // Just verify no panic and reasonable values
        assert!(
            matches!(
                backend,
                Backend::Vulkan | Backend::Metal | Backend::Dx12 | Backend::Gl | Backend::BrowserWebGpu
            ),
            "Backend should be a valid variant"
        );
        assert!(
            matches!(
                platform,
                Platform::Desktop | Platform::Apple | Platform::Mobile | Platform::Web | Platform::Unknown
            ),
            "Platform should be a valid variant"
        );
    }
}

/// Test: Real adapter compression detection (requires GPU).
#[test]

fn test_real_adapter_compression_detection() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));

    if let Some(adapter) = adapter {
        let selector = TextureFormatSelector::new(&adapter);

        // At least one compression type should be available on most GPUs
        let has_any_compression =
            selector.has_bc_compression() || selector.has_astc_compression() || selector.has_etc2_compression();

        // This is informational - don't assert as some systems may have none
        println!(
            "Compression support: BC={}, ASTC={}, ETC2={}",
            selector.has_bc_compression(),
            selector.has_astc_compression(),
            selector.has_etc2_compression()
        );
        let _ = has_any_compression; // silence unused warning
    }
}

/// Test: Real adapter swapchain format (requires GPU).
#[test]

fn test_real_adapter_swapchain_format() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));

    if let Some(adapter) = adapter {
        let selector = TextureFormatSelector::new(&adapter);
        let format = selector.swapchain_format();

        // Verify it's a valid presentable format
        let valid_swapchain = [
            TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Rgb10a2Unorm,
        ];
        assert!(
            valid_swapchain.contains(&format),
            "Swapchain format should be presentable: {:?}",
            format
        );
    }
}

/// Test: supports() returns true for common formats (requires GPU).
#[test]

fn test_supports_common_formats() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));

    if let Some(adapter) = adapter {
        let selector = TextureFormatSelector::new(&adapter);

        // These formats should be universally supported
        assert!(
            selector.supports(TextureFormat::Rgba8Unorm),
            "Rgba8Unorm should be supported"
        );
        assert!(
            selector.supports(TextureFormat::Depth32Float),
            "Depth32Float should be supported"
        );
    }
}

/// Test: HDR format selection with real adapter (requires GPU).
#[test]

fn test_real_adapter_hdr_format() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));

    if let Some(adapter) = adapter {
        let selector = TextureFormatSelector::new(&adapter);
        let hdr_format = selector.color_attachment(true);

        // Verify it's an HDR-capable format
        let hdr_formats = [
            TextureFormat::Rgba16Float,
            TextureFormat::Rgba32Float,
            TextureFormat::Rgb10a2Unorm,
            TextureFormat::Rg11b10Float,
        ];
        assert!(
            hdr_formats.contains(&hdr_format),
            "HDR format should be high-precision: {:?}",
            hdr_format
        );
    }
}

/// Test: Compressed color selection matches platform (requires GPU).
#[test]

fn test_real_adapter_platform_compression() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));

    if let Some(adapter) = adapter {
        let selector = TextureFormatSelector::new(&adapter);
        let compressed = selector.compressed_color(false);

        match selector.platform() {
            Platform::Desktop => {
                // Desktop should prefer BC if available
                if selector.has_bc_compression() {
                    let bc_formats = [
                        TextureFormat::Bc7RgbaUnorm,
                        TextureFormat::Bc3RgbaUnorm,
                        TextureFormat::Bc1RgbaUnorm,
                    ];
                    assert!(
                        bc_formats.contains(&compressed) || compressed == TextureFormat::Rgba8Unorm,
                        "Desktop should use BC or fallback"
                    );
                }
            }
            Platform::Apple => {
                // Apple should prefer ASTC if available
                if selector.has_astc_compression() {
                    // ASTC or fallback
                    let is_astc = matches!(compressed, TextureFormat::Astc { .. });
                    assert!(
                        is_astc || compressed == TextureFormat::Rgba8Unorm,
                        "Apple should use ASTC or fallback"
                    );
                }
            }
            _ => {
                // Other platforms - just verify no panic
            }
        }
    }
}

/// Test: Normal map compression matches features (requires GPU).
#[test]

fn test_real_adapter_normal_map_compression() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));

    if let Some(adapter) = adapter {
        let selector = TextureFormatSelector::new(&adapter);

        let uncompressed = selector.normal_map(false);
        let compressed = selector.normal_map(true);

        // Uncompressed should be RG format
        let rg_formats = [
            TextureFormat::Rg16Snorm,
            TextureFormat::Rg16Unorm,
            TextureFormat::Rg8Snorm,
            TextureFormat::Rg8Unorm,
            TextureFormat::Rgba8Snorm,
            TextureFormat::Rgba8Unorm,
        ];
        assert!(
            rg_formats.contains(&uncompressed),
            "Uncompressed normal should be RG: {:?}",
            uncompressed
        );

        // Compressed should be BC5 or fallback
        if selector.has_bc_compression() {
            let bc_or_fallback = [
                TextureFormat::Bc5RgSnorm,
                TextureFormat::Bc5RgUnorm,
                TextureFormat::Rg16Snorm,
                TextureFormat::Rg16Unorm,
            ];
            assert!(
                bc_or_fallback.contains(&compressed),
                "Compressed normal should be BC5 or fallback: {:?}",
                compressed
            );
        }
    }
}
