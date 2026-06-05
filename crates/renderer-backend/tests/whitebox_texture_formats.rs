//! Whitebox tests for T-WGPU-P2.3.2: Texture Formats
//!
//! These tests have FULL ACCESS to the texture_formats.rs implementation and thoroughly
//! test all public functions, types, and edge cases.
//!
//! Test Categories:
//! - Platform Detection (8 tests): Backend to platform mapping
//! - Color Attachment (10 tests): SDR/HDR format selection per platform
//! - Depth Format (8 tests): Depth32Float, Depth24Plus, stencil variants
//! - Normal Map (8 tests): Rg16Snorm, BC5, ETC2 fallbacks
//! - Compressed Format (10 tests): BC/ASTC/ETC2 selection
//! - Format Tables (8 tests): Table completeness and contents
//! - Feature Detection (6 tests): has_bc/astc/etc2, supports()

use renderer_backend::resources::texture_formats::{format_tables, Platform, TextureFormatSelector};
use wgpu::{AstcBlock, AstcChannel, Backend, Features, TextureFormat};

// ============================================================================
// HELPER: Create selector with specific backend and features
// ============================================================================

fn selector_with_features(backend: Backend, features: Features) -> TextureFormatSelector {
    TextureFormatSelector::with_backend_features(backend, features)
}

fn desktop_no_compression() -> TextureFormatSelector {
    selector_with_features(Backend::Vulkan, Features::empty())
}

fn desktop_bc() -> TextureFormatSelector {
    selector_with_features(Backend::Vulkan, Features::TEXTURE_COMPRESSION_BC)
}

fn desktop_dx12_bc() -> TextureFormatSelector {
    selector_with_features(Backend::Dx12, Features::TEXTURE_COMPRESSION_BC)
}

fn apple_no_compression() -> TextureFormatSelector {
    selector_with_features(Backend::Metal, Features::empty())
}

fn apple_astc() -> TextureFormatSelector {
    selector_with_features(Backend::Metal, Features::TEXTURE_COMPRESSION_ASTC)
}

fn mobile_etc2() -> TextureFormatSelector {
    selector_with_features(Backend::Gl, Features::TEXTURE_COMPRESSION_ETC2)
}

fn mobile_astc() -> TextureFormatSelector {
    selector_with_features(Backend::Gl, Features::TEXTURE_COMPRESSION_ASTC)
}

fn web_no_compression() -> TextureFormatSelector {
    selector_with_features(Backend::BrowserWebGpu, Features::empty())
}

fn all_compression() -> TextureFormatSelector {
    selector_with_features(
        Backend::Vulkan,
        Features::TEXTURE_COMPRESSION_BC
            | Features::TEXTURE_COMPRESSION_ASTC
            | Features::TEXTURE_COMPRESSION_ETC2,
    )
}

// ============================================================================
// CATEGORY 1: Platform Detection Tests (8 tests)
// ============================================================================

mod platform_detection_tests {
    use super::*;

    #[test]
    fn test_vulkan_maps_to_desktop() {
        assert_eq!(Platform::from_backend(Backend::Vulkan), Platform::Desktop);
    }

    #[test]
    fn test_dx12_maps_to_desktop() {
        assert_eq!(Platform::from_backend(Backend::Dx12), Platform::Desktop);
    }

    #[test]
    fn test_metal_maps_to_apple() {
        assert_eq!(Platform::from_backend(Backend::Metal), Platform::Apple);
    }

    #[test]
    fn test_gl_maps_to_mobile() {
        assert_eq!(Platform::from_backend(Backend::Gl), Platform::Mobile);
    }

    #[test]
    fn test_browser_webgpu_maps_to_web() {
        assert_eq!(
            Platform::from_backend(Backend::BrowserWebGpu),
            Platform::Web
        );
    }

    #[test]
    fn test_platform_is_apple_method() {
        assert!(Platform::Apple.is_apple());
        assert!(!Platform::Desktop.is_apple());
        assert!(!Platform::Mobile.is_apple());
        assert!(!Platform::Web.is_apple());
        assert!(!Platform::Unknown.is_apple());
    }

    #[test]
    fn test_platform_is_desktop_method() {
        assert!(Platform::Desktop.is_desktop());
        assert!(!Platform::Apple.is_desktop());
        assert!(!Platform::Mobile.is_desktop());
        assert!(!Platform::Web.is_desktop());
        assert!(!Platform::Unknown.is_desktop());
    }

    #[test]
    fn test_platform_is_mobile_and_is_web() {
        assert!(Platform::Mobile.is_mobile());
        assert!(!Platform::Desktop.is_mobile());

        assert!(Platform::Web.is_web());
        assert!(!Platform::Desktop.is_web());
    }
}

// ============================================================================
// CATEGORY 2: Color Attachment Tests (10 tests)
// ============================================================================

mod color_attachment_tests {
    use super::*;

    #[test]
    fn test_desktop_sdr_returns_rgba8_srgb() {
        let selector = desktop_no_compression();
        assert_eq!(
            selector.color_attachment(false),
            TextureFormat::Rgba8UnormSrgb
        );
    }

    #[test]
    fn test_dx12_sdr_returns_rgba8_srgb() {
        let selector = desktop_dx12_bc();
        assert_eq!(
            selector.color_attachment(false),
            TextureFormat::Rgba8UnormSrgb
        );
    }

    #[test]
    fn test_apple_sdr_returns_bgra8_unorm() {
        let selector = apple_no_compression();
        assert_eq!(
            selector.color_attachment(false),
            TextureFormat::Bgra8Unorm
        );
    }

    #[test]
    fn test_apple_with_astc_sdr_returns_bgra8() {
        let selector = apple_astc();
        assert_eq!(
            selector.color_attachment(false),
            TextureFormat::Bgra8Unorm
        );
    }

    #[test]
    fn test_mobile_sdr_returns_bgra8() {
        let selector = mobile_etc2();
        assert_eq!(
            selector.color_attachment(false),
            TextureFormat::Bgra8Unorm
        );
    }

    #[test]
    fn test_web_sdr_returns_bgra8() {
        let selector = web_no_compression();
        assert_eq!(
            selector.color_attachment(false),
            TextureFormat::Bgra8Unorm
        );
    }

    #[test]
    fn test_hdr_always_returns_rgba16float() {
        // All platforms should return Rgba16Float for HDR
        assert_eq!(
            desktop_no_compression().color_attachment(true),
            TextureFormat::Rgba16Float
        );
        assert_eq!(
            apple_no_compression().color_attachment(true),
            TextureFormat::Rgba16Float
        );
        assert_eq!(
            mobile_etc2().color_attachment(true),
            TextureFormat::Rgba16Float
        );
        assert_eq!(
            web_no_compression().color_attachment(true),
            TextureFormat::Rgba16Float
        );
    }

    #[test]
    fn test_hdr_ignores_compression_features() {
        let selector = all_compression();
        assert_eq!(selector.color_attachment(true), TextureFormat::Rgba16Float);
    }

    #[test]
    fn test_color_attachment_unknown_platform_returns_bgra8() {
        // Unknown platform uses fallback path (same as Mobile/Web)
        let selector = selector_with_features(Backend::Empty, Features::empty());
        assert_eq!(
            selector.color_attachment(false),
            TextureFormat::Bgra8Unorm
        );
    }

    #[test]
    fn test_color_attachment_platform_consistency() {
        // Verify that platform detection is consistent with color_attachment
        let vulkan = desktop_no_compression();
        assert_eq!(vulkan.platform(), Platform::Desktop);
        assert_eq!(
            vulkan.color_attachment(false),
            TextureFormat::Rgba8UnormSrgb
        );

        let metal = apple_no_compression();
        assert_eq!(metal.platform(), Platform::Apple);
        assert_eq!(
            metal.color_attachment(false),
            TextureFormat::Bgra8Unorm
        );
    }
}

// ============================================================================
// CATEGORY 3: Depth Format Tests (8 tests)
// ============================================================================

mod depth_format_tests {
    use super::*;

    #[test]
    fn test_depth_without_stencil_returns_depth32float() {
        let selector = desktop_no_compression();
        assert_eq!(selector.depth(false), TextureFormat::Depth32Float);
    }

    #[test]
    fn test_depth_with_stencil_returns_depth24plus_stencil8() {
        let selector = desktop_no_compression();
        assert_eq!(selector.depth(true), TextureFormat::Depth24PlusStencil8);
    }

    #[test]
    fn test_depth_fallback_without_stencil_returns_depth24plus() {
        let selector = desktop_no_compression();
        assert_eq!(selector.depth_fallback(false), TextureFormat::Depth24Plus);
    }

    #[test]
    fn test_depth_fallback_with_stencil_returns_depth24plus_stencil8() {
        let selector = desktop_no_compression();
        assert_eq!(
            selector.depth_fallback(true),
            TextureFormat::Depth24PlusStencil8
        );
    }

    #[test]
    fn test_depth_consistent_across_platforms() {
        // Depth format should be same across all platforms
        let platforms = [
            desktop_no_compression(),
            apple_no_compression(),
            mobile_etc2(),
            web_no_compression(),
        ];

        for selector in &platforms {
            assert_eq!(selector.depth(false), TextureFormat::Depth32Float);
            assert_eq!(selector.depth(true), TextureFormat::Depth24PlusStencil8);
        }
    }

    #[test]
    fn test_depth32float_support_check() {
        let selector = desktop_no_compression();
        // Depth32Float is in standard formats, should always be supported
        assert!(selector.supports(TextureFormat::Depth32Float));
    }

    #[test]
    fn test_depth32float_stencil8_support_check() {
        let selector = desktop_no_compression();
        // Depth32FloatStencil8 should be supported per implementation
        assert!(selector.supports(TextureFormat::Depth32FloatStencil8));
    }

    #[test]
    fn test_depth_fallback_different_from_primary() {
        let selector = desktop_no_compression();
        // Primary depth should be higher precision than fallback
        assert_ne!(selector.depth(false), selector.depth_fallback(false));
        assert_eq!(
            selector.depth(false),
            TextureFormat::Depth32Float
        );
        assert_eq!(
            selector.depth_fallback(false),
            TextureFormat::Depth24Plus
        );
    }
}

// ============================================================================
// CATEGORY 4: Normal Map Tests (8 tests)
// ============================================================================

mod normal_map_tests {
    use super::*;

    #[test]
    fn test_uncompressed_normal_map_returns_rg16snorm() {
        let selector = desktop_no_compression();
        assert_eq!(selector.normal_map(false), TextureFormat::Rg16Snorm);
    }

    #[test]
    fn test_compressed_normal_map_with_bc_returns_bc5() {
        let selector = desktop_bc();
        assert_eq!(selector.normal_map(true), TextureFormat::Bc5RgSnorm);
    }

    #[test]
    fn test_compressed_normal_map_with_etc2_returns_eac() {
        let selector = mobile_etc2();
        assert_eq!(selector.normal_map(true), TextureFormat::EacRg11Snorm);
    }

    #[test]
    fn test_compressed_normal_map_without_compression_falls_back() {
        let selector = desktop_no_compression();
        // Without any compression, should fall back to Rgba8Snorm
        assert_eq!(selector.normal_map(true), TextureFormat::Rgba8Snorm);
    }

    #[test]
    fn test_normal_map_bc_priority_over_etc2() {
        // When both BC and ETC2 are available, BC should be used
        let selector = all_compression();
        assert_eq!(selector.normal_map(true), TextureFormat::Bc5RgSnorm);
    }

    #[test]
    fn test_uncompressed_normal_map_supports_check() {
        let selector = desktop_no_compression();
        assert!(selector.supports(TextureFormat::Rg16Snorm));
    }

    #[test]
    fn test_normal_map_astc_not_used_for_normal_maps() {
        // ASTC alone (without BC or ETC2) should fall back to uncompressed
        // because ASTC is for color, not normal maps (no signed RG variant in our selector)
        let selector = selector_with_features(
            Backend::Metal,
            Features::TEXTURE_COMPRESSION_ASTC,
        );
        // Without BC or ETC2, compressed normal map falls back to Rgba8Snorm
        assert_eq!(selector.normal_map(true), TextureFormat::Rgba8Snorm);
    }

    #[test]
    fn test_normal_map_etc2_fallback_when_no_bc() {
        // On mobile with only ETC2, should use EAC RG11
        let selector = mobile_etc2();
        assert_eq!(selector.normal_map(true), TextureFormat::EacRg11Snorm);

        // Uncompressed should still be Rg16Snorm
        assert_eq!(selector.normal_map(false), TextureFormat::Rg16Snorm);
    }
}

// ============================================================================
// CATEGORY 5: Compressed Format Tests (10 tests)
// ============================================================================

mod compressed_format_tests {
    use super::*;

    #[test]
    fn test_bc_compression_srgb_returns_bc7() {
        let selector = desktop_bc();
        assert_eq!(
            selector.compressed_color(true),
            TextureFormat::Bc7RgbaUnormSrgb
        );
    }

    #[test]
    fn test_bc_compression_linear_returns_bc7() {
        let selector = desktop_bc();
        assert_eq!(
            selector.compressed_color(false),
            TextureFormat::Bc7RgbaUnorm
        );
    }

    #[test]
    fn test_astc_compression_srgb_returns_astc4x4() {
        let selector = apple_astc();
        assert_eq!(
            selector.compressed_color(true),
            TextureFormat::Astc {
                block: AstcBlock::B4x4,
                channel: AstcChannel::UnormSrgb,
            }
        );
    }

    #[test]
    fn test_astc_compression_linear_returns_astc4x4() {
        let selector = apple_astc();
        assert_eq!(
            selector.compressed_color(false),
            TextureFormat::Astc {
                block: AstcBlock::B4x4,
                channel: AstcChannel::Unorm,
            }
        );
    }

    #[test]
    fn test_etc2_compression_srgb() {
        let selector = mobile_etc2();
        assert_eq!(
            selector.compressed_color(true),
            TextureFormat::Etc2Rgba8UnormSrgb
        );
    }

    #[test]
    fn test_etc2_compression_linear() {
        let selector = mobile_etc2();
        assert_eq!(
            selector.compressed_color(false),
            TextureFormat::Etc2Rgba8Unorm
        );
    }

    #[test]
    fn test_no_compression_fallback_srgb() {
        let selector = desktop_no_compression();
        assert_eq!(
            selector.compressed_color(true),
            TextureFormat::Rgba8UnormSrgb
        );
    }

    #[test]
    fn test_no_compression_fallback_linear() {
        let selector = desktop_no_compression();
        assert_eq!(
            selector.compressed_color(false),
            TextureFormat::Rgba8Unorm
        );
    }

    #[test]
    fn test_compression_priority_bc_over_astc_over_etc2() {
        // When all compression formats are available, BC should be preferred
        let selector = all_compression();
        assert_eq!(
            selector.compressed_color(true),
            TextureFormat::Bc7RgbaUnormSrgb
        );
        assert_eq!(
            selector.compressed_color(false),
            TextureFormat::Bc7RgbaUnorm
        );
    }

    #[test]
    fn test_compression_astc_preferred_over_etc2() {
        // When ASTC and ETC2 are available (but not BC), ASTC should be preferred
        let selector = selector_with_features(
            Backend::Gl,
            Features::TEXTURE_COMPRESSION_ASTC | Features::TEXTURE_COMPRESSION_ETC2,
        );
        assert_eq!(
            selector.compressed_color(true),
            TextureFormat::Astc {
                block: AstcBlock::B4x4,
                channel: AstcChannel::UnormSrgb,
            }
        );
    }
}

// ============================================================================
// CATEGORY 6: Format Tables Tests (8 tests)
// ============================================================================

mod format_tables_tests {
    use super::*;

    #[test]
    fn test_color_formats_contains_expected_formats() {
        assert!(format_tables::COLOR_FORMATS.contains(&TextureFormat::Rgba8UnormSrgb));
        assert!(format_tables::COLOR_FORMATS.contains(&TextureFormat::Bgra8UnormSrgb));
        assert!(format_tables::COLOR_FORMATS.contains(&TextureFormat::Rgba8Unorm));
        assert!(format_tables::COLOR_FORMATS.contains(&TextureFormat::Bgra8Unorm));
        assert!(format_tables::COLOR_FORMATS.contains(&TextureFormat::Rgba16Float));
        assert!(format_tables::COLOR_FORMATS.contains(&TextureFormat::Rgb10a2Unorm));
    }

    #[test]
    fn test_hdr_color_formats_are_float_or_high_precision() {
        for fmt in format_tables::HDR_COLOR_FORMATS {
            match fmt {
                TextureFormat::Rgba16Float
                | TextureFormat::Rg11b10Float
                | TextureFormat::Rgba32Float => {
                    // Valid HDR formats
                }
                _ => panic!("Unexpected format in HDR_COLOR_FORMATS: {:?}", fmt),
            }
        }
    }

    #[test]
    fn test_depth_formats_completeness() {
        assert!(format_tables::DEPTH_FORMATS.contains(&TextureFormat::Depth32Float));
        assert!(format_tables::DEPTH_FORMATS.contains(&TextureFormat::Depth24Plus));
        assert!(format_tables::DEPTH_FORMATS.contains(&TextureFormat::Depth16Unorm));
        assert_eq!(format_tables::DEPTH_FORMATS.len(), 3);
    }

    #[test]
    fn test_depth_stencil_formats_completeness() {
        assert!(format_tables::DEPTH_STENCIL_FORMATS.contains(&TextureFormat::Depth32FloatStencil8));
        assert!(format_tables::DEPTH_STENCIL_FORMATS.contains(&TextureFormat::Depth24PlusStencil8));
        assert_eq!(format_tables::DEPTH_STENCIL_FORMATS.len(), 2);
    }

    #[test]
    fn test_compressed_bc_table_contains_all_bc_variants() {
        let bc_formats = format_tables::COMPRESSED_BC;
        // BC1, BC3, BC4, BC5, BC6H, BC7 (with sRGB variants)
        assert!(bc_formats.contains(&TextureFormat::Bc1RgbaUnorm));
        assert!(bc_formats.contains(&TextureFormat::Bc1RgbaUnormSrgb));
        assert!(bc_formats.contains(&TextureFormat::Bc3RgbaUnorm));
        assert!(bc_formats.contains(&TextureFormat::Bc3RgbaUnormSrgb));
        assert!(bc_formats.contains(&TextureFormat::Bc4RUnorm));
        assert!(bc_formats.contains(&TextureFormat::Bc4RSnorm));
        assert!(bc_formats.contains(&TextureFormat::Bc5RgUnorm));
        assert!(bc_formats.contains(&TextureFormat::Bc5RgSnorm));
        assert!(bc_formats.contains(&TextureFormat::Bc6hRgbUfloat));
        assert!(bc_formats.contains(&TextureFormat::Bc6hRgbFloat));
        assert!(bc_formats.contains(&TextureFormat::Bc7RgbaUnorm));
        assert!(bc_formats.contains(&TextureFormat::Bc7RgbaUnormSrgb));
    }

    #[test]
    fn test_compressed_astc_table_contains_block_sizes() {
        let astc_formats = format_tables::COMPRESSED_ASTC;
        // Should contain 4x4, 5x5, 6x6, 8x8 blocks with Unorm and UnormSrgb
        assert!(astc_formats.contains(&TextureFormat::Astc {
            block: AstcBlock::B4x4,
            channel: AstcChannel::Unorm,
        }));
        assert!(astc_formats.contains(&TextureFormat::Astc {
            block: AstcBlock::B4x4,
            channel: AstcChannel::UnormSrgb,
        }));
        assert!(astc_formats.contains(&TextureFormat::Astc {
            block: AstcBlock::B8x8,
            channel: AstcChannel::Unorm,
        }));
        assert_eq!(astc_formats.len(), 8); // 4 block sizes * 2 channels
    }

    #[test]
    fn test_compressed_etc2_table_completeness() {
        let etc2_formats = format_tables::COMPRESSED_ETC2;
        assert!(etc2_formats.contains(&TextureFormat::Etc2Rgba8Unorm));
        assert!(etc2_formats.contains(&TextureFormat::Etc2Rgba8UnormSrgb));
        assert!(etc2_formats.contains(&TextureFormat::Etc2Rgb8Unorm));
        assert!(etc2_formats.contains(&TextureFormat::Etc2Rgb8UnormSrgb));
        assert!(etc2_formats.contains(&TextureFormat::EacR11Unorm));
        assert!(etc2_formats.contains(&TextureFormat::EacR11Snorm));
        assert!(etc2_formats.contains(&TextureFormat::EacRg11Unorm));
        assert!(etc2_formats.contains(&TextureFormat::EacRg11Snorm));
    }

    #[test]
    fn test_normal_map_formats_prioritizes_signed() {
        let normal_formats = format_tables::NORMAL_MAP_FORMATS;
        // First entries should be signed formats for tangent-space normals
        assert_eq!(normal_formats[0], TextureFormat::Rg16Snorm);
        assert!(normal_formats.contains(&TextureFormat::Rgba8Snorm));
        assert!(normal_formats.contains(&TextureFormat::Bc5RgSnorm));
    }
}

// ============================================================================
// CATEGORY 7: Feature Detection Tests (6 tests)
// ============================================================================

mod feature_detection_tests {
    use super::*;

    #[test]
    fn test_has_bc_compression_with_bc_feature() {
        let selector = desktop_bc();
        assert!(selector.has_bc_compression());
        assert!(!selector.has_astc_compression());
        assert!(!selector.has_etc2_compression());
    }

    #[test]
    fn test_has_astc_compression_with_astc_feature() {
        let selector = apple_astc();
        assert!(!selector.has_bc_compression());
        assert!(selector.has_astc_compression());
        assert!(!selector.has_etc2_compression());
    }

    #[test]
    fn test_has_etc2_compression_with_etc2_feature() {
        let selector = mobile_etc2();
        assert!(!selector.has_bc_compression());
        assert!(!selector.has_astc_compression());
        assert!(selector.has_etc2_compression());
    }

    #[test]
    fn test_has_all_compression_types() {
        let selector = all_compression();
        assert!(selector.has_bc_compression());
        assert!(selector.has_astc_compression());
        assert!(selector.has_etc2_compression());
    }

    #[test]
    fn test_supports_returns_correct_for_compression_formats() {
        let bc_selector = desktop_bc();
        assert!(bc_selector.supports(TextureFormat::Bc7RgbaUnorm));
        assert!(bc_selector.supports(TextureFormat::Bc5RgSnorm));
        assert!(bc_selector.supports(TextureFormat::Bc1RgbaUnorm));
        assert!(!bc_selector.supports(TextureFormat::Etc2Rgba8Unorm));
        assert!(!bc_selector.supports(TextureFormat::Astc {
            block: AstcBlock::B4x4,
            channel: AstcChannel::Unorm,
        }));

        let etc2_selector = mobile_etc2();
        assert!(etc2_selector.supports(TextureFormat::Etc2Rgba8Unorm));
        assert!(etc2_selector.supports(TextureFormat::EacRg11Snorm));
        assert!(!etc2_selector.supports(TextureFormat::Bc7RgbaUnorm));

        let astc_selector = apple_astc();
        assert!(astc_selector.supports(TextureFormat::Astc {
            block: AstcBlock::B4x4,
            channel: AstcChannel::Unorm,
        }));
        assert!(!astc_selector.supports(TextureFormat::Bc7RgbaUnorm));
    }

    #[test]
    fn test_supports_standard_formats_always() {
        let selector = desktop_no_compression();
        // Standard uncompressed formats should always be supported
        assert!(selector.supports(TextureFormat::Rgba8Unorm));
        assert!(selector.supports(TextureFormat::Rgba8UnormSrgb));
        assert!(selector.supports(TextureFormat::Bgra8Unorm));
        assert!(selector.supports(TextureFormat::Rgba16Float));
        assert!(selector.supports(TextureFormat::Depth32Float));
        assert!(selector.supports(TextureFormat::Rg16Snorm));
    }
}

// ============================================================================
// CATEGORY 8: Accessor and Utility Tests (8 tests)
// ============================================================================

mod accessor_tests {
    use super::*;

    #[test]
    fn test_backend_accessor_returns_correct_backend() {
        assert_eq!(desktop_no_compression().backend(), Backend::Vulkan);
        assert_eq!(desktop_dx12_bc().backend(), Backend::Dx12);
        assert_eq!(apple_no_compression().backend(), Backend::Metal);
        assert_eq!(mobile_etc2().backend(), Backend::Gl);
        assert_eq!(web_no_compression().backend(), Backend::BrowserWebGpu);
    }

    #[test]
    fn test_features_accessor_returns_correct_features() {
        let selector = all_compression();
        assert!(selector.features().contains(Features::TEXTURE_COMPRESSION_BC));
        assert!(selector.features().contains(Features::TEXTURE_COMPRESSION_ASTC));
        assert!(selector.features().contains(Features::TEXTURE_COMPRESSION_ETC2));
    }

    #[test]
    fn test_platform_accessor_returns_correct_platform() {
        assert_eq!(desktop_no_compression().platform(), Platform::Desktop);
        assert_eq!(apple_no_compression().platform(), Platform::Apple);
        assert_eq!(mobile_etc2().platform(), Platform::Mobile);
        assert_eq!(web_no_compression().platform(), Platform::Web);
    }

    #[test]
    fn test_compression_scheme_bc() {
        assert_eq!(desktop_bc().compression_scheme(), "BC");
    }

    #[test]
    fn test_compression_scheme_astc() {
        assert_eq!(apple_astc().compression_scheme(), "ASTC");
    }

    #[test]
    fn test_compression_scheme_etc2() {
        assert_eq!(mobile_etc2().compression_scheme(), "ETC2");
    }

    #[test]
    fn test_compression_scheme_none() {
        assert_eq!(desktop_no_compression().compression_scheme(), "None");
        assert_eq!(web_no_compression().compression_scheme(), "None");
    }

    #[test]
    fn test_compression_scheme_priority() {
        // BC > ASTC > ETC2, so when all available, should report BC
        assert_eq!(all_compression().compression_scheme(), "BC");
    }
}

// ============================================================================
// CATEGORY 9: Swapchain and Enumeration Tests (8 tests)
// ============================================================================

mod swapchain_enumeration_tests {
    use super::*;

    #[test]
    fn test_swapchain_format_is_bgra8_srgb() {
        // All platforms should return Bgra8UnormSrgb for universal compatibility
        assert_eq!(
            desktop_no_compression().swapchain_format(),
            TextureFormat::Bgra8UnormSrgb
        );
        assert_eq!(
            apple_no_compression().swapchain_format(),
            TextureFormat::Bgra8UnormSrgb
        );
        assert_eq!(
            mobile_etc2().swapchain_format(),
            TextureFormat::Bgra8UnormSrgb
        );
        assert_eq!(
            web_no_compression().swapchain_format(),
            TextureFormat::Bgra8UnormSrgb
        );
    }

    #[test]
    fn test_supported_color_formats_not_empty() {
        let formats = desktop_no_compression().supported_color_formats();
        assert!(!formats.is_empty());
    }

    #[test]
    fn test_supported_color_formats_contains_standard_formats() {
        let formats = desktop_no_compression().supported_color_formats();
        assert!(formats.contains(&TextureFormat::Rgba8UnormSrgb));
        assert!(formats.contains(&TextureFormat::Rgba8Unorm));
        assert!(formats.contains(&TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn test_supported_compressed_formats_bc() {
        let formats = desktop_bc().supported_compressed_formats();
        assert!(!formats.is_empty());
        assert!(formats.contains(&TextureFormat::Bc7RgbaUnorm));
        assert!(formats.contains(&TextureFormat::Bc5RgSnorm));
    }

    #[test]
    fn test_supported_compressed_formats_astc() {
        let formats = apple_astc().supported_compressed_formats();
        assert!(!formats.is_empty());
        assert!(formats.contains(&TextureFormat::Astc {
            block: AstcBlock::B4x4,
            channel: AstcChannel::Unorm,
        }));
    }

    #[test]
    fn test_supported_compressed_formats_etc2() {
        let formats = mobile_etc2().supported_compressed_formats();
        assert!(!formats.is_empty());
        assert!(formats.contains(&TextureFormat::Etc2Rgba8Unorm));
        assert!(formats.contains(&TextureFormat::EacRg11Snorm));
    }

    #[test]
    fn test_supported_compressed_formats_none() {
        let formats = desktop_no_compression().supported_compressed_formats();
        assert!(formats.is_empty());

        let web_formats = web_no_compression().supported_compressed_formats();
        assert!(web_formats.is_empty());
    }

    #[test]
    fn test_supported_compressed_formats_all_types() {
        let formats = all_compression().supported_compressed_formats();
        // Should contain BC, ASTC, and ETC2 formats
        assert!(formats.contains(&TextureFormat::Bc7RgbaUnorm));
        assert!(formats.contains(&TextureFormat::Astc {
            block: AstcBlock::B4x4,
            channel: AstcChannel::Unorm,
        }));
        assert!(formats.contains(&TextureFormat::Etc2Rgba8Unorm));
    }
}

// ============================================================================
// CATEGORY 10: Edge Case and Boundary Tests (6 tests)
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_backend_maps_to_unknown() {
        let selector = selector_with_features(Backend::Empty, Features::empty());
        assert_eq!(selector.platform(), Platform::Unknown);
    }

    #[test]
    fn test_empty_features_no_compression() {
        let selector = selector_with_features(Backend::Vulkan, Features::empty());
        assert!(!selector.has_bc_compression());
        assert!(!selector.has_astc_compression());
        assert!(!selector.has_etc2_compression());
    }

    #[test]
    fn test_combined_features_work() {
        let features = Features::TEXTURE_COMPRESSION_BC | Features::TEXTURE_COMPRESSION_ASTC;
        let selector = selector_with_features(Backend::Vulkan, features);
        assert!(selector.has_bc_compression());
        assert!(selector.has_astc_compression());
        assert!(!selector.has_etc2_compression());
    }

    #[test]
    fn test_bc2_format_support_check() {
        // BC2 is in the supports() match arm
        let selector = desktop_bc();
        assert!(selector.supports(TextureFormat::Bc2RgbaUnorm));
        assert!(selector.supports(TextureFormat::Bc2RgbaUnormSrgb));
    }

    #[test]
    fn test_all_astc_blocks_supported_with_feature() {
        let selector = apple_astc();
        let blocks = [
            AstcBlock::B4x4,
            AstcBlock::B5x5,
            AstcBlock::B6x6,
            AstcBlock::B8x8,
        ];
        for block in &blocks {
            assert!(selector.supports(TextureFormat::Astc {
                block: *block,
                channel: AstcChannel::Unorm,
            }));
            assert!(selector.supports(TextureFormat::Astc {
                block: *block,
                channel: AstcChannel::UnormSrgb,
            }));
        }
    }

    #[test]
    fn test_srgb_linear_table_subsets() {
        // Verify sRGB tables are subsets of main tables with correct formats
        for fmt in format_tables::COMPRESSED_BC_SRGB {
            assert!(format_tables::COMPRESSED_BC.contains(fmt));
            match fmt {
                TextureFormat::Bc7RgbaUnormSrgb
                | TextureFormat::Bc3RgbaUnormSrgb
                | TextureFormat::Bc1RgbaUnormSrgb => {}
                _ => panic!("Non-sRGB format in BC_SRGB table: {:?}", fmt),
            }
        }

        for fmt in format_tables::COMPRESSED_BC_LINEAR {
            assert!(format_tables::COMPRESSED_BC.contains(fmt));
            match fmt {
                TextureFormat::Bc7RgbaUnorm
                | TextureFormat::Bc3RgbaUnorm
                | TextureFormat::Bc1RgbaUnorm => {}
                _ => panic!("sRGB format in BC_LINEAR table: {:?}", fmt),
            }
        }
    }
}

// ============================================================================
// CATEGORY 11: Single/Two Channel Format Tests (4 tests)
// ============================================================================

mod channel_format_tests {
    use super::*;

    #[test]
    fn test_single_channel_formats_table() {
        let formats = format_tables::SINGLE_CHANNEL_FORMATS;
        assert!(!formats.is_empty());
        assert!(formats.contains(&TextureFormat::R16Float));
        assert!(formats.contains(&TextureFormat::R32Float));
        assert!(formats.contains(&TextureFormat::R8Unorm));
        assert!(formats.contains(&TextureFormat::Bc4RUnorm));
        assert!(formats.contains(&TextureFormat::EacR11Unorm));
    }

    #[test]
    fn test_two_channel_formats_table() {
        let formats = format_tables::TWO_CHANNEL_FORMATS;
        assert!(!formats.is_empty());
        assert!(formats.contains(&TextureFormat::Rg16Snorm));
        assert!(formats.contains(&TextureFormat::Rg16Float));
        assert!(formats.contains(&TextureFormat::Rg8Snorm));
        assert!(formats.contains(&TextureFormat::Rg8Unorm));
    }

    #[test]
    fn test_normal_map_compressed_table() {
        let formats = format_tables::NORMAL_MAP_COMPRESSED;
        assert!(formats.contains(&TextureFormat::Bc5RgSnorm));
        assert!(formats.contains(&TextureFormat::Bc5RgUnorm));
        assert!(formats.contains(&TextureFormat::EacRg11Snorm));
        assert!(formats.contains(&TextureFormat::EacRg11Unorm));
    }

    #[test]
    fn test_etc2_srgb_and_linear_tables() {
        // sRGB table
        for fmt in format_tables::COMPRESSED_ETC2_SRGB {
            assert!(format_tables::COMPRESSED_ETC2.contains(fmt));
        }
        assert!(format_tables::COMPRESSED_ETC2_SRGB.contains(&TextureFormat::Etc2Rgba8UnormSrgb));

        // Linear table
        for fmt in format_tables::COMPRESSED_ETC2_LINEAR {
            assert!(format_tables::COMPRESSED_ETC2.contains(fmt));
        }
        assert!(format_tables::COMPRESSED_ETC2_LINEAR.contains(&TextureFormat::Etc2Rgba8Unorm));
    }
}

// ============================================================================
// CATEGORY 12: ASTC sRGB/Linear Table Tests (4 tests)
// ============================================================================

mod astc_table_tests {
    use super::*;

    #[test]
    fn test_astc_srgb_table_all_have_srgb_channel() {
        for fmt in format_tables::COMPRESSED_ASTC_SRGB {
            match fmt {
                TextureFormat::Astc { channel, .. } => {
                    assert_eq!(*channel, AstcChannel::UnormSrgb);
                }
                _ => panic!("Non-ASTC format in ASTC_SRGB table"),
            }
        }
    }

    #[test]
    fn test_astc_linear_table_all_have_unorm_channel() {
        for fmt in format_tables::COMPRESSED_ASTC_LINEAR {
            match fmt {
                TextureFormat::Astc { channel, .. } => {
                    assert_eq!(*channel, AstcChannel::Unorm);
                }
                _ => panic!("Non-ASTC format in ASTC_LINEAR table"),
            }
        }
    }

    #[test]
    fn test_astc_srgb_and_linear_tables_have_same_block_sizes() {
        // Both tables should have the same block sizes (4x4, 5x5, 6x6, 8x8)
        assert_eq!(
            format_tables::COMPRESSED_ASTC_SRGB.len(),
            format_tables::COMPRESSED_ASTC_LINEAR.len()
        );
        assert_eq!(format_tables::COMPRESSED_ASTC_SRGB.len(), 4);
    }

    #[test]
    fn test_astc_tables_are_subsets_of_main() {
        for fmt in format_tables::COMPRESSED_ASTC_SRGB {
            assert!(format_tables::COMPRESSED_ASTC.contains(fmt));
        }
        for fmt in format_tables::COMPRESSED_ASTC_LINEAR {
            assert!(format_tables::COMPRESSED_ASTC.contains(fmt));
        }
    }
}
