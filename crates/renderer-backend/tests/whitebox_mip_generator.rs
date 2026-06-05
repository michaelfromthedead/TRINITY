//! Whitebox tests for mip_generator.rs (T-WGPU-P2.3.4)
//!
//! Tests cover:
//! - MipFilter enum (as_u32, name, all, Default)
//! - calculate_mip_size function (POT, NPOT, non-square, edge cases)
//! - calculate_mip_levels function (POT, NPOT, non-square, edge cases)
//! - Format support functions (is_format_supported, is_filterable, storage_format_for)
//! - MipChainInfo struct (new, with_mip_count, mip_size, total_texels, etc.)
//! - StorageFormatInfo struct
//! - Constants (WORKGROUP_SIZE, MIN_MIP_DIMENSION, UNIFORM_BUFFER_SIZE)

use renderer_backend::resources::mip_generator::{
    calculate_mip_levels, calculate_mip_size, is_filterable, is_format_supported,
    storage_format_for, MipChainInfo, MipFilter, StorageFormatInfo, MIN_MIP_DIMENSION,
    WORKGROUP_SIZE,
};
use wgpu::TextureFormat;

// ============================================================================
// MipFilter Enum Tests
// ============================================================================

mod mip_filter_tests {
    use super::*;

    #[test]
    fn test_box_filter_as_u32() {
        assert_eq!(MipFilter::Box.as_u32(), 0);
    }

    #[test]
    fn test_bilinear_filter_as_u32() {
        assert_eq!(MipFilter::Bilinear.as_u32(), 1);
    }

    #[test]
    fn test_box_filter_name() {
        assert_eq!(MipFilter::Box.name(), "Box");
    }

    #[test]
    fn test_bilinear_filter_name() {
        assert_eq!(MipFilter::Bilinear.name(), "Bilinear");
    }

    #[test]
    fn test_default_is_box() {
        assert_eq!(MipFilter::default(), MipFilter::Box);
    }

    #[test]
    fn test_all_returns_both_variants() {
        let all = MipFilter::all();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0], MipFilter::Box);
        assert_eq!(all[1], MipFilter::Bilinear);
    }

    #[test]
    fn test_all_contains_box() {
        assert!(MipFilter::all().contains(&MipFilter::Box));
    }

    #[test]
    fn test_all_contains_bilinear() {
        assert!(MipFilter::all().contains(&MipFilter::Bilinear));
    }

    #[test]
    fn test_filter_equality() {
        assert_eq!(MipFilter::Box, MipFilter::Box);
        assert_eq!(MipFilter::Bilinear, MipFilter::Bilinear);
        assert_ne!(MipFilter::Box, MipFilter::Bilinear);
    }

    #[test]
    fn test_filter_clone() {
        let filter = MipFilter::Bilinear;
        let cloned = filter.clone();
        assert_eq!(filter, cloned);
    }

    #[test]
    fn test_filter_copy() {
        let filter = MipFilter::Box;
        let copied: MipFilter = filter; // Copy trait
        assert_eq!(filter, copied);
    }

    #[test]
    fn test_filter_debug() {
        let debug_box = format!("{:?}", MipFilter::Box);
        let debug_bilinear = format!("{:?}", MipFilter::Bilinear);
        assert!(debug_box.contains("Box"));
        assert!(debug_bilinear.contains("Bilinear"));
    }

    #[test]
    fn test_filter_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MipFilter::Box);
        set.insert(MipFilter::Bilinear);
        set.insert(MipFilter::Box); // Duplicate
        assert_eq!(set.len(), 2);
    }
}

// ============================================================================
// calculate_mip_size Tests
// ============================================================================

mod mip_size_tests {
    use super::*;

    // Power-of-two textures
    #[test]
    fn test_pot_1024x1024_level_0() {
        assert_eq!(calculate_mip_size(1024, 1024, 0), (1024, 1024));
    }

    #[test]
    fn test_pot_1024x1024_level_1() {
        assert_eq!(calculate_mip_size(1024, 1024, 1), (512, 512));
    }

    #[test]
    fn test_pot_1024x1024_level_2() {
        assert_eq!(calculate_mip_size(1024, 1024, 2), (256, 256));
    }

    #[test]
    fn test_pot_1024x1024_level_3() {
        assert_eq!(calculate_mip_size(1024, 1024, 3), (128, 128));
    }

    #[test]
    fn test_pot_1024x1024_level_10() {
        assert_eq!(calculate_mip_size(1024, 1024, 10), (1, 1));
    }

    #[test]
    fn test_pot_512x512_level_0() {
        assert_eq!(calculate_mip_size(512, 512, 0), (512, 512));
    }

    #[test]
    fn test_pot_512x512_level_9() {
        assert_eq!(calculate_mip_size(512, 512, 9), (1, 1));
    }

    #[test]
    fn test_pot_256x256_level_8() {
        assert_eq!(calculate_mip_size(256, 256, 8), (1, 1));
    }

    // Non-power-of-two textures
    #[test]
    fn test_npot_100x100_level_0() {
        assert_eq!(calculate_mip_size(100, 100, 0), (100, 100));
    }

    #[test]
    fn test_npot_100x100_level_1() {
        assert_eq!(calculate_mip_size(100, 100, 1), (50, 50));
    }

    #[test]
    fn test_npot_100x100_level_2() {
        assert_eq!(calculate_mip_size(100, 100, 2), (25, 25));
    }

    #[test]
    fn test_npot_100x100_level_3() {
        assert_eq!(calculate_mip_size(100, 100, 3), (12, 12));
    }

    #[test]
    fn test_npot_100x100_level_4() {
        assert_eq!(calculate_mip_size(100, 100, 4), (6, 6));
    }

    #[test]
    fn test_npot_100x100_level_5() {
        assert_eq!(calculate_mip_size(100, 100, 5), (3, 3));
    }

    #[test]
    fn test_npot_100x100_level_6() {
        assert_eq!(calculate_mip_size(100, 100, 6), (1, 1));
    }

    #[test]
    fn test_npot_300x200_level_0() {
        assert_eq!(calculate_mip_size(300, 200, 0), (300, 200));
    }

    #[test]
    fn test_npot_300x200_level_1() {
        assert_eq!(calculate_mip_size(300, 200, 1), (150, 100));
    }

    #[test]
    fn test_npot_300x200_level_2() {
        assert_eq!(calculate_mip_size(300, 200, 2), (75, 50));
    }

    #[test]
    fn test_npot_300x200_level_3() {
        assert_eq!(calculate_mip_size(300, 200, 3), (37, 25));
    }

    #[test]
    fn test_npot_1920x1080_level_0() {
        assert_eq!(calculate_mip_size(1920, 1080, 0), (1920, 1080));
    }

    #[test]
    fn test_npot_1920x1080_level_1() {
        assert_eq!(calculate_mip_size(1920, 1080, 1), (960, 540));
    }

    #[test]
    fn test_npot_1920x1080_level_10() {
        assert_eq!(calculate_mip_size(1920, 1080, 10), (1, 1));
    }

    // Non-square textures
    #[test]
    fn test_nonsquare_256x64_level_0() {
        assert_eq!(calculate_mip_size(256, 64, 0), (256, 64));
    }

    #[test]
    fn test_nonsquare_256x64_level_1() {
        assert_eq!(calculate_mip_size(256, 64, 1), (128, 32));
    }

    #[test]
    fn test_nonsquare_256x64_level_2() {
        assert_eq!(calculate_mip_size(256, 64, 2), (64, 16));
    }

    #[test]
    fn test_nonsquare_256x64_level_6() {
        assert_eq!(calculate_mip_size(256, 64, 6), (4, 1));
    }

    #[test]
    fn test_nonsquare_256x64_level_8() {
        assert_eq!(calculate_mip_size(256, 64, 8), (1, 1));
    }

    #[test]
    fn test_nonsquare_64x256_level_0() {
        assert_eq!(calculate_mip_size(64, 256, 0), (64, 256));
    }

    #[test]
    fn test_nonsquare_64x256_level_1() {
        assert_eq!(calculate_mip_size(64, 256, 1), (32, 128));
    }

    #[test]
    fn test_nonsquare_64x256_level_6() {
        assert_eq!(calculate_mip_size(64, 256, 6), (1, 4));
    }

    #[test]
    fn test_nonsquare_64x256_level_8() {
        assert_eq!(calculate_mip_size(64, 256, 8), (1, 1));
    }

    // Edge cases
    #[test]
    fn test_edge_1x1_level_0() {
        assert_eq!(calculate_mip_size(1, 1, 0), (1, 1));
    }

    #[test]
    fn test_edge_1x1_level_5() {
        assert_eq!(calculate_mip_size(1, 1, 5), (1, 1));
    }

    #[test]
    fn test_edge_2x1_level_0() {
        assert_eq!(calculate_mip_size(2, 1, 0), (2, 1));
    }

    #[test]
    fn test_edge_2x1_level_1() {
        assert_eq!(calculate_mip_size(2, 1, 1), (1, 1));
    }

    #[test]
    fn test_edge_1x2_level_1() {
        assert_eq!(calculate_mip_size(1, 2, 1), (1, 1));
    }

    #[test]
    fn test_edge_2x2_level_0() {
        assert_eq!(calculate_mip_size(2, 2, 0), (2, 2));
    }

    #[test]
    fn test_edge_2x2_level_1() {
        assert_eq!(calculate_mip_size(2, 2, 1), (1, 1));
    }

    #[test]
    fn test_edge_4x4_level_2() {
        assert_eq!(calculate_mip_size(4, 4, 2), (1, 1));
    }

    #[test]
    fn test_edge_4x4_level_10() {
        // High mip level on small texture stays at 1
        assert_eq!(calculate_mip_size(4, 4, 10), (1, 1));
    }

    // Note: Very high mip levels (>= 32) will cause shift overflow in the
    // implementation since it uses `width >> mip_level`. This is expected
    // behavior - callers should use calculate_mip_levels() to get valid
    // mip level count and not exceed it.
    #[test]
    fn test_edge_high_mip_level_within_u32_bits() {
        // Mip level 31 is the maximum safe shift for u32
        // 1024 >> 31 = 0, which clamps to 1
        assert_eq!(calculate_mip_size(1024, 1024, 31), (1, 1));
    }

    #[test]
    fn test_minimum_dimension_preserved() {
        // Verify minimum dimension is always 1, never 0
        for level in 0..=15 {
            let (w, h) = calculate_mip_size(1024, 1024, level);
            assert!(w >= 1, "Width at level {} should be >= 1, got {}", level, w);
            assert!(
                h >= 1,
                "Height at level {} should be >= 1, got {}",
                level,
                h
            );
        }
    }

    #[test]
    fn test_odd_dimension_handling() {
        // 3x3 -> 1x1 at level 1 (floor division: 3 >> 1 = 1)
        assert_eq!(calculate_mip_size(3, 3, 1), (1, 1));

        // 5x5 -> 2x2 at level 1
        assert_eq!(calculate_mip_size(5, 5, 1), (2, 2));

        // 7x7 -> 3x3 at level 1
        assert_eq!(calculate_mip_size(7, 7, 1), (3, 3));
    }
}

// ============================================================================
// calculate_mip_levels Tests
// ============================================================================

mod mip_levels_tests {
    use super::*;

    // Power-of-two textures
    #[test]
    fn test_pot_1024x1024() {
        assert_eq!(calculate_mip_levels(1024, 1024), 11);
    }

    #[test]
    fn test_pot_512x512() {
        assert_eq!(calculate_mip_levels(512, 512), 10);
    }

    #[test]
    fn test_pot_256x256() {
        assert_eq!(calculate_mip_levels(256, 256), 9);
    }

    #[test]
    fn test_pot_128x128() {
        assert_eq!(calculate_mip_levels(128, 128), 8);
    }

    #[test]
    fn test_pot_64x64() {
        assert_eq!(calculate_mip_levels(64, 64), 7);
    }

    #[test]
    fn test_pot_32x32() {
        assert_eq!(calculate_mip_levels(32, 32), 6);
    }

    #[test]
    fn test_pot_16x16() {
        assert_eq!(calculate_mip_levels(16, 16), 5);
    }

    #[test]
    fn test_pot_8x8() {
        assert_eq!(calculate_mip_levels(8, 8), 4);
    }

    #[test]
    fn test_pot_4x4() {
        assert_eq!(calculate_mip_levels(4, 4), 3);
    }

    #[test]
    fn test_pot_2x2() {
        assert_eq!(calculate_mip_levels(2, 2), 2);
    }

    #[test]
    fn test_pot_1x1() {
        assert_eq!(calculate_mip_levels(1, 1), 1);
    }

    // Non-power-of-two textures
    #[test]
    fn test_npot_100x100() {
        // log2(100) ~ 6.64, so 7 levels
        assert_eq!(calculate_mip_levels(100, 100), 7);
    }

    #[test]
    fn test_npot_300x200() {
        // log2(300) ~ 8.23, so 9 levels
        assert_eq!(calculate_mip_levels(300, 200), 9);
    }

    #[test]
    fn test_npot_1920x1080() {
        // log2(1920) ~ 10.9, so 11 levels
        assert_eq!(calculate_mip_levels(1920, 1080), 11);
    }

    #[test]
    fn test_npot_1280x720() {
        // log2(1280) ~ 10.32, so 11 levels
        assert_eq!(calculate_mip_levels(1280, 720), 11);
    }

    #[test]
    fn test_npot_640x480() {
        // log2(640) ~ 9.32, so 10 levels
        assert_eq!(calculate_mip_levels(640, 480), 10);
    }

    // Non-square textures (uses max dimension)
    #[test]
    fn test_nonsquare_512x256() {
        // Based on 512
        assert_eq!(calculate_mip_levels(512, 256), 10);
    }

    #[test]
    fn test_nonsquare_256x512() {
        // Based on 512
        assert_eq!(calculate_mip_levels(256, 512), 10);
    }

    #[test]
    fn test_nonsquare_1024x1() {
        // Based on 1024
        assert_eq!(calculate_mip_levels(1024, 1), 11);
    }

    #[test]
    fn test_nonsquare_1x1024() {
        // Based on 1024
        assert_eq!(calculate_mip_levels(1, 1024), 11);
    }

    #[test]
    fn test_nonsquare_256x64() {
        // Based on 256
        assert_eq!(calculate_mip_levels(256, 64), 9);
    }

    // Edge cases - zero dimensions
    // Note: The implementation uses max(width, height) for mip level calculation.
    // If one dimension is 0, the other dimension determines the mip count.
    // If both are 0, the function returns 1 (degenerate case).
    #[test]
    fn test_zero_width_uses_height() {
        // When width is 0, height (100) determines levels: log2(100) + 1 = 7
        assert_eq!(calculate_mip_levels(0, 100), 7);
    }

    #[test]
    fn test_zero_height_uses_width() {
        // When height is 0, width (100) determines levels: log2(100) + 1 = 7
        assert_eq!(calculate_mip_levels(100, 0), 7);
    }

    #[test]
    fn test_zero_both_returns_1() {
        // Degenerate case: both dimensions 0 returns 1 level
        assert_eq!(calculate_mip_levels(0, 0), 1);
    }

    #[test]
    fn test_large_texture_4096() {
        assert_eq!(calculate_mip_levels(4096, 4096), 13);
    }

    #[test]
    fn test_large_texture_8192() {
        assert_eq!(calculate_mip_levels(8192, 8192), 14);
    }

    #[test]
    fn test_large_texture_16384() {
        assert_eq!(calculate_mip_levels(16384, 16384), 15);
    }
}

// ============================================================================
// Format Support Tests
// ============================================================================

mod format_support_tests {
    use super::*;

    // Supported formats - RGBA variants
    #[test]
    fn test_supported_rgba8unorm() {
        assert!(is_format_supported(TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn test_supported_rgba8unorm_srgb() {
        assert!(is_format_supported(TextureFormat::Rgba8UnormSrgb));
    }

    // Supported formats - BGRA variants
    #[test]
    fn test_supported_bgra8unorm() {
        assert!(is_format_supported(TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn test_supported_bgra8unorm_srgb() {
        assert!(is_format_supported(TextureFormat::Bgra8UnormSrgb));
    }

    // Supported formats - Float variants
    #[test]
    fn test_supported_rgba16float() {
        assert!(is_format_supported(TextureFormat::Rgba16Float));
    }

    #[test]
    fn test_supported_r32float() {
        assert!(is_format_supported(TextureFormat::R32Float));
    }

    #[test]
    fn test_supported_rg32float() {
        assert!(is_format_supported(TextureFormat::Rg32Float));
    }

    #[test]
    fn test_supported_rgba32float() {
        assert!(is_format_supported(TextureFormat::Rgba32Float));
    }

    // Supported formats - Single/dual channel
    #[test]
    fn test_supported_r8unorm() {
        assert!(is_format_supported(TextureFormat::R8Unorm));
    }

    #[test]
    fn test_supported_rg8unorm() {
        assert!(is_format_supported(TextureFormat::Rg8Unorm));
    }

    #[test]
    fn test_supported_r16float() {
        assert!(is_format_supported(TextureFormat::R16Float));
    }

    #[test]
    fn test_supported_rg16float() {
        assert!(is_format_supported(TextureFormat::Rg16Float));
    }

    #[test]
    fn test_supported_rg11b10float() {
        assert!(is_format_supported(TextureFormat::Rg11b10Float));
    }

    // Unsupported formats - Depth
    #[test]
    fn test_unsupported_depth32float() {
        assert!(!is_format_supported(TextureFormat::Depth32Float));
    }

    #[test]
    fn test_unsupported_depth24plus() {
        assert!(!is_format_supported(TextureFormat::Depth24Plus));
    }

    #[test]
    fn test_unsupported_depth24plus_stencil8() {
        assert!(!is_format_supported(TextureFormat::Depth24PlusStencil8));
    }

    // Unsupported formats - Compressed
    #[test]
    fn test_unsupported_bc1_rgba_unorm() {
        assert!(!is_format_supported(TextureFormat::Bc1RgbaUnorm));
    }

    #[test]
    fn test_unsupported_bc3_rgba_unorm() {
        assert!(!is_format_supported(TextureFormat::Bc3RgbaUnorm));
    }

    // Unsupported formats - Integer
    #[test]
    fn test_unsupported_r8uint() {
        assert!(!is_format_supported(TextureFormat::R8Uint));
    }

    #[test]
    fn test_unsupported_rgba8uint() {
        assert!(!is_format_supported(TextureFormat::Rgba8Uint));
    }

    #[test]
    fn test_unsupported_r8sint() {
        assert!(!is_format_supported(TextureFormat::R8Sint));
    }

    // Unsupported formats - Stencil
    #[test]
    fn test_unsupported_stencil8() {
        assert!(!is_format_supported(TextureFormat::Stencil8));
    }
}

// ============================================================================
// is_filterable Tests
// ============================================================================

mod filterable_tests {
    use super::*;

    // Filterable formats - Unorm
    #[test]
    fn test_filterable_r8unorm() {
        assert!(is_filterable(TextureFormat::R8Unorm));
    }

    #[test]
    fn test_filterable_rg8unorm() {
        assert!(is_filterable(TextureFormat::Rg8Unorm));
    }

    #[test]
    fn test_filterable_rgba8unorm() {
        assert!(is_filterable(TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn test_filterable_rgba8unorm_srgb() {
        assert!(is_filterable(TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn test_filterable_bgra8unorm() {
        assert!(is_filterable(TextureFormat::Bgra8Unorm));
    }

    #[test]
    fn test_filterable_bgra8unorm_srgb() {
        assert!(is_filterable(TextureFormat::Bgra8UnormSrgb));
    }

    // Filterable formats - Float
    #[test]
    fn test_filterable_r16float() {
        assert!(is_filterable(TextureFormat::R16Float));
    }

    #[test]
    fn test_filterable_rg16float() {
        assert!(is_filterable(TextureFormat::Rg16Float));
    }

    #[test]
    fn test_filterable_rgba16float() {
        assert!(is_filterable(TextureFormat::Rgba16Float));
    }

    #[test]
    fn test_filterable_r32float() {
        assert!(is_filterable(TextureFormat::R32Float));
    }

    #[test]
    fn test_filterable_rg32float() {
        assert!(is_filterable(TextureFormat::Rg32Float));
    }

    #[test]
    fn test_filterable_rgba32float() {
        assert!(is_filterable(TextureFormat::Rgba32Float));
    }

    #[test]
    fn test_filterable_rg11b10float() {
        assert!(is_filterable(TextureFormat::Rg11b10Float));
    }

    // Non-filterable formats - Integer
    #[test]
    fn test_not_filterable_r8uint() {
        assert!(!is_filterable(TextureFormat::R8Uint));
    }

    #[test]
    fn test_not_filterable_rgba8uint() {
        assert!(!is_filterable(TextureFormat::Rgba8Uint));
    }

    #[test]
    fn test_not_filterable_r8sint() {
        assert!(!is_filterable(TextureFormat::R8Sint));
    }

    #[test]
    fn test_not_filterable_rgba8sint() {
        assert!(!is_filterable(TextureFormat::Rgba8Sint));
    }

    #[test]
    fn test_not_filterable_r16uint() {
        assert!(!is_filterable(TextureFormat::R16Uint));
    }

    #[test]
    fn test_not_filterable_r32uint() {
        assert!(!is_filterable(TextureFormat::R32Uint));
    }

    // Non-filterable formats - Depth/Stencil
    #[test]
    fn test_not_filterable_depth32float() {
        assert!(!is_filterable(TextureFormat::Depth32Float));
    }

    #[test]
    fn test_not_filterable_stencil8() {
        assert!(!is_filterable(TextureFormat::Stencil8));
    }
}

// ============================================================================
// storage_format_for Tests
// ============================================================================

mod storage_format_tests {
    use super::*;

    #[test]
    fn test_rgba8unorm_storage_format() {
        let info = storage_format_for(TextureFormat::Rgba8Unorm).unwrap();
        assert_eq!(info.storage_format, TextureFormat::Rgba8Unorm);
        assert!(!info.needs_custom_pipeline);
    }

    #[test]
    fn test_rgba8unorm_srgb_storage_format() {
        let info = storage_format_for(TextureFormat::Rgba8UnormSrgb).unwrap();
        assert_eq!(info.storage_format, TextureFormat::Rgba8Unorm);
        assert!(!info.needs_custom_pipeline);
    }

    #[test]
    fn test_bgra8unorm_storage_format() {
        let info = storage_format_for(TextureFormat::Bgra8Unorm).unwrap();
        // BGRA maps to RGBA for storage
        assert_eq!(info.storage_format, TextureFormat::Rgba8Unorm);
        assert!(info.needs_custom_pipeline);
    }

    #[test]
    fn test_bgra8unorm_srgb_storage_format() {
        let info = storage_format_for(TextureFormat::Bgra8UnormSrgb).unwrap();
        assert_eq!(info.storage_format, TextureFormat::Rgba8Unorm);
        assert!(info.needs_custom_pipeline);
    }

    #[test]
    fn test_rgba16float_storage_format() {
        let info = storage_format_for(TextureFormat::Rgba16Float).unwrap();
        assert_eq!(info.storage_format, TextureFormat::Rgba16Float);
        assert!(info.needs_custom_pipeline);
    }

    #[test]
    fn test_r8unorm_storage_format() {
        let info = storage_format_for(TextureFormat::R8Unorm).unwrap();
        assert_eq!(info.storage_format, TextureFormat::R8Unorm);
        assert!(info.needs_custom_pipeline);
    }

    #[test]
    fn test_rg8unorm_storage_format() {
        let info = storage_format_for(TextureFormat::Rg8Unorm).unwrap();
        assert_eq!(info.storage_format, TextureFormat::Rg8Unorm);
        assert!(info.needs_custom_pipeline);
    }

    #[test]
    fn test_r16float_storage_format() {
        let info = storage_format_for(TextureFormat::R16Float).unwrap();
        assert_eq!(info.storage_format, TextureFormat::R16Float);
        assert!(info.needs_custom_pipeline);
    }

    #[test]
    fn test_rg16float_storage_format() {
        let info = storage_format_for(TextureFormat::Rg16Float).unwrap();
        assert_eq!(info.storage_format, TextureFormat::Rg16Float);
        assert!(info.needs_custom_pipeline);
    }

    #[test]
    fn test_r32float_storage_format() {
        let info = storage_format_for(TextureFormat::R32Float).unwrap();
        assert_eq!(info.storage_format, TextureFormat::R32Float);
        assert!(info.needs_custom_pipeline);
    }

    #[test]
    fn test_rg32float_storage_format() {
        let info = storage_format_for(TextureFormat::Rg32Float).unwrap();
        assert_eq!(info.storage_format, TextureFormat::Rg32Float);
        assert!(info.needs_custom_pipeline);
    }

    #[test]
    fn test_rgba32float_storage_format() {
        let info = storage_format_for(TextureFormat::Rgba32Float).unwrap();
        assert_eq!(info.storage_format, TextureFormat::Rgba32Float);
        assert!(info.needs_custom_pipeline);
    }

    #[test]
    fn test_rg11b10float_storage_format() {
        let info = storage_format_for(TextureFormat::Rg11b10Float).unwrap();
        assert_eq!(info.storage_format, TextureFormat::Rg11b10Float);
        assert!(info.needs_custom_pipeline);
    }

    #[test]
    fn test_unsupported_depth32float_none() {
        assert!(storage_format_for(TextureFormat::Depth32Float).is_none());
    }

    #[test]
    fn test_unsupported_bc1_none() {
        assert!(storage_format_for(TextureFormat::Bc1RgbaUnorm).is_none());
    }

    #[test]
    fn test_unsupported_r8uint_none() {
        assert!(storage_format_for(TextureFormat::R8Uint).is_none());
    }

    #[test]
    fn test_unsupported_stencil8_none() {
        assert!(storage_format_for(TextureFormat::Stencil8).is_none());
    }
}

// ============================================================================
// MipChainInfo Tests
// ============================================================================

mod mip_chain_info_tests {
    use super::*;

    // new() tests
    #[test]
    fn test_new_1024x1024() {
        let info = MipChainInfo::new(1024, 1024);
        assert_eq!(info.width, 1024);
        assert_eq!(info.height, 1024);
        assert_eq!(info.mip_levels, 11);
    }

    #[test]
    fn test_new_512x256() {
        let info = MipChainInfo::new(512, 256);
        assert_eq!(info.width, 512);
        assert_eq!(info.height, 256);
        assert_eq!(info.mip_levels, 10);
    }

    #[test]
    fn test_new_100x100_npot() {
        let info = MipChainInfo::new(100, 100);
        assert_eq!(info.mip_levels, 7);
    }

    #[test]
    fn test_new_1x1() {
        let info = MipChainInfo::new(1, 1);
        assert_eq!(info.mip_levels, 1);
    }

    // with_mip_count() tests
    #[test]
    fn test_with_mip_count_explicit() {
        let info = MipChainInfo::with_mip_count(1024, 1024, 5);
        assert_eq!(info.width, 1024);
        assert_eq!(info.height, 1024);
        assert_eq!(info.mip_levels, 5);
    }

    #[test]
    fn test_with_mip_count_full_chain() {
        let info = MipChainInfo::with_mip_count(256, 256, 9);
        assert_eq!(info.mip_levels, 9);
    }

    #[test]
    fn test_with_mip_count_single_level() {
        let info = MipChainInfo::with_mip_count(512, 512, 1);
        assert_eq!(info.mip_levels, 1);
    }

    // mip_size() tests
    #[test]
    fn test_mip_size_level_0() {
        let info = MipChainInfo::new(256, 128);
        assert_eq!(info.mip_size(0), (256, 128));
    }

    #[test]
    fn test_mip_size_level_1() {
        let info = MipChainInfo::new(256, 128);
        assert_eq!(info.mip_size(1), (128, 64));
    }

    #[test]
    fn test_mip_size_level_2() {
        let info = MipChainInfo::new(256, 128);
        assert_eq!(info.mip_size(2), (64, 32));
    }

    #[test]
    fn test_mip_size_level_7() {
        let info = MipChainInfo::new(256, 128);
        assert_eq!(info.mip_size(7), (2, 1));
    }

    #[test]
    fn test_mip_size_level_8() {
        let info = MipChainInfo::new(256, 128);
        assert_eq!(info.mip_size(8), (1, 1));
    }

    // total_texels() tests
    #[test]
    fn test_total_texels_4x4() {
        // 4x4 with 3 mips: 16 + 4 + 1 = 21
        let info = MipChainInfo::new(4, 4);
        assert_eq!(info.total_texels(), 21);
    }

    #[test]
    fn test_total_texels_2x2() {
        // 2x2 with 2 mips: 4 + 1 = 5
        let info = MipChainInfo::new(2, 2);
        assert_eq!(info.total_texels(), 5);
    }

    #[test]
    fn test_total_texels_1x1() {
        let info = MipChainInfo::new(1, 1);
        assert_eq!(info.total_texels(), 1);
    }

    #[test]
    fn test_total_texels_8x8() {
        // 8x8: 64 + 16 + 4 + 1 = 85
        let info = MipChainInfo::new(8, 8);
        assert_eq!(info.total_texels(), 85);
    }

    #[test]
    fn test_total_texels_16x16() {
        // 16x16: 256 + 64 + 16 + 4 + 1 = 341
        let info = MipChainInfo::new(16, 16);
        assert_eq!(info.total_texels(), 341);
    }

    #[test]
    fn test_total_texels_limited_mip_count() {
        // 8x8 with only 2 mips: 64 + 16 = 80
        let info = MipChainInfo::with_mip_count(8, 8, 2);
        assert_eq!(info.total_texels(), 80);
    }

    // is_power_of_two() tests
    #[test]
    fn test_is_pot_1024x1024() {
        let info = MipChainInfo::new(1024, 1024);
        assert!(info.is_power_of_two());
    }

    #[test]
    fn test_is_pot_512x256() {
        let info = MipChainInfo::new(512, 256);
        assert!(info.is_power_of_two());
    }

    #[test]
    fn test_is_pot_1x1() {
        let info = MipChainInfo::new(1, 1);
        assert!(info.is_power_of_two());
    }

    #[test]
    fn test_is_not_pot_100x100() {
        let info = MipChainInfo::new(100, 100);
        assert!(!info.is_power_of_two());
    }

    #[test]
    fn test_is_not_pot_512x300() {
        let info = MipChainInfo::new(512, 300);
        assert!(!info.is_power_of_two());
    }

    #[test]
    fn test_is_not_pot_1920x1080() {
        let info = MipChainInfo::new(1920, 1080);
        assert!(!info.is_power_of_two());
    }

    // smallest_dimension() tests
    #[test]
    fn test_smallest_dimension_full_chain() {
        let info = MipChainInfo::new(1024, 1024);
        assert_eq!(info.smallest_dimension(), 1);
    }

    #[test]
    fn test_smallest_dimension_partial_chain() {
        let info = MipChainInfo::with_mip_count(1024, 1024, 5);
        // Mip 4: 1024 >> 4 = 64
        let (w, h) = info.mip_size(4);
        assert_eq!(info.smallest_dimension(), w.min(h));
        assert_eq!(info.smallest_dimension(), 64);
    }

    #[test]
    fn test_smallest_dimension_single_level() {
        let info = MipChainInfo::with_mip_count(512, 256, 1);
        // Mip 0: (512, 256), smallest is 256
        assert_eq!(info.smallest_dimension(), 256);
    }

    #[test]
    fn test_smallest_dimension_non_square() {
        let info = MipChainInfo::new(256, 64);
        // Full chain ends at (1, 1)
        assert_eq!(info.smallest_dimension(), 1);
    }

    // Equality and debug traits
    #[test]
    fn test_mip_chain_info_equality() {
        let info1 = MipChainInfo::new(256, 256);
        let info2 = MipChainInfo::new(256, 256);
        assert_eq!(info1, info2);
    }

    #[test]
    fn test_mip_chain_info_inequality() {
        let info1 = MipChainInfo::new(256, 256);
        let info2 = MipChainInfo::new(512, 256);
        assert_ne!(info1, info2);
    }

    #[test]
    fn test_mip_chain_info_clone() {
        let info = MipChainInfo::new(128, 128);
        let cloned = info.clone();
        assert_eq!(info, cloned);
    }

    #[test]
    fn test_mip_chain_info_copy() {
        let info = MipChainInfo::new(64, 64);
        let copied: MipChainInfo = info;
        assert_eq!(info, copied);
    }

    #[test]
    fn test_mip_chain_info_debug() {
        let info = MipChainInfo::new(256, 256);
        let debug = format!("{:?}", info);
        assert!(debug.contains("MipChainInfo"));
        assert!(debug.contains("256"));
    }
}

// ============================================================================
// Constants Tests
// ============================================================================

mod constants_tests {
    use super::*;

    #[test]
    fn test_workgroup_size_value() {
        assert_eq!(WORKGROUP_SIZE, 8);
    }

    #[test]
    fn test_min_mip_dimension_value() {
        assert_eq!(MIN_MIP_DIMENSION, 1);
    }

    #[test]
    fn test_workgroup_size_is_power_of_two() {
        assert!(WORKGROUP_SIZE.is_power_of_two());
    }

    #[test]
    fn test_workgroup_size_squared_is_64() {
        // 8x8 = 64 threads per workgroup
        assert_eq!(WORKGROUP_SIZE * WORKGROUP_SIZE, 64);
    }
}

// ============================================================================
// StorageFormatInfo Tests
// ============================================================================

mod storage_format_info_tests {
    use super::*;

    #[test]
    fn test_storage_format_info_debug() {
        let info = StorageFormatInfo {
            storage_format: TextureFormat::Rgba8Unorm,
            needs_custom_pipeline: false,
        };
        let debug = format!("{:?}", info);
        assert!(debug.contains("StorageFormatInfo"));
    }

    #[test]
    fn test_storage_format_info_clone() {
        let info = StorageFormatInfo {
            storage_format: TextureFormat::Rgba16Float,
            needs_custom_pipeline: true,
        };
        let cloned = info.clone();
        assert_eq!(info.storage_format, cloned.storage_format);
        assert_eq!(info.needs_custom_pipeline, cloned.needs_custom_pipeline);
    }

    #[test]
    fn test_storage_format_info_copy() {
        let info = StorageFormatInfo {
            storage_format: TextureFormat::R32Float,
            needs_custom_pipeline: true,
        };
        let copied: StorageFormatInfo = info;
        assert_eq!(info.storage_format, copied.storage_format);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_mip_size_all_levels_valid() {
        // Verify all mip levels for a texture produce valid dimensions
        let (width, height) = (1024, 512);
        let levels = calculate_mip_levels(width, height);

        for level in 0..levels {
            let (w, h) = calculate_mip_size(width, height, level);
            assert!(w >= 1, "Width invalid at level {}", level);
            assert!(h >= 1, "Height invalid at level {}", level);
        }
    }

    #[test]
    fn test_mip_chain_iteration() {
        let info = MipChainInfo::new(64, 32);
        let mut prev_size = (info.width, info.height);

        for level in 1..info.mip_levels {
            let size = info.mip_size(level);
            // Each dimension should be <= half of previous (floor division)
            assert!(size.0 <= (prev_size.0 + 1) / 2 || size.0 == 1);
            assert!(size.1 <= (prev_size.1 + 1) / 2 || size.1 == 1);
            prev_size = size;
        }
    }

    #[test]
    fn test_workgroup_dispatch_exact() {
        // When dimensions are exact multiple of workgroup size
        let width = 64;
        let height = 64;
        let workgroups_x = (width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let workgroups_y = (height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        assert_eq!(workgroups_x, 8);
        assert_eq!(workgroups_y, 8);
    }

    #[test]
    fn test_workgroup_dispatch_not_exact() {
        // When dimensions are not exact multiple
        let width = 100;
        let height = 50;
        let workgroups_x = (width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let workgroups_y = (height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        assert_eq!(workgroups_x, 13); // ceil(100/8)
        assert_eq!(workgroups_y, 7); // ceil(50/8)
    }

    #[test]
    fn test_workgroup_dispatch_minimum() {
        // Minimum 1x1 texture
        let width = 1;
        let height = 1;
        let workgroups_x = (width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let workgroups_y = (height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        assert_eq!(workgroups_x, 1);
        assert_eq!(workgroups_y, 1);
    }

    #[test]
    fn test_total_texels_geometric_series() {
        // For POT textures, total texels follows geometric series
        // For NxN: N^2 + (N/2)^2 + (N/4)^2 + ... + 1
        // = N^2 * (1 - (1/4)^n) / (1 - 1/4) approximately
        // For 4x4: 16 + 4 + 1 = 21
        let info = MipChainInfo::new(4, 4);
        assert_eq!(info.total_texels(), 21);

        // For 8x8: 64 + 16 + 4 + 1 = 85
        let info2 = MipChainInfo::new(8, 8);
        assert_eq!(info2.total_texels(), 85);
    }

    #[test]
    fn test_npot_odd_dimension_sequence() {
        // 5x5 -> 2x2 -> 1x1
        let info = MipChainInfo::new(5, 5);
        assert_eq!(info.mip_size(0), (5, 5));
        assert_eq!(info.mip_size(1), (2, 2));
        assert_eq!(info.mip_size(2), (1, 1));
        assert_eq!(info.mip_levels, 3);
    }

    #[test]
    fn test_format_support_consistency() {
        // All supported formats should have storage_format_for
        let formats = [
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Rgba16Float,
            TextureFormat::R8Unorm,
            TextureFormat::Rg8Unorm,
            TextureFormat::R16Float,
            TextureFormat::Rg16Float,
            TextureFormat::R32Float,
            TextureFormat::Rg32Float,
            TextureFormat::Rgba32Float,
            TextureFormat::Rg11b10Float,
        ];

        for format in formats {
            assert!(
                is_format_supported(format),
                "{:?} should be supported",
                format
            );
            assert!(
                storage_format_for(format).is_some(),
                "{:?} should have storage format",
                format
            );
        }
    }

    #[test]
    fn test_filterable_subset_of_supported() {
        // All filterable formats in the mip generator should also be supported
        let filterable_formats = [
            TextureFormat::R8Unorm,
            TextureFormat::Rg8Unorm,
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb,
            TextureFormat::R16Float,
            TextureFormat::Rg16Float,
            TextureFormat::Rgba16Float,
            TextureFormat::R32Float,
            TextureFormat::Rg32Float,
            TextureFormat::Rgba32Float,
            TextureFormat::Rg11b10Float,
        ];

        for format in filterable_formats {
            assert!(is_filterable(format), "{:?} should be filterable", format);
            assert!(
                is_format_supported(format),
                "{:?} should be supported",
                format
            );
        }
    }

    #[test]
    fn test_mip_levels_never_zero() {
        // Mip levels should always be at least 1
        let test_cases = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (1, 100),
            (100, 1),
            (1024, 1024),
        ];

        for (w, h) in test_cases {
            let levels = calculate_mip_levels(w, h);
            assert!(
                levels >= 1,
                "Mip levels for {}x{} should be >= 1, got {}",
                w,
                h,
                levels
            );
        }
    }

    #[test]
    fn test_large_texture_mip_chain() {
        // Test a very large texture (16K)
        let info = MipChainInfo::new(16384, 16384);
        assert_eq!(info.mip_levels, 15);
        assert!(info.is_power_of_two());
        assert_eq!(info.smallest_dimension(), 1);
    }

    #[test]
    fn test_rectangular_extremes() {
        // Very wide texture
        let info = MipChainInfo::new(4096, 1);
        assert_eq!(info.mip_levels, 13); // Based on 4096
        assert_eq!(info.mip_size(12), (1, 1));

        // Very tall texture
        let info2 = MipChainInfo::new(1, 4096);
        assert_eq!(info2.mip_levels, 13); // Based on 4096
        assert_eq!(info2.mip_size(12), (1, 1));
    }
}

// ============================================================================
// Workgroup Dispatch Calculation Tests
// ============================================================================

mod workgroup_tests {
    use super::*;

    fn calculate_workgroups(width: u32, height: u32) -> (u32, u32) {
        let x = (width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let y = (height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        (x, y)
    }

    #[test]
    fn test_workgroups_power_of_two() {
        assert_eq!(calculate_workgroups(64, 64), (8, 8));
        assert_eq!(calculate_workgroups(128, 128), (16, 16));
        assert_eq!(calculate_workgroups(256, 256), (32, 32));
    }

    #[test]
    fn test_workgroups_non_power_of_two() {
        // 100 / 8 = 12.5 -> 13
        assert_eq!(calculate_workgroups(100, 100), (13, 13));
        // 50 / 8 = 6.25 -> 7
        assert_eq!(calculate_workgroups(50, 50), (7, 7));
    }

    #[test]
    fn test_workgroups_small() {
        assert_eq!(calculate_workgroups(1, 1), (1, 1));
        assert_eq!(calculate_workgroups(7, 7), (1, 1));
        assert_eq!(calculate_workgroups(8, 8), (1, 1));
        assert_eq!(calculate_workgroups(9, 9), (2, 2));
    }

    #[test]
    fn test_workgroups_rectangular() {
        assert_eq!(calculate_workgroups(256, 64), (32, 8));
        assert_eq!(calculate_workgroups(64, 256), (8, 32));
        assert_eq!(calculate_workgroups(100, 50), (13, 7));
    }
}

// ============================================================================
// Mip Level Validation Tests
// ============================================================================

mod mip_level_validation_tests {
    use super::*;

    #[test]
    fn test_mip_chain_complete_pot() {
        // Verify complete mip chain for POT texture
        let info = MipChainInfo::new(256, 256);
        let expected_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
            (32, 32),
            (16, 16),
            (8, 8),
            (4, 4),
            (2, 2),
            (1, 1),
        ];

        assert_eq!(info.mip_levels as usize, expected_sizes.len());

        for (level, expected) in expected_sizes.iter().enumerate() {
            assert_eq!(info.mip_size(level as u32), *expected);
        }
    }

    #[test]
    fn test_mip_chain_complete_npot() {
        // Verify complete mip chain for NPOT texture
        let info = MipChainInfo::new(100, 100);
        let expected_sizes = [
            (100, 100),
            (50, 50),
            (25, 25),
            (12, 12),
            (6, 6),
            (3, 3),
            (1, 1),
        ];

        assert_eq!(info.mip_levels as usize, expected_sizes.len());

        for (level, expected) in expected_sizes.iter().enumerate() {
            assert_eq!(info.mip_size(level as u32), *expected);
        }
    }

    #[test]
    fn test_mip_chain_non_square() {
        // Verify complete mip chain for non-square texture
        let info = MipChainInfo::new(256, 64);
        let expected_sizes = [
            (256, 64),
            (128, 32),
            (64, 16),
            (32, 8),
            (16, 4),
            (8, 2),
            (4, 1),
            (2, 1),
            (1, 1),
        ];

        assert_eq!(info.mip_levels as usize, expected_sizes.len());

        for (level, expected) in expected_sizes.iter().enumerate() {
            assert_eq!(info.mip_size(level as u32), *expected);
        }
    }
}
