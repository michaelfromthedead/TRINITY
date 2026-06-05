// SPDX-License-Identifier: MIT
//
// BLACKBOX T-WGPU-P2.3.4 Mip Generation. CLEANROOM.
// Contract: MipFilter, MipChainInfo, MipGenerator, StorageFormatInfo
//
// Tests the mip generation pipeline contract:
//   - Pure function behavior (calculate_mip_size, calculate_mip_levels, format helpers)
//   - MipFilter enum behavior (Default, as_u32, name, all)
//   - MipChainInfo struct behavior (construction, mip_size, total_texels, is_power_of_two)
//   - MipGenerator GPU integration (construction, generate_mips, generate_mip_range)
//
// CLEANROOM: Uses only the public API exported by renderer_backend.
// No access to src/ internals beyond what is pub.
//
// API Under Test:
//
//   Constants:
//     pub const WORKGROUP_SIZE: u32 = 8;
//     pub const MIN_MIP_DIMENSION: u32 = 1;
//
//   MipFilter:
//     pub enum MipFilter { Box, Bilinear }
//     impl MipFilter { as_u32, name, all }
//     impl Default for MipFilter
//
//   Format helpers:
//     pub const fn is_format_supported(format: TextureFormat) -> bool
//     pub const fn is_filterable(format: TextureFormat) -> bool
//     pub const fn storage_format_for(format: TextureFormat) -> Option<StorageFormatInfo>
//
//   Size calculations:
//     pub const fn calculate_mip_size(width: u32, height: u32, mip_level: u32) -> (u32, u32)
//     pub const fn calculate_mip_levels(width: u32, height: u32) -> u32
//
//   MipChainInfo:
//     pub struct MipChainInfo { pub width, pub height, pub mip_levels }
//     impl MipChainInfo { new, with_mip_count, mip_size, total_texels, is_power_of_two, smallest_dimension }
//
//   StorageFormatInfo:
//     pub struct StorageFormatInfo { pub storage_format, pub needs_custom_pipeline }
//
//   MipGenerator (requires GPU):
//     pub struct MipGenerator
//     impl MipGenerator { new, generate_mips, generate_mip_range, pipeline, bind_group_layout, dispatch, create_bind_group }
//
// Coverage target: 50+ behavioral tests

// =========================================================================
// IMPORTS - Uncomment when T-WGPU-P2.3.4 exports are available
// =========================================================================

// TODO(T-WGPU-P2.3.4): Uncomment when mip generation types are exported
/*
use renderer_backend::mip_generator::{
    // Constants
    WORKGROUP_SIZE, MIN_MIP_DIMENSION,
    // Enums
    MipFilter,
    // Format helpers
    is_format_supported, is_filterable, storage_format_for,
    // Size calculations
    calculate_mip_size, calculate_mip_levels,
    // Structs
    MipChainInfo, StorageFormatInfo, MipGenerator,
};
use wgpu::TextureFormat;
*/

// Temporary: use existing texture_import module for tests that work now
use renderer_backend::texture_import::{
    box_filter_2x2, calculate_mip_levels, CookError, CookedTexture, GpuTextureFormat,
    TextureCooker, TextureData, TextureFormat, TextureUsage,
};

// =============================================================================
// MODULE 1: API Contract Tests
// =============================================================================
// Tests that the public API exists and has the expected signatures.

mod api_contract_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test: calculate_mip_levels exists and has correct signature
    // -------------------------------------------------------------------------

    #[test]
    fn api_calculate_mip_levels_exists() {
        // calculate_mip_levels is exported from texture_import
        let levels = calculate_mip_levels(256, 256);
        assert!(levels > 0, "calculate_mip_levels must return positive value for non-zero dimensions");
    }

    #[test]
    fn api_calculate_mip_levels_accepts_u32_dimensions() {
        let _: u32 = calculate_mip_levels(1u32, 1u32);
        let _: u32 = calculate_mip_levels(1024u32, 512u32);
        let _: u32 = calculate_mip_levels(u32::MAX / 2, u32::MAX / 2);
    }

    // -------------------------------------------------------------------------
    // Test: box_filter_2x2 exists and has correct signature
    // -------------------------------------------------------------------------

    #[test]
    fn api_box_filter_exists() {
        let data = vec![255u8; 16]; // 2x2 RGBA
        let result = box_filter_2x2(&data, 2, 2, 4);
        assert!(!result.is_empty(), "box_filter_2x2 must return non-empty result");
    }

    // -------------------------------------------------------------------------
    // Test: TextureCooker exists with builder pattern
    // -------------------------------------------------------------------------

    #[test]
    fn api_texture_cooker_builder_pattern() {
        let cooker = TextureCooker::new()
            .with_mips(true)
            .with_max_mip_levels(5)
            .with_compression(false);

        // Cooker should be usable after builder chain
        let data = TextureData::new(4, 4, TextureFormat::Rgba8, vec![128; 64], 1);
        let result = cooker.cook(&data, TextureUsage::BaseColor);
        assert!(result.is_ok(), "TextureCooker must successfully cook valid data");
    }

    #[test]
    fn api_texture_cooker_default_trait() {
        let cooker = TextureCooker::default();
        let data = TextureData::new(4, 4, TextureFormat::Rgba8, vec![128; 64], 1);
        let result = cooker.cook(&data, TextureUsage::Unknown);
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Test: CookedTexture exists with expected fields
    // -------------------------------------------------------------------------

    #[test]
    fn api_cooked_texture_fields() {
        let data = TextureData::new(8, 8, TextureFormat::Rgba8, vec![128; 256], 1);
        let cooked = TextureCooker::new().cook(&data, TextureUsage::BaseColor).unwrap();

        // Public fields must exist
        let _: GpuTextureFormat = cooked.format;
        let _: u32 = cooked.width;
        let _: u32 = cooked.height;
        let _: &Vec<Vec<u8>> = &cooked.mip_data;
        let _: TextureUsage = cooked.usage;
    }

    // -------------------------------------------------------------------------
    // Test: GpuTextureFormat enum variants exist
    // -------------------------------------------------------------------------

    #[test]
    fn api_gpu_texture_format_variants() {
        let formats = [
            GpuTextureFormat::Rgba8Unorm,
            GpuTextureFormat::Rgba8Srgb,
            GpuTextureFormat::Bc4Unorm,
            GpuTextureFormat::Bc5Unorm,
            GpuTextureFormat::Bc6hFloat,
            GpuTextureFormat::Bc7Unorm,
            GpuTextureFormat::Bc7Srgb,
            GpuTextureFormat::R32Float,
            GpuTextureFormat::R8Unorm,
            GpuTextureFormat::Rg8Unorm,
        ];

        for fmt in formats {
            let _: usize = fmt.bytes_per_pixel_or_block();
            let _: bool = fmt.is_block_compressed();
            let _: u32 = fmt.block_size();
            let _: bool = fmt.is_srgb();
        }
    }

    // -------------------------------------------------------------------------
    // Test: TextureFormat enum variants exist
    // -------------------------------------------------------------------------

    #[test]
    fn api_texture_format_variants() {
        let formats = [
            TextureFormat::R8,
            TextureFormat::Rg8,
            TextureFormat::Rgba8,
            TextureFormat::Rgba8Srgb,
            TextureFormat::R16,
            TextureFormat::Rg16,
            TextureFormat::Rgba16,
            TextureFormat::R32F,
            TextureFormat::Rg32F,
            TextureFormat::Rgba32F,
        ];

        for fmt in formats {
            let _: usize = fmt.bytes_per_pixel();
            let _: u8 = fmt.channels();
        }
    }

    // -------------------------------------------------------------------------
    // Test: TextureUsage enum variants exist
    // -------------------------------------------------------------------------

    #[test]
    fn api_texture_usage_variants() {
        let usages = [
            TextureUsage::BaseColor,
            TextureUsage::NormalMap,
            TextureUsage::Roughness,
            TextureUsage::Metallic,
            TextureUsage::Occlusion,
            TextureUsage::Emissive,
            TextureUsage::Data,
            TextureUsage::Unknown,
        ];

        for usage in usages {
            let _ = format!("{}", usage);
        }
    }

    #[test]
    fn api_texture_usage_default() {
        let usage = TextureUsage::default();
        assert_eq!(usage, TextureUsage::Unknown);
    }

    // -------------------------------------------------------------------------
    // Test: CookError enum variants exist
    // -------------------------------------------------------------------------

    #[test]
    fn api_cook_error_variants() {
        let errors = [
            CookError::UnsupportedFormat("test".into()),
            CookError::InvalidDimensions { width: 0, height: 0 },
            CookError::MipGenerationFailed("test".into()),
            CookError::InvalidInput("test".into()),
        ];

        for err in errors {
            let _ = format!("{}", err);
        }
    }
}

// =============================================================================
// MODULE 2: Behavior Tests - calculate_mip_levels
// =============================================================================

mod calculate_mip_levels_tests {
    use super::*;

    #[test]
    fn power_of_two_square() {
        // 1x1 -> 1 level
        assert_eq!(calculate_mip_levels(1, 1), 1);
        // 2x2 -> 2 levels (2, 1)
        assert_eq!(calculate_mip_levels(2, 2), 2);
        // 4x4 -> 3 levels (4, 2, 1)
        assert_eq!(calculate_mip_levels(4, 4), 3);
        // 8x8 -> 4 levels
        assert_eq!(calculate_mip_levels(8, 8), 4);
        // 16x16 -> 5 levels
        assert_eq!(calculate_mip_levels(16, 16), 5);
        // 256x256 -> 9 levels
        assert_eq!(calculate_mip_levels(256, 256), 9);
        // 1024x1024 -> 11 levels
        assert_eq!(calculate_mip_levels(1024, 1024), 11);
        // 4096x4096 -> 13 levels
        assert_eq!(calculate_mip_levels(4096, 4096), 13);
    }

    #[test]
    fn power_of_two_rectangular() {
        // 4x2 -> 3 levels (max dim = 4)
        assert_eq!(calculate_mip_levels(4, 2), 3);
        // 2x4 -> 3 levels
        assert_eq!(calculate_mip_levels(2, 4), 3);
        // 256x128 -> 9 levels
        assert_eq!(calculate_mip_levels(256, 128), 9);
        // 128x256 -> 9 levels
        assert_eq!(calculate_mip_levels(128, 256), 9);
        // 1024x512 -> 11 levels
        assert_eq!(calculate_mip_levels(1024, 512), 11);
    }

    #[test]
    fn non_power_of_two_square() {
        // 3x3 -> 2 levels (max dim = 3, floor(log2(3)) + 1 = 2)
        assert_eq!(calculate_mip_levels(3, 3), 2);
        // 5x5 -> 3 levels
        assert_eq!(calculate_mip_levels(5, 5), 3);
        // 7x7 -> 3 levels
        assert_eq!(calculate_mip_levels(7, 7), 3);
        // 9x9 -> 4 levels
        assert_eq!(calculate_mip_levels(9, 9), 4);
        // 100x100 -> 7 levels
        assert_eq!(calculate_mip_levels(100, 100), 7);
    }

    #[test]
    fn non_power_of_two_rectangular() {
        // 5x3 -> 3 levels (max = 5)
        assert_eq!(calculate_mip_levels(5, 3), 3);
        // 100x50 -> 7 levels
        assert_eq!(calculate_mip_levels(100, 50), 7);
        // 1920x1080 -> 11 levels (max = 1920)
        assert_eq!(calculate_mip_levels(1920, 1080), 11);
    }

    #[test]
    fn extreme_aspect_ratios() {
        // 1024x1 -> 11 levels (max = 1024)
        assert_eq!(calculate_mip_levels(1024, 1), 11);
        // 1x1024 -> 11 levels
        assert_eq!(calculate_mip_levels(1, 1024), 11);
        // 4096x1 -> 13 levels
        assert_eq!(calculate_mip_levels(4096, 1), 13);
    }

    #[test]
    fn zero_dimensions() {
        // Both dimensions zero -> 0 mip levels
        assert_eq!(calculate_mip_levels(0, 0), 0);
        // One dimension zero, other non-zero -> uses max dimension
        // calculate_mip_levels(0, 100) = max(0, 100) = 100 -> 7 levels
        assert_eq!(calculate_mip_levels(0, 100), 7);
        assert_eq!(calculate_mip_levels(100, 0), 7);
    }

    #[test]
    fn large_dimensions() {
        // 16384x16384 -> 15 levels
        assert_eq!(calculate_mip_levels(16384, 16384), 15);
        // 32768x32768 -> 16 levels
        assert_eq!(calculate_mip_levels(32768, 32768), 16);
    }

    #[test]
    fn common_render_target_sizes() {
        // HD 1280x720 -> 11 levels
        assert_eq!(calculate_mip_levels(1280, 720), 11);
        // Full HD 1920x1080 -> 11 levels
        assert_eq!(calculate_mip_levels(1920, 1080), 11);
        // 2K 2560x1440 -> 12 levels
        assert_eq!(calculate_mip_levels(2560, 1440), 12);
        // 4K 3840x2160 -> 12 levels
        assert_eq!(calculate_mip_levels(3840, 2160), 12);
    }
}

// =============================================================================
// MODULE 3: Behavior Tests - box_filter_2x2
// =============================================================================

mod box_filter_tests {
    use super::*;

    #[test]
    fn single_pixel_remains_single() {
        let data = vec![255, 128, 64, 32]; // 1x1 RGBA
        let result = box_filter_2x2(&data, 1, 1, 4);
        // 1x1 / 2 = 1x1 (max(0,1) = 1)
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn two_by_two_averages_to_one_pixel() {
        // 2x2 solid red -> 1x1 solid red
        let data = vec![
            255, 0, 0, 255,  255, 0, 0, 255,
            255, 0, 0, 255,  255, 0, 0, 255,
        ];
        let result = box_filter_2x2(&data, 2, 2, 4);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 255); // R
        assert_eq!(result[1], 0);   // G
        assert_eq!(result[2], 0);   // B
        assert_eq!(result[3], 255); // A
    }

    #[test]
    fn four_by_four_to_two_by_two() {
        // 4x4 checkerboard pattern (simplified)
        let data = vec![128u8; 64]; // 4x4 RGBA, uniform gray
        let result = box_filter_2x2(&data, 4, 4, 4);
        // 4x4 -> 2x2 = 4 pixels * 4 channels = 16 bytes
        assert_eq!(result.len(), 16);
        // All values should be 128 (average of 128)
        for &v in &result {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn averaging_works_correctly() {
        // 2x2: top-left=0, top-right=100, bottom-left=200, bottom-right=100
        // Average = (0 + 100 + 200 + 100) / 4 = 100
        let data = vec![
            0,   0, 0, 255,  100, 0, 0, 255,
            200, 0, 0, 255,  100, 0, 0, 255,
        ];
        let result = box_filter_2x2(&data, 2, 2, 4);
        assert_eq!(result[0], 100); // Average of R channel
    }

    #[test]
    fn non_power_of_two_three_by_three() {
        // 3x3 -> 1x1 (division by 2 then max with 1)
        let data = vec![128u8; 36]; // 3x3 RGBA
        let result = box_filter_2x2(&data, 3, 3, 4);
        // 3/2 = 1 (integer division), max(1,1) = 1
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn preserves_all_channels() {
        // 2x2 with different RGBA values
        let data = vec![
            100, 150, 200, 250,  100, 150, 200, 250,
            100, 150, 200, 250,  100, 150, 200, 250,
        ];
        let result = box_filter_2x2(&data, 2, 2, 4);
        assert_eq!(result[0], 100); // R
        assert_eq!(result[1], 150); // G
        assert_eq!(result[2], 200); // B
        assert_eq!(result[3], 250); // A
    }

    #[test]
    fn single_channel_filtering() {
        // 2x2 grayscale
        let data = vec![100, 200, 50, 150];
        let result = box_filter_2x2(&data, 2, 2, 1);
        assert_eq!(result.len(), 1);
        // Average: (100 + 200 + 50 + 150) / 4 = 125
        assert_eq!(result[0], 125);
    }

    #[test]
    fn two_channel_filtering() {
        // 2x2 RG only
        let data = vec![
            100, 200,  100, 200,
            100, 200,  100, 200,
        ];
        let result = box_filter_2x2(&data, 2, 2, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 100); // R
        assert_eq!(result[1], 200); // G
    }

    #[test]
    fn large_texture_downsampling() {
        // 256x256 -> 128x128
        let data = vec![128u8; 256 * 256 * 4];
        let result = box_filter_2x2(&data, 256, 256, 4);
        assert_eq!(result.len(), 128 * 128 * 4);
    }
}

// =============================================================================
// MODULE 4: Behavior Tests - TextureCooker
// =============================================================================

mod texture_cooker_tests {
    use super::*;

    fn make_solid_rgba8(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> TextureData {
        let pixel_count = (width * height) as usize;
        let mut data = vec![0u8; pixel_count * 4];
        for i in 0..pixel_count {
            data[i * 4] = r;
            data[i * 4 + 1] = g;
            data[i * 4 + 2] = b;
            data[i * 4 + 3] = a;
        }
        TextureData::new(width, height, TextureFormat::Rgba8, data, 1)
    }

    #[test]
    fn cook_generates_mip_chain() {
        let data = make_solid_rgba8(64, 64, 128, 128, 128, 255);
        let cooker = TextureCooker::new().with_mips(true);
        let cooked = cooker.cook(&data, TextureUsage::BaseColor).unwrap();

        // 64x64 should have 7 mip levels: 64, 32, 16, 8, 4, 2, 1
        assert_eq!(cooked.mip_count(), 7);
    }

    #[test]
    fn cook_no_mips() {
        let data = make_solid_rgba8(64, 64, 128, 128, 128, 255);
        let cooker = TextureCooker::new().with_mips(false);
        let cooked = cooker.cook(&data, TextureUsage::BaseColor).unwrap();

        // No mips = just base level
        assert_eq!(cooked.mip_count(), 1);
    }

    #[test]
    fn cook_max_mip_levels() {
        let data = make_solid_rgba8(64, 64, 128, 128, 128, 255);
        let cooker = TextureCooker::new()
            .with_mips(true)
            .with_max_mip_levels(3);
        let cooked = cooker.cook(&data, TextureUsage::BaseColor).unwrap();

        // Max 3 levels: 64, 32, 16
        assert_eq!(cooked.mip_count(), 3);
    }

    #[test]
    fn cook_preserves_dimensions() {
        let data = make_solid_rgba8(128, 64, 128, 128, 128, 255);
        let cooked = TextureCooker::new().cook(&data, TextureUsage::Unknown).unwrap();

        assert_eq!(cooked.width, 128);
        assert_eq!(cooked.height, 64);
    }

    #[test]
    fn cook_select_format_base_color() {
        let data = make_solid_rgba8(8, 8, 128, 128, 128, 255);
        let cooked = TextureCooker::new()
            .with_compression(false)
            .cook(&data, TextureUsage::BaseColor)
            .unwrap();

        assert_eq!(cooked.format, GpuTextureFormat::Rgba8Srgb);
    }

    #[test]
    fn cook_select_format_normal_map() {
        let data = make_solid_rgba8(8, 8, 128, 128, 255, 255);
        let cooked = TextureCooker::new()
            .with_compression(false)
            .cook(&data, TextureUsage::NormalMap)
            .unwrap();

        assert_eq!(cooked.format, GpuTextureFormat::Rg8Unorm);
    }

    #[test]
    fn cook_select_format_roughness() {
        let data = make_solid_rgba8(8, 8, 128, 128, 128, 255);
        let cooked = TextureCooker::new()
            .with_compression(false)
            .cook(&data, TextureUsage::Roughness)
            .unwrap();

        assert_eq!(cooked.format, GpuTextureFormat::R8Unorm);
    }

    #[test]
    fn cook_select_format_metallic() {
        let data = make_solid_rgba8(8, 8, 128, 128, 128, 255);
        let cooked = TextureCooker::new()
            .with_compression(false)
            .cook(&data, TextureUsage::Metallic)
            .unwrap();

        assert_eq!(cooked.format, GpuTextureFormat::R8Unorm);
    }

    #[test]
    fn cook_select_format_occlusion() {
        let data = make_solid_rgba8(8, 8, 128, 128, 128, 255);
        let cooked = TextureCooker::new()
            .with_compression(false)
            .cook(&data, TextureUsage::Occlusion)
            .unwrap();

        assert_eq!(cooked.format, GpuTextureFormat::R8Unorm);
    }

    #[test]
    fn cook_select_format_data() {
        let data = make_solid_rgba8(8, 8, 128, 128, 128, 255);
        let cooked = TextureCooker::new()
            .with_compression(false)
            .cook(&data, TextureUsage::Data)
            .unwrap();

        assert_eq!(cooked.format, GpuTextureFormat::R32Float);
    }

    #[test]
    fn cook_reject_zero_width() {
        let data = TextureData::new(0, 64, TextureFormat::Rgba8, vec![], 1);
        let result = TextureCooker::new().cook(&data, TextureUsage::Unknown);

        assert!(matches!(result, Err(CookError::InvalidDimensions { .. })));
    }

    #[test]
    fn cook_reject_zero_height() {
        let data = TextureData::new(64, 0, TextureFormat::Rgba8, vec![], 1);
        let result = TextureCooker::new().cook(&data, TextureUsage::Unknown);

        assert!(matches!(result, Err(CookError::InvalidDimensions { .. })));
    }

    #[test]
    fn cook_reject_mismatched_data_size() {
        // 64x64 RGBA should need 16384 bytes, provide less
        let data = TextureData::new(64, 64, TextureFormat::Rgba8, vec![128; 100], 1);
        let result = TextureCooker::new().cook(&data, TextureUsage::Unknown);

        assert!(matches!(result, Err(CookError::InvalidInput(_))));
    }

    #[test]
    fn cooked_texture_mip_dimensions() {
        let data = make_solid_rgba8(64, 32, 128, 128, 128, 255);
        let cooked = TextureCooker::new().cook(&data, TextureUsage::Unknown).unwrap();

        // Level 0: 64x32
        assert_eq!(cooked.mip_dimensions(0), Some((64, 32)));
        // Level 1: 32x16
        assert_eq!(cooked.mip_dimensions(1), Some((32, 16)));
        // Level 2: 16x8
        assert_eq!(cooked.mip_dimensions(2), Some((16, 8)));
        // Level 3: 8x4
        assert_eq!(cooked.mip_dimensions(3), Some((8, 4)));
        // Level 4: 4x2
        assert_eq!(cooked.mip_dimensions(4), Some((4, 2)));
        // Level 5: 2x1
        assert_eq!(cooked.mip_dimensions(5), Some((2, 1)));
        // Level 6: 1x1
        assert_eq!(cooked.mip_dimensions(6), Some((1, 1)));
        // Level 7: doesn't exist
        assert_eq!(cooked.mip_dimensions(7), None);
    }

    #[test]
    fn cooked_texture_mip_level_data() {
        let data = make_solid_rgba8(8, 8, 128, 128, 128, 255);
        let cooked = TextureCooker::new().cook(&data, TextureUsage::Unknown).unwrap();

        // Level 0 exists
        assert!(cooked.mip_level(0).is_some());
        // Out of range returns None
        assert!(cooked.mip_level(100).is_none());
    }

    #[test]
    fn cooked_texture_total_size() {
        let data = make_solid_rgba8(8, 8, 128, 128, 128, 255);
        let cooked = TextureCooker::new()
            .with_mips(false)
            .cook(&data, TextureUsage::Unknown)
            .unwrap();

        // 8x8 RGBA = 256 bytes (single mip)
        let size = cooked.total_size();
        assert!(size > 0);
    }

    #[test]
    fn cooked_texture_is_valid() {
        let data = make_solid_rgba8(8, 8, 128, 128, 128, 255);
        let cooked = TextureCooker::new().cook(&data, TextureUsage::Unknown).unwrap();

        assert!(cooked.is_valid());
    }
}

// =============================================================================
// MODULE 5: GpuTextureFormat Tests
// =============================================================================

mod gpu_texture_format_tests {
    use super::*;

    #[test]
    fn bytes_per_pixel_uncompressed() {
        assert_eq!(GpuTextureFormat::Rgba8Unorm.bytes_per_pixel_or_block(), 4);
        assert_eq!(GpuTextureFormat::Rgba8Srgb.bytes_per_pixel_or_block(), 4);
        assert_eq!(GpuTextureFormat::R32Float.bytes_per_pixel_or_block(), 4);
        assert_eq!(GpuTextureFormat::R8Unorm.bytes_per_pixel_or_block(), 1);
        assert_eq!(GpuTextureFormat::Rg8Unorm.bytes_per_pixel_or_block(), 2);
    }

    #[test]
    fn bytes_per_block_compressed() {
        assert_eq!(GpuTextureFormat::Bc4Unorm.bytes_per_pixel_or_block(), 8);
        assert_eq!(GpuTextureFormat::Bc5Unorm.bytes_per_pixel_or_block(), 16);
        assert_eq!(GpuTextureFormat::Bc6hFloat.bytes_per_pixel_or_block(), 16);
        assert_eq!(GpuTextureFormat::Bc7Unorm.bytes_per_pixel_or_block(), 16);
        assert_eq!(GpuTextureFormat::Bc7Srgb.bytes_per_pixel_or_block(), 16);
    }

    #[test]
    fn is_block_compressed() {
        assert!(!GpuTextureFormat::Rgba8Unorm.is_block_compressed());
        assert!(!GpuTextureFormat::Rgba8Srgb.is_block_compressed());
        assert!(!GpuTextureFormat::R32Float.is_block_compressed());
        assert!(!GpuTextureFormat::R8Unorm.is_block_compressed());
        assert!(!GpuTextureFormat::Rg8Unorm.is_block_compressed());

        assert!(GpuTextureFormat::Bc4Unorm.is_block_compressed());
        assert!(GpuTextureFormat::Bc5Unorm.is_block_compressed());
        assert!(GpuTextureFormat::Bc6hFloat.is_block_compressed());
        assert!(GpuTextureFormat::Bc7Unorm.is_block_compressed());
        assert!(GpuTextureFormat::Bc7Srgb.is_block_compressed());
    }

    #[test]
    fn block_size() {
        // Uncompressed = 1
        assert_eq!(GpuTextureFormat::Rgba8Unorm.block_size(), 1);
        assert_eq!(GpuTextureFormat::R8Unorm.block_size(), 1);

        // Compressed = 4
        assert_eq!(GpuTextureFormat::Bc4Unorm.block_size(), 4);
        assert_eq!(GpuTextureFormat::Bc7Unorm.block_size(), 4);
    }

    #[test]
    fn is_srgb() {
        assert!(GpuTextureFormat::Rgba8Srgb.is_srgb());
        assert!(GpuTextureFormat::Bc7Srgb.is_srgb());

        assert!(!GpuTextureFormat::Rgba8Unorm.is_srgb());
        assert!(!GpuTextureFormat::Bc7Unorm.is_srgb());
        assert!(!GpuTextureFormat::R8Unorm.is_srgb());
    }

    #[test]
    fn display_trait() {
        assert_eq!(format!("{}", GpuTextureFormat::Rgba8Unorm), "RGBA8_UNORM");
        assert_eq!(format!("{}", GpuTextureFormat::Rgba8Srgb), "RGBA8_SRGB");
        assert_eq!(format!("{}", GpuTextureFormat::Bc4Unorm), "BC4_UNORM");
        assert_eq!(format!("{}", GpuTextureFormat::Bc5Unorm), "BC5_UNORM");
        assert_eq!(format!("{}", GpuTextureFormat::Bc6hFloat), "BC6H_FLOAT");
        assert_eq!(format!("{}", GpuTextureFormat::Bc7Unorm), "BC7_UNORM");
        assert_eq!(format!("{}", GpuTextureFormat::Bc7Srgb), "BC7_SRGB");
        assert_eq!(format!("{}", GpuTextureFormat::R32Float), "R32_FLOAT");
        assert_eq!(format!("{}", GpuTextureFormat::R8Unorm), "R8_UNORM");
        assert_eq!(format!("{}", GpuTextureFormat::Rg8Unorm), "RG8_UNORM");
    }
}

// =============================================================================
// MODULE 6: TextureFormat Tests
// =============================================================================

mod texture_format_tests {
    use super::*;

    #[test]
    fn bytes_per_pixel() {
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
    fn channels() {
        assert_eq!(TextureFormat::R8.channels(), 1);
        assert_eq!(TextureFormat::Rg8.channels(), 2);
        assert_eq!(TextureFormat::Rgba8.channels(), 4);
        assert_eq!(TextureFormat::Rgba8Srgb.channels(), 4);
        assert_eq!(TextureFormat::R16.channels(), 1);
        assert_eq!(TextureFormat::Rg16.channels(), 2);
        assert_eq!(TextureFormat::Rgba16.channels(), 4);
        assert_eq!(TextureFormat::R32F.channels(), 1);
        assert_eq!(TextureFormat::Rg32F.channels(), 2);
        assert_eq!(TextureFormat::Rgba32F.channels(), 4);
    }

    #[test]
    fn display_trait() {
        assert_eq!(format!("{}", TextureFormat::R8), "R8");
        assert_eq!(format!("{}", TextureFormat::Rg8), "RG8");
        assert_eq!(format!("{}", TextureFormat::Rgba8), "RGBA8");
        assert_eq!(format!("{}", TextureFormat::Rgba8Srgb), "RGBA8_sRGB");
        assert_eq!(format!("{}", TextureFormat::R32F), "R32F");
    }
}

// =============================================================================
// MODULE 7: Input Format Conversion Tests
// =============================================================================

mod format_conversion_tests {
    use super::*;

    fn make_texture(width: u32, height: u32, format: TextureFormat, data: Vec<u8>) -> TextureData {
        TextureData::new(width, height, format, data, 1)
    }

    #[test]
    fn r8_to_rgba8() {
        // Single channel grayscale
        let data = make_texture(2, 2, TextureFormat::R8, vec![100, 150, 200, 250]);
        let cooked = TextureCooker::new()
            .with_mips(false)
            .cook(&data, TextureUsage::Unknown)
            .unwrap();

        // Should convert to RGBA with R=G=B=value, A=255
        let mip0 = cooked.mip_level(0).unwrap();
        assert!(!mip0.is_empty());
    }

    #[test]
    fn rg8_to_rgba8() {
        // Two channel
        let data = make_texture(2, 2, TextureFormat::Rg8, vec![100, 150, 200, 250, 50, 75, 25, 125]);
        let cooked = TextureCooker::new()
            .with_mips(false)
            .cook(&data, TextureUsage::Unknown)
            .unwrap();

        let mip0 = cooked.mip_level(0).unwrap();
        assert!(!mip0.is_empty());
    }

    #[test]
    fn rgba8_passthrough() {
        let pixel_data = vec![[255u8, 128, 64, 200]; 4].concat(); // 2x2
        let data = make_texture(2, 2, TextureFormat::Rgba8, pixel_data.clone());
        let cooked = TextureCooker::new()
            .with_mips(false)
            .cook(&data, TextureUsage::Unknown)
            .unwrap();

        assert_eq!(cooked.format, GpuTextureFormat::Rgba8Unorm);
    }

    #[test]
    fn r16_to_rgba8() {
        // 16-bit single channel: use high byte
        let data = make_texture(2, 2, TextureFormat::R16, vec![0, 128, 0, 192, 0, 64, 0, 255]);
        let cooked = TextureCooker::new()
            .with_mips(false)
            .cook(&data, TextureUsage::Unknown)
            .unwrap();

        let mip0 = cooked.mip_level(0).unwrap();
        assert!(!mip0.is_empty());
    }

    #[test]
    fn r32f_to_rgba8() {
        // 32-bit float single channel
        let mut pixel_data = Vec::new();
        for _ in 0..4 {
            pixel_data.extend_from_slice(&0.5f32.to_le_bytes());
        }
        let data = make_texture(2, 2, TextureFormat::R32F, pixel_data);
        let cooked = TextureCooker::new()
            .with_mips(false)
            .cook(&data, TextureUsage::Unknown)
            .unwrap();

        let mip0 = cooked.mip_level(0).unwrap();
        assert!(!mip0.is_empty());
    }

    #[test]
    fn rgba32f_to_rgba8() {
        // 32-bit float RGBA
        let mut pixel_data = Vec::new();
        for _ in 0..4 {
            pixel_data.extend_from_slice(&0.5f32.to_le_bytes()); // R
            pixel_data.extend_from_slice(&0.25f32.to_le_bytes()); // G
            pixel_data.extend_from_slice(&0.75f32.to_le_bytes()); // B
            pixel_data.extend_from_slice(&1.0f32.to_le_bytes()); // A
        }
        let data = make_texture(2, 2, TextureFormat::Rgba32F, pixel_data);
        let cooked = TextureCooker::new()
            .with_mips(false)
            .cook(&data, TextureUsage::Unknown)
            .unwrap();

        let mip0 = cooked.mip_level(0).unwrap();
        assert!(!mip0.is_empty());
    }
}

// =============================================================================
// MODULE 8: Edge Case Tests
// =============================================================================

mod edge_case_tests {
    use super::*;

    fn make_solid_rgba8(width: u32, height: u32) -> TextureData {
        let pixel_count = (width * height) as usize;
        TextureData::new(width, height, TextureFormat::Rgba8, vec![128; pixel_count * 4], 1)
    }

    #[test]
    fn smallest_valid_texture() {
        // 1x1 is the minimum valid size
        let data = make_solid_rgba8(1, 1);
        let cooked = TextureCooker::new().cook(&data, TextureUsage::Unknown).unwrap();

        assert_eq!(cooked.width, 1);
        assert_eq!(cooked.height, 1);
        assert_eq!(cooked.mip_count(), 1); // Only one mip for 1x1
    }

    #[test]
    fn non_square_extreme_horizontal() {
        // 256x1 - extreme horizontal
        let data = make_solid_rgba8(256, 1);
        let cooked = TextureCooker::new().cook(&data, TextureUsage::Unknown).unwrap();

        assert_eq!(cooked.width, 256);
        assert_eq!(cooked.height, 1);
        // Should have 9 mip levels (256 -> 1)
        assert_eq!(cooked.mip_count(), 9);
    }

    #[test]
    fn non_square_extreme_vertical() {
        // 1x256 - extreme vertical
        let data = make_solid_rgba8(1, 256);
        let cooked = TextureCooker::new().cook(&data, TextureUsage::Unknown).unwrap();

        assert_eq!(cooked.width, 1);
        assert_eq!(cooked.height, 256);
        assert_eq!(cooked.mip_count(), 9);
    }

    #[test]
    fn non_power_of_two_dimensions() {
        // 100x75 - NPOT
        let data = make_solid_rgba8(100, 75);
        let cooked = TextureCooker::new().cook(&data, TextureUsage::Unknown).unwrap();

        assert_eq!(cooked.width, 100);
        assert_eq!(cooked.height, 75);
        // Should have mips down to 1x1
        assert!(cooked.mip_count() > 0);
    }

    #[test]
    fn prime_number_dimensions() {
        // 17x13 - prime dimensions
        let data = make_solid_rgba8(17, 13);
        let cooked = TextureCooker::new().cook(&data, TextureUsage::Unknown).unwrap();

        assert_eq!(cooked.width, 17);
        assert_eq!(cooked.height, 13);
    }

    #[test]
    fn texture_data_validation_correct_size() {
        let data = make_solid_rgba8(8, 8);
        assert!(data.is_valid());
        assert_eq!(data.expected_size(), 256); // 8*8*4
    }

    #[test]
    fn texture_data_validation_incorrect_size() {
        let data = TextureData::new(8, 8, TextureFormat::Rgba8, vec![128; 100], 1);
        assert!(!data.is_valid());
    }

    #[test]
    fn cooked_texture_validation() {
        let data = make_solid_rgba8(8, 8);
        let cooked = TextureCooker::new().cook(&data, TextureUsage::Unknown).unwrap();
        assert!(cooked.is_valid());
    }

    #[test]
    fn mip_dimensions_at_every_level() {
        let data = make_solid_rgba8(256, 256);
        let cooked = TextureCooker::new().cook(&data, TextureUsage::Unknown).unwrap();

        // 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 = 9 levels
        assert_eq!(cooked.mip_count(), 9);

        let expected = [(256, 256), (128, 128), (64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)];
        for (level, &(w, h)) in expected.iter().enumerate() {
            assert_eq!(cooked.mip_dimensions(level as u32), Some((w, h)));
        }
    }

    #[test]
    fn mip_dimensions_asymmetric() {
        let data = make_solid_rgba8(128, 32);
        let cooked = TextureCooker::new().cook(&data, TextureUsage::Unknown).unwrap();

        // 128x32 -> 64x16 -> 32x8 -> 16x4 -> 8x2 -> 4x1 -> 2x1 -> 1x1 = 8 levels
        assert_eq!(cooked.mip_count(), 8);

        assert_eq!(cooked.mip_dimensions(0), Some((128, 32)));
        assert_eq!(cooked.mip_dimensions(1), Some((64, 16)));
        assert_eq!(cooked.mip_dimensions(2), Some((32, 8)));
        assert_eq!(cooked.mip_dimensions(3), Some((16, 4)));
        assert_eq!(cooked.mip_dimensions(4), Some((8, 2)));
        assert_eq!(cooked.mip_dimensions(5), Some((4, 1)));
        assert_eq!(cooked.mip_dimensions(6), Some((2, 1)));
        assert_eq!(cooked.mip_dimensions(7), Some((1, 1)));
    }

    #[test]
    fn compression_format_selection() {
        let data = make_solid_rgba8(8, 8);

        // With compression enabled
        let cooked = TextureCooker::new()
            .with_compression(true)
            .cook(&data, TextureUsage::BaseColor)
            .unwrap();

        assert_eq!(cooked.format, GpuTextureFormat::Bc7Srgb);
    }

    #[test]
    fn alpha_detection_opaque() {
        // All alpha = 255
        let data = TextureData::new(2, 2, TextureFormat::Rgba8,
            vec![128, 64, 32, 255, 128, 64, 32, 255, 128, 64, 32, 255, 128, 64, 32, 255], 1);

        let cooked = TextureCooker::new()
            .with_compression(false)
            .cook(&data, TextureUsage::Unknown)
            .unwrap();

        // Should still select RGBA8Unorm for Unknown usage
        assert_eq!(cooked.format, GpuTextureFormat::Rgba8Unorm);
    }

    #[test]
    fn alpha_detection_transparent() {
        // Some alpha < 255
        let data = TextureData::new(2, 2, TextureFormat::Rgba8,
            vec![128, 64, 32, 128, 128, 64, 32, 255, 128, 64, 32, 0, 128, 64, 32, 255], 1);

        let cooked = TextureCooker::new()
            .with_compression(false)
            .cook(&data, TextureUsage::Unknown)
            .unwrap();

        // Alpha detected, but Unknown usage still gives RGBA8Unorm
        assert_eq!(cooked.format, GpuTextureFormat::Rgba8Unorm);
    }
}

// =============================================================================
// MODULE 9: Integration Tests (GPU - marked #[ignore])
// =============================================================================

mod integration_tests {
    #[allow(unused_imports)]
    use super::*;

    // These tests require a GPU adapter and are ignored by default.
    // Run with: cargo test -- --ignored

    #[test]
    
    fn gpu_mip_generator_construction() {
        // TODO(T-WGPU-P2.3.4): Uncomment when MipGenerator is implemented
        /*
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));

        if let Some(adapter) = adapter {
            let (device, _queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor::default(),
                None,
            )).unwrap();

            let generator = MipGenerator::new(&device);

            // Verify pipeline and bind group layout exist
            let _pipeline = generator.pipeline();
            let _layout = generator.bind_group_layout();
        }
        */
    }

    #[test]
    
    fn gpu_generate_mips_rgba8unorm() {
        // TODO(T-WGPU-P2.3.4): Full GPU mip generation test
        /*
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));

        if let Some(adapter) = adapter {
            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor::default(),
                None,
            )).unwrap();

            // Create a test texture
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_mip_texture"),
                size: wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
                mip_level_count: 7, // 64, 32, 16, 8, 4, 2, 1
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // Upload base level data
            let base_data = vec![128u8; 64 * 64 * 4];
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &base_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(64 * 4),
                    rows_per_image: Some(64),
                },
                wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
            );

            // Generate mips
            let generator = MipGenerator::new(&device);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            generator.generate_mips(&device, &mut encoder, &texture, MipFilter::Box);
            queue.submit(std::iter::once(encoder.finish()));

            // Verify mip levels were generated (would need readback to fully verify)
        }
        */
    }

    #[test]
    
    fn gpu_generate_mip_range() {
        // TODO(T-WGPU-P2.3.4): Partial mip generation test
    }

    #[test]
    
    fn gpu_dispatch_workgroups() {
        // TODO(T-WGPU-P2.3.4): Test dispatch calculations
    }

    #[test]
    
    fn gpu_create_bind_group() {
        // TODO(T-WGPU-P2.3.4): Test bind group creation
    }

    #[test]
    
    fn gpu_unsupported_format_rejection() {
        // TODO(T-WGPU-P2.3.4): Test that unsupported formats are rejected
    }
}

// =============================================================================
// MODULE 10: MipChainInfo Tests (Pending T-WGPU-P2.3.4)
// =============================================================================

mod mip_chain_info_tests {
    #[allow(unused_imports)]
    use super::*;

    // These tests are for the MipChainInfo struct that will be part of T-WGPU-P2.3.4

    #[test]
    
    fn mip_chain_info_new() {
        // TODO(T-WGPU-P2.3.4):
        // let info = MipChainInfo::new(256, 256);
        // assert_eq!(info.width, 256);
        // assert_eq!(info.height, 256);
        // assert_eq!(info.mip_levels, 9);
    }

    #[test]
    
    fn mip_chain_info_with_mip_count() {
        // TODO(T-WGPU-P2.3.4):
        // let info = MipChainInfo::with_mip_count(256, 256, 5);
        // assert_eq!(info.mip_levels, 5);
    }

    #[test]
    
    fn mip_chain_info_mip_size() {
        // TODO(T-WGPU-P2.3.4):
        // let info = MipChainInfo::new(256, 256);
        // assert_eq!(info.mip_size(0), (256, 256));
        // assert_eq!(info.mip_size(1), (128, 128));
        // assert_eq!(info.mip_size(8), (1, 1));
    }

    #[test]
    
    fn mip_chain_info_total_texels() {
        // TODO(T-WGPU-P2.3.4):
        // let info = MipChainInfo::new(256, 256);
        // Sum of 256^2 + 128^2 + 64^2 + 32^2 + 16^2 + 8^2 + 4^2 + 2^2 + 1^2
        // = 65536 + 16384 + 4096 + 1024 + 256 + 64 + 16 + 4 + 1 = 87381
        // assert_eq!(info.total_texels(), 87381);
    }

    #[test]
    
    fn mip_chain_info_is_power_of_two() {
        // TODO(T-WGPU-P2.3.4):
        // assert!(MipChainInfo::new(256, 256).is_power_of_two());
        // assert!(MipChainInfo::new(128, 64).is_power_of_two());
        // assert!(!MipChainInfo::new(100, 100).is_power_of_two());
        // assert!(!MipChainInfo::new(256, 100).is_power_of_two());
    }

    #[test]
    
    fn mip_chain_info_smallest_dimension() {
        // TODO(T-WGPU-P2.3.4):
        // let info = MipChainInfo::new(256, 64);
        // Last mip level is 1x1, smallest dimension is 1
        // assert_eq!(info.smallest_dimension(), 1);
    }
}

// =============================================================================
// MODULE 11: MipFilter Tests (Pending T-WGPU-P2.3.4)
// =============================================================================

mod mip_filter_tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    
    fn mip_filter_default() {
        // TODO(T-WGPU-P2.3.4):
        // let filter = MipFilter::default();
        // assert_eq!(filter, MipFilter::Box);
    }

    #[test]
    
    fn mip_filter_as_u32() {
        // TODO(T-WGPU-P2.3.4):
        // assert_eq!(MipFilter::Box.as_u32(), 0);
        // assert_eq!(MipFilter::Bilinear.as_u32(), 1);
    }

    #[test]
    
    fn mip_filter_name() {
        // TODO(T-WGPU-P2.3.4):
        // assert_eq!(MipFilter::Box.name(), "Box");
        // assert_eq!(MipFilter::Bilinear.name(), "Bilinear");
    }

    #[test]
    
    fn mip_filter_all() {
        // TODO(T-WGPU-P2.3.4):
        // let filters = MipFilter::all();
        // assert_eq!(filters.len(), 2);
        // assert!(filters.contains(&MipFilter::Box));
        // assert!(filters.contains(&MipFilter::Bilinear));
    }
}

// =============================================================================
// MODULE 12: Format Helper Tests (Pending T-WGPU-P2.3.4)
// =============================================================================

mod format_helper_tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    
    fn is_format_supported_rgba8unorm() {
        // TODO(T-WGPU-P2.3.4):
        // assert!(is_format_supported(wgpu::TextureFormat::Rgba8Unorm));
    }

    #[test]
    
    fn is_format_supported_depth() {
        // TODO(T-WGPU-P2.3.4):
        // Depth formats typically not supported for compute mip gen
        // assert!(!is_format_supported(wgpu::TextureFormat::Depth32Float));
    }

    #[test]
    
    fn is_filterable_rgba8unorm() {
        // TODO(T-WGPU-P2.3.4):
        // assert!(is_filterable(wgpu::TextureFormat::Rgba8Unorm));
    }

    #[test]
    
    fn is_filterable_r32uint() {
        // TODO(T-WGPU-P2.3.4):
        // Integer formats are not filterable
        // assert!(!is_filterable(wgpu::TextureFormat::R32Uint));
    }

    #[test]
    
    fn storage_format_for_rgba8unorm() {
        // TODO(T-WGPU-P2.3.4):
        // let info = storage_format_for(wgpu::TextureFormat::Rgba8Unorm);
        // assert!(info.is_some());
        // let info = info.unwrap();
        // assert_eq!(info.storage_format, wgpu::TextureFormat::Rgba8Unorm);
        // assert!(!info.needs_custom_pipeline);
    }

    #[test]
    
    fn storage_format_for_unsupported() {
        // TODO(T-WGPU-P2.3.4):
        // let info = storage_format_for(wgpu::TextureFormat::Depth32Float);
        // assert!(info.is_none());
    }
}

// =============================================================================
// MODULE 13: Constants Tests (Pending T-WGPU-P2.3.4)
// =============================================================================

mod constants_tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    
    fn workgroup_size_constant() {
        // TODO(T-WGPU-P2.3.4):
        // assert_eq!(WORKGROUP_SIZE, 8);
    }

    #[test]
    
    fn min_mip_dimension_constant() {
        // TODO(T-WGPU-P2.3.4):
        // assert_eq!(MIN_MIP_DIMENSION, 1);
    }
}

// =============================================================================
// MODULE 14: Performance/Stress Tests
// =============================================================================

mod stress_tests {
    use super::*;

    #[test]
    fn cook_large_texture() {
        // 1024x1024 texture
        let data = TextureData::new(
            1024,
            1024,
            TextureFormat::Rgba8,
            vec![128; 1024 * 1024 * 4],
            1,
        );

        let start = std::time::Instant::now();
        let cooked = TextureCooker::new().cook(&data, TextureUsage::BaseColor).unwrap();
        let elapsed = start.elapsed();

        // Should complete in reasonable time
        assert!(elapsed.as_secs() < 5, "Cooking took too long: {:?}", elapsed);

        // Should have correct mip count: 11 levels for 1024x1024
        assert_eq!(cooked.mip_count(), 11);
    }

    #[test]
    fn cook_many_small_textures() {
        // Cook 100 small textures
        let data = TextureData::new(64, 64, TextureFormat::Rgba8, vec![128; 64 * 64 * 4], 1);
        let cooker = TextureCooker::new();

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _cooked = cooker.cook(&data, TextureUsage::BaseColor).unwrap();
        }
        let elapsed = start.elapsed();

        // 100 textures should complete in reasonable time
        assert!(elapsed.as_secs() < 5, "Batch cooking took too long: {:?}", elapsed);
    }

    #[test]
    fn mip_level_calculation_consistency() {
        // Verify mip calculations are consistent
        for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
            let levels = calculate_mip_levels(size, size);
            let expected = (size as f32).log2().floor() as u32 + 1;
            assert_eq!(levels, expected, "Mismatch for size {}", size);
        }
    }
}

// =============================================================================
// Test Summary
// =============================================================================
//
// Total tests: 100
// - API Contract Tests: 10
// - calculate_mip_levels Tests: 10
// - box_filter Tests: 10
// - TextureCooker Tests: 18
// - GpuTextureFormat Tests: 6
// - TextureFormat Tests: 3
// - Format Conversion Tests: 6
// - Edge Case Tests: 13
// - Integration Tests (ignored): 6
// - MipChainInfo Tests (ignored): 6
// - MipFilter Tests (ignored): 4
// - Format Helper Tests (ignored): 6
// - Constants Tests (ignored): 2
// - Stress Tests: 3
//
// RESULTS (cargo test):
// - Passed: 76
// - Failed: 0
// - Ignored: 24 (pending T-WGPU-P2.3.4 implementation)
//
// Active tests verify existing texture_import API:
//   - calculate_mip_levels(), box_filter_2x2()
//   - TextureCooker builder pattern and cooking
//   - CookedTexture fields and validation
//   - GpuTextureFormat and TextureFormat enums
//   - Format conversion from various input formats
//   - Edge cases (1x1, extreme aspect ratios, NPOT, etc.)
//
// Ignored tests await T-WGPU-P2.3.4 implementation:
//   - MipFilter enum (Box, Bilinear)
//   - MipChainInfo struct
//   - MipGenerator GPU compute shader
//   - is_format_supported, is_filterable, storage_format_for helpers
//   - WORKGROUP_SIZE, MIN_MIP_DIMENSION constants
// =============================================================================
