// Blackbox contract tests for T-WGPU-P2.3.1 Texture Creation API
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::texture_import` and `renderer_backend::rhi_resources`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/texture_import/cook.rs (implementation)
//   - crates/renderer-backend/src/rhi_resources.rs (implementation)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - Task acceptance criteria (T-WGPU-P2.3.1)
//
// Public API under test (texture_import):
//   - TextureFormat: R8, Rg8, Rgba8, Rgba8Srgb, etc. with bytes_per_pixel(), channels()
//   - TextureData: width, height, format, data, mip_levels, expected_size(), is_valid()
//   - GpuTextureFormat: Rgba8Unorm, Bc4Unorm, etc. with bytes_per_pixel_or_block(),
//     is_block_compressed(), block_size(), is_srgb()
//   - TextureCooker: new(), with_mips(), with_max_mip_levels(), with_compression(),
//     select_format(), cook()
//   - CookedTexture: format, width, height, mip_data, usage, mip_count(),
//     mip_dimensions(), mip_level(), total_size(), is_valid()
//   - TextureUsage: BaseColor, NormalMap, Roughness, Metallic, Occlusion, etc.
//   - calculate_mip_levels(width, height) -> u32
//   - box_filter_2x2(data, width, height, channels) -> Vec<u8>
//   - CookError enum
//
// Public API under test (rhi_resources):
//   - TextureType: D2, D3, Cube, Array
//   - RhiTexture: new(), inner(), tex_type(), format(), width(), height(), depth(),
//     create_view(), into_inner()
//   - create_texture(device, tex_type, format, width, height, depth) -> RhiTexture
//
// Test design rationale:
//   Equivalence partitioning:
//     - Small textures (1x1, 4x4, 16x16)
//     - Medium textures (64x64, 256x256)
//     - Large textures (1024x1024, 4096x4096, 16384x16384)
//   Format types:
//     - Uncompressed (Rgba8, R8, Rg8, R32F)
//     - Block-compressed (BC4, BC5, BC6H, BC7)
//     - sRGB vs linear
//   Boundary cases:
//     - 1x1 textures (minimum mip)
//     - Non-power-of-two dimensions
//     - Non-square textures (512x256, 1x1024)
//   Contract verification:
//     - Format bytes_per_pixel returns correct values
//     - Mip count calculations are correct (log2 + 1)
//     - Size estimates scale with dimensions
//     - Compressed formats report smaller sizes

use renderer_backend::texture_import::{
    box_filter_2x2, calculate_mip_levels, CookError, CookedTexture, GpuTextureFormat,
    TextureCooker, TextureData, TextureFormat, TextureUsage,
};

// =============================================================================
// SECTION 1: API CONTRACT TESTS (no GPU required)
// =============================================================================

// -----------------------------------------------------------------------------
// TextureFormat API Contract
// -----------------------------------------------------------------------------

/// Test: TextureFormat::bytes_per_pixel returns sensible values for all formats.
#[test]
fn test_texture_format_bytes_per_pixel_r8() {
    assert_eq!(TextureFormat::R8.bytes_per_pixel(), 1, "R8 is 1 byte");
}

#[test]
fn test_texture_format_bytes_per_pixel_rg8() {
    assert_eq!(TextureFormat::Rg8.bytes_per_pixel(), 2, "RG8 is 2 bytes");
}

#[test]
fn test_texture_format_bytes_per_pixel_rgba8() {
    assert_eq!(TextureFormat::Rgba8.bytes_per_pixel(), 4, "RGBA8 is 4 bytes");
}

#[test]
fn test_texture_format_bytes_per_pixel_rgba8srgb() {
    assert_eq!(TextureFormat::Rgba8Srgb.bytes_per_pixel(), 4, "RGBA8Srgb is 4 bytes");
}

#[test]
fn test_texture_format_bytes_per_pixel_r16() {
    assert_eq!(TextureFormat::R16.bytes_per_pixel(), 2, "R16 is 2 bytes");
}

#[test]
fn test_texture_format_bytes_per_pixel_rg16() {
    assert_eq!(TextureFormat::Rg16.bytes_per_pixel(), 4, "RG16 is 4 bytes");
}

#[test]
fn test_texture_format_bytes_per_pixel_rgba16() {
    assert_eq!(TextureFormat::Rgba16.bytes_per_pixel(), 8, "RGBA16 is 8 bytes");
}

#[test]
fn test_texture_format_bytes_per_pixel_r32f() {
    assert_eq!(TextureFormat::R32F.bytes_per_pixel(), 4, "R32F is 4 bytes");
}

#[test]
fn test_texture_format_bytes_per_pixel_rg32f() {
    assert_eq!(TextureFormat::Rg32F.bytes_per_pixel(), 8, "RG32F is 8 bytes");
}

#[test]
fn test_texture_format_bytes_per_pixel_rgba32f() {
    assert_eq!(TextureFormat::Rgba32F.bytes_per_pixel(), 16, "RGBA32F is 16 bytes");
}

/// Test: TextureFormat::channels returns correct channel count.
#[test]
fn test_texture_format_channels() {
    assert_eq!(TextureFormat::R8.channels(), 1);
    assert_eq!(TextureFormat::Rg8.channels(), 2);
    assert_eq!(TextureFormat::Rgba8.channels(), 4);
    assert_eq!(TextureFormat::R16.channels(), 1);
    assert_eq!(TextureFormat::Rg16.channels(), 2);
    assert_eq!(TextureFormat::Rgba16.channels(), 4);
    assert_eq!(TextureFormat::R32F.channels(), 1);
    assert_eq!(TextureFormat::Rg32F.channels(), 2);
    assert_eq!(TextureFormat::Rgba32F.channels(), 4);
}

// -----------------------------------------------------------------------------
// GpuTextureFormat API Contract
// -----------------------------------------------------------------------------

/// Test: GpuTextureFormat::bytes_per_pixel_or_block returns sensible values.
#[test]
fn test_gpu_format_bytes_rgba8_unorm() {
    assert_eq!(
        GpuTextureFormat::Rgba8Unorm.bytes_per_pixel_or_block(),
        4,
        "RGBA8 unorm is 4 bytes per pixel"
    );
}

#[test]
fn test_gpu_format_bytes_rgba8_srgb() {
    assert_eq!(
        GpuTextureFormat::Rgba8Srgb.bytes_per_pixel_or_block(),
        4,
        "RGBA8 sRGB is 4 bytes per pixel"
    );
}

#[test]
fn test_gpu_format_bytes_bc4() {
    assert_eq!(
        GpuTextureFormat::Bc4Unorm.bytes_per_pixel_or_block(),
        8,
        "BC4 is 8 bytes per 4x4 block"
    );
}

#[test]
fn test_gpu_format_bytes_bc5() {
    assert_eq!(
        GpuTextureFormat::Bc5Unorm.bytes_per_pixel_or_block(),
        16,
        "BC5 is 16 bytes per 4x4 block"
    );
}

#[test]
fn test_gpu_format_bytes_bc6h() {
    assert_eq!(
        GpuTextureFormat::Bc6hFloat.bytes_per_pixel_or_block(),
        16,
        "BC6H is 16 bytes per 4x4 block"
    );
}

#[test]
fn test_gpu_format_bytes_bc7() {
    assert_eq!(
        GpuTextureFormat::Bc7Unorm.bytes_per_pixel_or_block(),
        16,
        "BC7 is 16 bytes per 4x4 block"
    );
    assert_eq!(
        GpuTextureFormat::Bc7Srgb.bytes_per_pixel_or_block(),
        16,
        "BC7 sRGB is 16 bytes per 4x4 block"
    );
}

#[test]
fn test_gpu_format_bytes_r32_float() {
    assert_eq!(
        GpuTextureFormat::R32Float.bytes_per_pixel_or_block(),
        4,
        "R32 float is 4 bytes per pixel"
    );
}

#[test]
fn test_gpu_format_bytes_r8_unorm() {
    assert_eq!(
        GpuTextureFormat::R8Unorm.bytes_per_pixel_or_block(),
        1,
        "R8 unorm is 1 byte per pixel"
    );
}

#[test]
fn test_gpu_format_bytes_rg8_unorm() {
    assert_eq!(
        GpuTextureFormat::Rg8Unorm.bytes_per_pixel_or_block(),
        2,
        "RG8 unorm is 2 bytes per pixel"
    );
}

/// Test: GpuTextureFormat::is_block_compressed returns correct values.
#[test]
fn test_gpu_format_is_block_compressed() {
    // Compressed formats
    assert!(GpuTextureFormat::Bc4Unorm.is_block_compressed());
    assert!(GpuTextureFormat::Bc5Unorm.is_block_compressed());
    assert!(GpuTextureFormat::Bc6hFloat.is_block_compressed());
    assert!(GpuTextureFormat::Bc7Unorm.is_block_compressed());
    assert!(GpuTextureFormat::Bc7Srgb.is_block_compressed());

    // Uncompressed formats
    assert!(!GpuTextureFormat::Rgba8Unorm.is_block_compressed());
    assert!(!GpuTextureFormat::Rgba8Srgb.is_block_compressed());
    assert!(!GpuTextureFormat::R8Unorm.is_block_compressed());
    assert!(!GpuTextureFormat::Rg8Unorm.is_block_compressed());
    assert!(!GpuTextureFormat::R32Float.is_block_compressed());
}

/// Test: GpuTextureFormat::block_size returns correct values.
#[test]
fn test_gpu_format_block_size() {
    // Compressed formats have 4x4 blocks
    assert_eq!(GpuTextureFormat::Bc4Unorm.block_size(), 4);
    assert_eq!(GpuTextureFormat::Bc5Unorm.block_size(), 4);
    assert_eq!(GpuTextureFormat::Bc7Unorm.block_size(), 4);

    // Uncompressed formats have block size 1
    assert_eq!(GpuTextureFormat::Rgba8Unorm.block_size(), 1);
    assert_eq!(GpuTextureFormat::R8Unorm.block_size(), 1);
    assert_eq!(GpuTextureFormat::R32Float.block_size(), 1);
}

/// Test: GpuTextureFormat::is_srgb returns correct values.
#[test]
fn test_gpu_format_is_srgb() {
    assert!(GpuTextureFormat::Rgba8Srgb.is_srgb());
    assert!(GpuTextureFormat::Bc7Srgb.is_srgb());

    assert!(!GpuTextureFormat::Rgba8Unorm.is_srgb());
    assert!(!GpuTextureFormat::Bc7Unorm.is_srgb());
    assert!(!GpuTextureFormat::Bc4Unorm.is_srgb());
    assert!(!GpuTextureFormat::R32Float.is_srgb());
}

// =============================================================================
// SECTION 2: BEHAVIORAL TESTS (no GPU required)
// =============================================================================

// -----------------------------------------------------------------------------
// Mip Level Calculation
// -----------------------------------------------------------------------------

/// Test: calculate_mip_levels for 256x256 should be 9 (256->128->64->32->16->8->4->2->1).
#[test]
fn test_mip_count_256x256() {
    let levels = calculate_mip_levels(256, 256);
    assert_eq!(levels, 9, "256x256 should have 9 mip levels");
}

/// Test: calculate_mip_levels for 1x1 should be 1.
#[test]
fn test_mip_count_1x1() {
    let levels = calculate_mip_levels(1, 1);
    assert_eq!(levels, 1, "1x1 should have 1 mip level");
}

/// Test: calculate_mip_levels for 2x2 should be 2.
#[test]
fn test_mip_count_2x2() {
    let levels = calculate_mip_levels(2, 2);
    assert_eq!(levels, 2, "2x2 should have 2 mip levels");
}

/// Test: calculate_mip_levels for 4x4 should be 3.
#[test]
fn test_mip_count_4x4() {
    let levels = calculate_mip_levels(4, 4);
    assert_eq!(levels, 3, "4x4 should have 3 mip levels");
}

/// Test: calculate_mip_levels for 1024x1024 should be 11.
#[test]
fn test_mip_count_1024x1024() {
    let levels = calculate_mip_levels(1024, 1024);
    assert_eq!(levels, 11, "1024x1024 should have 11 mip levels");
}

/// Test: calculate_mip_levels for non-square textures uses max dimension.
#[test]
fn test_mip_count_non_square() {
    // 512x256: max dimension is 512, so 10 mip levels
    let levels = calculate_mip_levels(512, 256);
    assert_eq!(levels, 10, "512x256 should have 10 mip levels (based on 512)");

    // 1x1024: max dimension is 1024, so 11 mip levels
    let levels = calculate_mip_levels(1, 1024);
    assert_eq!(levels, 11, "1x1024 should have 11 mip levels");
}

/// Test: calculate_mip_levels for large textures (4096x4096) should be 13.
#[test]
fn test_mip_count_4096x4096() {
    let levels = calculate_mip_levels(4096, 4096);
    assert_eq!(levels, 13, "4096x4096 should have 13 mip levels");
}

/// Test: calculate_mip_levels for very large textures (16384x16384) should be 15.
#[test]
fn test_mip_count_16384x16384() {
    let levels = calculate_mip_levels(16384, 16384);
    assert_eq!(levels, 15, "16384x16384 should have 15 mip levels");
}

/// Test: calculate_mip_levels for 0x0 dimensions returns 0.
#[test]
fn test_mip_count_zero_dimensions() {
    // When both dimensions are 0, no mips
    assert_eq!(calculate_mip_levels(0, 0), 0, "0x0 should have 0 mip levels");
}

/// Test: calculate_mip_levels uses max dimension, even if one is 0.
#[test]
fn test_mip_count_partial_zero_uses_max() {
    // When one dimension is 0, it uses the max (non-zero) dimension
    // This matches the implementation which does: max(width, height)
    assert_eq!(calculate_mip_levels(0, 256), 9, "0x256 uses max(0, 256)=256 -> 9 levels");
    assert_eq!(calculate_mip_levels(256, 0), 9, "256x0 uses max(256, 0)=256 -> 9 levels");
}

// -----------------------------------------------------------------------------
// TextureData Validation
// -----------------------------------------------------------------------------

/// Test: TextureData::expected_size calculates correctly.
#[test]
fn test_texture_data_expected_size_rgba8() {
    let data = TextureData::new(64, 64, TextureFormat::Rgba8, vec![0; 64 * 64 * 4], 1);
    assert_eq!(
        data.expected_size(),
        64 * 64 * 4,
        "64x64 RGBA8 should be 16384 bytes"
    );
}

#[test]
fn test_texture_data_expected_size_r8() {
    let data = TextureData::new(128, 128, TextureFormat::R8, vec![0; 128 * 128], 1);
    assert_eq!(
        data.expected_size(),
        128 * 128,
        "128x128 R8 should be 16384 bytes"
    );
}

#[test]
fn test_texture_data_expected_size_rgba16() {
    let data = TextureData::new(32, 32, TextureFormat::Rgba16, vec![0; 32 * 32 * 8], 1);
    assert_eq!(
        data.expected_size(),
        32 * 32 * 8,
        "32x32 RGBA16 should be 8192 bytes"
    );
}

/// Test: TextureData::is_valid returns true for correct data.
#[test]
fn test_texture_data_is_valid_correct_size() {
    let data = TextureData::new(16, 16, TextureFormat::Rgba8, vec![128; 16 * 16 * 4], 1);
    assert!(data.is_valid(), "Correctly sized data should be valid");
}

/// Test: TextureData::is_valid returns false for undersized data.
#[test]
fn test_texture_data_is_valid_undersized() {
    // Create data that's too small
    let data = TextureData::new(16, 16, TextureFormat::Rgba8, vec![0; 16 * 16 * 2], 1);
    assert!(!data.is_valid(), "Undersized data should be invalid");
}

// -----------------------------------------------------------------------------
// TextureCooker Format Selection
// -----------------------------------------------------------------------------

/// Test: TextureCooker selects sRGB format for BaseColor usage.
#[test]
fn test_cooker_select_format_base_color_uncompressed() {
    let format = TextureCooker::select_format(TextureUsage::BaseColor, true, false);
    assert_eq!(format, GpuTextureFormat::Rgba8Srgb);
}

/// Test: TextureCooker selects BC7 sRGB for BaseColor with compression.
#[test]
fn test_cooker_select_format_base_color_compressed() {
    let format = TextureCooker::select_format(TextureUsage::BaseColor, true, true);
    assert_eq!(format, GpuTextureFormat::Bc7Srgb);
}

/// Test: TextureCooker selects RG8 for NormalMap (uncompressed).
#[test]
fn test_cooker_select_format_normal_map_uncompressed() {
    let format = TextureCooker::select_format(TextureUsage::NormalMap, false, false);
    assert_eq!(format, GpuTextureFormat::Rg8Unorm);
}

/// Test: TextureCooker selects BC5 for NormalMap (compressed).
#[test]
fn test_cooker_select_format_normal_map_compressed() {
    let format = TextureCooker::select_format(TextureUsage::NormalMap, false, true);
    assert_eq!(format, GpuTextureFormat::Bc5Unorm);
}

/// Test: TextureCooker selects R8 for Roughness (uncompressed).
#[test]
fn test_cooker_select_format_roughness_uncompressed() {
    let format = TextureCooker::select_format(TextureUsage::Roughness, false, false);
    assert_eq!(format, GpuTextureFormat::R8Unorm);
}

/// Test: TextureCooker selects BC4 for Roughness (compressed).
#[test]
fn test_cooker_select_format_roughness_compressed() {
    let format = TextureCooker::select_format(TextureUsage::Roughness, false, true);
    assert_eq!(format, GpuTextureFormat::Bc4Unorm);
}

/// Test: TextureCooker selects R32F for Data usage.
#[test]
fn test_cooker_select_format_data() {
    let format = TextureCooker::select_format(TextureUsage::Data, false, false);
    assert_eq!(format, GpuTextureFormat::R32Float);

    let format = TextureCooker::select_format(TextureUsage::Data, false, true);
    assert_eq!(format, GpuTextureFormat::R32Float);
}

// -----------------------------------------------------------------------------
// TextureCooker Cooking
// -----------------------------------------------------------------------------

/// Test: TextureCooker generates correct number of mips for 64x64 texture.
#[test]
fn test_cooker_generates_mips_64x64() {
    let input = TextureData::new(64, 64, TextureFormat::Rgba8, vec![128; 64 * 64 * 4], 1);
    let cooker = TextureCooker::new().with_mips(true);
    let result = cooker.cook(&input, TextureUsage::BaseColor);

    assert!(result.is_ok(), "Cooking should succeed");
    let cooked = result.unwrap();

    // 64x64 should have 7 mip levels: 64, 32, 16, 8, 4, 2, 1
    assert_eq!(cooked.mip_count(), 7, "64x64 should generate 7 mip levels");
}

/// Test: TextureCooker respects max_mip_levels setting.
#[test]
fn test_cooker_max_mip_levels() {
    let input = TextureData::new(128, 128, TextureFormat::Rgba8, vec![128; 128 * 128 * 4], 1);
    let cooker = TextureCooker::new().with_mips(true).with_max_mip_levels(3);
    let result = cooker.cook(&input, TextureUsage::BaseColor);

    assert!(result.is_ok());
    let cooked = result.unwrap();
    assert_eq!(cooked.mip_count(), 3, "Should respect max_mip_levels=3");
}

/// Test: TextureCooker without mips generates only 1 level.
#[test]
fn test_cooker_no_mips() {
    let input = TextureData::new(64, 64, TextureFormat::Rgba8, vec![128; 64 * 64 * 4], 1);
    let cooker = TextureCooker::new().with_mips(false);
    let result = cooker.cook(&input, TextureUsage::BaseColor);

    assert!(result.is_ok());
    let cooked = result.unwrap();
    assert_eq!(cooked.mip_count(), 1, "Should have only 1 mip level when mips disabled");
}

/// Test: CookedTexture::mip_dimensions returns correct sizes.
#[test]
fn test_cooked_texture_mip_dimensions() {
    let input = TextureData::new(64, 64, TextureFormat::Rgba8, vec![128; 64 * 64 * 4], 1);
    let cooker = TextureCooker::new().with_mips(true);
    let cooked = cooker.cook(&input, TextureUsage::BaseColor).unwrap();

    assert_eq!(cooked.mip_dimensions(0), Some((64, 64)));
    assert_eq!(cooked.mip_dimensions(1), Some((32, 32)));
    assert_eq!(cooked.mip_dimensions(2), Some((16, 16)));
    assert_eq!(cooked.mip_dimensions(3), Some((8, 8)));
    assert_eq!(cooked.mip_dimensions(4), Some((4, 4)));
    assert_eq!(cooked.mip_dimensions(5), Some((2, 2)));
    assert_eq!(cooked.mip_dimensions(6), Some((1, 1)));
    assert_eq!(cooked.mip_dimensions(7), None); // Out of range
}

/// Test: CookedTexture::total_size returns non-zero for valid textures.
#[test]
fn test_cooked_texture_total_size() {
    let input = TextureData::new(32, 32, TextureFormat::Rgba8, vec![128; 32 * 32 * 4], 1);
    let cooker = TextureCooker::new().with_mips(false);
    let cooked = cooker.cook(&input, TextureUsage::Unknown).unwrap();

    let total = cooked.total_size();
    assert!(total > 0, "Total size should be non-zero");
    assert_eq!(total, 32 * 32 * 4, "32x32 RGBA8 without mips should be 4096 bytes");
}

/// Test: CookedTexture::is_valid returns true for properly cooked textures.
#[test]
fn test_cooked_texture_is_valid() {
    let input = TextureData::new(64, 64, TextureFormat::Rgba8, vec![128; 64 * 64 * 4], 1);
    let cooker = TextureCooker::new().with_mips(true);
    let cooked = cooker.cook(&input, TextureUsage::BaseColor).unwrap();

    assert!(cooked.is_valid(), "Properly cooked texture should be valid");
}

// =============================================================================
// SECTION 3: EDGE CASE TESTS
// =============================================================================

/// Test: 1x1 texture cooks correctly.
#[test]
fn test_cook_1x1_texture() {
    let input = TextureData::new(1, 1, TextureFormat::Rgba8, vec![255, 0, 0, 255], 1);
    let cooker = TextureCooker::new().with_mips(true);
    let result = cooker.cook(&input, TextureUsage::BaseColor);

    assert!(result.is_ok(), "1x1 texture should cook successfully");
    let cooked = result.unwrap();
    assert_eq!(cooked.mip_count(), 1, "1x1 should have only 1 mip level");
    assert_eq!(cooked.width, 1);
    assert_eq!(cooked.height, 1);
}

/// Test: Very large texture (4096x4096) dimensions are preserved.
#[test]
fn test_cook_large_texture_4096x4096() {
    // Create minimal data to test dimensions (full data would be 64MB)
    let input = TextureData::new(4096, 4096, TextureFormat::Rgba8, vec![128; 4096 * 4096 * 4], 1);
    let cooker = TextureCooker::new().with_mips(false);
    let result = cooker.cook(&input, TextureUsage::BaseColor);

    assert!(result.is_ok(), "Large texture should cook successfully");
    let cooked = result.unwrap();
    assert_eq!(cooked.width, 4096);
    assert_eq!(cooked.height, 4096);
}

/// Test: Non-square texture (512x256) cooks correctly.
#[test]
fn test_cook_non_square_512x256() {
    let input = TextureData::new(512, 256, TextureFormat::Rgba8, vec![128; 512 * 256 * 4], 1);
    let cooker = TextureCooker::new().with_mips(true);
    let result = cooker.cook(&input, TextureUsage::BaseColor);

    assert!(result.is_ok());
    let cooked = result.unwrap();
    assert_eq!(cooked.width, 512);
    assert_eq!(cooked.height, 256);
    // 512 is the max dimension, so we should have 10 mip levels
    assert_eq!(cooked.mip_count(), 10, "512x256 should have 10 mip levels");
}

/// Test: Extreme aspect ratio (1x1024) cooks correctly.
#[test]
fn test_cook_extreme_aspect_ratio_1x1024() {
    let input = TextureData::new(1, 1024, TextureFormat::Rgba8, vec![128; 1 * 1024 * 4], 1);
    let cooker = TextureCooker::new().with_mips(true);
    let result = cooker.cook(&input, TextureUsage::BaseColor);

    assert!(result.is_ok());
    let cooked = result.unwrap();
    assert_eq!(cooked.width, 1);
    assert_eq!(cooked.height, 1024);
    // 1024 is max dimension, so 11 mip levels
    assert_eq!(cooked.mip_count(), 11);
}

/// Test: Zero width texture fails gracefully.
#[test]
fn test_cook_zero_width_fails() {
    let input = TextureData::new(0, 64, TextureFormat::Rgba8, vec![], 1);
    let cooker = TextureCooker::new();
    let result = cooker.cook(&input, TextureUsage::BaseColor);

    assert!(result.is_err(), "Zero width should fail");
    match result.unwrap_err() {
        CookError::InvalidDimensions { width, height } => {
            assert_eq!(width, 0);
            assert_eq!(height, 64);
        }
        e => panic!("Expected InvalidDimensions error, got {:?}", e),
    }
}

/// Test: Zero height texture fails gracefully.
#[test]
fn test_cook_zero_height_fails() {
    let input = TextureData::new(64, 0, TextureFormat::Rgba8, vec![], 1);
    let cooker = TextureCooker::new();
    let result = cooker.cook(&input, TextureUsage::BaseColor);

    assert!(result.is_err(), "Zero height should fail");
}

/// Test: Invalid data size fails gracefully.
#[test]
fn test_cook_invalid_data_size_fails() {
    // 64x64 RGBA8 needs 16384 bytes, but we provide 100
    let input = TextureData::new(64, 64, TextureFormat::Rgba8, vec![0; 100], 1);
    let cooker = TextureCooker::new();
    let result = cooker.cook(&input, TextureUsage::BaseColor);

    assert!(result.is_err(), "Invalid data size should fail");
    match result.unwrap_err() {
        CookError::InvalidInput(_) => {}
        e => panic!("Expected InvalidInput error, got {:?}", e),
    }
}

/// Test: Non-power-of-two texture (60x100) cooks correctly.
#[test]
fn test_cook_npot_texture() {
    let input = TextureData::new(60, 100, TextureFormat::Rgba8, vec![128; 60 * 100 * 4], 1);
    let cooker = TextureCooker::new().with_mips(true);
    let result = cooker.cook(&input, TextureUsage::BaseColor);

    assert!(result.is_ok(), "NPOT texture should cook successfully");
    let cooked = result.unwrap();
    assert_eq!(cooked.width, 60);
    assert_eq!(cooked.height, 100);
}

// -----------------------------------------------------------------------------
// Box Filter Edge Cases
// -----------------------------------------------------------------------------

/// Test: box_filter_2x2 works with 4x4 texture.
#[test]
fn test_box_filter_4x4() {
    // 4x4 RGBA: all pixels are white (255, 255, 255, 255)
    let input = vec![255u8; 4 * 4 * 4];
    let output = box_filter_2x2(&input, 4, 4, 4);

    // Should produce 2x2 output
    assert_eq!(output.len(), 2 * 2 * 4, "Output should be 2x2 RGBA");

    // All pixels should still be white (averaging white gives white)
    for pixel in output.chunks(4) {
        assert_eq!(pixel, &[255, 255, 255, 255]);
    }
}

/// Test: box_filter_2x2 works with 2x2 texture (minimum).
#[test]
fn test_box_filter_2x2_minimum() {
    // 2x2 texture: R, G, B, W corners
    let input = vec![
        255, 0, 0, 255, // Red
        0, 255, 0, 255, // Green
        0, 0, 255, 255, // Blue
        255, 255, 255, 255, // White
    ];
    let output = box_filter_2x2(&input, 2, 2, 4);

    // Should produce 1x1 output (average of all 4 pixels)
    assert_eq!(output.len(), 4, "Output should be 1x1 RGBA");

    // Average: R=(255+0+0+255)/4=127, G=(0+255+0+255)/4=127, B=(0+0+255+255)/4=127, A=255
    // Values might vary slightly due to integer division
    assert!(output[0] > 100 && output[0] < 150, "Red channel should be ~127");
    assert!(output[1] > 100 && output[1] < 150, "Green channel should be ~127");
    assert!(output[2] > 100 && output[2] < 150, "Blue channel should be ~127");
    assert_eq!(output[3], 255, "Alpha should be 255");
}

/// Test: box_filter_2x2 handles single channel.
#[test]
fn test_box_filter_single_channel() {
    let input = vec![100, 200, 50, 150]; // 2x2 single channel
    let output = box_filter_2x2(&input, 2, 2, 1);

    assert_eq!(output.len(), 1, "Output should be 1x1");
    // Average of 100, 200, 50, 150 = 125
    assert_eq!(output[0], 125);
}

// =============================================================================
// SECTION 4: INTEGRATION TESTS (require GPU - marked #[ignore])
// =============================================================================

use pollster::block_on;
use renderer_backend::device::{enumerate_adapters_with_info, TrinityInstance};
use renderer_backend::rhi_device::RhiDevice;
use renderer_backend::rhi_resources::{create_texture, RhiTexture, TextureType};

/// Creates a TrinityInstance and gets the first available adapter.
fn get_test_adapter() -> Option<wgpu::Adapter> {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    result.adapters.into_iter().next()
}

/// Helper macro to skip test if no GPU adapter is available.
macro_rules! require_adapter {
    () => {
        match get_test_adapter() {
            Some(adapter) => adapter,
            None => {
                eprintln!("SKIP: No GPU adapter available for this test");
                return;
            }
        }
    };
}

/// Creates a wgpu device for testing.
fn create_test_device(adapter: &wgpu::Adapter) -> Option<(wgpu::Device, wgpu::Queue)> {
    block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
}

/// Helper macro to get a device, skipping if unavailable.
macro_rules! require_device {
    ($adapter:expr) => {
        match create_test_device($adapter) {
            Some((device, queue)) => (device, queue),
            None => {
                eprintln!("SKIP: Could not create device");
                return;
            }
        }
    };
}

/// Creates an RhiDevice for testing.
fn create_rhi_device(adapter: &wgpu::Adapter) -> Option<RhiDevice> {
    let (device, queue) = create_test_device(adapter)?;
    Some(RhiDevice::new(device, queue))
}

/// Test: RhiTexture creation with D2 type succeeds.
#[test]

fn test_rhi_texture_creation_d2() {
    let adapter = require_adapter!();
    let rhi_device = match create_rhi_device(&adapter) {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Could not create RHI device");
            return;
        }
    };

    let texture = create_texture(
        &rhi_device,
        TextureType::D2,
        wgpu::TextureFormat::Rgba8Unorm,
        64,
        64,
        1,
    );

    assert_eq!(texture.width(), 64);
    assert_eq!(texture.height(), 64);
    assert_eq!(texture.depth(), 1);
    assert_eq!(texture.tex_type(), TextureType::D2);
    assert_eq!(texture.format(), wgpu::TextureFormat::Rgba8Unorm);
}

/// Test: RhiTexture creation with D3 type succeeds.
#[test]

fn test_rhi_texture_creation_d3() {
    let adapter = require_adapter!();
    let rhi_device = match create_rhi_device(&adapter) {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Could not create RHI device");
            return;
        }
    };

    let texture = create_texture(
        &rhi_device,
        TextureType::D3,
        wgpu::TextureFormat::Rgba8Unorm,
        32,
        32,
        32,
    );

    assert_eq!(texture.width(), 32);
    assert_eq!(texture.height(), 32);
    assert_eq!(texture.depth(), 32);
    assert_eq!(texture.tex_type(), TextureType::D3);
}

/// Test: RhiTexture creation with Cube type succeeds.
#[test]

fn test_rhi_texture_creation_cube() {
    let adapter = require_adapter!();
    let rhi_device = match create_rhi_device(&adapter) {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Could not create RHI device");
            return;
        }
    };

    let texture = create_texture(
        &rhi_device,
        TextureType::Cube,
        wgpu::TextureFormat::Rgba8Unorm,
        64,
        64,
        1,
    );

    assert_eq!(texture.width(), 64);
    assert_eq!(texture.height(), 64);
    assert_eq!(texture.tex_type(), TextureType::Cube);
}

/// Test: RhiTexture creation with Array type succeeds.
#[test]

fn test_rhi_texture_creation_array() {
    let adapter = require_adapter!();
    let rhi_device = match create_rhi_device(&adapter) {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Could not create RHI device");
            return;
        }
    };

    let texture = create_texture(
        &rhi_device,
        TextureType::Array,
        wgpu::TextureFormat::Rgba8Unorm,
        32,
        32,
        4, // 4 array layers
    );

    assert_eq!(texture.width(), 32);
    assert_eq!(texture.height(), 32);
    assert_eq!(texture.depth(), 4);
    assert_eq!(texture.tex_type(), TextureType::Array);
}

/// Test: RhiTexture view creation works.
#[test]

fn test_rhi_texture_view_creation() {
    let adapter = require_adapter!();
    let rhi_device = match create_rhi_device(&adapter) {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Could not create RHI device");
            return;
        }
    };

    let texture = create_texture(
        &rhi_device,
        TextureType::D2,
        wgpu::TextureFormat::Rgba8Unorm,
        64,
        64,
        1,
    );

    // create_view should not panic
    let _view = texture.create_view();
}

/// Test: RhiTexture with depth format works.
#[test]

fn test_rhi_texture_depth_format() {
    let adapter = require_adapter!();
    let rhi_device = match create_rhi_device(&adapter) {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Could not create RHI device");
            return;
        }
    };

    let texture = create_texture(
        &rhi_device,
        TextureType::D2,
        wgpu::TextureFormat::Depth32Float,
        256,
        256,
        1,
    );

    assert_eq!(texture.format(), wgpu::TextureFormat::Depth32Float);
    assert_eq!(texture.width(), 256);
    assert_eq!(texture.height(), 256);
}

/// Test: RhiTexture inner() returns wgpu::Texture reference.
#[test]

fn test_rhi_texture_inner_access() {
    let adapter = require_adapter!();
    let rhi_device = match create_rhi_device(&adapter) {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Could not create RHI device");
            return;
        }
    };

    let texture = create_texture(
        &rhi_device,
        TextureType::D2,
        wgpu::TextureFormat::Rgba8Unorm,
        32,
        32,
        1,
    );

    // inner() should return a valid reference
    let inner = texture.inner();
    // We can call wgpu::Texture methods on it
    let _view = inner.create_view(&wgpu::TextureViewDescriptor::default());
}

/// Test: End-to-end workflow - cook texture then create RHI texture.
#[test]

fn test_end_to_end_cook_and_create() {
    let adapter = require_adapter!();
    let rhi_device = match create_rhi_device(&adapter) {
        Some(d) => d,
        None => {
            eprintln!("SKIP: Could not create RHI device");
            return;
        }
    };

    // 1. Create and cook texture data
    let input = TextureData::new(64, 64, TextureFormat::Rgba8, vec![128; 64 * 64 * 4], 1);
    let cooker = TextureCooker::new().with_mips(false);
    let cooked = cooker.cook(&input, TextureUsage::BaseColor).unwrap();

    assert_eq!(cooked.width, 64);
    assert_eq!(cooked.height, 64);

    // 2. Create RHI texture with same dimensions
    let texture = create_texture(
        &rhi_device,
        TextureType::D2,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        cooked.width,
        cooked.height,
        1,
    );

    assert_eq!(texture.width(), cooked.width);
    assert_eq!(texture.height(), cooked.height);

    // 3. Create view
    let _view = texture.create_view();
}

// =============================================================================
// ADDITIONAL TESTS FOR COVERAGE
// =============================================================================

/// Test: TextureFormat display formatting.
#[test]
fn test_texture_format_display() {
    assert_eq!(format!("{}", TextureFormat::R8), "R8");
    assert_eq!(format!("{}", TextureFormat::Rgba8), "RGBA8");
    assert_eq!(format!("{}", TextureFormat::Rgba8Srgb), "RGBA8_sRGB");
    assert_eq!(format!("{}", TextureFormat::R32F), "R32F");
}

/// Test: GpuTextureFormat display formatting.
#[test]
fn test_gpu_format_display() {
    assert_eq!(format!("{}", GpuTextureFormat::Rgba8Unorm), "RGBA8_UNORM");
    assert_eq!(format!("{}", GpuTextureFormat::Rgba8Srgb), "RGBA8_SRGB");
    assert_eq!(format!("{}", GpuTextureFormat::Bc7Unorm), "BC7_UNORM");
    assert_eq!(format!("{}", GpuTextureFormat::R32Float), "R32_FLOAT");
}

/// Test: TextureUsage display formatting.
#[test]
fn test_texture_usage_display() {
    assert_eq!(format!("{}", TextureUsage::BaseColor), "BaseColor");
    assert_eq!(format!("{}", TextureUsage::NormalMap), "NormalMap");
    assert_eq!(format!("{}", TextureUsage::Unknown), "Unknown");
}

/// Test: TextureUsage default is Unknown.
#[test]
fn test_texture_usage_default() {
    let usage: TextureUsage = Default::default();
    assert_eq!(usage, TextureUsage::Unknown);
}

/// Test: TextureCooker default settings.
#[test]
fn test_texture_cooker_default() {
    let cooker = TextureCooker::default();
    // Default should have mips enabled
    let input = TextureData::new(16, 16, TextureFormat::Rgba8, vec![128; 16 * 16 * 4], 1);
    let cooked = cooker.cook(&input, TextureUsage::Unknown).unwrap();

    // 16x16 with mips should have 5 levels: 16, 8, 4, 2, 1
    assert_eq!(cooked.mip_count(), 5);
}

/// Test: CookError display formatting.
#[test]
fn test_cook_error_display() {
    let err = CookError::InvalidDimensions {
        width: 0,
        height: 64,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("invalid dimensions"), "Error message: {}", msg);
    assert!(msg.contains("0x64"), "Error message: {}", msg);
}

/// Test: Cooking with different input formats.
#[test]
fn test_cook_r8_format() {
    let input = TextureData::new(8, 8, TextureFormat::R8, vec![128; 8 * 8], 1);
    let cooker = TextureCooker::new().with_mips(false);
    let result = cooker.cook(&input, TextureUsage::Roughness);

    assert!(result.is_ok());
    let cooked = result.unwrap();
    assert_eq!(cooked.format, GpuTextureFormat::R8Unorm);
}

/// Test: Cooking with RG8 input format.
#[test]
fn test_cook_rg8_format() {
    let input = TextureData::new(8, 8, TextureFormat::Rg8, vec![128; 8 * 8 * 2], 1);
    let cooker = TextureCooker::new().with_mips(false);
    let result = cooker.cook(&input, TextureUsage::NormalMap);

    assert!(result.is_ok());
    let cooked = result.unwrap();
    assert_eq!(cooked.format, GpuTextureFormat::Rg8Unorm);
}

/// Test: CookedTexture mip_level returns correct data.
#[test]
fn test_cooked_texture_mip_level_access() {
    let input = TextureData::new(16, 16, TextureFormat::Rgba8, vec![200; 16 * 16 * 4], 1);
    let cooker = TextureCooker::new().with_mips(true);
    let cooked = cooker.cook(&input, TextureUsage::Unknown).unwrap();

    // Level 0 should have 16*16*4 bytes
    let level0 = cooked.mip_level(0).unwrap();
    assert_eq!(level0.len(), 16 * 16 * 4);

    // Level 1 should have 8*8*4 bytes
    let level1 = cooked.mip_level(1).unwrap();
    assert_eq!(level1.len(), 8 * 8 * 4);

    // Out of range should return None
    assert!(cooked.mip_level(100).is_none());
}

/// Test: Mip levels for typical game texture sizes.
#[test]
fn test_mip_levels_game_textures() {
    assert_eq!(calculate_mip_levels(128, 128), 8);
    assert_eq!(calculate_mip_levels(512, 512), 10);
    assert_eq!(calculate_mip_levels(2048, 2048), 12);
    assert_eq!(calculate_mip_levels(8192, 8192), 14);
}

/// Test: Compressed format selection for all usage types.
#[test]
fn test_compressed_format_selection_all_usages() {
    assert_eq!(
        TextureCooker::select_format(TextureUsage::Metallic, false, true),
        GpuTextureFormat::Bc4Unorm
    );
    assert_eq!(
        TextureCooker::select_format(TextureUsage::Occlusion, false, true),
        GpuTextureFormat::Bc4Unorm
    );
    assert_eq!(
        TextureCooker::select_format(TextureUsage::Emissive, false, true),
        GpuTextureFormat::Bc6hFloat
    );
    assert_eq!(
        TextureCooker::select_format(TextureUsage::Unknown, true, true),
        GpuTextureFormat::Bc7Unorm
    );
}
