//! Tests for the texture cooking pipeline.

use super::cook::*;
use super::{TextureData, TextureFormat};

// ---------------------------------------------------------------------------
// TextureUsage tests
// ---------------------------------------------------------------------------

#[test]
fn test_texture_usage_display() {
    assert_eq!(format!("{}", TextureUsage::BaseColor), "BaseColor");
    assert_eq!(format!("{}", TextureUsage::NormalMap), "NormalMap");
    assert_eq!(format!("{}", TextureUsage::Roughness), "Roughness");
    assert_eq!(format!("{}", TextureUsage::Metallic), "Metallic");
    assert_eq!(format!("{}", TextureUsage::Occlusion), "Occlusion");
    assert_eq!(format!("{}", TextureUsage::Emissive), "Emissive");
    assert_eq!(format!("{}", TextureUsage::Data), "Data");
    assert_eq!(format!("{}", TextureUsage::Unknown), "Unknown");
}

#[test]
fn test_texture_usage_default() {
    assert_eq!(TextureUsage::default(), TextureUsage::Unknown);
}

// ---------------------------------------------------------------------------
// GpuTextureFormat tests
// ---------------------------------------------------------------------------

#[test]
fn test_gpu_format_bytes_per_pixel() {
    assert_eq!(GpuTextureFormat::Rgba8Unorm.bytes_per_pixel_or_block(), 4);
    assert_eq!(GpuTextureFormat::R8Unorm.bytes_per_pixel_or_block(), 1);
    assert_eq!(GpuTextureFormat::Rg8Unorm.bytes_per_pixel_or_block(), 2);
    assert_eq!(GpuTextureFormat::R32Float.bytes_per_pixel_or_block(), 4);
    assert_eq!(GpuTextureFormat::Bc4Unorm.bytes_per_pixel_or_block(), 8);
    assert_eq!(GpuTextureFormat::Bc5Unorm.bytes_per_pixel_or_block(), 16);
    assert_eq!(GpuTextureFormat::Bc7Unorm.bytes_per_pixel_or_block(), 16);
}

#[test]
fn test_gpu_format_is_block_compressed() {
    assert!(!GpuTextureFormat::Rgba8Unorm.is_block_compressed());
    assert!(!GpuTextureFormat::R8Unorm.is_block_compressed());
    assert!(!GpuTextureFormat::R32Float.is_block_compressed());
    assert!(GpuTextureFormat::Bc4Unorm.is_block_compressed());
    assert!(GpuTextureFormat::Bc5Unorm.is_block_compressed());
    assert!(GpuTextureFormat::Bc6hFloat.is_block_compressed());
    assert!(GpuTextureFormat::Bc7Unorm.is_block_compressed());
    assert!(GpuTextureFormat::Bc7Srgb.is_block_compressed());
}

#[test]
fn test_gpu_format_block_size() {
    assert_eq!(GpuTextureFormat::Rgba8Unorm.block_size(), 1);
    assert_eq!(GpuTextureFormat::Bc4Unorm.block_size(), 4);
    assert_eq!(GpuTextureFormat::Bc7Srgb.block_size(), 4);
}

#[test]
fn test_gpu_format_is_srgb() {
    assert!(!GpuTextureFormat::Rgba8Unorm.is_srgb());
    assert!(GpuTextureFormat::Rgba8Srgb.is_srgb());
    assert!(!GpuTextureFormat::Bc7Unorm.is_srgb());
    assert!(GpuTextureFormat::Bc7Srgb.is_srgb());
}

#[test]
fn test_gpu_format_display() {
    assert_eq!(format!("{}", GpuTextureFormat::Rgba8Unorm), "RGBA8_UNORM");
    assert_eq!(format!("{}", GpuTextureFormat::Bc7Srgb), "BC7_SRGB");
    assert_eq!(format!("{}", GpuTextureFormat::R32Float), "R32_FLOAT");
}

// ---------------------------------------------------------------------------
// Format selection tests
// ---------------------------------------------------------------------------

#[test]
fn test_format_selection_base_color() {
    // Uncompressed
    let fmt = TextureCooker::select_format(TextureUsage::BaseColor, false, false);
    assert_eq!(fmt, GpuTextureFormat::Rgba8Srgb);

    // Compressed
    let fmt = TextureCooker::select_format(TextureUsage::BaseColor, false, true);
    assert_eq!(fmt, GpuTextureFormat::Bc7Srgb);
}

#[test]
fn test_format_selection_normal_map() {
    // Uncompressed
    let fmt = TextureCooker::select_format(TextureUsage::NormalMap, false, false);
    assert_eq!(fmt, GpuTextureFormat::Rg8Unorm);

    // Compressed
    let fmt = TextureCooker::select_format(TextureUsage::NormalMap, false, true);
    assert_eq!(fmt, GpuTextureFormat::Bc5Unorm);
}

#[test]
fn test_format_selection_roughness() {
    // Uncompressed
    let fmt = TextureCooker::select_format(TextureUsage::Roughness, false, false);
    assert_eq!(fmt, GpuTextureFormat::R8Unorm);

    // Compressed
    let fmt = TextureCooker::select_format(TextureUsage::Roughness, false, true);
    assert_eq!(fmt, GpuTextureFormat::Bc4Unorm);
}

#[test]
fn test_format_selection_metallic() {
    let fmt = TextureCooker::select_format(TextureUsage::Metallic, false, false);
    assert_eq!(fmt, GpuTextureFormat::R8Unorm);

    let fmt = TextureCooker::select_format(TextureUsage::Metallic, false, true);
    assert_eq!(fmt, GpuTextureFormat::Bc4Unorm);
}

#[test]
fn test_format_selection_occlusion() {
    let fmt = TextureCooker::select_format(TextureUsage::Occlusion, false, false);
    assert_eq!(fmt, GpuTextureFormat::R8Unorm);

    let fmt = TextureCooker::select_format(TextureUsage::Occlusion, false, true);
    assert_eq!(fmt, GpuTextureFormat::Bc4Unorm);
}

#[test]
fn test_format_selection_emissive() {
    let fmt = TextureCooker::select_format(TextureUsage::Emissive, false, false);
    assert_eq!(fmt, GpuTextureFormat::Rgba8Unorm); // Fallback for HDR

    let fmt = TextureCooker::select_format(TextureUsage::Emissive, false, true);
    assert_eq!(fmt, GpuTextureFormat::Bc6hFloat);
}

#[test]
fn test_format_selection_data() {
    let fmt = TextureCooker::select_format(TextureUsage::Data, false, false);
    assert_eq!(fmt, GpuTextureFormat::R32Float);

    let fmt = TextureCooker::select_format(TextureUsage::Data, false, true);
    assert_eq!(fmt, GpuTextureFormat::R32Float);
}

#[test]
fn test_format_selection_unknown() {
    let fmt = TextureCooker::select_format(TextureUsage::Unknown, true, false);
    assert_eq!(fmt, GpuTextureFormat::Rgba8Unorm);

    let fmt = TextureCooker::select_format(TextureUsage::Unknown, false, true);
    assert_eq!(fmt, GpuTextureFormat::Bc7Unorm);
}

// ---------------------------------------------------------------------------
// Mip generation tests
// ---------------------------------------------------------------------------

#[test]
fn test_mip_generation() {
    let data = TextureData::new(64, 64, TextureFormat::Rgba8, vec![128; 64 * 64 * 4], 1);
    let cooker = TextureCooker::new().with_mips(true);
    let cooked = cooker.cook(&data, TextureUsage::BaseColor).unwrap();

    // 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 = 7 levels
    assert_eq!(cooked.mip_count(), 7);
    assert_eq!(cooked.width, 64);
    assert_eq!(cooked.height, 64);
}

#[test]
fn test_mip_dimensions() {
    let data = TextureData::new(64, 32, TextureFormat::Rgba8, vec![128; 64 * 32 * 4], 1);
    let cooked = data.cook(TextureUsage::Unknown).unwrap();

    // 64x32 -> 32x16 -> 16x8 -> 8x4 -> 4x2 -> 2x1 -> 1x1 = 7 levels
    assert_eq!(cooked.mip_dimensions(0), Some((64, 32)));
    assert_eq!(cooked.mip_dimensions(1), Some((32, 16)));
    assert_eq!(cooked.mip_dimensions(2), Some((16, 8)));
    assert_eq!(cooked.mip_dimensions(3), Some((8, 4)));
    assert_eq!(cooked.mip_dimensions(4), Some((4, 2)));
    assert_eq!(cooked.mip_dimensions(5), Some((2, 1)));
    assert_eq!(cooked.mip_dimensions(6), Some((1, 1)));
    assert_eq!(cooked.mip_dimensions(7), None);
}

#[test]
fn test_cooker_no_mips() {
    let data = TextureData::new(16, 16, TextureFormat::Rgba8, vec![255; 16 * 16 * 4], 1);
    let cooker = TextureCooker::new().with_mips(false);
    let cooked = cooker.cook(&data, TextureUsage::BaseColor).unwrap();

    assert_eq!(cooked.mip_count(), 1);
    assert_eq!(cooked.mip_dimensions(0), Some((16, 16)));
}

#[test]
fn test_mip_max_levels() {
    let data = TextureData::new(256, 256, TextureFormat::Rgba8, vec![128; 256 * 256 * 4], 1);
    let cooker = TextureCooker::new().with_mips(true).with_max_mip_levels(3);
    let cooked = cooker.cook(&data, TextureUsage::BaseColor).unwrap();

    assert_eq!(cooked.mip_count(), 3);
    assert_eq!(cooked.mip_dimensions(0), Some((256, 256)));
    assert_eq!(cooked.mip_dimensions(1), Some((128, 128)));
    assert_eq!(cooked.mip_dimensions(2), Some((64, 64)));
}

// ---------------------------------------------------------------------------
// Cook roundtrip tests
// ---------------------------------------------------------------------------

#[test]
fn test_cook_roundtrip() {
    // Create test pattern
    let mut pixels = vec![0u8; 8 * 8 * 4];
    for y in 0..8u32 {
        for x in 0..8u32 {
            let idx = ((y * 8 + x) * 4) as usize;
            pixels[idx] = (x * 32) as u8;
            pixels[idx + 1] = (y * 32) as u8;
            pixels[idx + 2] = 128;
            pixels[idx + 3] = 255;
        }
    }

    let data = TextureData::new(8, 8, TextureFormat::Rgba8, pixels.clone(), 1);
    let cooked = data.cook(TextureUsage::BaseColor).unwrap();

    assert!(cooked.is_valid());
    assert_eq!(cooked.width, 8);
    assert_eq!(cooked.height, 8);

    // Level 0 should match input (for RGBA8 -> RGBA8 srgb)
    let level0 = cooked.mip_level(0).unwrap();
    assert_eq!(level0.len(), 8 * 8 * 4);
    // Data should be preserved
    assert_eq!(&level0[..], &pixels[..]);
}

#[test]
fn test_cook_single_channel() {
    let data = TextureData::new(4, 4, TextureFormat::R8, vec![128; 16], 1);
    let cooked = data.cook(TextureUsage::Roughness).unwrap();

    assert_eq!(cooked.format, GpuTextureFormat::R8Unorm);
    assert!(cooked.is_valid());
}

#[test]
fn test_cook_two_channel() {
    let data = TextureData::new(4, 4, TextureFormat::Rg8, vec![128; 32], 1);
    let cooked = data.cook(TextureUsage::NormalMap).unwrap();

    assert_eq!(cooked.format, GpuTextureFormat::Rg8Unorm);
    assert!(cooked.is_valid());
}

// ---------------------------------------------------------------------------
// Box filter tests
// ---------------------------------------------------------------------------

#[test]
fn test_box_filter_2x2_basic() {
    // 4x4 RGBA with constant values
    let input = vec![100u8; 4 * 4 * 4];
    let output = box_filter_2x2(&input, 4, 4, 4);

    assert_eq!(output.len(), 2 * 2 * 4);
    // All values should be 100 (average of 100s)
    for v in output {
        assert_eq!(v, 100);
    }
}

#[test]
fn test_box_filter_2x2_averaging() {
    // 2x2 RGBA: corners with different values
    let input = vec![
        0, 0, 0, 255, // top-left: black
        255, 255, 255, 255, // top-right: white
        255, 255, 255, 255, // bottom-left: white
        0, 0, 0, 255, // bottom-right: black
    ];
    let output = box_filter_2x2(&input, 2, 2, 4);

    assert_eq!(output.len(), 4);
    // Average of 0, 255, 255, 0 = 127.5 -> 127
    assert_eq!(output[0], 127);
    assert_eq!(output[1], 127);
    assert_eq!(output[2], 127);
    assert_eq!(output[3], 255);
}

#[test]
fn test_box_filter_minimum_size() {
    // 1x1 input -> 1x1 output
    let input = vec![50, 100, 150, 200];
    let output = box_filter_2x2(&input, 1, 1, 4);

    assert_eq!(output.len(), 4);
    assert_eq!(output, vec![50, 100, 150, 200]);
}

// ---------------------------------------------------------------------------
// calculate_mip_levels tests
// ---------------------------------------------------------------------------

#[test]
fn test_calculate_mip_levels() {
    assert_eq!(calculate_mip_levels(1, 1), 1);
    assert_eq!(calculate_mip_levels(2, 2), 2);
    assert_eq!(calculate_mip_levels(4, 4), 3);
    assert_eq!(calculate_mip_levels(8, 8), 4);
    assert_eq!(calculate_mip_levels(256, 256), 9);
    assert_eq!(calculate_mip_levels(512, 256), 10);
    assert_eq!(calculate_mip_levels(0, 0), 0);
}

// ---------------------------------------------------------------------------
// Error handling tests
// ---------------------------------------------------------------------------

#[test]
fn test_cook_zero_dimensions() {
    let data = TextureData::new(0, 10, TextureFormat::Rgba8, vec![], 1);
    let result = data.cook(TextureUsage::BaseColor);
    assert!(matches!(result, Err(CookError::InvalidDimensions { .. })));

    let data = TextureData::new(10, 0, TextureFormat::Rgba8, vec![], 1);
    let result = data.cook(TextureUsage::BaseColor);
    assert!(matches!(result, Err(CookError::InvalidDimensions { .. })));
}

#[test]
fn test_cook_invalid_data_size() {
    let data = TextureData::new(4, 4, TextureFormat::Rgba8, vec![0; 10], 1); // Wrong size
    let result = data.cook(TextureUsage::BaseColor);
    assert!(matches!(result, Err(CookError::InvalidInput(_))));
}

#[test]
fn test_cook_error_display() {
    let err = CookError::UnsupportedFormat("XYZ".to_string());
    assert_eq!(format!("{}", err), "unsupported format for cooking: XYZ");

    let err = CookError::InvalidDimensions {
        width: 0,
        height: 10,
    };
    assert_eq!(format!("{}", err), "invalid dimensions: 0x10");

    let err = CookError::MipGenerationFailed("test".to_string());
    assert_eq!(format!("{}", err), "mip generation failed: test");

    let err = CookError::InvalidInput("bad data".to_string());
    assert_eq!(format!("{}", err), "invalid input: bad data");
}

// ---------------------------------------------------------------------------
// CookedTexture validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_cooked_texture_total_size() {
    let data = TextureData::new(8, 8, TextureFormat::Rgba8, vec![128; 8 * 8 * 4], 1);
    let cooked = data.cook(TextureUsage::BaseColor).unwrap();

    // 8x8 + 4x4 + 2x2 + 1x1 = 64 + 16 + 4 + 1 = 85 pixels * 4 bytes = 340
    let expected = (64 + 16 + 4 + 1) * 4;
    assert_eq!(cooked.total_size(), expected);
}

#[test]
fn test_cooked_texture_mip_level_access() {
    let data = TextureData::new(4, 4, TextureFormat::Rgba8, vec![100; 4 * 4 * 4], 1);
    let cooked = data.cook(TextureUsage::Unknown).unwrap();

    assert!(cooked.mip_level(0).is_some());
    assert!(cooked.mip_level(1).is_some());
    assert!(cooked.mip_level(2).is_some());
    assert!(cooked.mip_level(3).is_none());
}

// ---------------------------------------------------------------------------
// TextureCooker builder tests
// ---------------------------------------------------------------------------

#[test]
fn test_cooker_builder() {
    let cooker = TextureCooker::new()
        .with_mips(false)
        .with_max_mip_levels(5)
        .with_compression(true);

    assert!(!cooker.generate_mips);
    assert_eq!(cooker.max_mip_levels, 5);
    assert!(cooker.use_compression);
}

#[test]
fn test_cooker_default() {
    let cooker = TextureCooker::default();
    assert!(cooker.generate_mips);
    assert_eq!(cooker.max_mip_levels, 0);
    assert!(!cooker.use_compression);
}

// ---------------------------------------------------------------------------
// Alpha detection tests
// ---------------------------------------------------------------------------

#[test]
fn test_alpha_detection_opaque() {
    let mut pixels = vec![0u8; 4 * 4 * 4];
    for i in 0..16 {
        pixels[i * 4 + 3] = 255; // All opaque
    }
    let data = TextureData::new(4, 4, TextureFormat::Rgba8, pixels, 1);
    assert!(!TextureCooker::detect_alpha(&data));
}

#[test]
fn test_alpha_detection_transparent() {
    let mut pixels = vec![0u8; 4 * 4 * 4];
    for i in 0..16 {
        pixels[i * 4 + 3] = if i == 0 { 128 } else { 255 }; // One transparent pixel
    }
    let data = TextureData::new(4, 4, TextureFormat::Rgba8, pixels, 1);
    assert!(TextureCooker::detect_alpha(&data));
}

// ---------------------------------------------------------------------------
// Compression output tests
// ---------------------------------------------------------------------------

#[test]
fn test_cook_with_compression() {
    let data = TextureData::new(8, 8, TextureFormat::Rgba8, vec![128; 8 * 8 * 4], 1);
    let cooker = TextureCooker::new().with_compression(true).with_mips(false);
    let cooked = cooker.cook(&data, TextureUsage::BaseColor).unwrap();

    assert_eq!(cooked.format, GpuTextureFormat::Bc7Srgb);
    // BC7: 16 bytes per 4x4 block, 8x8 = 4 blocks = 64 bytes
    assert_eq!(cooked.mip_level(0).unwrap().len(), 4 * 16);
}

#[test]
fn test_cook_bc4_compression() {
    let data = TextureData::new(8, 8, TextureFormat::Rgba8, vec![128; 8 * 8 * 4], 1);
    let cooker = TextureCooker::new().with_compression(true).with_mips(false);
    let cooked = cooker.cook(&data, TextureUsage::Roughness).unwrap();

    assert_eq!(cooked.format, GpuTextureFormat::Bc4Unorm);
    // BC4: 8 bytes per 4x4 block, 8x8 = 4 blocks = 32 bytes
    assert_eq!(cooked.mip_level(0).unwrap().len(), 4 * 8);
}

#[test]
fn test_cook_bc5_compression() {
    let data = TextureData::new(8, 8, TextureFormat::Rgba8, vec![128; 8 * 8 * 4], 1);
    let cooker = TextureCooker::new().with_compression(true).with_mips(false);
    let cooked = cooker.cook(&data, TextureUsage::NormalMap).unwrap();

    assert_eq!(cooked.format, GpuTextureFormat::Bc5Unorm);
    // BC5: 16 bytes per 4x4 block, 8x8 = 4 blocks = 64 bytes
    assert_eq!(cooked.mip_level(0).unwrap().len(), 4 * 16);
}

// ---------------------------------------------------------------------------
// Format conversion tests
// ---------------------------------------------------------------------------

#[test]
fn test_cook_r8_to_r8() {
    let data = TextureData::new(4, 4, TextureFormat::R8, vec![200; 16], 1);
    let cooker = TextureCooker::new().with_mips(false);
    let cooked = cooker.cook(&data, TextureUsage::Roughness).unwrap();

    assert_eq!(cooked.format, GpuTextureFormat::R8Unorm);
    let level0 = cooked.mip_level(0).unwrap();
    assert_eq!(level0.len(), 16);
    // All values should be 200 (gray -> luminance)
    for &v in level0 {
        assert_eq!(v, 200);
    }
}

#[test]
fn test_cook_r16_to_rgba8() {
    // 16-bit values (high byte matters)
    let mut r16_data = vec![0u8; 16 * 2];
    for i in 0..16 {
        r16_data[i * 2] = 0; // Low byte
        r16_data[i * 2 + 1] = 128; // High byte -> maps to 128
    }
    let data = TextureData::new(4, 4, TextureFormat::R16, r16_data, 1);
    let cooker = TextureCooker::new().with_mips(false);
    let cooked = cooker.cook(&data, TextureUsage::BaseColor).unwrap();

    let level0 = cooked.mip_level(0).unwrap();
    // All pixels should have R=G=B=128, A=255
    for chunk in level0.chunks(4) {
        assert_eq!(chunk[0], 128);
        assert_eq!(chunk[1], 128);
        assert_eq!(chunk[2], 128);
        assert_eq!(chunk[3], 255);
    }
}

#[test]
fn test_cook_r32f_to_rgba8() {
    // Float values 0.5 -> should map to ~128
    let mut r32f_data = vec![0u8; 16 * 4];
    for i in 0..16 {
        let bytes = 0.5f32.to_le_bytes();
        r32f_data[i * 4..i * 4 + 4].copy_from_slice(&bytes);
    }
    let data = TextureData::new(4, 4, TextureFormat::R32F, r32f_data, 1);
    let cooker = TextureCooker::new().with_mips(false);
    let cooked = cooker.cook(&data, TextureUsage::BaseColor).unwrap();

    let level0 = cooked.mip_level(0).unwrap();
    for chunk in level0.chunks(4) {
        // 0.5 * 255 = 127.5 -> 127 or 128
        assert!(chunk[0] >= 127 && chunk[0] <= 128);
    }
}

// ---------------------------------------------------------------------------
// TextureData::cook_with tests
// ---------------------------------------------------------------------------

#[test]
fn test_cook_with_custom_cooker() {
    let data = TextureData::new(16, 16, TextureFormat::Rgba8, vec![64; 16 * 16 * 4], 1);
    let cooker = TextureCooker::new().with_mips(true).with_max_mip_levels(2);
    let cooked = data.cook_with(&cooker, TextureUsage::BaseColor).unwrap();

    assert_eq!(cooked.mip_count(), 2);
}

// ---------------------------------------------------------------------------
// Edge case tests (BLACKBOX: T-MAT-9.2)
// ---------------------------------------------------------------------------

#[test]
fn test_non_power_of_two_dimensions() {
    // NPOT texture: 17x13
    let data = TextureData::new(17, 13, TextureFormat::Rgba8, vec![128; 17 * 13 * 4], 1);
    let cooked = data.cook(TextureUsage::BaseColor).unwrap();

    assert!(cooked.is_valid());
    assert_eq!(cooked.width, 17);
    assert_eq!(cooked.height, 13);
    // Mip levels: 17 -> 8 -> 4 -> 2 -> 1 = 5 levels (based on max dimension 17)
    assert_eq!(cooked.mip_count(), 5);

    // Verify all mip dimensions
    assert_eq!(cooked.mip_dimensions(0), Some((17, 13)));
    assert_eq!(cooked.mip_dimensions(1), Some((8, 6)));
    assert_eq!(cooked.mip_dimensions(2), Some((4, 3)));
    assert_eq!(cooked.mip_dimensions(3), Some((2, 1)));
    assert_eq!(cooked.mip_dimensions(4), Some((1, 1)));
}

#[test]
fn test_1x1_texture() {
    // Minimum valid texture size
    let data = TextureData::new(1, 1, TextureFormat::Rgba8, vec![255, 128, 64, 255], 1);
    let cooked = data.cook(TextureUsage::BaseColor).unwrap();

    assert!(cooked.is_valid());
    assert_eq!(cooked.width, 1);
    assert_eq!(cooked.height, 1);
    assert_eq!(cooked.mip_count(), 1); // Only 1 mip level for 1x1

    let level0 = cooked.mip_level(0).unwrap();
    assert_eq!(level0.len(), 4);
    assert_eq!(level0, &[255, 128, 64, 255]);
}

#[test]
fn test_1x1_texture_all_usage_types() {
    // Test 1x1 with each usage type
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
        let data = TextureData::new(1, 1, TextureFormat::Rgba8, vec![100, 100, 100, 255], 1);
        let cooked = data.cook(usage).unwrap();
        assert!(cooked.is_valid(), "1x1 texture failed for usage {:?}", usage);
        assert_eq!(cooked.mip_count(), 1);
    }
}

#[test]
fn test_maximum_mip_levels_large_texture() {
    // 2048x2048 = 2^11, should have 12 mip levels (2048 -> 1)
    let data = TextureData::new(2048, 2048, TextureFormat::Rgba8, vec![128; 2048 * 2048 * 4], 1);
    let cooker = TextureCooker::new().with_mips(true);
    let cooked = cooker.cook(&data, TextureUsage::BaseColor).unwrap();

    assert_eq!(cooked.mip_count(), 12);
    assert_eq!(cooked.mip_dimensions(11), Some((1, 1)));
}

#[test]
fn test_asymmetric_dimensions() {
    // Very asymmetric: 256x1
    let data = TextureData::new(256, 1, TextureFormat::Rgba8, vec![200; 256 * 1 * 4], 1);
    let cooked = data.cook(TextureUsage::Unknown).unwrap();

    assert!(cooked.is_valid());
    // Mip levels based on max dimension (256 = 2^8 -> 9 levels)
    assert_eq!(cooked.mip_count(), 9);
    assert_eq!(cooked.mip_dimensions(0), Some((256, 1)));
    assert_eq!(cooked.mip_dimensions(8), Some((1, 1)));
}

#[test]
fn test_npot_with_compression() {
    // NPOT with BC compression (requires block padding)
    let data = TextureData::new(5, 5, TextureFormat::Rgba8, vec![128; 5 * 5 * 4], 1);
    let cooker = TextureCooker::new().with_compression(true).with_mips(false);
    let cooked = cooker.cook(&data, TextureUsage::BaseColor).unwrap();

    assert_eq!(cooked.format, GpuTextureFormat::Bc7Srgb);
    // 5x5 needs 2x2 blocks = 4 blocks * 16 bytes = 64 bytes
    let level0 = cooked.mip_level(0).unwrap();
    assert_eq!(level0.len(), 4 * 16);
}

#[test]
fn test_3x3_bc4_compression_block_count() {
    // 3x3 -> 1x1 blocks, BC4 = 8 bytes/block
    let data = TextureData::new(3, 3, TextureFormat::Rgba8, vec![100; 3 * 3 * 4], 1);
    let cooker = TextureCooker::new().with_compression(true).with_mips(false);
    let cooked = cooker.cook(&data, TextureUsage::Roughness).unwrap();

    assert_eq!(cooked.format, GpuTextureFormat::Bc4Unorm);
    // 3x3 -> (3+3)/4 = 1 block each dimension = 1 block total
    let level0 = cooked.mip_level(0).unwrap();
    assert_eq!(level0.len(), 1 * 8);
}

#[test]
fn test_mip_chain_downscale_verification() {
    // Create gradient texture to verify box filtering
    let mut pixels = vec![0u8; 8 * 8 * 4];
    for y in 0..8u32 {
        for x in 0..8u32 {
            let idx = ((y * 8 + x) * 4) as usize;
            // R increases with X, G increases with Y
            pixels[idx] = (x * 32) as u8;
            pixels[idx + 1] = (y * 32) as u8;
            pixels[idx + 2] = 0;
            pixels[idx + 3] = 255;
        }
    }

    let data = TextureData::new(8, 8, TextureFormat::Rgba8, pixels, 1);
    let cooked = data.cook(TextureUsage::BaseColor).unwrap();

    // Verify mip 1 (4x4) has averaged values
    let mip1 = cooked.mip_level(1).unwrap();
    assert_eq!(mip1.len(), 4 * 4 * 4);

    // First pixel of mip1 should be average of top-left 2x2 of mip0
    // (0,0), (32,0), (0,32), (32,32) -> avg = (16, 16, 0, 255)
    assert!(mip1[0] <= 32); // R averaged
    assert!(mip1[1] <= 32); // G averaged
}

#[test]
fn test_bc6h_emissive_format() {
    let data = TextureData::new(8, 8, TextureFormat::Rgba8, vec![255; 8 * 8 * 4], 1);
    let cooker = TextureCooker::new().with_compression(true).with_mips(false);
    let cooked = cooker.cook(&data, TextureUsage::Emissive).unwrap();

    assert_eq!(cooked.format, GpuTextureFormat::Bc6hFloat);
    // 8x8 = 4 blocks * 16 bytes
    assert_eq!(cooked.mip_level(0).unwrap().len(), 4 * 16);
}

#[test]
fn test_data_usage_always_r32float() {
    // Data usage should always be R32Float regardless of compression setting
    let data = TextureData::new(4, 4, TextureFormat::Rgba8, vec![128; 4 * 4 * 4], 1);

    let cooker_no_compress = TextureCooker::new().with_compression(false).with_mips(false);
    let cooked1 = cooker_no_compress.cook(&data, TextureUsage::Data).unwrap();
    assert_eq!(cooked1.format, GpuTextureFormat::R32Float);

    let cooker_compress = TextureCooker::new().with_compression(true).with_mips(false);
    let cooked2 = cooker_compress.cook(&data, TextureUsage::Data).unwrap();
    assert_eq!(cooked2.format, GpuTextureFormat::R32Float);
}
