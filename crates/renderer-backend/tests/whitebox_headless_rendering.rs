// WHITEBOX tests for T-WGPU-P7.1.10 (Headless Rendering)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, and edge cases
// that are not visible through the public contract alone.
//
// Implementation under test: crates/renderer-backend/src/presentation/surface.rs
//   - HeadlessError enum: TextureCreationFailed, StagingBufferFailed, BufferMapFailed,
//                         NotInitialized, InvalidDimensions, ScreenshotSaveFailed, ResolveFailed
//   - HeadlessConfig struct: width, height, format, sample_count, usage, label
//   - HeadlessConfig methods: new(), with_format(), with_msaa(), with_readback(),
//                             with_usage(), with_label(), is_msaa_enabled(),
//                             supports_readback(), validate(), bytes_per_pixel(),
//                             aligned_bytes_per_row(), buffer_size()
//   - HeadlessTarget: new(), texture(), view(), resolve_view(), resolve_texture(),
//                     config(), width(), height(), dimensions(), format(), sample_count(),
//                     is_msaa_enabled(), aspect_ratio(), resize(), acquire_frame(),
//                     create_staging_buffer(), screenshot(), screenshot_packed()
//   - HeadlessFrame: view(), resolve_view(), width(), height(), dimensions(),
//                    format(), sample_count(), is_msaa_enabled(), aspect_ratio()
//   - ReadbackBuffer: buffer(), bytes_per_row(), bytes_per_pixel(), width(), height(),
//                     format(), size(), map_read(), map_read_packed()
//   - HeadlessRenderer: new(), acquire_frame(), target(), target_mut(), frame_count(),
//                       view(), resolve_view(), width(), height(), dimensions(),
//                       format(), resize(), screenshot(), screenshot_packed()
//
// WHITEBOX coverage plan:
//   - Category 1: HeadlessError - all variants, error messages, debug formatting
//   - Category 2: HeadlessConfig validation - dimensions, sample_count 1/4/8
//   - Category 3: Bytes per pixel - R8=1, Rg8=2, Rgba8=4, Bgra8=4, Rgba16Float=8, Rgba32=16
//   - Category 4: Row alignment to 256 bytes (COPY_BYTES_PER_ROW_ALIGNMENT)
//   - Category 5: Buffer size calculation with padding
//   - Category 6: MSAA sample count validation - clamping logic (0..=1->1, 2..=4->4, >4->8)
//   - Category 7: Texture usage flags (RENDER_ATTACHMENT | COPY_SRC)
//   - Category 8: Edge cases - zero dimensions, max dimensions, single pixel
//   - Category 9: HeadlessTarget - texture/view creation, MSAA resolve targets
//   - Category 10: HeadlessFrame - view accessors, dimension queries
//   - Category 11: ReadbackBuffer - size calculations, packed read logic
//   - Category 12: HeadlessRenderer - frame counting, delegation methods
//   - Category 13: Clone, Debug, Default trait implementations
//   - Category 14: Builder pattern chaining

use renderer_backend::presentation::{
    HeadlessConfig, HeadlessError, HeadlessFrame, HeadlessRenderer, HeadlessTarget,
    ReadbackBuffer,
};
use std::error::Error;
use wgpu::TextureFormat;

// ============================================================================
// Category 1: HeadlessError Tests
// ============================================================================

/// Test HeadlessError::TextureCreationFailed variant
#[test]
fn test_headless_error_texture_creation_failed_message() {
    let err = HeadlessError::TextureCreationFailed("GPU out of memory".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("GPU out of memory"));
    assert!(msg.contains("headless target"));
}

/// Test HeadlessError::TextureCreationFailed with empty message
#[test]
fn test_headless_error_texture_creation_failed_empty() {
    let err = HeadlessError::TextureCreationFailed(String::new());
    let msg = format!("{}", err);
    assert!(msg.contains("headless target"));
}

/// Test HeadlessError::StagingBufferFailed variant
#[test]
fn test_headless_error_staging_buffer_failed_message() {
    let err = HeadlessError::StagingBufferFailed("allocation failed".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("allocation failed"));
    assert!(msg.contains("staging buffer"));
}

/// Test HeadlessError::BufferMapFailed variant
#[test]
fn test_headless_error_buffer_map_failed_message() {
    let err = HeadlessError::BufferMapFailed("device lost".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("device lost"));
    assert!(msg.contains("mapping"));
}

/// Test HeadlessError::NotInitialized variant
#[test]
fn test_headless_error_not_initialized_message() {
    let err = HeadlessError::NotInitialized;
    let msg = format!("{}", err);
    assert!(msg.contains("not initialized"));
}

/// Test HeadlessError::InvalidDimensions with zero values
#[test]
fn test_headless_error_invalid_dimensions_zero() {
    let err = HeadlessError::invalid_dimensions(0, 0);
    let msg = format!("{}", err);
    assert!(msg.contains("invalid"));
    assert!(msg.contains("width=0"));
    assert!(msg.contains("height=0"));
}

/// Test HeadlessError::InvalidDimensions with one zero dimension
#[test]
fn test_headless_error_invalid_dimensions_one_zero_width() {
    let err = HeadlessError::invalid_dimensions(0, 1080);
    let msg = format!("{}", err);
    assert!(msg.contains("width=0"));
    assert!(msg.contains("height=1080"));
}

/// Test HeadlessError::InvalidDimensions with one zero dimension (height)
#[test]
fn test_headless_error_invalid_dimensions_one_zero_height() {
    let err = HeadlessError::invalid_dimensions(1920, 0);
    let msg = format!("{}", err);
    assert!(msg.contains("width=1920"));
    assert!(msg.contains("height=0"));
}

/// Test HeadlessError::InvalidDimensions with large values
#[test]
fn test_headless_error_invalid_dimensions_large_values() {
    let err = HeadlessError::invalid_dimensions(u32::MAX, u32::MAX);
    let msg = format!("{}", err);
    assert!(msg.contains(&u32::MAX.to_string()));
}

/// Test HeadlessError::ScreenshotSaveFailed variant
#[test]
fn test_headless_error_screenshot_save_failed_message() {
    let err = HeadlessError::ScreenshotSaveFailed("permission denied".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("permission denied"));
    assert!(msg.contains("screenshot"));
}

/// Test HeadlessError::ResolveFailed variant
#[test]
fn test_headless_error_resolve_failed_message() {
    let err = HeadlessError::ResolveFailed("MSAA resolve failed".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("MSAA"));
}

/// Test HeadlessError Debug trait for TextureCreationFailed
#[test]
fn test_headless_error_debug_texture_creation_failed() {
    let err = HeadlessError::TextureCreationFailed("test".to_string());
    let debug = format!("{:?}", err);
    assert!(debug.contains("TextureCreationFailed"));
}

/// Test HeadlessError Debug trait for StagingBufferFailed
#[test]
fn test_headless_error_debug_staging_buffer_failed() {
    let err = HeadlessError::StagingBufferFailed("test".to_string());
    let debug = format!("{:?}", err);
    assert!(debug.contains("StagingBufferFailed"));
}

/// Test HeadlessError Debug trait for BufferMapFailed
#[test]
fn test_headless_error_debug_buffer_map_failed() {
    let err = HeadlessError::BufferMapFailed("test".to_string());
    let debug = format!("{:?}", err);
    assert!(debug.contains("BufferMapFailed"));
}

/// Test HeadlessError Debug trait for NotInitialized
#[test]
fn test_headless_error_debug_not_initialized() {
    let err = HeadlessError::NotInitialized;
    let debug = format!("{:?}", err);
    assert!(debug.contains("NotInitialized"));
}

/// Test HeadlessError Debug trait for InvalidDimensions
#[test]
fn test_headless_error_debug_invalid_dimensions() {
    let err = HeadlessError::invalid_dimensions(100, 200);
    let debug = format!("{:?}", err);
    assert!(debug.contains("InvalidDimensions"));
    assert!(debug.contains("100"));
    assert!(debug.contains("200"));
}

/// Test HeadlessError Debug trait for ScreenshotSaveFailed
#[test]
fn test_headless_error_debug_screenshot_save_failed() {
    let err = HeadlessError::ScreenshotSaveFailed("test".to_string());
    let debug = format!("{:?}", err);
    assert!(debug.contains("ScreenshotSaveFailed"));
}

/// Test HeadlessError Debug trait for ResolveFailed
#[test]
fn test_headless_error_debug_resolve_failed() {
    let err = HeadlessError::ResolveFailed("test".to_string());
    let debug = format!("{:?}", err);
    assert!(debug.contains("ResolveFailed"));
}

/// Test HeadlessError implements std::error::Error
#[test]
fn test_headless_error_is_std_error() {
    fn assert_error<T: Error>() {}
    assert_error::<HeadlessError>();
}

// ============================================================================
// Category 2: HeadlessConfig - Basic Construction and Validation
// ============================================================================

/// Test HeadlessConfig::new with standard HD dimensions
#[test]
fn test_headless_config_new_hd() {
    let config = HeadlessConfig::new(1920, 1080);
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
}

/// Test HeadlessConfig::new with 4K dimensions
#[test]
fn test_headless_config_new_4k() {
    let config = HeadlessConfig::new(3840, 2160);
    assert_eq!(config.width, 3840);
    assert_eq!(config.height, 2160);
}

/// Test HeadlessConfig::new with 8K dimensions
#[test]
fn test_headless_config_new_8k() {
    let config = HeadlessConfig::new(7680, 4320);
    assert_eq!(config.width, 7680);
    assert_eq!(config.height, 4320);
}

/// Test HeadlessConfig::new clamps zero width to 1
#[test]
fn test_headless_config_new_clamps_zero_width() {
    let config = HeadlessConfig::new(0, 100);
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 100);
}

/// Test HeadlessConfig::new clamps zero height to 1
#[test]
fn test_headless_config_new_clamps_zero_height() {
    let config = HeadlessConfig::new(100, 0);
    assert_eq!(config.width, 100);
    assert_eq!(config.height, 1);
}

/// Test HeadlessConfig::new clamps both zero dimensions to 1
#[test]
fn test_headless_config_new_clamps_both_zero() {
    let config = HeadlessConfig::new(0, 0);
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 1);
}

/// Test HeadlessConfig::new with single pixel
#[test]
fn test_headless_config_new_single_pixel() {
    let config = HeadlessConfig::new(1, 1);
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 1);
}

/// Test HeadlessConfig::new default format is Rgba8Unorm
#[test]
fn test_headless_config_new_default_format() {
    let config = HeadlessConfig::new(1920, 1080);
    assert_eq!(config.format, TextureFormat::Rgba8Unorm);
}

/// Test HeadlessConfig::new default sample_count is 1 (no MSAA)
#[test]
fn test_headless_config_new_default_sample_count() {
    let config = HeadlessConfig::new(1920, 1080);
    assert_eq!(config.sample_count, 1);
}

/// Test HeadlessConfig::new default usage includes RENDER_ATTACHMENT
#[test]
fn test_headless_config_new_default_usage_render_attachment() {
    let config = HeadlessConfig::new(1920, 1080);
    assert!(config.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
}

/// Test HeadlessConfig::new default usage includes COPY_SRC
#[test]
fn test_headless_config_new_default_usage_copy_src() {
    let config = HeadlessConfig::new(1920, 1080);
    assert!(config.usage.contains(wgpu::TextureUsages::COPY_SRC));
}

/// Test HeadlessConfig::new default label is None
#[test]
fn test_headless_config_new_default_label() {
    let config = HeadlessConfig::new(1920, 1080);
    assert!(config.label.is_none());
}

/// Test HeadlessConfig::default uses 800x600
#[test]
fn test_headless_config_default() {
    let config = HeadlessConfig::default();
    assert_eq!(config.width, 800);
    assert_eq!(config.height, 600);
}

/// Test HeadlessConfig::default has same properties as new(800, 600)
#[test]
fn test_headless_config_default_matches_new() {
    let default = HeadlessConfig::default();
    let manual = HeadlessConfig::new(800, 600);
    assert_eq!(default.width, manual.width);
    assert_eq!(default.height, manual.height);
    assert_eq!(default.format, manual.format);
    assert_eq!(default.sample_count, manual.sample_count);
}

/// Test HeadlessConfig::validate succeeds for valid dimensions
#[test]
fn test_headless_config_validate_success() {
    let config = HeadlessConfig::new(1920, 1080);
    assert!(config.validate().is_ok());
}

/// Test HeadlessConfig::validate succeeds for single pixel
#[test]
fn test_headless_config_validate_success_single_pixel() {
    let config = HeadlessConfig::new(1, 1);
    assert!(config.validate().is_ok());
}

/// Test HeadlessConfig::validate fails for zero width (manually set)
#[test]
fn test_headless_config_validate_fails_zero_width() {
    let config = HeadlessConfig {
        width: 0,
        ..HeadlessConfig::new(100, 100)
    };
    let result = config.validate();
    assert!(result.is_err());
    if let Err(HeadlessError::InvalidDimensions { width, height }) = result {
        assert_eq!(width, 0);
        assert_eq!(height, 100);
    } else {
        panic!("Expected InvalidDimensions error");
    }
}

/// Test HeadlessConfig::validate fails for zero height (manually set)
#[test]
fn test_headless_config_validate_fails_zero_height() {
    let config = HeadlessConfig {
        height: 0,
        ..HeadlessConfig::new(100, 100)
    };
    let result = config.validate();
    assert!(result.is_err());
    if let Err(HeadlessError::InvalidDimensions { width, height }) = result {
        assert_eq!(width, 100);
        assert_eq!(height, 0);
    } else {
        panic!("Expected InvalidDimensions error");
    }
}

/// Test HeadlessConfig::validate fails for both dimensions zero (manually set)
#[test]
fn test_headless_config_validate_fails_both_zero() {
    let config = HeadlessConfig {
        width: 0,
        height: 0,
        ..HeadlessConfig::new(100, 100)
    };
    assert!(config.validate().is_err());
}

// ============================================================================
// Category 3: Bytes Per Pixel for Different Formats
// ============================================================================

/// Test bytes_per_pixel for R8Unorm (1 byte)
#[test]
fn test_bytes_per_pixel_r8unorm() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::R8Unorm);
    assert_eq!(config.bytes_per_pixel(), 1);
}

/// Test bytes_per_pixel for R8Snorm (1 byte)
#[test]
fn test_bytes_per_pixel_r8snorm() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::R8Snorm);
    assert_eq!(config.bytes_per_pixel(), 1);
}

/// Test bytes_per_pixel for R8Uint (1 byte)
#[test]
fn test_bytes_per_pixel_r8uint() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::R8Uint);
    assert_eq!(config.bytes_per_pixel(), 1);
}

/// Test bytes_per_pixel for R8Sint (1 byte)
#[test]
fn test_bytes_per_pixel_r8sint() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::R8Sint);
    assert_eq!(config.bytes_per_pixel(), 1);
}

/// Test bytes_per_pixel for Rg8Unorm (2 bytes)
#[test]
fn test_bytes_per_pixel_rg8unorm() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rg8Unorm);
    assert_eq!(config.bytes_per_pixel(), 2);
}

/// Test bytes_per_pixel for Rg8Snorm (2 bytes)
#[test]
fn test_bytes_per_pixel_rg8snorm() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rg8Snorm);
    assert_eq!(config.bytes_per_pixel(), 2);
}

/// Test bytes_per_pixel for Rg8Uint (2 bytes)
#[test]
fn test_bytes_per_pixel_rg8uint() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rg8Uint);
    assert_eq!(config.bytes_per_pixel(), 2);
}

/// Test bytes_per_pixel for Rg8Sint (2 bytes)
#[test]
fn test_bytes_per_pixel_rg8sint() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rg8Sint);
    assert_eq!(config.bytes_per_pixel(), 2);
}

/// Test bytes_per_pixel for Rgba8Unorm (4 bytes)
#[test]
fn test_bytes_per_pixel_rgba8unorm() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rgba8Unorm);
    assert_eq!(config.bytes_per_pixel(), 4);
}

/// Test bytes_per_pixel for Rgba8UnormSrgb (4 bytes)
#[test]
fn test_bytes_per_pixel_rgba8unorm_srgb() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rgba8UnormSrgb);
    assert_eq!(config.bytes_per_pixel(), 4);
}

/// Test bytes_per_pixel for Rgba8Snorm (4 bytes)
#[test]
fn test_bytes_per_pixel_rgba8snorm() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rgba8Snorm);
    assert_eq!(config.bytes_per_pixel(), 4);
}

/// Test bytes_per_pixel for Rgba8Uint (4 bytes)
#[test]
fn test_bytes_per_pixel_rgba8uint() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rgba8Uint);
    assert_eq!(config.bytes_per_pixel(), 4);
}

/// Test bytes_per_pixel for Rgba8Sint (4 bytes)
#[test]
fn test_bytes_per_pixel_rgba8sint() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rgba8Sint);
    assert_eq!(config.bytes_per_pixel(), 4);
}

/// Test bytes_per_pixel for Bgra8Unorm (4 bytes)
#[test]
fn test_bytes_per_pixel_bgra8unorm() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Bgra8Unorm);
    assert_eq!(config.bytes_per_pixel(), 4);
}

/// Test bytes_per_pixel for Bgra8UnormSrgb (4 bytes)
#[test]
fn test_bytes_per_pixel_bgra8unorm_srgb() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Bgra8UnormSrgb);
    assert_eq!(config.bytes_per_pixel(), 4);
}

/// Test bytes_per_pixel for Rgb10a2Unorm (4 bytes)
#[test]
fn test_bytes_per_pixel_rgb10a2unorm() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rgb10a2Unorm);
    assert_eq!(config.bytes_per_pixel(), 4);
}

/// Test bytes_per_pixel for Rgba16Float (8 bytes)
#[test]
fn test_bytes_per_pixel_rgba16float() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rgba16Float);
    assert_eq!(config.bytes_per_pixel(), 8);
}

/// Test bytes_per_pixel for Rgba16Uint (8 bytes)
#[test]
fn test_bytes_per_pixel_rgba16uint() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rgba16Uint);
    assert_eq!(config.bytes_per_pixel(), 8);
}

/// Test bytes_per_pixel for Rgba16Sint (8 bytes)
#[test]
fn test_bytes_per_pixel_rgba16sint() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rgba16Sint);
    assert_eq!(config.bytes_per_pixel(), 8);
}

/// Test bytes_per_pixel for Rgba16Unorm (8 bytes)
#[test]
fn test_bytes_per_pixel_rgba16unorm() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rgba16Unorm);
    assert_eq!(config.bytes_per_pixel(), 8);
}

/// Test bytes_per_pixel for Rgba16Snorm (8 bytes)
#[test]
fn test_bytes_per_pixel_rgba16snorm() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rgba16Snorm);
    assert_eq!(config.bytes_per_pixel(), 8);
}

/// Test bytes_per_pixel for Rgba32Float (16 bytes)
#[test]
fn test_bytes_per_pixel_rgba32float() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rgba32Float);
    assert_eq!(config.bytes_per_pixel(), 16);
}

/// Test bytes_per_pixel for Rgba32Uint (16 bytes)
#[test]
fn test_bytes_per_pixel_rgba32uint() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rgba32Uint);
    assert_eq!(config.bytes_per_pixel(), 16);
}

/// Test bytes_per_pixel for Rgba32Sint (16 bytes)
#[test]
fn test_bytes_per_pixel_rgba32sint() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Rgba32Sint);
    assert_eq!(config.bytes_per_pixel(), 16);
}

/// Test bytes_per_pixel defaults to 4 for unknown format (Depth32Float)
#[test]
fn test_bytes_per_pixel_unknown_format_defaults_4() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::Depth32Float);
    assert_eq!(config.bytes_per_pixel(), 4);
}

// ============================================================================
// Category 4: Row Alignment to 256 Bytes
// ============================================================================

/// Test aligned_bytes_per_row for 64 pixels * 4 bytes = 256 (already aligned)
#[test]
fn test_aligned_bytes_per_row_already_aligned_256() {
    let config = HeadlessConfig::new(64, 100);
    assert_eq!(config.aligned_bytes_per_row(), 256);
}

/// Test aligned_bytes_per_row for 128 pixels * 4 bytes = 512 (already aligned)
#[test]
fn test_aligned_bytes_per_row_already_aligned_512() {
    let config = HeadlessConfig::new(128, 100);
    assert_eq!(config.aligned_bytes_per_row(), 512);
}

/// Test aligned_bytes_per_row for 256 pixels * 4 bytes = 1024 (already aligned)
#[test]
fn test_aligned_bytes_per_row_already_aligned_1024() {
    let config = HeadlessConfig::new(256, 100);
    assert_eq!(config.aligned_bytes_per_row(), 1024);
}

/// Test aligned_bytes_per_row for 100 pixels * 4 bytes = 400 -> 512
#[test]
fn test_aligned_bytes_per_row_400_to_512() {
    let config = HeadlessConfig::new(100, 100);
    assert_eq!(config.aligned_bytes_per_row(), 512);
}

/// Test aligned_bytes_per_row for 1 pixel * 4 bytes = 4 -> 256
#[test]
fn test_aligned_bytes_per_row_single_pixel() {
    let config = HeadlessConfig::new(1, 1);
    assert_eq!(config.aligned_bytes_per_row(), 256);
}

/// Test aligned_bytes_per_row for 63 pixels * 4 bytes = 252 -> 256
#[test]
fn test_aligned_bytes_per_row_252_to_256() {
    let config = HeadlessConfig::new(63, 100);
    assert_eq!(config.aligned_bytes_per_row(), 256);
}

/// Test aligned_bytes_per_row for 65 pixels * 4 bytes = 260 -> 512
#[test]
fn test_aligned_bytes_per_row_260_to_512() {
    let config = HeadlessConfig::new(65, 100);
    assert_eq!(config.aligned_bytes_per_row(), 512);
}

/// Test aligned_bytes_per_row for 1920 pixels * 4 bytes = 7680 (already aligned)
#[test]
fn test_aligned_bytes_per_row_hd_width() {
    let config = HeadlessConfig::new(1920, 1080);
    // 1920 * 4 = 7680, 7680 is divisible by 256 (7680 / 256 = 30)
    assert_eq!(config.aligned_bytes_per_row(), 7680);
}

/// Test aligned_bytes_per_row for 1921 pixels * 4 bytes = 7684 -> 7936
#[test]
fn test_aligned_bytes_per_row_hd_width_plus_one() {
    let config = HeadlessConfig::new(1921, 1080);
    // 1921 * 4 = 7684, aligned to 256 = 7936 (7684 + 252 = 7936, or (7684 + 255) & !255)
    assert_eq!(config.aligned_bytes_per_row(), 7936);
}

/// Test aligned_bytes_per_row with R8 format (1 byte per pixel)
#[test]
fn test_aligned_bytes_per_row_r8_format() {
    let config = HeadlessConfig::new(256, 100).with_format(TextureFormat::R8Unorm);
    // 256 * 1 = 256 (already aligned)
    assert_eq!(config.aligned_bytes_per_row(), 256);
}

/// Test aligned_bytes_per_row with R8 format needing padding
#[test]
fn test_aligned_bytes_per_row_r8_format_needs_padding() {
    let config = HeadlessConfig::new(100, 100).with_format(TextureFormat::R8Unorm);
    // 100 * 1 = 100, aligned to 256
    assert_eq!(config.aligned_bytes_per_row(), 256);
}

/// Test aligned_bytes_per_row with Rgba16Float (8 bytes per pixel)
#[test]
fn test_aligned_bytes_per_row_rgba16float() {
    let config = HeadlessConfig::new(32, 100).with_format(TextureFormat::Rgba16Float);
    // 32 * 8 = 256 (already aligned)
    assert_eq!(config.aligned_bytes_per_row(), 256);
}

/// Test aligned_bytes_per_row with Rgba16Float needing padding
#[test]
fn test_aligned_bytes_per_row_rgba16float_needs_padding() {
    let config = HeadlessConfig::new(33, 100).with_format(TextureFormat::Rgba16Float);
    // 33 * 8 = 264, aligned to 512
    assert_eq!(config.aligned_bytes_per_row(), 512);
}

/// Test aligned_bytes_per_row with Rgba32Float (16 bytes per pixel)
#[test]
fn test_aligned_bytes_per_row_rgba32float() {
    let config = HeadlessConfig::new(16, 100).with_format(TextureFormat::Rgba32Float);
    // 16 * 16 = 256 (already aligned)
    assert_eq!(config.aligned_bytes_per_row(), 256);
}

// ============================================================================
// Category 5: Buffer Size Calculation with Padding
// ============================================================================

/// Test buffer_size for 64x100 Rgba8 (already aligned)
#[test]
fn test_buffer_size_aligned() {
    let config = HeadlessConfig::new(64, 100);
    // 64 * 4 = 256 bytes per row (aligned)
    // 256 * 100 = 25600 bytes
    assert_eq!(config.buffer_size(), 25600);
}

/// Test buffer_size for 100x100 Rgba8 (needs padding)
#[test]
fn test_buffer_size_with_padding() {
    let config = HeadlessConfig::new(100, 100);
    // 100 * 4 = 400, aligned to 512
    // 512 * 100 = 51200 bytes
    assert_eq!(config.buffer_size(), 51200);
}

/// Test buffer_size for single pixel
#[test]
fn test_buffer_size_single_pixel() {
    let config = HeadlessConfig::new(1, 1);
    // 1 * 4 = 4, aligned to 256
    // 256 * 1 = 256 bytes
    assert_eq!(config.buffer_size(), 256);
}

/// Test buffer_size for HD resolution
#[test]
fn test_buffer_size_hd() {
    let config = HeadlessConfig::new(1920, 1080);
    // 1920 * 4 = 7680 (already aligned)
    // 7680 * 1080 = 8,294,400 bytes
    assert_eq!(config.buffer_size(), 8_294_400);
}

/// Test buffer_size for 4K resolution
#[test]
fn test_buffer_size_4k() {
    let config = HeadlessConfig::new(3840, 2160);
    // 3840 * 4 = 15360 (already aligned, 15360 / 256 = 60)
    // 15360 * 2160 = 33,177,600 bytes
    assert_eq!(config.buffer_size(), 33_177_600);
}

/// Test buffer_size with R8 format
#[test]
fn test_buffer_size_r8_format() {
    let config = HeadlessConfig::new(256, 100).with_format(TextureFormat::R8Unorm);
    // 256 * 1 = 256 (aligned)
    // 256 * 100 = 25600 bytes
    assert_eq!(config.buffer_size(), 25600);
}

/// Test buffer_size with Rgba16Float format
#[test]
fn test_buffer_size_rgba16float_format() {
    let config = HeadlessConfig::new(32, 100).with_format(TextureFormat::Rgba16Float);
    // 32 * 8 = 256 (aligned)
    // 256 * 100 = 25600 bytes
    assert_eq!(config.buffer_size(), 25600);
}

/// Test buffer_size returns u64 for large textures
#[test]
fn test_buffer_size_large_texture() {
    let config = HeadlessConfig::new(8192, 8192);
    // 8192 * 4 = 32768 (aligned)
    // 32768 * 8192 = 268,435,456 bytes
    assert_eq!(config.buffer_size(), 268_435_456u64);
}

// ============================================================================
// Category 6: MSAA Sample Count Validation
// ============================================================================

/// Test with_msaa(0) clamps to 1
#[test]
fn test_msaa_clamp_0_to_1() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(0);
    assert_eq!(config.sample_count, 1);
}

/// Test with_msaa(1) stays 1
#[test]
fn test_msaa_1_stays_1() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(1);
    assert_eq!(config.sample_count, 1);
}

/// Test with_msaa(2) clamps to 4
#[test]
fn test_msaa_clamp_2_to_4() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(2);
    assert_eq!(config.sample_count, 4);
}

/// Test with_msaa(3) clamps to 4
#[test]
fn test_msaa_clamp_3_to_4() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(3);
    assert_eq!(config.sample_count, 4);
}

/// Test with_msaa(4) stays 4
#[test]
fn test_msaa_4_stays_4() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(4);
    assert_eq!(config.sample_count, 4);
}

/// Test with_msaa(5) clamps to 8
#[test]
fn test_msaa_clamp_5_to_8() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(5);
    assert_eq!(config.sample_count, 8);
}

/// Test with_msaa(6) clamps to 8
#[test]
fn test_msaa_clamp_6_to_8() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(6);
    assert_eq!(config.sample_count, 8);
}

/// Test with_msaa(7) clamps to 8
#[test]
fn test_msaa_clamp_7_to_8() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(7);
    assert_eq!(config.sample_count, 8);
}

/// Test with_msaa(8) stays 8
#[test]
fn test_msaa_8_stays_8() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(8);
    assert_eq!(config.sample_count, 8);
}

/// Test with_msaa(16) clamps to 8
#[test]
fn test_msaa_clamp_16_to_8() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(16);
    assert_eq!(config.sample_count, 8);
}

/// Test with_msaa(32) clamps to 8
#[test]
fn test_msaa_clamp_32_to_8() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(32);
    assert_eq!(config.sample_count, 8);
}

/// Test with_msaa(u32::MAX) clamps to 8
#[test]
fn test_msaa_clamp_max_to_8() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(u32::MAX);
    assert_eq!(config.sample_count, 8);
}

/// Test is_msaa_enabled returns false for sample_count 1
#[test]
fn test_is_msaa_enabled_false() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(1);
    assert!(!config.is_msaa_enabled());
}

/// Test is_msaa_enabled returns true for sample_count 4
#[test]
fn test_is_msaa_enabled_true_4x() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(4);
    assert!(config.is_msaa_enabled());
}

/// Test is_msaa_enabled returns true for sample_count 8
#[test]
fn test_is_msaa_enabled_true_8x() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(8);
    assert!(config.is_msaa_enabled());
}

// ============================================================================
// Category 7: Texture Usage Flags
// ============================================================================

/// Test default usage includes RENDER_ATTACHMENT
#[test]
fn test_usage_default_render_attachment() {
    let config = HeadlessConfig::new(1920, 1080);
    assert!(config.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
}

/// Test default usage includes COPY_SRC
#[test]
fn test_usage_default_copy_src() {
    let config = HeadlessConfig::new(1920, 1080);
    assert!(config.usage.contains(wgpu::TextureUsages::COPY_SRC));
}

/// Test with_readback adds COPY_SRC (idempotent)
#[test]
fn test_with_readback_adds_copy_src() {
    let config = HeadlessConfig::new(1920, 1080).with_readback();
    assert!(config.usage.contains(wgpu::TextureUsages::COPY_SRC));
}

/// Test with_usage adds RENDER_ATTACHMENT if not present
#[test]
fn test_with_usage_forces_render_attachment() {
    let config = HeadlessConfig::new(1920, 1080)
        .with_usage(wgpu::TextureUsages::TEXTURE_BINDING);
    assert!(config.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
    assert!(config.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
}

/// Test with_usage preserves additional flags
#[test]
fn test_with_usage_preserves_flags() {
    let config = HeadlessConfig::new(1920, 1080)
        .with_usage(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC);
    assert!(config.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
    assert!(config.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
    assert!(config.usage.contains(wgpu::TextureUsages::COPY_SRC));
}

/// Test supports_readback returns true by default
#[test]
fn test_supports_readback_default() {
    let config = HeadlessConfig::new(1920, 1080);
    assert!(config.supports_readback());
}

/// Test supports_readback returns false without COPY_SRC
#[test]
fn test_supports_readback_false_without_copy_src() {
    let config = HeadlessConfig {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        ..HeadlessConfig::new(1920, 1080)
    };
    assert!(!config.supports_readback());
}

/// Test with_readback on config without COPY_SRC
#[test]
fn test_with_readback_on_minimal_config() {
    let mut config = HeadlessConfig::new(1920, 1080);
    config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
    assert!(!config.supports_readback());

    config = config.with_readback();
    assert!(config.supports_readback());
}

// ============================================================================
// Category 8: Edge Cases
// ============================================================================

/// Test with very small width (1 pixel)
#[test]
fn test_edge_case_min_width() {
    let config = HeadlessConfig::new(1, 1080);
    assert_eq!(config.width, 1);
    assert_eq!(config.aligned_bytes_per_row(), 256);
}

/// Test with very small height (1 pixel)
#[test]
fn test_edge_case_min_height() {
    let config = HeadlessConfig::new(1920, 1);
    assert_eq!(config.height, 1);
    assert_eq!(config.buffer_size(), 7680); // 1920 * 4 = 7680, already aligned
}

/// Test aspect ratio for square texture
#[test]
fn test_aspect_ratio_square() {
    let config = HeadlessConfig::new(1000, 1000);
    // Note: aspect_ratio is not on HeadlessConfig but on HeadlessTarget/Frame
    // We test the calculation conceptually
    let ratio = config.width as f32 / config.height as f32;
    assert!((ratio - 1.0).abs() < 0.0001);
}

/// Test aspect ratio for wide texture
#[test]
fn test_aspect_ratio_wide() {
    let config = HeadlessConfig::new(1920, 1080);
    let ratio = config.width as f32 / config.height as f32;
    assert!((ratio - 16.0 / 9.0).abs() < 0.01);
}

/// Test aspect ratio for tall texture
#[test]
fn test_aspect_ratio_tall() {
    let config = HeadlessConfig::new(1080, 1920);
    let ratio = config.width as f32 / config.height as f32;
    assert!((ratio - 9.0 / 16.0).abs() < 0.01);
}

/// Test power-of-two dimensions
#[test]
fn test_power_of_two_dimensions() {
    let config = HeadlessConfig::new(1024, 512);
    assert_eq!(config.width, 1024);
    assert_eq!(config.height, 512);
    assert_eq!(config.aligned_bytes_per_row(), 4096);
}

/// Test non-power-of-two dimensions
#[test]
fn test_non_power_of_two_dimensions() {
    let config = HeadlessConfig::new(1000, 500);
    assert_eq!(config.width, 1000);
    assert_eq!(config.height, 500);
    // 1000 * 4 = 4000, aligned to 4096
    assert_eq!(config.aligned_bytes_per_row(), 4096);
}

/// Test very wide texture (minimal height)
#[test]
fn test_very_wide_texture() {
    let config = HeadlessConfig::new(16384, 1);
    assert_eq!(config.width, 16384);
    assert_eq!(config.height, 1);
    // Buffer size should be 16384 * 4 = 65536 for single row
    assert_eq!(config.buffer_size(), 65536);
}

/// Test very tall texture (minimal width)
#[test]
fn test_very_tall_texture() {
    let config = HeadlessConfig::new(1, 16384);
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 16384);
    // 256 bytes per row (aligned) * 16384 rows
    assert_eq!(config.buffer_size(), 256 * 16384);
}

// ============================================================================
// Category 9: HeadlessConfig Builder Pattern
// ============================================================================

/// Test with_format changes format
#[test]
fn test_with_format_changes_format() {
    let config = HeadlessConfig::new(1920, 1080)
        .with_format(TextureFormat::Bgra8UnormSrgb);
    assert_eq!(config.format, TextureFormat::Bgra8UnormSrgb);
}

/// Test with_label sets label
#[test]
fn test_with_label_sets_label() {
    let config = HeadlessConfig::new(1920, 1080)
        .with_label("my_render_target");
    assert_eq!(config.label, Some("my_render_target".to_string()));
}

/// Test with_label can be called multiple times (last wins)
#[test]
fn test_with_label_overwrites() {
    let config = HeadlessConfig::new(1920, 1080)
        .with_label("first")
        .with_label("second");
    assert_eq!(config.label, Some("second".to_string()));
}

/// Test chained builder pattern
#[test]
fn test_builder_chaining() {
    let config = HeadlessConfig::new(1920, 1080)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_msaa(4)
        .with_readback()
        .with_label("chained_target");

    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.format, TextureFormat::Bgra8Unorm);
    assert_eq!(config.sample_count, 4);
    assert!(config.supports_readback());
    assert_eq!(config.label, Some("chained_target".to_string()));
}

// ============================================================================
// Category 10: Clone and Debug Traits
// ============================================================================

/// Test HeadlessConfig Clone
#[test]
fn test_headless_config_clone() {
    let original = HeadlessConfig::new(1920, 1080)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_msaa(4)
        .with_label("original");

    let cloned = original.clone();

    assert_eq!(cloned.width, original.width);
    assert_eq!(cloned.height, original.height);
    assert_eq!(cloned.format, original.format);
    assert_eq!(cloned.sample_count, original.sample_count);
    assert_eq!(cloned.label, original.label);
    assert_eq!(cloned.usage, original.usage);
}

/// Test HeadlessConfig Debug output contains type name
#[test]
fn test_headless_config_debug_contains_type() {
    let config = HeadlessConfig::new(1920, 1080);
    let debug = format!("{:?}", config);
    assert!(debug.contains("HeadlessConfig"));
}

/// Test HeadlessConfig Debug output contains dimensions
#[test]
fn test_headless_config_debug_contains_dimensions() {
    let config = HeadlessConfig::new(1920, 1080);
    let debug = format!("{:?}", config);
    assert!(debug.contains("1920"));
    assert!(debug.contains("1080"));
}

/// Test HeadlessConfig Debug output contains format
#[test]
fn test_headless_config_debug_contains_format() {
    let config = HeadlessConfig::new(1920, 1080)
        .with_format(TextureFormat::Bgra8Unorm);
    let debug = format!("{:?}", config);
    assert!(debug.contains("Bgra8Unorm"));
}

/// Test HeadlessConfig Debug output contains sample_count
#[test]
fn test_headless_config_debug_contains_sample_count() {
    let config = HeadlessConfig::new(1920, 1080).with_msaa(4);
    let debug = format!("{:?}", config);
    assert!(debug.contains("sample_count"));
    assert!(debug.contains("4"));
}

// ============================================================================
// Category 11: HeadlessTarget API Signatures
// ============================================================================

/// Verify HeadlessTarget::new signature
#[test]
fn test_headless_target_new_api_signature() {
    fn _check_new<F: FnOnce(&wgpu::Device, HeadlessConfig) -> Result<HeadlessTarget, HeadlessError>>(_f: F) {}
    _check_new(HeadlessTarget::new);
}

/// Verify HeadlessTarget::view signature
#[test]
fn test_headless_target_view_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> &wgpu::TextureView>(_f: F) {}
    _check(|t| t.view());
}

/// Verify HeadlessTarget::texture signature
#[test]
fn test_headless_target_texture_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> &wgpu::Texture>(_f: F) {}
    _check(|t| t.texture());
}

/// Verify HeadlessTarget::resolve_view signature
#[test]
fn test_headless_target_resolve_view_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> Option<&wgpu::TextureView>>(_f: F) {}
    _check(|t| t.resolve_view());
}

/// Verify HeadlessTarget::resolve_texture signature
#[test]
fn test_headless_target_resolve_texture_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> Option<&wgpu::Texture>>(_f: F) {}
    _check(|t| t.resolve_texture());
}

/// Verify HeadlessTarget::config signature
#[test]
fn test_headless_target_config_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> &HeadlessConfig>(_f: F) {}
    _check(|t| t.config());
}

/// Verify HeadlessTarget::width signature
#[test]
fn test_headless_target_width_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> u32>(_f: F) {}
    _check(|t| t.width());
}

/// Verify HeadlessTarget::height signature
#[test]
fn test_headless_target_height_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> u32>(_f: F) {}
    _check(|t| t.height());
}

/// Verify HeadlessTarget::dimensions signature
#[test]
fn test_headless_target_dimensions_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> (u32, u32)>(_f: F) {}
    _check(|t| t.dimensions());
}

/// Verify HeadlessTarget::format signature
#[test]
fn test_headless_target_format_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> wgpu::TextureFormat>(_f: F) {}
    _check(|t| t.format());
}

/// Verify HeadlessTarget::sample_count signature
#[test]
fn test_headless_target_sample_count_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> u32>(_f: F) {}
    _check(|t| t.sample_count());
}

/// Verify HeadlessTarget::is_msaa_enabled signature
#[test]
fn test_headless_target_is_msaa_enabled_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> bool>(_f: F) {}
    _check(|t| t.is_msaa_enabled());
}

/// Verify HeadlessTarget::aspect_ratio signature
#[test]
fn test_headless_target_aspect_ratio_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> f32>(_f: F) {}
    _check(|t| t.aspect_ratio());
}

/// Verify HeadlessTarget::acquire_frame signature
#[test]
fn test_headless_target_acquire_frame_api_signature() {
    fn _check<'a, F: FnOnce(&'a HeadlessTarget) -> HeadlessFrame<'a>>(_f: F) {}
    _check(|t| t.acquire_frame());
}

/// Verify HeadlessTarget::create_staging_buffer signature
#[test]
fn test_headless_target_create_staging_buffer_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget, &wgpu::Device) -> ReadbackBuffer>(_f: F) {}
    _check(|t, d| t.create_staging_buffer(d));
}

/// Verify HeadlessTarget::screenshot signature
#[test]
fn test_headless_target_screenshot_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget, &wgpu::Device, &wgpu::Queue) -> Result<Vec<u8>, HeadlessError>>(_f: F) {}
    _check(|t, d, q| t.screenshot(d, q));
}

/// Verify HeadlessTarget::screenshot_packed signature
#[test]
fn test_headless_target_screenshot_packed_api_signature() {
    fn _check<F: FnOnce(&HeadlessTarget, &wgpu::Device, &wgpu::Queue) -> Result<Vec<u8>, HeadlessError>>(_f: F) {}
    _check(|t, d, q| t.screenshot_packed(d, q));
}

/// Verify HeadlessTarget implements Debug
#[test]
fn test_headless_target_implements_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<HeadlessTarget>();
}

// ============================================================================
// Category 12: HeadlessFrame API Signatures
// ============================================================================

/// Verify HeadlessFrame::view signature
#[test]
fn test_headless_frame_view_api_signature() {
    fn _check<'a, F: FnOnce(&'a HeadlessFrame<'a>) -> &'a wgpu::TextureView>(_f: F) {}
    _check(|f| f.view());
}

/// Verify HeadlessFrame::resolve_view signature
#[test]
fn test_headless_frame_resolve_view_api_signature() {
    fn _check<'a, F: FnOnce(&'a HeadlessFrame<'a>) -> Option<&'a wgpu::TextureView>>(_f: F) {}
    _check(|f| f.resolve_view());
}

/// Verify HeadlessFrame::width signature
#[test]
fn test_headless_frame_width_api_signature() {
    fn _check<'a, F: FnOnce(&HeadlessFrame<'a>) -> u32>(_f: F) {}
    _check(|f| f.width());
}

/// Verify HeadlessFrame::height signature
#[test]
fn test_headless_frame_height_api_signature() {
    fn _check<'a, F: FnOnce(&HeadlessFrame<'a>) -> u32>(_f: F) {}
    _check(|f| f.height());
}

/// Verify HeadlessFrame::dimensions signature
#[test]
fn test_headless_frame_dimensions_api_signature() {
    fn _check<'a, F: FnOnce(&HeadlessFrame<'a>) -> (u32, u32)>(_f: F) {}
    _check(|f| f.dimensions());
}

/// Verify HeadlessFrame::format signature
#[test]
fn test_headless_frame_format_api_signature() {
    fn _check<'a, F: FnOnce(&HeadlessFrame<'a>) -> wgpu::TextureFormat>(_f: F) {}
    _check(|f| f.format());
}

/// Verify HeadlessFrame::sample_count signature
#[test]
fn test_headless_frame_sample_count_api_signature() {
    fn _check<'a, F: FnOnce(&HeadlessFrame<'a>) -> u32>(_f: F) {}
    _check(|f| f.sample_count());
}

/// Verify HeadlessFrame::is_msaa_enabled signature
#[test]
fn test_headless_frame_is_msaa_enabled_api_signature() {
    fn _check<'a, F: FnOnce(&HeadlessFrame<'a>) -> bool>(_f: F) {}
    _check(|f| f.is_msaa_enabled());
}

/// Verify HeadlessFrame::aspect_ratio signature
#[test]
fn test_headless_frame_aspect_ratio_api_signature() {
    fn _check<'a, F: FnOnce(&HeadlessFrame<'a>) -> f32>(_f: F) {}
    _check(|f| f.aspect_ratio());
}

/// Verify HeadlessFrame implements Debug
#[test]
fn test_headless_frame_implements_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<HeadlessFrame<'_>>();
}

// ============================================================================
// Category 13: ReadbackBuffer API Signatures
// ============================================================================

/// Verify ReadbackBuffer::buffer signature
#[test]
fn test_readback_buffer_buffer_api_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer) -> &wgpu::Buffer>(_f: F) {}
    _check(|r| r.buffer());
}

/// Verify ReadbackBuffer::bytes_per_row signature
#[test]
fn test_readback_buffer_bytes_per_row_api_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer) -> u32>(_f: F) {}
    _check(|r| r.bytes_per_row());
}

/// Verify ReadbackBuffer::bytes_per_pixel signature
#[test]
fn test_readback_buffer_bytes_per_pixel_api_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer) -> u32>(_f: F) {}
    _check(|r| r.bytes_per_pixel());
}

/// Verify ReadbackBuffer::width signature
#[test]
fn test_readback_buffer_width_api_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer) -> u32>(_f: F) {}
    _check(|r| r.width());
}

/// Verify ReadbackBuffer::height signature
#[test]
fn test_readback_buffer_height_api_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer) -> u32>(_f: F) {}
    _check(|r| r.height());
}

/// Verify ReadbackBuffer::format signature
#[test]
fn test_readback_buffer_format_api_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer) -> wgpu::TextureFormat>(_f: F) {}
    _check(|r| r.format());
}

/// Verify ReadbackBuffer::size signature
#[test]
fn test_readback_buffer_size_api_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer) -> u64>(_f: F) {}
    _check(|r| r.size());
}

/// Verify ReadbackBuffer::map_read signature
#[test]
fn test_readback_buffer_map_read_api_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer, &wgpu::Device) -> Result<Vec<u8>, HeadlessError>>(_f: F) {}
    _check(|r, d| r.map_read(d));
}

/// Verify ReadbackBuffer::map_read_packed signature
#[test]
fn test_readback_buffer_map_read_packed_api_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer, &wgpu::Device) -> Result<Vec<u8>, HeadlessError>>(_f: F) {}
    _check(|r, d| r.map_read_packed(d));
}

/// Verify ReadbackBuffer implements Debug
#[test]
fn test_readback_buffer_implements_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<ReadbackBuffer>();
}

// ============================================================================
// Category 14: HeadlessRenderer API Signatures
// ============================================================================

/// Verify HeadlessRenderer::new signature
#[test]
fn test_headless_renderer_new_api_signature() {
    fn _check<F: FnOnce(&wgpu::Device, HeadlessConfig) -> Result<HeadlessRenderer, HeadlessError>>(_f: F) {}
    _check(HeadlessRenderer::new);
}

/// Verify HeadlessRenderer::acquire_frame signature
#[test]
fn test_headless_renderer_acquire_frame_api_signature() {
    fn _check<'a, F: FnOnce(&'a mut HeadlessRenderer) -> HeadlessFrame<'a>>(_f: F) {}
    _check(|r| r.acquire_frame());
}

/// Verify HeadlessRenderer::target signature
#[test]
fn test_headless_renderer_target_api_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> &HeadlessTarget>(_f: F) {}
    _check(|r| r.target());
}

/// Verify HeadlessRenderer::target_mut signature
#[test]
fn test_headless_renderer_target_mut_api_signature() {
    fn _check<F: FnOnce(&mut HeadlessRenderer) -> &mut HeadlessTarget>(_f: F) {}
    _check(|r| r.target_mut());
}

/// Verify HeadlessRenderer::frame_count signature
#[test]
fn test_headless_renderer_frame_count_api_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> u64>(_f: F) {}
    _check(|r| r.frame_count());
}

/// Verify HeadlessRenderer::view signature
#[test]
fn test_headless_renderer_view_api_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> &wgpu::TextureView>(_f: F) {}
    _check(|r| r.view());
}

/// Verify HeadlessRenderer::resolve_view signature
#[test]
fn test_headless_renderer_resolve_view_api_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> Option<&wgpu::TextureView>>(_f: F) {}
    _check(|r| r.resolve_view());
}

/// Verify HeadlessRenderer::width signature
#[test]
fn test_headless_renderer_width_api_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> u32>(_f: F) {}
    _check(|r| r.width());
}

/// Verify HeadlessRenderer::height signature
#[test]
fn test_headless_renderer_height_api_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> u32>(_f: F) {}
    _check(|r| r.height());
}

/// Verify HeadlessRenderer::dimensions signature
#[test]
fn test_headless_renderer_dimensions_api_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> (u32, u32)>(_f: F) {}
    _check(|r| r.dimensions());
}

/// Verify HeadlessRenderer::format signature
#[test]
fn test_headless_renderer_format_api_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> wgpu::TextureFormat>(_f: F) {}
    _check(|r| r.format());
}

/// Verify HeadlessRenderer::resize signature
#[test]
fn test_headless_renderer_resize_api_signature() {
    fn _check<F: FnOnce(&mut HeadlessRenderer, &wgpu::Device, u32, u32) -> Result<(), HeadlessError>>(_f: F) {}
    _check(|r, d, w, h| r.resize(d, w, h));
}

/// Verify HeadlessRenderer::screenshot signature
#[test]
fn test_headless_renderer_screenshot_api_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer, &wgpu::Device, &wgpu::Queue) -> Result<Vec<u8>, HeadlessError>>(_f: F) {}
    _check(|r, d, q| r.screenshot(d, q));
}

/// Verify HeadlessRenderer::screenshot_packed signature
#[test]
fn test_headless_renderer_screenshot_packed_api_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer, &wgpu::Device, &wgpu::Queue) -> Result<Vec<u8>, HeadlessError>>(_f: F) {}
    _check(|r, d, q| r.screenshot_packed(d, q));
}

/// Verify HeadlessRenderer implements Debug
#[test]
fn test_headless_renderer_implements_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<HeadlessRenderer>();
}

// ============================================================================
// Category 15: Additional Calculation Tests
// ============================================================================

/// Test buffer size calculation overflow safety
#[test]
fn test_buffer_size_no_overflow() {
    // Use u64 to prevent overflow for large textures
    let config = HeadlessConfig::new(16384, 16384);
    // 16384 * 4 = 65536 bytes per row
    // 65536 * 16384 = 1,073,741,824 bytes (1 GB)
    let size = config.buffer_size();
    assert!(size > 0);
    assert_eq!(size, 65536u64 * 16384u64);
}

/// Test aligned_bytes_per_row formula: (unaligned + 255) & !255
#[test]
fn test_alignment_formula_correctness() {
    // Test the alignment formula explicitly
    let widths = [1, 63, 64, 65, 100, 127, 128, 129, 256, 1000, 1920, 3840];

    for &width in &widths {
        let config = HeadlessConfig::new(width, 100);
        let unaligned = width * 4;
        let expected = (unaligned + 255) & !255;
        assert_eq!(
            config.aligned_bytes_per_row(),
            expected,
            "Width {} failed: got {}, expected {}",
            width,
            config.aligned_bytes_per_row(),
            expected
        );
    }
}

/// Test bytes_per_row is always a multiple of 256
#[test]
fn test_alignment_always_multiple_of_256() {
    let widths = [1, 50, 100, 200, 300, 500, 1000, 2000];

    for &width in &widths {
        let config = HeadlessConfig::new(width, 100);
        let aligned = config.aligned_bytes_per_row();
        assert_eq!(
            aligned % 256,
            0,
            "Width {} produced aligned_bytes_per_row {} which is not a multiple of 256",
            width,
            aligned
        );
    }
}

/// Test minimum aligned_bytes_per_row is 256
#[test]
fn test_minimum_aligned_bytes_per_row() {
    let config = HeadlessConfig::new(1, 1);
    assert!(config.aligned_bytes_per_row() >= 256);
}

/// Test MSAA enable consistency
#[test]
fn test_msaa_enable_consistency() {
    for sample_count in [1, 2, 3, 4, 5, 6, 7, 8, 16, 32] {
        let config = HeadlessConfig::new(1920, 1080).with_msaa(sample_count);
        let expected_enabled = config.sample_count > 1;
        assert_eq!(
            config.is_msaa_enabled(),
            expected_enabled,
            "MSAA check failed for input sample_count {}: actual={}, expected_enabled={}",
            sample_count,
            config.sample_count,
            expected_enabled
        );
    }
}

/// Test that all HeadlessError variants have distinct Display messages
#[test]
fn test_all_error_variants_have_distinct_messages() {
    let errors = [
        HeadlessError::TextureCreationFailed("x".to_string()),
        HeadlessError::StagingBufferFailed("x".to_string()),
        HeadlessError::BufferMapFailed("x".to_string()),
        HeadlessError::NotInitialized,
        HeadlessError::invalid_dimensions(1, 1),
        HeadlessError::ScreenshotSaveFailed("x".to_string()),
        HeadlessError::ResolveFailed("x".to_string()),
    ];

    let messages: Vec<String> = errors.iter().map(|e| format!("{}", e)).collect();

    // Check uniqueness
    for i in 0..messages.len() {
        for j in (i + 1)..messages.len() {
            assert_ne!(
                messages[i], messages[j],
                "Error variants {} and {} have the same message",
                i, j
            );
        }
    }
}

/// Test HeadlessConfig equality via field comparison
#[test]
fn test_config_equality_via_clone() {
    let config1 = HeadlessConfig::new(1920, 1080)
        .with_format(TextureFormat::Rgba8Unorm)
        .with_msaa(4)
        .with_label("test");

    let config2 = config1.clone();

    assert_eq!(config1.width, config2.width);
    assert_eq!(config1.height, config2.height);
    assert_eq!(config1.format, config2.format);
    assert_eq!(config1.sample_count, config2.sample_count);
    assert_eq!(config1.usage, config2.usage);
    assert_eq!(config1.label, config2.label);
}

/// Test bytes per pixel covers all standard render formats
#[test]
fn test_bytes_per_pixel_comprehensive() {
    // Map of format to expected bytes
    let format_bytes = [
        (TextureFormat::R8Unorm, 1),
        (TextureFormat::R8Snorm, 1),
        (TextureFormat::R8Uint, 1),
        (TextureFormat::R8Sint, 1),
        (TextureFormat::Rg8Unorm, 2),
        (TextureFormat::Rg8Snorm, 2),
        (TextureFormat::Rg8Uint, 2),
        (TextureFormat::Rg8Sint, 2),
        (TextureFormat::Rgba8Unorm, 4),
        (TextureFormat::Rgba8UnormSrgb, 4),
        (TextureFormat::Rgba8Snorm, 4),
        (TextureFormat::Rgba8Uint, 4),
        (TextureFormat::Rgba8Sint, 4),
        (TextureFormat::Bgra8Unorm, 4),
        (TextureFormat::Bgra8UnormSrgb, 4),
        (TextureFormat::Rgb10a2Unorm, 4),
        (TextureFormat::Rgba16Float, 8),
        (TextureFormat::Rgba16Uint, 8),
        (TextureFormat::Rgba16Sint, 8),
        (TextureFormat::Rgba16Unorm, 8),
        (TextureFormat::Rgba16Snorm, 8),
        (TextureFormat::Rgba32Float, 16),
        (TextureFormat::Rgba32Uint, 16),
        (TextureFormat::Rgba32Sint, 16),
    ];

    for (format, expected) in format_bytes {
        let config = HeadlessConfig::new(100, 100).with_format(format);
        assert_eq!(
            config.bytes_per_pixel(),
            expected,
            "Format {:?} expected {} bytes, got {}",
            format,
            expected,
            config.bytes_per_pixel()
        );
    }
}
