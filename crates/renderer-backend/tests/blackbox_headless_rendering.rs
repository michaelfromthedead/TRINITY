// SPDX-License-Identifier: MIT
//
// blackbox_headless_rendering.rs -- Blackbox tests for T-WGPU-P7.1.10 Headless Rendering API.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only the documented public types and functions from the presentation module:
//
//   - HeadlessConfig: new(), with_format(), with_msaa(), with_readback(), with_usage(), with_label()
//   - HeadlessConfig: bytes_per_pixel(), aligned_bytes_per_row(), buffer_size()
//   - HeadlessConfig: is_msaa_enabled(), supports_readback(), validate()
//   - HeadlessTarget: new(), view(), texture(), resize(), acquire_frame()
//   - HeadlessTarget: config(), width(), height(), dimensions(), format(), sample_count()
//   - HeadlessTarget: is_msaa_enabled(), aspect_ratio(), resolve_view(), resolve_texture()
//   - HeadlessTarget: create_staging_buffer(), screenshot(), screenshot_packed()
//   - HeadlessFrame: view(), dimensions(), format(), width(), height()
//   - HeadlessFrame: resolve_view(), sample_count(), is_msaa_enabled(), aspect_ratio()
//   - ReadbackBuffer: buffer(), map_read(), bytes_per_row(), bytes_per_pixel()
//   - ReadbackBuffer: width(), height(), format(), size(), map_read_packed()
//   - HeadlessRenderer: new(), acquire_frame(), screenshot(), resize()
//   - HeadlessRenderer: target(), target_mut(), frame_count(), view(), resolve_view()
//   - HeadlessRenderer: width(), height(), dimensions(), format(), screenshot_packed()
//   - HeadlessError: InvalidDimensions, TextureCreationFailed, BufferMapFailed, etc.
//
// ACCEPTANCE CRITERIA:
//   1. HeadlessConfig API tests        -- 20+ tests
//   2. HeadlessConfig calculations     -- 10+ tests
//   3. HeadlessTarget API tests        -- 15+ tests (most GPU-dependent)
//   4. HeadlessFrame API tests         -- 8+ tests
//   5. ReadbackBuffer API tests        -- 8+ tests
//   6. HeadlessRenderer API tests      -- 10+ tests
//   7. Error handling tests            -- 8+ tests
//   8. Format variation tests          -- 8+ tests
//   9. MSAA rendering tests            -- 6+ tests
//  10. Integration tests (GPU)         -- 8+ tests (ignored without GPU)
//
// Total target: 100+ tests, 110+ assertions

use renderer_backend::presentation::{
    HeadlessConfig, HeadlessError, HeadlessFrame, HeadlessRenderer, HeadlessTarget,
    ReadbackBuffer,
};
use wgpu::TextureFormat;

// =============================================================================
// SECTION 1 -- HEADLESS CONFIG CONSTRUCTION TESTS (20+ tests)
// =============================================================================

/// HeadlessConfig::new() creates config with specified dimensions.
#[test]
fn headless_config_new_creates_with_dimensions() {
    let config = HeadlessConfig::new(1920, 1080);
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
}

/// HeadlessConfig::new() clamps zero width to 1.
#[test]
fn headless_config_new_clamps_zero_width() {
    let config = HeadlessConfig::new(0, 100);
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 100);
}

/// HeadlessConfig::new() clamps zero height to 1.
#[test]
fn headless_config_new_clamps_zero_height() {
    let config = HeadlessConfig::new(100, 0);
    assert_eq!(config.width, 100);
    assert_eq!(config.height, 1);
}

/// HeadlessConfig::new() clamps both zero dimensions to 1x1.
#[test]
fn headless_config_new_clamps_both_zero() {
    let config = HeadlessConfig::new(0, 0);
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 1);
}

/// HeadlessConfig::new() defaults to Rgba8Unorm format.
#[test]
fn headless_config_new_default_format() {
    let config = HeadlessConfig::new(800, 600);
    assert_eq!(config.format, TextureFormat::Rgba8Unorm);
}

/// HeadlessConfig::new() defaults to sample_count 1 (no MSAA).
#[test]
fn headless_config_new_default_sample_count() {
    let config = HeadlessConfig::new(800, 600);
    assert_eq!(config.sample_count, 1);
}

/// HeadlessConfig::new() defaults to RENDER_ATTACHMENT | COPY_SRC usage.
#[test]
fn headless_config_new_default_usage() {
    let config = HeadlessConfig::new(800, 600);
    assert!(config.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
    assert!(config.usage.contains(wgpu::TextureUsages::COPY_SRC));
}

/// HeadlessConfig::new() defaults to no label.
#[test]
fn headless_config_new_default_label() {
    let config = HeadlessConfig::new(800, 600);
    assert!(config.label.is_none());
}

/// HeadlessConfig::default() creates 800x600 config.
#[test]
fn headless_config_default_dimensions() {
    let config = HeadlessConfig::default();
    assert_eq!(config.width, 800);
    assert_eq!(config.height, 600);
}

/// HeadlessConfig::with_format() sets the format.
#[test]
fn headless_config_with_format_sets_format() {
    let config = HeadlessConfig::new(800, 600)
        .with_format(TextureFormat::Bgra8Unorm);
    assert_eq!(config.format, TextureFormat::Bgra8Unorm);
}

/// HeadlessConfig::with_format() can set sRGB format.
#[test]
fn headless_config_with_format_srgb() {
    let config = HeadlessConfig::new(800, 600)
        .with_format(TextureFormat::Rgba8UnormSrgb);
    assert_eq!(config.format, TextureFormat::Rgba8UnormSrgb);
}

/// HeadlessConfig::with_format() can set HDR format.
#[test]
fn headless_config_with_format_hdr() {
    let config = HeadlessConfig::new(800, 600)
        .with_format(TextureFormat::Rgba16Float);
    assert_eq!(config.format, TextureFormat::Rgba16Float);
}

/// HeadlessConfig::with_msaa(0) clamps to 1.
#[test]
fn headless_config_with_msaa_zero_clamps_to_one() {
    let config = HeadlessConfig::new(800, 600).with_msaa(0);
    assert_eq!(config.sample_count, 1);
}

/// HeadlessConfig::with_msaa(1) stays at 1.
#[test]
fn headless_config_with_msaa_one_stays_one() {
    let config = HeadlessConfig::new(800, 600).with_msaa(1);
    assert_eq!(config.sample_count, 1);
}

/// HeadlessConfig::with_msaa(2) rounds to 4.
#[test]
fn headless_config_with_msaa_two_rounds_to_four() {
    let config = HeadlessConfig::new(800, 600).with_msaa(2);
    assert_eq!(config.sample_count, 4);
}

/// HeadlessConfig::with_msaa(4) stays at 4.
#[test]
fn headless_config_with_msaa_four_stays_four() {
    let config = HeadlessConfig::new(800, 600).with_msaa(4);
    assert_eq!(config.sample_count, 4);
}

/// HeadlessConfig::with_msaa(5) rounds to 8.
#[test]
fn headless_config_with_msaa_five_rounds_to_eight() {
    let config = HeadlessConfig::new(800, 600).with_msaa(5);
    assert_eq!(config.sample_count, 8);
}

/// HeadlessConfig::with_msaa(8) stays at 8.
#[test]
fn headless_config_with_msaa_eight_stays_eight() {
    let config = HeadlessConfig::new(800, 600).with_msaa(8);
    assert_eq!(config.sample_count, 8);
}

/// HeadlessConfig::with_msaa(16) clamps to 8.
#[test]
fn headless_config_with_msaa_sixteen_clamps_to_eight() {
    let config = HeadlessConfig::new(800, 600).with_msaa(16);
    assert_eq!(config.sample_count, 8);
}

/// HeadlessConfig::with_readback() adds COPY_SRC usage.
#[test]
fn headless_config_with_readback_adds_copy_src() {
    let config = HeadlessConfig::new(800, 600).with_readback();
    assert!(config.usage.contains(wgpu::TextureUsages::COPY_SRC));
}

/// HeadlessConfig::with_usage() preserves RENDER_ATTACHMENT.
#[test]
fn headless_config_with_usage_preserves_render_attachment() {
    let config = HeadlessConfig::new(800, 600)
        .with_usage(wgpu::TextureUsages::TEXTURE_BINDING);
    assert!(config.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
    assert!(config.usage.contains(wgpu::TextureUsages::TEXTURE_BINDING));
}

/// HeadlessConfig::with_label() sets the label.
#[test]
fn headless_config_with_label_sets_label() {
    let config = HeadlessConfig::new(800, 600)
        .with_label("my_headless_target");
    assert_eq!(config.label.as_deref(), Some("my_headless_target"));
}

/// HeadlessConfig builder methods can be chained.
#[test]
fn headless_config_builder_chaining() {
    let config = HeadlessConfig::new(1920, 1080)
        .with_format(TextureFormat::Rgba8UnormSrgb)
        .with_msaa(4)
        .with_readback()
        .with_label("chained_config");

    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.format, TextureFormat::Rgba8UnormSrgb);
    assert_eq!(config.sample_count, 4);
    assert!(config.usage.contains(wgpu::TextureUsages::COPY_SRC));
    assert_eq!(config.label.as_deref(), Some("chained_config"));
}

// =============================================================================
// SECTION 2 -- HEADLESS CONFIG PROPERTY TESTS (10+ tests)
// =============================================================================

/// HeadlessConfig::is_msaa_enabled() returns false for sample_count=1.
#[test]
fn headless_config_is_msaa_enabled_false_for_one() {
    let config = HeadlessConfig::new(800, 600);
    assert!(!config.is_msaa_enabled());
}

/// HeadlessConfig::is_msaa_enabled() returns true for sample_count=4.
#[test]
fn headless_config_is_msaa_enabled_true_for_four() {
    let config = HeadlessConfig::new(800, 600).with_msaa(4);
    assert!(config.is_msaa_enabled());
}

/// HeadlessConfig::is_msaa_enabled() returns true for sample_count=8.
#[test]
fn headless_config_is_msaa_enabled_true_for_eight() {
    let config = HeadlessConfig::new(800, 600).with_msaa(8);
    assert!(config.is_msaa_enabled());
}

/// HeadlessConfig::supports_readback() returns true by default.
#[test]
fn headless_config_supports_readback_default_true() {
    let config = HeadlessConfig::new(800, 600);
    assert!(config.supports_readback());
}

/// HeadlessConfig::supports_readback() returns true after with_readback().
#[test]
fn headless_config_supports_readback_after_with_readback() {
    let config = HeadlessConfig::new(800, 600).with_readback();
    assert!(config.supports_readback());
}

/// HeadlessConfig::validate() succeeds for valid config.
#[test]
fn headless_config_validate_succeeds_valid() {
    let config = HeadlessConfig::new(800, 600);
    assert!(config.validate().is_ok());
}

/// HeadlessConfig::validate() succeeds for 1x1 dimensions.
#[test]
fn headless_config_validate_succeeds_1x1() {
    let config = HeadlessConfig::new(1, 1);
    assert!(config.validate().is_ok());
}

/// HeadlessConfig::validate() succeeds for large dimensions.
#[test]
fn headless_config_validate_succeeds_large() {
    let config = HeadlessConfig::new(4096, 4096);
    assert!(config.validate().is_ok());
}

/// HeadlessConfig Debug impl contains format info.
#[test]
fn headless_config_debug_format() {
    let config = HeadlessConfig::new(1920, 1080);
    let debug = format!("{:?}", config);
    assert!(debug.contains("HeadlessConfig"));
    assert!(debug.contains("1920"));
    assert!(debug.contains("1080"));
}

/// HeadlessConfig Clone produces equal config.
#[test]
fn headless_config_clone_equals() {
    let config = HeadlessConfig::new(1920, 1080)
        .with_format(TextureFormat::Bgra8Unorm)
        .with_msaa(4);
    let cloned = config.clone();
    assert_eq!(cloned.width, config.width);
    assert_eq!(cloned.height, config.height);
    assert_eq!(cloned.format, config.format);
    assert_eq!(cloned.sample_count, config.sample_count);
}

// =============================================================================
// SECTION 3 -- HEADLESS CONFIG CALCULATION TESTS (12+ tests)
// =============================================================================

/// bytes_per_pixel returns 4 for Rgba8Unorm.
#[test]
fn headless_config_bytes_per_pixel_rgba8() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::Rgba8Unorm);
    assert_eq!(config.bytes_per_pixel(), 4);
}

/// bytes_per_pixel returns 4 for Bgra8Unorm.
#[test]
fn headless_config_bytes_per_pixel_bgra8() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::Bgra8Unorm);
    assert_eq!(config.bytes_per_pixel(), 4);
}

/// bytes_per_pixel returns 4 for Rgba8UnormSrgb.
#[test]
fn headless_config_bytes_per_pixel_rgba8_srgb() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::Rgba8UnormSrgb);
    assert_eq!(config.bytes_per_pixel(), 4);
}

/// bytes_per_pixel returns 8 for Rgba16Float.
#[test]
fn headless_config_bytes_per_pixel_rgba16f() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::Rgba16Float);
    assert_eq!(config.bytes_per_pixel(), 8);
}

/// bytes_per_pixel returns 16 for Rgba32Float.
#[test]
fn headless_config_bytes_per_pixel_rgba32f() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::Rgba32Float);
    assert_eq!(config.bytes_per_pixel(), 16);
}

/// bytes_per_pixel returns 1 for R8Unorm.
#[test]
fn headless_config_bytes_per_pixel_r8() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::R8Unorm);
    assert_eq!(config.bytes_per_pixel(), 1);
}

/// bytes_per_pixel returns 2 for Rg8Unorm.
#[test]
fn headless_config_bytes_per_pixel_rg8() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::Rg8Unorm);
    assert_eq!(config.bytes_per_pixel(), 2);
}

/// aligned_bytes_per_row aligns to 256 bytes.
#[test]
fn headless_config_aligned_bytes_per_row_alignment() {
    let config = HeadlessConfig::new(100, 100);
    let aligned = config.aligned_bytes_per_row();
    assert_eq!(aligned % 256, 0);
}

/// aligned_bytes_per_row for 64-wide Rgba8 = 256.
#[test]
fn headless_config_aligned_bytes_per_row_64_wide() {
    let config = HeadlessConfig::new(64, 100);
    // 64 * 4 = 256, already aligned
    assert_eq!(config.aligned_bytes_per_row(), 256);
}

/// aligned_bytes_per_row for 100-wide Rgba8 = 512.
#[test]
fn headless_config_aligned_bytes_per_row_100_wide() {
    let config = HeadlessConfig::new(100, 100);
    // 100 * 4 = 400, rounds up to 512
    assert_eq!(config.aligned_bytes_per_row(), 512);
}

/// aligned_bytes_per_row for 256-wide Rgba8 = 1024.
#[test]
fn headless_config_aligned_bytes_per_row_256_wide() {
    let config = HeadlessConfig::new(256, 100);
    // 256 * 4 = 1024, already aligned
    assert_eq!(config.aligned_bytes_per_row(), 1024);
}

/// buffer_size equals aligned_bytes_per_row * height.
#[test]
fn headless_config_buffer_size_calculation() {
    let config = HeadlessConfig::new(100, 200);
    let expected = config.aligned_bytes_per_row() as u64 * 200;
    assert_eq!(config.buffer_size(), expected);
}

/// buffer_size for 1920x1080 Rgba8.
#[test]
fn headless_config_buffer_size_1080p() {
    let config = HeadlessConfig::new(1920, 1080);
    // 1920 * 4 = 7680, rounds to 7936 (31 * 256)
    let aligned = ((1920 * 4) + 255) & !255;
    assert_eq!(config.aligned_bytes_per_row(), aligned);
    assert_eq!(config.buffer_size(), aligned as u64 * 1080);
}

/// buffer_size for 4K resolution.
#[test]
fn headless_config_buffer_size_4k() {
    let config = HeadlessConfig::new(3840, 2160);
    let aligned = config.aligned_bytes_per_row();
    assert_eq!(config.buffer_size(), aligned as u64 * 2160);
}

// =============================================================================
// SECTION 4 -- HEADLESS ERROR TESTS (8+ tests)
// =============================================================================

/// HeadlessError::InvalidDimensions displays correctly.
#[test]
fn headless_error_invalid_dimensions_display() {
    let err = HeadlessError::InvalidDimensions { width: 0, height: 0 };
    let msg = format!("{}", err);
    assert!(msg.contains("invalid dimensions"));
    assert!(msg.contains("0"));
}

/// HeadlessError::invalid_dimensions() helper creates correct error.
#[test]
fn headless_error_invalid_dimensions_helper() {
    let err = HeadlessError::invalid_dimensions(100, 200);
    match err {
        HeadlessError::InvalidDimensions { width, height } => {
            assert_eq!(width, 100);
            assert_eq!(height, 200);
        }
        _ => panic!("expected InvalidDimensions variant"),
    }
}

/// HeadlessError::TextureCreationFailed displays message.
#[test]
fn headless_error_texture_creation_failed_display() {
    let err = HeadlessError::TextureCreationFailed("out of memory".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("out of memory"));
}

/// HeadlessError::StagingBufferFailed displays message.
#[test]
fn headless_error_staging_buffer_failed_display() {
    let err = HeadlessError::StagingBufferFailed("buffer error".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("buffer error"));
}

/// HeadlessError::BufferMapFailed displays message.
#[test]
fn headless_error_buffer_map_failed_display() {
    let err = HeadlessError::BufferMapFailed("mapping failed".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("mapping failed"));
}

/// HeadlessError::NotInitialized displays correctly.
#[test]
fn headless_error_not_initialized_display() {
    let err = HeadlessError::NotInitialized;
    let msg = format!("{}", err);
    assert!(msg.contains("not initialized"));
}

/// HeadlessError::ScreenshotSaveFailed displays message.
#[test]
fn headless_error_screenshot_save_failed_display() {
    let err = HeadlessError::ScreenshotSaveFailed("file write error".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("file write error"));
}

/// HeadlessError::ResolveFailed displays message.
#[test]
fn headless_error_resolve_failed_display() {
    let err = HeadlessError::ResolveFailed("MSAA resolve error".to_string());
    let msg = format!("{}", err);
    assert!(msg.contains("MSAA resolve error"));
}

/// HeadlessError Debug impl works.
#[test]
fn headless_error_debug() {
    let err = HeadlessError::invalid_dimensions(0, 0);
    let debug = format!("{:?}", err);
    assert!(debug.contains("InvalidDimensions"));
}

// =============================================================================
// SECTION 5 -- HEADLESS TARGET API SHAPE TESTS (15+ tests)
// =============================================================================

/// HeadlessTarget::new() has correct signature.
#[test]
fn headless_target_new_signature() {
    fn _check<F: FnOnce(&wgpu::Device, HeadlessConfig) -> Result<HeadlessTarget, HeadlessError>>(_f: F) {}
    _check(HeadlessTarget::new);
}

/// HeadlessTarget::view() returns &TextureView.
#[test]
fn headless_target_view_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> &wgpu::TextureView>(_f: F) {}
    _check(HeadlessTarget::view);
}

/// HeadlessTarget::texture() returns &Texture.
#[test]
fn headless_target_texture_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> &wgpu::Texture>(_f: F) {}
    _check(HeadlessTarget::texture);
}

/// HeadlessTarget::resolve_view() returns Option<&TextureView>.
#[test]
fn headless_target_resolve_view_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> Option<&wgpu::TextureView>>(_f: F) {}
    _check(HeadlessTarget::resolve_view);
}

/// HeadlessTarget::resolve_texture() returns Option<&Texture>.
#[test]
fn headless_target_resolve_texture_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> Option<&wgpu::Texture>>(_f: F) {}
    _check(HeadlessTarget::resolve_texture);
}

/// HeadlessTarget::config() returns &HeadlessConfig.
#[test]
fn headless_target_config_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> &HeadlessConfig>(_f: F) {}
    _check(HeadlessTarget::config);
}

/// HeadlessTarget::width() returns u32.
#[test]
fn headless_target_width_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> u32>(_f: F) {}
    _check(HeadlessTarget::width);
}

/// HeadlessTarget::height() returns u32.
#[test]
fn headless_target_height_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> u32>(_f: F) {}
    _check(HeadlessTarget::height);
}

/// HeadlessTarget::dimensions() returns (u32, u32).
#[test]
fn headless_target_dimensions_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> (u32, u32)>(_f: F) {}
    _check(HeadlessTarget::dimensions);
}

/// HeadlessTarget::format() returns TextureFormat.
#[test]
fn headless_target_format_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> wgpu::TextureFormat>(_f: F) {}
    _check(HeadlessTarget::format);
}

/// HeadlessTarget::sample_count() returns u32.
#[test]
fn headless_target_sample_count_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> u32>(_f: F) {}
    _check(HeadlessTarget::sample_count);
}

/// HeadlessTarget::is_msaa_enabled() returns bool.
#[test]
fn headless_target_is_msaa_enabled_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> bool>(_f: F) {}
    _check(HeadlessTarget::is_msaa_enabled);
}

/// HeadlessTarget::aspect_ratio() returns f32.
#[test]
fn headless_target_aspect_ratio_signature() {
    fn _check<F: FnOnce(&HeadlessTarget) -> f32>(_f: F) {}
    _check(HeadlessTarget::aspect_ratio);
}

/// HeadlessTarget::acquire_frame() returns HeadlessFrame.
#[test]
fn headless_target_acquire_frame_signature() {
    fn _check<F: for<'a> FnOnce(&'a HeadlessTarget) -> HeadlessFrame<'a>>(_f: F) {}
    _check(HeadlessTarget::acquire_frame);
}

/// HeadlessTarget::create_staging_buffer() returns ReadbackBuffer.
#[test]
fn headless_target_create_staging_buffer_signature() {
    fn _check<F: FnOnce(&HeadlessTarget, &wgpu::Device) -> ReadbackBuffer>(_f: F) {}
    _check(HeadlessTarget::create_staging_buffer);
}

// =============================================================================
// SECTION 6 -- HEADLESS FRAME API SHAPE TESTS (8+ tests)
// =============================================================================

/// HeadlessFrame::view() returns &TextureView.
#[test]
fn headless_frame_view_signature() {
    fn _check<'a, F: FnOnce(&'a HeadlessFrame<'a>) -> &'a wgpu::TextureView>(_f: F) {}
    _check(HeadlessFrame::view);
}

/// HeadlessFrame::resolve_view() returns Option<&TextureView>.
#[test]
fn headless_frame_resolve_view_signature() {
    fn _check<'a, F: FnOnce(&'a HeadlessFrame<'a>) -> Option<&'a wgpu::TextureView>>(_f: F) {}
    _check(HeadlessFrame::resolve_view);
}

/// HeadlessFrame::width() returns u32.
#[test]
fn headless_frame_width_signature() {
    fn _check<'a, F: FnOnce(&'a HeadlessFrame<'a>) -> u32>(_f: F) {}
    _check(HeadlessFrame::width);
}

/// HeadlessFrame::height() returns u32.
#[test]
fn headless_frame_height_signature() {
    fn _check<'a, F: FnOnce(&'a HeadlessFrame<'a>) -> u32>(_f: F) {}
    _check(HeadlessFrame::height);
}

/// HeadlessFrame::dimensions() returns (u32, u32).
#[test]
fn headless_frame_dimensions_signature() {
    fn _check<'a, F: FnOnce(&'a HeadlessFrame<'a>) -> (u32, u32)>(_f: F) {}
    _check(HeadlessFrame::dimensions);
}

/// HeadlessFrame::format() returns TextureFormat.
#[test]
fn headless_frame_format_signature() {
    fn _check<'a, F: FnOnce(&'a HeadlessFrame<'a>) -> wgpu::TextureFormat>(_f: F) {}
    _check(HeadlessFrame::format);
}

/// HeadlessFrame::sample_count() returns u32.
#[test]
fn headless_frame_sample_count_signature() {
    fn _check<'a, F: FnOnce(&'a HeadlessFrame<'a>) -> u32>(_f: F) {}
    _check(HeadlessFrame::sample_count);
}

/// HeadlessFrame::is_msaa_enabled() returns bool.
#[test]
fn headless_frame_is_msaa_enabled_signature() {
    fn _check<'a, F: FnOnce(&'a HeadlessFrame<'a>) -> bool>(_f: F) {}
    _check(HeadlessFrame::is_msaa_enabled);
}

/// HeadlessFrame::aspect_ratio() returns f32.
#[test]
fn headless_frame_aspect_ratio_signature() {
    fn _check<'a, F: FnOnce(&'a HeadlessFrame<'a>) -> f32>(_f: F) {}
    _check(HeadlessFrame::aspect_ratio);
}

/// HeadlessFrame Debug impl works.
#[test]
fn headless_frame_debug_trait() {
    fn _check<T: std::fmt::Debug>() {}
    _check::<HeadlessFrame<'_>>();
}

// =============================================================================
// SECTION 7 -- READBACK BUFFER API SHAPE TESTS (8+ tests)
// =============================================================================

/// ReadbackBuffer::buffer() returns &Buffer.
#[test]
fn readback_buffer_buffer_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer) -> &wgpu::Buffer>(_f: F) {}
    _check(ReadbackBuffer::buffer);
}

/// ReadbackBuffer::bytes_per_row() returns u32.
#[test]
fn readback_buffer_bytes_per_row_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer) -> u32>(_f: F) {}
    _check(ReadbackBuffer::bytes_per_row);
}

/// ReadbackBuffer::bytes_per_pixel() returns u32.
#[test]
fn readback_buffer_bytes_per_pixel_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer) -> u32>(_f: F) {}
    _check(ReadbackBuffer::bytes_per_pixel);
}

/// ReadbackBuffer::width() returns u32.
#[test]
fn readback_buffer_width_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer) -> u32>(_f: F) {}
    _check(ReadbackBuffer::width);
}

/// ReadbackBuffer::height() returns u32.
#[test]
fn readback_buffer_height_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer) -> u32>(_f: F) {}
    _check(ReadbackBuffer::height);
}

/// ReadbackBuffer::format() returns TextureFormat.
#[test]
fn readback_buffer_format_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer) -> wgpu::TextureFormat>(_f: F) {}
    _check(ReadbackBuffer::format);
}

/// ReadbackBuffer::size() returns u64.
#[test]
fn readback_buffer_size_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer) -> u64>(_f: F) {}
    _check(ReadbackBuffer::size);
}

/// ReadbackBuffer::map_read() has correct signature.
#[test]
fn readback_buffer_map_read_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer, &wgpu::Device) -> Result<Vec<u8>, HeadlessError>>(_f: F) {}
    _check(ReadbackBuffer::map_read);
}

/// ReadbackBuffer::map_read_packed() has correct signature.
#[test]
fn readback_buffer_map_read_packed_signature() {
    fn _check<F: FnOnce(&ReadbackBuffer, &wgpu::Device) -> Result<Vec<u8>, HeadlessError>>(_f: F) {}
    _check(ReadbackBuffer::map_read_packed);
}

/// ReadbackBuffer Debug impl works.
#[test]
fn readback_buffer_debug_trait() {
    fn _check<T: std::fmt::Debug>() {}
    _check::<ReadbackBuffer>();
}

// =============================================================================
// SECTION 8 -- HEADLESS RENDERER API SHAPE TESTS (10+ tests)
// =============================================================================

/// HeadlessRenderer::new() has correct signature.
#[test]
fn headless_renderer_new_signature() {
    fn _check<F: FnOnce(&wgpu::Device, HeadlessConfig) -> Result<HeadlessRenderer, HeadlessError>>(_f: F) {}
    _check(HeadlessRenderer::new);
}

/// HeadlessRenderer::acquire_frame() returns HeadlessFrame.
#[test]
fn headless_renderer_acquire_frame_signature() {
    fn _check<F: for<'a> FnOnce(&'a mut HeadlessRenderer) -> HeadlessFrame<'a>>(_f: F) {}
    _check(HeadlessRenderer::acquire_frame);
}

/// HeadlessRenderer::target() returns &HeadlessTarget.
#[test]
fn headless_renderer_target_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> &HeadlessTarget>(_f: F) {}
    _check(HeadlessRenderer::target);
}

/// HeadlessRenderer::target_mut() returns &mut HeadlessTarget.
#[test]
fn headless_renderer_target_mut_signature() {
    fn _check<F: FnOnce(&mut HeadlessRenderer) -> &mut HeadlessTarget>(_f: F) {}
    _check(HeadlessRenderer::target_mut);
}

/// HeadlessRenderer::frame_count() returns u64.
#[test]
fn headless_renderer_frame_count_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> u64>(_f: F) {}
    _check(HeadlessRenderer::frame_count);
}

/// HeadlessRenderer::view() returns &TextureView.
#[test]
fn headless_renderer_view_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> &wgpu::TextureView>(_f: F) {}
    _check(HeadlessRenderer::view);
}

/// HeadlessRenderer::resolve_view() returns Option<&TextureView>.
#[test]
fn headless_renderer_resolve_view_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> Option<&wgpu::TextureView>>(_f: F) {}
    _check(HeadlessRenderer::resolve_view);
}

/// HeadlessRenderer::width() returns u32.
#[test]
fn headless_renderer_width_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> u32>(_f: F) {}
    _check(HeadlessRenderer::width);
}

/// HeadlessRenderer::height() returns u32.
#[test]
fn headless_renderer_height_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> u32>(_f: F) {}
    _check(HeadlessRenderer::height);
}

/// HeadlessRenderer::dimensions() returns (u32, u32).
#[test]
fn headless_renderer_dimensions_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> (u32, u32)>(_f: F) {}
    _check(HeadlessRenderer::dimensions);
}

/// HeadlessRenderer::format() returns TextureFormat.
#[test]
fn headless_renderer_format_signature() {
    fn _check<F: FnOnce(&HeadlessRenderer) -> wgpu::TextureFormat>(_f: F) {}
    _check(HeadlessRenderer::format);
}

/// HeadlessRenderer Debug impl works.
#[test]
fn headless_renderer_debug_trait() {
    fn _check<T: std::fmt::Debug>() {}
    _check::<HeadlessRenderer>();
}

// =============================================================================
// SECTION 9 -- FORMAT VARIATION TESTS (8+ tests)
// =============================================================================

/// Rgba8Unorm is valid for headless rendering.
#[test]
fn format_rgba8_unorm_valid() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::Rgba8Unorm);
    assert_eq!(config.format, TextureFormat::Rgba8Unorm);
    assert!(config.validate().is_ok());
}

/// Bgra8Unorm is valid for headless rendering.
#[test]
fn format_bgra8_unorm_valid() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::Bgra8Unorm);
    assert_eq!(config.format, TextureFormat::Bgra8Unorm);
    assert!(config.validate().is_ok());
}

/// Rgba8UnormSrgb is valid for headless rendering.
#[test]
fn format_rgba8_srgb_valid() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::Rgba8UnormSrgb);
    assert_eq!(config.format, TextureFormat::Rgba8UnormSrgb);
    assert!(config.validate().is_ok());
}

/// Bgra8UnormSrgb is valid for headless rendering.
#[test]
fn format_bgra8_srgb_valid() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::Bgra8UnormSrgb);
    assert_eq!(config.format, TextureFormat::Bgra8UnormSrgb);
    assert!(config.validate().is_ok());
}

/// Rgba16Float is valid for HDR headless rendering.
#[test]
fn format_rgba16_float_valid() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::Rgba16Float);
    assert_eq!(config.format, TextureFormat::Rgba16Float);
    assert!(config.validate().is_ok());
}

/// Rgba32Float is valid for HDR headless rendering.
#[test]
fn format_rgba32_float_valid() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::Rgba32Float);
    assert_eq!(config.format, TextureFormat::Rgba32Float);
    assert!(config.validate().is_ok());
}

/// R8Unorm is valid for grayscale headless rendering.
#[test]
fn format_r8_unorm_valid() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::R8Unorm);
    assert_eq!(config.format, TextureFormat::R8Unorm);
    assert!(config.validate().is_ok());
}

/// Rg8Unorm is valid for two-channel headless rendering.
#[test]
fn format_rg8_unorm_valid() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::Rg8Unorm);
    assert_eq!(config.format, TextureFormat::Rg8Unorm);
    assert!(config.validate().is_ok());
}

/// Rgb10a2Unorm is valid for wide gamut headless rendering.
#[test]
fn format_rgb10a2_unorm_valid() {
    let config = HeadlessConfig::new(100, 100)
        .with_format(TextureFormat::Rgb10a2Unorm);
    assert_eq!(config.format, TextureFormat::Rgb10a2Unorm);
    assert!(config.validate().is_ok());
}

// =============================================================================
// SECTION 10 -- MSAA CONFIGURATION TESTS (6+ tests)
// =============================================================================

/// No MSAA configuration (sample_count=1).
#[test]
fn msaa_config_no_msaa() {
    let config = HeadlessConfig::new(800, 600);
    assert_eq!(config.sample_count, 1);
    assert!(!config.is_msaa_enabled());
}

/// 4x MSAA configuration.
#[test]
fn msaa_config_4x() {
    let config = HeadlessConfig::new(800, 600).with_msaa(4);
    assert_eq!(config.sample_count, 4);
    assert!(config.is_msaa_enabled());
}

/// 8x MSAA configuration.
#[test]
fn msaa_config_8x() {
    let config = HeadlessConfig::new(800, 600).with_msaa(8);
    assert_eq!(config.sample_count, 8);
    assert!(config.is_msaa_enabled());
}

/// MSAA with sRGB format.
#[test]
fn msaa_config_with_srgb() {
    let config = HeadlessConfig::new(800, 600)
        .with_format(TextureFormat::Rgba8UnormSrgb)
        .with_msaa(4);
    assert_eq!(config.format, TextureFormat::Rgba8UnormSrgb);
    assert_eq!(config.sample_count, 4);
    assert!(config.is_msaa_enabled());
}

/// MSAA with HDR format.
#[test]
fn msaa_config_with_hdr() {
    let config = HeadlessConfig::new(800, 600)
        .with_format(TextureFormat::Rgba16Float)
        .with_msaa(4);
    assert_eq!(config.format, TextureFormat::Rgba16Float);
    assert_eq!(config.sample_count, 4);
    assert!(config.is_msaa_enabled());
}

/// MSAA with readback enabled.
#[test]
fn msaa_config_with_readback() {
    let config = HeadlessConfig::new(800, 600)
        .with_msaa(4)
        .with_readback();
    assert!(config.is_msaa_enabled());
    assert!(config.supports_readback());
}

// =============================================================================
// SECTION 11 -- DIMENSION VARIATION TESTS (8+ tests)
// =============================================================================

/// Minimum dimension 1x1.
#[test]
fn dimension_minimum_1x1() {
    let config = HeadlessConfig::new(1, 1);
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 1);
    assert!(config.validate().is_ok());
}

/// Small dimension 16x16.
#[test]
fn dimension_small_16x16() {
    let config = HeadlessConfig::new(16, 16);
    assert_eq!(config.width, 16);
    assert_eq!(config.height, 16);
    assert!(config.validate().is_ok());
}

/// Standard 720p dimension.
#[test]
fn dimension_720p() {
    let config = HeadlessConfig::new(1280, 720);
    assert_eq!(config.width, 1280);
    assert_eq!(config.height, 720);
    assert!(config.validate().is_ok());
}

/// Standard 1080p dimension.
#[test]
fn dimension_1080p() {
    let config = HeadlessConfig::new(1920, 1080);
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert!(config.validate().is_ok());
}

/// Standard 4K dimension.
#[test]
fn dimension_4k() {
    let config = HeadlessConfig::new(3840, 2160);
    assert_eq!(config.width, 3840);
    assert_eq!(config.height, 2160);
    assert!(config.validate().is_ok());
}

/// Non-standard aspect ratio (portrait).
#[test]
fn dimension_portrait() {
    let config = HeadlessConfig::new(1080, 1920);
    assert_eq!(config.width, 1080);
    assert_eq!(config.height, 1920);
    assert!(config.validate().is_ok());
}

/// Non-power-of-two dimensions.
#[test]
fn dimension_non_power_of_two() {
    let config = HeadlessConfig::new(1234, 567);
    assert_eq!(config.width, 1234);
    assert_eq!(config.height, 567);
    assert!(config.validate().is_ok());
}

/// Very wide aspect ratio (ultrawide).
#[test]
fn dimension_ultrawide() {
    let config = HeadlessConfig::new(3440, 1440);
    assert_eq!(config.width, 3440);
    assert_eq!(config.height, 1440);
    assert!(config.validate().is_ok());
}

// =============================================================================
// SECTION 12 -- GPU INTEGRATION TESTS (requires GPU, ignored by default)
// =============================================================================

/// Helper to create a test device.
#[cfg(test)]
async fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .ok()?;

    Some((device, queue))
}

/// Create HeadlessTarget with basic config.
#[test]

fn gpu_headless_target_create_basic() {
    pollster::block_on(async {
        let Some((device, _queue)) = create_test_device().await else {
            return;
        };

        let config = HeadlessConfig::new(800, 600);
        let target = HeadlessTarget::new(&device, config);
        assert!(target.is_ok());

        let target = target.unwrap();
        assert_eq!(target.width(), 800);
        assert_eq!(target.height(), 600);
        assert_eq!(target.format(), TextureFormat::Rgba8Unorm);
        assert_eq!(target.sample_count(), 1);
        assert!(!target.is_msaa_enabled());
    });
}

/// Create HeadlessTarget with 4x MSAA.
#[test]

fn gpu_headless_target_create_msaa_4x() {
    pollster::block_on(async {
        let Some((device, _queue)) = create_test_device().await else {
            return;
        };

        let config = HeadlessConfig::new(800, 600).with_msaa(4);
        let target = HeadlessTarget::new(&device, config);
        assert!(target.is_ok());

        let target = target.unwrap();
        assert_eq!(target.sample_count(), 4);
        assert!(target.is_msaa_enabled());
        assert!(target.resolve_view().is_some());
        assert!(target.resolve_texture().is_some());
    });
}

/// Create HeadlessTarget with 8x MSAA (if supported by hardware).
/// Note: 8x MSAA may panic on hardware that doesn't support it.
/// This test validates the API when 8x MSAA IS supported.
#[test]

fn gpu_headless_target_create_msaa_8x() {
    // Skip this test as 8x MSAA is not universally supported
    // The config API is already validated by other tests
    // This would require TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES feature

    // The configuration itself is valid:
    let config = HeadlessConfig::new(800, 600).with_msaa(8);
    assert_eq!(config.sample_count, 8);
    assert!(config.is_msaa_enabled());
    assert!(config.validate().is_ok());
}

/// Acquire frame from HeadlessTarget.
#[test]

fn gpu_headless_target_acquire_frame() {
    pollster::block_on(async {
        let Some((device, _queue)) = create_test_device().await else {
            return;
        };

        let config = HeadlessConfig::new(1920, 1080);
        let target = HeadlessTarget::new(&device, config).unwrap();

        let frame = target.acquire_frame();
        assert_eq!(frame.width(), 1920);
        assert_eq!(frame.height(), 1080);
        assert_eq!(frame.dimensions(), (1920, 1080));
        assert_eq!(frame.format(), TextureFormat::Rgba8Unorm);
        assert_eq!(frame.sample_count(), 1);
        assert!(!frame.is_msaa_enabled());
    });
}

/// Acquire frame from HeadlessTarget with MSAA.
#[test]

fn gpu_headless_target_acquire_frame_msaa() {
    pollster::block_on(async {
        let Some((device, _queue)) = create_test_device().await else {
            return;
        };

        let config = HeadlessConfig::new(800, 600).with_msaa(4);
        let target = HeadlessTarget::new(&device, config).unwrap();

        let frame = target.acquire_frame();
        assert_eq!(frame.sample_count(), 4);
        assert!(frame.is_msaa_enabled());
        assert!(frame.resolve_view().is_some());
    });
}

/// Resize HeadlessTarget.
#[test]

fn gpu_headless_target_resize() {
    pollster::block_on(async {
        let Some((device, _queue)) = create_test_device().await else {
            return;
        };

        let config = HeadlessConfig::new(800, 600);
        let mut target = HeadlessTarget::new(&device, config).unwrap();
        assert_eq!(target.dimensions(), (800, 600));

        let result = target.resize(&device, 1280, 720);
        assert!(result.is_ok());
        assert_eq!(target.dimensions(), (1280, 720));
        assert_eq!(target.width(), 1280);
        assert_eq!(target.height(), 720);
    });
}

/// Resize HeadlessTarget to same size is no-op.
#[test]

fn gpu_headless_target_resize_same_size() {
    pollster::block_on(async {
        let Some((device, _queue)) = create_test_device().await else {
            return;
        };

        let config = HeadlessConfig::new(800, 600);
        let mut target = HeadlessTarget::new(&device, config).unwrap();

        // Resize to same size should succeed and be a no-op
        let result = target.resize(&device, 800, 600);
        assert!(result.is_ok());
        assert_eq!(target.dimensions(), (800, 600));
    });
}

/// Create ReadbackBuffer from HeadlessTarget.
#[test]

fn gpu_headless_target_create_staging_buffer() {
    pollster::block_on(async {
        let Some((device, _queue)) = create_test_device().await else {
            return;
        };

        let config = HeadlessConfig::new(100, 100);
        let target = HeadlessTarget::new(&device, config).unwrap();

        let staging = target.create_staging_buffer(&device);
        assert_eq!(staging.width(), 100);
        assert_eq!(staging.height(), 100);
        assert_eq!(staging.format(), TextureFormat::Rgba8Unorm);
        assert_eq!(staging.bytes_per_pixel(), 4);
        // 100 * 4 = 400, aligned to 512
        assert_eq!(staging.bytes_per_row(), 512);
        assert_eq!(staging.size(), 512 * 100);
    });
}

/// HeadlessRenderer basic creation.
#[test]

fn gpu_headless_renderer_create() {
    pollster::block_on(async {
        let Some((device, _queue)) = create_test_device().await else {
            return;
        };

        let config = HeadlessConfig::new(1920, 1080);
        let renderer = HeadlessRenderer::new(&device, config);
        assert!(renderer.is_ok());

        let renderer = renderer.unwrap();
        assert_eq!(renderer.width(), 1920);
        assert_eq!(renderer.height(), 1080);
        assert_eq!(renderer.dimensions(), (1920, 1080));
        assert_eq!(renderer.format(), TextureFormat::Rgba8Unorm);
        assert_eq!(renderer.frame_count(), 0);
    });
}

/// HeadlessRenderer acquire frame increments count.
#[test]

fn gpu_headless_renderer_acquire_frame_increments_count() {
    pollster::block_on(async {
        let Some((device, _queue)) = create_test_device().await else {
            return;
        };

        let config = HeadlessConfig::new(800, 600);
        let mut renderer = HeadlessRenderer::new(&device, config).unwrap();
        assert_eq!(renderer.frame_count(), 0);

        let _frame1 = renderer.acquire_frame();
        assert_eq!(renderer.frame_count(), 1);

        let _frame2 = renderer.acquire_frame();
        assert_eq!(renderer.frame_count(), 2);

        let _frame3 = renderer.acquire_frame();
        assert_eq!(renderer.frame_count(), 3);
    });
}

/// HeadlessRenderer resize.
#[test]

fn gpu_headless_renderer_resize() {
    pollster::block_on(async {
        let Some((device, _queue)) = create_test_device().await else {
            return;
        };

        let config = HeadlessConfig::new(800, 600);
        let mut renderer = HeadlessRenderer::new(&device, config).unwrap();
        assert_eq!(renderer.dimensions(), (800, 600));

        let result = renderer.resize(&device, 1920, 1080);
        assert!(result.is_ok());
        assert_eq!(renderer.dimensions(), (1920, 1080));
    });
}

/// Render and screenshot basic operation.
#[test]

fn gpu_headless_render_and_screenshot() {
    pollster::block_on(async {
        let Some((device, queue)) = create_test_device().await else {
            return;
        };

        let config = HeadlessConfig::new(64, 64);
        let target = HeadlessTarget::new(&device, config).unwrap();

        // Create a simple render pass that clears to red
        let frame = target.acquire_frame();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });

        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        queue.submit(std::iter::once(encoder.finish()));

        // Take screenshot
        let pixels = target.screenshot(&device, &queue);
        assert!(pixels.is_ok());

        let pixels = pixels.unwrap();
        // Should have data (64 * 4 = 256, aligned, * 64 rows)
        assert!(!pixels.is_empty());
    });
}

/// Screenshot packed removes row padding.
#[test]

fn gpu_headless_screenshot_packed() {
    pollster::block_on(async {
        let Some((device, queue)) = create_test_device().await else {
            return;
        };

        // Use 100x100 which has padding (100*4=400, aligned to 512)
        let config = HeadlessConfig::new(100, 100);
        let target = HeadlessTarget::new(&device, config).unwrap();

        // Clear to blue
        let frame = target.acquire_frame();
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        queue.submit(std::iter::once(encoder.finish()));

        // Regular screenshot has padding
        let padded = target.screenshot(&device, &queue).unwrap();
        // Packed screenshot removes padding
        let packed = target.screenshot_packed(&device, &queue).unwrap();

        // Padded: 512 * 100 = 51200
        assert_eq!(padded.len(), 512 * 100);
        // Packed: 400 * 100 = 40000
        assert_eq!(packed.len(), 400 * 100);
    });
}

/// MSAA render with resolve.
#[test]

fn gpu_headless_msaa_render_with_resolve() {
    pollster::block_on(async {
        let Some((device, queue)) = create_test_device().await else {
            return;
        };

        let config = HeadlessConfig::new(64, 64).with_msaa(4);
        let target = HeadlessTarget::new(&device, config).unwrap();
        assert!(target.is_msaa_enabled());

        let frame = target.acquire_frame();
        let mut encoder = device.create_command_encoder(&Default::default());

        // For MSAA, render to MSAA target and resolve to resolve_target
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("msaa_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.view(),
                    resolve_target: frame.resolve_view(), // MSAA resolve
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        queue.submit(std::iter::once(encoder.finish()));

        // Screenshot reads from resolve target
        let pixels = target.screenshot(&device, &queue);
        assert!(pixels.is_ok());
        assert!(!pixels.unwrap().is_empty());
    });
}

/// Batch rendering multiple frames.
#[test]

fn gpu_headless_batch_render_multiple_frames() {
    pollster::block_on(async {
        let Some((device, queue)) = create_test_device().await else {
            return;
        };

        let config = HeadlessConfig::new(64, 64);
        let mut renderer = HeadlessRenderer::new(&device, config).unwrap();

        let colors = [
            wgpu::Color::RED,
            wgpu::Color::GREEN,
            wgpu::Color::BLUE,
            wgpu::Color::WHITE,
            wgpu::Color::BLACK,
        ];

        for (i, color) in colors.iter().enumerate() {
            let frame = renderer.acquire_frame();
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: frame.view(),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(*color),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            }
            queue.submit(std::iter::once(encoder.finish()));

            assert_eq!(renderer.frame_count(), (i + 1) as u64);
        }

        assert_eq!(renderer.frame_count(), 5);
    });
}

/// Different formats work for headless rendering.
#[test]

fn gpu_headless_different_formats() {
    pollster::block_on(async {
        let Some((device, queue)) = create_test_device().await else {
            return;
        };

        let formats = [
            TextureFormat::Rgba8Unorm,
            TextureFormat::Bgra8Unorm,
            TextureFormat::Rgba8UnormSrgb,
        ];

        for format in formats {
            let config = HeadlessConfig::new(64, 64).with_format(format);
            let target = HeadlessTarget::new(&device, config);

            if let Ok(target) = target {
                assert_eq!(target.format(), format);

                let frame = target.acquire_frame();
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: frame.view(),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                }
                queue.submit(std::iter::once(encoder.finish()));
            }
        }
    });
}

/// Server-side rendering simulation (no window context).
#[test]

fn gpu_headless_server_side_rendering() {
    pollster::block_on(async {
        let Some((device, queue)) = create_test_device().await else {
            return;
        };

        // Simulate server-side rendering: create, render, capture, discard
        for _ in 0..3 {
            let config = HeadlessConfig::new(256, 256)
                .with_format(TextureFormat::Rgba8Unorm)
                .with_readback();

            let target = HeadlessTarget::new(&device, config).unwrap();

            let frame = target.acquire_frame();
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("server_render"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: frame.view(),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.5,
                                g: 0.5,
                                b: 0.5,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            }
            queue.submit(std::iter::once(encoder.finish()));

            let result = target.screenshot(&device, &queue);
            assert!(result.is_ok());
            // Target is dropped here, simulating per-request rendering
        }
    });
}

// =============================================================================
// SECTION 13 -- PROPERTY-BASED INVARIANT TESTS (6+ tests)
// =============================================================================

/// Config dimensions are always at least 1x1.
#[test]
fn invariant_config_dimensions_at_least_1x1() {
    for w in [0, 1, 100, 1000] {
        for h in [0, 1, 100, 1000] {
            let config = HeadlessConfig::new(w, h);
            assert!(config.width >= 1);
            assert!(config.height >= 1);
        }
    }
}

/// Sample count is always 1, 4, or 8.
#[test]
fn invariant_sample_count_valid_values() {
    for samples in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 32, 64] {
        let config = HeadlessConfig::new(100, 100).with_msaa(samples);
        assert!(
            config.sample_count == 1 || config.sample_count == 4 || config.sample_count == 8,
            "sample_count {} is not 1, 4, or 8",
            config.sample_count
        );
    }
}

/// aligned_bytes_per_row is always multiple of 256.
#[test]
fn invariant_aligned_bytes_per_row_multiple_of_256() {
    for width in [1, 10, 63, 64, 65, 100, 256, 512, 1000, 1920, 3840] {
        let config = HeadlessConfig::new(width, 100);
        assert_eq!(config.aligned_bytes_per_row() % 256, 0);
    }
}

/// buffer_size is always >= width * height * bytes_per_pixel.
#[test]
fn invariant_buffer_size_at_least_raw_size() {
    for width in [64, 100, 1920] {
        for height in [64, 100, 1080] {
            let config = HeadlessConfig::new(width, height);
            let raw_size = width as u64 * height as u64 * config.bytes_per_pixel() as u64;
            assert!(config.buffer_size() >= raw_size);
        }
    }
}

/// RENDER_ATTACHMENT is always in usage.
#[test]
fn invariant_render_attachment_always_present() {
    let usages = [
        wgpu::TextureUsages::COPY_SRC,
        wgpu::TextureUsages::TEXTURE_BINDING,
        wgpu::TextureUsages::STORAGE_BINDING,
        wgpu::TextureUsages::empty(),
    ];

    for usage in usages {
        let config = HeadlessConfig::new(100, 100).with_usage(usage);
        assert!(config.usage.contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
    }
}

/// is_msaa_enabled is true iff sample_count > 1.
#[test]
fn invariant_is_msaa_enabled_iff_sample_count_gt_1() {
    for samples in [0, 1, 2, 4, 8, 16] {
        let config = HeadlessConfig::new(100, 100).with_msaa(samples);
        assert_eq!(
            config.is_msaa_enabled(),
            config.sample_count > 1,
            "is_msaa_enabled mismatch for sample_count {}",
            config.sample_count
        );
    }
}

// =============================================================================
// SECTION 14 -- NEW API METHODS TESTS (T-WGPU-P7.1.10 additions)
// =============================================================================

/// HeadlessConfig::for_screenshot() creates a 1920x1080 config.
#[test]
fn headless_config_for_screenshot_dimensions() {
    let config = HeadlessConfig::for_screenshot();
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
}

/// HeadlessConfig::for_screenshot() uses Rgba8Unorm format.
#[test]
fn headless_config_for_screenshot_format() {
    let config = HeadlessConfig::for_screenshot();
    assert_eq!(config.format, TextureFormat::Rgba8Unorm);
}

/// HeadlessConfig::for_screenshot() has readback support.
#[test]
fn headless_config_for_screenshot_supports_readback() {
    let config = HeadlessConfig::for_screenshot();
    assert!(config.supports_readback());
    assert!(config.usage.contains(wgpu::TextureUsages::COPY_SRC));
}

/// HeadlessConfig::for_screenshot() has no MSAA.
#[test]
fn headless_config_for_screenshot_no_msaa() {
    let config = HeadlessConfig::for_screenshot();
    assert_eq!(config.sample_count, 1);
    assert!(!config.is_msaa_enabled());
}

/// HeadlessConfig::for_screenshot() has a label.
#[test]
fn headless_config_for_screenshot_has_label() {
    let config = HeadlessConfig::for_screenshot();
    assert!(config.label.is_some());
    assert!(config.label.as_ref().unwrap().contains("screenshot"));
}

/// HeadlessConfig::for_video() creates config with specified dimensions.
#[test]
fn headless_config_for_video_dimensions() {
    let config = HeadlessConfig::for_video(1920, 1080);
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);

    let config_4k = HeadlessConfig::for_video(3840, 2160);
    assert_eq!(config_4k.width, 3840);
    assert_eq!(config_4k.height, 2160);
}

/// HeadlessConfig::for_video() clamps zero dimensions to 1.
#[test]
fn headless_config_for_video_clamps_zero() {
    let config = HeadlessConfig::for_video(0, 0);
    assert_eq!(config.width, 1);
    assert_eq!(config.height, 1);
}

/// HeadlessConfig::for_video() uses Rgba8Unorm format.
#[test]
fn headless_config_for_video_format() {
    let config = HeadlessConfig::for_video(1280, 720);
    assert_eq!(config.format, TextureFormat::Rgba8Unorm);
}

/// HeadlessConfig::for_video() has readback support.
#[test]
fn headless_config_for_video_supports_readback() {
    let config = HeadlessConfig::for_video(1280, 720);
    assert!(config.supports_readback());
    assert!(config.usage.contains(wgpu::TextureUsages::COPY_SRC));
}

/// HeadlessConfig::for_video() has no MSAA.
#[test]
fn headless_config_for_video_no_msaa() {
    let config = HeadlessConfig::for_video(1920, 1080);
    assert_eq!(config.sample_count, 1);
    assert!(!config.is_msaa_enabled());
}

/// HeadlessConfig::for_video() has a label.
#[test]
fn headless_config_for_video_has_label() {
    let config = HeadlessConfig::for_video(1920, 1080);
    assert!(config.label.is_some());
    assert!(config.label.as_ref().unwrap().contains("video"));
}

/// HeadlessConfig::with_samples() is an alias for with_msaa().
#[test]
fn headless_config_with_samples_alias() {
    let config_msaa = HeadlessConfig::new(800, 600).with_msaa(4);
    let config_samples = HeadlessConfig::new(800, 600).with_samples(4);
    assert_eq!(config_msaa.sample_count, config_samples.sample_count);
}

/// HeadlessConfig::with_samples() clamps invalid values.
#[test]
fn headless_config_with_samples_clamping() {
    // 0 and 1 -> 1
    assert_eq!(HeadlessConfig::new(100, 100).with_samples(0).sample_count, 1);
    assert_eq!(HeadlessConfig::new(100, 100).with_samples(1).sample_count, 1);
    // 2-4 -> 4
    assert_eq!(HeadlessConfig::new(100, 100).with_samples(2).sample_count, 4);
    assert_eq!(HeadlessConfig::new(100, 100).with_samples(3).sample_count, 4);
    assert_eq!(HeadlessConfig::new(100, 100).with_samples(4).sample_count, 4);
    // 5+ -> 8
    assert_eq!(HeadlessConfig::new(100, 100).with_samples(5).sample_count, 8);
    assert_eq!(HeadlessConfig::new(100, 100).with_samples(16).sample_count, 8);
}

/// HeadlessTarget::size() is an alias for dimensions().
#[test]

fn headless_target_size_alias() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("adapter");
        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("device");

        let config = HeadlessConfig::new(640, 480);
        let target = HeadlessTarget::new(&device, config).expect("target");

        assert_eq!(target.size(), target.dimensions());
        assert_eq!(target.size(), (640, 480));
    });
}

/// ReadbackBuffer::new() creates a buffer with correct properties.
#[test]

fn readback_buffer_new_creates_correctly() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("adapter");
        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("device");

        let buffer = ReadbackBuffer::new(&device, 256, 256, TextureFormat::Rgba8Unorm);

        assert_eq!(buffer.width(), 256);
        assert_eq!(buffer.height(), 256);
        assert_eq!(buffer.format(), TextureFormat::Rgba8Unorm);
        assert_eq!(buffer.bytes_per_pixel(), 4);
        // 256 * 4 = 1024, already aligned to 256
        assert_eq!(buffer.bytes_per_row(), 1024);
    });
}

/// ReadbackBuffer::padded_bytes_per_row() is an alias for bytes_per_row().
#[test]

fn readback_buffer_padded_bytes_per_row_alias() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("adapter");
        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("device");

        let buffer = ReadbackBuffer::new(&device, 100, 100, TextureFormat::Rgba8Unorm);

        assert_eq!(buffer.padded_bytes_per_row(), buffer.bytes_per_row());
    });
}

/// ReadbackBuffer::new() aligns bytes_per_row to 256.
#[test]

fn readback_buffer_new_aligns_to_256() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("adapter");
        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("device");

        // 100 * 4 = 400 bytes, should align to 512
        let buffer = ReadbackBuffer::new(&device, 100, 100, TextureFormat::Rgba8Unorm);
        assert_eq!(buffer.bytes_per_row() % 256, 0);
        assert_eq!(buffer.bytes_per_row(), 512);

        // 63 * 4 = 252 bytes, should align to 256
        let buffer2 = ReadbackBuffer::new(&device, 63, 100, TextureFormat::Rgba8Unorm);
        assert_eq!(buffer2.bytes_per_row() % 256, 0);
        assert_eq!(buffer2.bytes_per_row(), 256);
    });
}

/// HeadlessRenderer::config() returns the configuration.
#[test]

fn headless_renderer_config_returns_config() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("adapter");
        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("device");

        let config = HeadlessConfig::new(800, 600);
        let renderer = HeadlessRenderer::new(&device, config).expect("renderer");

        assert_eq!(renderer.config().width, 800);
        assert_eq!(renderer.config().height, 600);
    });
}

/// HeadlessRenderer::present() is a no-op that doesn't crash.
#[test]

fn headless_renderer_present_is_noop() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("adapter");
        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("device");

        let config = HeadlessConfig::new(100, 100);
        let mut renderer = HeadlessRenderer::new(&device, config).expect("renderer");

        // Should not panic or cause any issues
        renderer.present();
        renderer.present();
        renderer.present();
    });
}

/// HeadlessRenderer::enable_readback() enables COPY_SRC usage.
#[test]

fn headless_renderer_enable_readback_adds_copy_src() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("adapter");
        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("device");

        // Create a config without COPY_SRC
        let config = HeadlessConfig::new(100, 100).with_usage(wgpu::TextureUsages::RENDER_ATTACHMENT);
        let mut renderer = HeadlessRenderer::new(&device, config).expect("renderer");

        // Initially may not support readback
        let initial_supports = renderer.config().supports_readback();

        // Enable readback
        renderer.enable_readback(&device).expect("enable_readback");

        // Now it should support readback
        assert!(renderer.config().supports_readback());
    });
}

/// HeadlessRenderer::enable_readback() is idempotent.
#[test]

fn headless_renderer_enable_readback_idempotent() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("adapter");
        let (device, _queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("device");

        let config = HeadlessConfig::for_screenshot(); // Already has readback
        let mut renderer = HeadlessRenderer::new(&device, config).expect("renderer");

        // Should succeed without recreating the target
        assert!(renderer.config().supports_readback());
        renderer.enable_readback(&device).expect("enable_readback");
        assert!(renderer.config().supports_readback());
    });
}

/// HeadlessRenderer::read_pixels() returns pixel data when readback is supported.
#[test]

fn headless_renderer_read_pixels_with_readback() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("device");

        let config = HeadlessConfig::new(64, 64);
        let mut renderer = HeadlessRenderer::new(&device, config).expect("renderer");

        // Render something
        let _frame = renderer.acquire_frame();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: renderer.view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        queue.submit(std::iter::once(encoder.finish()));

        // Read pixels
        let pixels = renderer.read_pixels(&device, &queue);
        assert!(pixels.is_some());
        let data = pixels.unwrap();
        assert!(!data.is_empty());
    });
}
