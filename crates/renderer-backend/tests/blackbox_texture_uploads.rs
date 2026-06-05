// Blackbox contract tests for T-WGPU-P2.3.5 Texture Uploads.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::resources::*` -- no internal fields,
// no private methods, no implementation details.
//
// Public API under test:
//   - Constants: ROW_PITCH_ALIGNMENT, STAGING_THRESHOLD
//   - Alignment: calculate_row_pitch, align_to_256, is_row_pitch_aligned, pad_to_row_pitch, pad_to_row_pitch_3d
//   - Conversion: convert_rgb_to_rgba, convert_bgra_to_rgba, convert_rgba_to_bgra,
//                 convert_gray_to_rgba, convert_gray_alpha_to_rgba, premultiply_alpha
//   - Descriptors: TextureUploadDescriptor, TextureRegion
//   - Errors: TextureUploadError
//   - Uploader: TextureUploader
//   - Helpers: mip_size, bytes_per_pixel_for_format
//
// Coverage:
//   1.  Constants have expected values
//   2.  calculate_row_pitch returns 256-aligned values
//   3.  align_to_256 correctly aligns values
//   4.  is_row_pitch_aligned validates alignment
//   5.  pad_to_row_pitch pads data correctly
//   6.  pad_to_row_pitch_3d handles 3D textures
//   7.  convert_rgb_to_rgba adds alpha channel
//   8.  convert_bgra_to_rgba swaps R and B
//   9.  convert_rgba_to_bgra swaps R and B
//  10.  convert_gray_to_rgba expands grayscale
//  11.  convert_gray_alpha_to_rgba expands gray+alpha
//  12.  premultiply_alpha scales RGB by alpha
//  13.  TextureUploadDescriptor::full creates correct descriptor
//  14.  TextureUploadDescriptor::mip_level calculates mip sizes
//  15.  TextureRegion::full creates correct region
//  16.  TextureRegion::mip calculates mip dimensions
//  17.  TextureUploadError variants exist with correct fields
//  18.  TextureUploadError implements Display
//  19.  TextureUploadError implements std::error::Error
//  20.  TextureUploader::new accepts threshold
//  21.  mip_size calculates correct sizes
//  22.  bytes_per_pixel_for_format returns correct values

use renderer_backend::resources::{
    align_to_256, bytes_per_pixel_for_format, calculate_row_pitch, convert_bgra_to_rgba,
    convert_gray_alpha_to_rgba, convert_gray_to_rgba, convert_rgb_to_rgba, convert_rgba_to_bgra,
    is_row_pitch_aligned, mip_size, pad_to_row_pitch, pad_to_row_pitch_3d, premultiply_alpha,
    TextureRegion, TextureUploadDescriptor, TextureUploader, TextureUploadError,
    ROW_PITCH_ALIGNMENT, STAGING_THRESHOLD,
};
use wgpu::TextureFormat;

// ============================================================================
// SECTION 1 -- API Contract Tests (Constants and Basic Exports)
// ============================================================================

mod api_contract_tests {
    use super::*;

    #[test]
    fn row_pitch_alignment_is_256() {
        assert_eq!(ROW_PITCH_ALIGNMENT, 256);
    }

    #[test]
    fn staging_threshold_is_64kb() {
        assert_eq!(STAGING_THRESHOLD, 65536);
    }

    #[test]
    fn texture_upload_error_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TextureUploadError>();
    }

    #[test]
    fn texture_upload_error_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<TextureUploadError>();
    }

    #[test]
    fn texture_uploader_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TextureUploader>();
    }

    #[test]
    fn texture_uploader_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<TextureUploader>();
    }

    #[test]
    fn texture_region_is_clone() {
        let region = TextureRegion::full(100, 100, 1);
        let _cloned = region.clone();
    }

    #[test]
    fn texture_region_is_copy() {
        let region = TextureRegion::full(100, 100, 1);
        let copied = region;
        assert_eq!(copied.size, (100, 100, 1));
    }

    #[test]
    fn texture_upload_descriptor_is_clone() {
        let desc = TextureUploadDescriptor::full(100, 100, 4);
        let _cloned = desc.clone();
    }

    #[test]
    fn texture_upload_error_implements_debug() {
        let err = TextureUploadError::ZeroSizeRegion;
        let debug_str = format!("{:?}", err);
        assert!(!debug_str.is_empty());
    }
}

// ============================================================================
// SECTION 2 -- Alignment Behavior Tests
// ============================================================================

mod alignment_behavior_tests {
    use super::*;

    // -- calculate_row_pitch --

    #[test]
    fn calculate_row_pitch_width_1_bpp_4() {
        let pitch = calculate_row_pitch(1, 4);
        assert_eq!(pitch, 256);
        assert_eq!(pitch % ROW_PITCH_ALIGNMENT, 0);
    }

    #[test]
    fn calculate_row_pitch_width_64_bpp_4() {
        let pitch = calculate_row_pitch(64, 4);
        assert_eq!(pitch, 256); // 64*4=256, already aligned
        assert_eq!(pitch % ROW_PITCH_ALIGNMENT, 0);
    }

    #[test]
    fn calculate_row_pitch_width_65_bpp_4() {
        let pitch = calculate_row_pitch(65, 4);
        assert_eq!(pitch, 512); // 65*4=260, rounds up to 512
        assert_eq!(pitch % ROW_PITCH_ALIGNMENT, 0);
    }

    #[test]
    fn calculate_row_pitch_width_255_bpp_4() {
        let pitch = calculate_row_pitch(255, 4);
        // 255*4=1020, next 256-multiple is 1024
        assert_eq!(pitch, 1024);
        assert_eq!(pitch % ROW_PITCH_ALIGNMENT, 0);
    }

    #[test]
    fn calculate_row_pitch_width_256_bpp_4() {
        let pitch = calculate_row_pitch(256, 4);
        assert_eq!(pitch, 1024); // 256*4=1024, already aligned
        assert_eq!(pitch % ROW_PITCH_ALIGNMENT, 0);
    }

    #[test]
    fn calculate_row_pitch_width_257_bpp_4() {
        let pitch = calculate_row_pitch(257, 4);
        // 257*4=1028, next 256-multiple is 1280
        assert_eq!(pitch, 1280);
        assert_eq!(pitch % ROW_PITCH_ALIGNMENT, 0);
    }

    #[test]
    fn calculate_row_pitch_width_512_bpp_4() {
        let pitch = calculate_row_pitch(512, 4);
        assert_eq!(pitch, 2048); // 512*4=2048, already aligned
        assert_eq!(pitch % ROW_PITCH_ALIGNMENT, 0);
    }

    #[test]
    fn calculate_row_pitch_width_1024_bpp_4() {
        let pitch = calculate_row_pitch(1024, 4);
        assert_eq!(pitch, 4096); // 1024*4=4096, already aligned
        assert_eq!(pitch % ROW_PITCH_ALIGNMENT, 0);
    }

    #[test]
    fn calculate_row_pitch_width_1_bpp_1() {
        let pitch = calculate_row_pitch(1, 1);
        assert_eq!(pitch, 256); // 1*1=1, rounds up to 256
    }

    #[test]
    fn calculate_row_pitch_width_256_bpp_1() {
        let pitch = calculate_row_pitch(256, 1);
        assert_eq!(pitch, 256); // 256*1=256, already aligned
    }

    #[test]
    fn calculate_row_pitch_width_100_bpp_8() {
        let pitch = calculate_row_pitch(100, 8);
        // 100*8=800, next 256-multiple is 1024
        assert_eq!(pitch, 1024);
    }

    #[test]
    fn calculate_row_pitch_width_100_bpp_16() {
        let pitch = calculate_row_pitch(100, 16);
        // 100*16=1600, next 256-multiple is 1792
        assert_eq!(pitch, 1792);
    }

    // -- align_to_256 --

    #[test]
    fn align_to_256_zero() {
        assert_eq!(align_to_256(0), 0);
    }

    #[test]
    fn align_to_256_one() {
        assert_eq!(align_to_256(1), 256);
    }

    #[test]
    fn align_to_256_exactly_256() {
        assert_eq!(align_to_256(256), 256);
    }

    #[test]
    fn align_to_256_257() {
        assert_eq!(align_to_256(257), 512);
    }

    #[test]
    fn align_to_256_512() {
        assert_eq!(align_to_256(512), 512);
    }

    #[test]
    fn align_to_256_513() {
        assert_eq!(align_to_256(513), 768);
    }

    #[test]
    fn align_to_256_1000() {
        assert_eq!(align_to_256(1000), 1024);
    }

    #[test]
    fn align_to_256_1024() {
        assert_eq!(align_to_256(1024), 1024);
    }

    #[test]
    fn align_to_256_large_value() {
        assert_eq!(align_to_256(10000), 10240);
    }

    // -- is_row_pitch_aligned --

    #[test]
    fn is_row_pitch_aligned_256() {
        assert!(is_row_pitch_aligned(256));
    }

    #[test]
    fn is_row_pitch_aligned_512() {
        assert!(is_row_pitch_aligned(512));
    }

    #[test]
    fn is_row_pitch_aligned_1024() {
        assert!(is_row_pitch_aligned(1024));
    }

    #[test]
    fn is_row_pitch_aligned_255() {
        assert!(!is_row_pitch_aligned(255));
    }

    #[test]
    fn is_row_pitch_aligned_257() {
        assert!(!is_row_pitch_aligned(257));
    }

    #[test]
    fn is_row_pitch_aligned_100() {
        assert!(!is_row_pitch_aligned(100));
    }

    #[test]
    fn is_row_pitch_aligned_zero() {
        assert!(is_row_pitch_aligned(0));
    }

    // -- pad_to_row_pitch --

    #[test]
    fn pad_to_row_pitch_already_aligned() {
        // 64 pixels * 4 bpp = 256 bytes, already aligned
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let padded = pad_to_row_pitch(&data, 64, 1, 4);
        assert_eq!(padded.len(), 256);
    }

    #[test]
    fn pad_to_row_pitch_needs_padding() {
        // 10 pixels * 4 bpp = 40 bytes per row, needs padding to 256
        let data: Vec<u8> = vec![0u8; 40];
        let padded = pad_to_row_pitch(&data, 10, 1, 4);
        assert_eq!(padded.len(), 256);
    }

    #[test]
    fn pad_to_row_pitch_multiple_rows() {
        // 10 pixels * 4 bpp * 3 rows = 120 bytes
        // Each row needs padding to 256
        let data: Vec<u8> = vec![0u8; 120];
        let padded = pad_to_row_pitch(&data, 10, 3, 4);
        assert_eq!(padded.len(), 256 * 3);
    }

    #[test]
    fn pad_to_row_pitch_preserves_data() {
        // Create recognizable pattern
        let width = 10;
        let height = 2;
        let bpp = 4;
        let row_bytes = width * bpp;
        let mut data = vec![0u8; (row_bytes * height) as usize];
        // First row: 0-39
        for i in 0..row_bytes {
            data[i as usize] = i as u8;
        }
        // Second row: 100-139
        for i in 0..row_bytes {
            data[(row_bytes + i) as usize] = (100 + i) as u8;
        }

        let padded = pad_to_row_pitch(&data, width, height, bpp);

        // Check first row data preserved
        for i in 0..row_bytes as usize {
            assert_eq!(padded[i], i as u8);
        }
        // Check second row data preserved (at offset 256)
        for i in 0..row_bytes as usize {
            assert_eq!(padded[256 + i], (100 + i) as u8);
        }
    }

    #[test]
    fn pad_to_row_pitch_bpp_1() {
        let data: Vec<u8> = vec![0u8; 100]; // 100 pixels * 1 bpp
        let padded = pad_to_row_pitch(&data, 100, 1, 1);
        assert_eq!(padded.len(), 256); // Rounded up
    }

    #[test]
    fn pad_to_row_pitch_bpp_8() {
        let data: Vec<u8> = vec![0u8; 32 * 8]; // 32 pixels * 8 bpp = 256
        let padded = pad_to_row_pitch(&data, 32, 1, 8);
        assert_eq!(padded.len(), 256);
    }

    #[test]
    fn pad_to_row_pitch_bpp_16() {
        let data: Vec<u8> = vec![0u8; 16 * 16]; // 16 pixels * 16 bpp = 256
        let padded = pad_to_row_pitch(&data, 16, 1, 16);
        assert_eq!(padded.len(), 256);
    }

    // -- pad_to_row_pitch_3d --

    #[test]
    fn pad_to_row_pitch_3d_single_slice() {
        let data: Vec<u8> = vec![0u8; 40]; // 10 * 4 * 1 row
        let padded = pad_to_row_pitch_3d(&data, 10, 1, 1, 4);
        assert_eq!(padded.len(), 256);
    }

    #[test]
    fn pad_to_row_pitch_3d_multiple_slices() {
        // 10 pixels * 4 bpp * 2 rows * 3 slices = 240 bytes
        let data: Vec<u8> = vec![0u8; 240];
        let padded = pad_to_row_pitch_3d(&data, 10, 2, 3, 4);
        // 256 bytes per row * 2 rows * 3 slices
        assert_eq!(padded.len(), 256 * 2 * 3);
    }

    #[test]
    fn pad_to_row_pitch_3d_depth_1_same_as_2d() {
        let data: Vec<u8> = vec![0u8; 80]; // 10 * 4 * 2 rows
        let padded_2d = pad_to_row_pitch(&data, 10, 2, 4);
        let padded_3d = pad_to_row_pitch_3d(&data, 10, 2, 1, 4);
        assert_eq!(padded_2d.len(), padded_3d.len());
    }
}

// ============================================================================
// SECTION 3 -- Format Conversion Tests
// ============================================================================

mod format_conversion_tests {
    use super::*;

    // -- convert_rgb_to_rgba --

    #[test]
    fn convert_rgb_to_rgba_single_pixel() {
        let rgb = [255, 128, 64];
        let rgba = convert_rgb_to_rgba(&rgb);
        assert_eq!(rgba, vec![255, 128, 64, 255]);
    }

    #[test]
    fn convert_rgb_to_rgba_multiple_pixels() {
        let rgb = [255, 0, 0, 0, 255, 0, 0, 0, 255];
        let rgba = convert_rgb_to_rgba(&rgb);
        assert_eq!(rgba, vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
    }

    #[test]
    fn convert_rgb_to_rgba_alpha_is_255() {
        let rgb = [100, 150, 200];
        let rgba = convert_rgb_to_rgba(&rgb);
        assert_eq!(rgba[3], 255);
    }

    #[test]
    fn convert_rgb_to_rgba_black() {
        let rgb = [0, 0, 0];
        let rgba = convert_rgb_to_rgba(&rgb);
        assert_eq!(rgba, vec![0, 0, 0, 255]);
    }

    #[test]
    fn convert_rgb_to_rgba_white() {
        let rgb = [255, 255, 255];
        let rgba = convert_rgb_to_rgba(&rgb);
        assert_eq!(rgba, vec![255, 255, 255, 255]);
    }

    #[test]
    fn convert_rgb_to_rgba_empty() {
        let rgb: [u8; 0] = [];
        let rgba = convert_rgb_to_rgba(&rgb);
        assert!(rgba.is_empty());
    }

    // -- convert_bgra_to_rgba --

    #[test]
    fn convert_bgra_to_rgba_single_pixel() {
        let bgra = [64, 128, 255, 200]; // B=64, G=128, R=255, A=200
        let rgba = convert_bgra_to_rgba(&bgra);
        assert_eq!(rgba, vec![255, 128, 64, 200]); // R=255, G=128, B=64, A=200
    }

    #[test]
    fn convert_bgra_to_rgba_preserves_alpha() {
        let bgra = [0, 0, 0, 128];
        let rgba = convert_bgra_to_rgba(&bgra);
        assert_eq!(rgba[3], 128);
    }

    #[test]
    fn convert_bgra_to_rgba_swaps_rb() {
        let bgra = [10, 20, 30, 255]; // B=10, G=20, R=30
        let rgba = convert_bgra_to_rgba(&bgra);
        assert_eq!(rgba[0], 30); // R
        assert_eq!(rgba[1], 20); // G
        assert_eq!(rgba[2], 10); // B
    }

    #[test]
    fn convert_bgra_to_rgba_multiple_pixels() {
        let bgra = [10, 20, 30, 255, 40, 50, 60, 200];
        let rgba = convert_bgra_to_rgba(&bgra);
        assert_eq!(rgba, vec![30, 20, 10, 255, 60, 50, 40, 200]);
    }

    #[test]
    fn convert_bgra_to_rgba_empty() {
        let bgra: [u8; 0] = [];
        let rgba = convert_bgra_to_rgba(&bgra);
        assert!(rgba.is_empty());
    }

    // -- convert_rgba_to_bgra --

    #[test]
    fn convert_rgba_to_bgra_single_pixel() {
        let rgba = [255, 128, 64, 200];
        let bgra = convert_rgba_to_bgra(&rgba);
        assert_eq!(bgra, vec![64, 128, 255, 200]);
    }

    #[test]
    fn convert_rgba_to_bgra_roundtrip() {
        let original = [100, 150, 200, 128];
        let bgra = convert_rgba_to_bgra(&original);
        let back = convert_bgra_to_rgba(&bgra);
        assert_eq!(back, original.to_vec());
    }

    #[test]
    fn convert_rgba_to_bgra_preserves_alpha() {
        let rgba = [0, 0, 0, 77];
        let bgra = convert_rgba_to_bgra(&rgba);
        assert_eq!(bgra[3], 77);
    }

    // -- convert_gray_to_rgba --

    #[test]
    fn convert_gray_to_rgba_single_pixel() {
        let gray = [128];
        let rgba = convert_gray_to_rgba(&gray);
        assert_eq!(rgba, vec![128, 128, 128, 255]);
    }

    #[test]
    fn convert_gray_to_rgba_rgb_equal() {
        let gray = [200];
        let rgba = convert_gray_to_rgba(&gray);
        assert_eq!(rgba[0], rgba[1]);
        assert_eq!(rgba[1], rgba[2]);
        assert_eq!(rgba[0], 200);
    }

    #[test]
    fn convert_gray_to_rgba_alpha_is_255() {
        let gray = [50];
        let rgba = convert_gray_to_rgba(&gray);
        assert_eq!(rgba[3], 255);
    }

    #[test]
    fn convert_gray_to_rgba_black() {
        let gray = [0];
        let rgba = convert_gray_to_rgba(&gray);
        assert_eq!(rgba, vec![0, 0, 0, 255]);
    }

    #[test]
    fn convert_gray_to_rgba_white() {
        let gray = [255];
        let rgba = convert_gray_to_rgba(&gray);
        assert_eq!(rgba, vec![255, 255, 255, 255]);
    }

    #[test]
    fn convert_gray_to_rgba_multiple_pixels() {
        let gray = [0, 128, 255];
        let rgba = convert_gray_to_rgba(&gray);
        assert_eq!(rgba.len(), 12);
        assert_eq!(&rgba[0..4], &[0, 0, 0, 255]);
        assert_eq!(&rgba[4..8], &[128, 128, 128, 255]);
        assert_eq!(&rgba[8..12], &[255, 255, 255, 255]);
    }

    #[test]
    fn convert_gray_to_rgba_empty() {
        let gray: [u8; 0] = [];
        let rgba = convert_gray_to_rgba(&gray);
        assert!(rgba.is_empty());
    }

    // -- convert_gray_alpha_to_rgba --

    #[test]
    fn convert_gray_alpha_to_rgba_single_pixel() {
        let ga = [128, 200]; // Gray=128, Alpha=200
        let rgba = convert_gray_alpha_to_rgba(&ga);
        assert_eq!(rgba, vec![128, 128, 128, 200]);
    }

    #[test]
    fn convert_gray_alpha_to_rgba_preserves_alpha() {
        let ga = [100, 50];
        let rgba = convert_gray_alpha_to_rgba(&ga);
        assert_eq!(rgba[3], 50);
    }

    #[test]
    fn convert_gray_alpha_to_rgba_rgb_equal() {
        let ga = [77, 255];
        let rgba = convert_gray_alpha_to_rgba(&ga);
        assert_eq!(rgba[0], 77);
        assert_eq!(rgba[1], 77);
        assert_eq!(rgba[2], 77);
    }

    #[test]
    fn convert_gray_alpha_to_rgba_multiple_pixels() {
        let ga = [0, 255, 128, 128];
        let rgba = convert_gray_alpha_to_rgba(&ga);
        assert_eq!(rgba.len(), 8);
        assert_eq!(&rgba[0..4], &[0, 0, 0, 255]);
        assert_eq!(&rgba[4..8], &[128, 128, 128, 128]);
    }

    #[test]
    fn convert_gray_alpha_to_rgba_empty() {
        let ga: [u8; 0] = [];
        let rgba = convert_gray_alpha_to_rgba(&ga);
        assert!(rgba.is_empty());
    }

    // -- premultiply_alpha --

    #[test]
    fn premultiply_alpha_full_alpha() {
        let rgba = [100, 150, 200, 255];
        let result = premultiply_alpha(&rgba);
        // With alpha=255, values unchanged
        assert_eq!(result[0], 100);
        assert_eq!(result[1], 150);
        assert_eq!(result[2], 200);
        assert_eq!(result[3], 255);
    }

    #[test]
    fn premultiply_alpha_zero_alpha() {
        let rgba = [100, 150, 200, 0];
        let result = premultiply_alpha(&rgba);
        // With alpha=0, RGB becomes 0
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 0);
        assert_eq!(result[3], 0);
    }

    #[test]
    fn premultiply_alpha_half_alpha() {
        let rgba = [100, 200, 50, 128];
        let result = premultiply_alpha(&rgba);
        // RGB scaled by 128/255 ~ 0.502
        // 100 * 128 / 255 = 50 (rounded)
        // 200 * 128 / 255 = 100 (rounded)
        // 50 * 128 / 255 = 25 (rounded)
        assert!(result[0] >= 49 && result[0] <= 51);
        assert!(result[1] >= 99 && result[1] <= 101);
        assert!(result[2] >= 24 && result[2] <= 26);
        assert_eq!(result[3], 128);
    }

    #[test]
    fn premultiply_alpha_preserves_alpha() {
        let rgba = [255, 255, 255, 77];
        let result = premultiply_alpha(&rgba);
        assert_eq!(result[3], 77);
    }

    #[test]
    fn premultiply_alpha_multiple_pixels() {
        let rgba = [255, 255, 255, 255, 100, 100, 100, 0];
        let result = premultiply_alpha(&rgba);
        assert_eq!(result.len(), 8);
        // First pixel unchanged
        assert_eq!(&result[0..4], &[255, 255, 255, 255]);
        // Second pixel zeroed
        assert_eq!(&result[4..8], &[0, 0, 0, 0]);
    }

    #[test]
    fn premultiply_alpha_empty() {
        let rgba: [u8; 0] = [];
        let result = premultiply_alpha(&rgba);
        assert!(result.is_empty());
    }
}

// ============================================================================
// SECTION 4 -- Region and Descriptor Tests
// ============================================================================

mod region_tests {
    use super::*;

    // -- TextureRegion::full --

    #[test]
    fn texture_region_full_basic() {
        let region = TextureRegion::full(100, 200, 1);
        assert_eq!(region.offset, (0, 0, 0));
        assert_eq!(region.size, (100, 200, 1));
    }

    #[test]
    fn texture_region_full_3d() {
        let region = TextureRegion::full(64, 64, 32);
        assert_eq!(region.offset, (0, 0, 0));
        assert_eq!(region.size, (64, 64, 32));
    }

    #[test]
    fn texture_region_full_depth_1() {
        let region = TextureRegion::full(512, 512, 1);
        assert_eq!(region.size.2, 1);
    }

    // -- TextureRegion::mip --

    #[test]
    fn texture_region_mip_level_0() {
        let region = TextureRegion::mip(1024, 1024, 0);
        assert_eq!(region.size, (1024, 1024, 1));
    }

    #[test]
    fn texture_region_mip_level_1() {
        let region = TextureRegion::mip(1024, 1024, 1);
        assert_eq!(region.size, (512, 512, 1));
    }

    #[test]
    fn texture_region_mip_level_2() {
        let region = TextureRegion::mip(1024, 1024, 2);
        assert_eq!(region.size, (256, 256, 1));
    }

    #[test]
    fn texture_region_mip_level_3() {
        let region = TextureRegion::mip(1024, 1024, 3);
        assert_eq!(region.size, (128, 128, 1));
    }

    #[test]
    fn texture_region_mip_level_10() {
        let region = TextureRegion::mip(1024, 1024, 10);
        // 1024 >> 10 = 1
        assert_eq!(region.size, (1, 1, 1));
    }

    #[test]
    fn texture_region_mip_clamps_to_1() {
        let region = TextureRegion::mip(1024, 1024, 20);
        // Beyond max mip, should clamp to 1x1
        assert!(region.size.0 >= 1);
        assert!(region.size.1 >= 1);
    }

    #[test]
    fn texture_region_mip_non_power_of_2() {
        let region = TextureRegion::mip(100, 50, 1);
        // 100 >> 1 = 50, 50 >> 1 = 25
        assert_eq!(region.size, (50, 25, 1));
    }

    #[test]
    fn texture_region_mip_rectangular() {
        let region = TextureRegion::mip(1024, 512, 2);
        assert_eq!(region.size, (256, 128, 1));
    }

    // -- TextureUploadDescriptor::full --

    #[test]
    fn texture_upload_descriptor_full_basic() {
        let desc = TextureUploadDescriptor::full(100, 100, 4);
        assert_eq!(desc.offset, (0, 0, 0));
        assert_eq!(desc.size, (100, 100, 1));
        assert_eq!(desc.mip_level, 0);
        assert_eq!(desc.bytes_per_pixel, 4);
    }

    #[test]
    fn texture_upload_descriptor_full_large() {
        let desc = TextureUploadDescriptor::full(4096, 4096, 4);
        assert_eq!(desc.size, (4096, 4096, 1));
    }

    #[test]
    fn texture_upload_descriptor_full_bpp_1() {
        let desc = TextureUploadDescriptor::full(256, 256, 1);
        assert_eq!(desc.bytes_per_pixel, 1);
    }

    #[test]
    fn texture_upload_descriptor_full_bpp_16() {
        let desc = TextureUploadDescriptor::full(128, 128, 16);
        assert_eq!(desc.bytes_per_pixel, 16);
    }

    // -- TextureUploadDescriptor::mip_level --

    #[test]
    fn texture_upload_descriptor_mip_level_0() {
        let desc = TextureUploadDescriptor::mip_level(1024, 1024, 0, 4);
        assert_eq!(desc.size, (1024, 1024, 1));
        assert_eq!(desc.mip_level, 0);
    }

    #[test]
    fn texture_upload_descriptor_mip_level_1() {
        let desc = TextureUploadDescriptor::mip_level(1024, 1024, 1, 4);
        assert_eq!(desc.size, (512, 512, 1));
        assert_eq!(desc.mip_level, 1);
    }

    #[test]
    fn texture_upload_descriptor_mip_level_3() {
        let desc = TextureUploadDescriptor::mip_level(1024, 512, 3, 4);
        assert_eq!(desc.size, (128, 64, 1));
        assert_eq!(desc.mip_level, 3);
    }

    // -- mip_size helper --

    #[test]
    fn mip_size_level_0() {
        assert_eq!(mip_size(1024, 0), 1024);
    }

    #[test]
    fn mip_size_level_1() {
        assert_eq!(mip_size(1024, 1), 512);
    }

    #[test]
    fn mip_size_level_2() {
        assert_eq!(mip_size(1024, 2), 256);
    }

    #[test]
    fn mip_size_level_10() {
        assert_eq!(mip_size(1024, 10), 1);
    }

    #[test]
    fn mip_size_clamps_to_1() {
        assert_eq!(mip_size(1024, 20), 1);
    }

    #[test]
    fn mip_size_non_power_of_2() {
        assert_eq!(mip_size(100, 1), 50);
        assert_eq!(mip_size(100, 2), 25);
        assert_eq!(mip_size(100, 3), 12);
    }

    // -- bytes_per_pixel_for_format --

    #[test]
    fn bytes_per_pixel_rgba8_unorm() {
        let bpp = bytes_per_pixel_for_format(TextureFormat::Rgba8Unorm);
        assert_eq!(bpp, Some(4));
    }

    #[test]
    fn bytes_per_pixel_rgba8_srgb() {
        let bpp = bytes_per_pixel_for_format(TextureFormat::Rgba8UnormSrgb);
        assert_eq!(bpp, Some(4));
    }

    #[test]
    fn bytes_per_pixel_bgra8_unorm() {
        let bpp = bytes_per_pixel_for_format(TextureFormat::Bgra8Unorm);
        assert_eq!(bpp, Some(4));
    }

    #[test]
    fn bytes_per_pixel_r8_unorm() {
        let bpp = bytes_per_pixel_for_format(TextureFormat::R8Unorm);
        assert_eq!(bpp, Some(1));
    }

    #[test]
    fn bytes_per_pixel_rg8_unorm() {
        let bpp = bytes_per_pixel_for_format(TextureFormat::Rg8Unorm);
        assert_eq!(bpp, Some(2));
    }

    #[test]
    fn bytes_per_pixel_r16_float() {
        let bpp = bytes_per_pixel_for_format(TextureFormat::R16Float);
        assert_eq!(bpp, Some(2));
    }

    #[test]
    fn bytes_per_pixel_rgba16_float() {
        let bpp = bytes_per_pixel_for_format(TextureFormat::Rgba16Float);
        assert_eq!(bpp, Some(8));
    }

    #[test]
    fn bytes_per_pixel_r32_float() {
        let bpp = bytes_per_pixel_for_format(TextureFormat::R32Float);
        assert_eq!(bpp, Some(4));
    }

    #[test]
    fn bytes_per_pixel_rgba32_float() {
        let bpp = bytes_per_pixel_for_format(TextureFormat::Rgba32Float);
        assert_eq!(bpp, Some(16));
    }

    #[test]
    fn bytes_per_pixel_depth32_float() {
        // Depth formats may not have a bytes_per_pixel value since they're
        // not typically uploaded with user data
        let bpp = bytes_per_pixel_for_format(TextureFormat::Depth32Float);
        // The function may return None for depth-only formats
        let _ = bpp;
    }

    #[test]
    fn bytes_per_pixel_depth24_stencil8() {
        let bpp = bytes_per_pixel_for_format(TextureFormat::Depth24PlusStencil8);
        // This format might return None or a specific value
        // The test verifies the function doesn't panic
        let _ = bpp;
    }
}

// ============================================================================
// SECTION 5 -- Error Tests
// ============================================================================

mod error_tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn error_invalid_region_display() {
        let err = TextureUploadError::InvalidRegion {
            offset: (10, 10, 0),
            size: (100, 100, 1),
            texture_size: (64, 64, 1),
            mip_level: 0,
        };
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert!(msg.contains("region") || msg.contains("bounds") || msg.contains("invalid"));
    }

    #[test]
    fn error_buffer_too_small_display() {
        let err = TextureUploadError::BufferTooSmall {
            provided: 100,
            required: 200,
        };
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert!(msg.contains("100") || msg.contains("200") || msg.contains("small") || msg.contains("buffer"));
    }

    #[test]
    fn error_format_mismatch_display() {
        let err = TextureUploadError::FormatMismatch {
            expected: "RGBA8".to_string(),
            actual: "RGB8".to_string(),
        };
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
    }

    #[test]
    fn error_alignment_error_display() {
        let err = TextureUploadError::AlignmentError {
            row_size: 100,
            alignment: 256,
        };
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert!(msg.contains("alignment") || msg.contains("100") || msg.contains("256"));
    }

    #[test]
    fn error_zero_size_region_display() {
        let err = TextureUploadError::ZeroSizeRegion;
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert!(msg.contains("zero") || msg.contains("size"));
    }

    #[test]
    fn error_invalid_bytes_per_pixel_display() {
        let err = TextureUploadError::InvalidBytesPerPixel { value: 0 };
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
    }

    #[test]
    fn error_mip_level_out_of_range_display() {
        let err = TextureUploadError::MipLevelOutOfRange {
            level: 15,
            max_level: 10,
        };
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert!(msg.contains("mip") || msg.contains("level") || msg.contains("15") || msg.contains("10"));
    }

    #[test]
    fn error_implements_std_error() {
        let err = TextureUploadError::ZeroSizeRegion;
        let _: &dyn Error = &err;
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TextureUploadError>();
    }

    #[test]
    fn error_debug_impl() {
        let err = TextureUploadError::BufferTooSmall {
            provided: 50,
            required: 100,
        };
        let debug = format!("{:?}", err);
        assert!(!debug.is_empty());
    }

    #[test]
    fn error_clone() {
        let err = TextureUploadError::ZeroSizeRegion;
        let cloned = err.clone();
        assert!(matches!(cloned, TextureUploadError::ZeroSizeRegion));
    }

    #[test]
    fn error_partial_eq() {
        let err1 = TextureUploadError::ZeroSizeRegion;
        let err2 = TextureUploadError::ZeroSizeRegion;
        assert_eq!(err1, err2);
    }

    #[test]
    fn error_invalid_region_fields() {
        let err = TextureUploadError::InvalidRegion {
            offset: (5, 10, 0),
            size: (50, 100, 1),
            texture_size: (32, 32, 1),
            mip_level: 2,
        };
        if let TextureUploadError::InvalidRegion {
            offset,
            size,
            texture_size,
            mip_level,
        } = err
        {
            assert_eq!(offset, (5, 10, 0));
            assert_eq!(size, (50, 100, 1));
            assert_eq!(texture_size, (32, 32, 1));
            assert_eq!(mip_level, 2);
        } else {
            panic!("Expected InvalidRegion variant");
        }
    }

    #[test]
    fn error_buffer_too_small_fields() {
        let err = TextureUploadError::BufferTooSmall {
            provided: 123,
            required: 456,
        };
        if let TextureUploadError::BufferTooSmall { provided, required } = err {
            assert_eq!(provided, 123);
            assert_eq!(required, 456);
        } else {
            panic!("Expected BufferTooSmall variant");
        }
    }
}

// ============================================================================
// SECTION 6 -- TextureUploader Unit Tests (No GPU)
// ============================================================================

mod uploader_unit_tests {
    use super::*;

    #[test]
    fn texture_uploader_new_default_threshold() {
        let uploader = TextureUploader::new(STAGING_THRESHOLD);
        // Just verify construction doesn't panic
        let _ = uploader;
    }

    #[test]
    fn texture_uploader_new_custom_threshold() {
        let uploader = TextureUploader::new(1024);
        let _ = uploader;
    }

    #[test]
    fn texture_uploader_new_zero_threshold() {
        let uploader = TextureUploader::new(0);
        let _ = uploader;
    }

    #[test]
    fn texture_uploader_new_large_threshold() {
        let uploader = TextureUploader::new(u64::MAX);
        let _ = uploader;
    }

    #[test]
    fn texture_uploader_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TextureUploader>();
    }

    #[test]
    fn texture_uploader_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<TextureUploader>();
    }
}

// ============================================================================
// SECTION 7 -- Integration Tests (GPU Required)
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    // Helper to create a wgpu instance and device for testing
    async fn setup_gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
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

    #[test]
    
    fn integration_write_texture_small() {
        pollster::block_on(async {
            let Some((device, queue)) = setup_gpu().await else {
                eprintln!("GPU not available, skipping test");
                return;
            };

            let uploader = TextureUploader::new(STAGING_THRESHOLD);

            // Create a small texture
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_texture"),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // Create test data (64x64x4 = 16384 bytes)
            let data: Vec<u8> = (0..16384).map(|i| (i % 256) as u8).collect();
            let desc = TextureUploadDescriptor::full(64, 64, 4);

            let result = uploader.write_texture(&queue, &texture, &data, &desc);
            assert!(result.is_ok());

            let _ = device;
        });
    }

    #[test]
    
    fn integration_write_texture_with_padding() {
        pollster::block_on(async {
            let Some((device, queue)) = setup_gpu().await else {
                eprintln!("GPU not available, skipping test");
                return;
            };

            let uploader = TextureUploader::new(STAGING_THRESHOLD);

            // 10 pixels wide needs padding
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_texture_padded"),
                size: wgpu::Extent3d {
                    width: 10,
                    height: 10,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            let data: Vec<u8> = vec![128u8; 10 * 10 * 4];
            let desc = TextureUploadDescriptor::full(10, 10, 4);

            let result = uploader.write_texture(&queue, &texture, &data, &desc);
            assert!(result.is_ok());

            let _ = device;
        });
    }

    #[test]
    
    fn integration_upload_staged_large() {
        pollster::block_on(async {
            let Some((device, queue)) = setup_gpu().await else {
                eprintln!("GPU not available, skipping test");
                return;
            };

            let uploader = TextureUploader::new(1024); // Low threshold to force staging

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_texture_staged"),
                size: wgpu::Extent3d {
                    width: 256,
                    height: 256,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // Large data that exceeds threshold
            let data: Vec<u8> = vec![200u8; 256 * 256 * 4];
            let desc = TextureUploadDescriptor::full(256, 256, 4);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test_encoder"),
            });
            let result = uploader.upload_staged(&device, &mut encoder, &texture, &data, &desc);
            assert!(result.is_ok());

            queue.submit(std::iter::once(encoder.finish()));
        });
    }

    #[test]
    
    fn integration_upload_auto_selection_small() {
        pollster::block_on(async {
            let Some((device, queue)) = setup_gpu().await else {
                eprintln!("GPU not available, skipping test");
                return;
            };

            let uploader = TextureUploader::new(STAGING_THRESHOLD);

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_texture_auto"),
                size: wgpu::Extent3d {
                    width: 32,
                    height: 32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // Small data below threshold
            let data: Vec<u8> = vec![100u8; 32 * 32 * 4];
            let desc = TextureUploadDescriptor::full(32, 32, 4);

            let result = uploader.upload(&device, &queue, None, &texture, &data, &desc);
            assert!(result.is_ok());
        });
    }

    #[test]
    
    fn integration_upload_auto_selection_large() {
        pollster::block_on(async {
            let Some((device, queue)) = setup_gpu().await else {
                eprintln!("GPU not available, skipping test");
                return;
            };

            let uploader = TextureUploader::new(1024); // Low threshold

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_texture_auto_large"),
                size: wgpu::Extent3d {
                    width: 128,
                    height: 128,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // Data exceeding threshold should use staged upload
            let data: Vec<u8> = vec![150u8; 128 * 128 * 4];
            let desc = TextureUploadDescriptor::full(128, 128, 4);

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test_encoder"),
            });
            let result = uploader.upload(&device, &queue, Some(&mut encoder), &texture, &data, &desc);
            assert!(result.is_ok());

            queue.submit(std::iter::once(encoder.finish()));
        });
    }

    #[test]
    
    fn integration_upload_mip_level() {
        pollster::block_on(async {
            let Some((device, queue)) = setup_gpu().await else {
                eprintln!("GPU not available, skipping test");
                return;
            };

            let uploader = TextureUploader::new(STAGING_THRESHOLD);

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_texture_mips"),
                size: wgpu::Extent3d {
                    width: 256,
                    height: 256,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 5,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // Upload mip level 2 (64x64)
            let mip_data: Vec<u8> = vec![200u8; 64 * 64 * 4];
            let desc = TextureUploadDescriptor::mip_level(256, 256, 2, 4);

            let result = uploader.upload(&device, &queue, None, &texture, &mip_data, &desc);
            assert!(result.is_ok());
        });
    }

    #[test]
    
    fn integration_upload_3d_texture() {
        pollster::block_on(async {
            let Some((device, queue)) = setup_gpu().await else {
                eprintln!("GPU not available, skipping test");
                return;
            };

            let uploader = TextureUploader::new(STAGING_THRESHOLD);

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_texture_3d"),
                size: wgpu::Extent3d {
                    width: 32,
                    height: 32,
                    depth_or_array_layers: 8,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // 3D texture data
            let data: Vec<u8> = vec![100u8; 32 * 32 * 8 * 4];
            let desc = TextureUploadDescriptor {
                offset: (0, 0, 0),
                size: (32, 32, 8),
                mip_level: 0,
                bytes_per_pixel: 4,
                format: Some(TextureFormat::Rgba8Unorm),
            };

            let result = uploader.upload(&device, &queue, None, &texture, &data, &desc);
            assert!(result.is_ok());
        });
    }

    #[test]
    
    fn integration_upload_region() {
        pollster::block_on(async {
            let Some((device, queue)) = setup_gpu().await else {
                eprintln!("GPU not available, skipping test");
                return;
            };

            let uploader = TextureUploader::new(STAGING_THRESHOLD);

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_texture_region"),
                size: wgpu::Extent3d {
                    width: 256,
                    height: 256,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // Upload only a 64x64 region at offset (32, 32)
            let region_data: Vec<u8> = vec![255u8; 64 * 64 * 4];
            let desc = TextureUploadDescriptor {
                offset: (32, 32, 0),
                size: (64, 64, 1),
                mip_level: 0,
                bytes_per_pixel: 4,
                format: None,
            };

            let result = uploader.upload(&device, &queue, None, &texture, &region_data, &desc);
            assert!(result.is_ok());
        });
    }

    #[test]
    
    fn integration_error_buffer_too_small() {
        pollster::block_on(async {
            let Some((device, queue)) = setup_gpu().await else {
                eprintln!("GPU not available, skipping test");
                return;
            };

            let uploader = TextureUploader::new(STAGING_THRESHOLD);

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_texture_error"),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // Provide less data than required
            let data: Vec<u8> = vec![0u8; 100]; // Way too small
            let desc = TextureUploadDescriptor::full(64, 64, 4);

            let result = uploader.upload(&device, &queue, None, &texture, &data, &desc);
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                TextureUploadError::BufferTooSmall { .. }
            ));
        });
    }

    #[test]
    
    fn integration_error_zero_bytes_per_pixel() {
        pollster::block_on(async {
            let Some((device, queue)) = setup_gpu().await else {
                eprintln!("GPU not available, skipping test");
                return;
            };

            let uploader = TextureUploader::new(STAGING_THRESHOLD);

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_texture_zero_bpp"),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            let data: Vec<u8> = vec![0u8; 1000];
            let desc = TextureUploadDescriptor::full(64, 64, 0); // Invalid bpp

            let result = uploader.upload(&device, &queue, None, &texture, &data, &desc);
            assert!(result.is_err());
            assert!(matches!(
                result.unwrap_err(),
                TextureUploadError::InvalidBytesPerPixel { .. }
            ));
        });
    }

    #[test]
    
    fn integration_r8_format() {
        pollster::block_on(async {
            let Some((device, queue)) = setup_gpu().await else {
                eprintln!("GPU not available, skipping test");
                return;
            };

            let uploader = TextureUploader::new(STAGING_THRESHOLD);

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_texture_r8"),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            let data: Vec<u8> = vec![128u8; 64 * 64];
            let desc = TextureUploadDescriptor::full(64, 64, 1);

            let result = uploader.upload(&device, &queue, None, &texture, &data, &desc);
            assert!(result.is_ok());
        });
    }

    #[test]
    
    fn integration_rgba16_float_format() {
        pollster::block_on(async {
            let Some((device, queue)) = setup_gpu().await else {
                eprintln!("GPU not available, skipping test");
                return;
            };

            let uploader = TextureUploader::new(STAGING_THRESHOLD);

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("test_texture_rgba16f"),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // 8 bytes per pixel for RGBA16F
            let data: Vec<u8> = vec![0u8; 64 * 64 * 8];
            let desc = TextureUploadDescriptor::full(64, 64, 8);

            let result = uploader.upload(&device, &queue, None, &texture, &data, &desc);
            assert!(result.is_ok());
        });
    }
}

// ============================================================================
// SECTION 8 -- Edge Case and Stress Tests
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn alignment_edge_case_width_0() {
        // Width 0 should handle gracefully
        let pitch = calculate_row_pitch(0, 4);
        assert_eq!(pitch, 0);
    }

    #[test]
    fn alignment_edge_case_bpp_0() {
        let pitch = calculate_row_pitch(100, 0);
        assert_eq!(pitch, 0);
    }

    #[test]
    fn alignment_edge_case_max_u32() {
        // Large width shouldn't overflow
        let pitch = calculate_row_pitch(16777216, 4); // 64MB row
        assert!(pitch >= 16777216 * 4);
        assert_eq!(pitch % 256, 0);
    }

    #[test]
    fn mip_size_edge_case_width_1() {
        assert_eq!(mip_size(1, 0), 1);
        assert_eq!(mip_size(1, 1), 1);
        assert_eq!(mip_size(1, 10), 1);
    }

    #[test]
    fn mip_size_edge_case_width_2() {
        assert_eq!(mip_size(2, 0), 2);
        assert_eq!(mip_size(2, 1), 1);
        assert_eq!(mip_size(2, 2), 1);
    }

    #[test]
    fn texture_region_mip_edge_case_level_32() {
        let region = TextureRegion::mip(1024, 1024, 32);
        assert_eq!(region.size.0, 1);
        assert_eq!(region.size.1, 1);
    }

    #[test]
    fn texture_region_mip_edge_case_level_max() {
        let region = TextureRegion::mip(1024, 1024, u32::MAX);
        assert_eq!(region.size.0, 1);
        assert_eq!(region.size.1, 1);
    }

    #[test]
    fn convert_rgb_to_rgba_large_data() {
        let rgb: Vec<u8> = vec![128; 3 * 1024 * 1024]; // 1M pixels
        let rgba = convert_rgb_to_rgba(&rgb);
        assert_eq!(rgba.len(), 4 * 1024 * 1024);
    }

    #[test]
    fn pad_to_row_pitch_single_pixel() {
        let data = vec![255u8; 4];
        let padded = pad_to_row_pitch(&data, 1, 1, 4);
        assert_eq!(padded.len(), 256);
        assert_eq!(&padded[0..4], &[255, 255, 255, 255]);
    }

    #[test]
    fn texture_upload_descriptor_unpadded_data_size() {
        let desc = TextureUploadDescriptor::full(100, 100, 4);
        assert_eq!(desc.unpadded_data_size(), 100 * 100 * 4);
    }

    #[test]
    fn texture_upload_descriptor_unpadded_data_size_mip() {
        let desc = TextureUploadDescriptor::mip_level(1024, 1024, 2, 4);
        // Mip 2 = 256x256
        assert_eq!(desc.unpadded_data_size(), 256 * 256 * 4);
    }

    #[test]
    fn premultiply_alpha_single_channel_max() {
        let rgba = [255, 255, 255, 1];
        let result = premultiply_alpha(&rgba);
        // 255 * 1 / 255 = 1
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 1);
        assert_eq!(result[2], 1);
    }

    #[test]
    fn convert_gray_to_rgba_all_values() {
        for gray in 0..=255u8 {
            let result = convert_gray_to_rgba(&[gray]);
            assert_eq!(result[0], gray);
            assert_eq!(result[1], gray);
            assert_eq!(result[2], gray);
            assert_eq!(result[3], 255);
        }
    }
}
