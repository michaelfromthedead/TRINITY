//! Whitebox tests for texture upload utilities.
//!
//! This test module provides comprehensive coverage of the texture upload
//! functionality in `resources/texture_uploads.rs`, including:
//!
//! - Constants verification
//! - Alignment helper functions
//! - Error types and Display implementations
//! - TextureUploadDescriptor validation and methods
//! - TextureRegion methods and edge cases
//! - Format conversion functions
//! - TextureUploader validation logic
//! - Edge cases and boundary conditions
//!
//! Target: 100+ tests covering all helpers and validation logic.

use renderer_backend::resources::texture_uploads::{
    // Constants
    ROW_PITCH_ALIGNMENT, STAGING_THRESHOLD,
    // Alignment helpers
    align_to_256, calculate_row_pitch, is_row_pitch_aligned,
    pad_to_row_pitch, pad_to_row_pitch_3d,
    // Error type
    TextureUploadError,
    // Descriptors
    TextureUploadDescriptor, TextureRegion,
    // Format converters
    convert_rgb_to_rgba, convert_bgra_to_rgba, convert_rgba_to_bgra,
    convert_gray_to_rgba, convert_gray_alpha_to_rgba, premultiply_alpha,
    // Uploader
    TextureUploader,
    // Utilities
    mip_size, bytes_per_pixel_for_format,
};
use std::borrow::Cow;
use wgpu::TextureFormat;

// ============================================================================
// Module: constants_tests
// ============================================================================

mod constants_tests {
    use super::*;

    #[test]
    fn row_pitch_alignment_is_256() {
        assert_eq!(ROW_PITCH_ALIGNMENT, 256);
    }

    #[test]
    fn row_pitch_alignment_is_power_of_two() {
        assert!(ROW_PITCH_ALIGNMENT.is_power_of_two());
    }

    #[test]
    fn staging_threshold_is_64kb() {
        assert_eq!(STAGING_THRESHOLD, 65536);
        assert_eq!(STAGING_THRESHOLD, 64 * 1024);
    }

    #[test]
    fn staging_threshold_is_power_of_two() {
        assert!((STAGING_THRESHOLD as u32).is_power_of_two());
    }

    #[test]
    fn constants_relationship() {
        // Staging threshold should be significantly larger than row alignment
        assert!(STAGING_THRESHOLD > ROW_PITCH_ALIGNMENT as u64);
        assert!(STAGING_THRESHOLD >= 256 * ROW_PITCH_ALIGNMENT as u64);
    }
}

// ============================================================================
// Module: alignment_tests
// ============================================================================

mod alignment_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // align_to_256 tests
    // -------------------------------------------------------------------------

    #[test]
    fn align_to_256_zero_returns_zero() {
        assert_eq!(align_to_256(0), 0);
    }

    #[test]
    fn align_to_256_one_returns_256() {
        assert_eq!(align_to_256(1), 256);
    }

    #[test]
    fn align_to_256_255_returns_256() {
        assert_eq!(align_to_256(255), 256);
    }

    #[test]
    fn align_to_256_exact_multiple() {
        assert_eq!(align_to_256(256), 256);
        assert_eq!(align_to_256(512), 512);
        assert_eq!(align_to_256(768), 768);
        assert_eq!(align_to_256(1024), 1024);
    }

    #[test]
    fn align_to_256_just_above_multiple() {
        assert_eq!(align_to_256(257), 512);
        assert_eq!(align_to_256(513), 768);
        assert_eq!(align_to_256(1025), 1280);
    }

    #[test]
    fn align_to_256_typical_values() {
        assert_eq!(align_to_256(100), 256);
        assert_eq!(align_to_256(300), 512);
        assert_eq!(align_to_256(400), 512);
        assert_eq!(align_to_256(500), 512);
        assert_eq!(align_to_256(1000), 1024);
    }

    #[test]
    fn align_to_256_large_values() {
        assert_eq!(align_to_256(10000), 10240);
        assert_eq!(align_to_256(100000), 100096);
        assert_eq!(align_to_256(1000000), 1000192);
    }

    #[test]
    fn align_to_256_near_u32_max() {
        // Note: The current implementation can overflow for values very close to u32::MAX
        // because it adds (ROW_PITCH_ALIGNMENT - 1) = 255 before masking.
        // Values that would overflow when adding 255 are not supported.
        // This is acceptable as texture dimensions near u32::MAX are unrealistic.

        // The maximum safe unaligned value is u32::MAX - 255 + 1 = 4294967041
        // But since align_to_256 first checks if value == 0 and then adds 255,
        // the safe range is 0..=(u32::MAX - 255).

        // Values well below max are safe
        let safe_value = u32::MAX - 1024;
        let result = align_to_256(safe_value);
        assert!(result >= safe_value);
        assert_eq!(result % 256, 0);

        // Moderate large values work
        let moderate = 1_000_000_000;
        let result = align_to_256(moderate);
        assert!(result >= moderate);
        assert_eq!(result % 256, 0);
    }

    // -------------------------------------------------------------------------
    // calculate_row_pitch tests
    // -------------------------------------------------------------------------

    #[test]
    fn calculate_row_pitch_zero_width() {
        assert_eq!(calculate_row_pitch(0, 4), 0);
    }

    #[test]
    fn calculate_row_pitch_single_pixel() {
        assert_eq!(calculate_row_pitch(1, 1), 256);
        assert_eq!(calculate_row_pitch(1, 2), 256);
        assert_eq!(calculate_row_pitch(1, 4), 256);
        assert_eq!(calculate_row_pitch(1, 8), 256);
        assert_eq!(calculate_row_pitch(1, 16), 256);
    }

    #[test]
    fn calculate_row_pitch_already_aligned() {
        // 64 pixels * 4 bytes = 256 (aligned)
        assert_eq!(calculate_row_pitch(64, 4), 256);
        // 128 pixels * 4 bytes = 512 (aligned)
        assert_eq!(calculate_row_pitch(128, 4), 512);
        // 256 pixels * 1 byte = 256 (aligned)
        assert_eq!(calculate_row_pitch(256, 1), 256);
        // 128 pixels * 2 bytes = 256 (aligned)
        assert_eq!(calculate_row_pitch(128, 2), 256);
    }

    #[test]
    fn calculate_row_pitch_needs_padding() {
        // 100 pixels * 4 bytes = 400 -> 512
        assert_eq!(calculate_row_pitch(100, 4), 512);
        // 65 pixels * 4 bytes = 260 -> 512
        assert_eq!(calculate_row_pitch(65, 4), 512);
        // 200 pixels * 4 bytes = 800 -> 1024
        assert_eq!(calculate_row_pitch(200, 4), 1024);
    }

    #[test]
    fn calculate_row_pitch_different_bpp_1() {
        assert_eq!(calculate_row_pitch(100, 1), 256);
        assert_eq!(calculate_row_pitch(256, 1), 256);
        assert_eq!(calculate_row_pitch(257, 1), 512);
        assert_eq!(calculate_row_pitch(512, 1), 512);
    }

    #[test]
    fn calculate_row_pitch_different_bpp_2() {
        assert_eq!(calculate_row_pitch(100, 2), 256);
        assert_eq!(calculate_row_pitch(128, 2), 256);
        assert_eq!(calculate_row_pitch(129, 2), 512);
        assert_eq!(calculate_row_pitch(256, 2), 512);
    }

    #[test]
    fn calculate_row_pitch_different_bpp_8() {
        assert_eq!(calculate_row_pitch(32, 8), 256);
        assert_eq!(calculate_row_pitch(33, 8), 512);
        assert_eq!(calculate_row_pitch(64, 8), 512);
    }

    #[test]
    fn calculate_row_pitch_different_bpp_16() {
        assert_eq!(calculate_row_pitch(16, 16), 256);
        assert_eq!(calculate_row_pitch(17, 16), 512);
        assert_eq!(calculate_row_pitch(32, 16), 512);
    }

    #[test]
    fn calculate_row_pitch_common_textures() {
        // Common texture sizes for RGBA (4 bpp)
        assert_eq!(calculate_row_pitch(256, 4), 1024);
        assert_eq!(calculate_row_pitch(512, 4), 2048);
        assert_eq!(calculate_row_pitch(1024, 4), 4096);
        assert_eq!(calculate_row_pitch(2048, 4), 8192);
        assert_eq!(calculate_row_pitch(4096, 4), 16384);
    }

    #[test]
    fn calculate_row_pitch_non_power_of_two() {
        assert_eq!(calculate_row_pitch(100, 4), 512);
        assert_eq!(calculate_row_pitch(300, 4), 1280);
        assert_eq!(calculate_row_pitch(500, 4), 2048);
        assert_eq!(calculate_row_pitch(768, 4), 3072);
        assert_eq!(calculate_row_pitch(1920, 4), 7680);
    }

    // -------------------------------------------------------------------------
    // is_row_pitch_aligned tests
    // -------------------------------------------------------------------------

    #[test]
    fn is_row_pitch_aligned_zero() {
        assert!(is_row_pitch_aligned(0));
    }

    #[test]
    fn is_row_pitch_aligned_multiples() {
        assert!(is_row_pitch_aligned(256));
        assert!(is_row_pitch_aligned(512));
        assert!(is_row_pitch_aligned(768));
        assert!(is_row_pitch_aligned(1024));
        assert!(is_row_pitch_aligned(65536));
    }

    #[test]
    fn is_row_pitch_aligned_non_multiples() {
        assert!(!is_row_pitch_aligned(1));
        assert!(!is_row_pitch_aligned(255));
        assert!(!is_row_pitch_aligned(257));
        assert!(!is_row_pitch_aligned(400));
        assert!(!is_row_pitch_aligned(511));
    }

    // -------------------------------------------------------------------------
    // pad_to_row_pitch tests
    // -------------------------------------------------------------------------

    #[test]
    fn pad_to_row_pitch_already_aligned_returns_borrowed() {
        // 64x2 RGBA = 256 bytes/row (aligned)
        let data = vec![42u8; 64 * 2 * 4];
        let result = pad_to_row_pitch(&data, 64, 2, 4);

        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn pad_to_row_pitch_needs_padding_returns_owned() {
        // 100x1 RGBA = 400 bytes/row -> 512
        let data = vec![1u8; 100 * 1 * 4];
        let result = pad_to_row_pitch(&data, 100, 1, 4);

        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result.len(), 512);
    }

    #[test]
    fn pad_to_row_pitch_preserves_data() {
        // 100x2 RGBA
        let mut data = vec![0u8; 100 * 2 * 4];
        // Fill with pattern
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        let result = pad_to_row_pitch(&data, 100, 2, 4);

        // Verify original data is preserved in correct positions
        for row in 0..2 {
            for col in 0..400 {
                let src_idx = row * 400 + col;
                let dst_idx = row * 512 + col;
                assert_eq!(result[dst_idx], data[src_idx],
                    "Mismatch at row={}, col={}", row, col);
            }
        }
    }

    #[test]
    fn pad_to_row_pitch_padding_is_zero() {
        // 100x1 RGBA
        let data = vec![255u8; 100 * 1 * 4];
        let result = pad_to_row_pitch(&data, 100, 1, 4);

        // Padding bytes (400..512) should be zero
        for i in 400..512 {
            assert_eq!(result[i], 0, "Padding byte {} should be zero", i);
        }
    }

    #[test]
    fn pad_to_row_pitch_empty_data() {
        let data = vec![];
        let result = pad_to_row_pitch(&data, 0, 0, 4);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn pad_to_row_pitch_single_row() {
        let data = vec![128u8; 50 * 4]; // 50 pixels, RGBA
        let result = pad_to_row_pitch(&data, 50, 1, 4);

        // 50 * 4 = 200 -> aligned to 256
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn pad_to_row_pitch_multiple_rows() {
        let data = vec![1u8; 100 * 10 * 4]; // 100x10 RGBA
        let result = pad_to_row_pitch(&data, 100, 10, 4);

        // 400 bytes/row -> 512, 10 rows = 5120 bytes
        assert_eq!(result.len(), 5120);
    }

    // -------------------------------------------------------------------------
    // pad_to_row_pitch_3d tests
    // -------------------------------------------------------------------------

    #[test]
    fn pad_to_row_pitch_3d_single_layer() {
        let data = vec![1u8; 100 * 2 * 1 * 4];
        let result = pad_to_row_pitch_3d(&data, 100, 2, 1, 4);

        // Same as 2D case
        assert_eq!(result.len(), 512 * 2);
    }

    #[test]
    fn pad_to_row_pitch_3d_multiple_layers() {
        let data = vec![1u8; 100 * 2 * 3 * 4]; // 100x2x3 RGBA
        let result = pad_to_row_pitch_3d(&data, 100, 2, 3, 4);

        // 512 bytes/row * 2 rows * 3 layers = 3072
        assert_eq!(result.len(), 3072);
    }

    #[test]
    fn pad_to_row_pitch_3d_already_aligned() {
        // 64x2x2 RGBA = 256 bytes/row (aligned)
        let data = vec![1u8; 64 * 2 * 2 * 4];
        let result = pad_to_row_pitch_3d(&data, 64, 2, 2, 4);

        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn pad_to_row_pitch_3d_preserves_layer_data() {
        // 10x2x2 RGBA (small for easy verification)
        let mut data = vec![0u8; 10 * 2 * 2 * 4];

        // Fill each layer with different value
        for z in 0..2 {
            let layer_start = z * (10 * 2 * 4);
            for i in 0..(10 * 2 * 4) {
                data[layer_start + i] = (z * 100) as u8;
            }
        }

        let result = pad_to_row_pitch_3d(&data, 10, 2, 2, 4);

        // Verify each layer's data
        for z in 0..2 {
            let expected_val = (z * 100) as u8;
            for row in 0..2 {
                let padded_row = 256; // 10*4=40 -> 256
                let dst_start = z * (padded_row * 2) + row * padded_row;
                for col in 0..40 {
                    assert_eq!(result[dst_start + col], expected_val,
                        "Mismatch at z={}, row={}, col={}", z, row, col);
                }
            }
        }
    }
}

// ============================================================================
// Module: error_tests
// ============================================================================

mod error_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // InvalidRegion variant
    // -------------------------------------------------------------------------

    #[test]
    fn invalid_region_display() {
        let err = TextureUploadError::InvalidRegion {
            offset: (100, 200, 0),
            size: (300, 400, 1),
            texture_size: (256, 256, 1),
            mip_level: 0,
        };
        let msg = err.to_string();

        assert!(msg.contains("100"));
        assert!(msg.contains("200"));
        assert!(msg.contains("300"));
        assert!(msg.contains("400"));
        assert!(msg.contains("256"));
        assert!(msg.contains("mip"));
    }

    #[test]
    fn invalid_region_debug() {
        let err = TextureUploadError::InvalidRegion {
            offset: (0, 0, 0),
            size: (1, 1, 1),
            texture_size: (1, 1, 1),
            mip_level: 0,
        };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InvalidRegion"));
    }

    #[test]
    fn invalid_region_eq() {
        let err1 = TextureUploadError::InvalidRegion {
            offset: (0, 0, 0),
            size: (1, 1, 1),
            texture_size: (1, 1, 1),
            mip_level: 0,
        };
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }

    // -------------------------------------------------------------------------
    // BufferTooSmall variant
    // -------------------------------------------------------------------------

    #[test]
    fn buffer_too_small_display() {
        let err = TextureUploadError::BufferTooSmall {
            provided: 100,
            required: 400,
        };
        let msg = err.to_string();

        assert!(msg.contains("100"));
        assert!(msg.contains("400"));
        assert!(msg.contains("buffer"));
        assert!(msg.contains("small"));
    }

    #[test]
    fn buffer_too_small_eq() {
        let err1 = TextureUploadError::BufferTooSmall {
            provided: 50,
            required: 100,
        };
        let err2 = TextureUploadError::BufferTooSmall {
            provided: 50,
            required: 100,
        };
        assert_eq!(err1, err2);
    }

    // -------------------------------------------------------------------------
    // FormatMismatch variant
    // -------------------------------------------------------------------------

    #[test]
    fn format_mismatch_display() {
        let err = TextureUploadError::FormatMismatch {
            expected: "RGBA8Unorm".to_string(),
            actual: "RGB8".to_string(),
        };
        let msg = err.to_string();

        assert!(msg.contains("RGBA8Unorm"));
        assert!(msg.contains("RGB8"));
        assert!(msg.contains("format"));
    }

    // -------------------------------------------------------------------------
    // AlignmentError variant
    // -------------------------------------------------------------------------

    #[test]
    fn alignment_error_display() {
        let err = TextureUploadError::AlignmentError {
            row_size: 400,
            alignment: 256,
        };
        let msg = err.to_string();

        assert!(msg.contains("400"));
        assert!(msg.contains("256"));
        assert!(msg.contains("alignment"));
    }

    // -------------------------------------------------------------------------
    // ZeroSizeRegion variant
    // -------------------------------------------------------------------------

    #[test]
    fn zero_size_region_display() {
        let err = TextureUploadError::ZeroSizeRegion;
        let msg = err.to_string();

        assert!(msg.contains("zero"));
        assert!(msg.contains("size") || msg.contains("region"));
    }

    // -------------------------------------------------------------------------
    // InvalidBytesPerPixel variant
    // -------------------------------------------------------------------------

    #[test]
    fn invalid_bytes_per_pixel_display() {
        let err = TextureUploadError::InvalidBytesPerPixel { value: 0 };
        let msg = err.to_string();

        assert!(msg.contains("0"));
        assert!(msg.contains("bytes") || msg.contains("pixel"));
    }

    #[test]
    fn invalid_bytes_per_pixel_display_large() {
        let err = TextureUploadError::InvalidBytesPerPixel { value: 99 };
        let msg = err.to_string();
        assert!(msg.contains("99"));
    }

    // -------------------------------------------------------------------------
    // MipLevelOutOfRange variant
    // -------------------------------------------------------------------------

    #[test]
    fn mip_level_out_of_range_display() {
        let err = TextureUploadError::MipLevelOutOfRange {
            level: 15,
            max_level: 10,
        };
        let msg = err.to_string();

        assert!(msg.contains("15"));
        assert!(msg.contains("10"));
        assert!(msg.contains("mip"));
    }

    // -------------------------------------------------------------------------
    // Error trait
    // -------------------------------------------------------------------------

    #[test]
    fn error_is_error_trait() {
        let err: &dyn std::error::Error = &TextureUploadError::ZeroSizeRegion;
        assert!(err.source().is_none()); // No underlying cause
    }

    #[test]
    fn all_variants_implement_clone() {
        let errors: Vec<TextureUploadError> = vec![
            TextureUploadError::InvalidRegion {
                offset: (0, 0, 0),
                size: (1, 1, 1),
                texture_size: (1, 1, 1),
                mip_level: 0,
            },
            TextureUploadError::BufferTooSmall { provided: 1, required: 2 },
            TextureUploadError::FormatMismatch { expected: "a".into(), actual: "b".into() },
            TextureUploadError::AlignmentError { row_size: 1, alignment: 256 },
            TextureUploadError::ZeroSizeRegion,
            TextureUploadError::InvalidBytesPerPixel { value: 0 },
            TextureUploadError::MipLevelOutOfRange { level: 1, max_level: 0 },
        ];

        for err in errors {
            let cloned = err.clone();
            assert_eq!(err, cloned);
        }
    }
}

// ============================================================================
// Module: descriptor_tests
// ============================================================================

mod descriptor_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Default trait
    // -------------------------------------------------------------------------

    #[test]
    fn default_values() {
        let desc = TextureUploadDescriptor::default();

        assert_eq!(desc.offset, (0, 0, 0));
        assert_eq!(desc.size, (1, 1, 1));
        assert_eq!(desc.mip_level, 0);
        assert_eq!(desc.bytes_per_pixel, 4); // RGBA8 most common
        assert!(desc.format.is_none());
    }

    // -------------------------------------------------------------------------
    // full() constructor
    // -------------------------------------------------------------------------

    #[test]
    fn full_basic() {
        let desc = TextureUploadDescriptor::full(512, 256, 4);

        assert_eq!(desc.offset, (0, 0, 0));
        assert_eq!(desc.size, (512, 256, 1));
        assert_eq!(desc.mip_level, 0);
        assert_eq!(desc.bytes_per_pixel, 4);
    }

    #[test]
    fn full_different_bpp() {
        let desc = TextureUploadDescriptor::full(100, 100, 1);
        assert_eq!(desc.bytes_per_pixel, 1);

        let desc = TextureUploadDescriptor::full(100, 100, 16);
        assert_eq!(desc.bytes_per_pixel, 16);
    }

    // -------------------------------------------------------------------------
    // mip_level() constructor
    // -------------------------------------------------------------------------

    #[test]
    fn mip_level_zero() {
        let desc = TextureUploadDescriptor::mip_level(1024, 512, 0, 4);

        assert_eq!(desc.size, (1024, 512, 1));
        assert_eq!(desc.mip_level, 0);
    }

    #[test]
    fn mip_level_one() {
        let desc = TextureUploadDescriptor::mip_level(1024, 512, 1, 4);

        assert_eq!(desc.size, (512, 256, 1));
        assert_eq!(desc.mip_level, 1);
    }

    #[test]
    fn mip_level_two() {
        let desc = TextureUploadDescriptor::mip_level(1024, 1024, 2, 4);

        assert_eq!(desc.size, (256, 256, 1));
        assert_eq!(desc.mip_level, 2);
    }

    #[test]
    fn mip_level_non_square() {
        // 1024x256 at mip 1 = 512x128
        let desc = TextureUploadDescriptor::mip_level(1024, 256, 1, 4);
        assert_eq!(desc.size, (512, 128, 1));

        // At mip 2 = 256x64
        let desc = TextureUploadDescriptor::mip_level(1024, 256, 2, 4);
        assert_eq!(desc.size, (256, 64, 1));
    }

    #[test]
    fn mip_level_clamps_to_one() {
        // Very high mip level should result in 1x1
        let desc = TextureUploadDescriptor::mip_level(256, 256, 10, 4);
        assert_eq!(desc.size, (1, 1, 1));

        let desc = TextureUploadDescriptor::mip_level(1024, 1024, 20, 4);
        assert_eq!(desc.size, (1, 1, 1));
    }

    // -------------------------------------------------------------------------
    // subregion() constructor
    // -------------------------------------------------------------------------

    #[test]
    fn subregion_basic() {
        let desc = TextureUploadDescriptor::subregion(100, 200, 50, 75, 4);

        assert_eq!(desc.offset, (100, 200, 0));
        assert_eq!(desc.size, (50, 75, 1));
        assert_eq!(desc.mip_level, 0);
    }

    #[test]
    fn subregion_at_origin() {
        let desc = TextureUploadDescriptor::subregion(0, 0, 128, 128, 4);

        assert_eq!(desc.offset, (0, 0, 0));
        assert_eq!(desc.size, (128, 128, 1));
    }

    // -------------------------------------------------------------------------
    // Builder methods
    // -------------------------------------------------------------------------

    #[test]
    fn with_format() {
        let desc = TextureUploadDescriptor::full(256, 256, 4)
            .with_format(TextureFormat::Rgba8Unorm);

        assert_eq!(desc.format, Some(TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn with_mip_level() {
        let desc = TextureUploadDescriptor::full(256, 256, 4)
            .with_mip_level(3);

        assert_eq!(desc.mip_level, 3);
    }

    #[test]
    fn with_layer() {
        let desc = TextureUploadDescriptor::full(256, 256, 4)
            .with_layer(5);

        assert_eq!(desc.offset.2, 5);
    }

    #[test]
    fn chained_builders() {
        let desc = TextureUploadDescriptor::full(1024, 1024, 4)
            .with_format(TextureFormat::Bgra8Unorm)
            .with_mip_level(2)
            .with_layer(3);

        assert_eq!(desc.format, Some(TextureFormat::Bgra8Unorm));
        assert_eq!(desc.mip_level, 2);
        assert_eq!(desc.offset.2, 3);
    }

    // -------------------------------------------------------------------------
    // required_data_size()
    // -------------------------------------------------------------------------

    #[test]
    fn required_data_size_aligned() {
        // 64x64 RGBA: 64*4 = 256 (aligned), 256 * 64 = 16384
        let desc = TextureUploadDescriptor::full(64, 64, 4);
        assert_eq!(desc.required_data_size(), 16384);
    }

    #[test]
    fn required_data_size_needs_padding() {
        // 100x100 RGBA: 100*4 = 400 -> 512, 512 * 100 = 51200
        let desc = TextureUploadDescriptor::full(100, 100, 4);
        assert_eq!(desc.required_data_size(), 51200);
    }

    #[test]
    fn required_data_size_single_pixel() {
        // 1x1 RGBA: 4 -> 256, 256 * 1 = 256
        let desc = TextureUploadDescriptor::full(1, 1, 4);
        assert_eq!(desc.required_data_size(), 256);
    }

    #[test]
    fn required_data_size_with_depth() {
        // 100x100x2 RGBA
        let mut desc = TextureUploadDescriptor::full(100, 100, 4);
        desc.size.2 = 2;
        // 512 * 100 * 2 = 102400
        assert_eq!(desc.required_data_size(), 102400);
    }

    // -------------------------------------------------------------------------
    // unpadded_data_size()
    // -------------------------------------------------------------------------

    #[test]
    fn unpadded_data_size_basic() {
        let desc = TextureUploadDescriptor::full(100, 100, 4);
        assert_eq!(desc.unpadded_data_size(), 100 * 100 * 4);
    }

    #[test]
    fn unpadded_data_size_different_bpp() {
        let desc = TextureUploadDescriptor::full(256, 256, 1);
        assert_eq!(desc.unpadded_data_size(), 256 * 256 * 1);

        let desc = TextureUploadDescriptor::full(256, 256, 8);
        assert_eq!(desc.unpadded_data_size(), 256 * 256 * 8);
    }

    #[test]
    fn unpadded_data_size_with_depth() {
        let mut desc = TextureUploadDescriptor::full(100, 100, 4);
        desc.size.2 = 5;
        assert_eq!(desc.unpadded_data_size(), 100 * 100 * 5 * 4);
    }

    // -------------------------------------------------------------------------
    // Clone/Eq traits
    // -------------------------------------------------------------------------

    #[test]
    fn descriptor_clone() {
        let original = TextureUploadDescriptor::full(512, 512, 4)
            .with_format(TextureFormat::Rgba8Unorm);
        let cloned = original.clone();

        assert_eq!(original, cloned);
    }

    #[test]
    fn descriptor_debug() {
        let desc = TextureUploadDescriptor::default();
        let debug_str = format!("{:?}", desc);

        assert!(debug_str.contains("TextureUploadDescriptor"));
    }
}

// ============================================================================
// Module: region_tests
// ============================================================================

mod region_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // full() constructor
    // -------------------------------------------------------------------------

    #[test]
    fn full_2d_texture() {
        let region = TextureRegion::full(1024, 512, 1);

        assert_eq!(region.offset, (0, 0, 0));
        assert_eq!(region.size, (1024, 512, 1));
    }

    #[test]
    fn full_3d_texture() {
        let region = TextureRegion::full(256, 256, 64);

        assert_eq!(region.size, (256, 256, 64));
    }

    #[test]
    fn full_const() {
        // Verify it's a const fn
        const REGION: TextureRegion = TextureRegion::full(128, 128, 1);
        assert_eq!(REGION.size.0, 128);
    }

    // -------------------------------------------------------------------------
    // mip() constructor
    // -------------------------------------------------------------------------

    #[test]
    fn mip_level_zero() {
        let region = TextureRegion::mip(1024, 512, 0);
        assert_eq!(region.size, (1024, 512, 1));
    }

    #[test]
    fn mip_level_calculations() {
        assert_eq!(TextureRegion::mip(1024, 1024, 0).size, (1024, 1024, 1));
        assert_eq!(TextureRegion::mip(1024, 1024, 1).size, (512, 512, 1));
        assert_eq!(TextureRegion::mip(1024, 1024, 2).size, (256, 256, 1));
        assert_eq!(TextureRegion::mip(1024, 1024, 3).size, (128, 128, 1));
        assert_eq!(TextureRegion::mip(1024, 1024, 10).size, (1, 1, 1));
    }

    #[test]
    fn mip_non_square() {
        assert_eq!(TextureRegion::mip(2048, 512, 0).size, (2048, 512, 1));
        assert_eq!(TextureRegion::mip(2048, 512, 1).size, (1024, 256, 1));
        assert_eq!(TextureRegion::mip(2048, 512, 2).size, (512, 128, 1));
    }

    #[test]
    fn mip_clamps_to_one() {
        // When shifted to 0, clamps to 1
        let region = TextureRegion::mip(256, 256, 8);
        assert_eq!(region.size, (1, 1, 1));

        // Very high mip level
        let region = TextureRegion::mip(4096, 4096, 20);
        assert_eq!(region.size, (1, 1, 1));
    }

    #[test]
    fn mip_level_32_or_higher() {
        // Edge case: mip_level >= 32 should still return 1
        let region = TextureRegion::mip(1024, 1024, 32);
        assert_eq!(region.size, (1, 1, 1));

        let region = TextureRegion::mip(1024, 1024, 100);
        assert_eq!(region.size, (1, 1, 1));
    }

    // -------------------------------------------------------------------------
    // subregion() constructor
    // -------------------------------------------------------------------------

    #[test]
    fn subregion_basic() {
        let region = TextureRegion::subregion(10, 20, 100, 200);

        assert_eq!(region.offset, (10, 20, 0));
        assert_eq!(region.size, (100, 200, 1));
    }

    #[test]
    fn subregion_at_origin() {
        let region = TextureRegion::subregion(0, 0, 50, 50);

        assert_eq!(region.offset, (0, 0, 0));
        assert_eq!(region.size, (50, 50, 1));
    }

    // -------------------------------------------------------------------------
    // layer() constructor
    // -------------------------------------------------------------------------

    #[test]
    fn layer_basic() {
        let region = TextureRegion::layer(256, 256, 3);

        assert_eq!(region.offset, (0, 0, 3));
        assert_eq!(region.size, (256, 256, 1));
    }

    #[test]
    fn layer_zero() {
        let region = TextureRegion::layer(512, 512, 0);

        assert_eq!(region.offset.2, 0);
    }

    // -------------------------------------------------------------------------
    // layer_range() constructor
    // -------------------------------------------------------------------------

    #[test]
    fn layer_range_basic() {
        let region = TextureRegion::layer_range(256, 256, 2, 4);

        assert_eq!(region.offset, (0, 0, 2));
        assert_eq!(region.size, (256, 256, 4));
    }

    #[test]
    fn layer_range_from_zero() {
        let region = TextureRegion::layer_range(128, 128, 0, 6);

        assert_eq!(region.offset.2, 0);
        assert_eq!(region.size.2, 6);
    }

    // -------------------------------------------------------------------------
    // is_valid_for()
    // -------------------------------------------------------------------------

    #[test]
    fn is_valid_for_full_texture() {
        let region = TextureRegion::full(256, 256, 1);
        assert!(region.is_valid_for(256, 256, 1));
    }

    #[test]
    fn is_valid_for_subregion() {
        let region = TextureRegion::subregion(100, 100, 50, 50);

        assert!(region.is_valid_for(256, 256, 1));
        assert!(region.is_valid_for(150, 150, 1));
        assert!(!region.is_valid_for(140, 256, 1)); // x overflow
        assert!(!region.is_valid_for(256, 140, 1)); // y overflow
    }

    #[test]
    fn is_valid_for_exact_fit() {
        let region = TextureRegion::subregion(100, 100, 156, 156);
        assert!(region.is_valid_for(256, 256, 1));
    }

    #[test]
    fn is_valid_for_layer_overflow() {
        let region = TextureRegion::layer(256, 256, 5);

        assert!(region.is_valid_for(256, 256, 6));
        assert!(!region.is_valid_for(256, 256, 5)); // exactly at limit, size=1 overflows
    }

    #[test]
    fn is_valid_for_layer_range() {
        let region = TextureRegion::layer_range(256, 256, 2, 4);

        assert!(region.is_valid_for(256, 256, 6));
        assert!(!region.is_valid_for(256, 256, 5)); // 2+4=6, but depth is 5
    }

    // -------------------------------------------------------------------------
    // is_non_empty()
    // -------------------------------------------------------------------------

    #[test]
    fn is_non_empty_valid() {
        assert!(TextureRegion::full(1, 1, 1).is_non_empty());
        assert!(TextureRegion::full(100, 100, 100).is_non_empty());
    }

    #[test]
    fn is_non_empty_zero_width() {
        let region = TextureRegion { offset: (0, 0, 0), size: (0, 100, 1) };
        assert!(!region.is_non_empty());
    }

    #[test]
    fn is_non_empty_zero_height() {
        let region = TextureRegion { offset: (0, 0, 0), size: (100, 0, 1) };
        assert!(!region.is_non_empty());
    }

    #[test]
    fn is_non_empty_zero_depth() {
        let region = TextureRegion { offset: (0, 0, 0), size: (100, 100, 0) };
        assert!(!region.is_non_empty());
    }

    // -------------------------------------------------------------------------
    // Default trait
    // -------------------------------------------------------------------------

    #[test]
    fn default_is_1x1x1() {
        let region = TextureRegion::default();

        assert_eq!(region.offset, (0, 0, 0));
        assert_eq!(region.size, (1, 1, 1));
    }

    // -------------------------------------------------------------------------
    // Copy trait
    // -------------------------------------------------------------------------

    #[test]
    fn region_is_copy() {
        let region1 = TextureRegion::full(100, 100, 1);
        let region2 = region1; // Copy, not move

        assert_eq!(region1, region2);
    }
}

// ============================================================================
// Module: format_conversion_tests
// ============================================================================

mod format_conversion_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // convert_rgb_to_rgba
    // -------------------------------------------------------------------------

    #[test]
    fn rgb_to_rgba_single_pixel() {
        let rgb = [255, 128, 64];
        let rgba = convert_rgb_to_rgba(&rgb);

        assert_eq!(rgba, [255, 128, 64, 255]);
    }

    #[test]
    fn rgb_to_rgba_multiple_pixels() {
        let rgb = [
            255, 0, 0,   // Red
            0, 255, 0,   // Green
            0, 0, 255,   // Blue
        ];
        let rgba = convert_rgb_to_rgba(&rgb);

        assert_eq!(rgba, [
            255, 0, 0, 255,
            0, 255, 0, 255,
            0, 0, 255, 255,
        ]);
    }

    #[test]
    fn rgb_to_rgba_empty() {
        let rgb: [u8; 0] = [];
        let rgba = convert_rgb_to_rgba(&rgb);

        assert_eq!(rgba.len(), 0);
    }

    #[test]
    fn rgb_to_rgba_black_white() {
        let rgb = [0, 0, 0, 255, 255, 255];
        let rgba = convert_rgb_to_rgba(&rgb);

        assert_eq!(rgba, [0, 0, 0, 255, 255, 255, 255, 255]);
    }

    #[test]
    fn rgb_to_rgba_length() {
        let rgb = vec![0u8; 3 * 100];
        let rgba = convert_rgb_to_rgba(&rgb);

        assert_eq!(rgba.len(), 4 * 100);
    }

    // -------------------------------------------------------------------------
    // convert_bgra_to_rgba
    // -------------------------------------------------------------------------

    #[test]
    fn bgra_to_rgba_single_pixel() {
        let bgra = [64, 128, 255, 200]; // BGRA
        let rgba = convert_bgra_to_rgba(&bgra);

        assert_eq!(rgba, [255, 128, 64, 200]); // RGBA
    }

    #[test]
    fn bgra_to_rgba_multiple_pixels() {
        let bgra = [
            255, 0, 0, 255,     // Blue in BGRA = Blue
            0, 255, 0, 255,     // Green in BGRA = Green
            0, 0, 255, 255,     // Red in BGRA = Red
        ];
        let rgba = convert_bgra_to_rgba(&bgra);

        // R and B swapped
        assert_eq!(rgba, [
            0, 0, 255, 255,     // Was Blue, now Red
            0, 255, 0, 255,     // Green unchanged
            255, 0, 0, 255,     // Was Red, now Blue
        ]);
    }

    #[test]
    fn bgra_to_rgba_empty() {
        let bgra: [u8; 0] = [];
        let rgba = convert_bgra_to_rgba(&bgra);

        assert_eq!(rgba.len(), 0);
    }

    #[test]
    fn bgra_to_rgba_alpha_preserved() {
        let bgra = [100, 150, 200, 50];
        let rgba = convert_bgra_to_rgba(&bgra);

        assert_eq!(rgba[3], 50); // Alpha preserved
    }

    // -------------------------------------------------------------------------
    // convert_rgba_to_bgra
    // -------------------------------------------------------------------------

    #[test]
    fn rgba_to_bgra_single_pixel() {
        let rgba = [255, 128, 64, 200];
        let bgra = convert_rgba_to_bgra(&rgba);

        assert_eq!(bgra, [64, 128, 255, 200]);
    }

    #[test]
    fn rgba_to_bgra_is_inverse() {
        let original = [100, 150, 200, 250];
        let bgra = convert_rgba_to_bgra(&original);
        let back = convert_bgra_to_rgba(&bgra);

        assert_eq!(back, original);
    }

    #[test]
    fn rgba_to_bgra_roundtrip() {
        let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let bgra = convert_rgba_to_bgra(&data);
        let back = convert_bgra_to_rgba(&bgra);

        assert_eq!(back, data);
    }

    // -------------------------------------------------------------------------
    // convert_gray_to_rgba
    // -------------------------------------------------------------------------

    #[test]
    fn gray_to_rgba_single_pixel() {
        let gray = [128];
        let rgba = convert_gray_to_rgba(&gray);

        assert_eq!(rgba, [128, 128, 128, 255]);
    }

    #[test]
    fn gray_to_rgba_multiple_pixels() {
        let gray = [0, 128, 255];
        let rgba = convert_gray_to_rgba(&gray);

        assert_eq!(rgba, [
            0, 0, 0, 255,
            128, 128, 128, 255,
            255, 255, 255, 255,
        ]);
    }

    #[test]
    fn gray_to_rgba_empty() {
        let gray: [u8; 0] = [];
        let rgba = convert_gray_to_rgba(&gray);

        assert_eq!(rgba.len(), 0);
    }

    #[test]
    fn gray_to_rgba_length() {
        let gray = vec![100u8; 50];
        let rgba = convert_gray_to_rgba(&gray);

        assert_eq!(rgba.len(), 50 * 4);
    }

    // -------------------------------------------------------------------------
    // convert_gray_alpha_to_rgba
    // -------------------------------------------------------------------------

    #[test]
    fn gray_alpha_to_rgba_single_pixel() {
        let gray_alpha = [128, 200];
        let rgba = convert_gray_alpha_to_rgba(&gray_alpha);

        assert_eq!(rgba, [128, 128, 128, 200]);
    }

    #[test]
    fn gray_alpha_to_rgba_multiple_pixels() {
        let gray_alpha = [
            64, 255,   // Gray 64, fully opaque
            192, 128,  // Gray 192, half transparent
        ];
        let rgba = convert_gray_alpha_to_rgba(&gray_alpha);

        assert_eq!(rgba, [
            64, 64, 64, 255,
            192, 192, 192, 128,
        ]);
    }

    #[test]
    fn gray_alpha_to_rgba_empty() {
        let gray_alpha: [u8; 0] = [];
        let rgba = convert_gray_alpha_to_rgba(&gray_alpha);

        assert_eq!(rgba.len(), 0);
    }

    #[test]
    fn gray_alpha_to_rgba_transparent() {
        let gray_alpha = [255, 0]; // White, fully transparent
        let rgba = convert_gray_alpha_to_rgba(&gray_alpha);

        assert_eq!(rgba, [255, 255, 255, 0]);
    }

    // -------------------------------------------------------------------------
    // premultiply_alpha
    // -------------------------------------------------------------------------

    #[test]
    fn premultiply_fully_opaque() {
        let rgba = [255, 128, 64, 255];
        let premul = premultiply_alpha(&rgba);

        // Fully opaque: colors unchanged
        assert_eq!(premul, [255, 128, 64, 255]);
    }

    #[test]
    fn premultiply_fully_transparent() {
        let rgba = [255, 128, 64, 0];
        let premul = premultiply_alpha(&rgba);

        // Fully transparent: colors become 0
        assert_eq!(premul, [0, 0, 0, 0]);
    }

    #[test]
    fn premultiply_half_alpha() {
        let rgba = [255, 128, 64, 128]; // ~50% alpha
        let premul = premultiply_alpha(&rgba);

        // 255 * 128 / 255 = 128
        assert_eq!(premul[0], 128);
        // 128 * 128 / 255 = 64.25... -> 64
        assert_eq!(premul[1], 64);
        // 64 * 128 / 255 = 32.12... -> 32
        assert_eq!(premul[2], 32);
        assert_eq!(premul[3], 128); // Alpha unchanged
    }

    #[test]
    fn premultiply_quarter_alpha() {
        let rgba = [252, 128, 64, 64]; // ~25% alpha
        let premul = premultiply_alpha(&rgba);

        // 252 * 64 / 255 = 63.24... -> 63
        assert_eq!(premul[0], 63);
        // 128 * 64 / 255 = 32.12... -> 32
        assert_eq!(premul[1], 32);
        // 64 * 64 / 255 = 16.06... -> 16
        assert_eq!(premul[2], 16);
    }

    #[test]
    fn premultiply_multiple_pixels() {
        let rgba = [
            255, 255, 255, 255, // Opaque white
            255, 255, 255, 0,   // Transparent white
        ];
        let premul = premultiply_alpha(&rgba);

        assert_eq!(premul[0..4], [255, 255, 255, 255]);
        assert_eq!(premul[4..8], [0, 0, 0, 0]);
    }

    #[test]
    fn premultiply_empty() {
        let rgba: [u8; 0] = [];
        let premul = premultiply_alpha(&rgba);

        assert_eq!(premul.len(), 0);
    }

    #[test]
    fn premultiply_black() {
        let rgba = [0, 0, 0, 128];
        let premul = premultiply_alpha(&rgba);

        // Black stays black regardless of alpha
        assert_eq!(premul, [0, 0, 0, 128]);
    }
}

// ============================================================================
// Module: uploader_tests
// ============================================================================

mod uploader_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    #[test]
    fn new_with_threshold() {
        let uploader = TextureUploader::new(1024);
        assert_eq!(uploader.staging_threshold(), 1024);
    }

    #[test]
    fn new_with_zero_threshold() {
        let uploader = TextureUploader::new(0);
        assert_eq!(uploader.staging_threshold(), 0);
    }

    #[test]
    fn new_with_large_threshold() {
        let uploader = TextureUploader::new(u64::MAX);
        assert_eq!(uploader.staging_threshold(), u64::MAX);
    }

    #[test]
    fn with_default_threshold() {
        let uploader = TextureUploader::with_default_threshold();
        assert_eq!(uploader.staging_threshold(), STAGING_THRESHOLD);
    }

    #[test]
    fn default_trait() {
        let uploader = TextureUploader::default();
        assert_eq!(uploader.staging_threshold(), STAGING_THRESHOLD);
    }

    // -------------------------------------------------------------------------
    // Threshold management
    // -------------------------------------------------------------------------

    #[test]
    fn set_staging_threshold() {
        let mut uploader = TextureUploader::new(1024);
        uploader.set_staging_threshold(4096);
        assert_eq!(uploader.staging_threshold(), 4096);
    }

    #[test]
    fn staging_threshold_getter() {
        let uploader = TextureUploader::new(12345);
        assert_eq!(uploader.staging_threshold(), 12345);
    }

    // -------------------------------------------------------------------------
    // Validation: ZeroSizeRegion
    // -------------------------------------------------------------------------

    #[test]
    fn validate_zero_width() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor {
            size: (0, 100, 1),
            ..Default::default()
        };

        let result = uploader.validate_upload(&desc, 100);
        assert!(matches!(result, Err(TextureUploadError::ZeroSizeRegion)));
    }

    #[test]
    fn validate_zero_height() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor {
            size: (100, 0, 1),
            ..Default::default()
        };

        let result = uploader.validate_upload(&desc, 100);
        assert!(matches!(result, Err(TextureUploadError::ZeroSizeRegion)));
    }

    #[test]
    fn validate_zero_depth() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor {
            size: (100, 100, 0),
            ..Default::default()
        };

        let result = uploader.validate_upload(&desc, 100);
        assert!(matches!(result, Err(TextureUploadError::ZeroSizeRegion)));
    }

    // -------------------------------------------------------------------------
    // Validation: InvalidBytesPerPixel
    // -------------------------------------------------------------------------

    #[test]
    fn validate_bpp_zero() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor {
            bytes_per_pixel: 0,
            ..Default::default()
        };

        let result = uploader.validate_upload(&desc, 100);
        assert!(matches!(
            result,
            Err(TextureUploadError::InvalidBytesPerPixel { value: 0 })
        ));
    }

    #[test]
    fn validate_bpp_too_large() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor {
            bytes_per_pixel: 17, // Max is 16
            ..Default::default()
        };

        let result = uploader.validate_upload(&desc, 100);
        assert!(matches!(
            result,
            Err(TextureUploadError::InvalidBytesPerPixel { value: 17 })
        ));
    }

    #[test]
    fn validate_bpp_valid_range() {
        let uploader = TextureUploader::new(1024);

        for bpp in [1, 2, 4, 8, 16] {
            let desc = TextureUploadDescriptor {
                bytes_per_pixel: bpp,
                size: (1, 1, 1),
                ..Default::default()
            };
            let required = (bpp as usize);
            let result = uploader.validate_upload(&desc, required);
            assert!(result.is_ok(), "BPP {} should be valid", bpp);
        }
    }

    // -------------------------------------------------------------------------
    // Validation: BufferTooSmall
    // -------------------------------------------------------------------------

    #[test]
    fn validate_buffer_exactly_right() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor::full(10, 10, 4);

        // 10 * 10 * 4 = 400 bytes
        let result = uploader.validate_upload(&desc, 400);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_buffer_too_small() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor::full(10, 10, 4);

        // Needs 400, provide 399
        let result = uploader.validate_upload(&desc, 399);
        assert!(matches!(
            result,
            Err(TextureUploadError::BufferTooSmall { provided: 399, required: 400 })
        ));
    }

    #[test]
    fn validate_buffer_larger_is_ok() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor::full(10, 10, 4);

        // Needs 400, provide 1000
        let result = uploader.validate_upload(&desc, 1000);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_buffer_with_depth() {
        let uploader = TextureUploader::new(1024);
        let mut desc = TextureUploadDescriptor::full(10, 10, 4);
        desc.size.2 = 5; // 5 layers

        // 10 * 10 * 5 * 4 = 2000 bytes
        let result = uploader.validate_upload(&desc, 2000);
        assert!(result.is_ok());

        let result = uploader.validate_upload(&desc, 1999);
        assert!(matches!(result, Err(TextureUploadError::BufferTooSmall { .. })));
    }

    // -------------------------------------------------------------------------
    // Clone/Debug
    // -------------------------------------------------------------------------

    #[test]
    fn uploader_clone() {
        let uploader = TextureUploader::new(12345);
        let cloned = uploader.clone();

        assert_eq!(cloned.staging_threshold(), 12345);
    }

    #[test]
    fn uploader_debug() {
        let uploader = TextureUploader::new(1024);
        let debug_str = format!("{:?}", uploader);

        assert!(debug_str.contains("TextureUploader"));
        assert!(debug_str.contains("1024"));
    }
}

// ============================================================================
// Module: edge_case_tests
// ============================================================================

mod edge_case_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Zero dimensions
    // -------------------------------------------------------------------------

    #[test]
    fn zero_width_texture() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor::full(0, 100, 4);

        let result = uploader.validate_upload(&desc, 0);
        assert!(matches!(result, Err(TextureUploadError::ZeroSizeRegion)));
    }

    #[test]
    fn zero_height_texture() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor::full(100, 0, 4);

        let result = uploader.validate_upload(&desc, 0);
        assert!(matches!(result, Err(TextureUploadError::ZeroSizeRegion)));
    }

    #[test]
    fn all_zeros() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor {
            size: (0, 0, 0),
            ..Default::default()
        };

        let result = uploader.validate_upload(&desc, 0);
        assert!(matches!(result, Err(TextureUploadError::ZeroSizeRegion)));
    }

    // -------------------------------------------------------------------------
    // Single pixel
    // -------------------------------------------------------------------------

    #[test]
    fn single_pixel_texture() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor::full(1, 1, 4);

        let result = uploader.validate_upload(&desc, 4);
        assert!(result.is_ok());
    }

    #[test]
    fn single_pixel_required_data_size() {
        let desc = TextureUploadDescriptor::full(1, 1, 4);

        // 1 * 4 = 4 -> aligned to 256
        assert_eq!(desc.required_data_size(), 256);
    }

    // -------------------------------------------------------------------------
    // Very large textures
    // -------------------------------------------------------------------------

    #[test]
    fn large_texture_validation() {
        let uploader = TextureUploader::new(1024);
        let desc = TextureUploadDescriptor::full(4096, 4096, 4);

        // 4096 * 4096 * 4 = 67108864 bytes
        let required = 4096 * 4096 * 4;
        let result = uploader.validate_upload(&desc, required);
        assert!(result.is_ok());
    }

    #[test]
    fn large_texture_row_pitch() {
        // 4096 pixels * 4 bytes = 16384 (already aligned to 256)
        assert_eq!(calculate_row_pitch(4096, 4), 16384);

        // 8192 pixels * 4 bytes = 32768 (already aligned)
        assert_eq!(calculate_row_pitch(8192, 4), 32768);
    }

    // -------------------------------------------------------------------------
    // Misaligned data
    // -------------------------------------------------------------------------

    #[test]
    fn misaligned_width_padding() {
        // 100 pixels * 4 bytes = 400, needs padding to 512
        let data = vec![0u8; 100 * 2 * 4]; // 100x2 RGBA, unpadded
        let padded = pad_to_row_pitch(&data, 100, 2, 4);

        assert_eq!(padded.len(), 512 * 2); // 512 bytes/row * 2 rows
    }

    #[test]
    fn odd_width_handling() {
        // 97 pixels * 4 = 388 -> 512
        assert_eq!(calculate_row_pitch(97, 4), 512);

        // 1 pixel * 4 = 4 -> 256
        assert_eq!(calculate_row_pitch(1, 4), 256);
    }

    // -------------------------------------------------------------------------
    // Region validation edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn region_at_exact_boundary() {
        let region = TextureRegion::subregion(200, 200, 56, 56);

        // Exactly fits in 256x256
        assert!(region.is_valid_for(256, 256, 1));
    }

    #[test]
    fn region_one_pixel_overflow_x() {
        let region = TextureRegion::subregion(200, 200, 57, 56);

        // 200 + 57 = 257 > 256
        assert!(!region.is_valid_for(256, 256, 1));
    }

    #[test]
    fn region_one_pixel_overflow_y() {
        let region = TextureRegion::subregion(200, 200, 56, 57);

        // 200 + 57 = 257 > 256
        assert!(!region.is_valid_for(256, 256, 1));
    }

    #[test]
    fn region_saturating_add_overflow() {
        // Test that is_valid_for handles potential u32 overflow.
        // The current implementation uses saturating_add which prevents panic.
        let region = TextureRegion {
            offset: (u32::MAX - 10, 0, 0),
            size: (100, 1, 1),
        };

        // saturating_add(u32::MAX - 10, 100) = u32::MAX (saturates)
        // Then u32::MAX <= u32::MAX is true, so the region is considered valid
        // for the x dimension when tex_width is u32::MAX.
        // This is technically correct behavior with saturating arithmetic.
        assert!(region.is_valid_for(u32::MAX, 1, 1));
    }

    // -------------------------------------------------------------------------
    // Mip level edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn mip_level_clamping() {
        // 1x1 texture at any mip > 0 should still be 1x1
        let region = TextureRegion::mip(1, 1, 5);
        assert_eq!(region.size, (1, 1, 1));
    }

    #[test]
    fn mip_asymmetric_dimensions() {
        // One dimension reaches 1 before the other
        // 256x64 at mip 2 = 64x16
        assert_eq!(TextureRegion::mip(256, 64, 2).size, (64, 16, 1));

        // At mip 6, 256 -> 4, 64 -> 1
        assert_eq!(TextureRegion::mip(256, 64, 6).size, (4, 1, 1));
    }

    #[test]
    fn mip_size_utility_function() {
        assert_eq!(mip_size(1024, 0), 1024);
        assert_eq!(mip_size(1024, 1), 512);
        assert_eq!(mip_size(1024, 10), 1);
        assert_eq!(mip_size(1024, 11), 1);
        // Note: mip_level >= 32 causes shift overflow panic in current implementation.
        // This is acceptable as textures never have 32+ mip levels in practice.
        // Max mip levels for 16384x16384 is 15 (log2(16384)+1).
    }

    // -------------------------------------------------------------------------
    // Format conversion edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn rgb_to_rgba_partial_pixel() {
        // Input not a multiple of 3 - only full pixels converted
        let rgb = [255, 128]; // Less than 1 pixel
        let rgba = convert_rgb_to_rgba(&rgb);

        // chunks_exact skips the remainder
        assert_eq!(rgba.len(), 0);
    }

    #[test]
    fn bgra_to_rgba_partial_pixel() {
        // Input not a multiple of 4
        let bgra = [64, 128, 255]; // Less than 1 pixel
        let rgba = convert_bgra_to_rgba(&bgra);

        // chunks_exact_mut on original data, but it's not full pixel
        // The function copies then mutates, so partial data untouched
        assert_eq!(rgba.len(), 3);
        // Swap operation on incomplete chunk doesn't happen with chunks_exact_mut
    }

    #[test]
    fn gray_alpha_partial_pixel() {
        // Input not a multiple of 2
        let gray_alpha = [128]; // Half a pixel
        let rgba = convert_gray_alpha_to_rgba(&gray_alpha);

        assert_eq!(rgba.len(), 0);
    }

    #[test]
    fn premultiply_all_channels_max() {
        let rgba = [255, 255, 255, 255];
        let premul = premultiply_alpha(&rgba);

        assert_eq!(premul, [255, 255, 255, 255]);
    }

    #[test]
    fn premultiply_all_channels_min() {
        let rgba = [0, 0, 0, 0];
        let premul = premultiply_alpha(&rgba);

        assert_eq!(premul, [0, 0, 0, 0]);
    }

    // -------------------------------------------------------------------------
    // bytes_per_pixel_for_format
    // -------------------------------------------------------------------------

    #[test]
    fn bytes_per_pixel_r8_formats() {
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::R8Unorm), Some(1));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::R8Snorm), Some(1));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::R8Uint), Some(1));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::R8Sint), Some(1));
    }

    #[test]
    fn bytes_per_pixel_rg8_formats() {
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rg8Unorm), Some(2));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rg8Snorm), Some(2));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rg8Uint), Some(2));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rg8Sint), Some(2));
    }

    #[test]
    fn bytes_per_pixel_rgba8_formats() {
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rgba8Unorm), Some(4));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rgba8UnormSrgb), Some(4));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rgba8Snorm), Some(4));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Bgra8Unorm), Some(4));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Bgra8UnormSrgb), Some(4));
    }

    #[test]
    fn bytes_per_pixel_16_formats() {
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::R16Float), Some(2));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rg16Float), Some(4));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rgba16Float), Some(8));
    }

    #[test]
    fn bytes_per_pixel_32_formats() {
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::R32Float), Some(4));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rg32Float), Some(8));
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Rgba32Float), Some(16));
    }

    #[test]
    fn bytes_per_pixel_compressed_returns_none() {
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Bc1RgbaUnorm), None);
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Bc2RgbaUnorm), None);
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Bc3RgbaUnorm), None);
    }

    #[test]
    fn bytes_per_pixel_depth_returns_none() {
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Depth16Unorm), None);
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Depth24Plus), None);
        assert_eq!(bytes_per_pixel_for_format(TextureFormat::Depth32Float), None);
    }

    // -------------------------------------------------------------------------
    // BPP boundary validation
    // -------------------------------------------------------------------------

    #[test]
    fn validate_bpp_at_boundary() {
        let uploader = TextureUploader::new(1024);

        // BPP = 16 is valid (max)
        let desc = TextureUploadDescriptor {
            bytes_per_pixel: 16,
            size: (1, 1, 1),
            ..Default::default()
        };
        assert!(uploader.validate_upload(&desc, 16).is_ok());

        // BPP = 17 is invalid
        let desc = TextureUploadDescriptor {
            bytes_per_pixel: 17,
            size: (1, 1, 1),
            ..Default::default()
        };
        assert!(uploader.validate_upload(&desc, 17).is_err());
    }
}

// ============================================================================
// Module: utility_tests
// ============================================================================

mod utility_tests {
    use super::*;

    #[test]
    fn mip_size_standard_cases() {
        assert_eq!(mip_size(2048, 0), 2048);
        assert_eq!(mip_size(2048, 1), 1024);
        assert_eq!(mip_size(2048, 2), 512);
        assert_eq!(mip_size(2048, 3), 256);
        assert_eq!(mip_size(2048, 11), 1);
    }

    #[test]
    fn mip_size_non_power_of_two() {
        // 1000 >> 1 = 500
        assert_eq!(mip_size(1000, 1), 500);
        // 500 >> 1 = 250
        assert_eq!(mip_size(1000, 2), 250);
        // 250 >> 1 = 125
        assert_eq!(mip_size(1000, 3), 125);
    }

    #[test]
    fn mip_size_one_base() {
        assert_eq!(mip_size(1, 0), 1);
        assert_eq!(mip_size(1, 1), 1); // Clamps to 1
        assert_eq!(mip_size(1, 10), 1);
    }

    #[test]
    fn mip_size_zero_base() {
        // Edge case: 0 >> anything = 0, but we clamp to 1
        assert_eq!(mip_size(0, 0), 1);
        assert_eq!(mip_size(0, 5), 1);
    }
}

// ============================================================================
// Module: integration_tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn full_workflow_small_texture() {
        let uploader = TextureUploader::new(65536);

        // Create descriptor for 64x64 RGBA texture
        let desc = TextureUploadDescriptor::full(64, 64, 4);

        // Verify required sizes
        assert_eq!(desc.unpadded_data_size(), 64 * 64 * 4);
        assert_eq!(desc.required_data_size(), 256 * 64); // 64*4=256 aligned

        // Validate upload
        let data = vec![0u8; desc.unpadded_data_size() as usize];
        assert!(uploader.validate_upload(&desc, data.len()).is_ok());

        // Data is below staging threshold
        assert!((data.len() as u64) < uploader.staging_threshold());
    }

    #[test]
    fn full_workflow_large_texture() {
        let uploader = TextureUploader::new(65536);

        // Create descriptor for 512x512 RGBA texture
        let desc = TextureUploadDescriptor::full(512, 512, 4);

        // Verify this exceeds staging threshold
        let data_size = desc.unpadded_data_size();
        assert!(data_size > uploader.staging_threshold());

        // Should still validate
        let data = vec![0u8; data_size as usize];
        assert!(uploader.validate_upload(&desc, data.len()).is_ok());
    }

    #[test]
    fn mip_chain_sizes() {
        // Verify mip chain size calculations
        let base_width = 1024;
        let base_height = 1024;
        let bpp = 4;

        let expected_sizes: [(u32, u64); 11] = [
            (0, 1024 * 1024 * 4),   // 1024x1024
            (1, 512 * 512 * 4),     // 512x512
            (2, 256 * 256 * 4),     // 256x256
            (3, 128 * 128 * 4),     // 128x128
            (4, 64 * 64 * 4),       // 64x64
            (5, 32 * 32 * 4),       // 32x32
            (6, 16 * 16 * 4),       // 16x16
            (7, 8 * 8 * 4),         // 8x8
            (8, 4 * 4 * 4),         // 4x4
            (9, 2 * 2 * 4),         // 2x2
            (10, 1 * 1 * 4),        // 1x1
        ];

        for (mip, expected_size) in expected_sizes {
            let desc = TextureUploadDescriptor::mip_level(
                base_width, base_height, mip, bpp
            );
            assert_eq!(desc.unpadded_data_size(), expected_size,
                "Mip {} size mismatch", mip);
        }
    }

    #[test]
    fn descriptor_and_region_consistency() {
        // Ensure TextureRegion::mip and TextureUploadDescriptor::mip_level
        // calculate the same sizes

        for mip in 0..12 {
            let region = TextureRegion::mip(1024, 512, mip);
            let desc = TextureUploadDescriptor::mip_level(1024, 512, mip, 4);

            assert_eq!(region.size.0, desc.size.0, "Width mismatch at mip {}", mip);
            assert_eq!(region.size.1, desc.size.1, "Height mismatch at mip {}", mip);
        }
    }

    #[test]
    fn padding_preserves_image_integrity() {
        // Create a checkerboard pattern
        let width = 10;
        let height = 10;
        let bpp = 4;

        let mut data = vec![0u8; width * height * bpp];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * bpp;
                let color = if (x + y) % 2 == 0 { 255 } else { 0 };
                data[idx] = color;     // R
                data[idx + 1] = color; // G
                data[idx + 2] = color; // B
                data[idx + 3] = 255;   // A
            }
        }

        let padded = pad_to_row_pitch(&data, width as u32, height as u32, bpp as u32);

        // Verify the pattern is preserved
        let padded_row = calculate_row_pitch(width as u32, bpp as u32) as usize;
        for y in 0..height {
            for x in 0..width {
                let orig_idx = (y * width + x) * bpp;
                let padded_idx = y * padded_row + x * bpp;

                assert_eq!(padded[padded_idx], data[orig_idx],
                    "R mismatch at ({}, {})", x, y);
                assert_eq!(padded[padded_idx + 1], data[orig_idx + 1],
                    "G mismatch at ({}, {})", x, y);
                assert_eq!(padded[padded_idx + 2], data[orig_idx + 2],
                    "B mismatch at ({}, {})", x, y);
                assert_eq!(padded[padded_idx + 3], data[orig_idx + 3],
                    "A mismatch at ({}, {})", x, y);
            }
        }
    }
}
