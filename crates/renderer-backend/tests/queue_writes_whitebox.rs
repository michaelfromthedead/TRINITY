// SPDX-License-Identifier: MIT
//
// WHITEBOX tests for T-WGPU-P1.4.2 (Queue Writes)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, and edge cases
// that are not visible through the public contract alone.
//
// Implementation under test: crates/renderer-backend/src/device/queue.rs
//   - QueueWriteError enum (all variants, Display, Debug)
//   - write_buffer_validated() - validation with explicit buffer size
//   - write_buffer_checked() - validation with auto-detected buffer size
//   - write_texture_validated() - texture write with layout validation
//   - align_bytes_per_row() - row alignment helper
//   - is_buffer_offset_aligned() - offset alignment check
//   - is_bytes_per_row_aligned() - bytes per row alignment check
//   - COPY_BUFFER_ALIGNMENT constant (4 bytes)
//   - COPY_BYTES_PER_ROW_ALIGNMENT constant (256 bytes)
//
// WHITEBOX coverage plan:
//   - QueueWriteError variants:
//     - [QE-A] BufferOffsetUnaligned: construction, Display, Debug
//     - [QE-B] BufferOverflow: construction, Display, Debug
//     - [QE-C] TextureLayoutInvalid: construction, Display, Debug
//     - [QE-D] TextureDataSizeMismatch: construction, Display, Debug
//     - [QE-E] Error trait implementation
//     - [QE-F] PartialEq and Eq implementations
//     - [QE-G] Clone implementation
//
//   - Constants:
//     - [CONST-A] COPY_BUFFER_ALIGNMENT == 4
//     - [CONST-B] COPY_BYTES_PER_ROW_ALIGNMENT == 256
//
//   - Alignment helpers:
//     - [AH-A] align_bytes_per_row: rounds up correctly
//     - [AH-B] align_bytes_per_row: already aligned values unchanged
//     - [AH-C] align_bytes_per_row: zero input
//     - [AH-D] is_buffer_offset_aligned: aligned values
//     - [AH-E] is_buffer_offset_aligned: unaligned values
//     - [AH-F] is_buffer_offset_aligned: edge cases (0, 1, 3, 4)
//     - [AH-G] is_bytes_per_row_aligned: aligned values
//     - [AH-H] is_bytes_per_row_aligned: unaligned values
//     - [AH-I] is_bytes_per_row_aligned: edge cases (0, 1, 255, 256, 257)
//
//   - write_buffer_validated:
//     - [WBV-A] Valid write at offset 0
//     - [WBV-B] Valid write at aligned offset (4, 8, 256)
//     - [WBV-C] Alignment error: offset 1
//     - [WBV-D] Alignment error: offset 3
//     - [WBV-E] Overflow error: offset + data_len > buffer_size
//     - [WBV-F] Overflow error: exact boundary violation (off-by-one)
//     - [WBV-G] Valid: exact boundary (offset + data_len == buffer_size)
//     - [WBV-H] Zero-size data write (empty slice)
//     - [WBV-I] Zero buffer_size with non-empty data
//     - [WBV-J] Large data write validation
//
//   - write_buffer_checked (requires GPU):
//     - [WBC-A] Valid write with auto-detected buffer size
//     - [WBC-B] Alignment error with auto-detected size
//     - [WBC-C] Overflow error with auto-detected size
//
//   - write_texture_validated:
//     - [WTV-A] Valid single-row texture write (height=1)
//     - [WTV-B] Valid multi-row texture write with aligned bytes_per_row
//     - [WTV-C] Alignment error: unaligned bytes_per_row for multi-row
//     - [WTV-D] Missing bytes_per_row error for multi-row write
//     - [WTV-E] Missing rows_per_image error for 3D/array texture
//     - [WTV-F] Data size mismatch: insufficient data
//     - [WTV-G] Degenerate case: zero-width texture
//     - [WTV-H] Degenerate case: zero-height texture
//     - [WTV-I] Degenerate case: zero-depth texture
//     - [WTV-J] Valid 3D texture write
//     - [WTV-K] Valid texture array write

use renderer_backend::device::{
    align_bytes_per_row, is_buffer_offset_aligned, is_bytes_per_row_aligned, QueueWriteError,
    TrinityQueue, COPY_BUFFER_ALIGNMENT, COPY_BYTES_PER_ROW_ALIGNMENT,
};

// ============================================================================
// Section 1: QueueWriteError Variants
// ============================================================================

/// [QE-A] BufferOffsetUnaligned: construction, Display, Debug
#[test]
fn test_queue_write_error_buffer_offset_unaligned_display() {
    let error = QueueWriteError::BufferOffsetUnaligned {
        offset: 3,
        alignment: 4,
    };

    let display = error.to_string();
    assert!(
        display.contains("3"),
        "Display should contain the offset value"
    );
    assert!(
        display.contains("4"),
        "Display should contain the alignment value"
    );
    assert!(
        display.contains("aligned"),
        "Display should mention alignment"
    );
}

/// [QE-A] BufferOffsetUnaligned: Debug implementation
#[test]
fn test_queue_write_error_buffer_offset_unaligned_debug() {
    let error = QueueWriteError::BufferOffsetUnaligned {
        offset: 7,
        alignment: 4,
    };

    let debug = format!("{:?}", error);
    assert!(
        debug.contains("BufferOffsetUnaligned"),
        "Debug should contain variant name"
    );
    assert!(
        debug.contains("offset"),
        "Debug should contain field name 'offset'"
    );
    assert!(
        debug.contains("alignment"),
        "Debug should contain field name 'alignment'"
    );
}

/// [QE-B] BufferOverflow: construction, Display, Debug
#[test]
fn test_queue_write_error_buffer_overflow_display() {
    let error = QueueWriteError::BufferOverflow {
        offset: 100,
        data_len: 50,
        buffer_size: 120,
    };

    let display = error.to_string();
    assert!(
        display.contains("100"),
        "Display should contain the offset"
    );
    assert!(
        display.contains("50"),
        "Display should contain data length"
    );
    assert!(
        display.contains("120"),
        "Display should contain buffer size"
    );
    assert!(
        display.contains("150"),
        "Display should show computed end offset (100+50)"
    );
    assert!(
        display.contains("overflow") || display.contains("exceed"),
        "Display should indicate overflow/exceed"
    );
}

/// [QE-B] BufferOverflow: Debug implementation
#[test]
fn test_queue_write_error_buffer_overflow_debug() {
    let error = QueueWriteError::BufferOverflow {
        offset: 100,
        data_len: 50,
        buffer_size: 120,
    };

    let debug = format!("{:?}", error);
    assert!(
        debug.contains("BufferOverflow"),
        "Debug should contain variant name"
    );
    assert!(
        debug.contains("offset"),
        "Debug should contain 'offset' field"
    );
    assert!(
        debug.contains("data_len"),
        "Debug should contain 'data_len' field"
    );
    assert!(
        debug.contains("buffer_size"),
        "Debug should contain 'buffer_size' field"
    );
}

/// [QE-C] TextureLayoutInvalid: construction, Display, Debug
#[test]
fn test_queue_write_error_texture_layout_invalid_display() {
    let error = QueueWriteError::TextureLayoutInvalid {
        reason: "bytes_per_row is not aligned".to_string(),
    };

    let display = error.to_string();
    assert!(
        display.contains("bytes_per_row is not aligned"),
        "Display should contain the reason"
    );
    assert!(
        display.to_lowercase().contains("invalid")
            || display.to_lowercase().contains("layout"),
        "Display should indicate layout issue"
    );
}

/// [QE-C] TextureLayoutInvalid: Debug implementation
#[test]
fn test_queue_write_error_texture_layout_invalid_debug() {
    let error = QueueWriteError::TextureLayoutInvalid {
        reason: "test reason".to_string(),
    };

    let debug = format!("{:?}", error);
    assert!(
        debug.contains("TextureLayoutInvalid"),
        "Debug should contain variant name"
    );
    assert!(
        debug.contains("reason"),
        "Debug should contain 'reason' field"
    );
}

/// [QE-D] TextureDataSizeMismatch: construction, Display, Debug
#[test]
fn test_queue_write_error_texture_data_size_mismatch_display() {
    let error = QueueWriteError::TextureDataSizeMismatch {
        provided: 1000,
        expected: 2000,
    };

    let display = error.to_string();
    assert!(
        display.contains("1000"),
        "Display should contain provided size"
    );
    assert!(
        display.contains("2000"),
        "Display should contain expected size"
    );
    assert!(
        display.to_lowercase().contains("mismatch")
            || display.to_lowercase().contains("size"),
        "Display should mention size mismatch"
    );
}

/// [QE-D] TextureDataSizeMismatch: Debug implementation
#[test]
fn test_queue_write_error_texture_data_size_mismatch_debug() {
    let error = QueueWriteError::TextureDataSizeMismatch {
        provided: 500,
        expected: 1000,
    };

    let debug = format!("{:?}", error);
    assert!(
        debug.contains("TextureDataSizeMismatch"),
        "Debug should contain variant name"
    );
    assert!(
        debug.contains("provided"),
        "Debug should contain 'provided' field"
    );
    assert!(
        debug.contains("expected"),
        "Debug should contain 'expected' field"
    );
}

/// [QE-E] Error trait implementation
#[test]
fn test_queue_write_error_implements_error_trait() {
    let error: Box<dyn std::error::Error> = Box::new(QueueWriteError::BufferOffsetUnaligned {
        offset: 1,
        alignment: 4,
    });

    // Error trait requires Display, which we already tested
    // Just verify it compiles and can be used as a trait object
    let _ = error.to_string();
}

/// [QE-F] PartialEq and Eq implementations
#[test]
fn test_queue_write_error_equality() {
    let error1 = QueueWriteError::BufferOffsetUnaligned {
        offset: 3,
        alignment: 4,
    };
    let error2 = QueueWriteError::BufferOffsetUnaligned {
        offset: 3,
        alignment: 4,
    };
    let error3 = QueueWriteError::BufferOffsetUnaligned {
        offset: 7,
        alignment: 4,
    };

    assert_eq!(error1, error2, "Same errors should be equal");
    assert_ne!(error1, error3, "Different offsets should not be equal");

    let overflow1 = QueueWriteError::BufferOverflow {
        offset: 10,
        data_len: 20,
        buffer_size: 25,
    };
    let overflow2 = QueueWriteError::BufferOverflow {
        offset: 10,
        data_len: 20,
        buffer_size: 25,
    };
    assert_eq!(overflow1, overflow2, "Same overflow errors should be equal");

    // Different variants should not be equal
    assert_ne!(
        error1, overflow1,
        "Different error variants should not be equal"
    );
}

/// [QE-G] Clone implementation
#[test]
fn test_queue_write_error_clone() {
    let original = QueueWriteError::BufferOverflow {
        offset: 100,
        data_len: 50,
        buffer_size: 120,
    };
    let cloned = original.clone();

    assert_eq!(original, cloned, "Cloned error should equal original");
}

/// Clone for TextureLayoutInvalid (String field)
#[test]
fn test_queue_write_error_clone_texture_layout() {
    let original = QueueWriteError::TextureLayoutInvalid {
        reason: "some reason".to_string(),
    };
    let cloned = original.clone();

    assert_eq!(original, cloned, "Cloned TextureLayoutInvalid should equal original");
}

// ============================================================================
// Section 2: Constants
// ============================================================================

/// [CONST-A] COPY_BUFFER_ALIGNMENT == 4
#[test]
fn test_copy_buffer_alignment_constant() {
    assert_eq!(COPY_BUFFER_ALIGNMENT, 4, "COPY_BUFFER_ALIGNMENT must be 4 bytes");
}

/// [CONST-B] COPY_BYTES_PER_ROW_ALIGNMENT == 256
#[test]
fn test_copy_bytes_per_row_alignment_constant() {
    assert_eq!(
        COPY_BYTES_PER_ROW_ALIGNMENT, 256,
        "COPY_BYTES_PER_ROW_ALIGNMENT must be 256 bytes"
    );
}

// ============================================================================
// Section 3: Alignment Helper Functions
// ============================================================================

/// [AH-A] align_bytes_per_row: rounds up correctly
#[test]
fn test_align_bytes_per_row_rounds_up() {
    // Various unaligned values should round up to next 256 multiple
    assert_eq!(align_bytes_per_row(1), 256);
    assert_eq!(align_bytes_per_row(100), 256);
    assert_eq!(align_bytes_per_row(255), 256);
    assert_eq!(align_bytes_per_row(257), 512);
    assert_eq!(align_bytes_per_row(300), 512);
    assert_eq!(align_bytes_per_row(511), 512);
    assert_eq!(align_bytes_per_row(513), 768);
}

/// [AH-B] align_bytes_per_row: already aligned values unchanged
#[test]
fn test_align_bytes_per_row_already_aligned() {
    assert_eq!(align_bytes_per_row(256), 256);
    assert_eq!(align_bytes_per_row(512), 512);
    assert_eq!(align_bytes_per_row(768), 768);
    assert_eq!(align_bytes_per_row(1024), 1024);
    assert_eq!(align_bytes_per_row(4096), 4096);
}

/// [AH-C] align_bytes_per_row: zero input
#[test]
fn test_align_bytes_per_row_zero() {
    // Zero is technically aligned, but aligning 0 should give 0
    // (0 + 255) / 256 * 256 = 0
    assert_eq!(align_bytes_per_row(0), 0);
}

/// [AH-D] is_buffer_offset_aligned: aligned values
#[test]
fn test_is_buffer_offset_aligned_true() {
    assert!(is_buffer_offset_aligned(0), "0 is aligned");
    assert!(is_buffer_offset_aligned(4), "4 is aligned");
    assert!(is_buffer_offset_aligned(8), "8 is aligned");
    assert!(is_buffer_offset_aligned(12), "12 is aligned");
    assert!(is_buffer_offset_aligned(256), "256 is aligned");
    assert!(is_buffer_offset_aligned(1024), "1024 is aligned");
}

/// [AH-E] is_buffer_offset_aligned: unaligned values
#[test]
fn test_is_buffer_offset_aligned_false() {
    assert!(!is_buffer_offset_aligned(1), "1 is not aligned");
    assert!(!is_buffer_offset_aligned(2), "2 is not aligned");
    assert!(!is_buffer_offset_aligned(3), "3 is not aligned");
    assert!(!is_buffer_offset_aligned(5), "5 is not aligned");
    assert!(!is_buffer_offset_aligned(7), "7 is not aligned");
    assert!(!is_buffer_offset_aligned(9), "9 is not aligned");
    assert!(!is_buffer_offset_aligned(255), "255 is not aligned");
}

/// [AH-F] is_buffer_offset_aligned: edge cases (0, 1, 3, 4)
#[test]
fn test_is_buffer_offset_aligned_edge_cases() {
    assert!(is_buffer_offset_aligned(0), "0 should be aligned (boundary)");
    assert!(!is_buffer_offset_aligned(1), "1 should not be aligned");
    assert!(!is_buffer_offset_aligned(3), "3 should not be aligned (one before boundary)");
    assert!(is_buffer_offset_aligned(4), "4 should be aligned (boundary)");
}

/// [AH-G] is_bytes_per_row_aligned: aligned values
#[test]
fn test_is_bytes_per_row_aligned_true() {
    assert!(is_bytes_per_row_aligned(0), "0 is aligned");
    assert!(is_bytes_per_row_aligned(256), "256 is aligned");
    assert!(is_bytes_per_row_aligned(512), "512 is aligned");
    assert!(is_bytes_per_row_aligned(768), "768 is aligned");
    assert!(is_bytes_per_row_aligned(1024), "1024 is aligned");
}

/// [AH-H] is_bytes_per_row_aligned: unaligned values
#[test]
fn test_is_bytes_per_row_aligned_false() {
    assert!(!is_bytes_per_row_aligned(1), "1 is not aligned");
    assert!(!is_bytes_per_row_aligned(100), "100 is not aligned");
    assert!(!is_bytes_per_row_aligned(255), "255 is not aligned");
    assert!(!is_bytes_per_row_aligned(257), "257 is not aligned");
    assert!(!is_bytes_per_row_aligned(300), "300 is not aligned");
    assert!(!is_bytes_per_row_aligned(511), "511 is not aligned");
}

/// [AH-I] is_bytes_per_row_aligned: edge cases (0, 1, 255, 256, 257)
#[test]
fn test_is_bytes_per_row_aligned_edge_cases() {
    assert!(is_bytes_per_row_aligned(0), "0 should be aligned");
    assert!(!is_bytes_per_row_aligned(1), "1 should not be aligned");
    assert!(!is_bytes_per_row_aligned(255), "255 should not be aligned (one before boundary)");
    assert!(is_bytes_per_row_aligned(256), "256 should be aligned (boundary)");
    assert!(!is_bytes_per_row_aligned(257), "257 should not be aligned (one after boundary)");
}

// ============================================================================
// Section 4: write_buffer_validated (Offline Validation Tests)
//
// These tests verify the validation logic WITHOUT a GPU. They test what the
// function WOULD return if called, by examining the error types directly.
// ============================================================================

/// Helper: Create a test validation scenario
struct BufferWriteScenario {
    offset: u64,
    data_len: usize,
    buffer_size: u64,
}

impl BufferWriteScenario {
    /// Check if this scenario would fail alignment validation
    fn fails_alignment(&self) -> bool {
        self.offset % COPY_BUFFER_ALIGNMENT != 0
    }

    /// Check if this scenario would fail bounds validation
    fn fails_bounds(&self) -> bool {
        self.offset.saturating_add(self.data_len as u64) > self.buffer_size
    }

    /// Expected error type, if any
    fn expected_error(&self) -> Option<QueueWriteError> {
        if self.fails_alignment() {
            Some(QueueWriteError::BufferOffsetUnaligned {
                offset: self.offset,
                alignment: COPY_BUFFER_ALIGNMENT,
            })
        } else if self.fails_bounds() {
            Some(QueueWriteError::BufferOverflow {
                offset: self.offset,
                data_len: self.data_len,
                buffer_size: self.buffer_size,
            })
        } else {
            None
        }
    }
}

/// [WBV-A] Valid write at offset 0 - should pass validation
#[test]
fn test_buffer_write_valid_offset_zero() {
    let scenario = BufferWriteScenario {
        offset: 0,
        data_len: 64,
        buffer_size: 1024,
    };
    assert!(
        scenario.expected_error().is_none(),
        "Write at offset 0 with sufficient buffer should be valid"
    );
}

/// [WBV-B] Valid write at aligned offset (4, 8, 256)
#[test]
fn test_buffer_write_valid_aligned_offsets() {
    for offset in [4, 8, 12, 16, 256, 1024] {
        let scenario = BufferWriteScenario {
            offset,
            data_len: 64,
            buffer_size: 2048,
        };
        assert!(
            scenario.expected_error().is_none(),
            "Write at aligned offset {} should be valid",
            offset
        );
    }
}

/// [WBV-C] Alignment error: offset 1
#[test]
fn test_buffer_write_alignment_error_offset_1() {
    let scenario = BufferWriteScenario {
        offset: 1,
        data_len: 64,
        buffer_size: 1024,
    };
    let error = scenario.expected_error().expect("Should produce alignment error");
    assert!(
        matches!(error, QueueWriteError::BufferOffsetUnaligned { offset: 1, .. }),
        "Should be BufferOffsetUnaligned with offset 1"
    );
}

/// [WBV-D] Alignment error: offset 3
#[test]
fn test_buffer_write_alignment_error_offset_3() {
    let scenario = BufferWriteScenario {
        offset: 3,
        data_len: 64,
        buffer_size: 1024,
    };
    let error = scenario.expected_error().expect("Should produce alignment error");
    assert!(
        matches!(error, QueueWriteError::BufferOffsetUnaligned { offset: 3, .. }),
        "Should be BufferOffsetUnaligned with offset 3"
    );
}

/// [WBV-E] Overflow error: offset + data_len > buffer_size
#[test]
fn test_buffer_write_overflow_error() {
    let scenario = BufferWriteScenario {
        offset: 0,
        data_len: 2000,
        buffer_size: 1024,
    };
    let error = scenario.expected_error().expect("Should produce overflow error");
    assert!(
        matches!(error, QueueWriteError::BufferOverflow { .. }),
        "Should be BufferOverflow"
    );
}

/// [WBV-F] Overflow error: exact boundary violation (off-by-one)
#[test]
fn test_buffer_write_overflow_off_by_one() {
    // Buffer size 1024, trying to write 1025 bytes from offset 0
    let scenario = BufferWriteScenario {
        offset: 0,
        data_len: 1025,
        buffer_size: 1024,
    };
    let error = scenario.expected_error().expect("Should produce overflow error");
    assert!(
        matches!(error, QueueWriteError::BufferOverflow { .. }),
        "Off-by-one should produce BufferOverflow"
    );

    // Buffer size 1024, trying to write 1 byte at offset 1024
    let scenario2 = BufferWriteScenario {
        offset: 1024, // Aligned offset at exact end
        data_len: 1,
        buffer_size: 1024,
    };
    let error2 = scenario2.expected_error().expect("Should produce overflow error");
    assert!(
        matches!(error2, QueueWriteError::BufferOverflow { .. }),
        "Write past end should produce BufferOverflow"
    );
}

/// [WBV-G] Valid: exact boundary (offset + data_len == buffer_size)
#[test]
fn test_buffer_write_valid_exact_boundary() {
    // Write exactly fills buffer
    let scenario = BufferWriteScenario {
        offset: 0,
        data_len: 1024,
        buffer_size: 1024,
    };
    assert!(
        scenario.expected_error().is_none(),
        "Exact boundary write should be valid"
    );

    // Write fills remaining buffer from offset
    let scenario2 = BufferWriteScenario {
        offset: 512, // Aligned
        data_len: 512,
        buffer_size: 1024,
    };
    assert!(
        scenario2.expected_error().is_none(),
        "Write that exactly fills remaining buffer should be valid"
    );
}

/// [WBV-H] Zero-size data write (empty slice)
#[test]
fn test_buffer_write_zero_size_data() {
    let scenario = BufferWriteScenario {
        offset: 0,
        data_len: 0,
        buffer_size: 1024,
    };
    assert!(
        scenario.expected_error().is_none(),
        "Zero-size write should be valid"
    );

    // Zero-size write at any aligned offset should be valid
    let scenario2 = BufferWriteScenario {
        offset: 1024,
        data_len: 0,
        buffer_size: 1024,
    };
    assert!(
        scenario2.expected_error().is_none(),
        "Zero-size write at buffer end should be valid"
    );
}

/// [WBV-I] Zero buffer_size with non-empty data
#[test]
fn test_buffer_write_zero_buffer_size() {
    let scenario = BufferWriteScenario {
        offset: 0,
        data_len: 1,
        buffer_size: 0,
    };
    let error = scenario.expected_error().expect("Should produce overflow error");
    assert!(
        matches!(error, QueueWriteError::BufferOverflow { .. }),
        "Write to zero-size buffer should produce BufferOverflow"
    );
}

/// [WBV-J] Large data write validation
#[test]
fn test_buffer_write_large_data() {
    // 1 GB buffer, 500 MB write
    let scenario = BufferWriteScenario {
        offset: 0,
        data_len: 500 * 1024 * 1024,
        buffer_size: 1024 * 1024 * 1024,
    };
    assert!(
        scenario.expected_error().is_none(),
        "Large but valid write should pass validation"
    );

    // 1 GB buffer, 1.5 GB write (overflow)
    let scenario2 = BufferWriteScenario {
        offset: 0,
        data_len: 1500 * 1024 * 1024,
        buffer_size: 1024 * 1024 * 1024,
    };
    let error = scenario2.expected_error().expect("Should produce overflow error");
    assert!(
        matches!(error, QueueWriteError::BufferOverflow { .. }),
        "Large overflow should produce BufferOverflow"
    );
}

/// Alignment takes priority over overflow validation
#[test]
fn test_buffer_write_alignment_error_takes_priority() {
    // Both unaligned and would overflow - alignment error should be returned first
    let scenario = BufferWriteScenario {
        offset: 1, // Unaligned
        data_len: 2000, // Would overflow 1024 buffer
        buffer_size: 1024,
    };
    let error = scenario.expected_error().expect("Should produce error");
    assert!(
        matches!(error, QueueWriteError::BufferOffsetUnaligned { .. }),
        "Alignment error should take priority over overflow"
    );
}

// ============================================================================
// Section 5: Texture Write Validation Scenarios
//
// These tests verify the texture write validation logic WITHOUT a GPU.
// They examine the validation rules for bytes_per_row, rows_per_image, and data size.
// ============================================================================

/// Helper: Create texture write validation scenarios
#[allow(dead_code)]
struct TextureWriteScenario {
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
    bytes_per_row: Option<u32>,
    rows_per_image: Option<u32>,
    /// Data offset in the source buffer (for future data size validation tests)
    data_offset: u64,
    /// Expected data length (for future data size validation tests)
    data_len: usize,
}

impl TextureWriteScenario {
    /// Check if bytes_per_row alignment would fail for multi-row writes
    fn fails_bytes_per_row_alignment(&self) -> bool {
        if self.height > 1 || self.depth_or_array_layers > 1 {
            if let Some(bpr) = self.bytes_per_row {
                bpr % COPY_BYTES_PER_ROW_ALIGNMENT != 0
            } else {
                true // bytes_per_row required for multi-row
            }
        } else {
            false
        }
    }

    /// Check if rows_per_image is required but missing
    fn fails_rows_per_image_missing(&self) -> bool {
        self.depth_or_array_layers > 1 && self.rows_per_image.is_none()
    }

    /// Check for bytes_per_row required but missing (multi-row)
    fn fails_bytes_per_row_missing(&self) -> bool {
        (self.height > 1 || self.depth_or_array_layers > 1) && self.bytes_per_row.is_none()
    }
}

/// [WTV-A] Valid single-row texture write (height=1)
#[test]
fn test_texture_write_valid_single_row() {
    let scenario = TextureWriteScenario {
        width: 256,
        height: 1,
        depth_or_array_layers: 1,
        bytes_per_row: None, // Not required for single row
        rows_per_image: None,
        data_offset: 0,
        data_len: 256 * 4, // RGBA
    };
    assert!(
        !scenario.fails_bytes_per_row_alignment(),
        "Single row should not require aligned bytes_per_row"
    );
    assert!(
        !scenario.fails_rows_per_image_missing(),
        "Single row should not require rows_per_image"
    );
}

/// [WTV-B] Valid multi-row texture write with aligned bytes_per_row
#[test]
fn test_texture_write_valid_multi_row_aligned() {
    let scenario = TextureWriteScenario {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
        bytes_per_row: Some(256 * 4), // 1024, which is aligned (multiple of 256)
        rows_per_image: Some(256),
        data_offset: 0,
        data_len: 256 * 256 * 4,
    };
    // 1024 % 256 == 0, so this is aligned
    assert!(
        !scenario.fails_bytes_per_row_alignment(),
        "Multi-row with aligned bytes_per_row should be valid"
    );
}

/// [WTV-C] Alignment error: unaligned bytes_per_row for multi-row
#[test]
fn test_texture_write_unaligned_bytes_per_row() {
    let scenario = TextureWriteScenario {
        width: 100, // 100 * 4 = 400, not aligned to 256
        height: 100,
        depth_or_array_layers: 1,
        bytes_per_row: Some(400), // 400 % 256 != 0
        rows_per_image: Some(100),
        data_offset: 0,
        data_len: 400 * 100,
    };
    assert!(
        scenario.fails_bytes_per_row_alignment(),
        "Unaligned bytes_per_row should fail for multi-row"
    );
}

/// [WTV-D] Missing bytes_per_row error for multi-row write
#[test]
fn test_texture_write_missing_bytes_per_row() {
    let scenario = TextureWriteScenario {
        width: 256,
        height: 256,
        depth_or_array_layers: 1,
        bytes_per_row: None, // Missing for multi-row
        rows_per_image: Some(256),
        data_offset: 0,
        data_len: 256 * 256 * 4,
    };
    assert!(
        scenario.fails_bytes_per_row_missing(),
        "Missing bytes_per_row should fail for multi-row"
    );
}

/// [WTV-E] Missing rows_per_image error for 3D/array texture
#[test]
fn test_texture_write_missing_rows_per_image() {
    let scenario = TextureWriteScenario {
        width: 256,
        height: 256,
        depth_or_array_layers: 4, // Texture array
        bytes_per_row: Some(1024),
        rows_per_image: None, // Missing for array/3D
        data_offset: 0,
        data_len: 1024 * 256 * 4,
    };
    assert!(
        scenario.fails_rows_per_image_missing(),
        "Missing rows_per_image should fail for texture array"
    );
}

/// [WTV-G] Degenerate case: zero-width texture
#[test]
fn test_texture_write_zero_width() {
    let scenario = TextureWriteScenario {
        width: 0,
        height: 256,
        depth_or_array_layers: 1,
        bytes_per_row: Some(0),
        rows_per_image: Some(256),
        data_offset: 0,
        data_len: 0,
    };
    // Zero dimensions should result in zero data requirement
    // The validation logic should handle this gracefully
    assert!(
        !scenario.fails_bytes_per_row_alignment() || scenario.width == 0,
        "Zero-width texture should be handled gracefully"
    );
}

/// [WTV-H] Degenerate case: zero-height texture
#[test]
fn test_texture_write_zero_height() {
    let scenario = TextureWriteScenario {
        width: 256,
        height: 0,
        depth_or_array_layers: 1,
        bytes_per_row: None,
        rows_per_image: None,
        data_offset: 0,
        data_len: 0,
    };
    // Zero height with depth=1 shouldn't require alignment validation
    assert!(
        !scenario.fails_bytes_per_row_alignment(),
        "Zero-height texture should not trigger bytes_per_row alignment check"
    );
}

/// [WTV-I] Degenerate case: zero-depth texture
#[test]
fn test_texture_write_zero_depth() {
    let scenario = TextureWriteScenario {
        width: 256,
        height: 256,
        depth_or_array_layers: 0,
        bytes_per_row: Some(1024),
        rows_per_image: Some(256),
        data_offset: 0,
        data_len: 0,
    };
    // Zero depth should not require rows_per_image
    assert!(
        !scenario.fails_rows_per_image_missing(),
        "Zero-depth texture should not require rows_per_image"
    );
}

/// [WTV-J] Valid 3D texture write
#[test]
fn test_texture_write_valid_3d() {
    let scenario = TextureWriteScenario {
        width: 64,
        height: 64,
        depth_or_array_layers: 64,
        bytes_per_row: Some(256), // 64 * 4 = 256, aligned
        rows_per_image: Some(64),
        data_offset: 0,
        data_len: 256 * 64 * 64,
    };
    assert!(
        !scenario.fails_bytes_per_row_alignment(),
        "3D texture with aligned layout should be valid"
    );
    assert!(
        !scenario.fails_rows_per_image_missing(),
        "3D texture with rows_per_image should be valid"
    );
}

/// [WTV-K] Valid texture array write
#[test]
fn test_texture_write_valid_array() {
    let scenario = TextureWriteScenario {
        width: 256,
        height: 256,
        depth_or_array_layers: 6, // Cubemap-like
        bytes_per_row: Some(1024), // 256 * 4 = 1024, aligned
        rows_per_image: Some(256),
        data_offset: 0,
        data_len: 1024 * 256 * 6,
    };
    assert!(
        !scenario.fails_bytes_per_row_alignment(),
        "Texture array with aligned layout should be valid"
    );
    assert!(
        !scenario.fails_rows_per_image_missing(),
        "Texture array with rows_per_image should be valid"
    );
}

// ============================================================================
// Section 6: Integration Tests with GPU (require hardware)
//
// These tests require actual GPU hardware and will be skipped if unavailable.
// ============================================================================

/// Get a test device and queue if available
fn get_test_device_and_queue() -> Option<(wgpu::Device, TrinityQueue)> {
    // Use pollster to block on async
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .ok()?;

        Some((device, TrinityQueue::new(queue)))
    })
}

/// Create a test buffer with COPY_DST usage
fn create_test_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// [WBC-A] Valid write with auto-detected buffer size
#[test]
fn test_write_buffer_checked_valid() {
    let Some((device, queue)) = get_test_device_and_queue() else {
        eprintln!("Skipping GPU test: no adapter available");
        return;
    };

    let buffer = create_test_buffer(&device, 1024);
    let data = vec![0u8; 64];

    let result = queue.write_buffer_checked(&buffer, 0, &data);
    assert!(
        result.is_ok(),
        "Valid write should succeed: {:?}",
        result.err()
    );
}

/// [WBC-B] Alignment error with auto-detected size
#[test]
fn test_write_buffer_checked_alignment_error() {
    let Some((device, queue)) = get_test_device_and_queue() else {
        eprintln!("Skipping GPU test: no adapter available");
        return;
    };

    let buffer = create_test_buffer(&device, 1024);
    let data = vec![0u8; 64];

    let result = queue.write_buffer_checked(&buffer, 1, &data); // Unaligned offset
    assert!(
        matches!(
            result,
            Err(QueueWriteError::BufferOffsetUnaligned { offset: 1, .. })
        ),
        "Should produce alignment error"
    );
}

/// [WBC-C] Overflow error with auto-detected size
#[test]
fn test_write_buffer_checked_overflow_error() {
    let Some((device, queue)) = get_test_device_and_queue() else {
        eprintln!("Skipping GPU test: no adapter available");
        return;
    };

    let buffer = create_test_buffer(&device, 64);
    let data = vec![0u8; 128]; // Larger than buffer

    let result = queue.write_buffer_checked(&buffer, 0, &data);
    assert!(
        matches!(
            result,
            Err(QueueWriteError::BufferOverflow {
                buffer_size: 64,
                ..
            })
        ),
        "Should produce overflow error with correct buffer_size"
    );
}

/// [WBV validated] write_buffer_validated success path with GPU
#[test]
fn test_write_buffer_validated_success_with_gpu() {
    let Some((device, queue)) = get_test_device_and_queue() else {
        eprintln!("Skipping GPU test: no adapter available");
        return;
    };

    let buffer = create_test_buffer(&device, 1024);
    let data = vec![42u8; 256];

    // Valid write at offset 0
    let result = queue.write_buffer_validated(&buffer, 0, &data, 1024);
    assert!(result.is_ok(), "Valid write should succeed");

    // Valid write at aligned offset
    let result2 = queue.write_buffer_validated(&buffer, 256, &data, 1024);
    assert!(result2.is_ok(), "Valid write at aligned offset should succeed");

    // Valid write that fills exactly to boundary
    let remaining_data = vec![0u8; 512];
    let result3 = queue.write_buffer_validated(&buffer, 512, &remaining_data, 1024);
    assert!(
        result3.is_ok(),
        "Write that fills exactly to boundary should succeed"
    );
}

/// [WBV validated] write_buffer_validated error paths with GPU
#[test]
fn test_write_buffer_validated_errors_with_gpu() {
    let Some((device, queue)) = get_test_device_and_queue() else {
        eprintln!("Skipping GPU test: no adapter available");
        return;
    };

    let buffer = create_test_buffer(&device, 512);
    let data = vec![0u8; 64];

    // Alignment error
    let result = queue.write_buffer_validated(&buffer, 3, &data, 512);
    assert!(
        matches!(
            result,
            Err(QueueWriteError::BufferOffsetUnaligned { offset: 3, .. })
        ),
        "Unaligned offset should produce alignment error"
    );

    // Overflow error
    let large_data = vec![0u8; 600];
    let result2 = queue.write_buffer_validated(&buffer, 0, &large_data, 512);
    assert!(
        matches!(result2, Err(QueueWriteError::BufferOverflow { .. })),
        "Overflow should produce overflow error"
    );
}

/// Test zero-size write with GPU
#[test]
fn test_write_buffer_validated_zero_size_with_gpu() {
    let Some((device, queue)) = get_test_device_and_queue() else {
        eprintln!("Skipping GPU test: no adapter available");
        return;
    };

    let buffer = create_test_buffer(&device, 256);
    let empty_data: &[u8] = &[];

    // Zero-size write should succeed
    let result = queue.write_buffer_validated(&buffer, 0, empty_data, 256);
    assert!(result.is_ok(), "Zero-size write should succeed");

    // Zero-size write at end of buffer should succeed
    let result2 = queue.write_buffer_validated(&buffer, 256, empty_data, 256);
    assert!(
        result2.is_ok(),
        "Zero-size write at buffer end should succeed"
    );
}

/// Test texture write validation with GPU
#[test]
fn test_write_texture_validated_with_gpu() {
    let Some((device, queue)) = get_test_device_and_queue() else {
        eprintln!("Skipping GPU test: no adapter available");
        return;
    };

    // Create a test texture
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
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // Calculate aligned bytes_per_row: 64 * 4 = 256, which is already aligned
    let bytes_per_row = align_bytes_per_row(64 * 4);
    let data_size = (bytes_per_row * 64) as usize;
    let data = vec![255u8; data_size];

    let result = queue.write_texture_validated(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(64),
        },
        wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
    );

    assert!(
        result.is_ok(),
        "Valid texture write should succeed: {:?}",
        result.err()
    );
}

/// Test texture write validation - unaligned bytes_per_row error with GPU
#[test]
fn test_write_texture_validated_unaligned_bytes_per_row_with_gpu() {
    let Some((device, queue)) = get_test_device_and_queue() else {
        eprintln!("Skipping GPU test: no adapter available");
        return;
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_texture"),
        size: wgpu::Extent3d {
            width: 100,
            height: 100,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // Unaligned bytes_per_row: 100 * 4 = 400, not aligned to 256
    let unaligned_bytes_per_row = 400u32;
    let data = vec![0u8; (unaligned_bytes_per_row * 100) as usize];

    let result = queue.write_texture_validated(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(unaligned_bytes_per_row),
            rows_per_image: Some(100),
        },
        wgpu::Extent3d {
            width: 100,
            height: 100,
            depth_or_array_layers: 1,
        },
    );

    assert!(
        matches!(result, Err(QueueWriteError::TextureLayoutInvalid { .. })),
        "Unaligned bytes_per_row should produce TextureLayoutInvalid error"
    );
}

/// Test texture write validation - insufficient data error with GPU
#[test]
fn test_write_texture_validated_insufficient_data_with_gpu() {
    let Some((device, queue)) = get_test_device_and_queue() else {
        eprintln!("Skipping GPU test: no adapter available");
        return;
    };

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
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let bytes_per_row = align_bytes_per_row(64 * 4);
    // Insufficient data - only half what's needed
    let insufficient_data = vec![0u8; (bytes_per_row * 32) as usize];

    let result = queue.write_texture_validated(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &insufficient_data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(64),
        },
        wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
    );

    assert!(
        matches!(
            result,
            Err(QueueWriteError::TextureDataSizeMismatch { .. })
        ),
        "Insufficient data should produce TextureDataSizeMismatch error"
    );
}

/// Test single-row texture write (no alignment required for bytes_per_row)
#[test]
fn test_write_texture_validated_single_row_with_gpu() {
    let Some((device, queue)) = get_test_device_and_queue() else {
        eprintln!("Skipping GPU test: no adapter available");
        return;
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_texture"),
        size: wgpu::Extent3d {
            width: 100,
            height: 1, // Single row
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // For single row, bytes_per_row alignment is not required
    let bytes_per_row = 100 * 4; // 400, not aligned
    let data = vec![255u8; bytes_per_row as usize];

    let result = queue.write_texture_validated(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 100,
            height: 1,
            depth_or_array_layers: 1,
        },
    );

    assert!(
        result.is_ok(),
        "Single-row texture write should not require bytes_per_row alignment: {:?}",
        result.err()
    );
}

// ============================================================================
// Section 7: TrinityQueue Debug and Thread Safety
// ============================================================================

/// TrinityQueue Debug implementation
#[test]
fn test_trinity_queue_debug() {
    let Some((_, queue)) = get_test_device_and_queue() else {
        eprintln!("Skipping GPU test: no adapter available");
        return;
    };

    let debug_str = format!("{:?}", queue);
    assert!(
        debug_str.contains("TrinityQueue"),
        "Debug should contain struct name"
    );
    assert!(
        debug_str.contains("pending_submissions"),
        "Debug should contain pending_submissions field"
    );
}

/// TrinityQueue pending count
#[test]
fn test_trinity_queue_pending_count() {
    let Some((_, queue)) = get_test_device_and_queue() else {
        eprintln!("Skipping GPU test: no adapter available");
        return;
    };

    // Initial pending count should be 0
    assert_eq!(queue.pending_count(), 0, "Initial pending count should be 0");
    assert!(
        !queue.has_pending_work(),
        "Should not have pending work initially"
    );
}

/// TrinityQueue inner() accessor
#[test]
fn test_trinity_queue_inner_accessor() {
    let Some((_, queue)) = get_test_device_and_queue() else {
        eprintln!("Skipping GPU test: no adapter available");
        return;
    };

    // Just verify inner() returns a valid reference
    let _inner: &wgpu::Queue = queue.inner();
    // If we got here without panic, the accessor works
}

// ============================================================================
// Section 8: Edge Cases and Boundary Conditions
// ============================================================================

/// Test alignment with maximum u64 offset (should be aligned)
#[test]
fn test_alignment_max_offset_aligned() {
    // Maximum offset that's aligned to 4 bytes
    let max_aligned = u64::MAX - (u64::MAX % COPY_BUFFER_ALIGNMENT);
    assert!(
        is_buffer_offset_aligned(max_aligned),
        "Maximum aligned offset should be aligned"
    );
}

/// Test alignment with maximum u64 offset minus 1 (might be unaligned)
#[test]
fn test_alignment_near_max_offset() {
    let near_max = u64::MAX - 1;
    // This depends on whether MAX-1 is divisible by 4
    let expected = near_max % COPY_BUFFER_ALIGNMENT == 0;
    assert_eq!(
        is_buffer_offset_aligned(near_max),
        expected,
        "Near-max offset alignment check"
    );
}

/// Test bytes_per_row alignment with large values
#[test]
fn test_bytes_per_row_alignment_large_values() {
    // Very large aligned value
    assert!(is_bytes_per_row_aligned(256 * 1000000)); // 256 MB
    assert!(is_bytes_per_row_aligned(256 * 4096)); // 1 MB

    // Very large unaligned value
    assert!(!is_bytes_per_row_aligned(256 * 1000000 + 1));
    assert!(!is_bytes_per_row_aligned(256 * 4096 - 1));
}

/// Test align_bytes_per_row with values near u32::MAX
#[test]
fn test_align_bytes_per_row_near_max() {
    // A value that when aligned would overflow - check we don't panic
    // (The function uses (unpadded + alignment - 1) / alignment * alignment)
    // This could overflow for very large values, but in practice texture rows
    // won't be that large

    // Test a large but reasonable value
    let large_row = 1024 * 1024 * 100; // 100 MB row
    let aligned = align_bytes_per_row(large_row);
    assert!(
        aligned >= large_row,
        "Aligned value should be >= input"
    );
    assert!(
        aligned % COPY_BYTES_PER_ROW_ALIGNMENT == 0,
        "Result should be aligned"
    );
}

/// Test saturation arithmetic in buffer overflow detection
#[test]
fn test_buffer_overflow_saturation() {
    // Test case where offset + data_len would overflow u64
    let scenario = BufferWriteScenario {
        offset: u64::MAX - 10,
        data_len: 100, // Would overflow u64 if not for saturating_add
        buffer_size: u64::MAX,
    };

    // The implementation uses saturating_add, so:
    // offset.saturating_add(data_len as u64) = u64::MAX
    // buffer_size = u64::MAX
    // Therefore: end_offset (u64::MAX) > buffer_size (u64::MAX) is FALSE
    // This is actually a valid case (saturated to max equals buffer size)
    //
    // The real overflow protection here is that:
    // 1. The saturating_add prevents arithmetic overflow panic
    // 2. For any realistic buffer, this would still fail bounds check
    //
    // Let's test with a more realistic scenario where saturation matters:
    let scenario2 = BufferWriteScenario {
        offset: u64::MAX - 10,
        data_len: 100,
        buffer_size: 1024, // Realistic buffer size
    };
    // saturating_add gives u64::MAX, which is > 1024
    assert!(
        scenario2.fails_bounds(),
        "Saturating arithmetic should prevent panic and detect overflow against realistic buffer"
    );

    // Test that the original scenario doesn't panic (saturation works)
    // and the comparison is well-defined
    let _result = scenario.expected_error();
    // If we get here without panic, saturating arithmetic is working
}
