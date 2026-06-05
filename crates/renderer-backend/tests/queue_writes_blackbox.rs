// Blackbox contract tests for T-WGPU-P1.4.2 Queue Writes
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::device`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/device/queue.rs (implementation)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_TODO.md (T-WGPU-P1.4.2)
//   - docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_ARCH.md (Queue Management section)
//
// Acceptance criteria (T-WGPU-P1.4.2):
//   - Buffer write with offset and data
//   - Texture write with layout specification
//   - Alignment validation
//   - Size validation
//
// Test design rationale:
//   Equivalence partitioning:
//     - Valid aligned offsets vs unaligned offsets
//     - Data within buffer bounds vs exceeding bounds
//     - Valid texture layouts vs invalid layouts
//   Boundary cases:
//     - Zero offset (minimum valid)
//     - Alignment boundary values (4, 8, 252, 256)
//     - Empty data array
//     - Maximum offset at buffer size
//   Error cases:
//     - Unaligned buffer offset
//     - Unaligned bytes per row
//     - Offset + data.len() > buffer.size()
//     - Invalid texture extent

use renderer_backend::device::{
    align_bytes_per_row, is_buffer_offset_aligned, is_bytes_per_row_aligned, QueueWriteError,
    COPY_BUFFER_ALIGNMENT, COPY_BYTES_PER_ROW_ALIGNMENT,
};

// =============================================================================
// 1. Alignment Constants Contract Tests
// =============================================================================

/// Verifies COPY_BUFFER_ALIGNMENT is 4 bytes per wgpu spec.
///
/// Contract: Buffer write offsets must be aligned to 4 bytes.
#[test]
fn test_copy_buffer_alignment_is_4_bytes() {
    assert_eq!(
        COPY_BUFFER_ALIGNMENT, 4,
        "COPY_BUFFER_ALIGNMENT should be 4 bytes per wgpu specification"
    );
}

/// Verifies COPY_BYTES_PER_ROW_ALIGNMENT is 256 bytes per wgpu spec.
///
/// Contract: Texture bytes_per_row must be aligned to 256 bytes.
#[test]
fn test_copy_bytes_per_row_alignment_is_256_bytes() {
    assert_eq!(
        COPY_BYTES_PER_ROW_ALIGNMENT, 256,
        "COPY_BYTES_PER_ROW_ALIGNMENT should be 256 bytes per wgpu specification"
    );
}

/// Verifies alignment constants are powers of 2.
///
/// Contract: Alignment values must be powers of 2 for efficient masking.
#[test]
fn test_alignment_constants_are_powers_of_2() {
    assert!(
        COPY_BUFFER_ALIGNMENT.is_power_of_two(),
        "COPY_BUFFER_ALIGNMENT must be a power of 2"
    );
    assert!(
        COPY_BYTES_PER_ROW_ALIGNMENT.is_power_of_two(),
        "COPY_BYTES_PER_ROW_ALIGNMENT must be a power of 2"
    );
}

// =============================================================================
// 2. is_buffer_offset_aligned() Contract Tests
// =============================================================================

/// Verifies that zero offset is aligned.
///
/// Contract: Zero is always aligned to any power of 2.
#[test]
fn test_is_buffer_offset_aligned_zero_is_aligned() {
    assert!(
        is_buffer_offset_aligned(0),
        "Zero offset should be aligned"
    );
}

/// Verifies that multiples of COPY_BUFFER_ALIGNMENT are aligned.
///
/// Contract: Any multiple of 4 should be aligned.
#[test]
fn test_is_buffer_offset_aligned_multiples_of_4_are_aligned() {
    for multiple in [4, 8, 12, 16, 20, 100, 1000, 4096] {
        assert!(
            is_buffer_offset_aligned(multiple),
            "Offset {} should be aligned (multiple of {})",
            multiple,
            COPY_BUFFER_ALIGNMENT
        );
    }
}

/// Verifies that non-multiples of COPY_BUFFER_ALIGNMENT are not aligned.
///
/// Contract: Values not divisible by 4 should be unaligned.
#[test]
fn test_is_buffer_offset_aligned_non_multiples_are_unaligned() {
    for non_multiple in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 101, 255] {
        assert!(
            !is_buffer_offset_aligned(non_multiple),
            "Offset {} should NOT be aligned (not a multiple of {})",
            non_multiple,
            COPY_BUFFER_ALIGNMENT
        );
    }
}

/// Verifies boundary values around alignment.
///
/// Contract: Boundary value analysis at alignment points.
#[test]
fn test_is_buffer_offset_aligned_boundary_values() {
    // At boundary
    assert!(is_buffer_offset_aligned(4), "4 should be aligned");
    assert!(is_buffer_offset_aligned(8), "8 should be aligned");

    // Just below boundary
    assert!(!is_buffer_offset_aligned(3), "3 should NOT be aligned");
    assert!(!is_buffer_offset_aligned(7), "7 should NOT be aligned");

    // Just above boundary (still needs to be multiple)
    assert!(!is_buffer_offset_aligned(5), "5 should NOT be aligned");
    assert!(!is_buffer_offset_aligned(9), "9 should NOT be aligned");
}

/// Verifies large aligned values.
///
/// Contract: Large values that are multiples of 4 should be aligned.
#[test]
fn test_is_buffer_offset_aligned_large_values() {
    assert!(
        is_buffer_offset_aligned(1024 * 1024),
        "1MB should be aligned"
    );
    assert!(
        is_buffer_offset_aligned(256 * 1024 * 1024),
        "256MB should be aligned"
    );
    assert!(
        is_buffer_offset_aligned(u64::MAX - 3),
        "u64::MAX - 3 should be aligned (divisible by 4)"
    );
}

// =============================================================================
// 3. is_bytes_per_row_aligned() Contract Tests
// =============================================================================

/// Verifies that zero bytes_per_row is aligned.
///
/// Contract: Zero is always aligned.
#[test]
fn test_is_bytes_per_row_aligned_zero_is_aligned() {
    assert!(
        is_bytes_per_row_aligned(0),
        "Zero bytes_per_row should be aligned"
    );
}

/// Verifies that multiples of 256 are aligned.
///
/// Contract: Any multiple of 256 should be aligned.
#[test]
fn test_is_bytes_per_row_aligned_multiples_of_256_are_aligned() {
    for multiple in [256, 512, 768, 1024, 2048, 4096, 65536] {
        assert!(
            is_bytes_per_row_aligned(multiple),
            "bytes_per_row {} should be aligned (multiple of {})",
            multiple,
            COPY_BYTES_PER_ROW_ALIGNMENT
        );
    }
}

/// Verifies that non-multiples of 256 are not aligned.
///
/// Contract: Values not divisible by 256 should be unaligned.
#[test]
fn test_is_bytes_per_row_aligned_non_multiples_are_unaligned() {
    for non_multiple in [1, 4, 128, 255, 257, 300, 500, 1000] {
        assert!(
            !is_bytes_per_row_aligned(non_multiple),
            "bytes_per_row {} should NOT be aligned (not a multiple of {})",
            non_multiple,
            COPY_BYTES_PER_ROW_ALIGNMENT
        );
    }
}

/// Verifies boundary values around 256.
///
/// Contract: Boundary value analysis at alignment points.
#[test]
fn test_is_bytes_per_row_aligned_boundary_values() {
    // At boundary
    assert!(is_bytes_per_row_aligned(256), "256 should be aligned");
    assert!(is_bytes_per_row_aligned(512), "512 should be aligned");

    // Just below boundary
    assert!(!is_bytes_per_row_aligned(255), "255 should NOT be aligned");
    assert!(!is_bytes_per_row_aligned(511), "511 should NOT be aligned");

    // Just above boundary
    assert!(!is_bytes_per_row_aligned(257), "257 should NOT be aligned");
    assert!(!is_bytes_per_row_aligned(513), "513 should NOT be aligned");
}

// =============================================================================
// 4. align_bytes_per_row() Contract Tests
// =============================================================================

/// Verifies that zero remains zero when aligned.
///
/// Contract: Aligning zero should return zero.
#[test]
fn test_align_bytes_per_row_zero_returns_zero() {
    assert_eq!(align_bytes_per_row(0), 0, "Aligning 0 should return 0");
}

/// Verifies that already-aligned values remain unchanged.
///
/// Contract: Values that are already multiples of 256 should be unchanged.
#[test]
fn test_align_bytes_per_row_already_aligned_unchanged() {
    for aligned in [256, 512, 1024, 2048, 65536] {
        assert_eq!(
            align_bytes_per_row(aligned),
            aligned,
            "Already aligned value {} should be unchanged",
            aligned
        );
    }
}

/// Verifies that unaligned values are rounded up.
///
/// Contract: Unaligned values should round up to next multiple of 256.
#[test]
fn test_align_bytes_per_row_rounds_up() {
    // 1 should round up to 256
    assert_eq!(
        align_bytes_per_row(1),
        256,
        "1 should round up to 256"
    );

    // 255 should round up to 256
    assert_eq!(
        align_bytes_per_row(255),
        256,
        "255 should round up to 256"
    );

    // 257 should round up to 512
    assert_eq!(
        align_bytes_per_row(257),
        512,
        "257 should round up to 512"
    );

    // 128 should round up to 256
    assert_eq!(
        align_bytes_per_row(128),
        256,
        "128 should round up to 256"
    );
}

/// Verifies boundary value alignment.
///
/// Contract: Boundary value analysis for alignment function.
#[test]
fn test_align_bytes_per_row_boundary_values() {
    // Just below 256 -> 256
    assert_eq!(
        align_bytes_per_row(255),
        256,
        "255 should align to 256"
    );

    // Exactly 256 -> 256
    assert_eq!(
        align_bytes_per_row(256),
        256,
        "256 should remain 256"
    );

    // Just above 256 -> 512
    assert_eq!(
        align_bytes_per_row(257),
        512,
        "257 should align to 512"
    );

    // Just below 512 -> 512
    assert_eq!(
        align_bytes_per_row(511),
        512,
        "511 should align to 512"
    );
}

/// Verifies that aligned result is always aligned.
///
/// Contract: The return value should always pass is_bytes_per_row_aligned().
#[test]
fn test_align_bytes_per_row_result_is_always_aligned() {
    for input in [1, 100, 200, 255, 256, 257, 300, 500, 1000, 2000, 5000] {
        let aligned = align_bytes_per_row(input);
        assert!(
            is_bytes_per_row_aligned(aligned),
            "align_bytes_per_row({}) = {} should be aligned",
            input,
            aligned
        );
    }
}

/// Verifies that aligned result is >= input.
///
/// Contract: Alignment rounds up, never down.
#[test]
fn test_align_bytes_per_row_result_is_greater_or_equal() {
    for input in [0, 1, 100, 255, 256, 257, 1000, 10000] {
        let aligned = align_bytes_per_row(input);
        assert!(
            aligned >= input,
            "align_bytes_per_row({}) = {} should be >= input",
            input,
            aligned
        );
    }
}

// =============================================================================
// 5. QueueWriteError Contract Tests
// =============================================================================

/// Verifies that QueueWriteError type exists and implements Debug.
///
/// Contract: Error types should implement Debug for diagnostic purposes.
#[test]
fn test_queue_write_error_implements_debug() {
    // This test verifies the type exists and implements Debug
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<QueueWriteError>();
}

/// Verifies that QueueWriteError implements std::error::Error.
///
/// Contract: Error types should implement the Error trait.
#[test]
fn test_queue_write_error_implements_error() {
    fn assert_error<T: std::error::Error>() {}
    assert_error::<QueueWriteError>();
}

/// Verifies that QueueWriteError implements Display.
///
/// Contract: Error types should implement Display for user-facing messages.
#[test]
fn test_queue_write_error_implements_display() {
    fn assert_display<T: std::fmt::Display>() {}
    assert_display::<QueueWriteError>();
}

/// Verifies that QueueWriteError implements Clone.
///
/// Contract: Error types should be clonable for error propagation patterns.
#[test]
fn test_queue_write_error_implements_clone() {
    fn assert_clone<T: Clone>() {}
    assert_clone::<QueueWriteError>();
}

/// Verifies that QueueWriteError implements PartialEq.
///
/// Contract: Error types should be comparable for testing.
#[test]
fn test_queue_write_error_implements_partial_eq() {
    fn assert_partial_eq<T: PartialEq>() {}
    assert_partial_eq::<QueueWriteError>();
}

// =============================================================================
// 6. Property-Based Tests for Alignment Functions
// =============================================================================

/// Verifies alignment idempotence: aligning an aligned value returns the same value.
///
/// Contract: align(align(x)) == align(x)
#[test]
fn test_align_bytes_per_row_idempotent() {
    for input in [0, 1, 100, 255, 256, 257, 500, 1000, 10000] {
        let once = align_bytes_per_row(input);
        let twice = align_bytes_per_row(once);
        assert_eq!(
            once, twice,
            "Alignment should be idempotent: align(align({})) = {} but align({}) = {}",
            input, twice, input, once
        );
    }
}

/// Verifies that alignment increases by at most (alignment - 1).
///
/// Contract: The increase from input to aligned should be less than alignment.
#[test]
fn test_align_bytes_per_row_increase_is_bounded() {
    let alignment = COPY_BYTES_PER_ROW_ALIGNMENT;
    for input in [1, 100, 200, 255, 257, 500, 1000] {
        let aligned = align_bytes_per_row(input);
        let increase = aligned - input;
        assert!(
            increase < alignment,
            "align_bytes_per_row({}) increased by {}, which should be < {}",
            input,
            increase,
            alignment
        );
    }
}

/// Verifies that buffer offset alignment check matches manual calculation.
///
/// Contract: is_buffer_offset_aligned(x) == (x % 4 == 0)
#[test]
fn test_is_buffer_offset_aligned_matches_modulo() {
    for offset in 0..100 {
        let expected = offset % COPY_BUFFER_ALIGNMENT == 0;
        let actual = is_buffer_offset_aligned(offset);
        assert_eq!(
            actual, expected,
            "is_buffer_offset_aligned({}) returned {}, expected {}",
            offset, actual, expected
        );
    }
}

/// Verifies that bytes_per_row alignment check matches manual calculation.
///
/// Contract: is_bytes_per_row_aligned(x) == (x % 256 == 0)
#[test]
fn test_is_bytes_per_row_aligned_matches_modulo() {
    for bpr in (0..1024).step_by(10) {
        let expected = bpr % COPY_BYTES_PER_ROW_ALIGNMENT == 0;
        let actual = is_bytes_per_row_aligned(bpr);
        assert_eq!(
            actual, expected,
            "is_bytes_per_row_aligned({}) returned {}, expected {}",
            bpr, actual, expected
        );
    }
}

// =============================================================================
// 7. Integration with TrinityQueue (requires device)
// =============================================================================

#[cfg(not(target_arch = "wasm32"))]
mod integration_tests {
    use renderer_backend::device::{TrinityInstance, TrinityQueue};
    use wgpu::{BufferDescriptor, BufferUsages};

    /// Helper to create a device and queue for testing.
    /// Returns owned (Device, Queue) tuple that can be used to construct TrinityQueue.
    fn create_test_device_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = TrinityInstance::new();
        let adapters = instance.inner().enumerate_adapters(instance.backends());

        if adapters.is_empty() {
            return None;
        }

        pollster::block_on(adapters[0].request_device(
            &wgpu::DeviceDescriptor {
                label: Some("test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .ok()
    }

    /// Verifies TrinityQueue can be created from wgpu::Queue.
    ///
    /// Contract: TrinityQueue wraps wgpu::Queue for extended functionality.
    #[test]
    fn test_trinity_queue_can_be_created() {
        if let Some((_device, queue)) = create_test_device_queue() {
            let trinity_queue = TrinityQueue::new(queue);
            // Just verify it compiles and doesn't panic
            let _ = trinity_queue;
        }
    }

    /// Verifies write_buffer_validated rejects unaligned offset.
    ///
    /// Contract: Buffer offset must be aligned to COPY_BUFFER_ALIGNMENT.
    #[test]
    fn test_write_buffer_validated_rejects_unaligned_offset() {
        if let Some((device, queue)) = create_test_device_queue() {
            let buffer_size = 256u64;
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("test_buffer"),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });

            let trinity_queue = TrinityQueue::new(queue);
            let data = [0u8; 16];

            // Unaligned offset of 1 should fail
            let result = trinity_queue.write_buffer_validated(&buffer, 1, &data, buffer_size);
            assert!(
                result.is_err(),
                "write_buffer_validated should reject unaligned offset 1"
            );

            if let Err(e) = result {
                // Verify error message mentions alignment
                let msg = format!("{}", e);
                assert!(
                    msg.to_lowercase().contains("align"),
                    "Error message should mention alignment: {}",
                    msg
                );
            }
        }
    }

    /// Verifies write_buffer_validated rejects offset 3 (unaligned).
    ///
    /// Contract: Offset 3 is not divisible by 4.
    #[test]
    fn test_write_buffer_validated_rejects_offset_3() {
        if let Some((device, queue)) = create_test_device_queue() {
            let buffer_size = 256u64;
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("test_buffer"),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });

            let trinity_queue = TrinityQueue::new(queue);
            let data = [0u8; 8];

            let result = trinity_queue.write_buffer_validated(&buffer, 3, &data, buffer_size);
            assert!(
                result.is_err(),
                "write_buffer_validated should reject unaligned offset 3"
            );
        }
    }

    /// Verifies write_buffer_validated accepts aligned offset.
    ///
    /// Contract: Offset 4 is aligned to COPY_BUFFER_ALIGNMENT.
    #[test]
    fn test_write_buffer_validated_accepts_aligned_offset() {
        if let Some((device, queue)) = create_test_device_queue() {
            let buffer_size = 256u64;
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("test_buffer"),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });

            let trinity_queue = TrinityQueue::new(queue);
            let data = [0u8; 16];

            // Aligned offset of 4 should succeed
            let result = trinity_queue.write_buffer_validated(&buffer, 4, &data, buffer_size);
            assert!(
                result.is_ok(),
                "write_buffer_validated should accept aligned offset 4: {:?}",
                result
            );
        }
    }

    /// Verifies write_buffer_validated accepts zero offset.
    ///
    /// Contract: Zero is always aligned.
    #[test]
    fn test_write_buffer_validated_accepts_zero_offset() {
        if let Some((device, queue)) = create_test_device_queue() {
            let buffer_size = 256u64;
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("test_buffer"),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });

            let trinity_queue = TrinityQueue::new(queue);
            let data = [0u8; 16];

            let result = trinity_queue.write_buffer_validated(&buffer, 0, &data, buffer_size);
            assert!(
                result.is_ok(),
                "write_buffer_validated should accept zero offset: {:?}",
                result
            );
        }
    }

    /// Verifies write_buffer_validated rejects data exceeding buffer size.
    ///
    /// Contract: offset + data.len() must be <= buffer.size()
    #[test]
    fn test_write_buffer_validated_rejects_overflow() {
        if let Some((device, queue)) = create_test_device_queue() {
            let buffer_size = 64u64; // Small buffer
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("test_buffer"),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });

            let trinity_queue = TrinityQueue::new(queue);
            let data = [0u8; 128]; // Larger than buffer

            let result = trinity_queue.write_buffer_validated(&buffer, 0, &data, buffer_size);
            assert!(
                result.is_err(),
                "write_buffer_validated should reject data larger than buffer"
            );

            if let Err(e) = result {
                let msg = format!("{}", e);
                assert!(
                    msg.to_lowercase().contains("size")
                        || msg.to_lowercase().contains("overflow")
                        || msg.to_lowercase().contains("exceed")
                        || msg.to_lowercase().contains("bound"),
                    "Error message should mention size/overflow: {}",
                    msg
                );
            }
        }
    }

    /// Verifies write_buffer_validated rejects offset + data exceeding buffer.
    ///
    /// Contract: Even with valid offset, total must fit in buffer.
    #[test]
    fn test_write_buffer_validated_rejects_offset_plus_data_overflow() {
        if let Some((device, queue)) = create_test_device_queue() {
            let buffer_size = 64u64;
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("test_buffer"),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });

            let trinity_queue = TrinityQueue::new(queue);
            let data = [0u8; 32];

            // Offset 48 + 32 bytes = 80 bytes, exceeds 64 byte buffer
            let result = trinity_queue.write_buffer_validated(&buffer, 48, &data, buffer_size);
            assert!(
                result.is_err(),
                "write_buffer_validated should reject offset + data exceeding buffer size"
            );
        }
    }

    /// Verifies write_buffer_validated accepts data exactly fitting buffer.
    ///
    /// Contract: Boundary case where offset + data.len() == buffer.size()
    #[test]
    fn test_write_buffer_validated_accepts_exact_fit() {
        if let Some((device, queue)) = create_test_device_queue() {
            let buffer_size = 64u64;
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("test_buffer"),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });

            let trinity_queue = TrinityQueue::new(queue);
            let data = [0u8; 64];

            // Offset 0 + 64 bytes = 64 bytes, exactly fits 64 byte buffer
            let result = trinity_queue.write_buffer_validated(&buffer, 0, &data, buffer_size);
            assert!(
                result.is_ok(),
                "write_buffer_validated should accept data exactly fitting buffer: {:?}",
                result
            );
        }
    }

    /// Verifies write_buffer_validated accepts empty data.
    ///
    /// Contract: Empty writes should be valid (no-op).
    #[test]
    fn test_write_buffer_validated_accepts_empty_data() {
        if let Some((device, queue)) = create_test_device_queue() {
            let buffer_size = 64u64;
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("test_buffer"),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });

            let trinity_queue = TrinityQueue::new(queue);
            let data: [u8; 0] = [];

            let result = trinity_queue.write_buffer_validated(&buffer, 0, &data, buffer_size);
            assert!(
                result.is_ok(),
                "write_buffer_validated should accept empty data: {:?}",
                result
            );
        }
    }

    /// Verifies write_buffer_validated works with various aligned offsets.
    ///
    /// Contract: Multiple valid aligned offsets should work.
    #[test]
    fn test_write_buffer_validated_various_aligned_offsets() {
        if let Some((device, queue)) = create_test_device_queue() {
            let buffer_size = 256u64;
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("test_buffer"),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });

            let trinity_queue = TrinityQueue::new(queue);
            let data = [0u8; 16];

            for offset in [0, 4, 8, 16, 32, 64, 128, 240] {
                let result = trinity_queue.write_buffer_validated(&buffer, offset, &data, buffer_size);
                assert!(
                    result.is_ok(),
                    "write_buffer_validated should accept aligned offset {}: {:?}",
                    offset,
                    result
                );
            }
        }
    }

    /// Verifies write_buffer_validated rejects various unaligned offsets.
    ///
    /// Contract: Multiple invalid unaligned offsets should fail.
    #[test]
    fn test_write_buffer_validated_various_unaligned_offsets() {
        if let Some((device, queue)) = create_test_device_queue() {
            let buffer_size = 256u64;
            let buffer = device.create_buffer(&BufferDescriptor {
                label: Some("test_buffer"),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });

            let trinity_queue = TrinityQueue::new(queue);
            let data = [0u8; 16];

            for offset in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17] {
                let result = trinity_queue.write_buffer_validated(&buffer, offset, &data, buffer_size);
                assert!(
                    result.is_err(),
                    "write_buffer_validated should reject unaligned offset {}",
                    offset
                );
            }
        }
    }
}

// =============================================================================
// Summary
// =============================================================================
//
// Tests cover:
//   1. Alignment constants (COPY_BUFFER_ALIGNMENT = 4, COPY_BYTES_PER_ROW_ALIGNMENT = 256)
//   2. is_buffer_offset_aligned() - zero, multiples, non-multiples, boundaries
//   3. is_bytes_per_row_aligned() - zero, multiples, non-multiples, boundaries
//   4. align_bytes_per_row() - zero, already aligned, rounding up, idempotence
//   5. QueueWriteError - Debug, Error, Display, Clone, PartialEq traits
//   6. Property-based tests for alignment functions
//   7. Integration tests with real device (conditional on hardware availability):
//      - TrinityQueue creation
//      - write_buffer_validated alignment validation
//      - write_buffer_validated size validation (overflow)
//      - Boundary cases (exact fit, empty data)
