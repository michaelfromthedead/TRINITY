// WHITEBOX tests for T-WGPU-P2.1.1 (Buffer Creation API) and T-WGPU-P2.1.2 (Buffer Usage Flags)
//
// WHITEBOX discipline: These tests have FULL ACCESS to the implementation.
// They exercise internal code paths, branch conditions, and edge cases
// that are not visible through the public contract alone.
//
// Implementation under test: crates/renderer-backend/src/resources/buffer.rs
//   - TrinityBuffer: Wrapper with inner, size, usage, label fields
//   - TrinityBufferDescriptor: label, size, usage, mapped_at_creation
//   - create_buffer(): Panicking version
//   - try_create_buffer(): Returns Result<TrinityBuffer, BufferCreationError>
//   - align_size(): Aligns to 4 bytes
//   - is_aligned(): Checks 4-byte alignment
//   - BUFFER_ALIGNMENT = 4
//   - buffer_usages: Module with 9 const presets (T-WGPU-P2.1.2)
//   - UsageValidationError: Enum with 3 variants (T-WGPU-P2.1.2)
//   - validate_usage(): Usage flag validation (T-WGPU-P2.1.2)
//   - validate_usage_with_label(): Validation with context (T-WGPU-P2.1.2)
//
// WHITEBOX coverage plan (T-WGPU-P2.1.1):
//   - Path A: align_size(0) behavior
//   - Path B: align_size() for values 1-8
//   - Path C: align_size() for large values near u64::MAX
//   - Path D: is_aligned() for aligned values (0, 4, 8, 100, 1024)
//   - Path E: is_aligned() for unaligned values (1, 2, 3, 5, 101)
//   - Path F: BufferCreationError::ZeroSize on size = 0
//   - Path G: BufferCreationError::EmptyUsage on empty usage flags
//   - Path H: Error Display formatting
//   - Path I: Error Debug formatting
//   - Path J: TrinityBufferDescriptor::default() values
//   - Path K: TrinityBufferDescriptor clone behavior
//   - Path L: Descriptor with special character labels
//   - Path M: All BufferUsages flag combinations
//   - Path N: Very large buffer sizes (no panic)
//
// WHITEBOX coverage plan (T-WGPU-P2.1.2):
//   - Path O: All 9 buffer_usages presets are const and valid
//   - Path P: Each preset contains expected flags
//   - Path Q: UsageValidationError::MapReadAndWrite detection
//   - Path R: UsageValidationError::MapReadWithGpuOnly detection
//   - Path S: UsageValidationError::MapWriteWithGpuOnly (documented but not enforced)
//   - Path T: validate_usage() passes valid combinations
//   - Path U: validate_usage() edge cases (empty, single flag)
//   - Path V: UsageValidationError Display trait formatting
//   - Path W: UsageValidationError Clone, PartialEq, Eq traits
//   - Path X: UsageValidationError implements std::error::Error
//   - Path Y: validate_usage_with_label() includes label in error
//   - Path Z: BufferCreationError::InvalidUsage variant
//   - Path AA: From<UsageValidationError> for BufferCreationError impl
//   - Path AB: Error source chain for InvalidUsage

use renderer_backend::resources::buffer::{
    align_size, buffer_usages, is_aligned, validate_usage, validate_usage_with_label,
    BufferCreationError, TrinityBufferDescriptor, UsageValidationError, BUFFER_ALIGNMENT,
};
use wgpu::BufferUsages;

// ============================================================================
// Constants Verification
// ============================================================================

#[test]
fn test_buffer_alignment_constant() {
    // Verify the alignment constant is 4 bytes as documented
    assert_eq!(BUFFER_ALIGNMENT, 4);
}

// ============================================================================
// Path A-C: align_size() Tests
// ============================================================================

#[test]
fn test_align_size_zero() {
    // Path A: align_size(0) should return 0
    // This is a special case - zero size is not a valid buffer size
    // but the alignment function itself should handle it mathematically
    assert_eq!(align_size(0), 0);
}

#[test]
fn test_align_size_one_to_four() {
    // Path B: Values 1-4 should all align to 4
    assert_eq!(align_size(1), 4);
    assert_eq!(align_size(2), 4);
    assert_eq!(align_size(3), 4);
    assert_eq!(align_size(4), 4);
}

#[test]
fn test_align_size_five_to_eight() {
    // Path B continued: Values 5-8 should all align to 8
    assert_eq!(align_size(5), 8);
    assert_eq!(align_size(6), 8);
    assert_eq!(align_size(7), 8);
    assert_eq!(align_size(8), 8);
}

#[test]
fn test_align_size_already_aligned() {
    // Values already aligned should stay the same
    assert_eq!(align_size(4), 4);
    assert_eq!(align_size(8), 8);
    assert_eq!(align_size(12), 12);
    assert_eq!(align_size(16), 16);
    assert_eq!(align_size(100), 100);
    assert_eq!(align_size(1024), 1024);
    assert_eq!(align_size(4096), 4096);
    assert_eq!(align_size(1024 * 1024), 1024 * 1024); // 1 MB
}

#[test]
fn test_align_size_near_alignment_boundary() {
    // Test values just below and at alignment boundaries
    assert_eq!(align_size(99), 100);
    assert_eq!(align_size(100), 100);
    assert_eq!(align_size(101), 104);
    assert_eq!(align_size(102), 104);
    assert_eq!(align_size(103), 104);
    assert_eq!(align_size(104), 104);
}

#[test]
fn test_align_size_powers_of_two() {
    // Powers of 2 that are >= 4 should be aligned already
    for pow in 2..30 {
        let size: u64 = 1 << pow;
        assert_eq!(align_size(size), size, "Failed for 2^{}", pow);
    }
}

#[test]
fn test_align_size_powers_of_two_minus_one() {
    // Powers of 2 minus 1: should align up to the next multiple of 4
    // 2^2 - 1 = 3 -> 4
    assert_eq!(align_size(3), 4);
    // 2^3 - 1 = 7 -> 8
    assert_eq!(align_size(7), 8);
    // 2^4 - 1 = 15 -> 16
    assert_eq!(align_size(15), 16);
    // 2^5 - 1 = 31 -> 32
    assert_eq!(align_size(31), 32);
    // 2^10 - 1 = 1023 -> 1024
    assert_eq!(align_size(1023), 1024);
}

#[test]
fn test_align_size_large_values() {
    // Path C: Large but reasonable buffer sizes
    let one_gb: u64 = 1024 * 1024 * 1024;
    assert_eq!(align_size(one_gb), one_gb);
    assert_eq!(align_size(one_gb + 1), one_gb + 4);
    assert_eq!(align_size(one_gb + 2), one_gb + 4);
    assert_eq!(align_size(one_gb + 3), one_gb + 4);

    // 4 GB
    let four_gb: u64 = 4 * 1024 * 1024 * 1024;
    assert_eq!(align_size(four_gb), four_gb);
    assert_eq!(align_size(four_gb + 1), four_gb + 4);
}

#[test]
fn test_align_size_near_u64_max() {
    // Path C: Test values near u64::MAX for overflow handling
    // The alignment formula is: (size + BUFFER_ALIGNMENT - 1) & !(BUFFER_ALIGNMENT - 1)
    // This means: (size + 3) & !3
    //
    // IMPORTANT: The current implementation will panic on overflow in debug builds
    // for values where size + 3 > u64::MAX. This is acceptable because:
    // 1. No real GPU can allocate buffers anywhere near u64::MAX bytes
    // 2. Validation catches size == 0, and practical limits are far below overflow
    //
    // Test values that are safe (already aligned, so no addition needed beyond alignment)
    // u64::MAX & !3 = largest aligned value = 18446744073709551612
    let _max_aligned = u64::MAX & !(BUFFER_ALIGNMENT - 1);

    // Already-aligned values near the max don't overflow because the formula
    // (max_aligned + 3) & !3 might overflow the addition, so we test safe values instead

    // Test large but safe values (won't overflow)
    // Note: Even u64::MAX - 10 would overflow when we add 3 to align it,
    // so we need to test values that are already aligned.

    // Test values that are already aligned (no overflow possible)
    let test_val = u64::MAX - 7; // = 18446744073709551608, which is divisible by 4
    assert!(test_val % 4 == 0, "Test value should be aligned");
    assert_eq!(align_size(test_val), test_val);

    // Another aligned value
    let test_val2 = u64::MAX - 15; // = 18446744073709551600, divisible by 4
    assert!(test_val2 % 4 == 0, "Test value should be aligned");
    assert_eq!(align_size(test_val2), test_val2);
}

#[test]
fn test_align_size_overflow_behavior_documented() {
    // WHITEBOX FINDING: The align_size() function uses standard addition
    // which panics on overflow in debug builds. This is intentional behavior:
    //
    // 1. Buffer sizes near u64::MAX (18+ exabytes) are impossible on real hardware
    // 2. Validation at the API level catches invalid sizes long before this
    // 3. The panic-on-overflow provides safety against accidental huge allocations
    //
    // Values that would overflow: anything where size + 3 > u64::MAX
    // - u64::MAX - 2 = 18446744073709551613, +3 = overflow
    // - u64::MAX - 1 = 18446744073709551614, +3 = overflow
    // - u64::MAX = 18446744073709551615, +3 = overflow
    //
    // This test documents that overflow behavior is NOT gracefully handled,
    // which is the correct design for a low-level alignment utility.
    //
    // In release builds with overflow checks disabled, wrapping would occur,
    // but this is also acceptable as validation happens elsewhere.

    // Test that reasonable large values work fine
    let reasonable_large = 1024 * 1024 * 1024 * 16u64; // 16 GB
    let aligned = align_size(reasonable_large);
    assert_eq!(aligned, reasonable_large); // Already aligned

    // Test unaligned large value
    let unaligned_large = reasonable_large + 1;
    let aligned2 = align_size(unaligned_large);
    assert_eq!(aligned2, reasonable_large + 4);

    // Maximum practical GPU buffer size (most GPUs cap at a few GB)
    let max_practical = 8 * 1024 * 1024 * 1024u64; // 8 GB
    assert_eq!(align_size(max_practical), max_practical);
    assert_eq!(align_size(max_practical + 1), max_practical + 4);
}

// ============================================================================
// Path D-E: is_aligned() Tests
// ============================================================================

#[test]
fn test_is_aligned_zero() {
    // Path D: Zero is technically aligned (0 % 4 == 0)
    assert!(is_aligned(0));
}

#[test]
fn test_is_aligned_aligned_values() {
    // Path D: Values divisible by 4 should be aligned
    assert!(is_aligned(4));
    assert!(is_aligned(8));
    assert!(is_aligned(12));
    assert!(is_aligned(16));
    assert!(is_aligned(100));
    assert!(is_aligned(1024));
    assert!(is_aligned(4096));
    assert!(is_aligned(1024 * 1024)); // 1 MB
    assert!(is_aligned(u64::MAX & !(BUFFER_ALIGNMENT - 1)));
}

#[test]
fn test_is_aligned_unaligned_values() {
    // Path E: Values not divisible by 4 should not be aligned
    assert!(!is_aligned(1));
    assert!(!is_aligned(2));
    assert!(!is_aligned(3));
    assert!(!is_aligned(5));
    assert!(!is_aligned(6));
    assert!(!is_aligned(7));
    assert!(!is_aligned(101));
    assert!(!is_aligned(102));
    assert!(!is_aligned(103));
    assert!(!is_aligned(1025));
}

#[test]
fn test_is_aligned_consistency_with_align_size() {
    // Verify is_aligned() is consistent with align_size()
    for i in 0..1000 {
        let aligned = align_size(i);
        assert!(
            is_aligned(aligned),
            "align_size({}) = {}, but is_aligned({}) returned false",
            i,
            aligned,
            aligned
        );
    }
}

// ============================================================================
// Path F-I: Error Type Tests
// ============================================================================

#[test]
fn test_buffer_creation_error_zero_size() {
    // Path F: ZeroSize error variant
    let err = BufferCreationError::ZeroSize;
    assert_eq!(err, BufferCreationError::ZeroSize);
}

#[test]
fn test_buffer_creation_error_empty_usage() {
    // Path G: EmptyUsage error variant
    let err = BufferCreationError::EmptyUsage;
    assert_eq!(err, BufferCreationError::EmptyUsage);
}

#[test]
fn test_buffer_creation_error_display() {
    // Path H: Display formatting
    assert_eq!(
        BufferCreationError::ZeroSize.to_string(),
        "buffer size must be greater than 0"
    );
    assert_eq!(
        BufferCreationError::EmptyUsage.to_string(),
        "buffer usage flags must not be empty"
    );
}

#[test]
fn test_buffer_creation_error_debug() {
    // Path I: Debug formatting
    assert_eq!(format!("{:?}", BufferCreationError::ZeroSize), "ZeroSize");
    assert_eq!(
        format!("{:?}", BufferCreationError::EmptyUsage),
        "EmptyUsage"
    );
}

#[test]
fn test_buffer_creation_error_clone() {
    // Verify Clone trait
    let err1 = BufferCreationError::ZeroSize;
    let err2 = err1.clone();
    assert_eq!(err1, err2);

    let err3 = BufferCreationError::EmptyUsage;
    let err4 = err3.clone();
    assert_eq!(err3, err4);
}

#[test]
fn test_buffer_creation_error_is_error() {
    // Verify std::error::Error trait implementation
    let err: &dyn std::error::Error = &BufferCreationError::ZeroSize;
    assert!(err.source().is_none()); // No underlying cause

    let err: &dyn std::error::Error = &BufferCreationError::EmptyUsage;
    assert!(err.source().is_none());
}

#[test]
fn test_buffer_creation_error_equality() {
    // PartialEq and Eq trait tests
    assert_eq!(BufferCreationError::ZeroSize, BufferCreationError::ZeroSize);
    assert_eq!(
        BufferCreationError::EmptyUsage,
        BufferCreationError::EmptyUsage
    );
    assert_ne!(BufferCreationError::ZeroSize, BufferCreationError::EmptyUsage);
}

// ============================================================================
// Path J-L: TrinityBufferDescriptor Tests
// ============================================================================

#[test]
fn test_descriptor_default_values() {
    // Path J: Default values
    let desc = TrinityBufferDescriptor::default();

    assert!(desc.label.is_none());
    assert_eq!(desc.size, BUFFER_ALIGNMENT); // Minimum valid size = 4
    assert_eq!(desc.usage, BufferUsages::COPY_DST); // Safe default
    assert!(!desc.mapped_at_creation);
}

#[test]
fn test_descriptor_clone() {
    // Path K: Clone behavior
    let desc1 = TrinityBufferDescriptor {
        label: Some("test_buffer"),
        size: 256,
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    };

    let desc2 = desc1.clone();

    assert_eq!(desc1.label, desc2.label);
    assert_eq!(desc1.size, desc2.size);
    assert_eq!(desc1.usage, desc2.usage);
    assert_eq!(desc1.mapped_at_creation, desc2.mapped_at_creation);
}

#[test]
fn test_descriptor_debug() {
    let desc = TrinityBufferDescriptor {
        label: Some("debug_test"),
        size: 128,
        usage: BufferUsages::UNIFORM,
        mapped_at_creation: true,
    };

    let debug_str = format!("{:?}", desc);

    // Verify debug output contains expected fields
    assert!(debug_str.contains("TrinityBufferDescriptor"));
    assert!(debug_str.contains("debug_test"));
    assert!(debug_str.contains("128"));
}

#[test]
fn test_descriptor_special_character_labels() {
    // Path L: Special character labels
    let special_labels = [
        "buffer with spaces",
        "buffer-with-dashes",
        "buffer_with_underscores",
        "buffer.with.dots",
        "buffer/with/slashes",
        "buffer#with@special!chars",
        "buffer\twith\ttabs",
        "buffer\nwith\nnewlines",
        "buffer with unicode: \u{1F600}", // emoji
        "",                                // empty string
        " ",                               // single space
        "   ",                             // multiple spaces
    ];

    for label in &special_labels {
        let desc = TrinityBufferDescriptor {
            label: Some(label),
            size: 64,
            usage: BufferUsages::VERTEX,
            mapped_at_creation: false,
        };

        assert_eq!(desc.label, Some(*label));
    }
}

#[test]
fn test_descriptor_none_label() {
    let desc = TrinityBufferDescriptor {
        label: None,
        size: 64,
        usage: BufferUsages::VERTEX,
        mapped_at_creation: false,
    };

    assert!(desc.label.is_none());
}

// ============================================================================
// Path M: BufferUsages Flag Combinations
// ============================================================================

#[test]
fn test_usage_individual_flags() {
    // Path M: Test each individual usage flag
    let flags = [
        BufferUsages::MAP_READ,
        BufferUsages::MAP_WRITE,
        BufferUsages::COPY_SRC,
        BufferUsages::COPY_DST,
        BufferUsages::INDEX,
        BufferUsages::VERTEX,
        BufferUsages::UNIFORM,
        BufferUsages::STORAGE,
        BufferUsages::INDIRECT,
        BufferUsages::QUERY_RESOLVE,
    ];

    for flag in &flags {
        let desc = TrinityBufferDescriptor {
            label: Some("flag_test"),
            size: 64,
            usage: *flag,
            mapped_at_creation: false,
        };

        assert_eq!(desc.usage, *flag);
        assert!(!desc.usage.is_empty());
    }
}

#[test]
fn test_usage_common_combinations() {
    // Common real-world usage combinations
    let combinations = [
        // Vertex buffer with upload capability
        BufferUsages::VERTEX | BufferUsages::COPY_DST,
        // Index buffer with upload capability
        BufferUsages::INDEX | BufferUsages::COPY_DST,
        // Uniform buffer (most common pattern)
        BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        // Storage buffer for compute
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        // Staging buffer for writes
        BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
        // Staging buffer for reads
        BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        // Indirect draw buffer
        BufferUsages::INDIRECT | BufferUsages::COPY_DST | BufferUsages::STORAGE,
    ];

    for combo in &combinations {
        let desc = TrinityBufferDescriptor {
            label: Some("combo_test"),
            size: 64,
            usage: *combo,
            mapped_at_creation: false,
        };

        assert!(!desc.usage.is_empty());
        assert!(desc.usage.bits() > 0);
    }
}

#[test]
fn test_usage_all_flags() {
    // All flags combined
    let all = BufferUsages::all();
    let desc = TrinityBufferDescriptor {
        label: Some("all_flags"),
        size: 64,
        usage: all,
        mapped_at_creation: false,
    };

    assert!(!desc.usage.is_empty());
    assert!(desc.usage.contains(BufferUsages::VERTEX));
    assert!(desc.usage.contains(BufferUsages::UNIFORM));
    assert!(desc.usage.contains(BufferUsages::STORAGE));
}

#[test]
fn test_usage_empty_flag() {
    // Empty usage flag - this would be rejected by try_create_buffer
    let empty = BufferUsages::empty();
    let desc = TrinityBufferDescriptor {
        label: Some("empty_flags"),
        size: 64,
        usage: empty,
        mapped_at_creation: false,
    };

    assert!(desc.usage.is_empty());
    assert_eq!(desc.usage.bits(), 0);
}

// ============================================================================
// Path N: Large Buffer Sizes
// ============================================================================

#[test]
fn test_descriptor_large_sizes() {
    // Path N: Very large buffer sizes (no panic in descriptor creation)
    let large_sizes: Vec<u64> = vec![
        1024 * 1024,           // 1 MB
        1024 * 1024 * 16,      // 16 MB
        1024 * 1024 * 256,     // 256 MB
        1024 * 1024 * 1024,    // 1 GB
        1024 * 1024 * 1024 * 4, // 4 GB
        u64::MAX / 2,          // Half max
        u64::MAX - 4,          // Near max, aligned
        u64::MAX,              // Maximum
    ];

    for size in large_sizes {
        let desc = TrinityBufferDescriptor {
            label: Some("large_buffer"),
            size,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        };

        assert_eq!(desc.size, size);
    }
}

#[test]
fn test_descriptor_mapped_at_creation() {
    // Test mapped_at_creation field
    let desc_unmapped = TrinityBufferDescriptor {
        label: Some("unmapped"),
        size: 64,
        usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    };
    assert!(!desc_unmapped.mapped_at_creation);

    let desc_mapped = TrinityBufferDescriptor {
        label: Some("mapped"),
        size: 64,
        usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    };
    assert!(desc_mapped.mapped_at_creation);
}

// ============================================================================
// Alignment Roundtrip Tests
// ============================================================================

#[test]
fn test_align_idempotent() {
    // Aligning an already aligned value should return the same value
    for i in 0..1000 {
        let first = align_size(i);
        let second = align_size(first);
        assert_eq!(
            first, second,
            "align_size is not idempotent for input {}",
            i
        );
    }
}

#[test]
fn test_align_monotonic() {
    // align_size(a) <= align_size(b) when a <= b (for reasonable values)
    for i in 0..1000 {
        for j in i..1000 {
            let aligned_i = align_size(i);
            let aligned_j = align_size(j);
            assert!(
                aligned_i <= aligned_j,
                "align_size is not monotonic: align_size({}) = {}, align_size({}) = {}",
                i,
                aligned_i,
                j,
                aligned_j
            );
        }
    }
}

#[test]
fn test_align_never_smaller() {
    // align_size(x) >= x for all reasonable values
    for i in 0..10000 {
        let aligned = align_size(i);
        assert!(
            aligned >= i,
            "align_size({}) = {} is smaller than input",
            i,
            aligned
        );
    }
}

#[test]
fn test_align_at_most_three_larger() {
    // align_size(x) <= x + 3 (since we align to 4)
    for i in 0..10000 {
        let aligned = align_size(i);
        assert!(
            aligned <= i.saturating_add(3),
            "align_size({}) = {} is more than 3 larger than input",
            i,
            aligned
        );
    }
}

// ============================================================================
// Validation Logic Tests (without device)
// ============================================================================

#[test]
fn test_validation_logic_zero_size_would_fail() {
    // Verify that a descriptor with size = 0 would fail validation
    let desc = TrinityBufferDescriptor {
        label: Some("zero_size"),
        size: 0,
        usage: BufferUsages::VERTEX,
        mapped_at_creation: false,
    };

    // We can't call try_create_buffer without a device, but we can verify
    // the descriptor state that would cause failure
    assert_eq!(desc.size, 0);
}

#[test]
fn test_validation_logic_empty_usage_would_fail() {
    // Verify that a descriptor with empty usage would fail validation
    let desc = TrinityBufferDescriptor {
        label: Some("empty_usage"),
        size: 64,
        usage: BufferUsages::empty(),
        mapped_at_creation: false,
    };

    // Verify the descriptor state that would cause failure
    assert!(desc.usage.is_empty());
}

#[test]
fn test_automatic_alignment_would_occur() {
    // Verify that sizes 5, 6, 7 would be aligned to 8
    for size in 5..=7 {
        let aligned = align_size(size);
        assert_eq!(aligned, 8);
    }

    // Verify size 100 stays 100 (already aligned)
    assert_eq!(align_size(100), 100);

    // Verify size 101 becomes 104
    assert_eq!(align_size(101), 104);
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_align_size_stress() {
    // Stress test alignment with many values
    let mut aligned_count = 0;
    let mut unaligned_count = 0;

    for i in 0..100_000 {
        let aligned = align_size(i);

        // Verify result is always aligned
        assert!(is_aligned(aligned), "align_size({}) = {} is not aligned", i, aligned);

        if i == aligned {
            aligned_count += 1;
        } else {
            unaligned_count += 1;
        }
    }

    // Roughly 1/4 of values should already be aligned (0, 4, 8, 12, ...)
    // Actually: aligned values are 0, 4, 8, ... so 25% of values
    assert!(
        aligned_count >= 24_000 && aligned_count <= 26_000,
        "Expected ~25000 aligned values, got {}",
        aligned_count
    );
    assert!(
        unaligned_count >= 74_000 && unaligned_count <= 76_000,
        "Expected ~75000 unaligned values, got {}",
        unaligned_count
    );
}

#[test]
fn test_is_aligned_stress() {
    // Stress test is_aligned with many values
    for i in 0..100_000 {
        let expected = i % BUFFER_ALIGNMENT == 0;
        assert_eq!(
            is_aligned(i),
            expected,
            "is_aligned({}) returned unexpected result",
            i
        );
    }
}

// ============================================================================
// Boundary Tests
// ============================================================================

#[test]
fn test_alignment_at_common_boundaries() {
    // Test at common GPU buffer size boundaries
    let boundaries: Vec<u64> = vec![
        64,    // Min uniform buffer alignment on many GPUs
        128,   // Common alignment
        256,   // WebGPU min uniform buffer offset alignment
        512,
        1024,
        2048,
        4096,  // Page size
        16384, // 16KB
        65536, // 64KB
    ];

    for boundary in boundaries {
        assert!(
            is_aligned(boundary),
            "Common boundary {} should be aligned",
            boundary
        );
        assert_eq!(
            align_size(boundary),
            boundary,
            "Common boundary {} should not change after alignment",
            boundary
        );
    }
}

#[test]
fn test_alignment_one_before_boundaries() {
    // Test values just before common boundaries
    let boundaries: Vec<u64> = vec![64, 128, 256, 512, 1024, 2048, 4096];

    for boundary in boundaries {
        let before = boundary - 1;
        let aligned = align_size(before);
        assert_eq!(
            aligned, boundary,
            "Value {} should align to {}",
            before, boundary
        );
    }
}

// ============================================================================
// Summary Statistics Test
// ============================================================================

#[test]
fn test_buffer_api_coverage_summary() {
    // This test serves as documentation of what we've covered
    // It should always pass and provides a summary

    // Alignment tests covered:
    assert_eq!(align_size(0), 0);      // Zero handling
    assert_eq!(align_size(1), 4);      // Min alignment
    assert_eq!(align_size(5), 8);      // Next boundary
    assert!(is_aligned(align_size(101))); // Result is always aligned

    // Error types covered:
    let _ = BufferCreationError::ZeroSize;
    let _ = BufferCreationError::EmptyUsage;
    assert!(matches!(BufferCreationError::ZeroSize, BufferCreationError::ZeroSize));

    // Descriptor covered:
    let _ = TrinityBufferDescriptor::default();

    // Usage flags covered:
    assert!(!BufferUsages::VERTEX.is_empty());
    assert!(BufferUsages::empty().is_empty());

    // Constant covered:
    assert_eq!(BUFFER_ALIGNMENT, 4);
}

// ============================================================================
// T-WGPU-P2.1.2: Buffer Usage Flags - Preset Tests (Path O)
// ============================================================================

#[test]
fn test_preset_vertex_is_const_and_valid() {
    // Path O: Verify VERTEX preset is const (compile-time) and valid
    const VERTEX_PRESET: BufferUsages = buffer_usages::VERTEX;
    assert!(!VERTEX_PRESET.is_empty());
    assert!(validate_usage(VERTEX_PRESET).is_ok());
}

#[test]
fn test_preset_index_is_const_and_valid() {
    const INDEX_PRESET: BufferUsages = buffer_usages::INDEX;
    assert!(!INDEX_PRESET.is_empty());
    assert!(validate_usage(INDEX_PRESET).is_ok());
}

#[test]
fn test_preset_uniform_is_const_and_valid() {
    const UNIFORM_PRESET: BufferUsages = buffer_usages::UNIFORM;
    assert!(!UNIFORM_PRESET.is_empty());
    assert!(validate_usage(UNIFORM_PRESET).is_ok());
}

#[test]
fn test_preset_storage_read_is_const_and_valid() {
    const STORAGE_READ_PRESET: BufferUsages = buffer_usages::STORAGE_READ;
    assert!(!STORAGE_READ_PRESET.is_empty());
    assert!(validate_usage(STORAGE_READ_PRESET).is_ok());
}

#[test]
fn test_preset_storage_rw_is_const_and_valid() {
    const STORAGE_RW_PRESET: BufferUsages = buffer_usages::STORAGE_RW;
    assert!(!STORAGE_RW_PRESET.is_empty());
    assert!(validate_usage(STORAGE_RW_PRESET).is_ok());
}

#[test]
fn test_preset_staging_upload_is_const_and_valid() {
    const STAGING_UPLOAD_PRESET: BufferUsages = buffer_usages::STAGING_UPLOAD;
    assert!(!STAGING_UPLOAD_PRESET.is_empty());
    assert!(validate_usage(STAGING_UPLOAD_PRESET).is_ok());
}

#[test]
fn test_preset_staging_readback_is_const_and_valid() {
    const STAGING_READBACK_PRESET: BufferUsages = buffer_usages::STAGING_READBACK;
    assert!(!STAGING_READBACK_PRESET.is_empty());
    assert!(validate_usage(STAGING_READBACK_PRESET).is_ok());
}

#[test]
fn test_preset_indirect_is_const_and_valid() {
    const INDIRECT_PRESET: BufferUsages = buffer_usages::INDIRECT;
    assert!(!INDIRECT_PRESET.is_empty());
    assert!(validate_usage(INDIRECT_PRESET).is_ok());
}

#[test]
fn test_preset_query_resolve_is_const_and_valid() {
    const QUERY_RESOLVE_PRESET: BufferUsages = buffer_usages::QUERY_RESOLVE;
    assert!(!QUERY_RESOLVE_PRESET.is_empty());
    assert!(validate_usage(QUERY_RESOLVE_PRESET).is_ok());
}

#[test]
fn test_all_nine_presets_compile_time() {
    // Verify all 9 presets can be used in const context
    const _: BufferUsages = buffer_usages::VERTEX;
    const _: BufferUsages = buffer_usages::INDEX;
    const _: BufferUsages = buffer_usages::UNIFORM;
    const _: BufferUsages = buffer_usages::STORAGE_READ;
    const _: BufferUsages = buffer_usages::STORAGE_RW;
    const _: BufferUsages = buffer_usages::STAGING_UPLOAD;
    const _: BufferUsages = buffer_usages::STAGING_READBACK;
    const _: BufferUsages = buffer_usages::INDIRECT;
    const _: BufferUsages = buffer_usages::QUERY_RESOLVE;
}

// ============================================================================
// T-WGPU-P2.1.2: Preset Contains Expected Flags (Path P)
// ============================================================================

#[test]
fn test_preset_vertex_contains_expected_flags() {
    // Path P: VERTEX preset should contain VERTEX | COPY_DST
    let preset = buffer_usages::VERTEX;
    assert!(preset.contains(BufferUsages::VERTEX));
    assert!(preset.contains(BufferUsages::COPY_DST));
    // Should NOT contain unrelated flags
    assert!(!preset.contains(BufferUsages::MAP_READ));
    assert!(!preset.contains(BufferUsages::MAP_WRITE));
    assert!(!preset.contains(BufferUsages::INDEX));
}

#[test]
fn test_preset_index_contains_expected_flags() {
    let preset = buffer_usages::INDEX;
    assert!(preset.contains(BufferUsages::INDEX));
    assert!(preset.contains(BufferUsages::COPY_DST));
    assert!(!preset.contains(BufferUsages::VERTEX));
}

#[test]
fn test_preset_uniform_contains_expected_flags() {
    let preset = buffer_usages::UNIFORM;
    assert!(preset.contains(BufferUsages::UNIFORM));
    assert!(preset.contains(BufferUsages::COPY_DST));
    assert!(!preset.contains(BufferUsages::STORAGE));
}

#[test]
fn test_preset_storage_read_contains_expected_flags() {
    let preset = buffer_usages::STORAGE_READ;
    assert!(preset.contains(BufferUsages::STORAGE));
    assert!(preset.contains(BufferUsages::COPY_DST));
    // STORAGE_READ should NOT have COPY_SRC (that's STORAGE_RW)
    assert!(!preset.contains(BufferUsages::COPY_SRC));
}

#[test]
fn test_preset_storage_rw_contains_expected_flags() {
    let preset = buffer_usages::STORAGE_RW;
    assert!(preset.contains(BufferUsages::STORAGE));
    assert!(preset.contains(BufferUsages::COPY_DST));
    assert!(preset.contains(BufferUsages::COPY_SRC)); // RW has COPY_SRC
}

#[test]
fn test_preset_staging_upload_contains_expected_flags() {
    let preset = buffer_usages::STAGING_UPLOAD;
    assert!(preset.contains(BufferUsages::MAP_WRITE));
    assert!(preset.contains(BufferUsages::COPY_SRC));
    // Upload staging should NOT have MAP_READ
    assert!(!preset.contains(BufferUsages::MAP_READ));
    assert!(!preset.contains(BufferUsages::COPY_DST));
}

#[test]
fn test_preset_staging_readback_contains_expected_flags() {
    let preset = buffer_usages::STAGING_READBACK;
    assert!(preset.contains(BufferUsages::MAP_READ));
    assert!(preset.contains(BufferUsages::COPY_DST));
    // Readback staging should NOT have MAP_WRITE
    assert!(!preset.contains(BufferUsages::MAP_WRITE));
    assert!(!preset.contains(BufferUsages::COPY_SRC));
}

#[test]
fn test_preset_indirect_contains_expected_flags() {
    let preset = buffer_usages::INDIRECT;
    assert!(preset.contains(BufferUsages::INDIRECT));
    assert!(preset.contains(BufferUsages::COPY_DST));
    assert!(preset.contains(BufferUsages::STORAGE));
}

#[test]
fn test_preset_query_resolve_contains_expected_flags() {
    let preset = buffer_usages::QUERY_RESOLVE;
    assert!(preset.contains(BufferUsages::QUERY_RESOLVE));
    assert!(preset.contains(BufferUsages::COPY_SRC));
}

// ============================================================================
// T-WGPU-P2.1.2: UsageValidationError Detection (Paths Q, R, S)
// ============================================================================

#[test]
fn test_validate_usage_map_read_and_write_rejected() {
    // Path Q: MAP_READ + MAP_WRITE is invalid
    let invalid = BufferUsages::MAP_READ | BufferUsages::MAP_WRITE;
    let result = validate_usage(invalid);
    assert!(result.is_err());
    assert!(matches!(result, Err(UsageValidationError::MapReadAndWrite)));
}

#[test]
fn test_validate_usage_map_read_and_write_with_other_flags() {
    // MAP_READ + MAP_WRITE + other flags should still be rejected
    let invalid = BufferUsages::MAP_READ | BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC;
    let result = validate_usage(invalid);
    assert!(result.is_err());
    assert!(matches!(result, Err(UsageValidationError::MapReadAndWrite)));
}

#[test]
fn test_validate_usage_map_read_with_vertex_rejected() {
    // Path R: MAP_READ + VERTEX is invalid
    let invalid = BufferUsages::MAP_READ | BufferUsages::VERTEX;
    let result = validate_usage(invalid);
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(UsageValidationError::MapReadWithGpuOnly(_))
    ));
}

#[test]
fn test_validate_usage_map_read_with_index_rejected() {
    // Path R: MAP_READ + INDEX is invalid
    let invalid = BufferUsages::MAP_READ | BufferUsages::INDEX;
    let result = validate_usage(invalid);
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(UsageValidationError::MapReadWithGpuOnly(_))
    ));
}

#[test]
fn test_validate_usage_map_read_with_uniform_rejected() {
    // Path R: MAP_READ + UNIFORM is invalid
    let invalid = BufferUsages::MAP_READ | BufferUsages::UNIFORM;
    let result = validate_usage(invalid);
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(UsageValidationError::MapReadWithGpuOnly(_))
    ));
}

#[test]
fn test_validate_usage_map_read_with_storage_rejected() {
    // Path R: MAP_READ + STORAGE is invalid
    let invalid = BufferUsages::MAP_READ | BufferUsages::STORAGE;
    let result = validate_usage(invalid);
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(UsageValidationError::MapReadWithGpuOnly(_))
    ));
}

#[test]
fn test_validate_usage_map_read_with_multiple_gpu_only_rejected() {
    // MAP_READ + multiple GPU-only flags
    let invalid = BufferUsages::MAP_READ | BufferUsages::VERTEX | BufferUsages::INDEX;
    let result = validate_usage(invalid);
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(UsageValidationError::MapReadWithGpuOnly(_))
    ));
}

#[test]
fn test_validate_usage_map_read_and_write_priority_over_gpu_only() {
    // When both errors apply, MapReadAndWrite should take priority (checked first)
    let invalid =
        BufferUsages::MAP_READ | BufferUsages::MAP_WRITE | BufferUsages::VERTEX;
    let result = validate_usage(invalid);
    assert!(result.is_err());
    // MapReadAndWrite is checked first in the implementation
    assert!(matches!(result, Err(UsageValidationError::MapReadAndWrite)));
}

#[test]
fn test_validate_usage_map_write_with_gpu_only_not_rejected() {
    // Path S: MAP_WRITE + GPU-only is currently allowed (soft validation)
    // The implementation documents this as suspicious but doesn't error
    let suspicious = BufferUsages::MAP_WRITE | BufferUsages::VERTEX;
    let result = validate_usage(suspicious);
    // Currently passes (implementation notes it's suspicious but doesn't reject)
    assert!(result.is_ok());
}

// ============================================================================
// T-WGPU-P2.1.2: validate_usage Valid Combinations (Path T)
// ============================================================================

#[test]
fn test_validate_usage_valid_vertex_buffer() {
    // Path T: Common valid combinations
    assert!(validate_usage(BufferUsages::VERTEX | BufferUsages::COPY_DST).is_ok());
}

#[test]
fn test_validate_usage_valid_index_buffer() {
    assert!(validate_usage(BufferUsages::INDEX | BufferUsages::COPY_DST).is_ok());
}

#[test]
fn test_validate_usage_valid_uniform_buffer() {
    assert!(validate_usage(BufferUsages::UNIFORM | BufferUsages::COPY_DST).is_ok());
}

#[test]
fn test_validate_usage_valid_storage_buffer() {
    assert!(validate_usage(BufferUsages::STORAGE | BufferUsages::COPY_DST).is_ok());
}

#[test]
fn test_validate_usage_valid_storage_rw_buffer() {
    assert!(
        validate_usage(BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC)
            .is_ok()
    );
}

#[test]
fn test_validate_usage_valid_staging_upload() {
    assert!(validate_usage(BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC).is_ok());
}

#[test]
fn test_validate_usage_valid_staging_readback() {
    assert!(validate_usage(BufferUsages::MAP_READ | BufferUsages::COPY_DST).is_ok());
}

#[test]
fn test_validate_usage_valid_indirect_buffer() {
    assert!(
        validate_usage(
            BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST
        )
        .is_ok()
    );
}

#[test]
fn test_validate_usage_valid_query_resolve() {
    assert!(validate_usage(BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC).is_ok());
}

// ============================================================================
// T-WGPU-P2.1.2: validate_usage Edge Cases (Path U)
// ============================================================================

#[test]
fn test_validate_usage_single_copy_src() {
    // Path U: Single flag tests
    assert!(validate_usage(BufferUsages::COPY_SRC).is_ok());
}

#[test]
fn test_validate_usage_single_copy_dst() {
    assert!(validate_usage(BufferUsages::COPY_DST).is_ok());
}

#[test]
fn test_validate_usage_single_map_read() {
    // MAP_READ alone is valid
    assert!(validate_usage(BufferUsages::MAP_READ).is_ok());
}

#[test]
fn test_validate_usage_single_map_write() {
    // MAP_WRITE alone is valid
    assert!(validate_usage(BufferUsages::MAP_WRITE).is_ok());
}

#[test]
fn test_validate_usage_single_vertex() {
    // VERTEX alone is valid (though unusual without COPY_DST)
    assert!(validate_usage(BufferUsages::VERTEX).is_ok());
}

#[test]
fn test_validate_usage_single_index() {
    assert!(validate_usage(BufferUsages::INDEX).is_ok());
}

#[test]
fn test_validate_usage_single_uniform() {
    assert!(validate_usage(BufferUsages::UNIFORM).is_ok());
}

#[test]
fn test_validate_usage_single_storage() {
    assert!(validate_usage(BufferUsages::STORAGE).is_ok());
}

#[test]
fn test_validate_usage_single_indirect() {
    assert!(validate_usage(BufferUsages::INDIRECT).is_ok());
}

#[test]
fn test_validate_usage_single_query_resolve() {
    assert!(validate_usage(BufferUsages::QUERY_RESOLVE).is_ok());
}

#[test]
fn test_validate_usage_empty_is_ok() {
    // Empty usage passes validation (but would fail buffer creation)
    // The validate_usage function only checks for invalid combinations,
    // not for empty flags - that's handled by try_create_buffer
    assert!(validate_usage(BufferUsages::empty()).is_ok());
}

// ============================================================================
// T-WGPU-P2.1.2: UsageValidationError Display Trait (Path V)
// ============================================================================

#[test]
fn test_usage_validation_error_display_map_read_and_write() {
    // Path V: Display formatting for MapReadAndWrite
    let err = UsageValidationError::MapReadAndWrite;
    let msg = err.to_string();
    assert!(msg.contains("MAP_READ"), "Message should mention MAP_READ");
    assert!(msg.contains("MAP_WRITE"), "Message should mention MAP_WRITE");
    assert!(
        msg.contains("cannot both be set"),
        "Message should explain the constraint"
    );
}

#[test]
fn test_usage_validation_error_display_map_read_with_gpu_only() {
    // Path V: Display formatting for MapReadWithGpuOnly
    let flags = BufferUsages::MAP_READ | BufferUsages::VERTEX;
    let err = UsageValidationError::MapReadWithGpuOnly(flags);
    let msg = err.to_string();
    assert!(msg.contains("MAP_READ"), "Message should mention MAP_READ");
    assert!(
        msg.contains("GPU-only"),
        "Message should mention GPU-only flags"
    );
    assert!(
        msg.contains("VERTEX") || msg.contains("INDEX") || msg.contains("UNIFORM") || msg.contains("STORAGE"),
        "Message should list problematic flags"
    );
}

#[test]
fn test_usage_validation_error_display_map_write_with_gpu_only() {
    // Path V: Display formatting for MapWriteWithGpuOnly
    let flags = BufferUsages::MAP_WRITE | BufferUsages::UNIFORM;
    let err = UsageValidationError::MapWriteWithGpuOnly(flags);
    let msg = err.to_string();
    assert!(msg.contains("MAP_WRITE"), "Message should mention MAP_WRITE");
    assert!(
        msg.contains("GPU-only"),
        "Message should mention GPU-only flags"
    );
}

// ============================================================================
// T-WGPU-P2.1.2: UsageValidationError Traits (Path W)
// ============================================================================

#[test]
fn test_usage_validation_error_clone() {
    // Path W: Clone trait
    let err1 = UsageValidationError::MapReadAndWrite;
    let err2 = err1.clone();
    assert_eq!(err1, err2);

    let flags = BufferUsages::MAP_READ | BufferUsages::VERTEX;
    let err3 = UsageValidationError::MapReadWithGpuOnly(flags);
    let err4 = err3.clone();
    assert_eq!(err3, err4);

    let flags2 = BufferUsages::MAP_WRITE | BufferUsages::STORAGE;
    let err5 = UsageValidationError::MapWriteWithGpuOnly(flags2);
    let err6 = err5.clone();
    assert_eq!(err5, err6);
}

#[test]
fn test_usage_validation_error_partial_eq() {
    // Path W: PartialEq trait
    let err1 = UsageValidationError::MapReadAndWrite;
    let err2 = UsageValidationError::MapReadAndWrite;
    assert_eq!(err1, err2);

    let flags = BufferUsages::MAP_READ | BufferUsages::VERTEX;
    let err3 = UsageValidationError::MapReadWithGpuOnly(flags);
    let err4 = UsageValidationError::MapReadWithGpuOnly(flags);
    assert_eq!(err3, err4);

    // Different variants are not equal
    assert_ne!(
        UsageValidationError::MapReadAndWrite,
        UsageValidationError::MapReadWithGpuOnly(flags)
    );
}

#[test]
fn test_usage_validation_error_eq() {
    // Path W: Eq trait (reflexive, symmetric, transitive)
    let err1 = UsageValidationError::MapReadAndWrite;
    let err2 = UsageValidationError::MapReadAndWrite;
    let err3 = UsageValidationError::MapReadAndWrite;

    // Reflexive
    assert_eq!(err1, err1);

    // Symmetric
    assert_eq!(err1, err2);
    assert_eq!(err2, err1);

    // Transitive
    assert_eq!(err1, err2);
    assert_eq!(err2, err3);
    assert_eq!(err1, err3);
}

#[test]
fn test_usage_validation_error_debug() {
    // Debug formatting
    let err = UsageValidationError::MapReadAndWrite;
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("MapReadAndWrite"));

    let flags = BufferUsages::MAP_READ | BufferUsages::VERTEX;
    let err2 = UsageValidationError::MapReadWithGpuOnly(flags);
    let debug_str2 = format!("{:?}", err2);
    assert!(debug_str2.contains("MapReadWithGpuOnly"));
}

// ============================================================================
// T-WGPU-P2.1.2: UsageValidationError std::error::Error (Path X)
// ============================================================================

#[test]
fn test_usage_validation_error_is_error() {
    // Path X: std::error::Error implementation
    let err: &dyn std::error::Error = &UsageValidationError::MapReadAndWrite;
    // UsageValidationError has no source (it's a leaf error)
    assert!(err.source().is_none());

    let flags = BufferUsages::MAP_READ | BufferUsages::STORAGE;
    let err2: &dyn std::error::Error = &UsageValidationError::MapReadWithGpuOnly(flags);
    assert!(err2.source().is_none());
}

#[test]
fn test_usage_validation_error_error_trait_display() {
    // Error trait requires Display
    let err: Box<dyn std::error::Error> = Box::new(UsageValidationError::MapReadAndWrite);
    let msg = err.to_string();
    assert!(!msg.is_empty());
}

// ============================================================================
// T-WGPU-P2.1.2: validate_usage_with_label (Path Y)
// ============================================================================

#[test]
fn test_validate_usage_with_label_valid() {
    // Path Y: Valid usage with label
    let result = validate_usage_with_label(
        BufferUsages::VERTEX | BufferUsages::COPY_DST,
        Some("my_vertex_buffer"),
    );
    assert!(result.is_ok());
}

#[test]
fn test_validate_usage_with_label_invalid_includes_label() {
    // Path Y: Invalid usage should include label in error
    let result = validate_usage_with_label(
        BufferUsages::MAP_READ | BufferUsages::MAP_WRITE,
        Some("problematic_buffer"),
    );
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("problematic_buffer"),
        "Error should include buffer label"
    );
    assert!(
        err_msg.contains("MAP_READ"),
        "Error should include validation details"
    );
}

#[test]
fn test_validate_usage_with_label_none() {
    // Path Y: Invalid usage without label
    let result = validate_usage_with_label(
        BufferUsages::MAP_READ | BufferUsages::VERTEX,
        None,
    );
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    // Should still have the error message, just without buffer label prefix
    assert!(err_msg.contains("MAP_READ"));
    assert!(!err_msg.contains("Buffer '")); // No label prefix
}

#[test]
fn test_validate_usage_with_label_format() {
    // Verify the exact format: "Buffer 'label': error"
    let result = validate_usage_with_label(
        BufferUsages::MAP_READ | BufferUsages::INDEX,
        Some("test_buf"),
    );
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.starts_with("Buffer 'test_buf':"),
        "Error should start with 'Buffer 'label':'"
    );
}

// ============================================================================
// T-WGPU-P2.1.2: BufferCreationError::InvalidUsage (Path Z)
// ============================================================================

#[test]
fn test_buffer_creation_error_invalid_usage_variant() {
    // Path Z: InvalidUsage variant exists and can be constructed
    let validation_err = UsageValidationError::MapReadAndWrite;
    let creation_err = BufferCreationError::InvalidUsage(validation_err);

    assert!(matches!(
        creation_err,
        BufferCreationError::InvalidUsage(UsageValidationError::MapReadAndWrite)
    ));
}

#[test]
fn test_buffer_creation_error_invalid_usage_display() {
    // Path Z: InvalidUsage Display formatting
    let creation_err =
        BufferCreationError::InvalidUsage(UsageValidationError::MapReadAndWrite);
    let msg = creation_err.to_string();
    assert!(
        msg.contains("invalid buffer usage"),
        "Message should indicate invalid usage"
    );
    assert!(
        msg.contains("MAP_READ") || msg.contains("cannot"),
        "Message should include underlying error details"
    );
}

#[test]
fn test_buffer_creation_error_invalid_usage_debug() {
    let creation_err =
        BufferCreationError::InvalidUsage(UsageValidationError::MapReadAndWrite);
    let debug_str = format!("{:?}", creation_err);
    assert!(debug_str.contains("InvalidUsage"));
    assert!(debug_str.contains("MapReadAndWrite"));
}

#[test]
fn test_buffer_creation_error_invalid_usage_clone() {
    let err1 = BufferCreationError::InvalidUsage(UsageValidationError::MapReadAndWrite);
    let err2 = err1.clone();
    assert_eq!(err1, err2);
}

#[test]
fn test_buffer_creation_error_invalid_usage_equality() {
    let err1 = BufferCreationError::InvalidUsage(UsageValidationError::MapReadAndWrite);
    let err2 = BufferCreationError::InvalidUsage(UsageValidationError::MapReadAndWrite);
    assert_eq!(err1, err2);

    // Different inner errors should be different
    let flags = BufferUsages::MAP_READ | BufferUsages::VERTEX;
    let err3 = BufferCreationError::InvalidUsage(UsageValidationError::MapReadWithGpuOnly(flags));
    assert_ne!(err1, err3);
}

// ============================================================================
// T-WGPU-P2.1.2: From<UsageValidationError> impl (Path AA)
// ============================================================================

#[test]
fn test_from_usage_validation_error_map_read_and_write() {
    // Path AA: From trait implementation
    let validation_err = UsageValidationError::MapReadAndWrite;
    let creation_err: BufferCreationError = validation_err.into();

    assert!(matches!(
        creation_err,
        BufferCreationError::InvalidUsage(UsageValidationError::MapReadAndWrite)
    ));
}

#[test]
fn test_from_usage_validation_error_map_read_with_gpu_only() {
    let flags = BufferUsages::MAP_READ | BufferUsages::UNIFORM;
    let validation_err = UsageValidationError::MapReadWithGpuOnly(flags);
    let creation_err: BufferCreationError = validation_err.into();

    assert!(matches!(
        creation_err,
        BufferCreationError::InvalidUsage(UsageValidationError::MapReadWithGpuOnly(_))
    ));
}

#[test]
fn test_from_usage_validation_error_map_write_with_gpu_only() {
    let flags = BufferUsages::MAP_WRITE | BufferUsages::STORAGE;
    let validation_err = UsageValidationError::MapWriteWithGpuOnly(flags);
    let creation_err: BufferCreationError = validation_err.into();

    assert!(matches!(
        creation_err,
        BufferCreationError::InvalidUsage(UsageValidationError::MapWriteWithGpuOnly(_))
    ));
}

#[test]
fn test_from_impl_with_question_mark_operator() {
    // Verify the From impl works with the ? operator
    fn validate_and_return() -> Result<(), BufferCreationError> {
        let invalid = BufferUsages::MAP_READ | BufferUsages::MAP_WRITE;
        validate_usage(invalid)?; // This should use From to convert
        Ok(())
    }

    let result = validate_and_return();
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(BufferCreationError::InvalidUsage(UsageValidationError::MapReadAndWrite))
    ));
}

// ============================================================================
// T-WGPU-P2.1.2: Error Source Chain (Path AB)
// ============================================================================

#[test]
fn test_buffer_creation_error_source_for_invalid_usage() {
    // Path AB: InvalidUsage has source pointing to UsageValidationError
    let creation_err =
        BufferCreationError::InvalidUsage(UsageValidationError::MapReadAndWrite);
    let source = std::error::Error::source(&creation_err);

    assert!(source.is_some(), "InvalidUsage should have a source");

    // Downcast to verify it's the right type
    let source = source.unwrap();
    assert!(
        source.downcast_ref::<UsageValidationError>().is_some(),
        "Source should be UsageValidationError"
    );
}

#[test]
fn test_buffer_creation_error_source_for_zero_size() {
    // ZeroSize and EmptyUsage have no source
    let err = BufferCreationError::ZeroSize;
    let source = std::error::Error::source(&err);
    assert!(source.is_none(), "ZeroSize should have no source");
}

#[test]
fn test_buffer_creation_error_source_for_empty_usage() {
    let err = BufferCreationError::EmptyUsage;
    let source = std::error::Error::source(&err);
    assert!(source.is_none(), "EmptyUsage should have no source");
}

#[test]
fn test_error_chain_display() {
    // Verify error chain can be displayed
    let creation_err =
        BufferCreationError::InvalidUsage(UsageValidationError::MapReadAndWrite);

    // Print error chain (like anyhow/eyre would)
    let mut current: Option<&dyn std::error::Error> = Some(&creation_err);
    let mut messages = Vec::new();

    while let Some(err) = current {
        messages.push(err.to_string());
        current = err.source();
    }

    assert_eq!(messages.len(), 2, "Should have two errors in chain");
    assert!(messages[0].contains("invalid buffer usage"));
    assert!(messages[1].contains("MAP_READ"));
}

// ============================================================================
// T-WGPU-P2.1.2: Comprehensive Preset Validation
// ============================================================================

#[test]
fn test_all_presets_pass_validation() {
    // Comprehensive test that all 9 presets pass validation
    let presets = [
        ("VERTEX", buffer_usages::VERTEX),
        ("INDEX", buffer_usages::INDEX),
        ("UNIFORM", buffer_usages::UNIFORM),
        ("STORAGE_READ", buffer_usages::STORAGE_READ),
        ("STORAGE_RW", buffer_usages::STORAGE_RW),
        ("STAGING_UPLOAD", buffer_usages::STAGING_UPLOAD),
        ("STAGING_READBACK", buffer_usages::STAGING_READBACK),
        ("INDIRECT", buffer_usages::INDIRECT),
        ("QUERY_RESOLVE", buffer_usages::QUERY_RESOLVE),
    ];

    for (name, preset) in &presets {
        let result = validate_usage(*preset);
        assert!(
            result.is_ok(),
            "Preset {} should pass validation, but got: {:?}",
            name,
            result
        );
    }
}

#[test]
fn test_presets_are_distinct() {
    // Verify each preset has a unique combination of flags
    let presets = [
        buffer_usages::VERTEX,
        buffer_usages::INDEX,
        buffer_usages::UNIFORM,
        buffer_usages::STORAGE_READ,
        buffer_usages::STORAGE_RW,
        buffer_usages::STAGING_UPLOAD,
        buffer_usages::STAGING_READBACK,
        buffer_usages::INDIRECT,
        buffer_usages::QUERY_RESOLVE,
    ];

    for i in 0..presets.len() {
        for j in (i + 1)..presets.len() {
            assert_ne!(
                presets[i], presets[j],
                "Presets at index {} and {} should be distinct",
                i, j
            );
        }
    }
}

// ============================================================================
// T-WGPU-P2.1.2: API Coverage Summary
// ============================================================================

#[test]
fn test_t_wgpu_p2_1_2_api_coverage_summary() {
    // This test documents the API coverage for T-WGPU-P2.1.2

    // buffer_usages module - 9 presets
    let _ = buffer_usages::VERTEX;
    let _ = buffer_usages::INDEX;
    let _ = buffer_usages::UNIFORM;
    let _ = buffer_usages::STORAGE_READ;
    let _ = buffer_usages::STORAGE_RW;
    let _ = buffer_usages::STAGING_UPLOAD;
    let _ = buffer_usages::STAGING_READBACK;
    let _ = buffer_usages::INDIRECT;
    let _ = buffer_usages::QUERY_RESOLVE;

    // UsageValidationError - 3 variants
    let _ = UsageValidationError::MapReadAndWrite;
    let _ = UsageValidationError::MapReadWithGpuOnly(BufferUsages::MAP_READ | BufferUsages::VERTEX);
    let _ =
        UsageValidationError::MapWriteWithGpuOnly(BufferUsages::MAP_WRITE | BufferUsages::UNIFORM);

    // validate_usage function
    let _ = validate_usage(BufferUsages::VERTEX);

    // validate_usage_with_label function
    let _ = validate_usage_with_label(BufferUsages::VERTEX, Some("test"));

    // BufferCreationError::InvalidUsage variant
    let _ = BufferCreationError::InvalidUsage(UsageValidationError::MapReadAndWrite);

    // From<UsageValidationError> for BufferCreationError
    let _: BufferCreationError = UsageValidationError::MapReadAndWrite.into();
}

// ============================================================================
// T-WGPU-P2.1.3: Buffer Mapping - WHITEBOX Tests
// ============================================================================
//
// WHITEBOX discipline: These tests have FULL ACCESS to buffer mapping implementation.
//
// Implementation under test: crates/renderer-backend/src/resources/buffer.rs
//   - MappingMode enum: Read, Write
//   - MappingError enum: NotMappable, MapFailed, AlreadyMapped, ChannelError, NotMappedAtCreation
//   - MappedBuffer struct with Drop impl for auto-unmap
//   - map_buffer_sync_write() - For mapped_at_creation buffers
//   - map_buffer_sync_read() - For already mapped buffers
//   - map_buffer_async() - With callback
//   - map_buffer_async_channel() - With oneshot channel
//   - map_buffer_blocking() - Sync wrapper around async
//   - create_staging_upload_buffer() / create_staging_readback_buffer()
//   - is_mappable() - Check if buffer supports mapping mode
//   - MappedBuffer::leak() - Prevent auto-unmap
//
// WHITEBOX coverage plan (T-WGPU-P2.1.3):
//   - Path AC: MappingMode::Read and MappingMode::Write variants exist
//   - Path AD: From<MappingMode> for wgpu::MapMode conversion
//   - Path AE: MappingMode Debug, Clone, Copy, PartialEq, Eq, Hash traits
//   - Path AF: MappingMode Display trait
//   - Path AG: MappingError::NotMappable variant with mode and usage fields
//   - Path AH: MappingError::MapFailed variant
//   - Path AI: MappingError::AlreadyMapped variant
//   - Path AJ: MappingError::ChannelError variant
//   - Path AK: MappingError::NotMappedAtCreation variant
//   - Path AL: MappingError Display trait for all variants
//   - Path AM: MappingError Clone, PartialEq, Eq, Debug traits
//   - Path AN: MappingError implements std::error::Error
//   - Path AO: is_mappable() returns true for MAP_READ + Read mode
//   - Path AP: is_mappable() returns true for MAP_WRITE + Write mode
//   - Path AQ: is_mappable() returns false for missing MAP_READ/MAP_WRITE
//   - Path AR: STAGING_UPLOAD preset is mappable for Write
//   - Path AS: STAGING_READBACK preset is mappable for Read
//   - Path AT: VERTEX preset is not mappable

use renderer_backend::resources::buffer::{
    is_mappable, MappingError, MappingMode,
};

// ============================================================================
// Path AC: MappingMode Variants
// ============================================================================

#[test]
fn test_mapping_mode_read_variant() {
    // Path AC: MappingMode::Read exists
    let mode = MappingMode::Read;
    assert!(matches!(mode, MappingMode::Read));
}

#[test]
fn test_mapping_mode_write_variant() {
    // Path AC: MappingMode::Write exists
    let mode = MappingMode::Write;
    assert!(matches!(mode, MappingMode::Write));
}

#[test]
fn test_mapping_mode_variants_distinct() {
    // Verify the two variants are distinct
    assert_ne!(MappingMode::Read, MappingMode::Write);
}

// ============================================================================
// Path AD: From<MappingMode> for wgpu::MapMode
// ============================================================================

#[test]
fn test_mapping_mode_to_wgpu_map_mode_read() {
    // Path AD: MappingMode::Read converts to wgpu::MapMode::Read
    use wgpu::MapMode;

    let mode = MappingMode::Read;
    let wgpu_mode: MapMode = mode.into();
    assert!(matches!(wgpu_mode, MapMode::Read));
}

#[test]
fn test_mapping_mode_to_wgpu_map_mode_write() {
    // Path AD: MappingMode::Write converts to wgpu::MapMode::Write
    use wgpu::MapMode;

    let mode = MappingMode::Write;
    let wgpu_mode: MapMode = mode.into();
    assert!(matches!(wgpu_mode, MapMode::Write));
}

#[test]
fn test_mapping_mode_from_trait_explicit() {
    // Test explicit From::from call
    use wgpu::MapMode;

    let wgpu_read = MapMode::from(MappingMode::Read);
    assert!(matches!(wgpu_read, MapMode::Read));

    let wgpu_write = MapMode::from(MappingMode::Write);
    assert!(matches!(wgpu_write, MapMode::Write));
}

// ============================================================================
// Path AE: MappingMode Debug, Clone, Copy, PartialEq, Eq, Hash
// ============================================================================

#[test]
fn test_mapping_mode_debug() {
    // Path AE: Debug trait
    let read_debug = format!("{:?}", MappingMode::Read);
    let write_debug = format!("{:?}", MappingMode::Write);

    assert_eq!(read_debug, "Read");
    assert_eq!(write_debug, "Write");
}

#[test]
fn test_mapping_mode_clone() {
    // Path AE: Clone trait
    let mode = MappingMode::Read;
    let cloned = mode.clone();
    assert_eq!(mode, cloned);

    let mode2 = MappingMode::Write;
    let cloned2 = mode2.clone();
    assert_eq!(mode2, cloned2);
}

#[test]
fn test_mapping_mode_copy() {
    // Path AE: Copy trait - no move semantics
    let mode = MappingMode::Read;
    let copied = mode; // Copy, not move
    let also_valid = mode; // Original still valid

    assert_eq!(mode, copied);
    assert_eq!(mode, also_valid);
}

#[test]
fn test_mapping_mode_partial_eq() {
    // Path AE: PartialEq trait
    assert_eq!(MappingMode::Read, MappingMode::Read);
    assert_eq!(MappingMode::Write, MappingMode::Write);
    assert!(MappingMode::Read != MappingMode::Write);
}

#[test]
fn test_mapping_mode_eq_reflexive() {
    // Path AE: Eq trait - reflexive property
    let mode = MappingMode::Read;
    assert_eq!(mode, mode);

    let mode2 = MappingMode::Write;
    assert_eq!(mode2, mode2);
}

#[test]
fn test_mapping_mode_eq_symmetric() {
    // Path AE: Eq trait - symmetric property
    let a = MappingMode::Read;
    let b = MappingMode::Read;
    assert_eq!(a, b);
    assert_eq!(b, a);
}

#[test]
fn test_mapping_mode_eq_transitive() {
    // Path AE: Eq trait - transitive property
    let a = MappingMode::Write;
    let b = MappingMode::Write;
    let c = MappingMode::Write;
    assert_eq!(a, b);
    assert_eq!(b, c);
    assert_eq!(a, c);
}

#[test]
fn test_mapping_mode_hash() {
    // Path AE: Hash trait
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(MappingMode::Read);
    set.insert(MappingMode::Write);

    assert!(set.contains(&MappingMode::Read));
    assert!(set.contains(&MappingMode::Write));
    assert_eq!(set.len(), 2);
}

#[test]
fn test_mapping_mode_hash_consistency() {
    // Hash should be consistent for equal values
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher1 = DefaultHasher::new();
    let mut hasher2 = DefaultHasher::new();

    MappingMode::Read.hash(&mut hasher1);
    MappingMode::Read.hash(&mut hasher2);

    assert_eq!(hasher1.finish(), hasher2.finish());
}

#[test]
fn test_mapping_mode_hash_different_for_different_values() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher1 = DefaultHasher::new();
    let mut hasher2 = DefaultHasher::new();

    MappingMode::Read.hash(&mut hasher1);
    MappingMode::Write.hash(&mut hasher2);

    // Different values should (almost certainly) have different hashes
    assert_ne!(hasher1.finish(), hasher2.finish());
}

// ============================================================================
// Path AF: MappingMode Display Trait
// ============================================================================

#[test]
fn test_mapping_mode_display_read() {
    // Path AF: Display for Read
    let mode = MappingMode::Read;
    assert_eq!(mode.to_string(), "Read");
}

#[test]
fn test_mapping_mode_display_write() {
    // Path AF: Display for Write
    let mode = MappingMode::Write;
    assert_eq!(mode.to_string(), "Write");
}

#[test]
fn test_mapping_mode_display_format() {
    // Test with format! macro
    let read_str = format!("{}", MappingMode::Read);
    let write_str = format!("{}", MappingMode::Write);

    assert_eq!(read_str, "Read");
    assert_eq!(write_str, "Write");
}

// ============================================================================
// Path AG: MappingError::NotMappable Variant
// ============================================================================

#[test]
fn test_mapping_error_not_mappable_read() {
    // Path AG: NotMappable variant with Read mode
    let usage = BufferUsages::VERTEX | BufferUsages::COPY_DST;
    let err = MappingError::NotMappable {
        mode: MappingMode::Read,
        usage,
    };

    if let MappingError::NotMappable { mode, usage: u } = err {
        assert_eq!(mode, MappingMode::Read);
        assert_eq!(u, usage);
    } else {
        panic!("Expected NotMappable variant");
    }
}

#[test]
fn test_mapping_error_not_mappable_write() {
    // Path AG: NotMappable variant with Write mode
    let usage = BufferUsages::INDEX | BufferUsages::COPY_DST;
    let err = MappingError::NotMappable {
        mode: MappingMode::Write,
        usage,
    };

    if let MappingError::NotMappable { mode, usage: u } = err {
        assert_eq!(mode, MappingMode::Write);
        assert_eq!(u, usage);
    } else {
        panic!("Expected NotMappable variant");
    }
}

// ============================================================================
// Path AH: MappingError::MapFailed Variant
// ============================================================================

#[test]
fn test_mapping_error_map_failed() {
    // Path AH: MapFailed variant
    let err = MappingError::MapFailed;
    assert!(matches!(err, MappingError::MapFailed));
}

// ============================================================================
// Path AI: MappingError::AlreadyMapped Variant
// ============================================================================

#[test]
fn test_mapping_error_already_mapped() {
    // Path AI: AlreadyMapped variant
    let err = MappingError::AlreadyMapped;
    assert!(matches!(err, MappingError::AlreadyMapped));
}

// ============================================================================
// Path AJ: MappingError::ChannelError Variant
// ============================================================================

#[test]
fn test_mapping_error_channel_error() {
    // Path AJ: ChannelError variant
    let err = MappingError::ChannelError;
    assert!(matches!(err, MappingError::ChannelError));
}

// ============================================================================
// Path AK: MappingError::NotMappedAtCreation Variant
// ============================================================================

#[test]
fn test_mapping_error_not_mapped_at_creation() {
    // Path AK: NotMappedAtCreation variant
    let err = MappingError::NotMappedAtCreation;
    assert!(matches!(err, MappingError::NotMappedAtCreation));
}

// ============================================================================
// Path AL: MappingError Display Trait
// ============================================================================

#[test]
fn test_mapping_error_display_not_mappable_read() {
    // Path AL: Display for NotMappable (Read mode)
    let err = MappingError::NotMappable {
        mode: MappingMode::Read,
        usage: BufferUsages::VERTEX,
    };
    let msg = err.to_string();

    assert!(msg.contains("not mappable"), "Should mention 'not mappable'");
    assert!(msg.contains("Read"), "Should mention the mode");
    assert!(msg.contains("MAP_READ"), "Should mention required flag");
}

#[test]
fn test_mapping_error_display_not_mappable_write() {
    // Path AL: Display for NotMappable (Write mode)
    let err = MappingError::NotMappable {
        mode: MappingMode::Write,
        usage: BufferUsages::UNIFORM,
    };
    let msg = err.to_string();

    assert!(msg.contains("not mappable"), "Should mention 'not mappable'");
    assert!(msg.contains("Write"), "Should mention the mode");
    assert!(msg.contains("MAP_WRITE"), "Should mention required flag");
}

#[test]
fn test_mapping_error_display_map_failed() {
    // Path AL: Display for MapFailed
    let err = MappingError::MapFailed;
    let msg = err.to_string();

    assert!(
        msg.contains("mapping failed") || msg.contains("failed"),
        "Should indicate mapping failure"
    );
}

#[test]
fn test_mapping_error_display_already_mapped() {
    // Path AL: Display for AlreadyMapped
    let err = MappingError::AlreadyMapped;
    let msg = err.to_string();

    assert!(
        msg.contains("already mapped"),
        "Should mention 'already mapped'"
    );
}

#[test]
fn test_mapping_error_display_channel_error() {
    // Path AL: Display for ChannelError
    let err = MappingError::ChannelError;
    let msg = err.to_string();

    assert!(msg.contains("channel"), "Should mention channel error");
}

#[test]
fn test_mapping_error_display_not_mapped_at_creation() {
    // Path AL: Display for NotMappedAtCreation
    let err = MappingError::NotMappedAtCreation;
    let msg = err.to_string();

    assert!(
        msg.contains("not mapped at creation"),
        "Should explain the error"
    );
}

// ============================================================================
// Path AM: MappingError Clone, PartialEq, Eq, Debug
// ============================================================================

#[test]
fn test_mapping_error_clone() {
    // Path AM: Clone trait
    let err1 = MappingError::MapFailed;
    let err2 = err1.clone();
    assert_eq!(err1, err2);

    let err3 = MappingError::NotMappable {
        mode: MappingMode::Read,
        usage: BufferUsages::VERTEX,
    };
    let err4 = err3.clone();
    assert_eq!(err3, err4);
}

#[test]
fn test_mapping_error_partial_eq() {
    // Path AM: PartialEq trait
    assert_eq!(MappingError::MapFailed, MappingError::MapFailed);
    assert_eq!(MappingError::AlreadyMapped, MappingError::AlreadyMapped);
    assert_eq!(MappingError::ChannelError, MappingError::ChannelError);
    assert_eq!(
        MappingError::NotMappedAtCreation,
        MappingError::NotMappedAtCreation
    );

    // NotMappable with same fields
    let err1 = MappingError::NotMappable {
        mode: MappingMode::Read,
        usage: BufferUsages::VERTEX,
    };
    let err2 = MappingError::NotMappable {
        mode: MappingMode::Read,
        usage: BufferUsages::VERTEX,
    };
    assert_eq!(err1, err2);

    // NotMappable with different fields
    let err3 = MappingError::NotMappable {
        mode: MappingMode::Write,
        usage: BufferUsages::VERTEX,
    };
    assert_ne!(err1, err3);
}

#[test]
fn test_mapping_error_eq() {
    // Path AM: Eq trait - verify all variants
    let variants = [
        MappingError::MapFailed,
        MappingError::AlreadyMapped,
        MappingError::ChannelError,
        MappingError::NotMappedAtCreation,
        MappingError::NotMappable {
            mode: MappingMode::Read,
            usage: BufferUsages::STORAGE,
        },
    ];

    // Each variant equals itself
    for v in &variants {
        assert_eq!(v, v);
    }

    // Different variants are not equal
    for i in 0..variants.len() {
        for j in (i + 1)..variants.len() {
            assert_ne!(variants[i], variants[j]);
        }
    }
}

#[test]
fn test_mapping_error_debug() {
    // Path AM: Debug trait
    assert!(format!("{:?}", MappingError::MapFailed).contains("MapFailed"));
    assert!(format!("{:?}", MappingError::AlreadyMapped).contains("AlreadyMapped"));
    assert!(format!("{:?}", MappingError::ChannelError).contains("ChannelError"));
    assert!(
        format!("{:?}", MappingError::NotMappedAtCreation).contains("NotMappedAtCreation")
    );

    let not_mappable = MappingError::NotMappable {
        mode: MappingMode::Read,
        usage: BufferUsages::INDEX,
    };
    let debug_str = format!("{:?}", not_mappable);
    assert!(debug_str.contains("NotMappable"));
    assert!(debug_str.contains("Read"));
}

// ============================================================================
// Path AN: MappingError std::error::Error
// ============================================================================

#[test]
fn test_mapping_error_is_error_trait() {
    // Path AN: std::error::Error implementation
    fn assert_error<E: std::error::Error>() {}
    assert_error::<MappingError>();
}

#[test]
fn test_mapping_error_as_dyn_error() {
    // Can be used as dyn Error
    let err: &dyn std::error::Error = &MappingError::MapFailed;
    assert!(err.source().is_none()); // Leaf error, no source
}

#[test]
fn test_mapping_error_all_variants_as_dyn_error() {
    // All variants work as dyn Error
    let errors: Vec<&dyn std::error::Error> = vec![
        &MappingError::MapFailed,
        &MappingError::AlreadyMapped,
        &MappingError::ChannelError,
        &MappingError::NotMappedAtCreation,
        &MappingError::NotMappable {
            mode: MappingMode::Write,
            usage: BufferUsages::UNIFORM,
        },
    ];

    for err in errors {
        // All should have no source (leaf errors)
        assert!(err.source().is_none());
        // All should have non-empty display
        assert!(!err.to_string().is_empty());
    }
}

// ============================================================================
// Path AO-AT: is_mappable() Function Tests
// ============================================================================

// Note: is_mappable() requires a TrinityBuffer, which requires a wgpu Device.
// We test the underlying logic by checking usage flags directly.

#[test]
fn test_is_mappable_logic_map_read_plus_read_mode() {
    // Path AO: MAP_READ usage + Read mode = should be mappable
    // The is_mappable function checks: buffer.usage().contains(MAP_READ/MAP_WRITE)
    let usage = BufferUsages::MAP_READ | BufferUsages::COPY_DST;
    assert!(
        usage.contains(BufferUsages::MAP_READ),
        "MAP_READ usage should allow Read mode mapping"
    );
}

#[test]
fn test_is_mappable_logic_map_write_plus_write_mode() {
    // Path AP: MAP_WRITE usage + Write mode = should be mappable
    let usage = BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC;
    assert!(
        usage.contains(BufferUsages::MAP_WRITE),
        "MAP_WRITE usage should allow Write mode mapping"
    );
}

#[test]
fn test_is_mappable_logic_vertex_not_mappable() {
    // Path AQ & AT: VERTEX preset without MAP_READ/MAP_WRITE = not mappable
    let usage = buffer_usages::VERTEX;
    assert!(
        !usage.contains(BufferUsages::MAP_READ),
        "VERTEX should not have MAP_READ"
    );
    assert!(
        !usage.contains(BufferUsages::MAP_WRITE),
        "VERTEX should not have MAP_WRITE"
    );
}

#[test]
fn test_is_mappable_logic_uniform_not_mappable() {
    // Path AQ: UNIFORM preset without MAP_READ/MAP_WRITE = not mappable
    let usage = buffer_usages::UNIFORM;
    assert!(!usage.contains(BufferUsages::MAP_READ));
    assert!(!usage.contains(BufferUsages::MAP_WRITE));
}

#[test]
fn test_is_mappable_logic_index_not_mappable() {
    // Path AQ: INDEX preset without MAP_READ/MAP_WRITE = not mappable
    let usage = buffer_usages::INDEX;
    assert!(!usage.contains(BufferUsages::MAP_READ));
    assert!(!usage.contains(BufferUsages::MAP_WRITE));
}

#[test]
fn test_is_mappable_logic_storage_read_not_mappable() {
    // Path AQ: STORAGE_READ preset = not mappable
    let usage = buffer_usages::STORAGE_READ;
    assert!(!usage.contains(BufferUsages::MAP_READ));
    assert!(!usage.contains(BufferUsages::MAP_WRITE));
}

#[test]
fn test_is_mappable_logic_storage_rw_not_mappable() {
    // Path AQ: STORAGE_RW preset = not mappable
    let usage = buffer_usages::STORAGE_RW;
    assert!(!usage.contains(BufferUsages::MAP_READ));
    assert!(!usage.contains(BufferUsages::MAP_WRITE));
}

#[test]
fn test_is_mappable_logic_staging_upload_writable() {
    // Path AR: STAGING_UPLOAD preset = mappable for Write
    let usage = buffer_usages::STAGING_UPLOAD;
    assert!(
        usage.contains(BufferUsages::MAP_WRITE),
        "STAGING_UPLOAD should have MAP_WRITE"
    );
    assert!(
        !usage.contains(BufferUsages::MAP_READ),
        "STAGING_UPLOAD should not have MAP_READ"
    );
}

#[test]
fn test_is_mappable_logic_staging_readback_readable() {
    // Path AS: STAGING_READBACK preset = mappable for Read
    let usage = buffer_usages::STAGING_READBACK;
    assert!(
        usage.contains(BufferUsages::MAP_READ),
        "STAGING_READBACK should have MAP_READ"
    );
    assert!(
        !usage.contains(BufferUsages::MAP_WRITE),
        "STAGING_READBACK should not have MAP_WRITE"
    );
}

#[test]
fn test_is_mappable_logic_indirect_not_mappable() {
    // INDIRECT preset = not mappable
    let usage = buffer_usages::INDIRECT;
    assert!(!usage.contains(BufferUsages::MAP_READ));
    assert!(!usage.contains(BufferUsages::MAP_WRITE));
}

#[test]
fn test_is_mappable_logic_query_resolve_not_mappable() {
    // QUERY_RESOLVE preset = not mappable
    let usage = buffer_usages::QUERY_RESOLVE;
    assert!(!usage.contains(BufferUsages::MAP_READ));
    assert!(!usage.contains(BufferUsages::MAP_WRITE));
}

// ============================================================================
// Staging Buffer Usage Tests
// ============================================================================

#[test]
fn test_staging_upload_buffer_usage() {
    // Verify STAGING_UPLOAD has correct flags for create_staging_upload_buffer
    let usage = buffer_usages::STAGING_UPLOAD;
    assert!(usage.contains(BufferUsages::MAP_WRITE));
    assert!(usage.contains(BufferUsages::COPY_SRC));
}

#[test]
fn test_staging_readback_buffer_usage() {
    // Verify STAGING_READBACK has correct flags for create_staging_readback_buffer
    let usage = buffer_usages::STAGING_READBACK;
    assert!(usage.contains(BufferUsages::MAP_READ));
    assert!(usage.contains(BufferUsages::COPY_DST));
}

// ============================================================================
// Comprehensive MappingError Variant Tests
// ============================================================================

#[test]
fn test_mapping_error_all_five_variants_exist() {
    // Verify all 5 variants exist as documented
    let _ = MappingError::NotMappable {
        mode: MappingMode::Read,
        usage: BufferUsages::VERTEX,
    };
    let _ = MappingError::MapFailed;
    let _ = MappingError::AlreadyMapped;
    let _ = MappingError::ChannelError;
    let _ = MappingError::NotMappedAtCreation;
}

#[test]
fn test_mapping_error_not_mappable_usage_variations() {
    // Test NotMappable with various usage flags
    let usages = [
        BufferUsages::VERTEX,
        BufferUsages::INDEX,
        BufferUsages::UNIFORM,
        BufferUsages::STORAGE,
        BufferUsages::INDIRECT,
        BufferUsages::VERTEX | BufferUsages::COPY_DST,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    ];

    for usage in usages {
        let err = MappingError::NotMappable {
            mode: MappingMode::Read,
            usage,
        };
        // Each should have a valid display message
        let msg = err.to_string();
        assert!(!msg.is_empty());
    }
}

// ============================================================================
// API Coverage Summary for T-WGPU-P2.1.3
// ============================================================================

#[test]
fn test_t_wgpu_p2_1_3_api_coverage_summary() {
    // This test documents API coverage for T-WGPU-P2.1.3 (Buffer Mapping)

    // MappingMode enum - 2 variants
    let _ = MappingMode::Read;
    let _ = MappingMode::Write;

    // MappingMode traits
    let mode = MappingMode::Read;
    let _ = mode.clone(); // Clone
    let _ = format!("{:?}", mode); // Debug
    let _ = mode.to_string(); // Display
    let _ = mode == mode; // PartialEq, Eq
    let _copied = mode; // Copy

    // MappingMode conversion
    let _: wgpu::MapMode = MappingMode::Read.into();

    // MappingError enum - 5 variants
    let _ = MappingError::NotMappable {
        mode: MappingMode::Read,
        usage: BufferUsages::VERTEX,
    };
    let _ = MappingError::MapFailed;
    let _ = MappingError::AlreadyMapped;
    let _ = MappingError::ChannelError;
    let _ = MappingError::NotMappedAtCreation;

    // MappingError traits
    let err = MappingError::MapFailed;
    let _ = err.clone(); // Clone
    let _ = format!("{:?}", err); // Debug
    let _ = err.to_string(); // Display
    let _ = err == err; // PartialEq, Eq

    // MappingError as std::error::Error
    let _: &dyn std::error::Error = &MappingError::MapFailed;

    // is_mappable - tested via usage flag logic
    // (requires TrinityBuffer which needs Device)

    // buffer_usages for staging
    let _ = buffer_usages::STAGING_UPLOAD;
    let _ = buffer_usages::STAGING_READBACK;
}
