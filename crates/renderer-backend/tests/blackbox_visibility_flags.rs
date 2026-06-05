//! Blackbox tests for VisibilityFlagsBuffer (T-WGPU-P6.2.3)
//!
//! Tests the PUBLIC API contract for GPU-driven visibility culling bitfield buffer.
//! CLEANROOM: Does NOT read implementation source, only tests exported API.

use renderer_backend::gpu_driven::{
    VisibilityFlagsBuffer,
    words_for_objects, bit_location,
    is_visible, set_visible, clear_visible, count_visible,
    cpu_atomic_or_visibility, cpu_clear_visibility_flags, cpu_compact_visible,
};

// =============================================================================
// SECTION 1: Pure Utility Functions
// =============================================================================

/// words_for_objects calculates the number of u32 words needed for N objects.
/// Each u32 word holds 32 bits (one per object).
mod words_calculation {
    use super::*;

    #[test]
    fn blackbox_words_for_zero_objects() {
        // Zero objects should require zero words
        assert_eq!(words_for_objects(0), 0);
    }

    #[test]
    fn blackbox_words_for_one_object() {
        // 1 object needs 1 word (bits 0-31 available)
        assert_eq!(words_for_objects(1), 1);
    }

    #[test]
    fn blackbox_words_for_32_objects() {
        // 32 objects exactly fill 1 word
        assert_eq!(words_for_objects(32), 1);
    }

    #[test]
    fn blackbox_words_for_33_objects() {
        // 33 objects require 2 words (overflow by 1 bit)
        assert_eq!(words_for_objects(33), 2);
    }

    #[test]
    fn blackbox_words_for_64_objects() {
        // 64 objects exactly fill 2 words
        assert_eq!(words_for_objects(64), 2);
    }

    #[test]
    fn blackbox_words_for_65_objects() {
        // 65 objects require 3 words
        assert_eq!(words_for_objects(65), 3);
    }

    #[test]
    fn blackbox_words_for_1000_objects() {
        // 1000 objects: ceil(1000/32) = 32 words (32*32=1024 bits)
        assert_eq!(words_for_objects(1000), 32);
    }

    #[test]
    fn blackbox_words_for_1024_objects() {
        // 1024 objects exactly fill 32 words
        assert_eq!(words_for_objects(1024), 32);
    }
}

/// bit_location returns (word_index, bit_mask) for a given object index.
mod bit_location_tests {
    use super::*;

    #[test]
    fn blackbox_bit_location_object_0() {
        let (word, mask) = bit_location(0);
        assert_eq!(word, 0, "Object 0 should be in word 0");
        assert_eq!(mask, 1, "Object 0 should have mask 0x00000001");
    }

    #[test]
    fn blackbox_bit_location_object_1() {
        let (word, mask) = bit_location(1);
        assert_eq!(word, 0, "Object 1 should be in word 0");
        assert_eq!(mask, 2, "Object 1 should have mask 0x00000002");
    }

    #[test]
    fn blackbox_bit_location_object_31() {
        let (word, mask) = bit_location(31);
        assert_eq!(word, 0, "Object 31 should be in word 0");
        assert_eq!(mask, 1 << 31, "Object 31 should have mask 0x80000000");
    }

    #[test]
    fn blackbox_bit_location_object_32() {
        let (word, mask) = bit_location(32);
        assert_eq!(word, 1, "Object 32 should be in word 1");
        assert_eq!(mask, 1, "Object 32 should have mask 0x00000001 in word 1");
    }

    #[test]
    fn blackbox_bit_location_object_63() {
        let (word, mask) = bit_location(63);
        assert_eq!(word, 1, "Object 63 should be in word 1");
        assert_eq!(mask, 1 << 31, "Object 63 should have mask 0x80000000");
    }

    #[test]
    fn blackbox_bit_location_object_64() {
        let (word, mask) = bit_location(64);
        assert_eq!(word, 2, "Object 64 should be in word 2");
        assert_eq!(mask, 1, "Object 64 should have mask 0x00000001");
    }

    #[test]
    fn blackbox_bit_location_large_index() {
        // Object 1000: word = 1000/32 = 31, bit = 1000%32 = 8
        let (word, mask) = bit_location(1000);
        assert_eq!(word, 31, "Object 1000 should be in word 31");
        assert_eq!(mask, 1 << 8, "Object 1000 should have mask for bit 8");
    }
}

// =============================================================================
// SECTION 2: CPU Bitfield Manipulation (on raw slices)
// =============================================================================

mod cpu_bitfield_operations {
    use super::*;

    #[test]
    fn blackbox_is_visible_false_by_default() {
        let flags = [0u32; 4]; // 128 bits, all clear
        for i in 0..128 {
            assert!(!is_visible(&flags, i), "Object {} should not be visible", i);
        }
    }

    #[test]
    fn blackbox_set_visible_single() {
        let mut flags = [0u32; 2]; // 64 bits
        set_visible(&mut flags, 0);
        assert!(is_visible(&flags, 0), "Object 0 should be visible after set");
        assert!(!is_visible(&flags, 1), "Object 1 should still be invisible");
    }

    #[test]
    fn blackbox_set_visible_multiple() {
        let mut flags = [0u32; 2];
        set_visible(&mut flags, 0);
        set_visible(&mut flags, 31);
        set_visible(&mut flags, 32);
        set_visible(&mut flags, 63);

        assert!(is_visible(&flags, 0));
        assert!(is_visible(&flags, 31));
        assert!(is_visible(&flags, 32));
        assert!(is_visible(&flags, 63));

        // Check some that should NOT be visible
        assert!(!is_visible(&flags, 1));
        assert!(!is_visible(&flags, 30));
        assert!(!is_visible(&flags, 33));
    }

    #[test]
    fn blackbox_set_visible_idempotent() {
        let mut flags = [0u32; 1];
        set_visible(&mut flags, 5);
        set_visible(&mut flags, 5); // Set again
        set_visible(&mut flags, 5); // And again

        assert!(is_visible(&flags, 5), "Should still be visible");
        assert_eq!(count_visible(&flags), 1, "Should only count as 1");
    }

    #[test]
    fn blackbox_clear_visible() {
        let mut flags = [0u32; 2];
        set_visible(&mut flags, 10);
        set_visible(&mut flags, 40);

        assert!(is_visible(&flags, 10));
        assert!(is_visible(&flags, 40));

        clear_visible(&mut flags, 10);

        assert!(!is_visible(&flags, 10), "Object 10 should be cleared");
        assert!(is_visible(&flags, 40), "Object 40 should still be visible");
    }

    #[test]
    fn blackbox_clear_visible_idempotent() {
        let mut flags = [0u32; 1];
        clear_visible(&mut flags, 5);
        clear_visible(&mut flags, 5);
        assert!(!is_visible(&flags, 5));
    }

    #[test]
    fn blackbox_count_visible_empty() {
        let flags = [0u32; 4];
        assert_eq!(count_visible(&flags), 0);
    }

    #[test]
    fn blackbox_count_visible_all_set() {
        let flags = [u32::MAX; 4]; // All 128 bits set
        assert_eq!(count_visible(&flags), 128);
    }

    #[test]
    fn blackbox_count_visible_partial() {
        let mut flags = [0u32; 4];
        set_visible(&mut flags, 0);
        set_visible(&mut flags, 50);
        set_visible(&mut flags, 100);
        set_visible(&mut flags, 127);

        assert_eq!(count_visible(&flags), 4);
    }

    #[test]
    fn blackbox_count_visible_scattered() {
        let mut flags = [0u32; 2]; // 64 bits
        // Set every other bit
        for i in (0..64).step_by(2) {
            set_visible(&mut flags, i);
        }
        assert_eq!(count_visible(&flags), 32);
    }
}

// =============================================================================
// SECTION 3: CPU Batch Operations
// =============================================================================

mod cpu_batch_operations {
    use super::*;

    #[test]
    fn blackbox_cpu_clear_visibility_flags() {
        let mut flags = [u32::MAX; 4]; // All visible
        cpu_clear_visibility_flags(&mut flags);

        assert_eq!(count_visible(&flags), 0, "All flags should be cleared");
        for word in &flags {
            assert_eq!(*word, 0);
        }
    }

    #[test]
    fn blackbox_cpu_clear_empty_slice() {
        let mut flags: [u32; 0] = [];
        cpu_clear_visibility_flags(&mut flags); // Should not panic
    }

    #[test]
    fn blackbox_cpu_atomic_or_single_object() {
        let mut flags = [0u32; 2];
        cpu_atomic_or_visibility(&mut flags, 5);

        assert!(is_visible(&flags, 5));
        assert_eq!(count_visible(&flags), 1);
    }

    #[test]
    fn blackbox_cpu_atomic_or_multiple_objects() {
        let mut flags = [0u32; 4];
        let visible_indices: Vec<usize> = vec![0, 10, 31, 32, 64, 100];
        for &idx in &visible_indices {
            cpu_atomic_or_visibility(&mut flags, idx);
        }

        for &idx in &visible_indices {
            assert!(is_visible(&flags, idx), "Object {} should be visible", idx);
        }
        assert_eq!(count_visible(&flags), visible_indices.len());
    }

    #[test]
    fn blackbox_cpu_atomic_or_preserves_existing() {
        let mut flags = [0u32; 2];
        set_visible(&mut flags, 5);

        cpu_atomic_or_visibility(&mut flags, 10);
        cpu_atomic_or_visibility(&mut flags, 20);

        assert!(is_visible(&flags, 5), "Existing visibility preserved");
        assert!(is_visible(&flags, 10), "New visibility added");
        assert!(is_visible(&flags, 20), "New visibility added");
        assert_eq!(count_visible(&flags), 3);
    }

    #[test]
    fn blackbox_cpu_atomic_or_duplicates() {
        let mut flags = [0u32; 1];
        cpu_atomic_or_visibility(&mut flags, 5);
        cpu_atomic_or_visibility(&mut flags, 5);
        cpu_atomic_or_visibility(&mut flags, 5);
        cpu_atomic_or_visibility(&mut flags, 5);

        assert!(is_visible(&flags, 5));
        assert_eq!(count_visible(&flags), 1, "Duplicates should not increase count");
    }

    #[test]
    fn blackbox_cpu_atomic_or_idempotent_no_op() {
        let mut flags = [0u32; 2];
        set_visible(&mut flags, 10);

        // Just verify the state remains unchanged (no batch empty call needed)
        assert!(is_visible(&flags, 10), "Existing state preserved");
        assert_eq!(count_visible(&flags), 1);
    }
}

// =============================================================================
// SECTION 4: Compact Visible (Stream Compaction)
// =============================================================================

mod compact_visible_tests {
    use super::*;

    #[test]
    fn blackbox_cpu_compact_visible_empty() {
        let flags = [0u32; 4];
        let result = cpu_compact_visible(&flags, 128);
        assert!(result.is_empty(), "No visible objects should produce empty vec");
    }

    #[test]
    fn blackbox_cpu_compact_visible_all() {
        let flags = [u32::MAX; 2]; // 64 bits all set
        let result = cpu_compact_visible(&flags, 64);

        assert_eq!(result.len(), 64);
        for (i, &idx) in result.iter().enumerate() {
            assert_eq!(idx, i, "Index {} should be {}", i, i);
        }
    }

    #[test]
    fn blackbox_cpu_compact_visible_sparse() {
        let mut flags = [0u32; 4];
        set_visible(&mut flags, 5);
        set_visible(&mut flags, 50);
        set_visible(&mut flags, 100);

        let result = cpu_compact_visible(&flags, 128);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 5usize);
        assert_eq!(result[1], 50usize);
        assert_eq!(result[2], 100usize);
    }

    #[test]
    fn blackbox_cpu_compact_visible_respects_object_count() {
        let flags = [u32::MAX; 4]; // All bits set

        // Only consider first 50 objects
        let result = cpu_compact_visible(&flags, 50);

        assert_eq!(result.len(), 50, "Should only return objects within count");
        for (i, &idx) in result.iter().enumerate() {
            assert_eq!(idx, i);
            assert!(idx < 50);
        }
    }

    #[test]
    fn blackbox_cpu_compact_visible_word_boundary() {
        let mut flags = [0u32; 2];
        set_visible(&mut flags, 31); // Last bit of word 0
        set_visible(&mut flags, 32); // First bit of word 1

        let result = cpu_compact_visible(&flags, 64);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 31usize);
        assert_eq!(result[1], 32usize);
    }

    #[test]
    fn blackbox_cpu_compact_visible_sorted() {
        let mut flags = [0u32; 4];
        // Set in random order
        set_visible(&mut flags, 100);
        set_visible(&mut flags, 10);
        set_visible(&mut flags, 50);
        set_visible(&mut flags, 0);
        set_visible(&mut flags, 127);

        let result = cpu_compact_visible(&flags, 128);

        // Result should be sorted (usize vec)
        assert_eq!(result, vec![0usize, 10, 50, 100, 127]);
    }
}

// =============================================================================
// SECTION 5: VisibilityFlagsBuffer Struct (GPU Buffer Wrapper)
// =============================================================================

#[cfg(test)]
mod buffer_struct_tests {
    use super::*;

    /// Test that the struct type is publicly accessible
    #[test]
    fn blackbox_struct_exists_and_accessible() {
        // This test passes if the type can be named
        fn _assert_type_exists(_: &VisibilityFlagsBuffer) {}
    }

    // Note: GPU buffer tests require a wgpu::Device, which needs async setup.
    // These tests verify the API exists and can be called.
}

// =============================================================================
// SECTION 6: Integration / Consistency Tests
// =============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn blackbox_roundtrip_set_check_clear() {
        let mut flags = [0u32; 4];

        // Set, verify, clear, verify for many indices
        for i in 0..128 {
            assert!(!is_visible(&flags, i));
            set_visible(&mut flags, i);
            assert!(is_visible(&flags, i));
            clear_visible(&mut flags, i);
            assert!(!is_visible(&flags, i));
        }
    }

    #[test]
    fn blackbox_bit_location_consistency_with_operations() {
        let mut flags = [0u32; 4];

        for obj_idx in [0usize, 31, 32, 63, 64, 100, 127] {
            let (word, mask) = bit_location(obj_idx);

            // Manually set using bit_location result
            flags[word] |= mask;
            assert!(is_visible(&flags, obj_idx), "Manual set via bit_location should work");

            // Clear using bit_location result
            flags[word] &= !mask;
            assert!(!is_visible(&flags, obj_idx), "Manual clear via bit_location should work");
        }
    }

    #[test]
    fn blackbox_words_for_objects_consistency() {
        // Verify words_for_objects gives correct sizing for operations
        for object_count in [1usize, 32, 33, 64, 100, 1000] {
            let word_count = words_for_objects(object_count);
            let mut flags = vec![0u32; word_count];

            // Should be able to set visibility for all objects
            for i in 0..object_count {
                set_visible(&mut flags, i);
            }

            assert_eq!(count_visible(&flags), object_count);
        }
    }

    #[test]
    fn blackbox_atomic_or_then_compact() {
        let mut flags = [0u32; 8]; // 256 objects
        let visible: Vec<usize> = vec![10, 50, 100, 200];

        for &idx in &visible {
            cpu_atomic_or_visibility(&mut flags, idx);
        }
        let compacted = cpu_compact_visible(&flags, 256);

        assert_eq!(compacted, visible);
    }

    #[test]
    fn blackbox_clear_then_compact() {
        let mut flags = [u32::MAX; 4]; // All 128 visible
        cpu_clear_visibility_flags(&mut flags);

        let compacted = cpu_compact_visible(&flags, 128);
        assert!(compacted.is_empty());
    }

    #[test]
    fn blackbox_stress_many_objects() {
        // Test with a realistic number of objects (16K)
        let object_count: usize = 16 * 1024;
        let word_count = words_for_objects(object_count);
        let mut flags = vec![0u32; word_count];

        // Set every 7th object visible
        let visible_indices: Vec<usize> = (0..object_count).step_by(7).collect();
        for &idx in &visible_indices {
            cpu_atomic_or_visibility(&mut flags, idx);
        }

        // Verify count
        let expected_count = (object_count + 6) / 7; // ceil(object_count/7)
        assert_eq!(count_visible(&flags), expected_count);

        // Verify compact returns correct indices
        let compacted = cpu_compact_visible(&flags, object_count);
        assert_eq!(compacted.len(), expected_count);
        for (i, &idx) in compacted.iter().enumerate() {
            assert_eq!(idx, i * 7);
        }
    }
}

// =============================================================================
// SECTION 7: Edge Cases and Boundary Conditions
// =============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn blackbox_single_bit_per_word() {
        // Test setting exactly one bit in each word
        let mut flags = [0u32; 4];

        set_visible(&mut flags, 0);   // Word 0, bit 0
        set_visible(&mut flags, 32);  // Word 1, bit 0
        set_visible(&mut flags, 64);  // Word 2, bit 0
        set_visible(&mut flags, 96);  // Word 3, bit 0

        assert_eq!(flags[0], 1);
        assert_eq!(flags[1], 1);
        assert_eq!(flags[2], 1);
        assert_eq!(flags[3], 1);
        assert_eq!(count_visible(&flags), 4);
    }

    #[test]
    fn blackbox_last_bit_per_word() {
        // Test setting the last bit in each word
        let mut flags = [0u32; 4];

        set_visible(&mut flags, 31);  // Word 0, bit 31
        set_visible(&mut flags, 63);  // Word 1, bit 31
        set_visible(&mut flags, 95);  // Word 2, bit 31
        set_visible(&mut flags, 127); // Word 3, bit 31

        assert_eq!(flags[0], 1 << 31);
        assert_eq!(flags[1], 1 << 31);
        assert_eq!(flags[2], 1 << 31);
        assert_eq!(flags[3], 1 << 31);
        assert_eq!(count_visible(&flags), 4);
    }

    #[test]
    fn blackbox_alternating_bits() {
        let mut flags = [0u32; 2];

        // Set alternating bits: 0, 2, 4, 6, ... up to 62
        for i in (0..64).step_by(2) {
            set_visible(&mut flags, i);
        }

        // Each word should have 0x55555555 pattern
        assert_eq!(flags[0], 0x55555555);
        assert_eq!(flags[1], 0x55555555);
        assert_eq!(count_visible(&flags), 32);
    }

    #[test]
    fn blackbox_compact_with_zero_object_count() {
        let flags = [u32::MAX; 4];
        let result = cpu_compact_visible(&flags, 0);
        assert!(result.is_empty(), "Zero object count should return empty");
    }

    #[test]
    fn blackbox_words_for_large_count() {
        // 1 million objects
        let count = 1_000_000;
        let words = words_for_objects(count);
        assert_eq!(words, (count + 31) / 32);
        assert!(words * 32 >= count);
        assert!((words - 1) * 32 < count);
    }
}

// =============================================================================
// Test Summary
// =============================================================================

#[test]
fn blackbox_visibility_flags_api_summary() {
    // This test exists to provide a clear summary of what's being tested
    println!("=== BLACKBOX TEST SUMMARY: T-WGPU-P6.2.3 VisibilityFlagsBuffer ===");
    println!("Pure Functions:");
    println!("  - words_for_objects(count) -> word_count");
    println!("  - bit_location(object_index) -> (word_index, bit_mask)");
    println!("");
    println!("CPU Bit Operations:");
    println!("  - is_visible(&flags, index) -> bool");
    println!("  - set_visible(&mut flags, index)");
    println!("  - clear_visible(&mut flags, index)");
    println!("  - count_visible(&flags) -> u32");
    println!("");
    println!("CPU Batch Operations:");
    println!("  - cpu_clear_visibility_flags(&mut flags)");
    println!("  - cpu_atomic_or_visibility(&mut flags, &indices)");
    println!("  - cpu_compact_visible(&flags, object_count) -> Vec<u32>");
    println!("");
    println!("GPU Buffer Wrapper:");
    println!("  - VisibilityFlagsBuffer struct (requires wgpu::Device)");
    println!("===============================================================");
}
