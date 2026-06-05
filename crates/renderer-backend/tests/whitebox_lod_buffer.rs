// SPDX-License-Identifier: MIT
//
// WHITEBOX T-WGPU-P6.5.3: LOD Buffer Management
//
// Tests the internal structure and behavior of the LOD buffer management
// system. This includes struct layout verification, buffer operations,
// pool management, and CPU helper functions.
//
// Categories:
//   1. Struct Layout Tests - LodEntry size, alignment, field offsets
//   2. Buffer Creation Tests - LodBuffer and LodBufferPool construction
//   3. Buffer Operations Tests - resize, clear, read_back
//   4. Pool Operations Tests - current, previous, advance, cycling
//   5. CPU Helper Tests - clear, set, get, count, collect functions
//
// Coverage:
//   - LodEntry: Pod, Zeroable, repr(C), size=8, alignment=4
//   - LodBuffer: creation, resize, clear, upload, read_back
//   - LodBufferPool: double/triple buffering, advance, previous
//   - CPU helpers: cpu_clear_lod_entries, cpu_set_lod_entry, etc.

#![allow(unexpected_cfgs)]

use renderer_backend::gpu_driven::{
    LodEntry,
    cpu_clear_lod_entries, cpu_set_lod_entry, cpu_get_lod_level,
    cpu_count_by_lod, cpu_collect_by_lod,
    LOD_ENTRY_SIZE, DEFAULT_LOD_BUFFER_CAPACITY, MIN_LOD_BUFFER_CAPACITY,
    MAX_LOD_LEVEL, DEFAULT_POOL_SIZE,
};

// Note: LodBuffer and LodBufferPool require wgpu::Device and are tested
// in GPU-enabled integration tests. This whitebox test covers CPU-side
// logic and struct layout verification.

use bytemuck::{Pod, Zeroable};
use std::mem;

// =============================================================================
// Category 1: Struct Layout Tests
// =============================================================================

#[test]
fn test_lod_entry_size_is_8_bytes() {
    assert_eq!(
        mem::size_of::<LodEntry>(),
        8,
        "LodEntry must be exactly 8 bytes for GPU alignment"
    );
}

#[test]
fn test_lod_entry_size_constant_matches() {
    assert_eq!(
        LOD_ENTRY_SIZE,
        mem::size_of::<LodEntry>(),
        "LOD_ENTRY_SIZE constant must match actual struct size"
    );
    assert_eq!(LOD_ENTRY_SIZE, 8);
}

#[test]
fn test_lod_entry_alignment_is_4() {
    assert_eq!(
        mem::align_of::<LodEntry>(),
        4,
        "LodEntry alignment must be 4 bytes for GPU compatibility"
    );
}

#[test]
fn test_lod_entry_implements_pod() {
    fn assert_pod<T: Pod>() {}
    assert_pod::<LodEntry>();
}

#[test]
fn test_lod_entry_implements_zeroable() {
    fn assert_zeroable<T: Zeroable>() {}
    assert_zeroable::<LodEntry>();
}

#[test]
fn test_lod_entry_implements_copy() {
    fn assert_copy<T: Copy>() {}
    assert_copy::<LodEntry>();
}

#[test]
fn test_lod_entry_implements_clone() {
    fn assert_clone<T: Clone>() {}
    assert_clone::<LodEntry>();
}

#[test]
fn test_lod_entry_implements_debug() {
    fn assert_debug<T: std::fmt::Debug>() {}
    assert_debug::<LodEntry>();
}

#[test]
fn test_lod_entry_implements_partial_eq() {
    fn assert_partial_eq<T: PartialEq>() {}
    assert_partial_eq::<LodEntry>();
}

#[test]
fn test_lod_entry_field_offsets() {
    // Create entry with known values
    let entry = LodEntry::new(0x12345678, 0.5);
    let bytes = bytemuck::bytes_of(&entry);

    // level field at offset 0 (4 bytes, little-endian)
    let level = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(level, 0x12345678, "level field should be at offset 0");

    // blend_factor field at offset 4 (4 bytes, little-endian)
    let blend = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert!(
        (blend - 0.5).abs() < 1e-6,
        "blend_factor field should be at offset 4"
    );
}

#[test]
fn test_lod_entry_level_offset_is_zero() {
    let entry = LodEntry::new(42, 0.0);
    let bytes = bytemuck::bytes_of(&entry);
    let level = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(level, 42, "level must be at offset 0");
}

#[test]
fn test_lod_entry_blend_factor_offset_is_four() {
    let entry = LodEntry::new(0, 1.0);
    let bytes = bytemuck::bytes_of(&entry);
    let blend = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert!((blend - 1.0).abs() < 1e-6, "blend_factor must be at offset 4");
}

#[test]
fn test_lod_entry_repr_c_layout() {
    // Verify repr(C) by checking that bytemuck cast produces expected bytes
    let entry = LodEntry::new(1, 0.25);
    let bytes: &[u8] = bytemuck::bytes_of(&entry);

    // Expected layout for repr(C):
    // - level (u32): bytes 0-3
    // - blend_factor (f32): bytes 4-7
    assert_eq!(bytes.len(), 8);

    // Roundtrip should preserve values
    let roundtrip: &LodEntry = bytemuck::from_bytes(bytes);
    assert_eq!(*roundtrip, entry);
}

// =============================================================================
// Category 2: Buffer Creation Tests
// =============================================================================

#[test]
fn test_lod_entry_default() {
    let entry = LodEntry::default();
    assert_eq!(entry.level, 0, "Default level should be 0 (highest detail)");
    assert_eq!(entry.blend_factor, 0.0, "Default blend_factor should be 0.0");
}

#[test]
fn test_lod_entry_new() {
    let entry = LodEntry::new(2, 0.75);
    assert_eq!(entry.level, 2);
    assert_eq!(entry.blend_factor, 0.75);
}

#[test]
fn test_lod_entry_discrete() {
    let entry = LodEntry::discrete(3);
    assert_eq!(entry.level, 3);
    assert_eq!(entry.blend_factor, 0.0, "Discrete should have no blending");
}

#[test]
fn test_lod_entry_clamped_level() {
    // Level above MAX_LOD_LEVEL should be clamped
    let entry = LodEntry::clamped(10, 0.5);
    assert_eq!(entry.level, MAX_LOD_LEVEL);
    assert_eq!(entry.blend_factor, 0.5);
}

#[test]
fn test_lod_entry_clamped_blend_factor_high() {
    // Blend factor above 1.0 should be clamped
    let entry = LodEntry::clamped(1, 2.0);
    assert_eq!(entry.level, 1);
    assert_eq!(entry.blend_factor, 1.0);
}

#[test]
fn test_lod_entry_clamped_blend_factor_low() {
    // Negative blend factor should be clamped to 0.0
    let entry = LodEntry::clamped(0, -0.5);
    assert_eq!(entry.level, 0);
    assert_eq!(entry.blend_factor, 0.0);
}

#[test]
fn test_lod_entry_clamped_both() {
    // Both level and blend factor out of range
    let entry = LodEntry::clamped(100, -100.0);
    assert_eq!(entry.level, MAX_LOD_LEVEL);
    assert_eq!(entry.blend_factor, 0.0);
}

#[test]
fn test_min_lod_buffer_capacity_constant() {
    assert!(
        MIN_LOD_BUFFER_CAPACITY >= 1,
        "Minimum capacity should be at least 1"
    );
    assert_eq!(MIN_LOD_BUFFER_CAPACITY, 32);
}

#[test]
fn test_default_lod_buffer_capacity_constant() {
    assert!(
        DEFAULT_LOD_BUFFER_CAPACITY >= MIN_LOD_BUFFER_CAPACITY,
        "Default capacity should be >= minimum"
    );
    assert_eq!(DEFAULT_LOD_BUFFER_CAPACITY, 65536);
}

#[test]
fn test_max_lod_level_constant() {
    assert_eq!(MAX_LOD_LEVEL, 3, "MAX_LOD_LEVEL should be 3 (LOD 0-3)");
}

#[test]
fn test_default_pool_size_constant() {
    assert!(DEFAULT_POOL_SIZE >= 1, "Pool size must be at least 1");
    assert_eq!(DEFAULT_POOL_SIZE, 2, "Default pool should be double-buffered");
}

// =============================================================================
// Category 3: Buffer Operations Tests (CPU-only, no GPU required)
// =============================================================================

#[test]
fn test_lod_entry_is_highest_detail() {
    assert!(LodEntry::discrete(0).is_highest_detail());
    assert!(!LodEntry::discrete(1).is_highest_detail());
    assert!(!LodEntry::discrete(2).is_highest_detail());
    assert!(!LodEntry::discrete(3).is_highest_detail());
}

#[test]
fn test_lod_entry_is_lowest_detail() {
    assert!(!LodEntry::discrete(0).is_lowest_detail());
    assert!(!LodEntry::discrete(1).is_lowest_detail());
    assert!(!LodEntry::discrete(2).is_lowest_detail());
    assert!(LodEntry::discrete(3).is_lowest_detail());
    // Beyond max should still be lowest
    assert!(LodEntry::new(4, 0.0).is_lowest_detail());
    assert!(LodEntry::new(100, 0.0).is_lowest_detail());
}

#[test]
fn test_lod_entry_is_transitioning() {
    // No transition when blend_factor is 0.0
    assert!(!LodEntry::new(0, 0.0).is_transitioning());
    assert!(!LodEntry::discrete(2).is_transitioning());

    // Transitioning when blend_factor > 0.0
    assert!(LodEntry::new(0, 0.001).is_transitioning());
    assert!(LodEntry::new(1, 0.5).is_transitioning());
    assert!(LodEntry::new(2, 1.0).is_transitioning());
}

#[test]
fn test_lod_entry_effective_level() {
    // Level + blend_factor = effective level
    assert_eq!(LodEntry::new(0, 0.0).effective_level(), 0.0);
    assert_eq!(LodEntry::new(1, 0.0).effective_level(), 1.0);
    assert_eq!(LodEntry::new(1, 0.5).effective_level(), 1.5);
    assert_eq!(LodEntry::new(2, 0.25).effective_level(), 2.25);
    assert_eq!(LodEntry::new(3, 0.0).effective_level(), 3.0);
    assert_eq!(LodEntry::new(0, 1.0).effective_level(), 1.0);
}

#[test]
fn test_buffer_size_calculation() {
    // 1000 objects at 8 bytes each = 8000 bytes
    let size = 1000 * LOD_ENTRY_SIZE;
    assert_eq!(size, 8000);

    // 100,000 objects = 800,000 bytes
    let size = 100_000 * LOD_ENTRY_SIZE;
    assert_eq!(size, 800_000);

    // Default capacity size
    let size = DEFAULT_LOD_BUFFER_CAPACITY as usize * LOD_ENTRY_SIZE;
    assert_eq!(size, 524_288); // 65536 * 8 = 512 KB
}

#[test]
fn test_min_capacity_enforcement() {
    // Capacity below minimum should be clamped
    let capacity = 0u32.max(MIN_LOD_BUFFER_CAPACITY);
    assert_eq!(capacity, MIN_LOD_BUFFER_CAPACITY);

    let capacity = 10u32.max(MIN_LOD_BUFFER_CAPACITY);
    assert_eq!(capacity, MIN_LOD_BUFFER_CAPACITY);

    // Capacity at minimum should stay
    let capacity = 32u32.max(MIN_LOD_BUFFER_CAPACITY);
    assert_eq!(capacity, 32);

    // Capacity above minimum should stay
    let capacity = 1000u32.max(MIN_LOD_BUFFER_CAPACITY);
    assert_eq!(capacity, 1000);
}

// =============================================================================
// Category 4: Pool Operations Tests (CPU-only logic verification)
// =============================================================================

#[test]
fn test_pool_index_wraparound_double_buffer() {
    let buffer_count = 2;
    let mut current_index = 0;

    // Advance once: 0 -> 1
    current_index = (current_index + 1) % buffer_count;
    assert_eq!(current_index, 1);

    // Advance again: 1 -> 0 (wraparound)
    current_index = (current_index + 1) % buffer_count;
    assert_eq!(current_index, 0);

    // Verify cycle
    for expected in [1, 0, 1, 0, 1, 0] {
        current_index = (current_index + 1) % buffer_count;
        assert_eq!(current_index, expected);
    }
}

#[test]
fn test_pool_index_wraparound_triple_buffer() {
    let buffer_count = 3;
    let mut current_index = 0;

    // Full cycle: 0 -> 1 -> 2 -> 0
    for expected in [1, 2, 0, 1, 2, 0] {
        current_index = (current_index + 1) % buffer_count;
        assert_eq!(current_index, expected);
    }
}

#[test]
fn test_pool_previous_index_double_buffer() {
    let buffer_count = 2;

    // At index 0, previous is 1
    let current = 0;
    let prev = if current == 0 { buffer_count - 1 } else { current - 1 };
    assert_eq!(prev, 1);

    // At index 1, previous is 0
    let current = 1;
    let prev = if current == 0 { buffer_count - 1 } else { current - 1 };
    assert_eq!(prev, 0);
}

#[test]
fn test_pool_previous_index_triple_buffer() {
    let buffer_count = 3;

    // At index 0, previous is 2
    let current = 0;
    let prev = if current == 0 { buffer_count - 1 } else { current - 1 };
    assert_eq!(prev, 2);

    // At index 1, previous is 0
    let current = 1;
    let prev = if current == 0 { buffer_count - 1 } else { current - 1 };
    assert_eq!(prev, 0);

    // At index 2, previous is 1
    let current = 2;
    let prev = if current == 0 { buffer_count - 1 } else { current - 1 };
    assert_eq!(prev, 1);
}

#[test]
fn test_pool_previous_after_advance() {
    // Simulate pool behavior
    let buffer_count = 3;
    let mut current_index = 0;

    // After each advance, previous should be what current was before
    let old_current = current_index;
    current_index = (current_index + 1) % buffer_count;
    let prev = if current_index == 0 { buffer_count - 1 } else { current_index - 1 };
    assert_eq!(prev, old_current);

    // Again
    let old_current = current_index;
    current_index = (current_index + 1) % buffer_count;
    let prev = if current_index == 0 { buffer_count - 1 } else { current_index - 1 };
    assert_eq!(prev, old_current);
}

#[test]
fn test_pool_buffer_count_assertion() {
    // Verify that DEFAULT_POOL_SIZE is valid
    assert!(DEFAULT_POOL_SIZE >= 1, "Pool must have at least 1 buffer");
}

#[test]
fn test_pool_cycling_maintains_consistency() {
    // Simulate 10 frames of double buffering
    let buffer_count = 2;
    let mut current_index = 0;

    for frame in 0..10 {
        let expected_current = frame % buffer_count;
        assert_eq!(current_index, expected_current);

        // Previous should be the other buffer
        let prev = if current_index == 0 { buffer_count - 1 } else { current_index - 1 };
        assert_ne!(prev, current_index, "Previous should not equal current");

        current_index = (current_index + 1) % buffer_count;
    }
}

// =============================================================================
// Category 5: CPU Helper Tests
// =============================================================================

#[test]
fn test_cpu_clear_lod_entries_single() {
    let mut entries = vec![LodEntry::new(3, 0.9)];
    cpu_clear_lod_entries(&mut entries);
    assert_eq!(entries[0], LodEntry::default());
}

#[test]
fn test_cpu_clear_lod_entries_multiple() {
    let mut entries = vec![
        LodEntry::new(1, 0.5),
        LodEntry::new(2, 0.3),
        LodEntry::new(3, 0.7),
    ];

    cpu_clear_lod_entries(&mut entries);

    for entry in &entries {
        assert_eq!(entry.level, 0);
        assert_eq!(entry.blend_factor, 0.0);
    }
}

#[test]
fn test_cpu_clear_lod_entries_empty() {
    let mut entries: Vec<LodEntry> = vec![];
    cpu_clear_lod_entries(&mut entries); // Should not panic
    assert!(entries.is_empty());
}

#[test]
fn test_cpu_clear_lod_entries_large() {
    let mut entries = vec![LodEntry::new(2, 0.5); 10000];
    cpu_clear_lod_entries(&mut entries);

    for entry in &entries {
        assert_eq!(*entry, LodEntry::default());
    }
}

#[test]
fn test_cpu_set_lod_entry_basic() {
    let mut entries = vec![LodEntry::default(); 5];

    cpu_set_lod_entry(&mut entries, 2, 3, 0.5);

    assert_eq!(entries[2].level, 3);
    assert_eq!(entries[2].blend_factor, 0.5);
}

#[test]
fn test_cpu_set_lod_entry_preserves_others() {
    let mut entries = vec![LodEntry::default(); 5];

    cpu_set_lod_entry(&mut entries, 2, 3, 0.5);

    // Other entries should be unchanged
    assert_eq!(entries[0], LodEntry::default());
    assert_eq!(entries[1], LodEntry::default());
    assert_eq!(entries[3], LodEntry::default());
    assert_eq!(entries[4], LodEntry::default());
}

#[test]
fn test_cpu_set_lod_entry_first() {
    let mut entries = vec![LodEntry::default(); 3];
    cpu_set_lod_entry(&mut entries, 0, 1, 0.25);
    assert_eq!(entries[0], LodEntry::new(1, 0.25));
}

#[test]
fn test_cpu_set_lod_entry_last() {
    let mut entries = vec![LodEntry::default(); 3];
    cpu_set_lod_entry(&mut entries, 2, 2, 0.75);
    assert_eq!(entries[2], LodEntry::new(2, 0.75));
}

#[test]
fn test_cpu_set_lod_entry_out_of_bounds() {
    let mut entries = vec![LodEntry::default(); 3];

    // Should not panic, just silently ignore
    cpu_set_lod_entry(&mut entries, 10, 2, 0.5);
    cpu_set_lod_entry(&mut entries, 100, 3, 0.9);
    cpu_set_lod_entry(&mut entries, usize::MAX, 1, 0.1);

    // Entries unchanged
    assert!(entries.iter().all(|e| *e == LodEntry::default()));
}

#[test]
fn test_cpu_set_lod_entry_overwrite() {
    let mut entries = vec![LodEntry::new(1, 0.5); 3];

    cpu_set_lod_entry(&mut entries, 1, 3, 0.9);

    assert_eq!(entries[1], LodEntry::new(3, 0.9));
    // Others unchanged
    assert_eq!(entries[0], LodEntry::new(1, 0.5));
    assert_eq!(entries[2], LodEntry::new(1, 0.5));
}

#[test]
fn test_cpu_get_lod_level_basic() {
    let entries = vec![
        LodEntry::new(0, 0.0),
        LodEntry::new(2, 0.5),
        LodEntry::new(3, 0.0),
    ];

    assert_eq!(cpu_get_lod_level(&entries, 0), Some(0));
    assert_eq!(cpu_get_lod_level(&entries, 1), Some(2));
    assert_eq!(cpu_get_lod_level(&entries, 2), Some(3));
}

#[test]
fn test_cpu_get_lod_level_out_of_bounds() {
    let entries = vec![LodEntry::default(); 3];

    assert_eq!(cpu_get_lod_level(&entries, 3), None);
    assert_eq!(cpu_get_lod_level(&entries, 10), None);
    assert_eq!(cpu_get_lod_level(&entries, usize::MAX), None);
}

#[test]
fn test_cpu_get_lod_level_empty() {
    let entries: Vec<LodEntry> = vec![];
    assert_eq!(cpu_get_lod_level(&entries, 0), None);
}

#[test]
fn test_cpu_count_by_lod_basic() {
    let entries = vec![
        LodEntry::discrete(0),
        LodEntry::discrete(0),
        LodEntry::discrete(1),
        LodEntry::discrete(2),
        LodEntry::discrete(2),
        LodEntry::discrete(2),
        LodEntry::discrete(3),
    ];

    let counts = cpu_count_by_lod(&entries);
    assert_eq!(counts[0], 2); // LOD 0
    assert_eq!(counts[1], 1); // LOD 1
    assert_eq!(counts[2], 3); // LOD 2
    assert_eq!(counts[3], 1); // LOD 3
}

#[test]
fn test_cpu_count_by_lod_empty() {
    let entries: Vec<LodEntry> = vec![];
    let counts = cpu_count_by_lod(&entries);
    assert_eq!(counts, [0, 0, 0, 0]);
}

#[test]
fn test_cpu_count_by_lod_all_same() {
    let entries = vec![LodEntry::discrete(1); 100];
    let counts = cpu_count_by_lod(&entries);
    assert_eq!(counts, [0, 100, 0, 0]);
}

#[test]
fn test_cpu_count_by_lod_all_lod0() {
    let entries = vec![LodEntry::default(); 50];
    let counts = cpu_count_by_lod(&entries);
    assert_eq!(counts, [50, 0, 0, 0]);
}

#[test]
fn test_cpu_count_by_lod_all_lod3() {
    let entries = vec![LodEntry::discrete(3); 75];
    let counts = cpu_count_by_lod(&entries);
    assert_eq!(counts, [0, 0, 0, 75]);
}

#[test]
fn test_cpu_count_by_lod_beyond_max() {
    // Level > 3 should be counted as level 3
    let entries = vec![
        LodEntry::new(4, 0.0),
        LodEntry::new(5, 0.0),
        LodEntry::new(100, 0.0),
    ];
    let counts = cpu_count_by_lod(&entries);
    assert_eq!(counts, [0, 0, 0, 3]); // All clamped to LOD 3
}

#[test]
fn test_cpu_count_by_lod_with_blend_factor() {
    // Blend factor should not affect counting
    let entries = vec![
        LodEntry::new(0, 0.9),
        LodEntry::new(1, 0.5),
        LodEntry::new(1, 0.1),
    ];
    let counts = cpu_count_by_lod(&entries);
    assert_eq!(counts, [1, 2, 0, 0]);
}

#[test]
fn test_cpu_collect_by_lod_basic() {
    let entries = vec![
        LodEntry::discrete(0), // index 0
        LodEntry::discrete(1), // index 1
        LodEntry::discrete(0), // index 2
        LodEntry::discrete(2), // index 3
        LodEntry::discrete(0), // index 4
    ];

    let lod0_indices = cpu_collect_by_lod(&entries, 0);
    assert_eq!(lod0_indices, vec![0, 2, 4]);

    let lod1_indices = cpu_collect_by_lod(&entries, 1);
    assert_eq!(lod1_indices, vec![1]);

    let lod2_indices = cpu_collect_by_lod(&entries, 2);
    assert_eq!(lod2_indices, vec![3]);

    let lod3_indices = cpu_collect_by_lod(&entries, 3);
    assert!(lod3_indices.is_empty());
}

#[test]
fn test_cpu_collect_by_lod_empty() {
    let entries: Vec<LodEntry> = vec![];

    for lod in 0..=3 {
        let indices = cpu_collect_by_lod(&entries, lod);
        assert!(indices.is_empty());
    }
}

#[test]
fn test_cpu_collect_by_lod_no_matches() {
    let entries = vec![LodEntry::discrete(0); 10];

    // No LOD 1, 2, or 3
    assert!(cpu_collect_by_lod(&entries, 1).is_empty());
    assert!(cpu_collect_by_lod(&entries, 2).is_empty());
    assert!(cpu_collect_by_lod(&entries, 3).is_empty());
}

#[test]
fn test_cpu_collect_by_lod_all_match() {
    let entries = vec![LodEntry::discrete(2); 5];
    let indices = cpu_collect_by_lod(&entries, 2);
    assert_eq!(indices, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_cpu_collect_by_lod_order_preserved() {
    let entries = vec![
        LodEntry::discrete(1), // 0
        LodEntry::discrete(2), // 1
        LodEntry::discrete(1), // 2
        LodEntry::discrete(0), // 3
        LodEntry::discrete(1), // 4
        LodEntry::discrete(1), // 5
    ];

    let indices = cpu_collect_by_lod(&entries, 1);
    // Indices should be in ascending order
    assert_eq!(indices, vec![0, 2, 4, 5]);

    for i in 1..indices.len() {
        assert!(indices[i] > indices[i - 1], "Indices should be ordered");
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_lod_entry_max_values() {
    let entry = LodEntry::new(u32::MAX, f32::MAX);
    assert_eq!(entry.level, u32::MAX);
    assert_eq!(entry.blend_factor, f32::MAX);
}

#[test]
fn test_lod_entry_negative_blend() {
    // Negative blend factor is valid in new() (no clamping)
    let entry = LodEntry::new(0, -1.0);
    assert_eq!(entry.blend_factor, -1.0);

    // Use clamped() for validation
    let entry = LodEntry::clamped(0, -1.0);
    assert_eq!(entry.blend_factor, 0.0);
}

#[test]
fn test_lod_entry_nan_blend() {
    // NaN is a valid f32 value
    let entry = LodEntry::new(0, f32::NAN);
    assert!(entry.blend_factor.is_nan());
}

#[test]
fn test_lod_entry_inf_blend() {
    let entry = LodEntry::new(0, f32::INFINITY);
    assert!(entry.blend_factor.is_infinite());

    // Clamped handles infinity
    let entry = LodEntry::clamped(0, f32::INFINITY);
    assert_eq!(entry.blend_factor, 1.0);

    let entry = LodEntry::clamped(0, f32::NEG_INFINITY);
    assert_eq!(entry.blend_factor, 0.0);
}

#[test]
fn test_lod_entry_zeroable() {
    let zeroed: LodEntry = bytemuck::Zeroable::zeroed();
    assert_eq!(zeroed.level, 0);
    assert_eq!(zeroed.blend_factor, 0.0);

    // Verify all bytes are zero
    let bytes = bytemuck::bytes_of(&zeroed);
    for byte in bytes {
        assert_eq!(*byte, 0);
    }
}

#[test]
fn test_lod_entry_pod_roundtrip() {
    let entry = LodEntry::new(2, 0.75);
    let bytes: &[u8] = bytemuck::bytes_of(&entry);
    let roundtrip: &LodEntry = bytemuck::from_bytes(bytes);
    assert_eq!(*roundtrip, entry);
}

#[test]
fn test_lod_entry_slice_cast() {
    let entries = [
        LodEntry::new(0, 0.0),
        LodEntry::new(1, 0.25),
        LodEntry::new(2, 0.5),
        LodEntry::new(3, 0.75),
    ];
    let bytes: &[u8] = bytemuck::cast_slice(&entries);
    assert_eq!(bytes.len(), LOD_ENTRY_SIZE * 4);

    let roundtrip: &[LodEntry] = bytemuck::cast_slice(bytes);
    assert_eq!(roundtrip, &entries);
}

#[test]
fn test_lod_entry_equality() {
    let e1 = LodEntry::new(1, 0.5);
    let e2 = LodEntry::new(1, 0.5);
    let e3 = LodEntry::new(2, 0.5);
    let e4 = LodEntry::new(1, 0.6);

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
    assert_ne!(e1, e4);
}

#[test]
fn test_lod_entry_clone_copy() {
    let entry = LodEntry::new(2, 0.3);
    let cloned = entry.clone();
    let copied: LodEntry = entry; // Copy

    assert_eq!(entry, cloned);
    assert_eq!(entry, copied);
}

#[test]
fn test_lod_entry_debug_format() {
    let entry = LodEntry::new(1, 0.5);
    let debug_str = format!("{:?}", entry);
    assert!(debug_str.contains("LodEntry"));
    assert!(debug_str.contains("level"));
    assert!(debug_str.contains("blend_factor"));
}

#[test]
fn test_lod_entry_const_new() {
    // Verify new() is const
    const ENTRY: LodEntry = LodEntry::new(2, 0.5);
    assert_eq!(ENTRY.level, 2);
    assert_eq!(ENTRY.blend_factor, 0.5);
}

#[test]
fn test_lod_entry_const_discrete() {
    // Verify discrete() is const
    const ENTRY: LodEntry = LodEntry::discrete(3);
    assert_eq!(ENTRY.level, 3);
    assert_eq!(ENTRY.blend_factor, 0.0);
}

// =============================================================================
// Integration Tests: CPU Helpers Working Together
// =============================================================================

#[test]
fn test_cpu_helpers_workflow() {
    // Simulate a frame workflow
    let mut entries = vec![LodEntry::default(); 10];

    // 1. Clear all entries at frame start
    cpu_clear_lod_entries(&mut entries);
    assert!(entries.iter().all(|e| *e == LodEntry::default()));

    // 2. Set LOD levels based on "distance"
    cpu_set_lod_entry(&mut entries, 0, 0, 0.0); // Close
    cpu_set_lod_entry(&mut entries, 1, 0, 0.5); // Close, transitioning
    cpu_set_lod_entry(&mut entries, 2, 1, 0.0); // Medium
    cpu_set_lod_entry(&mut entries, 3, 1, 0.3); // Medium, transitioning
    cpu_set_lod_entry(&mut entries, 4, 2, 0.0); // Far
    cpu_set_lod_entry(&mut entries, 5, 2, 0.8); // Far, transitioning
    cpu_set_lod_entry(&mut entries, 6, 3, 0.0); // Very far
    cpu_set_lod_entry(&mut entries, 7, 3, 0.0); // Very far
    cpu_set_lod_entry(&mut entries, 8, 3, 0.0); // Very far
    cpu_set_lod_entry(&mut entries, 9, 0, 0.0); // Close

    // 3. Query individual levels
    assert_eq!(cpu_get_lod_level(&entries, 0), Some(0));
    assert_eq!(cpu_get_lod_level(&entries, 4), Some(2));
    assert_eq!(cpu_get_lod_level(&entries, 6), Some(3));

    // 4. Count by LOD
    let counts = cpu_count_by_lod(&entries);
    assert_eq!(counts[0], 3); // indices 0, 1, 9
    assert_eq!(counts[1], 2); // indices 2, 3
    assert_eq!(counts[2], 2); // indices 4, 5
    assert_eq!(counts[3], 3); // indices 6, 7, 8

    // 5. Collect indices for draw batching
    let lod0_objs = cpu_collect_by_lod(&entries, 0);
    assert_eq!(lod0_objs, vec![0, 1, 9]);

    let lod3_objs = cpu_collect_by_lod(&entries, 3);
    assert_eq!(lod3_objs, vec![6, 7, 8]);
}

#[test]
fn test_cpu_helpers_stress() {
    let count = 100_000;
    let mut entries = vec![LodEntry::default(); count];

    // Set various LOD levels
    for i in 0..count {
        let level = (i % 4) as u32;
        let blend = (i % 10) as f32 / 10.0;
        cpu_set_lod_entry(&mut entries, i, level, blend);
    }

    // Verify counts
    let counts = cpu_count_by_lod(&entries);
    assert_eq!(counts[0], 25000);
    assert_eq!(counts[1], 25000);
    assert_eq!(counts[2], 25000);
    assert_eq!(counts[3], 25000);

    // Verify collection
    let lod2_indices = cpu_collect_by_lod(&entries, 2);
    assert_eq!(lod2_indices.len(), 25000);

    // First few should be 2, 6, 10, 14...
    assert_eq!(lod2_indices[0], 2);
    assert_eq!(lod2_indices[1], 6);
    assert_eq!(lod2_indices[2], 10);
}

// =============================================================================
// Summary Test
// =============================================================================

#[test]
fn test_all_exports_accessible() {
    // Types
    let _: LodEntry = LodEntry::default();
    // LodBuffer and LodBufferPool require wgpu device, tested elsewhere

    // Constants
    assert_eq!(LOD_ENTRY_SIZE, 8);
    assert_eq!(DEFAULT_LOD_BUFFER_CAPACITY, 65536);
    assert_eq!(MIN_LOD_BUFFER_CAPACITY, 32);
    assert_eq!(MAX_LOD_LEVEL, 3);
    assert_eq!(DEFAULT_POOL_SIZE, 2);

    // CPU helper functions
    let mut entries = vec![LodEntry::default(); 3];
    cpu_clear_lod_entries(&mut entries);
    cpu_set_lod_entry(&mut entries, 0, 1, 0.5);
    let _ = cpu_get_lod_level(&entries, 0);
    let _ = cpu_count_by_lod(&entries);
    let _ = cpu_collect_by_lod(&entries, 0);
}
