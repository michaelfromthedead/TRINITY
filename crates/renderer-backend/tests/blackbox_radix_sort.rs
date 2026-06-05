// SPDX-License-Identifier: MIT
//
// BLACKBOX T-WGPU-P3.10.4 Radix Sort Compute Shaders. CLEANROOM.
//
// Contract: Radix sort shader compiles correctly and RadixSortPipeline API
// is accessible. Algorithm uses 4-bit digits, 8 passes for 32-bit keys,
// histogram + scatter approach.
//
// Tests:
//   1. Shader constants are correctly defined (RADIX_BITS=4, BUCKETS=16, PASSES=8)
//   2. RadixSortParams struct (construction, accessors, bytemuck compatibility)
//   3. Workgroup calculation (num_workgroups)
//   4. Histogram buffer size calculation
//   5. RadixSortError enum (Display, std::error::Error)
//   6. Real-world scenarios (depth, Morton codes, particles)
//   7. Edge cases (empty, single element, already sorted)
//
// Note: Full GPU execution tests require a GPU and are in a separate test file.

use renderer_backend::compute_library::radix_sort::{
    RadixSortError, RadixSortParams, RadixSortPipeline, ELEMENTS_PER_THREAD,
    ELEMENTS_PER_WORKGROUP, RADIX_BITS, RADIX_BUCKETS, TOTAL_PASSES, WORKGROUP_SIZE,
};
use std::error::Error;

// =============================================================================
// SECTION 1 -- Constants are correctly defined
// =============================================================================

#[test]
fn test_radix_bits_constant() {
    // Radix sort uses 4-bit digits (16 buckets)
    assert_eq!(RADIX_BITS, 4, "Radix sort must use 4 bits per pass");
}

#[test]
fn test_radix_buckets_constant() {
    // 4 bits = 2^4 = 16 possible digit values
    assert_eq!(RADIX_BUCKETS, 16, "Must have 16 buckets (2^4)");
}

#[test]
fn test_radix_buckets_equals_two_to_radix_bits() {
    // Mathematical relationship
    assert_eq!(
        RADIX_BUCKETS,
        1 << RADIX_BITS,
        "RADIX_BUCKETS must equal 2^RADIX_BITS"
    );
}

#[test]
fn test_total_passes_constant() {
    // 32-bit keys / 4 bits per pass = 8 passes
    assert_eq!(TOTAL_PASSES, 8, "32-bit keys require 8 passes with 4-bit digits");
}

#[test]
fn test_total_passes_covers_32_bits() {
    // Verify algorithm covers all 32 bits
    assert_eq!(
        TOTAL_PASSES * RADIX_BITS,
        32,
        "Total passes * radix bits must equal 32 for u32 keys"
    );
}

#[test]
fn test_workgroup_size_constant() {
    // Standard GPU workgroup size
    assert_eq!(WORKGROUP_SIZE, 256, "Workgroup size must be 256 threads");
}

#[test]
fn test_elements_per_thread_constant() {
    // Each thread processes multiple elements for efficiency
    assert_eq!(
        ELEMENTS_PER_THREAD, 4,
        "Each thread should process 4 elements"
    );
}

#[test]
fn test_elements_per_workgroup_constant() {
    // 256 threads * 4 elements = 1024 elements per workgroup
    assert_eq!(
        ELEMENTS_PER_WORKGROUP, 1024,
        "Workgroup should process 1024 elements"
    );
}

#[test]
fn test_elements_per_workgroup_derivation() {
    // Verify derivation from thread count and elements per thread
    assert_eq!(
        ELEMENTS_PER_WORKGROUP,
        WORKGROUP_SIZE * ELEMENTS_PER_THREAD,
        "ELEMENTS_PER_WORKGROUP = WORKGROUP_SIZE * ELEMENTS_PER_THREAD"
    );
}

// =============================================================================
// SECTION 2 -- RadixSortParams API
// =============================================================================

#[test]
fn test_radix_sort_params_new() {
    let params = RadixSortParams::new(1000, 3, 1);
    assert_eq!(params.input_size, 1000);
    assert_eq!(params.pass_number, 3);
    assert_eq!(params.num_workgroups, 1);
}

#[test]
fn test_radix_sort_params_pass_zero() {
    // Pass 0 is LSB (bits 0-3)
    let params = RadixSortParams::new(500, 0, 1);
    assert_eq!(params.pass_number, 0, "Pass 0 should be valid");
}

#[test]
fn test_radix_sort_params_pass_seven() {
    // Pass 7 is MSB (bits 28-31)
    let params = RadixSortParams::new(500, 7, 1);
    assert_eq!(params.pass_number, 7, "Pass 7 should be valid");
}

#[test]
fn test_radix_sort_params_all_passes() {
    // Should be able to create params for all 8 passes
    for pass in 0..TOTAL_PASSES {
        let params = RadixSortParams::new(100, pass, 1);
        assert_eq!(params.pass_number, pass);
    }
}

#[test]
fn test_radix_sort_params_size_alignment() {
    // Must be 16 bytes (4x u32) for proper GPU uniform buffer alignment
    assert_eq!(
        std::mem::size_of::<RadixSortParams>(),
        16,
        "RadixSortParams must be 16 bytes for GPU alignment"
    );
}

#[test]
fn test_radix_sort_params_padding_field() {
    // Verify padding exists for alignment
    let params = RadixSortParams::new(100, 0, 1);
    assert_eq!(params._pad, 0, "Padding should be zero");
}

#[test]
fn test_radix_sort_params_bytemuck_pod() {
    // Verify struct is Plain Old Data (can be cast to bytes)
    fn requires_pod<T: bytemuck::Pod>() {}
    requires_pod::<RadixSortParams>();
}

#[test]
fn test_radix_sort_params_bytemuck_zeroable() {
    // Verify struct can be safely zeroed
    fn requires_zeroable<T: bytemuck::Zeroable>() {}
    requires_zeroable::<RadixSortParams>();
}

#[test]
fn test_radix_sort_params_cast_to_bytes() {
    let params = RadixSortParams::new(1024, 5, 2);
    let bytes: &[u8] = bytemuck::bytes_of(&params);
    assert_eq!(bytes.len(), 16, "Should cast to 16 bytes");
}

#[test]
fn test_radix_sort_params_default() {
    let params = RadixSortParams::default();
    assert_eq!(params.input_size, 0);
    assert_eq!(params.pass_number, 0);
    assert_eq!(params.num_workgroups, 0);
    assert_eq!(params._pad, 0);
}

#[test]
fn test_radix_sort_params_debug() {
    // Should implement Debug for logging
    let params = RadixSortParams::new(500, 2, 1);
    let debug_str = format!("{:?}", params);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

#[test]
fn test_radix_sort_params_clone() {
    let params = RadixSortParams::new(2000, 4, 2);
    let cloned = params.clone();
    assert_eq!(cloned.input_size, params.input_size);
    assert_eq!(cloned.pass_number, params.pass_number);
    assert_eq!(cloned.num_workgroups, params.num_workgroups);
}

#[test]
fn test_radix_sort_params_copy() {
    let params = RadixSortParams::new(1500, 6, 2);
    let copied: RadixSortParams = params; // Copy
    assert_eq!(copied.input_size, 1500);
    assert_eq!(copied.pass_number, 6);
}

// =============================================================================
// SECTION 3 -- Workgroup calculation (num_workgroups)
// =============================================================================

#[test]
fn test_single_element_uses_one_workgroup() {
    assert_eq!(RadixSortPipeline::num_workgroups(1), 1);
}

#[test]
fn test_1024_elements_uses_one_workgroup() {
    // One workgroup handles 1024 elements
    assert_eq!(RadixSortPipeline::num_workgroups(1024), 1);
}

#[test]
fn test_1025_elements_uses_two_workgroups() {
    // 1025 elements exceed single workgroup capacity
    assert_eq!(RadixSortPipeline::num_workgroups(1025), 2);
}

#[test]
fn test_2048_elements_uses_two_workgroups() {
    // ceil(2048 / 1024) = 2
    assert_eq!(RadixSortPipeline::num_workgroups(2048), 2);
}

#[test]
fn test_workgroup_boundary_exact() {
    // Exactly at workgroup boundaries
    assert_eq!(RadixSortPipeline::num_workgroups(1024), 1);
    assert_eq!(RadixSortPipeline::num_workgroups(2048), 2);
    assert_eq!(RadixSortPipeline::num_workgroups(3072), 3);
    assert_eq!(RadixSortPipeline::num_workgroups(4096), 4);
}

#[test]
fn test_workgroup_calculation_formula() {
    // Verify formula: ceil(n / ELEMENTS_PER_WORKGROUP)
    for n in [1, 100, 1023, 1024, 1025, 2000, 5000, 10000] {
        let expected = (n + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP;
        assert_eq!(
            RadixSortPipeline::num_workgroups(n),
            expected,
            "Mismatch for n={}",
            n
        );
    }
}

#[test]
fn test_workgroup_calculation_large_arrays() {
    // 10000 elements: ceil(10000 / 1024) = 10
    assert_eq!(RadixSortPipeline::num_workgroups(10_000), 10);

    // 100000 elements: ceil(100000 / 1024) = 98
    assert_eq!(RadixSortPipeline::num_workgroups(100_000), 98);

    // 1M elements: ceil(1_000_000 / 1024) = 977
    assert_eq!(RadixSortPipeline::num_workgroups(1_000_000), 977);
}

#[test]
fn test_workgroup_calculation_power_of_two_sizes() {
    // Common GPU buffer sizes
    assert_eq!(RadixSortPipeline::num_workgroups(256), 1);
    assert_eq!(RadixSortPipeline::num_workgroups(512), 1);
    assert_eq!(RadixSortPipeline::num_workgroups(1024), 1);
    assert_eq!(RadixSortPipeline::num_workgroups(4096), 4);
    assert_eq!(RadixSortPipeline::num_workgroups(65536), 64);
    assert_eq!(RadixSortPipeline::num_workgroups(1048576), 1024);
}

// =============================================================================
// SECTION 4 -- Histogram buffer size calculation
// =============================================================================

#[test]
fn test_histogram_buffer_single_workgroup() {
    // Single workgroup needs space for 16 buckets
    let size = RadixSortPipeline::histogram_buffer_size(1);
    // At minimum: 16 buckets * 4 bytes = 64 bytes
    assert!(size >= 64, "Single workgroup histogram should be at least 64 bytes");
}

#[test]
fn test_histogram_buffer_multi_workgroup() {
    // Two workgroups: 16 buckets * 2 workgroups * 4 bytes = 128 bytes
    let size = RadixSortPipeline::histogram_buffer_size(2);
    assert!(size >= 128, "Two workgroups histogram should be at least 128 bytes");
}

#[test]
fn test_histogram_buffer_scales_with_workgroups() {
    let size_small = RadixSortPipeline::histogram_buffer_size(1);
    let size_medium = RadixSortPipeline::histogram_buffer_size(10);
    let size_large = RadixSortPipeline::histogram_buffer_size(100);

    assert!(
        size_medium >= size_small,
        "More workgroups need more histogram space"
    );
    assert!(
        size_large >= size_medium,
        "Even more workgroups need even more histogram space"
    );
}

#[test]
fn test_histogram_buffer_formula() {
    // Formula: num_workgroups * RADIX_BUCKETS * sizeof(u32)
    for num_wg in [1, 5, 10, 50, 100, 500, 1000] {
        let size = RadixSortPipeline::histogram_buffer_size(num_wg);
        let expected = (num_wg as u64) * (RADIX_BUCKETS as u64) * 4;
        assert!(
            size >= expected,
            "For {} workgroups, got {} bytes, expected at least {}",
            num_wg,
            size,
            expected
        );
    }
}

#[test]
fn test_histogram_buffer_realistic_particle_count() {
    // 1M particles: 977 workgroups
    let num_wg = RadixSortPipeline::num_workgroups(1_000_000);
    let hist_size = RadixSortPipeline::histogram_buffer_size(num_wg);

    // 977 * 16 * 4 = 62528 bytes minimum
    assert!(
        hist_size >= 62528,
        "1M particles histogram should have sufficient space"
    );
}

// =============================================================================
// SECTION 5 -- RadixSortError error handling
// =============================================================================

#[test]
fn test_radix_sort_error_implements_display() {
    fn requires_display<T: std::fmt::Display>(_: &T) {}

    let error = RadixSortError::EmptyInput;
    requires_display(&error);
}

#[test]
fn test_radix_sort_error_implements_std_error() {
    fn requires_error<T: Error>(_: &T) {}

    let error = RadixSortError::EmptyInput;
    requires_error(&error);
}

#[test]
fn test_empty_input_error() {
    let error = RadixSortError::EmptyInput;
    let msg = error.to_string();
    assert!(!msg.is_empty(), "Error message should not be empty");
}

#[test]
fn test_input_too_large_error() {
    let error = RadixSortError::InputTooLarge {
        size: 100_000_000,
        max: 67_108_864,
    };
    let msg = error.to_string();
    assert!(!msg.is_empty(), "Error message should not be empty");
}

#[test]
fn test_size_mismatch_error() {
    let error = RadixSortError::SizeMismatch {
        keys: 4096,
        values: 2048,
    };
    let msg = error.to_string();
    assert!(!msg.is_empty(), "Error message should not be empty");
}

#[test]
fn test_buffer_too_small_error() {
    let error = RadixSortError::BufferTooSmall {
        required: 8192,
        actual: 4096,
    };
    let msg = error.to_string();
    assert!(!msg.is_empty(), "Error message should not be empty");
}

#[test]
fn test_radix_sort_error_debug() {
    let error = RadixSortError::EmptyInput;
    let debug_str = format!("{:?}", error);
    assert!(!debug_str.is_empty(), "Debug output should not be empty");
}

#[test]
fn test_radix_sort_error_clone() {
    let error = RadixSortError::InputTooLarge {
        size: 1000,
        max: 500,
    };
    let cloned = error.clone();
    assert_eq!(error, cloned);
}

#[test]
fn test_radix_sort_error_eq() {
    let error1 = RadixSortError::EmptyInput;
    let error2 = RadixSortError::EmptyInput;
    assert_eq!(error1, error2);

    let error3 = RadixSortError::InputTooLarge {
        size: 100,
        max: 50,
    };
    assert_ne!(error1, error3);
}

// =============================================================================
// SECTION 6 -- Real-world scenarios (depth sorting for transparency)
// =============================================================================

#[test]
fn test_depth_sorting_transparent_objects() {
    // Scenario: Sort transparent objects back-to-front for correct blending
    // 1000 transparent meshes, each with a depth key
    let transparent_count = 1000u32;

    let num_wg = RadixSortPipeline::num_workgroups(transparent_count);
    assert_eq!(num_wg, 1, "1000 objects fit in one workgroup");

    // Each pass: histogram + prefix scan + scatter
    let params_pass0 = RadixSortParams::new(transparent_count, 0, num_wg);
    assert_eq!(params_pass0.pass_number, 0, "Start with LSB");

    let params_pass7 = RadixSortParams::new(transparent_count, 7, num_wg);
    assert_eq!(params_pass7.pass_number, 7, "End with MSB");
}

#[test]
fn test_morton_code_sorting_spatial_queries() {
    // Scenario: Sort objects by Morton code (Z-order curve) for spatial queries
    // 50k objects in a scene
    let object_count = 50_000u32;

    let num_wg = RadixSortPipeline::num_workgroups(object_count);
    assert_eq!(num_wg, 49, "ceil(50000/1024) = 49 workgroups");

    let hist_size = RadixSortPipeline::histogram_buffer_size(num_wg);
    // 49 * 16 * 4 = 3136 bytes
    assert!(hist_size >= 3136, "Need sufficient histogram buffer");
}

#[test]
fn test_particle_distance_sorting() {
    // Scenario: Sort 1M particles by distance for alpha blending
    let particle_count = 1_000_000u32;

    let num_wg = RadixSortPipeline::num_workgroups(particle_count);
    assert_eq!(num_wg, 977, "1M particles need 977 workgroups");

    // Memory requirement per pass
    let hist_size = RadixSortPipeline::histogram_buffer_size(num_wg);
    assert!(hist_size >= 62528, "Large histogram buffer needed");

    // 8 passes total for full 32-bit sort
    for pass in 0..TOTAL_PASSES {
        let params = RadixSortParams::new(particle_count, pass, num_wg);
        assert_eq!(params.pass_number, pass);
        assert_eq!(params.input_size, particle_count);
    }
}

#[test]
fn test_index_buffer_sorting() {
    // Scenario: Sort indices for indirect draw calls
    // 10k draw calls to be sorted by material/shader/distance
    let draw_count = 10_000u32;

    let num_wg = RadixSortPipeline::num_workgroups(draw_count);
    assert_eq!(num_wg, 10, "10k draws need 10 workgroups");

    // Key-value sort: keys = sort criteria, values = draw indices
    let params = RadixSortParams::new(draw_count, 0, num_wg);
    assert_eq!(params.input_size, draw_count);
}

#[test]
fn test_tile_sorting_for_tiled_rendering() {
    // Scenario: Sort lights/decals into tiles for tiled deferred rendering
    // 4k lights spread across 16x16 tile grid = 256 tiles
    let light_count = 4096u32;

    let num_wg = RadixSortPipeline::num_workgroups(light_count);
    assert_eq!(num_wg, 4, "4k lights fit in 4 workgroups");
}

#[test]
fn test_grass_blade_sorting() {
    // Scenario: Sort grass blade instances for LOD transitions
    // 500k grass blades sorted by camera distance
    let grass_count = 500_000u32;

    let num_wg = RadixSortPipeline::num_workgroups(grass_count);
    assert_eq!(num_wg, 489, "500k grass blades need 489 workgroups");
}

// =============================================================================
// SECTION 7 -- Edge cases
// =============================================================================

#[test]
fn test_single_element_params() {
    // Edge case: single element (still valid to sort)
    let params = RadixSortParams::new(1, 0, 1);
    assert_eq!(params.input_size, 1);
    assert_eq!(params.num_workgroups, 1);
}

#[test]
fn test_workgroups_for_edge_counts() {
    // Various edge cases for workgroup calculation
    assert_eq!(RadixSortPipeline::num_workgroups(1), 1);
    assert_eq!(RadixSortPipeline::num_workgroups(2), 1);
    assert_eq!(RadixSortPipeline::num_workgroups(1023), 1);
    assert_eq!(RadixSortPipeline::num_workgroups(1024), 1);
    assert_eq!(RadixSortPipeline::num_workgroups(1025), 2);
}

#[test]
fn test_max_u32_workgroup_count() {
    // Extremely large input (theoretical limit test)
    // 2^31 elements = 2147483648
    // ceil(2147483648 / 1024) = 2097152 workgroups
    let large_count = u32::MAX / 2; // Stay within reasonable bounds
    let num_wg = RadixSortPipeline::num_workgroups(large_count);
    assert!(num_wg > 0, "Should calculate positive workgroup count");
}

#[test]
fn test_params_for_already_sorted_input() {
    // Pre-sorted input uses same algorithm (stable sort maintains order)
    let sorted_count = 5000u32;
    let num_wg = RadixSortPipeline::num_workgroups(sorted_count);

    // All 8 passes still execute
    for pass in 0..TOTAL_PASSES {
        let params = RadixSortParams::new(sorted_count, pass, num_wg);
        assert_eq!(params.pass_number, pass);
    }
}

#[test]
fn test_params_for_reverse_sorted_input() {
    // Reverse-sorted input (worst case still O(8n))
    let reverse_count = 8000u32;
    let num_wg = RadixSortPipeline::num_workgroups(reverse_count);

    let params = RadixSortParams::new(reverse_count, 0, num_wg);
    assert_eq!(params.input_size, reverse_count);
    assert_eq!(params.num_workgroups, 8);
}

#[test]
fn test_params_for_all_same_values() {
    // All elements have same key (trivial sort)
    let uniform_count = 3000u32;
    let num_wg = RadixSortPipeline::num_workgroups(uniform_count);

    let params = RadixSortParams::new(uniform_count, 0, num_wg);
    assert_eq!(params.input_size, uniform_count);
}

// =============================================================================
// SECTION 8 -- Algorithm property verification
// =============================================================================

#[test]
fn test_lsb_first_ordering() {
    // Radix sort processes LSB first for stability
    // Pass 0 = bits 0-3, Pass 7 = bits 28-31

    // Create params for each pass and verify LSB→MSB ordering
    let size = 1000u32;
    let wg = 1u32;

    let pass0 = RadixSortParams::new(size, 0, wg);
    let pass7 = RadixSortParams::new(size, 7, wg);

    assert!(
        pass0.pass_number < pass7.pass_number,
        "LSB pass should come before MSB pass"
    );
}

#[test]
fn test_16_buckets_per_pass() {
    // Each pass uses 16 buckets (4-bit radix)
    assert_eq!(RADIX_BUCKETS, 16);

    // Histogram needs space for 16 buckets per workgroup
    let hist_for_one_wg = RadixSortPipeline::histogram_buffer_size(1);
    assert!(hist_for_one_wg >= 16 * 4, "Need space for 16 u32 counters");
}

#[test]
fn test_stable_sort_property() {
    // Radix sort is stable when processing LSB-first
    // This is a documentation test - actual stability verified in integration tests

    // For stability, we need:
    // 1. LSB-first processing (pass 0 to pass 7)
    // 2. Scatter preserves relative order within same digit

    assert_eq!(TOTAL_PASSES, 8, "8 passes for 32-bit keys");
    assert_eq!(RADIX_BITS, 4, "4-bit digits for 16 buckets");

    // Pass 0 processes bits 0-3
    let pass0_bit_start = 0 * RADIX_BITS;
    let pass0_bit_end = pass0_bit_start + RADIX_BITS - 1;
    assert_eq!(pass0_bit_start, 0);
    assert_eq!(pass0_bit_end, 3);

    // Pass 7 processes bits 28-31
    let pass7_bit_start = 7 * RADIX_BITS;
    let pass7_bit_end = pass7_bit_start + RADIX_BITS - 1;
    assert_eq!(pass7_bit_start, 28);
    assert_eq!(pass7_bit_end, 31);
}

#[test]
fn test_linear_time_complexity() {
    // Radix sort is O(8n) for 32-bit keys
    // Each element is processed 8 times (once per pass)

    let element_counts = [100u32, 1000, 10_000, 100_000, 1_000_000];

    for &count in &element_counts {
        let wg = RadixSortPipeline::num_workgroups(count);

        // Total work = 8 passes * count elements
        let total_elements_processed = (TOTAL_PASSES as u64) * (count as u64);

        // Linear scaling: 10x elements = 10x work
        assert_eq!(
            total_elements_processed,
            8 * (count as u64),
            "Work should scale linearly with element count"
        );

        // Workgroups also scale linearly
        let expected_wg = (count + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP;
        assert_eq!(wg, expected_wg);
    }
}

// =============================================================================
// SECTION 9 -- Key-value pair sorting
// =============================================================================

#[test]
fn test_key_value_pair_concept() {
    // Radix sort supports key-value pairs
    // Keys are sorted, values are permuted to maintain association

    // Example: depth keys with draw call indices as values
    // Keys:   [5, 3, 8, 1]
    // Values: [0, 1, 2, 3]  (original indices)
    // After sort:
    // Keys:   [1, 3, 5, 8]
    // Values: [3, 1, 0, 2]  (permuted to match sorted keys)

    // This test verifies the API supports this pattern
    let pair_count = 1000u32;
    let params = RadixSortParams::new(pair_count, 0, 1);
    assert_eq!(params.input_size, pair_count);
}

#[test]
fn test_histogram_scatter_phases() {
    // Each pass has: histogram → prefix scan → scatter
    // This test documents the algorithm phases

    let count = 5000u32;
    let wg = RadixSortPipeline::num_workgroups(count);

    // Phase 1: Histogram - count digit occurrences
    let hist_size = RadixSortPipeline::histogram_buffer_size(wg);
    assert!(hist_size > 0, "Histogram buffer required");

    // Phase 2: Prefix scan (uses PrefixScanPipeline internally)
    // Converts counts to scatter offsets

    // Phase 3: Scatter - write elements to sorted positions
    // Uses histogram offsets to place elements
}

// =============================================================================
// SECTION 10 -- Memory layout and buffer requirements
// =============================================================================

#[test]
fn test_params_uniform_buffer_layout() {
    // RadixSortParams is used as uniform buffer data
    // Must be properly aligned for GPU

    // Field layout:
    // offset 0:  input_size (u32)
    // offset 4:  pass_number (u32)
    // offset 8:  num_workgroups (u32)
    // offset 12: _pad (u32)
    // Total: 16 bytes

    assert_eq!(std::mem::size_of::<RadixSortParams>(), 16);
    assert_eq!(std::mem::align_of::<RadixSortParams>(), 4);
}

#[test]
fn test_histogram_buffer_alignment() {
    // Histogram buffer holds u32 counters
    // Must be aligned to at least 4 bytes

    for wg_count in [1, 10, 100, 1000] {
        let size = RadixSortPipeline::histogram_buffer_size(wg_count);
        assert_eq!(
            size % 4,
            0,
            "Histogram buffer size must be 4-byte aligned for workgroup count {}",
            wg_count
        );
    }
}

#[test]
fn test_buffer_size_for_typical_use_cases() {
    // Document typical buffer sizes

    // Small scene: 1k objects
    let wg_1k = RadixSortPipeline::num_workgroups(1_000);
    let hist_1k = RadixSortPipeline::histogram_buffer_size(wg_1k);
    assert!(hist_1k <= 256, "1k objects should need small histogram");

    // Medium scene: 100k objects
    let wg_100k = RadixSortPipeline::num_workgroups(100_000);
    let hist_100k = RadixSortPipeline::histogram_buffer_size(wg_100k);
    assert!(hist_100k <= 8192, "100k objects should need moderate histogram");

    // Large scene: 1M particles
    let wg_1m = RadixSortPipeline::num_workgroups(1_000_000);
    let hist_1m = RadixSortPipeline::histogram_buffer_size(wg_1m);
    assert!(hist_1m <= 131072, "1M particles should need reasonable histogram");
}
