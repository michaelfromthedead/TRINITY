// SPDX-License-Identifier: MIT
//
// BLACKBOX T-WGPU-P3.10.2 Prefix Scan Compute Shaders. CLEANROOM.
//
// Contract: Prefix scan (parallel scan) shaders compile correctly and
// PrefixScanPipeline API is accessible.
//
// Tests:
//   1. Shader constants are correctly defined
//   2. Workgroup calculation is correct
//   3. Block sums buffer size calculation
//   4. ScanParams struct (exclusive/inclusive modes, offset)
//   5. PrefixScanError enum (Display, std::error::Error)
//   6. Pipeline creation with headless wgpu device
//   7. Real-world scenarios (particles, stream compaction, histograms)
//
// Note: Full GPU execution tests require a GPU and are in a separate test file.

use renderer_backend::compute_library::prefix_scan::{
    PrefixScanError, PrefixScanPipeline, ScanParams, ELEMENTS_PER_WORKGROUP, WORKGROUP_SIZE,
};
use std::error::Error;

// =============================================================================
// SECTION 1 -- Constants are correctly defined
// =============================================================================

#[test]
fn test_workgroup_size_constant() {
    assert_eq!(WORKGROUP_SIZE, 256, "Workgroup size must be 256 threads");
}

#[test]
fn test_elements_per_workgroup_constant() {
    assert_eq!(
        ELEMENTS_PER_WORKGROUP,
        512,
        "Each workgroup processes 512 elements (2 per thread)"
    );
}

#[test]
fn test_elements_per_workgroup_is_double_workgroup_size() {
    assert_eq!(
        ELEMENTS_PER_WORKGROUP,
        WORKGROUP_SIZE * 2,
        "Each thread processes 2 elements in work-efficient scan"
    );
}

// =============================================================================
// SECTION 2 -- ScanParams API
// =============================================================================

#[test]
fn test_scan_params_exclusive_creation() {
    let params = ScanParams::exclusive(1000);
    assert_eq!(params.input_size, 1000);
    assert_eq!(params.is_inclusive, 0, "Exclusive scan should have is_inclusive = 0");
}

#[test]
fn test_scan_params_inclusive_creation() {
    let params = ScanParams::inclusive(1000);
    assert_eq!(params.input_size, 1000);
    assert_eq!(params.is_inclusive, 1, "Inclusive scan should have is_inclusive = 1");
}

#[test]
fn test_scan_params_with_offset() {
    let params = ScanParams::exclusive(500).with_offset(128);
    assert_eq!(params.input_size, 500);
    assert_eq!(params.block_offset, 128);
    assert_eq!(params.is_inclusive, 0);
}

#[test]
fn test_scan_params_default_offset_is_zero() {
    let params = ScanParams::exclusive(100);
    assert_eq!(params.block_offset, 0, "Default block_offset should be 0");
}

#[test]
fn test_scan_params_size_alignment() {
    // Must be 16 bytes (4x u32) for proper GPU alignment
    assert_eq!(
        std::mem::size_of::<ScanParams>(),
        16,
        "ScanParams must be 16 bytes for GPU alignment"
    );
}

#[test]
fn test_scan_params_small_sizes() {
    // Edge case: single element
    let params = ScanParams::exclusive(1);
    assert_eq!(params.input_size, 1);

    // Two elements
    let params = ScanParams::inclusive(2);
    assert_eq!(params.input_size, 2);
}

#[test]
fn test_scan_params_large_sizes() {
    // Large particle count
    let params = ScanParams::exclusive(1_000_000);
    assert_eq!(params.input_size, 1_000_000);

    // Very large
    let params = ScanParams::inclusive(10_000_000);
    assert_eq!(params.input_size, 10_000_000);
}

#[test]
fn test_scan_params_builder_chain() {
    // Chaining should work
    let params = ScanParams::inclusive(256)
        .with_offset(64);
    assert_eq!(params.input_size, 256);
    assert_eq!(params.block_offset, 64);
    assert_eq!(params.is_inclusive, 1);
}

// =============================================================================
// SECTION 3 -- Workgroup calculation (num_workgroups)
// =============================================================================

#[test]
fn test_single_element_uses_one_workgroup() {
    assert_eq!(PrefixScanPipeline::num_workgroups(1), 1);
}

#[test]
fn test_512_elements_uses_one_workgroup() {
    // One workgroup handles 512 elements
    assert_eq!(PrefixScanPipeline::num_workgroups(512), 1);
}

#[test]
fn test_513_elements_uses_two_workgroups() {
    // 513 elements exceed single workgroup capacity
    assert_eq!(PrefixScanPipeline::num_workgroups(513), 2);
}

#[test]
fn test_1024_elements_uses_two_workgroups() {
    // ceil(1024 / 512) = 2
    assert_eq!(PrefixScanPipeline::num_workgroups(1024), 2);
}

#[test]
fn test_workgroup_boundary_exact() {
    // Exactly at workgroup boundary
    assert_eq!(PrefixScanPipeline::num_workgroups(512), 1);
    assert_eq!(PrefixScanPipeline::num_workgroups(1024), 2);
    assert_eq!(PrefixScanPipeline::num_workgroups(1536), 3);
    assert_eq!(PrefixScanPipeline::num_workgroups(2048), 4);
}

#[test]
fn test_workgroup_calculation_large_arrays() {
    // 10000 elements: ceil(10000 / 512) = 20
    assert_eq!(PrefixScanPipeline::num_workgroups(10_000), 20);

    // 100000 elements: ceil(100000 / 512) = 196
    assert_eq!(PrefixScanPipeline::num_workgroups(100_000), 196);

    // 1M elements: ceil(1_000_000 / 512) = 1954
    assert_eq!(PrefixScanPipeline::num_workgroups(1_000_000), 1954);
}

#[test]
fn test_workgroup_calculation_formula() {
    // Verify formula: ceil(n / ELEMENTS_PER_WORKGROUP)
    for n in [1, 100, 511, 512, 513, 1000, 5000, 10000] {
        let expected = (n + ELEMENTS_PER_WORKGROUP - 1) / ELEMENTS_PER_WORKGROUP;
        assert_eq!(
            PrefixScanPipeline::num_workgroups(n),
            expected,
            "Mismatch for n={}", n
        );
    }
}

// =============================================================================
// SECTION 4 -- Block sums buffer size calculation
// =============================================================================

#[test]
fn test_block_sums_buffer_single_block() {
    // Single block doesn't need block sums
    // But API might return minimum size
    let size = PrefixScanPipeline::block_sums_buffer_size(512);
    // At minimum, should be able to store partial sums
    assert!(size >= 4, "Buffer should hold at least one u32/f32");
}

#[test]
fn test_block_sums_buffer_multi_block() {
    // Two blocks need space for 2 partial sums
    let size = PrefixScanPipeline::block_sums_buffer_size(1024);
    // 2 blocks * 4 bytes per sum = 8 bytes minimum
    assert!(size >= 8, "Two blocks need at least 8 bytes");
}

#[test]
fn test_block_sums_buffer_scales_with_workgroups() {
    // More workgroups require more space for partial sums
    let size_small = PrefixScanPipeline::block_sums_buffer_size(1024);
    let size_large = PrefixScanPipeline::block_sums_buffer_size(10_000);

    // Large input should need more space
    assert!(
        size_large >= size_small,
        "Larger input should need at least as much block sums space"
    );
}

#[test]
fn test_block_sums_buffer_realistic_sizes() {
    // 1M particles: 1954 workgroups
    let size_1m = PrefixScanPipeline::block_sums_buffer_size(1_000_000);
    // Should be at least 1954 * sizeof(u32) = 7816 bytes
    assert!(
        size_1m >= 1954 * 4,
        "1M elements should have sufficient block sums buffer"
    );
}

// =============================================================================
// SECTION 5 -- PrefixScanError error handling
// =============================================================================

#[test]
fn test_prefix_scan_error_implements_display() {
    // Verify Error enum implements Display
    fn requires_display<T: std::fmt::Display>(_: &T) {}

    let error = PrefixScanError::EmptyInput;
    requires_display(&error);
}

#[test]
fn test_prefix_scan_error_implements_std_error() {
    // Verify Error enum implements std::error::Error
    fn requires_error<T: Error>(_: &T) {}

    let error = PrefixScanError::EmptyInput;
    requires_error(&error);
}

#[test]
fn test_empty_input_error_message() {
    let error = PrefixScanError::EmptyInput;
    let msg = error.to_string();

    // Should contain meaningful message
    assert!(!msg.is_empty(), "Error message should not be empty");
}

#[test]
fn test_input_too_large_error() {
    let error = PrefixScanError::InputTooLarge {
        size: 100_000_000,
        max: 67_108_864, // 64M typical limit
    };
    let msg = error.to_string();

    // Should mention the sizes
    assert!(!msg.is_empty(), "Error message should not be empty");
    // Could verify it contains the actual values, but that's implementation detail
}

#[test]
fn test_buffer_too_small_error() {
    let error = PrefixScanError::BufferTooSmall {
        required: 4096,
        actual: 1024,
    };
    let msg = error.to_string();

    assert!(!msg.is_empty(), "Error message should not be empty");
}

// =============================================================================
// SECTION 6 -- Real-world scenarios (API usage patterns)
// =============================================================================

#[test]
fn test_particle_index_generation_params() {
    // Common scenario: generate indices for 100k particles
    let particle_count = 100_000u32;

    let params = ScanParams::exclusive(particle_count);
    assert_eq!(params.input_size, particle_count);
    assert_eq!(params.is_inclusive, 0); // Exclusive for index generation

    let workgroups = PrefixScanPipeline::num_workgroups(particle_count);
    assert_eq!(workgroups, 196); // ceil(100000/512)
}

#[test]
fn test_stream_compaction_prep_params() {
    // Stream compaction: scan flags to get output indices
    // Inclusive scan to get final count
    let flag_count = 50_000u32;

    let params = ScanParams::inclusive(flag_count);
    assert_eq!(params.input_size, flag_count);
    assert_eq!(params.is_inclusive, 1);
}

#[test]
fn test_histogram_prefix_sum_params() {
    // Histogram prefix sums for radix sort
    // Typically 256 buckets per pass
    let bucket_count = 256u32;

    let params = ScanParams::exclusive(bucket_count);
    assert_eq!(params.input_size, bucket_count);

    // Single workgroup handles 256 buckets
    let workgroups = PrefixScanPipeline::num_workgroups(bucket_count);
    assert_eq!(workgroups, 1);
}

#[test]
fn test_multi_block_histogram_params() {
    // Multi-block histogram: 256 buckets * 4096 blocks = 1M entries
    let histogram_size = 256u32 * 4096;

    let _params = ScanParams::exclusive(histogram_size);
    let workgroups = PrefixScanPipeline::num_workgroups(histogram_size);

    // ceil(1048576 / 512) = 2048 workgroups
    assert_eq!(workgroups, 2048);

    // Need block sums buffer
    let block_sums_size = PrefixScanPipeline::block_sums_buffer_size(histogram_size);
    assert!(block_sums_size >= 2048 * 4);
}

#[test]
fn test_culling_result_compaction() {
    // GPU culling: scan visibility flags
    // 10k draw calls, compact visible ones
    let draw_count = 10_000u32;

    // Exclusive scan on visibility flags
    let _params = ScanParams::exclusive(draw_count);

    let workgroups = PrefixScanPipeline::num_workgroups(draw_count);
    assert_eq!(workgroups, 20);
}

// =============================================================================
// SECTION 7 -- Headless device test (requires GPU or software renderer)
// =============================================================================

/// This test requires a wgpu adapter. It will be skipped if no adapter
/// is available (e.g., in headless CI without GPU).
#[test]
fn test_pipeline_creation_with_headless_device() {
    // Try to create a headless wgpu instance
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        dx12_shader_compiler: Default::default(),
        flags: wgpu::InstanceFlags::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::default(),
    });

    // Request adapter
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));

    // Skip test if no adapter available
    let adapter = match adapter {
        Some(a) => a,
        None => {
            eprintln!("SKIPPED: No wgpu adapter available for headless test");
            return;
        }
    };

    // Request device
    let (device, _queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("prefix_scan_test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .expect("Failed to create device");

    // Create prefix scan pipeline - this compiles all shaders
    let pipeline = PrefixScanPipeline::new(&device);

    // Verify bind group layout was created
    let _layout = pipeline.bind_group_layout();
}

#[test]
fn test_pipeline_components_with_headless_device() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        dx12_shader_compiler: Default::default(),
        flags: wgpu::InstanceFlags::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::default(),
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));

    let adapter = match adapter {
        Some(a) => a,
        None => {
            eprintln!("SKIPPED: No wgpu adapter available");
            return;
        }
    };

    let (device, _queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("prefix_scan_test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .expect("Failed to create device");

    let pipeline = PrefixScanPipeline::new(&device);

    // Test that all pipeline components are accessible
    // These verify the up-sweep, down-sweep, and block operations are compiled
    let _single_block = pipeline.single_block_pipeline();
    let _up_sweep = pipeline.up_sweep_pipeline();
    let _down_sweep = pipeline.down_sweep_pipeline();
    let _add_block_sums = pipeline.add_block_sums_pipeline();

    // All pipelines exist - the 4 phases of Blelloch scan
}

// =============================================================================
// SECTION 8 -- Multi-block scan requirements
// =============================================================================

#[test]
fn test_multi_block_scan_requires_phases() {
    // For arrays larger than ELEMENTS_PER_WORKGROUP (512), we need:
    // 1. Up-sweep (local): each block computes local prefix sums
    // 2. Block sums: store partial sums from each block
    // 3. Scan block sums: recursive scan on block partial sums
    // 4. Down-sweep: add scanned block sums back

    let large_input = 10_000u32;
    let workgroups = PrefixScanPipeline::num_workgroups(large_input);

    // Multiple workgroups means we need multi-block algorithm
    assert!(workgroups > 1, "Large inputs need multiple workgroups");

    // Block sums buffer needed
    let block_buffer_size = PrefixScanPipeline::block_sums_buffer_size(large_input);
    assert!(block_buffer_size > 0, "Multi-block scan needs block sums buffer");
}

#[test]
fn test_single_block_scan_is_simpler() {
    // For arrays fitting in one workgroup, single dispatch suffices
    let small_input = 256u32;
    let workgroups = PrefixScanPipeline::num_workgroups(small_input);

    assert_eq!(workgroups, 1, "Small input fits in single workgroup");
}

// =============================================================================
// SECTION 9 -- Boundary conditions and edge cases
// =============================================================================

#[test]
fn test_exact_workgroup_boundary_512() {
    let params = ScanParams::exclusive(512);
    let workgroups = PrefixScanPipeline::num_workgroups(512);

    assert_eq!(params.input_size, 512);
    assert_eq!(workgroups, 1, "Exactly 512 elements fits in one workgroup");
}

#[test]
fn test_one_past_workgroup_boundary() {
    let workgroups = PrefixScanPipeline::num_workgroups(513);
    assert_eq!(workgroups, 2, "513 elements needs two workgroups");
}

#[test]
fn test_power_of_two_sizes() {
    // Common power-of-two buffer sizes
    assert_eq!(PrefixScanPipeline::num_workgroups(256), 1);
    assert_eq!(PrefixScanPipeline::num_workgroups(512), 1);
    assert_eq!(PrefixScanPipeline::num_workgroups(1024), 2);
    assert_eq!(PrefixScanPipeline::num_workgroups(2048), 4);
    assert_eq!(PrefixScanPipeline::num_workgroups(4096), 8);
    assert_eq!(PrefixScanPipeline::num_workgroups(8192), 16);
    assert_eq!(PrefixScanPipeline::num_workgroups(65536), 128);
}

#[test]
fn test_non_power_of_two_sizes() {
    // Verify handling of arbitrary sizes
    assert_eq!(PrefixScanPipeline::num_workgroups(1000), 2);
    assert_eq!(PrefixScanPipeline::num_workgroups(1500), 3);
    assert_eq!(PrefixScanPipeline::num_workgroups(7777), 16);
}

// =============================================================================
// SECTION 10 -- API consistency checks
// =============================================================================

#[test]
fn test_scan_params_modes_are_distinct() {
    let exclusive = ScanParams::exclusive(100);
    let inclusive = ScanParams::inclusive(100);

    assert_ne!(
        exclusive.is_inclusive,
        inclusive.is_inclusive,
        "Exclusive and inclusive modes should differ"
    );
}

#[test]
fn test_offset_builder_doesnt_change_mode() {
    let exclusive = ScanParams::exclusive(100).with_offset(10);
    let inclusive = ScanParams::inclusive(100).with_offset(10);

    assert_eq!(exclusive.is_inclusive, 0);
    assert_eq!(inclusive.is_inclusive, 1);
}

#[test]
fn test_offset_builder_doesnt_change_size() {
    let params = ScanParams::exclusive(500).with_offset(64);
    assert_eq!(params.input_size, 500);
}
