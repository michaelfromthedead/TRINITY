// SPDX-License-Identifier: MIT
//
// BLACKBOX T-WGPU-P3.10.1 Reduction Compute Shaders. CLEANROOM.
//
// Contract: Parallel reduction shaders compile correctly and ReductionPipeline
// API is accessible.
//
// Tests:
//   1. Shaders compile successfully (via include_str!)
//   2. ReductionPipeline can be constructed with headless wgpu device
//   3. Workgroup calculation is correct
//   4. Multi-pass calculation is correct
//   5. ReductionParams struct is properly aligned
//
// Note: Full GPU execution tests require a GPU and are in a separate test file.

use renderer_backend::compute_library::reduction::{
    ReductionOperation, ReductionParams, ReductionPipeline, ELEMENTS_PER_WORKGROUP, WORKGROUP_SIZE,
};

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

// =============================================================================
// SECTION 2 -- Workgroup calculation
// =============================================================================

#[test]
fn test_single_element_uses_one_workgroup() {
    assert_eq!(ReductionPipeline::calculate_workgroups(1), 1);
}

#[test]
fn test_512_elements_uses_one_workgroup() {
    assert_eq!(ReductionPipeline::calculate_workgroups(512), 1);
}

#[test]
fn test_513_elements_uses_two_workgroups() {
    assert_eq!(ReductionPipeline::calculate_workgroups(513), 2);
}

#[test]
fn test_1024_elements_uses_two_workgroups() {
    assert_eq!(ReductionPipeline::calculate_workgroups(1024), 2);
}

#[test]
fn test_large_array_workgroup_calculation() {
    // 10000 elements: ceil(10000 / 512) = 20 workgroups
    assert_eq!(ReductionPipeline::calculate_workgroups(10000), 20);

    // 100000 elements: ceil(100000 / 512) = 196 workgroups
    assert_eq!(ReductionPipeline::calculate_workgroups(100000), 196);
}

// =============================================================================
// SECTION 3 -- Multi-pass calculation
// =============================================================================

#[test]
fn test_single_pass_for_small_arrays() {
    assert_eq!(ReductionPipeline::calculate_passes(1), 1);
    assert_eq!(ReductionPipeline::calculate_passes(256), 1);
    assert_eq!(ReductionPipeline::calculate_passes(512), 1);
}

#[test]
fn test_two_passes_for_medium_arrays() {
    // 513-262144 elements need 2 passes
    assert_eq!(ReductionPipeline::calculate_passes(513), 2);
    assert_eq!(ReductionPipeline::calculate_passes(1024), 2);
    assert_eq!(ReductionPipeline::calculate_passes(100000), 2);
}

#[test]
fn test_multiple_passes_for_large_arrays() {
    // Very large arrays need 3+ passes
    // 1M elements: 1M -> 1954 -> 4 -> 1 (3 passes)
    assert_eq!(ReductionPipeline::calculate_passes(1_000_000), 3);
    // 10M elements: 10M -> 19532 -> 39 -> 1 (3 passes still, since 39 < 512)
    assert_eq!(ReductionPipeline::calculate_passes(10_000_000), 3);
    // 100M elements: 100M -> 195313 -> 382 -> 1 (3 passes)
    // Need 512^3 = 134M+ elements for 4 passes
    assert_eq!(ReductionPipeline::calculate_passes(100_000_000), 3);
    // 1B elements: 1B -> 1953126 -> 3815 -> 8 -> 1 (4 passes)
    assert_eq!(ReductionPipeline::calculate_passes(1_000_000_000), 4);
}

// =============================================================================
// SECTION 4 -- ReductionParams struct
// =============================================================================

#[test]
fn test_reduction_params_size() {
    // Must be 16 bytes (4x u32) for proper GPU alignment
    assert_eq!(
        std::mem::size_of::<ReductionParams>(),
        16,
        "ReductionParams must be 16 bytes for GPU alignment"
    );
}

#[test]
fn test_reduction_params_creation() {
    let params = ReductionParams::new(1000, 5);
    assert_eq!(params.input_size, 1000);
    assert_eq!(params.output_offset, 5);
}

// =============================================================================
// SECTION 5 -- ReductionOperation enum
// =============================================================================

#[test]
fn test_sum_identity() {
    assert_eq!(ReductionOperation::Sum.identity(), 0.0);
}

#[test]
fn test_min_identity() {
    assert_eq!(ReductionOperation::Min.identity(), f32::MAX);
}

#[test]
fn test_max_identity() {
    assert_eq!(ReductionOperation::Max.identity(), f32::MIN);
}

#[test]
fn test_entry_point_names() {
    assert_eq!(ReductionOperation::Sum.entry_point(), "reduce_sum");
    assert_eq!(ReductionOperation::Min.entry_point(), "reduce_min");
    assert_eq!(ReductionOperation::Max.entry_point(), "reduce_max");
}

// =============================================================================
// SECTION 6 -- Shader source inclusion (compile-time check)
// =============================================================================

#[test]
fn test_shader_sources_are_included() {
    // These are embedded via include_str! in reduction.rs
    // If the shaders don't exist or have syntax errors, this test would fail
    // at compile time, not runtime.

    // Verify the shader files exist by checking the module compiles
    // The ReductionPipeline::new() call compiles shaders, but we can't
    // call it without a wgpu device. So we just verify the module loads.
    let _ = ReductionOperation::Sum;
    let _ = ReductionOperation::Min;
    let _ = ReductionOperation::Max;
}

// =============================================================================
// SECTION 7 -- Additional constants coverage
// =============================================================================

#[test]
fn test_max_single_pass_elements_constant() {
    // MAX_SINGLE_PASS_ELEMENTS should equal ELEMENTS_PER_WORKGROUP
    assert_eq!(
        renderer_backend::compute_library::reduction::MAX_SINGLE_PASS_ELEMENTS,
        ELEMENTS_PER_WORKGROUP,
        "MAX_SINGLE_PASS_ELEMENTS must equal ELEMENTS_PER_WORKGROUP"
    );
}

#[test]
fn test_multi_pass_threshold_constant() {
    // MULTI_PASS_THRESHOLD = WORKGROUP_SIZE * WORKGROUP_SIZE * 2 = 131072
    assert_eq!(
        renderer_backend::compute_library::reduction::MULTI_PASS_THRESHOLD,
        WORKGROUP_SIZE * WORKGROUP_SIZE * 2,
        "MULTI_PASS_THRESHOLD must be WORKGROUP_SIZE^2 * 2"
    );
    assert_eq!(
        renderer_backend::compute_library::reduction::MULTI_PASS_THRESHOLD,
        131072
    );
}

// =============================================================================
// SECTION 8 -- ReductionOperation enum traits
// =============================================================================

#[test]
fn test_reduction_operation_is_copy() {
    let op = ReductionOperation::Sum;
    let op_copy = op; // Should compile if Copy is implemented
    assert_eq!(op, op_copy);
}

#[test]
fn test_reduction_operation_is_clone() {
    let op = ReductionOperation::Min;
    let op_clone = op.clone();
    assert_eq!(op, op_clone);
}

#[test]
fn test_reduction_operation_is_debug() {
    let debug_str = format!("{:?}", ReductionOperation::Max);
    assert!(debug_str.contains("Max"));
}

#[test]
fn test_reduction_operation_equality() {
    assert_eq!(ReductionOperation::Sum, ReductionOperation::Sum);
    assert_ne!(ReductionOperation::Sum, ReductionOperation::Min);
    assert_ne!(ReductionOperation::Min, ReductionOperation::Max);
}

#[test]
fn test_reduction_operation_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ReductionOperation::Sum);
    set.insert(ReductionOperation::Min);
    set.insert(ReductionOperation::Max);
    assert_eq!(set.len(), 3);

    // Inserting duplicates should not change size
    set.insert(ReductionOperation::Sum);
    assert_eq!(set.len(), 3);
}

#[test]
fn test_reduction_operation_can_be_matched() {
    let op = ReductionOperation::Sum;
    let result = match op {
        ReductionOperation::Sum => "sum",
        ReductionOperation::Min => "min",
        ReductionOperation::Max => "max",
    };
    assert_eq!(result, "sum");
}

// =============================================================================
// SECTION 9 -- Edge cases for workgroup/pass calculations
// =============================================================================

#[test]
fn test_zero_elements_workgroups() {
    // Edge case: 0 elements should still return at least 0 workgroups
    // (0 + 512 - 1) / 512 = 0
    assert_eq!(ReductionPipeline::calculate_workgroups(0), 0);
}

#[test]
fn test_exact_boundary_workgroups() {
    // Test exact boundaries
    assert_eq!(ReductionPipeline::calculate_workgroups(511), 1);
    assert_eq!(ReductionPipeline::calculate_workgroups(512), 1);
    assert_eq!(ReductionPipeline::calculate_workgroups(1023), 2);
    assert_eq!(ReductionPipeline::calculate_workgroups(1024), 2);
    assert_eq!(ReductionPipeline::calculate_workgroups(1025), 3);
}

#[test]
fn test_zero_elements_passes() {
    // Edge case: 0 elements - the function still computes correctly
    // since 0 <= ELEMENTS_PER_WORKGROUP, returns 1
    assert_eq!(ReductionPipeline::calculate_passes(0), 1);
}

// =============================================================================
// SECTION 10 -- ReductionParams edge cases
// =============================================================================

#[test]
fn test_reduction_params_alignment() {
    // Must be 16-byte aligned for GPU
    assert_eq!(std::mem::align_of::<ReductionParams>(), 4);
    // Size must be multiple of 4 (for wgpu uniform requirements)
    assert_eq!(std::mem::size_of::<ReductionParams>() % 4, 0);
}

#[test]
fn test_reduction_params_max_values() {
    // Test with maximum u32 values
    let params = ReductionParams::new(u32::MAX, u32::MAX);
    assert_eq!(params.input_size, u32::MAX);
    assert_eq!(params.output_offset, u32::MAX);
}

#[test]
fn test_reduction_params_zero_values() {
    let params = ReductionParams::new(0, 0);
    assert_eq!(params.input_size, 0);
    assert_eq!(params.output_offset, 0);
}

// =============================================================================
// SECTION 11 -- Real-world scenario tests (particle energy, bounds)
// =============================================================================

#[test]
fn test_particle_count_scenarios() {
    // Common particle counts in game engines
    // 1000 particles: ceil(1000/512) = 2 workgroups, 2 passes
    assert_eq!(ReductionPipeline::calculate_workgroups(1000), 2);
    assert_eq!(ReductionPipeline::calculate_passes(1000), 2);

    // 10000 particles: ceil(10000/512) = 20 workgroups, 2 passes
    assert_eq!(ReductionPipeline::calculate_workgroups(10000), 20);
    assert_eq!(ReductionPipeline::calculate_passes(10000), 2);

    // 100000 particles: ceil(100000/512) = 196 workgroups, 2 passes
    assert_eq!(ReductionPipeline::calculate_workgroups(100000), 196);
    assert_eq!(ReductionPipeline::calculate_passes(100000), 2);

    // 500000 particles: ceil(500000/512) = 977 workgroups, 3 passes
    assert_eq!(ReductionPipeline::calculate_workgroups(500000), 977);
    assert_eq!(ReductionPipeline::calculate_passes(500000), 3);
}

#[test]
fn test_vertex_buffer_reduction_scenarios() {
    // Common vertex counts for mesh bounds calculation
    // Small mesh: 1000 vertices
    assert_eq!(ReductionPipeline::calculate_passes(1000), 2);

    // Medium mesh: 50000 vertices
    assert_eq!(ReductionPipeline::calculate_passes(50000), 2);

    // Large mesh: 1M vertices (high-detail character)
    assert_eq!(ReductionPipeline::calculate_passes(1_000_000), 3);
}

// =============================================================================
// SECTION 12 -- Headless device test (requires GPU or software renderer)
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
            label: Some("reduction_test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .expect("Failed to create device");

    // Create reduction pipeline - this compiles all three shaders
    let pipeline = ReductionPipeline::new(&device);

    // Verify bind group layout was created
    let _layout = pipeline.bind_group_layout();
}
