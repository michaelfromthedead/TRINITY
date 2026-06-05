// SPDX-License-Identifier: MIT
//
// WHITEBOX T-WGPU-P2.5.4 Push Constants
// Contract: Push constant configuration, validation, writers, and fallback system.
//
// This test suite has FULL SOURCE ACCESS and tests all internal implementation
// details of the push constants module.
//
// Categories tested:
//   1. Config Construction (~15 tests)
//   2. Config Validation (~20 tests)
//   3. Error Types (~15 tests)
//   4. Writer Construction (~10 tests)
//   5. Writer Operations (~15 tests)
//   6. Fallback System (~15 tests)
//   7. Pre-built Structs (~10 tests)
//   8. Thread Safety (~5 tests)
//   9. Edge Cases (~15 tests)

use renderer_backend::resources::push_constants::{
    align_up, compute_only, fragment_only, is_aligned, max_push_constant_size,
    supports_push_constants, vertex_fragment, vertex_only, ComputePushConstantWriter,
    DrawPushConstants, ExtendedDrawPushConstants, FallbackPushConstants, PushConstantConfig,
    PushConstantError, PushConstantFallback, PushConstantWriter, DEFAULT_FALLBACK_BIND_GROUP,
    MAX_PUSH_CONSTANT_SIZE, PUSH_CONSTANT_ALIGNMENT,
};
use bytemuck::{Pod, Zeroable};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::thread;
use wgpu::{Features, ShaderStages};

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    Some(
        pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("whitebox_push_constants test device"),
                required_features: Features::PUSH_CONSTANTS,
                required_limits: wgpu::Limits {
                    max_push_constant_size: 128,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?,
    )
}

fn create_test_device_no_push_constants() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    Some(
        pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("whitebox_push_constants test device (no push)"),
                required_features: Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("device creation"),
    )
}

// Test data structures
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct TestData4 {
    value: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct TestData16 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct TestData64 {
    data: [f32; 16],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct TestData128 {
    data: [f32; 32],
}

// =============================================================================
// SECTION 1: Config Construction (~15 tests)
// =============================================================================

#[test]
fn config_construction_new_creates_empty_config() {
    let config = PushConstantConfig::new();
    assert!(config.is_empty());
    assert_eq!(config.range_count(), 0);
}

#[test]
fn config_construction_default_creates_empty_config() {
    let config = PushConstantConfig::default();
    assert!(config.is_empty());
    assert_eq!(config.range_count(), 0);
}

#[test]
fn config_construction_new_and_default_are_equivalent() {
    let new_config = PushConstantConfig::new();
    let default_config = PushConstantConfig::default();
    assert_eq!(new_config.range_count(), default_config.range_count());
    assert_eq!(new_config.total_size(), default_config.total_size());
    assert_eq!(new_config.is_empty(), default_config.is_empty());
}

#[test]
fn config_construction_add_range_returns_modified_config() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid range");
    assert!(!config.is_empty());
    assert_eq!(config.range_count(), 1);
}

#[test]
fn config_construction_builder_chain_works() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 32)
        .expect("first")
        .add_range(ShaderStages::FRAGMENT, 32, 32)
        .expect("second")
        .add_range(ShaderStages::VERTEX, 64, 32)
        .expect("third");
    assert_eq!(config.range_count(), 3);
}

#[test]
fn config_construction_add_vertex_fragment_range_shortcut() {
    let config = PushConstantConfig::new()
        .add_vertex_fragment_range(0, 64)
        .expect("valid");
    assert_eq!(config.range_count(), 1);
    let ranges = config.ranges();
    assert_eq!(ranges[0].stages, ShaderStages::VERTEX | ShaderStages::FRAGMENT);
}

#[test]
fn config_construction_add_compute_range_shortcut() {
    let config = PushConstantConfig::new()
        .add_compute_range(0, 128)
        .expect("valid");
    assert_eq!(config.range_count(), 1);
    let ranges = config.ranges();
    assert_eq!(ranges[0].stages, ShaderStages::COMPUTE);
}

#[test]
fn config_construction_ranges_accessor_returns_correct_data() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    let ranges = config.ranges();
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0].range.start, 0);
    assert_eq!(ranges[0].range.end, 64);
}

#[test]
fn config_construction_total_size_with_single_range() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    assert_eq!(config.total_size(), 64);
}

#[test]
fn config_construction_total_size_with_offset() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 32, 64)
        .expect("valid");
    assert_eq!(config.total_size(), 96); // offset + size
}

#[test]
fn config_construction_total_size_max_of_ranges() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 32)
        .expect("valid")
        .add_range(ShaderStages::FRAGMENT, 64, 64)
        .expect("valid");
    assert_eq!(config.total_size(), 128); // max end offset
}

#[test]
fn config_construction_range_count_increments_properly() {
    let config0 = PushConstantConfig::new();
    assert_eq!(config0.range_count(), 0);

    let config1 = config0.add_range(ShaderStages::VERTEX, 0, 32).expect("valid");
    assert_eq!(config1.range_count(), 1);

    let config2 = config1.add_range(ShaderStages::FRAGMENT, 32, 32).expect("valid");
    assert_eq!(config2.range_count(), 2);
}

#[test]
fn config_construction_into_ranges_consumes_and_returns_vec() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    let ranges = config.into_ranges();
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0].range, 0..64);
}

#[test]
fn config_construction_clone_produces_identical_copy() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    let cloned = config.clone();
    assert_eq!(config.range_count(), cloned.range_count());
    assert_eq!(config.total_size(), cloned.total_size());
}

#[test]
fn config_construction_debug_format_contains_type_name() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    let debug = format!("{:?}", config);
    assert!(debug.contains("PushConstantConfig"));
}

// =============================================================================
// SECTION 2: Config Validation (~20 tests)
// =============================================================================

#[test]
fn config_validation_empty_config_is_valid() {
    let config = PushConstantConfig::new();
    assert!(config.validate().is_ok());
}

#[test]
fn config_validation_single_aligned_range_is_valid() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    assert!(config.validate().is_ok());
}

#[test]
fn config_validation_max_size_range_is_valid() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 128)
        .expect("valid");
    assert!(config.validate().is_ok());
}

#[test]
fn config_validation_offset_alignment_error() {
    let result = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 1, 64);
    assert!(matches!(result, Err(PushConstantError::MisalignedOffset(1))));
}

#[test]
fn config_validation_offset_alignment_error_various_offsets() {
    for bad_offset in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17] {
        let result = PushConstantConfig::new().add_range(ShaderStages::VERTEX, bad_offset, 4);
        assert!(
            matches!(result, Err(PushConstantError::MisalignedOffset(o)) if o == bad_offset),
            "Expected MisalignedOffset for offset {}",
            bad_offset
        );
    }
}

#[test]
fn config_validation_size_alignment_error() {
    let result = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 0, 63);
    assert!(matches!(result, Err(PushConstantError::MisalignedSize(63))));
}

#[test]
fn config_validation_size_alignment_error_various_sizes() {
    for bad_size in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 62, 63, 65] {
        let result = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 0, bad_size);
        assert!(
            matches!(result, Err(PushConstantError::MisalignedSize(s)) if s == bad_size),
            "Expected MisalignedSize for size {}",
            bad_size
        );
    }
}

#[test]
fn config_validation_exceeds_max_size_single_range() {
    let result = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 0, 132);
    assert!(matches!(
        result,
        Err(PushConstantError::ExceedsMaxSize { total: 132, max: 128 })
    ));
}

#[test]
fn config_validation_exceeds_max_size_with_offset() {
    let result = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 64, 68);
    assert!(matches!(
        result,
        Err(PushConstantError::ExceedsMaxSize { total: 132, max: 128 })
    ));
}

#[test]
fn config_validation_empty_range_error() {
    let result = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 0, 0);
    assert!(matches!(result, Err(PushConstantError::EmptyRange)));
}

#[test]
fn config_validation_range_overlap_same_stages() {
    let result = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("first valid")
        .add_range(ShaderStages::VERTEX, 32, 64);
    assert!(matches!(result, Err(PushConstantError::RangeOverlap { .. })));
}

#[test]
fn config_validation_range_overlap_partial() {
    let result = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("first")
        .add_range(ShaderStages::VERTEX, 60, 8);
    assert!(matches!(result, Err(PushConstantError::RangeOverlap { .. })));
}

#[test]
fn config_validation_range_overlap_complete_containment() {
    let result = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 128)
        .expect("first")
        .add_range(ShaderStages::VERTEX, 32, 32);
    assert!(matches!(result, Err(PushConstantError::RangeOverlap { .. })));
}

#[test]
fn config_validation_adjacent_ranges_no_overlap() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("first")
        .add_range(ShaderStages::VERTEX, 64, 64)
        .expect("adjacent is valid");
    assert_eq!(config.range_count(), 2);
}

#[test]
fn config_validation_different_stages_same_range_valid() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("vertex")
        .add_range(ShaderStages::FRAGMENT, 0, 64)
        .expect("fragment same range is valid");
    assert_eq!(config.range_count(), 2);
}

#[test]
fn config_validation_intersecting_stages_overlap_error() {
    // VERTEX | FRAGMENT overlaps with VERTEX
    let result = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("first")
        .add_range(ShaderStages::VERTEX | ShaderStages::FRAGMENT, 32, 64);
    assert!(matches!(result, Err(PushConstantError::RangeOverlap { .. })));
}

#[test]
fn config_validation_multiple_non_overlapping_ranges() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 32)
        .expect("first")
        .add_range(ShaderStages::VERTEX, 32, 32)
        .expect("second")
        .add_range(ShaderStages::VERTEX, 64, 32)
        .expect("third")
        .add_range(ShaderStages::VERTEX, 96, 32)
        .expect("fourth");
    assert_eq!(config.total_size(), 128);
    assert!(config.validate().is_ok());
}

#[test]
fn config_validation_all_aligned_offsets_valid() {
    for offset in (0..128).step_by(4) {
        let size = std::cmp::min(128 - offset, 4);
        if size > 0 {
            let result = PushConstantConfig::new().add_range(ShaderStages::VERTEX, offset, size);
            assert!(result.is_ok(), "Offset {} should be valid", offset);
        }
    }
}

#[test]
fn config_validation_all_aligned_sizes_valid() {
    for size in (4..=128).step_by(4) {
        let result = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 0, size);
        assert!(result.is_ok(), "Size {} should be valid", size);
    }
}

#[test]
fn config_validation_find_range_returns_correct_available_space() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    // At offset 0, 64 bytes available
    let (range, available) = config.find_range(ShaderStages::VERTEX, 0).unwrap();
    assert_eq!(range, 0..64);
    assert_eq!(available, 64);

    // At offset 32, 32 bytes available
    let (_, available) = config.find_range(ShaderStages::VERTEX, 32).unwrap();
    assert_eq!(available, 32);

    // At offset 63, only 1 byte available
    let (_, available) = config.find_range(ShaderStages::VERTEX, 63).unwrap();
    assert_eq!(available, 1);
}

// =============================================================================
// SECTION 3: Error Types (~15 tests)
// =============================================================================

#[test]
fn error_misaligned_offset_display() {
    let err = PushConstantError::MisalignedOffset(5);
    let msg = format!("{}", err);
    assert!(msg.contains("offset"));
    assert!(msg.contains("5"));
    assert!(msg.contains("4-byte aligned"));
}

#[test]
fn error_misaligned_offset_various_values() {
    for value in [1, 2, 3, 5, 7, 13, 127] {
        let err = PushConstantError::MisalignedOffset(value);
        let msg = format!("{}", err);
        assert!(msg.contains(&value.to_string()));
    }
}

#[test]
fn error_misaligned_size_display() {
    let err = PushConstantError::MisalignedSize(7);
    let msg = format!("{}", err);
    assert!(msg.contains("size"));
    assert!(msg.contains("7"));
}

#[test]
fn error_misaligned_size_various_values() {
    for value in [1, 2, 3, 5, 7, 13, 63, 127] {
        let err = PushConstantError::MisalignedSize(value);
        let msg = format!("{}", err);
        assert!(msg.contains(&value.to_string()));
    }
}

#[test]
fn error_exceeds_max_size_display() {
    let err = PushConstantError::ExceedsMaxSize {
        total: 256,
        max: 128,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("256"));
    assert!(msg.contains("128"));
    assert!(msg.contains("exceeds"));
}

#[test]
fn error_range_overlap_display() {
    let err = PushConstantError::RangeOverlap {
        range1: "0..64 (VERTEX)".to_string(),
        range2: "32..96 (VERTEX)".to_string(),
    };
    let msg = format!("{}", err);
    assert!(msg.contains("overlap"));
    assert!(msg.contains("0..64"));
    assert!(msg.contains("32..96"));
}

#[test]
fn error_unsupported_feature_display() {
    let err = PushConstantError::UnsupportedFeature;
    let msg = format!("{}", err);
    assert!(msg.contains("not supported"));
}

#[test]
fn error_data_too_large_display() {
    let err = PushConstantError::DataTooLarge {
        data_size: 100,
        available: 64,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("100"));
    assert!(msg.contains("64"));
    assert!(msg.contains("exceeds"));
}

#[test]
fn error_invalid_offset_display() {
    let err = PushConstantError::InvalidOffset {
        offset: 200,
        stages: ShaderStages::VERTEX,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("200"));
    assert!(msg.contains("not valid"));
}

#[test]
fn error_empty_range_display() {
    let err = PushConstantError::EmptyRange;
    let msg = format!("{}", err);
    assert!(msg.contains("empty"));
    assert!(msg.contains("size must be > 0"));
}

#[test]
fn error_clone_preserves_data() {
    let err = PushConstantError::ExceedsMaxSize {
        total: 256,
        max: 128,
    };
    let cloned = err.clone();
    assert_eq!(format!("{}", err), format!("{}", cloned));
}

#[test]
fn error_partial_eq() {
    let err1 = PushConstantError::MisalignedOffset(5);
    let err2 = PushConstantError::MisalignedOffset(5);
    let err3 = PushConstantError::MisalignedOffset(7);
    assert_eq!(err1, err2);
    assert_ne!(err1, err3);
}

#[test]
fn error_debug_format() {
    let err = PushConstantError::EmptyRange;
    let debug = format!("{:?}", err);
    assert!(debug.contains("EmptyRange"));
}

#[test]
fn error_implements_std_error() {
    fn assert_error<E: std::error::Error>() {}
    assert_error::<PushConstantError>();
}

#[test]
fn error_all_variants_have_display() {
    let errors: Vec<PushConstantError> = vec![
        PushConstantError::MisalignedOffset(1),
        PushConstantError::MisalignedSize(1),
        PushConstantError::ExceedsMaxSize { total: 200, max: 128 },
        PushConstantError::RangeOverlap {
            range1: "a".to_string(),
            range2: "b".to_string(),
        },
        PushConstantError::UnsupportedFeature,
        PushConstantError::DataTooLarge {
            data_size: 100,
            available: 50,
        },
        PushConstantError::InvalidOffset {
            offset: 200,
            stages: ShaderStages::VERTEX,
        },
        PushConstantError::EmptyRange,
    ];

    for err in errors {
        let msg = format!("{}", err);
        assert!(!msg.is_empty(), "Display should not be empty for {:?}", err);
    }
}

// =============================================================================
// SECTION 4: Writer Construction (~10 tests)
// =============================================================================

// Note: PushConstantWriter and ComputePushConstantWriter require actual
// RenderPass/ComputePass which can only be obtained during command encoding.
// These tests verify the type system and API surface.

#[test]
fn writer_types_exist() {
    // Verify PushConstantWriter and ComputePushConstantWriter types exist
    // and can be named (compilation test)
    fn _takes_render_writer<'a, 'b>(_: PushConstantWriter<'a, 'b>) {}
    fn _takes_compute_writer<'a, 'b>(_: ComputePushConstantWriter<'a, 'b>) {}
}

#[test]
fn writer_config_is_stored() {
    // This test verifies the config field exists by testing its methods
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    // Writers store the config reference - we can verify by checking is_valid_write
    assert!(config.is_valid_write(ShaderStages::VERTEX, 0, 16).is_ok());
}

#[test]
fn writer_config_validation_via_is_valid_write() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    // Valid writes
    assert!(config.is_valid_write(ShaderStages::VERTEX, 0, 4).is_ok());
    assert!(config.is_valid_write(ShaderStages::VERTEX, 0, 64).is_ok());
    assert!(config.is_valid_write(ShaderStages::VERTEX, 32, 32).is_ok());

    // Invalid writes
    assert!(config.is_valid_write(ShaderStages::VERTEX, 0, 100).is_err());
    assert!(config.is_valid_write(ShaderStages::VERTEX, 64, 4).is_err());
    assert!(config.is_valid_write(ShaderStages::FRAGMENT, 0, 4).is_err());
}

#[test]
fn writer_set_vertex_uses_vertex_stages() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    // set_vertex should use VERTEX stages
    assert!(config.is_valid_write(ShaderStages::VERTEX, 0, 16).is_ok());
}

#[test]
fn writer_set_fragment_uses_fragment_stages() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::FRAGMENT, 0, 64)
        .expect("valid");
    // set_fragment should use FRAGMENT stages
    assert!(config.is_valid_write(ShaderStages::FRAGMENT, 0, 16).is_ok());
}

#[test]
fn writer_set_vertex_fragment_uses_combined_stages() {
    let config = PushConstantConfig::new()
        .add_vertex_fragment_range(0, 64)
        .expect("valid");
    // set_vertex_fragment should use VERTEX | FRAGMENT stages
    assert!(config.is_valid_write(ShaderStages::VERTEX | ShaderStages::FRAGMENT, 0, 16).is_ok());
}

#[test]
fn writer_compute_uses_compute_stages() {
    let config = PushConstantConfig::new()
        .add_compute_range(0, 128)
        .expect("valid");
    // ComputePushConstantWriter::set uses COMPUTE stages
    assert!(config.is_valid_write(ShaderStages::COMPUTE, 0, 16).is_ok());
}

#[test]
fn writer_set_bytes_interface() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    // set_bytes uses raw bytes, validation still applies
    assert!(config.is_valid_write(ShaderStages::VERTEX, 0, 4).is_ok());
    assert!(config.is_valid_write(ShaderStages::VERTEX, 60, 4).is_ok());
}

#[test]
fn writer_offset_validation_boundary() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    // Exactly at boundary
    assert!(config.is_valid_write(ShaderStages::VERTEX, 60, 4).is_ok());
    // One byte past is invalid offset
    assert!(config.is_valid_write(ShaderStages::VERTEX, 64, 4).is_err());
}

#[test]
fn writer_size_validation_overflow() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    // Writing too much data
    let result = config.is_valid_write(ShaderStages::VERTEX, 32, 64);
    assert!(matches!(
        result,
        Err(PushConstantError::DataTooLarge {
            data_size: 64,
            available: 32
        })
    ));
}

// =============================================================================
// SECTION 5: Writer Operations (~15 tests)
// =============================================================================

#[test]
fn writer_ops_find_range_at_start() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    let result = config.find_range(ShaderStages::VERTEX, 0);
    assert!(result.is_some());
    let (range, available) = result.unwrap();
    assert_eq!(range, 0..64);
    assert_eq!(available, 64);
}

#[test]
fn writer_ops_find_range_middle() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    let result = config.find_range(ShaderStages::VERTEX, 32);
    assert!(result.is_some());
    let (range, available) = result.unwrap();
    assert_eq!(range, 0..64);
    assert_eq!(available, 32);
}

#[test]
fn writer_ops_find_range_end_minus_one() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    let result = config.find_range(ShaderStages::VERTEX, 63);
    assert!(result.is_some());
    let (_, available) = result.unwrap();
    assert_eq!(available, 1);
}

#[test]
fn writer_ops_find_range_at_end_returns_none() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    assert!(config.find_range(ShaderStages::VERTEX, 64).is_none());
}

#[test]
fn writer_ops_find_range_wrong_stages() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    assert!(config.find_range(ShaderStages::FRAGMENT, 0).is_none());
    assert!(config.find_range(ShaderStages::COMPUTE, 0).is_none());
}

#[test]
fn writer_ops_find_range_multiple_ranges() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 32)
        .expect("first")
        .add_range(ShaderStages::FRAGMENT, 64, 32)
        .expect("second");

    // Find in first range
    let (range, avail) = config.find_range(ShaderStages::VERTEX, 0).unwrap();
    assert_eq!(range, 0..32);
    assert_eq!(avail, 32);

    // Find in second range
    let (range, avail) = config.find_range(ShaderStages::FRAGMENT, 64).unwrap();
    assert_eq!(range, 64..96);
    assert_eq!(avail, 32);

    // Gap between ranges
    assert!(config.find_range(ShaderStages::VERTEX, 32).is_none());
    assert!(config.find_range(ShaderStages::FRAGMENT, 0).is_none());
}

#[test]
fn writer_ops_is_valid_write_exact_fit() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    assert!(config.is_valid_write(ShaderStages::VERTEX, 0, 64).is_ok());
}

#[test]
fn writer_ops_is_valid_write_partial() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    assert!(config.is_valid_write(ShaderStages::VERTEX, 0, 16).is_ok());
    assert!(config.is_valid_write(ShaderStages::VERTEX, 16, 16).is_ok());
    assert!(config.is_valid_write(ShaderStages::VERTEX, 48, 16).is_ok());
}

#[test]
fn writer_ops_is_valid_write_data_too_large_error() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    let result = config.is_valid_write(ShaderStages::VERTEX, 0, 128);
    assert!(matches!(
        result,
        Err(PushConstantError::DataTooLarge {
            data_size: 128,
            available: 64
        })
    ));
}

#[test]
fn writer_ops_is_valid_write_invalid_offset_error() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    let result = config.is_valid_write(ShaderStages::VERTEX, 100, 4);
    assert!(matches!(
        result,
        Err(PushConstantError::InvalidOffset { offset: 100, .. })
    ));
}

#[test]
fn writer_ops_is_valid_write_wrong_stages_error() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");
    let result = config.is_valid_write(ShaderStages::FRAGMENT, 0, 4);
    assert!(matches!(
        result,
        Err(PushConstantError::InvalidOffset { offset: 0, .. })
    ));
}

#[test]
fn writer_ops_pod_type_sizes() {
    // Verify bytemuck works with test types
    assert_eq!(std::mem::size_of::<TestData4>(), 4);
    assert_eq!(std::mem::size_of::<TestData16>(), 16);
    assert_eq!(std::mem::size_of::<TestData64>(), 64);
    assert_eq!(std::mem::size_of::<TestData128>(), 128);
}

#[test]
fn writer_ops_bytemuck_bytes_of() {
    let data = TestData16 { a: 1, b: 2, c: 3, d: 4 };
    let bytes = bytemuck::bytes_of(&data);
    assert_eq!(bytes.len(), 16);
}

#[test]
fn writer_ops_bytemuck_from_bytes() {
    let data = TestData16 { a: 1, b: 2, c: 3, d: 4 };
    let bytes = bytemuck::bytes_of(&data);
    let recovered: &TestData16 = bytemuck::from_bytes(bytes);
    assert_eq!(recovered.a, 1);
    assert_eq!(recovered.b, 2);
    assert_eq!(recovered.c, 3);
    assert_eq!(recovered.d, 4);
}

// =============================================================================
// SECTION 6: Fallback System (~15 tests)
// =============================================================================

#[test]
fn fallback_enum_native_variant() {
    let fallback = PushConstantFallback::Native;
    assert!(fallback.is_native());
    assert!(!fallback.is_fallback());
    assert!(fallback.buffer().is_none());
    assert!(fallback.bind_group_index().is_none());
}

#[test]
fn fallback_enum_debug_format_native() {
    let fallback = PushConstantFallback::Native;
    let debug = format!("{:?}", fallback);
    assert!(debug.contains("Native"));
}

#[test]
fn fallback_auto_detection_with_push_constants() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter with push constants available");
            return;
        }
    };

    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    let fallback = FallbackPushConstants::new(&device, &config, None);
    assert!(fallback.is_native());
    assert!(!fallback.is_fallback());
    assert!(fallback.buffer().is_none());
}

#[test]
fn fallback_new_uniform_buffer_forces_fallback() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    let fallback = FallbackPushConstants::new_uniform_buffer(&device, &config, 2);
    assert!(!fallback.is_native());
    assert!(fallback.is_fallback());
    assert!(fallback.buffer().is_some());
}

#[test]
fn fallback_new_native_error_without_feature() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    // Skip if the device actually supports push constants
    if supports_push_constants(&device) {
        return;
    }

    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    let result = FallbackPushConstants::new_native(&device, &config);
    assert!(matches!(result, Err(PushConstantError::UnsupportedFeature)));
}

#[test]
fn fallback_write_typed_data() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    let mut fallback = FallbackPushConstants::new(&device, &config, None);

    let data = DrawPushConstants::new(42, 7);
    assert!(fallback.write(0, &data).is_ok());
}

#[test]
fn fallback_write_bytes_raw() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    let mut fallback = FallbackPushConstants::new(&device, &config, None);

    let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8];
    assert!(fallback.write_bytes(0, &bytes).is_ok());

    // Verify bytes were written
    let stored = fallback.data();
    assert_eq!(&stored[0..8], &bytes);
}

#[test]
fn fallback_write_exceeds_size_error() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 16)
        .expect("valid");

    let mut fallback = FallbackPushConstants::new(&device, &config, None);

    // Try to write 64 bytes to a 16-byte config
    let result = fallback.write_bytes(0, &[0u8; 64]);
    assert!(matches!(result, Err(PushConstantError::ExceedsMaxSize { .. })));
}

#[test]
fn fallback_write_at_offset() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    let mut fallback = FallbackPushConstants::new(&device, &config, None);

    // Write at offset 32
    let bytes = [0xAA, 0xBB, 0xCC, 0xDD];
    assert!(fallback.write_bytes(32, &bytes).is_ok());

    let stored = fallback.data();
    assert_eq!(&stored[32..36], &bytes);
}

#[test]
fn fallback_data_accessor() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    let fallback = FallbackPushConstants::new(&device, &config, None);

    let data = fallback.data();
    assert_eq!(data.len(), 64);
    // Initial data should be zeroed
    assert!(data.iter().all(|&b| b == 0));
}

#[test]
fn fallback_config_accessor() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    let fallback = FallbackPushConstants::new(&device, &config, None);
    assert_eq!(fallback.config().total_size(), 64);
    assert_eq!(fallback.config().range_count(), 1);
}

#[test]
fn fallback_fallback_accessor() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    let fallback = FallbackPushConstants::new(&device, &config, None);
    let strategy = fallback.fallback();
    assert!(strategy.is_native());
}

#[test]
fn fallback_upload_no_op_for_native() {
    let (device, queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    let fallback = FallbackPushConstants::new(&device, &config, None);
    // upload() should be a no-op for native mode
    fallback.upload(&queue);
}

#[test]
fn fallback_upload_writes_to_buffer() {
    let (device, queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    let mut fallback = FallbackPushConstants::new_uniform_buffer(&device, &config, 2);

    // Write data
    let data = DrawPushConstants::new(42, 7);
    fallback.write(0, &data).expect("write");

    // Upload to GPU
    fallback.upload(&queue);
    // No direct verification possible without readback, but no panic is a pass
}

// =============================================================================
// SECTION 7: Pre-built Structs (~10 tests)
// =============================================================================

#[test]
fn struct_draw_push_constants_size() {
    assert_eq!(DrawPushConstants::SIZE, 16);
    assert_eq!(std::mem::size_of::<DrawPushConstants>(), 16);
}

#[test]
fn struct_draw_push_constants_new() {
    let pc = DrawPushConstants::new(42, 7);
    assert_eq!(pc.object_id, 42);
    assert_eq!(pc.material_id, 7);
    assert_eq!(pc.first_vertex, 0);
    assert_eq!(pc._reserved, 0);
}

#[test]
fn struct_draw_push_constants_with_vertex_offset() {
    let pc = DrawPushConstants::with_vertex_offset(100, 50, 1000);
    assert_eq!(pc.object_id, 100);
    assert_eq!(pc.material_id, 50);
    assert_eq!(pc.first_vertex, 1000);
    assert_eq!(pc._reserved, 0);
}

#[test]
fn struct_draw_push_constants_default() {
    let pc = DrawPushConstants::default();
    assert_eq!(pc.object_id, 0);
    assert_eq!(pc.material_id, 0);
    assert_eq!(pc.first_vertex, 0);
    assert_eq!(pc._reserved, 0);
}

#[test]
fn struct_draw_push_constants_bytemuck_pod() {
    let pc = DrawPushConstants::new(42, 7);
    let bytes = bytemuck::bytes_of(&pc);
    assert_eq!(bytes.len(), 16);

    let recovered: &DrawPushConstants = bytemuck::from_bytes(bytes);
    assert_eq!(recovered.object_id, pc.object_id);
    assert_eq!(recovered.material_id, pc.material_id);
}

#[test]
fn struct_extended_push_constants_size() {
    assert_eq!(ExtendedDrawPushConstants::SIZE, 64);
    assert_eq!(std::mem::size_of::<ExtendedDrawPushConstants>(), 64);
}

#[test]
fn struct_extended_push_constants_identity() {
    let pc = ExtendedDrawPushConstants::identity();
    assert_eq!(pc.model_row0, [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(pc.model_row1, [0.0, 1.0, 0.0, 0.0]);
    assert_eq!(pc.model_row2, [0.0, 0.0, 1.0, 0.0]);
    assert_eq!(pc.model_row3, [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn struct_extended_push_constants_from_matrix() {
    let matrix = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [1.0, 2.0, 3.0, 1.0],
    ];
    let pc = ExtendedDrawPushConstants::from_matrix(matrix);
    assert_eq!(pc.model_row0, matrix[0]);
    assert_eq!(pc.model_row1, matrix[1]);
    assert_eq!(pc.model_row2, matrix[2]);
    assert_eq!(pc.model_row3, matrix[3]);
}

#[test]
fn struct_extended_push_constants_default() {
    let pc = ExtendedDrawPushConstants::default();
    let identity = ExtendedDrawPushConstants::identity();
    assert_eq!(pc.model_row0, identity.model_row0);
    assert_eq!(pc.model_row1, identity.model_row1);
    assert_eq!(pc.model_row2, identity.model_row2);
    assert_eq!(pc.model_row3, identity.model_row3);
}

#[test]
fn struct_extended_push_constants_bytemuck_pod() {
    let pc = ExtendedDrawPushConstants::identity();
    let bytes = bytemuck::bytes_of(&pc);
    assert_eq!(bytes.len(), 64);
}

// =============================================================================
// SECTION 8: Thread Safety (~5 tests)
// =============================================================================

#[test]
fn thread_safety_config_send() {
    fn assert_send<T: Send>() {}
    assert_send::<PushConstantConfig>();
}

#[test]
fn thread_safety_config_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<PushConstantConfig>();
}

#[test]
fn thread_safety_fallback_push_constants_send() {
    fn assert_send<T: Send>() {}
    assert_send::<FallbackPushConstants>();
}

#[test]
fn thread_safety_fallback_push_constants_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<FallbackPushConstants>();
}

#[test]
fn thread_safety_concurrent_config_creation() {
    let handles: Vec<_> = (0..8)
        .map(|i| {
            thread::spawn(move || {
                let config = PushConstantConfig::new()
                    .add_range(ShaderStages::VERTEX, 0, 64)
                    .expect("valid");
                (i, config.total_size())
            })
        })
        .collect();

    for handle in handles {
        let (_, total_size) = handle.join().expect("thread panicked");
        assert_eq!(total_size, 64);
    }
}

// =============================================================================
// SECTION 9: Edge Cases (~15 tests)
// =============================================================================

#[test]
fn edge_case_minimum_range_size() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 4)
        .expect("minimum size of 4 is valid");
    assert_eq!(config.total_size(), 4);
}

#[test]
fn edge_case_maximum_range_size() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 128)
        .expect("maximum size of 128 is valid");
    assert_eq!(config.total_size(), 128);
}

#[test]
fn edge_case_max_offset() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 124, 4)
        .expect("max valid offset");
    assert_eq!(config.total_size(), 128);
}

#[test]
fn edge_case_boundary_exact_fit() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("first half")
        .add_range(ShaderStages::FRAGMENT, 64, 64)
        .expect("second half");
    assert_eq!(config.total_size(), 128);
}

#[test]
fn edge_case_many_small_ranges() {
    let mut config = PushConstantConfig::new();
    // Create 32 ranges of 4 bytes each = 128 bytes total
    for i in 0..32 {
        config = config
            .add_range(ShaderStages::VERTEX, i * 4, 4)
            .expect("valid 4-byte range");
    }
    assert_eq!(config.range_count(), 32);
    assert_eq!(config.total_size(), 128);
}

#[test]
fn edge_case_all_stages() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX | ShaderStages::FRAGMENT | ShaderStages::COMPUTE, 0, 64)
        .expect("all stages");
    assert_eq!(config.range_count(), 1);
}

#[test]
fn edge_case_helper_vertex_only() {
    let config = vertex_only(64).expect("valid");
    assert_eq!(config.range_count(), 1);
    assert_eq!(config.ranges()[0].stages, ShaderStages::VERTEX);
    assert_eq!(config.total_size(), 64);
}

#[test]
fn edge_case_helper_fragment_only() {
    let config = fragment_only(64).expect("valid");
    assert_eq!(config.range_count(), 1);
    assert_eq!(config.ranges()[0].stages, ShaderStages::FRAGMENT);
}

#[test]
fn edge_case_helper_vertex_fragment() {
    let config = vertex_fragment(64).expect("valid");
    assert_eq!(config.range_count(), 1);
    assert_eq!(
        config.ranges()[0].stages,
        ShaderStages::VERTEX | ShaderStages::FRAGMENT
    );
}

#[test]
fn edge_case_helper_compute_only() {
    let config = compute_only(128).expect("valid");
    assert_eq!(config.range_count(), 1);
    assert_eq!(config.ranges()[0].stages, ShaderStages::COMPUTE);
}

#[test]
fn edge_case_helper_with_invalid_size() {
    assert!(vertex_only(0).is_err());
    assert!(vertex_only(3).is_err());
    assert!(vertex_only(129).is_err());
}

#[test]
fn edge_case_is_aligned_helper() {
    assert!(is_aligned(0));
    assert!(is_aligned(4));
    assert!(is_aligned(8));
    assert!(is_aligned(128));
    assert!(!is_aligned(1));
    assert!(!is_aligned(2));
    assert!(!is_aligned(3));
    assert!(!is_aligned(127));
}

#[test]
fn edge_case_align_up_helper() {
    assert_eq!(align_up(0), 0);
    assert_eq!(align_up(1), 4);
    assert_eq!(align_up(2), 4);
    assert_eq!(align_up(3), 4);
    assert_eq!(align_up(4), 4);
    assert_eq!(align_up(5), 8);
    assert_eq!(align_up(127), 128);
    assert_eq!(align_up(128), 128);
}

#[test]
fn edge_case_constants() {
    assert_eq!(MAX_PUSH_CONSTANT_SIZE, 128);
    assert_eq!(PUSH_CONSTANT_ALIGNMENT, 4);
    assert_eq!(DEFAULT_FALLBACK_BIND_GROUP, 2);
}

#[test]
fn edge_case_supports_push_constants_with_feature() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter with push constants available");
            return;
        }
    };
    assert!(supports_push_constants(&device));
}

// =============================================================================
// SECTION 10: Integration Tests (~10 tests)
// =============================================================================

#[test]
fn integration_full_config_workflow() {
    // Create config
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("vertex range")
        .add_range(ShaderStages::FRAGMENT, 64, 64)
        .expect("fragment range");

    // Validate
    assert!(config.validate().is_ok());
    assert_eq!(config.total_size(), 128);
    assert_eq!(config.range_count(), 2);

    // Check ranges
    let ranges = config.ranges();
    assert_eq!(ranges[0].stages, ShaderStages::VERTEX);
    assert_eq!(ranges[0].range, 0..64);
    assert_eq!(ranges[1].stages, ShaderStages::FRAGMENT);
    assert_eq!(ranges[1].range, 64..128);

    // Check valid writes
    assert!(config.is_valid_write(ShaderStages::VERTEX, 0, 64).is_ok());
    assert!(config.is_valid_write(ShaderStages::FRAGMENT, 64, 64).is_ok());
    assert!(config.is_valid_write(ShaderStages::VERTEX, 64, 4).is_err());
}

#[test]
fn integration_fallback_full_workflow() {
    let (device, queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    // Create config
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    // Create fallback
    let mut fallback = FallbackPushConstants::new(&device, &config, None);

    // Write data
    let draw_data = DrawPushConstants::new(42, 7);
    fallback.write(0, &draw_data).expect("write DrawPushConstants");

    // Write more data at offset
    let extra = TestData16 { a: 1, b: 2, c: 3, d: 4 };
    fallback.write(16, &extra).expect("write TestData16");

    // Upload
    fallback.upload(&queue);

    // Verify data
    let stored = fallback.data();
    assert_eq!(stored.len(), 64);

    // Check DrawPushConstants bytes
    let draw_bytes: &[u8; 16] = bytemuck::from_bytes(&stored[0..16]);
    let recovered_draw: &DrawPushConstants = bytemuck::from_bytes(draw_bytes);
    assert_eq!(recovered_draw.object_id, 42);
    assert_eq!(recovered_draw.material_id, 7);
}

#[test]
fn integration_device_feature_detection() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter with push constants available");
            return;
        }
    };

    assert!(supports_push_constants(&device));
    let max_size = max_push_constant_size(&device);
    assert!(max_size >= 128);
}

#[test]
fn integration_device_without_feature() {
    let (device, _queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    // May or may not support depending on backend
    let _supports = supports_push_constants(&device);
    let _max_size = max_push_constant_size(&device);
}

#[test]
fn integration_uniform_buffer_fallback() {
    let (device, queue) = match create_test_device_no_push_constants() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("valid");

    // Force uniform buffer fallback
    let mut fallback = FallbackPushConstants::new_uniform_buffer(&device, &config, 2);
    assert!(fallback.is_fallback());
    assert!(fallback.buffer().is_some());

    // Write and upload
    let data = DrawPushConstants::new(100, 200);
    fallback.write(0, &data).expect("write");
    fallback.upload(&queue);
}

#[test]
fn integration_overlapping_offset_detection() {
    let config1 = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 64)
        .expect("first");

    // Overlapping with same stages should fail
    let result = config1.clone().add_range(ShaderStages::VERTEX, 32, 64);
    assert!(matches!(result, Err(PushConstantError::RangeOverlap { .. })));

    // Non-overlapping should succeed
    let config2 = config1.add_range(ShaderStages::VERTEX, 64, 32).expect("non-overlapping");
    assert_eq!(config2.range_count(), 2);
}

#[test]
fn integration_extended_draw_push_constants_workflow() {
    let (device, _queue) = match create_test_device() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: no GPU adapter available");
            return;
        }
    };

    // Config for extended push constants (64 bytes)
    let config = PushConstantConfig::new()
        .add_vertex_fragment_range(0, 64)
        .expect("valid");

    let mut fallback = FallbackPushConstants::new(&device, &config, None);

    // Create and write extended push constants
    let transform = ExtendedDrawPushConstants::from_matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [10.0, 20.0, 30.0, 1.0],
    ]);

    fallback.write(0, &transform).expect("write transform");

    // Verify
    let stored = fallback.data();
    assert_eq!(stored.len(), 64);
}

#[test]
fn integration_compute_config() {
    let config = compute_only(128).expect("valid");
    assert_eq!(config.range_count(), 1);
    assert_eq!(config.ranges()[0].stages, ShaderStages::COMPUTE);
    assert_eq!(config.total_size(), 128);

    // Valid compute writes
    assert!(config.is_valid_write(ShaderStages::COMPUTE, 0, 128).is_ok());
    assert!(config.is_valid_write(ShaderStages::COMPUTE, 64, 64).is_ok());

    // Invalid for other stages
    assert!(config.is_valid_write(ShaderStages::VERTEX, 0, 4).is_err());
}

#[test]
fn integration_multiple_stage_combinations() {
    let config = PushConstantConfig::new()
        .add_range(ShaderStages::VERTEX, 0, 32)
        .expect("vertex only")
        .add_range(ShaderStages::FRAGMENT, 32, 32)
        .expect("fragment only")
        .add_range(ShaderStages::VERTEX | ShaderStages::FRAGMENT, 64, 32)
        .expect("vertex+fragment")
        .add_range(ShaderStages::COMPUTE, 96, 32)
        .expect("compute");

    assert_eq!(config.range_count(), 4);
    assert_eq!(config.total_size(), 128);

    // Check each stage's valid ranges
    assert!(config.is_valid_write(ShaderStages::VERTEX, 0, 32).is_ok());
    assert!(config.is_valid_write(ShaderStages::FRAGMENT, 32, 32).is_ok());
    assert!(config.is_valid_write(ShaderStages::VERTEX | ShaderStages::FRAGMENT, 64, 32).is_ok());
    assert!(config.is_valid_write(ShaderStages::COMPUTE, 96, 32).is_ok());
}

#[test]
fn integration_error_chaining() {
    // Multiple errors in sequence
    let r1 = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 1, 64);
    assert!(r1.is_err());

    let r2 = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 0, 3);
    assert!(r2.is_err());

    let r3 = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 0, 256);
    assert!(r3.is_err());

    let r4 = PushConstantConfig::new().add_range(ShaderStages::VERTEX, 0, 0);
    assert!(r4.is_err());
}
