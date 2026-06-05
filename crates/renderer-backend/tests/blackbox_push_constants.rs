//! Blackbox tests for push constants (T-WGPU-P2.5.4)
//!
//! CLEANROOM: Tests ONLY the public API as exported from `renderer_backend::resources`.
//! Does NOT read implementation details from push_constants.rs.
//!
//! Public API under test:
//! - Constants: MAX_PUSH_CONSTANT_SIZE (from pipeline_layout), PUSH_CONSTANT_ALIGNMENT, DEFAULT_FALLBACK_BIND_GROUP
//! - Types: DrawPushConstants, ExtendedDrawPushConstants, PushConstantConfig, PushConstantError, PushConstantFallback, FallbackPushConstants
//! - Writers: PushConstantWriter, ComputePushConstantWriter
//! - Functions: supports_push_constants, max_push_constant_size, vertex_only, fragment_only, compute_only, vertex_fragment
//! - Helpers: align_push_constant (alias for align_up), is_push_constant_aligned (alias for is_aligned)

use renderer_backend::resources::{
    // Constants
    MAX_PUSH_CONSTANT_SIZE,
    PUSH_CONSTANT_ALIGNMENT,
    DEFAULT_FALLBACK_BIND_GROUP,
    // Types
    DrawPushConstants,
    ExtendedDrawPushConstants,
    PushConstantConfig,
    PushConstantError,
    PushConstantFallback,
    FallbackPushConstants,
    // Writers
    PushConstantWriter,
    ComputePushConstantWriter,
    // Functions
    supports_push_constants,
    max_push_constant_size,
    vertex_only,
    fragment_only,
    compute_only,
    vertex_fragment,
    align_push_constant,
    is_push_constant_aligned,
};
use std::mem::size_of;

// =============================================================================
// CATEGORY 1: Constants (~5 tests)
// =============================================================================

mod constants {
    use super::*;

    #[test]
    fn max_push_constant_size_is_128_bytes() {
        // wgpu requires push constants to be at most 128 bytes
        assert_eq!(MAX_PUSH_CONSTANT_SIZE, 128);
    }

    #[test]
    fn push_constant_alignment_is_4_bytes() {
        // Push constants must be 4-byte aligned per wgpu spec
        assert_eq!(PUSH_CONSTANT_ALIGNMENT, 4);
    }

    #[test]
    fn default_fallback_bind_group_is_2() {
        // Default bind group index for uniform buffer fallback (OBJECT = 2)
        assert_eq!(DEFAULT_FALLBACK_BIND_GROUP, 2);
    }

    #[test]
    fn max_push_constant_size_is_power_of_two() {
        assert!(MAX_PUSH_CONSTANT_SIZE.is_power_of_two());
    }

    #[test]
    fn alignment_divides_max_size() {
        // Alignment should evenly divide max size
        assert_eq!(MAX_PUSH_CONSTANT_SIZE % PUSH_CONSTANT_ALIGNMENT, 0);
    }
}

// =============================================================================
// CATEGORY 2: PushConstantConfig API (~20 tests)
// =============================================================================

mod push_constant_config {
    use super::*;

    #[test]
    fn new_creates_empty_config() {
        let config = PushConstantConfig::new();
        assert!(config.is_empty());
    }

    #[test]
    fn total_size_of_empty_config_is_zero() {
        let config = PushConstantConfig::new();
        assert_eq!(config.total_size(), 0);
    }

    #[test]
    fn add_range_returns_result() {
        let config = PushConstantConfig::new();
        let result = config.add_range(wgpu::ShaderStages::VERTEX, 0, 16);
        // Valid range should succeed
        assert!(result.is_ok());
    }

    #[test]
    fn add_range_with_valid_offset_and_size() {
        let config = PushConstantConfig::new();
        // 0 offset, 64 bytes, 4-byte aligned - should work
        assert!(config.add_range(wgpu::ShaderStages::VERTEX, 0, 64).is_ok());
    }

    #[test]
    fn add_range_with_misaligned_offset_returns_error() {
        let config = PushConstantConfig::new();
        // Offset 1 is not 4-byte aligned
        let result = config.add_range(wgpu::ShaderStages::VERTEX, 1, 16);
        assert!(result.is_err());
    }

    #[test]
    fn add_range_with_offset_3_returns_error() {
        let config = PushConstantConfig::new();
        // Offset 3 is not 4-byte aligned
        let result = config.add_range(wgpu::ShaderStages::VERTEX, 3, 16);
        assert!(result.is_err());
    }

    #[test]
    fn add_range_with_misaligned_size_returns_error() {
        let config = PushConstantConfig::new();
        // Size 17 is not 4-byte aligned
        let result = config.add_range(wgpu::ShaderStages::VERTEX, 0, 17);
        assert!(result.is_err());
    }

    #[test]
    fn add_range_with_size_1_returns_error() {
        let config = PushConstantConfig::new();
        // Size 1 is not 4-byte aligned
        let result = config.add_range(wgpu::ShaderStages::VERTEX, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn add_range_exceeding_max_size_returns_error() {
        let config = PushConstantConfig::new();
        // 256 bytes exceeds the 128-byte limit
        let result = config.add_range(wgpu::ShaderStages::VERTEX, 0, 256);
        assert!(result.is_err());
    }

    #[test]
    fn add_range_at_max_size_succeeds() {
        let config = PushConstantConfig::new();
        // 128 bytes is the maximum allowed
        let result = config.add_range(wgpu::ShaderStages::VERTEX, 0, 128);
        assert!(result.is_ok());
    }

    #[test]
    fn add_range_just_over_max_fails() {
        let config = PushConstantConfig::new();
        // 132 bytes (128 + 4) exceeds max
        let result = config.add_range(wgpu::ShaderStages::VERTEX, 0, 132);
        assert!(result.is_err());
    }

    #[test]
    fn ranges_returns_slice_of_added_ranges() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 16).unwrap()
            .add_range(wgpu::ShaderStages::FRAGMENT, 16, 16).unwrap();

        let ranges = config.ranges();
        assert_eq!(ranges.len(), 2);
    }

    #[test]
    fn total_size_returns_sum_of_ranges() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 16).unwrap()
            .add_range(wgpu::ShaderStages::FRAGMENT, 16, 32).unwrap();

        // Total should be at least 48 (16 + 32), but may be computed differently
        assert!(config.total_size() >= 48);
    }

    #[test]
    fn builder_pattern_chaining_works() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 16).unwrap()
            .add_range(wgpu::ShaderStages::FRAGMENT, 16, 16).unwrap();

        assert_eq!(config.ranges().len(), 2);
    }

    #[test]
    fn add_multiple_vertex_ranges_works() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 32).unwrap()
            .add_range(wgpu::ShaderStages::VERTEX, 32, 32).unwrap();

        assert_eq!(config.range_count(), 2);
    }

    #[test]
    fn add_vertex_and_fragment_ranges_works() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 32).unwrap()
            .add_range(wgpu::ShaderStages::FRAGMENT, 32, 32).unwrap();

        assert_eq!(config.range_count(), 2);
    }

    #[test]
    fn add_compute_range_works() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::COMPUTE, 0, 64).unwrap();

        assert_eq!(config.range_count(), 1);
    }

    #[test]
    fn add_vertex_fragment_combined_range_works() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, 64).unwrap();

        assert_eq!(config.range_count(), 1);
    }

    #[test]
    fn zero_size_range_is_invalid() {
        let config = PushConstantConfig::new();
        let result = config.add_range(wgpu::ShaderStages::VERTEX, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn config_with_max_ranges_at_boundary() {
        // Add 128 bytes in one range
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 128).unwrap();

        assert_eq!(config.total_size(), 128);
    }

    #[test]
    fn range_count_method_works() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 16).unwrap()
            .add_range(wgpu::ShaderStages::FRAGMENT, 16, 16).unwrap()
            .add_range(wgpu::ShaderStages::COMPUTE, 32, 16).unwrap();

        assert_eq!(config.range_count(), 3);
    }

    #[test]
    fn is_empty_returns_true_for_new_config() {
        let config = PushConstantConfig::new();
        assert!(config.is_empty());
    }

    #[test]
    fn is_empty_returns_false_after_adding_range() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 16).unwrap();

        assert!(!config.is_empty());
    }
}

// =============================================================================
// CATEGORY 3: PushConstantError API (~10 tests)
// =============================================================================

mod push_constant_error {
    use super::*;

    #[test]
    fn error_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<PushConstantError>();
    }

    #[test]
    fn error_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<PushConstantError>();
    }

    #[test]
    fn error_implements_debug() {
        let config = PushConstantConfig::new();
        let err = config.add_range(wgpu::ShaderStages::VERTEX, 1, 16).unwrap_err();
        let debug_str = format!("{:?}", err);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn error_implements_display() {
        let config = PushConstantConfig::new();
        let err = config.add_range(wgpu::ShaderStages::VERTEX, 1, 16).unwrap_err();
        let display_str = format!("{}", err);
        assert!(!display_str.is_empty());
    }

    #[test]
    fn misaligned_offset_error_message_contains_alignment_info() {
        let config = PushConstantConfig::new();
        let err = config.add_range(wgpu::ShaderStages::VERTEX, 5, 16).unwrap_err();
        let msg = format!("{}", err);
        // Error message should mention alignment or offset
        assert!(msg.to_lowercase().contains("align") || msg.contains("5") || msg.to_lowercase().contains("offset"));
    }

    #[test]
    fn misaligned_size_error_message_contains_size_info() {
        let config = PushConstantConfig::new();
        let err = config.add_range(wgpu::ShaderStages::VERTEX, 0, 7).unwrap_err();
        let msg = format!("{}", err);
        // Error message should mention size or alignment
        assert!(msg.to_lowercase().contains("align") || msg.contains("7") || msg.to_lowercase().contains("size"));
    }

    #[test]
    fn exceeds_max_error_message_mentions_limit() {
        let config = PushConstantConfig::new();
        let err = config.add_range(wgpu::ShaderStages::VERTEX, 0, 256).unwrap_err();
        let msg = format!("{}", err);
        // Error message should mention the limit or max
        assert!(msg.to_lowercase().contains("max") || msg.to_lowercase().contains("limit") || msg.to_lowercase().contains("exceed") || msg.contains("128"));
    }

    #[test]
    fn different_errors_are_distinct() {
        let config1 = PushConstantConfig::new();
        let err1 = config1.add_range(wgpu::ShaderStages::VERTEX, 1, 16).unwrap_err(); // Misaligned offset

        let config2 = PushConstantConfig::new();
        let err2 = config2.add_range(wgpu::ShaderStages::VERTEX, 0, 256).unwrap_err(); // Exceeds max

        // Different error types should produce different messages
        let msg1 = format!("{:?}", err1);
        let msg2 = format!("{:?}", err2);
        assert_ne!(msg1, msg2);
    }

    #[test]
    fn error_can_be_converted_to_string() {
        let config = PushConstantConfig::new();
        let err = config.add_range(wgpu::ShaderStages::VERTEX, 1, 16).unwrap_err();
        let _msg = err.to_string();
    }

    #[test]
    fn zero_size_error_is_distinguishable() {
        let config = PushConstantConfig::new();
        let err = config.add_range(wgpu::ShaderStages::VERTEX, 0, 0).unwrap_err();
        let msg = format!("{}", err);
        // Should indicate something is wrong with size
        assert!(msg.to_lowercase().contains("size") || msg.to_lowercase().contains("zero") || msg.to_lowercase().contains("invalid"));
    }
}

// =============================================================================
// CATEGORY 4: DrawPushConstants API (~10 tests)
// =============================================================================

mod draw_push_constants {
    use super::*;
    use bytemuck::{Pod, Zeroable};

    #[test]
    fn size_is_exactly_16_bytes() {
        assert_eq!(size_of::<DrawPushConstants>(), 16);
    }

    #[test]
    fn implements_default() {
        let pc = DrawPushConstants::default();
        let _size = size_of_val(&pc);
    }

    #[test]
    fn implements_clone() {
        let pc = DrawPushConstants::default();
        let pc2 = pc.clone();
        let _size = size_of_val(&pc2);
    }

    #[test]
    fn implements_copy() {
        let pc = DrawPushConstants::default();
        let pc2 = pc;
        let _pc3 = pc; // Can use pc again because it's Copy
        let _size = size_of_val(&pc2);
    }

    #[test]
    fn is_pod_castable_to_bytes() {
        let pc = DrawPushConstants::default();
        let bytes: &[u8] = bytemuck::bytes_of(&pc);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn is_zeroable() {
        let pc = DrawPushConstants::zeroed();
        let bytes: &[u8] = bytemuck::bytes_of(&pc);
        // All bytes should be zero
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn can_cast_from_bytes() {
        let bytes = [0u8; 16];
        let pc: &DrawPushConstants = bytemuck::from_bytes(&bytes);
        let _size = size_of_val(pc);
    }

    #[test]
    fn size_fits_in_push_constant_limit() {
        assert!(size_of::<DrawPushConstants>() <= MAX_PUSH_CONSTANT_SIZE as usize);
    }

    #[test]
    fn size_is_4_byte_aligned() {
        assert_eq!(size_of::<DrawPushConstants>() % 4, 0);
    }

    #[test]
    fn alignment_is_at_least_4() {
        assert!(std::mem::align_of::<DrawPushConstants>() >= 4);
    }

    #[test]
    fn size_constant_matches_sizeof() {
        assert_eq!(DrawPushConstants::SIZE as usize, size_of::<DrawPushConstants>());
    }

    #[test]
    fn new_constructor_creates_instance() {
        let pc = DrawPushConstants::new(1, 2);
        let _size = size_of_val(&pc);
    }

    #[test]
    fn with_vertex_offset_constructor_works() {
        let pc = DrawPushConstants::with_vertex_offset(1, 2, 100);
        let _size = size_of_val(&pc);
    }
}

// =============================================================================
// CATEGORY 5: ExtendedDrawPushConstants API (~10 tests)
// =============================================================================

mod extended_draw_push_constants {
    use super::*;
    use bytemuck::{Pod, Zeroable};

    #[test]
    fn size_is_exactly_64_bytes() {
        assert_eq!(size_of::<ExtendedDrawPushConstants>(), 64);
    }

    #[test]
    fn implements_default() {
        let pc = ExtendedDrawPushConstants::default();
        let _size = size_of_val(&pc);
    }

    #[test]
    fn implements_clone() {
        let pc = ExtendedDrawPushConstants::default();
        let pc2 = pc.clone();
        let _size = size_of_val(&pc2);
    }

    #[test]
    fn implements_copy() {
        let pc = ExtendedDrawPushConstants::default();
        let pc2 = pc;
        let _pc3 = pc; // Can use pc again because it's Copy
        let _size = size_of_val(&pc2);
    }

    #[test]
    fn is_pod_castable_to_bytes() {
        let pc = ExtendedDrawPushConstants::default();
        let bytes: &[u8] = bytemuck::bytes_of(&pc);
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn is_zeroable() {
        let pc = ExtendedDrawPushConstants::zeroed();
        let bytes: &[u8] = bytemuck::bytes_of(&pc);
        // All bytes should be zero
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn can_cast_from_bytes() {
        let bytes = [0u8; 64];
        let pc: &ExtendedDrawPushConstants = bytemuck::from_bytes(&bytes);
        let _size = size_of_val(pc);
    }

    #[test]
    fn size_fits_in_push_constant_limit() {
        assert!(size_of::<ExtendedDrawPushConstants>() <= MAX_PUSH_CONSTANT_SIZE as usize);
    }

    #[test]
    fn size_is_4_byte_aligned() {
        assert_eq!(size_of::<ExtendedDrawPushConstants>() % 4, 0);
    }

    #[test]
    fn alignment_is_at_least_4() {
        assert!(std::mem::align_of::<ExtendedDrawPushConstants>() >= 4);
    }

    #[test]
    fn size_constant_matches_sizeof() {
        assert_eq!(ExtendedDrawPushConstants::SIZE as usize, size_of::<ExtendedDrawPushConstants>());
    }

    #[test]
    fn identity_constructor_creates_instance() {
        let pc = ExtendedDrawPushConstants::identity();
        let _size = size_of_val(&pc);
    }

    #[test]
    fn from_matrix_constructor_works() {
        let matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let pc = ExtendedDrawPushConstants::from_matrix(matrix);
        let _size = size_of_val(&pc);
    }
}

// =============================================================================
// CATEGORY 6: PushConstantFallback API (~10 tests)
// =============================================================================

mod push_constant_fallback {
    use super::*;

    #[test]
    fn native_variant_exists() {
        let fallback = PushConstantFallback::Native;
        assert!(fallback.is_native());
    }

    #[test]
    fn native_is_not_fallback() {
        let fallback = PushConstantFallback::Native;
        assert!(!fallback.is_fallback());
    }

    #[test]
    fn native_buffer_returns_none() {
        let fallback = PushConstantFallback::Native;
        assert!(fallback.buffer().is_none());
    }

    #[test]
    fn native_bind_group_index_returns_none() {
        let fallback = PushConstantFallback::Native;
        assert!(fallback.bind_group_index().is_none());
    }

    #[test]
    fn implements_debug() {
        let fallback = PushConstantFallback::Native;
        let debug_str = format!("{:?}", fallback);
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn native_debug_contains_native() {
        let fallback = PushConstantFallback::Native;
        let debug_str = format!("{:?}", fallback);
        assert!(debug_str.contains("Native"));
    }

    #[test]
    fn implements_clone() {
        let fallback = PushConstantFallback::Native;
        let cloned = fallback.clone();
        assert!(cloned.is_native());
    }

    #[test]
    fn is_native_method_works() {
        let native = PushConstantFallback::Native;
        assert!(native.is_native());
    }

    #[test]
    fn is_fallback_method_works_for_native() {
        let native = PushConstantFallback::Native;
        assert!(!native.is_fallback());
    }

    #[test]
    fn type_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<PushConstantFallback>();
    }

    #[test]
    fn type_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<PushConstantFallback>();
    }
}

// =============================================================================
// CATEGORY 7: FallbackPushConstants API (~5 tests)
// =============================================================================

mod fallback_push_constants {
    use super::*;

    #[test]
    fn type_exists() {
        fn assert_type_exists<T>() {}
        assert_type_exists::<FallbackPushConstants>();
    }

    #[test]
    fn is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<FallbackPushConstants>();
    }

    #[test]
    fn is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<FallbackPushConstants>();
    }

    #[test]
    fn size_is_reasonable() {
        // FallbackPushConstants should have a reasonable size (not huge)
        assert!(size_of::<FallbackPushConstants>() <= 1024);
    }

    #[test]
    
    fn new_native_requires_device() {
        // FallbackPushConstants::new_native(&device, &config)
    }

    #[test]
    
    fn new_uniform_buffer_requires_device() {
        // FallbackPushConstants::new_uniform_buffer(...)
    }
}

// =============================================================================
// CATEGORY 8: Helper Functions (~10 tests)
// =============================================================================

mod helper_functions {
    use super::*;

    #[test]
    fn vertex_only_returns_config_result() {
        let result = vertex_only(16);
        assert!(result.is_ok());
    }

    #[test]
    fn vertex_only_config_has_vertex_stage() {
        let config = vertex_only(16).unwrap();
        let ranges = config.ranges();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].stages, wgpu::ShaderStages::VERTEX);
    }

    #[test]
    fn fragment_only_returns_config_result() {
        let result = fragment_only(32);
        assert!(result.is_ok());
    }

    #[test]
    fn fragment_only_config_has_fragment_stage() {
        let config = fragment_only(32).unwrap();
        let ranges = config.ranges();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].stages, wgpu::ShaderStages::FRAGMENT);
    }

    #[test]
    fn compute_only_returns_config_result() {
        let result = compute_only(64);
        assert!(result.is_ok());
    }

    #[test]
    fn compute_only_config_has_compute_stage() {
        let config = compute_only(64).unwrap();
        let ranges = config.ranges();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].stages, wgpu::ShaderStages::COMPUTE);
    }

    #[test]
    fn vertex_fragment_returns_config_result() {
        let result = vertex_fragment(16);
        assert!(result.is_ok());
    }

    #[test]
    fn vertex_fragment_config_has_both_stages() {
        let config = vertex_fragment(16).unwrap();
        let ranges = config.ranges();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].stages, wgpu::ShaderStages::VERTEX_FRAGMENT);
    }

    #[test]
    fn align_push_constant_aligns_to_4() {
        assert_eq!(align_push_constant(0), 0);
        assert_eq!(align_push_constant(1), 4);
        assert_eq!(align_push_constant(3), 4);
        assert_eq!(align_push_constant(4), 4);
        assert_eq!(align_push_constant(5), 8);
    }

    #[test]
    fn is_push_constant_aligned_checks_alignment() {
        assert!(is_push_constant_aligned(0));
        assert!(!is_push_constant_aligned(1));
        assert!(!is_push_constant_aligned(2));
        assert!(!is_push_constant_aligned(3));
        assert!(is_push_constant_aligned(4));
        assert!(is_push_constant_aligned(128));
    }

    #[test]
    fn align_push_constant_handles_large_values() {
        assert_eq!(align_push_constant(100), 100);
        assert_eq!(align_push_constant(101), 104);
        assert_eq!(align_push_constant(127), 128);
    }

    #[test]
    fn helper_configs_produce_valid_sizes() {
        let v = vertex_only(16).unwrap();
        let f = fragment_only(32).unwrap();
        let vf = vertex_fragment(64).unwrap();
        let c = compute_only(128).unwrap();

        assert_eq!(v.total_size(), 16);
        assert_eq!(f.total_size(), 32);
        assert_eq!(vf.total_size(), 64);
        assert_eq!(c.total_size(), 128);
    }

    #[test]
    fn helper_with_invalid_size_returns_error() {
        // Misaligned size
        assert!(vertex_only(17).is_err());
        assert!(fragment_only(7).is_err());
        assert!(compute_only(1).is_err());
    }

    #[test]
    fn helper_with_zero_size_returns_error() {
        assert!(vertex_only(0).is_err());
        assert!(fragment_only(0).is_err());
        assert!(compute_only(0).is_err());
        assert!(vertex_fragment(0).is_err());
    }

    #[test]
    fn helper_with_exceeding_size_returns_error() {
        assert!(vertex_only(256).is_err());
        assert!(fragment_only(256).is_err());
        assert!(compute_only(256).is_err());
        assert!(vertex_fragment(256).is_err());
    }
}

// =============================================================================
// CATEGORY 9: Writer Type Existence (~5 tests)
// =============================================================================

mod writer_types {
    use super::*;

    #[test]
    fn push_constant_writer_type_exists() {
        fn assert_type_exists<T>() {}
        assert_type_exists::<PushConstantWriter>();
    }

    #[test]
    fn compute_push_constant_writer_type_exists() {
        fn assert_type_exists<T>() {}
        assert_type_exists::<ComputePushConstantWriter>();
    }

    #[test]
    fn push_constant_writer_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<PushConstantWriter>();
    }

    #[test]
    fn push_constant_writer_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<PushConstantWriter>();
    }

    #[test]
    fn compute_push_constant_writer_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ComputePushConstantWriter>();
    }
}

// =============================================================================
// CATEGORY 10: Feature Detection (~5 tests, require GPU)
// =============================================================================

mod feature_detection {
    use super::*;

    #[test]
    
    fn supports_push_constants_returns_bool() {
        // This would require a real device
        // let supports = supports_push_constants(&device);
        // assert!(supports == true || supports == false);
    }

    #[test]
    
    fn max_push_constant_size_returns_u32() {
        // This would require a real device
        // let size = max_push_constant_size(&device);
        // assert!(size >= 0);
    }

    #[test]
    fn supports_push_constants_function_signature() {
        // Verify the function exists and has the right signature
        fn _check_signature<F: Fn(&wgpu::Device) -> bool>(_f: F) {}
        _check_signature(supports_push_constants);
    }

    #[test]
    fn max_push_constant_size_function_signature() {
        // Verify the function exists and has the right signature
        fn _check_signature<F: Fn(&wgpu::Device) -> u32>(_f: F) {}
        _check_signature(max_push_constant_size);
    }

    #[test]
    
    fn max_push_constant_size_is_at_least_128() {
        // Per wgpu spec, push constants should support at least 128 bytes
        // let size = max_push_constant_size(&device);
        // assert!(size >= 128);
    }
}

// =============================================================================
// CATEGORY 11: Integration Scenarios (~10 tests, some require GPU)
// =============================================================================

mod integration_scenarios {
    use super::*;

    #[test]
    fn create_config_with_draw_push_constants_size() {
        let config = PushConstantConfig::new()
            .add_range(
                wgpu::ShaderStages::VERTEX,
                0,
                size_of::<DrawPushConstants>() as u32,
            );
        assert!(config.is_ok());
    }

    #[test]
    fn create_config_with_extended_push_constants_size() {
        let config = PushConstantConfig::new()
            .add_range(
                wgpu::ShaderStages::VERTEX_FRAGMENT,
                0,
                size_of::<ExtendedDrawPushConstants>() as u32,
            );
        assert!(config.is_ok());
    }

    #[test]
    fn create_config_with_builder_pattern() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 16).unwrap()
            .add_range(wgpu::ShaderStages::FRAGMENT, 16, 16).unwrap();

        assert_eq!(config.ranges().len(), 2);
    }

    #[test]
    fn split_vertex_fragment_config() {
        // Vertex gets first 32 bytes, fragment gets next 32
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 32).unwrap()
            .add_range(wgpu::ShaderStages::FRAGMENT, 32, 32).unwrap();

        assert_eq!(config.total_size(), 64);
    }

    #[test]
    fn overlapping_stages_config() {
        // Same data visible to both vertex and fragment
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, 64).unwrap();

        let ranges = config.ranges();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].stages, wgpu::ShaderStages::VERTEX_FRAGMENT);
    }

    #[test]
    fn compute_shader_only_config() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::COMPUTE, 0, 64).unwrap();

        let ranges = config.ranges();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].stages, wgpu::ShaderStages::COMPUTE);
    }

    #[test]
    fn config_total_size_matches_struct_size() {
        let draw_size = size_of::<DrawPushConstants>() as u32;
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, draw_size).unwrap();

        assert_eq!(config.total_size(), draw_size);
    }

    #[test]
    fn config_validates_successfully() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 64).unwrap();

        assert!(config.validate().is_ok());
    }

    #[test]
    fn into_ranges_consumes_config() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 16).unwrap();

        let ranges = config.into_ranges();
        assert_eq!(ranges.len(), 1);
    }

    #[test]
    
    fn create_writer_and_set_values() {
        // Would require RenderPass and device
        // let writer = PushConstantWriter::new(&render_pass, &config);
        // writer.set::<DrawPushConstants>(0, &data);
    }

    #[test]
    
    fn create_compute_writer_and_set_values() {
        // Would require ComputePass and device
        // let writer = ComputePushConstantWriter::new(&compute_pass, &config);
        // writer.set::<SomeComputeData>(0, &data);
    }
}

// =============================================================================
// CATEGORY 12: Edge Cases (~10 tests)
// =============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn minimum_valid_range_size() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 4);
        // 4 bytes is the minimum valid size
        assert!(config.is_ok());
    }

    #[test]
    fn maximum_valid_single_range() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 128);
        assert!(config.is_ok());
    }

    #[test]
    fn all_shader_stages() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::all(), 0, 64);
        // All stages at once
        assert!(config.is_ok());
    }

    #[test]
    fn boundary_offset_at_124() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 124, 4);
        // 124 offset + 4 size = 128 total, should be valid
        assert!(config.is_ok());
    }

    #[test]
    fn boundary_offset_at_max_fails() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 128, 4);
        // 128 offset + any size exceeds limit
        assert!(config.is_err());
    }

    #[test]
    fn draw_push_constants_fits_multiple_times() {
        // 128 / 16 = 8 DrawPushConstants instances could fit
        let count = MAX_PUSH_CONSTANT_SIZE as usize / size_of::<DrawPushConstants>();
        assert_eq!(count, 8);
    }

    #[test]
    fn extended_push_constants_fits_twice() {
        // 128 / 64 = 2 ExtendedDrawPushConstants instances could fit
        let count = MAX_PUSH_CONSTANT_SIZE as usize / size_of::<ExtendedDrawPushConstants>();
        assert_eq!(count, 2);
    }

    #[test]
    fn ranges_can_be_non_contiguous() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 16).unwrap()
            // Skip 16-32, add at 32
            .add_range(wgpu::ShaderStages::FRAGMENT, 32, 16).unwrap();

        assert_eq!(config.ranges().len(), 2);
    }

    #[test]
    fn empty_config_produces_empty_ranges() {
        let config = PushConstantConfig::new();
        assert!(config.ranges().is_empty());
        assert_eq!(config.total_size(), 0);
    }

    #[test]
    fn alignment_helper_works_at_boundaries() {
        assert_eq!(align_push_constant(0), 0);
        assert_eq!(align_push_constant(128), 128);
        assert_eq!(align_push_constant(256), 256);
    }

    #[test]
    fn find_range_method_exists() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 16).unwrap();

        let found = config.find_range(wgpu::ShaderStages::VERTEX, 0);
        assert!(found.is_some());
    }

    #[test]
    fn find_range_returns_none_for_missing() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 16).unwrap();

        // Fragment stage not added
        let found = config.find_range(wgpu::ShaderStages::FRAGMENT, 0);
        assert!(found.is_none());
    }
}

// =============================================================================
// CATEGORY 13: Trait Implementations (~5 tests)
// =============================================================================

mod trait_implementations {
    use super::*;

    #[test]
    fn draw_push_constants_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DrawPushConstants>();
    }

    #[test]
    fn draw_push_constants_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<DrawPushConstants>();
    }

    #[test]
    fn extended_draw_push_constants_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ExtendedDrawPushConstants>();
    }

    #[test]
    fn extended_draw_push_constants_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ExtendedDrawPushConstants>();
    }

    #[test]
    fn config_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<PushConstantConfig>();
    }

    #[test]
    fn config_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<PushConstantConfig>();
    }
}

// =============================================================================
// CATEGORY 14: Config Methods (~8 tests)
// =============================================================================

mod config_methods {
    use super::*;

    #[test]
    fn add_vertex_fragment_range_convenience_method() {
        let config = PushConstantConfig::new()
            .add_vertex_fragment_range(0, 32);

        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.ranges()[0].stages, wgpu::ShaderStages::VERTEX_FRAGMENT);
    }

    #[test]
    fn add_compute_range_convenience_method() {
        let config = PushConstantConfig::new()
            .add_compute_range(0, 64);

        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.ranges()[0].stages, wgpu::ShaderStages::COMPUTE);
    }

    #[test]
    fn validate_empty_config_succeeds() {
        let config = PushConstantConfig::new();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_valid_config_succeeds() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 64).unwrap();

        assert!(config.validate().is_ok());
    }

    #[test]
    fn is_valid_write_method_exists() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 16).unwrap();

        // Check if a write at offset 0 with size 16 would be valid
        let is_valid = config.is_valid_write(wgpu::ShaderStages::VERTEX, 0, 16);
        assert!(is_valid.is_ok());
    }

    #[test]
    fn is_valid_write_fails_for_wrong_stage() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 16).unwrap();

        // Fragment not in config
        let is_valid = config.is_valid_write(wgpu::ShaderStages::FRAGMENT, 0, 16);
        assert!(is_valid.is_err());
    }

    #[test]
    fn is_valid_write_fails_for_out_of_range() {
        let config = PushConstantConfig::new()
            .add_range(wgpu::ShaderStages::VERTEX, 0, 16).unwrap();

        // Offset 16 is past the range
        let is_valid = config.is_valid_write(wgpu::ShaderStages::VERTEX, 16, 16);
        assert!(is_valid.is_err());
    }

    #[test]
    fn data_method_on_fallback_returns_slice() {
        // FallbackPushConstants should have a data() method
        // (test structure only, actual instance requires device)
        fn _check_method<T>()
        where
            T: Fn() -> &'static [u8],
        {
        }
    }
}
