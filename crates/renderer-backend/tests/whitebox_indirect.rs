//! Whitebox tests for indirect buffer support (T-WGPU-P2.1.7).
//!
//! This module tests the indirect draw and dispatch argument structs, validation
//! helpers, and buffer creation utilities with full access to implementation details.

use renderer_backend::resources::indirect::{
    // Exact wgpu-compatible types
    DrawIndexedIndirectArgs,
    DispatchIndirectArgs,
    // Re-exported from storage (16-byte)
    DrawIndirectArgs,
    // Padded versions for storage buffers
    DrawIndexedIndirectArgsPadded,
    DispatchIndirectArgsPadded,
    // Validation
    validate_draw_indirect_args,
    validate_draw_indexed_indirect_args,
    validate_dispatch_indirect_args,
    ValidationOptions,
    IndirectValidationError,
    // Buffer helpers
    indirect_buffer_size,
    // Multi-draw
    MultiDrawInfo,
};

// ============================================================================
// Category 1: Struct Size Tests
// ============================================================================

mod struct_size_tests {
    use super::*;

    #[test]
    fn test_draw_indirect_args_is_16_bytes() {
        // DrawIndirectArgs (vertex_count, instance_count, first_vertex, first_instance)
        // 4 + 4 + 4 + 4 = 16 bytes
        assert_eq!(std::mem::size_of::<DrawIndirectArgs>(), 16);
        assert_eq!(DrawIndirectArgs::SIZE, 16);
    }

    #[test]
    fn test_draw_indexed_indirect_args_is_20_bytes_exact() {
        // DrawIndexedIndirectArgs (exact wgpu layout, NO padding)
        // index_count: u32 (4) + instance_count: u32 (4) + first_index: u32 (4) +
        // base_vertex: i32 (4) + first_instance: u32 (4) = 20 bytes
        assert_eq!(std::mem::size_of::<DrawIndexedIndirectArgs>(), 20);
        assert_eq!(DrawIndexedIndirectArgs::SIZE, 20);
    }

    #[test]
    fn test_dispatch_indirect_args_is_12_bytes_exact() {
        // DispatchIndirectArgs (exact wgpu layout, NO padding)
        // x: u32 (4) + y: u32 (4) + z: u32 (4) = 12 bytes
        assert_eq!(std::mem::size_of::<DispatchIndirectArgs>(), 12);
        assert_eq!(DispatchIndirectArgs::SIZE, 12);
    }

    #[test]
    fn test_padded_draw_indexed_indirect_args_is_24_bytes() {
        // Padded version for storage buffer arrays (16-byte aligned)
        // 20 bytes + 4 bytes padding = 24 bytes
        assert_eq!(std::mem::size_of::<DrawIndexedIndirectArgsPadded>(), 24);
        assert_eq!(DrawIndexedIndirectArgsPadded::SIZE, 24);
    }

    #[test]
    fn test_padded_dispatch_indirect_args_is_16_bytes() {
        // Padded version for storage buffer arrays (16-byte aligned)
        // 12 bytes + 4 bytes padding = 16 bytes
        assert_eq!(std::mem::size_of::<DispatchIndirectArgsPadded>(), 16);
        assert_eq!(DispatchIndirectArgsPadded::SIZE, 16);
    }

    #[test]
    fn test_exact_vs_padded_size_difference() {
        // DrawIndexedIndirectArgs: exact 20 vs padded 24 (4 bytes difference)
        assert_eq!(
            DrawIndexedIndirectArgsPadded::SIZE - DrawIndexedIndirectArgs::SIZE,
            4
        );

        // DispatchIndirectArgs: exact 12 vs padded 16 (4 bytes difference)
        assert_eq!(
            DispatchIndirectArgsPadded::SIZE - DispatchIndirectArgs::SIZE,
            4
        );

        // DrawIndirectArgs: same size (already 16-byte aligned)
        // Note: DrawIndirectArgs is re-exported from storage, so it's always 16 bytes
        assert_eq!(DrawIndirectArgs::SIZE, 16);
    }

    #[test]
    fn test_struct_sizes_const_fn() {
        // Ensure SIZE constants are usable in const contexts
        const DRAW_SIZE: u64 = DrawIndirectArgs::SIZE;
        const INDEXED_SIZE: u64 = DrawIndexedIndirectArgs::SIZE;
        const DISPATCH_SIZE: u64 = DispatchIndirectArgs::SIZE;

        assert_eq!(DRAW_SIZE, 16);
        assert_eq!(INDEXED_SIZE, 20);
        assert_eq!(DISPATCH_SIZE, 12);
    }
}

// ============================================================================
// Category 2: Field Layout Tests
// ============================================================================

mod field_layout_tests {
    use super::*;

    #[test]
    fn test_draw_indirect_args_field_offsets() {
        let args = DrawIndirectArgs::new(100, 10, 5, 2);

        // Verify each field is accessible and correct
        assert_eq!(args.vertex_count, 100);
        assert_eq!(args.instance_count, 10);
        assert_eq!(args.first_vertex, 5);
        assert_eq!(args.first_instance, 2);

        // Verify byte layout via bytemuck
        let bytes: &[u8] = bytemuck::bytes_of(&args);

        // vertex_count at offset 0 (u32 LE)
        assert_eq!(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 100);
        // instance_count at offset 4
        assert_eq!(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 10);
        // first_vertex at offset 8
        assert_eq!(u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]), 5);
        // first_instance at offset 12
        assert_eq!(u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]), 2);
    }

    #[test]
    fn test_draw_indexed_indirect_args_field_offsets() {
        let args = DrawIndexedIndirectArgs::new(36, 5, 10, -100, 3);

        // Verify each field
        assert_eq!(args.index_count, 36);
        assert_eq!(args.instance_count, 5);
        assert_eq!(args.first_index, 10);
        assert_eq!(args.base_vertex, -100);
        assert_eq!(args.first_instance, 3);

        // Verify byte layout
        let bytes: &[u8] = bytemuck::bytes_of(&args);

        // index_count at offset 0
        assert_eq!(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 36);
        // instance_count at offset 4
        assert_eq!(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 5);
        // first_index at offset 8
        assert_eq!(u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]), 10);
        // base_vertex at offset 12 (i32, signed!)
        assert_eq!(i32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]), -100);
        // first_instance at offset 16
        assert_eq!(u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]), 3);
    }

    #[test]
    fn test_base_vertex_is_i32_not_u32() {
        // Critical: base_vertex must be i32 to support negative offsets
        // This is required by the wgpu spec

        // Test with positive value
        let args_positive = DrawIndexedIndirectArgs::new(36, 1, 0, 100, 0);
        assert_eq!(args_positive.base_vertex, 100);

        // Test with negative value
        let args_negative = DrawIndexedIndirectArgs::new(36, 1, 0, -100, 0);
        assert_eq!(args_negative.base_vertex, -100);

        // Test with min i32
        let args_min = DrawIndexedIndirectArgs::new(36, 1, 0, i32::MIN, 0);
        assert_eq!(args_min.base_vertex, i32::MIN);

        // Test with max i32
        let args_max = DrawIndexedIndirectArgs::new(36, 1, 0, i32::MAX, 0);
        assert_eq!(args_max.base_vertex, i32::MAX);
    }

    #[test]
    fn test_base_vertex_negative_value_bytes() {
        // Verify negative base_vertex is correctly represented in bytes
        let args = DrawIndexedIndirectArgs::new(36, 1, 0, -1, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&args);

        // -1 in i32 LE is [0xFF, 0xFF, 0xFF, 0xFF]
        assert_eq!([bytes[12], bytes[13], bytes[14], bytes[15]], [0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn test_base_vertex_roundtrip_with_negative() {
        let original = DrawIndexedIndirectArgs::new(36, 1, 0, -12345, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&original);
        let recovered: &DrawIndexedIndirectArgs = bytemuck::from_bytes(bytes);

        assert_eq!(recovered.base_vertex, -12345);
    }

    #[test]
    fn test_dispatch_indirect_args_field_offsets() {
        let args = DispatchIndirectArgs::new(64, 32, 16);

        // Verify each field
        assert_eq!(args.x, 64);
        assert_eq!(args.y, 32);
        assert_eq!(args.z, 16);

        // Verify byte layout
        let bytes: &[u8] = bytemuck::bytes_of(&args);

        // x at offset 0
        assert_eq!(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 64);
        // y at offset 4
        assert_eq!(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 32);
        // z at offset 8
        assert_eq!(u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]), 16);
    }

    #[test]
    fn test_repr_c_guarantees_layout() {
        // All indirect args structs use #[repr(C)] which guarantees:
        // - Fields are laid out in declaration order
        // - No reordering by the compiler
        // - Predictable padding

        // Verify by checking that all bytes are accounted for
        let draw = DrawIndirectArgs::new(1, 2, 3, 4);
        let draw_bytes: &[u8] = bytemuck::bytes_of(&draw);
        assert_eq!(draw_bytes.len(), 16);

        let indexed = DrawIndexedIndirectArgs::new(1, 2, 3, 4, 5);
        let indexed_bytes: &[u8] = bytemuck::bytes_of(&indexed);
        assert_eq!(indexed_bytes.len(), 20); // No padding in exact version

        let dispatch = DispatchIndirectArgs::new(1, 2, 3);
        let dispatch_bytes: &[u8] = bytemuck::bytes_of(&dispatch);
        assert_eq!(dispatch_bytes.len(), 12); // No padding in exact version
    }
}

// ============================================================================
// Category 3: Validation Tests
// ============================================================================

mod validation_tests {
    use super::*;

    // DrawIndirectArgs validation

    #[test]
    fn test_validate_draw_indirect_args_valid() {
        let args = DrawIndirectArgs::new(36, 1, 0, 0);
        let result = validate_draw_indirect_args(&args, &ValidationOptions::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_draw_indirect_args_reasonable_limits() {
        // Large but reasonable values should pass
        let args = DrawIndirectArgs::new(1_000_000, 1000, 0, 0);
        let result = validate_draw_indirect_args(&args, &ValidationOptions::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_draw_indirect_args_vertex_count_too_large() {
        let args = DrawIndirectArgs::new(u32::MAX, 1, 0, 0);
        let result = validate_draw_indirect_args(&args, &ValidationOptions::default());
        assert!(matches!(
            result,
            Err(IndirectValidationError::DrawCountTooLarge { field: "vertex_count", .. })
        ));
    }

    #[test]
    fn test_validate_draw_indirect_args_instance_count_too_large() {
        let args = DrawIndirectArgs::new(36, u32::MAX, 0, 0);
        let result = validate_draw_indirect_args(&args, &ValidationOptions::default());
        assert!(matches!(
            result,
            Err(IndirectValidationError::DrawCountTooLarge { field: "instance_count", .. })
        ));
    }

    #[test]
    fn test_validate_draw_indirect_args_zero_warn() {
        let options = ValidationOptions {
            warn_on_zero: true,
            ..Default::default()
        };

        // Zero instance count
        let args = DrawIndirectArgs::new(36, 0, 0, 0);
        let result = validate_draw_indirect_args(&args, &options);
        assert!(matches!(
            result,
            Err(IndirectValidationError::ZeroCount { field: "instance_count" })
        ));

        // Zero vertex count
        let args2 = DrawIndirectArgs::new(0, 1, 0, 0);
        let result2 = validate_draw_indirect_args(&args2, &options);
        assert!(matches!(
            result2,
            Err(IndirectValidationError::ZeroCount { field: "vertex_count" })
        ));
    }

    #[test]
    fn test_validate_draw_indirect_args_zero_no_warn() {
        let options = ValidationOptions {
            warn_on_zero: false,
            ..Default::default()
        };

        // Zero counts should pass when warn_on_zero is false
        let args = DrawIndirectArgs::new(0, 0, 0, 0);
        let result = validate_draw_indirect_args(&args, &options);
        assert!(result.is_ok());
    }

    // DrawIndexedIndirectArgs validation

    #[test]
    fn test_validate_draw_indexed_indirect_args_valid() {
        let args = DrawIndexedIndirectArgs::new(36, 1, 0, 0, 0);
        let result = validate_draw_indexed_indirect_args(&args, &ValidationOptions::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_draw_indexed_indirect_args_negative_base_vertex_valid() {
        // Negative base_vertex is valid and should pass
        let args = DrawIndexedIndirectArgs::new(36, 1, 0, -100, 0);
        let result = validate_draw_indexed_indirect_args(&args, &ValidationOptions::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_draw_indexed_indirect_args_index_count_too_large() {
        let args = DrawIndexedIndirectArgs::new(u32::MAX, 1, 0, 0, 0);
        let result = validate_draw_indexed_indirect_args(&args, &ValidationOptions::default());
        assert!(matches!(
            result,
            Err(IndirectValidationError::DrawCountTooLarge { field: "index_count", .. })
        ));
    }

    #[test]
    fn test_validate_draw_indexed_indirect_args_instance_count_too_large() {
        let args = DrawIndexedIndirectArgs::new(36, u32::MAX, 0, 0, 0);
        let result = validate_draw_indexed_indirect_args(&args, &ValidationOptions::default());
        assert!(matches!(
            result,
            Err(IndirectValidationError::DrawCountTooLarge { field: "instance_count", .. })
        ));
    }

    // DispatchIndirectArgs validation

    #[test]
    fn test_validate_dispatch_indirect_args_valid() {
        let args = DispatchIndirectArgs::new(64, 64, 1);
        let result = validate_dispatch_indirect_args(&args, &ValidationOptions::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_dispatch_indirect_args_max_per_dimension() {
        // Default max is 65535 per dimension
        let args = DispatchIndirectArgs::new(65535, 65535, 65535);
        let result = validate_dispatch_indirect_args(&args, &ValidationOptions::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_dispatch_indirect_args_x_too_large() {
        let args = DispatchIndirectArgs::new(100000, 1, 1);
        let result = validate_dispatch_indirect_args(&args, &ValidationOptions::default());
        assert!(matches!(
            result,
            Err(IndirectValidationError::WorkgroupCountTooLarge { dimension: 'X', .. })
        ));
    }

    #[test]
    fn test_validate_dispatch_indirect_args_y_too_large() {
        let args = DispatchIndirectArgs::new(1, 100000, 1);
        let result = validate_dispatch_indirect_args(&args, &ValidationOptions::default());
        assert!(matches!(
            result,
            Err(IndirectValidationError::WorkgroupCountTooLarge { dimension: 'Y', .. })
        ));
    }

    #[test]
    fn test_validate_dispatch_indirect_args_z_too_large() {
        let args = DispatchIndirectArgs::new(1, 1, 100000);
        let result = validate_dispatch_indirect_args(&args, &ValidationOptions::default());
        assert!(matches!(
            result,
            Err(IndirectValidationError::WorkgroupCountTooLarge { dimension: 'Z', .. })
        ));
    }

    #[test]
    fn test_validate_dispatch_indirect_args_overflow() {
        // Large values that could overflow when multiplied
        // Note: This will first fail the per-dimension check with default options
        let options = ValidationOptions {
            max_workgroups_per_dim: u32::MAX, // Allow any per-dimension value
            ..Default::default()
        };
        let args = DispatchIndirectArgs::new(u32::MAX, u32::MAX, u32::MAX);
        let result = validate_dispatch_indirect_args(&args, &options);
        assert!(matches!(
            result,
            Err(IndirectValidationError::WorkgroupOverflow { .. })
        ));
    }

    #[test]
    fn test_validate_dispatch_indirect_args_zero_warn() {
        let options = ValidationOptions {
            warn_on_zero: true,
            ..Default::default()
        };

        // Zero in any dimension
        let args = DispatchIndirectArgs::new(64, 0, 1);
        let result = validate_dispatch_indirect_args(&args, &options);
        assert!(matches!(
            result,
            Err(IndirectValidationError::ZeroCount { field: "workgroup dimension" })
        ));
    }

    // Custom validation options

    #[test]
    fn test_validation_options_custom_limits() {
        let options = ValidationOptions {
            max_vertex_count: 1000,
            max_instance_count: 100,
            max_workgroups_per_dim: 1000,
            warn_on_zero: false,
        };

        // Exceeds custom vertex limit
        let draw = DrawIndirectArgs::new(2000, 1, 0, 0);
        let result = validate_draw_indirect_args(&draw, &options);
        assert!(result.is_err());

        // Exceeds custom instance limit
        let draw2 = DrawIndirectArgs::new(100, 200, 0, 0);
        let result2 = validate_draw_indirect_args(&draw2, &options);
        assert!(result2.is_err());

        // Exceeds custom workgroup limit
        let dispatch = DispatchIndirectArgs::new(2000, 1, 1);
        let result3 = validate_dispatch_indirect_args(&dispatch, &options);
        assert!(result3.is_err());
    }

    #[test]
    fn test_validation_options_default_values() {
        let options = ValidationOptions::default();

        assert_eq!(options.max_vertex_count, 16 * 1024 * 1024); // 16M
        assert_eq!(options.max_instance_count, 1024 * 1024); // 1M
        assert_eq!(options.max_workgroups_per_dim, 65535);
        assert!(!options.warn_on_zero);
    }

    #[test]
    fn test_indirect_validation_error_display() {
        let error = IndirectValidationError::DrawCountTooLarge {
            field: "vertex_count",
            value: 100_000_000,
            max: 16_777_216,
        };
        let display = format!("{}", error);
        assert!(display.contains("vertex_count"));
        assert!(display.contains("100000000"));
        assert!(display.contains("16777216"));

        let error2 = IndirectValidationError::WorkgroupCountTooLarge {
            dimension: 'X',
            value: 100000,
            max: 65535,
        };
        let display2 = format!("{}", error2);
        assert!(display2.contains("X"));
        assert!(display2.contains("100000"));

        let error3 = IndirectValidationError::ZeroCount { field: "instance_count" };
        let display3 = format!("{}", error3);
        assert!(display3.contains("instance_count"));
        assert!(display3.contains("zero"));
    }
}

// ============================================================================
// Category 4: Buffer Helper Tests
// ============================================================================

mod buffer_helper_tests {
    use super::*;

    #[test]
    fn test_indirect_buffer_size_draw() {
        assert_eq!(indirect_buffer_size::<DrawIndirectArgs>(0), 0);
        assert_eq!(indirect_buffer_size::<DrawIndirectArgs>(1), 16);
        assert_eq!(indirect_buffer_size::<DrawIndirectArgs>(10), 160);
        assert_eq!(indirect_buffer_size::<DrawIndirectArgs>(100), 1600);
    }

    #[test]
    fn test_indirect_buffer_size_draw_indexed() {
        assert_eq!(indirect_buffer_size::<DrawIndexedIndirectArgs>(0), 0);
        assert_eq!(indirect_buffer_size::<DrawIndexedIndirectArgs>(1), 20);
        assert_eq!(indirect_buffer_size::<DrawIndexedIndirectArgs>(10), 200);
        assert_eq!(indirect_buffer_size::<DrawIndexedIndirectArgs>(100), 2000);
    }

    #[test]
    fn test_indirect_buffer_size_dispatch() {
        assert_eq!(indirect_buffer_size::<DispatchIndirectArgs>(0), 0);
        assert_eq!(indirect_buffer_size::<DispatchIndirectArgs>(1), 12);
        assert_eq!(indirect_buffer_size::<DispatchIndirectArgs>(10), 120);
        assert_eq!(indirect_buffer_size::<DispatchIndirectArgs>(100), 1200);
    }

    #[test]
    fn test_indirect_buffer_size_padded() {
        // Padded versions have different sizes
        assert_eq!(indirect_buffer_size::<DrawIndexedIndirectArgsPadded>(10), 240); // 10 * 24
        assert_eq!(indirect_buffer_size::<DispatchIndirectArgsPadded>(10), 160); // 10 * 16
    }

    #[test]
    fn test_indirect_buffer_size_const_fn() {
        // Ensure indirect_buffer_size is usable in const contexts
        const SIZE_1: u64 = indirect_buffer_size::<DrawIndirectArgs>(100);
        const SIZE_2: u64 = indirect_buffer_size::<DrawIndexedIndirectArgs>(100);
        const SIZE_3: u64 = indirect_buffer_size::<DispatchIndirectArgs>(100);

        assert_eq!(SIZE_1, 1600);
        assert_eq!(SIZE_2, 2000);
        assert_eq!(SIZE_3, 1200);
    }

    #[test]
    fn test_indirect_buffer_size_large_count() {
        // Test with large counts (no overflow in u64)
        let large_count: u32 = 1_000_000;
        assert_eq!(
            indirect_buffer_size::<DrawIndirectArgs>(large_count),
            16_000_000
        );
        assert_eq!(
            indirect_buffer_size::<DrawIndexedIndirectArgs>(large_count),
            20_000_000
        );
        assert_eq!(
            indirect_buffer_size::<DispatchIndirectArgs>(large_count),
            12_000_000
        );
    }

    #[test]
    fn test_multi_draw_info_draw() {
        let info = MultiDrawInfo::draw(0, 100);
        assert_eq!(info.offset, 0);
        assert_eq!(info.count, 100);

        let info_offset = MultiDrawInfo::draw(10, 50);
        assert_eq!(info_offset.offset, 160); // 10 * 16 bytes
        assert_eq!(info_offset.count, 50);

        let info_large = MultiDrawInfo::draw(1000, 500);
        assert_eq!(info_large.offset, 16000); // 1000 * 16 bytes
        assert_eq!(info_large.count, 500);
    }

    #[test]
    fn test_multi_draw_info_draw_indexed() {
        let info = MultiDrawInfo::draw_indexed(0, 100);
        assert_eq!(info.offset, 0);
        assert_eq!(info.count, 100);

        let info_offset = MultiDrawInfo::draw_indexed(10, 50);
        assert_eq!(info_offset.offset, 200); // 10 * 20 bytes
        assert_eq!(info_offset.count, 50);

        let info_large = MultiDrawInfo::draw_indexed(1000, 500);
        assert_eq!(info_large.offset, 20000); // 1000 * 20 bytes
        assert_eq!(info_large.count, 500);
    }

    #[test]
    fn test_multi_draw_info_const() {
        // Ensure MultiDrawInfo::draw and draw_indexed are const
        const INFO: MultiDrawInfo = MultiDrawInfo::draw(0, 100);
        const INFO_INDEXED: MultiDrawInfo = MultiDrawInfo::draw_indexed(5, 50);

        assert_eq!(INFO.offset, 0);
        assert_eq!(INFO.count, 100);
        assert_eq!(INFO_INDEXED.offset, 100); // 5 * 20
        assert_eq!(INFO_INDEXED.count, 50);
    }
}

// ============================================================================
// Category 5: bytemuck Tests
// ============================================================================

mod bytemuck_tests {
    use super::*;

    #[test]
    fn test_draw_indirect_args_pod_zeroable() {
        // Pod trait: type can be safely transmuted to/from bytes
        let args = DrawIndirectArgs::new(36, 1, 0, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 16);

        // Zeroable trait: zero-initialized is valid
        let zeroed: DrawIndirectArgs = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.vertex_count, 0);
        assert_eq!(zeroed.instance_count, 0);
    }

    #[test]
    fn test_draw_indexed_indirect_args_pod_zeroable() {
        let args = DrawIndexedIndirectArgs::new(36, 1, 0, -5, 0);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 20);

        let zeroed: DrawIndexedIndirectArgs = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.index_count, 0);
        assert_eq!(zeroed.base_vertex, 0);
    }

    #[test]
    fn test_dispatch_indirect_args_pod_zeroable() {
        let args = DispatchIndirectArgs::new(64, 64, 1);
        let bytes: &[u8] = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 12);

        let zeroed: DispatchIndirectArgs = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.x, 0);
        assert_eq!(zeroed.y, 0);
        assert_eq!(zeroed.z, 0);
    }

    #[test]
    fn test_bytes_of_roundtrip_draw() {
        let original = DrawIndirectArgs::new(100, 10, 5, 2);
        let bytes: &[u8] = bytemuck::bytes_of(&original);
        let recovered: &DrawIndirectArgs = bytemuck::from_bytes(bytes);

        assert_eq!(recovered.vertex_count, 100);
        assert_eq!(recovered.instance_count, 10);
        assert_eq!(recovered.first_vertex, 5);
        assert_eq!(recovered.first_instance, 2);
    }

    #[test]
    fn test_bytes_of_roundtrip_draw_indexed() {
        let original = DrawIndexedIndirectArgs::new(36, 5, 10, -100, 3);
        let bytes: &[u8] = bytemuck::bytes_of(&original);
        let recovered: &DrawIndexedIndirectArgs = bytemuck::from_bytes(bytes);

        assert_eq!(recovered.index_count, 36);
        assert_eq!(recovered.instance_count, 5);
        assert_eq!(recovered.first_index, 10);
        assert_eq!(recovered.base_vertex, -100); // Verify signed value preserved
        assert_eq!(recovered.first_instance, 3);
    }

    #[test]
    fn test_bytes_of_roundtrip_dispatch() {
        let original = DispatchIndirectArgs::new(256, 128, 64);
        let bytes: &[u8] = bytemuck::bytes_of(&original);
        let recovered: &DispatchIndirectArgs = bytemuck::from_bytes(bytes);

        assert_eq!(recovered.x, 256);
        assert_eq!(recovered.y, 128);
        assert_eq!(recovered.z, 64);
    }

    #[test]
    fn test_cast_slice_draw_array() {
        let commands = [
            DrawIndirectArgs::new(36, 1, 0, 0),
            DrawIndirectArgs::new(24, 2, 36, 1),
            DrawIndirectArgs::new(12, 3, 60, 2),
        ];

        let bytes: &[u8] = bytemuck::cast_slice(&commands);
        assert_eq!(bytes.len(), 48); // 3 * 16 bytes

        // Cast back
        let recovered: &[DrawIndirectArgs] = bytemuck::cast_slice(bytes);
        assert_eq!(recovered.len(), 3);
        assert_eq!(recovered[0].vertex_count, 36);
        assert_eq!(recovered[1].vertex_count, 24);
        assert_eq!(recovered[2].vertex_count, 12);
    }

    #[test]
    fn test_cast_slice_draw_indexed_array() {
        let commands = [
            DrawIndexedIndirectArgs::new(36, 1, 0, 0, 0),
            DrawIndexedIndirectArgs::new(24, 1, 36, -10, 1),
        ];

        let bytes: &[u8] = bytemuck::cast_slice(&commands);
        assert_eq!(bytes.len(), 40); // 2 * 20 bytes

        // Cast back
        let recovered: &[DrawIndexedIndirectArgs] = bytemuck::cast_slice(bytes);
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].index_count, 36);
        assert_eq!(recovered[1].index_count, 24);
        assert_eq!(recovered[1].base_vertex, -10); // Signed preserved
    }

    #[test]
    fn test_cast_slice_dispatch_array() {
        let commands = [
            DispatchIndirectArgs::new(8, 8, 1),
            DispatchIndirectArgs::new(16, 16, 2),
            DispatchIndirectArgs::new(32, 32, 4),
            DispatchIndirectArgs::new(64, 64, 8),
        ];

        let bytes: &[u8] = bytemuck::cast_slice(&commands);
        assert_eq!(bytes.len(), 48); // 4 * 12 bytes

        // Cast back
        let recovered: &[DispatchIndirectArgs] = bytemuck::cast_slice(bytes);
        assert_eq!(recovered.len(), 4);
        assert_eq!(recovered[0].x, 8);
        assert_eq!(recovered[3].z, 8);
    }

    #[test]
    fn test_try_cast_slice_alignment() {
        // Verify that cast_slice handles alignment correctly
        let commands = vec![
            DrawIndirectArgs::new(36, 1, 0, 0),
            DrawIndirectArgs::new(24, 1, 36, 0),
        ];

        // Vec guarantees proper alignment for the element type
        let bytes: &[u8] = bytemuck::cast_slice(&commands);
        let recovered: &[DrawIndirectArgs] = bytemuck::cast_slice(bytes);
        assert_eq!(recovered.len(), 2);
    }

    #[test]
    fn test_zeroed_struct_values() {
        // Verify that Zeroable::zeroed produces all-zero structs
        let draw: DrawIndirectArgs = bytemuck::Zeroable::zeroed();
        let draw_bytes: &[u8] = bytemuck::bytes_of(&draw);
        assert!(draw_bytes.iter().all(|&b| b == 0));

        let indexed: DrawIndexedIndirectArgs = bytemuck::Zeroable::zeroed();
        let indexed_bytes: &[u8] = bytemuck::bytes_of(&indexed);
        assert!(indexed_bytes.iter().all(|&b| b == 0));

        let dispatch: DispatchIndirectArgs = bytemuck::Zeroable::zeroed();
        let dispatch_bytes: &[u8] = bytemuck::bytes_of(&dispatch);
        assert!(dispatch_bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_from_bytes_exact_size() {
        // Verify from_bytes requires exact size
        let bytes_16 = [0u8; 16];
        let _: &DrawIndirectArgs = bytemuck::from_bytes(&bytes_16);

        let bytes_20 = [0u8; 20];
        let _: &DrawIndexedIndirectArgs = bytemuck::from_bytes(&bytes_20);

        let bytes_12 = [0u8; 12];
        let _: &DispatchIndirectArgs = bytemuck::from_bytes(&bytes_12);
    }
}

// ============================================================================
// Category 6: Constructor and Helper Method Tests
// ============================================================================

mod constructor_tests {
    use super::*;

    #[test]
    fn test_draw_indexed_indirect_args_zeroed() {
        let args = DrawIndexedIndirectArgs::zeroed();
        assert_eq!(args.index_count, 0);
        assert_eq!(args.instance_count, 0);
        assert_eq!(args.first_index, 0);
        assert_eq!(args.base_vertex, 0);
        assert_eq!(args.first_instance, 0);
    }

    #[test]
    fn test_draw_indexed_indirect_args_to_padded() {
        let exact = DrawIndexedIndirectArgs::new(36, 10, 5, -3, 2);
        let padded = exact.to_padded();

        assert_eq!(padded.index_count, 36);
        assert_eq!(padded.instance_count, 10);
        assert_eq!(padded.first_index, 5);
        assert_eq!(padded.base_vertex, -3);
        assert_eq!(padded.first_instance, 2);

        // Padded version is larger
        assert_eq!(std::mem::size_of_val(&padded), 24);
    }

    #[test]
    fn test_dispatch_indirect_args_linear() {
        let args = DispatchIndirectArgs::linear(256);
        assert_eq!(args.x, 256);
        assert_eq!(args.y, 1);
        assert_eq!(args.z, 1);
    }

    #[test]
    fn test_dispatch_indirect_args_grid_2d() {
        let args = DispatchIndirectArgs::grid_2d(16, 16);
        assert_eq!(args.x, 16);
        assert_eq!(args.y, 16);
        assert_eq!(args.z, 1);
    }

    #[test]
    fn test_dispatch_indirect_args_zeroed() {
        let args = DispatchIndirectArgs::zeroed();
        assert_eq!(args.x, 0);
        assert_eq!(args.y, 0);
        assert_eq!(args.z, 0);
    }

    #[test]
    fn test_dispatch_indirect_args_to_padded() {
        let exact = DispatchIndirectArgs::new(64, 32, 16);
        let padded = exact.to_padded();

        assert_eq!(padded.x, 64);
        assert_eq!(padded.y, 32);
        assert_eq!(padded.z, 16);

        // Padded version is larger
        assert_eq!(std::mem::size_of_val(&padded), 16);
    }

    #[test]
    fn test_dispatch_indirect_args_total_workgroups() {
        let args = DispatchIndirectArgs::new(8, 8, 2);
        assert_eq!(args.total_workgroups(), Some(128));

        let args_2d = DispatchIndirectArgs::grid_2d(1000, 1000);
        assert_eq!(args_2d.total_workgroups(), Some(1_000_000));

        let args_linear = DispatchIndirectArgs::linear(65535);
        assert_eq!(args_linear.total_workgroups(), Some(65535));
    }

    #[test]
    fn test_dispatch_indirect_args_total_workgroups_overflow() {
        // Very large values that would overflow u64
        let args = DispatchIndirectArgs::new(u32::MAX, u32::MAX, u32::MAX);
        assert!(args.total_workgroups().is_none());

        // Edge case: just before overflow
        let args2 = DispatchIndirectArgs::new(65535, 65535, 65535);
        assert!(args2.total_workgroups().is_some());
    }

    #[test]
    fn test_dispatch_indirect_args_max_workgroups_constant() {
        assert_eq!(DispatchIndirectArgs::MAX_WORKGROUPS_PER_DIMENSION, 65535);
    }
}

// ============================================================================
// Category 7: Default and PartialEq Tests
// ============================================================================

mod trait_tests {
    use super::*;

    #[test]
    fn test_draw_indirect_args_default() {
        let default = DrawIndirectArgs::default();
        assert_eq!(default.vertex_count, 0);
        assert_eq!(default.instance_count, 0);
        assert_eq!(default.first_vertex, 0);
        assert_eq!(default.first_instance, 0);
    }

    #[test]
    fn test_draw_indexed_indirect_args_default() {
        let default = DrawIndexedIndirectArgs::default();
        assert_eq!(default.index_count, 0);
        assert_eq!(default.instance_count, 0);
        assert_eq!(default.first_index, 0);
        assert_eq!(default.base_vertex, 0);
        assert_eq!(default.first_instance, 0);
    }

    #[test]
    fn test_dispatch_indirect_args_default() {
        let default = DispatchIndirectArgs::default();
        assert_eq!(default.x, 0);
        assert_eq!(default.y, 0);
        assert_eq!(default.z, 0);
    }

    #[test]
    fn test_draw_indexed_indirect_args_partial_eq() {
        let a = DrawIndexedIndirectArgs::new(36, 1, 0, 0, 0);
        let b = DrawIndexedIndirectArgs::new(36, 1, 0, 0, 0);
        let c = DrawIndexedIndirectArgs::new(24, 1, 0, 0, 0);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_dispatch_indirect_args_partial_eq() {
        let a = DispatchIndirectArgs::new(8, 8, 1);
        let b = DispatchIndirectArgs::new(8, 8, 1);
        let c = DispatchIndirectArgs::new(16, 8, 1);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_multi_draw_info_partial_eq() {
        let a = MultiDrawInfo::draw(0, 100);
        let b = MultiDrawInfo::draw(0, 100);
        let c = MultiDrawInfo::draw(10, 100);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_draw_indexed_indirect_args_copy_clone() {
        let original = DrawIndexedIndirectArgs::new(36, 1, 0, -5, 0);
        let copied = original;
        let cloned = original.clone();

        assert_eq!(original, copied);
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_dispatch_indirect_args_copy_clone() {
        let original = DispatchIndirectArgs::new(64, 32, 16);
        let copied = original;
        let cloned = original.clone();

        assert_eq!(original, copied);
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_multi_draw_info_copy_clone() {
        let original = MultiDrawInfo::draw_indexed(5, 50);
        let copied = original;
        let cloned = original.clone();

        assert_eq!(original, copied);
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_debug_impl() {
        // Verify Debug is implemented and produces output
        let draw = DrawIndexedIndirectArgs::new(36, 1, 0, -5, 0);
        let debug = format!("{:?}", draw);
        assert!(debug.contains("DrawIndexedIndirectArgs"));
        assert!(debug.contains("36"));
        assert!(debug.contains("-5"));

        let dispatch = DispatchIndirectArgs::new(64, 32, 16);
        let debug2 = format!("{:?}", dispatch);
        assert!(debug2.contains("DispatchIndirectArgs"));
        assert!(debug2.contains("64"));
    }
}
