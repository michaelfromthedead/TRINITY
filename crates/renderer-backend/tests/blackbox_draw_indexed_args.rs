// Blackbox contract tests for T-WGPU-P6.1.2: DrawIndexedIndirectArgs Struct
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::gpu_driven::DrawIndexedIndirectArgs`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/gpu_driven/indirect_draw.rs (implementation)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - Task acceptance criteria (T-WGPU-P6.1.2)
//   - wgpu 25.x DrawIndexedIndirectArgs specification
//   - base_vertex MUST be i32 to support negative vertex offsets
//
// Public API under test:
//   - DrawIndexedIndirectArgs struct with 5 fields
//   - SIZE constant = 20 bytes
//   - base_vertex field accepts negative values (i32, not u32)
//   - Bytemuck compatibility (Pod + Zeroable)
//   - Default implementation

use renderer_backend::gpu_driven::{
    DrawIndexedIndirectArgs, INDIRECT_DRAW_INDEXED_ARGS_SIZE,
};

// =============================================================================
// TEST: STRUCT EXISTS AND IS CONSTRUCTIBLE
// =============================================================================

/// Test: DrawIndexedIndirectArgs struct exists and can be instantiated.
#[test]
fn blackbox_struct_exists_and_default() {
    let args: DrawIndexedIndirectArgs = DrawIndexedIndirectArgs::default();
    // Default values should be zero
    assert_eq!(args.index_count, 0);
    assert_eq!(args.instance_count, 0);
    assert_eq!(args.first_index, 0);
    assert_eq!(args.base_vertex, 0);
    assert_eq!(args.first_instance, 0);
}

/// Test: DrawIndexedIndirectArgs can be constructed with field initialization.
#[test]
fn blackbox_struct_field_construction() {
    let args = DrawIndexedIndirectArgs {
        index_count: 36,
        instance_count: 1,
        first_index: 0,
        base_vertex: 0,
        first_instance: 0,
    };
    assert_eq!(args.index_count, 36);
    assert_eq!(args.instance_count, 1);
    assert_eq!(args.first_index, 0);
    assert_eq!(args.base_vertex, 0);
    assert_eq!(args.first_instance, 0);
}

// =============================================================================
// TEST: SIZE CONSTANT = 20 BYTES
// =============================================================================

/// Test: SIZE constant equals 20 bytes per wgpu specification.
#[test]
fn blackbox_size_constant_20() {
    // wgpu DrawIndexedIndirectArgs is 5 x u32/i32 = 20 bytes
    assert_eq!(
        INDIRECT_DRAW_INDEXED_ARGS_SIZE, 20,
        "INDIRECT_DRAW_INDEXED_ARGS_SIZE must be 20 bytes per wgpu spec"
    );
}

/// Test: std::mem::size_of matches SIZE constant.
#[test]
fn blackbox_sizeof_matches_constant() {
    let size = std::mem::size_of::<DrawIndexedIndirectArgs>();
    assert_eq!(
        size, INDIRECT_DRAW_INDEXED_ARGS_SIZE as usize,
        "size_of<DrawIndexedIndirectArgs> must match SIZE constant"
    );
}

/// Test: Struct size is exactly 20 bytes (no padding).
#[test]
fn blackbox_struct_size_20_bytes() {
    let size = std::mem::size_of::<DrawIndexedIndirectArgs>();
    assert_eq!(
        size, 20,
        "DrawIndexedIndirectArgs must be exactly 20 bytes (5 x 4-byte fields)"
    );
}

// =============================================================================
// TEST: FIVE FIELDS ACCESSIBLE
// =============================================================================

/// Test: All five fields exist and are accessible.
#[test]
fn blackbox_five_fields_accessible() {
    let args = DrawIndexedIndirectArgs {
        index_count: 100,
        instance_count: 10,
        first_index: 50,
        base_vertex: -50, // CRITICAL: negative value
        first_instance: 5,
    };

    // Verify all fields accessible and hold expected values
    assert_eq!(args.index_count, 100, "index_count field must exist");
    assert_eq!(args.instance_count, 10, "instance_count field must exist");
    assert_eq!(args.first_index, 50, "first_index field must exist");
    assert_eq!(args.base_vertex, -50, "base_vertex field must exist and accept negative");
    assert_eq!(args.first_instance, 5, "first_instance field must exist");
}

// =============================================================================
// TEST: SIGNED base_vertex (CRITICAL)
// =============================================================================

/// Test: base_vertex accepts negative value -50.
#[test]
fn blackbox_base_vertex_negative_50() {
    let args = DrawIndexedIndirectArgs {
        index_count: 36,
        instance_count: 1,
        first_index: 0,
        base_vertex: -50,
        first_instance: 0,
    };
    assert_eq!(
        args.base_vertex, -50,
        "base_vertex MUST accept -50 (wgpu 25.x requires i32)"
    );
}

/// Test: base_vertex accepts negative value -100.
#[test]
fn blackbox_base_vertex_negative_100() {
    let args = DrawIndexedIndirectArgs {
        index_count: 36,
        instance_count: 1,
        first_index: 0,
        base_vertex: -100,
        first_instance: 0,
    };
    assert_eq!(
        args.base_vertex, -100,
        "base_vertex MUST accept -100 (wgpu 25.x requires i32)"
    );
}

/// Test: base_vertex accepts i32::MIN (extreme negative).
#[test]
fn blackbox_base_vertex_i32_min() {
    let args = DrawIndexedIndirectArgs {
        index_count: 36,
        instance_count: 1,
        first_index: 0,
        base_vertex: i32::MIN,
        first_instance: 0,
    };
    assert_eq!(
        args.base_vertex,
        i32::MIN,
        "base_vertex MUST accept i32::MIN (-2147483648)"
    );
}

/// Test: base_vertex accepts i32::MAX (extreme positive).
#[test]
fn blackbox_base_vertex_i32_max() {
    let args = DrawIndexedIndirectArgs {
        index_count: 36,
        instance_count: 1,
        first_index: 0,
        base_vertex: i32::MAX,
        first_instance: 0,
    };
    assert_eq!(
        args.base_vertex,
        i32::MAX,
        "base_vertex MUST accept i32::MAX (2147483647)"
    );
}

/// Test: base_vertex handles typical negative mesh offset values.
#[test]
fn blackbox_base_vertex_typical_negative_offsets() {
    // Test common negative offset scenarios in merged mesh rendering
    let test_values: &[i32] = &[-1, -10, -100, -1000, -10000, -100000];

    for &val in test_values {
        let args = DrawIndexedIndirectArgs {
            index_count: 36,
            instance_count: 1,
            first_index: 0,
            base_vertex: val,
            first_instance: 0,
        };
        assert_eq!(
            args.base_vertex, val,
            "base_vertex must accept negative value {}", val
        );
    }
}

// =============================================================================
// TEST: BYTEMUCK COMPATIBILITY
// =============================================================================

/// Test: Struct implements bytemuck::Pod (can cast to bytes).
#[test]
fn blackbox_bytemuck_pod_bytes_of() {
    let args = DrawIndexedIndirectArgs {
        index_count: 36,
        instance_count: 1,
        first_index: 0,
        base_vertex: -50,
        first_instance: 0,
    };

    let bytes: &[u8] = bytemuck::bytes_of(&args);
    assert_eq!(
        bytes.len(), 20,
        "bytemuck::bytes_of must return 20 bytes"
    );
}

/// Test: Struct implements bytemuck::Zeroable (can create zeroed).
#[test]
fn blackbox_bytemuck_zeroable() {
    let zeroed: DrawIndexedIndirectArgs = bytemuck::Zeroable::zeroed();
    assert_eq!(zeroed.index_count, 0);
    assert_eq!(zeroed.instance_count, 0);
    assert_eq!(zeroed.first_index, 0);
    assert_eq!(zeroed.base_vertex, 0);
    assert_eq!(zeroed.first_instance, 0);
}

/// Test: Bytemuck round-trip preserves negative base_vertex.
#[test]
fn blackbox_bytemuck_roundtrip_negative_base_vertex() {
    let original = DrawIndexedIndirectArgs {
        index_count: 100,
        instance_count: 5,
        first_index: 10,
        base_vertex: -500,
        first_instance: 2,
    };

    // Convert to bytes and back
    let bytes: &[u8] = bytemuck::bytes_of(&original);
    let recovered: &DrawIndexedIndirectArgs = bytemuck::from_bytes(bytes);

    assert_eq!(recovered.index_count, original.index_count);
    assert_eq!(recovered.instance_count, original.instance_count);
    assert_eq!(recovered.first_index, original.first_index);
    assert_eq!(
        recovered.base_vertex, original.base_vertex,
        "Negative base_vertex must survive bytemuck roundtrip"
    );
    assert_eq!(recovered.first_instance, original.first_instance);
}

/// Test: Byte layout matches expected wgpu specification order.
/// Fields: index_count, instance_count, first_index, base_vertex, first_instance
/// All 4-byte aligned, total 20 bytes.
#[test]
fn blackbox_byte_layout_wgpu_spec() {
    let args = DrawIndexedIndirectArgs {
        index_count: 0x01020304,
        instance_count: 0x05060708,
        first_index: 0x090A0B0C,
        base_vertex: -1, // 0xFFFFFFFF in two's complement
        first_instance: 0x11121314,
    };

    let bytes: &[u8] = bytemuck::bytes_of(&args);

    // index_count at offset 0 (little-endian)
    assert_eq!(bytes[0..4], [0x04, 0x03, 0x02, 0x01], "index_count at offset 0");

    // instance_count at offset 4
    assert_eq!(bytes[4..8], [0x08, 0x07, 0x06, 0x05], "instance_count at offset 4");

    // first_index at offset 8
    assert_eq!(bytes[8..12], [0x0C, 0x0B, 0x0A, 0x09], "first_index at offset 8");

    // base_vertex at offset 12 (-1 as i32 = 0xFFFFFFFF)
    assert_eq!(bytes[12..16], [0xFF, 0xFF, 0xFF, 0xFF], "base_vertex at offset 12");

    // first_instance at offset 16
    assert_eq!(bytes[16..20], [0x14, 0x13, 0x12, 0x11], "first_instance at offset 16");
}

// =============================================================================
// TEST: COPY AND CLONE TRAITS
// =============================================================================

/// Test: Struct implements Copy trait.
#[test]
fn blackbox_copy_trait() {
    let args = DrawIndexedIndirectArgs {
        index_count: 36,
        instance_count: 1,
        first_index: 0,
        base_vertex: -100,
        first_instance: 0,
    };

    let copy = args; // Copy
    let _ = args;    // Original still valid (proves Copy)
    assert_eq!(copy.base_vertex, -100);
}

/// Test: Struct implements Clone trait.
#[test]
fn blackbox_clone_trait() {
    let args = DrawIndexedIndirectArgs {
        index_count: 36,
        instance_count: 1,
        first_index: 0,
        base_vertex: -100,
        first_instance: 0,
    };

    let cloned = args.clone();
    assert_eq!(cloned.base_vertex, -100);
    assert_eq!(cloned.index_count, 36);
}

// =============================================================================
// TEST: DEBUG TRAIT
// =============================================================================

/// Test: Struct implements Debug trait for formatting.
#[test]
fn blackbox_debug_trait() {
    let args = DrawIndexedIndirectArgs {
        index_count: 36,
        instance_count: 1,
        first_index: 0,
        base_vertex: -100,
        first_instance: 0,
    };

    let debug_str = format!("{:?}", args);

    // Debug output should contain field names and values
    assert!(debug_str.contains("36"), "Debug should show index_count value");
    assert!(
        debug_str.contains("-100") || debug_str.contains("base_vertex"),
        "Debug should show base_vertex or its negative value"
    );
}

// =============================================================================
// TEST: ALIGNMENT
// =============================================================================

/// Test: Struct alignment is 4 bytes (standard for GPU upload).
#[test]
fn blackbox_alignment_4_bytes() {
    let align = std::mem::align_of::<DrawIndexedIndirectArgs>();
    assert_eq!(
        align, 4,
        "DrawIndexedIndirectArgs alignment should be 4 bytes"
    );
}

// =============================================================================
// TEST: EDGE CASES
// =============================================================================

/// Test: Large field values don't overflow.
#[test]
fn blackbox_large_values() {
    let args = DrawIndexedIndirectArgs {
        index_count: u32::MAX,
        instance_count: u32::MAX,
        first_index: u32::MAX,
        base_vertex: i32::MIN,
        first_instance: u32::MAX,
    };

    assert_eq!(args.index_count, u32::MAX);
    assert_eq!(args.instance_count, u32::MAX);
    assert_eq!(args.first_index, u32::MAX);
    assert_eq!(args.base_vertex, i32::MIN);
    assert_eq!(args.first_instance, u32::MAX);
}

/// Test: Array of args has correct stride.
#[test]
fn blackbox_array_stride() {
    let args_array: [DrawIndexedIndirectArgs; 3] = [
        DrawIndexedIndirectArgs::default(),
        DrawIndexedIndirectArgs::default(),
        DrawIndexedIndirectArgs::default(),
    ];

    let array_size = std::mem::size_of_val(&args_array);
    assert_eq!(
        array_size, 60,
        "Array of 3 DrawIndexedIndirectArgs should be 60 bytes (3 x 20)"
    );
}

/// Test: Slice of args can be cast with bytemuck.
#[test]
fn blackbox_slice_cast() {
    let args_array: [DrawIndexedIndirectArgs; 2] = [
        DrawIndexedIndirectArgs {
            index_count: 36,
            instance_count: 1,
            first_index: 0,
            base_vertex: -10,
            first_instance: 0,
        },
        DrawIndexedIndirectArgs {
            index_count: 72,
            instance_count: 2,
            first_index: 36,
            base_vertex: -20,
            first_instance: 1,
        },
    ];

    let bytes: &[u8] = bytemuck::cast_slice(&args_array);
    assert_eq!(bytes.len(), 40, "2 args = 40 bytes");

    // Cast back
    let recovered: &[DrawIndexedIndirectArgs] = bytemuck::cast_slice(bytes);
    assert_eq!(recovered.len(), 2);
    assert_eq!(recovered[0].base_vertex, -10);
    assert_eq!(recovered[1].base_vertex, -20);
}
