// Blackbox contract tests for T-WGPU-P6.1.1: DrawIndirectArgs Struct
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::gpu_driven`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/gpu_driven/indirect_draw.rs (implementation)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - Task acceptance criteria (T-WGPU-P6.1.1)
//   - wgpu 25.x DrawIndirectArgs specification
//
// Public API under test:
//   - DrawIndirectArgs struct with fields: vertex_count, instance_count, first_vertex, first_instance
//   - DrawIndirectArgs::SIZE constant (16 bytes)
//   - DrawIndirectArgs::new() constructor
//   - Trait implementations: Clone, Copy, Debug, PartialEq, Default, bytemuck::Pod

use std::mem::size_of;

use renderer_backend::gpu_driven::DrawIndirectArgs;

// =============================================================================
// 1. PUBLIC API CONTRACT TESTS (4 tests)
// =============================================================================

/// Test: DrawIndirectArgs struct exists and can be instantiated with Default.
#[test]
fn blackbox_struct_exists() {
    let _args: DrawIndirectArgs = DrawIndirectArgs::default();
}

/// Test: DrawIndirectArgs::SIZE constant is publicly accessible and equals 16 bytes.
/// Per wgpu 25.x specification, DrawIndirectArgs is exactly 16 bytes (4 x u32).
#[test]
fn blackbox_size_constant_public() {
    assert_eq!(
        DrawIndirectArgs::SIZE, 16,
        "DrawIndirectArgs::SIZE must be 16 bytes per wgpu 25.x spec"
    );
}

/// Test: All four fields are publicly accessible for direct struct construction.
/// Fields: vertex_count, instance_count, first_vertex, first_instance (all u32).
#[test]
fn blackbox_fields_accessible() {
    let args = DrawIndirectArgs {
        vertex_count: 100,
        instance_count: 5,
        first_vertex: 10,
        first_instance: 2,
    };
    assert_eq!(args.vertex_count, 100, "vertex_count field should be accessible");
    assert_eq!(args.instance_count, 5, "instance_count field should be accessible");
    assert_eq!(args.first_vertex, 10, "first_vertex field should be accessible");
    assert_eq!(args.first_instance, 2, "first_instance field should be accessible");
}

/// Test: DrawIndirectArgs::new() constructor creates instance with specified values.
/// The new() constructor should accept at minimum (vertex_count, instance_count).
#[test]
fn blackbox_constructor_new() {
    // Test the full constructor with all fields
    let args = DrawIndirectArgs::new(36, 1, 0, 0);
    assert_eq!(args.vertex_count, 36, "new() should set vertex_count");
    assert_eq!(args.instance_count, 1, "new() should set instance_count");
    assert_eq!(args.first_vertex, 0, "new() should set first_vertex");
    assert_eq!(args.first_instance, 0, "new() should set first_instance");
}

// =============================================================================
// 2. TRAIT IMPLEMENTATIONS TESTS (4 tests)
// =============================================================================

/// Test: DrawIndirectArgs implements Clone trait.
#[test]
fn blackbox_clone() {
    let a = DrawIndirectArgs::new(10, 1, 5, 2);
    let b = a.clone();
    assert_eq!(a, b, "Cloned value should equal original");
}

/// Test: DrawIndirectArgs implements Copy trait.
/// After copy, both original and copy should be usable.
#[test]
fn blackbox_copy() {
    let a = DrawIndirectArgs::new(10, 1, 5, 2);
    let b = a; // Copy, not move
    // If Copy is implemented, both a and b should be accessible
    assert_eq!(a.vertex_count, b.vertex_count, "Copy should preserve vertex_count");
    assert_eq!(a.instance_count, b.instance_count, "Copy should preserve instance_count");
    assert_eq!(a.first_vertex, b.first_vertex, "Copy should preserve first_vertex");
    assert_eq!(a.first_instance, b.first_instance, "Copy should preserve first_instance");
}

/// Test: DrawIndirectArgs implements Debug trait.
/// Debug output should contain the field values.
#[test]
fn blackbox_debug() {
    let args = DrawIndirectArgs::new(10, 1, 5, 2);
    let debug = format!("{:?}", args);
    assert!(debug.contains("10"), "Debug output should contain vertex_count value");
    assert!(!debug.is_empty(), "Debug output should not be empty");
}

/// Test: DrawIndirectArgs implements PartialEq trait.
/// Equal structs should compare equal, different structs should not.
#[test]
fn blackbox_partial_eq() {
    let a = DrawIndirectArgs::new(10, 1, 5, 2);
    let b = DrawIndirectArgs::new(10, 1, 5, 2);
    let c = DrawIndirectArgs::new(20, 1, 5, 2); // Different vertex_count
    assert_eq!(a, b, "Identical DrawIndirectArgs should be equal");
    assert_ne!(a, c, "Different DrawIndirectArgs should not be equal");
}

// =============================================================================
// 3. BUFFER COMPATIBILITY TESTS (2 tests)
// =============================================================================

/// Test: DrawIndirectArgs is bytemuck::Pod compatible.
/// bytes_of() should return exactly 16 bytes.
#[test]
fn blackbox_bytemuck_pod() {
    let args = DrawIndirectArgs::new(10, 1, 0, 0);
    let bytes: &[u8] = bytemuck::bytes_of(&args);
    assert_eq!(
        bytes.len(), 16,
        "DrawIndirectArgs should serialize to exactly 16 bytes"
    );
}

/// Test: Slice of DrawIndirectArgs can be cast to bytes with bytemuck.
/// Two DrawIndirectArgs should convert to exactly 32 bytes.
#[test]
fn blackbox_slice_to_bytes() {
    let args = [
        DrawIndirectArgs::new(10, 1, 0, 0),
        DrawIndirectArgs::new(20, 2, 10, 1),
    ];
    let bytes: &[u8] = bytemuck::cast_slice(&args);
    assert_eq!(
        bytes.len(), 32,
        "Two DrawIndirectArgs should serialize to exactly 32 bytes"
    );
}

// =============================================================================
// 4. ADDITIONAL CONTRACT VERIFICATION TESTS
// =============================================================================

/// Test: sizeof(DrawIndirectArgs) matches SIZE constant.
#[test]
fn blackbox_sizeof_matches_size_constant() {
    assert_eq!(
        size_of::<DrawIndirectArgs>(),
        DrawIndirectArgs::SIZE,
        "std::mem::size_of should match SIZE constant"
    );
}

/// Test: DrawIndirectArgs implements Default with zero values.
#[test]
fn blackbox_default_values() {
    let args = DrawIndirectArgs::default();
    assert_eq!(args.vertex_count, 0, "Default vertex_count should be 0");
    assert_eq!(args.instance_count, 0, "Default instance_count should be 0");
    assert_eq!(args.first_vertex, 0, "Default first_vertex should be 0");
    assert_eq!(args.first_instance, 0, "Default first_instance should be 0");
}

/// Test: DrawIndirectArgs with maximum u32 values.
/// Struct should handle maximum field values without panic.
#[test]
fn blackbox_max_values() {
    let args = DrawIndirectArgs {
        vertex_count: u32::MAX,
        instance_count: u32::MAX,
        first_vertex: u32::MAX,
        first_instance: u32::MAX,
    };
    assert_eq!(args.vertex_count, u32::MAX);
    assert_eq!(args.instance_count, u32::MAX);
    assert_eq!(args.first_vertex, u32::MAX);
    assert_eq!(args.first_instance, u32::MAX);
}

/// Test: DrawIndirectArgs byte layout is correct (little-endian u32s).
/// This verifies wgpu 25.x layout compatibility.
#[test]
fn blackbox_byte_layout() {
    let args = DrawIndirectArgs {
        vertex_count: 0x04030201,
        instance_count: 0x08070605,
        first_vertex: 0x0C0B0A09,
        first_instance: 0x100F0E0D,
    };
    let bytes: &[u8] = bytemuck::bytes_of(&args);

    // Verify little-endian layout for each u32 field
    // vertex_count (bytes 0-3)
    assert_eq!(bytes[0], 0x01);
    assert_eq!(bytes[1], 0x02);
    assert_eq!(bytes[2], 0x03);
    assert_eq!(bytes[3], 0x04);

    // instance_count (bytes 4-7)
    assert_eq!(bytes[4], 0x05);
    assert_eq!(bytes[5], 0x06);
    assert_eq!(bytes[6], 0x07);
    assert_eq!(bytes[7], 0x08);

    // first_vertex (bytes 8-11)
    assert_eq!(bytes[8], 0x09);
    assert_eq!(bytes[9], 0x0A);
    assert_eq!(bytes[10], 0x0B);
    assert_eq!(bytes[11], 0x0C);

    // first_instance (bytes 12-15)
    assert_eq!(bytes[12], 0x0D);
    assert_eq!(bytes[13], 0x0E);
    assert_eq!(bytes[14], 0x0F);
    assert_eq!(bytes[15], 0x10);
}

/// Test: DrawIndirectArgs implements Eq trait (implied by PartialEq + Copy).
#[test]
fn blackbox_eq_trait() {
    let a = DrawIndirectArgs::new(10, 1, 0, 0);
    let b = DrawIndirectArgs::new(10, 1, 0, 0);
    // Eq implies reflexive equality
    assert!(a == a, "Eq requires reflexive equality");
    assert!(a == b, "Eq requires symmetric equality");
}

/// Test: with_counts convenience constructor if available.
#[test]
fn blackbox_with_counts_constructor() {
    // Test the with_counts convenience constructor
    let args = DrawIndirectArgs::with_counts(100, 5);
    assert_eq!(args.vertex_count, 100, "with_counts should set vertex_count");
    assert_eq!(args.instance_count, 5, "with_counts should set instance_count");
    // first_vertex and first_instance should default to 0
    assert_eq!(args.first_vertex, 0, "with_counts should default first_vertex to 0");
    assert_eq!(args.first_instance, 0, "with_counts should default first_instance to 0");
}
