// Blackbox contract tests for T-WGPU-P6.1.3: DispatchIndirectArgs Struct
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::gpu_driven`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/gpu_driven/indirect_draw.rs (implementation)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - Task acceptance criteria (T-WGPU-P6.1.3)
//   - wgpu 25.x DispatchIndirectArgs specification
//
// Public API under test:
//   - DispatchIndirectArgs struct (alias for IndirectDispatchArgs)
//   - Fields: workgroup_count_x, workgroup_count_y, workgroup_count_z (all u32)
//   - SIZE constant: 12 bytes (3 x u32)
//   - Constructors: new(x, y, z), single(x), linear(x), grid_2d(x, y), zeroed()
//   - Accessors: x(), y(), z()
//   - Methods: is_active(), total_workgroups(), for_elements()
//   - Traits: Default, Copy, Clone, Pod, Zeroable

use std::mem::size_of;

use renderer_backend::gpu_driven::DispatchIndirectArgs;

// =============================================================================
// API CONTRACT TESTS: STRUCT EXISTS AND HAS CORRECT SIZE
// =============================================================================

/// Test: DispatchIndirectArgs struct exists and can be instantiated with default.
#[test]
fn blackbox_dispatch_struct_exists() {
    let _args: DispatchIndirectArgs = DispatchIndirectArgs::default();
}

/// Test: SIZE constant equals 12 bytes (3 x u32) per wgpu specification.
#[test]
fn blackbox_dispatch_size_constant_12() {
    assert_eq!(
        DispatchIndirectArgs::SIZE,
        12,
        "DispatchIndirectArgs::SIZE must be exactly 12 bytes (3 x u32)"
    );
}

/// Test: sizeof(DispatchIndirectArgs) matches SIZE constant.
#[test]
fn blackbox_dispatch_sizeof_matches_size_constant() {
    assert_eq!(
        size_of::<DispatchIndirectArgs>(),
        DispatchIndirectArgs::SIZE,
        "size_of::<DispatchIndirectArgs>() must equal SIZE constant"
    );
}

/// Test: sizeof(DispatchIndirectArgs) is exactly 12 bytes.
#[test]
fn blackbox_dispatch_sizeof_is_12_bytes() {
    assert_eq!(
        size_of::<DispatchIndirectArgs>(),
        12,
        "DispatchIndirectArgs must be exactly 12 bytes (3 x u32)"
    );
}

// =============================================================================
// API CONTRACT TESTS: THREE FIELDS WITH CORRECT NAMES
// =============================================================================

/// Test: DispatchIndirectArgs has three public fields with correct names.
#[test]
fn blackbox_dispatch_three_fields() {
    let args = DispatchIndirectArgs {
        workgroup_count_x: 64,
        workgroup_count_y: 64,
        workgroup_count_z: 1,
    };
    assert_eq!(args.workgroup_count_x, 64);
    assert_eq!(args.workgroup_count_y, 64);
    assert_eq!(args.workgroup_count_z, 1);
}

/// Test: Fields are accessible and modifiable.
#[test]
fn blackbox_dispatch_fields_mutable() {
    let mut args = DispatchIndirectArgs::default();
    args.workgroup_count_x = 128;
    args.workgroup_count_y = 256;
    args.workgroup_count_z = 4;
    assert_eq!(args.workgroup_count_x, 128);
    assert_eq!(args.workgroup_count_y, 256);
    assert_eq!(args.workgroup_count_z, 4);
}

// =============================================================================
// API CONTRACT TESTS: CONSTRUCTORS
// =============================================================================

/// Test: new() constructor creates args with specified dimensions.
#[test]
fn blackbox_dispatch_new_constructor() {
    let args = DispatchIndirectArgs::new(32, 32, 4);
    assert_eq!(args.workgroup_count_x, 32);
    assert_eq!(args.workgroup_count_y, 32);
    assert_eq!(args.workgroup_count_z, 4);
}

/// Test: single() constructor creates 1D dispatch (y=1, z=1).
#[test]
fn blackbox_dispatch_single_constructor() {
    let args = DispatchIndirectArgs::single(128);
    assert_eq!(args.workgroup_count_x, 128);
    assert_eq!(args.workgroup_count_y, 1);
    assert_eq!(args.workgroup_count_z, 1);
}

/// Test: linear() constructor creates 1D dispatch (alias for single).
#[test]
fn blackbox_dispatch_linear_constructor() {
    let args = DispatchIndirectArgs::linear(256);
    assert_eq!(args.workgroup_count_x, 256);
    assert_eq!(args.workgroup_count_y, 1);
    assert_eq!(args.workgroup_count_z, 1);
}

/// Test: grid_2d() constructor creates 2D dispatch (z=1).
#[test]
fn blackbox_dispatch_grid_2d_constructor() {
    let args = DispatchIndirectArgs::grid_2d(16, 16);
    assert_eq!(args.workgroup_count_x, 16);
    assert_eq!(args.workgroup_count_y, 16);
    assert_eq!(args.workgroup_count_z, 1);
}

/// Test: zeroed() constructor creates dispatch with all zeros.
#[test]
fn blackbox_dispatch_zeroed_constructor() {
    let args = DispatchIndirectArgs::zeroed();
    assert_eq!(args.workgroup_count_x, 0);
    assert_eq!(args.workgroup_count_y, 0);
    assert_eq!(args.workgroup_count_z, 0);
}

/// Test: new() is const fn (compile-time usable).
#[test]
fn blackbox_dispatch_new_is_const() {
    const ARGS: DispatchIndirectArgs = DispatchIndirectArgs::new(8, 8, 8);
    assert_eq!(ARGS.workgroup_count_x, 8);
    assert_eq!(ARGS.workgroup_count_y, 8);
    assert_eq!(ARGS.workgroup_count_z, 8);
}

/// Test: single() is const fn.
#[test]
fn blackbox_dispatch_single_is_const() {
    const ARGS: DispatchIndirectArgs = DispatchIndirectArgs::single(64);
    assert_eq!(ARGS.workgroup_count_x, 64);
}

// =============================================================================
// API CONTRACT TESTS: ACCESSORS
// =============================================================================

/// Test: x() accessor returns workgroup_count_x.
#[test]
fn blackbox_dispatch_x_accessor() {
    let args = DispatchIndirectArgs::new(100, 200, 300);
    assert_eq!(args.x(), 100);
}

/// Test: y() accessor returns workgroup_count_y.
#[test]
fn blackbox_dispatch_y_accessor() {
    let args = DispatchIndirectArgs::new(100, 200, 300);
    assert_eq!(args.y(), 200);
}

/// Test: z() accessor returns workgroup_count_z.
#[test]
fn blackbox_dispatch_z_accessor() {
    let args = DispatchIndirectArgs::new(100, 200, 300);
    assert_eq!(args.z(), 300);
}

// =============================================================================
// API CONTRACT TESTS: METHODS
// =============================================================================

/// Test: is_active() returns true for non-zero dispatch.
#[test]
fn blackbox_dispatch_is_active_true() {
    let args = DispatchIndirectArgs::new(1, 1, 1);
    assert!(args.is_active(), "1x1x1 dispatch should be active");
}

/// Test: is_active() returns false for zero dispatch.
#[test]
fn blackbox_dispatch_is_active_false_all_zero() {
    let args = DispatchIndirectArgs::zeroed();
    assert!(!args.is_active(), "0x0x0 dispatch should not be active");
}

/// Test: is_active() returns false if any dimension is zero.
#[test]
fn blackbox_dispatch_is_active_false_partial_zero() {
    let args_x_zero = DispatchIndirectArgs::new(0, 1, 1);
    assert!(
        !args_x_zero.is_active(),
        "0xNxN dispatch should not be active"
    );

    let args_y_zero = DispatchIndirectArgs::new(1, 0, 1);
    assert!(
        !args_y_zero.is_active(),
        "Nx0xN dispatch should not be active"
    );

    let args_z_zero = DispatchIndirectArgs::new(1, 1, 0);
    assert!(
        !args_z_zero.is_active(),
        "NxNx0 dispatch should not be active"
    );
}

/// Test: total_workgroups() calculates x * y * z.
#[test]
fn blackbox_dispatch_total_workgroups() {
    let args = DispatchIndirectArgs::new(4, 5, 6);
    assert_eq!(
        args.total_workgroups(),
        120,
        "4 * 5 * 6 = 120 total workgroups"
    );
}

/// Test: total_workgroups() returns 0 for zeroed dispatch.
#[test]
fn blackbox_dispatch_total_workgroups_zero() {
    let args = DispatchIndirectArgs::zeroed();
    assert_eq!(args.total_workgroups(), 0);
}

/// Test: total_workgroups() uses u64 to avoid overflow.
#[test]
fn blackbox_dispatch_total_workgroups_large() {
    let args = DispatchIndirectArgs::new(65535, 65535, 1);
    let expected = 65535u64 * 65535u64;
    assert_eq!(
        args.total_workgroups(),
        expected,
        "Large workgroup count should use u64"
    );
}

/// Test: for_elements() creates dispatch for given element count and workgroup size.
#[test]
fn blackbox_dispatch_for_elements() {
    // 1024 elements with workgroup size 64 = 16 workgroups
    let args = DispatchIndirectArgs::for_elements(1024, 64);
    assert_eq!(args.workgroup_count_x, 16);
    assert_eq!(args.workgroup_count_y, 1);
    assert_eq!(args.workgroup_count_z, 1);
}

/// Test: for_elements() rounds up when elements don't divide evenly.
#[test]
fn blackbox_dispatch_for_elements_rounds_up() {
    // 1000 elements with workgroup size 64 = ceil(1000/64) = 16 workgroups
    let args = DispatchIndirectArgs::for_elements(1000, 64);
    assert_eq!(args.workgroup_count_x, 16, "Should round up: ceil(1000/64) = 16");
}

/// Test: for_elements() with exact division.
#[test]
fn blackbox_dispatch_for_elements_exact() {
    // 256 elements with workgroup size 256 = 1 workgroup
    let args = DispatchIndirectArgs::for_elements(256, 256);
    assert_eq!(args.workgroup_count_x, 1);
}

/// Test: for_elements() with single element.
#[test]
fn blackbox_dispatch_for_elements_single() {
    // 1 element with workgroup size 64 = 1 workgroup
    let args = DispatchIndirectArgs::for_elements(1, 64);
    assert_eq!(args.workgroup_count_x, 1);
}

// =============================================================================
// API CONTRACT TESTS: DEFAULT TRAIT
// =============================================================================

/// Test: Default trait is implemented.
#[test]
fn blackbox_dispatch_default_trait() {
    let args = DispatchIndirectArgs::default();
    // Default should be all zeros (no dispatch)
    assert_eq!(args.workgroup_count_x, 0);
    assert_eq!(args.workgroup_count_y, 0);
    assert_eq!(args.workgroup_count_z, 0);
}

/// Test: Default equals zeroed().
#[test]
fn blackbox_dispatch_default_equals_zeroed() {
    let default_args = DispatchIndirectArgs::default();
    let zeroed_args = DispatchIndirectArgs::zeroed();
    assert_eq!(default_args.workgroup_count_x, zeroed_args.workgroup_count_x);
    assert_eq!(default_args.workgroup_count_y, zeroed_args.workgroup_count_y);
    assert_eq!(default_args.workgroup_count_z, zeroed_args.workgroup_count_z);
}

// =============================================================================
// API CONTRACT TESTS: COPY/CLONE TRAITS
// =============================================================================

/// Test: Copy trait is implemented.
#[test]
fn blackbox_dispatch_copy_trait() {
    let args = DispatchIndirectArgs::new(64, 32, 16);
    let copied = args; // Copy
    let _ = args; // Original still usable if Copy is implemented
    assert_eq!(copied.workgroup_count_x, 64);
}

/// Test: Clone trait is implemented.
#[test]
fn blackbox_dispatch_clone_trait() {
    let args = DispatchIndirectArgs::new(64, 32, 16);
    let cloned = args.clone();
    assert_eq!(cloned.workgroup_count_x, 64);
    assert_eq!(cloned.workgroup_count_y, 32);
    assert_eq!(cloned.workgroup_count_z, 16);
}

// =============================================================================
// API CONTRACT TESTS: BYTEMUCK COMPATIBILITY
// =============================================================================

/// Test: bytemuck::bytes_of produces exactly 12 bytes.
#[test]
fn blackbox_dispatch_bytemuck_12_bytes() {
    let args = DispatchIndirectArgs::new(64, 64, 1);
    let bytes: &[u8] = bytemuck::bytes_of(&args);
    assert_eq!(bytes.len(), 12, "bytemuck::bytes_of should produce 12 bytes");
}

/// Test: bytemuck round-trip preserves values.
#[test]
fn blackbox_dispatch_bytemuck_roundtrip() {
    let original = DispatchIndirectArgs::new(123, 456, 789);
    let bytes: &[u8] = bytemuck::bytes_of(&original);
    let restored: &DispatchIndirectArgs = bytemuck::from_bytes(bytes);
    assert_eq!(restored.workgroup_count_x, 123);
    assert_eq!(restored.workgroup_count_y, 456);
    assert_eq!(restored.workgroup_count_z, 789);
}

/// Test: bytemuck::cast_slice works for arrays.
#[test]
fn blackbox_dispatch_bytemuck_slice() {
    let args = vec![
        DispatchIndirectArgs::new(1, 2, 3),
        DispatchIndirectArgs::new(4, 5, 6),
    ];
    let bytes: &[u8] = bytemuck::cast_slice(&args);
    assert_eq!(bytes.len(), 24, "Two DispatchIndirectArgs should be 24 bytes");
}

/// Test: bytemuck::zeroed produces zeroed instance.
#[test]
fn blackbox_dispatch_bytemuck_zeroed() {
    let zeroed: DispatchIndirectArgs = bytemuck::Zeroable::zeroed();
    assert_eq!(zeroed.workgroup_count_x, 0);
    assert_eq!(zeroed.workgroup_count_y, 0);
    assert_eq!(zeroed.workgroup_count_z, 0);
}

// =============================================================================
// EDGE CASE TESTS: BOUNDARY VALUES
// =============================================================================

/// Test: Maximum u32 values in all dimensions.
#[test]
fn blackbox_dispatch_max_values() {
    let args = DispatchIndirectArgs::new(u32::MAX, u32::MAX, u32::MAX);
    assert_eq!(args.workgroup_count_x, u32::MAX);
    assert_eq!(args.workgroup_count_y, u32::MAX);
    assert_eq!(args.workgroup_count_z, u32::MAX);
}

/// Test: Minimum non-zero dispatch (1x1x1).
#[test]
fn blackbox_dispatch_min_active() {
    let args = DispatchIndirectArgs::new(1, 1, 1);
    assert!(args.is_active());
    assert_eq!(args.total_workgroups(), 1);
}

/// Test: Large but valid dispatch dimensions.
#[test]
fn blackbox_dispatch_large_valid() {
    // wgpu default limit is 65535 per dimension
    let args = DispatchIndirectArgs::new(65535, 65535, 65535);
    assert!(args.is_active());
}

/// Test: Typical compute shader dispatch sizes.
#[test]
fn blackbox_dispatch_typical_sizes() {
    // 1D compute: process 1M elements with workgroup size 256
    let args_1d = DispatchIndirectArgs::for_elements(1_000_000, 256);
    assert!(args_1d.workgroup_count_x > 0);
    assert_eq!(args_1d.workgroup_count_y, 1);
    assert_eq!(args_1d.workgroup_count_z, 1);

    // 2D compute: 1024x1024 image with 8x8 workgroups
    let args_2d = DispatchIndirectArgs::grid_2d(128, 128);
    assert_eq!(args_2d.workgroup_count_x, 128);
    assert_eq!(args_2d.workgroup_count_y, 128);
    assert_eq!(args_2d.workgroup_count_z, 1);

    // 3D compute: volume processing
    let args_3d = DispatchIndirectArgs::new(16, 16, 16);
    assert_eq!(args_3d.total_workgroups(), 4096);
}

// =============================================================================
// MEMORY LAYOUT TESTS
// =============================================================================

/// Test: Fields are laid out contiguously (no padding).
#[test]
fn blackbox_dispatch_no_padding() {
    // 3 * sizeof(u32) = 12 bytes, which should equal struct size
    let field_size = 3 * size_of::<u32>();
    assert_eq!(
        size_of::<DispatchIndirectArgs>(),
        field_size,
        "Struct should have no padding (3 x u32 = 12 bytes)"
    );
}

/// Test: First field is at offset 0.
#[test]
fn blackbox_dispatch_first_field_offset() {
    let args = DispatchIndirectArgs::new(0x12345678, 0, 0);
    let bytes: &[u8] = bytemuck::bytes_of(&args);
    // Little-endian: first 4 bytes should be 0x78, 0x56, 0x34, 0x12
    let first_u32 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(first_u32, 0x12345678, "workgroup_count_x should be at offset 0");
}

/// Test: Second field is at offset 4.
#[test]
fn blackbox_dispatch_second_field_offset() {
    let args = DispatchIndirectArgs::new(0, 0xAABBCCDD, 0);
    let bytes: &[u8] = bytemuck::bytes_of(&args);
    let second_u32 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(second_u32, 0xAABBCCDD, "workgroup_count_y should be at offset 4");
}

/// Test: Third field is at offset 8.
#[test]
fn blackbox_dispatch_third_field_offset() {
    let args = DispatchIndirectArgs::new(0, 0, 0xDEADBEEF);
    let bytes: &[u8] = bytemuck::bytes_of(&args);
    let third_u32 = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    assert_eq!(third_u32, 0xDEADBEEF, "workgroup_count_z should be at offset 8");
}

// =============================================================================
// WGPU 25.X COMPATIBILITY TESTS
// =============================================================================

/// Test: Struct layout matches wgpu DispatchIndirectArgs specification.
/// Per wgpu spec: struct { x: u32, y: u32, z: u32 } = 12 bytes, tightly packed.
#[test]
fn blackbox_dispatch_wgpu_spec_compliance() {
    // wgpu spec requires:
    // 1. Exactly 12 bytes
    // 2. Three u32 fields (x, y, z workgroup counts)
    // 3. Tightly packed (no padding)

    assert_eq!(
        size_of::<DispatchIndirectArgs>(),
        12,
        "Must be 12 bytes per wgpu spec"
    );
    assert_eq!(
        DispatchIndirectArgs::SIZE,
        12,
        "SIZE constant must be 12 per wgpu spec"
    );

    // Verify we can construct with all fields
    let args = DispatchIndirectArgs {
        workgroup_count_x: 1,
        workgroup_count_y: 1,
        workgroup_count_z: 1,
    };
    assert!(args.is_active());
}

/// Test: Buffer alignment is suitable for INDIRECT usage.
#[test]
fn blackbox_dispatch_buffer_alignment() {
    // wgpu requires indirect buffers to be at least 4-byte aligned
    let alignment = std::mem::align_of::<DispatchIndirectArgs>();
    assert!(
        alignment >= 4,
        "DispatchIndirectArgs must be at least 4-byte aligned for indirect buffers"
    );
}
