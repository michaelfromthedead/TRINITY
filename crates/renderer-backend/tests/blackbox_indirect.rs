// Blackbox contract tests for T-WGPU-P2.1.7 Indirect Buffers
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::resources::indirect`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/resources/indirect.rs (implementation)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - Task acceptance criteria (T-WGPU-P2.1.7)
//   - wgpu DrawIndirect / DrawIndexedIndirect / DispatchIndirect specifications
//
// Public API under test:
//   - DrawIndirectArgs: vertex_count, instance_count, first_vertex, first_instance (16 bytes)
//   - DrawIndexedIndirectArgs: index_count, instance_count, first_index, base_vertex, first_instance (20 bytes)
//   - DispatchIndirectArgs: x, y, z (12 bytes)
//   - validate_draw_indirect_args(args, max_vertices, max_instances) -> Result
//   - validate_draw_indexed_indirect_args(args, max_indices, max_vertices, max_instances) -> Result
//   - validate_dispatch_indirect_args(args, max_workgroups) -> Result
//   - create_indirect_buffer(device, contents, label) -> TrinityBuffer
//   - create_empty_indirect_buffer(device, count, label) -> TrinityBuffer
//   - create_typed_indirect_buffer<T>(device, args, label) -> TrinityBuffer
//   - indirect_buffer_size::<T>(count) -> u64
//   - MultiDrawInfo struct
//   - IndirectValidationError enum
//   - ValidationOptions struct
//
// Test design rationale:
//   API Contract Tests:
//     - Struct sizes match wgpu specification (16, 20, 12 bytes)
//     - Functions accept documented parameter types
//     - Return types match documentation
//   Behavioral Tests:
//     - Validation functions return Ok/Err appropriately
//     - Size calculations are consistent
//     - base_vertex accepts negative values (i32)
//   Integration Tests (require wgpu device, marked with #[ignore]):
//     - Create actual indirect buffer with INDIRECT usage
//     - Write indirect args to buffer
//     - Buffer can be used for draw_indirect

use std::mem::size_of;

use pollster::block_on;
use renderer_backend::device::{enumerate_adapters_with_info, TrinityInstance};
use renderer_backend::resources::indirect::{
    create_empty_indirect_buffer, create_indirect_buffer, create_typed_indirect_buffer,
    indirect_buffer_size, validate_dispatch_indirect_args, validate_draw_indexed_indirect_args,
    validate_draw_indirect_args, DispatchIndirectArgs, DrawIndexedIndirectArgs, DrawIndirectArgs,
    IndirectValidationError, MultiDrawInfo, ValidationOptions,
};
use renderer_backend::resources::indirect::{
    DispatchIndirectArgsPadded, DrawIndexedIndirectArgsPadded,
};
use wgpu::BufferUsages;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Creates a TrinityInstance and gets the first available adapter.
fn get_test_adapter() -> Option<wgpu::Adapter> {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    result.adapters.into_iter().next()
}

/// Helper macro to skip test if no GPU adapter is available.
macro_rules! require_adapter {
    () => {
        match get_test_adapter() {
            Some(adapter) => adapter,
            None => {
                eprintln!("SKIP: No GPU adapter available for this test");
                return;
            }
        }
    };
}

/// Creates a wgpu device for testing.
fn create_test_device(adapter: &wgpu::Adapter) -> Option<(wgpu::Device, wgpu::Queue)> {
    block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
}

/// Helper macro to get a device, skipping if unavailable.
macro_rules! require_device {
    ($adapter:expr) => {
        match create_test_device($adapter) {
            Some((device, queue)) => (device, queue),
            None => {
                eprintln!("SKIP: Could not create device");
                return;
            }
        }
    };
}

// =============================================================================
// API CONTRACT TESTS: STRUCT SIZES
// =============================================================================

/// Test: DrawIndirectArgs is exactly 16 bytes (4 x u32) per wgpu spec.
#[test]
fn test_draw_indirect_args_size_is_16_bytes() {
    assert_eq!(
        size_of::<DrawIndirectArgs>(),
        16,
        "DrawIndirectArgs must be exactly 16 bytes (4 x u32)"
    );
}

/// Test: DrawIndexedIndirectArgs is exactly 20 bytes (5 x u32/i32) per wgpu spec.
#[test]
fn test_draw_indexed_indirect_args_size_is_20_bytes() {
    assert_eq!(
        size_of::<DrawIndexedIndirectArgs>(),
        20,
        "DrawIndexedIndirectArgs must be exactly 20 bytes (5 x u32/i32)"
    );
}

/// Test: DispatchIndirectArgs is exactly 12 bytes (3 x u32) per wgpu spec.
#[test]
fn test_dispatch_indirect_args_size_is_12_bytes() {
    assert_eq!(
        size_of::<DispatchIndirectArgs>(),
        12,
        "DispatchIndirectArgs must be exactly 12 bytes (3 x u32)"
    );
}

/// Test: Padded dispatch args align to 16 bytes for storage buffer arrays.
#[test]
fn test_dispatch_indirect_args_padded_size_is_16_bytes() {
    assert_eq!(
        size_of::<DispatchIndirectArgsPadded>(),
        16,
        "DispatchIndirectArgsPadded must be 16 bytes for storage buffer alignment"
    );
}

/// Test: Padded draw indexed args align for storage buffer arrays.
#[test]
fn test_draw_indexed_indirect_args_padded_alignment() {
    // DrawIndexedIndirectArgs is 20 bytes, padded version adds padding for WGSL struct alignment
    let size = size_of::<DrawIndexedIndirectArgsPadded>();
    // The padded version should be >= 20 bytes and properly aligned (at least 4-byte aligned)
    assert!(
        size >= 20 && size % 4 == 0,
        "DrawIndexedIndirectArgsPadded ({} bytes) should be at least 4-byte aligned and >= 20 bytes",
        size
    );
}

// =============================================================================
// API CONTRACT TESTS: STRUCT FIELDS
// =============================================================================

/// Test: DrawIndirectArgs has all required fields constructable.
#[test]
fn test_draw_indirect_args_field_construction() {
    let args = DrawIndirectArgs {
        vertex_count: 100,
        instance_count: 10,
        first_vertex: 0,
        first_instance: 0,
    };
    assert_eq!(args.vertex_count, 100);
    assert_eq!(args.instance_count, 10);
    assert_eq!(args.first_vertex, 0);
    assert_eq!(args.first_instance, 0);
}

/// Test: DrawIndexedIndirectArgs has all required fields constructable.
#[test]
fn test_draw_indexed_indirect_args_field_construction() {
    let args = DrawIndexedIndirectArgs {
        index_count: 1000,
        instance_count: 50,
        first_index: 0,
        base_vertex: 0,
        first_instance: 0,
    };
    assert_eq!(args.index_count, 1000);
    assert_eq!(args.instance_count, 50);
    assert_eq!(args.first_index, 0);
    assert_eq!(args.base_vertex, 0);
    assert_eq!(args.first_instance, 0);
}

/// Test: DrawIndexedIndirectArgs base_vertex can be negative (i32).
#[test]
fn test_draw_indexed_indirect_args_base_vertex_negative() {
    let args = DrawIndexedIndirectArgs {
        index_count: 100,
        instance_count: 1,
        first_index: 0,
        base_vertex: -100, // Negative base vertex is valid per wgpu spec
        first_instance: 0,
    };
    assert_eq!(
        args.base_vertex, -100,
        "base_vertex must accept negative values (i32)"
    );
}

/// Test: DispatchIndirectArgs has all required fields constructable.
#[test]
fn test_dispatch_indirect_args_field_construction() {
    let args = DispatchIndirectArgs { x: 64, y: 32, z: 16 };
    assert_eq!(args.x, 64);
    assert_eq!(args.y, 32);
    assert_eq!(args.z, 16);
}

// =============================================================================
// API CONTRACT TESTS: DEFAULT IMPLEMENTATIONS
// =============================================================================

/// Test: DrawIndirectArgs implements Default.
#[test]
fn test_draw_indirect_args_default() {
    let args = DrawIndirectArgs::default();
    // Default should be all zeros (no draw)
    assert_eq!(args.vertex_count, 0);
    assert_eq!(args.instance_count, 0);
    assert_eq!(args.first_vertex, 0);
    assert_eq!(args.first_instance, 0);
}

/// Test: DrawIndexedIndirectArgs implements Default.
#[test]
fn test_draw_indexed_indirect_args_default() {
    let args = DrawIndexedIndirectArgs::default();
    // Default should be all zeros (no draw)
    assert_eq!(args.index_count, 0);
    assert_eq!(args.instance_count, 0);
    assert_eq!(args.first_index, 0);
    assert_eq!(args.base_vertex, 0);
    assert_eq!(args.first_instance, 0);
}

/// Test: DispatchIndirectArgs implements Default.
#[test]
fn test_dispatch_indirect_args_default() {
    let args = DispatchIndirectArgs::default();
    // Default should be all zeros (no dispatch)
    assert_eq!(args.x, 0);
    assert_eq!(args.y, 0);
    assert_eq!(args.z, 0);
}

// =============================================================================
// API CONTRACT TESTS: COPY/CLONE TRAITS
// =============================================================================

/// Test: DrawIndirectArgs implements Copy.
#[test]
fn test_draw_indirect_args_copy() {
    let args = DrawIndirectArgs {
        vertex_count: 100,
        instance_count: 10,
        first_vertex: 5,
        first_instance: 1,
    };
    let copied = args; // Copy
    let _ = args; // Original still usable if Copy is implemented
    assert_eq!(copied.vertex_count, 100);
}

/// Test: DrawIndexedIndirectArgs implements Copy.
#[test]
fn test_draw_indexed_indirect_args_copy() {
    let args = DrawIndexedIndirectArgs {
        index_count: 1000,
        instance_count: 50,
        first_index: 10,
        base_vertex: -5,
        first_instance: 2,
    };
    let copied = args; // Copy
    let _ = args; // Original still usable if Copy is implemented
    assert_eq!(copied.index_count, 1000);
}

/// Test: DispatchIndirectArgs implements Copy.
#[test]
fn test_dispatch_indirect_args_copy() {
    let args = DispatchIndirectArgs { x: 64, y: 32, z: 16 };
    let copied = args; // Copy
    let _ = args; // Original still usable if Copy is implemented
    assert_eq!(copied.x, 64);
}

// =============================================================================
// BEHAVIORAL TESTS: SIZE CALCULATIONS
// =============================================================================

/// Test: indirect_buffer_size returns correct size for single DrawIndirectArgs.
#[test]
fn test_indirect_buffer_size_draw_single() {
    let size = indirect_buffer_size::<DrawIndirectArgs>(1);
    assert_eq!(size, 16, "Single DrawIndirectArgs buffer should be 16 bytes");
}

/// Test: indirect_buffer_size returns correct size for multiple DrawIndirectArgs.
#[test]
fn test_indirect_buffer_size_draw_multiple() {
    let size = indirect_buffer_size::<DrawIndirectArgs>(10);
    assert_eq!(
        size, 160,
        "10 DrawIndirectArgs should be 160 bytes (10 * 16)"
    );
}

/// Test: indirect_buffer_size returns correct size for single DrawIndexedIndirectArgs.
#[test]
fn test_indirect_buffer_size_draw_indexed_single() {
    let size = indirect_buffer_size::<DrawIndexedIndirectArgs>(1);
    assert_eq!(
        size, 20,
        "Single DrawIndexedIndirectArgs buffer should be 20 bytes"
    );
}

/// Test: indirect_buffer_size returns correct size for multiple DrawIndexedIndirectArgs.
#[test]
fn test_indirect_buffer_size_draw_indexed_multiple() {
    let size = indirect_buffer_size::<DrawIndexedIndirectArgs>(5);
    assert_eq!(
        size, 100,
        "5 DrawIndexedIndirectArgs should be 100 bytes (5 * 20)"
    );
}

/// Test: indirect_buffer_size returns correct size for single DispatchIndirectArgs.
#[test]
fn test_indirect_buffer_size_dispatch_single() {
    let size = indirect_buffer_size::<DispatchIndirectArgs>(1);
    assert_eq!(
        size, 12,
        "Single DispatchIndirectArgs buffer should be 12 bytes"
    );
}

/// Test: indirect_buffer_size returns correct size for multiple DispatchIndirectArgs.
#[test]
fn test_indirect_buffer_size_dispatch_multiple() {
    let size = indirect_buffer_size::<DispatchIndirectArgs>(8);
    assert_eq!(
        size, 96,
        "8 DispatchIndirectArgs should be 96 bytes (8 * 12)"
    );
}

/// Test: indirect_buffer_size handles zero count.
#[test]
fn test_indirect_buffer_size_zero_count() {
    let size = indirect_buffer_size::<DrawIndirectArgs>(0);
    assert_eq!(size, 0, "Zero count should return 0 size");
}

/// Test: indirect_buffer_size handles large counts.
#[test]
fn test_indirect_buffer_size_large_count() {
    let count = 100_000;
    let size = indirect_buffer_size::<DrawIndirectArgs>(count);
    assert_eq!(
        size,
        (count as u64) * 16,
        "Large count should scale linearly"
    );
}

// =============================================================================
// BEHAVIORAL TESTS: VALIDATION - DRAW INDIRECT
// =============================================================================

/// Test: validate_draw_indirect_args accepts valid draw call.
#[test]
fn test_validate_draw_indirect_args_valid() {
    let args = DrawIndirectArgs {
        vertex_count: 100,
        instance_count: 10,
        first_vertex: 0,
        first_instance: 0,
    };
    let options = ValidationOptions::default();
    let result = validate_draw_indirect_args(&args, &options);
    assert!(result.is_ok(), "Valid draw args should pass validation");
}

/// Test: validate_draw_indirect_args accepts zero-count draw (no-op).
#[test]
fn test_validate_draw_indirect_args_zero_count_noop() {
    let args = DrawIndirectArgs {
        vertex_count: 0,
        instance_count: 0,
        first_vertex: 0,
        first_instance: 0,
    };
    let options = ValidationOptions::default();
    let result = validate_draw_indirect_args(&args, &options);
    assert!(
        result.is_ok(),
        "Zero-count draw should be valid (no-op draw)"
    );
}

/// Test: validate_draw_indirect_args with first_vertex offset.
#[test]
fn test_validate_draw_indirect_args_with_first_vertex() {
    let args = DrawIndirectArgs {
        vertex_count: 50,
        instance_count: 1,
        first_vertex: 100,
        first_instance: 0,
    };
    let options = ValidationOptions {
        max_vertex_count: 200,
        ..Default::default()
    };
    let result = validate_draw_indirect_args(&args, &options);
    assert!(
        result.is_ok(),
        "first_vertex + vertex_count within bounds should be valid"
    );
}

/// Test: validate_draw_indirect_args rejects vertex count exceeding max.
#[test]
fn test_validate_draw_indirect_args_vertex_count_exceeds_max() {
    let args = DrawIndirectArgs {
        vertex_count: 2000, // Exceeds max_vertex_count
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    let options = ValidationOptions {
        max_vertex_count: 1000,
        ..Default::default()
    };
    let result = validate_draw_indirect_args(&args, &options);
    assert!(
        result.is_err(),
        "vertex_count exceeding max_vertex_count should fail"
    );
}

/// Test: validate_draw_indirect_args rejects instance count exceeding max.
#[test]
fn test_validate_draw_indirect_args_instance_count_exceeds_max() {
    let args = DrawIndirectArgs {
        vertex_count: 100,
        instance_count: 2000, // Exceeds max_instance_count
        first_vertex: 0,
        first_instance: 0,
    };
    let options = ValidationOptions {
        max_instance_count: 1000,
        ..Default::default()
    };
    let result = validate_draw_indirect_args(&args, &options);
    assert!(
        result.is_err(),
        "instance_count exceeding max_instance_count should fail"
    );
}

// =============================================================================
// BEHAVIORAL TESTS: VALIDATION - DRAW INDEXED INDIRECT
// =============================================================================

/// Test: validate_draw_indexed_indirect_args accepts valid indexed draw call.
#[test]
fn test_validate_draw_indexed_indirect_args_valid() {
    let args = DrawIndexedIndirectArgs {
        index_count: 1000,
        instance_count: 10,
        first_index: 0,
        base_vertex: 0,
        first_instance: 0,
    };
    let options = ValidationOptions::default();
    let result = validate_draw_indexed_indirect_args(&args, &options);
    assert!(
        result.is_ok(),
        "Valid indexed draw args should pass validation"
    );
}

/// Test: validate_draw_indexed_indirect_args accepts negative base_vertex.
#[test]
fn test_validate_draw_indexed_indirect_args_negative_base_vertex() {
    let args = DrawIndexedIndirectArgs {
        index_count: 100,
        instance_count: 1,
        first_index: 0,
        base_vertex: -50,
        first_instance: 0,
    };
    let options = ValidationOptions {
        max_vertex_count: 1000,
        ..Default::default()
    };
    let result = validate_draw_indexed_indirect_args(&args, &options);
    // Negative base_vertex is valid per wgpu spec (used for merged meshes)
    assert!(
        result.is_ok(),
        "Negative base_vertex should be valid per wgpu spec"
    );
}

/// Test: validate_draw_indexed_indirect_args rejects instance count exceeding max.
#[test]
fn test_validate_draw_indexed_indirect_args_instance_count_exceeds_max() {
    let args = DrawIndexedIndirectArgs {
        index_count: 100,
        instance_count: 2000, // Exceeds max_instance_count
        first_index: 0,
        base_vertex: 0,
        first_instance: 0,
    };
    let options = ValidationOptions {
        max_instance_count: 1000,
        ..Default::default()
    };
    let result = validate_draw_indexed_indirect_args(&args, &options);
    assert!(
        result.is_err(),
        "instance_count exceeding max_instance_count should fail"
    );
}

/// Test: validate_draw_indexed_indirect_args with valid offsets.
#[test]
fn test_validate_draw_indexed_indirect_args_valid_offsets() {
    let args = DrawIndexedIndirectArgs {
        index_count: 100,
        instance_count: 5,
        first_index: 200,
        base_vertex: 500,
        first_instance: 10,
    };
    let options = ValidationOptions {
        max_vertex_count: 2000,
        max_instance_count: 100,
        ..Default::default()
    };
    let result = validate_draw_indexed_indirect_args(&args, &options);
    assert!(
        result.is_ok(),
        "All offsets within bounds should be valid"
    );
}

// =============================================================================
// BEHAVIORAL TESTS: VALIDATION - DISPATCH INDIRECT
// =============================================================================

/// Test: validate_dispatch_indirect_args accepts valid compute dispatch.
#[test]
fn test_validate_dispatch_indirect_args_valid() {
    let args = DispatchIndirectArgs { x: 64, y: 32, z: 16 };
    let options = ValidationOptions::default();
    let result = validate_dispatch_indirect_args(&args, &options);
    assert!(
        result.is_ok(),
        "Valid dispatch args should pass validation"
    );
}

/// Test: validate_dispatch_indirect_args accepts zero dispatch (no-op).
#[test]
fn test_validate_dispatch_indirect_args_zero_noop() {
    let args = DispatchIndirectArgs { x: 0, y: 0, z: 0 };
    let options = ValidationOptions::default();
    let result = validate_dispatch_indirect_args(&args, &options);
    assert!(result.is_ok(), "Zero dispatch should be valid (no-op)");
}

/// Test: validate_dispatch_indirect_args accepts partial zero (1D dispatch).
#[test]
fn test_validate_dispatch_indirect_args_1d_dispatch() {
    let args = DispatchIndirectArgs { x: 256, y: 1, z: 1 };
    let options = ValidationOptions::default();
    let result = validate_dispatch_indirect_args(&args, &options);
    assert!(result.is_ok(), "1D dispatch (y=z=1) should be valid");
}

/// Test: validate_dispatch_indirect_args rejects workgroup overflow.
#[test]
fn test_validate_dispatch_indirect_args_workgroup_overflow() {
    let args = DispatchIndirectArgs {
        x: 65536, // wgpu default limit is 65535
        y: 1,
        z: 1,
    };
    let options = ValidationOptions {
        max_workgroups_per_dim: 65535,
        ..Default::default()
    };
    let result = validate_dispatch_indirect_args(&args, &options);
    assert!(
        result.is_err(),
        "Workgroup count exceeding max should fail"
    );
}

/// Test: validate_dispatch_indirect_args checks all dimensions.
#[test]
fn test_validate_dispatch_indirect_args_all_dimensions_checked() {
    // Check Y dimension
    let args_y = DispatchIndirectArgs {
        x: 100,
        y: 70000,
        z: 1,
    };
    let options = ValidationOptions {
        max_workgroups_per_dim: 65535,
        ..Default::default()
    };
    let result_y = validate_dispatch_indirect_args(&args_y, &options);
    assert!(result_y.is_err(), "Y dimension exceeding max should fail");

    // Check Z dimension
    let args_z = DispatchIndirectArgs {
        x: 100,
        y: 100,
        z: 70000,
    };
    let result_z = validate_dispatch_indirect_args(&args_z, &options);
    assert!(result_z.is_err(), "Z dimension exceeding max should fail");
}

// =============================================================================
// BEHAVIORAL TESTS: VALIDATION OPTIONS
// =============================================================================

/// Test: ValidationOptions has sensible defaults.
#[test]
fn test_validation_options_default() {
    let options = ValidationOptions::default();
    // Defaults should be permissive (u32::MAX or similar)
    // The exact default value is implementation-defined, just verify it's usable
    let _ = options.max_vertex_count;
    let _ = options.max_instance_count;
    let _ = options.max_workgroups_per_dim;
}

/// Test: ValidationOptions fields are configurable.
#[test]
fn test_validation_options_configurable() {
    let options = ValidationOptions {
        max_vertex_count: 10000,
        max_instance_count: 1000,
        max_workgroups_per_dim: 1024,
        warn_on_zero: false,
    };
    assert_eq!(options.max_vertex_count, 10000);
    assert_eq!(options.max_instance_count, 1000);
    assert_eq!(options.max_workgroups_per_dim, 1024);
    assert!(!options.warn_on_zero);
}

// =============================================================================
// BEHAVIORAL TESTS: INDIRECT VALIDATION ERROR
// =============================================================================

/// Test: IndirectValidationError is an enum (implements Debug).
#[test]
fn test_indirect_validation_error_debug() {
    let args = DrawIndirectArgs {
        vertex_count: u32::MAX,
        instance_count: 1,
        first_vertex: 1,
        first_instance: 0,
    };
    let options = ValidationOptions {
        max_vertex_count: 100,
        ..Default::default()
    };
    let result = validate_draw_indirect_args(&args, &options);
    if let Err(e) = result {
        let debug_str = format!("{:?}", e);
        assert!(
            !debug_str.is_empty(),
            "IndirectValidationError should implement Debug"
        );
    }
}

// =============================================================================
// BEHAVIORAL TESTS: MULTI-DRAW INFO
// =============================================================================

/// Test: MultiDrawInfo struct is constructable.
#[test]
fn test_multi_draw_info_construction() {
    let info = MultiDrawInfo {
        offset: 0,
        count: 100,
    };
    assert_eq!(info.offset, 0);
    assert_eq!(info.count, 100);
}

/// Test: MultiDrawInfo with non-zero offset.
#[test]
fn test_multi_draw_info_with_offset() {
    let info = MultiDrawInfo {
        offset: 64,  // Skip first 4 DrawIndirectArgs (4 * 16 bytes)
        count: 10,
    };
    assert_eq!(info.offset, 64);
    assert_eq!(info.count, 10);
}

/// Test: MultiDrawInfo offset calculation for DrawIndirectArgs.
#[test]
fn test_multi_draw_info_offset_draw_indirect() {
    // Offset to skip first N DrawIndirectArgs
    let skip_count = 5u64;
    let offset = skip_count * (size_of::<DrawIndirectArgs>() as u64);
    let info = MultiDrawInfo {
        offset,
        count: 10,
    };
    assert_eq!(info.offset, 80, "Offset for 5 DrawIndirectArgs should be 80 bytes");
}

/// Test: MultiDrawInfo offset calculation for DrawIndexedIndirectArgs.
#[test]
fn test_multi_draw_info_offset_draw_indexed_indirect() {
    // Offset to skip first N DrawIndexedIndirectArgs
    let skip_count = 5u64;
    let offset = skip_count * (size_of::<DrawIndexedIndirectArgs>() as u64);
    let info = MultiDrawInfo {
        offset,
        count: 10,
    };
    assert_eq!(info.offset, 100, "Offset for 5 DrawIndexedIndirectArgs should be 100 bytes");
}

// =============================================================================
// INTEGRATION TESTS: BUFFER CREATION (REQUIRE GPU)
// =============================================================================

/// Test: create_indirect_buffer creates buffer with INDIRECT usage.
#[test]

fn test_create_indirect_buffer_usage() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let args = DrawIndirectArgs {
        vertex_count: 100,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    let bytes = bytemuck::bytes_of(&args);
    let buffer = create_indirect_buffer(&device, "test_indirect", bytes);

    assert!(
        buffer.usage().contains(BufferUsages::INDIRECT),
        "Indirect buffer must have INDIRECT usage flag"
    );
}

/// Test: create_indirect_buffer size matches data.
#[test]

fn test_create_indirect_buffer_size() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let args = DrawIndirectArgs {
        vertex_count: 100,
        instance_count: 1,
        first_vertex: 0,
        first_instance: 0,
    };
    let bytes = bytemuck::bytes_of(&args);
    let buffer = create_indirect_buffer(&device, "test_indirect", bytes);

    assert!(
        buffer.size() >= 16,
        "Indirect buffer size should be at least 16 bytes"
    );
}

/// Test: create_empty_indirect_buffer creates correctly sized buffer.
#[test]

fn test_create_empty_indirect_buffer_size() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let count = 10u64;
    let expected_size = count * 16; // 16 bytes per DrawIndirectArgs
    let buffer = create_empty_indirect_buffer(&device, "empty_indirect", expected_size);

    // Size should be at least count * sizeof(DrawIndirectArgs)
    assert!(
        buffer.size() >= expected_size,
        "Empty indirect buffer should hold {} args (>= {} bytes), got {} bytes",
        count,
        expected_size,
        buffer.size()
    );
}

/// Test: create_typed_indirect_buffer creates buffer with correct content.
#[test]

fn test_create_typed_indirect_buffer_content() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let args = vec![
        DrawIndirectArgs {
            vertex_count: 100,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        },
        DrawIndirectArgs {
            vertex_count: 200,
            instance_count: 2,
            first_vertex: 100,
            first_instance: 1,
        },
    ];
    let buffer = create_typed_indirect_buffer(&device, "typed_indirect", &args);

    assert!(
        buffer.size() >= 32,
        "Typed indirect buffer should hold 2 args (>= 32 bytes)"
    );
    assert!(
        buffer.usage().contains(BufferUsages::INDIRECT),
        "Typed indirect buffer must have INDIRECT usage"
    );
}

/// Test: create_typed_indirect_buffer for DispatchIndirectArgs.
#[test]

fn test_create_typed_indirect_buffer_dispatch() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let args = vec![DispatchIndirectArgs { x: 64, y: 32, z: 16 }];
    let buffer = create_typed_indirect_buffer(&device, "dispatch_indirect", &args);

    assert!(
        buffer.size() >= 12,
        "Dispatch indirect buffer should be at least 12 bytes"
    );
    assert!(
        buffer.usage().contains(BufferUsages::INDIRECT),
        "Dispatch indirect buffer must have INDIRECT usage"
    );
}

/// Test: create_typed_indirect_buffer for DrawIndexedIndirectArgs.
#[test]

fn test_create_typed_indirect_buffer_draw_indexed() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let args = vec![DrawIndexedIndirectArgs {
        index_count: 1000,
        instance_count: 1,
        first_index: 0,
        base_vertex: -50, // Test negative base_vertex
        first_instance: 0,
    }];
    let buffer = create_typed_indirect_buffer(&device, "indexed_indirect", &args);

    assert!(
        buffer.size() >= 20,
        "Draw indexed indirect buffer should be at least 20 bytes"
    );
}

/// Test: create_empty_indirect_buffer for DrawIndexedIndirectArgs.
#[test]

fn test_create_empty_indirect_buffer_draw_indexed() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let count = 5u64;
    let expected_size = count * 20; // 20 bytes per DrawIndexedIndirectArgs
    let buffer = create_empty_indirect_buffer(&device, "empty_indexed", expected_size);

    assert!(
        buffer.size() >= expected_size,
        "Empty indexed indirect buffer should hold {} args (>= {} bytes)",
        count,
        expected_size
    );
}

/// Test: create_empty_indirect_buffer for DispatchIndirectArgs.
#[test]

fn test_create_empty_indirect_buffer_dispatch() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let count = 8u64;
    let expected_size = count * 12; // 12 bytes per DispatchIndirectArgs
    let buffer = create_empty_indirect_buffer(&device, "empty_dispatch", expected_size);

    assert!(
        buffer.size() >= expected_size,
        "Empty dispatch indirect buffer should hold {} args (>= {} bytes)",
        count,
        expected_size
    );
}

// =============================================================================
// INTEGRATION TESTS: BUFFER PROPERTIES (REQUIRE GPU)
// =============================================================================

/// Test: Indirect buffer has COPY_DST for updates.
#[test]

fn test_indirect_buffer_has_copy_dst() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let args = DrawIndirectArgs::default();
    let bytes = bytemuck::bytes_of(&args);
    let buffer = create_indirect_buffer(&device, "indirect_copy_dst", bytes);

    assert!(
        buffer.usage().contains(BufferUsages::COPY_DST),
        "Indirect buffer should have COPY_DST for queue.write_buffer"
    );
}

// =============================================================================
// EDGE CASE TESTS: BOUNDARY VALUES
// =============================================================================

/// Test: Maximum u32 values in draw args.
#[test]
fn test_draw_indirect_args_max_values() {
    let args = DrawIndirectArgs {
        vertex_count: u32::MAX,
        instance_count: u32::MAX,
        first_vertex: u32::MAX,
        first_instance: u32::MAX,
    };
    // Should construct without panic
    assert_eq!(args.vertex_count, u32::MAX);
    assert_eq!(args.instance_count, u32::MAX);
}

/// Test: Maximum i32 values for base_vertex.
#[test]
fn test_draw_indexed_indirect_args_max_base_vertex() {
    let args_pos = DrawIndexedIndirectArgs {
        index_count: 0,
        instance_count: 0,
        first_index: 0,
        base_vertex: i32::MAX,
        first_instance: 0,
    };
    assert_eq!(args_pos.base_vertex, i32::MAX);

    let args_neg = DrawIndexedIndirectArgs {
        index_count: 0,
        instance_count: 0,
        first_index: 0,
        base_vertex: i32::MIN,
        first_instance: 0,
    };
    assert_eq!(args_neg.base_vertex, i32::MIN);
}

/// Test: Maximum dispatch dimensions.
#[test]
fn test_dispatch_indirect_args_max_values() {
    let args = DispatchIndirectArgs {
        x: u32::MAX,
        y: u32::MAX,
        z: u32::MAX,
    };
    // Should construct without panic
    assert_eq!(args.x, u32::MAX);
    assert_eq!(args.y, u32::MAX);
    assert_eq!(args.z, u32::MAX);
}

/// Test: Size calculation doesn't overflow for reasonable counts.
#[test]
fn test_indirect_buffer_size_no_overflow() {
    // Maximum practical indirect buffer: 1 million draw calls
    let count = 1_000_000;
    let size = indirect_buffer_size::<DrawIndirectArgs>(count);
    assert_eq!(
        size,
        16_000_000,
        "1M draw calls should be 16MB"
    );

    let indexed_size = indirect_buffer_size::<DrawIndexedIndirectArgs>(count);
    assert_eq!(
        indexed_size,
        20_000_000,
        "1M indexed draw calls should be 20MB"
    );
}

// =============================================================================
// BYTEMUCK COMPATIBILITY TESTS
// =============================================================================

/// Test: DrawIndirectArgs is Pod (Plain Old Data) for bytemuck.
#[test]
fn test_draw_indirect_args_bytemuck_pod() {
    let args = DrawIndirectArgs {
        vertex_count: 100,
        instance_count: 10,
        first_vertex: 0,
        first_instance: 0,
    };
    let bytes = bytemuck::bytes_of(&args);
    assert_eq!(bytes.len(), 16, "DrawIndirectArgs bytes should be 16");
}

/// Test: DrawIndexedIndirectArgs is Pod for bytemuck.
#[test]
fn test_draw_indexed_indirect_args_bytemuck_pod() {
    let args = DrawIndexedIndirectArgs {
        index_count: 1000,
        instance_count: 50,
        first_index: 0,
        base_vertex: -100,
        first_instance: 0,
    };
    let bytes = bytemuck::bytes_of(&args);
    assert_eq!(bytes.len(), 20, "DrawIndexedIndirectArgs bytes should be 20");
}

/// Test: DispatchIndirectArgs is Pod for bytemuck.
#[test]
fn test_dispatch_indirect_args_bytemuck_pod() {
    let args = DispatchIndirectArgs { x: 64, y: 32, z: 16 };
    let bytes = bytemuck::bytes_of(&args);
    assert_eq!(bytes.len(), 12, "DispatchIndirectArgs bytes should be 12");
}

/// Test: DrawIndirectArgs slice conversion with bytemuck.
#[test]
fn test_draw_indirect_args_bytemuck_slice() {
    let args = vec![
        DrawIndirectArgs {
            vertex_count: 100,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        },
        DrawIndirectArgs {
            vertex_count: 200,
            instance_count: 2,
            first_vertex: 100,
            first_instance: 1,
        },
    ];
    let bytes: &[u8] = bytemuck::cast_slice(&args);
    assert_eq!(bytes.len(), 32, "Two DrawIndirectArgs should be 32 bytes");
}

/// Test: DispatchIndirectArgs slice conversion with bytemuck.
#[test]
fn test_dispatch_indirect_args_bytemuck_slice() {
    let args = vec![
        DispatchIndirectArgs { x: 64, y: 32, z: 16 },
        DispatchIndirectArgs { x: 128, y: 1, z: 1 },
    ];
    let bytes: &[u8] = bytemuck::cast_slice(&args);
    assert_eq!(bytes.len(), 24, "Two DispatchIndirectArgs should be 24 bytes");
}
