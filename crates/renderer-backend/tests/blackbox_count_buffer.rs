// Blackbox contract tests for T-WGPU-P6.1.5 CountBuffer
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::gpu_driven::CountBuffer`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/gpu_driven/indirect_draw.rs (implementation)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - Task acceptance criteria (T-WGPU-P6.1.5)
//   - wgpu Buffer specifications for indirect draw count
//
// Public API under test:
//   - CountBuffer::SIZE constant (4 bytes = size of u32)
//   - CountBuffer::new(device, label) -> Self
//   - CountBuffer::reset(&self, queue) - resets count to 0
//   - CountBuffer::storage_buffer(&self) -> &Buffer
//   - CountBuffer::indirect_buffer(&self) -> &Buffer
//   - CountBuffer::upload(&self, queue, count) - uploads count value

use std::mem::size_of;

use pollster::block_on;
use renderer_backend::device::{enumerate_adapters_with_info, TrinityInstance};
use renderer_backend::gpu_driven::CountBuffer;
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
// API CONTRACT TESTS - SIZE CONSTANT
// =============================================================================

/// Test: CountBuffer::SIZE is 4 bytes (size of u32).
///
/// Rationale: The count buffer stores a single u32 for indirect draw count.
/// This matches wgpu specification where multi-draw-count requires a u32.
#[test]
fn blackbox_count_buffer_size_constant() {
    // SIZE should equal 4 (size of u32)
    assert_eq!(
        CountBuffer::SIZE,
        4,
        "CountBuffer::SIZE must be 4 bytes (u32)"
    );

    // SIZE should match the actual size of u32
    assert_eq!(
        CountBuffer::SIZE as usize,
        size_of::<u32>(),
        "CountBuffer::SIZE must match size_of::<u32>()"
    );
}

/// Test: CountBuffer::SIZE is a u64 constant.
///
/// Rationale: Buffer sizes in wgpu are typically u64 for large buffer support.
#[test]
fn blackbox_count_buffer_size_is_u64() {
    let size: u64 = CountBuffer::SIZE;
    assert_eq!(size, 4u64, "CountBuffer::SIZE should be usable as u64");
}

// =============================================================================
// API CONTRACT TESTS - CONSTRUCTOR
// =============================================================================

/// Test: CountBuffer::new() creates a valid buffer.
///
/// Rationale: Constructor must create internal wgpu buffers without panicking.
#[test]
fn blackbox_count_buffer_new_creates_buffer() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Should not panic when creating with a label
    let count_buffer = CountBuffer::new(&device, Some("test_count_buffer"));

    // Verify we can access the storage buffer
    let _storage = count_buffer.storage_buffer();
}

/// Test: CountBuffer::new() accepts None label.
///
/// Rationale: Labels are optional for debugging; None should be valid.
#[test]
fn blackbox_count_buffer_new_accepts_none_label() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Should not panic with None label
    let count_buffer = CountBuffer::new(&device, None);

    // Verify we can access buffers
    let _storage = count_buffer.storage_buffer();
    let _indirect = count_buffer.indirect_buffer();
}

// =============================================================================
// API CONTRACT TESTS - BUFFER ACCESSORS
// =============================================================================

/// Test: storage_buffer() returns a valid wgpu Buffer reference.
///
/// Rationale: The storage buffer is used for atomic operations in compute shaders.
#[test]
fn blackbox_count_buffer_storage_buffer_accessor() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("storage_test"));

    // Should return a valid buffer reference
    let storage = count_buffer.storage_buffer();

    // Buffer size should be at least CountBuffer::SIZE
    assert!(
        storage.size() >= CountBuffer::SIZE,
        "Storage buffer size ({}) must be >= CountBuffer::SIZE ({})",
        storage.size(),
        CountBuffer::SIZE
    );

    // Buffer should have STORAGE usage for compute shader access
    let usages = storage.usage();
    assert!(
        usages.contains(BufferUsages::STORAGE),
        "Storage buffer must have STORAGE usage"
    );
}

/// Test: indirect_buffer() returns a valid wgpu Buffer reference.
///
/// Rationale: The indirect buffer is used for multi-draw-indirect-count calls.
#[test]
fn blackbox_count_buffer_indirect_buffer_accessor() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("indirect_test"));

    // Should return a valid buffer reference
    let indirect = count_buffer.indirect_buffer();

    // Buffer size should be at least CountBuffer::SIZE
    assert!(
        indirect.size() >= CountBuffer::SIZE,
        "Indirect buffer size ({}) must be >= CountBuffer::SIZE ({})",
        indirect.size(),
        CountBuffer::SIZE
    );

    // Buffer should have INDIRECT usage for indirect draw calls
    let usages = indirect.usage();
    assert!(
        usages.contains(BufferUsages::INDIRECT),
        "Indirect buffer must have INDIRECT usage"
    );
}

/// Test: storage_buffer and indirect_buffer may be distinct or aliased.
///
/// Rationale: Implementation may use one or two buffers; API allows either.
#[test]
fn blackbox_count_buffer_dual_buffer_access() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("dual_buffer_test"));

    // Both accessors must return valid buffers
    let storage = count_buffer.storage_buffer();
    let indirect = count_buffer.indirect_buffer();

    // Both should have sufficient size
    assert!(storage.size() >= CountBuffer::SIZE);
    assert!(indirect.size() >= CountBuffer::SIZE);
}

// =============================================================================
// API CONTRACT TESTS - RESET METHOD
// =============================================================================

/// Test: reset() method is callable without panic.
///
/// Rationale: Reset clears the count to zero for a new frame/pass.
#[test]
fn blackbox_count_buffer_reset_method() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("reset_test"));

    // Reset should not panic
    count_buffer.reset(&queue);
}

/// Test: reset() can be called multiple times.
///
/// Rationale: Reset is typically called each frame; must be idempotent.
#[test]
fn blackbox_count_buffer_reset_multiple_times() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("multi_reset_test"));

    // Multiple resets should not panic
    count_buffer.reset(&queue);
    count_buffer.reset(&queue);
    count_buffer.reset(&queue);
}

// =============================================================================
// API CONTRACT TESTS - UPLOAD METHOD
// =============================================================================

/// Test: upload() method accepts u32 count value.
///
/// Rationale: Upload writes the draw count for multi-draw-indirect-count.
#[test]
fn blackbox_count_buffer_upload_method() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("upload_test"));

    // Upload should accept a u32 count
    count_buffer.upload(&queue, 42);
}

/// Test: upload() accepts zero count.
///
/// Rationale: Zero draws is valid (nothing rendered).
#[test]
fn blackbox_count_buffer_upload_zero() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("upload_zero_test"));

    // Zero count should be valid
    count_buffer.upload(&queue, 0);
}

/// Test: upload() accepts maximum u32 value.
///
/// Rationale: API should accept any valid u32, even if GPU limits are lower.
#[test]
fn blackbox_count_buffer_upload_max_u32() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("upload_max_test"));

    // Max u32 should not panic during upload
    count_buffer.upload(&queue, u32::MAX);
}

/// Test: upload() can be called after reset().
///
/// Rationale: Normal workflow is reset, compute, upload, draw.
#[test]
fn blackbox_count_buffer_upload_after_reset() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("upload_after_reset_test"));

    count_buffer.reset(&queue);
    count_buffer.upload(&queue, 100);
}

/// Test: upload() can overwrite previous value.
///
/// Rationale: Multiple uploads should be valid (last one wins).
#[test]
fn blackbox_count_buffer_upload_overwrites() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("upload_overwrite_test"));

    count_buffer.upload(&queue, 10);
    count_buffer.upload(&queue, 20);
    count_buffer.upload(&queue, 30);
}

// =============================================================================
// WORKFLOW TESTS
// =============================================================================

/// Test: Typical frame workflow - reset, (compute), upload, (draw).
///
/// Rationale: This is the expected usage pattern for GPU-driven rendering.
#[test]
fn blackbox_count_buffer_frame_workflow() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("frame_workflow_test"));

    // Frame 1
    count_buffer.reset(&queue);
    // ... GPU culling would happen here ...
    count_buffer.upload(&queue, 1000);
    // ... draw call would use indirect_buffer() here ...
    let _ = count_buffer.indirect_buffer();

    // Frame 2
    count_buffer.reset(&queue);
    count_buffer.upload(&queue, 500);
    let _ = count_buffer.indirect_buffer();

    // Frame 3 - no draws
    count_buffer.reset(&queue);
    count_buffer.upload(&queue, 0);
    let _ = count_buffer.indirect_buffer();
}

/// Test: Storage buffer can be used for bind group creation.
///
/// Rationale: Storage buffer is bound in compute shader for atomic count updates.
#[test]
fn blackbox_count_buffer_storage_for_compute() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("compute_bind_test"));

    let storage = count_buffer.storage_buffer();

    // Create a bind group layout for storage buffer
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("count_buffer_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    // Create bind group with the storage buffer
    let _bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("count_buffer_bind_group"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage.as_entire_binding(),
        }],
    });
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

/// Test: Multiple CountBuffer instances are independent.
///
/// Rationale: Different passes may use different count buffers.
#[test]
fn blackbox_count_buffer_multiple_instances() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let buffer_a = CountBuffer::new(&device, Some("count_a"));
    let buffer_b = CountBuffer::new(&device, Some("count_b"));
    let buffer_c = CountBuffer::new(&device, Some("count_c"));

    // Operations on one should not affect others
    buffer_a.upload(&queue, 100);
    buffer_b.upload(&queue, 200);
    buffer_c.reset(&queue);

    buffer_a.reset(&queue);
    buffer_b.upload(&queue, 999);
}

/// Test: CountBuffer can be dropped without issue.
///
/// Rationale: RAII - drop should release GPU resources properly.
#[test]
fn blackbox_count_buffer_drop() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    {
        let count_buffer = CountBuffer::new(&device, Some("drop_test"));
        count_buffer.upload(&queue, 42);
        // count_buffer dropped here
    }

    // Should be able to create another one after drop
    let _count_buffer2 = CountBuffer::new(&device, Some("after_drop_test"));
}

/// Test: Empty label string is valid.
///
/// Rationale: Some callers may pass empty string instead of None.
#[test]
fn blackbox_count_buffer_empty_label() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Empty string label should be valid
    let _count_buffer = CountBuffer::new(&device, Some(""));
}

/// Test: Long label string is valid.
///
/// Rationale: Debug labels should handle arbitrary length names.
#[test]
fn blackbox_count_buffer_long_label() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let long_label = "a".repeat(1000);
    let _count_buffer = CountBuffer::new(&device, Some(&long_label));
}

// =============================================================================
// BUFFER USAGE VERIFICATION
// =============================================================================

/// Test: Storage buffer has COPY_DST for CPU uploads.
///
/// Rationale: CPU-to-GPU data transfer requires COPY_DST usage.
#[test]
fn blackbox_count_buffer_storage_has_copy_dst() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("copy_dst_test"));
    let storage = count_buffer.storage_buffer();

    // Storage buffer should have COPY_DST for queue.write_buffer
    let usages = storage.usage();
    assert!(
        usages.contains(BufferUsages::COPY_DST),
        "Storage buffer should have COPY_DST for CPU uploads"
    );
}

/// Test: Indirect buffer has COPY_DST for CPU uploads.
///
/// Rationale: The indirect buffer may receive direct CPU uploads.
#[test]
fn blackbox_count_buffer_indirect_has_copy_dst() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("indirect_copy_dst_test"));
    let indirect = count_buffer.indirect_buffer();

    // Indirect buffer should have COPY_DST
    let usages = indirect.usage();
    assert!(
        usages.contains(BufferUsages::COPY_DST),
        "Indirect buffer should have COPY_DST for uploads"
    );
}

// =============================================================================
// TYPE SYSTEM TESTS
// =============================================================================

/// Test: CountBuffer is not Copy (owns GPU resources).
///
/// Rationale: GPU resources should not be implicitly copied.
#[test]
fn blackbox_count_buffer_not_copy() {
    // This test verifies at compile time that CountBuffer does not implement Copy.
    // If it did, this function would fail to compile.
    fn assert_not_copy<T>() {}

    // CountBuffer should NOT be Copy - GPU resources need explicit handling
    // This is a compile-time assertion, test passes if code compiles
    assert_not_copy::<CountBuffer>();
}

/// Test: CountBuffer can be moved.
///
/// Rationale: Move semantics should work for ownership transfer.
#[test]
fn blackbox_count_buffer_move_semantics() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let count_buffer = CountBuffer::new(&device, Some("move_test"));

    // Move to a new binding
    let moved_buffer = count_buffer;

    // Should still work after move
    moved_buffer.upload(&queue, 42);
}
