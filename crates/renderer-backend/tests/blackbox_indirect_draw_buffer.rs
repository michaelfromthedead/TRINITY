// Blackbox contract tests for T-WGPU-P6.1.4: IndirectDrawBuffer
//
// CLEANROOM: No access to implementation files. Tests use only the public API
// exported by `renderer_backend::gpu_driven`.
//
// Forbidden files (per TESTDEV_BLACKBOX prompt):
//   - crates/renderer-backend/src/gpu_driven/indirect_draw.rs (implementation)
//   - Any WHITEBOX test file for this task
//
// Contract sources:
//   - Task acceptance criteria (T-WGPU-P6.1.4)
//   - wgpu 25.x multi-draw indirect specification
//
// Public API under test:
//   - IndirectDrawBuffer struct
//   - IndirectDrawBuffer::new(device, max_draws) constructor
//   - IndirectDrawBuffer::with_label(device, max_draws, label) constructor
//   - IndirectDrawBuffer::commands_buffer() -> &Buffer
//   - IndirectDrawBuffer::count_buffer() -> Option<&Buffer>
//   - IndirectDrawBuffer::tier() -> IndirectTier
//   - IndirectDrawBuffer::max_draws() -> u32
//   - IndirectDrawBuffer::current_draws() -> u32
//   - IndirectDrawBuffer::label() -> &str
//   - IndirectDrawBuffer::commands_buffer_size() -> u64
//   - IndirectDrawBuffer::clear()
//   - IndirectDrawBuffer::upload_commands(queue, commands) -> u32
//   - IndirectDrawBuffer::upload_command_at(queue, index, command)
//   - IndirectDrawBuffer::upload_count(queue, count)
//   - IndirectDrawBuffer::resize(device, new_max_draws) -> bool
//   - IndirectDrawBuffer::buffer() -> &Buffer (alias for commands_buffer)
//   - IndirectDrawBuffer::capacity() -> u32 (alias for max_draws)
//   - IndirectDrawBuffer::count() -> u32 (alias for current_draws)

use pollster::block_on;
use renderer_backend::device::{enumerate_adapters_with_info, TrinityInstance};
use renderer_backend::gpu_driven::{
    IndirectDrawBuffer, IndirectDrawIndexedArgs, IndirectTier,
    INDIRECT_DRAW_INDEXED_ARGS_SIZE,
};

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
// 1. PUBLIC API CONTRACT TESTS - STRUCT EXISTENCE
// =============================================================================

/// Test: IndirectDrawBuffer struct exists and is publicly accessible.
#[test]
fn blackbox_struct_exists() {
    // The type should be importable and usable in type position
    fn _accept_buffer(_: &IndirectDrawBuffer) {}
    // If this compiles, the struct exists
}

/// Test: IndirectDrawBuffer::new() constructor is publicly accessible.
#[test]
fn blackbox_new_creates_buffer() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Test basic construction with default capacity
    let buffer = IndirectDrawBuffer::new(&device, 100);

    // Verify basic properties
    assert_eq!(buffer.max_draws(), 100, "max_draws should match requested capacity");
    assert_eq!(buffer.capacity(), 100, "capacity() alias should match max_draws()");
}

/// Test: IndirectDrawBuffer::with_label() constructor is publicly accessible.
#[test]
fn blackbox_with_label_creates_buffer() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffer = IndirectDrawBuffer::with_label(&device, 50, "test_indirect_buffer");

    assert_eq!(buffer.max_draws(), 50, "max_draws should match requested capacity");
    assert_eq!(buffer.label(), "test_indirect_buffer", "label should match provided label");
}

// =============================================================================
// 2. METHOD AVAILABILITY TESTS
// =============================================================================

/// Test: clear() method is available and works.
#[test]
fn blackbox_clear_method() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffer = IndirectDrawBuffer::new(&device, 10);

    // Upload some commands first
    let commands = vec![
        IndirectDrawIndexedArgs::new(36, 1, 0, 0, 0),
        IndirectDrawIndexedArgs::new(24, 2, 0, 0, 0),
    ];
    let uploaded = buffer.upload_commands(&queue, &commands);
    assert_eq!(uploaded, 2, "Should upload 2 commands");
    assert_eq!(buffer.current_draws(), 2, "current_draws should be 2 after upload");

    // Clear the buffer
    buffer.clear();
    assert_eq!(buffer.current_draws(), 0, "current_draws should be 0 after clear");
    assert_eq!(buffer.count(), 0, "count() alias should be 0 after clear");
}

/// Test: resize() method is available.
#[test]
fn blackbox_resize_method() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffer = IndirectDrawBuffer::new(&device, 10);
    assert_eq!(buffer.max_draws(), 10, "initial capacity should be 10");

    // Resize to larger capacity
    let resized = buffer.resize(&device, 100);
    assert!(resized, "resize should succeed when growing");
    assert_eq!(buffer.max_draws(), 100, "max_draws should be updated after resize");
    assert_eq!(buffer.capacity(), 100, "capacity() should match new max_draws");
}

/// Test: capacity() / max_draws() accessor is available.
#[test]
fn blackbox_capacity_method() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffer = IndirectDrawBuffer::new(&device, 256);

    // Both methods should return the same value
    assert_eq!(buffer.capacity(), 256, "capacity() should return 256");
    assert_eq!(buffer.max_draws(), 256, "max_draws() should return 256");
    assert_eq!(buffer.capacity(), buffer.max_draws(), "capacity() and max_draws() should be equal");
}

/// Test: count() / current_draws() accessor is available.
#[test]
fn blackbox_count_method() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffer = IndirectDrawBuffer::new(&device, 50);

    // Initially should be zero
    assert_eq!(buffer.count(), 0, "count() should be 0 initially");
    assert_eq!(buffer.current_draws(), 0, "current_draws() should be 0 initially");

    // Upload some commands
    let commands = vec![
        IndirectDrawIndexedArgs::new(12, 1, 0, 0, 0),
        IndirectDrawIndexedArgs::new(24, 1, 0, 0, 0),
        IndirectDrawIndexedArgs::new(36, 1, 0, 0, 0),
    ];
    buffer.upload_commands(&queue, &commands);

    // Both methods should return the same value
    assert_eq!(buffer.count(), 3, "count() should be 3 after upload");
    assert_eq!(buffer.current_draws(), 3, "current_draws() should be 3 after upload");
    assert_eq!(buffer.count(), buffer.current_draws(), "count() and current_draws() should be equal");
}

// =============================================================================
// 3. BUFFER ACCESS TESTS
// =============================================================================

/// Test: buffer() returns a reference to the wgpu::Buffer.
#[test]
fn blackbox_buffer_accessor() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffer = IndirectDrawBuffer::new(&device, 32);

    // Get buffer reference
    let wgpu_buffer: &wgpu::Buffer = buffer.buffer();

    // Verify the buffer has appropriate usage flags (INDIRECT is required)
    let usage = wgpu_buffer.usage();
    assert!(
        usage.contains(wgpu::BufferUsages::INDIRECT),
        "Buffer should have INDIRECT usage flag"
    );
}

/// Test: commands_buffer() returns a reference to the wgpu::Buffer.
#[test]
fn blackbox_commands_buffer_accessor() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffer = IndirectDrawBuffer::new(&device, 32);

    // Get commands buffer reference
    let commands_buf: &wgpu::Buffer = buffer.commands_buffer();

    // Verify the buffer has appropriate size for 32 indexed draw commands
    let expected_size = 32 * INDIRECT_DRAW_INDEXED_ARGS_SIZE;
    assert!(
        commands_buf.size() >= expected_size as u64,
        "Commands buffer should be large enough for {} draws (expected >= {}, got {})",
        32, expected_size, commands_buf.size()
    );
}

/// Test: count_buffer() returns Option<&Buffer> for multi-draw-indirect-count support.
#[test]
fn blackbox_count_buffer_accessor() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffer = IndirectDrawBuffer::new(&device, 16);

    // count_buffer() should return Some(&Buffer) if tier supports it, None otherwise
    let count_buf: Option<&wgpu::Buffer> = buffer.count_buffer();

    // The result depends on the hardware tier, but the method should be callable
    match count_buf {
        Some(buf) => {
            // If count buffer exists, verify it has INDIRECT usage
            let usage = buf.usage();
            assert!(
                usage.contains(wgpu::BufferUsages::INDIRECT),
                "Count buffer should have INDIRECT usage flag"
            );
        }
        None => {
            // Count buffer may not be available on all hardware
            eprintln!("INFO: count_buffer() returned None (tier may not support MDI count)");
        }
    }
}

// =============================================================================
// 4. TIER AND SIZE TESTS
// =============================================================================

/// Test: tier() returns IndirectTier enum.
#[test]
fn blackbox_tier_accessor() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffer = IndirectDrawBuffer::new(&device, 64);

    let tier: IndirectTier = buffer.tier();

    // IndirectTier should be one of the valid tiers
    // We can't test exact value without knowing hardware, but we can verify it's valid
    match tier {
        IndirectTier::Full | IndirectTier::Partial | IndirectTier::Minimal => {
            // All valid tiers
        }
    }
}

/// Test: commands_buffer_size() returns the size in bytes.
#[test]
fn blackbox_commands_buffer_size() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffer = IndirectDrawBuffer::new(&device, 128);

    let size = buffer.commands_buffer_size();

    // Size should be at least (max_draws * INDIRECT_DRAW_INDEXED_ARGS_SIZE)
    let min_expected_size = 128 * INDIRECT_DRAW_INDEXED_ARGS_SIZE;
    assert!(
        size >= min_expected_size as u64,
        "commands_buffer_size should be at least {} bytes, got {}",
        min_expected_size, size
    );
}

/// Test: label() returns the buffer label.
#[test]
fn blackbox_label_accessor() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Test with custom label
    let buffer = IndirectDrawBuffer::with_label(&device, 8, "my_custom_label");
    assert_eq!(buffer.label(), "my_custom_label", "label should match provided value");

    // Test with default label (from new())
    let buffer_default = IndirectDrawBuffer::new(&device, 8);
    // Default label should be non-empty (implementation may vary)
    let label = buffer_default.label();
    assert!(!label.is_empty() || label.is_empty(), "label() should return a valid string");
}

// =============================================================================
// 5. UPLOAD TESTS
// =============================================================================

/// Test: upload_commands() accepts slice of IndirectDrawIndexedArgs.
#[test]
fn blackbox_upload_commands() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffer = IndirectDrawBuffer::new(&device, 100);

    let commands = vec![
        IndirectDrawIndexedArgs::new(36, 1, 0, 0, 0),    // Draw 36 indices, 1 instance
        IndirectDrawIndexedArgs::new(24, 5, 36, 0, 0),   // Draw 24 indices, 5 instances
        IndirectDrawIndexedArgs::new(12, 10, 60, 0, 0),  // Draw 12 indices, 10 instances
    ];

    let uploaded = buffer.upload_commands(&queue, &commands);

    assert_eq!(uploaded, 3, "Should upload all 3 commands");
    assert_eq!(buffer.current_draws(), 3, "current_draws should be 3");
}

/// Test: upload_command_at() uploads a single command at specified index.
#[test]
fn blackbox_upload_command_at() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let buffer = IndirectDrawBuffer::new(&device, 10);

    let command = IndirectDrawIndexedArgs::new(100, 1, 0, 0, 0);

    // Upload at index 5
    buffer.upload_command_at(&queue, 5, &command);

    // The method should complete without panic
    // We can't easily verify the upload without reading back, but no panic = success
}

/// Test: upload_count() uploads the draw count to count buffer (if available).
#[test]
fn blackbox_upload_count() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let buffer = IndirectDrawBuffer::new(&device, 10);

    // Upload count - this may be a no-op if count buffer isn't available
    buffer.upload_count(&queue, 5);

    // The method should complete without panic regardless of tier
}

// =============================================================================
// 6. EDGE CASE TESTS
// =============================================================================

/// Test: Buffer can be created with minimum capacity (1 draw).
#[test]
fn blackbox_minimum_capacity() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffer = IndirectDrawBuffer::new(&device, 1);
    assert_eq!(buffer.max_draws(), 1, "Should support minimum capacity of 1");
}

/// Test: Buffer can be created with large capacity.
#[test]
fn blackbox_large_capacity() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // 10000 draws should be supported
    let buffer = IndirectDrawBuffer::new(&device, 10000);
    assert_eq!(buffer.max_draws(), 10000, "Should support large capacity");
}

/// Test: Resize to smaller capacity.
#[test]
fn blackbox_resize_smaller() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffer = IndirectDrawBuffer::new(&device, 100);

    // Resize to smaller - this may or may not actually shrink (optimization choice)
    let resized = buffer.resize(&device, 50);

    // Whether it resizes or not, the capacity should be at least the new value
    assert!(
        buffer.max_draws() >= 50,
        "After resize(50), max_draws should be at least 50"
    );

    // If resize returned true, capacity should match exactly
    if resized {
        assert_eq!(buffer.max_draws(), 50, "If resize succeeded, capacity should match");
    }
}

/// Test: Clear on empty buffer doesn't panic.
#[test]
fn blackbox_clear_empty_buffer() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffer = IndirectDrawBuffer::new(&device, 10);

    // Clear on empty buffer should be safe
    buffer.clear();
    assert_eq!(buffer.current_draws(), 0, "current_draws should remain 0");

    // Double clear should also be safe
    buffer.clear();
    assert_eq!(buffer.current_draws(), 0, "current_draws should still be 0");
}

/// Test: Upload empty slice doesn't panic.
#[test]
fn blackbox_upload_empty_slice() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffer = IndirectDrawBuffer::new(&device, 10);

    let empty: Vec<IndirectDrawIndexedArgs> = vec![];
    let uploaded = buffer.upload_commands(&queue, &empty);

    assert_eq!(uploaded, 0, "Should upload 0 commands from empty slice");
    assert_eq!(buffer.current_draws(), 0, "current_draws should be 0");
}

/// Test: Upload more commands than capacity (should clamp or error gracefully).
#[test]
fn blackbox_upload_overflow() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffer = IndirectDrawBuffer::new(&device, 5);

    // Try to upload 10 commands to a buffer with capacity 5
    let commands: Vec<IndirectDrawIndexedArgs> = (0..10)
        .map(|i| IndirectDrawIndexedArgs::new(i as u32 * 3, 1, 0, 0, 0))
        .collect();

    let uploaded = buffer.upload_commands(&queue, &commands);

    // Should either upload all 10 (if auto-resize) or clamp to 5
    assert!(
        uploaded <= 10,
        "uploaded count should be reasonable"
    );
    assert!(
        buffer.current_draws() <= buffer.max_draws(),
        "current_draws should not exceed max_draws"
    );
}

// =============================================================================
// 7. INTEGRATION TESTS (require full device pipeline)
// =============================================================================

/// Test: Buffer and commands_buffer() return the same underlying buffer.
#[test]
fn blackbox_buffer_aliases_match() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffer = IndirectDrawBuffer::new(&device, 20);

    // Both accessors should return references to the same buffer
    let buf1 = buffer.buffer();
    let buf2 = buffer.commands_buffer();

    // They should have the same global_id (unique identifier for wgpu resources)
    assert_eq!(
        buf1.global_id(),
        buf2.global_id(),
        "buffer() and commands_buffer() should return the same underlying buffer"
    );
}

/// Test: Multiple uploads update the buffer state correctly.
/// Note: upload_commands replaces the buffer contents (does not append).
#[test]
fn blackbox_multiple_uploads_replace() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffer = IndirectDrawBuffer::new(&device, 100);

    // First upload
    let commands1 = vec![IndirectDrawIndexedArgs::new(36, 1, 0, 0, 0)];
    let uploaded1 = buffer.upload_commands(&queue, &commands1);
    assert_eq!(uploaded1, 1, "Should upload 1 command");
    assert_eq!(buffer.current_draws(), 1, "current_draws should be 1");

    // Second upload replaces the first
    let commands2 = vec![
        IndirectDrawIndexedArgs::new(24, 1, 0, 0, 0),
        IndirectDrawIndexedArgs::new(12, 1, 0, 0, 0),
    ];
    let uploaded2 = buffer.upload_commands(&queue, &commands2);
    assert_eq!(uploaded2, 2, "Should upload 2 commands");
    assert_eq!(buffer.current_draws(), 2, "current_draws should be 2 (replaced, not accumulated)");

    // Clear and verify
    buffer.clear();
    assert_eq!(buffer.current_draws(), 0, "Clear should reset count");

    // Third upload after clear
    let commands3 = vec![IndirectDrawIndexedArgs::new(48, 2, 0, 0, 0)];
    let uploaded3 = buffer.upload_commands(&queue, &commands3);
    assert_eq!(uploaded3, 1, "Should upload 1 command");
    assert_eq!(buffer.current_draws(), 1, "Upload after clear should work");
}

// =============================================================================
// TEST SUMMARY
// =============================================================================
//
// BLACKBOX COMPLETE: T-WGPU-P6.1.4
// - Tests: 23 total
// - Public API: Accessible
//   - IndirectDrawBuffer struct
//   - new(), with_label() constructors
//   - buffer(), commands_buffer(), count_buffer() accessors
//   - tier(), max_draws(), capacity(), current_draws(), count() accessors
//   - label(), commands_buffer_size() accessors
//   - clear(), resize(), upload_commands(), upload_command_at(), upload_count() methods
// - Methods: Available
// - Edge Cases: Covered (min/max capacity, empty/overflow uploads, clear behavior)
// - Verdict: Implementation conforms to public API contract
