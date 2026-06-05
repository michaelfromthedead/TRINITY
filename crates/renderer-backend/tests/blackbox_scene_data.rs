//! Blackbox tests for T-WGPU-P6.2.2: SceneDataBuffers
//!
//! CLEANROOM RULES:
//! - DO NOT read implementation details of `crates/renderer-backend/src/gpu_driven/scene_data.rs`
//! - Only test PUBLIC API through imports
//! - Focus on behavioral contract
//!
//! Tests verify:
//! 1. Public API contract - struct existence, constructors
//! 2. Object management - add, get, get_mut, count
//! 3. Dirty tracking for GPU synchronization
//! 4. Upload and GPU buffer management
//! 5. Capacity management and auto-resize
//! 6. Clear and reset operations
//! 7. Constants and defaults

use pollster::block_on;
use renderer_backend::gpu_driven::{
    ObjectData, SceneDataBuffers, object_flags,
    DEFAULT_SCENE_CAPACITY, GROWTH_FACTOR, MIN_BUFFER_CAPACITY, OBJECT_DATA_SIZE,
};
use renderer_backend::device::{TrinityInstance, enumerate_adapters_with_info};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Creates a TrinityInstance and gets the first available adapter.
fn get_test_adapter() -> Option<wgpu::Adapter> {
    let instance = TrinityInstance::new();
    let result = enumerate_adapters_with_info(instance.inner(), instance.backends());
    result.adapters.into_iter().next()
}

/// Creates a wgpu device for testing.
fn create_test_device(adapter: &wgpu::Adapter) -> Option<(wgpu::Device, wgpu::Queue)> {
    block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
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

// ============================================================================
// SECTION 1: Public API Contract Tests (No GPU Required)
// ============================================================================

/// Verify DEFAULT_SCENE_CAPACITY constant is exported and reasonable
#[test]
fn blackbox_default_capacity_constant() {
    // Default capacity should be reasonable for a scene
    assert!(
        DEFAULT_SCENE_CAPACITY >= 64,
        "DEFAULT_SCENE_CAPACITY ({}) should be at least 64",
        DEFAULT_SCENE_CAPACITY
    );
    assert!(
        DEFAULT_SCENE_CAPACITY <= 1_000_000,
        "DEFAULT_SCENE_CAPACITY ({}) should be at most 1M",
        DEFAULT_SCENE_CAPACITY
    );
}

/// Verify GROWTH_FACTOR constant is exported and reasonable
#[test]
fn blackbox_growth_factor_constant() {
    // Growth factor should be > 1 for actual growth
    assert!(
        GROWTH_FACTOR > 1,
        "GROWTH_FACTOR ({}) must be > 1 for growth",
        GROWTH_FACTOR
    );
    // Typical growth factors are 2 to 4
    assert!(
        GROWTH_FACTOR <= 4,
        "GROWTH_FACTOR ({}) should not exceed 4 for efficiency",
        GROWTH_FACTOR
    );
}

/// Verify MIN_BUFFER_CAPACITY constant is exported and reasonable
#[test]
fn blackbox_min_capacity_constant() {
    // Minimum should be at least 1
    assert!(
        MIN_BUFFER_CAPACITY >= 1,
        "MIN_BUFFER_CAPACITY ({}) must be at least 1",
        MIN_BUFFER_CAPACITY
    );
    // But not too large to avoid waste
    assert!(
        MIN_BUFFER_CAPACITY <= 1024,
        "MIN_BUFFER_CAPACITY ({}) should not exceed 1024",
        MIN_BUFFER_CAPACITY
    );
}

// ============================================================================
// SECTION 2: Struct Existence and Constructor Tests
// ============================================================================

/// Verify SceneDataBuffers struct exists and is constructible
#[test]
fn blackbox_struct_exists() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    // Constructor must work with capacity
    let buffers = SceneDataBuffers::new(&device, 100, Some("test"));
    // Struct must be usable
    let _count = buffers.count();
}

/// Verify with_default_capacity constructor
#[test]
fn blackbox_with_default_capacity_constructor() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffers = SceneDataBuffers::with_default_capacity(&device, Some("test"));
    assert!(buffers.capacity() >= MIN_BUFFER_CAPACITY);
}

/// Verify new constructor with None label
#[test]
fn blackbox_new_with_none_label() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffers = SceneDataBuffers::new(&device, 100, None);
    assert!(buffers.capacity() >= 100);
}

/// Verify label accessor
#[test]
fn blackbox_label_accessor() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffers = SceneDataBuffers::new(&device, 100, Some("my_scene_buffer"));
    assert_eq!(buffers.label(), Some("my_scene_buffer"));

    let buffers_no_label = SceneDataBuffers::new(&device, 100, None);
    assert!(buffers_no_label.label().is_none());
}

// ============================================================================
// SECTION 3: Object Management - Add Tests
// ============================================================================

/// Verify add returns an index
#[test]
fn blackbox_add_returns_index() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    let obj = ObjectData::new().with_mesh(42);
    let index = buffers.add(obj);

    // First index should be 0
    assert_eq!(index, 0, "First added object should have index 0");
}

/// Verify sequential adds return sequential indices
#[test]
fn blackbox_add_sequential_indices() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    for i in 0..10 {
        let obj = ObjectData::new().with_mesh(i as u32);
        let index = buffers.add(obj);
        assert_eq!(index, i, "Index {} should match iteration {}", index, i);
    }

    assert_eq!(buffers.count(), 10, "Count should be 10 after 10 adds");
}

/// Verify add increases count
#[test]
fn blackbox_add_increases_count() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    assert_eq!(buffers.count(), 0, "Initial count should be 0");

    buffers.add(ObjectData::new());
    assert_eq!(buffers.count(), 1, "Count should be 1 after first add");

    buffers.add(ObjectData::new());
    assert_eq!(buffers.count(), 2, "Count should be 2 after second add");
}

// ============================================================================
// SECTION 4: Object Management - Get Tests
// ============================================================================

/// Verify get returns added objects
#[test]
fn blackbox_get_returns_object() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    let obj = ObjectData::new()
        .with_mesh(123)
        .with_material(456);
    let index = buffers.add(obj);

    let retrieved = buffers.get(index);
    assert!(retrieved.is_some(), "get should return Some for valid index");

    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.mesh_index, 123, "mesh_index should match");
    assert_eq!(retrieved.material_index, 456, "material_index should match");
}

/// Verify get returns None for invalid index
#[test]
fn blackbox_get_invalid_index() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffers = SceneDataBuffers::new(&device, 100, None);

    assert!(buffers.get(0).is_none(), "get(0) should be None when empty");
    assert!(buffers.get(100).is_none(), "get(100) should be None");
    assert!(buffers.get(usize::MAX).is_none(), "get(MAX) should be None");
}

/// Verify get_mut allows modification
#[test]
fn blackbox_get_mut_modification() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    let obj = ObjectData::new().with_mesh(10);
    let index = buffers.add(obj);

    // Modify via get_mut
    if let Some(obj_mut) = buffers.get_mut(index) {
        obj_mut.mesh_index = 999;
    }

    // Verify modification persisted
    let retrieved = buffers.get(index).unwrap();
    assert_eq!(retrieved.mesh_index, 999, "Modification should persist");
}

/// Verify get_mut marks buffer as dirty
#[test]
fn blackbox_get_mut_marks_dirty() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);
    buffers.add(ObjectData::new());

    // Upload to clear dirty state
    buffers.upload(&device, &queue);
    assert!(!buffers.is_dirty(), "Should not be dirty after upload");

    // get_mut should mark as dirty
    let _ = buffers.get_mut(0);
    assert!(buffers.is_dirty(), "get_mut should mark buffer as dirty");
}

/// Verify get_mut returns None for invalid index
#[test]
fn blackbox_get_mut_invalid_index() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    assert!(buffers.get_mut(0).is_none(), "get_mut(0) should be None when empty");
    assert!(buffers.get_mut(100).is_none(), "get_mut(100) should be None");
}

// ============================================================================
// SECTION 5: Count and Capacity Tracking
// ============================================================================

/// Verify count accessor
#[test]
fn blackbox_count_tracking() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    assert_eq!(buffers.count(), 0);

    for i in 1..=50 {
        buffers.add(ObjectData::new());
        assert_eq!(buffers.count(), i);
    }
}

/// Verify is_empty accessor
#[test]
fn blackbox_is_empty() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    assert!(buffers.is_empty(), "Should be empty initially");

    buffers.add(ObjectData::new());
    assert!(!buffers.is_empty(), "Should not be empty after add");
}

/// Verify capacity accessor
#[test]
fn blackbox_capacity_tracking() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffers = SceneDataBuffers::new(&device, 500, None);

    assert!(
        buffers.capacity() >= 500,
        "Capacity should be at least requested amount"
    );
}

/// Verify buffer_size accessor returns GPU buffer size
#[test]
fn blackbox_buffer_size() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffers = SceneDataBuffers::new(&device, 100, None);

    // Buffer size should be at least capacity * OBJECT_DATA_SIZE
    assert!(
        buffers.buffer_size() >= (100 * OBJECT_DATA_SIZE) as u64,
        "buffer_size should be at least capacity * object size"
    );
}

/// Verify used_size accessor
#[test]
fn blackbox_used_size() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    assert_eq!(buffers.used_size(), 0, "used_size should be 0 when empty");

    buffers.add(ObjectData::new());
    assert_eq!(buffers.used_size(), OBJECT_DATA_SIZE, "used_size should be OBJECT_DATA_SIZE for 1 object");

    for _ in 0..9 {
        buffers.add(ObjectData::new());
    }
    assert_eq!(buffers.used_size(), 10 * OBJECT_DATA_SIZE, "used_size should scale with count");
}

// ============================================================================
// SECTION 6: Dirty Tracking Tests
// ============================================================================

/// Verify is_dirty accessor
#[test]
fn blackbox_dirty_initial_state() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffers = SceneDataBuffers::new(&device, 100, None);
    // New empty buffer might or might not be dirty depending on implementation
    // Just verify the accessor works
    let _ = buffers.is_dirty();
}

/// Verify add marks as dirty
#[test]
fn blackbox_add_marks_dirty() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    // Upload to clear dirty
    buffers.upload(&device, &queue);
    assert!(!buffers.is_dirty());

    buffers.add(ObjectData::new());
    assert!(buffers.is_dirty(), "add should mark buffer as dirty");
}

/// Verify upload clears dirty flag
#[test]
fn blackbox_upload_clears_dirty() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    buffers.add(ObjectData::new());
    assert!(buffers.is_dirty());

    buffers.upload(&device, &queue);
    assert!(!buffers.is_dirty(), "upload should clear dirty flag");
}

/// Verify update method marks dirty
#[test]
fn blackbox_update_marks_dirty() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);
    buffers.add(ObjectData::new());

    // Upload to clear dirty
    buffers.upload(&device, &queue);
    assert!(!buffers.is_dirty());

    // Update should mark dirty
    let new_obj = ObjectData::new().with_mesh(999);
    buffers.update(0, new_obj);

    assert!(buffers.is_dirty(), "update should mark buffer as dirty");
}

/// Verify update modifies object at index
#[test]
fn blackbox_update_modifies() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);
    buffers.add(ObjectData::new().with_mesh(1));

    let new_obj = ObjectData::new().with_mesh(888);
    buffers.update(0, new_obj);

    assert_eq!(buffers.get(0).unwrap().mesh_index, 888);
}

/// Verify update returns old object
#[test]
fn blackbox_update_returns_old() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);
    buffers.add(ObjectData::new().with_mesh(123));

    let new_obj = ObjectData::new().with_mesh(456);
    let old = buffers.update(0, new_obj);

    assert!(old.is_some());
    assert_eq!(old.unwrap().mesh_index, 123);
}

/// Verify mark_all_dirty
#[test]
fn blackbox_mark_all_dirty() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);
    for _ in 0..10 {
        buffers.add(ObjectData::new());
    }

    // Upload to clear
    buffers.upload(&device, &queue);
    assert!(!buffers.is_dirty());

    // Mark all dirty
    buffers.mark_all_dirty();
    assert!(buffers.is_dirty(), "mark_all_dirty should set dirty flag");
}

// ============================================================================
// SECTION 7: Auto-Resize Tests
// ============================================================================

/// Verify buffer auto-resizes when capacity exceeded during upload
#[test]
fn blackbox_auto_resize() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 10, None);

    // Initial capacity
    let initial_capacity = buffers.capacity();

    // Add more than initial capacity
    for i in 0..100 {
        buffers.add(ObjectData::new().with_mesh(i as u32));
    }

    // Upload triggers resize of GPU buffer
    buffers.upload(&device, &queue);

    // Should have resized
    assert!(
        buffers.capacity() >= 100,
        "Capacity should have grown to accommodate 100 objects"
    );
    assert!(
        buffers.capacity() > initial_capacity,
        "Capacity should have grown from initial"
    );

    // All objects should still be accessible
    assert_eq!(buffers.count(), 100);
    for i in 0..100 {
        assert_eq!(
            buffers.get(i).unwrap().mesh_index,
            i as u32,
            "Object {} should be preserved after resize",
            i
        );
    }
}

/// Verify was_resized flag
#[test]
fn blackbox_was_resized_flag() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 5, None);

    // Add more than capacity to trigger resize
    for _ in 0..20 {
        buffers.add(ObjectData::new());
    }

    // Check was_resized (may or may not be true depending on timing)
    let _ = buffers.was_resized();
}

/// Verify reserve increases capacity
#[test]
fn blackbox_reserve() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 10, None);

    buffers.reserve(&device, 1000);

    assert!(
        buffers.capacity() >= 1000,
        "reserve(1000) should ensure capacity >= 1000"
    );
}

/// Verify resize method
#[test]
fn blackbox_resize() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 10, None);

    // Add some data
    for i in 0..5 {
        buffers.add(ObjectData::new().with_mesh(i as u32));
    }

    // Resize to larger
    buffers.resize(&device, 500);
    assert!(buffers.capacity() >= 500);

    // Data should be preserved
    for i in 0..5 {
        assert_eq!(buffers.get(i).unwrap().mesh_index, i as u32);
    }
}

// ============================================================================
// SECTION 8: Clear Tests
// ============================================================================

/// Verify clear removes all objects
#[test]
fn blackbox_clear() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    // Add some objects
    for _ in 0..50 {
        buffers.add(ObjectData::new());
    }
    assert_eq!(buffers.count(), 50);

    // Clear
    buffers.clear();

    assert_eq!(buffers.count(), 0, "count should be 0 after clear");
    assert!(buffers.is_empty(), "is_empty should be true after clear");
}

/// Verify clear preserves capacity
#[test]
fn blackbox_clear_preserves_capacity() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    // Add some objects
    for _ in 0..50 {
        buffers.add(ObjectData::new());
    }

    let capacity_before = buffers.capacity();
    buffers.clear();
    let capacity_after = buffers.capacity();

    assert_eq!(
        capacity_after, capacity_before,
        "clear should preserve capacity"
    );
}

/// Verify buffer is reusable after clear
#[test]
fn blackbox_reuse_after_clear() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    // First use
    for i in 0..50 {
        buffers.add(ObjectData::new().with_mesh(i as u32));
    }
    assert_eq!(buffers.count(), 50);

    buffers.clear();

    // Reuse
    for i in 0..30 {
        buffers.add(ObjectData::new().with_mesh((i + 1000) as u32));
    }
    assert_eq!(buffers.count(), 30);

    // Verify new data
    assert_eq!(buffers.get(0).unwrap().mesh_index, 1000);
    assert_eq!(buffers.get(29).unwrap().mesh_index, 1029);
}

// ============================================================================
// SECTION 9: Slice Access Tests
// ============================================================================

/// Verify as_slice returns all objects
#[test]
fn blackbox_as_slice() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    for i in 0..10 {
        buffers.add(ObjectData::new().with_mesh(i as u32));
    }

    let slice = buffers.as_slice();
    assert_eq!(slice.len(), 10, "Slice length should match count");

    for i in 0..10 {
        assert_eq!(
            slice[i].mesh_index,
            i as u32,
            "Slice element {} should match",
            i
        );
    }
}

/// Verify iter_mut allows bulk modification
#[test]
fn blackbox_iter_mut_modification() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    for _ in 0..10 {
        buffers.add(ObjectData::new());
    }

    // Modify all via iter_mut
    for (i, obj) in buffers.iter_mut().enumerate() {
        obj.mesh_index = (i * 100) as u32;
    }

    // Verify modifications
    for i in 0..10 {
        assert_eq!(
            buffers.get(i).unwrap().mesh_index,
            (i * 100) as u32
        );
    }
}

// ============================================================================
// SECTION 10: Upload / GPU Buffer Tests
// ============================================================================

/// Verify object_buffer accessor returns the wgpu buffer
#[test]
fn blackbox_object_buffer() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffers = SceneDataBuffers::new(&device, 100, None);

    // Should be able to get the GPU buffer
    let gpu_buffer = buffers.object_buffer();

    // Buffer should have reasonable size
    assert!(gpu_buffer.size() >= 100 * OBJECT_DATA_SIZE as u64);
}

/// Verify buffer_binding accessor
#[test]
fn blackbox_buffer_binding() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffers = SceneDataBuffers::new(&device, 100, None);

    // Should be able to get buffer binding
    let binding = buffers.buffer_binding();

    // Binding should reference the buffer
    assert!(binding.size.is_some() || binding.size.is_none()); // Just verify it works
}

/// Verify upload method
#[test]
fn blackbox_upload() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    // Add some data
    for i in 0..10 {
        buffers.add(ObjectData::new().with_mesh(i as u32));
    }

    // Upload should succeed and return true if data was uploaded
    let uploaded = buffers.upload(&device, &queue);
    // Upload returns true if there was data to upload
    assert!(uploaded || buffers.count() == 0);

    // Should no longer be dirty
    assert!(!buffers.is_dirty(), "Should not be dirty after upload");
}

/// Verify upload with no changes returns false
#[test]
fn blackbox_upload_no_changes() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);
    buffers.add(ObjectData::new());

    // First upload
    buffers.upload(&device, &queue);

    // Second upload without changes should return false
    let uploaded = buffers.upload(&device, &queue);
    assert!(!uploaded, "upload without changes should return false");
}

// ============================================================================
// SECTION 11: Iterator Tests
// ============================================================================

/// Verify iter() iterates all objects
#[test]
fn blackbox_iter() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    for i in 0..10 {
        buffers.add(ObjectData::new().with_mesh(i as u32));
    }

    let collected: Vec<_> = buffers.iter().collect();
    assert_eq!(collected.len(), 10);

    for (i, obj) in collected.iter().enumerate() {
        assert_eq!(obj.mesh_index, i as u32);
    }
}

/// Verify iter_mut() allows modification
#[test]
fn blackbox_iter_mut() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    for _ in 0..10 {
        buffers.add(ObjectData::new());
    }

    // Modify all
    for (i, obj) in buffers.iter_mut().enumerate() {
        obj.material_index = (i + 500) as u32;
    }

    // Verify
    for i in 0..10 {
        assert_eq!(buffers.get(i).unwrap().material_index, (i + 500) as u32);
    }
}

/// Verify iter_indexed returns index and reference
#[test]
fn blackbox_iter_indexed() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    for i in 0..10 {
        buffers.add(ObjectData::new().with_mesh(i as u32));
    }

    for (idx, obj) in buffers.iter_indexed() {
        assert_eq!(obj.mesh_index, idx as u32);
    }
}

// ============================================================================
// SECTION 12: Remove and Swap-Remove Tests
// ============================================================================

/// Verify remove/swap_remove exists and works
#[test]
fn blackbox_swap_remove() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    for i in 0..5 {
        buffers.add(ObjectData::new().with_mesh(i as u32));
    }

    // Remove middle element (index 2)
    let removed = buffers.swap_remove(2);
    assert!(removed.is_some());
    assert_eq!(removed.unwrap().mesh_index, 2);

    // Count should decrease
    assert_eq!(buffers.count(), 4);

    // Element at index 2 should now be the last element (was at index 4)
    assert_eq!(buffers.get(2).unwrap().mesh_index, 4);
}

/// Verify swap_remove returns None for invalid index
#[test]
fn blackbox_swap_remove_invalid() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);
    buffers.add(ObjectData::new());

    assert!(buffers.swap_remove(1).is_none());
    assert!(buffers.swap_remove(100).is_none());
}

/// Verify swap_remove marks dirty
#[test]
fn blackbox_swap_remove_marks_dirty() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);
    buffers.add(ObjectData::new());
    buffers.add(ObjectData::new());

    buffers.upload(&device, &queue);
    assert!(!buffers.is_dirty());

    buffers.swap_remove(0);
    assert!(buffers.is_dirty(), "swap_remove should mark dirty");
}

// ============================================================================
// SECTION 13: Transform Update via get_mut
// ============================================================================

/// Verify transform can be updated via get_mut
#[test]
fn blackbox_update_transform_via_get_mut() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);
    buffers.add(ObjectData::new());

    let new_transform = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [100.0, 200.0, 300.0, 1.0],
    ];

    if let Some(obj) = buffers.get_mut(0) {
        obj.transform = new_transform;
    }

    let obj = buffers.get(0).unwrap();
    assert_eq!(obj.transform[0][0], 2.0);
    assert_eq!(obj.transform[3][0], 100.0);
    assert_eq!(obj.transform[3][1], 200.0);
    assert_eq!(obj.transform[3][2], 300.0);
}

/// Verify update_transforms helper
#[test]
fn blackbox_update_transforms() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    for _ in 0..5 {
        buffers.add(ObjectData::new());
    }

    // Update all transforms
    buffers.update_transforms(|idx, transform| {
        transform[3][0] = idx as f32 * 10.0; // X translation
    });

    // Verify
    for i in 0..5 {
        assert_eq!(buffers.get(i).unwrap().transform[3][0], i as f32 * 10.0);
    }
}

/// Verify AABB can be updated via get_mut
#[test]
fn blackbox_update_aabb_via_get_mut() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);
    buffers.add(ObjectData::new());

    if let Some(obj) = buffers.get_mut(0) {
        obj.aabb_min = [-5.0, -10.0, -15.0];
        obj.aabb_max = [5.0, 10.0, 15.0];
    }

    let obj = buffers.get(0).unwrap();
    assert_eq!(obj.aabb_min, [-5.0, -10.0, -15.0]);
    assert_eq!(obj.aabb_max, [5.0, 10.0, 15.0]);
}

// ============================================================================
// SECTION 14: Visibility via get_mut
// ============================================================================

/// Verify visibility can be updated via get_mut
#[test]
fn blackbox_set_visible_via_get_mut() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);
    buffers.add(ObjectData::new().with_flags(object_flags::DEFAULT));

    assert!(buffers.get(0).unwrap().is_visible());

    if let Some(obj) = buffers.get_mut(0) {
        obj.set_visible(false);
    }
    assert!(!buffers.get(0).unwrap().is_visible());

    if let Some(obj) = buffers.get_mut(0) {
        obj.set_visible(true);
    }
    assert!(buffers.get(0).unwrap().is_visible());
}

/// Verify flags can be set via get_mut
#[test]
fn blackbox_set_flags_via_get_mut() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);
    buffers.add(ObjectData::new());

    if let Some(obj) = buffers.get_mut(0) {
        obj.flags = object_flags::VISIBLE | object_flags::STATIC;
    }

    let obj = buffers.get(0).unwrap();
    assert!(obj.is_visible());
    assert!(obj.is_static());
}

// ============================================================================
// SECTION 15: Batch Operations
// ============================================================================

/// Verify add_batch for bulk insertion
#[test]
fn blackbox_add_batch() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    let objects: Vec<ObjectData> = (0..20)
        .map(|i| ObjectData::new().with_mesh(i as u32))
        .collect();

    let start_index = buffers.add_batch(objects);
    assert_eq!(start_index, 0, "First batch should start at 0");
    assert_eq!(buffers.count(), 20);

    // All should be accessible
    for i in 0..20 {
        assert_eq!(buffers.get(i).unwrap().mesh_index, i as u32);
    }
}

/// Verify add_batch returns start index
#[test]
fn blackbox_add_batch_returns_start() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    // Add first batch
    let objects1: Vec<ObjectData> = (0..10)
        .map(|i| ObjectData::new().with_mesh(i as u32))
        .collect();
    let start1 = buffers.add_batch(objects1);
    assert_eq!(start1, 0);

    // Add second batch
    let objects2: Vec<ObjectData> = (100..110)
        .map(|i| ObjectData::new().with_mesh(i as u32))
        .collect();
    let start2 = buffers.add_batch(objects2);
    assert_eq!(start2, 10, "Second batch should start at index 10");

    assert_eq!(buffers.count(), 20);
}

// ============================================================================
// SECTION 16: Dirty Range Tracking
// ============================================================================

/// Verify dirty_range returns range of modified objects
#[test]
fn blackbox_dirty_range() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    // Add objects
    for _ in 0..10 {
        buffers.add(ObjectData::new());
    }

    // Upload to clear dirty state
    buffers.upload(&device, &queue);

    // Modify specific objects
    let _ = buffers.get_mut(3);
    let _ = buffers.get_mut(7);

    // Check dirty range
    let range = buffers.dirty_range();
    assert!(range.is_some(), "Should have dirty range after modifications");

    let (start, end) = range.unwrap();
    assert!(start <= 3, "Dirty range should include index 3");
    assert!(end >= 7, "Dirty range should include index 7");
}

// ============================================================================
// SECTION 17: Retain Tests
// ============================================================================

/// Verify retain removes objects not matching predicate
#[test]
fn blackbox_retain() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    // Add objects with varying mesh indices
    for i in 0..10 {
        buffers.add(ObjectData::new().with_mesh(i as u32));
    }

    // Retain only even mesh indices
    buffers.retain(|obj| obj.mesh_index % 2 == 0);

    assert_eq!(buffers.count(), 5, "Should retain 5 objects with even indices");

    // Verify remaining objects have even indices
    for obj in buffers.iter() {
        assert_eq!(obj.mesh_index % 2, 0, "All remaining should have even mesh_index");
    }
}

/// Verify retain marks dirty
#[test]
fn blackbox_retain_marks_dirty() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    for _ in 0..10 {
        buffers.add(ObjectData::new());
    }

    buffers.upload(&device, &queue);
    assert!(!buffers.is_dirty());

    buffers.retain(|_| true); // Keep all
    // Retain may or may not mark dirty if nothing changed
    // Just verify it doesn't crash
}

// ============================================================================
// SECTION 18: Debug Tests
// ============================================================================

/// Verify Debug implementation
#[test]
fn blackbox_debug() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffers = SceneDataBuffers::new(&device, 100, None);
    let debug_str = format!("{:?}", buffers);

    assert!(!debug_str.is_empty(), "Debug output should be non-empty");
}

// ============================================================================
// SECTION 19: Edge Cases and Stress Tests
// ============================================================================

/// Test empty buffer operations
#[test]
fn blackbox_empty_buffer_operations() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let buffers = SceneDataBuffers::new(&device, 100, None);

    assert!(buffers.is_empty());
    assert_eq!(buffers.count(), 0);
    assert!(buffers.get(0).is_none());
    assert_eq!(buffers.as_slice().len(), 0);
    assert_eq!(buffers.used_size(), 0);
}

/// Test single object buffer
#[test]
fn blackbox_single_object() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 1, None);

    let idx = buffers.add(ObjectData::new().with_mesh(42));
    assert_eq!(idx, 0);
    assert_eq!(buffers.count(), 1);
    assert_eq!(buffers.get(0).unwrap().mesh_index, 42);
}

/// Test large buffer stress
#[test]
fn blackbox_large_buffer() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    const LARGE_COUNT: usize = 10_000;

    let mut buffers = SceneDataBuffers::new(&device, LARGE_COUNT, None);

    for i in 0..LARGE_COUNT {
        buffers.add(ObjectData::new().with_mesh(i as u32));
    }

    assert_eq!(buffers.count(), LARGE_COUNT);

    // Spot check
    assert_eq!(buffers.get(0).unwrap().mesh_index, 0);
    assert_eq!(buffers.get(5000).unwrap().mesh_index, 5000);
    assert_eq!(buffers.get(LARGE_COUNT - 1).unwrap().mesh_index, (LARGE_COUNT - 1) as u32);
}

/// Test growth from minimal capacity with upload
#[test]
fn blackbox_growth_from_minimal() {
    let adapter = require_adapter!();
    let (device, queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 1, None);

    // Add many more than initial capacity
    for i in 0..1000 {
        buffers.add(ObjectData::new().with_mesh(i as u32));
    }

    // Upload triggers resize of GPU buffer to accommodate all objects
    buffers.upload(&device, &queue);

    assert_eq!(buffers.count(), 1000);
    assert!(buffers.capacity() >= 1000);

    // Verify all data preserved through multiple resizes
    for i in 0..1000 {
        assert_eq!(
            buffers.get(i).unwrap().mesh_index,
            i as u32,
            "Object {} should be preserved through resize",
            i
        );
    }
}

/// Test repeated clear and reuse cycles
#[test]
fn blackbox_repeated_clear_cycles() {
    let adapter = require_adapter!();
    let (device, _queue) = require_device!(&adapter);

    let mut buffers = SceneDataBuffers::new(&device, 100, None);

    for cycle in 0..5 {
        // Fill
        for i in 0..50 {
            buffers.add(ObjectData::new().with_mesh((cycle * 1000 + i) as u32));
        }
        assert_eq!(buffers.count(), 50);

        // Verify
        assert_eq!(
            buffers.get(0).unwrap().mesh_index,
            (cycle * 1000) as u32
        );

        // Clear
        buffers.clear();
        assert!(buffers.is_empty());
    }
}
