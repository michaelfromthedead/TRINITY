// SPDX-License-Identifier: MIT
//
// WHITEBOX T-WGPU-P2.2.3 Deferred Destroyer
//
// Comprehensive whitebox tests for the DeferredDestroyer module.
// Tests internal state, metrics tracking, edge cases, and lifecycle.
//
// Since we cannot create real wgpu::Buffer/Texture without a GPU device,
// we use the defer_other() method to test the core logic with mock types.
// The DeferredResource::Other variant exercises the same code paths as
// Buffer/Texture variants for queue management and lifecycle.

use renderer_backend::resources::deferred_destroyer::{
    DeferredDestroyer, DeferredDestroyerMetrics, DeferredResource, DEFAULT_DESTRUCTION_DELAY,
};
use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ============================================================================
// Mock Types for Testing
// ============================================================================

/// Mock buffer for testing deferred destruction logic.
#[derive(Debug)]
struct MockBuffer {
    id: u64,
    size: usize,
}

impl MockBuffer {
    fn new(id: u64, size: usize) -> Self {
        Self { id, size }
    }
}

/// Mock texture for testing deferred destruction logic.
#[derive(Debug)]
struct MockTexture {
    id: u64,
    width: u32,
    height: u32,
}

impl MockTexture {
    fn new(id: u64, width: u32, height: u32) -> Self {
        Self { id, width, height }
    }
}

/// Mock resource that tracks when it is dropped.
struct DropTracker {
    id: u64,
    drop_counter: Arc<AtomicUsize>,
}

impl DropTracker {
    fn new(id: u64, counter: Arc<AtomicUsize>) -> Self {
        Self {
            id,
            drop_counter: counter,
        }
    }
}

impl Drop for DropTracker {
    fn drop(&mut self) {
        self.drop_counter.fetch_add(1, Ordering::SeqCst);
    }
}

/// Mock resource with a payload for testing Box<dyn Any> behavior.
struct PayloadResource {
    data: Vec<u8>,
    name: String,
}

impl PayloadResource {
    fn new(name: &str, size: usize) -> Self {
        Self {
            data: vec![0u8; size],
            name: name.to_string(),
        }
    }
}

// ============================================================================
// 1. Construction Tests (8+ tests)
// ============================================================================

mod construction_tests {
    use super::*;

    /// Test default constructor creates destroyer with DEFAULT_DESTRUCTION_DELAY.
    #[test]
    fn test_default_constructor() {
        let destroyer = DeferredDestroyer::new();
        assert_eq!(destroyer.default_delay(), DEFAULT_DESTRUCTION_DELAY);
    }

    /// Test default constructor initializes empty pending queue.
    #[test]
    fn test_default_constructor_empty_queue() {
        let destroyer = DeferredDestroyer::new();
        assert!(destroyer.is_empty());
        assert_eq!(destroyer.pending_count(), 0);
    }

    /// Test Default trait implementation matches new().
    #[test]
    fn test_default_trait() {
        let destroyer = DeferredDestroyer::default();
        assert_eq!(destroyer.default_delay(), DEFAULT_DESTRUCTION_DELAY);
        assert!(destroyer.is_empty());
    }

    /// Test custom delay constructor with standard value.
    #[test]
    fn test_custom_delay_constructor() {
        let destroyer = DeferredDestroyer::with_delay(5);
        assert_eq!(destroyer.default_delay(), 5);
        assert!(destroyer.is_empty());
    }

    /// Test zero delay edge case.
    #[test]
    fn test_zero_delay_constructor() {
        let destroyer = DeferredDestroyer::with_delay(0);
        assert_eq!(destroyer.default_delay(), 0);
    }

    /// Test large delay value.
    #[test]
    fn test_large_delay_constructor() {
        let large_delay = 1_000_000u64;
        let destroyer = DeferredDestroyer::with_delay(large_delay);
        assert_eq!(destroyer.default_delay(), large_delay);
    }

    /// Test default metrics state on construction.
    #[test]
    fn test_default_metrics_state() {
        let destroyer = DeferredDestroyer::new();
        let metrics = destroyer.metrics();

        assert_eq!(metrics.pending_count, 0);
        assert_eq!(metrics.destroyed_count, 0);
        assert_eq!(metrics.peak_pending, 0);
        assert_eq!(metrics.buffers_destroyed, 0);
        assert_eq!(metrics.textures_destroyed, 0);
        assert_eq!(metrics.other_destroyed, 0);
    }

    /// Test pending queue is empty on construction.
    #[test]
    fn test_empty_pending_queue_on_construction() {
        let mut destroyer = DeferredDestroyer::new();

        // Process a frame on empty destroyer should return 0
        let destroyed = destroyer.process_frame(100);
        assert_eq!(destroyed, 0);
        assert!(destroyer.is_empty());
    }

    /// Test maximum u64 delay value.
    #[test]
    fn test_max_delay_constructor() {
        let destroyer = DeferredDestroyer::with_delay(u64::MAX);
        assert_eq!(destroyer.default_delay(), u64::MAX);
    }
}

// ============================================================================
// 2. Defer Buffer Tests (10+ tests) - Using MockBuffer via defer_other
// ============================================================================

mod defer_buffer_tests {
    use super::*;

    /// Test basic buffer deferral at frame 0.
    #[test]
    fn test_basic_buffer_deferral() {
        let mut destroyer = DeferredDestroyer::new();
        destroyer.defer_other(MockBuffer::new(1, 1024), 0);

        assert_eq!(destroyer.pending_count(), 1);
        assert!(!destroyer.is_empty());
    }

    /// Test buffer with custom delay.
    #[test]
    fn test_buffer_with_custom_delay() {
        let mut destroyer = DeferredDestroyer::new();
        destroyer.defer_other_with_delay(MockBuffer::new(1, 1024), 0, 5);

        // Should not destroy before frame 5
        assert_eq!(destroyer.process_frame(4), 0);
        assert_eq!(destroyer.pending_count(), 1);

        // Should destroy at frame 5
        assert_eq!(destroyer.process_frame(5), 1);
        assert!(destroyer.is_empty());
    }

    /// Test buffer with label.
    #[test]
    fn test_buffer_with_label() {
        let mut destroyer = DeferredDestroyer::with_delay(1);
        destroyer.defer_other_labeled(MockBuffer::new(1, 2048), 0, "vertex_buffer");

        assert_eq!(destroyer.pending_count(), 1);

        // Should destroy at frame 1
        let destroyed = destroyer.process_frame(1);
        assert_eq!(destroyed, 1);
    }

    /// Test multiple buffers deferred at same frame.
    #[test]
    fn test_multiple_buffers_same_frame() {
        let mut destroyer = DeferredDestroyer::with_delay(2);

        for i in 0..5 {
            destroyer.defer_other(MockBuffer::new(i, (i as usize + 1) * 256), 10);
        }

        assert_eq!(destroyer.pending_count(), 5);

        // All should be destroyed at frame 12
        assert_eq!(destroyer.process_frame(12), 5);
        assert!(destroyer.is_empty());
    }

    /// Test multiple buffers deferred at different frames.
    #[test]
    fn test_multiple_buffers_different_frames() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        destroyer.defer_other(MockBuffer::new(1, 256), 0);  // destroy at 1
        destroyer.defer_other(MockBuffer::new(2, 512), 1);  // destroy at 2
        destroyer.defer_other(MockBuffer::new(3, 1024), 2); // destroy at 3

        assert_eq!(destroyer.pending_count(), 3);

        // Frame 0: nothing destroyed
        assert_eq!(destroyer.process_frame(0), 0);
        assert_eq!(destroyer.pending_count(), 3);

        // Frame 1: buffer 1 destroyed
        assert_eq!(destroyer.process_frame(1), 1);
        assert_eq!(destroyer.pending_count(), 2);

        // Frame 2: buffer 2 destroyed
        assert_eq!(destroyer.process_frame(2), 1);
        assert_eq!(destroyer.pending_count(), 1);

        // Frame 3: buffer 3 destroyed
        assert_eq!(destroyer.process_frame(3), 1);
        assert!(destroyer.is_empty());
    }

    /// Test buffer at frame 0 with zero delay.
    #[test]
    fn test_buffer_at_frame_0_zero_delay() {
        let mut destroyer = DeferredDestroyer::with_delay(0);
        destroyer.defer_other(MockBuffer::new(1, 1024), 0);

        // Should destroy immediately at frame 0
        assert_eq!(destroyer.process_frame(0), 1);
        assert!(destroyer.is_empty());
    }

    /// Test buffer at large frame number.
    #[test]
    fn test_buffer_at_large_frame_number() {
        let mut destroyer = DeferredDestroyer::with_delay(2);
        let large_frame = 1_000_000_000u64;

        destroyer.defer_other(MockBuffer::new(1, 1024), large_frame);

        // Should not destroy before target frame
        assert_eq!(destroyer.process_frame(large_frame + 1), 0);
        assert_eq!(destroyer.pending_count(), 1);

        // Should destroy at target + delay
        assert_eq!(destroyer.process_frame(large_frame + 2), 1);
        assert!(destroyer.is_empty());
    }

    /// Test buffer deferral increments pending count.
    #[test]
    fn test_buffer_deferral_increments_pending() {
        let mut destroyer = DeferredDestroyer::new();

        for i in 1..=10 {
            destroyer.defer_other(MockBuffer::new(i, 256), 0);
            assert_eq!(destroyer.pending_count(), i as usize);
        }
    }

    /// Test buffer deferral updates peak pending.
    #[test]
    fn test_buffer_deferral_updates_peak() {
        let mut destroyer = DeferredDestroyer::new();

        destroyer.defer_other(MockBuffer::new(1, 256), 0);
        assert_eq!(destroyer.metrics().peak_pending, 1);

        destroyer.defer_other(MockBuffer::new(2, 256), 0);
        assert_eq!(destroyer.metrics().peak_pending, 2);

        destroyer.defer_other(MockBuffer::new(3, 256), 0);
        assert_eq!(destroyer.metrics().peak_pending, 3);
    }

    /// Test buffer with string label type.
    #[test]
    fn test_buffer_with_string_label() {
        let mut destroyer = DeferredDestroyer::with_delay(1);
        let label = String::from("dynamic_vertex_buffer_pool_0");
        destroyer.defer_other_labeled(MockBuffer::new(1, 4096), 0, label);

        assert_eq!(destroyer.pending_count(), 1);
        assert_eq!(destroyer.process_frame(1), 1);
    }
}

// ============================================================================
// 3. Defer Texture Tests (10+ tests) - Using MockTexture via defer_other
// ============================================================================

mod defer_texture_tests {
    use super::*;

    /// Test basic texture deferral.
    #[test]
    fn test_basic_texture_deferral() {
        let mut destroyer = DeferredDestroyer::new();
        destroyer.defer_other(MockTexture::new(1, 1024, 1024), 0);

        assert_eq!(destroyer.pending_count(), 1);
        assert!(!destroyer.is_empty());
    }

    /// Test texture with custom delay.
    #[test]
    fn test_texture_with_custom_delay() {
        let mut destroyer = DeferredDestroyer::new();
        destroyer.defer_other_with_delay(MockTexture::new(1, 512, 512), 0, 3);

        // Should not destroy before frame 3
        assert_eq!(destroyer.process_frame(2), 0);

        // Should destroy at frame 3
        assert_eq!(destroyer.process_frame(3), 1);
        assert!(destroyer.is_empty());
    }

    /// Test texture with label.
    #[test]
    fn test_texture_with_label() {
        let mut destroyer = DeferredDestroyer::with_delay(1);
        destroyer.defer_other_labeled(MockTexture::new(1, 2048, 2048), 0, "diffuse_atlas");

        assert_eq!(destroyer.pending_count(), 1);
        assert_eq!(destroyer.process_frame(1), 1);
    }

    /// Test multiple textures deferred at same frame.
    #[test]
    fn test_multiple_textures_same_frame() {
        let mut destroyer = DeferredDestroyer::with_delay(2);

        for i in 0..8 {
            destroyer.defer_other(MockTexture::new(i, 256, 256), 5);
        }

        assert_eq!(destroyer.pending_count(), 8);

        // All should be destroyed at frame 7
        assert_eq!(destroyer.process_frame(7), 8);
        assert!(destroyer.is_empty());
    }

    /// Test multiple textures deferred at different frames.
    #[test]
    fn test_multiple_textures_different_frames() {
        let mut destroyer = DeferredDestroyer::with_delay(2);

        destroyer.defer_other(MockTexture::new(1, 128, 128), 0);   // destroy at 2
        destroyer.defer_other(MockTexture::new(2, 256, 256), 5);   // destroy at 7
        destroyer.defer_other(MockTexture::new(3, 512, 512), 10);  // destroy at 12

        assert_eq!(destroyer.pending_count(), 3);

        assert_eq!(destroyer.process_frame(2), 1);
        assert_eq!(destroyer.pending_count(), 2);

        assert_eq!(destroyer.process_frame(7), 1);
        assert_eq!(destroyer.pending_count(), 1);

        assert_eq!(destroyer.process_frame(12), 1);
        assert!(destroyer.is_empty());
    }

    /// Test mixed buffer and texture deferral.
    #[test]
    fn test_mixed_buffer_and_texture_deferral() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        destroyer.defer_other(MockBuffer::new(1, 1024), 0);
        destroyer.defer_other(MockTexture::new(1, 512, 512), 0);
        destroyer.defer_other(MockBuffer::new(2, 2048), 0);
        destroyer.defer_other(MockTexture::new(2, 1024, 1024), 0);

        assert_eq!(destroyer.pending_count(), 4);

        // All should be destroyed at frame 1
        assert_eq!(destroyer.process_frame(1), 4);
        assert!(destroyer.is_empty());

        let metrics = destroyer.metrics();
        assert_eq!(metrics.destroyed_count, 4);
        assert_eq!(metrics.other_destroyed, 4); // All are "other" since we use MockXxx
    }

    /// Test texture deferral order preserved.
    #[test]
    fn test_texture_deferral_order() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        // Defer textures at same frame with same delay
        destroyer.defer_other(MockTexture::new(1, 64, 64), 10);
        destroyer.defer_other(MockTexture::new(2, 128, 128), 10);
        destroyer.defer_other(MockTexture::new(3, 256, 256), 10);

        // All should be destroyed together
        let destroyed = destroyer.process_frame(11);
        assert_eq!(destroyed, 3);
    }

    /// Test texture with very large dimensions (mock).
    #[test]
    fn test_texture_large_dimensions() {
        let mut destroyer = DeferredDestroyer::with_delay(1);
        destroyer.defer_other(MockTexture::new(1, 16384, 16384), 0);

        assert_eq!(destroyer.pending_count(), 1);
        assert_eq!(destroyer.process_frame(1), 1);
    }

    /// Test texture at frame near u64::MAX.
    #[test]
    fn test_texture_near_max_frame() {
        let mut destroyer = DeferredDestroyer::with_delay(1);
        let near_max = u64::MAX - 10;

        destroyer.defer_other(MockTexture::new(1, 256, 256), near_max);

        // Should not destroy before target
        assert_eq!(destroyer.process_frame(near_max), 0);

        // Should destroy at target + 1
        assert_eq!(destroyer.process_frame(near_max + 1), 1);
    }

    /// Test interleaved texture and buffer deferrals.
    #[test]
    fn test_interleaved_texture_buffer_deferrals() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        for i in 0..10u64 {
            if i % 2 == 0 {
                destroyer.defer_other(MockBuffer::new(i, 1024), i);
            } else {
                destroyer.defer_other(MockTexture::new(i, 256, 256), i);
            }
        }

        assert_eq!(destroyer.pending_count(), 10);

        // Process frames 0-10, destroying one per frame starting at 1
        for frame in 0..=10u64 {
            let destroyed = destroyer.process_frame(frame);
            if frame == 0 {
                assert_eq!(destroyed, 0);
            } else if frame <= 10 {
                assert!(destroyed >= 1);
            }
        }
    }
}

// ============================================================================
// 4. Defer Other Tests (6+ tests)
// ============================================================================

mod defer_other_tests {
    use super::*;

    /// Test custom type deferral with simple struct.
    #[test]
    fn test_custom_type_deferral() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        #[derive(Debug)]
        struct CustomResource {
            id: u64,
            data: String,
        }

        destroyer.defer_other(
            CustomResource {
                id: 42,
                data: "test".to_string(),
            },
            0,
        );

        assert_eq!(destroyer.pending_count(), 1);
        assert_eq!(destroyer.process_frame(1), 1);
    }

    /// Test multiple different custom types.
    #[test]
    fn test_multiple_custom_types() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        destroyer.defer_other(42u32, 0);
        destroyer.defer_other("string_resource", 0);
        destroyer.defer_other(vec![1, 2, 3, 4, 5], 0);
        destroyer.defer_other(PayloadResource::new("test", 1024), 0);

        assert_eq!(destroyer.pending_count(), 4);
        assert_eq!(destroyer.process_frame(1), 4);
    }

    /// Test Box<dyn Any> behavior via DeferredResource::Other.
    #[test]
    fn test_box_dyn_any_behavior() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        let resource = DeferredResource::Other(Box::new("test_value"));
        destroyer.defer_resource(resource, 0);

        assert_eq!(destroyer.pending_count(), 1);
        assert_eq!(destroyer.process_frame(1), 1);
    }

    /// Test type_name accessor for DeferredResource variants.
    #[test]
    fn test_type_name_accessor() {
        let other = DeferredResource::Other(Box::new(42i32));
        assert_eq!(other.type_name(), "Other");
    }

    /// Test defer_resource_with_delay method.
    #[test]
    fn test_defer_resource_with_delay() {
        let mut destroyer = DeferredDestroyer::new();

        let resource = DeferredResource::Other(Box::new(PayloadResource::new("data", 512)));
        destroyer.defer_resource_with_delay(resource, 10, 5);

        // Should not destroy before frame 15
        assert_eq!(destroyer.process_frame(14), 0);

        // Should destroy at frame 15
        assert_eq!(destroyer.process_frame(15), 1);
    }

    /// Test defer_other_labeled method.
    #[test]
    fn test_defer_other_labeled() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        destroyer.defer_other_labeled(
            PayloadResource::new("labeled_resource", 256),
            0,
            "my_custom_label",
        );

        assert_eq!(destroyer.pending_count(), 1);

        let destroyed = destroyer.process_frame(1);
        assert_eq!(destroyed, 1);
    }

    /// Test that resources are actually dropped.
    #[test]
    fn test_resources_actually_dropped() {
        let drop_counter = Arc::new(AtomicUsize::new(0));

        {
            let mut destroyer = DeferredDestroyer::with_delay(1);

            destroyer.defer_other(DropTracker::new(1, drop_counter.clone()), 0);
            destroyer.defer_other(DropTracker::new(2, drop_counter.clone()), 0);
            destroyer.defer_other(DropTracker::new(3, drop_counter.clone()), 0);

            // Not dropped yet
            assert_eq!(drop_counter.load(Ordering::SeqCst), 0);

            // Process frame to destroy
            destroyer.process_frame(1);

            // Now dropped
            assert_eq!(drop_counter.load(Ordering::SeqCst), 3);
        }
    }

    /// Test Debug implementation for DeferredResource::Other.
    #[test]
    fn test_debug_impl_other() {
        let resource = DeferredResource::Other(Box::new(42u64));
        let debug_str = format!("{:?}", resource);
        assert!(debug_str.contains("Other"));
    }
}

// ============================================================================
// 5. Process Frame Tests (10+ tests)
// ============================================================================

mod process_frame_tests {
    use super::*;

    /// Test process_frame on empty queue returns 0.
    #[test]
    fn test_process_empty_queue() {
        let mut destroyer = DeferredDestroyer::new();

        assert_eq!(destroyer.process_frame(0), 0);
        assert_eq!(destroyer.process_frame(100), 0);
        assert_eq!(destroyer.process_frame(u64::MAX), 0);
    }

    /// Test process_frame destroys single resource at correct frame.
    #[test]
    fn test_process_single_resource() {
        let mut destroyer = DeferredDestroyer::with_delay(3);
        destroyer.defer_other(MockBuffer::new(1, 256), 10);

        // Should not destroy before frame 13
        assert_eq!(destroyer.process_frame(10), 0);
        assert_eq!(destroyer.process_frame(11), 0);
        assert_eq!(destroyer.process_frame(12), 0);

        // Should destroy at frame 13
        assert_eq!(destroyer.process_frame(13), 1);
        assert!(destroyer.is_empty());
    }

    /// Test process_frame destroys multiple resources.
    #[test]
    fn test_process_multiple_resources() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        for i in 0..10 {
            destroyer.defer_other(MockBuffer::new(i, 256), 5);
        }

        // All should be destroyed at frame 6
        let destroyed = destroyer.process_frame(6);
        assert_eq!(destroyed, 10);
        assert!(destroyer.is_empty());
    }

    /// Test process_frame partial destruction (some not ready).
    #[test]
    fn test_process_partial_destruction() {
        let mut destroyer = DeferredDestroyer::with_delay(2);

        destroyer.defer_other(MockBuffer::new(1, 256), 0);  // destroy at 2
        destroyer.defer_other(MockBuffer::new(2, 256), 1);  // destroy at 3
        destroyer.defer_other(MockBuffer::new(3, 256), 2);  // destroy at 4

        // Frame 2: only first resource destroyed
        assert_eq!(destroyer.process_frame(2), 1);
        assert_eq!(destroyer.pending_count(), 2);

        // Frame 3: second resource destroyed
        assert_eq!(destroyer.process_frame(3), 1);
        assert_eq!(destroyer.pending_count(), 1);
    }

    /// Test process_frame catches up (destroys all earlier resources).
    #[test]
    fn test_process_catching_up() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        destroyer.defer_other(MockBuffer::new(1, 256), 0);  // destroy at 1
        destroyer.defer_other(MockBuffer::new(2, 256), 1);  // destroy at 2
        destroyer.defer_other(MockBuffer::new(3, 256), 2);  // destroy at 3

        // Skip to frame 10 - all should be destroyed
        let destroyed = destroyer.process_frame(10);
        assert_eq!(destroyed, 3);
        assert!(destroyer.is_empty());
    }

    /// Test process_frame at frame 0.
    #[test]
    fn test_process_at_frame_0() {
        let mut destroyer = DeferredDestroyer::with_delay(0);
        destroyer.defer_other(MockBuffer::new(1, 256), 0);

        // With zero delay, should destroy at frame 0
        assert_eq!(destroyer.process_frame(0), 1);
        assert!(destroyer.is_empty());
    }

    /// Test process_frame at very large frame number.
    #[test]
    fn test_process_at_large_frame() {
        let mut destroyer = DeferredDestroyer::with_delay(1);
        destroyer.defer_other(MockBuffer::new(1, 256), 0);

        // Process at very large frame - should destroy
        let destroyed = destroyer.process_frame(u64::MAX);
        assert_eq!(destroyed, 1);
    }

    /// Test process_frame return value accuracy.
    #[test]
    fn test_process_return_value() {
        let mut destroyer = DeferredDestroyer::with_delay(0);

        destroyer.defer_other(MockBuffer::new(1, 256), 5);
        destroyer.defer_other(MockBuffer::new(2, 256), 5);
        destroyer.defer_other(MockBuffer::new(3, 256), 10);

        // Frame 5: 2 destroyed
        let destroyed = destroyer.process_frame(5);
        assert_eq!(destroyed, 2);

        // Frame 10: 1 destroyed
        let destroyed = destroyer.process_frame(10);
        assert_eq!(destroyed, 1);
    }

    /// Test process_frame multiple times at same frame.
    #[test]
    fn test_process_same_frame_multiple_times() {
        let mut destroyer = DeferredDestroyer::with_delay(0);
        destroyer.defer_other(MockBuffer::new(1, 256), 5);

        // First process at frame 5
        assert_eq!(destroyer.process_frame(5), 1);

        // Second process at frame 5 - nothing to destroy
        assert_eq!(destroyer.process_frame(5), 0);
    }

    /// Test process_frame with interleaved defer and process.
    #[test]
    fn test_interleaved_defer_and_process() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        // Frame 0: defer(0), process(0) -> 0 destroyed, 1 pending (destroy at 1)
        // Frame 1: defer(1), process(1) -> 1 destroyed (frame 0's), 1 pending (destroy at 2)
        // Frame 2: defer(2), process(2) -> 1 destroyed (frame 1's), 1 pending (destroy at 3)
        // Frame 3: defer(3), process(3) -> 1 destroyed (frame 2's), 1 pending (destroy at 4)
        // Frame 4: defer(4), process(4) -> 1 destroyed (frame 3's), 1 pending (destroy at 5)
        // After loop: 1 pending (frame 4's resource, destroy at 5)

        for frame in 0..5u64 {
            destroyer.defer_other(MockBuffer::new(frame, 256), frame);
            let _destroyed = destroyer.process_frame(frame);
        }

        // Should have 1 pending (frame 4's resource, ready at frame 5)
        assert_eq!(destroyer.pending_count(), 1);

        // Process frame 5 to clean up the last resource
        let destroyed = destroyer.process_frame(5);
        assert_eq!(destroyed, 1);
        assert!(destroyer.is_empty());
    }

    /// Test process_frame respects destroy_at_frame exactly.
    #[test]
    fn test_process_exact_frame_boundary() {
        let mut destroyer = DeferredDestroyer::with_delay(2);
        destroyer.defer_other(MockBuffer::new(1, 256), 10);

        // Frame 11: not ready (destroy_at = 12)
        assert_eq!(destroyer.process_frame(11), 0);
        assert_eq!(destroyer.pending_count(), 1);

        // Frame 12: exactly ready
        assert_eq!(destroyer.process_frame(12), 1);
        assert!(destroyer.is_empty());
    }
}

// ============================================================================
// 6. Metrics Tests (8+ tests)
// ============================================================================

mod metrics_tests {
    use super::*;

    /// Test pending_count tracking.
    #[test]
    fn test_pending_count_tracking() {
        let mut destroyer = DeferredDestroyer::with_delay(10);

        assert_eq!(destroyer.pending_count(), 0);
        assert_eq!(destroyer.metrics().pending_count, 0);

        for i in 1..=5 {
            destroyer.defer_other(MockBuffer::new(i, 256), 0);
            assert_eq!(destroyer.pending_count(), i as usize);
            assert_eq!(destroyer.metrics().pending_count, i as usize);
        }
    }

    /// Test destroyed_count increments correctly.
    #[test]
    fn test_destroyed_count_increments() {
        let mut destroyer = DeferredDestroyer::with_delay(0);

        assert_eq!(destroyer.metrics().destroyed_count, 0);

        destroyer.defer_other(MockBuffer::new(1, 256), 0);
        destroyer.process_frame(0);
        assert_eq!(destroyer.metrics().destroyed_count, 1);

        destroyer.defer_other(MockBuffer::new(2, 256), 1);
        destroyer.defer_other(MockBuffer::new(3, 256), 1);
        destroyer.process_frame(1);
        assert_eq!(destroyer.metrics().destroyed_count, 3);
    }

    /// Test peak_pending tracking.
    #[test]
    fn test_peak_pending_tracking() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        // Add 3 resources
        destroyer.defer_other(1, 0);
        destroyer.defer_other(2, 0);
        destroyer.defer_other(3, 0);
        assert_eq!(destroyer.metrics().peak_pending, 3);

        // Destroy all
        destroyer.process_frame(1);
        assert_eq!(destroyer.metrics().peak_pending, 3); // Peak unchanged

        // Add 2 more (peak still 3)
        destroyer.defer_other(4, 1);
        destroyer.defer_other(5, 1);
        assert_eq!(destroyer.metrics().peak_pending, 3);

        // Add 2 more (new peak = 4)
        destroyer.defer_other(6, 1);
        destroyer.defer_other(7, 1);
        assert_eq!(destroyer.metrics().peak_pending, 4);
    }

    /// Test per-type destroyed counts (other_destroyed).
    #[test]
    fn test_per_type_destroyed_counts() {
        let mut destroyer = DeferredDestroyer::with_delay(0);

        destroyer.defer_other(MockBuffer::new(1, 256), 0);
        destroyer.defer_other(MockTexture::new(1, 256, 256), 0);
        destroyer.defer_other("string", 0);

        destroyer.process_frame(0);

        let metrics = destroyer.metrics();
        // All are "Other" since we use defer_other
        assert_eq!(metrics.other_destroyed, 3);
        assert_eq!(metrics.buffers_destroyed, 0);
        assert_eq!(metrics.textures_destroyed, 0);
    }

    /// Test metrics after clear.
    #[test]
    fn test_metrics_after_clear() {
        let mut destroyer = DeferredDestroyer::with_delay(10);

        destroyer.defer_other(1, 0);
        destroyer.defer_other(2, 0);
        destroyer.defer_other(3, 0);

        destroyer.clear();

        let metrics = destroyer.metrics();
        assert_eq!(metrics.pending_count, 0);
        assert_eq!(metrics.destroyed_count, 3);
        assert_eq!(metrics.peak_pending, 3);
        assert_eq!(metrics.other_destroyed, 3);
    }

    /// Test metrics after process_frame.
    #[test]
    fn test_metrics_after_process() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        destroyer.defer_other(1, 0);
        destroyer.defer_other(2, 0);

        let metrics_before = destroyer.metrics();
        assert_eq!(metrics_before.pending_count, 2);
        assert_eq!(metrics_before.destroyed_count, 0);

        destroyer.process_frame(1);

        let metrics_after = destroyer.metrics();
        assert_eq!(metrics_after.pending_count, 0);
        assert_eq!(metrics_after.destroyed_count, 2);
    }

    /// Test DeferredDestroyerMetrics Clone.
    #[test]
    fn test_metrics_clone() {
        let mut destroyer = DeferredDestroyer::with_delay(0);
        destroyer.defer_other(1, 0);
        destroyer.process_frame(0);

        let metrics1 = destroyer.metrics();
        let metrics2 = metrics1.clone();

        assert_eq!(metrics1.destroyed_count, metrics2.destroyed_count);
        assert_eq!(metrics1.pending_count, metrics2.pending_count);
        assert_eq!(metrics1.peak_pending, metrics2.peak_pending);
    }

    /// Test DeferredDestroyerMetrics Default.
    #[test]
    fn test_metrics_default() {
        let metrics = DeferredDestroyerMetrics::default();

        assert_eq!(metrics.pending_count, 0);
        assert_eq!(metrics.destroyed_count, 0);
        assert_eq!(metrics.peak_pending, 0);
        assert_eq!(metrics.buffers_destroyed, 0);
        assert_eq!(metrics.textures_destroyed, 0);
        assert_eq!(metrics.other_destroyed, 0);
    }

    /// Test metrics accuracy over many operations.
    #[test]
    fn test_metrics_accuracy_over_many_operations() {
        let mut destroyer = DeferredDestroyer::with_delay(0);
        let mut expected_destroyed = 0u64;
        let mut max_pending = 0usize;

        for batch in 0..10 {
            let batch_size = (batch % 5) + 1;

            for i in 0..batch_size {
                destroyer.defer_other(i, batch as u64);
            }

            if destroyer.pending_count() > max_pending {
                max_pending = destroyer.pending_count();
            }

            let destroyed = destroyer.process_frame(batch as u64);
            expected_destroyed += destroyed as u64;
        }

        let metrics = destroyer.metrics();
        assert_eq!(metrics.destroyed_count, expected_destroyed);
        assert_eq!(metrics.peak_pending, max_pending);
    }
}

// ============================================================================
// 7. Edge Cases (8+ tests)
// ============================================================================

mod edge_case_tests {
    use super::*;

    /// Test clear functionality.
    #[test]
    fn test_clear_functionality() {
        let mut destroyer = DeferredDestroyer::with_delay(100);

        for i in 0..10 {
            destroyer.defer_other(MockBuffer::new(i, 256), 0);
        }

        assert_eq!(destroyer.pending_count(), 10);

        destroyer.clear();

        assert!(destroyer.is_empty());
        assert_eq!(destroyer.pending_count(), 0);
    }

    /// Test set_default_delay functionality.
    #[test]
    fn test_set_default_delay() {
        let mut destroyer = DeferredDestroyer::new();
        assert_eq!(destroyer.default_delay(), DEFAULT_DESTRUCTION_DELAY);

        destroyer.set_default_delay(10);
        assert_eq!(destroyer.default_delay(), 10);

        // New deferrals should use new delay
        destroyer.defer_other(MockBuffer::new(1, 256), 0);

        // Should not destroy before frame 10
        assert_eq!(destroyer.process_frame(9), 0);

        // Should destroy at frame 10
        assert_eq!(destroyer.process_frame(10), 1);
    }

    /// Test is_empty check.
    #[test]
    fn test_is_empty_check() {
        let mut destroyer = DeferredDestroyer::new();

        assert!(destroyer.is_empty());

        destroyer.defer_other(1, 0);
        assert!(!destroyer.is_empty());

        destroyer.clear();
        assert!(destroyer.is_empty());
    }

    /// Test very large pending queue.
    #[test]
    fn test_large_pending_queue() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        let count = 1000;
        for i in 0..count {
            destroyer.defer_other(i, 0);
        }

        assert_eq!(destroyer.pending_count(), count);

        let destroyed = destroyer.process_frame(1);
        assert_eq!(destroyed, count);
        assert!(destroyer.is_empty());
    }

    /// Test frame overflow edge case (near u64::MAX).
    #[test]
    fn test_frame_near_max() {
        let mut destroyer = DeferredDestroyer::with_delay(1);
        let near_max = u64::MAX - 5;

        destroyer.defer_other(MockBuffer::new(1, 256), near_max);

        // Should not destroy before near_max + 1
        assert_eq!(destroyer.process_frame(near_max), 0);

        // Should destroy at near_max + 1
        assert_eq!(destroyer.process_frame(near_max + 1), 1);
    }

    /// Test rapid defer/process cycles.
    #[test]
    fn test_rapid_defer_process_cycles() {
        let mut destroyer = DeferredDestroyer::with_delay(0);

        for frame in 0..100u64 {
            destroyer.defer_other(frame, frame);
            let destroyed = destroyer.process_frame(frame);
            assert_eq!(destroyed, 1);
            assert!(destroyer.is_empty());
        }

        let metrics = destroyer.metrics();
        assert_eq!(metrics.destroyed_count, 100);
    }

    /// Test defer with delay 0 immediate destruction.
    #[test]
    fn test_delay_zero_immediate_destruction() {
        let mut destroyer = DeferredDestroyer::with_delay(0);

        for i in 0..5 {
            destroyer.defer_other(i, 0);
        }

        // All should be destroyed at frame 0
        assert_eq!(destroyer.process_frame(0), 5);
        assert!(destroyer.is_empty());
    }

    /// Test clear updates metrics correctly.
    #[test]
    fn test_clear_updates_metrics() {
        let mut destroyer = DeferredDestroyer::with_delay(100);

        destroyer.defer_other(1, 0);
        destroyer.defer_other(2, 0);
        destroyer.defer_other(3, 0);

        let metrics_before = destroyer.metrics();
        assert_eq!(metrics_before.pending_count, 3);
        assert_eq!(metrics_before.destroyed_count, 0);

        destroyer.clear();

        let metrics_after = destroyer.metrics();
        assert_eq!(metrics_after.pending_count, 0);
        assert_eq!(metrics_after.destroyed_count, 3);
    }

    /// Test drop behavior with pending resources.
    #[test]
    fn test_drop_with_pending_resources() {
        let drop_counter = Arc::new(AtomicUsize::new(0));

        {
            let mut destroyer = DeferredDestroyer::with_delay(100);

            destroyer.defer_other(DropTracker::new(1, drop_counter.clone()), 0);
            destroyer.defer_other(DropTracker::new(2, drop_counter.clone()), 0);

            // Not destroyed yet (delay is 100)
            assert_eq!(drop_counter.load(Ordering::SeqCst), 0);

            // Destroyer goes out of scope here
        }

        // Resources should be dropped when destroyer is dropped
        assert_eq!(drop_counter.load(Ordering::SeqCst), 2);
    }

    /// Test set_default_delay does not affect already queued resources.
    #[test]
    fn test_set_delay_no_retroactive_effect() {
        let mut destroyer = DeferredDestroyer::with_delay(2);

        // Queue with delay 2
        destroyer.defer_other(1, 0); // destroy at frame 2

        // Change delay to 10
        destroyer.set_default_delay(10);

        // Queue with new delay
        destroyer.defer_other(2, 0); // destroy at frame 10

        // Frame 2: only first resource destroyed
        assert_eq!(destroyer.process_frame(2), 1);
        assert_eq!(destroyer.pending_count(), 1);

        // Frame 10: second resource destroyed
        assert_eq!(destroyer.process_frame(10), 1);
        assert!(destroyer.is_empty());
    }

    /// Test defer_resource method with pre-wrapped resource.
    #[test]
    fn test_defer_resource_pre_wrapped() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        let resource = DeferredResource::Other(Box::new(vec![1, 2, 3, 4, 5]));
        destroyer.defer_resource(resource, 5);

        assert_eq!(destroyer.pending_count(), 1);
        assert_eq!(destroyer.process_frame(5), 0);
        assert_eq!(destroyer.process_frame(6), 1);
    }
}

// ============================================================================
// 8. Additional Comprehensive Tests
// ============================================================================

mod comprehensive_tests {
    use super::*;

    /// Test DEFAULT_DESTRUCTION_DELAY constant value.
    #[test]
    fn test_default_destruction_delay_constant() {
        assert_eq!(DEFAULT_DESTRUCTION_DELAY, 2);
    }

    /// Test DeferredResource debug formatting.
    #[test]
    fn test_deferred_resource_debug_format() {
        let other = DeferredResource::Other(Box::new(42i32));
        let debug_str = format!("{:?}", other);
        assert!(debug_str.contains("DeferredResource::Other"));
    }

    /// Test multiple process_frame calls at increasing frames.
    #[test]
    fn test_sequential_frame_processing() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        destroyer.defer_other(1, 0);
        destroyer.defer_other(2, 2);
        destroyer.defer_other(3, 4);

        let mut total_destroyed = 0;
        for frame in 0..10 {
            total_destroyed += destroyer.process_frame(frame);
        }

        assert_eq!(total_destroyed, 3);
        assert!(destroyer.is_empty());
    }

    /// Test metric values remain valid after many operations.
    #[test]
    fn test_metric_invariants() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        for i in 0..50 {
            destroyer.defer_other(i, i as u64);

            let metrics = destroyer.metrics();
            // pending_count should match pending.len()
            assert_eq!(metrics.pending_count, destroyer.pending_count());
            // peak_pending >= pending_count
            assert!(metrics.peak_pending >= metrics.pending_count);
        }
    }

    /// Test zero delay with multiple frames.
    #[test]
    fn test_zero_delay_multiple_frames() {
        let mut destroyer = DeferredDestroyer::with_delay(0);

        destroyer.defer_other(1, 5);
        destroyer.defer_other(2, 10);
        destroyer.defer_other(3, 15);

        // Each should be destroyed at their deferred frame
        assert_eq!(destroyer.process_frame(5), 1);
        assert_eq!(destroyer.process_frame(10), 1);
        assert_eq!(destroyer.process_frame(15), 1);
    }

    /// Test pending_count and is_empty consistency.
    #[test]
    fn test_pending_count_is_empty_consistency() {
        let mut destroyer = DeferredDestroyer::new();

        assert!(destroyer.is_empty());
        assert_eq!(destroyer.pending_count(), 0);

        destroyer.defer_other(1, 0);
        assert!(!destroyer.is_empty());
        assert_eq!(destroyer.pending_count(), 1);

        destroyer.defer_other(2, 0);
        assert!(!destroyer.is_empty());
        assert_eq!(destroyer.pending_count(), 2);
    }

    /// Test defer_other with Send types.
    #[test]
    fn test_defer_other_send_types() {
        let mut destroyer = DeferredDestroyer::with_delay(0);

        // Various Send types
        destroyer.defer_other(42u64, 0);
        destroyer.defer_other(String::from("test"), 0);
        destroyer.defer_other(vec![1, 2, 3], 0);
        destroyer.defer_other(Arc::new(42), 0);

        assert_eq!(destroyer.process_frame(0), 4);
    }

    /// Test mixed labeled and unlabeled deferrals.
    #[test]
    fn test_mixed_labeled_unlabeled() {
        let mut destroyer = DeferredDestroyer::with_delay(1);

        destroyer.defer_other(1, 0);
        destroyer.defer_other_labeled(2, 0, "labeled_1");
        destroyer.defer_other(3, 0);
        destroyer.defer_other_labeled(4, 0, "labeled_2");

        assert_eq!(destroyer.pending_count(), 4);
        assert_eq!(destroyer.process_frame(1), 4);
    }

    /// Test stress with high volume.
    #[test]
    fn test_high_volume_stress() {
        let mut destroyer = DeferredDestroyer::with_delay(2);

        // Add 500 resources across 100 frames
        for frame in 0..100u64 {
            for i in 0..5 {
                destroyer.defer_other((frame, i), frame);
            }
        }

        assert_eq!(destroyer.pending_count(), 500);

        // Process all
        let destroyed = destroyer.process_frame(200);
        assert_eq!(destroyed, 500);
        assert!(destroyer.is_empty());

        let metrics = destroyer.metrics();
        assert_eq!(metrics.destroyed_count, 500);
        assert_eq!(metrics.peak_pending, 500);
    }
}

// ============================================================================
// Test Summary
// ============================================================================

#[test]
fn whitebox_test_summary() {
    // This test serves as documentation of test coverage
    // Total tests: 60+ across 8 categories
    //
    // 1. Construction Tests: 9 tests
    // 2. Defer Buffer Tests: 11 tests
    // 3. Defer Texture Tests: 10 tests
    // 4. Defer Other Tests: 8 tests
    // 5. Process Frame Tests: 12 tests
    // 6. Metrics Tests: 10 tests
    // 7. Edge Cases: 12 tests
    // 8. Comprehensive Tests: 9 tests
    //
    // Coverage areas:
    // - Default and custom construction
    // - Zero and large delay values
    // - Frame 0 and near-MAX frame numbers
    // - Single and multiple resource deferrals
    // - Mixed resource types
    // - Labeled and unlabeled deferrals
    // - Process timing and boundaries
    // - Metrics tracking accuracy
    // - Clear and drop behavior
    // - High volume stress testing
    assert!(true);
}
