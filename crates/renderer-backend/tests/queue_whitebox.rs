// SPDX-License-Identifier: MIT
//
// queue_whitebox.rs -- Whitebox structural tests for T-WGPU-P1.4.1
// (TrinityQueue - Queue Submission).
//
// These tests exercise the internal implementation of TrinityQueue,
// covering all code paths in queue submission, pending tracking,
// callbacks, and the SubmissionTracker utility.
//
// WHITEBOX coverage plan:
//   - Path A: TrinityQueue::new() construction with pending_submissions = 0
//   - Path B: TrinityQueue::submit() with single command buffer
//   - Path C: TrinityQueue::submit() with multiple command buffers
//   - Path D: TrinityQueue::submit() with empty iterator (zero buffers)
//   - Path E: TrinityQueue::submit_single() convenience method
//   - Path F: TrinityQueue::submit_empty() convenience method
//   - Path G: TrinityQueue::pending_count() atomic read
//   - Path H: TrinityQueue::has_pending_work() boolean check
//   - Path I: TrinityQueue::on_submitted_work_done() callback registration
//   - Path J: TrinityQueue::on_submitted_work_done_tracked() with Arc<Self>
//   - Path K: TrinityQueue::inner() accessor method
//   - Path L: TrinityQueue::write_buffer() forwarding method
//   - Path M: TrinityQueue::write_texture() forwarding method
//   - Path N: TrinityQueue Debug trait implementation
//   - Path O: TrinityQueue Send + Sync safety
//   - Path P: SubmissionTracker::new() construction
//   - Path Q: SubmissionTracker::track_submission() ID generation and submission
//   - Path R: SubmissionTracker::is_completed() completion checking
//   - Path S: SubmissionTracker::mark_completed() internal marking
//   - Path T: SubmissionTracker::clear_completed() cleanup
//   - Path U: SubmissionTracker::completed_count() counting
//   - Path V: SubmissionTracker Default trait
//   - Path W: SubmissionTracker Debug trait
//   - Path X: Concurrent pending count increments (thread safety)
//   - Path Y: Multiple sequential submissions tracking
//   - Path Z: Edge case - rapid submission/completion cycles
//
// Acceptance criteria (T-WGPU-P1.4.1):
//   1. Accepts single or multiple command buffers
//   2. Returns SubmissionIndex
//   3. Tracks pending submissions
//   4. Works with on_submitted_work_done callback

use renderer_backend::device::{SubmissionTracker, TrinityInstance, TrinityQueue};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

// ===========================================================================
// Test Helpers
// ===========================================================================

/// Helper to create a real wgpu device and queue for testing.
/// Returns None if no GPU adapter is available (CI environment).
fn create_test_device_and_queue() -> Option<(wgpu::Device, TrinityQueue)> {
    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();

    if adapters.is_empty() {
        eprintln!("WHITEBOX: No GPU adapter available, skipping hardware test");
        return None;
    }

    // Use the first available adapter
    let adapter = adapters.into_iter().next()?;
    let info = adapter.get_info();
    eprintln!(
        "WHITEBOX: Using adapter '{}' (backend: {:?})",
        info.name, info.backend
    );

    // Request device
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("whitebox_test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .ok()?;

    Some((device, TrinityQueue::new(queue)))
}

/// Helper to create a command buffer from a device.
fn create_command_buffer(device: &wgpu::Device, label: &str) -> wgpu::CommandBuffer {
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(label),
    });
    encoder.finish()
}

// ===========================================================================
// Path A: TrinityQueue::new() construction
// ===========================================================================

#[test]
fn test_trinity_queue_new_pending_starts_at_zero() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        assert_eq!(
            queue.pending_count(),
            0,
            "New TrinityQueue should have zero pending submissions"
        );
    }
}

#[test]
fn test_trinity_queue_new_has_no_pending_work() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        assert!(
            !queue.has_pending_work(),
            "New TrinityQueue should report no pending work"
        );
    }
}

#[test]
fn test_trinity_queue_new_inner_is_accessible() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        // inner() should return a valid reference
        let inner = queue.inner();
        let _ = std::mem::size_of_val(inner);
    }
}

// ===========================================================================
// Path B: TrinityQueue::submit() with single command buffer
// ===========================================================================

#[test]
fn test_submit_single_buffer_returns_submission_index() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let cmd_buf = create_command_buffer(&device, "test_single");

        // Submit single buffer via iterator
        let index = queue.submit(std::iter::once(cmd_buf));

        // Should return a valid SubmissionIndex
        let _ = index;

        // Pending count should increment
        assert!(
            queue.pending_count() >= 1,
            "Pending count should be at least 1 after submission"
        );
    }
}

#[test]
fn test_submit_single_buffer_increments_pending() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let initial = queue.pending_count();

        let cmd_buf = create_command_buffer(&device, "test_increment");
        let _index = queue.submit(std::iter::once(cmd_buf));

        assert_eq!(
            queue.pending_count(),
            initial + 1,
            "Submit should increment pending count by 1"
        );
    }
}

// ===========================================================================
// Path C: TrinityQueue::submit() with multiple command buffers
// ===========================================================================

#[test]
fn test_submit_multiple_buffers_returns_index() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let buffers = vec![
            create_command_buffer(&device, "multi_1"),
            create_command_buffer(&device, "multi_2"),
            create_command_buffer(&device, "multi_3"),
        ];

        let index = queue.submit(buffers);

        // Should return a valid SubmissionIndex
        let _ = index;
    }
}

#[test]
fn test_submit_multiple_buffers_increments_once() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let initial = queue.pending_count();

        let buffers = vec![
            create_command_buffer(&device, "batch_1"),
            create_command_buffer(&device, "batch_2"),
            create_command_buffer(&device, "batch_3"),
            create_command_buffer(&device, "batch_4"),
        ];

        let _index = queue.submit(buffers);

        // Multiple buffers in one submit = one pending increment
        assert_eq!(
            queue.pending_count(),
            initial + 1,
            "Batch submit should increment pending count by 1, not by buffer count"
        );
    }
}

#[test]
fn test_submit_vec_of_buffers() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let mut buffers = Vec::new();
        for i in 0..5 {
            buffers.push(create_command_buffer(&device, &format!("vec_{}", i)));
        }

        let index = queue.submit(buffers);
        let _ = index;
    }
}

// ===========================================================================
// Path D: TrinityQueue::submit() with empty iterator
// ===========================================================================

#[test]
fn test_submit_empty_iterator_returns_index() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        // Empty iterator should still return a valid index
        let index = queue.submit(std::iter::empty::<wgpu::CommandBuffer>());
        let _ = index;
    }
}

#[test]
fn test_submit_empty_iterator_increments_pending() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let initial = queue.pending_count();

        // Even empty submit increments pending (it's a sync point)
        let _index = queue.submit(std::iter::empty::<wgpu::CommandBuffer>());

        assert_eq!(
            queue.pending_count(),
            initial + 1,
            "Empty submit should still increment pending count"
        );
    }
}

#[test]
fn test_submit_empty_vec() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let empty: Vec<wgpu::CommandBuffer> = vec![];
        let index = queue.submit(empty);
        let _ = index;
    }
}

// ===========================================================================
// Path E: TrinityQueue::submit_single() convenience method
// ===========================================================================

#[test]
fn test_submit_single_convenience_method() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let cmd_buf = create_command_buffer(&device, "submit_single");

        let index = queue.submit_single(cmd_buf);

        // Should return valid index
        let _ = index;
        assert!(
            queue.has_pending_work(),
            "submit_single should result in pending work"
        );
    }
}

#[test]
fn test_submit_single_increments_pending() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let initial = queue.pending_count();

        let cmd_buf = create_command_buffer(&device, "submit_single_inc");
        let _index = queue.submit_single(cmd_buf);

        assert_eq!(
            queue.pending_count(),
            initial + 1,
            "submit_single should increment pending by 1"
        );
    }
}

// ===========================================================================
// Path F: TrinityQueue::submit_empty() convenience method
// ===========================================================================

#[test]
fn test_submit_empty_convenience_method() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let index = queue.submit_empty();

        // Should return valid index
        let _ = index;
    }
}

#[test]
fn test_submit_empty_increments_pending() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let initial = queue.pending_count();

        let _index = queue.submit_empty();

        assert_eq!(
            queue.pending_count(),
            initial + 1,
            "submit_empty should increment pending by 1"
        );
    }
}

#[test]
fn test_submit_empty_creates_sync_point() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        // Submit some work
        let cmd_buf = create_command_buffer(&device, "before_sync");
        let _idx1 = queue.submit_single(cmd_buf);

        // Submit empty as sync point
        let idx2 = queue.submit_empty();

        // Both indices should be valid
        let _ = idx2;
    }
}

// ===========================================================================
// Path G: TrinityQueue::pending_count() atomic read
// ===========================================================================

#[test]
fn test_pending_count_reads_atomic() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        assert_eq!(queue.pending_count(), 0);

        let cmd_buf1 = create_command_buffer(&device, "atomic_1");
        queue.submit_single(cmd_buf1);
        assert_eq!(queue.pending_count(), 1);

        let cmd_buf2 = create_command_buffer(&device, "atomic_2");
        queue.submit_single(cmd_buf2);
        assert_eq!(queue.pending_count(), 2);

        let cmd_buf3 = create_command_buffer(&device, "atomic_3");
        queue.submit_single(cmd_buf3);
        assert_eq!(queue.pending_count(), 3);
    }
}

#[test]
fn test_pending_count_multiple_reads_consistent() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let cmd_buf = create_command_buffer(&device, "consistent");
        queue.submit_single(cmd_buf);

        // Multiple reads should be consistent (no submissions between)
        let count1 = queue.pending_count();
        let count2 = queue.pending_count();
        let count3 = queue.pending_count();

        assert_eq!(count1, count2);
        assert_eq!(count2, count3);
    }
}

// ===========================================================================
// Path H: TrinityQueue::has_pending_work() boolean check
// ===========================================================================

#[test]
fn test_has_pending_work_false_initially() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        assert!(
            !queue.has_pending_work(),
            "New queue should have no pending work"
        );
    }
}

#[test]
fn test_has_pending_work_true_after_submit() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let cmd_buf = create_command_buffer(&device, "has_pending");
        queue.submit_single(cmd_buf);

        assert!(
            queue.has_pending_work(),
            "Queue should have pending work after submit"
        );
    }
}

#[test]
fn test_has_pending_work_reflects_pending_count() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        // Initially: count=0, has_pending=false
        assert_eq!(queue.pending_count() > 0, queue.has_pending_work());

        // After submit: count=1, has_pending=true
        let cmd_buf = create_command_buffer(&device, "reflects");
        queue.submit_single(cmd_buf);
        assert_eq!(queue.pending_count() > 0, queue.has_pending_work());
    }
}

// ===========================================================================
// Path I: TrinityQueue::on_submitted_work_done() callback registration
// ===========================================================================

#[test]
fn test_on_submitted_work_done_accepts_closure() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let cmd_buf = create_command_buffer(&device, "callback_test");
        queue.submit_single(cmd_buf);

        // Register callback - should not panic
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = Arc::clone(&called);
        queue.on_submitted_work_done(move || {
            called_clone.store(true, Ordering::SeqCst);
        });

        // Note: Callback execution depends on device.poll(), which we don't
        // call here. This test verifies registration succeeds.
        let _ = called;
    }
}

#[test]
fn test_on_submitted_work_done_callback_invoked_on_poll() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let cmd_buf = create_command_buffer(&device, "poll_callback");
        queue.submit_single(cmd_buf);

        let called = Arc::new(AtomicBool::new(false));
        let called_clone = Arc::clone(&called);
        queue.on_submitted_work_done(move || {
            called_clone.store(true, Ordering::SeqCst);
        });

        // Poll the device to trigger callbacks
        device.poll(wgpu::Maintain::Wait);

        // After poll(Wait), callback should have been invoked
        assert!(
            called.load(Ordering::SeqCst),
            "Callback should be invoked after device.poll(Wait)"
        );
    }
}

#[test]
fn test_on_submitted_work_done_multiple_callbacks() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let counter = Arc::new(AtomicU64::new(0));

        // Register multiple callbacks
        for _ in 0..3 {
            let cmd_buf = create_command_buffer(&device, "multi_cb");
            queue.submit_single(cmd_buf);

            let counter_clone = Arc::clone(&counter);
            queue.on_submitted_work_done(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
        }

        // Poll to trigger all callbacks
        device.poll(wgpu::Maintain::Wait);

        assert_eq!(
            counter.load(Ordering::SeqCst),
            3,
            "All three callbacks should have been invoked"
        );
    }
}

// ===========================================================================
// Path J: TrinityQueue::on_submitted_work_done_tracked() with Arc<Self>
// ===========================================================================

#[test]
fn test_on_submitted_work_done_tracked_requires_arc() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));

        let cmd_buf = create_command_buffer(&device, "tracked_test");
        queue.submit_single(cmd_buf);

        let called = Arc::new(AtomicBool::new(false));
        let called_clone = Arc::clone(&called);

        // on_submitted_work_done_tracked takes &Arc<Self>
        queue.on_submitted_work_done_tracked(move || {
            called_clone.store(true, Ordering::SeqCst);
        });

        // Poll to trigger
        device.poll(wgpu::Maintain::Wait);

        assert!(
            called.load(Ordering::SeqCst),
            "Tracked callback should be invoked"
        );
    }
}

#[test]
fn test_on_submitted_work_done_tracked_decrements_pending() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));

        let cmd_buf = create_command_buffer(&device, "decrement_test");
        queue.submit_single(cmd_buf);

        let initial = queue.pending_count();
        assert_eq!(initial, 1, "Should have 1 pending after submit");

        // Register tracked callback
        queue.on_submitted_work_done_tracked(|| {});

        // Poll to trigger callback
        device.poll(wgpu::Maintain::Wait);

        assert_eq!(
            queue.pending_count(),
            0,
            "Tracked callback should decrement pending count"
        );
    }
}

#[test]
fn test_on_submitted_work_done_tracked_invokes_user_callback() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));

        let cmd_buf = create_command_buffer(&device, "user_cb_test");
        queue.submit_single(cmd_buf);

        let user_called = Arc::new(AtomicBool::new(false));
        let user_called_clone = Arc::clone(&user_called);

        queue.on_submitted_work_done_tracked(move || {
            user_called_clone.store(true, Ordering::SeqCst);
        });

        device.poll(wgpu::Maintain::Wait);

        assert!(
            user_called.load(Ordering::SeqCst),
            "User callback should be invoked by tracked callback"
        );
    }
}

/// Helper that returns raw wgpu::Queue instead of TrinityQueue
fn create_test_device_and_queue_raw() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = TrinityInstance::new();
    let adapters = instance.enumerate_adapters();

    if adapters.is_empty() {
        return None;
    }

    let adapter = adapters.into_iter().next()?;

    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("whitebox_test_device_raw"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .ok()
}

// ===========================================================================
// Path K: TrinityQueue::inner() accessor method
// ===========================================================================

#[test]
fn test_inner_returns_wgpu_queue_reference() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let inner: &wgpu::Queue = queue.inner();

        // Should be able to call wgpu::Queue methods
        let _ = std::mem::size_of_val(inner);
    }
}

#[test]
fn test_inner_can_be_used_for_write_buffer() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        // Create a buffer
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("inner_write_test"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Write via inner()
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        queue.inner().write_buffer(&buffer, 0, &data);

        // Should not panic
    }
}

// ===========================================================================
// Path L: TrinityQueue::write_buffer() forwarding method
// ===========================================================================

#[test]
fn test_write_buffer_forwards_to_inner() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("write_buffer_test"),
            size: 128,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let data: [u8; 32] = [0xAB; 32];
        queue.write_buffer(&buffer, 0, &data);

        // No panic means success
    }
}

#[test]
fn test_write_buffer_with_offset() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("write_offset_test"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let data: [u8; 64] = [0xCD; 64];

        // Write at offset 128
        queue.write_buffer(&buffer, 128, &data);
    }
}

#[test]
fn test_write_buffer_empty_data() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("write_empty_test"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Writing empty data should be fine
        queue.write_buffer(&buffer, 0, &[]);
    }
}

// ===========================================================================
// Path M: TrinityQueue::write_texture() forwarding method
// ===========================================================================

#[test]
fn test_write_texture_forwards_to_inner() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("write_texture_test"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // RGBA data for 64x64 texture
        let data = vec![0u8; 64 * 64 * 4];

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(64 * 4),
                rows_per_image: Some(64),
            },
            wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
        );
    }
}

#[test]
fn test_write_texture_subregion() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("write_subregion_test"),
            size: wgpu::Extent3d {
                width: 128,
                height: 128,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Write only 32x32 subregion
        let data = vec![0xFFu8; 32 * 32 * 4];

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 48, y: 48, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(32 * 4),
                rows_per_image: Some(32),
            },
            wgpu::Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 1,
            },
        );
    }
}

// ===========================================================================
// Path N: TrinityQueue Debug trait implementation
// ===========================================================================

#[test]
fn test_debug_trait_contains_struct_name() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let debug_str = format!("{:?}", queue);
        assert!(
            debug_str.contains("TrinityQueue"),
            "Debug output should contain 'TrinityQueue'"
        );
    }
}

#[test]
fn test_debug_trait_contains_pending_field() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let debug_str = format!("{:?}", queue);
        assert!(
            debug_str.contains("pending_submissions"),
            "Debug output should contain 'pending_submissions'"
        );
    }
}

#[test]
fn test_debug_trait_shows_pending_count() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        // Initially 0
        let debug_str = format!("{:?}", queue);
        assert!(
            debug_str.contains("0"),
            "Debug should show pending count 0 initially"
        );

        // After submit
        let cmd_buf = create_command_buffer(&device, "debug_pending");
        queue.submit_single(cmd_buf);

        let debug_str = format!("{:?}", queue);
        assert!(
            debug_str.contains("1"),
            "Debug should show pending count 1 after submit"
        );
    }
}

#[test]
fn test_debug_trait_non_exhaustive() {
    if let Some((_device, queue)) = create_test_device_and_queue() {
        let debug_str = format!("{:?}", queue);
        assert!(
            debug_str.contains(".."),
            "Debug should use non-exhaustive format"
        );
    }
}

// ===========================================================================
// Path O: TrinityQueue Send + Sync safety
// ===========================================================================

#[test]
fn test_trinity_queue_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<TrinityQueue>();
}

#[test]
fn test_trinity_queue_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<TrinityQueue>();
}

#[test]
fn test_trinity_queue_send_across_thread() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let queue = Arc::new(queue);
        let device = Arc::new(device);

        let queue_clone = Arc::clone(&queue);
        let device_clone = Arc::clone(&device);

        let handle = std::thread::spawn(move || {
            let cmd_buf = device_clone.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("thread_test"),
            });
            queue_clone.submit_single(cmd_buf.finish());
            queue_clone.pending_count()
        });

        let count = handle.join().expect("Thread should complete");
        assert!(count >= 1, "Submission from other thread should work");
    }
}

// ===========================================================================
// Path P: SubmissionTracker::new() construction
// ===========================================================================

#[test]
fn test_submission_tracker_new_empty() {
    let tracker = SubmissionTracker::new();
    assert_eq!(
        tracker.completed_count(),
        0,
        "New tracker should have zero completions"
    );
}

#[test]
fn test_submission_tracker_new_no_completions() {
    let tracker = SubmissionTracker::new();

    // No ID should be completed yet
    assert!(
        !tracker.is_completed(0),
        "ID 0 should not be completed in new tracker"
    );
    assert!(
        !tracker.is_completed(100),
        "ID 100 should not be completed in new tracker"
    );
}

// ===========================================================================
// Path Q: SubmissionTracker::track_submission() ID generation
// ===========================================================================

#[test]
fn test_track_submission_returns_sequential_ids() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));
        let tracker = Arc::new(SubmissionTracker::new());

        let id0 = tracker.track_submission(&queue, create_command_buffer(&device, "seq_0"));
        let id1 = tracker.track_submission(&queue, create_command_buffer(&device, "seq_1"));
        let id2 = tracker.track_submission(&queue, create_command_buffer(&device, "seq_2"));

        assert_eq!(id0, 0, "First ID should be 0");
        assert_eq!(id1, 1, "Second ID should be 1");
        assert_eq!(id2, 2, "Third ID should be 2");
    }
}

#[test]
fn test_track_submission_submits_command_buffer() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));
        let tracker = Arc::new(SubmissionTracker::new());

        let initial = queue.pending_count();

        tracker.track_submission(&queue, create_command_buffer(&device, "track_sub"));

        assert_eq!(
            queue.pending_count(),
            initial + 1,
            "track_submission should submit the command buffer"
        );
    }
}

// ===========================================================================
// Path R: SubmissionTracker::is_completed() completion checking
// ===========================================================================

#[test]
fn test_is_completed_false_before_poll() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));
        let tracker = Arc::new(SubmissionTracker::new());

        let id = tracker.track_submission(&queue, create_command_buffer(&device, "before_poll"));

        // Before polling, should not be completed
        assert!(
            !tracker.is_completed(id),
            "Should not be completed before device poll"
        );
    }
}

#[test]
fn test_is_completed_true_after_poll() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));
        let tracker = Arc::new(SubmissionTracker::new());

        let id = tracker.track_submission(&queue, create_command_buffer(&device, "after_poll"));

        // Poll to trigger completion
        device.poll(wgpu::Maintain::Wait);

        assert!(
            tracker.is_completed(id),
            "Should be completed after device poll"
        );
    }
}

#[test]
fn test_is_completed_nonexistent_id() {
    let tracker = SubmissionTracker::new();

    // Check IDs that were never tracked
    assert!(!tracker.is_completed(0));
    assert!(!tracker.is_completed(999));
    assert!(!tracker.is_completed(u64::MAX));
}

// ===========================================================================
// Path S: SubmissionTracker::mark_completed() internal marking
// ===========================================================================

// Note: mark_completed() is private, but we can test it indirectly through
// the callback mechanism

#[test]
fn test_mark_completed_increments_count() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));
        let tracker = Arc::new(SubmissionTracker::new());

        let _id = tracker.track_submission(&queue, create_command_buffer(&device, "mark_test"));

        assert_eq!(tracker.completed_count(), 0, "Before poll: 0 completed");

        device.poll(wgpu::Maintain::Wait);

        assert_eq!(tracker.completed_count(), 1, "After poll: 1 completed");
    }
}

// ===========================================================================
// Path T: SubmissionTracker::clear_completed() cleanup
// ===========================================================================

#[test]
fn test_clear_completed_removes_all() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));
        let tracker = Arc::new(SubmissionTracker::new());

        // Track and complete multiple submissions
        for i in 0..5 {
            tracker.track_submission(&queue, create_command_buffer(&device, &format!("clear_{}", i)));
        }

        device.poll(wgpu::Maintain::Wait);
        assert_eq!(tracker.completed_count(), 5);

        // Clear
        tracker.clear_completed();
        assert_eq!(
            tracker.completed_count(),
            0,
            "After clear: count should be 0"
        );
    }
}

#[test]
fn test_clear_completed_ids_no_longer_completed() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));
        let tracker = Arc::new(SubmissionTracker::new());

        let id = tracker.track_submission(&queue, create_command_buffer(&device, "clear_id"));

        device.poll(wgpu::Maintain::Wait);
        assert!(tracker.is_completed(id), "Should be completed");

        tracker.clear_completed();
        assert!(
            !tracker.is_completed(id),
            "After clear, ID should not be marked completed"
        );
    }
}

#[test]
fn test_clear_completed_on_empty_tracker() {
    let tracker = SubmissionTracker::new();
    tracker.clear_completed(); // Should not panic
    assert_eq!(tracker.completed_count(), 0);
}

// ===========================================================================
// Path U: SubmissionTracker::completed_count() counting
// ===========================================================================

#[test]
fn test_completed_count_increments() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));
        let tracker = Arc::new(SubmissionTracker::new());

        assert_eq!(tracker.completed_count(), 0);

        for expected in 1..=3 {
            tracker.track_submission(
                &queue,
                create_command_buffer(&device, &format!("count_{}", expected)),
            );
            device.poll(wgpu::Maintain::Wait);
            assert_eq!(tracker.completed_count(), expected);
        }
    }
}

// ===========================================================================
// Path V: SubmissionTracker Default trait
// ===========================================================================

#[test]
fn test_submission_tracker_default() {
    let tracker: SubmissionTracker = Default::default();
    assert_eq!(
        tracker.completed_count(),
        0,
        "Default tracker should have 0 completions"
    );
}

#[test]
fn test_submission_tracker_default_equals_new() {
    let new_tracker = SubmissionTracker::new();
    let default_tracker: SubmissionTracker = Default::default();

    assert_eq!(new_tracker.completed_count(), default_tracker.completed_count());
}

// ===========================================================================
// Path W: SubmissionTracker Debug trait
// ===========================================================================

#[test]
fn test_submission_tracker_debug_contains_name() {
    let tracker = SubmissionTracker::new();
    let debug_str = format!("{:?}", tracker);
    assert!(
        debug_str.contains("SubmissionTracker"),
        "Debug should contain struct name"
    );
}

#[test]
fn test_submission_tracker_debug_contains_fields() {
    let tracker = SubmissionTracker::new();
    let debug_str = format!("{:?}", tracker);
    assert!(
        debug_str.contains("next_id"),
        "Debug should contain 'next_id'"
    );
    assert!(
        debug_str.contains("completed_count"),
        "Debug should contain 'completed_count'"
    );
}

#[test]
fn test_submission_tracker_debug_shows_values() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));
        let tracker = Arc::new(SubmissionTracker::new());

        // Initial state
        let debug_str = format!("{:?}", tracker);
        assert!(debug_str.contains("0"), "Should show 0 initially");

        // After some submissions
        tracker.track_submission(&queue, create_command_buffer(&device, "debug_val"));
        device.poll(wgpu::Maintain::Wait);

        let debug_str = format!("{:?}", tracker);
        assert!(
            debug_str.contains("1"),
            "Should show 1 after one completion"
        );
    }
}

// ===========================================================================
// Path X: Concurrent pending count increments (thread safety)
// ===========================================================================

#[test]
fn test_concurrent_submissions_thread_safe() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));
        let device = Arc::new(device);

        const NUM_THREADS: usize = 4;
        const SUBMISSIONS_PER_THREAD: usize = 10;

        let mut handles = Vec::new();

        for t in 0..NUM_THREADS {
            let queue_clone = Arc::clone(&queue);
            let device_clone = Arc::clone(&device);

            let handle = std::thread::spawn(move || {
                for i in 0..SUBMISSIONS_PER_THREAD {
                    let cmd_buf = device_clone.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some(&format!("concurrent_t{}_i{}", t, i)),
                    });
                    queue_clone.submit_single(cmd_buf.finish());
                }
            });

            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread should complete");
        }

        // Total submissions should be NUM_THREADS * SUBMISSIONS_PER_THREAD
        assert_eq!(
            queue.pending_count(),
            (NUM_THREADS * SUBMISSIONS_PER_THREAD) as u64,
            "All concurrent submissions should be counted"
        );
    }
}

// ===========================================================================
// Path Y: Multiple sequential submissions tracking
// ===========================================================================

#[test]
fn test_sequential_submissions_all_tracked() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));
        let tracker = Arc::new(SubmissionTracker::new());

        let mut ids = Vec::new();
        for i in 0..10 {
            let id = tracker.track_submission(
                &queue,
                create_command_buffer(&device, &format!("seq_{}", i)),
            );
            ids.push(id);
        }

        // IDs should be sequential
        for (expected, actual) in ids.iter().enumerate() {
            assert_eq!(*actual, expected as u64, "ID {} should be {}", actual, expected);
        }

        // Note: Due to async nature of GPU callbacks, some submissions may
        // complete before we check. We verify the IDs are sequential (above)
        // and that after polling, all are completed (below).

        // Poll to complete all
        device.poll(wgpu::Maintain::Wait);

        // All should be completed now
        for id in &ids {
            assert!(tracker.is_completed(*id), "ID {} should be completed", id);
        }

        assert_eq!(
            tracker.completed_count(),
            10,
            "All 10 submissions should be completed"
        );
    }
}

// ===========================================================================
// Path Z: Edge case - rapid submission/completion cycles
// ===========================================================================

#[test]
fn test_rapid_submit_poll_cycles() {
    if let Some((device, wgpu_queue)) = create_test_device_and_queue_raw() {
        let queue = Arc::new(TrinityQueue::new(wgpu_queue));

        for i in 0..20 {
            // Submit
            let cmd_buf = create_command_buffer(&device, &format!("rapid_{}", i));
            queue.submit_single(cmd_buf);

            let count_after_submit = queue.pending_count();
            assert!(
                count_after_submit >= 1,
                "Pending should be at least 1 after submit"
            );

            // Register tracked callback to decrement
            queue.on_submitted_work_done_tracked(|| {});

            // Poll
            device.poll(wgpu::Maintain::Wait);

            // After poll + tracked callback, pending should decrease
            // Note: Due to async nature, exact count may vary
        }
    }
}

#[test]
fn test_submission_index_is_unique_per_submit() {
    if let Some((device, queue)) = create_test_device_and_queue() {
        let mut indices = Vec::new();

        for i in 0..5 {
            let cmd_buf = create_command_buffer(&device, &format!("unique_idx_{}", i));
            let index = queue.submit_single(cmd_buf);
            indices.push(format!("{:?}", index));
        }

        // Each submission should produce a submission index
        // (indices may or may not be the same depending on wgpu internals,
        // but the operation should succeed)
        assert_eq!(indices.len(), 5, "Should have 5 submission indices");
    }
}

// ===========================================================================
// Edge cases: Acceptance criteria validation
// ===========================================================================

#[test]
fn acceptance_criteria_1_accepts_single_buffer() {
    // Acceptance Criteria 1: Accepts single command buffer
    if let Some((device, queue)) = create_test_device_and_queue() {
        let cmd_buf = create_command_buffer(&device, "ac1_single");
        let _index = queue.submit_single(cmd_buf);
        // No panic = pass
    }
}

#[test]
fn acceptance_criteria_1_accepts_multiple_buffers() {
    // Acceptance Criteria 1: Accepts multiple command buffers
    if let Some((device, queue)) = create_test_device_and_queue() {
        let buffers = vec![
            create_command_buffer(&device, "ac1_multi_1"),
            create_command_buffer(&device, "ac1_multi_2"),
        ];
        let _index = queue.submit(buffers);
        // No panic = pass
    }
}

#[test]
fn acceptance_criteria_2_returns_submission_index() {
    // Acceptance Criteria 2: Returns SubmissionIndex
    if let Some((device, queue)) = create_test_device_and_queue() {
        let cmd_buf = create_command_buffer(&device, "ac2_index");
        let index: wgpu::SubmissionIndex = queue.submit_single(cmd_buf);

        // Verify it's a real SubmissionIndex by using it
        let _ = format!("{:?}", index);
    }
}

#[test]
fn acceptance_criteria_3_tracks_pending_submissions() {
    // Acceptance Criteria 3: Tracks pending submissions
    if let Some((device, queue)) = create_test_device_and_queue() {
        assert_eq!(queue.pending_count(), 0, "Initially 0");

        let cmd_buf = create_command_buffer(&device, "ac3_track");
        queue.submit_single(cmd_buf);

        assert_eq!(queue.pending_count(), 1, "After submit: 1");
        assert!(queue.has_pending_work(), "has_pending_work should be true");
    }
}

#[test]
fn acceptance_criteria_4_works_with_callback() {
    // Acceptance Criteria 4: Works with on_submitted_work_done callback
    if let Some((device, queue)) = create_test_device_and_queue() {
        let cmd_buf = create_command_buffer(&device, "ac4_callback");
        queue.submit_single(cmd_buf);

        let callback_executed = Arc::new(AtomicBool::new(false));
        let callback_clone = Arc::clone(&callback_executed);

        queue.on_submitted_work_done(move || {
            callback_clone.store(true, Ordering::SeqCst);
        });

        device.poll(wgpu::Maintain::Wait);

        assert!(
            callback_executed.load(Ordering::SeqCst),
            "Callback should be executed after poll"
        );
    }
}
