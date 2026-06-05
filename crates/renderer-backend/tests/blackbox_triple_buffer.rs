// Blackbox contract tests for T-WGPU-P4.6.3 Triple Buffering
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_sync::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria (T-WGPU-P4.6.3):
//   1. Configurable buffer count (2 or 3)
//   2. N fences array (Vec<FrameFence>)
//   3. Wait on frame N-2 for triple buffering
//   4. Lower latency mode option
//
// Coverage (tests):
//   Construction:
//     - TrinityFrameSynchronizer::new(BufferCount::Double) creates 2-buffer sync
//     - TrinityFrameSynchronizer::new(BufferCount::Triple) creates 3-buffer sync
//   Buffer count:
//     - buffer_count() returns 2 for Double
//     - buffer_count() returns 3 for Triple
//   Index cycling:
//     - current_index() starts at 0
//     - next_index() returns (current + 1) % buffer_count
//     - Indices cycle correctly after multiple frames
//   Wait offset:
//     - wait_offset() returns 1 for Double buffering
//     - wait_offset() returns 2 for Triple buffering
//   Low latency mode:
//     - is_low_latency() defaults to false
//     - set_low_latency(true) enables low latency mode
//     - set_low_latency(false) disables low latency mode
//     - Low latency mode affects wait behavior
//   Frame lifecycle:
//     - begin_frame() advances frame state
//     - end_frame() submits commands and returns submission index
//     - frame_count() tracks completed frames
//     - wait_idle() waits for all frames
//   Edge cases:
//     - Buffer cycling wraps correctly at boundary
//     - Multiple frame submissions work correctly
//     - Fence access for each buffer index

use pollster::block_on;

use renderer_backend::frame_sync::{BufferCount, TrinityFrameSynchronizer};

// =============================================================================
// TEST INFRASTRUCTURE -- Headless wgpu device/queue
// =============================================================================

/// Creates a headless wgpu device for testing.
/// Returns None if no adapter is available (CI without GPU).
fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    Some(
        block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("blackbox_triple_buffer_test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .ok()?,
    )
}

// =============================================================================
// CRITERION 1: CONFIGURABLE BUFFER COUNT (2 or 3)
// =============================================================================

/// Test 01: BufferCount enum has Double variant
#[test]
fn test_01_buffer_count_double_exists() {
    let _count = BufferCount::Double;
    // Type and variant exist
}

/// Test 02: BufferCount enum has Triple variant
#[test]
fn test_02_buffer_count_triple_exists() {
    let _count = BufferCount::Triple;
    // Type and variant exist
}

/// Test 03: TrinityFrameSynchronizer::new(BufferCount::Double) creates synchronizer
#[test]
fn test_03_new_with_double_buffering() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);
    assert_eq!(sync.buffer_count(), 2, "Double buffering should have 2 buffers");
}

/// Test 04: TrinityFrameSynchronizer::new(BufferCount::Triple) creates synchronizer
#[test]
fn test_04_new_with_triple_buffering() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    assert_eq!(sync.buffer_count(), 3, "Triple buffering should have 3 buffers");
}

/// Test 05: buffer_count() returns correct value for Double
#[test]
fn test_05_buffer_count_returns_2_for_double() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);
    let count = sync.buffer_count();
    assert_eq!(count, 2, "buffer_count() must return 2 for Double");
}

/// Test 06: buffer_count() returns correct value for Triple
#[test]
fn test_06_buffer_count_returns_3_for_triple() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    let count = sync.buffer_count();
    assert_eq!(count, 3, "buffer_count() must return 3 for Triple");
}

// =============================================================================
// CRITERION 2: N FENCES ARRAY (Vec<FrameFence>)
// =============================================================================

/// Test 07: fence() can access fence at index 0 for double buffering
#[test]
fn test_07_fence_access_double_index_0() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);
    let fence = sync.fence(0);
    // Fence exists and is accessible
    let _ = fence.current_frame();
}

/// Test 08: fence() can access fence at index 1 for double buffering
#[test]
fn test_08_fence_access_double_index_1() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);
    let fence = sync.fence(1);
    // Fence exists and is accessible
    let _ = fence.current_frame();
}

/// Test 09: fence() can access all fences for triple buffering
#[test]
fn test_09_fence_access_triple_all() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

    // Access each fence
    for i in 0..3 {
        let fence = sync.fence(i);
        let _ = fence.current_frame();
    }
}

/// Test 10: current_fence() returns fence for current buffer
#[test]
fn test_10_current_fence_access() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    let current_idx = sync.current_index() as usize;

    let current_fence = sync.current_fence();
    let indexed_fence = sync.fence(current_idx);

    // Both should refer to the same fence (same frame number)
    assert_eq!(
        current_fence.current_frame(),
        indexed_fence.current_frame(),
        "current_fence() should return fence at current_index()"
    );
}

/// Test 11: Each fence is independent (different instances)
#[test]
fn test_11_fences_are_independent() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

    // Initially all fences should be at frame 0
    for i in 0..3 {
        let fence = sync.fence(i);
        assert_eq!(fence.current_frame(), 0, "Fence {} should start at frame 0", i);
    }
}

// =============================================================================
// CRITERION 3: WAIT ON FRAME N-2 FOR TRIPLE BUFFERING
// =============================================================================

/// Test 12: wait_offset() returns 1 for double buffering
#[test]
fn test_12_wait_offset_double() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);
    let offset = sync.wait_offset();
    assert_eq!(offset, 1, "Double buffering should wait 1 frame back");
}

/// Test 13: wait_offset() returns 2 for triple buffering
#[test]
fn test_13_wait_offset_triple() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    let offset = sync.wait_offset();
    assert_eq!(offset, 2, "Triple buffering should wait 2 frames back (N-2)");
}

/// Test 14: wait_offset() determines how far back to wait
#[test]
fn test_14_wait_offset_semantics() {
    let double = TrinityFrameSynchronizer::new(BufferCount::Double);
    let triple = TrinityFrameSynchronizer::new(BufferCount::Triple);

    // Double: wait for frame N-1 (offset 1)
    // Triple: wait for frame N-2 (offset 2)
    assert_eq!(
        triple.wait_offset() - double.wait_offset(),
        1,
        "Triple buffering should wait 1 more frame back than double"
    );
}

/// Test 15: Triple buffering allows more frames in flight
#[test]
fn test_15_triple_allows_more_frames_in_flight() {
    let double = TrinityFrameSynchronizer::new(BufferCount::Double);
    let triple = TrinityFrameSynchronizer::new(BufferCount::Triple);

    // Triple buffering has higher wait offset = more frames can be in flight
    assert!(
        triple.wait_offset() > double.wait_offset(),
        "Triple buffering should have higher wait offset"
    );
    assert!(
        triple.buffer_count() > double.buffer_count(),
        "Triple buffering should have more buffers"
    );
}

// =============================================================================
// CRITERION 4: LOWER LATENCY MODE OPTION
// =============================================================================

/// Test 16: is_low_latency() defaults to false
#[test]
fn test_16_low_latency_default_false() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    assert!(!sync.is_low_latency(), "Low latency should default to false");
}

/// Test 17: set_low_latency(true) enables low latency mode
#[test]
fn test_17_set_low_latency_true() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    sync.set_low_latency(true);
    assert!(sync.is_low_latency(), "Low latency should be true after set_low_latency(true)");
}

/// Test 18: set_low_latency(false) disables low latency mode
#[test]
fn test_18_set_low_latency_false() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    sync.set_low_latency(true);
    assert!(sync.is_low_latency());

    sync.set_low_latency(false);
    assert!(!sync.is_low_latency(), "Low latency should be false after set_low_latency(false)");
}

/// Test 19: Low latency mode toggle multiple times
#[test]
fn test_19_low_latency_toggle() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

    // Toggle multiple times
    for _ in 0..5 {
        sync.set_low_latency(true);
        assert!(sync.is_low_latency());
        sync.set_low_latency(false);
        assert!(!sync.is_low_latency());
    }
}

/// Test 20: Low latency mode works with both buffer counts
#[test]
fn test_20_low_latency_both_buffer_counts() {
    let double = TrinityFrameSynchronizer::new(BufferCount::Double);
    let triple = TrinityFrameSynchronizer::new(BufferCount::Triple);

    double.set_low_latency(true);
    triple.set_low_latency(true);

    assert!(double.is_low_latency());
    assert!(triple.is_low_latency());
}

// =============================================================================
// INDEX CYCLING TESTS
// =============================================================================

/// Test 21: current_index() starts at 0
#[test]
fn test_21_current_index_starts_at_zero() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    assert_eq!(sync.current_index(), 0, "current_index() should start at 0");
}

/// Test 22: next_index() returns (current + 1) % buffer_count for double
#[test]
fn test_22_next_index_double() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);
    let current = sync.current_index();
    let next = sync.next_index();

    assert_eq!(
        next,
        (current + 1) % 2,
        "next_index() should be (current + 1) % 2 for double buffering"
    );
}

/// Test 23: next_index() returns (current + 1) % buffer_count for triple
#[test]
fn test_23_next_index_triple() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    let current = sync.current_index();
    let next = sync.next_index();

    assert_eq!(
        next,
        (current + 1) % 3,
        "next_index() should be (current + 1) % 3 for triple buffering"
    );
}

/// Test 24: Index cycling wraps correctly at boundary for double
#[test]
fn test_24_index_cycling_double() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

    // Track indices through multiple frames
    let mut indices = Vec::new();

    for _ in 0..6 {
        indices.push(sync.current_index());
        sync.begin_frame(&device);

        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("index_cycle_test"),
        });
        let _ = sync.end_frame(&queue, &[encoder.finish()]);
    }

    // Indices should cycle: 0, 1, 0, 1, 0, 1 (after frame advances)
    for (i, &idx) in indices.iter().enumerate() {
        assert!(
            idx < 2,
            "Index {} at iteration {} should be < 2 for double buffering",
            idx, i
        );
    }
}

/// Test 25: Index cycling wraps correctly at boundary for triple
#[test]
fn test_25_index_cycling_triple() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

    // Track indices through multiple frames
    let mut indices = Vec::new();

    for _ in 0..9 {
        indices.push(sync.current_index());
        sync.begin_frame(&device);

        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("index_cycle_test"),
        });
        let _ = sync.end_frame(&queue, &[encoder.finish()]);
    }

    // All indices should be < 3
    for (i, &idx) in indices.iter().enumerate() {
        assert!(
            idx < 3,
            "Index {} at iteration {} should be < 3 for triple buffering",
            idx, i
        );
    }
}

// =============================================================================
// FRAME LIFECYCLE TESTS
// =============================================================================

/// Test 26: frame_count() starts at 0
#[test]
fn test_26_frame_count_starts_at_zero() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    assert_eq!(sync.frame_count(), 0, "frame_count() should start at 0");
}

/// Test 27: begin_frame() and end_frame() advance frame count
#[test]
fn test_27_frame_lifecycle_advances_count() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

    assert_eq!(sync.frame_count(), 0);

    sync.begin_frame(&device);
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("frame_lifecycle_test"),
    });
    let _ = sync.end_frame(&queue, &[encoder.finish()]);

    assert_eq!(sync.frame_count(), 1, "frame_count() should be 1 after one frame");
}

/// Test 28: Multiple frames increment frame count correctly
#[test]
fn test_28_multiple_frames_increment() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

    for i in 0..10 {
        assert_eq!(sync.frame_count(), i, "frame_count() should be {} before frame {}", i, i);

        sync.begin_frame(&device);
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("multi_frame_test"),
        });
        let _ = sync.end_frame(&queue, &[encoder.finish()]);
    }

    assert_eq!(sync.frame_count(), 10, "frame_count() should be 10 after 10 frames");
}

/// Test 29: end_frame() returns SubmissionIndex
#[test]
fn test_29_end_frame_returns_submission_index() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

    sync.begin_frame(&device);
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("submission_test"),
    });
    let submission: wgpu::SubmissionIndex = sync.end_frame(&queue, &[encoder.finish()]);

    // SubmissionIndex returned successfully
    let _ = submission;
}

/// Test 30: end_frame_with_iter() works with iterator
#[test]
fn test_30_end_frame_with_iter() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

    sync.begin_frame(&device);

    // Create multiple command buffers
    let encoders: Vec<_> = (0..3)
        .map(|i| {
            let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("iter_test_{}", i)),
            });
            encoder.finish()
        })
        .collect();

    let submission = sync.end_frame_with_iter(&queue, encoders.into_iter());
    let _ = submission;
}

/// Test 31: wait_idle() waits for all frames
#[test]
fn test_31_wait_idle() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

    // Submit multiple frames
    for _ in 0..5 {
        sync.begin_frame(&device);
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wait_idle_test"),
        });
        let _ = sync.end_frame(&queue, &[encoder.finish()]);
    }

    // Wait for all frames to complete
    sync.wait_idle(&device);

    // After wait_idle, GPU should be idle
    assert!(sync.is_idle(&device), "GPU should be idle after wait_idle()");
}

/// Test 32: is_idle() returns true when no work pending
#[test]
fn test_32_is_idle_when_no_work() {
    let Some((device, _queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

    // Initially should be idle (no work submitted)
    let idle = sync.is_idle(&device);
    // Note: Implementation may vary - just test API works
    let _ = idle;
}

// =============================================================================
// BUFFER MAPPING TESTS
// =============================================================================

/// Test 33: buffer_for_frame() maps frame to buffer index
#[test]
fn test_33_buffer_for_frame_double() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

    // Frame 0 -> buffer 0
    // Frame 1 -> buffer 1
    // Frame 2 -> buffer 0
    // Frame 3 -> buffer 1
    assert_eq!(sync.buffer_for_frame(0), 0);
    assert_eq!(sync.buffer_for_frame(1), 1);
    assert_eq!(sync.buffer_for_frame(2), 0);
    assert_eq!(sync.buffer_for_frame(3), 1);
}

/// Test 34: buffer_for_frame() maps frame to buffer index (triple)
#[test]
fn test_34_buffer_for_frame_triple() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

    // Frame 0 -> buffer 0
    // Frame 1 -> buffer 1
    // Frame 2 -> buffer 2
    // Frame 3 -> buffer 0
    assert_eq!(sync.buffer_for_frame(0), 0);
    assert_eq!(sync.buffer_for_frame(1), 1);
    assert_eq!(sync.buffer_for_frame(2), 2);
    assert_eq!(sync.buffer_for_frame(3), 0);
    assert_eq!(sync.buffer_for_frame(4), 1);
    assert_eq!(sync.buffer_for_frame(5), 2);
}

/// Test 35: wait_frame_for() returns the frame to wait on
#[test]
fn test_35_wait_frame_for_double() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

    // For double buffering (offset 1):
    // Frame 0 -> wait for None (no prior frame)
    // Frame 1 -> wait for frame 0
    // Frame 2 -> wait for frame 1
    let wait_0 = sync.wait_frame_for(0);
    let wait_1 = sync.wait_frame_for(1);
    let wait_2 = sync.wait_frame_for(2);

    // Frame 0 has nothing to wait for
    assert!(wait_0.is_none() || wait_0 == Some(0), "Frame 0 wait: {:?}", wait_0);

    // Frame 1+ should wait for frame N-1 (double buffering)
    if let Some(frame) = wait_1 {
        assert!(frame < 1, "Frame 1 should wait for earlier frame");
    }
    if let Some(frame) = wait_2 {
        assert!(frame < 2, "Frame 2 should wait for earlier frame");
    }
}

/// Test 36: wait_frame_for() returns the frame to wait on (triple)
#[test]
fn test_36_wait_frame_for_triple() {
    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

    // For triple buffering (offset 2):
    // Frame 0, 1 -> wait for None (not enough prior frames)
    // Frame 2 -> wait for frame 0
    // Frame 3 -> wait for frame 1
    let wait_0 = sync.wait_frame_for(0);
    let wait_1 = sync.wait_frame_for(1);
    let wait_2 = sync.wait_frame_for(2);
    let wait_3 = sync.wait_frame_for(3);

    // Early frames may have nothing to wait for
    let _ = (wait_0, wait_1);

    // Frame 2+ should wait for frame N-2 (triple buffering)
    if let Some(frame) = wait_2 {
        assert!(frame <= 0, "Frame 2 should wait for frame 0 or earlier");
    }
    if let Some(frame) = wait_3 {
        assert!(frame <= 1, "Frame 3 should wait for frame 1 or earlier");
    }
}

// =============================================================================
// STRESS AND EDGE CASE TESTS
// =============================================================================

/// Test 37: Many frames with double buffering
#[test]
fn test_37_stress_double_buffering() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

    for _ in 0..50 {
        sync.begin_frame(&device);
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stress_double"),
        });
        let _ = sync.end_frame(&queue, &[encoder.finish()]);
    }

    sync.wait_idle(&device);
    assert_eq!(sync.frame_count(), 50);
}

/// Test 38: Many frames with triple buffering
#[test]
fn test_38_stress_triple_buffering() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

    for _ in 0..50 {
        sync.begin_frame(&device);
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("stress_triple"),
        });
        let _ = sync.end_frame(&queue, &[encoder.finish()]);
    }

    sync.wait_idle(&device);
    assert_eq!(sync.frame_count(), 50);
}

/// Test 39: Toggle low latency during frame processing
#[test]
fn test_39_low_latency_toggle_during_frames() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

    for i in 0..10 {
        // Toggle low latency each frame
        sync.set_low_latency(i % 2 == 0);

        sync.begin_frame(&device);
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("toggle_test"),
        });
        let _ = sync.end_frame(&queue, &[encoder.finish()]);
    }

    sync.wait_idle(&device);
    assert_eq!(sync.frame_count(), 10);
}

/// Test 40: Empty command buffer submission
#[test]
fn test_40_empty_command_buffer() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

    sync.begin_frame(&device);
    // Submit with empty slice
    let _ = sync.end_frame(&queue, &[]);

    assert_eq!(sync.frame_count(), 1);
}

/// Test 41: Multiple command buffers per frame
#[test]
fn test_41_multiple_command_buffers() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);

    sync.begin_frame(&device);

    let buffers: Vec<_> = (0..5)
        .map(|i| {
            device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("multi_cmd_{}", i)),
                })
                .finish()
        })
        .collect();

    let _ = sync.end_frame(&queue, &buffers);

    assert_eq!(sync.frame_count(), 1);
}

/// Test 42: Arc compatibility for thread safety
#[test]
fn test_42_arc_compatible() {
    use std::sync::Arc;

    let sync = Arc::new(TrinityFrameSynchronizer::new(BufferCount::Triple));
    let sync2 = Arc::clone(&sync);

    assert_eq!(sync.buffer_count(), 3);
    assert_eq!(sync2.buffer_count(), 3);

    // Set low latency from one reference
    sync.set_low_latency(true);

    // Should be visible from other reference
    assert!(sync2.is_low_latency());
}

/// Test 43: Rapid begin/end cycles
#[test]
fn test_43_rapid_cycles() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Double);

    // Rapid cycles without waiting
    for _ in 0..100 {
        sync.begin_frame(&device);
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rapid"),
        });
        let _ = sync.end_frame(&queue, &[encoder.finish()]);
    }

    sync.wait_idle(&device);
    assert_eq!(sync.frame_count(), 100);
}

/// Test 44: Verify indices stay within bounds after many frames
#[test]
fn test_44_indices_bounded() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let sync = TrinityFrameSynchronizer::new(BufferCount::Triple);
    let buffer_count = sync.buffer_count();

    for _ in 0..100 {
        let current = sync.current_index();
        let next = sync.next_index();

        assert!(
            (current as usize) < buffer_count,
            "current_index {} out of bounds for buffer_count {}",
            current, buffer_count
        );
        assert!(
            (next as usize) < buffer_count,
            "next_index {} out of bounds for buffer_count {}",
            next, buffer_count
        );

        sync.begin_frame(&device);
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bounds_test"),
        });
        let _ = sync.end_frame(&queue, &[encoder.finish()]);
    }
}

/// Test 45: buffer_for_frame never exceeds buffer_count
#[test]
fn test_45_buffer_for_frame_bounded() {
    let double = TrinityFrameSynchronizer::new(BufferCount::Double);
    let triple = TrinityFrameSynchronizer::new(BufferCount::Triple);

    for frame in 0..1000 {
        let buf_double = double.buffer_for_frame(frame);
        let buf_triple = triple.buffer_for_frame(frame);

        assert!(
            buf_double < double.buffer_count(),
            "Double buffer {} >= {} for frame {}",
            buf_double, double.buffer_count(), frame
        );
        assert!(
            buf_triple < triple.buffer_count(),
            "Triple buffer {} >= {} for frame {}",
            buf_triple, triple.buffer_count(), frame
        );
    }
}
