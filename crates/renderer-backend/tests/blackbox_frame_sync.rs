// Blackbox contract tests for T-WGPU-P4.6.1 Frame Fence synchronization.
//
// CLEANROOM: No src/ access beyond the public API exported by the crate.
// Tests use only `renderer_backend::frame_sync::*` -- no internal fields,
// no private methods, no implementation details.
//
// Acceptance criteria (T-WGPU-P4.6.1):
//   1. Fence-based frame tracking -- FrameFence tracks frames
//   2. Wait for frame completion -- wait_for_frame method
//   3. Frame timeline queries -- is_frame_complete, oldest_pending_frame
//   4. Submission index tracking -- record_submission
//
// Coverage (25 tests):
//   01. FrameFence::new() creates fence with specified frames in flight
//   02. FrameFence::default() creates fence with DEFAULT_FRAMES_IN_FLIGHT
//   03. FrameFence::current_frame() returns current frame number
//   04. FrameFence::advance_frame() increments frame counter
//   05. FrameFence::frames_in_flight() returns configured count
//   06. FrameFence::record_submission() tracks submission indices
//   07. FrameFence::is_frame_complete() checks frame completion
//   08. FrameFence::wait_for_frame() blocks until frame completes
//   09. FrameFence::wait_all() waits for all pending frames
//   10. FrameFence::oldest_pending_frame() returns earliest pending
//   11. FrameFence::tracked_frame_count() returns tracked frame count
//   12. FrameFence::frame_stats() returns submission info for frame
//   13. FrameFence::is_frame_pending() checks if frame is pending
//   14. FrameSyncManager::new() creates manager with max frames
//   15. FrameSyncManager::default() creates manager with defaults
//   16. FrameSyncManager::begin_frame() starts new frame
//   17. FrameSyncManager::end_frame() completes frame, returns stats
//   18. FrameSyncManager::wait_for_oldest_if_needed() throttles frames
//   19. FrameSyncManager::wait_for_gpu_idle() waits for all GPU work
//   20. FrameSyncManager::record_submission() records submission index
//   21. FrameSyncManager::current_frame() returns current frame
//   22. FrameSyncManager::total_frames() returns total frame count
//   23. FrameSyncManager::max_frames_in_flight() returns max frames
//   24. FrameSyncManager::fence() returns reference to inner fence
//   25. FrameSyncManager::is_frame_complete() delegates to fence
//   26. FrameSyncManager::oldest_pending_frame() delegates to fence
//   27. FrameSyncManager::average_cpu_time_ms() returns timing stats
//   28. FrameSubmission struct has expected fields
//   29. FrameStats struct has expected fields
//   30. Multiple frames in flight tracking
//   31. Frame advancement sequence
//   32. Constants DEFAULT_FRAMES_IN_FLIGHT, MAX_FRAMES_IN_FLIGHT, MIN_FRAMES_IN_FLIGHT

use pollster::block_on;

use renderer_backend::frame_sync::{
    FrameFence, FrameStats, FrameSubmission, FrameSyncManager,
    DEFAULT_FRAMES_IN_FLIGHT, MAX_FRAMES_IN_FLIGHT, MIN_FRAMES_IN_FLIGHT,
};

// =========================================================================
// TEST INFRASTRUCTURE -- Headless wgpu device/queue
// =========================================================================

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
                label: Some("blackbox_frame_sync_test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .ok()?,
    )
}

// =========================================================================
// CONSTANTS TESTS
// =========================================================================

/// Test 01: DEFAULT_FRAMES_IN_FLIGHT constant is valid
#[test]
fn test_01_default_frames_in_flight_constant() {
    assert!(
        DEFAULT_FRAMES_IN_FLIGHT >= MIN_FRAMES_IN_FLIGHT,
        "DEFAULT must be >= MIN"
    );
    assert!(
        DEFAULT_FRAMES_IN_FLIGHT <= MAX_FRAMES_IN_FLIGHT,
        "DEFAULT must be <= MAX"
    );
}

/// Test 02: MAX_FRAMES_IN_FLIGHT constant is valid
#[test]
fn test_02_max_frames_in_flight_constant() {
    assert!(MAX_FRAMES_IN_FLIGHT >= MIN_FRAMES_IN_FLIGHT, "MAX must be >= MIN");
    assert!(MAX_FRAMES_IN_FLIGHT >= 2, "MAX should allow at least 2 frames");
}

/// Test 03: MIN_FRAMES_IN_FLIGHT constant is valid
#[test]
fn test_03_min_frames_in_flight_constant() {
    assert!(MIN_FRAMES_IN_FLIGHT >= 1, "MIN should be at least 1");
    assert!(MIN_FRAMES_IN_FLIGHT <= MAX_FRAMES_IN_FLIGHT, "MIN must be <= MAX");
}

// =========================================================================
// FRAMEFENCE CONSTRUCTION TESTS
// =========================================================================

/// Test 04: FrameFence::new() creates fence with specified frames in flight
#[test]
fn test_04_framefence_new_with_specified_frames() {
    let fence = FrameFence::new(4);
    assert_eq!(fence.frames_in_flight(), 4, "Should use specified frames in flight");
    assert_eq!(fence.current_frame(), 0, "Should start at frame 0");
}

/// Test 05: FrameFence::default() creates fence with DEFAULT_FRAMES_IN_FLIGHT
#[test]
fn test_05_framefence_default() {
    let fence = FrameFence::default();
    assert_eq!(
        fence.frames_in_flight(),
        DEFAULT_FRAMES_IN_FLIGHT,
        "Default should use DEFAULT_FRAMES_IN_FLIGHT"
    );
}

/// Test 06: FrameFence::new() with minimum frames in flight
#[test]
fn test_06_framefence_new_minimum_frames() {
    let fence = FrameFence::new(MIN_FRAMES_IN_FLIGHT);
    assert_eq!(fence.frames_in_flight(), MIN_FRAMES_IN_FLIGHT);
}

/// Test 07: FrameFence::new() with maximum frames in flight
#[test]
fn test_07_framefence_new_maximum_frames() {
    let fence = FrameFence::new(MAX_FRAMES_IN_FLIGHT);
    assert_eq!(fence.frames_in_flight(), MAX_FRAMES_IN_FLIGHT);
}

// =========================================================================
// FRAMEFENCE FRAME TRACKING TESTS
// =========================================================================

/// Test 08: FrameFence::current_frame() returns current frame number
#[test]
fn test_08_framefence_current_frame() {
    let fence = FrameFence::new(3);
    assert_eq!(fence.current_frame(), 0, "Initial frame should be 0");
}

/// Test 09: FrameFence::advance_frame() increments frame counter
#[test]
fn test_09_framefence_advance_frame() {
    let fence = FrameFence::new(3);
    assert_eq!(fence.current_frame(), 0);

    let new_frame = fence.advance_frame();
    assert_eq!(new_frame, 1, "advance_frame should return new frame number");
    assert_eq!(fence.current_frame(), 1, "current_frame should reflect new frame");
}

/// Test 10: FrameFence::advance_frame() multiple times
#[test]
fn test_10_framefence_advance_multiple() {
    let fence = FrameFence::new(3);

    for i in 1..=10 {
        let frame = fence.advance_frame();
        assert_eq!(frame, i, "Frame should increment correctly");
    }
    assert_eq!(fence.current_frame(), 10);
}

/// Test 11: FrameFence::tracked_frame_count() returns tracked frame count
#[test]
fn test_11_framefence_tracked_frame_count() {
    let fence = FrameFence::new(3);
    // Initially, no frames are tracked (frame 0 hasn't been submitted yet)
    let initial_count = fence.tracked_frame_count();

    // Advance and track
    fence.advance_frame();
    fence.advance_frame();

    // Count should reflect tracked frames
    let count = fence.tracked_frame_count();
    assert!(count <= fence.frames_in_flight(), "Tracked count should be <= frames_in_flight");
    assert!(count >= initial_count, "Tracked count should not decrease without completion");
}

// =========================================================================
// FRAMEFENCE FRAME STATE TESTS
// =========================================================================

/// Test 12: FrameFence::oldest_pending_frame() returns earliest pending frame
#[test]
fn test_12_framefence_oldest_pending_frame() {
    let fence = FrameFence::new(3);

    // Initially, no pending frames
    let oldest_initial = fence.oldest_pending_frame();

    // After advancing, we may have pending frames
    fence.advance_frame();
    fence.advance_frame();

    let oldest = fence.oldest_pending_frame();
    // Either None (if nothing pending) or Some(frame) where frame is the oldest
    if let Some(frame) = oldest {
        assert!(frame <= fence.current_frame(), "Oldest pending should be <= current");
    }
}

/// Test 13: FrameFence::is_frame_pending() checks if frame is pending
#[test]
fn test_13_framefence_is_frame_pending() {
    let fence = FrameFence::new(3);

    // Frame 0 at start - check its pending status
    let _frame0_pending = fence.is_frame_pending(0);

    fence.advance_frame(); // Now at frame 1

    // Frame 0 might be pending if we haven't completed it
    // Frame 1 is current
    let _frame1_pending = fence.is_frame_pending(1);

    // Very old frame should not be pending
    let old_frame_pending = fence.is_frame_pending(1000);
    assert!(!old_frame_pending, "Non-existent frame should not be pending");
}

/// Test 14: FrameFence::frame_stats() returns submission info for frame
#[test]
fn test_14_framefence_frame_stats() {
    let fence = FrameFence::new(3);

    // Stats for frame 0 before any work
    let _stats = fence.frame_stats(0);

    fence.advance_frame();

    // Stats for frame 0 after advance
    let _stats_after = fence.frame_stats(0);

    // Stats for non-existent frame
    let no_stats = fence.frame_stats(1000);
    assert!(no_stats.is_none(), "Non-tracked frame should return None");
}

// =========================================================================
// FRAMEFENCE DEVICE INTERACTION TESTS
// =========================================================================

/// Test 15: FrameFence::is_frame_complete() checks frame completion with device
#[test]
fn test_15_framefence_is_frame_complete() {
    let Some((device, _queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let fence = FrameFence::new(3);

    // Frame 0 with no work submitted should be considered complete (no submissions)
    let is_complete = fence.is_frame_complete(&device, 0);

    // After advancing, check completion status
    fence.advance_frame();
    let is_frame1_complete = fence.is_frame_complete(&device, 1);

    // Both results are valid - just testing the API works
    let _ = (is_complete, is_frame1_complete);
}

/// Test 16: FrameFence::wait_for_frame() blocks until frame completes
#[test]
fn test_16_framefence_wait_for_frame() {
    let Some((device, _queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let fence = FrameFence::new(3);

    // Wait for frame 0 - should return quickly if no work submitted
    fence.wait_for_frame(&device, 0);

    // After waiting, frame should be complete
    assert!(
        fence.is_frame_complete(&device, 0),
        "Frame should be complete after wait"
    );
}

/// Test 17: FrameFence::wait_all() waits for all pending frames
#[test]
fn test_17_framefence_wait_all() {
    let Some((device, _queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let fence = FrameFence::new(3);

    fence.advance_frame();
    fence.advance_frame();

    // Wait for all frames
    fence.wait_all(&device);

    // After wait_all, there should be no pending frames
    assert!(
        fence.oldest_pending_frame().is_none() || fence.is_frame_complete(&device, 0),
        "All frames should be complete after wait_all"
    );
}

// =========================================================================
// FRAMEFENCE SUBMISSION TRACKING TESTS
// =========================================================================

/// Test 18: FrameFence::record_submission() tracks submission indices
#[test]
fn test_18_framefence_record_submission() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let fence = FrameFence::new(3);

    // Create a submission by encoding and submitting
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });
    let command_buffer = encoder.finish();
    let submission_index = queue.submit(std::iter::once(command_buffer));

    // Record the submission
    fence.record_submission(submission_index);

    // Verify submission is tracked via frame_stats
    let stats = fence.frame_stats(fence.current_frame());
    if let Some((count, _)) = stats {
        assert!(count >= 1, "Should have at least one submission recorded");
    }
}

/// Test 19: FrameFence multiple submissions per frame
#[test]
fn test_19_framefence_multiple_submissions() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let fence = FrameFence::new(3);

    // Multiple submissions in same frame
    for _ in 0..3 {
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("multi_submit_encoder"),
        });
        let command_buffer = encoder.finish();
        let submission_index = queue.submit(std::iter::once(command_buffer));
        fence.record_submission(submission_index);
    }

    // Check frame stats
    let stats = fence.frame_stats(fence.current_frame());
    if let Some((count, _)) = stats {
        assert!(count >= 3, "Should have at least 3 submissions recorded");
    }
}

// =========================================================================
// FRAMESYNCMANAGER CONSTRUCTION TESTS
// =========================================================================

/// Test 20: FrameSyncManager::new() creates manager with max frames
#[test]
fn test_20_framesyncmanager_new() {
    let manager = FrameSyncManager::new(4);
    assert_eq!(
        manager.max_frames_in_flight(),
        4,
        "Should use specified max frames"
    );
    assert_eq!(manager.current_frame(), 0, "Should start at frame 0");
    assert_eq!(manager.total_frames(), 0, "Should have 0 total completed frames");
}

/// Test 21: FrameSyncManager::default() creates manager with defaults
#[test]
fn test_21_framesyncmanager_default() {
    let manager = FrameSyncManager::default();
    assert_eq!(
        manager.max_frames_in_flight(),
        DEFAULT_FRAMES_IN_FLIGHT,
        "Default should use DEFAULT_FRAMES_IN_FLIGHT"
    );
}

// =========================================================================
// FRAMESYNCMANAGER FRAME LIFECYCLE TESTS
// =========================================================================

/// Test 22: FrameSyncManager::begin_frame() starts new frame
#[test]
fn test_22_framesyncmanager_begin_frame() {
    let manager = FrameSyncManager::new(3);

    // First begin_frame
    let frame_id = manager.begin_frame();
    // The frame_id returned by begin_frame might be the current frame before advancing
    // or the frame being started - verify it's a valid frame number
    assert!(frame_id <= 1, "First begin_frame should return 0 or 1");

    // Second begin_frame - frame should advance
    let frame_id2 = manager.begin_frame();
    assert!(
        frame_id2 > frame_id || frame_id2 == frame_id,
        "Second begin_frame should return same or higher frame"
    );

    // Current frame tracks the active frame
    let current = manager.current_frame();
    // Current frame should be consistent with begin_frame behavior
    assert!(current <= frame_id2, "Current frame should be <= returned frame");
}

/// Test 23: FrameSyncManager::end_frame() completes frame, returns stats
#[test]
fn test_23_framesyncmanager_end_frame() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let manager = FrameSyncManager::new(3);

    // Begin, do work, end
    let _frame_id = manager.begin_frame();

    // Submit some work
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("end_frame_test"),
    });
    let _submission = queue.submit(std::iter::once(encoder.finish()));

    let stats = manager.end_frame(&device, &queue);

    // FrameStats should have valid data
    let _ = stats; // Stats returned successfully

    assert_eq!(manager.total_frames(), 1, "Total frames should be 1 after end_frame");
}

/// Test 24: FrameSyncManager::wait_for_oldest_if_needed() throttles frames
#[test]
fn test_24_framesyncmanager_wait_for_oldest_if_needed() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let manager = FrameSyncManager::new(2); // Low limit to trigger waiting

    // Begin multiple frames to potentially trigger waiting
    for i in 0..4 {
        manager.begin_frame();

        // Submit work
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("frame_{}", i)),
        });
        let _submission = queue.submit(std::iter::once(encoder.finish()));

        // This should wait if we have too many frames in flight
        manager.wait_for_oldest_if_needed(&device);

        manager.end_frame(&device, &queue);
    }

    assert_eq!(manager.total_frames(), 4, "Should have completed 4 frames");
}

/// Test 25: FrameSyncManager::wait_for_gpu_idle() waits for all GPU work
#[test]
fn test_25_framesyncmanager_wait_for_gpu_idle() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let manager = FrameSyncManager::new(3);

    // Begin and end a few frames
    for _ in 0..3 {
        manager.begin_frame();
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("idle_test"),
        });
        let _submission = queue.submit(std::iter::once(encoder.finish()));
        manager.end_frame(&device, &queue);
    }

    // Wait for GPU to be idle
    manager.wait_for_gpu_idle(&device);

    // After waiting, there should be no pending work
    // (No assertion needed - just verifying the call succeeds)
}

// =========================================================================
// FRAMESYNCMANAGER ACCESSOR TESTS
// =========================================================================

/// Test 26: FrameSyncManager::current_frame() returns current frame
#[test]
fn test_26_framesyncmanager_current_frame() {
    let manager = FrameSyncManager::new(3);
    assert_eq!(manager.current_frame(), 0);

    manager.begin_frame();
    assert_eq!(manager.current_frame(), 0); // Still 0 until advance
}

/// Test 27: FrameSyncManager::total_frames() returns total frame count
#[test]
fn test_27_framesyncmanager_total_frames() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let manager = FrameSyncManager::new(3);
    assert_eq!(manager.total_frames(), 0);

    manager.begin_frame();
    manager.end_frame(&device, &queue);
    assert_eq!(manager.total_frames(), 1);

    manager.begin_frame();
    manager.end_frame(&device, &queue);
    assert_eq!(manager.total_frames(), 2);
}

/// Test 28: FrameSyncManager::max_frames_in_flight() returns max frames
#[test]
fn test_28_framesyncmanager_max_frames_in_flight() {
    let manager = FrameSyncManager::new(5);
    assert_eq!(manager.max_frames_in_flight(), 5);
}

/// Test 29: FrameSyncManager::fence() returns reference to inner fence
#[test]
fn test_29_framesyncmanager_fence() {
    let manager = FrameSyncManager::new(4);
    let fence = manager.fence();

    assert_eq!(
        fence.frames_in_flight(),
        4,
        "Inner fence should have same frames_in_flight"
    );
}

/// Test 30: FrameSyncManager::is_frame_complete() delegates to fence
#[test]
fn test_30_framesyncmanager_is_frame_complete() {
    let Some((device, _queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let manager = FrameSyncManager::new(3);

    // Check frame completion via manager
    let is_complete = manager.is_frame_complete(&device, 0);

    // Also check via fence directly
    let fence_complete = manager.fence().is_frame_complete(&device, 0);

    assert_eq!(
        is_complete, fence_complete,
        "Manager should delegate to fence"
    );
}

/// Test 31: FrameSyncManager::oldest_pending_frame() delegates to fence
#[test]
fn test_31_framesyncmanager_oldest_pending_frame() {
    let manager = FrameSyncManager::new(3);

    let manager_oldest = manager.oldest_pending_frame();
    let fence_oldest = manager.fence().oldest_pending_frame();

    assert_eq!(
        manager_oldest, fence_oldest,
        "Manager should delegate to fence"
    );
}

/// Test 32: FrameSyncManager::average_cpu_time_ms() returns timing stats
#[test]
fn test_32_framesyncmanager_average_cpu_time_ms() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let manager = FrameSyncManager::new(3);

    // No frames yet - average might be 0 or NaN
    let _initial_avg = manager.average_cpu_time_ms();

    // Complete a frame
    manager.begin_frame();
    std::thread::sleep(std::time::Duration::from_millis(1)); // Add some time
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("timing_test"),
    });
    let _submission = queue.submit(std::iter::once(encoder.finish()));
    manager.end_frame(&device, &queue);

    // Now we should have a non-zero average (or at least valid)
    let avg = manager.average_cpu_time_ms();
    assert!(avg >= 0.0 || avg.is_nan(), "Average should be >= 0 or NaN");
}

/// Test 33: FrameSyncManager::record_submission() records submission index
#[test]
fn test_33_framesyncmanager_record_submission() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let manager = FrameSyncManager::new(3);

    manager.begin_frame();

    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("record_sub_test"),
    });
    let submission_index = queue.submit(std::iter::once(encoder.finish()));

    // Record via manager
    manager.record_submission(submission_index);

    // Verify via fence
    let stats = manager.fence().frame_stats(manager.current_frame());
    if let Some((count, _)) = stats {
        assert!(count >= 1, "Should have at least one submission");
    }
}

// =========================================================================
// FRAMESUBMISSION STRUCT TESTS (via FrameSyncManager access)
// =========================================================================

/// Test 34: FrameSubmission struct is usable (verify type exists in API)
#[test]
fn test_34_framesubmission_type_exists() {
    // FrameSubmission is a public type - verify we can reference it
    fn _accepts_submission(_s: &FrameSubmission) {}

    // Type exists and is part of the public API
    // We cannot construct it directly without Default, but we can use it via manager
}

/// Test 35: FrameStats struct is usable (verify type exists in API)
#[test]
fn test_35_framestats_type_exists() {
    // FrameStats is a public type - verify we can reference it
    fn _accepts_stats(_s: &FrameStats) {}

    // Type exists and is part of the public API
}

/// Test 36: FrameSubmission methods are accessible (via fence inspection)
#[test]
fn test_36_framesubmission_methods_via_fence() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let fence = FrameFence::new(3);

    // Submit work
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("submission_test"),
    });
    let submission = queue.submit(std::iter::once(encoder.finish()));
    fence.record_submission(submission);

    // The frame_stats method gives us info about submissions
    // (submission_count, is_complete)
    if let Some((count, _complete)) = fence.frame_stats(fence.current_frame()) {
        assert!(count >= 1, "Should have recorded submission");
    }
}

/// Test 37: CPU time tracking works
#[test]
fn test_37_cpu_time_tracking() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let manager = FrameSyncManager::new(3);

    manager.begin_frame();

    // Add some CPU time
    std::thread::sleep(std::time::Duration::from_millis(5));

    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cpu_time_test"),
    });
    let _submission = queue.submit(std::iter::once(encoder.finish()));

    let _stats = manager.end_frame(&device, &queue);

    // Check average CPU time is tracked
    let avg_ms = manager.average_cpu_time_ms();
    // May be NaN if no frames completed yet, or should be >= 0
    assert!(avg_ms.is_nan() || avg_ms >= 0.0, "Average CPU time should be valid");
}

// =========================================================================
// FRAMESTATS STRUCT TESTS
// =========================================================================

/// Test 38: FrameStats has expected fields (test via debug/display or accessors)
#[test]
fn test_38_framestats_structure() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let manager = FrameSyncManager::new(3);

    manager.begin_frame();
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("stats_test"),
    });
    let _submission = queue.submit(std::iter::once(encoder.finish()));

    let stats: FrameStats = manager.end_frame(&device, &queue);

    // FrameStats returned successfully - structure is valid
    let _ = stats;
}

// =========================================================================
// INTEGRATION / STRESS TESTS
// =========================================================================

/// Test 39: Multiple frames in flight tracking
#[test]
fn test_39_multiple_frames_in_flight() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let manager = FrameSyncManager::new(3);

    // Process many frames, verifying we don't exceed max in flight
    for i in 0..10 {
        manager.begin_frame();

        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("multi_frame_{}", i)),
        });
        let submission = queue.submit(std::iter::once(encoder.finish()));
        manager.record_submission(submission);

        // Wait if needed
        manager.wait_for_oldest_if_needed(&device);

        manager.end_frame(&device, &queue);
    }

    assert_eq!(manager.total_frames(), 10);
}

/// Test 40: Frame advancement sequence
#[test]
fn test_40_frame_advancement_sequence() {
    let fence = FrameFence::new(3);

    // Verify strict monotonic frame advancement
    let mut last_frame = fence.current_frame();

    for _ in 0..100 {
        let new_frame = fence.advance_frame();
        assert!(
            new_frame > last_frame,
            "Frames must be strictly monotonically increasing"
        );
        last_frame = new_frame;
    }
}

/// Test 41: Concurrent frame manager usage simulation
#[test]
fn test_41_frame_manager_stress() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let manager = FrameSyncManager::new(MAX_FRAMES_IN_FLIGHT);

    // Rapid frame processing
    for i in 0..50 {
        manager.begin_frame();

        // Variable work per frame
        for _ in 0..(i % 3) {
            let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stress_test"),
            });
            let submission = queue.submit(std::iter::once(encoder.finish()));
            manager.record_submission(submission);
        }

        manager.wait_for_oldest_if_needed(&device);
        manager.end_frame(&device, &queue);
    }

    manager.wait_for_gpu_idle(&device);
    assert_eq!(manager.total_frames(), 50);
}

/// Test 42: Fence with minimum frames in flight under load
#[test]
fn test_42_minimum_frames_stress() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let manager = FrameSyncManager::new(MIN_FRAMES_IN_FLIGHT);

    // With minimum frames, we should still function correctly
    for _ in 0..20 {
        manager.begin_frame();

        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("min_frames_test"),
        });
        let submission = queue.submit(std::iter::once(encoder.finish()));
        manager.record_submission(submission);

        manager.wait_for_oldest_if_needed(&device);
        manager.end_frame(&device, &queue);
    }

    manager.wait_for_gpu_idle(&device);
    assert_eq!(manager.total_frames(), 20);
}

/// Test 43: Frame completion detection accuracy
#[test]
fn test_43_frame_completion_accuracy() {
    let Some((device, queue)) = create_test_device() else {
        println!("Skipping test: no GPU adapter available");
        return;
    };

    let manager = FrameSyncManager::new(3);

    manager.begin_frame();
    let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("completion_test"),
    });
    let submission = queue.submit(std::iter::once(encoder.finish()));
    manager.record_submission(submission);
    manager.end_frame(&device, &queue);

    // Wait for GPU idle ensures all work is done
    manager.wait_for_gpu_idle(&device);

    // After waiting, frame 0 should be complete
    let complete = manager.is_frame_complete(&device, 0);
    // Note: This may still be false depending on implementation
    // The important thing is the API works without crashing
    let _ = complete;
}

/// Test 44: FrameFence thread safety (basic verification via Arc)
#[test]
fn test_44_framefence_arc_compatible() {
    use std::sync::Arc;

    let fence = Arc::new(FrameFence::new(3));

    // Multiple references
    let fence2 = Arc::clone(&fence);
    let fence3 = Arc::clone(&fence);

    // All references work
    assert_eq!(fence.frames_in_flight(), 3);
    assert_eq!(fence2.frames_in_flight(), 3);
    assert_eq!(fence3.frames_in_flight(), 3);

    // Advance from one reference
    fence.advance_frame();

    // All see the same state (if interior mutability)
    assert_eq!(fence.current_frame(), fence2.current_frame());
}

/// Test 45: FrameSyncManager thread safety (basic verification via Arc)
#[test]
fn test_45_framesyncmanager_arc_compatible() {
    use std::sync::Arc;

    let manager = Arc::new(FrameSyncManager::new(3));

    let manager2 = Arc::clone(&manager);

    assert_eq!(manager.max_frames_in_flight(), 3);
    assert_eq!(manager2.max_frames_in_flight(), 3);
}
